#!/bin/bash
# ============================================================
# GSA Data Flywheel — Full Stack Launch (vLLM Edition)
# ============================================================
# Usage:
#   bash scripts/run_gsa_flywheel.sh                    # Start all (vLLM + services)
#   bash scripts/run_gsa_flywheel.sh --continuous       # Continuous data production
#   bash scripts/run_gsa_flywheel.sh --mode api         # Use remote API instead of local vLLM
#   bash scripts/run_gsa_flywheel.sh --stop             # Stop all
#   bash scripts/run_gsa_flywheel.sh --status           # Check health
#   bash scripts/run_gsa_flywheel.sh --viewer-only      # Only start viewer
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ============================================================
# Configuration
# ============================================================
LOCAL_MODEL_PATH="/path/to/environment/hf_models/Qwen/Qwen3.6-27B"
LOCAL_MODEL_NAME="Qwen3.6-27B"
LOCAL_PYTHON="/path/to/environment/env/qwen36/bin/python"

VLLM_PORT_A=10301
VLLM_PORT_B=10302
PROXY_PORT=10300

API_URL="${VLM_URL:-YOUR_VLM_ENDPOINT}"
API_MODEL="Qwen3.5-397B-A17B"
API_KEY="${VLM_API_KEY:-YOUR_VLM_API_KEY}"

GENERATOR_PORT=10305
VERIFIER_PORT=10306
VIEWER_PORT=10307
LOG_DIR="$PROJECT_ROOT/logs/gsa_flywheel"
PID_DIR="$LOG_DIR/pids"
OUTPUT_DIR="$PROJECT_ROOT/data/output/gsa_flywheel_exp"

WATCHDOG_INTERVAL=30

# ============================================================
# Argument Parsing
# ============================================================
MODE="${GSA_MODE:-local}"
MAX_IMAGES="${GSA_MAX_IMAGES:-50}"
CONTINUOUS=0
STOP=0
STATUS=0
VIEWER_ONLY=0
MAX_WORKERS="${VLM_WORKERS:-32}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --stop) STOP=1; shift ;;
        --status) STATUS=1; shift ;;
        --images) MAX_IMAGES="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --continuous) CONTINUOUS=1; shift ;;
        --viewer-only) VIEWER_ONLY=1; shift ;;
        --workers) MAX_WORKERS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================
# Functions
# ============================================================

stop_all() {
    echo -e "${YELLOW}Stopping all GSA Flywheel processes...${NC}"

    # Stop vLLM instances
    bash "$SCRIPT_DIR/start_vllm_serving.sh" --stop 2>/dev/null || true

    # Stop application services
    for pidfile in "$PID_DIR"/*.pid; do
        [ -f "$pidfile" ] || continue
        pid=$(cat "$pidfile")
        name=$(basename "$pidfile" .pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo -e "  ${RED}●${NC} Stopped $name (PID $pid)"
        fi
        rm -f "$pidfile"
    done

    # Kill by port as fallback
    for port in $PROXY_PORT $GENERATOR_PORT $VERIFIER_PORT $VIEWER_PORT; do
        fuser -k "$port/tcp" 2>/dev/null || true
    done

    echo -e "${GREEN}All processes stopped.${NC}"
}

show_status() {
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  GSA Data Flywheel — Status${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo ""

    # vLLM status
    echo -e "${CYAN}  vLLM Instances:${NC}"
    for port in $VLLM_PORT_A $VLLM_PORT_B; do
        resp=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/v1/models" 2>/dev/null || echo "000")
        if [ "$resp" = "200" ]; then
            echo -e "  ${GREEN}●${NC} vLLM :$port — healthy"
        else
            echo -e "  ${RED}●${NC} vLLM :$port — down"
        fi
    done
    echo ""

    # Application services
    echo -e "${CYAN}  Application Services:${NC}"
    for pidfile in "$PID_DIR"/*.pid; do
        [ -f "$pidfile" ] || continue
        pid=$(cat "$pidfile")
        name=$(basename "$pidfile" .pid)
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} $name (PID $pid) — running"
        else
            echo -e "  ${RED}●${NC} $name (PID $pid) — dead"
        fi
    done
    echo ""

    # Service endpoints
    echo -e "${CYAN}  Endpoints:${NC}"
    for port in $GENERATOR_PORT $VERIFIER_PORT $VIEWER_PORT; do
        resp=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null || echo "000")
        if [ "$resp" = "200" ]; then
            echo -e "  ${GREEN}●${NC} Port $port — OK"
        else
            echo -e "  ${RED}●${NC} Port $port — down"
        fi
    done
    echo ""

    # Generator status
    echo -e "${CYAN}  Generator Status:${NC}"
    curl -s "http://localhost:$GENERATOR_PORT/status" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  Not reachable"
    echo ""

    # GPU utilization
    echo -e "${CYAN}  GPU Utilization:${NC}"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
        while IFS=, read -r idx util mem_used mem_total; do
            echo -e "  GPU $idx: util=${util}, mem=${mem_used}/${mem_total}"
        done
}

watchdog() {
    echo -e "${CYAN}Watchdog started (interval=${WATCHDOG_INTERVAL}s)${NC}"
    while true; do
        sleep $WATCHDOG_INTERVAL

        # Check generator
        if [ -f "$PID_DIR/generator.pid" ]; then
            pid=$(cat "$PID_DIR/generator.pid")
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "  ${YELLOW}●${NC} Generator died, restarting..."
                start_generator
            fi
        fi

        # Check verifier
        if [ -f "$PID_DIR/verifier.pid" ]; then
            pid=$(cat "$PID_DIR/verifier.pid")
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "  ${YELLOW}●${NC} Verifier died, restarting..."
                start_verifier
            fi
        fi

        # Check viewer
        if [ -f "$PID_DIR/viewer.pid" ]; then
            pid=$(cat "$PID_DIR/viewer.pid")
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "  ${YELLOW}●${NC} Viewer died, restarting..."
                start_viewer
            fi
        fi
    done
}

start_generator() {
    GSA_AUTO_START=1 GSA_MAX_IMAGES=$MAX_IMAGES GSA_CONTINUOUS=$( [ $CONTINUOUS -eq 1 ] && echo "1" || echo "0" ) \
        VLM_URL_A="http://localhost:${VLLM_PORT_A}/v1" \
        VLM_URL_B="http://localhost:${VLLM_PORT_B}/v1" \
        VLM_URL="http://localhost:${PROXY_PORT}/v1" \
        VLM_MODEL="$VLM_MODEL" VLM_API_KEY="$VLM_API_KEY" \
        VLM_WORKERS="$MAX_WORKERS" \
        python scripts/generator_service.py \
        >"$LOG_DIR/generator.log" 2>&1 &
    echo $! > "$PID_DIR/generator.pid"
    echo -e "  ${GREEN}●${NC} Generator started (PID $(cat $PID_DIR/generator.pid), workers=$MAX_WORKERS)"
}

start_verifier() {
    python scripts/verifier_service.py \
        >"$LOG_DIR/verifier.log" 2>&1 &
    echo $! > "$PID_DIR/verifier.pid"
    echo -e "  ${GREEN}●${NC} Verifier started (PID $(cat $PID_DIR/verifier.pid))"
}

start_viewer() {
    python scripts/viewer_service.py \
        >"$LOG_DIR/viewer.log" 2>&1 &
    echo $! > "$PID_DIR/viewer.pid"
    echo -e "  ${GREEN}●${NC} Viewer started (PID $(cat $PID_DIR/viewer.pid))"
}

# ============================================================
# Main Logic
# ============================================================

if [ $STOP -eq 1 ]; then stop_all; exit 0; fi
if [ $STATUS -eq 1 ]; then show_status; exit 0; fi

# Stop existing processes
stop_all 2>/dev/null || true

mkdir -p "$LOG_DIR" "$PID_DIR" "$OUTPUT_DIR"

echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  GSA Data Flywheel — Starting (vLLM Edition)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "  Mode:       ${MODE}"
echo -e "  Continuous: $( [ $CONTINUOUS -eq 1 ] && echo 'YES' || echo 'NO' )"
echo -e "  Workers:    ${MAX_WORKERS}"
echo -e "  Images:     ${MAX_IMAGES} per dimension"
echo ""

# Set VLM environment based on mode
if [ "$MODE" = "local" ]; then
    export VLM_MODEL="$LOCAL_MODEL_NAME"
    export VLM_API_KEY=""

    # [0/4] Start vLLM dual instances
    if [ $VIEWER_ONLY -eq 0 ]; then
        echo -e "[0/4] Starting vLLM dual instances (8x A100)..."
        bash "$SCRIPT_DIR/start_vllm_serving.sh" 2>&1 | sed 's/^/  /'
        echo ""
    fi
else
    export VLM_URL="$API_URL"
    export VLM_MODEL="$API_MODEL"
    export VLM_API_KEY="$API_KEY"
    echo -e "  Using API mode: ${VLM_MODEL}"
    echo ""
fi

if [ $VIEWER_ONLY -eq 1 ]; then
    echo -e "[1/1] Starting Viewer only..."
    start_viewer
    echo ""
    echo -e "${GREEN}Viewer ready at http://$(hostname -I | awk '{print $1}'):${VIEWER_PORT}/${NC}"
    exit 0
fi

# [1/4] Generator
echo -e "[1/4] Starting Generator (port ${GENERATOR_PORT}, ${MAX_WORKERS} workers)..."
start_generator
sleep 2

# [2/4] Verifier
echo -e "[2/4] Starting Verifier (port ${VERIFIER_PORT}, 10s scan)..."
start_verifier
sleep 1

# [3/4] Viewer
echo -e "[3/4] Starting Viewer (port ${VIEWER_PORT})..."
start_viewer
sleep 1

# [4/4] Watchdog
echo -e "[4/4] Starting Watchdog (${WATCHDOG_INTERVAL}s interval)..."
watchdog &
echo $! > "$PID_DIR/watchdog.pid"
echo -e "  ${GREEN}●${NC} Watchdog started (PID $(cat $PID_DIR/watchdog.pid))"

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All services launched!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo -e "  Generator: http://${IP}:${GENERATOR_PORT}/status"
echo -e "  Verifier:  http://${IP}:${VERIFIER_PORT}/status"
echo -e "  Viewer:    http://${IP}:${VIEWER_PORT}/"
echo ""
echo -e "  Monitor:   tail -f ${LOG_DIR}/generator.log"
echo -e "  Status:    bash scripts/run_gsa_flywheel.sh --status"
echo -e "  Stop:      bash scripts/run_gsa_flywheel.sh --stop"
echo ""
if [ $CONTINUOUS -eq 1 ]; then
    echo -e "  ${YELLOW}CONTINUOUS MODE: Data production will run indefinitely.${NC}"
    echo -e "  ${YELLOW}Use --stop to gracefully shutdown.${NC}"
fi
echo ""
