#!/bin/bash
# ============================================================
# vLLM Dual-Instance Serving — 8x A100 Full Utilization
# ============================================================
# Deploys two vLLM instances with tensor-parallel=4 each,
# covering all 8 GPUs for maximum throughput.
#
# Instance A: GPU 0-3, port 10301
# Instance B: GPU 4-7, port 10302
#
# Usage:
#   bash scripts/start_vllm_serving.sh          # Start both instances
#   bash scripts/start_vllm_serving.sh --stop   # Stop all
#   bash scripts/start_vllm_serving.sh --status # Check health
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================
# Configuration
# ============================================================
MODEL_PATH="/path/to/environment/hf_models/Qwen/Qwen3.6-27B"
MODEL_NAME="Qwen3.6-27B"
PYTHON="/path/to/environment/env/qwen36/bin/python"

INSTANCE_A_PORT=10301
INSTANCE_B_PORT=10302
INSTANCE_A_GPUS="0,1,2,3"
INSTANCE_B_GPUS="4,5,6,7"
TP_SIZE=4

GPU_MEM_UTIL=0.92
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=64

LOG_DIR="$PROJECT_ROOT/logs/vllm"
PID_DIR="$LOG_DIR/pids"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================
# Functions
# ============================================================

start_instance() {
    local name="$1"
    local port="$2"
    local gpus="$3"
    local log_file="$LOG_DIR/${name}.log"
    local pid_file="$PID_DIR/${name}.pid"

    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo -e "  ${GREEN}●${NC} $name already running (PID $(cat "$pid_file"))"
        return 0
    fi

    echo -e "  Starting $name on port $port (GPU: $gpus)..."

    CUDA_VISIBLE_DEVICES="$gpus" "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --trust-remote-code \
        --enable-prefix-caching \
        --disable-log-requests \
        --host 0.0.0.0 \
        --port "$port" \
        >"$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    echo -e "  ${YELLOW}●${NC} $name starting (PID $pid), log: $log_file"
}

wait_for_instance() {
    local name="$1"
    local port="$2"
    local max_wait="${3:-300}"

    echo -n "  Waiting for $name"
    for i in $(seq 1 "$max_wait"); do
        if curl -s "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            echo -e "\n  ${GREEN}●${NC} $name ready! (~${i}s)"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo -e "\n  ${RED}●${NC} $name failed to start within ${max_wait}s"
    return 1
}

stop_all() {
    echo -e "${YELLOW}Stopping vLLM instances...${NC}"
    for pidfile in "$PID_DIR"/*.pid; do
        [ -f "$pidfile" ] || continue
        local pid
        pid=$(cat "$pidfile")
        local name
        name=$(basename "$pidfile" .pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo -e "  ${RED}●${NC} Stopped $name (PID $pid)"
        fi
        rm -f "$pidfile"
    done
    for port in $INSTANCE_A_PORT $INSTANCE_B_PORT; do
        fuser -k "$port/tcp" 2>/dev/null || true
    done
    echo -e "${GREEN}All vLLM instances stopped.${NC}"
}

show_status() {
    echo -e "${CYAN}  vLLM Serving Status${NC}"
    echo ""
    for pidfile in "$PID_DIR"/*.pid; do
        [ -f "$pidfile" ] || continue
        local pid
        pid=$(cat "$pidfile")
        local name
        name=$(basename "$pidfile" .pid)
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} $name (PID $pid) — running"
        else
            echo -e "  ${RED}●${NC} $name (PID $pid) — dead"
        fi
    done
    echo ""
    for port in $INSTANCE_A_PORT $INSTANCE_B_PORT; do
        local resp
        resp=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/v1/models" 2>/dev/null || echo "000")
        if [ "$resp" = "200" ]; then
            echo -e "  ${GREEN}●${NC} Port $port — healthy (HTTP 200)"
        else
            echo -e "  ${RED}●${NC} Port $port — down (HTTP $resp)"
        fi
    done
    echo ""

    # Show GPU utilization
    echo -e "${CYAN}  GPU Utilization:${NC}"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
        while IFS=, read -r idx util mem_used mem_total; do
            echo -e "  GPU $idx: util=${util}, mem=${mem_used}/${mem_total}"
        done
}

# ============================================================
# Main
# ============================================================

case "${1:-start}" in
    --stop|-s)
        stop_all
        exit 0
        ;;
    --status|-t)
        show_status
        exit 0
        ;;
    start|--start)
        ;;
    *)
        echo "Usage: $0 [--start|--stop|--status]"
        exit 1
        ;;
esac

# Validate environment
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Model not found at $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo -e "${RED}ERROR: Python not found at $PYTHON${NC}"
    exit 1
fi

# Create directories
mkdir -p "$LOG_DIR" "$PID_DIR"

echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  vLLM Dual-Instance Serving — 8x A100${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "  Model:     $MODEL_NAME"
echo -e "  TP Size:   $TP_SIZE"
echo -e "  GPU Mem:   ${GPU_MEM_UTIL} utilization"
echo -e "  Max Seqs:  $MAX_NUM_SEQS concurrent"
echo -e "  Max Len:   $MAX_MODEL_LEN tokens"
echo ""

# Stop any existing instances
stop_all 2>/dev/null || true
echo ""

# Start Instance A
echo -e "[1/2] Instance A (GPU ${INSTANCE_A_GPUS}, port ${INSTANCE_A_PORT})"
start_instance "vllm_instance_a" "$INSTANCE_A_PORT" "$INSTANCE_A_GPUS"

# Start Instance B
echo -e "[2/2] Instance B (GPU ${INSTANCE_B_GPUS}, port ${INSTANCE_B_PORT})"
start_instance "vllm_instance_b" "$INSTANCE_B_PORT" "$INSTANCE_B_GPUS"

echo ""
echo -e "${YELLOW}Waiting for instances to load model...${NC}"
echo -e "${YELLOW}(Qwen3.6-27B ~54GB, expect 60-120s per instance)${NC}"
echo ""

# Wait for both instances
wait_for_instance "Instance A" "$INSTANCE_A_PORT" 300
A_OK=$?
wait_for_instance "Instance B" "$INSTANCE_B_PORT" 300
B_OK=$?

echo ""
if [ $A_OK -eq 0 ] && [ $B_OK -eq 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Both vLLM instances ready!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
else
    echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  Partial startup (check logs)${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"
fi

echo ""
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo -e "  Instance A: http://${IP}:${INSTANCE_A_PORT}/v1/models"
echo -e "  Instance B: http://${IP}:${INSTANCE_B_PORT}/v1/models"
echo -e "  Logs:       $LOG_DIR/"
echo ""
