#!/bin/bash
# Start/use a local vLLM VLM server, then run the GSA flywheel against it.
# Usage:
#   bash scripts/run_gsa_with_local_vllm.sh --max-images 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

VLLM_ENV="${VLLM_ENV:-/path/to/workspace/foundation/contributor/envs/rex-vllm-local}"
VLLM_MODEL="${VLLM_MODEL:-/path/to/workspace/foundation/contributor/hf_models/Qwen/Qwen2.5-VL-7B-Instruct}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-10301}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$VLLM_MODEL}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
if [ -n "$VLLM_CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES"
fi
VLLM_LOG="${VLLM_LOG:-data/output/gsa_v0/local_vllm_${VLLM_PORT}.log}"
CONFIG="${CONFIG:-scripts/configs/gsa_vllm_local.yaml}"
MAX_IMAGES=""
KEEP_VLLM="${KEEP_VLLM:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --max-images) MAX_IMAGES="$2"; shift 2 ;;
        --stop-vllm-on-exit) KEEP_VLLM=0; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ ! -x "$VLLM_ENV/bin/python" ]; then
    echo "ERROR: VLLM_ENV python not found: $VLLM_ENV/bin/python" >&2
    exit 1
fi
if [ ! -d "$VLLM_MODEL" ]; then
    echo "ERROR: VLLM_MODEL directory not found: $VLLM_MODEL" >&2
    exit 1
fi

export VLLM_ENV VLLM_MODEL VLLM_HOST VLLM_PORT VLLM_BASE_URL VLLM_SERVED_MODEL_NAME VLLM_CUDA_VISIBLE_DEVICES
VLLM_CACHE_HOME="${VLLM_CACHE_HOME:-/path/to/workspace/foundation/contributor/cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$VLLM_CACHE_HOME}"
export HF_HOME="${HF_HOME:-/path/to/workspace/foundation/contributor/hf-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torch_inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PY="${PY:-$VLLM_ENV/bin/python}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$VLLM_LOG")"

health_check() {
    "$VLLM_ENV/bin/python" scripts/check_vllm_health.py \
        --base-url "$VLLM_BASE_URL" \
        --model "$VLLM_SERVED_MODEL_NAME" \
        --timeout 30 >/dev/null 2>&1
}

STARTED_VLLM=0
if health_check; then
    echo "Local vLLM is already healthy at $VLLM_BASE_URL"
else
    echo "Starting local vLLM server..."
    echo "  model: $VLLM_MODEL"
    echo "  url:   $VLLM_BASE_URL"
    echo "  gpu:   ${CUDA_VISIBLE_DEVICES:-all}"
    echo "  tp:    ${VLLM_TENSOR_PARALLEL_SIZE:-auto}"
    echo "  log:   $VLLM_LOG"
    bash scripts/run_local_vllm.sh >"$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    STARTED_VLLM=1

    for i in $(seq 1 120); do
        if health_check; then
            echo "Local vLLM is healthy."
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM server exited during startup. Log tail:" >&2
            tail -80 "$VLLM_LOG" >&2 || true
            exit 1
        fi
        sleep 5
        if [ "$i" -eq 120 ]; then
            echo "ERROR: timed out waiting for local vLLM health check. Log tail:" >&2
            tail -80 "$VLLM_LOG" >&2 || true
            exit 1
        fi
    done
fi

if [ "$STARTED_VLLM" -eq 1 ] && [ "$KEEP_VLLM" -eq 0 ]; then
    trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
fi

ARGS=(--config "$CONFIG")
if [ -n "$MAX_IMAGES" ]; then
    ARGS+=(--max-images "$MAX_IMAGES")
fi

exec bash scripts/run_gsa_v0.sh "${ARGS[@]}"
