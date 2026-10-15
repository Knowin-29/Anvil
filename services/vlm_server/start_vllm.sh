#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# vLLM OpenAI-compatible server for Qwen3.6 models
# Supports: Qwen3.6-27B (dense) and Qwen3.6-35B-A3B (MoE, active 3B)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

MODEL_NAME="${1:-qwen3.6-35b-a3b}"  # default to MoE model (faster)
PORT="${2:-10301}"
TP_SIZE="${3:-8}"

MODEL_DIR="/path/to/environment/hf_models/Qwen"

case "${MODEL_NAME}" in
    "qwen3.6-35b-a3b"|"35b-a3b"|"moe")
        MODEL_PATH="${MODEL_DIR}/Qwen3.6-35B-A3B"
        SERVED_NAME="qwen3.6-35b-a3b"
        MAX_MODEL_LEN=4096
        MAX_NUM_SEQS=128  # MoE can handle more concurrent requests
        ;;
    "qwen3.6-27b"|"27b"|"dense")
        MODEL_PATH="${MODEL_DIR}/Qwen3.6-27B"
        SERVED_NAME="qwen3.6-27b"
        MAX_MODEL_LEN=4096
        MAX_NUM_SEQS=64
        ;;
    "qwen2.5-vl-32b"|"32b"|"legacy")
        MODEL_PATH="${MODEL_DIR}/Qwen2.5-VL-32B-Instruct"
        SERVED_NAME="qwen2.5-vl-32b-instruct"
        MAX_MODEL_LEN=4096
        MAX_NUM_SEQS=64
        ;;
    *)
        echo "Unknown model: ${MODEL_NAME}"
        echo "Available: qwen3.6-35b-a3b, qwen3.6-27b, qwen2.5-vl-32b"
        exit 1
        ;;
esac

echo "═══════════════════════════════════════════════════════════"
echo " vLLM Server: ${SERVED_NAME}"
echo " Model Path: ${MODEL_PATH}"
echo " Port: ${PORT}"
echo " Tensor Parallel: ${TP_SIZE} GPUs"
echo " Max Model Len: ${MAX_MODEL_LEN}"
echo " Max Concurrent: ${MAX_NUM_SEQS} (continuous batching)"
echo "═══════════════════════════════════════════════════════════"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    exit 1
fi

exec conda run --no-capture-output -p /path/to/environment/env/dataflywheel python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port ${PORT} \
    --host 0.0.0.0 \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --trust-remote-code \
    --dtype bfloat16 \
    --served-model-name "${SERVED_NAME}" \
    --disable-log-requests
