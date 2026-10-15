#!/usr/bin/env bash
# Launch one vLLM OpenAI-compatible Qwen2.5-VL-7B server on a single GPU.
# Usage: serve_vllm_qwen25vl.sh <GPU_ID> <PORT>
set -euo pipefail

GPU="${1:-0}"
PORT="${2:-30000}"

VENV=/path/to/environment/.cache/vllm_venv
MODEL=/path/to/environment/models/Qwen2.5-VL-7B-Instruct-patched
CACHE=/path/to/environment/.cache
LOGDIR=/path/to/Anvil/logs/vlm_serve
mkdir -p "$LOGDIR"
LOG="$LOGDIR/vllm_gpu${GPU}_p${PORT}.log"

unset PYTHONPATH || true
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$CACHE/hf"
export VLLM_CACHE_ROOT="$CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor"
export TRITON_CACHE_DIR="$CACHE/triton"
export OUTLINES_CACHE_DIR="$CACHE/outlines"

exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name Qwen2.5-VL-7B-Instruct \
  --port "$PORT" --host 127.0.0.1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --limit-mm-per-prompt image=1 \
  >> "$LOG" 2>&1
