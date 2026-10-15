#!/usr/bin/env bash
# Launch ONE transformers-backed OpenAI-compatible Qwen3.6-27B server, sharded
# across two GPUs (device_map=auto over the two visible cards ~= TP=2 placement).
# vLLM 0.8.5 and sglang 0.5.12 both lack working qwen3_5 support on this box, so
# this transformers 5.5.0 path is the reliable fallback.
#
# Usage: serve_tf_qwen36_27b.sh "<GPU_A> <GPU_B>" <PORT>
#   e.g. serve_tf_qwen36_27b.sh "1 2" 30000
set -euo pipefail

GPUS="${1:-1 2}"
PORT="${2:-30000}"
GPU_CSV="$(echo "$GPUS" | tr ' ' ',')"

PYBIN=/path/to/environment/env/gsa_train/bin/python
MODEL=/path/to/environment/hf_models/Qwen/Qwen3.6-27B
CACHE=/path/to/environment/.cache
REPO=/path/to/Anvil
LOGDIR="$REPO/logs/vlm_serve"
PIDFILE="$LOGDIR/tf_qwen36.pids"
mkdir -p "$LOGDIR"

unset PYTHONPATH || true
export CUDA_VISIBLE_DEVICES="$GPU_CSV"
export HF_HOME="$CACHE/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor"
export TRITON_CACHE_DIR="$CACHE/triton"
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=PIX

export TF_MODEL="$MODEL"
export TF_SERVED_NAME=Qwen3.6-27B
export TF_HOST=127.0.0.1
export TF_PORT="$PORT"
export TF_MAX_NEW_TOKENS=1024

LOG="$LOGDIR/tf_qwen36_gpu${GPU_CSV//,/_}_p${PORT}.log"
echo "[tf] launching Qwen3.6-27B on GPUs=$GPU_CSV port=$PORT -> $LOG"
setsid "$PYBIN" "$REPO/scripts/tf_openai_server.py" >> "$LOG" 2>&1 < /dev/null &
echo "$! GPUS=$GPU_CSV PORT=$PORT" >> "$PIDFILE"
echo "[tf] started pid $! ; pidfile $PIDFILE"
echo "[tf] readiness: grep -m1 'serving' '$LOG' ; then curl 127.0.0.1:$PORT/health"
