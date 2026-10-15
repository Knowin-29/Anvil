#!/usr/bin/env bash
# Start N single-GPU Qwen2.5-VL-7B replicas as background daemons, one per free GPU.
# Avoids GPU4 (external tenant). Each replica is an independent OpenAI-compatible
# endpoint on 127.0.0.1:<BASE_PORT+i>. Use scripts/stop_vllm_fleet.sh to stop.
#
# Usage: serve_vllm_fleet.sh ["0 1 2 3 5 6 7"] [BASE_PORT]
set -euo pipefail

GPUS="${1:-0 1 2 3 5 6 7}"     # free GPUs (GPU4 excluded)
BASE_PORT="${2:-30000}"

VENV=/path/to/environment/.cache/vllm_venv
MODEL=/path/to/environment/models/Qwen2.5-VL-7B-Instruct-patched
CACHE=/path/to/environment/.cache
LOGDIR=/path/to/Anvil/logs/vlm_serve
PIDFILE=/path/to/Anvil/logs/vlm_serve/fleet.pids
mkdir -p "$LOGDIR"
: > "$PIDFILE"

unset PYTHONPATH || true
export HF_HOME="$CACHE/hf"
export VLLM_CACHE_ROOT="$CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor"
export TRITON_CACHE_DIR="$CACHE/triton"
export OUTLINES_CACHE_DIR="$CACHE/outlines"

i=0
for g in $GPUS; do
  port=$((BASE_PORT + i))
  log="$LOGDIR/vllm_gpu${g}_p${port}.log"
  echo "[fleet] launching replica on GPU${g} port${port} -> $log"
  CUDA_VISIBLE_DEVICES="$g" setsid "$VENV/bin/python" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name Qwen2.5-VL-7B-Instruct \
    --port "$port" --host 127.0.0.1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --limit-mm-per-prompt image=1 \
    >> "$log" 2>&1 < /dev/null &
  echo "$! GPU${g} $port" >> "$PIDFILE"
  i=$((i + 1))
done
echo "[fleet] started $i replicas; pids in $PIDFILE"
echo "[fleet] wait for readiness: grep -l 'Application startup complete' $LOGDIR/vllm_gpu*_p*.log"
