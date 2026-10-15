#!/usr/bin/env bash
# Fan out generate_multiform.py across N local vLLM replicas with zero router:
# each shard process gets its own index slice (--offset/--limit) and its own
# GSA_VLM_URL (one replica). This is the most robust load balancing for the
# flywheel because vlm_client reads a single URL from the environment.
#
# Usage: run_flywheel_sharded.sh <INDEX> <OUT_ROOT> <BASE_PORT> <N_REPLICAS> <TARGET_TOTAL> [WORKERS_PER_SHARD]
set -euo pipefail

INDEX="${1:-configs/data/embodied_images.index}"
OUT_ROOT="${2:-data/flywheel_local}"
BASE_PORT="${3:-30000}"
N="${4:-7}"
TARGET_TOTAL="${5:-7000000}"
WORKERS="${6:-48}"

REPO="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
# shellcheck disable=SC1091
source "$REPO/.env" 2>/dev/null || true
PY="${GSA_PYTHON_BIN:-python}"

TOTAL_IMAGES=$(grep -c . "$INDEX")
PER=$(( (TOTAL_IMAGES + N - 1) / N ))
PER_TARGET=$(( (TARGET_TOTAL + N - 1) / N ))
mkdir -p logs/flywheel

echo "[shard] images=$TOTAL_IMAGES replicas=$N per_shard_images=$PER per_shard_target=$PER_TARGET"
for i in $(seq 0 $((N - 1))); do
  port=$((BASE_PORT + i))
  off=$((i * PER))
  log="logs/flywheel/shard${i}_p${port}.log"
  echo "[shard] shard$i offset=$off limit=$PER url=127.0.0.1:$port -> $log"
  GSA_VLM_URL="http://127.0.0.1:${port}/v1/chat/completions" \
  GSA_VLM_MODEL="Qwen2.5-VL-7B-Instruct" \
  GSA_VLM_KEY="dummy" \
  GSA_VLM_CACHE="${REPO}/logs/vlm_cache_shard${i}" \
  setsid "$PY" trains/flywheel/generate_multiform.py \
    --index "$INDEX" \
    --out "${OUT_ROOT}/shard${i}" \
    --offset "$off" --limit "$PER" \
    --workers "$WORKERS" --target "$PER_TARGET" --resume \
    >> "$log" 2>&1 < /dev/null &
  echo "$! shard${i} $port" >> logs/flywheel/shards.pids
done
echo "[shard] launched $N shards; pids in logs/flywheel/shards.pids"
echo "[shard] monitor: tail -f logs/flywheel/shard0_p${BASE_PORT}.log"
