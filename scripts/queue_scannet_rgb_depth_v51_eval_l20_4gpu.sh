#!/usr/bin/env bash
# Queue the formal40 paired G -> S evaluation until all four local L20s are idle.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"

MODEL="${MODEL:-/path/to/environment/hf_models/Qwen/Qwen3.5-9B}"
OUT="${OUT:-$W/output/scannet_rgb_depth_v51_formal40_qwen35_9b_20260823_v1}"
QUEUE_DIR="${QUEUE_DIR:-$W/artifacts/local4_queue/scannet_rgb_depth_v51_qwen35_v1}"
STATUS="$QUEUE_DIR/status.tsv"
RUN_LOG="$QUEUE_DIR/run.log"
LOCK="$W/artifacts/local4_queue/queue.lock"

mkdir -p "$QUEUE_DIR"
exec 9>"$LOCK"
flock -n 9 || {
  echo "FATAL: another project 4-GPU queue holds $LOCK" >&2
  exit 3
}

record() {
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" | tee -a "$STATUS"
}

four_l20s_present() {
  mapfile -t rows < <(
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null
  )
  [[ "${#rows[@]}" == 4 ]] || return 1
  local row name memory
  for row in "${rows[@]}"; do
    name=${row%,*}
    memory=${row##*,}
    memory=${memory// /}
    [[ "${name^^}" == *L20* && "$memory" -ge 45056 ]] || return 1
  done
}

gpus_idle() {
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF')" ]]
}

record QUEUED "waiting for exactly four idle physical L20 GPUs"
while true; do
  if four_l20s_present && gpus_idle; then
    sleep 10
    if four_l20s_present && gpus_idle; then
      break
    fi
  fi
  sleep 30
done

record STARTING "model=$MODEL out=$OUT"
set +e
THINKING=off MODEL="$MODEL" OUT="$OUT" \
  bash "$W/scripts/run_scannet_rgb_depth_v51_eval_l20_4gpu.sh" \
  >>"$RUN_LOG" 2>&1
rc=$?
set -e

if [[ "$rc" == 0 ]]; then
  record SUCCEEDED "result=$OUT/result.json"
else
  record FAILED "rc=$rc log=$RUN_LOG"
fi
exit "$rc"
