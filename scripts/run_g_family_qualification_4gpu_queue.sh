#!/usr/bin/env bash
# Sequential local queue: matched 4xL20 training followed by frozen G qualification eval.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEED="${SEED:-42}"
case "$SEED" in
  42|43|44) ;;
  *) echo "FATAL: qualification seed must be 42, 43, or 44" >&2; exit 2 ;;
esac

TRAIN_OUT="${TRAIN_OUT:-$W/output/g_family_v3_scoped_v2_qualification_lora_4gpu_seed${SEED}}"
EVAL_OUT="${EVAL_OUT:-$W/output/g_family_v3_scoped_v2_qualification_eval_4gpu_seed${SEED}}"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  SEED="$SEED" OUT="$TRAIN_OUT" PREFLIGHT_ONLY=1 \
    bash "$W/scripts/run_g_family_qualification_l20_4gpu_v2.sh"
  echo "G_V2_4GPU_QUEUE_PREFLIGHT_READY: evaluation awaits a completed checkpoint-53"
  exit 0
fi

SEED="$SEED" OUT="$TRAIN_OUT" \
  bash "$W/scripts/run_g_family_qualification_l20_4gpu_v2.sh"

SEED="$SEED" G_SOURCE="$TRAIN_OUT" OUT="$EVAL_OUT" \
  bash "$W/scripts/run_g_family_qualification_eval_l20_4gpu.sh"
