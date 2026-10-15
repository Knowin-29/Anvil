#!/usr/bin/env bash
# Evaluate the G-family v2 pilot on 4xL20 while preserving eight logical shards.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEED="${SEED:-42}"
case "$SEED" in
  42|43|44) ;;
  *) echo "FATAL: qualification seed must be 42, 43, or 44" >&2; exit 2 ;;
esac

export GSA_EVAL_WORLD_SIZE=4
export G_SOURCE="${G_SOURCE:-$W/output/g_family_v3_scoped_v2_qualification_lora_4gpu_seed${SEED}}"
export OUT="${OUT:-$W/output/g_family_v3_scoped_v2_qualification_eval_4gpu_seed${SEED}}"
export BATCH="${BATCH:-4}"

exec bash "$W/scripts/run_g_family_qualification_eval_l20.sh"
