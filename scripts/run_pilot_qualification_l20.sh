#!/usr/bin/env bash
# Run exactly one source-clean qualification stage after semantic audit.
#
#   STAGE=joint bash scripts/run_pilot_qualification_l20.sh
#   STAGE=G     bash scripts/run_pilot_qualification_l20.sh
#   STAGE=S     bash scripts/run_pilot_qualification_l20.sh
#   STAGE=A     bash scripts/run_pilot_qualification_l20.sh
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

STAGE="${STAGE:?set STAGE=joint|G|S|A}"
case "$STAGE" in joint|G|S|A) ;; *) echo "FATAL: STAGE must be joint, G, S, or A" >&2; exit 2 ;; esac

AUDIT="${SEMANTIC_AUDIT:-$W/artifacts/audits/pilot_semantic_audit_composite.json}"
BENCH_DIR="${BENCH_DIR:-$W/trains/gsa_bench_v4_composite_pilot_clean}"
V3_PACK="${V3_AUDIT_PACK:-$W/data/semantic_audit_v3}"
V3_ADJ="${V3_ADJUDICATION:-$W/data/semantic_audit_v3/adjudication.jsonl}"
V4_PACK="${V4_AUDIT_PACK:-$W/data/semantic_audit_v4_depth}"
V4_ADJ="${V4_ADJUDICATION:-$W/data/semantic_audit_v4_depth/adjudication.jsonl}"

# This is intentionally composite-only. A historical global v3 PASS cannot
# substitute for the failed v3 depth row or release any qualification training.
python "$W/scripts/verify_composite_qualification.py" \
  --audit "$AUDIT" \
  --manifest "$BENCH_DIR/manifest.json" \
  --v3-pack "$V3_PACK" --v3-adjudication "$V3_ADJ" \
  --v4-pack "$V4_PACK" --v4-adjudication "$V4_ADJ"

mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF' | sort -u)
if [[ "${#GPU_PIDS[@]}" -gt 0 && "${ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "FATAL: visible GPUs already have compute processes: ${GPU_PIDS[*]}" >&2
  echo "Wait for the allocation to become idle; do not kill unrelated jobs." >&2
  exit 3
fi

NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
[[ "$NG" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: torch sees no visible GPUs" >&2; exit 2;
}
BS="${BS:-1}"
GA="${GA:-1}"
for value in "$BS" "$GA"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: BS and GA must be positive integers" >&2; exit 2;
  }
done
GLOBAL_BATCH=$(( NG * BS * GA ))

if [[ "$STAGE" == "joint" ]]; then
  ROWS=$(wc -l < "$BENCH_DIR/train.jsonl")
  EPOCHS="${EPOCHS:-3}"
  [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: EPOCHS must be a positive integer" >&2; exit 2;
  }
  STEPS="${STEPS:-$(( (ROWS * EPOCHS + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))}"
  EVAL_STEPS="${EVAL_STEPS:-$STEPS}"
  SAVE_STEPS="${SAVE_STEPS:-$STEPS}"
  TRAIN="$BENCH_DIR/train.jsonl" VAL="$BENCH_DIR/val.jsonl" \
    OUT="${OUT:-$W/output/pilot_clean_joint}" LR="${LR:-1e-5}" STEPS="$STEPS" BS="$BS" GA="$GA" \
    EVAL_STEPS="$EVAL_STEPS" SAVE_STEPS="$SAVE_STEPS" \
    bash "$W/scripts/run_sft_9b_lora_l20_dlc.sh"
else
  ROWS=$(wc -l < "$BENCH_DIR/train_$STAGE.jsonl")
  case "$STAGE" in
    G) DEFAULT_EPOCHS=3 ;;
    S|A) DEFAULT_EPOCHS=5 ;;
  esac
  EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
  [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: EPOCHS must be a positive integer" >&2; exit 2;
  }
  STEPS="${STEPS:-$(( (ROWS * EPOCHS + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))}"
  MIN_STEPS="${MIN_STEPS:-1}"
  [[ "$MIN_STEPS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: MIN_STEPS must be a positive integer" >&2; exit 2;
  }
  [[ "$STEPS" -lt "$MIN_STEPS" ]] && STEPS="$MIN_STEPS"
  EVAL_STEPS="${EVAL_STEPS:-$STEPS}"
  SAVE_STEPS="${SAVE_STEPS:-$STEPS}"
  BENCH_DIR="$BENCH_DIR" DIM="$STAGE" TUNER=lora EPOCHS="$EPOCHS" MIN_STEPS="$MIN_STEPS" DS="${DS:-zero2}" \
    OUT="${OUT:-$W/output/pilot_clean_teacher_$STAGE}" STEPS="$STEPS" BS="$BS" GA="$GA" \
    EVAL_STEPS="$EVAL_STEPS" SAVE_STEPS="$SAVE_STEPS" \
    bash "$W/scripts/run_teacher_dim_full_dlc.sh"
fi
