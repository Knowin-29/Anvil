#!/usr/bin/env bash
# run_sweep_eval_l20_3gpu.sh — merge one sweep adapter and score the three fast cells.
#
#   CFG=<name> bash scripts/run_sweep_eval_l20_3gpu.sh
#   MODEL_DIR=<merged-or-base> TAG=<tag> bash scripts/run_sweep_eval_l20_3gpu.sh
#
# Cells: refspatial_bench (277, of which the 100-row `location` split is the headline),
# cv_bench (2638) and gsa_bench_test (1463). Protocol is the frozen one: schema 7,
# gsa-eval-v7, seed 0, thinking off, batch 16 — identical to the recorded base and
# G-only-seed42 runs, so the deltas are comparable to them without a re-run.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || exit 1
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs/sweep"

BENCHES="${BENCHES:-refspatial_bench cv_bench gsa_bench_test}"
BATCH="${BATCH:-16}"
GPUS="${GPUS:-5,6,7}"
SUBSET="${SUBSET:-0}"

if [[ -n "${MODEL_DIR:-}" ]]; then
  TAG="${TAG:?set TAG with MODEL_DIR}"
  MERGED="$MODEL_DIR"
else
  CFG="${CFG:?set CFG or MODEL_DIR}"
  TAG="${TAG:-sweep_$CFG}"
  SRC="${SRC:-$W/output/sweep_v1/$CFG}"
  ADAPTER=$(find "$SRC" -type f -name adapter_config.json -printf '%h\n' 2>/dev/null \
    | sed 's|.*/checkpoint-\([0-9]\{1,\}\)$|\1 &|' | sort -n -k1 \
    | awk 'NF==2' | tail -1 | cut -d' ' -f2)
  [[ -n "${ADAPTER:-}" ]] || ADAPTER=$(find "$SRC" -type f -name adapter_config.json \
    -printf '%h\n' 2>/dev/null | tail -1)
  [[ -n "${ADAPTER:-}" && -f "$ADAPTER/adapter_config.json" ]] || {
    echo "FATAL: no LoRA adapter under $SRC" >&2; exit 2; }
  # Merged weights are 19 GiB each. They go to /path/to/workspace/demo (18T free), never to the
  # container overlay root, which has ~112G total.
  MERGED="${MERGED_DIR:-$W/output/sweep_v1/_merged/$CFG}"
  if [[ ! -f "$MERGED/config.json" ]]; then
    echo "[eval] merging $ADAPTER -> $MERGED"
    CUDA_VISIBLE_DEVICES="" swift export --adapters "$ADAPTER" --merge_lora true \
      --output_dir "$MERGED" > "$W/logs/sweep/merge_${CFG}.log" 2>&1 || {
        echo "FATAL: merge failed; see logs/sweep/merge_${CFG}.log" >&2
        tail -20 "$W/logs/sweep/merge_${CFG}.log" >&2; exit 3; }
  fi
  [[ -f "$MERGED/config.json" ]] || { echo "FATAL: no merged config.json" >&2; exit 3; }
fi

echo "[eval] tag=$TAG model=$MERGED benches='$BENCHES' gpus=$GPUS"
CUDA_VISIBLE_DEVICES="$GPUS" NUM_GPUS=3 THINKING=off CELL_TIMEOUT="${CELL_TIMEOUT:-4h}" \
  bash "$W/scripts/run_eval_dp.sh" "$TAG" "$MERGED" "$BENCHES" "$SUBSET" "$BATCH"
RC=$?

# Land the row immediately: an in-memory-only result is lost when the shell dies, and
# the sweep table is rebuilt from these files.
python "$W/tools/collect_sweep_results.py" --tag "$TAG" || true
echo "[eval] $TAG rc=$RC"
exit "$RC"
