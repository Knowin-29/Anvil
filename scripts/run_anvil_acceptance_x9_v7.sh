#!/usr/bin/env bash
# Anvil acceptance runbook: expand the frozen gsa-eval-v7 benchmark suite from 5 to 9
# public cells, rerun every table-eligible model under the unified protocol, and
# produce the seed-43 continuation + retention needed for the three-seed interval.
#
# This script LAUNCHES real GPU jobs. It is intentionally guarded: it refuses to start
# while the 8 L20s are still pinned by leaked processes (each card shows ~45.5 GiB held
# by a dead PID). Reclaim the cards first (GPU reset or node reboot on the shared host),
# then run the phase you want:
#
#   bash scripts/run_anvil_acceptance_x9_v7.sh preflight     # GPU + path checks only
#   bash scripts/run_anvil_acceptance_x9_v7.sh smoke         # 8-sample smoke on new sets
#   bash scripts/run_anvil_acceptance_x9_v7.sh evals         # full x9 for every model
#   bash scripts/run_anvil_acceptance_x9_v7.sh seed43        # train + eval + retention
#   bash scripts/run_anvil_acceptance_x9_v7.sh sota          # local embodied adversaries
#
# Nothing here relaxes the v7 contract: THINKING=off, subset=0, batch 16, frozen list.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || { echo "FATAL: $W missing"; exit 1; }
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

# Frozen benchmark lists. OLD = the five already in the paper; NEW = the four public
# sets added in this round (RoboSpatial-Home contributes two cells: yes/no MC + context
# pointing), all self-hosted from local files with no network access.
OLD="cv_bench embspatial refspatial_bench where2place gsa_bench_test"
NEW="erqa blink_relative_depth robospatial_home robospatial_home_context vabench_point"
ALL="$OLD $NEW"

BASE="${MODEL_9B:?set MODEL_9B in .env}"
P4_MERGED="$W/output/merged_anvil_p4_imgdis_seed17_final_v7"

# Local table-eligible models (weights already on disk; no download).
declare -A BASELINES=(
  [qwen3_5_9b_base_x9_v7]="$BASE"
  [qwen3_vl_8b_instruct_x9_v7]="/path/to/environment/models/Qwen3-VL-8B-Instruct"
  [internvl3_5_8b_hf_x9_v7]="/path/to/environment/models/InternVL3_5-8B-HF"
  [llava_ov_qwen2_7b_x9_v7]="/path/to/environment/models/llava-onevision-qwen2-7b-ov-hf"
  [qwen2_5_vl_7b_x9_v7]="/path/to/environment/models/Qwen2.5-VL-7B-Instruct"
  [qwen2_5_vl_3b_x9_v7]="/path/to/environment/models/Qwen2.5-VL-3B-Instruct"
)
declare -A SOTA=(
  [robobrain2_5_8b_nv_local_x9_v7]="/path/to/models/RoboBrain2.5-8B-NV"
  [robobrain2_0_7b_local_x9_v7]="/path/to/datasets/models/RoboBrain2.0-7B"
  [mimo_embodied_7b_local_x9_v7]="/path/to/datasets/models/MiMo-Embodied-7B"
)

gpu_is_clear() {
  # A card is usable only when no compute process holds memory on it. Leaked memory from
  # dead PIDs still shows in compute-apps, so treat any listed app as "not clear".
  local n
  n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  [[ "$n" == "0" ]]
}

require_clear_gpus() {
  if ! gpu_is_clear; then
    echo "FATAL: GPUs still hold memory from other/leaked processes:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
    echo "Reclaim the cards (reset/reboot) before launching real jobs." >&2
    exit 3
  fi
}

phase_preflight() {
  echo "== path checks =="
  [[ -f "$BASE/config.json" ]] && echo "OK base $BASE" || { echo "MISS base"; exit 2; }
  [[ -f "$P4_MERGED/config.json" ]] && echo "OK merged P4" || echo "WARN merged P4 missing (rebuild from adapter first)"
  for tag in "${!BASELINES[@]}"; do
    p="${BASELINES[$tag]}"; [[ -f "$p/config.json" ]] && echo "OK $tag" || echo "MISS $tag :: $p"
  done
  for tag in "${!SOTA[@]}"; do
    p="${SOTA[$tag]}"; [[ -f "$p/config.json" ]] && echo "OK $tag" || echo "MISS $tag :: $p"
  done
  echo "== benchmark registry =="
  python eval/run_benchmark.py --list
  echo "== gpu state =="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  gpu_is_clear && echo "GPUs CLEAR" || echo "GPUs NOT clear (see above)"
}

phase_smoke() {
  require_clear_gpus
  for b in $NEW; do
    echo "== smoke $b =="
    CUDA_VISIBLE_DEVICES=0 python eval/run_benchmark.py --model "$BASE" \
      --benchmark "$b" --limit 8 --batch-size 2 --no-thinking \
      --out "/tmp/smoke_${b}.json"
  done
  python eval/verify_shard_scoring.py
}

phase_evals() {
  require_clear_gpus
  # Base + every open baseline over the full nine-cell suite.
  for tag in "${!BASELINES[@]}"; do
    echo "== eval $tag =="
    CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 THINKING=off CELL_TIMEOUT=6h \
      bash scripts/run_eval_dp.sh "$tag" "${BASELINES[$tag]}" "$ALL" 0 16
  done
  # Anvil P4 seed17: the OLD cells are already registered; only the NEW cells are missing.
  echo "== eval anvil_p4_seed17 (new cells) =="
  CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 THINKING=off CELL_TIMEOUT=6h \
    bash scripts/run_eval_dp.sh anvil_p4_seed17_x9new_v7 "$P4_MERGED" "$NEW" 0 16
}

phase_seed43() {
  require_clear_gpus
  # Third seed for the paired interval. Uses the sweep-winning coordinate convention on
  # the full P4 corpus; evaluate only the final preregistered checkpoint, never a
  # test-selected one.
  CFG=p4_seed43 TRAIN="$W/trains/corpus_source_clean_v1/p4/train.jsonl" \
    VAL="$W/trains/corpus_source_clean_v1/p4/dev.jsonl" \
    LR=7e-5 RANK=64 ALPHA=128 EPOCHS=1 SEED=43 HOURS=30 \
    bash scripts/run_sweep_train_l20_3gpu.sh

  local RUN="$W/output/sweep_v1/p4_seed43"
  local MERGED="$W/output/merged_anvil_p4_seed43_final_v7"
  RUN="$RUN" MERGED_DIR="$MERGED" \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 THINKING=off CELL_TIMEOUT=6h \
    bash scripts/run_eval_adapter_dp.sh anvil_p4_seed43_final_v7 "$RUN" "$ALL" 0 16

  # Paired general-capability retention against the base (fills the P4 retention gap).
  CUDA_VISIBLE_DEVICES=0,1,2,3 BASE_MODEL="$BASE" E1_MODEL="$MERGED" \
    OUT="$W/output/general_retention_formal_v3_base_vs_anvil_p4_seed43" \
    RUN_STAGE=paired RESUME=0 THINKING=off CELL_TIMEOUT=8h \
    bash scripts/run_general_retention_dp.sh

  echo "== three-seed paired comparison (seed17 / seed23 / seed43) =="
  echo "run: python eval/compare_models.py  (see --help for the paired-interval flags)"
}

phase_sota() {
  require_clear_gpus
  for tag in "${!SOTA[@]}"; do
    p="${SOTA[$tag]}"
    echo "== smoke $tag chat-template =="
    CUDA_VISIBLE_DEVICES=0 python eval/run_benchmark.py --model "$p" \
      --benchmark cv_bench --limit 8 --batch-size 2 --no-thinking \
      --out "/tmp/smoke_${tag}.json"
    echo "== eval $tag =="
    CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 THINKING=off CELL_TIMEOUT=6h \
      bash scripts/run_eval_dp.sh "$tag" "$p" "$ALL" 0 16
  done
}

case "${1:-preflight}" in
  preflight) phase_preflight ;;
  smoke)     phase_smoke ;;
  evals)     phase_evals ;;
  seed43)    phase_seed43 ;;
  sota)      phase_sota ;;
  *) echo "usage: $0 {preflight|smoke|evals|seed43|sota}"; exit 2 ;;
esac
