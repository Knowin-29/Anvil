#!/usr/bin/env bash
# Extract all remaining source-ready official-val scenes in three atomic CPU batches.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
[[ -r "$W/.env" ]] || { echo "FATAL: missing $W/.env" >&2; exit 2; }
# shellcheck disable=SC1091
source "$W/.env"

PYTHON_BIN="${PYTHON_BIN:-/path/to/environment/env/gsa_train/bin/python}"
EXPECTED_ENV="/path/to/environment/env/gsa_train"
[[ "$($PYTHON_BIN -c 'import sys; print(sys.prefix)')" == "$EXPECTED_ENV" ]] || {
  echo "FATAL: expected Python environment $EXPECTED_ENV" >&2
  exit 2
}

SCANNET_ROOT="/path/to/datasets/downloads/ScanNet-official/scans"
SPLIT_ROOT="$W/configs/scannet_splits"
SOURCE_EXCLUSIONS="$W/configs/scannet_source_exclusions_v1.json"
OPEN_EQA_ROOT="/path/to/datasets/datasets/transferred_datasets/EmbodiedAI_benchmark/openeqa/scannet-v0"
BASE_ROOTS=(
  "$OPEN_EQA_ROOT"
  "$W/data/scannet_val_frame_bundles/canary_scene0011_v1"
  "$W/data/scannet_val_frame_bundles/formal_pool_batch01_v1"
  "$W/data/scannet_val_frame_bundles/formal_pool_batch02_v1"
  "$W/data/scannet_val_frame_bundles/formal_pool_batch03_v1"
)
NAMES=(formal_pool_batch04a_v1 formal_pool_batch04b_v1 formal_pool_batch04c_v1)
LISTS=(
  "$W/configs/scannet_v52_formal_pool_batch04a_v1.txt"
  "$W/configs/scannet_v52_formal_pool_batch04b_v1.txt"
  "$W/configs/scannet_v52_formal_pool_batch04c_v1.txt"
)
ORDINALS=(300000 400000 500000)
QUEUE_DIR="$W/artifacts/cpu_queue/scannet_v52_frame_bundles_20260823_v1"
STATUS="$QUEUE_DIR/status.tsv"
mkdir -p "$QUEUE_DIR"

record() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" \
    | tee -a "$STATUS"
}

export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
completed_roots=()
record QUEUED queue "three batches, 141 source-ready official-val scenes"
for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  scene_list="${LISTS[$index]}"
  output="$W/data/scannet_val_frame_bundles/$name"
  log="$QUEUE_DIR/$name.log"
  if [[ -s "$output/manifest.json" && -s "$output/manifest.sha256" ]]; then
    record REUSED "$name" "output=$output"
    completed_roots+=("$output")
    continue
  fi
  [[ ! -e "$output" ]] || {
    record FAILED "$name" "noncanonical partial output exists: $output"
    exit 2
  }
  args=(
    --stage batch
    --scene-list "$scene_list"
    --scannet-root "$SCANNET_ROOT"
    --split-root "$SPLIT_ROOT"
    --source-exclusions "$SOURCE_EXCLUSIONS"
    --output-root "$output"
    --max-frames 64
    --start-ordinal "${ORDINALS[$index]}"
  )
  for root in "${BASE_ROOTS[@]}" "${completed_roots[@]}"; do
    args+=(--exclude-frame-root "$root")
  done
  record STARTED "$name" "scenes=$(wc -l < "$scene_list") output=$output"
  if "$PYTHON_BIN" tools/extract_scannet_val_frame_bundles.py "${args[@]}" >"$log" 2>&1; then
    record SUCCEEDED "$name" "output=$output"
    completed_roots+=("$output")
  else
    rc=$?
    record FAILED "$name" "rc=$rc log=$log"
    exit "$rc"
  fi
done
record COMPLETE queue "batches=${NAMES[*]}"
