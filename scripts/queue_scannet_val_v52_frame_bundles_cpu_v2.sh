#!/usr/bin/env bash
# Queue source-ready official-val scenes into isolated v2 frame-bundle roots.
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

EXTRACTOR="$W/tools/extract_scannet_val_frame_bundles_v2.py"
SCHEMA="gsa-scannet-official-val-frame-bundle-v2"
PROTOCOL="gsa-scannet-val-frame-bundle-extraction-v2"
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
NAMES=(formal_pool_batch04a_v2 formal_pool_batch04b_v2 formal_pool_batch04c_v2)
LISTS=(
  "$W/configs/scannet_v52_formal_pool_batch04a_v1.txt"
  "$W/configs/scannet_v52_formal_pool_batch04b_v1.txt"
  "$W/configs/scannet_v52_formal_pool_batch04c_v1.txt"
)
ORDINALS=(300000 400000 500000)
QUEUE_DIR="$W/artifacts/cpu_queue/scannet_v52_frame_bundles_20260823_v2"
STATUS="$QUEUE_DIR/status.tsv"
mkdir -p "$QUEUE_DIR"

record() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" \
    | tee -a "$STATUS"
}

verify_v2_output() {
  "$PYTHON_BIN" - "$1" "$EXTRACTOR" "$SCHEMA" "$PROTOCOL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
extractor = Path(sys.argv[2])
schema = sys.argv[3]
protocol = sys.argv[4]
manifest_path = root / "manifest.json"
checksum_path = root / "manifest.sha256"
if not manifest_path.is_file() or not checksum_path.is_file():
    raise SystemExit(2)
payload = manifest_path.read_bytes()
document = json.loads(payload)
expected_manifest_sha = checksum_path.read_text(encoding="ascii").split()[0]
if hashlib.sha256(payload).hexdigest() != expected_manifest_sha:
    raise SystemExit(2)
binding = document.get("extractor", {})
if document.get("schema") != schema or document.get("protocol") != protocol:
    raise SystemExit(2)
if Path(binding.get("extractor_path", "")).resolve() != extractor.resolve():
    raise SystemExit(2)
if binding.get("extractor_sha256") != hashlib.sha256(extractor.read_bytes()).hexdigest():
    raise SystemExit(2)
PY
}

export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
completed_roots=()
record QUEUED queue "three v2 batches; no v1 output roots are writable"
for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  scene_list="${LISTS[$index]}"
  output="$W/data/scannet_val_frame_bundles_v2/$name"
  log="$QUEUE_DIR/$name.log"
  if [[ -d "$output" ]] && verify_v2_output "$output"; then
    record REUSED "$name" "verified_v2_output=$output"
    completed_roots+=("$output")
    continue
  fi
  [[ ! -e "$output" ]] || {
    record FAILED "$name" "unverified or partial output exists: $output"
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
  if "$PYTHON_BIN" "$EXTRACTOR" "${args[@]}" >"$log" 2>&1; then
    if verify_v2_output "$output"; then
      record SUCCEEDED "$name" "verified_v2_output=$output"
      completed_roots+=("$output")
    else
      record FAILED "$name" "v2 manifest self-binding verification failed: $output"
      exit 2
    fi
  else
    rc=$?
    record FAILED "$name" "rc=$rc log=$log"
    exit "$rc"
  fi
done
record COMPLETE queue "batches=${NAMES[*]}"
