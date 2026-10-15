#!/usr/bin/env bash
# Queue source-ready official-val scenes into isolated pose-valid v3 roots.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
[[ -r "$W/.env" ]] || { echo "FATAL: missing $W/.env" >&2; exit 2; }
# shellcheck disable=SC1091
source "$W/.env"

PYTHON_BIN="${PYTHON_BIN:-/path/to/environment/env/gsa_train/bin/python}"
EXPECTED_ENV="/path/to/environment/env/gsa_train"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || {
  echo "FATAL: PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 2
}
[[ "$($PYTHON_BIN -c 'import sys; print(sys.prefix)')" == "$EXPECTED_ENV" ]] || {
  echo "FATAL: expected Python environment $EXPECTED_ENV" >&2
  exit 2
}

QUEUE_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
EXTRACTOR="$W/tools/extract_scannet_val_frame_bundles_v3.py"
LAUNCHER="$W/scripts/run_scannet_val_frame_bundles_v3.sh"
SCHEMA="gsa-scannet-official-val-frame-bundle-v3"
PROTOCOL="gsa-scannet-val-frame-bundle-extraction-v3"
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
NAMES=(formal_pool_batch04a_v3 formal_pool_batch04b_v3 formal_pool_batch04c_v3)
LISTS=(
  "$W/configs/scannet_v52_formal_pool_batch04a_v1.txt"
  "$W/configs/scannet_v52_formal_pool_batch04b_v1.txt"
  "$W/configs/scannet_v52_formal_pool_batch04c_v1.txt"
)
SPLIT_FILES=(
  "$SPLIT_ROOT/scannetv2_train.txt"
  "$SPLIT_ROOT/scannetv2_val.txt"
  "$SPLIT_ROOT/scannetv2_test.txt"
)
ORDINALS=(300000 400000 500000)
QUEUE_DIR="$W/artifacts/cpu_queue/scannet_v52_frame_bundles_20260823_v3"
STATUS="$QUEUE_DIR/status.tsv"
BINDING="$QUEUE_DIR/execution_binding.json"
mkdir -p "$QUEUE_DIR"

for path in "$QUEUE_SCRIPT" "$EXTRACTOR" "$LAUNCHER" "$SOURCE_EXCLUSIONS" "${LISTS[@]}" "${SPLIT_FILES[@]}"; do
  [[ -s "$path" ]] || { echo "FATAL: missing bound input: $path" >&2; exit 2; }
done

write_or_verify_binding() {
  "$PYTHON_BIN" - "$BINDING" "$QUEUE_SCRIPT" "$EXTRACTOR" "$LAUNCHER" "$SCHEMA" "$PROTOCOL" "$SOURCE_EXCLUSIONS" "${LISTS[@]}" "${SPLIT_FILES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

target = Path(sys.argv[1])
queue, extractor, launcher = map(Path, sys.argv[2:5])
schema, protocol = sys.argv[5:7]
source_exclusions = Path(sys.argv[7])
scene_lists = [Path(value) for value in sys.argv[8:11]]
split_manifests = [Path(value) for value in sys.argv[11:]]

def document(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }

binding = {
    "schema": "gsa-scannet-v3-cpu-queue-binding-v1",
    "extractor_schema": schema,
    "extractor_protocol": protocol,
    "queue": document(queue),
    "extractor": document(extractor),
    "launcher": document(launcher),
    "source_exclusions": document(source_exclusions),
    "scene_lists": [document(path) for path in scene_lists],
    "split_manifests": [document(path) for path in split_manifests],
}
payload = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8")
if target.exists():
    if target.read_bytes() != payload:
        raise SystemExit("FATAL: queue execution binding no longer matches inputs")
else:
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
PY
}

record() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" \
    | tee -a "$STATUS"
}

verify_v3_output() {
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
if document.get("output_contract", {}).get("namespace") != "scannet_val_frame_bundles_v3":
    raise SystemExit(2)
if Path(binding.get("extractor_path", "")).resolve() != extractor.resolve():
    raise SystemExit(2)
if binding.get("extractor_sha256") != hashlib.sha256(extractor.read_bytes()).hexdigest():
    raise SystemExit(2)
for dependency in binding.get("implementation_dependencies", []):
    path = Path(dependency.get("path", ""))
    if not path.is_file():
        raise SystemExit(2)
    if dependency.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise SystemExit(2)
integrity = document.pop("integrity", None)
if not isinstance(integrity, dict):
    raise SystemExit(2)
canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
if integrity.get("manifest_payload_sha256") != hashlib.sha256(canonical).hexdigest():
    raise SystemExit(2)
PY
}

write_or_verify_binding
QUEUE_SHA256="$(sha256sum "$QUEUE_SCRIPT" | awk '{print $1}')"
EXTRACTOR_SHA256="$(sha256sum "$EXTRACTOR" | awk '{print $1}')"
LAUNCHER_SHA256="$(sha256sum "$LAUNCHER" | awk '{print $1}')"
record BOUND queue "queue_sha256=$QUEUE_SHA256 extractor_sha256=$EXTRACTOR_SHA256 launcher_sha256=$LAUNCHER_SHA256 binding=$BINDING"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  record PREFLIGHT_PASS queue "no scenes inspected; no dataset outputs written"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
completed_roots=()
record QUEUED queue "three v3 batches; v1/v2 output roots are not writable"
for index in "${!NAMES[@]}"; do
  write_or_verify_binding
  name="${NAMES[$index]}"
  scene_list="${LISTS[$index]}"
  output="$W/data/scannet_val_frame_bundles_v3/$name"
  log="$QUEUE_DIR/$name.log"
  if [[ -d "$output" ]] && verify_v3_output "$output"; then
    record REUSED "$name" "verified_v3_output=$output"
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
  if nice -n 10 "$PYTHON_BIN" "$EXTRACTOR" "${args[@]}" >"$log" 2>&1; then
    write_or_verify_binding
    if verify_v3_output "$output"; then
      record SUCCEEDED "$name" "verified_v3_output=$output"
      completed_roots+=("$output")
    else
      record FAILED "$name" "v3 manifest self-binding verification failed: $output"
      exit 2
    fi
  else
    rc=$?
    record FAILED "$name" "rc=$rc log=$log"
    exit "$rc"
  fi
done
write_or_verify_binding
record COMPLETE queue "batches=${NAMES[*]}"
