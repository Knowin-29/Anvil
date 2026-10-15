#!/usr/bin/env bash
# Retry batch05 without the single source that failed the frozen v3 IMU contract.
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
[[ "$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')" == "$EXPECTED_ENV" ]] || {
  echo "FATAL: expected Python environment $EXPECTED_ENV" >&2
  exit 2
}

QUEUE_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
EXTRACTOR="$W/tools/extract_scannet_val_frame_bundles_v3.py"
SCENE_LIST="$W/configs/scannet_v52_formal_pool_batch05b_v1.txt"
PARENT_BINDING="$W/artifacts/cpu_queue/scannet_v52_frame_bundles_20260824_batch05_v3/execution_binding.json"
PARENT_BINDING_SHA256="3dc8eea588fe8a2053fa02c5e2a5e29abeb7170dd796286821c3a00a73f77e13"
SCANNET_ROOT="/path/to/datasets/downloads/ScanNet-official/scans"
SPLIT_ROOT="$W/configs/scannet_splits"
SOURCE_EXCLUSIONS="$W/configs/scannet_source_exclusions_v1.json"
V3_ROOTS=(
  "$W/data/scannet_val_frame_bundles_v3/formal_pool_batch04a_v3"
  "$W/data/scannet_val_frame_bundles_v3/formal_pool_batch04b_v3"
  "$W/data/scannet_val_frame_bundles_v3/formal_pool_batch04c_v3"
)
OUTPUT="$W/data/scannet_val_frame_bundles_v3/formal_pool_batch05b_v3"
QUEUE_DIR="$W/artifacts/cpu_queue/scannet_v52_frame_bundles_20260824_batch05b_v3"
STATUS="$QUEUE_DIR/status.tsv"
BINDING="$QUEUE_DIR/execution_binding.json"
LOG="$QUEUE_DIR/formal_pool_batch05b_v3.log"
mkdir -p "$QUEUE_DIR"

for path in "$QUEUE_SCRIPT" "$EXTRACTOR" "$SCENE_LIST" "$PARENT_BINDING" \
  "$SOURCE_EXCLUSIONS"; do
  [[ -s "$path" ]] || { echo "FATAL: missing bound input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$PARENT_BINDING" | awk '{print $1}')" == "$PARENT_BINDING_SHA256" ]] || {
  echo "FATAL: failed batch05 parent binding drift" >&2
  exit 2
}

write_or_verify_binding() {
  "$PYTHON_BIN" - "$BINDING" "$QUEUE_SCRIPT" "$EXTRACTOR" "$SCENE_LIST" \
    "$PARENT_BINDING" "$PARENT_BINDING_SHA256" "$SCANNET_ROOT" \
    "${V3_ROOTS[@]}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from tools import build_scannet_rgb_depth_v52_unique_referent_formal as v52

target = Path(sys.argv[1])
queue = Path(sys.argv[2])
extractor = Path(sys.argv[3])
scene_list = Path(sys.argv[4])
parent_path = Path(sys.argv[5])
parent_sha256 = sys.argv[6]
scannet_root = Path(sys.argv[7])
v3_roots = [Path(value) for value in sys.argv[8:]]

def document(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }

parent_bytes = parent_path.read_bytes()
if hashlib.sha256(parent_bytes).hexdigest() != parent_sha256:
    raise SystemExit("FATAL: failed batch05 parent binding drift")
parent = json.loads(parent_bytes)
if parent.get("schema") != "gsa-scannet-v52-batch05-v3-cpu-queue-binding-v1":
    raise SystemExit("FATAL: failed batch05 parent schema drift")

scene_ids = scene_list.read_text().split()
parent_scenes = set(parent.get("scene_ids") or [])
if (
    len(scene_ids) != 10
    or len(scene_ids) != len(set(scene_ids))
    or scene_ids != sorted(scene_ids)
    or not set(scene_ids) < parent_scenes
    or parent_scenes - set(scene_ids) != {"scene0549_00"}
):
    raise SystemExit("FATAL: batch05b must be the exact parent list minus scene0549_00")

v51_files = parent["v51_formal_binding"]
v51 = v52.load_v51_reuse_binding(
    Path(v51_files["manifest"]["path"]).parent,
    Path(v51_files["report"]["path"]),
)
prior = v52._prior_scene_ids(v51)
if set(scene_ids) & prior:
    raise SystemExit("FATAL: batch05b overlaps the frozen prior-audit scene set")

v3_bindings = [v52.load_v3_frame_root(root, verify_artifacts=False) for root in v3_roots]
v3_scenes = {scene for binding in v3_bindings for scene in binding["scene_ids"]}
if set(scene_ids) & v3_scenes:
    raise SystemExit("FATAL: batch05b overlaps an existing v3 scene")

for scene_id in scene_ids:
    scene_root = scannet_root / scene_id
    required = (
        scene_root / f"{scene_id}.sens",
        scene_root / f"{scene_id}.txt",
        scene_root / f"{scene_id}.aggregation.json",
        scene_root / f"{scene_id}_2d-instance-filt.zip",
        scene_root / f"{scene_id}_2d-label-filt.zip",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise SystemExit(f"FATAL: batch05b source is incomplete: {scene_id}")

binding = {
    "schema": "gsa-scannet-v52-batch05b-v3-cpu-queue-binding-v1",
    "purpose": "fail-closed retry after excluding the one invalid IMU-count source",
    "training_materialization_eligible": False,
    "queue": document(queue),
    "extractor": document(extractor),
    "scene_list": document(scene_list),
    "scene_ids": scene_ids,
    "scene_ids_sha256": v52.canonical_hash(scene_ids),
    "parent_failed_round": document(parent_path),
    "excluded_source": {
        "scene_id": "scene0549_00",
        "reason": "metadata_numIMUmeasurements_934_but_sens_fixed_width_records_933",
    },
    "prior_audit_scene_count": len(prior),
    "prior_audit_scene_ids_sha256": v52.canonical_hash(sorted(prior)),
    "v3_exclusion_roots": v3_bindings,
    "selection_policy": parent["selection_policy"],
}
payload = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8")
if target.exists():
    if target.read_bytes() != payload:
        raise SystemExit("FATAL: batch05b execution binding no longer matches inputs")
else:
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
PY
}

verify_output() {
  "$PYTHON_BIN" - "$1" "$SCENE_LIST" <<'PY'
import sys
from pathlib import Path
from tools import build_scannet_rgb_depth_v52_unique_referent_formal as v52

root = Path(sys.argv[1])
expected = sorted(Path(sys.argv[2]).read_text().split())
binding = v52.load_v3_frame_root(root, verify_artifacts=True)
if binding["scene_ids"] != expected or binding["scene_count"] != len(expected):
    raise SystemExit("FATAL: batch05b output scene identity drift")
PY
}

record() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" \
    | tee -a "$STATUS"
}

write_or_verify_binding
record BOUND batch05b "binding=$BINDING scene_list_sha256=$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  record PREFLIGHT_PASS batch05b "10 official-val scenes; no dataset output written"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -d "$OUTPUT" ]] && verify_output "$OUTPUT"; then
  record REUSED batch05b "verified_v3_output=$OUTPUT"
  exit 0
fi
[[ ! -e "$OUTPUT" ]] || {
  record FAILED batch05b "unverified or partial output exists: $OUTPUT"
  exit 2
}

args=(
  --stage batch
  --scene-list "$SCENE_LIST"
  --scannet-root "$SCANNET_ROOT"
  --split-root "$SPLIT_ROOT"
  --source-exclusions "$SOURCE_EXCLUSIONS"
  --output-root "$OUTPUT"
  --max-frames 64
  --start-ordinal 610000
)
for root in "${V3_ROOTS[@]}"; do
  args+=(--exclude-frame-root "$root")
done
record STARTED batch05b "scenes=10 output=$OUTPUT cpu=only network=disabled"
if nice -n 10 "$PYTHON_BIN" "$EXTRACTOR" "${args[@]}" >"$LOG" 2>&1; then
  write_or_verify_binding
  if verify_output "$OUTPUT"; then
    record SUCCEEDED batch05b "verified_v3_output=$OUTPUT log=$LOG"
  else
    record FAILED batch05b "v3 output verification failed: $OUTPUT"
    exit 2
  fi
else
  rc=$?
  record FAILED batch05b "rc=$rc log=$LOG"
  exit "$rc"
fi
