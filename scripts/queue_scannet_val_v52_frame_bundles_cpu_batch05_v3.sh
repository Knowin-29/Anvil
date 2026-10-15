#!/usr/bin/env bash
# Re-extract legacy-v1-only official-val scenes under the frozen pose-valid v3 contract.
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
SCENE_LIST="$W/configs/scannet_v52_formal_pool_batch05_v1.txt"
V51_PACK="$W/data/semantic_audit_v51_scannet_rgb_auditable_depth_formal"
V51_REPORT="$W/artifacts/audits/scannet_rgb_depth_v51_formal.json"
SCANNET_ROOT="/path/to/datasets/downloads/ScanNet-official/scans"
SPLIT_ROOT="$W/configs/scannet_splits"
SOURCE_EXCLUSIONS="$W/configs/scannet_source_exclusions_v1.json"
V3_ROOTS=(
  "$W/data/scannet_val_frame_bundles_v3/formal_pool_batch04a_v3"
  "$W/data/scannet_val_frame_bundles_v3/formal_pool_batch04b_v3"
  "$W/data/scannet_val_frame_bundles_v3/formal_pool_batch04c_v3"
)
LEGACY_V1_ROOTS=(
  "$W/data/scannet_val_frame_bundles/canary_scene0011_v1"
  "$W/data/scannet_val_frame_bundles/formal_pool_batch01_v1"
  "$W/data/scannet_val_frame_bundles/formal_pool_batch02_v1"
  "$W/data/scannet_val_frame_bundles/formal_pool_batch03_v1"
)
SPLIT_FILES=(
  "$SPLIT_ROOT/scannetv2_train.txt"
  "$SPLIT_ROOT/scannetv2_val.txt"
  "$SPLIT_ROOT/scannetv2_test.txt"
)
OUTPUT="$W/data/scannet_val_frame_bundles_v3/formal_pool_batch05_v3"
QUEUE_DIR="$W/artifacts/cpu_queue/scannet_v52_frame_bundles_20260824_batch05_v3"
STATUS="$QUEUE_DIR/status.tsv"
BINDING="$QUEUE_DIR/execution_binding.json"
LOG="$QUEUE_DIR/formal_pool_batch05_v3.log"
mkdir -p "$QUEUE_DIR"

for path in "$QUEUE_SCRIPT" "$EXTRACTOR" "$SCENE_LIST" "$SOURCE_EXCLUSIONS" \
  "$V51_PACK/manifest.json" "$V51_REPORT" "${SPLIT_FILES[@]}"; do
  [[ -s "$path" ]] || { echo "FATAL: missing bound input: $path" >&2; exit 2; }
done
for root in "${V3_ROOTS[@]}" "${LEGACY_V1_ROOTS[@]}"; do
  [[ -d "$root" ]] || { echo "FATAL: missing bound frame root: $root" >&2; exit 2; }
done

write_or_verify_binding() {
  "$PYTHON_BIN" - "$BINDING" "$QUEUE_SCRIPT" "$EXTRACTOR" "$SCENE_LIST" \
    "$V51_PACK" "$V51_REPORT" "$SOURCE_EXCLUSIONS" "$SCANNET_ROOT" \
    "${V3_ROOTS[@]}" --legacy "${LEGACY_V1_ROOTS[@]}" --splits "${SPLIT_FILES[@]}" <<'PY'
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
v51_pack = Path(sys.argv[5])
v51_report = Path(sys.argv[6])
source_exclusions = Path(sys.argv[7])
scannet_root = Path(sys.argv[8])
legacy_marker = sys.argv.index("--legacy")
split_marker = sys.argv.index("--splits")
v3_roots = [Path(value) for value in sys.argv[9:legacy_marker]]
legacy_roots = [Path(value) for value in sys.argv[legacy_marker + 1:split_marker]]
split_files = [Path(value) for value in sys.argv[split_marker + 1:]]

def document(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }

scene_ids = [line.strip() for line in scene_list.read_text().splitlines() if line.strip()]
if len(scene_ids) != 11 or len(scene_ids) != len(set(scene_ids)):
    raise SystemExit("FATAL: batch05 scene list must contain 11 unique scenes")
if scene_ids != sorted(scene_ids):
    raise SystemExit("FATAL: batch05 scene list must remain lexicographically ordered")

v51 = v52.load_v51_reuse_binding(v51_pack, v51_report)
prior = v52._prior_scene_ids(v51)
if set(scene_ids) & prior:
    raise SystemExit("FATAL: batch05 overlaps the frozen prior-audit scene set")

v3_bindings = [v52.load_v3_frame_root(root, verify_artifacts=False) for root in v3_roots]
v3_scenes = {scene for binding in v3_bindings for scene in binding["scene_ids"]}
if set(scene_ids) & v3_scenes:
    raise SystemExit("FATAL: batch05 overlaps an existing v3 scene")

legacy_bindings = v51["manifest"]["inventory"].get(
    "additional_frame_root_bindings"
)
if not isinstance(legacy_bindings, list) or len(legacy_bindings) != len(legacy_roots):
    raise SystemExit("FATAL: v5.1 legacy-v1 root binding coverage drift")
if {binding.get("path") for binding in legacy_bindings} != {
    str(root.resolve()) for root in legacy_roots
}:
    raise SystemExit("FATAL: v5.1 legacy-v1 root paths drift")
for binding in legacy_bindings:
    manifest_path = Path(str(binding.get("manifest_path", ""))).resolve()
    if (
        not manifest_path.is_file()
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != binding.get("manifest_sha256")
    ):
        raise SystemExit(f"FATAL: legacy-v1 manifest hash drift: {manifest_path}")
legacy_scenes = {scene for binding in legacy_bindings for scene in binding["scene_ids"]}
if not set(scene_ids) <= legacy_scenes:
    raise SystemExit("FATAL: batch05 contains a scene outside the bound legacy-v1 roots")

official_val = set(split_files[1].read_text().split())
if not set(scene_ids) <= official_val:
    raise SystemExit("FATAL: batch05 contains a non-val scene")
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
        raise SystemExit(f"FATAL: batch05 source is incomplete: {scene_id}")

binding = {
    "schema": "gsa-scannet-v52-batch05-v3-cpu-queue-binding-v1",
    "purpose": "pose-valid v3 re-extraction of legacy-v1-only held-out val scenes",
    "training_materialization_eligible": False,
    "queue": document(queue),
    "extractor": document(extractor),
    "scene_list": document(scene_list),
    "scene_ids": scene_ids,
    "scene_ids_sha256": v52.canonical_hash(scene_ids),
    "prior_audit_scene_count": len(prior),
    "prior_audit_scene_ids_sha256": v52.canonical_hash(sorted(prior)),
    "v51_formal_binding": v51["files"],
    "source_exclusions": document(source_exclusions),
    "split_manifests": [document(path) for path in split_files],
    "v3_exclusion_roots": v3_bindings,
    "legacy_v1_source_roots": legacy_bindings,
    "selection_policy": {
        "mechanical_only": True,
        "source": "legacy-v1 scenes passing the frozen v5.1 strict RGB/sensor gate",
        "model_results_used": False,
        "human_votes_used": False,
        "openeqa_legacy_scenes_used": False,
    },
}
payload = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8")
if target.exists():
    if target.read_bytes() != payload:
        raise SystemExit("FATAL: batch05 execution binding no longer matches inputs")
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
    raise SystemExit("FATAL: batch05 output scene identity drift")
PY
}

record() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" \
    | tee -a "$STATUS"
}

write_or_verify_binding
record BOUND batch05 "binding=$BINDING scene_list_sha256=$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  record PREFLIGHT_PASS batch05 "11 official-val scenes; no dataset output written"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -d "$OUTPUT" ]] && verify_output "$OUTPUT"; then
  record REUSED batch05 "verified_v3_output=$OUTPUT"
  exit 0
fi
[[ ! -e "$OUTPUT" ]] || {
  record FAILED batch05 "unverified or partial output exists: $OUTPUT"
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
  --start-ordinal 600000
)
for root in "${V3_ROOTS[@]}"; do
  args+=(--exclude-frame-root "$root")
done
record STARTED batch05 "scenes=11 output=$OUTPUT cpu=only network=disabled"
if nice -n 10 "$PYTHON_BIN" "$EXTRACTOR" "${args[@]}" >"$LOG" 2>&1; then
  write_or_verify_binding
  if verify_output "$OUTPUT"; then
    record SUCCEEDED batch05 "verified_v3_output=$OUTPUT log=$LOG"
  else
    record FAILED batch05 "v3 output verification failed: $OUTPUT"
    exit 2
  fi
else
  rc=$?
  record FAILED batch05 "rc=$rc log=$LOG"
  exit "$rc"
fi
