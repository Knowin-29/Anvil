#!/usr/bin/env bash
# Offline ScanNet sensor-depth builder. It never starts a GPU job or VLM client.
set -euo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$W"
[[ -s "$W/.env" ]] || { echo "FATAL: missing environment file: $W/.env" >&2; exit 2; }
source "$W/.env"

STAGE="${STAGE:-${1:-check}}"
case "$STAGE" in
  check) DEFAULT_OUT="$W/data/depth_v50_scannet_sensor/check_v4" ;;
  canary) DEFAULT_OUT="$W/data/depth_v50_scannet_sensor/canary_v4" ;;
  full) DEFAULT_OUT="$W/data/depth_v50_scannet_sensor/full_v3" ;;
  *) echo "FATAL: STAGE must be check, canary, or full" >&2; exit 2 ;;
esac

OPENEQA_ROOT="${OPENEQA_ROOT:-/path/to/datasets/datasets/transferred_datasets/EmbodiedAI_benchmark/openeqa/scannet-v0}"
SCANNET_ROOT="${SCANNET_ROOT:-/path/to/datasets/downloads/ScanNet-official/scans}"
LABEL_MAP="${LABEL_MAP:-/path/to/datasets/downloads/ScanNet-official/tasks/scannetv2-labels.combined.tsv}"
SPLIT_ROOT="${SPLIT_ROOT:-$W/configs/scannet_splits}"
SOURCE_EXCLUSIONS="${SOURCE_EXCLUSIONS:-$W/configs/scannet_source_exclusions_v1.json}"
CANARY_SCENES="${CANARY_SCENES:-$W/configs/scannet_sensor_canary_scenes_v1.json}"
OUT="${OUT:-$DEFAULT_OUT}"
CANARY_REPORT="${CANARY_REPORT:-$W/data/depth_v50_scannet_sensor/canary_v4/generation_report.json}"
TARGET="${TARGET:-120}"
MAX_FRAMES_PER_SCENE="${MAX_FRAMES_PER_SCENE:-64}"
MAX_RECORDS_PER_SCENE="${MAX_RECORDS_PER_SCENE:-2}"
CHECK_SCENES="${CHECK_SCENES:-10}"
EXPECTED_ENV="${GSA_ENV_ROOT:-/path/to/environment/env}/gsa_train"
PYTHON="${PYTHON:-$EXPECTED_ENV/bin/python}"

for value in "$TARGET" "$MAX_FRAMES_PER_SCENE" "$MAX_RECORDS_PER_SCENE" "$CHECK_SCENES"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: numeric builder settings must be positive integers" >&2
    exit 2
  }
done
for path in "$OPENEQA_ROOT" "$SCANNET_ROOT" "$SPLIT_ROOT"; do
  [[ -d "$path" ]] || { echo "FATAL: missing directory: $path" >&2; exit 2; }
done
[[ -s "$LABEL_MAP" ]] || { echo "FATAL: missing label map: $LABEL_MAP" >&2; exit 2; }
[[ -s "$SOURCE_EXCLUSIONS" ]] || {
  echo "FATAL: missing source exclusion manifest: $SOURCE_EXCLUSIONS" >&2
  exit 2
}
[[ -s "$CANARY_SCENES" ]] || {
  echo "FATAL: missing frozen canary scene manifest: $CANARY_SCENES" >&2
  exit 2
}
[[ -x "$PYTHON" ]] || { echo "FATAL: missing gsa_train Python: $PYTHON" >&2; exit 2; }
PYTHON_PREFIX="$($PYTHON -c 'import sys; print(sys.prefix)')"
[[ "$PYTHON_PREFIX" == "$EXPECTED_ENV" ]] || {
  echo "FATAL: Python prefix must be $EXPECTED_ENV, got $PYTHON_PREFIX" >&2
  exit 2
}
if [[ "$STAGE" == full ]]; then
  (( TARGET >= 120 )) || { echo "FATAL: full target must be at least 120" >&2; exit 2; }
  [[ -s "$CANARY_REPORT" ]] || {
    echo "FATAL: full stage requires canary report: $CANARY_REPORT" >&2
    exit 2
  }
fi

ARGS=(
  --stage "$STAGE"
  --openeqa-root "$OPENEQA_ROOT"
  --scannet-root "$SCANNET_ROOT"
  --label-map "$LABEL_MAP"
  --split-root "$SPLIT_ROOT"
  --source-exclusions "$SOURCE_EXCLUSIONS"
  --canary-scenes "$CANARY_SCENES"
  --output-dir "$OUT"
  --target "$TARGET"
  --max-frames-per-scene "$MAX_FRAMES_PER_SCENE"
  --max-records-per-scene "$MAX_RECORDS_PER_SCENE"
  --check-scenes "$CHECK_SCENES"
)
if [[ "$STAGE" == full ]]; then
  ARGS+=(--canary-report "$CANARY_REPORT")
fi

printf 'scannet_sensor_depth stage=%s target=%s output=%s gpu=disabled external_calls=disabled\n' \
  "$STAGE" "$TARGET" "$OUT"
exec "$PYTHON" tools/build_depth_v50_scannet_sensor.py "${ARGS[@]}"
