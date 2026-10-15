#!/usr/bin/env bash
# CPU-only official-val ScanNet frame-bundle check/canary/batch launcher.
set -euo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$W"
[[ -s "$W/.env" ]] || { echo "FATAL: missing environment file: $W/.env" >&2; exit 2; }
source "$W/.env"

CHECK_ONLY="${CHECK_ONLY:-1}"
STAGE="${STAGE:-canary}"
SCENE_ID="${SCENE_ID:-scene0011_00}"
SCENE_LIST="${SCENE_LIST:-}"
SCANNET_ROOT="${SCANNET_ROOT:-/path/to/datasets/downloads/ScanNet-official/scans}"
SPLIT_ROOT="${SPLIT_ROOT:-$W/configs/scannet_splits}"
SOURCE_EXCLUSIONS="${SOURCE_EXCLUSIONS:-$W/configs/scannet_source_exclusions_v1.json}"
EXCLUDE_FRAME_ROOT="${EXCLUDE_FRAME_ROOT:-/path/to/datasets/datasets/transferred_datasets/EmbodiedAI_benchmark/openeqa/scannet-v0}"
OUT="${OUT:-$W/data/scannet_val_frame_bundles/canary_v1}"
MAX_FRAMES="${MAX_FRAMES:-4}"
START_ORDINAL="${START_ORDINAL:-0}"
EXPECTED_ENV="${GSA_ENV_ROOT:-/path/to/environment/env}/gsa_train"
PYTHON="${PYTHON:-$EXPECTED_ENV/bin/python}"

[[ "$CHECK_ONLY" == 0 || "$CHECK_ONLY" == 1 ]] || {
  echo "FATAL: CHECK_ONLY must be 0 or 1" >&2
  exit 2
}
[[ "$STAGE" == canary || "$STAGE" == batch ]] || {
  echo "FATAL: STAGE must be canary or batch" >&2
  exit 2
}
[[ "$SCENE_ID" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] || {
  echo "FATAL: invalid SCENE_ID: $SCENE_ID" >&2
  exit 2
}
[[ "$MAX_FRAMES" =~ ^[0-9]+$ ]] && (( MAX_FRAMES >= 1 && MAX_FRAMES <= 64 )) || {
  echo "FATAL: MAX_FRAMES must be an integer in [1,64]" >&2
  exit 2
}
[[ "$START_ORDINAL" =~ ^[0-9]+$ ]] || {
  echo "FATAL: START_ORDINAL must be a non-negative integer" >&2
  exit 2
}
for path in "$SCANNET_ROOT" "$SPLIT_ROOT" "$EXCLUDE_FRAME_ROOT"; do
  [[ -d "$path" ]] || { echo "FATAL: missing directory: $path" >&2; exit 2; }
done
[[ -s "$SOURCE_EXCLUSIONS" ]] || {
  echo "FATAL: missing source exclusions: $SOURCE_EXCLUSIONS" >&2
  exit 2
}
[[ -x "$PYTHON" ]] || { echo "FATAL: missing gsa_train Python: $PYTHON" >&2; exit 2; }
PYTHON_PREFIX="$($PYTHON -c 'import sys; print(sys.prefix)')"
[[ "$PYTHON_PREFIX" == "$EXPECTED_ENV" ]] || {
  echo "FATAL: Python prefix must be $EXPECTED_ENV, got $PYTHON_PREFIX" >&2
  exit 2
}

if [[ "$CHECK_ONLY" == 1 ]]; then
  RUN_STAGE=check
elif [[ "$STAGE" == batch ]]; then
  RUN_STAGE=batch
  [[ -s "$SCENE_LIST" ]] || {
    echo "FATAL: batch requires nonempty SCENE_LIST" >&2
    exit 2
  }
else
  RUN_STAGE=canary
fi

ARGS=(
  --stage "$RUN_STAGE"
  --scannet-root "$SCANNET_ROOT"
  --split-root "$SPLIT_ROOT"
  --source-exclusions "$SOURCE_EXCLUSIONS"
  --exclude-frame-root "$EXCLUDE_FRAME_ROOT"
  --max-frames "$MAX_FRAMES"
  --start-ordinal "$START_ORDINAL"
)
if [[ "$RUN_STAGE" == batch ]]; then
  ARGS+=(--scene-list "$SCENE_LIST" --output-root "$OUT")
else
  ARGS+=(--scene-id "$SCENE_ID")
  if [[ "$RUN_STAGE" == canary ]]; then
    ARGS+=(--output-root "$OUT")
  fi
fi

printf 'scannet_val_frame_bundle stage=%s scene=%s scene_list=%s frames=%s cpu=only network=disabled output=%s\n' \
  "$RUN_STAGE" "$SCENE_ID" "${SCENE_LIST:-none}" "$MAX_FRAMES" "$OUT"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$PYTHON" tools/extract_scannet_val_frame_bundles.py "${ARGS[@]}"
