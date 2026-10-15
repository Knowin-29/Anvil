#!/usr/bin/env bash
# CPU-only, offline adapter canary for one official-train ScanNet .sens frame.
set -euo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$W"
[[ -s "$W/.env" ]] || { echo "FATAL: missing environment file: $W/.env" >&2; exit 2; }
source "$W/.env"

STAGE="${STAGE:-${1:-check}}"
SCENE_ID="${SCENE_ID:-scene0191_00}"
FRAME_INDEX="${FRAME_INDEX:-0}"
SCANNET_ROOT="${SCANNET_ROOT:-/path/to/datasets/downloads/ScanNet-official/scans}"
SPLIT_ROOT="${SPLIT_ROOT:-$W/configs/scannet_splits}"
LABEL_MAP="${LABEL_MAP:-/path/to/datasets/downloads/ScanNet-official/tasks/scannetv2-labels.combined.tsv}"
OUT="${OUT:-$W/data/scannet_train_sens_adapter/canary_v1}"
EXPECTED_ENV="${GSA_ENV_ROOT:-/path/to/environment/env}/gsa_train"
PYTHON="${PYTHON:-$EXPECTED_ENV/bin/python}"

[[ "$STAGE" == check || "$STAGE" == canary ]] || {
  echo "FATAL: STAGE must be check or canary" >&2
  exit 2
}
[[ "$SCENE_ID" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] || {
  echo "FATAL: invalid SCENE_ID: $SCENE_ID" >&2
  exit 2
}
[[ "$FRAME_INDEX" =~ ^[0-9]+$ ]] || {
  echo "FATAL: FRAME_INDEX must be a non-negative integer" >&2
  exit 2
}
for path in "$SCANNET_ROOT" "$SPLIT_ROOT"; do
  [[ -d "$path" ]] || { echo "FATAL: missing directory: $path" >&2; exit 2; }
done
[[ -s "$LABEL_MAP" ]] || { echo "FATAL: missing label map: $LABEL_MAP" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "FATAL: missing gsa_train Python: $PYTHON" >&2; exit 2; }
PYTHON_PREFIX="$($PYTHON -c 'import sys; print(sys.prefix)')"
[[ "$PYTHON_PREFIX" == "$EXPECTED_ENV" ]] || {
  echo "FATAL: Python prefix must be $EXPECTED_ENV, got $PYTHON_PREFIX" >&2
  exit 2
}

ARGS=(
  --stage "$STAGE"
  --scene-id "$SCENE_ID"
  --frame-index "$FRAME_INDEX"
  --scannet-root "$SCANNET_ROOT"
  --split-root "$SPLIT_ROOT"
  --label-map "$LABEL_MAP"
  --output-dir "$OUT"
)
if [[ -n "${LICENSE_EVIDENCE:-}" ]]; then
  ARGS+=(--license-evidence "$LICENSE_EVIDENCE")
fi

printf 'scannet_train_sens_adapter stage=%s scene=%s frame=%s cpu=only network=disabled output=%s\n' \
  "$STAGE" "$SCENE_ID" "$FRAME_INDEX" "$OUT"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$PYTHON" tools/extract_scannet_train_sens.py "${ARGS[@]}"
