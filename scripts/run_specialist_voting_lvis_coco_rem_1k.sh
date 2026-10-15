#!/usr/bin/env bash
set -euo pipefail

# Specialist voting verifier runner for LVIS/COCO-ReM 1k.
# Prefer exporting the real key before running:
#   export VOLC_API_KEY="your-volc-api-key"
: "${VOLC_API_KEY:=<PUT_VOLC_API_KEY_HERE>}"
export VOLC_API_KEY
if [[ "${VOLC_API_KEY}" == "<PUT_VOLC_API_KEY_HERE>" || -z "${VOLC_API_KEY}" ]]; then
  echo "VOLC_API_KEY is not configured. Export it first or edit this script." >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/workspace/foundation/anaconda3_new/bin/python}"
WORKER_COUNT="${WORKER_COUNT:-4}"
INPUT_PATH="${PROJECT_ROOT}/data/input/grounding_verifier_lvis_coco_rem_1k/lvis_coco_rem_grounding_1k.jsonl"
CONFIG_PATH="${PROJECT_ROOT}/configs/grounding_verifier_rules.yaml"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/data/input/grounding_verifier_lvis_coco_rem_1k/verifier_report_specialist_voting_$(date +%Y%m%d_%H%M).json}"

cd "${PROJECT_ROOT}"
echo "Running specialist voting verifier"
echo "  input: ${INPUT_PATH}"
echo "  config: ${CONFIG_PATH}"
echo "  workers: ${WORKER_COUNT}"
echo "  output: ${OUTPUT_PATH}"

"${PYTHON_BIN}" scripts/evaluate_specialist_voting.py \
  --input "${INPUT_PATH}" \
  --config "${CONFIG_PATH}" \
  --output "${OUTPUT_PATH}" \
  --vote-policy any \
  --worker-count "${WORKER_COUNT}"

echo "Wrote ${OUTPUT_PATH}"
