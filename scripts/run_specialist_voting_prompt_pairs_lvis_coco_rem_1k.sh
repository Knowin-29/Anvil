#!/usr/bin/env bash
set -euo pipefail

# Batch specialist voting verifier runner for paired bbox/pointing prompts.
# It runs prompt pairs 0-0, 1-1, ..., 9-9 by default.
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
WORKER_COUNT="${WORKER_COUNT:-32}"
PROMPT_INDEXES="${PROMPT_INDEXES:-0 1 2 3 4 5 6 7 8 9}"
INPUT_PATH="${PROJECT_ROOT}/data/input/grounding_verifier_lvis_coco_rem_1k/lvis_coco_rem_grounding_1k.jsonl"
CONFIG_PATH="${PROJECT_ROOT}/configs/grounding_verifier_rules.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/input/grounding_verifier_lvis_coco_rem_1k}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M)_workers${WORKER_COUNT}}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/specialist_voting_prompt_pairs_${RUN_TAG}.log}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
echo "Running specialist voting prompt-pair sweep"
echo "  input: ${INPUT_PATH}"
echo "  config: ${CONFIG_PATH}"
echo "  workers: ${WORKER_COUNT}"
echo "  prompt indexes: ${PROMPT_INDEXES}"
echo "  output dir: ${OUTPUT_DIR}"
echo "  log: ${LOG_PATH}"

: > "${LOG_PATH}"
for idx in ${PROMPT_INDEXES}; do
  OUTPUT_PATH="${OUTPUT_DIR}/verifier_report_specialist_voting_${RUN_TAG}_bbox${idx}_pointing${idx}.json"
  echo "[prompt-pair ${idx}-${idx}] start $(date +%F_%T) -> ${OUTPUT_PATH}" | tee -a "${LOG_PATH}"
  "${PYTHON_BIN}" scripts/evaluate_specialist_voting.py \
    --input "${INPUT_PATH}" \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_PATH}" \
    --vote-policy any \
    --worker-count "${WORKER_COUNT}" \
    --bbox-prompt-index "${idx}" \
    --pointing-prompt-index "${idx}" \
    >> "${LOG_PATH}" 2>&1
  echo "[prompt-pair ${idx}-${idx}] done $(date +%F_%T)" | tee -a "${LOG_PATH}"
done

echo "Completed specialist voting prompt-pair sweep"
