#!/bin/bash
# GSA Data Flywheel v0 — Run Script
# Usage: bash scripts/run_gsa_v0.sh [--config CONFIG_PATH] [--max-images N]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${PROJECT_ROOT}/scripts/configs/gsa_v0.yaml"
MAX_IMAGES=""
INGEST_ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --max-images) MAX_IMAGES="$2"; shift 2 ;;
        --ingest-only) INGEST_ONLY=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== GSA Data Flywheel v0 ==="
echo "Config: $CONFIG"
echo "Project: $PROJECT_ROOT"
echo ""

# Check VLM API key only for remote API configs. Local vLLM configs do not need it.
if grep -q "apigateway\|https://" "$CONFIG" && [ -z "${VLM_API_KEY:-}" ]; then
    echo "WARNING: VLM_API_KEY not set. Set it for remote API access."
    echo "  export VLM_API_KEY='your-api-key'"
fi

# Run the pipeline
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

if [ -n "$INGEST_ONLY" ]; then
    export GSA_INGEST_ONLY=1
    echo "Mode: ingest-only (no VLM image pipeline)"
fi

if [ -n "$MAX_IMAGES" ]; then
    export GSA_MAX_IMAGES="$MAX_IMAGES"
fi

# Prefer project rex env when available
PY="${PY:-}"
if [ -z "$PY" ]; then
    if [ -x "/path/to/workspace/foundation/contributor/envs/rex/bin/python" ]; then
        PY="/path/to/workspace/foundation/contributor/envs/rex/bin/python"
    elif [ -x "/path/to/environment/env/rex/bin/python" ]; then
        PY="/path/to/environment/env/rex/bin/python"
    else
        PY="python"
    fi
fi

echo "Starting GSA v0 pipeline..."
"$PY" -c "
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

from apps.data_flywheel.gsa.runner import GSAv0Runner

runner = GSAv0Runner(config_path='$CONFIG')
if '$MAX_IMAGES':
    runner.config.setdefault('input', {})['max_images'] = int('$MAX_IMAGES')
summary = runner.run()

print()
print('=== GSA v0 Run Complete ===')
print(f'Run ID: {summary[\"run_id\"]}')
print(f'Images: {summary[\"images_processed\"]}')
print(f'Labels: {summary[\"total_labels\"]}')
print(f'Reviews: {summary[\"total_reviews\"]}')
print(f'Verifiable Rate: {summary[\"eval_results\"][\"overall_verifiable_rate\"]:.2%}')
print()
print('Output directory:')
print(f'  {runner.output_root}')
print()
print('Reports:')
print(f'  {runner.output_root}/reports/gsa_gap_report_v0.html')
print(f'  {runner.output_root}/eval/gsa_eval_results.json')
"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: GSA v0 pipeline failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "=== Done ==="
