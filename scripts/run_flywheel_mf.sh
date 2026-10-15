#!/usr/bin/env bash
# Multi-form flywheel generation (8 task forms, validated). Long-running.
#   OFFSET=0 LIMIT=0 TARGET=200000 bash scripts/run_flywheel_mf.sh
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then source .env; fi
exec python trains/flywheel/generate_multiform.py \
  --index "${INDEX:-configs/data/embodied_images.index}" \
  --out "${OUT:-data/flywheel_mf}" \
  --offset "${OFFSET:-0}" --limit "${LIMIT:-0}" \
  --workers "${WORKERS:-56}" --target "${TARGET:-90000}"
