#!/usr/bin/env bash
# Clean-v1 flywheel generation over the recovered BridgeData V2 image pool.
# Emits verified/needs_review/rejected JSONL per GSA dimension.
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
INDEX="${INDEX:-configs/data/bridge_images.index}"
OUT="${OUT:-data/flywheel_v1}"
LIMIT="${LIMIT:-0}"
WORKERS="${WORKERS:-40}"
python trains/flywheel/generate.py --index "$INDEX" --out "$OUT" --limit "$LIMIT" --workers "$WORKERS"
python trains/flywheel/rescan.py --dir "$OUT" --rounds 1 --limit 2000
