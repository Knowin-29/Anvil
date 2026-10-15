#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
python trains/flywheel_to_swift.py --in-glob 'data/flywheel_v1/verified/*.jsonl' --out trains/gsa_sft_clean_v1.jsonl --min-scan 1
