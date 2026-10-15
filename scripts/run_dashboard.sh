#!/usr/bin/env bash
# Serve the GSA-Forge data dashboard for human audit.
#   bash scripts/run_dashboard.sh            # data/flywheel_mf on :10307
#   DATA=... PORT=... bash scripts/run_dashboard.sh
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then source .env; fi
export NO_PROXY=127.0.0.1,localhost
exec python eval/dashboard.py --data "${DATA:-data/flywheel_mf}" --port "${PORT:-10307}"
