#!/usr/bin/env bash
# Launch the ESTABLISHED GSA viewer (dashboard + cases + human annotation platform)
# against our multi-form flywheel data, adapted into its expected layout.
set -uo pipefail
cd "$(dirname "$0")/.."
source .env
export NO_PROXY=127.0.0.1,localhost
export GSA_DATA_ROOT="${GSA_DATA_ROOT:-$PWD/data/viewer_root}"
export VIEWER_PORT="${VIEWER_PORT:-10310}"
FLY=/path/to/Anvil
cd "$FLY"
exec python scripts/viewer_service.py
