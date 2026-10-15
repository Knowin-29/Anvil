#!/usr/bin/env bash
# Non-gating machine triage. Use PROMPT_ONLY=1 to render all requests without HTTP.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
[[ -f "$W/.env" ]] || { echo "FATAL: missing $W/.env" >&2; exit 1; }
# shellcheck disable=SC1091
source "$W/.env"
exec python "$W/tools/semantic_machine_triage.py" "$@"
