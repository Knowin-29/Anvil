#!/usr/bin/env bash
set -euo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANNOTATOR="${ANNOTATOR:?set ANNOTATOR=a or ANNOTATOR=b}"
case "$ANNOTATOR" in
  a) DEFAULT_PORT=10320 ;;
  b) DEFAULT_PORT=10321 ;;
  *) echo "ANNOTATOR must be a or b" >&2; exit 2 ;;
esac

ARGS=(
  --pack "${PACK:-$W/data/semantic_audit_v2}"
  --annotator "$ANNOTATOR"
  --host "${HOST:-127.0.0.1}"
  --port "${PORT:-$DEFAULT_PORT}"
)
if [[ -n "${TOKEN:-}" ]]; then
  ARGS+=(--token "$TOKEN")
fi
if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  ARGS+=(--check)
fi

exec "${PYTHON_BIN:-python3}" "$W/tools/quick_semantic_audit_server.py" "${ARGS[@]}"
