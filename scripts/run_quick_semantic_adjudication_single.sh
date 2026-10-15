#!/usr/bin/env bash
set -euo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARGS=(
  --pack "${PACK:-$W/data/semantic_adjudication_v3}"
  --host "${HOST:-127.0.0.1}"
  --port "${PORT:-10322}"
)
if [[ -n "${TOKEN:-}" ]]; then
  ARGS+=(--token "$TOKEN")
fi
if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  ARGS+=(--check)
fi
exec "${PYTHON_BIN:-python3}" "$W/tools/quick_semantic_adjudication_single_server.py" "${ARGS[@]}"
