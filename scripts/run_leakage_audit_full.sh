#!/usr/bin/env bash
set -euo pipefail

ROOT="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/path/to/environment/env/gsa_train/bin/python}"
OUTPUT_DIR="${LEAKAGE_OUTPUT_DIR:-$ROOT/artifacts/leakage_audit_full_v1}"

cd "$ROOT"
exec "$PYTHON_BIN" tools/audit_multibench_leakage.py \
  --output-dir "$OUTPUT_DIR" \
  "$@"
