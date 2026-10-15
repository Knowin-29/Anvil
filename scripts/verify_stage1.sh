#!/usr/bin/env bash
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"

# shellcheck disable=SC1091
source "$W/.env" >/dev/null 2>&1 || true
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$W/.cache/mpl"
mkdir -p "$MPLCONFIGDIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "FATAL: python not found; source .env or set PYTHON_BIN" >&2
  exit 1
}

for required in paper eval scripts trains experiments runs artifacts; do
  [[ -d "$required" ]] || { echo "FATAL: missing $required/" >&2; exit 1; }
done

while IFS= read -r script; do
  bash -n "$script"
done < <(find scripts eval -type f -name '*.sh' -not -path '*/archive/*' -print)
echo "[ok] shell syntax"

"$PYTHON_BIN" - <<'PY'
import ast
from pathlib import Path

roots = [Path("eval"), Path("scripts"), Path("trains"), Path("analysis")]
files = [p for root in roots for p in root.rglob("*.py")
         if "archive" not in p.parts and "__pycache__" not in p.parts]
for path in files:
    ast.parse(path.read_text(errors="strict"), filename=str(path))
print(f"[ok] parsed {len(files)} Python files")
PY

"$PYTHON_BIN" eval/test_harness_offline.py
"$PYTHON_BIN" eval/verify_f1f2f3.py
"$PYTHON_BIN" eval/verify_shard_scoring.py
PYTHONPATH="$W" "$PYTHON_BIN" -m pytest -q \
  tests/test_gsa_contracts.py \
  tests/test_leakage_audit.py \
  tests/test_mopd_routing.py \
  tests/test_eval_protocol_v7.py
echo "[ok] GSA contracts, leakage audit, and MOPD routing"

TMP_TABLES="$(mktemp -d /tmp/gsa-stage1-tables.XXXXXX)"
trap 'rm -rf "$TMP_TABLES"' EXIT
"$PYTHON_BIN" eval/backfill_tables.py \
  --results runs/eval_v2_20260817 --out-dir "$TMP_TABLES"
# The historical base CV score predates the truncation field and exceeded the
# current 15% gate; it must remain withheld while the trusted SFT score survives.
if rg -q '78\.3' "$TMP_TABLES/tab_main.tex"; then
  echo "FATAL: untrusted legacy base CV score entered the table" >&2; exit 1
fi
rg -q '77.5' "$TMP_TABLES/tab_main.tex"
echo "[ok] v2 result backfill"

"$PYTHON_BIN" analysis/plot_eval_snapshot.py
echo "[ok] diagnostic figure"

if command -v pdflatex >/dev/null && command -v bibtex >/dev/null; then
  bash paper/compile_check.sh
  echo "[ok] paper compile"
else
  echo "[skip] paper compile (pdflatex/bibtex unavailable)"
fi

echo "stage-1 verification passed"
