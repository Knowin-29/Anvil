#!/usr/bin/env bash
# Merge the public-source corpus with all flywheel output into the training corpus.
# Safe to re-run at any time while the long-run generator keeps producing: it simply
# re-snapshots the current flywheel state as one shard.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source .env

SRC_DIR="${SRC_DIR:-trains/corpus_v2}"
OUT="$SRC_DIR/shard_flywheel.jsonl"
FILTERED="$OUT.filtered"
mkdir -p "$SRC_DIR"

# Re-run the current validator at the corpus boundary. Older Swift snapshots can
# contain records emitted before a validator fix (or audit-only candidates); a raw
# concatenation must never promote them into training.
python - "$FILTERED" <<'PY'
import glob
import json
import sys
from pathlib import Path

from trains.flywheel.contracts import validate_task_columns
from trains.flywheel.validators import VALIDATORS, validate

out = Path(sys.argv[1])
accepted = rejected = 0
reasons = {}
with out.open("w", encoding="utf-8") as handle:
    paths = sorted(glob.glob("data/flywheel_long/shard*/swift/*.jsonl"))
    paths += sorted(glob.glob("data/flywheel_*/swift/*.jsonl"))
    paths += sorted(glob.glob("data/anvil_flywheel_v1/shard*/swift/*.jsonl"))
    for path in paths:
        with Path(path).open(errors="ignore") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    form = (row.get("gsa_meta") or {}).get("form") or row.get("gsa_form")
                    messages = row.get("messages") or []
                    payload = json.loads(messages[-1]["content"])
                    context = row.get("gsa_context")
                except Exception:
                    rejected += 1
                    reasons["malformed"] = reasons.get("malformed", 0) + 1
                    continue
                columns_ok, columns_reason = validate_task_columns(row)
                if not columns_ok:
                    rejected += 1
                    key = f"contract:{columns_reason}"
                    reasons[key] = reasons.get(key, 0) + 1
                    continue
                if form not in VALIDATORS:
                    rejected += 1
                    reasons["unknown_form"] = reasons.get("unknown_form", 0) + 1
                    continue
                ok, reason = validate(form, payload, context)
                if (row.get("_audit_only")
                        or (row.get("gsa_meta") or {}).get("audit_only")):
                    rejected += 1
                    reasons["audit_only"] = reasons.get("audit_only", 0) + 1
                    continue
                if not ok:
                    rejected += 1
                    reasons[reason or "validator_reject"] = reasons.get(
                        reason or "validator_reject", 0) + 1
                    continue
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                accepted += 1
print(f"validated flywheel rows: accepted={accepted} rejected={rejected} reasons={dict(sorted(reasons.items()))}")
PY
sort -u "$FILTERED" > "$OUT.tmp"
rm -f "$FILTERED"
mv "$OUT.tmp" "$OUT"

FW=$(wc -l < "$OUT")
TOT=$(cat "$SRC_DIR"/shard_*.jsonl | wc -l)
echo "flywheel records : $FW"
echo "corpus total     : $TOT"

# report balance
python3 - "$SRC_DIR" <<'PY'
import json, sys, glob
from collections import Counter
scene, form, src = Counter(), Counter(), Counter()
n = 0
for f in glob.glob(sys.argv[1] + '/shard_*.jsonl'):
    for ln in open(f, errors='ignore'):
        try:
            m = json.loads(ln).get('gsa_meta') or {}
        except Exception:
            continue
        n += 1
        scene[m.get('scene', '?')] += 1
        form[m.get('form', '?')] += 1
        src[m.get('source', '?')] += 1
print(f"\nrecords with meta: {n:,}")
emb = 1 - scene.get('general_visual', 0) / max(n, 1)
print(f"embodied fraction: {emb:.3f}")
top = src.most_common(1)
if top:
    print(f"largest source   : {top[0][0]} {top[0][1]/max(n,1):.1%}")
print("scenes:", dict(scene.most_common()))
print("forms :", dict(form.most_common()))
PY
