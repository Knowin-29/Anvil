#!/usr/bin/env python3
"""Audit the shared parsers against real dumped predictions.

Run offline (no GPU): reads save/logs/preds_*.jsonl and reports, per file, how often
the answer span is empty, how often the parser recovers points/boxes/choices, and
which format the model actually emitted. Exists because a parser that is friendlier
to one checkpoint's output style than another's turns a formatting difference into an
apparent capability difference.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.gsa_eval.parsing import (  # noqa: E402
    parse_boxes, parse_choice, parse_points, strip_reasoning)

_JSON_ARRAY_HEAD = re.compile(r"^\s*(?:```(?:json)?\s*)?\[", re.MULTILINE)


def classify(text: str) -> str:
    low = text.lower()
    has_open = "<think>" in low
    has_close = "</think>" in low
    if has_open and not has_close:
        return "truncated_think"
    if has_open:
        return "think_then_answer"
    if _JSON_ARRAY_HEAD.match(text or ""):
        return "bare_json"
    return "bare_prose"


def audit(path: Path, kind: str) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    shapes = Counter()
    empty_answer = 0
    recovered = 0
    n = 0
    lost_examples = []
    for r in rows:
        text = r.get("response") or ""
        n += 1
        shapes[classify(text)] += 1
        answer = strip_reasoning(text)
        if not answer.strip():
            empty_answer += 1
        if kind == "points":
            ok = bool(parse_points(text))
        elif kind == "boxes":
            ok = bool(parse_boxes(text))
        else:
            ok = parse_choice(text, 4) is not None
        if ok:
            recovered += 1
        elif len(lost_examples) < 3:
            lost_examples.append({"uid": r.get("uid"),
                                  "tail": text[-260:]})
    return {"file": path.name, "n": n, "shapes": dict(shapes),
            "empty_answer_span": empty_answer,
            "parsed_ok": recovered,
            "parse_rate_pct": round(100.0 * recovered / n, 2) if n else 0.0,
            "lost_examples": lost_examples}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=None)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()
    root = Path(a.logs) if a.logs else (Path(__file__).resolve().parents[1]
                                        / "save" / "logs")
    out = []
    for p in sorted(root.glob("preds_*.jsonl")):
        kind = "choice" if "cv_bench" in p.name or "embspatial" in p.name else "points"
        if "gsa_bench_test" in p.name:
            kind = "boxes"
        out.append(audit(p, kind))
    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    for r in out:
        print(f"\n=== {r['file']}  n={r['n']} ===")
        print(f"  shapes           : {r['shapes']}")
        print(f"  empty answer span: {r['empty_answer_span']}")
        print(f"  parsed ok        : {r['parsed_ok']}  ({r['parse_rate_pct']}%)")
        for ex in r["lost_examples"]:
            print(f"  LOST uid={ex['uid']}: ...{ex['tail'][-200:]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
