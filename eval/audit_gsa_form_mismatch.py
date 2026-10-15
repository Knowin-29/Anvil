"""How much of the gsa_bench_test parse-failure rate is a PROMPT defect?

Measured hypothesis: the bbox-group prompts are `Locate "X" in the image.` and never
say which geometric form to emit, so both models answer with a POINT. The scorer asks
`parse_boxes` for a box, finds none, and books a parse failure -- for a response that
did contain a well-formed answer, just of the other form.

This script separates three distinct outcomes per response, for both models:
  no_answer_at_all   nothing geometric came out
  wrong_form         a valid answer of the OTHER form (prompt never specified one)
  right_form         an answer of the form the scorer wanted

It also measures how many rows only parse once `coordinate_2d` / `coordinate` are
recognised as point keys, which is a plain parser-coverage gap.

Usage:
    python eval/audit_gsa_form_mismatch.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gsa_eval.parsing import (parse_boxes, parse_json_items, parse_points,  # noqa: E402
                              strip_reasoning)

DUMP = ROOT.parent / "save" / "logs"
# The keys parse_points recognised BEFORE this audit, so the delta is attributable.
_OLD_POINT_KEYS = ("point_2d", "point", "coordinates", "coord")
_NEW_POINT_KEYS = ("coordinate_2d", "coordinate", "point_2d_normalized", "position",
                   "center", "center_2d")


def _points_with_keys(text: str, keys) -> list:
    """Re-implement parse_points' dict branch over a chosen key set."""
    out = []
    for item in parse_json_items(strip_reasoning(text)):
        if isinstance(item, dict):
            for k in keys:
                v = item.get(k)
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    try:
                        out.append((float(v[0]), float(v[1])))
                    except (TypeError, ValueError):
                        pass
                    break
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                out.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                pass
    return out


def audit(model: str) -> dict:
    path = DUMP / f"preds_{model}_gsa_bench_test.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    tally: dict[str, Counter] = {"bbox": Counter(), "point": Counter()}
    key_hist: dict[str, Counter] = {"bbox": Counter(), "point": Counter()}
    gained = Counter()
    for r in rows:
        g = r["group"]
        if g not in tally:
            continue
        text = r["response"]
        boxes = parse_boxes(text)
        pts_old = _points_with_keys(text, _OLD_POINT_KEYS)
        pts_new = _points_with_keys(text, _OLD_POINT_KEYS + _NEW_POINT_KEYS)
        for item in parse_json_items(strip_reasoning(text)):
            if isinstance(item, dict):
                key_hist[g].update(item.keys())
        want_box = (g == "bbox")
        has_want = bool(boxes) if want_box else bool(pts_new)
        has_other = bool(pts_new) if want_box else bool(boxes)
        if has_want:
            tally[g]["right_form"] += 1
        elif has_other:
            tally[g]["wrong_form"] += 1
        else:
            tally[g]["no_answer_at_all"] += 1
        # Rows the old key set missed but the extended one catches.
        if not pts_old and pts_new:
            gained[g] += 1
    return {"tally": tally, "keys": key_hist, "key_gain": gained, "n": len(rows)}


def main() -> int:
    for model in ("base", "sft_9b_lora"):
        a = audit(model)
        print(f"\n{'=' * 66}\n{model}  (n={a['n']})\n{'=' * 66}")
        for g in ("bbox", "point"):
            c = a["tally"][g]
            n = sum(c.values())
            print(f"  {g} (n={n}):")
            for k in ("right_form", "wrong_form", "no_answer_at_all"):
                v = c.get(k, 0)
                print(f"    {k:20s} {v:4d}  {100.0 * v / n if n else 0:5.2f}%")
            print(f"    top keys: " + ", ".join(
                f"{k}={v}" for k, v in a["keys"][g].most_common(8)))
        print(f"  rows newly parseable from extended point keys: {dict(a['key_gain'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
