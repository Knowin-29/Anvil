"""Why do 38.83% of the SFT model's gsa_bench_test answers fail to parse?

Reads the recorded dumps (no inference) and, for every response the scorer counted as
a parse failure, classifies WHY: empty answer span, unterminated <think>, JSON that
loads but carries no recognised key, JSON that does not load at all, and so on.

Usage:
    python eval/audit_gsa_parsefail.py [--model sft_9b_lora] [--show 15]
"""
from __future__ import annotations

import argparse
import ast
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


def _loadable(blob: str):
    for loader in (json.loads, ast.literal_eval):
        try:
            return True, loader(blob)
        except Exception:
            continue
    return False, None


def classify(group: str, response: str) -> tuple[str, dict]:
    """Return (reason, evidence) for a response whose answer could not be parsed."""
    ev: dict = {}
    if not response:
        return "empty_response", ev
    low = response.lower()
    if "<think>" in low and "</think>" not in low:
        return "unclosed_think", ev
    answer = strip_reasoning(response)
    ev["answer_len"] = len(answer)
    if not answer.strip():
        return "empty_answer_span", ev
    items = parse_json_items(answer)
    ev["n_json_items"] = len(items)
    if not items:
        ok, _ = _loadable(answer.strip())
        if ok:
            return "json_loads_but_not_list_or_dict", ev
        if re.search(r"[\[{]", answer):
            return "json_present_but_unloadable", ev
        return "no_json_structure", ev
    keys: Counter = Counter()
    for it in items:
        if isinstance(it, dict):
            keys.update(it.keys())
        else:
            keys.update([f"<{type(it).__name__}>"])
    ev["item_keys"] = dict(keys)
    want = ({"bbox_2d", "bbox", "box", "bounding_box"} if group == "bbox"
            else {"point_2d", "point", "coordinates", "coord"})
    if not (set(keys) & want):
        return "json_items_lack_expected_key", ev
    return "key_present_but_value_rejected", ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sft_9b_lora")
    ap.add_argument("--bench", default="gsa_bench_test")
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()

    path = DUMP / f"preds_{a.model}_{a.bench}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"{path.name}: {len(rows)} rows")

    reasons: Counter = Counter()
    by_group: dict[str, Counter] = {}
    examples: dict[str, list] = {}
    n_fail = 0
    for r in rows:
        g = r.get("group") or ""
        parsed = parse_boxes(r["response"]) if g == "bbox" else parse_points(r["response"])
        if parsed:
            continue
        n_fail += 1
        reason, ev = classify(g, r["response"])
        reasons[reason] += 1
        by_group.setdefault(g, Counter())[reason] += 1
        examples.setdefault(f"{g}/{reason}", []).append((r["uid"], ev, r["response"]))

    print(f"\nparse failures: {n_fail}/{len(rows)} = {100.0 * n_fail / len(rows):.2f}%")
    print("\nby reason:")
    for k, v in reasons.most_common():
        print(f"  {v:5d}  {k}")
    print("\nby group x reason:")
    for g, c in sorted(by_group.items()):
        print(f"  {g}: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))

    print("\n=== examples ===")
    for key, rows_ in sorted(examples.items()):
        print(f"\n--- {key} ({len(rows_)}) ---")
        for uid, ev, resp in rows_[:a.show]:
            print(f"[{uid}] {ev}")
            print("  " + repr(resp[:400]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
