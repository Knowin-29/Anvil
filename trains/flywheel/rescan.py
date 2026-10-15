#!/usr/bin/env python3
"""Verified@k re-scan: re-verify each 'verified' record with a fresh stochastic
generation of its bbox (temperature>0), incrementing _scan_count on survival.

This yields a per-sample process reward WITHOUT human labels: a record that keeps
matching under independent re-generation is high-confidence; one that flips is a
soft-negative. Emits survival counts for the paper's process-reward figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trains.flywheel import prompts, verifier  # noqa: E402
from trains.flywheel.vlm_client import chat_json  # noqa: E402
from trains.flywheel.taxonomy import BBOX_DIMS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="flywheel out dir (has verified/)")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    vdir = Path(args.dir) / "verified"
    survival = Counter()
    for jf in sorted(vdir.glob("*.jsonl")):
        dim = jf.stem
        recs = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
        if args.limit:
            recs = recs[: args.limit]
        for rec in recs:
            k = rec.get("_scan_count", 1)
            if dim in BBOX_DIMS and rec.get("bbox"):
                for _ in range(args.rounds):
                    cc = chat_json(rec["image_path"],
                                   prompts.cross_check(rec.get("object_class", "object"), rec["bbox"]),
                                   temperature=0.7, use_cache=False)
                    if isinstance(cc, dict) and cc.get("match", False):
                        k += 1
                    else:
                        break
            rec["_scan_count"] = k
            survival[min(k, 20)] += 1
        jf.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")

    out = Path(args.dir) / "verified_at_k.json"
    out.write_text(json.dumps({"survival_by_k": dict(sorted(survival.items()))}, indent=2))
    print("Verified@k survival:", dict(sorted(survival.items())))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
