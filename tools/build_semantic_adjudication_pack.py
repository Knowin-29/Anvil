#!/usr/bin/env python3
"""Freeze A/B disagreements into a blind reviewer-C adjudication pack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_adjudication import build_pack


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pack", type=Path, default=ROOT / "data/semantic_audit_v3")
    parser.add_argument("--out", type=Path, default=ROOT / "data/semantic_adjudication_v3")
    parser.add_argument("--exclude-form", action="append", default=["depth"])
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="allow schema5 depth disagreements into the reviewer-C queue",
    )
    parser.add_argument("--expected-count", type=int, default=18)
    parser.add_argument("--reviewer-key", default="reviewer-c-v3")
    args = parser.parse_args()
    excluded_forms = set(args.exclude_form)
    if args.include_depth:
        excluded_forms.discard("depth")
    manifest = build_pack(
        args.source_pack,
        args.out,
        excluded_forms=excluded_forms,
        expected_count=args.expected_count,
        reviewer_key=args.reviewer_key,
    )
    print(json.dumps({
        "adjudication_id": manifest["adjudication_id"],
        "case_count": manifest["case_count"],
        "excluded_forms": manifest["excluded_forms"],
        "out": str(args.out.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
