#!/usr/bin/env python3
"""Prepare the 100-sample A_action_interface acceptance review package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.data_flywheel.gsa.a_action_interface_acceptance import prepare_acceptance_review_set


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT
        / "data/output/gsa_v0/a_action_interface_overnight/review_priority/review_priority_candidates.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/output/gsa_v0/a_action_interface_overnight/acceptance_review_100",
    )
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--max-per-object-name", type=int, default=24)
    parser.add_argument("--max-per-image", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if not args.candidates.exists():
        print(f"ACCEPTANCE_REVIEW_FAILED: candidates file not found: {args.candidates}")
        return 2
    summary = prepare_acceptance_review_set(
        candidates_path=args.candidates,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        max_per_object_name=args.max_per_object_name,
        max_per_image=args.max_per_image,
    )
    print("A_ACTION_INTERFACE_ACCEPTANCE_REVIEW_OK")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
