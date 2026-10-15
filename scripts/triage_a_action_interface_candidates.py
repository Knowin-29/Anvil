#!/usr/bin/env python3
"""Build a prioritized A_action_interface review set from existing candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.data_flywheel.gsa.a_action_interface_triage import write_triage_outputs


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT
        / "data/output/gsa_v0/a_action_interface_overnight/merged_A_action_interface_candidates.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/output/gsa_v0/a_action_interface_overnight/review_priority",
    )
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--max-per-object-name", type=int, default=24)
    parser.add_argument("--max-per-image", type=int, default=6)
    parser.add_argument(
        "--include-unknown-objects",
        action="store_true",
        help="Use unknown object names to fill the review set after preferred manipulable objects.",
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    if not args.candidates.exists():
        print(f"TRIAGE_FAILED: candidates file not found: {args.candidates}")
        return 2
    summary = write_triage_outputs(
        candidates_path=args.candidates,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
        max_per_object_name=args.max_per_object_name,
        max_per_image=args.max_per_image,
        include_unknown_objects=args.include_unknown_objects,
    )
    print("A_ACTION_INTERFACE_TRIAGE_OK")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
