#!/usr/bin/env python3
"""Append a complete, frozen reviewer-C stream to canonical adjudication JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_adjudication import finalize


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=ROOT / "data/semantic_adjudication_v3")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/semantic_audit_v3/adjudication.jsonl",
    )
    args = parser.parse_args()
    print(json.dumps(finalize(args.pack, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
