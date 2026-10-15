#!/usr/bin/env python3
"""Validate a frozen near-duplicate review file without changing audit evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.audit_multibench_leakage import (  # noqa: E402
    FullLeakageError,
    adjudicate_near_candidates,
    load_adjudications,
)


def load_candidates(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("pair_id"):
                raise FullLeakageError(f"candidate lacks pair_id at line {line_number}")
            rows.append(row)
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise FullLeakageError("candidate file contains duplicate pair_id values")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args()
    try:
        unresolved, leaks, cleared = adjudicate_near_candidates(
            load_candidates(args.candidates), load_adjudications(args.decisions)
        )
    except (OSError, ValueError, json.JSONDecodeError, FullLeakageError) as exc:
        print(json.dumps({"state": "INCOMPLETE", "error": str(exc)}, indent=2))
        return 4
    state = "LEAKAGE_DETECTED" if leaks else ("BLOCKED" if unresolved else "PASS")
    print(json.dumps({
        "state": state,
        "candidates": len(unresolved) + len(leaks) + len(cleared),
        "unresolved": len(unresolved),
        "leaks": len(leaks),
        "cleared": len(cleared),
    }, indent=2))
    return 0 if state == "PASS" else (3 if state == "LEAKAGE_DETECTED" else 4)


if __name__ == "__main__":
    raise SystemExit(main())
