#!/usr/bin/env python3
"""Merge strict mask repairs into a fresh, provenance-preserving raw root."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FORMS = (
    "action_cot", "affordance", "bbox", "depth",
    "detection", "dof", "mask", "point",
)


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty merge output: {args.out}")
    repaired = {
        row["image_sha256"]: row
        for row in read_rows(args.repair / "raw" / "mask.jsonl")
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    stats = {"base_records": 0, "repaired_masks": 0, "forms": {}}
    for form in FORMS:
        rows = read_rows(args.base / "raw" / f"{form}.jsonl")
        if form == "mask":
            merged = []
            for row in rows:
                replacement = repaired.get(row.get("image_sha256"))
                if replacement is not None:
                    merged.append(replacement)
                    stats["repaired_masks"] += 1
                else:
                    merged.append(row)
        else:
            merged = rows
        if merged:
            (args.out / "raw" / f"{form}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n"
                        for row in merged),
                encoding="utf-8",
            )
        stats["base_records"] += len(rows)
        stats["forms"][form] = len(merged)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    (args.out / "provenance.json").write_text(json.dumps({
        "base": str(args.base),
        "repair": str(args.repair),
        "replacement_key": "image_sha256",
        "repaired_mask_count": stats["repaired_masks"],
    }, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
