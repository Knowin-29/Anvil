#!/usr/bin/env python3
"""Repair masks for images that already have a complete v3 audit cascade."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trains.flywheel import prompts_ext, verifier
from trains.flywheel.contracts import context_for
from trains.flywheel.validators import validate
from trains.flywheel.vlm_client import chat_json


def load_rows(root: Path) -> dict[str, dict[str, dict]]:
    forms = ("bbox", "detection", "mask", "depth",
             "affordance", "dof", "action_cot", "point")
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for form in forms:
        path = root / "raw" / f"{form}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["image_sha256"]][form] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty repair output: {args.out}")
    rows = load_rows(args.data)
    complete = [
        image_rows for image_rows in rows.values()
        if len(image_rows) == 8
    ]
    if args.max_images:
        complete = complete[:args.max_images]
    out_rows = []
    stats = {"images": len(complete), "repaired": 0, "failed": 0}
    for image_rows in complete:
        anchor = image_rows["bbox"]["gsa_context"]["primary"]
        context = image_rows["bbox"]["gsa_context"]
        image = image_rows["bbox"]["image_path"]
        cross_check = image_rows["detection"].get("_cross_check", [])
        base = prompts_ext.mask_polygon(anchor)
        feedback = (
            "\nThis is a repair pass. The previous mask was rejected because it "
            "was a rectangle or had too few vertices. Trace the visible silhouette "
            "with 8-16 distinct vertices; a four-corner box is invalid. If the "
            "silhouette cannot be seen, return [] instead."
        )
        accepted = None
        last_reason = "no_response"
        for attempt in range(3):
            prompt = base + feedback
            response = chat_json(
                image, prompt, max_tokens=1100,
                temperature=0.0 + 0.15 * attempt,
                use_cache=(attempt == 0),
            )
            if response is None:
                last_reason = "request_failed"
                continue
            payload = response if isinstance(response, list) else [response]
            ok, reason = validate("mask", payload, context_for(
                "mask", context.get("groundings", [])))
            if ok:
                accepted = payload
                break
            last_reason = reason
            feedback = (
                f"\nPrevious repair failed deterministic check {reason}. "
                "Return a non-rectangular polygon with at least 6 nested [x,y] "
                "vertices inside the same object box."
            )
        if accepted is None:
            stats["failed"] += 1
            continue
        raw = {
            "form": "mask",
            "image_path": image,
            "image_sha256": image_rows["bbox"]["image_sha256"],
            "source": image_rows["bbox"].get("source", "unknown"),
            "scene": image_rows["bbox"].get("scene", "unknown"),
            "label_version": image_rows["bbox"]["label_version"],
            "prompt_version": (
                image_rows["bbox"]["prompt_version"] + "-mask-repair"
            ),
            "payload": accepted,
            "gsa_context": context,
            "_cross_check": cross_check,
            "_verification": verifier.score_form(
                "mask", accepted, context_for(
                    "mask", context.get("groundings", [])), cross_check),
            "_semantic_gate": {
                "repair": {
                    "source_prompt_version": image_rows["bbox"]["prompt_version"],
                    "attempts_max": 3,
                    "strict_validator": True,
                }
            },
        }
        out_rows.append(raw)
        stats["repaired"] += 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    (args.out / "raw" / "mask.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out_rows),
        encoding="utf-8",
    )
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0 if stats["repaired"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
