#!/usr/bin/env python3
"""Render the exact v3 prompt family without making a model request."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trains.flywheel import prompts, prompts_ext


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("artifacts/manifests/v3_prompt_preview.md"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing file: {args.out}")

    objects = [
        {"object_class": "red mug", "bbox": [180, 240, 430, 720]},
        {"object_class": "blue towel", "bbox": [560, 390, 880, 760]},
    ]
    cross_check_objects = [
        {"object_id": f"o{index + 1}", **item}
        for index, item in enumerate(objects)
    ]
    depth_candidate = [
        {"label": "red mug", "rank": 1, "relative_depth": 0.2},
        {"label": "blue towel", "rank": 2, "relative_depth": 0.7},
    ]
    affordance = [{
        "label": "mug rim",
        "affordance": "grasp",
        "bbox_2d": [210, 250, 390, 330],
        "contact_point_2d": [300, 285],
    }]
    dof = [{
        "label": "red mug",
        "grasp_point_2d": [300, 285],
        "jaw_axis_2d": [0.87, -0.49],
        "approach": "top",
        "wrist_deg": -29,
    }]
    action = [{
        "goal": "grasp and lift the red mug",
        "steps": ["approach the rim", "close the jaws", "lift the mug"],
        "preconditions": ["the rim is unobstructed"],
        "failure_modes": ["the grasp slips from the rim"],
        "target_point_2d": [300, 285],
    }]
    sections = [
        ("Scene eligibility", prompts_ext.SCENE_ELIGIBILITY),
        ("Detection", prompts_ext.DETECTION),
        ("Detection cross-check", prompts.cross_check_set(cross_check_objects)),
        ("Mask", prompts_ext.mask_polygon(objects[0])),
        ("Depth ordering", prompts_ext.depth_order(objects)),
        ("Depth verification", prompts_ext.verify_depth_order(objects, depth_candidate)),
        ("Affordance", prompts_ext.affordance_region(
            objects[0], {"depth_order": depth_candidate})),
        ("DoF", prompts_ext.dof_interface(
            objects[0], {"depth_order": depth_candidate}, affordance)),
        ("Action plan", prompts_ext.action_cot(
            objects[0], {"depth_order": depth_candidate}, affordance, dof)),
        ("Joint manipulation verification", prompts_ext.verify_manipulation(
            objects[0], affordance, dof, action)),
    ]
    lines = [
        "# GSA Forge semantic generation v3 prompt preview",
        "",
        f"- prompt_version: {prompts_ext.PROMPT_VERSION}",
        f"- label_version: {prompts_ext.LABEL_VERSION}",
        "- This file is rendered locally and makes zero API requests.",
        "- Scene eligibility and detection are the exact static prompts.",
        "- Later sections are the exact prompt functions instantiated with an "
        "illustrative red-mug/blue-towel cascade. Runtime detections replace those "
        "fields, and every actual instantiated prompt is written to the local "
        "request audit log.",
        "",
    ]
    for title, value in sections:
        lines.extend([
            f"## {title}",
            "",
            f"SHA256: {digest(value)}",
            "",
            "~~~text",
            value,
            "~~~",
            "",
        ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "prompt_version": prompts_ext.PROMPT_VERSION,
        "sections": len(sections),
        "section_hashes": {title: digest(value) for title, value in sections},
        "api_calls": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
