#!/usr/bin/env python3
"""Run the A_action_interface full GSA loop on BridgeDataV2 samples."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.data_flywheel.gsa.a_action_interface_review import (
    load_action_interface_labels,
    load_review_decisions,
    make_review_decision_template,
    normalize_action_interface_candidates,
    promote_action_interface_candidates,
    write_jsonl,
    write_review_html,
)
from apps.data_flywheel.gsa.data_adapters.bridge_adapter import discover_bridge_images
from apps.data_flywheel.gsa.runner import GSAv0Runner


DEFAULT_EXCLUDE_OBJECT_NAMES = [
    "wall",
    "table",
    "cabinet",
    "robot",
    "robotic_arm",
    "robot_arm",
    "arm",
    "gripper",
    "hand",
    "microwave",
    "faucet",
    "stove",
    "burner",
    "knob",
    "sink",
    "door",
    "stand",
    "shelf",
    "drawer",
    "counter",
    "handle",
    "surface",
    "floor",
    "oven",
]

DEFAULT_PREFERRED_OBJECT_NAMES = [
    "can",
    "bowl",
    "cup",
    "mug",
    "spoon",
    "fork",
    "knife",
    "spatula",
    "carrot",
    "banana",
    "apple",
    "tomato",
    "sushi",
    "bottle",
    "brush",
    "cloth",
    "towel",
    "pot",
    "pan",
    "plate",
    "lid",
    "block",
    "toy",
    "container",
    "box",
    "bag",
    "sponge",
]


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/configs/gsa_v0.yaml")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--max-objects-per-image", type=int, default=3)
    parser.add_argument("--max-bbox-area-ratio", type=float, default=0.45)
    parser.add_argument(
        "--exclude-object-name",
        action="append",
        default=list(DEFAULT_EXCLUDE_OBJECT_NAMES),
    )
    parser.add_argument(
        "--preferred-object-name",
        action="append",
        default=list(DEFAULT_PREFERRED_OBJECT_NAMES),
    )
    parser.add_argument(
        "--online-all-context",
        action="store_true",
        help="Keep VLM calls enabled for all non-target GSA context pipelines.",
    )
    parser.add_argument(
        "--use-consensus-grounding",
        action="store_true",
        help="Run two-pass grounding consensus instead of one-pass detection.",
    )
    parser.add_argument(
        "--enable-json-repair",
        action="store_true",
        help="Allow A_action_interface to make a second text-only repair call when JSON is missing.",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--decisions", type=Path, default=None)
    parser.add_argument("--promote-only", action="store_true")
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = _args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    _load_env(ROOT / ".env")

    if args.promote_only:
        if not args.run_dir:
            print("FULL_LOOP_FAILED: --promote-only requires --run-dir")
            return 2
        run_dir = args.run_dir
        if not run_dir.exists():
            print(f"FULL_LOOP_FAILED: run dir not found: {run_dir}")
            return 2
    else:
        if not os.environ.get("VLM_API_KEY"):
            print("FULL_LOOP_FAILED: VLM_API_KEY is not set")
            return 2
        run_dir = _run_full_gsa(args)

    outputs = _write_review_and_promotion_outputs(run_dir, args.decisions)
    print("A_ACTION_INTERFACE_FULL_LOOP_OK")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


def _run_full_gsa(args: argparse.Namespace) -> Path:
    runner = GSAv0Runner(config_path=args.config)
    _configure_optimized_runner(runner, args)
    run_dir = args.run_dir or runner.output_root
    if args.run_dir:
        runner.output_root = args.run_dir
        runner.run_id = args.run_dir.name

    labels_path = run_dir / "labels" / "A_action_interface" / "A_action_interface.jsonl"
    if labels_path.exists() and not args.force_run:
        logging.info("Reusing existing GSA run outputs in %s", run_dir)
        return run_dir

    image_limit = args.image_offset + args.max_images
    images = discover_bridge_images(max_images=image_limit)
    if not images:
        raise RuntimeError("BridgeDataV2 images were not found")
    images = images[args.image_offset : args.image_offset + args.max_images]
    if not images:
        raise RuntimeError(
            f"No BridgeDataV2 images selected for offset={args.image_offset}, "
            f"max_images={args.max_images}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = [
        {
            "index": args.image_offset + idx,
            "chunk_index": idx,
            "image_path": image_path,
            "source_dataset": "BridgeDataV2",
        }
        for idx, image_path in enumerate(images)
    ]
    write_jsonl(run_dir / "selected_images.jsonl", selected_rows)

    summary = runner.run(image_paths=images)
    summary_path = run_dir / "a_action_interface_full_loop_summary.json"
    summary_payload: Dict[str, Any] = {
        "run_id": runner.run_id,
        "run_dir": str(run_dir),
        "image_offset": args.image_offset,
        "selected_image_count": len(images),
        "selected_images_path": str(run_dir / "selected_images.jsonl"),
        "gsa_summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def _configure_optimized_runner(runner: GSAv0Runner, args: argparse.Namespace) -> None:
    input_cfg = runner.config.setdefault("input", {})
    input_cfg["max_objects_per_image"] = args.max_objects_per_image
    input_cfg["max_downstream_bbox_area_ratio"] = args.max_bbox_area_ratio
    input_cfg["downstream_exclude_object_names"] = args.exclude_object_name or []
    input_cfg["downstream_preferred_object_names"] = args.preferred_object_name or []

    runner.g_worker.use_consensus = bool(args.use_consensus_grounding)
    runner.a_action_interface.enable_repair = bool(args.enable_json_repair)
    if not args.online_all_context:
        runner.g_worker.simple_only = True
        runner.s_worker.vlm = None
        runner.a_state_pipeline.vlm_client = None
        runner.a_action_pipeline.vlm_client = None
        runner.a_identity_attr.vlm_client = None
        runner.a_obj_state.vlm_client = None
        runner.a_robot_state.vlm_client = None
        runner.a_action_precondition.vlm_client = None


def _write_review_and_promotion_outputs(
    run_dir: Path,
    decisions_path: Optional[Path],
) -> Dict[str, Any]:
    review_dir = run_dir / "review_queue" / "A_action_interface"
    reports_dir = run_dir / "reports"
    packs_dir = run_dir / "packs"

    raw_labels = load_action_interface_labels(run_dir)
    candidates = normalize_action_interface_candidates(raw_labels)
    candidates_path = review_dir / "A_action_interface_candidates.jsonl"
    template_path = review_dir / "review_decisions.template.jsonl"
    default_decisions_path = review_dir / "review_decisions.jsonl"
    html_path = reports_dir / "a_action_interface_review.html"
    report_path = reports_dir / "a_action_interface_promotion_report.json"
    pack_path = packs_dir / "A_action_interface_pack_manifest.jsonl"
    reviewed_candidates_path = review_dir / "A_action_interface_reviewed_candidates.jsonl"

    write_jsonl(candidates_path, candidates)
    write_jsonl(template_path, make_review_decision_template(candidates))
    write_review_html(html_path, candidates)

    active_decisions_path = decisions_path or (default_decisions_path if default_decisions_path.exists() else None)
    decisions = load_review_decisions(active_decisions_path)
    promotion = promote_action_interface_candidates(candidates, decisions)
    write_jsonl(reviewed_candidates_path, promotion.candidates)
    write_jsonl(pack_path, promotion.stable_labels)

    report = {
        "run_dir": str(run_dir),
        "candidates_path": str(candidates_path),
        "review_decisions_template_path": str(template_path),
        "active_decisions_path": str(active_decisions_path) if active_decisions_path else None,
        "review_html_path": str(html_path),
        "reviewed_candidates_path": str(reviewed_candidates_path),
        "stable_pack_path": str(pack_path),
        **promotion.summary,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["promotion_report_path"] = str(report_path)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
