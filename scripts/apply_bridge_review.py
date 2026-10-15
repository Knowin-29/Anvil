#!/usr/bin/env python3
"""Apply Bridge human review decisions to A_robot_state labels JSONL.

Reads review_decisions/bridge_review_decisions.jsonl and writes an updated
labels file with human-verified Bridge entries.

Usage:
    PYTHONPATH=. python scripts/apply_bridge_review.py --run-id gsa_v0_1780219771
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.data_flywheel.gsa.a_pipeline_schemas import validate_robot_state, ARobotStateLabel

logger = logging.getLogger("apply_bridge_review")


def resolve_run_dir(output_root: Path, run_id: str) -> Path:
    if run_id:
        run_dir = output_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    runs = sorted(output_root.glob("gsa_v0_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No runs under {output_root}")
    return runs[-1]


def load_decisions(path: Path) -> Dict[str, Dict[str, Any]]:
    """Keep latest decision per sample_id."""
    decisions: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return decisions
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if sample_id:
                decisions[sample_id] = row
    return decisions


def apply_decision(label: Dict[str, Any], decision: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    if decision.get("review_decision") != "approve":
        return label, False

    updated = dict(label)
    updated["state_type"] = decision.get("state_type", label.get("state_type"))
    updated["subtype"] = updated["state_type"]
    updated["state_value"] = decision.get("state_value", label.get("state_value"))
    target = decision.get("target_object_class")
    updated["target_object_class"] = target if target else None
    updated["verifier_status"] = "auto_verifiable"
    updated["label_status"] = "human_verified"
    updated["stable_training_label"] = True
    updated["confidence"] = 1.0
    updated["quality_flags"] = []
    if decision.get("reviewer"):
        updated["reviewer"] = decision["reviewer"]
    if decision.get("notes"):
        updated["review_notes"] = decision["notes"]

    probe = ARobotStateLabel(
        sample_id=updated.get("sample_id", ""),
        camera_view=updated.get("camera_view", "wrist"),
        end_effector=updated.get("end_effector", "parallel_jaw"),
        state_type=updated.get("state_type", ""),
        state_value=updated.get("state_value", ""),
        target_object_class=updated.get("target_object_class"),
    )
    result = validate_robot_state(probe)
    if not result.valid:
        logger.warning(
            "Invalid decision for %s: %s",
            updated.get("sample_id"),
            result.errors,
        )
        return label, False
    return updated, True


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Bridge review decisions")
    parser.add_argument("--output-root", default="data/output/gsa_v0")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--decisions-file",
        default="",
        help="Default: {run}/review_decisions/bridge_review_decisions.jsonl",
    )
    parser.add_argument(
        "--labels-in",
        default="",
        help="Default: {run}/labels/A_robot_state/A_robot_state.jsonl",
    )
    parser.add_argument(
        "--labels-out",
        default="",
        help="Default: {run}/labels/A_robot_state/A_robot_state_human.jsonl",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    run_dir = resolve_run_dir(ROOT / args.output_root, args.run_id)
    decisions_path = (
        Path(args.decisions_file)
        if args.decisions_file
        else run_dir / "review_decisions/bridge_review_decisions.jsonl"
    )
    labels_in = (
        Path(args.labels_in)
        if args.labels_in
        else run_dir / "labels/A_robot_state/A_robot_state.jsonl"
    )
    labels_out = (
        Path(args.labels_out)
        if args.labels_out
        else run_dir / "labels/A_robot_state/A_robot_state_human.jsonl"
    )

    decisions = load_decisions(decisions_path)
    if not decisions_path.exists():
        logger.error("Decisions file not found: %s", decisions_path)
        return 1

    approved = sum(1 for d in decisions.values() if d.get("review_decision") == "approve")
    skipped = sum(1 for d in decisions.values() if d.get("review_decision") == "skip")
    logger.info("Loaded %d decisions (%d approve, %d skip)", len(decisions), approved, skipped)

    labels: List[Dict[str, Any]] = []
    with open(labels_in, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                labels.append(json.loads(line))

    exact_state_samples = {
        label.get("sample_id")
        for label in labels
        if label.get("source_dataset") == "bridge_data_v2"
        and label.get("sample_id") in decisions
        and label.get("state_type") == decisions[label.get("sample_id")].get("state_type")
    }

    updated_labels: List[Dict[str, Any]] = []
    applied = 0
    legacy_applied: set[str] = set()
    for label in labels:
        sample_id = label.get("sample_id")
        decision = decisions.get(sample_id)
        if decision and label.get("source_dataset") == "bridge_data_v2":
            exact_match = sample_id in exact_state_samples
            legacy_match = sample_id not in exact_state_samples and sample_id not in legacy_applied
            if exact_match and label.get("state_type") == decision.get("state_type"):
                label, changed = apply_decision(label, decision)
                if changed:
                    applied += 1
            elif legacy_match:
                label, changed = apply_decision(label, decision)
                legacy_applied.add(sample_id)
                if changed:
                    applied += 1
        updated_labels.append(label)

    labels_out.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_out, "w", encoding="utf-8") as f:
        for label in updated_labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")

    stable = sum(
        1 for l in updated_labels
        if l.get("source_dataset") == "bridge_data_v2" and l.get("stable_training_label")
    )
    logger.info("Applied %d bridge approvals", applied)
    logger.info("Bridge stable_training_label count: %d", stable)
    logger.info("Output: %s", labels_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
