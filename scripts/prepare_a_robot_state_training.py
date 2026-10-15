#!/usr/bin/env python3
"""Convert A_robot_state JSONL to Teacher SFT conversation format.

Usage:
    PYTHONPATH=. python scripts/prepare_a_robot_state_training.py \\
        --run-id gsa_v0_1234567890 \\
        --include-review   # optional: also convert review_required labels
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("prepare.a_robot_state")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.teacher_training.data_prep import gsa_label_to_conversation


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare A_robot_state SFT data")
    parser.add_argument("--output-root", default="data/output/gsa_v0")
    parser.add_argument("--run-id", default="", help="Run directory name; latest if empty")
    parser.add_argument(
        "--output",
        default="",
        help="Output jsonl path (default: {run}/packs/A_robot_state_sft.jsonl)",
    )
    parser.add_argument(
        "--labels-file",
        default="",
        help="Explicit input labels jsonl, e.g. {run}/labels/A_robot_state/A_robot_state_human.jsonl",
    )
    parser.add_argument(
        "--include-auto-verifiable",
        action="store_true",
        help="Experimental: include auto_verifiable labels even when stable_training_label is false",
    )
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Debug only: include review_required labels; not recommended for training",
    )
    args = parser.parse_args()

    out_root = Path(args.output_root)
    if args.run_id:
        run_dir = out_root / args.run_id
    else:
        runs = sorted(out_root.glob("gsa_v0_*"), key=lambda p: p.stat().st_mtime)
        if not runs:
            logger.error("No runs under %s", out_root)
            return 1
        run_dir = runs[-1]

    labels_path = Path(args.labels_file) if args.labels_file else (
        run_dir / "labels/A_robot_state/A_robot_state.jsonl"
    )
    if not labels_path.exists():
        logger.error("Missing %s", labels_path)
        return 1

    output_path = Path(args.output) if args.output else (
        run_dir / "packs/A_robot_state_sft.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    skipped = 0
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            label = json.loads(line)
            label.setdefault("bucket", "A_robot_state")
            is_stable = bool(label.get("stable_training_label"))
            is_auto = label.get("verifier_status") == "auto_verifiable"
            is_review = label.get("verifier_status") == "review_required"
            if not is_stable:
                if args.include_review and is_review:
                    pass
                elif args.include_auto_verifiable and is_auto:
                    pass
                else:
                    skipped += 1
                    continue
            conv = gsa_label_to_conversation(label)
            if conv and conv.get("conversations"):
                samples.append(conv)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Run: %s", run_dir.name)
    logger.info("Input labels: %s", labels_path)
    logger.info("Prepared %d samples (skipped %d)", len(samples), skipped)
    logger.info("Output: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
