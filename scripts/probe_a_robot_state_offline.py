#!/usr/bin/env python3
"""Offline probe for A_robot_state — subtype coverage without VLM.

Usage:
    PYTHONPATH=. python scripts/probe_a_robot_state_offline.py \\
        --max-samples 500 \\
        --output-dir data/output/a_robot_state_probe
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("probe.a_robot_state")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.data_flywheel.gsa.data_adapters.robot_state_ingest import build_robot_state_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe A_robot_state ingest coverage")
    parser.add_argument("--config", default=str(ROOT / "scripts/configs/gsa_v0.yaml"))
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="data/output/a_robot_state_probe")
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rs_cfg = cfg.get("robot_state_datasets", {})
    enabled = rs_cfg.get("enabled") or ["sharerobot"]
    overrides = rs_cfg.get("nas_overrides", {})
    max_per = min(args.max_samples, rs_cfg.get("max_samples_per_source", 2000))

    registry = build_robot_state_registry()
    labels = registry.ingest_all(enabled, overrides, max_per)

    subtype_counts = Counter(l.state_type for l in labels)
    source_counts = Counter(l.source_dataset for l in labels)
    verifier_counts = Counter(l.verifier_status for l in labels)

    report = {
        "enabled_sources": enabled,
        "total_labels": len(labels),
        "subtype_counts": dict(subtype_counts),
        "source_counts": dict(source_counts),
        "verifier_status_counts": dict(verifier_counts),
        "subtypes_missing": [
            st for st in (
                "gripper_empty",
                "holding_object",
                "object_between_fingers",
                "contact_state",
                "tool_ready",
            )
            if subtype_counts.get(st, 0) == 0
        ],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    jsonl_path = out_dir / "A_robot_state_candidates.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for label in labels:
            d = label.to_dict()
            d["bucket"] = "A_robot_state"
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    logger.info("Total labels: %d", len(labels))
    logger.info("Subtype counts: %s", dict(subtype_counts))
    logger.info("Report: %s", out_dir / "probe_report.json")
    logger.info("JSONL: %s", jsonl_path)
    if report["subtypes_missing"]:
        logger.warning("Missing subtypes: %s", report["subtypes_missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
