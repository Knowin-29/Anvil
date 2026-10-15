#!/usr/bin/env python3
"""Summarize human review and optionally run VLM-judge verifier eval."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.data_flywheel.gsa.a_action_interface_acceptance import (
    AActionInterfaceVLMJudge,
    evaluate_verifier_predictions,
    render_verifier_eval_markdown,
    run_acceptance_evaluation,
    run_vlm_judge,
)
from apps.data_flywheel.gsa.a_action_interface_review import read_jsonl, write_jsonl
from apps.data_flywheel.workers.vlm_client import VLMClient


DEFAULT_OUTPUT_DIR = ROOT / "data/output/gsa_v0/a_action_interface_overnight/acceptance_review_100"


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
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "acceptance_review_candidates.jsonl",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "review_decisions.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--run-vlm-judge", action="store_true")
    parser.add_argument("--max-vlm-cases", type=int, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/configs/gsa_v0.yaml")
    return parser.parse_args()


def main() -> int:
    args = _args()
    _load_env(ROOT / ".env")
    if not args.candidates.exists():
        print(f"ACCEPTANCE_EVAL_FAILED: candidates file not found: {args.candidates}")
        return 2

    decisions_path = args.decisions if args.decisions.exists() else None
    predictions_path = args.predictions
    result = run_acceptance_evaluation(
        candidates_path=args.candidates,
        decisions_path=decisions_path,
        output_dir=args.output_dir,
        predictions_path=predictions_path,
    )

    if args.run_vlm_judge:
        eval_set_path = Path(result["verifier_eval_set_path"])
        eval_set = read_jsonl(eval_set_path)
        if not eval_set:
            result["vlm_judge_status"] = "skipped_no_human_reviewed_eval_set"
        else:
            judge = AActionInterfaceVLMJudge(_make_vlm_client(args.config))
            predictions = run_vlm_judge(eval_set, judge, max_cases=args.max_vlm_cases)
            predictions_path = args.output_dir / "vlm_judge_predictions.jsonl"
            write_jsonl(predictions_path, predictions)
            verifier_report = evaluate_verifier_predictions(eval_set, predictions)
            report_path = args.output_dir / "vlm_judge_verifier_eval_report.json"
            markdown_path = args.output_dir / "vlm_judge_verifier_eval_report.md"
            report_path.write_text(json.dumps(verifier_report, ensure_ascii=False, indent=2), encoding="utf-8")
            markdown_path.write_text(render_verifier_eval_markdown(verifier_report), encoding="utf-8")
            result["vlm_judge_predictions_path"] = str(predictions_path)
            result["verifier_eval_report_path"] = str(report_path)
            result["verifier_eval_markdown_path"] = str(markdown_path)
            result["verifier_eval_report"] = verifier_report

    print("A_ACTION_INTERFACE_ACCEPTANCE_EVAL_OK")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _make_vlm_client(config_path: Path) -> VLMClient:
    cfg = _load_yaml(config_path)
    vlm_cfg: Dict[str, Any] = dict(cfg.get("vlm", {}))
    api_key = vlm_cfg.get("api_key")
    if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
        api_key = os.environ.get(api_key[2:-1])
    else:
        api_key = api_key or os.environ.get("VLM_API_KEY")
    return VLMClient(
        base_url=vlm_cfg.get("base_url", "http://localhost:10301"),
        model=vlm_cfg.get("model", "qwen2.5-vl-32b"),
        api_key=api_key,
        max_image_size=int(vlm_cfg.get("max_image_size", 960)),
        timeout=int(vlm_cfg.get("timeout", 300)),
        max_retries=int(vlm_cfg.get("max_retries", 3)),
    )


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    raise SystemExit(main())
