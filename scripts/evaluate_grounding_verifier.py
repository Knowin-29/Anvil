"""Direct G_grounding verifier pass-rate evaluator.

This CLI bypasses the generator and archival scanner. It accepts user-provided
image paths, object names, and bboxes, then reports each verifier rule's pass
rate using the same rule implementation as verifier_service.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from verifier_service import (
    SPECIALIST_MODES,
    _deep_merge_dict,
    evaluate_grounding_batch,
    load_grounding_verifier_config,
)


def load_labels(input_path: Path) -> list[dict[str, Any]]:
    """Load labels from JSONL, a JSON array, or a JSON object with items."""
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if input_path.suffix.lower() == ".jsonl":
        labels = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            labels.append(item)
        return labels

    payload = json.loads(text)
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("JSON array items must be objects")
        return payload
    if isinstance(payload, dict):
        items = payload.get("items")
        if items is None:
            return [payload]
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise ValueError("JSON object field 'items' must be a list of objects")
        return items
    raise ValueError("Input must be JSONL, a JSON array, or a JSON object")


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    override: dict[str, Any] = {}
    cursor = override
    for key in path[:-1]:
        cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value
    return _deep_merge_dict(config, override)


def _flatten_diff(base: dict[str, Any], candidate: dict[str, Any], prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key, candidate_value in candidate.items():
        path = prefix + (key,)
        base_value = base.get(key)
        if isinstance(candidate_value, dict) and isinstance(base_value, dict):
            diff.update(_flatten_diff(base_value, candidate_value, path))
        elif candidate_value != base_value:
            diff[".".join(path)] = {"from": base_value, "to": candidate_value}
    return diff


def generate_search_configs(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate conservative G_grounding tuning candidates for target pass-rate search."""
    search_space: dict[tuple[str, ...], list[Any]] = {
        ("tag_thresholds", "verified_auto"): [0.85, 0.80, 0.75, 0.70, 0.65],
        ("content_quality", "generic_name_score"): [0.0, 0.3, 0.6, 1.0],
        ("content_quality", "single_short_word_score"): [0.6, 0.8, 1.0],
        ("geometry_valid", "too_large_area_ratio"): [0.40, 0.50, 0.60],
        ("geometry_valid", "large_area_ratio"): [0.25, 0.30, 0.40],
        ("confidence_check", "template_score"): [0.6, 0.8, 1.0],
        ("confidence_check", "medium_confidence_score"): [0.7, 0.85, 1.0],
    }

    paths = list(search_space)
    configs = []
    for values in itertools.product(*(search_space[path] for path in paths)):
        candidate = base_config
        for path, value in zip(paths, values):
            candidate = _set_nested(candidate, path, value)
        configs.append(candidate)
    return configs


def auto_search_grounding_config(
    labels: list[dict[str, Any]],
    base_config: dict[str, Any],
    target_passrate: float,
    specialist_mode: str = "off",
    rule_pass_threshold: float = 1.0,
) -> dict[str, Any]:
    """Return the candidate config whose verified_auto rate is closest to target_passrate."""
    best: dict[str, Any] | None = None
    candidates = generate_search_configs(base_config)
    for candidate in candidates:
        report = evaluate_grounding_batch(
            labels,
            specialist_mode=specialist_mode,
            rule_pass_threshold=rule_pass_threshold,
            config=candidate,
        )
        actual = report["overall"]["verified_auto"]["rate"]
        distance = abs(actual - target_passrate)
        score = (distance, len(_flatten_diff(base_config, candidate)))
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "target_passrate": target_passrate,
                "actual_passrate": actual,
                "distance": round(distance, 6),
                "candidates_evaluated": len(candidates),
                "recommended_config_diff": _flatten_diff(base_config, candidate),
                "recommended_report": {
                    "overall": report["overall"],
                    "rule_passrate": report["rule_passrate"],
                    "cumulative_passrate": report["cumulative_passrate"],
                    "specialist_passrate": report["specialist_passrate"],
                },
            }
    if best is None:
        raise ValueError("No candidate configurations generated")
    best.pop("score")
    return best


def search_grounding_config_for_target(
    labels: list[dict[str, Any]],
    base_config: dict[str, Any],
    target_passrate: float,
    specialist_mode: str = "off",
    rule_pass_threshold: float = 1.0,
) -> dict[str, Any]:
    """Compatibility wrapper returning a compact best-candidate structure."""
    best = auto_search_grounding_config(
        labels,
        base_config,
        target_passrate,
        specialist_mode=specialist_mode,
        rule_pass_threshold=rule_pass_threshold,
    )
    direct_diff = {
        path: change["to"]
        for path, change in best["recommended_config_diff"].items()
    }
    return {
        "best": {
            "target_passrate": best["target_passrate"],
            "actual_passrate": best["actual_passrate"],
            "distance": best["distance"],
            "diff": direct_diff,
            "report": best["recommended_report"],
        },
        "candidates_evaluated": best["candidates_evaluated"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct G_grounding verifier pass rates.")
    parser.add_argument("--input", required=True, type=Path, help="Input .jsonl or .json file.")
    parser.add_argument("--output", type=Path, help="Optional output JSON path. Defaults to stdout.")
    parser.add_argument("--config", type=Path, help="Optional grounding verifier tuning YAML.")
    parser.add_argument(
        "--specialist",
        default="off",
        choices=sorted(SPECIALIST_MODES),
        help="Specialist cross-verification mode.",
    )
    parser.add_argument(
        "--rule-pass-threshold",
        type=float,
        default=1.0,
        help="Minimum per-rule score counted as passing that rule.",
    )
    parser.add_argument("--target-passrate", type=float, help="Desired verified_auto passrate, e.g. 0.75.")
    parser.add_argument("--auto-search", action="store_true", help="Search conservative config candidates near target passrate.")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON instead of pretty JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = load_labels(args.input)
    config = load_grounding_verifier_config(args.config)
    report = evaluate_grounding_batch(
        labels,
        specialist_mode=args.specialist,
        rule_pass_threshold=args.rule_pass_threshold,
        config=config,
    )
    output: dict[str, Any] = {"report": report}
    if args.auto_search:
        if args.target_passrate is None:
            raise ValueError("--auto-search requires --target-passrate")
        output["auto_search"] = auto_search_grounding_config(
            labels,
            config,
            args.target_passrate,
            specialist_mode=args.specialist,
            rule_pass_threshold=args.rule_pass_threshold,
        )

    indent = None if args.compact else 2
    text = json.dumps(output, ensure_ascii=False, indent=indent)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
