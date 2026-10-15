#!/usr/bin/env python3
"""Render verified ICLR claim evidence into editable tables and figures.

The renderer accepts only an ``iclr-claim-evidence-v3`` payload whose identity
and live compiler binding validate. It writes a fresh, atomic artifact bundle
and never edits manuscript sources. When v5.2 evidence is unavailable, the
bundle contains only General Retention outputs; no placeholder paired result is
created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import compile_iclr_claim_evidence as compiler  # noqa: E402


RENDER_PROTOCOL = "iclr-claim-artifacts-v3"
CONTRACT_PATH = ROOT / "experiments/claim_evidence_visual_contract.yaml"
DATASET_ORDER = ("ai2d_test", "scienceqa_test", "textvqa_val")
DATASET_LABELS = {
    "ai2d_test": "AI2D",
    "scienceqa_test": "ScienceQA",
    "textvqa_val": "TextVQA",
}
EXPECTED_DATASET_COUNTS = {
    "ai2d_test": 3088,
    "scienceqa_test": 2017,
    "textvqa_val": 5000,
}
V52_METRICS = {
    "g_valid_pair_coverage",
    "g_both_box_iou_at_0_5",
    "g_mean_min_pair_iou",
    "s_oracle_accuracy",
    "s_oracle_parse_coverage",
    "s_absent_accuracy",
    "s_absent_parse_coverage",
    "s_corrupted_accuracy",
    "s_corrupted_parse_coverage",
    "s_pred_accuracy_given_emitted_request",
    "s_pred_parse_coverage_all_scenes",
    "s_pred_correct_all_scenes_invalid_g_zero",
    "s_pred_strict_e2e_accuracy",
    "paired_dependency_gap_oracle_minus_predicted",
    "paired_context_gain_oracle_minus_absent",
    "paired_context_gain_oracle_minus_corrupted",
    "positive_oracle_headroom",
    "bounded_oracle_recovery",
}
PREDICTED_TRANSITIONS = {
    "oracle_correct_to_predicted_correct",
    "oracle_correct_to_predicted_wrong",
    "oracle_wrong_to_predicted_correct",
    "oracle_wrong_to_predicted_wrong",
}
STRICT_E2E_TRANSITIONS = {
    "oracle_correct_to_e2e_correct",
    "oracle_correct_to_e2e_wrong",
    "oracle_wrong_to_e2e_correct",
    "oracle_wrong_to_e2e_wrong",
}
INK = "#202124"
GROUNDING = "#0072B2"
SPATIAL = "#009E73"
STRESS = "#E69F00"
LOSS = "#A33A3A"
RULE = "#C7CBD1"
NEUTRAL = "#6B7075"


class RenderError(ValueError):
    """Raised when evidence is not admissible for publication rendering."""


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RenderError(f"missing required file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _file_hash(path)}


def _relative_binding(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": _file_hash(path),
    }


def _load_json_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RenderError(f"missing claim evidence: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RenderError("invalid claim evidence UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise RenderError(f"invalid claim evidence JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise RenderError("claim evidence must be a JSON object")
    binding = {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return value, binding


def _finite(value: Any, label: str, *, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RenderError(f"{label} must be finite")
    if low is not None and number < low:
        raise RenderError(f"{label} is below {low}")
    if high is not None and number > high:
        raise RenderError(f"{label} is above {high}")
    return number


def _integer(value: Any, label: str, *, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise RenderError(f"{label} must be an integer >= {low}")
    return value


def _count_from_rate(value: float, denominator: int, label: str) -> int:
    count = round(value * denominator)
    if not math.isclose(value, count / denominator, abs_tol=1e-6, rel_tol=0.0):
        raise RenderError(f"{label} is not an exact count/{denominator} rate")
    return count


def _ci(value: Any, label: str, *, low: float, high: float) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RenderError(f"{label} must contain two endpoints")
    left = _finite(value[0], f"{label}[0]", low=low, high=high)
    right = _finite(value[1], f"{label}[1]", low=low, high=high)
    if left > right:
        raise RenderError(f"{label} endpoints are reversed")
    return left, right


def _verify_compiler_binding(payload: dict[str, Any]) -> None:
    observed = payload.get("compiler")
    live = _file_binding(Path(compiler.__file__))
    if observed != live:
        raise RenderError("claim evidence compiler binding is stale or foreign")


def _verify_general(general: Any) -> None:
    if not isinstance(general, dict):
        raise RenderError("General Retention evidence is missing")
    if (
        general.get("availability") != "available"
        or general.get("admission") != "PAIRED_GENERAL_RETENTION_ADMITTED"
    ):
        raise RenderError("General Retention evidence is not admitted")
    counts = general.get("dataset_counts")
    if counts != EXPECTED_DATASET_COUNTS or general.get("records") != sum(counts.values()):
        raise RenderError("General Retention dataset counts drift")
    datasets = general.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASET_ORDER):
        raise RenderError("General Retention dataset coverage drift")

    base_scores: list[float] = []
    e1_scores: list[float] = []
    for dataset in DATASET_ORDER:
        cell = datasets[dataset]
        if not isinstance(cell, dict) or cell.get("n") != EXPECTED_DATASET_COUNTS[dataset]:
            raise RenderError(f"{dataset}: row count drift")
        base = _finite(cell.get("base_score_percent"), f"{dataset}.base", low=0, high=100)
        e1 = _finite(cell.get("e1_score_percent"), f"{dataset}.e1", low=0, high=100)
        delta = _finite(cell.get("e1_minus_base_points"), f"{dataset}.delta", low=-100, high=100)
        if not math.isclose(delta, e1 - base, abs_tol=1e-6, rel_tol=0.0):
            raise RenderError(f"{dataset}: delta does not equal E1 minus Base")
        _ci(cell.get("e1_minus_base_points_ci95"), f"{dataset}.ci95", low=-100, high=100)
        retention = cell.get("retention_ratio_percent")
        if base == 0:
            if retention is not None:
                raise RenderError(f"{dataset}: retention must be undefined when Base is zero")
        else:
            retention_value = _finite(retention, f"{dataset}.retention", low=0)
            if not math.isclose(retention_value, 100.0 * e1 / base, abs_tol=1e-6, rel_tol=0.0):
                raise RenderError(f"{dataset}: retention ratio drift")
        transitions = cell.get("paired_transitions")
        if not isinstance(transitions, dict) or set(transitions) != {"improved", "regressed", "unchanged"}:
            raise RenderError(f"{dataset}: transition schema drift")
        if sum(_integer(transitions[key], f"{dataset}.{key}") for key in transitions) != cell["n"]:
            raise RenderError(f"{dataset}: paired transitions do not sum to n")
        validity = cell.get("output_validity")
        required_validity = {
            "base_percent",
            "e1_percent",
            "base_malformed_count",
            "e1_malformed_count",
            "base_truncation_count",
            "e1_truncation_count",
            "base_health_admitted",
            "e1_health_admitted",
        }
        if not isinstance(validity, dict) or set(validity) != required_validity:
            raise RenderError(f"{dataset}: output-validity schema drift")
        base_validity = _finite(validity["base_percent"], f"{dataset}.base_validity", low=0, high=100)
        e1_validity = _finite(validity["e1_percent"], f"{dataset}.e1_validity", low=0, high=100)
        count_keys = required_validity - {
            "base_percent",
            "e1_percent",
            "base_health_admitted",
            "e1_health_admitted",
        }
        validity_counts = {
            key: _integer(validity[key], f"{dataset}.{key}")
            for key in count_keys
        }
        if any(count > cell["n"] for count in validity_counts.values()):
            raise RenderError(f"{dataset}: output-health count exceeds n")
        expected_base_validity = (
            100.0 * (cell["n"] - validity_counts["base_malformed_count"]) / cell["n"]
        )
        expected_e1_validity = (
            100.0 * (cell["n"] - validity_counts["e1_malformed_count"]) / cell["n"]
        )
        if not math.isclose(base_validity, expected_base_validity, abs_tol=1e-6, rel_tol=0.0):
            raise RenderError(f"{dataset}: Base validity disagrees with malformed count")
        if not math.isclose(e1_validity, expected_e1_validity, abs_tol=1e-6, rel_tol=0.0):
            raise RenderError(f"{dataset}: E1 validity disagrees with malformed count")
        if validity["base_health_admitted"] is not True or validity["e1_health_admitted"] is not True:
            raise RenderError(f"{dataset}: output health is not admitted")
        base_scores.append(base)
        e1_scores.append(e1)

    macro = general.get("macro")
    if not isinstance(macro, dict):
        raise RenderError("General Retention macro is missing")
    base_macro = _finite(macro.get("base_general_avg_percent"), "macro.base", low=0, high=100)
    e1_macro = _finite(macro.get("e1_general_avg_percent"), "macro.e1", low=0, high=100)
    delta_macro = _finite(macro.get("e1_minus_base_points"), "macro.delta", low=-100, high=100)
    if not math.isclose(base_macro, sum(base_scores) / len(base_scores), abs_tol=1e-6, rel_tol=0.0):
        raise RenderError("General Retention Base macro drift")
    if not math.isclose(e1_macro, sum(e1_scores) / len(e1_scores), abs_tol=1e-6, rel_tol=0.0):
        raise RenderError("General Retention E1 macro drift")
    if not math.isclose(delta_macro, e1_macro - base_macro, abs_tol=1e-6, rel_tol=0.0):
        raise RenderError("General Retention macro delta drift")
    _ci(macro.get("e1_minus_base_points_ci95"), "macro.ci95", low=-100, high=100)


def _verify_v52(v52: Any) -> None:
    if not isinstance(v52, dict):
        raise RenderError("v5.2 evidence is missing")
    availability = v52.get("availability")
    if availability == "unavailable":
        if not isinstance(v52.get("status_state"), str) or not isinstance(v52.get("reason"), str):
            raise RenderError("unavailable v5.2 evidence lacks its exact disposition")
        return
    if availability != "available" or v52.get("admission") != "FORMAL_SCORE_COMPLETE":
        raise RenderError("v5.2 evidence is neither unavailable nor formally admitted")
    if v52.get("records") != 40:
        raise RenderError("v5.2 evidence is not exact40")
    metrics = v52.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != V52_METRICS:
        raise RenderError("v5.2 metric coverage drift")
    optional_metrics = {"s_pred_accuracy_given_emitted_request", "bounded_oracle_recovery"}
    for key, value in metrics.items():
        if key in optional_metrics and value is None:
            continue
        low = -1.0 if key.startswith("paired_") else 0.0
        _finite(value, f"v5.2.{key}", low=low, high=1.0)
    gap = float(metrics["paired_dependency_gap_oracle_minus_predicted"])
    oracle = float(metrics["s_oracle_accuracy"])
    absent = float(metrics["s_absent_accuracy"])
    corrupted = float(metrics["s_corrupted_accuracy"])
    predicted = float(metrics["s_pred_correct_all_scenes_invalid_g_zero"])
    e2e = float(metrics["s_pred_strict_e2e_accuracy"])
    if not math.isclose(gap, oracle - predicted, abs_tol=1e-6, rel_tol=0.0):
        raise RenderError("v5.2 dependency gap drift")
    if not math.isclose(
        float(metrics["paired_context_gain_oracle_minus_absent"]),
        oracle - absent,
        abs_tol=1e-6,
        rel_tol=0.0,
    ):
        raise RenderError("v5.2 absent-context gain drift")
    if not math.isclose(
        float(metrics["paired_context_gain_oracle_minus_corrupted"]),
        oracle - corrupted,
        abs_tol=1e-6,
        rel_tol=0.0,
    ):
        raise RenderError("v5.2 corrupted-context gain drift")
    if not math.isclose(float(metrics["positive_oracle_headroom"]), max(0.0, gap), abs_tol=1e-6, rel_tol=0.0):
        raise RenderError("v5.2 positive headroom drift")
    expected_recovery = min(1.0, max(0.0, predicted / oracle)) if oracle else None
    observed_recovery = metrics["bounded_oracle_recovery"]
    if expected_recovery is None:
        if observed_recovery is not None:
            raise RenderError("v5.2 recovery must be undefined when S-oracle accuracy is zero")
    elif observed_recovery is None or not math.isclose(
        float(observed_recovery), expected_recovery, abs_tol=1e-6, rel_tol=0.0
    ):
        raise RenderError("v5.2 bounded recovery drift")
    transitions = v52.get("transition_counts")
    if not isinstance(transitions, dict) or not set(transitions).issubset(PREDICTED_TRANSITIONS):
        raise RenderError("v5.2 predicted-context transition schema drift")
    if sum(_integer(value, f"transition.{key}") for key, value in transitions.items()) != 40:
        raise RenderError("v5.2 transitions do not sum to 40")
    counts = {key: int(transitions.get(key, 0)) for key in PREDICTED_TRANSITIONS}
    oracle_from_transitions = (
        counts["oracle_correct_to_predicted_correct"]
        + counts["oracle_correct_to_predicted_wrong"]
    ) / 40
    predicted_from_transitions = (
        counts["oracle_correct_to_predicted_correct"]
        + counts["oracle_wrong_to_predicted_correct"]
    ) / 40
    if not math.isclose(oracle, oracle_from_transitions, abs_tol=1e-6, rel_tol=0.0):
        raise RenderError("v5.2 transitions disagree with S-oracle accuracy")
    if not math.isclose(predicted, predicted_from_transitions, abs_tol=1e-6, rel_tol=0.0):
        raise RenderError("v5.2 transitions disagree with predicted-context accuracy")
    strict_transitions = v52.get("strict_e2e_transition_counts")
    if not isinstance(strict_transitions, dict) or not set(strict_transitions).issubset(
        STRICT_E2E_TRANSITIONS
    ):
        raise RenderError("v5.2 strict-E2E transition schema drift")
    if sum(
        _integer(value, f"strict_transition.{key}")
        for key, value in strict_transitions.items()
    ) != 40:
        raise RenderError("v5.2 strict-E2E transitions do not sum to 40")
    strict_counts = {
        key: int(strict_transitions.get(key, 0)) for key in STRICT_E2E_TRANSITIONS
    }
    strict_oracle = (
        strict_counts["oracle_correct_to_e2e_correct"]
        + strict_counts["oracle_correct_to_e2e_wrong"]
    ) / 40
    strict_e2e = (
        strict_counts["oracle_correct_to_e2e_correct"]
        + strict_counts["oracle_wrong_to_e2e_correct"]
    ) / 40
    if not math.isclose(oracle, strict_oracle, abs_tol=1e-6, rel_tol=0.0):
        raise RenderError("v5.2 strict-E2E transitions disagree with S-oracle accuracy")
    if not math.isclose(e2e, strict_e2e, abs_tol=1e-6, rel_tol=0.0):
        raise RenderError("v5.2 strict-E2E transitions disagree with strict E2E accuracy")
    g_valid = float(metrics["g_valid_pair_coverage"])
    g_iou = float(metrics["g_both_box_iou_at_0_5"])
    pred_parse = float(metrics["s_pred_parse_coverage_all_scenes"])
    pred_all = float(metrics["s_pred_correct_all_scenes_invalid_g_zero"])
    oracle_parse = float(metrics["s_oracle_parse_coverage"])
    absent_parse = float(metrics["s_absent_parse_coverage"])
    corrupted_parse = float(metrics["s_corrupted_parse_coverage"])
    mean_min_iou = float(metrics["g_mean_min_pair_iou"])
    conditional_pred = metrics["s_pred_accuracy_given_emitted_request"]
    if g_iou > g_valid + 1e-6 or pred_parse > g_valid + 1e-6:
        raise RenderError("v5.2 downstream coverage exceeds valid grounding coverage")
    if e2e > g_iou + 1e-6 or e2e > pred_all + 1e-6:
        raise RenderError("v5.2 strict E2E exceeds a required component")
    if (
        pred_all > pred_parse + 1e-6
        or oracle > oracle_parse + 1e-6
        or absent > absent_parse + 1e-6
        or corrupted > corrupted_parse + 1e-6
    ):
        raise RenderError("v5.2 accuracy exceeds parse coverage")
    if mean_min_iou > g_valid + 1e-6:
        raise RenderError("v5.2 mean minimum IoU exceeds valid grounding coverage")
    exact40_rates = {
        "g_valid_pair_coverage": g_valid,
        "g_both_box_iou_at_0_5": g_iou,
        "s_oracle_accuracy": oracle,
        "s_oracle_parse_coverage": oracle_parse,
        "s_absent_accuracy": absent,
        "s_absent_parse_coverage": absent_parse,
        "s_corrupted_accuracy": corrupted,
        "s_corrupted_parse_coverage": corrupted_parse,
        "s_pred_parse_coverage_all_scenes": pred_parse,
        "s_pred_correct_all_scenes_invalid_g_zero": pred_all,
        "s_pred_strict_e2e_accuracy": e2e,
        "paired_dependency_gap_oracle_minus_predicted": gap,
        "positive_oracle_headroom": float(metrics["positive_oracle_headroom"]),
    }
    exact_counts = {
        key: _count_from_rate(value, 40, f"v5.2.{key}")
        for key, value in exact40_rates.items()
    }
    if g_valid == 0:
        if conditional_pred is not None or pred_all != 0 or pred_parse != 0:
            raise RenderError("v5.2 emitted-request metrics drift when no G request is valid")
    elif conditional_pred is None:
        raise RenderError("v5.2 conditional S-pred accuracy is undefined despite valid G requests")
    else:
        emitted_correct = _count_from_rate(
            float(conditional_pred), exact_counts["g_valid_pair_coverage"],
            "v5.2.s_pred_accuracy_given_emitted_request",
        )
        if emitted_correct != exact_counts["s_pred_correct_all_scenes_invalid_g_zero"]:
            raise RenderError("v5.2 conditional S-pred accuracy does not match all-scene credit")
    bootstrap = v52.get("scene_bootstrap")
    if (
        not isinstance(bootstrap, dict)
        or bootstrap.get("metric") != "paired_dependency_gap_oracle_minus_predicted"
        or bootstrap.get("unit") != "scene"
        or bootstrap.get("paired_scenes") != 40
        or bootstrap.get("confidence") != 0.95
    ):
        raise RenderError("v5.2 scene-bootstrap contract drift")
    _integer(bootstrap.get("samples"), "v5.2.bootstrap.samples", low=1)
    _integer(bootstrap.get("seed"), "v5.2.bootstrap.seed", low=0)
    _ci(bootstrap.get("ci"), "v5.2.bootstrap.ci", low=-1, high=1)
    controls = v52.get("context_control_bootstrap")
    expected_controls = {
        "oracle_minus_absent": "paired_context_gain_oracle_minus_absent",
        "oracle_minus_corrupted": "paired_context_gain_oracle_minus_corrupted",
    }
    if not isinstance(controls, dict) or set(controls) != set(expected_controls):
        raise RenderError("v5.2 context-control bootstrap coverage drift")
    for key, metric in expected_controls.items():
        cell = controls[key]
        if (
            not isinstance(cell, dict)
            or cell.get("metric") != metric
            or cell.get("unit") != "scene"
            or cell.get("paired_scenes") != 40
            or cell.get("confidence") != 0.95
            or cell.get("samples") != bootstrap.get("samples")
            or cell.get("seed") != bootstrap.get("seed")
        ):
            raise RenderError(f"v5.2 {key} bootstrap contract drift")
        _ci(cell.get("ci"), f"v5.2.{key}.ci", low=-1, high=1)
    parse_status = v52.get("parse_status")
    expected_parse_tracks = {"G", "S_oracle", "S_absent", "S_corrupted", "S_pred"}
    if not isinstance(parse_status, dict) or set(parse_status) != expected_parse_tracks:
        raise RenderError("v5.2 parse-status schema drift")
    for track, counts in parse_status.items():
        if not isinstance(counts, dict) or sum(_integer(v, f"parse.{track}.{k}") for k, v in counts.items()) != 40:
            raise RenderError(f"v5.2 {track} parse counts do not sum to 40")
    if int(parse_status["G"].get("parsed_exact", 0)) != exact_counts["g_valid_pair_coverage"]:
        raise RenderError("v5.2 G parse counts disagree with valid coverage")
    if int(parse_status["S_oracle"].get("parsed_exact", 0)) != exact_counts["s_oracle_parse_coverage"]:
        raise RenderError("v5.2 S-oracle parse counts disagree with parse coverage")
    if int(parse_status["S_absent"].get("parsed_exact", 0)) != exact_counts["s_absent_parse_coverage"]:
        raise RenderError("v5.2 S-absent parse counts disagree with parse coverage")
    if int(parse_status["S_corrupted"].get("parsed_exact", 0)) != exact_counts["s_corrupted_parse_coverage"]:
        raise RenderError("v5.2 S-corrupted parse counts disagree with parse coverage")
    if int(parse_status["S_pred"].get("parsed_exact", 0)) != exact_counts["s_pred_parse_coverage_all_scenes"]:
        raise RenderError("v5.2 S-pred parse counts disagree with parse coverage")
    caps = v52.get("token_cap_counts")
    if not isinstance(caps, dict) or set(caps) != expected_parse_tracks:
        raise RenderError("v5.2 token-cap schema drift")
    for track, count in caps.items():
        if _integer(count, f"caps.{track}") > 40:
            raise RenderError(f"v5.2 {track} token-cap count exceeds 40")
        if count != int(parse_status[track].get("hit_token_cap", 0)):
            raise RenderError(f"v5.2 {track} token-cap count disagrees with parse status")
    attribution = v52.get("attribution")
    states = ["Attributed(G)", "Local(S)", "Inconclusive"]
    if not isinstance(attribution, dict) or attribution.get("states") != states:
        raise RenderError("v5.2 attribution-state schema drift")
    attribution_counts = attribution.get("counts")
    if not isinstance(attribution_counts, dict) or set(attribution_counts) != set(states):
        raise RenderError("v5.2 attribution count schema drift")
    if sum(_integer(attribution_counts[state], f"attribution.{state}") for state in states) != 40:
        raise RenderError("v5.2 attribution counts do not sum to 40")
    if attribution.get("conclusive_failures") != (
        attribution_counts["Attributed(G)"] + attribution_counts["Local(S)"]
    ):
        raise RenderError("v5.2 conclusive attribution count drift")
    if attribution_counts["Attributed(G)"] > int(
        transitions.get("oracle_correct_to_predicted_wrong", 0)
    ):
        raise RenderError("v5.2 Attributed(G) exceeds oracle-repaired failures")
    if attribution_counts["Attributed(G)"] > 40 - exact_counts["g_both_box_iou_at_0_5"]:
        raise RenderError("v5.2 Attributed(G) exceeds failed-G scenes")
    if attribution_counts["Local(S)"] > int(
        transitions.get("oracle_wrong_to_predicted_wrong", 0)
    ):
        raise RenderError("v5.2 Local(S) exceeds oracle-context S failures")
    if attribution_counts["Local(S)"] > exact_counts["g_both_box_iou_at_0_5"]:
        raise RenderError("v5.2 Local(S) exceeds correct-G scenes")
    reasons = attribution.get("reason_counts")
    if not isinstance(reasons, dict) or sum(
        _integer(value, f"attribution_reason.{key}") for key, value in reasons.items()
    ) != 40:
        raise RenderError("v5.2 attribution reasons do not sum to 40")
    policy = attribution.get("policy")
    if not isinstance(policy, dict) or policy != _file_binding(Path(str(policy.get("path", "")))):
        raise RenderError("v5.2 attribution policy binding drift")


def _verify_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1 or payload.get("protocol") != compiler.EVIDENCE_PROTOCOL:
        raise RenderError("unsupported claim-evidence protocol")
    if payload.get("status") != "CLAIM_EVIDENCE_VERIFIED":
        raise RenderError("claim evidence is not verified")
    identity = payload.get("identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    if not isinstance(identity, str) or compiler.content_hash(unsigned) != identity:
        raise RenderError("claim-evidence identity mismatch")
    _verify_compiler_binding(payload)
    policy = payload.get("publication_policy")
    if policy != {
        "paper_was_not_modified": True,
        "console_logs_used_as_evidence": False,
        "missing_values_filled": False,
        "only_deeply_recomputed_values_exported": True,
    }:
        raise RenderError("claim-evidence publication policy drift")
    _verify_general(payload.get("general_retention"))
    _verify_v52(payload.get("scannet_v52_paired_gs"))


def _verified_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, binding = _load_json_snapshot(path)
    _verify_payload(payload)
    return payload, binding


def verify_evidence(path: Path) -> dict[str, Any]:
    payload, _ = _verified_snapshot(path)
    return payload


def _fmt(value: float, digits: int = 2, *, signed: bool = False) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:{'+' if signed else ''}.{digits}f}"


def _general_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    general = payload["general_retention"]
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        cell = general["datasets"][dataset]
        rows.append({"label": DATASET_LABELS[dataset], **cell})
    macro = general["macro"]
    totals = {key: 0 for key in ("improved", "regressed", "unchanged")}
    for cell in general["datasets"].values():
        for key in totals:
            totals[key] += cell["paired_transitions"][key]
    rows.append(
        {
            "label": "Macro",
            "n": None,
            "base_score_percent": macro["base_general_avg_percent"],
            "e1_score_percent": macro["e1_general_avg_percent"],
            "e1_minus_base_points": macro["e1_minus_base_points"],
            "e1_minus_base_points_ci95": macro["e1_minus_base_points_ci95"],
            "output_validity": None,
            "paired_transitions": None,
            "total_records": general["records"],
            "total_transitions": totals,
        }
    )
    return rows


def render_general_table(payload: dict[str, Any]) -> str:
    lines = [
        "% Generated from verified claim evidence; do not edit numeric cells.",
        "\\begingroup",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\noindent",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrccc@{}}",
        "\\toprule",
        "Dataset & $n$ & Base (\\%) & E1 (\\%) & \\shortstack{$\\Delta$ (pp) \\\\ {[95\\% CI]}} & Valid B/E (\\%) & I/R/U \\\\",
        "\\midrule",
    ]
    for index, row in enumerate(_general_rows(payload)):
        ci = row["e1_minus_base_points_ci95"]
        validity = row["output_validity"]
        valid_text = "--" if validity is None else (
            f"{_fmt(float(validity['base_percent']), 1)}/{_fmt(float(validity['e1_percent']), 1)}"
        )
        transition = row["paired_transitions"]
        if index == 3:
            lines.append("\\midrule")
        delta_ci = "\\shortstack{{{delta} \\\\ {{[{lo}, {hi}]}}}}".format(
            delta=_fmt(float(row["e1_minus_base_points"]), signed=True),
            lo=_fmt(float(ci[0]), signed=True),
            hi=_fmt(float(ci[1]), signed=True),
        )
        lines.append(
            "{label} & {n} & {base} & {e1} & {delta_ci} & {valid} & {transitions} \\\\".format(
                label="\\textbf{Macro}$^{\\dagger}$" if index == 3 else row["label"],
                n="--" if row["n"] is None else f"{row['n']:,}",
                base=_fmt(float(row["base_score_percent"])),
                e1=_fmt(float(row["e1_score_percent"])),
                delta_ci=delta_ci,
                valid=valid_text,
                transitions=(
                    "--"
                    if transition is None
                    else f"{transition['improved']}/{transition['regressed']}/{transition['unchanged']}"
                ),
            )
        )
    total = _general_rows(payload)[-1]
    totals = total["total_transitions"]
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "{\\footnotesize $^{\\dagger}$Unweighted across datasets. "
            f"Total paired items: {total['total_records']:,}; I/R/U: "
            f"{totals['improved']}/{totals['regressed']}/{totals['unchanged']}.\\par}}",
            "\\endgroup",
            "",
        ]
    )
    return "\n".join(lines)


def render_v52_table(payload: dict[str, Any]) -> str:
    v52 = payload["scannet_v52_paired_gs"]
    if v52["availability"] != "available":
        raise RenderError("refusing to render a v5.2 table without formal evidence")
    metrics = v52["metrics"]
    ci = v52["scene_bootstrap"]["ci"]
    control_ci = v52["context_control_bootstrap"]
    transitions = v52["transition_counts"]
    strict_transitions = v52["strict_e2e_transition_counts"]
    attribution = v52["attribution"]["counts"]
    t = {key: transitions.get(key, 0) for key in PREDICTED_TRANSITIONS}
    e = {key: strict_transitions.get(key, 0) for key in STRICT_E2E_TRANSITIONS}
    lines = [
        "% Generated from verified exact40 v5.2 evidence; do not edit numeric cells.",
        "\\begingroup",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\noindent",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}rrrrrrrr@{}}",
        "\\toprule",
        "$n$ & G$@.5$ & S-abs. & S-corr. & S-oracle & S-pred & E2E & Attr. G/S/I \\\\",
        "\\midrule",
        "{n} & {giou} & {absent} & {corrupted} & {oracle} & {pred} & {e2e} & {ag}/{ls}/{inc} \\\\".format(
            n=v52["records"],
            giou=_fmt(100 * float(metrics["g_both_box_iou_at_0_5"]), 1),
            absent=_fmt(100 * float(metrics["s_absent_accuracy"]), 1),
            corrupted=_fmt(100 * float(metrics["s_corrupted_accuracy"]), 1),
            oracle=_fmt(100 * float(metrics["s_oracle_accuracy"]), 1),
            pred=_fmt(100 * float(metrics["s_pred_correct_all_scenes_invalid_g_zero"]), 1),
            e2e=_fmt(100 * float(metrics["s_pred_strict_e2e_accuracy"]), 1),
            ag=attribution["Attributed(G)"],
            ls=attribution["Local(S)"],
            inc=attribution["Inconclusive"],
        ),
        "\\bottomrule",
        "\\end{tabular*}",
        "{{\\footnotesize Gap: {gap} pp [{lo}, {hi}]; headroom: {headroom} pp; recovery: {recovery}.\\par}}".format(
            gap=_fmt(100 * float(metrics["paired_dependency_gap_oracle_minus_predicted"]), 1, signed=True),
            lo=_fmt(100 * float(ci[0]), 1, signed=True),
            hi=_fmt(100 * float(ci[1]), 1, signed=True),
            headroom=_fmt(100 * float(metrics["positive_oracle_headroom"]), 1),
            recovery=(
                "\\textit{n/a}"
                if metrics["bounded_oracle_recovery"] is None
                else _fmt(100 * float(metrics["bounded_oracle_recovery"]), 1) + "\\%"
            ),
        ),
        "{{\\footnotesize Oracle-minus-absent: {absent} pp [{abs_lo}, {abs_hi}]; "
        "oracle-minus-corrupted: {corrupt} pp [{cor_lo}, {cor_hi}].\\par}}".format(
            absent=_fmt(100 * float(metrics["paired_context_gain_oracle_minus_absent"]), 1, signed=True),
            abs_lo=_fmt(100 * float(control_ci["oracle_minus_absent"]["ci"][0]), 1, signed=True),
            abs_hi=_fmt(100 * float(control_ci["oracle_minus_absent"]["ci"][1]), 1, signed=True),
            corrupt=_fmt(100 * float(metrics["paired_context_gain_oracle_minus_corrupted"]), 1, signed=True),
            cor_lo=_fmt(100 * float(control_ci["oracle_minus_corrupted"]["ci"][0]), 1, signed=True),
            cor_hi=_fmt(100 * float(control_ci["oracle_minus_corrupted"]["ci"][1]), 1, signed=True),
        ),
        "{{\\footnotesize Oracle/predicted transitions: CC={cc}, CW={cw}, WC={wc}, WW={ww}.\\par}}".format(
            cc=t["oracle_correct_to_predicted_correct"],
            cw=t["oracle_correct_to_predicted_wrong"],
            wc=t["oracle_wrong_to_predicted_correct"],
            ww=t["oracle_wrong_to_predicted_wrong"],
        ),
        "{{\\footnotesize Oracle/E2E transitions: CC={ecc}, CW={ecw}, WC={ewc}, WW={eww}.\\par}}".format(
            ecc=e["oracle_correct_to_e2e_correct"],
            ecw=e["oracle_correct_to_e2e_wrong"],
            ewc=e["oracle_wrong_to_e2e_correct"],
            eww=e["oracle_wrong_to_e2e_wrong"],
        ),
        "\\endgroup",
        "",
    ]
    return "\n".join(lines)


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="both", length=0, labelsize=7.5, colors=INK)
    ax.grid(axis="x", color=RULE, linewidth=0.55, alpha=0.60, zorder=0)


def _draw_general(ax: plt.Axes, payload: dict[str, Any]) -> None:
    rows = _general_rows(payload)
    labels = [row["label"] for row in rows]
    values = np.asarray([row["e1_minus_base_points"] for row in rows], dtype=float)
    cis = np.asarray([row["e1_minus_base_points_ci95"] for row in rows], dtype=float)
    y = np.arange(len(rows), dtype=float)
    colors = [GROUNDING, SPATIAL, STRESS, INK]
    markers = ["o", "s", "^", "D"]
    ax.axvline(0, color=NEUTRAL, linewidth=0.8, zorder=1)
    for index, (value, interval, color, marker) in enumerate(zip(values, cis, colors, markers, strict=True)):
        ax.hlines(y[index], interval[0], interval[1], color=color, linewidth=1.15, zorder=2)
        ax.vlines([interval[0], interval[1]], y[index] - 0.06, y[index] + 0.06, color=color, linewidth=1.0, zorder=2)
        ax.scatter(
            value,
            y[index],
            marker=marker,
            s=25,
            facecolor="white" if index < 3 else color,
            edgecolor=color,
            linewidth=1.0,
            zorder=3,
        )
        offset = 4 if value >= 0 else -4
        ax.annotate(
            _fmt(float(value), signed=True),
            (value, y[index]),
            xytext=(offset, -8),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
            fontsize=7.3,
            color=color,
        )
    lower = min(float(cis.min()), 0.0)
    upper = max(float(cis.max()), 0.0)
    padding = max(2.0, 0.16 * (upper - lower or 1.0))
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_yticks(y, labels)
    ax.set_xlabel("E1 - Base (points; 95% CI)", fontsize=7.7, color=INK)
    ax.set_title("(a) General retention", loc="left", fontsize=8.8, fontweight="bold", color=INK, pad=5)
    _style_axis(ax)


def _draw_v52(ax: plt.Axes, payload: dict[str, Any]) -> None:
    v52 = payload["scannet_v52_paired_gs"]
    metrics = v52["metrics"]
    labels = [
        "S-absent",
        "S-corrupted",
        "S-oracle",
        "S-pred (all scenes)",
        "Strict E2E",
        "Dependency gap (pp)",
    ]
    values = np.asarray(
        [
            100 * float(metrics["s_absent_accuracy"]),
            100 * float(metrics["s_corrupted_accuracy"]),
            100 * float(metrics["s_oracle_accuracy"]),
            100 * float(metrics["s_pred_correct_all_scenes_invalid_g_zero"]),
            100 * float(metrics["s_pred_strict_e2e_accuracy"]),
            100 * float(metrics["paired_dependency_gap_oracle_minus_predicted"]),
        ]
    )
    gap_ci = np.asarray(v52["scene_bootstrap"]["ci"], dtype=float) * 100
    y = np.arange(6, dtype=float)
    markers = ["x", "X", "o", "s", "^"]
    colors = [NEUTRAL, LOSS, SPATIAL, GROUNDING, INK]
    for index in range(5):
        style = (
            {"facecolor": "white", "edgecolor": colors[index]}
            if index in {3, 4}
            else {"color": colors[index]}
        )
        ax.scatter(
            values[index], y[index], marker=markers[index], s=34,
            linewidth=1.2, zorder=3, **style,
        )
    ax.hlines(y[5], gap_ci[0], gap_ci[1], color=STRESS, linewidth=1.2, zorder=2)
    ax.vlines([gap_ci[0], gap_ci[1]], y[5] - 0.06, y[5] + 0.06, color=STRESS, linewidth=1.0, zorder=2)
    ax.scatter(values[5], y[5], marker="D", s=27, color=STRESS, zorder=3)
    for index, value in enumerate(values):
        label_x = gap_ci[1] if index == 5 else value
        align_right = label_x >= 90.0
        ax.annotate(
            _fmt(float(value), 1, signed=index == 5),
            (label_x, y[index]),
            xytext=(-5 if align_right else 5, -3),
            textcoords="offset points",
            ha="right" if align_right else "left",
            fontsize=7.5,
            color=[NEUTRAL, LOSS, SPATIAL, GROUNDING, INK, STRESS][index],
        )
    transitions = v52["transition_counts"]
    cc = transitions.get("oracle_correct_to_predicted_correct", 0)
    cw = transitions.get("oracle_correct_to_predicted_wrong", 0)
    wc = transitions.get("oracle_wrong_to_predicted_correct", 0)
    ww = transitions.get("oracle_wrong_to_predicted_wrong", 0)
    attribution = v52["attribution"]["counts"]
    ax.text(
        0.5,
        -0.24,
        "Pred. CC/CW/WC/WW = "
        f"{cc}/{cw}/{wc}/{ww}\nAttr. G/S/I = "
        f"{attribution['Attributed(G)']}/{attribution['Local(S)']}/{attribution['Inconclusive']}",
        transform=ax.transAxes,
        fontsize=6.8,
        color=NEUTRAL,
        ha="center",
        va="top",
    )
    ax.set_xlim(min(-5.0, float(gap_ci[0]) - 5.0), max(112.0, float(values.max()) + 12.0, float(gap_ci[1]) + 12.0))
    ax.set_ylim(5.48, -0.48)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Accuracy (%) / gap (pp)", fontsize=7.7, color=INK)
    ax.set_title("(b) Paired G-to-S", loc="left", fontsize=8.8, fontweight="bold", color=INK, pad=5)
    _style_axis(ax)


def render_figure(payload: dict[str, Any], output_prefix: Path) -> list[Path]:
    available = payload["scannet_v52_paired_gs"]["availability"] == "available"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "svg.hashsalt": payload["identity_sha256"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if available:
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.85), gridspec_kw={"width_ratios": [1.05, 0.95]})
        _draw_general(axes[0], payload)
        _draw_v52(axes[1], payload)
        fig.subplots_adjust(left=0.15, right=0.985, bottom=0.24, top=0.88, wspace=0.69)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.45))
        _draw_general(ax, payload)
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.24, top=0.86)
    fig.patch.set_facecolor("white")
    outputs = [output_prefix.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png")]
    fig.savefig(
        outputs[0],
        format="svg",
        metadata={
            "Title": "Verified ICLR claim evidence",
            "Description": f"Evidence identity {payload['identity_sha256']}",
            "Creator": RENDER_PROTOCOL,
            "Date": None,
        },
    )
    fig.savefig(
        outputs[1],
        format="pdf",
        metadata={"Title": "Verified ICLR claim evidence", "Creator": RENDER_PROTOCOL, "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        outputs[2],
        format="png",
        dpi=240,
        metadata={"Software": RENDER_PROTOCOL, "evidence_identity_sha256": payload["identity_sha256"]},
    )
    plt.close(fig)
    return outputs


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def render_bundle(evidence_path: Path, output_dir: Path) -> dict[str, Any]:
    evidence_path = evidence_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    try:
        output_dir.relative_to((ROOT / "paper").resolve())
    except ValueError:
        pass
    else:
        raise RenderError("renderer outputs must not be written inside paper/")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{output_dir.name}.render.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RenderError(f"render lock already exists: {lock_path}") from exc
    staging: Path | None = None
    try:
        if output_dir.exists():
            raise RenderError(f"refusing to overwrite existing output directory: {output_dir}")
        payload, evidence_binding = _verified_snapshot(evidence_path)
        renderer_binding = _file_binding(Path(__file__))
        contract_binding = _file_binding(CONTRACT_PATH)
        staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
        general_table = staging / "tab_general_retention.tex"
        _write_text(general_table, render_general_table(payload))
        outputs: dict[str, dict[str, Any]] = {
            "general_retention_table": _relative_binding(general_table, staging)
        }
        if payload["scannet_v52_paired_gs"]["availability"] == "available":
            v52_table = staging / "tab_scannet_v52_paired_gs.tex"
            _write_text(v52_table, render_v52_table(payload))
            outputs["scannet_v52_paired_gs_table"] = _relative_binding(v52_table, staging)
        figure_paths = render_figure(payload, staging / "fig_claim_evidence")
        for path in figure_paths:
            outputs[f"claim_figure_{path.suffix[1:]}"] = _relative_binding(path, staging)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "protocol": RENDER_PROTOCOL,
            "status": "CLAIM_ARTIFACTS_RENDERED",
            "evidence": {
                **evidence_binding,
                "identity_sha256": payload["identity_sha256"],
            },
            "compiler": payload["compiler"],
            "renderer": renderer_binding,
            "visual_contract": contract_binding,
            "v52_availability": payload["scannet_v52_paired_gs"]["availability"],
            "outputs": outputs,
            "publication_policy": {
                "paper_was_not_modified": True,
                "missing_values_filled": False,
                "unavailable_v52_table_created": False,
                "editable_vector_output_created": True,
            },
        }
        manifest["identity_sha256"] = compiler.content_hash(manifest)
        _write_text(
            staging / "render_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        final_payload, final_binding = _verified_snapshot(evidence_path)
        if final_binding != evidence_binding or final_payload["identity_sha256"] != payload["identity_sha256"]:
            raise RenderError("claim evidence changed during rendering")
        if _file_binding(Path(__file__)) != renderer_binding:
            raise RenderError("renderer changed during rendering")
        if _file_binding(CONTRACT_PATH) != contract_binding:
            raise RenderError("visual contract changed during rendering")
        if output_dir.exists():
            raise RenderError(f"output directory appeared during render: {output_dir}")
        staging.rename(output_dir)
        staging = None
        return manifest
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = render_bundle(args.evidence, args.output_dir)
    except (RenderError, OSError) as exc:
        print(f"CLAIM_RENDER_REJECTED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "identity_sha256": manifest["identity_sha256"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "v52_availability": manifest["v52_availability"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
