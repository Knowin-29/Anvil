#!/usr/bin/env python3
"""Compile the five-model formal40 v5.1 S-oracle leaderboard.

This compiler is deliberately separate from the frozen single-model scorer.  It
accepts only complete, mutually comparable formal results, rehashes every file
binding reachable from those results, and derives all leaderboard and paired
statistics from the common per-item records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULT_PROTOCOL = "gsa-depth-v5.1-formal40-s-oracle-result-v1"
EVALUATION_PROTOCOL = "gsa-depth-v5.1-formal40-paired-gs-eval-v1"
RUN_PROTOCOL = "gsa-depth-v5.1-formal40-paired-gs-dlc-v1"
LEADERBOARD_PROTOCOL = "gsa-depth-v5.1-formal40-s-oracle-five-model-leaderboard-v1"
RESULT_STATUS = "FORMAL_S_ORACLE_SCORE_COMPLETE"
LEADERBOARD_STATUS = "FORMAL_FIVE_MODEL_LEADERBOARD_COMPLETE"
TRACK = "S_oracle_grounded_pair"
RELEASE = "formal40"
EXPECTED_MODELS = 5
EXPECTED_RECORDS = 40
EXPECTED_LABEL_BALANCE = {"A": 20, "B": 20}
BOOTSTRAP_SEED = 51051
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95
REFERENCE_KEY = "qwen3_5_9b"
DEFAULT_OUT = ROOT / "artifacts/results/scannet_v51_s_oracle_leaderboard.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class LeaderboardError(ValueError):
    """Raised when formal results cannot be compiled without weakening trust."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    role: str
    result_path: Path
    expected_model_inventory_identity_sha256: str


DEFAULT_MODEL_SPECS = (
    ModelSpec(
        key=REFERENCE_KEY,
        display_name="Qwen3.5-9B",
        role="reference_general_model",
        result_path=(
            ROOT
            / "output/scannet_rgb_depth_v51_formal40_qwen35_9b_20260823_v1/s_oracle/result.json"
        ),
        expected_model_inventory_identity_sha256=(
            "dfad48fe82fcf17f8bad486b15c8077086eb8a10beba7cde8f6ce94bb2811f65"
        ),
    ),
    ModelSpec(
        key="qwen3_vl_8b",
        display_name="Qwen3-VL-8B-Instruct",
        role="general_model",
        result_path=(
            ROOT
            / "output/scannet_rgb_depth_v51_s_oracle_qwen3vl8b_20260823_v1/s_oracle/result.json"
        ),
        expected_model_inventory_identity_sha256=(
            "684da50c4ec7c8a16e9e2d3d11ab55e23b4b6467b48eb24860e2d2048f70fc1e"
        ),
    ),
    ModelSpec(
        key="e1_posttrained",
        display_name="E1 (Qwen3.5-9B post-trained)",
        role="posttrained_candidate",
        result_path=(
            ROOT
            / "output/scannet_rgb_depth_v51_s_oracle_e1_20260823_v1/s_oracle/result.json"
        ),
        expected_model_inventory_identity_sha256=(
            "1314f0fd60a0153b2e0736d55d2ba3d4ac4d12127c3e80113c2a19dae98a7b2e"
        ),
    ),
    ModelSpec(
        key="qwen2_5_vl_7b",
        display_name="Qwen2.5-VL-7B-Instruct",
        role="general_model",
        result_path=(
            ROOT
            / "output/scannet_rgb_depth_v51_s_oracle_qwen25vl7b_20260823_v1/s_oracle/result.json"
        ),
        expected_model_inventory_identity_sha256=(
            "b59483ae1d148deb21a0687520593632e5fa4a58f07d843d3bf63dfe9bfee98b"
        ),
    ),
    ModelSpec(
        key="qwen2_5_vl_3b",
        display_name="Qwen2.5-VL-3B-Instruct",
        role="general_model",
        result_path=(
            ROOT
            / "output/scannet_rgb_depth_v51_s_oracle_qwen25vl3b_20260823_v1/s_oracle/result.json"
        ),
        expected_model_inventory_identity_sha256=(
            "f911f757d751f2c93756019d99b3b2b7d0c5fbbf19bcec61640077a69220a06b"
        ),
    ),
)


EXPECTED_TRUST = {
    "score_available": True,
    "complete_bound_s_oracle_bundle": True,
    "formal_source_reverified": True,
    "all_token_cap_metadata_complete": True,
    "zero_capped_responses": True,
    "held_out_evaluation_only": True,
    "training_materialization_eligible": False,
}

EXPECTED_TASK_CONTRACT = {
    "scope": "held_out_s_oracle_only",
    "primary_unit": "scene",
    "all_item_denominator": EXPECTED_RECORDS,
    "capped_response_credit": 0,
    "oracle_context_source": "frozen_human_ground_truth_boxes",
    "training_materialization_eligible": False,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _round(value: float) -> float:
    return round(float(value), 6)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise LeaderboardError("cannot form a ratio with a non-positive denominator")
    return _round(numerator / denominator)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise LeaderboardError("cannot compute a percentile from no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_ci(values: list[int], *, paired: bool) -> dict[str, Any]:
    if not values or any(value not in (-1, 0, 1) for value in values):
        raise LeaderboardError("bootstrap inputs must be nonempty values in {-1,0,1}")
    if not paired and any(value not in (0, 1) for value in values):
        raise LeaderboardError("binary bootstrap inputs must be in {0,1}")
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    ci = [
        _round(_percentile(means, (1.0 - CONFIDENCE) / 2.0)),
        _round(_percentile(means, 1.0 - (1.0 - CONFIDENCE) / 2.0)),
    ]
    report: dict[str, Any] = {
        "method": "deterministic_percentile_bootstrap",
        "unit": "scene",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "confidence": CONFIDENCE,
        "ci": ci,
    }
    report["paired_scenes" if paired else "scenes"] = n
    return report


def _wilson_ci(successes: int, total: int) -> dict[str, Any]:
    if total <= 0 or not 0 <= successes <= total:
        raise LeaderboardError("invalid Wilson interval inputs")
    z = NormalDist().inv_cdf(0.5 + CONFIDENCE / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "method": "wilson_score_binomial",
        "successes": successes,
        "trials": total,
        "confidence": CONFIDENCE,
        "ci": [_round(max(0.0, centre - radius)), _round(min(1.0, centre + radius))],
    }


def _verify_identity(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise LeaderboardError(f"{label} must be an object")
    identity = value.get("identity_sha256")
    if not isinstance(identity, str) or not _SHA256.fullmatch(identity):
        raise LeaderboardError(f"{label} identity is absent or invalid")
    unsigned = {key: item for key, item in value.items() if key != "identity_sha256"}
    if canonical_hash(unsigned) != identity:
        raise LeaderboardError(f"{label} identity drift")


class RehashLedger:
    """Deduplicate full-file hashing while preserving fail-closed bindings."""

    def __init__(self) -> None:
        self._files: dict[Path, dict[str, Any]] = {}
        self.binding_occurrences = 0
        self.explicit_result_files = 0

    @staticmethod
    def _hash_path(path: Path) -> tuple[int, int, str]:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise LeaderboardError(f"file changed during live rehash: {path}")
        return after.st_size, after.st_mtime_ns, digest.hexdigest()

    def _live(self, path: Path) -> dict[str, Any]:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise LeaderboardError(f"bound file is missing: {path}")
        cached = self._files.get(path)
        if cached is not None:
            stat = path.stat()
            if stat.st_size != cached["bytes"] or stat.st_mtime_ns != cached["mtime_ns"]:
                raise LeaderboardError(f"file changed after live rehash: {path}")
            return cached
        size, mtime_ns, digest = self._hash_path(path)
        live = {
            "path": str(path),
            "bytes": size,
            "sha256": digest,
            "mtime_ns": mtime_ns,
        }
        self._files[path] = live
        return live

    def read_result(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        path = path.expanduser().resolve()
        live = self._live(path)
        self.explicit_result_files += 1
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LeaderboardError(f"invalid result JSON: {path}") from exc
        if not isinstance(value, dict):
            raise LeaderboardError(f"result must be a JSON object: {path}")
        stat = path.stat()
        if stat.st_size != live["bytes"] or stat.st_mtime_ns != live["mtime_ns"]:
            raise LeaderboardError(f"result changed after live rehash: {path}")
        binding = {key: live[key] for key in ("path", "bytes", "sha256")}
        return value, binding

    def verify_binding(self, binding: Any, label: str) -> None:
        self.binding_occurrences += 1
        if not isinstance(binding, dict):
            raise LeaderboardError(f"{label}: file binding must be an object")
        allowed = {"path", "bytes", "sha256", "mtime_ns"}
        if not {"path", "bytes", "sha256"}.issubset(binding) or not set(binding).issubset(
            allowed
        ):
            raise LeaderboardError(f"{label}: file binding schema drift")
        path_value = binding.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise LeaderboardError(f"{label}: invalid bound path")
        path = Path(path_value).expanduser().resolve()
        if str(path) != path_value:
            raise LeaderboardError(f"{label}: bound path is not canonical: {path_value}")
        expected_sha = binding.get("sha256")
        if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
            raise LeaderboardError(f"{label}: invalid bound SHA-256")
        live = self._live(path)
        if binding.get("bytes") != live["bytes"]:
            raise LeaderboardError(f"{label}: bound file size drift: {path}")
        if expected_sha != live["sha256"]:
            raise LeaderboardError(f"{label}: bound file hash drift: {path}")
        if "mtime_ns" in binding and binding["mtime_ns"] != live["mtime_ns"]:
            raise LeaderboardError(f"{label}: bound file mtime drift: {path}")

    def verify_all_bindings(self, value: Any, label: str) -> None:
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"}.issubset(value):
                self.verify_binding(value, label)
                return
            for key, item in value.items():
                self.verify_all_bindings(item, f"{label}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self.verify_all_bindings(item, f"{label}[{index}]")

    def report(self) -> dict[str, Any]:
        bindings = [
            {key: live[key] for key in ("path", "bytes", "sha256")}
            for _, live in sorted(self._files.items(), key=lambda item: str(item[0]))
        ]
        return {
            "status": "LIVE_REHASH_PASS",
            "algorithm": "sha256_full_bytes",
            "explicit_result_files_rehashed": self.explicit_result_files,
            "binding_occurrences_verified": self.binding_occurrences,
            "unique_files_rehashed": len(bindings),
            "unique_bytes_rehashed": sum(item["bytes"] for item in bindings),
            "verified_file_set_sha256": canonical_hash(bindings),
            "all_size_sha256_and_bound_weight_mtime_checks_match": True,
        }


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise LeaderboardError(f"{label} drift: expected {expected!r}, got {actual!r}")


def _validate_metric_block(
    block: Any, *, total: int, correct: int, label: str
) -> None:
    if not isinstance(block, dict):
        raise LeaderboardError(f"{label} must be an object")
    expected = {
        "n_scenes": total,
        "correct": correct,
        "accuracy": _ratio(correct, total),
        "parsed_responses": total,
        "parse_coverage": 1.0,
        "uncapped_parsed_responses": total,
        "uncapped_parse_coverage": 1.0,
        "capped_responses": 0,
        "cap_rate": 0.0,
        "accuracy_binomial_ci": _wilson_ci(correct, total),
    }
    _require_equal(block, expected, label)


def _validate_items(result: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    raw_items = result.get("per_item")
    if not isinstance(raw_items, list) or len(raw_items) != EXPECTED_RECORDS:
        raise LeaderboardError(f"{label}: per_item must contain exactly 40 records")
    items: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict) or set(item) != {
            "eval_id",
            "target_stratum",
            "parse_status",
            "parsed",
            "hit_token_cap",
            "correct",
        }:
            raise LeaderboardError(f"{label}: per_item[{index}] schema drift")
        eval_id = item.get("eval_id")
        if not isinstance(eval_id, str) or not eval_id or eval_id in items:
            raise LeaderboardError(f"{label}: missing or duplicate eval_id")
        if item.get("target_stratum") not in {"A", "B"}:
            raise LeaderboardError(f"{label}/{eval_id}: target is not A/B")
        if item.get("parsed") is not True or item.get("hit_token_cap") is not False:
            raise LeaderboardError(f"{label}/{eval_id}: result is not fully parsed and uncapped")
        if not isinstance(item.get("parse_status"), str) or not item["parse_status"]:
            raise LeaderboardError(f"{label}/{eval_id}: parse status is invalid")
        if not isinstance(item.get("correct"), bool):
            raise LeaderboardError(f"{label}/{eval_id}: correctness is not Boolean")
        status_counts[item["parse_status"]] += 1
        items[eval_id] = item

    target_counts = Counter(item["target_stratum"] for item in items.values())
    _require_equal(dict(sorted(target_counts.items())), EXPECTED_LABEL_BALANCE, f"{label} balance")
    correct = sum(int(item["correct"]) for item in items.values())
    _validate_metric_block(result.get("metrics"), total=EXPECTED_RECORDS, correct=correct, label=f"{label}.metrics")
    strata = result.get("ab_strata")
    if not isinstance(strata, dict) or set(strata) != {"A", "B"}:
        raise LeaderboardError(f"{label}: A/B strata schema drift")
    for target in ("A", "B"):
        target_items = [item for item in items.values() if item["target_stratum"] == target]
        target_correct = sum(int(item["correct"]) for item in target_items)
        _validate_metric_block(
            strata[target], total=len(target_items), correct=target_correct, label=f"{label}.ab_strata.{target}"
        )
    _require_equal(result.get("parse_status"), dict(sorted(status_counts.items())), f"{label}.parse_status")
    expected_intervals = {
        "accuracy_scene_bootstrap": _bootstrap_ci(
            # The frozen scorer bootstraps in frozen source/per_item order.  With
            # a finite deterministic draw sequence, sorting the same outcomes
            # can move an endpoint by one 1/40 step.
            [int(item["correct"]) for item in items.values()], paired=False
        ),
        "accuracy_binomial": _wilson_ci(correct, EXPECTED_RECORDS),
    }
    _require_equal(result.get("confidence_intervals"), expected_intervals, f"{label}.confidence_intervals")
    return items


def _validate_result(
    spec: ModelSpec, result: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    label = spec.key
    _verify_identity(result, f"{label}.result")
    _require_equal(result.get("result_schema_version"), 1, f"{label}.result_schema_version")
    _require_equal(result.get("protocol"), RESULT_PROTOCOL, f"{label}.protocol")
    _require_equal(
        result.get("evaluation_protocol"), EVALUATION_PROTOCOL, f"{label}.evaluation_protocol"
    )
    _require_equal(result.get("release"), RELEASE, f"{label}.release")
    _require_equal(result.get("status"), RESULT_STATUS, f"{label}.status")
    _require_equal(
        result.get("track"),
        {"name": TRACK, "status": "FORMAL_SCORE_COMPLETE", "context": "human_ground_truth_bounding_boxes"},
        f"{label}.track",
    )
    _require_equal(result.get("task_contract"), EXPECTED_TASK_CONTRACT, f"{label}.task_contract")
    _require_equal(result.get("trust"), EXPECTED_TRUST, f"{label}.trust")

    provenance = result.get("provenance")
    _verify_identity(provenance, f"{label}.provenance")
    _require_equal(provenance.get("verified"), True, f"{label}.provenance.verified")
    _require_equal(
        provenance.get("training_materialization_eligible"),
        False,
        f"{label}.provenance.training_materialization_eligible",
    )
    formal_source = provenance.get("formal_source")
    _verify_identity(formal_source, f"{label}.formal_source")
    for key, expected in (
        ("records", EXPECTED_RECORDS),
        ("unique_scenes", EXPECTED_RECORDS),
        ("unique_images", EXPECTED_RECORDS),
        ("official_split", "val"),
        ("human_audit_status", "FORMAL_PASS"),
        ("training_materialization_eligible", False),
    ):
        _require_equal(formal_source.get(key), expected, f"{label}.formal_source.{key}")

    input_receipt = provenance.get("input_receipt")
    _verify_identity(input_receipt, f"{label}.input_receipt")
    _require_equal(input_receipt.get("protocol"), RUN_PROTOCOL, f"{label}.input_receipt.protocol")
    _require_equal(
        input_receipt.get("evaluation_protocol"),
        EVALUATION_PROTOCOL,
        f"{label}.input_receipt.evaluation_protocol",
    )
    _require_equal(
        input_receipt.get("training_materialization_eligible"),
        False,
        f"{label}.input_receipt.training_materialization_eligible",
    )
    frozen_code = provenance.get("frozen_evaluation_code")
    if not isinstance(frozen_code, dict) or not isinstance(frozen_code.get("files"), dict):
        raise LeaderboardError(f"{label}: frozen evaluation code binding is absent")
    _require_equal(
        frozen_code.get("identity_sha256"),
        canonical_hash(frozen_code["files"]),
        f"{label}.frozen_evaluation_code.identity",
    )

    inventory = provenance.get("model_inventory")
    _verify_identity(inventory, f"{label}.model_inventory")
    _require_equal(
        inventory.get("identity_sha256"),
        spec.expected_model_inventory_identity_sha256,
        f"{label}.model_inventory.identity",
    )
    _require_equal(
        inventory.get("weight_bytes_fully_sha256_hashed"),
        True,
        f"{label}.model_inventory.full_weight_hash",
    )
    _require_equal(
        provenance.get("model_binding_identity_sha256"),
        (provenance.get("s_oracle_bundle") or {}).get("model_binding_identity_sha256"),
        f"{label}.model_binding_identity",
    )

    bundle = provenance.get("s_oracle_bundle")
    _verify_identity(bundle, f"{label}.s_oracle_bundle")
    for key, expected in (
        ("verified", True),
        ("protocol", RUN_PROTOCOL),
        ("evaluation_protocol", EVALUATION_PROTOCOL),
        ("track", TRACK),
        ("n_predictions", EXPECTED_RECORDS),
    ):
        _require_equal(bundle.get(key), expected, f"{label}.s_oracle_bundle.{key}")
    _require_equal(
        bundle.get("model_inventory_identity_sha256"),
        inventory.get("identity_sha256"),
        f"{label}.s_oracle_bundle.model_inventory_identity",
    )
    return _validate_items(result, label)


def _common_context(result: dict[str, Any]) -> dict[str, Any]:
    provenance = result["provenance"]
    bundle = provenance["s_oracle_bundle"]
    receipt_binding = bundle["input_receipt"]
    return {
        "result_protocol": result["protocol"],
        "evaluation_protocol": result["evaluation_protocol"],
        "release": result["release"],
        "track": result["track"],
        "task_contract": result["task_contract"],
        "related_tracks": result["related_tracks"],
        "formal_source": provenance["formal_source"],
        "request_release": provenance["request_release"],
        "request_release_identity_sha256": provenance["request_release_identity_sha256"],
        "input_receipt": provenance["input_receipt"],
        "input_receipt_content_binding": {
            "bytes": receipt_binding["bytes"],
            "sha256": receipt_binding["sha256"],
        },
        "scorer": provenance["scorer"],
        "frozen_evaluation_code": provenance["frozen_evaluation_code"],
        "referent_interface_audit_identity_sha256": provenance[
            "referent_interface_audit_identity_sha256"
        ],
        "run_protocol": bundle["protocol"],
        "request_manifest": bundle["request_manifest"],
        "requests": bundle["requests"],
        "request_records_sha256": bundle["request_records_sha256"],
        "decoding": bundle["decoding"],
    }


def _prediction_distribution(items: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    predicted_a = 0
    for item in items.values():
        target = item["target_stratum"]
        predicted = target if item["correct"] else ("B" if target == "A" else "A")
        predicted_a += int(predicted == "A")
    total = len(items)
    rate_a = _ratio(predicted_a, total)
    rate_b = _ratio(total - predicted_a, total)
    return {
        "derivation": "binary_target_plus_fully_parsed_uncapped_correctness",
        "predicted_A": predicted_a,
        "predicted_A_rate": rate_a,
        "predicted_B": total - predicted_a,
        "predicted_B_rate": rate_b,
        "position_bias": {
            "signed_A_rate_minus_balanced": _round(rate_a - 0.5),
            "absolute_rate_deviation_from_balanced": _round(abs(rate_a - 0.5)),
            "signed_A_minus_B_rate": _round(rate_a - rate_b),
        },
    }


def _paired_comparison(
    items: Mapping[str, dict[str, Any]], reference: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    if set(items) != set(reference):
        raise LeaderboardError("paired comparison eval ID sets differ")
    deltas: list[int] = []
    discordance = Counter()
    for eval_id in sorted(reference):
        ref_correct = bool(reference[eval_id]["correct"])
        candidate_correct = bool(items[eval_id]["correct"])
        deltas.append(int(candidate_correct) - int(ref_correct))
        if ref_correct and candidate_correct:
            discordance["both_correct"] += 1
        elif not ref_correct and not candidate_correct:
            discordance["both_wrong"] += 1
        elif candidate_correct:
            discordance["candidate_only_correct"] += 1
        else:
            discordance["reference_only_correct"] += 1
    observed = _round(sum(deltas) / len(deltas))
    bootstrap = _bootstrap_ci(deltas, paired=True)
    bootstrap["observed_accuracy_delta"] = observed
    counts = {
        "both_correct": discordance["both_correct"],
        "both_wrong": discordance["both_wrong"],
        "candidate_only_correct": discordance["candidate_only_correct"],
        "reference_only_correct": discordance["reference_only_correct"],
        "discordant_total": (
            discordance["candidate_only_correct"] + discordance["reference_only_correct"]
        ),
        "net_candidate_correct_advantage": (
            discordance["candidate_only_correct"] - discordance["reference_only_correct"]
        ),
    }
    if sum(counts[key] for key in ("both_correct", "both_wrong", "candidate_only_correct", "reference_only_correct")) != len(deltas):
        raise LeaderboardError("discordance accounting does not cover all paired scenes")
    return {
        "reference_model_key": REFERENCE_KEY,
        "accuracy_delta_candidate_minus_reference": observed,
        "paired_scene_bootstrap_95_ci": bootstrap,
        "discordance": counts,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def compile_leaderboard(
    specs: Sequence[ModelSpec] = DEFAULT_MODEL_SPECS,
) -> dict[str, Any]:
    if len(specs) != EXPECTED_MODELS:
        raise LeaderboardError("formal leaderboard requires exactly five model results")
    keys = [spec.key for spec in specs]
    if len(set(keys)) != len(keys) or REFERENCE_KEY not in keys:
        raise LeaderboardError("model keys must be unique and include the Qwen3.5 reference")
    paths = [spec.result_path.expanduser().resolve() for spec in specs]
    if len(set(paths)) != len(paths):
        raise LeaderboardError("the five model results must be distinct files")

    ledger = RehashLedger()
    results: dict[str, dict[str, Any]] = {}
    result_bindings: dict[str, dict[str, Any]] = {}
    item_maps: dict[str, dict[str, dict[str, Any]]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    inventory_ids: set[str] = set()
    for spec in specs:
        result, binding = ledger.read_result(spec.result_path)
        item_maps[spec.key] = _validate_result(spec, result)
        inventory_id = result["provenance"]["model_inventory"]["identity_sha256"]
        if inventory_id in inventory_ids:
            raise LeaderboardError("the five rows do not represent five distinct model inventories")
        inventory_ids.add(inventory_id)
        results[spec.key] = result
        result_bindings[spec.key] = binding
        contexts[spec.key] = _common_context(result)

    reference_context = contexts[REFERENCE_KEY]
    for spec in specs:
        _require_equal(
            contexts[spec.key], reference_context, f"{spec.key}.formal_comparability_context"
        )
    reference_items = item_maps[REFERENCE_KEY]
    eval_ids = sorted(reference_items)
    target_map = {eval_id: reference_items[eval_id]["target_stratum"] for eval_id in eval_ids}
    for spec in specs:
        candidate = item_maps[spec.key]
        _require_equal(sorted(candidate), eval_ids, f"{spec.key}.eval_ids")
        _require_equal(
            {eval_id: candidate[eval_id]["target_stratum"] for eval_id in eval_ids},
            target_map,
            f"{spec.key}.target_map",
        )

    # Expensive verification occurs only after all cheap structural and
    # comparability checks pass.  Shared release/source files are hashed once.
    for spec in specs:
        ledger.verify_all_bindings(results[spec.key], spec.key)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        result = results[spec.key]
        items = item_maps[spec.key]
        correct = sum(int(item["correct"]) for item in items.values())
        strata = {
            target: {
                "n": EXPECTED_LABEL_BALANCE[target],
                "correct": sum(
                    int(item["correct"])
                    for item in items.values()
                    if item["target_stratum"] == target
                ),
            }
            for target in ("A", "B")
        }
        for value in strata.values():
            value["accuracy"] = _ratio(value["correct"], value["n"])
        inventory = result["provenance"]["model_inventory"]
        rows.append(
            {
                "model_key": spec.key,
                "display_name": spec.display_name,
                "role": spec.role,
                "model": {
                    "path": inventory["path"],
                    "architecture": inventory["architecture"],
                    "model_type": inventory["model_type"],
                    "inventory_identity_sha256": inventory["identity_sha256"],
                    "binding_identity_sha256": result["provenance"][
                        "model_binding_identity_sha256"
                    ],
                },
                "result": {
                    "file": result_bindings[spec.key],
                    "identity_sha256": result["identity_sha256"],
                },
                "n": EXPECTED_RECORDS,
                "correct": correct,
                "accuracy": _ratio(correct, EXPECTED_RECORDS),
                "A": strata["A"],
                "B": strata["B"],
                "prediction_distribution": _prediction_distribution(items),
                "vs_qwen3_5_9b": _paired_comparison(items, reference_items),
            }
        )

    rows.sort(key=lambda row: (-row["accuracy"], row["model_key"]))
    previous_accuracy: float | None = None
    previous_rank = 0
    for index, row in enumerate(rows, 1):
        if previous_accuracy is None or row["accuracy"] != previous_accuracy:
            previous_rank = index
            previous_accuracy = row["accuracy"]
        row["accuracy_rank"] = previous_rank

    common = {
        "result_protocol": RESULT_PROTOCOL,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "run_protocol": RUN_PROTOCOL,
        "release": RELEASE,
        "track": TRACK,
        "records": EXPECTED_RECORDS,
        "eval_ids_sha256": canonical_hash(eval_ids),
        "target_map_sha256": canonical_hash(target_map),
        "label_balance": EXPECTED_LABEL_BALANCE,
        "formal_source_identity_sha256": reference_context["formal_source"][
            "identity_sha256"
        ],
        "request_release_identity_sha256": reference_context[
            "request_release_identity_sha256"
        ],
        "input_receipt_identity_sha256": reference_context["input_receipt"][
            "identity_sha256"
        ],
        "request_records_sha256": reference_context["request_records_sha256"],
        "decoding": reference_context["decoding"],
        "context_identity_sha256": canonical_hash(reference_context),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": LEADERBOARD_PROTOCOL,
        "status": LEADERBOARD_STATUS,
        "reference_model_key": REFERENCE_KEY,
        "comparability": {
            "status": "PASS",
            "same_release_source_protocol_and_eval_ids": True,
            "all_five_results_have_complete_formal_trust": True,
            "common": common,
        },
        "statistics_contract": {
            "accuracy_unit": "scene",
            "paired_delta_direction": "candidate_minus_qwen3_5_9b",
            "paired_bootstrap": {
                "method": "deterministic_percentile_bootstrap",
                "unit": "scene",
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "confidence": CONFIDENCE,
            },
            "prediction_A_rate_derivation": (
                "The formal binary outputs are fully parsed and uncapped; a wrong prediction "
                "is therefore the opposite of target_stratum."
            ),
        },
        "leaderboard": rows,
        "live_rehash": ledger.report(),
        "trust": {
            "score_available": True,
            "leaderboard_rows": EXPECTED_MODELS,
            "paired_scenes_per_comparison": EXPECTED_RECORDS,
            "formal_held_out_evaluation_only": True,
            "training_materialization_eligible": False,
        },
    }
    report["identity_sha256"] = canonical_hash(report)
    return report


def _specs_with_overrides(values: Iterable[str]) -> tuple[ModelSpec, ...]:
    overrides: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or not path or key in overrides:
            raise LeaderboardError("--result must be a unique MODEL_KEY=/absolute/result.json")
        overrides[key] = Path(path)
    expected = {spec.key for spec in DEFAULT_MODEL_SPECS}
    if overrides and set(overrides) != expected:
        raise LeaderboardError(f"--result overrides must cover exactly: {', '.join(sorted(expected))}")
    return tuple(
        ModelSpec(
            key=spec.key,
            display_name=spec.display_name,
            role=spec.role,
            result_path=overrides.get(spec.key, spec.result_path),
            expected_model_inventory_identity_sha256=spec.expected_model_inventory_identity_sha256,
        )
        for spec in DEFAULT_MODEL_SPECS
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="MODEL_KEY=PATH",
        help="override all five default result paths while retaining pinned model identities",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        specs = _specs_with_overrides(args.result)
        report = compile_leaderboard(specs)
        output = args.out.expanduser().resolve()
        if output.exists():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise LeaderboardError(f"existing output is invalid: {output}") from exc
            if existing != report:
                raise LeaderboardError(f"refusing to overwrite a different leaderboard: {output}")
        else:
            _atomic_json(output, report)
        print(json.dumps({"status": report["status"], "out": str(output), "identity_sha256": report["identity_sha256"]}))
        return 0
    except LeaderboardError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
