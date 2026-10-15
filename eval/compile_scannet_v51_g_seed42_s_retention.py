#!/usr/bin/env python3
"""Compile the seed-42 G-only checkpoint's formal40 S-oracle retention diagnostic.

This compiler is intentionally independent of the five-model S-oracle leaderboard.
It accepts one frozen Qwen3.5-9B reference result and one G-only post-trained result,
rehashes their complete provenance, verifies the 4xL20 execution receipt and G-only
training lineage, and derives paired scene statistics from the common 40 records.

The resulting artifact is diagnostic evidence only.  It does not establish teacher
quality, FirstFault performance, multi-seed retention, or general anti-forgetting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
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
EXECUTION_PROTOCOL = "gsa-depth-v5.1-formal40-s-oracle-execution-v1"
TRAINING_PROTOCOL = "semantic-audit-v3-g-family-scoped-v2-independent-qualification"
QUALIFICATION_PROTOCOL = "g-family-specialist-qualification-pilot-v2"
DIAGNOSTIC_PROTOCOL = "gsa-depth-v5.1-formal40-g-seed42-s-retention-diagnostic-v1"
RESULT_STATUS = "FORMAL_S_ORACLE_SCORE_COMPLETE"
DIAGNOSTIC_STATUS = "SINGLE_SEED_S_RETENTION_DIAGNOSTIC_COMPLETE"
TRACK = "S_oracle_grounded_pair"
RELEASE = "formal40"
EXPECTED_RECORDS = 40
EXPECTED_LABEL_BALANCE = {"A": 20, "B": 20}
BOOTSTRAP_SEED = 51042
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95
EXPECTED_TRAIN_ROWS = 84
EXPECTED_TRAIN_IMAGES = 24
EXPECTED_TRAIN_SEED = 42
EXPECTED_TRAIN_STEPS = 53
_SHA256 = re.compile(r"[0-9a-f]{64}")

DEFAULT_BASE_RESULT = (
    ROOT / "output/scannet_rgb_depth_v51_formal40_qwen35_9b_20260823_v1/s_oracle/result.json"
)
DEFAULT_CANDIDATE_RESULT = (
    ROOT / "output/scannet_rgb_depth_v51_s_oracle_g_seed42_20260823_v1/s_oracle/result.json"
)
DEFAULT_EXECUTION_RECEIPT = (
    ROOT
    / "output/scannet_rgb_depth_v51_s_oracle_g_seed42_20260823_v1/s_oracle/execution_receipt.json"
)
DEFAULT_TRAINING_PROVENANCE = (
    ROOT
    / "output/g_family_v3_scoped_v2_qualification_eval_4gpu_seed42_local4_20260821_125929_tmux/g_family_qualification_eval_inputs_v2.json"
)
DEFAULT_QUALIFICATION_BINDING = (
    ROOT
    / "output/g_family_v3_scoped_v2_qualification_eval_4gpu_seed42_local4_20260821_125929_tmux/model_bindings/specialist.json"
)
DEFAULT_OUT = ROOT / "artifacts/results/scannet_v51_g_seed42_s_retention_diagnostic.json"


class DiagnosticError(ValueError):
    """Raised when the diagnostic cannot be compiled without weakening trust."""


@dataclass(frozen=True)
class DiagnosticInputs:
    base_result: Path
    candidate_result: Path
    execution_receipt: Path
    training_provenance: Path
    qualification_binding: Path
    expected_base_result_identity: str
    expected_candidate_result_identity: str
    expected_base_model_identity: str
    expected_candidate_model_identity: str
    expected_execution_identity: str
    expected_training_provenance_identity: str
    expected_qualification_binding_identity: str
    expected_adapter_identity: str


DEFAULT_INPUTS = DiagnosticInputs(
    base_result=DEFAULT_BASE_RESULT,
    candidate_result=DEFAULT_CANDIDATE_RESULT,
    execution_receipt=DEFAULT_EXECUTION_RECEIPT,
    training_provenance=DEFAULT_TRAINING_PROVENANCE,
    qualification_binding=DEFAULT_QUALIFICATION_BINDING,
    expected_base_result_identity="87ccaff7811332628395ba85ae227bccaae07801642c65e4f3cd80bc24637ef8",
    expected_candidate_result_identity="ae700ce2c4a377e3a789019bc7ed0c5b65a27b2cd303ceafa2d36a7da13cc1fd",
    expected_base_model_identity="dfad48fe82fcf17f8bad486b15c8077086eb8a10beba7cde8f6ce94bb2811f65",
    expected_candidate_model_identity="731d59a0181a8d7e2fae9d942049548149a1c2fc8a8ca2001693d94aa71bd34f",
    expected_execution_identity="d57071fdfeb10f536409ff8314f8a5e88aba235d54eeb218ba615946f87493b0",
    expected_training_provenance_identity=(
        "6f5dff52695b2297059a1b9f5730baa7aa98787a410a201cc4722b26d0a5cf8c"
    ),
    expected_qualification_binding_identity=(
        "701d142fec438f15d03d7aec14f166c35d6112474499d8662d1424fa03e2d1a7"
    ),
    expected_adapter_identity="79a727945e66a70301019770db06b2654fa7a5ba939ad846fbc4285c4fe7922b",
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
        raise DiagnosticError("cannot form a ratio with a non-positive denominator")
    return _round(numerator / denominator)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise DiagnosticError(f"{label} drift: expected {expected!r}, got {actual!r}")


def _verify_identity(value: Any, label: str, *, expected: str | None = None) -> None:
    if not isinstance(value, dict):
        raise DiagnosticError(f"{label} must be an object")
    identity = value.get("identity_sha256")
    if not isinstance(identity, str) or not _SHA256.fullmatch(identity):
        raise DiagnosticError(f"{label} identity is absent or invalid")
    unsigned = {key: item for key, item in value.items() if key != "identity_sha256"}
    _require_equal(identity, canonical_hash(unsigned), f"{label}.identity_sha256")
    if expected is not None:
        _require_equal(identity, expected, f"{label}.pinned_identity_sha256")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise DiagnosticError("cannot compute a percentile from no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_ci(values: Sequence[int], *, paired: bool) -> dict[str, Any]:
    allowed = {-1, 0, 1} if paired else {0, 1}
    if not values or any(value not in allowed for value in values):
        raise DiagnosticError(f"invalid {'paired' if paired else 'binary'} bootstrap inputs")
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    alpha = (1.0 - CONFIDENCE) / 2.0
    report: dict[str, Any] = {
        "method": "deterministic_percentile_bootstrap",
        "unit": "scene",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "confidence": CONFIDENCE,
        "ci": [
            _round(_percentile(means, alpha)),
            _round(_percentile(means, 1.0 - alpha)),
        ],
    }
    report["paired_scenes" if paired else "scenes"] = n
    return report


def _wilson_ci(successes: int, total: int) -> dict[str, Any]:
    if total <= 0 or not 0 <= successes <= total:
        raise DiagnosticError("invalid Wilson interval inputs")
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


class RehashLedger:
    """Rehash each unique file once and reject size, digest, or mtime drift."""

    def __init__(self) -> None:
        self._files: dict[Path, dict[str, Any]] = {}
        self.binding_occurrences = 0
        self.explicit_json_files = 0

    @staticmethod
    def _hash_path(path: Path) -> tuple[int, int, str]:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise DiagnosticError(f"file changed during live rehash: {path}")
        return after.st_size, after.st_mtime_ns, digest.hexdigest()

    def live(self, path: Path) -> dict[str, Any]:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise DiagnosticError(f"bound file is missing: {path}")
        cached = self._files.get(path)
        if cached is not None:
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (cached["bytes"], cached["mtime_ns"]):
                raise DiagnosticError(f"file changed after live rehash: {path}")
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

    def binding_for(self, path: Path) -> dict[str, Any]:
        live = self.live(path)
        return {key: live[key] for key in ("path", "bytes", "sha256")}

    def read_json(self, path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
        path = path.expanduser().resolve()
        live = self.live(path)
        self.explicit_json_files += 1
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DiagnosticError(f"{label} is not valid JSON: {path}") from exc
        if not isinstance(value, dict):
            raise DiagnosticError(f"{label} must be a JSON object")
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (live["bytes"], live["mtime_ns"]):
            raise DiagnosticError(f"{label} changed after live rehash: {path}")
        return value, {key: live[key] for key in ("path", "bytes", "sha256")}

    def verify_binding(self, binding: Any, label: str) -> None:
        self.binding_occurrences += 1
        if not isinstance(binding, dict):
            raise DiagnosticError(f"{label}: file binding must be an object")
        allowed = {"path", "bytes", "sha256", "mtime_ns"}
        if not {"path", "bytes", "sha256"}.issubset(binding) or not set(binding).issubset(
            allowed
        ):
            raise DiagnosticError(f"{label}: file binding schema drift")
        path_value = binding.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise DiagnosticError(f"{label}: invalid bound path")
        path = Path(path_value).expanduser().resolve()
        if str(path) != path_value:
            raise DiagnosticError(f"{label}: bound path is not canonical")
        expected_sha = binding.get("sha256")
        if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
            raise DiagnosticError(f"{label}: invalid bound SHA-256")
        live = self.live(path)
        _require_equal(binding.get("bytes"), live["bytes"], f"{label}.bytes")
        _require_equal(expected_sha, live["sha256"], f"{label}.sha256")
        if "mtime_ns" in binding:
            _require_equal(binding["mtime_ns"], live["mtime_ns"], f"{label}.mtime_ns")

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
            "explicit_json_files_rehashed": self.explicit_json_files,
            "binding_occurrences_verified": self.binding_occurrences,
            "unique_files_rehashed": len(bindings),
            "unique_bytes_rehashed": sum(item["bytes"] for item in bindings),
            "verified_file_set_sha256": canonical_hash(bindings),
            "all_size_sha256_and_bound_weight_mtime_checks_match": True,
        }


def _validate_metric_block(block: Any, *, total: int, correct: int, label: str) -> None:
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


def _validate_items(result: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    raw_items = result.get("per_item")
    if not isinstance(raw_items, list) or len(raw_items) != EXPECTED_RECORDS:
        raise DiagnosticError(f"{label}.per_item must contain exactly 40 records")
    items: dict[str, dict[str, Any]] = {}
    parse_status: Counter[str] = Counter()
    expected_keys = {
        "eval_id",
        "target_stratum",
        "parse_status",
        "parsed",
        "hit_token_cap",
        "correct",
    }
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise DiagnosticError(f"{label}.per_item[{index}] schema drift")
        eval_id = item.get("eval_id")
        if not isinstance(eval_id, str) or not eval_id or eval_id in items:
            raise DiagnosticError(f"{label}: missing or duplicate eval_id")
        if item.get("target_stratum") not in {"A", "B"}:
            raise DiagnosticError(f"{label}/{eval_id}: target is not A/B")
        if item.get("parsed") is not True or item.get("hit_token_cap") is not False:
            raise DiagnosticError(f"{label}/{eval_id}: response is not parsed and uncapped")
        if not isinstance(item.get("parse_status"), str) or not item["parse_status"]:
            raise DiagnosticError(f"{label}/{eval_id}: invalid parse status")
        if not isinstance(item.get("correct"), bool):
            raise DiagnosticError(f"{label}/{eval_id}: correctness is not Boolean")
        items[eval_id] = dict(item)
        parse_status[item["parse_status"]] += 1

    balance = Counter(item["target_stratum"] for item in items.values())
    _require_equal(dict(sorted(balance.items())), EXPECTED_LABEL_BALANCE, f"{label}.label_balance")
    correct = sum(int(item["correct"]) for item in items.values())
    _validate_metric_block(result.get("metrics"), total=40, correct=correct, label=f"{label}.metrics")
    strata = result.get("ab_strata")
    if not isinstance(strata, dict) or set(strata) != {"A", "B"}:
        raise DiagnosticError(f"{label}.ab_strata schema drift")
    for target in ("A", "B"):
        selected = [item for item in items.values() if item["target_stratum"] == target]
        target_correct = sum(int(item["correct"]) for item in selected)
        _validate_metric_block(
            strata[target], total=len(selected), correct=target_correct, label=f"{label}.{target}"
        )
    _require_equal(result.get("parse_status"), dict(sorted(parse_status.items())), f"{label}.parse_status")
    return items


def _validate_result(
    result: dict[str, Any],
    *,
    label: str,
    expected_result_identity: str,
    expected_model_identity: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    _verify_identity(result, f"{label}.result", expected=expected_result_identity)
    for key, expected in (
        ("result_schema_version", 1),
        ("protocol", RESULT_PROTOCOL),
        ("evaluation_protocol", EVALUATION_PROTOCOL),
        ("release", RELEASE),
        ("status", RESULT_STATUS),
        ("task_contract", EXPECTED_TASK_CONTRACT),
        ("trust", EXPECTED_TRUST),
    ):
        _require_equal(result.get(key), expected, f"{label}.{key}")
    _require_equal(
        result.get("track"),
        {
            "name": TRACK,
            "status": "FORMAL_SCORE_COMPLETE",
            "context": "human_ground_truth_bounding_boxes",
        },
        f"{label}.track",
    )

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
        ("records", 40),
        ("unique_scenes", 40),
        ("unique_images", 40),
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
    inventory = provenance.get("model_inventory")
    _verify_identity(inventory, f"{label}.model_inventory", expected=expected_model_identity)
    _require_equal(
        inventory.get("weight_bytes_fully_sha256_hashed"),
        True,
        f"{label}.model_inventory.weight_bytes_fully_sha256_hashed",
    )
    bundle = provenance.get("s_oracle_bundle")
    _verify_identity(bundle, f"{label}.s_oracle_bundle")
    for key, expected in (
        ("verified", True),
        ("protocol", RUN_PROTOCOL),
        ("evaluation_protocol", EVALUATION_PROTOCOL),
        ("track", TRACK),
        ("n_predictions", 40),
    ):
        _require_equal(bundle.get(key), expected, f"{label}.s_oracle_bundle.{key}")
    _require_equal(
        bundle.get("model_inventory_identity_sha256"),
        expected_model_identity,
        f"{label}.s_oracle_bundle.model_inventory_identity_sha256",
    )
    _require_equal(
        provenance.get("model_binding_identity_sha256"),
        bundle.get("model_binding_identity_sha256"),
        f"{label}.model_binding_identity_sha256",
    )
    items = _validate_items(result, label)
    context = {
        "formal_source_identity_sha256": formal_source["identity_sha256"],
        "input_receipt_identity_sha256": input_receipt["identity_sha256"],
        "request_release_identity_sha256": provenance.get("request_release_identity_sha256"),
        "request_records_sha256": bundle.get("request_records_sha256"),
        "requests_sha256": (bundle.get("requests") or {}).get("sha256"),
        "decoding": bundle.get("decoding"),
    }
    return items, context


def _binary_prediction(item: Mapping[str, Any]) -> str:
    target = str(item["target_stratum"])
    if bool(item["correct"]):
        return target
    return "B" if target == "A" else "A"


def _position_summary(items: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    predicted = Counter(_binary_prediction(item) for item in items.values())
    strata = {
        target: {
            "n": sum(item["target_stratum"] == target for item in items.values()),
            "correct": sum(
                item["target_stratum"] == target and bool(item["correct"])
                for item in items.values()
            ),
        }
        for target in ("A", "B")
    }
    for value in strata.values():
        value["accuracy"] = _ratio(value["correct"], value["n"])
    return {
        "inference": "binary_target_plus_correctness_on_fully_parsed_A_B_items",
        "predicted_A": predicted["A"],
        "predicted_B": predicted["B"],
        "predicted_A_rate": _ratio(predicted["A"], len(items)),
        "A_rate_minus_balanced_target": _round(_ratio(predicted["A"], len(items)) - 0.5),
        "A_minus_B_prediction_rate": _round((predicted["A"] - predicted["B"]) / len(items)),
        "strata": strata,
        "A_minus_B_accuracy": _round(strata["A"]["accuracy"] - strata["B"]["accuracy"]),
    }


def _paired_summary(
    base_items: Mapping[str, Mapping[str, Any]],
    candidate_items: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_equal(set(candidate_items), set(base_items), "paired.eval_ids")
    records: list[dict[str, Any]] = []
    for eval_id in sorted(base_items):
        base = base_items[eval_id]
        candidate = candidate_items[eval_id]
        _require_equal(
            candidate["target_stratum"], base["target_stratum"], f"paired.{eval_id}.target"
        )
        base_correct = int(bool(base["correct"]))
        candidate_correct = int(bool(candidate["correct"]))
        records.append(
            {
                "eval_id": eval_id,
                "target_stratum": base["target_stratum"],
                "base_correct": bool(base_correct),
                "candidate_correct": bool(candidate_correct),
                "candidate_minus_base": candidate_correct - base_correct,
                "base_inferred_prediction": _binary_prediction(base),
                "candidate_inferred_prediction": _binary_prediction(candidate),
            }
        )
    differences = [record["candidate_minus_base"] for record in records]
    both_correct = sum(record["base_correct"] and record["candidate_correct"] for record in records)
    both_wrong = sum(not record["base_correct"] and not record["candidate_correct"] for record in records)
    candidate_only = sum(
        record["candidate_correct"] and not record["base_correct"] for record in records
    )
    base_only = sum(record["base_correct"] and not record["candidate_correct"] for record in records)
    base_correct = sum(int(record["base_correct"]) for record in records)
    candidate_correct = sum(int(record["candidate_correct"]) for record in records)
    summary = {
        "n_scenes": len(records),
        "base": {
            "correct": base_correct,
            "accuracy": _ratio(base_correct, len(records)),
        },
        "candidate": {
            "correct": candidate_correct,
            "accuracy": _ratio(candidate_correct, len(records)),
        },
        "accuracy_delta_candidate_minus_base": _round(
            (candidate_correct - base_correct) / len(records)
        ),
        "paired_scene_bootstrap_95_ci": _bootstrap_ci(differences, paired=True),
        "discordance": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "candidate_only_correct": candidate_only,
            "base_only_correct": base_only,
            "discordant_total": candidate_only + base_only,
            "net_candidate_correct_advantage": candidate_only - base_only,
        },
        "identical_scene_correctness_vector": all(value == 0 for value in differences),
        "identical_inferred_binary_predictions": all(
            record["base_inferred_prediction"] == record["candidate_inferred_prediction"]
            for record in records
        ),
    }
    return summary, records


def _validate_execution_receipt(
    receipt: dict[str, Any],
    *,
    expected_identity: str,
    candidate_binding: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    ledger: RehashLedger,
) -> dict[str, Any]:
    _verify_identity(receipt, "execution_receipt", expected=expected_identity)
    _require_equal(receipt.get("schema_version"), 1, "execution_receipt.schema_version")
    _require_equal(receipt.get("protocol"), EXECUTION_PROTOCOL, "execution_receipt.protocol")
    _require_equal(receipt.get("physical_gpu_count"), 4, "execution_receipt.physical_gpu_count")
    _require_equal(
        receipt.get("training_materialization_eligible"),
        False,
        "execution_receipt.training_materialization_eligible",
    )
    _require_equal(receipt.get("formal_result"), candidate_binding, "execution_receipt.formal_result")
    candidate_provenance = candidate_result["provenance"]
    candidate_bundle = candidate_provenance["s_oracle_bundle"]
    for receipt_key, expected in (
        ("model_binding", candidate_provenance["model_binding"]),
        ("predictions", candidate_bundle["merged_predictions"]),
        ("prediction_manifest", candidate_bundle["prediction_manifest"]),
        ("input_receipt", candidate_bundle["input_receipt"]),
    ):
        _require_equal(receipt.get(receipt_key), expected, f"execution_receipt.{receipt_key}")

    hardware_binding = receipt.get("hardware_inventory")
    if not isinstance(hardware_binding, dict):
        raise DiagnosticError("execution_receipt.hardware_inventory is absent")
    hardware_path = Path(hardware_binding["path"])
    with hardware_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [[field.strip() for field in row] for row in csv.reader(handle) if row]
    if len(rows) != 4:
        raise DiagnosticError("execution hardware inventory must contain exactly four GPUs")
    if [row[0] for row in rows] != ["0", "1", "2", "3"]:
        raise DiagnosticError("execution hardware inventory indices drift")
    if any(len(row) != 5 or "L20" not in row[2].upper() or int(row[3]) < 44_000 for row in rows):
        raise DiagnosticError("execution hardware inventory is not four >=44GB L20 GPUs")
    if len({row[1] for row in rows}) != 4:
        raise DiagnosticError("execution hardware inventory repeats a physical GPU UUID")
    ledger.live(hardware_path)
    return {
        "protocol": receipt["protocol"],
        "physical_gpu_count": 4,
        "hardware": "4xNVIDIA_L20",
        "unique_gpu_uuids": 4,
        "started_at_utc": receipt.get("started_at_utc"),
        "finished_at_utc": receipt.get("finished_at_utc"),
        "identity_sha256": receipt["identity_sha256"],
    }


def _load_json_binding(binding: Mapping[str, Any], label: str, ledger: RehashLedger) -> dict[str, Any]:
    ledger.verify_binding(binding, label)
    value, observed = ledger.read_json(Path(str(binding["path"])), label)
    _require_equal(observed, {k: binding[k] for k in ("path", "bytes", "sha256")}, label)
    return value


def _validate_training_lineage(
    training: dict[str, Any],
    qualification_binding: dict[str, Any],
    *,
    inputs: DiagnosticInputs,
    candidate_result: Mapping[str, Any],
    training_binding: Mapping[str, Any],
    ledger: RehashLedger,
) -> dict[str, Any]:
    _verify_identity(
        training,
        "training_provenance",
        expected=inputs.expected_training_provenance_identity,
    )
    for key, expected in (
        ("schema_version", 1),
        ("protocol", QUALIFICATION_PROTOCOL),
        ("scope", "G_family_bbox_detection_mask_point_only"),
        ("publication_claim_eligible", False),
        ("authorization", "evaluation_only"),
    ):
        _require_equal(training.get(key), expected, f"training_provenance.{key}")
    data = training.get("data")
    if not isinstance(data, dict):
        raise DiagnosticError("training_provenance.data is absent")
    for key, expected in (
        ("train_rows", EXPECTED_TRAIN_ROWS),
        ("train_images", EXPECTED_TRAIN_IMAGES),
        ("cross_split_image_sha256_overlap", 0),
        ("held_out_from_gradient_updates", True),
        ("used_during_training_for_validation_and_model_selection", False),
    ):
        _require_equal(data.get(key), expected, f"training_provenance.data.{key}")
    training_config = training.get("training")
    if not isinstance(training_config, dict):
        raise DiagnosticError("training_provenance.training is absent")
    for key, expected in (
        ("hardware", "4xL20"),
        ("world_size", 4),
        ("max_steps", EXPECTED_TRAIN_STEPS),
        ("seed", EXPECTED_TRAIN_SEED),
        ("seed_binding", "receipt_and_args"),
    ):
        _require_equal(training_config.get(key), expected, f"training_provenance.training.{key}")

    training_receipt = _load_json_binding(
        training["training_receipt"], "training_receipt", ledger
    )
    for key, expected in (
        ("protocol", TRAINING_PROTOCOL),
        ("hardware", "4xL20"),
        ("world_size", 4),
        ("train_rows", EXPECTED_TRAIN_ROWS),
        ("train_images", EXPECTED_TRAIN_IMAGES),
        ("seed", EXPECTED_TRAIN_SEED),
        ("data_seed", EXPECTED_TRAIN_SEED),
        ("max_steps", EXPECTED_TRAIN_STEPS),
        ("qualification_exposed_to_trainer", False),
    ):
        _require_equal(training_receipt.get(key), expected, f"training_receipt.{key}")
    train_binding = data.get("train")
    manifest_binding = data.get("manifest")
    if not isinstance(train_binding, dict) or not isinstance(manifest_binding, dict):
        raise DiagnosticError("training data bindings are absent")
    _require_equal(training_receipt.get("train"), {
        "path": train_binding["path"], "sha256": train_binding["sha256"]
    }, "training_receipt.train")
    _require_equal(training_receipt.get("manifest"), {
        "path": manifest_binding["path"], "sha256": manifest_binding["sha256"]
    }, "training_receipt.manifest")

    args = _load_json_binding(training["training_args"], "training_args", ledger)
    for key, expected in (
        ("seed", 42),
        ("data_seed", 42),
        ("max_steps", 53),
        ("per_device_train_batch_size", 1),
        ("gradient_accumulation_steps", 2),
    ):
        _require_equal(args.get(key), expected, f"training_args.{key}")
    _require_equal(args.get("dataset"), [train_binding["path"]], "training_args.dataset")
    deepspeed = args.get("deepspeed") or {}
    _require_equal(
        (deepspeed.get("zero_optimization") or {}).get("stage"), 2, "training_args.deepspeed_stage"
    )

    adapter = ((training.get("models") or {}).get("specialist_adapter") or {})
    _verify_identity(adapter, "training_provenance.specialist_adapter", expected=inputs.expected_adapter_identity)
    _require_equal(adapter.get("step"), 53, "training_provenance.specialist_adapter.step")
    trainer_state = _load_json_binding(adapter["trainer_state"], "trainer_state", ledger)
    _require_equal(trainer_state.get("global_step"), 53, "trainer_state.global_step")
    _require_equal(trainer_state.get("max_steps"), 53, "trainer_state.max_steps")

    train_path = Path(train_binding["path"])
    rows: list[dict[str, Any]] = []
    with train_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DiagnosticError(f"invalid training JSONL row {line_number}") from exc
            if not isinstance(row, dict):
                raise DiagnosticError(f"training row {line_number} is not an object")
            rows.append(row)
    _require_equal(len(rows), 84, "live_training_rows")
    if len({row.get("sample_id") for row in rows}) != 84:
        raise DiagnosticError("training sample ids are missing or duplicated")
    if any(row.get("gsa_family") != "G" for row in rows):
        raise DiagnosticError("training corpus contains a non-G row")
    if any(row.get("gsa_form") not in {"bbox", "detection", "mask", "point"} for row in rows):
        raise DiagnosticError("training corpus contains an out-of-scope form")
    image_paths: set[Path] = set()
    for row in rows:
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise DiagnosticError("each training row must bind exactly one image")
        image_paths.add(Path(images[0]).expanduser().resolve())
    _require_equal(len(image_paths), 24, "live_training_images")
    manifest = _load_json_binding(manifest_binding, "training_manifest", ledger)
    _require_equal(manifest.get("protocol"), TRAINING_PROTOCOL, "training_manifest.protocol")
    _require_equal((manifest.get("counts") or {}).get("train"), 84, "training_manifest.counts.train")
    _require_equal(
        (manifest.get("unique_images") or {}).get("train"), 24, "training_manifest.unique_images.train"
    )
    live_image_hashes = sorted(ledger.live(path)["sha256"] for path in image_paths)
    manifest_hashes = sorted((manifest.get("image_split") or {}).get("train_image_sha256") or [])
    _require_equal(live_image_hashes, manifest_hashes, "training_manifest.train_image_sha256")

    _verify_identity(
        qualification_binding,
        "qualification_model_binding",
        expected=inputs.expected_qualification_binding_identity,
    )
    _require_equal(
        qualification_binding.get("protocol"),
        QUALIFICATION_PROTOCOL,
        "qualification_model_binding.protocol",
    )
    _require_equal(
        qualification_binding.get("role"), "specialist", "qualification_model_binding.role"
    )
    _require_equal(
        qualification_binding.get("input_receipt"),
        dict(training_binding),
        "qualification_model_binding.input_receipt",
    )
    qualification_model = qualification_binding.get("model") or {}
    candidate_model = candidate_result["provenance"]["model_inventory"]
    _require_equal(
        qualification_model.get("path"), candidate_model.get("path"), "qualification_model.path"
    )
    expected_dirname = f"specialist_{inputs.expected_adapter_identity[:16]}"
    _require_equal(Path(candidate_model["path"]).name, expected_dirname, "merged_model.dirname")
    for name, binding in (qualification_model.get("metadata_files") or {}).items():
        _require_equal(
            (candidate_model.get("metadata_files") or {}).get(name),
            binding,
            f"qualification_model.metadata_files.{name}",
        )
    qualification_weights = qualification_model.get("weight_files") or []
    candidate_weights = candidate_model.get("weight_files") or []
    if len(qualification_weights) != len(candidate_weights) or not qualification_weights:
        raise DiagnosticError("qualification and formal-eval weight inventories differ")
    for index, (qualified, candidate) in enumerate(zip(qualification_weights, candidate_weights)):
        _require_equal(qualified.get("path"), candidate.get("path"), f"weight[{index}].path")
        _require_equal(qualified.get("bytes"), candidate.get("bytes"), f"weight[{index}].bytes")

    return {
        "scope": training["scope"],
        "seed": 42,
        "seeds": 1,
        "train_rows": 84,
        "train_images": 24,
        "train_forms": dict(sorted(Counter(row["gsa_form"] for row in rows).items())),
        "max_steps": 53,
        "hardware": "4xL20",
        "adapter_identity_sha256": adapter["identity_sha256"],
        "merged_model_inventory_identity_sha256": candidate_model["identity_sha256"],
        "qualification_binding_identity_sha256": qualification_binding["identity_sha256"],
        "publication_claim_eligible": False,
        "authorization": "evaluation_only",
        "lineage_status": "BOUND_TRAINING_RECEIPT_ADAPTER_MERGE_AND_FORMAL_MODEL_MATCH",
    }


def compile_diagnostic(inputs: DiagnosticInputs = DEFAULT_INPUTS) -> dict[str, Any]:
    ledger = RehashLedger()
    base_result, base_binding = ledger.read_json(inputs.base_result, "base_result")
    candidate_result, candidate_binding = ledger.read_json(
        inputs.candidate_result, "candidate_result"
    )
    execution_receipt, execution_binding = ledger.read_json(
        inputs.execution_receipt, "execution_receipt"
    )
    training, training_binding = ledger.read_json(
        inputs.training_provenance, "training_provenance"
    )
    qualification_binding, qualification_file_binding = ledger.read_json(
        inputs.qualification_binding, "qualification_model_binding"
    )

    base_items, base_context = _validate_result(
        base_result,
        label="base",
        expected_result_identity=inputs.expected_base_result_identity,
        expected_model_identity=inputs.expected_base_model_identity,
    )
    candidate_items, candidate_context = _validate_result(
        candidate_result,
        label="candidate",
        expected_result_identity=inputs.expected_candidate_result_identity,
        expected_model_identity=inputs.expected_candidate_model_identity,
    )
    _require_equal(candidate_context, base_context, "formal_comparability_context")
    ledger.verify_all_bindings(base_result, "base_result")
    ledger.verify_all_bindings(candidate_result, "candidate_result")
    ledger.verify_all_bindings(execution_receipt, "execution_receipt")
    ledger.verify_all_bindings(training, "training_provenance")
    ledger.verify_all_bindings(qualification_binding, "qualification_model_binding")

    execution = _validate_execution_receipt(
        execution_receipt,
        expected_identity=inputs.expected_execution_identity,
        candidate_binding=candidate_binding,
        candidate_result=candidate_result,
        ledger=ledger,
    )
    training_lineage = _validate_training_lineage(
        training,
        qualification_binding,
        inputs=inputs,
        candidate_result=candidate_result,
        training_binding=training_binding,
        ledger=ledger,
    )
    paired, paired_records = _paired_summary(base_items, candidate_items)
    base_position = _position_summary(base_items)
    candidate_position = _position_summary(candidate_items)

    common_target_map = [
        {"eval_id": eval_id, "target_stratum": base_items[eval_id]["target_stratum"]}
        for eval_id in sorted(base_items)
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": DIAGNOSTIC_PROTOCOL,
        "status": DIAGNOSTIC_STATUS,
        "release": RELEASE,
        "track": TRACK,
        "evidence_tier": "single_seed_post_training_S_retention_diagnostic",
        "comparability": {
            "status": "PASS",
            "same_40_scene_eval_ids_targets_release_source_and_decoding": True,
            "records": 40,
            "label_balance": EXPECTED_LABEL_BALANCE,
            "eval_ids_sha256": canonical_hash(sorted(base_items)),
            "target_map_sha256": canonical_hash(common_target_map),
            **base_context,
        },
        "paired_result": paired,
        "position_bias": {
            "base": base_position,
            "candidate": candidate_position,
            "candidate_minus_base_predicted_A_rate": _round(
                candidate_position["predicted_A_rate"] - base_position["predicted_A_rate"]
            ),
            "candidate_minus_base_A_minus_B_accuracy": _round(
                candidate_position["A_minus_B_accuracy"] - base_position["A_minus_B_accuracy"]
            ),
        },
        "paired_scene_records": paired_records,
        "training_lineage": training_lineage,
        "execution": execution,
        "claim_scope": {
            "supported": (
                "On this frozen 40-scene S-oracle panel, the seed-42 G-only checkpoint "
                "matches its Qwen3.5-9B base scene by scene."
            ),
            "not_supported": [
                "teacher quality or teacher eligibility",
                "FirstFault or routed-distillation performance",
                "multi-seed spatial retention",
                "general anti-forgetting",
                "SOTA performance",
            ],
            "single_seed": True,
            "diagnostic_only": True,
        },
        "provenance": {
            "base_result": base_binding,
            "candidate_result": candidate_binding,
            "execution_receipt": execution_binding,
            "training_provenance": training_binding,
            "qualification_model_binding": qualification_file_binding,
            "base_result_identity_sha256": base_result["identity_sha256"],
            "candidate_result_identity_sha256": candidate_result["identity_sha256"],
            "base_model_inventory_identity_sha256": inputs.expected_base_model_identity,
            "candidate_model_inventory_identity_sha256": inputs.expected_candidate_model_identity,
            "training_materialization_eligible": False,
        },
        "live_rehash": ledger.report(),
    }
    report["identity_sha256"] = canonical_hash(report)
    return report


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
    temporary.replace(path)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    report = compile_diagnostic()
    _atomic_write_json(args.out, report)
    paired = report["paired_result"]
    position = report["position_bias"]["candidate"]
    print(f"wrote: {args.out.expanduser().resolve()}")
    print(
        "candidate/base: "
        f"{paired['candidate']['correct']}/40 vs {paired['base']['correct']}/40; "
        f"delta={paired['accuracy_delta_candidate_minus_base']:.3f}; "
        f"paired95={paired['paired_scene_bootstrap_95_ci']['ci']}"
    )
    print(
        f"discordant={paired['discordance']['discordant_total']}; "
        f"predicted_A={position['predicted_A']}/40; "
        f"identity={report['identity_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
