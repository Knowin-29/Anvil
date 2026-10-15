#!/usr/bin/env python3
"""Fail-closed scorer for the FirstFault paired-context evidence contract.

This module aggregates already stage-scored intervention outputs. It does not run a
model and does not implement task-native semantic metrics. Domain evaluators must
emit one bound cell per required probe; malformed model responses are represented as
``output_status=malformed`` and receive aggregate sample score zero here. Any item
with a malformed or capped dependency-closure probe is inconclusive for routing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from firstfault.routing import (  # noqa: E402
    RouteKind,
    confirm_state_use,
    response_scores_consistent,
    route_first_fault,
)

DEFAULT_CONTRACT = ROOT / "experiments/firstfault_intervention_v1.yaml"
SHA256_LEN = 64

EXPECTED_COMPARABILITY = {
    "same_item": "required",
    "same_observation": "required",
    "same_checkpoint": "required",
    "same_decoding": "required",
    "same_downstream_target": "required",
    "deterministic_descendant_regeneration": "required",
}
EXPECTED_LINEAGE = {
    "predicted_state_source": "exact_upstream_cell_emitted_state",
    "static_state_source": "manifest_bound_oracle_absent_corrupted_state",
    "context_identity": "canonical_sha256_of_ordered_context_sources",
    "malformed_emitted_state": "hashable_and_propagated_but_semantic_sample_zero",
}


class ContractError(ValueError):
    """Raised when an input cannot be interpreted under the frozen contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != SHA256_LEN:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    _require(actual == expected, f"{label} keys drift: expected {sorted(expected)}, got {sorted(actual)}")
    return value


EXPECTED_PROBES: dict[str, tuple[str, str | None, str]] = {
    "G.local": ("G", None, "none"),
    "S.predicted_G": ("S", "G", "predicted"),
    "S.oracle_G": ("S", "G", "oracle"),
    "S.absent_G": ("S", "G", "absent"),
    "S.corrupted_G": ("S", "G", "corrupted"),
    "A.predicted_GS": ("A", "G", "predicted"),
    "A.repair_G": ("A", "G", "oracle"),
    "A.oracle_GS": ("A", "S", "oracle"),
    "A.absent_G": ("A", "G", "absent"),
    "A.corrupted_G": ("A", "G", "corrupted"),
    "A.absent_S": ("A", "S", "absent"),
    "A.corrupted_S": ("A", "S", "corrupted"),
}

EXPECTED_PROBE_ROLES = {
    "G.local": "local_G_under_no_parents",
    "S.predicted_G": "S_under_predicted_G",
    "S.oracle_G": "S_under_oracle_G",
    "S.absent_G": "S_with_G_removed",
    "S.corrupted_G": "S_with_type_valid_corrupted_G",
    "A.predicted_GS": "A_under_predicted_G_then_predicted_S",
    "A.repair_G": "A_after_oracle_G_and_regenerated_predicted_S",
    "A.oracle_GS": "A_under_oracle_G_and_oracle_S",
    "A.absent_G": "A_after_G_removed_and_descendants_regenerated",
    "A.corrupted_G": "A_after_type_valid_corrupted_G_and_descendants_regenerated",
    "A.absent_S": "A_under_oracle_G_with_S_removed",
    "A.corrupted_S": "A_under_oracle_G_with_type_valid_corrupted_S",
}

EXPECTED_CHAINS: dict[str, tuple[str, ...]] = {
    "G_S": (
        "G.local",
        "S.predicted_G",
        "S.oracle_G",
        "S.absent_G",
        "S.corrupted_G",
    ),
    "G_S_A": tuple(EXPECTED_PROBES),
}

# Ordered context lineage for every probe. A predicted source names the upstream
# cell whose emitted state must be supplied; static modes bind a manifest state.
EXPECTED_CONTEXT_SOURCES: dict[str, tuple[tuple[str, str, str | None], ...]] = {
    "G.local": (),
    "S.predicted_G": (("G", "predicted", "G.local"),),
    "S.oracle_G": (("G", "oracle", None),),
    "S.absent_G": (("G", "absent", None),),
    "S.corrupted_G": (("G", "corrupted", None),),
    "A.predicted_GS": (
        ("G", "predicted", "G.local"),
        ("S", "predicted", "S.predicted_G"),
    ),
    "A.repair_G": (
        ("G", "oracle", None),
        ("S", "predicted", "S.oracle_G"),
    ),
    "A.oracle_GS": (("G", "oracle", None), ("S", "oracle", None)),
    "A.absent_G": (
        ("G", "absent", None),
        ("S", "predicted", "S.absent_G"),
    ),
    "A.corrupted_G": (
        ("G", "corrupted", None),
        ("S", "predicted", "S.corrupted_G"),
    ),
    "A.absent_S": (("G", "oracle", None), ("S", "absent", None)),
    "A.corrupted_S": (("G", "oracle", None), ("S", "corrupted", None)),
}

COST_FIELDS = (
    "model_queries",
    "teacher_queries",
    "model_input_tokens",
    "model_output_tokens",
    "teacher_input_tokens",
    "teacher_output_tokens",
    "specialist_scored_tokens",
    "model_cost_microusd",
    "teacher_cost_microusd",
)

TRANSITION_STATES = ("wrong", "correct")
TRANSITION_LABELS = tuple(
    f"{before}_to_{after}" for before in TRANSITION_STATES for after in TRANSITION_STATES
)
TRANSITION_EDGES = {"G_S": ("G_repair",), "G_S_A": ("G_repair", "S_repair")}

CONTRACT_KEYS = {
    "schema_version",
    "contract_id",
    "result_protocol",
    "evidence_role",
    "comparability",
    "lineage",
    "score_semantics",
    "cell_admission",
    "bootstrap",
    "cost_fields",
    "cost_accounting",
    "probe_definitions",
    "chains",
    "first_fault_states",
    "first_fault_transitions",
}


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    payload = _require_keys(payload, CONTRACT_KEYS, "contract")
    _require(payload.get("schema_version") == 1, "unsupported contract schema_version")
    _require(payload.get("contract_id") == "firstfault-intervention-v1", "contract_id drift")
    _require(payload.get("result_protocol") == "firstfault-intervention-result-v1", "result_protocol drift")
    _require(payload.get("evidence_role") == "paired_context_diagnosis", "evidence_role drift")
    _require(payload.get("comparability") == EXPECTED_COMPARABILITY, "comparability contract drift")
    _require(payload.get("lineage") == EXPECTED_LINEAGE, "lineage contract drift")
    probes = payload.get("probe_definitions")
    _require(isinstance(probes, dict) and set(probes) == set(EXPECTED_PROBES), "probe definition set drift")
    for probe_id, (target, intervention, mode) in EXPECTED_PROBES.items():
        probe = _require_keys(
            probes[probe_id],
            {
                "target_stage",
                "intervention_stage",
                "context_mode",
                "semantic_role",
                "semantic_metric_source",
            },
            f"probe_definitions.{probe_id}",
        )
        _require(probe.get("target_stage") == target, f"{probe_id}.target_stage drift")
        _require(probe.get("intervention_stage") == intervention, f"{probe_id}.intervention_stage drift")
        _require(probe.get("context_mode") == mode, f"{probe_id}.context_mode drift")
        _require(probe.get("semantic_role") == EXPECTED_PROBE_ROLES[probe_id], f"{probe_id}.semantic_role drift")
        _require(
            probe.get("semantic_metric_source") == "target.semantic_metric_contract",
            f"{probe_id}.semantic_metric_source drift",
        )
    chains = payload.get("chains")
    _require(isinstance(chains, dict) and set(chains) == set(EXPECTED_CHAINS), "chain definition set drift")
    for chain, probes_expected in EXPECTED_CHAINS.items():
        chain_spec = _require_keys(
            chains[chain], {"stages", "required_probes", "primary_target"}, f"chains.{chain}"
        )
        expected_stages = ["G", "S"] if chain == "G_S" else ["G", "S", "A"]
        _require(chain_spec["stages"] == expected_stages, f"{chain} stage order drift")
        _require(tuple(chain_spec["required_probes"]) == probes_expected, f"{chain} probe order drift")
        _require(chain_spec["primary_target"] == expected_stages[-1], f"{chain} primary target drift")
    _require(tuple(payload.get("cost_fields", ())) == COST_FIELDS, "cost field contract drift")
    _require(
        payload.get("cost_accounting")
        == {
            "receipt_scope": "per_cell",
            "monetary_unit": "micro_usd",
            "rate_policy_binding": "bundle_cost_policy_identity_sha256",
            "aggregation": "exact_integer_sum",
        },
        "cost accounting contract drift",
    )
    scoring = _require_keys(
        payload.get("score_semantics"),
        {
            "range",
            "correctness_threshold_source",
            "context_sensitivity_threshold_source",
            "monotonic_tolerance_source",
            "malformed_response",
            "malformed_semantic_score",
            "dependency_gap",
            "oracle_headroom",
            "headroom_recovery",
        },
        "score_semantics",
    )
    _require(scoring.get("malformed_response") == "sample_zero", "malformed response must remain sample zero")
    _require(scoring.get("malformed_semantic_score") is None, "malformed semantic score must remain null")
    _require(scoring.get("range") == [0.0, 1.0], "semantic score range drift")
    _require(
        scoring.get("correctness_threshold_source")
        == "manifest_target_semantic_metric_contract",
        "correctness threshold source drift",
    )
    _require(
        scoring.get("context_sensitivity_threshold_source")
        == "manifest_target_semantic_metric_contract",
        "context sensitivity threshold source drift",
    )
    _require(
        scoring.get("monotonic_tolerance_source")
        == "manifest_target_semantic_metric_contract",
        "monotonic tolerance source drift",
    )
    _require(scoring.get("dependency_gap") == "oracle_minus_predicted", "dependency gap definition drift")
    _require(
        scoring.get("oracle_headroom") == "positive_part_of_oracle_minus_predicted",
        "oracle headroom definition drift",
    )
    _require(
        scoring.get("headroom_recovery") == "bounded_repair_gain_over_positive_oracle_headroom",
        "headroom recovery definition drift",
    )
    admission = _require_keys(
        payload.get("cell_admission"),
        {"withhold_only_for", "max_malformed_rate", "max_truncation_rate", "gate_scope"},
        "cell_admission",
    )
    _require(
        admission.get("withhold_only_for")
        == [
            "protocol_or_binding_mismatch",
            "missing_duplicate_or_unexpected_uid",
            "incomplete_or_incorrect_shard_coverage",
            "preregistered_output_health_gate_failure",
        ],
        "cell withholding categories drift",
    )
    _require(admission.get("gate_scope") == "chain_target_probe", "cell admission gate scope drift")
    for field in ("max_malformed_rate", "max_truncation_rate"):
        value = admission.get(field)
        _require(not isinstance(value, bool) and isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0, f"invalid {field}")
    _require(float(admission["max_malformed_rate"]) == 0.25, "max_malformed_rate drift")
    _require(float(admission["max_truncation_rate"]) == 0.15, "max_truncation_rate drift")
    bootstrap = _require_keys(
        payload.get("bootstrap"),
        {"unit", "method", "confidence", "resamples", "seed"},
        "bootstrap",
    )
    _require(bootstrap.get("unit") == "scene_uid", "bootstrap unit must be scene_uid")
    _require(bootstrap.get("method") == "deterministic_percentile_cluster_bootstrap", "bootstrap method drift")
    _require(
        not isinstance(bootstrap.get("resamples"), bool)
        and bootstrap.get("resamples") == 10000,
        "bootstrap resamples drift",
    )
    _require(
        not isinstance(bootstrap.get("seed"), bool) and bootstrap.get("seed") == 51051,
        "bootstrap seed drift",
    )
    confidence = bootstrap.get("confidence")
    _require(
        not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and float(confidence) == 0.95,
        "bootstrap confidence drift",
    )
    _require(
        payload.get("first_fault_states")
        == ["Attributed", "Local", "Inconclusive"],
        "first-fault state set/order drift",
    )
    _require(
        payload.get("first_fault_transitions")
        == {
            "G_S": ["G_repair"],
            "G_S_A": ["G_repair", "S_repair"],
            "states": ["wrong", "correct"],
        },
        "first-fault transition contract drift",
    )
    return dict(payload)


def make_cell_uid(
    protocol: str,
    run_uid: str,
    checkpoint_identity_sha256: str,
    decoding_identity_sha256: str,
    item_uid: str,
    probe_id: str,
) -> str:
    return canonical_hash(
        {
            "protocol": protocol,
            "run_uid": run_uid,
            "checkpoint_identity_sha256": checkpoint_identity_sha256,
            "decoding_identity_sha256": decoding_identity_sha256,
            "item_uid": item_uid,
            "probe_id": probe_id,
        }
    )


def expected_shard(cell_uid: str, num_shards: int) -> int:
    _require(_is_sha256(cell_uid), "cell_uid must be lowercase SHA-256")
    _require(not isinstance(num_shards, bool) and isinstance(num_shards, int) and num_shards > 0, "num_shards must be positive")
    return int(cell_uid[:16], 16) % num_shards


def _validate_identity(payload: Mapping[str, Any], label: str) -> None:
    identity = payload.get("identity_sha256")
    _require(_is_sha256(identity), f"{label}.identity_sha256 is invalid")
    unsigned = {key: value for key, value in payload.items() if key != "identity_sha256"}
    _require(identity == canonical_hash(unsigned), f"{label} canonical identity mismatch")


MANIFEST_KEYS = {"schema_version", "protocol", "panel_uid", "items", "identity_sha256"}
ITEM_KEYS = {
    "item_uid",
    "scene_uid",
    "chain",
    "observation_identity_sha256",
    "targets",
    "static_states",
    "context_recipes",
}
TARGET_KEYS = {
    "target_uid",
    "identity_sha256",
    "prompt_template_identity_sha256",
    "semantic_metric_contract",
}
SEMANTIC_METRIC_CONTRACT_KEYS = {
    "metric_id",
    "correct_threshold",
    "context_sensitivity_minimum",
    "monotonic_tolerance",
    "identity_sha256",
}
STATIC_STATE_KEYS = {"oracle", "absent", "corrupted"}


def load_manifest(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    manifest = _require_keys(payload, MANIFEST_KEYS, "manifest")
    _validate_identity(manifest, "manifest")
    _require(manifest["schema_version"] == 1, "unsupported manifest schema_version")
    _require(manifest["protocol"] == contract["contract_id"], "manifest protocol mismatch")
    _require(isinstance(manifest["panel_uid"], str) and manifest["panel_uid"], "panel_uid missing")
    items = manifest["items"]
    _require(isinstance(items, list) and items, "manifest must contain at least one item")
    seen: set[str] = set()
    metric_contracts_by_id: dict[str, str] = {}
    for index, raw in enumerate(items):
        item = _require_keys(raw, ITEM_KEYS, f"manifest.items[{index}]")
        item_uid = item["item_uid"]
        _require(isinstance(item_uid, str) and item_uid and item_uid not in seen, "missing or duplicate item_uid")
        seen.add(item_uid)
        _require(isinstance(item["scene_uid"], str) and item["scene_uid"], f"{item_uid}: scene_uid missing")
        chain = item["chain"]
        _require(chain in EXPECTED_CHAINS, f"{item_uid}: unsupported chain")
        _require(_is_sha256(item["observation_identity_sha256"]), f"{item_uid}: observation identity invalid")
        stages = ("G", "S") if chain == "G_S" else ("G", "S", "A")
        targets = item["targets"]
        _require(isinstance(targets, dict) and set(targets) == set(stages), f"{item_uid}: target stage set drift")
        target_uids: set[str] = set()
        for stage in stages:
            target = _require_keys(targets[stage], TARGET_KEYS, f"{item_uid}.targets.{stage}")
            _require(isinstance(target["target_uid"], str) and target["target_uid"], f"{item_uid}/{stage}: target_uid missing")
            _require(target["target_uid"] not in target_uids, f"{item_uid}: target_uid reused across stages")
            target_uids.add(target["target_uid"])
            for field in ("identity_sha256", "prompt_template_identity_sha256"):
                _require(_is_sha256(target[field]), f"{item_uid}/{stage}: {field} invalid")
            metric = _require_keys(
                target["semantic_metric_contract"],
                SEMANTIC_METRIC_CONTRACT_KEYS,
                f"{item_uid}.targets.{stage}.semantic_metric_contract",
            )
            _validate_identity(metric, f"{item_uid}.targets.{stage}.semantic_metric_contract")
            metric_id = metric["metric_id"]
            threshold = metric["correct_threshold"]
            _require(
                isinstance(metric_id, str) and metric_id,
                f"{item_uid}/{stage}: semantic metric_id missing",
            )
            _require(
                not isinstance(threshold, bool)
                and isinstance(threshold, (int, float))
                and math.isfinite(float(threshold))
                and 0.0 < float(threshold) <= 1.0,
                f"{item_uid}/{stage}: invalid semantic correct_threshold",
            )
            for field in ("context_sensitivity_minimum", "monotonic_tolerance"):
                value = metric[field]
                _require(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= 1.0,
                    f"{item_uid}/{stage}: invalid semantic {field}",
                )
            prior_identity = metric_contracts_by_id.setdefault(
                metric_id, metric["identity_sha256"]
            )
            _require(
                prior_identity == metric["identity_sha256"],
                f"semantic metric contract drift for metric_id {metric_id}",
            )
        static_stages = ("G",) if chain == "G_S" else ("G", "S")
        static_states = item["static_states"]
        _require(
            isinstance(static_states, dict) and set(static_states) == set(static_stages),
            f"{item_uid}: static state stage set drift",
        )
        for stage in static_stages:
            states = static_states[stage]
            _require(
                isinstance(states, dict) and set(states) == STATIC_STATE_KEYS,
                f"{item_uid}/{stage}: static state modes drift",
            )
            for mode, identity in states.items():
                _require(_is_sha256(identity), f"{item_uid}/{stage}/{mode}: state identity invalid")
            _require(
                len(set(states.values())) == len(STATIC_STATE_KEYS),
                f"{item_uid}/{stage}: oracle/absent/corrupted states must be distinct",
            )
        contexts = item["context_recipes"]
        expected = EXPECTED_CHAINS[chain]
        _require(isinstance(contexts, dict) and set(contexts) == set(expected), f"{item_uid}: context recipe set drift")
        for probe_id, identity in contexts.items():
            _require(_is_sha256(identity), f"{item_uid}/{probe_id}: context recipe identity invalid")
    return dict(manifest)


BUNDLE_KEYS = {
    "schema_version",
    "protocol",
    "run_uid",
    "panel_identity_sha256",
    "checkpoint_identity_sha256",
    "decoding_identity_sha256",
    "cost_policy_identity_sha256",
    "num_shards",
    "cells",
    "identity_sha256",
}
CELL_KEYS = {
    "protocol",
    "cell_uid",
    "item_uid",
    "scene_uid",
    "chain",
    "probe_id",
    "target_stage",
    "target_uid",
    "target_identity_sha256",
    "observation_identity_sha256",
    "context_recipe_identity_sha256",
    "context_identity_sha256",
    "context_sources",
    "prompt_template_identity_sha256",
    "checkpoint_identity_sha256",
    "decoding_identity_sha256",
    "shard_index",
    "num_shards",
    "output_status",
    "semantic_score",
    "hit_token_cap",
    "response_identity_sha256",
    "emitted_state_identity_sha256",
    "cost",
}
CONTEXT_SOURCE_KEYS = {"stage", "source_mode", "source_cell_uid", "state_identity_sha256"}


def load_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    bundle = _require_keys(payload, BUNDLE_KEYS, "prediction bundle")
    _validate_identity(bundle, "prediction bundle")
    _require(bundle["schema_version"] == 1, "unsupported prediction bundle schema_version")
    _require(isinstance(bundle["run_uid"], str) and bundle["run_uid"], "run_uid missing")
    for field in (
        "panel_identity_sha256",
        "checkpoint_identity_sha256",
        "decoding_identity_sha256",
        "cost_policy_identity_sha256",
    ):
        _require(_is_sha256(bundle[field]), f"prediction bundle {field} invalid")
    _require(not isinstance(bundle["num_shards"], bool) and isinstance(bundle["num_shards"], int) and bundle["num_shards"] > 0, "num_shards invalid")
    _require(isinstance(bundle["cells"], list), "prediction bundle cells must be a list")
    return dict(bundle)


def _cell_group(chain: str, probe_id: str) -> str:
    return f"{chain}/{probe_id}"


def _parse_cells(
    bundle: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[str]], list[str]]:
    items = {item["item_uid"]: item for item in manifest["items"]}
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    seen_keys: set[tuple[str, str]] = set()
    group_errors: dict[str, list[str]] = defaultdict(list)
    global_errors: list[str] = []
    expected_protocol = contract["result_protocol"]
    if bundle["protocol"] != expected_protocol:
        global_errors.append("protocol_or_binding_mismatch:bundle_protocol")
    if bundle["panel_identity_sha256"] != manifest["identity_sha256"]:
        global_errors.append("protocol_or_binding_mismatch:panel_identity")

    for index, raw in enumerate(bundle["cells"]):
        if not isinstance(raw, dict):
            global_errors.append(f"protocol_or_binding_mismatch:cell_{index}_not_object")
            continue
        item_uid = raw.get("item_uid")
        probe_id = raw.get("probe_id")
        item = items.get(item_uid) if isinstance(item_uid, str) else None
        if (
            item is None
            or not isinstance(probe_id, str)
            or probe_id not in EXPECTED_CHAINS[item["chain"]]
        ):
            global_errors.append("missing_duplicate_or_unexpected_uid:unknown_item_or_probe")
            continue
        chain = item["chain"]
        group = _cell_group(str(chain), str(probe_id))
        if set(raw) != CELL_KEYS:
            group_errors[group].append("protocol_or_binding_mismatch:cell_schema")
            continue
        key = (item_uid, probe_id)
        if key in seen_keys:
            group_errors[group].append("missing_duplicate_or_unexpected_uid:duplicate_cell")
            continue
        seen_keys.add(key)
        target_stage, _, _ = EXPECTED_PROBES[probe_id]
        target = item["targets"][target_stage]
        expected_uid = make_cell_uid(
            expected_protocol,
            bundle["run_uid"],
            bundle["checkpoint_identity_sha256"],
            bundle["decoding_identity_sha256"],
            item_uid,
            probe_id,
        )
        binding_checks = {
            "protocol": expected_protocol,
            "cell_uid": expected_uid,
            "scene_uid": item["scene_uid"],
            "chain": item["chain"],
            "target_stage": target_stage,
            "target_uid": target["target_uid"],
            "target_identity_sha256": target["identity_sha256"],
            "observation_identity_sha256": item["observation_identity_sha256"],
            "context_recipe_identity_sha256": item["context_recipes"][probe_id],
            "prompt_template_identity_sha256": target["prompt_template_identity_sha256"],
            "checkpoint_identity_sha256": bundle["checkpoint_identity_sha256"],
            "decoding_identity_sha256": bundle["decoding_identity_sha256"],
            "num_shards": bundle["num_shards"],
        }
        mismatches = [field for field, expected in binding_checks.items() if raw.get(field) != expected]
        if mismatches:
            group_errors[group].append("protocol_or_binding_mismatch:" + ",".join(sorted(mismatches)))
            continue
        if isinstance(raw["shard_index"], bool) or not isinstance(raw["shard_index"], int) or raw["shard_index"] != expected_shard(expected_uid, bundle["num_shards"]):
            group_errors[group].append("incomplete_or_incorrect_shard_coverage:assignment")
            continue
        if raw["output_status"] not in {"ok", "malformed"}:
            group_errors[group].append("preregistered_output_health_gate_failure:unknown_status")
            continue
        if (
            not isinstance(raw["hit_token_cap"], bool)
            or not _is_sha256(raw["response_identity_sha256"])
            or not _is_sha256(raw["emitted_state_identity_sha256"])
        ):
            group_errors[group].append("preregistered_output_health_gate_failure:metadata")
            continue
        sources = raw["context_sources"]
        if not isinstance(sources, list) or any(
            not isinstance(source, dict) or set(source) != CONTEXT_SOURCE_KEYS
            for source in sources
        ):
            group_errors[group].append("protocol_or_binding_mismatch:context_source_schema")
            continue
        semantic = raw["semantic_score"]
        if raw["output_status"] == "malformed":
            if semantic is not None:
                group_errors[group].append("preregistered_output_health_gate_failure:malformed_score_not_null")
                continue
            score = 0.0
        else:
            if isinstance(semantic, bool) or not isinstance(semantic, (int, float)) or not math.isfinite(float(semantic)) or not 0.0 <= float(semantic) <= 1.0:
                group_errors[group].append("preregistered_output_health_gate_failure:invalid_semantic_score")
                continue
            score = float(semantic)
        cost = raw["cost"]
        if not isinstance(cost, dict) or set(cost) != set(COST_FIELDS):
            group_errors[group].append("protocol_or_binding_mismatch:cost_schema")
            continue
        if any(isinstance(cost[field], bool) or not isinstance(cost[field], int) or cost[field] < 0 for field in COST_FIELDS):
            group_errors[group].append("protocol_or_binding_mismatch:cost_value")
            continue
        if cost["model_queries"] != 1:
            group_errors[group].append("protocol_or_binding_mismatch:model_query_count")
            continue
        cell = dict(raw)
        cell["score"] = score
        candidates[key] = cell

    # Validate dynamic lineage in topological probe order. A downstream cell is
    # admitted only when every predicted context points to the accepted upstream
    # cell from this exact run and carries that cell's emitted state identity.
    parsed: dict[tuple[str, str], dict[str, Any]] = {}
    for item in manifest["items"]:
        item_uid = item["item_uid"]
        chain = item["chain"]
        for probe_id in EXPECTED_CHAINS[chain]:
            key = (item_uid, probe_id)
            cell = candidates.get(key)
            if cell is None:
                continue
            expected_sources: list[dict[str, Any]] = []
            lineage_failed = False
            for stage, source_mode, source_probe in EXPECTED_CONTEXT_SOURCES[probe_id]:
                if source_mode == "predicted":
                    source_key = (item_uid, str(source_probe))
                    source_cell = parsed.get(source_key)
                    if source_cell is None:
                        group_errors[_cell_group(chain, probe_id)].append(
                            "protocol_or_binding_mismatch:missing_upstream_lineage"
                        )
                        lineage_failed = True
                        break
                    source_cell_uid: str | None = source_cell["cell_uid"]
                    state_identity = source_cell["emitted_state_identity_sha256"]
                else:
                    source_cell_uid = None
                    state_identity = item["static_states"][stage][source_mode]
                expected_sources.append(
                    {
                        "stage": stage,
                        "source_mode": source_mode,
                        "source_cell_uid": source_cell_uid,
                        "state_identity_sha256": state_identity,
                    }
                )
            if lineage_failed:
                continue
            if cell["context_sources"] != expected_sources:
                group_errors[_cell_group(chain, probe_id)].append(
                    "protocol_or_binding_mismatch:context_lineage"
                )
                continue
            if cell["context_identity_sha256"] != canonical_hash(expected_sources):
                group_errors[_cell_group(chain, probe_id)].append(
                    "protocol_or_binding_mismatch:context_identity"
                )
                continue
            parsed[key] = cell
    return parsed, dict(group_errors), global_errors


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def scene_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    value_fn: Callable[[Mapping[str, Any]], tuple[float, ...]],
    statistic: Callable[[Sequence[tuple[float, ...]]], float | None],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    by_scene: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_uid"])].append(value_fn(row))
    scenes = sorted(by_scene)
    _require(scenes, "bootstrap requires at least one scene")
    rng = random.Random(int(config["seed"]))
    estimates: list[float] = []
    for _ in range(int(config["resamples"])):
        sampled: list[tuple[float, ...]] = []
        for _ in scenes:
            sampled.extend(by_scene[rng.choice(scenes)])
        estimate = statistic(sampled)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(float(estimate))
    confidence = float(config["confidence"])
    alpha = (1.0 - confidence) / 2.0
    estimates.sort()
    return {
        "method": config["method"],
        "paired_on": "item_uid",
        "unit": "scene_uid",
        "confidence": confidence,
        "seed": int(config["seed"]),
        "requested_resamples": int(config["resamples"]),
        "valid_resamples": len(estimates),
        "n_scenes": len(scenes),
        "ci": (
            [round(_percentile(estimates, alpha), 6), round(_percentile(estimates, 1.0 - alpha), 6)]
            if estimates
            else None
        ),
    }


def _mean_stat(values: Sequence[tuple[float, ...]]) -> float | None:
    return sum(value[0] for value in values) / len(values) if values else None


def _recovery_stat(values: Sequence[tuple[float, ...]]) -> float | None:
    denominator = sum(value[1] for value in values)
    return sum(value[0] for value in values) / denominator if denominator > 0.0 else None


def _probe_summary(
    chain: str,
    probe_id: str,
    item_rows: Sequence[Mapping[str, Any]],
    parsed: Mapping[tuple[str, str], Mapping[str, Any]],
    group_errors: Mapping[str, Sequence[str]],
    global_errors: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    group = _cell_group(chain, probe_id)
    expected_ids = [str(item["item_uid"]) for item in item_rows]
    cells = [parsed[(item_uid, probe_id)] for item_uid in expected_ids if (item_uid, probe_id) in parsed]
    missing = len(expected_ids) - len(cells)
    errors = list(global_errors) + list(group_errors.get(group, ()))
    if missing:
        errors.append(f"missing_duplicate_or_unexpected_uid:missing={missing}")
    malformed = sum(cell["output_status"] == "malformed" for cell in cells)
    truncated = sum(bool(cell["hit_token_cap"]) for cell in cells)
    malformed_rate = malformed / len(expected_ids) if expected_ids else 0.0
    truncation_rate = truncated / len(expected_ids) if expected_ids else 0.0
    health_failure = (
        malformed_rate > float(contract["cell_admission"]["max_malformed_rate"])
        or truncation_rate > float(contract["cell_admission"]["max_truncation_rate"])
    )
    if health_failure:
        errors.append("preregistered_output_health_gate_failure:aggregate_rate")
    allowed_categories = set(contract["cell_admission"]["withhold_only_for"])
    for error in errors:
        _require(
            error.split(":", 1)[0] in allowed_categories,
            f"unregistered cell-withholding category: {error}",
        )
    structural = any(not error.startswith("preregistered_output_health_gate_failure") for error in errors)
    if structural:
        admission = "withheld"
    elif errors:
        admission = "diagnostic_only"
    else:
        admission = "admitted"
    diagnostic_score = _mean([float(cell["score"]) for cell in cells]) if not structural and len(cells) == len(expected_ids) else None
    return {
        "chain": chain,
        "probe_id": probe_id,
        "target_stage": EXPECTED_PROBES[probe_id][0],
        "intervention_stage": EXPECTED_PROBES[probe_id][1],
        "context_mode": EXPECTED_PROBES[probe_id][2],
        "expected_n": len(expected_ids),
        "observed_valid_n": len(cells),
        "malformed_n": malformed,
        "malformed_rate": round(malformed_rate, 6),
        "truncated_n": truncated,
        "truncation_rate": round(truncation_rate, 6),
        "admission": admission,
        "score": diagnostic_score if admission == "admitted" else None,
        "diagnostic_score": diagnostic_score,
        "withheld_reasons": sorted(set(errors)),
    }


def _rows_for(
    item_rows: Sequence[Mapping[str, Any]],
    parsed: Mapping[tuple[str, str], Mapping[str, Any]],
    probes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in item_rows:
        row: dict[str, Any] = {"item_uid": item["item_uid"], "scene_uid": item["scene_uid"]}
        for probe in probes:
            row[probe] = float(parsed[(item["item_uid"], probe)]["score"])
        rows.append(row)
    return rows


def _predicted_lineage_closure(probes: set[str]) -> set[str]:
    """Include every dynamically generated source required by the probe set."""
    closure = set(probes)
    pending = list(probes)
    while pending:
        probe_id = pending.pop()
        for _, source_mode, source_probe in EXPECTED_CONTEXT_SOURCES[probe_id]:
            if source_mode != "predicted" or source_probe is None or source_probe in closure:
                continue
            closure.add(source_probe)
            pending.append(source_probe)
    return closure


def _metric_block(
    chain: str,
    target_stage: str,
    items: Sequence[Mapping[str, Any]],
    parsed: Mapping[tuple[str, str], Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if target_stage == "S":
        predicted, oracle = "S.predicted_G", "S.oracle_G"
        repair_probes = {"G": oracle}
        controls = {"G": ("S.absent_G", "S.corrupted_G")}
        oracle_ceiling_probe = None
    else:
        predicted, oracle = "A.predicted_GS", "A.oracle_GS"
        repair_probes = {"G": "A.repair_G"}
        oracle_ceiling_probe = oracle
        controls = {
            "G": ("A.absent_G", "A.corrupted_G"),
            "S": ("A.absent_S", "A.corrupted_S"),
        }
    direct_needed = {
        predicted,
        oracle,
        *repair_probes.values(),
        *(probe for pair in controls.values() for probe in pair),
    }
    needed = _predicted_lineage_closure(direct_needed)
    failed = [probe for probe in sorted(needed) if summaries[probe]["admission"] != "admitted"]
    if failed:
        return {
            "chain": chain,
            "target_stage": target_stage,
            "status": "WITHHELD",
            "required_admitted_probes": sorted(needed),
            "failed_probes": failed,
            "dependency_gap": None,
            "oracle_headroom": None,
            "repair_effects": None,
            "headroom_recovery": None,
            "oracle_ceiling": None,
            "context_sensitivity": None,
            "semantic_conflicts": None,
        }
    semantic_conflicts = [
        {
            "item_uid": item["item_uid"],
            "conflicts": conflicts,
        }
        for item in items
        if (
            conflicts := _response_score_conflicts(
                str(item["item_uid"]), sorted(needed), parsed
            )
        )
    ]
    if semantic_conflicts:
        return {
            "chain": chain,
            "target_stage": target_stage,
            "status": "WITHHELD",
            "required_admitted_probes": sorted(needed),
            "failed_probes": [],
            "dependency_gap": None,
            "oracle_headroom": None,
            "repair_effects": None,
            "headroom_recovery": None,
            "oracle_ceiling": None,
            "context_sensitivity": None,
            "semantic_conflicts": semantic_conflicts,
        }
    rows = _rows_for(items, parsed, sorted(needed))
    items_by_uid = {str(item["item_uid"]): item for item in items}
    gap_values = [row[oracle] - row[predicted] for row in rows]
    positive_headroom = [max(value, 0.0) for value in gap_values]
    dependency = {
        "definition": "oracle_minus_predicted",
        "estimate": _mean(gap_values),
        "paired_scene_bootstrap": scene_bootstrap_ci(
            rows,
            lambda row: (row[oracle] - row[predicted],),
            _mean_stat,
            contract["bootstrap"],
        ),
    }
    headroom = {
        "definition": "max(oracle_minus_predicted,0) per item",
        "positive_items": sum(value > 0.0 for value in positive_headroom),
        "total": round(sum(positive_headroom), 6),
        "mean": _mean(positive_headroom),
        "paired_scene_bootstrap": scene_bootstrap_ci(
            rows,
            lambda row: (max(row[oracle] - row[predicted], 0.0),),
            _mean_stat,
            contract["bootstrap"],
        ),
    }
    repair_effects: dict[str, Any] = {}
    recoveries: dict[str, Any] = {}
    for label, repair in repair_probes.items():
        effects = [row[repair] - row[predicted] for row in rows]
        bounded = [
            min(max(row[repair] - row[predicted], 0.0), max(row[oracle] - row[predicted], 0.0))
            for row in rows
        ]
        repair_effects[label] = {
            "repair_probe": repair,
            "signed_mean": _mean(effects),
            "paired_scene_bootstrap": scene_bootstrap_ci(
                rows,
                lambda row, repair=repair: (row[repair] - row[predicted],),
                _mean_stat,
                contract["bootstrap"],
            ),
        }
        recovery = _ratio(sum(bounded), sum(positive_headroom))
        recoveries[label] = {
            "definition": "sum(clamp(repair-predicted,0,positive_oracle_headroom))/sum(positive_oracle_headroom)",
            "eligible_positive_headroom_items": sum(value > 0.0 for value in positive_headroom),
            "estimate": recovery,
            "status": "defined" if recovery is not None else "undefined_no_positive_oracle_headroom",
            "paired_scene_bootstrap": scene_bootstrap_ci(
                rows,
                lambda row, repair=repair: (
                    min(max(row[repair] - row[predicted], 0.0), max(row[oracle] - row[predicted], 0.0)),
                    max(row[oracle] - row[predicted], 0.0),
                ),
                _recovery_stat,
                contract["bootstrap"],
            ),
        }
    oracle_ceiling = None
    if oracle_ceiling_probe is not None:
        recovery = 1.0 if sum(positive_headroom) > 0.0 else None
        oracle_ceiling = {
            "reference_probe": oracle_ceiling_probe,
            "role": "algebraic_oracle_ceiling_not_empirical_ablation",
            "signed_effect_equals_dependency_gap": dependency["estimate"],
            "headroom_recovery": recovery,
            "status": (
                "defined_identity" if recovery is not None else "undefined_no_positive_oracle_headroom"
            ),
        }
    sensitivity: dict[str, Any] = {}
    for stage, (absent, corrupted) in controls.items():
        repaired = repair_probes["G"] if stage == "G" else oracle
        values = [row[repaired] - 0.5 * (row[absent] + row[corrupted]) for row in rows]
        absent_effects = [row[repaired] - row[absent] for row in rows]
        corrupted_effects = [row[repaired] - row[corrupted] for row in rows]
        corrupted_estimate = _mean(corrupted_effects)
        sensitivity_thresholds = [
            float(
                items_by_uid[str(row["item_uid"])]["targets"][target_stage][
                    "semantic_metric_contract"
                ]["context_sensitivity_minimum"]
            )
            for row in rows
        ]
        corrupted_margins = [
            effect - threshold
            for effect, threshold in zip(corrupted_effects, sensitivity_thresholds)
        ]
        sensitivity[stage] = {
            "definition": "repaired_minus_mean_absent_corrupted",
            "estimate": _mean(values),
            "absent_effect": {
                "estimate": _mean(absent_effects),
                "paired_scene_bootstrap": scene_bootstrap_ci(
                    rows,
                    lambda row, repaired=repaired, absent=absent: (row[repaired] - row[absent],),
                    _mean_stat,
                    contract["bootstrap"],
                ),
            },
            "corrupted_effect": {
                "estimate": corrupted_estimate,
                "threshold_source": "manifest_target_semantic_metric_contract",
                "mean_minimum_effect": _mean(sensitivity_thresholds),
                "mean_margin_over_minimum": _mean(corrupted_margins),
                "directionally_consistent": _mean(corrupted_margins) > 0.0,
                "paired_scene_bootstrap": scene_bootstrap_ci(
                    rows,
                    lambda row, repaired=repaired, corrupted=corrupted: (
                        row[repaired] - row[corrupted],
                    ),
                    _mean_stat,
                    contract["bootstrap"],
                ),
            },
            "paired_scene_bootstrap": scene_bootstrap_ci(
                rows,
                lambda row, repaired=repaired, absent=absent, corrupted=corrupted: (
                    row[repaired] - 0.5 * (row[absent] + row[corrupted]),
                ),
                _mean_stat,
                contract["bootstrap"],
            ),
        }
    return {
        "chain": chain,
        "target_stage": target_stage,
        "status": "ADMITTED",
        "n_items": len(rows),
        "n_scenes": len({row["scene_uid"] for row in rows}),
        "predicted_probe": predicted,
        "oracle_probe": oracle,
        "dependency_gap": dependency,
        "oracle_headroom": headroom,
        "repair_effects": repair_effects,
        "headroom_recovery": recoveries,
        "oracle_ceiling": oracle_ceiling,
        "context_sensitivity": sensitivity,
        "semantic_conflicts": [],
    }


def _transition(before: bool, after: bool) -> str:
    return ("correct" if before else "wrong") + "_to_" + ("correct" if after else "wrong")


def _complete_transition_counts(
    chain: str, counts: Mapping[str, Counter[str]]
) -> dict[str, dict[str, int]]:
    return {
        edge: {label: int(counts[edge].get(label, 0)) for label in TRANSITION_LABELS}
        for edge in TRANSITION_EDGES[chain]
    }


def _transition_matrix(
    complete_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        edge: {
            before: {
                after: int(edge_counts[f"{before}_to_{after}"])
                for after in TRANSITION_STATES
            }
            for before in TRANSITION_STATES
        }
        for edge, edge_counts in complete_counts.items()
    }


def _semantic_criterion(
    item: Mapping[str, Any], probe_id: str
) -> dict[str, Any]:
    target_stage = EXPECTED_PROBES[probe_id][0]
    metric = item["targets"][target_stage]["semantic_metric_contract"]
    return {
        "metric_id": str(metric["metric_id"]),
        "correct_threshold": float(metric["correct_threshold"]),
        "context_sensitivity_minimum": float(
            metric["context_sensitivity_minimum"]
        ),
        "monotonic_tolerance": float(metric["monotonic_tolerance"]),
        "identity_sha256": str(metric["identity_sha256"]),
    }


def _response_score_conflicts(
    item_uid: str,
    probes: Sequence[str],
    parsed: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_target: dict[str, list[str]] = defaultdict(list)
    for probe_id in probes:
        by_target[EXPECTED_PROBES[probe_id][0]].append(probe_id)

    conflicts: list[dict[str, Any]] = []
    for target_stage, target_probes in sorted(by_target.items()):
        scores = {
            probe_id: float(parsed[(item_uid, probe_id)]["score"])
            for probe_id in target_probes
        }
        identities = {
            probe_id: str(
                parsed[(item_uid, probe_id)]["response_identity_sha256"]
            )
            for probe_id in target_probes
        }
        if response_scores_consistent(scores, identities):
            continue
        seen: dict[str, tuple[str, float]] = {}
        for probe_id in target_probes:
            identity = identities[probe_id]
            score = scores[probe_id]
            previous = seen.setdefault(identity, (probe_id, score))
            if previous[1] != score:
                conflicts.append(
                    {
                        "target_stage": target_stage,
                        "response_identity_sha256": identity,
                        "probe_ids": [previous[0], probe_id],
                        "semantic_scores": [previous[1], score],
                    }
                )
    return conflicts


def _item_probe_health_failures(
    item_uid: str,
    probes: Sequence[str],
    parsed: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for probe_id in probes:
        cell = parsed[(item_uid, probe_id)]
        if cell["output_status"] == "malformed":
            failures.append({"probe_id": probe_id, "reason": "malformed"})
        if cell["hit_token_cap"]:
            failures.append({"probe_id": probe_id, "reason": "token_cap"})
    return failures


def _first_fault_block(
    chain: str,
    items: Sequence[Mapping[str, Any]],
    parsed: Mapping[tuple[str, str], Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    required = set(EXPECTED_CHAINS[chain])
    failed = [probe for probe in sorted(required) if summaries[probe]["admission"] != "admitted"]
    if failed:
        return {
            "chain": chain,
            "status": "WITHHELD",
            "failed_probes": failed,
            "per_item": [],
            "counts": {},
            "routing": None,
            "transition_counts": None,
            "transition_matrix": None,
        }
    ordered_required = sorted(required)
    rows = _rows_for(items, parsed, ordered_required)
    items_by_uid = {str(item["item_uid"]): item for item in items}
    decisions: list[dict[str, Any]] = []
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        item = items_by_uid[str(row["item_uid"])]
        stages = ("G", "S") if chain == "G_S" else ("G", "S", "A")
        edges = TRANSITION_EDGES[chain]
        criteria = {
            probe_id: _semantic_criterion(item, probe_id)
            for probe_id in ordered_required
        }
        semantic_criteria = {
            probe_id: dict(criterion)
            for probe_id, criterion in criteria.items()
        }
        health_failures = _item_probe_health_failures(
            str(row["item_uid"]), ordered_required, parsed
        )
        if health_failures:
            decisions.append(
                {
                    "item_uid": row["item_uid"],
                    "scene_uid": row["scene_uid"],
                    "state": "Inconclusive",
                    "fault_stage": None,
                    "reason": "invalid_or_capped_probe",
                    "routing_eligible": False,
                    "routing_stage": None,
                    "transition_evaluable": False,
                    "probe_health": {
                        "valid_uncapped": False,
                        "failures": health_failures,
                    },
                    "semantic_consistency": {"valid": None, "conflicts": []},
                    "semantic_criteria": semantic_criteria,
                    "local_correct": {stage: None for stage in stages},
                    "state_use_confirmed": {
                        stage: None for stage in stages[:-1]
                    },
                    "transitions": {edge: None for edge in edges},
                }
            )
            continue

        response_conflicts = _response_score_conflicts(
            str(row["item_uid"]), ordered_required, parsed
        )
        if response_conflicts:
            decisions.append(
                {
                    "item_uid": row["item_uid"],
                    "scene_uid": row["scene_uid"],
                    "state": RouteKind.INCONCLUSIVE.value,
                    "fault_stage": None,
                    "reason": "response_score_inconsistency",
                    "routing_eligible": False,
                    "routing_stage": None,
                    "transition_evaluable": False,
                    "probe_health": {"valid_uncapped": True, "failures": []},
                    "semantic_consistency": {
                        "valid": False,
                        "conflicts": response_conflicts,
                    },
                    "semantic_criteria": semantic_criteria,
                    "local_correct": {stage: None for stage in stages},
                    "state_use_confirmed": {
                        stage: None for stage in stages[:-1]
                    },
                    "transitions": {edge: None for edge in edges},
                }
            )
            continue

        def correct(probe_id: str) -> bool:
            return row[probe_id] >= criteria[probe_id]["correct_threshold"]

        cells = {
            probe_id: parsed[(str(row["item_uid"]), probe_id)]
            for probe_id in ordered_required
        }

        g_ok = correct("G.local")
        s_ok = correct("S.oracle_G")
        if chain == "G_S":
            oracle_ok = s_ok
            target_criterion = criteria["S.predicted_G"]
            use_g = confirm_state_use(
                repaired_target_score=row["S.oracle_G"],
                absent_target_score=row["S.absent_G"],
                corrupted_target_score=row["S.corrupted_G"],
                minimum_effect=target_criterion["context_sensitivity_minimum"],
                previous_response_identity=cells["S.predicted_G"][
                    "response_identity_sha256"
                ],
                repaired_response_identity=cells["S.oracle_G"][
                    "response_identity_sha256"
                ],
            )
            route = route_first_fault(
                stage_order=stages,
                local_correct={"G": g_ok, "S": oracle_ok},
                predicted_target_score=row["S.predicted_G"],
                repair_target_scores={"G": row["S.oracle_G"]},
                predicted_target_response_identity=cells["S.predicted_G"][
                    "response_identity_sha256"
                ],
                repair_target_response_identities={
                    "G": cells["S.oracle_G"]["response_identity_sha256"]
                },
                state_use_confirmed={"G": use_g},
                probes_valid=True,
                hit_token_cap=False,
                correct_threshold=target_criterion["correct_threshold"],
                monotonic_tolerance=target_criterion["monotonic_tolerance"],
            )
            pred_ok = correct("S.predicted_G")
            trans_g = _transition(pred_ok, oracle_ok)
            transition_counts["G_repair"][trans_g] += 1
            transitions = {"G_repair": trans_g}
            state_use = {"G": use_g}
        else:
            oracle_ok = correct("A.oracle_GS")
            target_criterion = criteria["A.predicted_GS"]
            use_g = confirm_state_use(
                repaired_target_score=row["A.repair_G"],
                absent_target_score=row["A.absent_G"],
                corrupted_target_score=row["A.corrupted_G"],
                minimum_effect=target_criterion["context_sensitivity_minimum"],
                previous_response_identity=cells["A.predicted_GS"][
                    "response_identity_sha256"
                ],
                repaired_response_identity=cells["A.repair_G"][
                    "response_identity_sha256"
                ],
            )
            use_s = confirm_state_use(
                repaired_target_score=row["A.oracle_GS"],
                absent_target_score=row["A.absent_S"],
                corrupted_target_score=row["A.corrupted_S"],
                minimum_effect=target_criterion["context_sensitivity_minimum"],
                previous_response_identity=cells["A.repair_G"][
                    "response_identity_sha256"
                ],
                repaired_response_identity=cells["A.oracle_GS"][
                    "response_identity_sha256"
                ],
            )
            route = route_first_fault(
                stage_order=stages,
                local_correct={"G": g_ok, "S": s_ok, "A": oracle_ok},
                predicted_target_score=row["A.predicted_GS"],
                repair_target_scores={
                    "G": row["A.repair_G"],
                    "S": row["A.oracle_GS"],
                },
                predicted_target_response_identity=cells["A.predicted_GS"][
                    "response_identity_sha256"
                ],
                repair_target_response_identities={
                    "G": cells["A.repair_G"]["response_identity_sha256"],
                    "S": cells["A.oracle_GS"]["response_identity_sha256"],
                },
                state_use_confirmed={"G": use_g, "S": use_s},
                probes_valid=True,
                hit_token_cap=False,
                correct_threshold=target_criterion["correct_threshold"],
                monotonic_tolerance=target_criterion["monotonic_tolerance"],
            )
            pred_ok = correct("A.predicted_GS")
            repair_g_ok = correct("A.repair_G")
            trans_g = _transition(pred_ok, repair_g_ok)
            trans_s = _transition(repair_g_ok, oracle_ok)
            transition_counts["G_repair"][trans_g] += 1
            transition_counts["S_repair"][trans_s] += 1
            transitions = {"G_repair": trans_g, "S_repair": trans_s}
            state_use = {"G": use_g, "S": use_s}
        routing_eligible = route.eligible
        decisions.append(
            {
                "item_uid": row["item_uid"],
                "scene_uid": row["scene_uid"],
                "state": route.kind.value,
                "fault_stage": route.stage,
                "reason": route.reason,
                "routing_eligible": routing_eligible,
                "routing_stage": route.stage if routing_eligible else None,
                "transition_evaluable": True,
                "probe_health": {"valid_uncapped": True, "failures": []},
                "semantic_consistency": {"valid": True, "conflicts": []},
                "semantic_criteria": semantic_criteria,
                "local_correct": {"G": g_ok, "S": s_ok, **({"A": oracle_ok} if chain == "G_S_A" else {})},
                "state_use_confirmed": state_use,
                "transitions": transitions,
            }
        )
    counts = Counter(row["state"] for row in decisions)
    routed = [row for row in decisions if row["routing_eligible"]]
    routed_by_stage = Counter(str(row["routing_stage"]) for row in routed)
    complete_transitions = _complete_transition_counts(chain, transition_counts)
    return {
        "chain": chain,
        "status": "ADMITTED",
        "n_items": len(decisions),
        "counts": {key: counts.get(key, 0) for key in contract["first_fault_states"]},
        "routing": {
            "eligible_n": len(routed),
            "ineligible_n": len(decisions) - len(routed),
            "by_stage": {
                stage: routed_by_stage.get(stage, 0)
                for stage in ("G", "S", "A")
            },
        },
        "transition_evaluable_n": sum(
            row["transition_evaluable"] for row in decisions
        ),
        "transition_counts": complete_transitions,
        "transition_matrix": _transition_matrix(complete_transitions),
        "per_item": decisions,
    }


def _cost_block(
    parsed: Mapping[tuple[str, str], Mapping[str, Any]],
    expected_cells: int,
    expected_items: int,
    expected_items_by_chain: Mapping[str, int],
    observed_bundle_cells: int,
    structurally_comparable: bool,
    cost_policy_identity_sha256: str,
) -> dict[str, Any]:
    totals = {field: 0 for field in COST_FIELDS}
    by_probe: dict[str, dict[str, int]] = defaultdict(lambda: {field: 0 for field in COST_FIELDS})
    by_chain: dict[str, dict[str, int]] = defaultdict(lambda: {field: 0 for field in COST_FIELDS})
    by_chain_probe: dict[str, dict[str, int]] = defaultdict(
        lambda: {field: 0 for field in COST_FIELDS}
    )
    for cell in parsed.values():
        for field in COST_FIELDS:
            value = int(cell["cost"][field])
            totals[field] += value
            by_probe[cell["probe_id"]][field] += value
            by_chain[cell["chain"]][field] += value
            by_chain_probe[_cell_group(cell["chain"], cell["probe_id"])][field] += value
    observed = len(parsed)
    budget_comparable = (
        structurally_comparable
        and observed == expected_cells
        and observed_bundle_cells == expected_cells
    )
    model_tokens = totals["model_input_tokens"] + totals["model_output_tokens"]
    teacher_tokens = totals["teacher_input_tokens"] + totals["teacher_output_tokens"]
    total_cost_microusd = totals["model_cost_microusd"] + totals["teacher_cost_microusd"]
    return {
        "status": "COMPLETE" if budget_comparable else "PARTIAL_NOT_BUDGET_COMPARABLE",
        "totals_scope": "structurally_valid_bound_cells_only",
        "cost_policy_identity_sha256": cost_policy_identity_sha256,
        "expected_cells": expected_cells,
        "observed_bundle_cells": observed_bundle_cells,
        "observed_structurally_valid_cells": observed,
        "coverage": _ratio(observed, expected_cells),
        "totals": totals,
        "derived_totals": {
            "model_tokens": model_tokens,
            "teacher_tokens": teacher_tokens,
            "total_queries": totals["model_queries"] + totals["teacher_queries"],
            "total_tokens": model_tokens + teacher_tokens,
            "total_cost_microusd": total_cost_microusd,
            "total_cost_usd": round(total_cost_microusd / 1_000_000.0, 6),
        },
        "per_item": {field: _ratio(value, expected_items) for field, value in totals.items()},
        "per_item_by_chain": {
            chain: {
                field: _ratio(value, expected_items_by_chain[chain])
                for field, value in values.items()
            }
            for chain, values in sorted(by_chain.items())
        },
        "by_chain": {chain: values for chain, values in sorted(by_chain.items())},
        "by_probe": {probe: values for probe, values in sorted(by_probe.items())},
        "by_chain_probe": {
            group: values for group, values in sorted(by_chain_probe.items())
        },
    }


def score_bundle(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parsed, group_errors, global_errors = _parse_cells(bundle, manifest, contract)
    items_by_chain = {
        chain: [item for item in manifest["items"] if item["chain"] == chain]
        for chain in EXPECTED_CHAINS
    }
    summaries_by_chain: dict[str, dict[str, Any]] = {}
    evaluations: list[dict[str, Any]] = []
    first_fault: list[dict[str, Any]] = []
    expected_cells = 0
    for chain, items in items_by_chain.items():
        if not items:
            continue
        expected_cells += len(items) * len(EXPECTED_CHAINS[chain])
        summaries = {
            probe: _probe_summary(chain, probe, items, parsed, group_errors, global_errors, contract)
            for probe in EXPECTED_CHAINS[chain]
        }
        summaries_by_chain[chain] = summaries
        evaluations.append(_metric_block(chain, "S", items, parsed, summaries, contract))
        if chain == "G_S_A":
            evaluations.append(_metric_block(chain, "A", items, parsed, summaries, contract))
        first_fault.append(_first_fault_block(chain, items, parsed, summaries, contract))
    admitted = all(
        summary["admission"] == "admitted"
        for summaries in summaries_by_chain.values()
        for summary in summaries.values()
    ) and all(evaluation["status"] == "ADMITTED" for evaluation in evaluations)
    all_errors = list(global_errors) + [
        error
        for errors in group_errors.values()
        for error in errors
    ]
    bindings_verified = not any(
        error.startswith("protocol_or_binding_mismatch") for error in all_errors
    )
    coverage_verified = not any(
        error.startswith("missing_duplicate_or_unexpected_uid")
        or error.startswith("incomplete_or_incorrect_shard_coverage")
        for error in all_errors
    )
    structurally_comparable = not any(
        not error.startswith("preregistered_output_health_gate_failure")
        for error in all_errors
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": contract["result_protocol"],
        "status": "PAIRED_SCORE_COMPLETE" if admitted else "WITHHELD",
        "prediction_generation_performed_by_scorer": False,
        "note": "The scorer aggregates supplied stage-native scores; evidence tier is determined by the bound panel and run, not by this program.",
        "bindings": dict(bindings or {}),
        "run": {
            "run_uid": bundle["run_uid"],
            "panel_uid": manifest["panel_uid"],
            "panel_identity_sha256": manifest["identity_sha256"],
            "checkpoint_identity_sha256": bundle["checkpoint_identity_sha256"],
            "decoding_identity_sha256": bundle["decoding_identity_sha256"],
            "cost_policy_identity_sha256": bundle["cost_policy_identity_sha256"],
            "num_shards": bundle["num_shards"],
        },
        "comparability": {
            "same_item_observation_checkpoint_decoding_and_downstream_target": (
                bindings_verified and coverage_verified
            ),
            "upstream_output_lineage_verified": bindings_verified,
            "uid_and_shard_coverage_verified": coverage_verified,
            "manifest_bound_context_recipe_and_static_states": True,
            "response_score_consistency_verified": all(
                evaluation["status"] == "ADMITTED"
                and not evaluation.get("semantic_conflicts")
                for evaluation in evaluations
            ),
            "type_validity_and_answer_freedom_verified_by_scorer": False,
            "global_errors": sorted(set(global_errors)),
        },
        "condition_cells": summaries_by_chain,
        "evaluations": evaluations,
        "first_causal_fault": first_fault,
        "query_and_token_cost": _cost_block(
            parsed,
            expected_cells,
            len(manifest["items"]),
            {chain: len(items) for chain, items in items_by_chain.items() if items},
            len(bundle["cells"]),
            structurally_comparable,
            bundle["cost_policy_identity_sha256"],
        ),
    }
    report["identity_sha256"] = canonical_hash(report)
    return report


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only and args.out is not None:
        parser.error("--check-only and --out are mutually exclusive")
    if not args.check_only and args.out is None:
        parser.error("--out is required unless --check-only is used")
    contract = load_contract(args.contract)
    manifest = load_manifest(args.manifest, contract)
    bundle = load_bundle(args.predictions)
    bindings = {
        "contract": {**file_binding(args.contract), "identity_sha256": canonical_hash(contract)},
        "manifest": file_binding(args.manifest),
        "predictions": file_binding(args.predictions),
        "scorer": file_binding(Path(__file__)),
    }
    report = score_bundle(contract, manifest, bundle, bindings=bindings)
    if args.check_only:
        print(json.dumps({"status": report["status"], "identity_sha256": report["identity_sha256"]}, sort_keys=True))
    else:
        _atomic_json(args.out, report)
        print(json.dumps({"out": str(args.out.expanduser().resolve()), "status": report["status"], "identity_sha256": report["identity_sha256"]}, sort_keys=True))
    return 0 if report["status"] == "PAIRED_SCORE_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
