from __future__ import annotations

import copy
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import score_firstfault_interventions as ff  # noqa: E402
from firstfault.routing import route_first_fault  # noqa: E402


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _signed(payload: dict) -> dict:
    payload = copy.deepcopy(payload)
    payload["identity_sha256"] = ff.canonical_hash(payload)
    return payload


def _test_contract() -> dict:
    contract = ff.load_contract()
    contract = copy.deepcopy(contract)
    contract["bootstrap"]["resamples"] = 200
    return contract


def _semantic_metric_contract(stage: str) -> dict:
    return _signed(
        {
            "metric_id": "synthetic-g-iou-v1" if stage == "G" else f"synthetic-{stage.lower()}-exact-v1",
            "correct_threshold": 0.5 if stage == "G" else 1.0,
            "context_sensitivity_minimum": 0.0,
            "monotonic_tolerance": 0.0,
        }
    )


def _manifest(chains: list[str]) -> dict:
    items = []
    for index, chain in enumerate(chains):
        item_uid = f"item-{index:03d}"
        stages = ("G", "S") if chain == "G_S" else ("G", "S", "A")
        targets = {
            stage: {
                "target_uid": f"{item_uid}-{stage}",
                "identity_sha256": _digest(f"target/{item_uid}/{stage}"),
                "prompt_template_identity_sha256": _digest(f"prompt/{stage}"),
                "semantic_metric_contract": _semantic_metric_contract(stage),
            }
            for stage in stages
        }
        contexts = {
            probe: _digest(f"context/{item_uid}/{probe}")
            for probe in ff.EXPECTED_CHAINS[chain]
        }
        static_stages = ("G",) if chain == "G_S" else ("G", "S")
        static_states = {
            stage: {
                mode: _digest(f"static/{item_uid}/{stage}/{mode}")
                for mode in ("oracle", "absent", "corrupted")
            }
            for stage in static_stages
        }
        items.append(
            {
                "item_uid": item_uid,
                "scene_uid": f"scene-{index // 2:03d}",
                "chain": chain,
                "observation_identity_sha256": _digest(f"observation/{item_uid}"),
                "targets": targets,
                "static_states": static_states,
                "context_recipes": contexts,
            }
        )
    return _signed(
        {
            "schema_version": 1,
            "protocol": "firstfault-intervention-v1",
            "panel_uid": "synthetic-panel",
            "items": items,
        }
    )


def _default_scores(chain: str) -> dict[str, float]:
    scores = {
        "G.local": 1.0,
        "S.predicted_G": 0.0,
        "S.oracle_G": 1.0,
        "S.absent_G": 0.0,
        "S.corrupted_G": 0.0,
    }
    if chain == "G_S_A":
        scores.update(
            {
                "A.predicted_GS": 0.0,
                "A.repair_G": 1.0,
                "A.oracle_GS": 1.0,
                "A.absent_G": 0.0,
                "A.corrupted_G": 0.0,
                "A.absent_S": 0.0,
                "A.corrupted_S": 0.0,
            }
        )
    return scores


def _bundle(
    manifest: dict,
    overrides: dict[tuple[str, str], float | None] | None = None,
    malformed: set[tuple[str, str]] | None = None,
    capped: set[tuple[str, str]] | None = None,
) -> dict:
    overrides = overrides or {}
    malformed = malformed or set()
    capped = capped or set()
    protocol = "firstfault-intervention-result-v1"
    run_uid = "synthetic-run"
    checkpoint = _digest("checkpoint")
    decoding = _digest("decoding")
    num_shards = 3
    cells = []
    for item in manifest["items"]:
        scores = _default_scores(item["chain"])
        emitted_by_probe: dict[str, tuple[str, str]] = {}
        for probe_id in ff.EXPECTED_CHAINS[item["chain"]]:
            key = (item["item_uid"], probe_id)
            target_stage = ff.EXPECTED_PROBES[probe_id][0]
            target = item["targets"][target_stage]
            cell_uid = ff.make_cell_uid(
                protocol,
                run_uid,
                checkpoint,
                decoding,
                item["item_uid"],
                probe_id,
            )
            is_malformed = key in malformed
            score = overrides.get(key, scores[probe_id])
            context_sources = []
            for stage, source_mode, source_probe in ff.EXPECTED_CONTEXT_SOURCES[probe_id]:
                if source_mode == "predicted":
                    source_cell_uid, state_identity = emitted_by_probe[str(source_probe)]
                else:
                    source_cell_uid = None
                    state_identity = item["static_states"][stage][source_mode]
                context_sources.append(
                    {
                        "stage": stage,
                        "source_mode": source_mode,
                        "source_cell_uid": source_cell_uid,
                        "state_identity_sha256": state_identity,
                    }
                )
            emitted_state = _digest(f"emitted/{item['item_uid']}/{probe_id}")
            cells.append(
                {
                    "protocol": protocol,
                    "cell_uid": cell_uid,
                    "item_uid": item["item_uid"],
                    "scene_uid": item["scene_uid"],
                    "chain": item["chain"],
                    "probe_id": probe_id,
                    "target_stage": target_stage,
                    "target_uid": target["target_uid"],
                    "target_identity_sha256": target["identity_sha256"],
                    "observation_identity_sha256": item["observation_identity_sha256"],
                    "context_recipe_identity_sha256": item["context_recipes"][probe_id],
                    "context_identity_sha256": ff.canonical_hash(context_sources),
                    "context_sources": context_sources,
                    "prompt_template_identity_sha256": target["prompt_template_identity_sha256"],
                    "checkpoint_identity_sha256": checkpoint,
                    "decoding_identity_sha256": decoding,
                    "shard_index": ff.expected_shard(cell_uid, num_shards),
                    "num_shards": num_shards,
                    "output_status": "malformed" if is_malformed else "ok",
                    "semantic_score": None if is_malformed else score,
                    "hit_token_cap": key in capped,
                    "response_identity_sha256": _digest(f"response/{item['item_uid']}/{probe_id}"),
                    "emitted_state_identity_sha256": emitted_state,
                    "cost": {
                        "model_queries": 1,
                        "teacher_queries": 1,
                        "model_input_tokens": 10,
                        "model_output_tokens": 2,
                        "teacher_input_tokens": 7,
                        "teacher_output_tokens": 3,
                        "specialist_scored_tokens": 2,
                        "model_cost_microusd": 11,
                        "teacher_cost_microusd": 13,
                    },
                }
            )
            emitted_by_probe[probe_id] = (cell_uid, emitted_state)
    return _signed(
        {
            "schema_version": 1,
            "protocol": protocol,
            "run_uid": run_uid,
            "panel_identity_sha256": manifest["identity_sha256"],
            "checkpoint_identity_sha256": checkpoint,
            "decoding_identity_sha256": decoding,
            "cost_policy_identity_sha256": _digest("cost-policy"),
            "num_shards": num_shards,
            "cells": cells,
        }
    )


def _resign(bundle: dict) -> None:
    bundle["identity_sha256"] = ff.canonical_hash(
        {key: value for key, value in bundle.items() if key != "identity_sha256"}
    )


def _set_metric_thresholds(
    manifest: dict,
    item_uid: str,
    stage: str,
    *,
    correct_threshold: float,
    context_sensitivity_minimum: float,
    monotonic_tolerance: float = 0.0,
) -> None:
    item = next(row for row in manifest["items"] if row["item_uid"] == item_uid)
    metric = item["targets"][stage]["semantic_metric_contract"]
    metric["correct_threshold"] = correct_threshold
    metric["context_sensitivity_minimum"] = context_sensitivity_minimum
    metric["monotonic_tolerance"] = monotonic_tolerance
    _resign(metric)
    _resign(manifest)


def _evaluation(report: dict, chain: str, target_stage: str) -> dict:
    return next(
        row
        for row in report["evaluations"]
        if row["chain"] == chain and row["target_stage"] == target_stage
    )


def test_contract_freezes_probe_bridge_and_zero_semantics() -> None:
    contract = ff.load_contract()
    assert contract["score_semantics"]["malformed_response"] == "sample_zero"
    assert (
        contract["score_semantics"]["correctness_threshold_source"]
        == "manifest_target_semantic_metric_contract"
    )
    assert (
        contract["score_semantics"]["context_sensitivity_threshold_source"]
        == "manifest_target_semantic_metric_contract"
    )
    assert (
        contract["score_semantics"]["monotonic_tolerance_source"]
        == "manifest_target_semantic_metric_contract"
    )
    assert contract["probe_definitions"]["A.repair_G"]["context_mode"] == "oracle"
    assert contract["probe_definitions"]["A.repair_G"]["semantic_role"].startswith(
        "A_after_oracle_G"
    )
    assert ff.EXPECTED_CHAINS["G_S_A"].index("A.repair_G") < ff.EXPECTED_CHAINS[
        "G_S_A"
    ].index("A.oracle_GS")
    assert contract["first_fault_states"] == ["Attributed", "Local", "Inconclusive"]


def test_malformed_is_sample_zero_below_health_gate_and_cost_is_complete() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"] * 5)
    overrides = {
        (item["item_uid"], "S.predicted_G"): 1.0 for item in manifest["items"]
    }
    malformed = {(manifest["items"][0]["item_uid"], "S.predicted_G")}
    report = ff.score_bundle(contract, manifest, _bundle(manifest, overrides, malformed))
    cell = report["condition_cells"]["G_S"]["S.predicted_G"]
    assert report["status"] == "PAIRED_SCORE_COMPLETE"
    assert cell["admission"] == "admitted"
    assert cell["malformed_n"] == 1
    assert cell["score"] == 0.8
    metric = _evaluation(report, "G_S", "S")
    assert metric["status"] == "ADMITTED"
    assert metric["dependency_gap"]["estimate"] == 0.2
    assert metric["dependency_gap"]["paired_scene_bootstrap"]["n_scenes"] == 3
    assert metric["oracle_headroom"]["positive_items"] == 1
    assert metric["headroom_recovery"]["G"]["estimate"] == 1.0
    cost = report["query_and_token_cost"]
    assert cost["status"] == "COMPLETE"
    assert cost["expected_cells"] == 25
    assert cost["totals"]["model_queries"] == 25
    assert cost["totals"]["teacher_input_tokens"] == 175
    assert cost["totals"]["model_cost_microusd"] == 275
    assert cost["totals"]["teacher_cost_microusd"] == 325
    assert cost["derived_totals"]["total_cost_microusd"] == 600
    assert cost["derived_totals"]["total_cost_usd"] == 0.0006
    assert cost["per_item"]["model_queries"] == 5.0


def test_output_health_gate_withholds_aggregate_but_keeps_zero_diagnostic() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"] * 5)
    overrides = {
        (item["item_uid"], "S.predicted_G"): 1.0 for item in manifest["items"]
    }
    malformed = {
        (manifest["items"][0]["item_uid"], "S.predicted_G"),
        (manifest["items"][1]["item_uid"], "S.predicted_G"),
    }
    report = ff.score_bundle(contract, manifest, _bundle(manifest, overrides, malformed))
    cell = report["condition_cells"]["G_S"]["S.predicted_G"]
    assert report["status"] == "WITHHELD"
    assert cell["admission"] == "diagnostic_only"
    assert cell["score"] is None
    assert cell["diagnostic_score"] == 0.6
    assert cell["malformed_rate"] == 0.4
    assert cell["withheld_reasons"] == [
        "preregistered_output_health_gate_failure:aggregate_rate"
    ]
    assert _evaluation(report, "G_S", "S")["status"] == "WITHHELD"
    assert report["query_and_token_cost"]["status"] == "COMPLETE"


def test_malformed_upstream_state_remains_zero_and_lineage_is_still_bound() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"] * 5)
    malformed = {(manifest["items"][0]["item_uid"], "G.local")}
    report = ff.score_bundle(contract, manifest, _bundle(manifest, malformed=malformed))
    assert report["status"] == "PAIRED_SCORE_COMPLETE"
    local = report["condition_cells"]["G_S"]["G.local"]
    predicted = report["condition_cells"]["G_S"]["S.predicted_G"]
    assert local["admission"] == "admitted"
    assert local["score"] == 0.8
    assert predicted["admission"] == "admitted"
    assert report["comparability"]["upstream_output_lineage_verified"] is True


@pytest.mark.parametrize(("anomaly", "n_items"), (("malformed", 5), ("token_cap", 10)))
def test_single_item_output_health_failure_is_inconclusive_and_never_routed(
    anomaly: str, n_items: int
) -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"] * n_items)
    item_uids = [item["item_uid"] for item in manifest["items"]]
    overrides = {(item_uid, "G.local"): 0.0 for item_uid in item_uids}
    affected = {(item_uids[0], "S.absent_G")}
    bundle = _bundle(
        manifest,
        overrides,
        malformed=affected if anomaly == "malformed" else None,
        capped=affected if anomaly == "token_cap" else None,
    )
    report = ff.score_bundle(contract, manifest, bundle)
    assert report["status"] == "PAIRED_SCORE_COMPLETE"
    fault = report["first_causal_fault"][0]
    assert fault["status"] == "ADMITTED"
    affected_row = next(row for row in fault["per_item"] if row["item_uid"] == item_uids[0])
    assert affected_row["state"] == "Inconclusive"
    assert affected_row["reason"] == "invalid_or_capped_probe"
    assert affected_row["routing_eligible"] is False
    assert affected_row["routing_stage"] is None
    assert affected_row["local_correct"] == {"G": None, "S": None}
    assert affected_row["probe_health"]["failures"] == [
        {"probe_id": "S.absent_G", "reason": anomaly}
    ]
    healthy_row = next(row for row in fault["per_item"] if row["item_uid"] == item_uids[1])
    assert healthy_row["state"] == "Attributed"
    assert healthy_row["fault_stage"] == "G"
    assert healthy_row["routing_eligible"] is True
    assert fault["routing"]["eligible_n"] == n_items - 1
    assert fault["transition_evaluable_n"] == n_items - 1


def test_first_fault_correctness_uses_bound_metric_threshold_not_stage_exact_match() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"])
    item_uid = manifest["items"][0]["item_uid"]
    overrides = {
        (item_uid, "G.local"): 0.6,
        (item_uid, "S.predicted_G"): 0.0,
        (item_uid, "S.oracle_G"): 0.0,
    }
    fault = ff.score_bundle(contract, manifest, _bundle(manifest, overrides))[
        "first_causal_fault"
    ][0]
    row = fault["per_item"][0]
    assert row["local_correct"]["G"] is True
    assert row["semantic_criteria"]["G.local"]["correct_threshold"] == 0.5
    assert row["state"] == "Local"
    assert row["fault_stage"] == "S"
    assert row["routing_stage"] == "S"


def test_manifest_rejects_metric_threshold_drift_within_one_metric_id(tmp_path: Path) -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S", "G_S"])
    metric = manifest["items"][1]["targets"]["G"]["semantic_metric_contract"]
    metric["correct_threshold"] = 0.75
    metric["identity_sha256"] = ff.canonical_hash(
        {key: value for key, value in metric.items() if key != "identity_sha256"}
    )
    manifest["identity_sha256"] = ff.canonical_hash(
        {key: value for key, value in manifest.items() if key != "identity_sha256"}
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ff.ContractError, match="semantic metric contract drift"):
        ff.load_manifest(path, contract)


def test_gsa_first_fault_transitions_gap_recovery_and_bootstrap_are_deterministic() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S_A"] * 4)
    item = [row["item_uid"] for row in manifest["items"]]
    overrides = {
        # G-origin failure: repairing G repairs A.
        (item[0], "G.local"): 0.0,
        (item[0], "S.oracle_G"): 1.0,
        (item[0], "A.predicted_GS"): 0.0,
        (item[0], "A.repair_G"): 1.0,
        (item[0], "A.oracle_GS"): 1.0,
        # S-origin failure: G repair is insufficient; S repair fixes A.
        (item[1], "G.local"): 1.0,
        (item[1], "S.oracle_G"): 0.0,
        (item[1], "A.predicted_GS"): 0.0,
        (item[1], "A.repair_G"): 0.0,
        (item[1], "A.oracle_GS"): 1.0,
        # Local A failure remains wrong under full oracle context.
        (item[2], "G.local"): 1.0,
        (item[2], "S.oracle_G"): 1.0,
        (item[2], "A.predicted_GS"): 0.0,
        (item[2], "A.repair_G"): 0.0,
        (item[2], "A.oracle_GS"): 0.0,
        # Upstream G is wrong, but the predicted-context A target is already correct.
        (item[3], "G.local"): 0.0,
        (item[3], "S.oracle_G"): 1.0,
        (item[3], "A.predicted_GS"): 1.0,
        (item[3], "A.repair_G"): 1.0,
        (item[3], "A.oracle_GS"): 1.0,
    }
    bundle = _bundle(manifest, overrides)
    first = ff.score_bundle(contract, manifest, bundle)
    second = ff.score_bundle(contract, manifest, bundle)
    assert first == second
    metric = _evaluation(first, "G_S_A", "A")
    assert metric["status"] == "ADMITTED"
    assert metric["dependency_gap"]["estimate"] == 0.5
    assert metric["oracle_headroom"]["total"] == 2.0
    assert metric["repair_effects"]["G"]["signed_mean"] == 0.25
    assert metric["headroom_recovery"]["G"]["estimate"] == 0.5
    assert set(metric["headroom_recovery"]) == {"G"}
    assert metric["oracle_ceiling"] == {
        "reference_probe": "A.oracle_GS",
        "role": "algebraic_oracle_ceiling_not_empirical_ablation",
        "signed_effect_equals_dependency_gap": 0.5,
        "headroom_recovery": 1.0,
        "status": "defined_identity",
    }
    fault = first["first_causal_fault"][0]
    assert fault["counts"] == {"Attributed": 2, "Local": 1, "Inconclusive": 1}
    assert fault["routing"] == {
        "eligible_n": 3,
        "ineligible_n": 1,
        "by_stage": {"G": 1, "S": 1, "A": 1},
    }
    assert [row["fault_stage"] for row in fault["per_item"]] == ["G", "S", "A", None]
    assert fault["per_item"][3]["routing_eligible"] is False
    assert fault["transition_counts"]["G_repair"] == {
        "wrong_to_wrong": 2,
        "wrong_to_correct": 1,
        "correct_to_wrong": 0,
        "correct_to_correct": 1,
    }
    assert fault["transition_counts"]["S_repair"] == {
        "wrong_to_wrong": 1,
        "wrong_to_correct": 1,
        "correct_to_wrong": 0,
        "correct_to_correct": 2,
    }
    assert fault["transition_matrix"]["G_repair"] == {
        "wrong": {"wrong": 2, "correct": 1},
        "correct": {"wrong": 0, "correct": 1},
    }


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("target_identity_sha256", _digest("wrong-target"), "protocol_or_binding_mismatch"),
        ("observation_identity_sha256", _digest("wrong-observation"), "protocol_or_binding_mismatch"),
        ("checkpoint_identity_sha256", _digest("wrong-checkpoint"), "protocol_or_binding_mismatch"),
        ("decoding_identity_sha256", _digest("wrong-decoding"), "protocol_or_binding_mismatch"),
        ("prompt_template_identity_sha256", _digest("wrong-prompt"), "protocol_or_binding_mismatch"),
        ("shard_index", -1, "incomplete_or_incorrect_shard_coverage"),
        ("cell_uid", _digest("wrong-cell"), "protocol_or_binding_mismatch"),
    ),
)
def test_binding_uid_or_shard_drift_withholds_only_the_affected_cell_group(
    field: str, value: object, reason: str
) -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"] * 4)
    bundle = _bundle(manifest)
    row = next(cell for cell in bundle["cells"] if cell["probe_id"] == "S.predicted_G")
    row[field] = value
    _resign(bundle)
    report = ff.score_bundle(contract, manifest, bundle)
    cells = report["condition_cells"]["G_S"]
    assert report["status"] == "WITHHELD"
    assert cells["S.predicted_G"]["admission"] == "withheld"
    assert any(entry.startswith(reason) for entry in cells["S.predicted_G"]["withheld_reasons"])
    assert cells["S.oracle_G"]["admission"] == "admitted"
    assert _evaluation(report, "G_S", "S")["status"] == "WITHHELD"
    assert report["query_and_token_cost"]["status"] == "PARTIAL_NOT_BUDGET_COMPARABLE"


def test_dynamic_context_lineage_drift_withholds_downstream_probe_only() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S_A"] * 3)
    bundle = _bundle(manifest)
    row = next(cell for cell in bundle["cells"] if cell["probe_id"] == "A.repair_G")
    row["context_sources"][1]["state_identity_sha256"] = _digest("wrong-upstream-state")
    row["context_identity_sha256"] = ff.canonical_hash(row["context_sources"])
    _resign(bundle)
    report = ff.score_bundle(contract, manifest, bundle)
    cells = report["condition_cells"]["G_S_A"]
    assert cells["A.repair_G"]["admission"] == "withheld"
    assert cells["S.oracle_G"]["admission"] == "admitted"
    assert cells["A.oracle_GS"]["admission"] == "admitted"
    assert cells["A.repair_G"]["withheld_reasons"] == [
        "missing_duplicate_or_unexpected_uid:missing=1",
        "protocol_or_binding_mismatch:context_lineage",
    ]
    assert report["comparability"]["upstream_output_lineage_verified"] is False


def test_static_context_lineage_drift_withholds_only_that_probe() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"] * 3)
    bundle = _bundle(manifest)
    row = next(cell for cell in bundle["cells"] if cell["probe_id"] == "S.oracle_G")
    row["context_sources"][0]["state_identity_sha256"] = _digest("wrong-static-state")
    row["context_identity_sha256"] = ff.canonical_hash(row["context_sources"])
    _resign(bundle)
    report = ff.score_bundle(contract, manifest, bundle)
    cells = report["condition_cells"]["G_S"]
    assert cells["S.oracle_G"]["admission"] == "withheld"
    assert cells["S.predicted_G"]["admission"] == "admitted"
    assert cells["S.oracle_G"]["withheld_reasons"] == [
        "missing_duplicate_or_unexpected_uid:missing=1",
        "protocol_or_binding_mismatch:context_lineage",
    ]


def test_token_cap_gate_is_probe_local_and_does_not_erase_costs() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"] * 5)
    bundle = _bundle(manifest)
    row = next(cell for cell in bundle["cells"] if cell["probe_id"] == "S.corrupted_G")
    row["hit_token_cap"] = True
    _resign(bundle)
    report = ff.score_bundle(contract, manifest, bundle)
    summary = report["condition_cells"]["G_S"]["S.corrupted_G"]
    assert summary["admission"] == "diagnostic_only"
    assert summary["truncation_rate"] == 0.2
    assert summary["diagnostic_score"] == 0.0
    assert report["condition_cells"]["G_S"]["S.absent_G"]["admission"] == "admitted"
    assert report["query_and_token_cost"]["status"] == "COMPLETE"


def test_malformed_rate_equal_to_gate_remains_sample_zero_and_admitted() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"] * 4)
    item_uid = manifest["items"][0]["item_uid"]
    bundle = _bundle(
        manifest,
        {(item["item_uid"], "S.predicted_G"): 1.0 for item in manifest["items"]},
        {(item_uid, "S.predicted_G")},
    )
    report = ff.score_bundle(contract, manifest, bundle)
    summary = report["condition_cells"]["G_S"]["S.predicted_G"]
    assert summary["malformed_rate"] == 0.25
    assert summary["score"] == 0.75
    assert summary["admission"] == "admitted"


def test_zero_headroom_is_undefined_and_recovery_is_bounded() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S", "G_S_A"])
    gs_uid, gsa_uid = [item["item_uid"] for item in manifest["items"]]
    overrides = {
        (gs_uid, "S.predicted_G"): 1.0,
        (gs_uid, "S.oracle_G"): 1.0,
        (gsa_uid, "A.predicted_GS"): 0.0,
        (gsa_uid, "A.repair_G"): 1.0,
        (gsa_uid, "A.oracle_GS"): 0.5,
    }
    report = ff.score_bundle(contract, manifest, _bundle(manifest, overrides))
    gs_metric = _evaluation(report, "G_S", "S")
    assert gs_metric["oracle_headroom"]["total"] == 0.0
    assert gs_metric["headroom_recovery"]["G"]["status"] == "undefined_no_positive_oracle_headroom"
    assert gs_metric["headroom_recovery"]["G"]["estimate"] is None
    assert gs_metric["headroom_recovery"]["G"]["paired_scene_bootstrap"]["ci"] is None
    gsa_metric = _evaluation(report, "G_S_A", "A")
    assert gsa_metric["oracle_headroom"]["total"] == 0.5
    assert gsa_metric["headroom_recovery"]["G"]["estimate"] == 1.0


def test_gs_transition_matrix_has_all_four_cells() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"] * 4)
    item = [row["item_uid"] for row in manifest["items"]]
    overrides = {
        (item[0], "S.predicted_G"): 0.0,
        (item[0], "S.oracle_G"): 0.0,
        (item[1], "S.predicted_G"): 0.0,
        (item[1], "S.oracle_G"): 1.0,
        (item[2], "S.predicted_G"): 1.0,
        (item[2], "S.oracle_G"): 0.0,
        (item[3], "S.predicted_G"): 1.0,
        (item[3], "S.oracle_G"): 1.0,
    }
    fault = ff.score_bundle(contract, manifest, _bundle(manifest, overrides))[
        "first_causal_fault"
    ][0]
    assert fault["transition_counts"]["G_repair"] == {
        "wrong_to_wrong": 1,
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
        "correct_to_correct": 1,
    }
    assert fault["transition_matrix"]["G_repair"] == {
        "wrong": {"wrong": 1, "correct": 1},
        "correct": {"wrong": 1, "correct": 1},
    }


def test_duplicate_cell_withholds_score_and_budget_comparability() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"])
    bundle = _bundle(manifest)
    bundle["cells"].append(copy.deepcopy(bundle["cells"][0]))
    _resign(bundle)
    report = ff.score_bundle(contract, manifest, bundle)
    cost = report["query_and_token_cost"]
    assert report["status"] == "WITHHELD"
    assert cost["status"] == "PARTIAL_NOT_BUDGET_COMPARABLE"
    assert cost["expected_cells"] == 5
    assert cost["observed_bundle_cells"] == 6
    assert cost["observed_structurally_valid_cells"] == 5


def test_cost_accounting_is_separate_for_each_chain() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S", "G_S_A"])
    report = ff.score_bundle(contract, manifest, _bundle(manifest))
    cost = report["query_and_token_cost"]
    assert cost["status"] == "COMPLETE"
    assert cost["by_chain"]["G_S"]["model_queries"] == 5
    assert cost["by_chain"]["G_S_A"]["model_queries"] == 12
    assert cost["per_item_by_chain"]["G_S"]["model_queries"] == 5.0
    assert cost["per_item_by_chain"]["G_S_A"]["model_queries"] == 12.0


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("comparability", "same_checkpoint", "optional", "comparability contract drift"),
        (
            "score_semantics",
            "correctness_threshold_source",
            "stage_constant",
            "correctness threshold source drift",
        ),
        (
            "score_semantics",
            "context_sensitivity_threshold_source",
            "contract_constant",
            "context sensitivity threshold source drift",
        ),
        ("score_semantics", "dependency_gap", "predicted_minus_oracle", "dependency gap definition drift"),
        ("cell_admission", "max_malformed_rate", 0.3, "max_malformed_rate drift"),
        ("bootstrap", "seed", 1, "bootstrap seed drift"),
        ("cost_accounting", "monetary_unit", "usd", "cost accounting contract drift"),
    ),
)
def test_contract_rejects_evidence_semantic_drift(
    tmp_path: Path, section: str, field: str, value: object, message: str
) -> None:
    payload = yaml.safe_load(ff.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    payload[section][field] = value
    path = tmp_path / "drifted.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ff.ContractError, match=message):
        ff.load_contract(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("correct_threshold", 0.0),
        ("context_sensitivity_minimum", -0.1),
        ("monotonic_tolerance", 1.1),
    ),
)
def test_manifest_rejects_invalid_bound_routing_thresholds(
    tmp_path: Path, field: str, value: float
) -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"])
    metric = manifest["items"][0]["targets"]["S"]["semantic_metric_contract"]
    metric[field] = value
    _resign(metric)
    _resign(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ff.ContractError, match=f"invalid semantic {field}"):
        ff.load_manifest(path, contract)


def test_bundle_protocol_mismatch_withholds_all_cells_without_scoring() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S", "G_S_A"])
    bundle = _bundle(manifest)
    bundle["protocol"] = "different-protocol"
    _resign(bundle)
    report = ff.score_bundle(contract, manifest, bundle)
    assert report["status"] == "WITHHELD"
    assert report["comparability"][
        "same_item_observation_checkpoint_decoding_and_downstream_target"
    ] is False
    assert all(
        cell["admission"] == "withheld"
        for chain in report["condition_cells"].values()
        for cell in chain.values()
    )
    assert all(metric["status"] == "WITHHELD" for metric in report["evaluations"])


def test_unexpected_orphan_cell_forces_cli_withheld(tmp_path: Path) -> None:
    manifest = _manifest(["G_S"])
    bundle = _bundle(manifest)
    bundle["cells"].append({"chain": "BOGUS", "probe_id": "BOGUS"})
    bundle = _signed({key: value for key, value in bundle.items() if key != "identity_sha256"})
    manifest_path = tmp_path / "manifest.json"
    bundle_path = tmp_path / "bundle.json"
    output_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    assert (
        ff.main(
            [
                "--manifest",
                str(manifest_path),
                "--predictions",
                str(bundle_path),
                "--out",
                str(output_path),
            ]
        )
        == 2
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "WITHHELD"
    assert report["comparability"]["uid_and_shard_coverage_verified"] is False
    assert "missing_duplicate_or_unexpected_uid:unknown_item_or_probe" in report[
        "comparability"
    ]["global_errors"]


def test_upstream_health_failure_withholds_dependent_s_metric() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"] * 5)
    malformed = {
        (manifest["items"][0]["item_uid"], "G.local"),
        (manifest["items"][1]["item_uid"], "G.local"),
    }
    report = ff.score_bundle(contract, manifest, _bundle(manifest, malformed=malformed))
    metric = _evaluation(report, "G_S", "S")
    assert metric["status"] == "WITHHELD"
    assert "G.local" in metric["failed_probes"]


def test_upstream_health_failure_withholds_dependent_a_metric() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S_A"] * 5)
    malformed = {
        (manifest["items"][0]["item_uid"], "S.oracle_G"),
        (manifest["items"][1]["item_uid"], "S.oracle_G"),
    }
    report = ff.score_bundle(contract, manifest, _bundle(manifest, malformed=malformed))
    metric = _evaluation(report, "G_S_A", "A")
    assert metric["status"] == "WITHHELD"
    assert "S.oracle_G" in metric["failed_probes"]


@pytest.mark.parametrize("chain", ("G_S", "G_S_A"))
def test_nonmonotonic_oracle_degradation_is_inconclusive(chain: str) -> None:
    contract = _test_contract()
    manifest = _manifest([chain])
    item_uid = manifest["items"][0]["item_uid"]
    if chain == "G_S":
        overrides = {
            (item_uid, "G.local"): 1.0,
            (item_uid, "S.predicted_G"): 1.0,
            (item_uid, "S.oracle_G"): 0.0,
        }
    else:
        overrides = {
            (item_uid, "G.local"): 1.0,
            (item_uid, "S.oracle_G"): 1.0,
            (item_uid, "A.predicted_GS"): 1.0,
            (item_uid, "A.oracle_GS"): 0.0,
        }
    report = ff.score_bundle(contract, manifest, _bundle(manifest, overrides))
    fault = next(row for row in report["first_causal_fault"] if row["chain"] == chain)
    assert fault["per_item"][0]["state"] == "Inconclusive"
    assert fault["per_item"][0]["reason"] == "oracle_degradation"
    assert fault["per_item"][0]["routing_eligible"] is False
    assert fault["per_item"][0]["routing_stage"] is None


@pytest.mark.parametrize("chain", ("G_S", "G_S_A"))
def test_same_side_continuous_oracle_degradation_is_inconclusive(chain: str) -> None:
    contract = _test_contract()
    manifest = _manifest([chain])
    item_uid = manifest["items"][0]["item_uid"]
    if chain == "G_S":
        overrides = {
            (item_uid, "G.local"): 1.0,
            (item_uid, "S.predicted_G"): 0.4,
            (item_uid, "S.oracle_G"): 0.3,
        }
    else:
        overrides = {
            (item_uid, "G.local"): 1.0,
            (item_uid, "S.oracle_G"): 0.0,
            (item_uid, "A.predicted_GS"): 0.4,
            (item_uid, "A.repair_G"): 0.3,
            (item_uid, "A.oracle_GS"): 1.0,
        }
    row = ff.score_bundle(contract, manifest, _bundle(manifest, overrides))[
        "first_causal_fault"
    ][0]["per_item"][0]
    assert row["state"] == "Inconclusive"
    assert row["reason"] == "oracle_degradation"
    assert row["routing_eligible"] is False


def test_multiple_upstream_faults_never_route_when_g_repair_fixes_a() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S_A"])
    item_uid = manifest["items"][0]["item_uid"]
    overrides = {
        (item_uid, "G.local"): 0.0,
        (item_uid, "S.oracle_G"): 0.0,
        (item_uid, "A.predicted_GS"): 0.0,
        (item_uid, "A.repair_G"): 1.0,
        (item_uid, "A.oracle_GS"): 1.0,
    }
    row = ff.score_bundle(contract, manifest, _bundle(manifest, overrides))[
        "first_causal_fault"
    ][0]["per_item"][0]
    assert row["local_correct"] == {"G": False, "S": False, "A": True}
    assert row["state"] == "Inconclusive"
    assert row["reason"] == "multiple_unresolved_faults"
    assert row["routing_stage"] is None


def test_same_response_with_different_scores_is_inconclusive() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"])
    bundle = _bundle(manifest)
    predicted = next(
        cell for cell in bundle["cells"] if cell["probe_id"] == "S.predicted_G"
    )
    oracle = next(
        cell for cell in bundle["cells"] if cell["probe_id"] == "S.oracle_G"
    )
    oracle["response_identity_sha256"] = predicted["response_identity_sha256"]
    _resign(bundle)
    report = ff.score_bundle(contract, manifest, bundle)
    row = report["first_causal_fault"][0]["per_item"][0]
    assert report["status"] == "WITHHELD"
    assert report["evaluations"][0]["status"] == "WITHHELD"
    assert report["comparability"]["response_score_consistency_verified"] is False
    assert row["state"] == "Inconclusive"
    assert row["reason"] == "response_score_inconsistency"
    assert row["semantic_consistency"]["valid"] is False
    assert row["transition_evaluable"] is False


def test_scorer_uses_bound_nonunit_target_and_context_thresholds() -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"])
    item_uid = manifest["items"][0]["item_uid"]
    _set_metric_thresholds(
        manifest,
        item_uid,
        "S",
        correct_threshold=0.5,
        context_sensitivity_minimum=0.25,
    )
    overrides = {
        (item_uid, "G.local"): 0.0,
        (item_uid, "S.predicted_G"): 0.4,
        (item_uid, "S.oracle_G"): 0.6,
        (item_uid, "S.absent_G"): 0.4,
        (item_uid, "S.corrupted_G"): 0.4,
    }
    row = ff.score_bundle(contract, manifest, _bundle(manifest, overrides))[
        "first_causal_fault"
    ][0]["per_item"][0]
    assert row["semantic_criteria"]["S.predicted_G"] == {
        "metric_id": "synthetic-s-exact-v1",
        "correct_threshold": 0.5,
        "context_sensitivity_minimum": 0.25,
        "monotonic_tolerance": 0.0,
        "identity_sha256": manifest["items"][0]["targets"]["S"][
            "semantic_metric_contract"
        ]["identity_sha256"],
    }
    assert row["local_correct"] == {"G": False, "S": True}
    assert row["state_use_confirmed"] == {"G": False}
    assert row["state"] == "Inconclusive"
    assert row["reason"] == "context_insensitive_repair"

    _set_metric_thresholds(
        manifest,
        item_uid,
        "S",
        correct_threshold=0.5,
        context_sensitivity_minimum=0.1,
    )
    admitted = ff.score_bundle(contract, manifest, _bundle(manifest, overrides))[
        "first_causal_fault"
    ][0]["per_item"][0]
    assert admitted["state_use_confirmed"] == {"G": True}
    assert admitted["state"] == "Attributed"
    assert admitted["fault_stage"] == "G"


def test_gsa_scorer_and_shared_router_have_exhaustive_binary_parity() -> None:
    contract = _test_contract()
    contract["bootstrap"]["resamples"] = 1
    manifest = _manifest(["G_S_A"])
    item_uid = manifest["items"][0]["item_uid"]
    for g_score, s_score, predicted, repair_g, oracle in product((0.0, 1.0), repeat=5):
        overrides = {
            (item_uid, "G.local"): g_score,
            (item_uid, "S.oracle_G"): s_score,
            (item_uid, "A.predicted_GS"): predicted,
            (item_uid, "A.repair_G"): repair_g,
            (item_uid, "A.oracle_GS"): oracle,
        }
        bundle = _bundle(manifest, overrides)
        report_row = ff.score_bundle(contract, manifest, bundle)[
            "first_causal_fault"
        ][0]["per_item"][0]
        cells = {cell["probe_id"]: cell for cell in bundle["cells"]}
        direct = route_first_fault(
            stage_order=("G", "S", "A"),
            local_correct={
                "G": bool(g_score),
                "S": bool(s_score),
                "A": bool(oracle),
            },
            predicted_target_score=predicted,
            repair_target_scores={"G": repair_g, "S": oracle},
            predicted_target_response_identity=cells["A.predicted_GS"][
                "response_identity_sha256"
            ],
            repair_target_response_identities={
                "G": cells["A.repair_G"]["response_identity_sha256"],
                "S": cells["A.oracle_GS"]["response_identity_sha256"],
            },
            state_use_confirmed=report_row["state_use_confirmed"],
            probes_valid=True,
            hit_token_cap=False,
            correct_threshold=1.0,
            monotonic_tolerance=0.0,
        )
        assert (
            report_row["state"],
            report_row["fault_stage"],
            report_row["reason"],
        ) == (direct.kind.value, direct.stage, direct.reason)


def test_gs_scorer_and_shared_router_have_exhaustive_binary_parity() -> None:
    contract = _test_contract()
    contract["bootstrap"]["resamples"] = 1
    manifest = _manifest(["G_S"])
    item_uid = manifest["items"][0]["item_uid"]
    for g_score, predicted, oracle in product((0.0, 1.0), repeat=3):
        overrides = {
            (item_uid, "G.local"): g_score,
            (item_uid, "S.predicted_G"): predicted,
            (item_uid, "S.oracle_G"): oracle,
        }
        bundle = _bundle(manifest, overrides)
        report_row = ff.score_bundle(contract, manifest, bundle)[
            "first_causal_fault"
        ][0]["per_item"][0]
        cells = {cell["probe_id"]: cell for cell in bundle["cells"]}
        direct = route_first_fault(
            stage_order=("G", "S"),
            local_correct={"G": bool(g_score), "S": bool(oracle)},
            predicted_target_score=predicted,
            repair_target_scores={"G": oracle},
            predicted_target_response_identity=cells["S.predicted_G"][
                "response_identity_sha256"
            ],
            repair_target_response_identities={
                "G": cells["S.oracle_G"]["response_identity_sha256"]
            },
            state_use_confirmed=report_row["state_use_confirmed"],
            probes_valid=True,
            hit_token_cap=False,
            correct_threshold=1.0,
            monotonic_tolerance=0.0,
        )
        assert (
            report_row["state"],
            report_row["fault_stage"],
            report_row["reason"],
        ) == (direct.kind.value, direct.stage, direct.reason)


def test_manifest_and_bundle_canonical_identity_are_fail_closed(tmp_path: Path) -> None:
    contract = _test_contract()
    manifest = _manifest(["G_S"])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert ff.load_manifest(manifest_path, contract)["identity_sha256"] == manifest["identity_sha256"]
    bundle = _bundle(manifest)
    bundle["cells"][0]["semantic_score"] = 0.5
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ff.ContractError, match="canonical identity mismatch"):
        ff.load_bundle(bundle_path)


def test_cli_writes_an_identity_bound_report(tmp_path: Path) -> None:
    manifest = _manifest(["G_S"])
    bundle = _bundle(manifest)
    manifest_path = tmp_path / "manifest.json"
    bundle_path = tmp_path / "bundle.json"
    output_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    assert (
        ff.main(
            [
                "--manifest",
                str(manifest_path),
                "--predictions",
                str(bundle_path),
                "--out",
                str(output_path),
            ]
        )
        == 0
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in report.items() if key != "identity_sha256"}
    assert report["identity_sha256"] == ff.canonical_hash(unsigned)
    assert report["bindings"]["contract"]["identity_sha256"] == ff.canonical_hash(
        ff.load_contract()
    )
