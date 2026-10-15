from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import compile_scannet_v51_g_seed42_s_retention as retention


def _signed(payload: dict) -> dict:
    value = dict(payload)
    value["identity_sha256"] = retention.canonical_hash(payload)
    return value


def _binding(path: Path) -> dict:
    path = path.resolve()
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _items(*, candidate: bool = False) -> dict[str, dict]:
    # One shared B error yields 39/40, A/B=20/19, and inferred A=21/40.
    rows = {}
    for index in range(40):
        target = "A" if index < 20 else "B"
        rows[f"scene-{index:02d}"] = {
            "eval_id": f"scene-{index:02d}",
            "target_stratum": target,
            "parse_status": "direct",
            "parsed": True,
            "hit_token_cap": False,
            "correct": index != 20,
        }
    if candidate:
        return {key: dict(value) for key, value in reversed(tuple(rows.items()))}
    return rows


def _metric(correct: int, total: int) -> dict:
    return {
        "n_scenes": total,
        "correct": correct,
        "accuracy": round(correct / total, 6),
        "parsed_responses": total,
        "parse_coverage": 1.0,
        "uncapped_parsed_responses": total,
        "uncapped_parse_coverage": 1.0,
        "capped_responses": 0,
        "cap_rate": 0.0,
        "accuracy_binomial_ci": retention._wilson_ci(correct, total),
    }


def test_paired_retention_stats_are_scene_aligned_and_order_independent() -> None:
    base = _items()
    candidate = _items(candidate=True)
    paired, records = retention._paired_summary(base, candidate)

    assert len(records) == 40
    assert paired["base"] == {"correct": 39, "accuracy": 0.975}
    assert paired["candidate"] == {"correct": 39, "accuracy": 0.975}
    assert paired["accuracy_delta_candidate_minus_base"] == 0.0
    assert paired["paired_scene_bootstrap_95_ci"]["ci"] == [0.0, 0.0]
    assert paired["discordance"] == {
        "both_correct": 39,
        "both_wrong": 1,
        "candidate_only_correct": 0,
        "base_only_correct": 0,
        "discordant_total": 0,
        "net_candidate_correct_advantage": 0,
    }
    assert paired["identical_scene_correctness_vector"] is True
    assert paired["identical_inferred_binary_predictions"] is True


def test_position_summary_reports_binary_A_preference_without_overclaim() -> None:
    summary = retention._position_summary(_items())
    assert summary["predicted_A"] == 21
    assert summary["predicted_B"] == 19
    assert summary["predicted_A_rate"] == 0.525
    assert summary["A_rate_minus_balanced_target"] == 0.025
    assert summary["A_minus_B_prediction_rate"] == 0.05
    assert summary["strata"]["A"] == {"n": 20, "correct": 20, "accuracy": 1.0}
    assert summary["strata"]["B"] == {"n": 20, "correct": 19, "accuracy": 0.95}
    assert summary["A_minus_B_accuracy"] == 0.05


def test_result_item_validator_recomputes_metrics_and_rejects_drift() -> None:
    items = list(_items().values())
    result = {
        "per_item": items,
        "metrics": _metric(39, 40),
        "ab_strata": {"A": _metric(20, 20), "B": _metric(19, 20)},
        "parse_status": {"direct": 40},
    }
    observed = retention._validate_items(result, "fixture")
    assert len(observed) == 40

    result["metrics"] = _metric(38, 40)
    with pytest.raises(retention.DiagnosticError, match="fixture.metrics drift"):
        retention._validate_items(result, "fixture")


def test_rehash_ledger_rejects_live_bound_file_drift(tmp_path: Path) -> None:
    path = tmp_path / "bound.json"
    path.write_text('{"ok":true}\n', encoding="utf-8")
    binding = _binding(path)
    path.write_text('{"ok":false}\n', encoding="utf-8")

    ledger = retention.RehashLedger()
    with pytest.raises(retention.DiagnosticError, match=r"(bytes|sha256) drift"):
        ledger.verify_binding(binding, "fixture")


def test_execution_receipt_requires_four_distinct_L20s(tmp_path: Path) -> None:
    hardware = tmp_path / "hardware.csv"
    hardware.write_text(
        "\n".join(
            f"{index}, GPU-{index}, NVIDIA L20, 46068, 535.154.05" for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_file = tmp_path / "result.json"
    candidate_file.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text("{}\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    input_receipt = tmp_path / "inputs.json"
    input_receipt.write_text("{}\n", encoding="utf-8")

    candidate_binding = _binding(candidate_file)
    model_binding = _binding(model)
    prediction_binding = _binding(predictions)
    manifest_binding = _binding(manifest)
    input_binding = _binding(input_receipt)
    candidate_result = {
        "provenance": {
            "model_binding": model_binding,
            "s_oracle_bundle": {
                "merged_predictions": prediction_binding,
                "prediction_manifest": manifest_binding,
                "input_receipt": input_binding,
            },
        }
    }
    payload = {
        "schema_version": 1,
        "protocol": retention.EXECUTION_PROTOCOL,
        "physical_gpu_count": 4,
        "training_materialization_eligible": False,
        "formal_result": candidate_binding,
        "model_binding": model_binding,
        "predictions": prediction_binding,
        "prediction_manifest": manifest_binding,
        "input_receipt": input_binding,
        "hardware_inventory": _binding(hardware),
        "started_at_utc": "2026-08-23T00:00:00Z",
        "finished_at_utc": "2026-08-23T00:01:00Z",
    }
    receipt = _signed(payload)
    observed = retention._validate_execution_receipt(
        receipt,
        expected_identity=receipt["identity_sha256"],
        candidate_binding=candidate_binding,
        candidate_result=candidate_result,
        ledger=retention.RehashLedger(),
    )
    assert observed["hardware"] == "4xNVIDIA_L20"

    bad_payload = {**payload, "physical_gpu_count": 3}
    bad = _signed(bad_payload)
    with pytest.raises(retention.DiagnosticError, match="physical_gpu_count drift"):
        retention._validate_execution_receipt(
            bad,
            expected_identity=bad["identity_sha256"],
            candidate_binding=candidate_binding,
            candidate_result=candidate_result,
            ledger=retention.RehashLedger(),
        )
