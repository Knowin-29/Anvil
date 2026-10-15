from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import compile_scannet_v51_s_oracle_leaderboard as leaderboard


MODEL_KEYS = (
    leaderboard.REFERENCE_KEY,
    "qwen3_vl_8b",
    "e1_posttrained",
    "qwen2_5_vl_7b",
    "qwen2_5_vl_3b",
)


def _identity(payload: dict) -> dict:
    value = dict(payload)
    value["identity_sha256"] = leaderboard.canonical_hash(payload)
    return value


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path.resolve()


def _binding(path: Path, *, mtime: bool = False) -> dict:
    path = path.resolve()
    stat = path.stat()
    value = {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if mtime:
        value["mtime_ns"] = stat.st_mtime_ns
    return value


def _metric_block(correct: int, total: int) -> dict:
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
        "accuracy_binomial_ci": leaderboard._wilson_ci(correct, total),
    }


def _make_fixture(tmp_path: Path) -> tuple[tuple[leaderboard.ModelSpec, ...], dict[str, Path]]:
    shared = tmp_path / "shared"
    scorer = _binding(_write(shared / "scorer.py", b"# frozen scorer\n"))
    release = _binding(_write(shared / "release.manifest.json", b'{"release":"formal40"}\n'))
    requests = _binding(_write(shared / "s_oracle_requests.jsonl", b'{"eval_id":"fixture"}\n'))
    source_file = _binding(_write(shared / "source.jsonl", b'{"source":"fixture"}\n'))
    frozen_files = {"scorer.py": scorer}
    frozen_code = {"files": frozen_files, "identity_sha256": leaderboard.canonical_hash(frozen_files)}
    formal_source = _identity(
        {
            "release": "fixture_formal40",
            "source_wrappers": source_file,
            "records": 40,
            "unique_scenes": 40,
            "unique_images": 40,
            "official_split": "val",
            "human_audit_status": "FORMAL_PASS",
            "training_materialization_eligible": False,
        }
    )
    receipt = _identity(
        {
            "schema_version": 1,
            "protocol": leaderboard.RUN_PROTOCOL,
            "evaluation_protocol": leaderboard.EVALUATION_PROTOCOL,
            "request_release": {"manifest": release, "identity_sha256": "a" * 64},
            "formal_source": formal_source,
            "generation": {"same_bound_model_all_stages": True},
            "evaluation_code": frozen_code,
            "training_materialization_eligible": False,
        }
    )
    receipt_path = _write(
        shared / "formal40_inputs.json", (json.dumps(receipt, sort_keys=True) + "\n").encode()
    )
    receipt_binding = _binding(receipt_path)

    wrong = {
        leaderboard.REFERENCE_KEY: {20},
        "qwen3_vl_8b": {0, 20, 21},
        "e1_posttrained": {0, 1, 20, 21},
        "qwen2_5_vl_7b": {0, 1, 2, 20, 21},
        "qwen2_5_vl_3b": {0, 1, 2, 3, 20, 21, 22, 23, 24},
    }
    specs: list[leaderboard.ModelSpec] = []
    result_paths: dict[str, Path] = {}
    for key in MODEL_KEYS:
        model_dir = tmp_path / "models" / key
        config = _binding(_write(model_dir / "config.json", b'{"model_type":"fixture_vl"}\n'))
        weight = _binding(_write(model_dir / "model.safetensors", f"weights-{key}".encode()), mtime=True)
        inventory = _identity(
            {
                "path": str(model_dir.resolve()),
                "architecture": "FixtureVLForConditionalGeneration",
                "model_type": "fixture_vl",
                "source_kind": "standalone_merged_weights",
                "metadata_files": {"config.json": config},
                "weight_files": [weight],
                "weight_bytes_fully_sha256_hashed": True,
            }
        )
        model_binding_payload = _identity({"role": "evaluated", "model": inventory})
        model_binding_path = _write(
            tmp_path / key / "model_binding.json",
            (json.dumps(model_binding_payload, sort_keys=True) + "\n").encode(),
        )
        model_binding = _binding(model_binding_path)
        predictions = _binding(
            _write(tmp_path / key / "predictions.jsonl", b'{"eval_id":"fixture","response":"A"}\n')
        )
        shard = _binding(
            _write(tmp_path / key / "shard_0.jsonl", b'{"eval_id":"fixture","response":"A"}\n')
        )
        manifest_payload = _identity({"predictions": predictions, "records": 40})
        manifest_path = _write(
            tmp_path / key / "predictions.manifest.json",
            (json.dumps(manifest_payload, sort_keys=True) + "\n").encode(),
        )
        manifest = _binding(manifest_path)
        bundle = _identity(
            {
                "verified": True,
                "protocol": leaderboard.RUN_PROTOCOL,
                "evaluation_protocol": leaderboard.EVALUATION_PROTOCOL,
                "track": leaderboard.TRACK,
                "prediction_manifest": manifest,
                "prediction_manifest_identity_sha256": manifest_payload["identity_sha256"],
                "merged_predictions": predictions,
                "prediction_shards": [shard],
                "model_binding": model_binding,
                "model_binding_identity_sha256": model_binding_payload["identity_sha256"],
                "model_inventory_identity_sha256": inventory["identity_sha256"],
                "input_receipt": receipt_binding,
                "request_manifest": release,
                "requests": requests,
                "request_records_sha256": "b" * 64,
                "decoding": {
                    "thinking": False,
                    "temperature": 0.0,
                    "max_tokens": 16,
                    "batch_size_per_gpu": 4,
                },
                "n_predictions": 40,
            }
        )
        provenance = _identity(
            {
                "verified": True,
                "scorer": scorer,
                "frozen_evaluation_code": frozen_code,
                "request_release": release,
                "request_release_identity_sha256": "a" * 64,
                "formal_source": formal_source,
                "input_receipt": receipt,
                "model_binding": model_binding,
                "model_binding_identity_sha256": model_binding_payload["identity_sha256"],
                "model_inventory": inventory,
                "s_oracle_bundle": bundle,
                "referent_interface_audit_identity_sha256": "c" * 64,
                "training_materialization_eligible": False,
            }
        )
        items = [
            {
                "eval_id": f"eval-{index:02d}",
                "target_stratum": "A" if index < 20 else "B",
                "parse_status": "direct",
                "parsed": True,
                "hit_token_cap": False,
                "correct": index not in wrong[key],
            }
            for index in range(40)
        ]
        correct = sum(int(item["correct"]) for item in items)
        a_correct = sum(int(item["correct"]) for item in items[:20])
        b_correct = sum(int(item["correct"]) for item in items[20:])
        report = _identity(
            {
                "result_schema_version": 1,
                "protocol": leaderboard.RESULT_PROTOCOL,
                "evaluation_protocol": leaderboard.EVALUATION_PROTOCOL,
                "release": leaderboard.RELEASE,
                "status": leaderboard.RESULT_STATUS,
                "track": {
                    "name": leaderboard.TRACK,
                    "status": "FORMAL_SCORE_COMPLETE",
                    "context": "human_ground_truth_bounding_boxes",
                },
                "related_tracks": {
                    "G_named_pair_bbox": {"status": "INTERFACE_BLOCKED_NONUNIQUE_REFERENTS"}
                },
                "task_contract": leaderboard.EXPECTED_TASK_CONTRACT,
                "metrics": _metric_block(correct, 40),
                "ab_strata": {
                    "A": _metric_block(a_correct, 20),
                    "B": _metric_block(b_correct, 20),
                },
                "confidence_intervals": {
                    "accuracy_scene_bootstrap": leaderboard._bootstrap_ci(
                        [int(item["correct"]) for item in items], paired=False
                    ),
                    "accuracy_binomial": leaderboard._wilson_ci(correct, 40),
                },
                "parse_status": {"direct": 40},
                "per_item": items,
                "provenance": provenance,
                "trust": leaderboard.EXPECTED_TRUST,
            }
        )
        result_path = _write(
            tmp_path / key / "result.json", (json.dumps(report, sort_keys=True) + "\n").encode()
        )
        result_paths[key] = result_path
        specs.append(
            leaderboard.ModelSpec(
                key=key,
                display_name=key,
                role="fixture",
                result_path=result_path,
                expected_model_inventory_identity_sha256=inventory["identity_sha256"],
            )
        )
    return tuple(specs), {"source": Path(source_file["path"]), **result_paths}


def _rewrite_result(path: Path, mutate) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    mutate(report)
    unsigned = {key: value for key, value in report.items() if key != "identity_sha256"}
    report["identity_sha256"] = leaderboard.canonical_hash(unsigned)
    path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")


def test_compiles_metrics_position_bias_paired_delta_and_discordance(tmp_path: Path) -> None:
    specs, _ = _make_fixture(tmp_path)
    first = leaderboard.compile_leaderboard(specs)
    second = leaderboard.compile_leaderboard(specs)
    assert first == second
    unsigned = {key: value for key, value in first.items() if key != "identity_sha256"}
    assert first["identity_sha256"] == leaderboard.canonical_hash(unsigned)
    assert first["comparability"]["status"] == "PASS"
    assert first["comparability"]["common"]["records"] == 40
    assert first["live_rehash"]["status"] == "LIVE_REHASH_PASS"
    assert first["live_rehash"]["explicit_result_files_rehashed"] == 5

    rows = {row["model_key"]: row for row in first["leaderboard"]}
    assert [rows[key]["correct"] for key in MODEL_KEYS] == [39, 37, 36, 35, 31]
    assert rows["qwen3_vl_8b"]["accuracy"] == 0.925
    assert rows["qwen3_vl_8b"]["A"] == {"n": 20, "correct": 19, "accuracy": 0.95}
    assert rows["qwen3_vl_8b"]["B"] == {"n": 20, "correct": 18, "accuracy": 0.9}
    assert rows["qwen3_vl_8b"]["prediction_distribution"]["predicted_A_rate"] == 0.525
    comparison = rows["qwen3_vl_8b"]["vs_qwen3_5_9b"]
    assert comparison["accuracy_delta_candidate_minus_reference"] == -0.05
    assert comparison["paired_scene_bootstrap_95_ci"]["seed"] == 51051
    assert comparison["discordance"] == {
        "both_correct": 37,
        "both_wrong": 1,
        "candidate_only_correct": 0,
        "reference_only_correct": 2,
        "discordant_total": 2,
        "net_candidate_correct_advantage": -2,
    }


def test_rejects_live_bound_file_hash_drift(tmp_path: Path) -> None:
    specs, paths = _make_fixture(tmp_path)
    paths["source"].write_bytes(b"tampered source\n")
    with pytest.raises(leaderboard.LeaderboardError, match="bound file (size|hash) drift"):
        leaderboard.compile_leaderboard(specs)


def test_rejects_incomplete_trust_before_rehash(tmp_path: Path) -> None:
    specs, paths = _make_fixture(tmp_path)
    _rewrite_result(
        paths["qwen3_vl_8b"],
        lambda report: report["trust"].__setitem__("score_available", False),
    )
    with pytest.raises(leaderboard.LeaderboardError, match=r"qwen3_vl_8b\.trust drift"):
        leaderboard.compile_leaderboard(specs)


def test_rejects_eval_id_and_formal_source_mismatch(tmp_path: Path) -> None:
    specs, paths = _make_fixture(tmp_path)
    _rewrite_result(
        paths["qwen3_vl_8b"],
        lambda report: report["per_item"][-1].__setitem__("eval_id", "different-eval-id"),
    )
    with pytest.raises(leaderboard.LeaderboardError, match="eval_ids drift"):
        leaderboard.compile_leaderboard(specs)

    specs, paths = _make_fixture(tmp_path / "source-mismatch")

    def change_source(report: dict) -> None:
        source = report["provenance"]["formal_source"]
        source["release"] = "different_formal_source"
        source["identity_sha256"] = leaderboard.canonical_hash(
            {key: value for key, value in source.items() if key != "identity_sha256"}
        )
        provenance = report["provenance"]
        provenance["identity_sha256"] = leaderboard.canonical_hash(
            {key: value for key, value in provenance.items() if key != "identity_sha256"}
        )

    _rewrite_result(paths["qwen3_vl_8b"], change_source)
    with pytest.raises(leaderboard.LeaderboardError, match="formal_comparability_context drift"):
        leaderboard.compile_leaderboard(specs)
