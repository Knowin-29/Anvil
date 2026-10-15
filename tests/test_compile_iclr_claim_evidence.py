from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


from eval import compile_iclr_claim_evidence as compiler
from eval import general_retention_formal_v3 as general
from eval import scannet_rgb_depth_v52_eval as v52_eval


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_status(path: Path, run_root: Path, v52_state: str) -> None:
    rows = [
        "timestamp_utc\tstage\tstate\trc\tdetail",
        "2026-08-27T01:00:00Z\tgeneral_retention\tSUCCEEDED\t0\tformal artifacts complete",
    ]
    if v52_state == "pending":
        report = {"status": "PENDING_HUMAN_REVIEW"}
        report["identity_sha256"] = v52_eval.canonical_hash(report)
        _write_json(run_root / "v52_formal_compile.json", report)
        rows.extend(
            [
                "2026-08-27T01:00:01Z\tv52_formal\tBLOCKED\t0\tstatus=PENDING_HUMAN_REVIEW report="
                + str((run_root / "v52_formal_compile.json").resolve()),
                "2026-08-27T01:00:02Z\tv52_paired\tSKIPPED_PENDING_ANNOTATION\t0\tcompile_status=PENDING_HUMAN_REVIEW",
            ]
        )
    elif v52_state == "complete":
        rows.append(
            "2026-08-27T01:00:02Z\tv52_paired\tSUCCEEDED\t0\tformal exact40 complete"
        )
    else:
        raise AssertionError(v52_state)
    rows.append(
        "2026-08-27T01:00:03Z\tpipeline\tCOMPLETE_WITH_BLOCKED\t0\tclaim-bearing stages complete"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _project(tmp_path: Path, *, v52_state: str = "pending") -> tuple[Path, Path]:
    project = tmp_path / "project"
    for index, relative in enumerate(compiler.EXPECTED_TRACKED_CODE):
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture code {index}\n", encoding="utf-8")
    run_root = project / "artifacts/dlc_runs/synthetic_complete"
    run_root.mkdir(parents=True)
    tracked = {
        relative: compiler.file_hash(project / relative)
        for relative in compiler.EXPECTED_TRACKED_CODE
    }
    _write_json(
        run_root / "run_manifest.json",
        {
            "schema_version": 2,
            "run_id": run_root.name,
            "root": str(project.resolve()),
            "git_commit": "a" * 40,
            "tracked_code": tracked,
            "environment": {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "PYTHON_BIN": "/fixture/python",
                "BASE_MODEL": "/fixture/base",
                "E1_MODEL": "/fixture/e1",
                "HF_HUB_OFFLINE": "1",
                "WANDB_MODE": "offline",
            },
            "resume_count": 0,
        },
    )
    _write_status(run_root / "status.tsv", run_root, v52_state)
    return project, run_root


def _general_fixture(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, int]:
    root = run_root / "general_retention_base_e1"
    receipt_path = root / "inputs/input_receipt.json"
    _write_json(receipt_path, {"fixture": "receipt"})
    bindings = {}
    shards_by_role = {}
    for role in ("base", "e1"):
        binding_path = root / f"model_bindings/{role}.json"
        _write_json(binding_path, {"fixture": role})
        bindings[role] = binding_path
        shards = []
        for shard in range(4):
            path = root / f"predictions/{role}/shard_{shard}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{role}-shard-{shard}\n", encoding="utf-8")
            shards.append(path)
        shards_by_role[role] = shards
    for name in ("base", "e1", "base_vs_e1"):
        _write_json(root / f"results/{name}.json", {"fixture": name, "untampered": True})

    counts = {dataset: 4 for dataset in general.DATASET_ORDER}
    runtime_rows = []
    base_items = []
    e1_items = []
    base_credits = {
        "ai2d_test": [0.0, 1.0, 1.0, 0.0],
        "scienceqa_test": [1.0, 1.0, 0.0, 0.0],
        "textvqa_val": [0.0, 0.3, 0.6, 1.0],
    }
    e1_credits = {
        "ai2d_test": [1.0, 1.0, 1.0, 0.0],
        "scienceqa_test": [1.0, 0.0, 1.0, 0.0],
        "textvqa_val": [0.3, 0.3, 0.9, 1.0],
    }
    for dataset in general.DATASET_ORDER:
        for index in range(4):
            uid = f"{dataset}:{index}"
            runtime_rows.append({"dataset": dataset, "global_uid": uid})
            base_items.append(
                {"dataset": dataset, "global_uid": uid, "credit": base_credits[dataset][index]}
            )
            e1_items.append(
                {"dataset": dataset, "global_uid": uid, "credit": e1_credits[dataset][index]}
            )

    def cell(dataset: str, credits: list[float]) -> dict:
        return {
            "metric": "fixture_metric",
            "n": 4,
            "raw_score_percent": 100.0 * sum(credits) / 4,
            "malformed_count": 0,
            "malformed_percent": 0.0,
            "output_validity_percent": 100.0,
            "truncation_count": 0,
            "truncation_percent": 0.0,
            "health_admitted": True,
            "health_violations": [],
            "malformed_response_credit": 0.0,
        }

    base_benchmarks = {
        dataset: cell(dataset, base_credits[dataset]) for dataset in general.DATASET_ORDER
    }
    e1_benchmarks = {
        dataset: cell(dataset, e1_credits[dataset]) for dataset in general.DATASET_ORDER
    }
    base_macro = sum(x["raw_score_percent"] for x in base_benchmarks.values()) / 3
    e1_macro = sum(x["raw_score_percent"] for x in e1_benchmarks.values()) / 3

    def result(role: str) -> dict:
        benchmarks = base_benchmarks if role == "base" else e1_benchmarks
        items = base_items if role == "base" else e1_items
        macro = base_macro if role == "base" else e1_macro
        return {
            "status": "FORMAL_SCORE_ADMITTED",
            "protocol": general.PROTOCOL,
            "model_role": role,
            "input_receipt": general.file_binding(receipt_path),
            "model_binding": general.file_binding(bindings[role]),
            "model_binding_identity_sha256": f"{1 if role == 'base' else 2:064x}",
            "model_checkpoint_content_identity_sha256": f"{3 if role == 'base' else 4:064x}",
            "prediction_shards": [general.file_binding(path) for path in shards_by_role[role]],
            "benchmarks": benchmarks,
            "raw_general_avg_percent": macro,
            "general_avg_percent": macro,
            "per_sample": items,
        }

    base_result, e1_result = result("base"), result("e1")
    paired_cells = {}
    for dataset in general.DATASET_ORDER:
        left, right = base_credits[dataset], e1_credits[dataset]
        paired_cells[dataset] = {
            "n": 4,
            "base_raw_score_percent": base_benchmarks[dataset]["raw_score_percent"],
            "e1_raw_score_percent": e1_benchmarks[dataset]["raw_score_percent"],
            "e1_minus_base_points": 100.0 * sum(b - a for a, b in zip(left, right)) / 4,
            "retention_ratio_percent": (
                100.0
                * e1_benchmarks[dataset]["raw_score_percent"]
                / base_benchmarks[dataset]["raw_score_percent"]
                if base_benchmarks[dataset]["raw_score_percent"]
                else None
            ),
            "paired_improved": sum(b > a for a, b in zip(left, right)),
            "paired_regressed": sum(b < a for a, b in zip(left, right)),
            "paired_unchanged": sum(b == a for a, b in zip(left, right)),
            "base_health_admitted": True,
            "e1_health_admitted": True,
        }
    comparison = {
        "status": "PAIRED_GENERAL_RETENTION_ADMITTED",
        "protocol": general.PROTOCOL,
        "benchmarks": paired_cells,
        "raw_general_avg_delta_points": e1_macro - base_macro,
    }

    expected_results = {"base": base_result, "e1": e1_result}

    def verify_result(path: Path, **_kwargs: object) -> dict:
        disk = json.loads(path.read_text(encoding="utf-8"))
        if disk.get("untampered") is not True:
            raise general.FormalProtocolError("fixture result tamper")
        role = path.stem
        for binding in expected_results[role]["prediction_shards"]:
            general.verify_file_binding(binding, deep=True)
        return expected_results[role]

    def verify_comparison(path: Path, **_kwargs: object) -> dict:
        disk = json.loads(path.read_text(encoding="utf-8"))
        if disk.get("untampered") is not True:
            raise general.FormalProtocolError("fixture comparison tamper")
        return comparison

    def verify_model(path: Path, **_kwargs: object) -> dict:
        role = path.stem
        return {"identity_sha256": expected_results[role]["model_binding_identity_sha256"]}

    monkeypatch.setattr(general, "_verify_result", verify_result)
    monkeypatch.setattr(general, "verify_comparison", verify_comparison)
    monkeypatch.setattr(general, "verify_model_binding", verify_model)
    monkeypatch.setattr(general, "verify_input_receipt", lambda *_args, **_kwargs: {"fixture": True})
    monkeypatch.setattr(general, "_runtime_rows", lambda _receipt: runtime_rows)
    return counts


def _v52_fixture(run_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = run_root / "scannet_rgb_depth_v52_paired"
    paths = {
        "result": root / "result.json",
        "release": root / "release/release.manifest.json",
        "model_binding": root / "model_binding/model.json",
        "g_predictions": root / "g/predictions.jsonl",
        "g_manifest": root / "g/predictions.manifest.json",
        "s_oracle_predictions": root / "s_oracle/predictions.jsonl",
        "s_oracle_manifest": root / "s_oracle/predictions.manifest.json",
        "s_absent_predictions": root / "s_absent/predictions.jsonl",
        "s_absent_manifest": root / "s_absent/predictions.manifest.json",
        "s_corrupted_predictions": root / "s_corrupted/predictions.jsonl",
        "s_corrupted_manifest": root / "s_corrupted/predictions.manifest.json",
        "s_pred_requests": root / "s_pred/requests.manifest.json",
        "s_pred_predictions": root / "s_pred/predictions.jsonl",
        "s_pred_manifest": root / "s_pred/predictions.manifest.json",
    }
    for key, path in paths.items():
        if key != "result":
            _write_json(path, {"fixture": key})

    pairs = []
    g_rows = []
    oracle_rows = []
    absent_rows = []
    corrupted_rows = []
    pred_rows = []
    good_g = '{"A":[100,100,300,500],"B":[500,100,800,600]}'
    for index in range(v52_eval.EXPECTED_RECORDS):
        eval_id = f"gsa-sdepth-v52-{index:020x}"
        pairs.append(
            v52_eval.PairExample(
                eval_id=eval_id,
                sample_id=f"sample-{index}",
                scene_id=f"scene{index:04d}_00",
                source_image=Path(f"/fixture/{index}.png"),
                image_sha256=f"{index + 1:064x}",
                entity_a=v52_eval.Entity("chair", "chair", (100, 100, 300, 500)),
                entity_b=v52_eval.Entity("table", "table", (500, 100, 800, 600)),
                nearer_choice="A",
                label={},
            )
        )
        g_rows.append({"eval_id": eval_id, "response": good_g, "hit_token_cap": False})
        oracle_rows.append({"eval_id": eval_id, "response": "A", "hit_token_cap": False})
        absent_rows.append({"eval_id": eval_id, "response": "B", "hit_token_cap": False})
        corrupted_rows.append({"eval_id": eval_id, "response": "B", "hit_token_cap": False})
        pred_rows.append({"eval_id": eval_id, "response": "A", "hit_token_cap": False})

    release = SimpleNamespace(
        manifest={"identity_sha256": "9" * 64},
        source=SimpleNamespace(pairs=pairs, binding={"fixture": "formal-source"}),
    )
    binding = {
        "identity_sha256": "8" * 64,
        "model": {
            "identity_sha256": "7" * 64,
            "checkpoint_content_identity_sha256": "6" * 64,
        },
    }
    receipt = {"identity_sha256": "5" * 64}

    def bundle(track: str, rows: list[dict]) -> dict:
        return {
            "rows": rows,
            "receipt": receipt,
            "provenance": {
                "track": track,
                "model_binding_identity_sha256": binding["identity_sha256"],
            },
        }

    bundles = {
        v52_eval.TRACK_G: bundle(v52_eval.TRACK_G, g_rows),
        v52_eval.TRACK_S_ORACLE: bundle(v52_eval.TRACK_S_ORACLE, oracle_rows),
        v52_eval.TRACK_S_ABSENT: bundle(v52_eval.TRACK_S_ABSENT, absent_rows),
        v52_eval.TRACK_S_CORRUPTED: bundle(v52_eval.TRACK_S_CORRUPTED, corrupted_rows),
        v52_eval.TRACK_S_PRED: bundle(v52_eval.TRACK_S_PRED, pred_rows),
    }
    monkeypatch.setattr(v52_eval, "validate_request_release", lambda *_args, **_kwargs: release)
    monkeypatch.setattr(
        compiler.v52_dlc, "verify_model_binding", lambda *_args, **_kwargs: binding
    )
    monkeypatch.setattr(
        compiler.v52_dlc,
        "verify_prediction_bundle",
        lambda *args, **kwargs: bundles[kwargs["expected_track"]],
    )
    monkeypatch.setattr(
        compiler.v52_dlc,
        "verify_predicted_request_release",
        lambda **_kwargs: {
            "manifest": {"g_prediction_manifest": v52_eval.file_binding(paths["g_manifest"])}
        },
    )

    provenance = {
        "verified": True,
        "request_release": v52_eval.file_binding(paths["release"]),
        "request_release_identity_sha256": release.manifest["identity_sha256"],
        "formal_source": release.source.binding,
        "input_receipt": receipt,
        "model_binding": v52_eval.file_binding(paths["model_binding"]),
        "model_binding_identity_sha256": binding["identity_sha256"],
        "model_inventory": binding["model"],
        "stages": {
            v52_eval.TRACK_G: bundles[v52_eval.TRACK_G]["provenance"],
            v52_eval.TRACK_S_ORACLE: bundles[v52_eval.TRACK_S_ORACLE]["provenance"],
            v52_eval.TRACK_S_ABSENT: bundles[v52_eval.TRACK_S_ABSENT]["provenance"],
            v52_eval.TRACK_S_CORRUPTED: bundles[v52_eval.TRACK_S_CORRUPTED]["provenance"],
            v52_eval.TRACK_S_PRED: bundles[v52_eval.TRACK_S_PRED]["provenance"],
        },
        "predicted_request_release": v52_eval.file_binding(paths["s_pred_requests"]),
        "same_model_binding_all_stages": True,
        "training_materialization_eligible": False,
    }
    provenance["identity_sha256"] = v52_eval.canonical_hash(provenance)
    result = v52_eval.score_paired_predictions(
        pairs,
        g_rows,
        oracle_rows,
        absent_rows,
        corrupted_rows,
        pred_rows,
        provenance=provenance,
    )
    _write_json(paths["result"], result)


def test_pending_v52_exports_verified_general_and_exact_unavailable_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, run_root = _project(tmp_path, v52_state="pending")
    counts = _general_fixture(run_root, monkeypatch)
    payload = compiler.compile_evidence(
        run_root,
        project_root=project,
        allow_test_fixture=True,
        expected_dataset_counts=counts,
    )
    assert payload["status"] == "CLAIM_EVIDENCE_VERIFIED"
    assert payload["general_retention"]["records"] == 12
    assert payload["general_retention"]["macro"]["e1_minus_base_points_ci95"]
    assert payload["scannet_v52_paired_gs"] == {
        "availability": "unavailable",
        "status_state": "SKIPPED_PENDING_ANNOTATION",
        "reason": "compile_status=PENDING_HUMAN_REVIEW",
        "status_event": {
            "timestamp_utc": "2026-08-27T01:00:02Z",
            "stage": "v52_paired",
            "state": "SKIPPED_PENDING_ANNOTATION",
            "rc": "0",
            "detail": "compile_status=PENDING_HUMAN_REVIEW",
        },
        "formal_gate_report": compiler.file_binding(run_root / "v52_formal_compile.json"),
        "formal_gate_status": "PENDING_HUMAN_REVIEW",
    }
    assert payload["publication_policy"]["console_logs_used_as_evidence"] is False


@pytest.mark.parametrize(
    "state", ["BLOCKED_NO_V52_EVALUATOR", "BLOCKED_PREFLIGHT_FAILED"]
)
def test_formal_pass_implementation_failure_cannot_be_exported_as_unavailable(
    tmp_path: Path, state: str
) -> None:
    rows = [
        compiler.StatusRow(
            "2026-08-27T01:00:00Z",
            "v52_paired",
            state,
            "0",
            "FORMAL_PASS implementation failure",
        )
    ]
    with pytest.raises(compiler.EvidenceError, match="did not complete or cleanly skip"):
        compiler._formal_v52_unavailable(tmp_path, rows)


def test_complete_v52_recomputes_exact40_and_exports_dependency_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, run_root = _project(tmp_path, v52_state="complete")
    counts = _general_fixture(run_root, monkeypatch)
    _v52_fixture(run_root, monkeypatch)
    payload = compiler.compile_evidence(
        run_root,
        project_root=project,
        allow_test_fixture=True,
        expected_dataset_counts=counts,
    )
    v52 = payload["scannet_v52_paired_gs"]
    assert v52["availability"] == "available"
    assert v52["records"] == 40
    assert v52["metrics"]["g_both_box_iou_at_0_5"] == 1.0
    assert v52["metrics"]["s_oracle_accuracy"] == 1.0
    assert v52["metrics"]["s_absent_accuracy"] == 0.0
    assert v52["metrics"]["s_corrupted_accuracy"] == 0.0
    assert v52["metrics"]["s_pred_strict_e2e_accuracy"] == 1.0
    assert v52["scene_bootstrap"]["paired_scenes"] == 40
    assert sum(v52["transition_counts"].values()) == 40
    assert v52["attribution"]["counts"]["Inconclusive"] == 40


@pytest.mark.parametrize(
    "failure", ["missing_shard", "tampered_shard", "missing_result", "tampered_result"]
)
def test_general_missing_or_tampered_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    project, run_root = _project(tmp_path)
    counts = _general_fixture(run_root, monkeypatch)
    root = run_root / "general_retention_base_e1"
    if failure == "missing_shard":
        (root / "predictions/base/shard_2.jsonl").unlink()
    elif failure == "tampered_shard":
        (root / "predictions/base/shard_2.jsonl").write_text(
            "tampered\n", encoding="utf-8"
        )
    elif failure == "missing_result":
        (root / "results/base.json").unlink()
    else:
        _write_json(root / "results/base.json", {"fixture": "base", "untampered": False})
    with pytest.raises(compiler.EvidenceError, match="General Retention deep verification failed"):
        compiler.compile_evidence(
            run_root,
            project_root=project,
            allow_test_fixture=True,
            expected_dataset_counts=counts,
        )


def test_wrong_terminal_status_fails_before_artifact_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, run_root = _project(tmp_path)
    counts = _general_fixture(run_root, monkeypatch)
    status = (run_root / "status.tsv").read_text(encoding="utf-8").replace(
        "pipeline\tCOMPLETE_WITH_BLOCKED\t0", "pipeline\tFAILED\t6"
    )
    (run_root / "status.tsv").write_text(status, encoding="utf-8")
    with pytest.raises(compiler.EvidenceError, match="terminal status"):
        compiler.compile_evidence(
            run_root,
            project_root=project,
            allow_test_fixture=True,
            expected_dataset_counts=counts,
        )


def test_wrong_code_binding_fails_before_artifact_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, run_root = _project(tmp_path)
    counts = _general_fixture(run_root, monkeypatch)
    relative = compiler.EXPECTED_TRACKED_CODE[0]
    (project / relative).write_text("post-run mutation\n", encoding="utf-8")
    with pytest.raises(compiler.EvidenceError, match="stale or tampered"):
        compiler.compile_evidence(
            run_root,
            project_root=project,
            allow_test_fixture=True,
            expected_dataset_counts=counts,
        )


def test_tampered_v52_result_fails_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, run_root = _project(tmp_path, v52_state="complete")
    counts = _general_fixture(run_root, monkeypatch)
    _v52_fixture(run_root, monkeypatch)
    path = run_root / "scannet_rgb_depth_v52_paired/result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["metrics"]["s_oracle_accuracy"] = 0.0
    _write_json(path, result)
    with pytest.raises(compiler.EvidenceError, match="does not recompute"):
        compiler.compile_evidence(
            run_root,
            project_root=project,
            allow_test_fixture=True,
            expected_dataset_counts=counts,
        )


def test_fresh_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    compiler._write_fresh(path, "first\n")
    with pytest.raises(compiler.EvidenceError, match="overwrite"):
        compiler._write_fresh(path, "second\n")
    assert path.read_text(encoding="utf-8") == "first\n"
