from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.g_family_qualification import (  # noqa: E402
    PROTOCOL,
    _load_prediction_files,
    compile_comparison,
    content_hash,
    file_hash,
    resolve_checkpoint,
    score_predictions,
    validate_release_inputs,
)
from scripts.verify_g_family_scoped_v2 import verify  # noqa: E402


CORPUS = ROOT / "trains/gsa_bench_v3_g_family_scoped_v2"
SOURCE = ROOT / "trains/gsa_bench_v3_g_family_scoped_v1"
SCOPED = ROOT / "artifacts/audits/pilot_semantic_audit_v3_g_family_scoped_v1.json"
FULL = ROOT / "artifacts/audits/pilot_semantic_audit_v3_adjudicated.json"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _bind(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_hash(path)}


def test_native_diagnostic_binds_shuffled_predictions_and_retains_failures(tmp_path: Path) -> None:
    from eval.g_family_qualification import native_diagnostic

    image = tmp_path / "image.png"
    image.write_bytes(b"image identity fixture")
    source, predicted = tmp_path / "source.jsonl", tmp_path / "predicted.jsonl"
    records, outputs = [], []
    payload = [{"label": "mug", "bbox_2d": [100, 100, 400, 500]}]
    answer = "<GROUND>" + json.dumps(payload)
    for i in range(2):
        messages = [{"role": "user", "content": f"Locate mug {i}"}]
        records.append({"sample_id": str(i), "images": [str(image)], "gsa_payload": payload,
                        "gsa_form": "bbox", "gsa_family": "G", "gsa_context": {},
                        "gsa_meta": {"image_sha256": file_hash(image)},
                        "messages": messages + [{"role": "assistant", "content": answer}]})
        outputs.append({"images": [{"path": str(image), "bytes": None}], "labels": answer,
                        "messages": [{**messages[0], "loss": None}],
                        "response": answer if i == 0 else "<GROUND>[]"})
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    predicted.write_text("".join(json.dumps(row) + "\n" for row in reversed(outputs)))
    result = native_diagnostic(source, [("candidate", predicted)])
    assert result["teacher_eligible"] is False and result["publication_claim_eligible"] is False
    form = result["models"]["candidate"]["per_form"]["bbox"]
    assert form["n"] == 2 and form["strict_contract_pass"] == 1
    assert form["reference_diagnostics"]["bbox_iou"] == 0.5
    outputs[0]["labels"] = "wrong reference"
    predicted.write_text("".join(json.dumps(row) + "\n" for row in outputs))
    with pytest.raises(ValueError, match="reference binding drift"):
        native_diagnostic(source, [("candidate", predicted)])
    outputs[0]["labels"] = answer
    outputs[0]["messages"][0]["content"] = "wrong prompt"
    predicted.write_text("".join(json.dumps(row) + "\n" for row in outputs))
    with pytest.raises(ValueError, match="coverage differs"):
        native_diagnostic(source, [("candidate", predicted)])


def test_native_action_geometry_respects_jaw_symmetry_and_zero_components() -> None:
    import copy
    import math
    from eval.g_family_qualification import native_action_reference_scores

    reference = {"label": "mug", "grasp_point_2d": [250, 300], "jaw_axis_2d": [1, 0],
                 "wrist_deg": 0, "approach": "side"}
    row = {"gsa_form": "dof", "gsa_payload": [reference]}
    flipped = {**reference, "jaw_axis_2d": [-1, 0]}
    assert native_action_reference_scores(row, [flipped], usable=True)["scores"]["single_reference_dof_score"] == 1
    offset = {**reference, "grasp_point_2d": [300, 300],
              "jaw_axis_2d": [math.cos(math.pi / 12), math.sin(math.pi / 12)], "wrist_deg": 15}
    measured = native_action_reference_scores(row, [offset], usable=True)
    assert measured["scores"]["single_reference_dof_score"] == pytest.approx(0.5946035575)
    offset["approach"] = "top"
    assert native_action_reference_scores(row, [offset], usable=True)["scores"]["single_reference_dof_score"] == 0
    assert all(value == 0 for value in native_action_reference_scores(row, [], usable=False)["scores"].values())
    affordance = {"gsa_form": "affordance", "gsa_payload": [{"label": "handle", "affordance": "grasp",
                   "bbox_2d": [100, 100, 200, 200], "contact_point_2d": [150, 150]}]}
    wrong_action = copy.deepcopy(affordance["gsa_payload"])
    wrong_action[0]["affordance"] = "pull"
    assert native_action_reference_scores(affordance, wrong_action, usable=True)["scores"]["single_reference_affordance_score"] == 0
    plan = {"gsa_form": "action_cot", "gsa_payload": [{"target_point_2d": [150, 150]}]}
    action = native_action_reference_scores(plan, [{"target_point_2d": [150, 150], "steps": ["nonsense"]}], usable=True)
    assert action["scores"]["target_point_score"] == 1 and action["plan_semantic_score"] is None


def test_native_action_diagnostic_keeps_wrong_stage_in_denominator(tmp_path: Path) -> None:
    from eval.g_family_qualification import native_diagnostic

    image = tmp_path / "image.png"
    image.write_bytes(b"bound image fixture")
    payload = [{"label": "handle", "affordance": "grasp", "bbox_2d": [100, 100, 200, 200],
                "contact_point_2d": [150, 150]}]
    answer = "<ACT>" + json.dumps(payload)
    sources, outputs = [], []
    for index in range(2):
        messages = [{"role": "user", "content": f"Locate contact region {index}"}]
        sources.append({"sample_id": str(index), "images": [str(image)], "gsa_payload": payload,
                        "gsa_form": "affordance", "gsa_family": "A", "gsa_context": {},
                        "gsa_meta": {"image_sha256": file_hash(image)},
                        "messages": messages + [{"role": "assistant", "content": answer}]})
        outputs.append({"images": [str(image)], "messages": messages, "labels": answer,
                        "response": answer if index == 0 else json.dumps(payload)})
    source, predicted = tmp_path / "source.jsonl", tmp_path / "predicted.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in sources))
    predicted.write_text("".join(json.dumps(row) + "\n" for row in outputs))
    result = native_diagnostic(source, [("candidate", predicted)])
    metrics = result["models"]["candidate"]["per_form"]["affordance"]
    assert metrics["n"] == 2
    assert metrics["stage_delimiter_correct"] == 1
    assert metrics["reference_diagnostics"]["single_reference_affordance_score"] == .5
    assert result["action_reference_protocol"]["plan_semantic_score"] is None
    assert result["publication_claim_eligible"] is False


def test_native_context_cascade_never_fills_failed_parents_from_references(tmp_path: Path) -> None:
    import copy
    from eval.g_family_qualification import native_context_rows, native_diagnostic
    from trains.flywheel.contracts import training_instruction

    image = tmp_path / "fixture.png"
    image.write_bytes(b"bound context fixture")
    groundings = [{"object_class": "mug", "bbox": [100, 100, 800, 800]},
                  {"object_class": "plate", "bbox": [500, 500, 900, 900]}]
    payloads = {
        "bbox": [{"label": "mug", "bbox_2d": [100, 100, 800, 800]}],
        "detection": [{"label": g["object_class"], "bbox_2d": g["bbox"]} for g in groundings],
        "depth": [{"label": "mug", "rank": 1, "relative_depth": 0}, {"label": "plate", "rank": 2, "relative_depth": 1}],
        "affordance": [{"label": "handle", "affordance": "grasp", "bbox_2d": [150, 200, 250, 300], "contact_point_2d": [200, 250]}],
        "dof": [{"label": "mug", "grasp_point_2d": [200, 250], "jaw_axis_2d": [1, 0], "approach": "side", "wrist_deg": 0}],
        "action_cot": [{"target_point_2d": [200, 250]}],
    }
    context = {"contract_version": "gsa-contract-v2", "primary": groundings[0], "groundings": groundings,
               "spatial": {"depth_order": payloads["depth"], "affordance": payloads["affordance"], "dof": payloads["dof"]}}
    sources, generated = [], {}
    for form, payload in payloads.items():
        family = "S" if form == "depth" else "A" if form in ("affordance", "dof", "action_cot") else "G"
        prefix = {"G": "<GROUND>", "S": "<RELATE>", "A": "<ACT>"}[family]
        messages = [{"role": "user", "content": "<image>" + training_instruction(form, context).replace(
                    "Return only a JSON array;", f"Return {prefix} followed by a JSON array;")},
                    {"role": "assistant", "content": prefix + json.dumps(payload)}]
        sources.append({"sample_id": form, "gsa_form": form, "gsa_family": family, "messages": messages,
                        "images": [str(image)], "gsa_context": context, "gsa_payload": payload,
                        "gsa_meta": {"image_sha256": file_hash(image)}})
        generated[form] = {"response": prefix + json.dumps(payload)}
    before = copy.deepcopy(sources)
    oracle = native_context_rows(sources, "oracle")
    assert len(oracle) == 4 and not any(row["gsa_context_intervention"]["visible_input_changed"] for row in oracle)
    student = native_context_rows(sources, "student_cascade", generated)
    dof = next(row for row in student if row["gsa_form"] == "dof")
    assert "dof" not in dof["gsa_provided_context"]["spatial"]
    generated["dof"]["response"] = "<ACT>not-json"
    assert next(row for row in native_context_rows(sources, "student_cascade", generated)
                if row["gsa_form"] == "dof")["messages"] == dof["messages"]
    corrupted = native_context_rows(sources, "corrupted")
    assert all(row["gsa_context_intervention"]["visible_input_changed"] for row in corrupted)
    assert [row["gsa_payload"] for row in corrupted] == [row["gsa_payload"] for row in oracle]
    generated["detection"]["finish_reason"] = "length"
    capped = native_context_rows(sources, "student_cascade", generated)
    assert all(any(reason["reason"] == "parent_token_cap" for reason in row["gsa_eval_blocked_upstream"]) for row in capped)
    del generated["detection"]["finish_reason"]
    generated["detection"]["response"] = "<GROUND>not-json"
    blocked = native_context_rows(sources, "student_cascade", generated)
    assert len(blocked) == 4 and all(row["gsa_eval_blocked_upstream"] for row in blocked)
    assert all(row["gsa_provided_context"] is None for row in blocked)
    source, predicted = tmp_path / "blocked.jsonl", tmp_path / "empty.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in blocked))
    predicted.write_text("")
    result = native_diagnostic(source, [("candidate", predicted)])
    assert result["models"]["candidate"]["coverage"] == 0
    assert len(result["models"]["candidate"]["items"]) == 4
    for form in result["models"]["candidate"]["per_form"].values():
        assert form["n"] == 1 and form["blocked_upstream"] == 1
        assert all(value == 0 for value in form["reference_diagnostics"].values())
    assert sources == before


def _fake_run(
    tmp_path: Path,
    *,
    trainer_includes_qualification: bool = False,
    world_size: int = 8,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    run = tmp_path / "run"
    checkpoint = run / "v0-fixture/checkpoint-53"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base.resolve())}) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture-adapter")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 53}) + "\n", encoding="utf-8"
    )
    manifest = CORPUS / "manifest.json"
    train = CORPUS / "train_G_only.jsonl"
    dev = CORPUS / "dev_G_only.jsonl"
    qualification = CORPUS / "qualification_G_only.jsonl"
    trainer_paths = [str(train.resolve()), str(dev.resolve())]
    if trainer_includes_qualification:
        trainer_paths.append(str(qualification.resolve()))
    bs, ga = (1, 2) if world_size == 4 else (1, 1)
    trainer_args = {
        "local_world_size": world_size,
        "per_device_train_batch_size": bs,
        "gradient_accumulation_steps": ga,
        "seed": seed,
        "data_seed": seed,
        "max_steps": 53,
        "learning_rate": 1e-5,
        "max_length": 3072,
        "dataset": [str(train.resolve())],
        "val_dataset": [str(dev.resolve())],
        "dataloader_num_workers": 4 if world_size == 4 else 12,
        "dataset_num_proc": 4 if world_size == 4 else 24,
        "deepspeed": {"zero_optimization": {"stage": 2}},
    }
    (checkpoint.parent / "args.json").write_text(
        json.dumps(trainer_args, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "protocol": "semantic-audit-v3-g-family-scoped-v2-independent-qualification",
        "authorization": "G_specialist_qualification_only",
        "global_audit_verdict": "BLOCKED",
        "manifest": _bind(manifest),
        "scoped_report": _bind(SCOPED),
        "full_report": _bind(FULL),
        "model_config": _bind(base / "config.json"),
        "train": _bind(train), "dev": _bind(dev), "qualification": _bind(qualification),
        "train_rows": 84, "dev_rows": 32, "qualification_rows": 32,
        "hardware": f"{world_size}xL20", "world_size": world_size,
        "per_device_batch_size": bs, "gradient_accumulation_steps": ga,
        "effective_batch_size": world_size * bs * ga,
        "epochs_for_step_calculation": 5, "max_steps": 53,
        "seed": seed, "data_seed": seed, "deepspeed": "zero2",
        "dataloader_num_workers": trainer_args["dataloader_num_workers"],
        "dataset_num_proc": trainer_args["dataset_num_proc"],
        "wandb_mode": "offline" if world_size == 4 else "online",
        "qualification_exposed_to_trainer": trainer_includes_qualification,
        "trainer_dataset_paths": trainer_paths,
    }
    (run / "g_family_qualification_inputs_v2.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return run, checkpoint, base


def test_v2_partition_replays_and_preserves_independent_qualification() -> None:
    result = verify(
        manifest_path=CORPUS / "manifest.json",
        source=SOURCE,
        scoped_report=SCOPED,
        full_report=FULL,
    )
    assert result["counts"] == {"train": 84, "dev": 32, "qualification": 32}
    assert result["unique_images"] == {"train": 24, "dev": 8, "qualification": 8}
    assert result["qualification_exposed_to_trainer"] is False


def test_cpu_preflight_binds_exact_v2_corpus_audit_and_checkpoint(tmp_path: Path) -> None:
    run, checkpoint, base = _fake_run(tmp_path)
    receipt = validate_release_inputs(
        manifest_path=CORPUS / "manifest.json",
        scoped_report_path=SCOPED,
        full_report_path=FULL,
        base_model=base,
        checkpoint=checkpoint,
        source=run,
    )
    assert receipt["data"]["train_rows"] == 84
    assert receipt["data"]["dev_rows"] == 32
    assert receipt["data"]["qualification_rows"] == 32
    assert receipt["data"]["qualification_images"] == 8
    assert receipt["data"]["used_during_training_for_validation_and_model_selection"] is False
    assert receipt["publication_claim_eligible"] is False
    assert receipt["teacher_authorization"] == "NONE"


def test_cpu_preflight_accepts_matched_4gpu_topology(tmp_path: Path) -> None:
    run, checkpoint, base = _fake_run(tmp_path, world_size=4, seed=43)
    receipt = validate_release_inputs(
        manifest_path=CORPUS / "manifest.json",
        scoped_report_path=SCOPED,
        full_report_path=FULL,
        base_model=base,
        checkpoint=checkpoint,
        source=run,
        eval_world_size=4,
        batch_size=4,
    )
    assert receipt["training"] == {
        "hardware": "4xL20",
        "world_size": 4,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 8,
        "epochs": 5,
        "max_steps": 53,
        "seed": 43,
        "seed_binding": "receipt_and_args",
        "deepspeed": "zero2",
    }
    assert receipt["generation"]["eval_world_size"] == 4
    assert receipt["generation"]["num_shards"] == 8
    assert receipt["generation"]["max_concurrent_workers"] == 4
    assert receipt["generation"]["batch_size"] == 4


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("receipt", "gradient_accumulation_steps", 1, "effective batch drift"),
        ("receipt", "seed", 44, "seed differs"),
        ("args", "local_world_size", 8, "world size differs"),
        ("args", "data_seed", 44, "seed/data_seed drift"),
        ("args", "max_steps", 52, "max steps drift"),
        ("args", "dataset", [], "dataset paths"),
    ],
)
def test_cpu_preflight_rejects_receipt_or_trainer_args_drift(
    tmp_path: Path, target: str, field: str, value: object, message: str
) -> None:
    run, checkpoint, base = _fake_run(tmp_path, world_size=4)
    path = run / "g_family_qualification_inputs_v2.json" if target == "receipt" else checkpoint.parent / "args.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_release_inputs(
            manifest_path=CORPUS / "manifest.json",
            scoped_report_path=SCOPED,
            full_report_path=FULL,
            base_model=base,
            checkpoint=checkpoint,
            source=run,
            eval_world_size=4,
        )


def test_cpu_preflight_rejects_non_zero2_trainer_args(tmp_path: Path) -> None:
    run, checkpoint, base = _fake_run(tmp_path, world_size=4)
    path = checkpoint.parent / "args.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["deepspeed"]["zero_optimization"]["stage"] = 3
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ZeRO-2"):
        validate_release_inputs(
            manifest_path=CORPUS / "manifest.json",
            scoped_report_path=SCOPED,
            full_report_path=FULL,
            base_model=base,
            checkpoint=checkpoint,
            source=run,
            eval_world_size=4,
        )


def test_cpu_preflight_rejects_qualification_exposure(tmp_path: Path) -> None:
    run, checkpoint, base = _fake_run(tmp_path, trainer_includes_qualification=True)
    with pytest.raises(ValueError, match="exposed qualification"):
        validate_release_inputs(
            manifest_path=CORPUS / "manifest.json",
            scoped_report_path=SCOPED,
            full_report_path=FULL,
            base_model=base,
            checkpoint=checkpoint,
            source=run,
        )


def test_eval_launcher_cpu_preflight_stops_before_merge_or_gpu(tmp_path: Path) -> None:
    run, _, base = _fake_run(tmp_path)
    out = tmp_path / "eval"
    env = {
        **os.environ,
        "PREFLIGHT_ONLY": "1",
        "G_SOURCE": str(run),
        "MODEL": str(base),
        "OUT": str(out),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_g_family_qualification_eval_l20.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "G_V2_EVAL_PREFLIGHT_READY: no model merge or GPU process was started" in result.stdout
    assert (out / "g_family_qualification_eval_inputs_v2.json").is_file()
    assert not (out / "model_bindings/base.json").exists()


def test_4gpu_eval_launcher_cpu_preflight_binds_four_workers_eight_shards(tmp_path: Path) -> None:
    run, _, base = _fake_run(tmp_path, world_size=4)
    out = tmp_path / "eval4"
    env = {
        **os.environ,
        "PREFLIGHT_ONLY": "1",
        "G_SOURCE": str(run),
        "MODEL": str(base),
        "OUT": str(out),
        "SEED": "42",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_g_family_qualification_eval_l20_4gpu.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "G_V2_EVAL_PREFLIGHT_READY: no model merge or GPU process was started" in result.stdout
    receipt = json.loads((out / "g_family_qualification_eval_inputs_v2.json").read_text())
    assert receipt["generation"]["hardware"] == "exactly_4xL20"
    assert receipt["generation"]["num_shards"] == 8
    assert receipt["generation"]["max_concurrent_workers"] == 4


def test_4gpu_train_and_queue_cpu_preflight_never_touch_cuda(tmp_path: Path) -> None:
    env = {**os.environ, "PREFLIGHT_ONLY": "1", "SEED": "42", "OUT": str(tmp_path / "train")}
    direct = subprocess.run(
        ["bash", str(ROOT / "scripts/run_g_family_qualification_l20_4gpu_v2.sh")],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    assert direct.returncode == 0, direct.stderr
    assert "G_V2_4GPU_TRAIN_PREFLIGHT_READY: no GPU process was started" in direct.stdout
    assert not (tmp_path / "train").exists()
    queued = subprocess.run(
        ["bash", str(ROOT / "scripts/run_g_family_qualification_4gpu_queue.sh")],
        cwd=ROOT,
        env={**os.environ, "PREFLIGHT_ONLY": "1", "SEED": "42", "TRAIN_OUT": str(tmp_path / "queued")},
        text=True, capture_output=True, check=False,
    )
    assert queued.returncode == 0, queued.stderr
    assert "G_V2_4GPU_QUEUE_PREFLIGHT_READY" in queued.stdout
    assert not (tmp_path / "queued").exists()


def test_latest_checkpoint_is_numeric_and_ambiguous_ties_fail(tmp_path: Path) -> None:
    for branch, step in (("v0-a", 2), ("v0-a", 10)):
        checkpoint = tmp_path / branch / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "adapter_config.json").write_text("{}")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"x")
    assert resolve_checkpoint(tmp_path).name == "checkpoint-10"
    tied = tmp_path / "v0-b/checkpoint-10"
    tied.mkdir(parents=True)
    (tied / "adapter_config.json").write_text("{}")
    (tied / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="ambiguous latest checkpoint-10"):
        resolve_checkpoint(tmp_path)


def test_task_metrics_score_perfect_structured_predictions() -> None:
    rows = _rows(CORPUS / "qualification_G_only.jsonl")[:4]
    assert [row["gsa_form"] for row in rows] == ["bbox", "detection", "mask", "point"]
    predictions = {
        row["sample_id"]: {
            "response": row["messages"][-1]["content"],
            "hit_token_cap": False,
        }
        for row in rows
    }
    result = score_predictions(rows, predictions)
    assert result["macro_primary"] == 1.0
    assert result["forms"]["bbox"]["mean_iou"] == 1.0
    assert result["forms"]["detection"]["label_aware_f1_at_0_5"] == 1.0
    assert result["forms"]["mask"]["mean_polygon_iou"] == 1.0
    assert result["forms"]["point"]["inside_primary_bbox"] == 1.0
    assert result["trust"]["status"] == "TRUSTED"


def _prediction_shards(tmp_path: Path) -> tuple[Path, dict, list[Path]]:
    qualification = CORPUS / "qualification_G_only.jsonl"
    rows = _rows(qualification)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({
            "generation": {"num_shards": 8},
            "data": {"qualification": _bind(qualification)},
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    binding = {"identity_sha256": "model-id", "role": "specialist"}
    receipt_sha = file_hash(receipt)
    paths: list[Path] = []
    for shard in range(8):
        path = tmp_path / f"shard_{shard}.jsonl"
        records = []
        for index in range(shard, len(rows), 8):
            row = rows[index]
            records.append({
                "protocol": PROTOCOL,
                "sample_index": index,
                "sample_id": row["sample_id"],
                "form": row["gsa_form"],
                "image_sha256": row["gsa_meta"]["image_sha256"],
                "response": row["messages"][-1]["content"],
                "hit_token_cap": False,
                "model_role": "specialist",
                "model_identity_sha256": "model-id",
                "input_receipt_sha256": receipt_sha,
                "shard": shard,
                "num_shards": 8,
            })
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        paths.append(path)
    return receipt, binding, paths


def test_prediction_loader_replays_all_eight_logical_shards(tmp_path: Path) -> None:
    receipt, binding, paths = _prediction_shards(tmp_path)
    predictions = _load_prediction_files(paths, receipt, binding)
    assert len(predictions) == 32


@pytest.mark.parametrize("mutation", ("wrong_modulo", "wrong_form", "wrong_num_shards"))
def test_prediction_loader_rejects_row_provenance_drift(tmp_path: Path, mutation: str) -> None:
    receipt, binding, paths = _prediction_shards(tmp_path)
    path = paths[0]
    records = _rows(path)
    if mutation == "wrong_modulo":
        records[0]["sample_index"] = 1
    elif mutation == "wrong_form":
        records[0]["form"] = "point"
    else:
        records[0]["num_shards"] = 4
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="shard|provenance"):
        _load_prediction_files(paths, receipt, binding)


def _result(role: str, primary: float) -> dict:
    samples = []
    for image in range(8):
        for form in ("bbox", "detection", "mask", "point"):
            samples.append({
                "sample_id": f"{form}-{image}",
                "image_sha256": f"image-{image}",
                "form": form,
                "primary": primary,
            })
    result = {
        "protocol": PROTOCOL,
        "model_role": role,
        "input_receipt": {"path": "/fixture/receipt", "sha256": "same"},
        "n_records": 32,
        "n_unique_images": 8,
        "macro_primary": primary,
        "forms": {form: {"primary_value": primary} for form in ("bbox", "detection", "mask", "point")},
        "trust": {"status": "TRUSTED"},
        "samples": samples,
    }
    result["identity_sha256"] = content_hash(result)
    return result


def test_directional_comparison_never_becomes_teacher_or_publication_pass(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    specialist_path = tmp_path / "specialist.json"
    base_path.write_text(json.dumps(_result("base", 0.0)))
    specialist_path.write_text(json.dumps(_result("specialist", 1.0)))
    report = compile_comparison(base_path, specialist_path)
    assert report["engineering_decision"] == "G_PILOT_DIRECTIONAL_CONTINUE"
    assert report["evidence_status"] == "PILOT_ONLY_PUBLICATION_CLAIM_WITHHELD"
    assert report["publication_claim_eligible"] is False
    assert report["teacher_authorization"] == "NONE"
    assert report["global_teacher_eligibility_status"] == "NOT_EVALUATED"
    assert report["joint_training_authorized"] is False
    assert report["mopd_training_authorized"] is False


def test_v2_training_launcher_passes_only_train_and_dev_to_swift() -> None:
    text = (ROOT / "scripts/run_g_family_qualification_l20_v2.sh").read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert 'TRAIN="$TRAIN" VAL="$DEV"' in executable
    assert 'VAL="$QUALIFICATION"' not in executable
    assert '"qualification_exposed_to_trainer": False' in text
    assert "joint_training_authorized" in text
    assert "mopd_training_authorized" in text


def test_4gpu_launchers_freeze_matched_batch_seed_and_two_wave_eval() -> None:
    train = (ROOT / "scripts/run_g_family_qualification_l20_4gpu_v2.sh").read_text(encoding="utf-8")
    evaluator = (ROOT / "scripts/run_g_family_qualification_eval_l20.sh").read_text(encoding="utf-8")
    thin_eval = (ROOT / "scripts/run_g_family_qualification_eval_l20_4gpu.sh").read_text(encoding="utf-8")
    generic = (ROOT / "scripts/run_sft_9b_lora_l20_dlc.sh").read_text(encoding="utf-8")
    executable = "\n".join(line for line in train.splitlines() if not line.lstrip().startswith("#"))
    assert 'WORLD_SIZE=4' in train
    assert 'BS="${BS:-1}"' in train and 'GA="${GA:-2}"' in train
    assert '"effective_batch_size": 4 * bs * ga' in train
    assert '"seed": seed, "data_seed": seed' in train
    assert '"wandb_mode": "offline"' in train
    assert 'TRAIN="$TRAIN" VAL="$DEV"' in executable
    assert 'VAL="$QUALIFICATION"' not in executable
    assert '--seed "$SEED" --data_seed "$SEED"' in generic
    assert 'DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"' in generic
    assert 'export GSA_EVAL_WORLD_SIZE=4' in thin_eval
    assert 'NUM_SHARDS=8' in evaluator
    assert 'wave_start += EVAL_WORLD_SIZE' in evaluator
    assert 'CUDA_VISIBLE_DEVICES="${GPU_IDS[$slot]}"' in evaluator


def test_native_depth_context_reports_entity_loss_without_dropping_cases():
    from copy import deepcopy
    from eval.compare_models import native_depth_context_availability

    reference = [{"label": "mug", "rank": 1}, {"label": "plate", "rank": 2}]
    context = {"groundings": [{"object_class": "mug"}, {"object_class": "plate"}]}
    source = [{"sample_id": str(i), "gsa_form": "depth", "gsa_payload": reference,
               "gsa_context": context, "gsa_provided_context": deepcopy(context)} for i in range(4)]
    source[1]["gsa_provided_context"]["groundings"][1]["object_class"] = "bowl"
    source[2]["gsa_provided_context"]["groundings"].append({"object_class": "plate"})
    source[3]["gsa_provided_context"] = None
    source[3]["gsa_eval_blocked_upstream"] = ["G"]
    result = native_depth_context_availability(source)
    assert result["n"] == 4 and result["exact_entity_set_available"] == 1
    assert result["changes_scoring_denominator"] is False
    assert result["items"][1]["missing_reference_labels"] == ["plate"]
    assert result["items"][1]["extra_provided_labels"] == ["bowl"]
    assert not result["items"][2]["provided_labels_unique_and_nonempty"]
    assert result["items"][3]["blocked_upstream"]


def test_native_paired_context_difference_preserves_blocked_denominator():
    from copy import deepcopy
    from eval.compare_models import native_paired_difference
    source = [{"sample_id": f"sample{i}", "gsa_form": "depth", "gsa_payload": [{"rank": 1}],
               "gsa_meta": {"image_sha256": str(i) * 64}} for i in range(2)]
    left = {"items": [{"sample_id": row["sample_id"], "form": "depth",
                       "depth_pairs": {"correct": 1, "total": 1}} for row in source]}
    changed = deepcopy(source)
    right = deepcopy(left)
    for row, item in zip(changed, right["items"]):
        row["gsa_eval_original_sample_id"] = row["sample_id"]
        row["sample_id"] += "::student_cascade"
        item["sample_id"] = row["sample_id"]
    right["items"][0].update(generation_status="BLOCKED_UPSTREAM", depth_pairs={"correct": 0, "total": 1})
    result = native_paired_difference(source, left, changed, right)
    depth = result["per_form"]["depth"]
    assert depth["n"] == depth["n_independent_images"] == 2
    assert depth["left_percent"] == 100 and depth["right_percent"] == 50 and depth["delta_pp"] == -50
    assert depth["regressed"] == 1 and depth["unchanged"] == 1
    assert result["A_family_semantic_macro"] is None and result["significance"] is None
    assert result["publication_inference_allowed"] is False
    right["items"][0]["depth_pairs"]["correct"] = 1
    with pytest.raises(ValueError, match="blocked native case"):
        native_paired_difference(source, left, changed, right)


def test_native_paired_comparison_rejects_missing_or_changed_targets():
    from copy import deepcopy
    from eval.compare_models import native_paired_difference
    source = [{"sample_id": f"sample{i}", "gsa_form": "bbox", "gsa_payload": [{"x": 1}],
               "gsa_meta": {"image_sha256": "a" * 64}} for i in range(2)]
    model = {"items": [{"sample_id": row["sample_id"], "form": "bbox", "g_reference_scores": {"bbox_iou": 0.5}}
                       for row in source]}
    assert native_paired_difference(source, model, source, model)["per_form"]["bbox"]["n_independent_images"] == 1
    incomplete = {"items": model["items"][:1]}
    with pytest.raises(ValueError, match="coverage mismatch"):
        native_paired_difference(source, model, source, incomplete)
    changed = deepcopy(source)
    changed[0]["gsa_payload"][0]["x"] = 2
    with pytest.raises(ValueError, match="reference payload drift"):
        native_paired_difference(source, model, changed, model)
    changed = deepcopy(source)
    changed[0]["gsa_meta"]["image_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="coverage or reference payload drift"):
        native_paired_difference(source, model, changed, model)


def test_native_refresh_comparison_does_not_admit_incomplete_receipts(tmp_path):
    from eval.compare_models import compile_native_refresh
    run = tmp_path / "G_control_r1"
    run.mkdir()
    (run / "launch.json").write_text(json.dumps({"status": "TRAINED_EVALUATING"}))
    result = compile_native_refresh(tmp_path)
    assert result["roles"]["G"]["receipt_statuses"]["control"] == "TRAINED_EVALUATING"
    assert result["roles"]["G"]["comparisons"] == {}
    assert result["roles"]["S"]["receipt_statuses"]["control"] == "MISSING_RECEIPT"
    assert result["publication_claim_eligible"] is False and result["teacher_eligible"] is False


def test_native_interval_keeps_all_forms_in_the_same_image_draw():
    from eval.compare_models import native_cluster_interval
    rows = [{"cluster_id": "image:" + image, "form": form, "delta_pp": delta}
            for image, delta in (("a", -100.0), ("b", 100.0))
            for form in ("bbox", "detection", "mask", "point")]
    result = native_cluster_interval(rows)
    assert result["n_clusters"] == 2 and result["n_rows"] == 8
    assert result["observed_delta_pp"] == 0 and result["ci_pp"] == [-100, 100]
    assert not result["training_randomness_covered"] and not result["publication_inference_allowed"]
    # Extra repeated outputs must not increase the effective sample size or
    # narrow the interval when they carry no new images/scenes.
    repeated = native_cluster_interval(rows * 4)
    assert repeated["n_clusters"] == 2 and repeated["ci_pp"] == result["ci_pp"]


def test_native_interval_recomputes_macro_and_does_not_condition_on_coverage():
    from eval.compare_models import native_cluster_interval
    rows = [{"cluster_id": "image:a", "form": "bbox", "delta_pp": 0.0},
            {"cluster_id": "image:b", "form": "bbox", "delta_pp": 100.0},
            {"cluster_id": "image:b", "form": "point", "delta_pp": 100.0}]
    rows += [{"cluster_id": "image:a", "form": "point", "delta_pp": 0.0}] * 10
    result = native_cluster_interval(rows)
    assert result["observed_delta_pp"] == pytest.approx((50 + 100 / 11) / 2)
    disjoint = native_cluster_interval([rows[0], rows[2]])
    assert disjoint["status"] == "RESAMPLES_MISSING_FORM_COVERAGE"
    assert disjoint["replicates_missing_form_coverage"] > 0 and disjoint["ci_pp"] is None


def test_native_comparison_clusters_repeated_physical_scenes():
    from copy import deepcopy
    from eval.compare_models import native_paired_difference
    source = [{"sample_id": str(i), "gsa_form": "bbox", "gsa_payload": [],
               "gsa_meta": {"image_sha256": str(i) * 64, "source": "sensor", "scene_group": "scene1"}}
              for i in range(2)]
    model = {"items": [{"sample_id": row["sample_id"], "form": "bbox",
                        "g_reference_scores": {"bbox_iou": 0.5}} for row in source]}
    result = native_paired_difference(source, model, source, model)["per_form"]["bbox"]
    assert result["n_independent_images"] == 2
    interval = result["paired_cluster_interval"]
    assert interval["n_clusters"] == 1 and interval["status"] == "INSUFFICIENT_CLUSTERS"
    assert interval["ci_pp"] is None
    changed = deepcopy(source)
    changed[0]["gsa_meta"]["scene_group"] = "scene2"
    with pytest.raises(ValueError, match="reference payload drift"):
        native_paired_difference(source, model, changed, model)


def test_native_specialist_joint_positive_pilot_does_not_imply_qualification():
    from eval.compare_models import native_specialist_joint_comparisons
    rows = [{"sample_id": str(i), "gsa_form": "depth", "gsa_payload": [{"rank": 1}],
             "gsa_meta": {"image_sha256": str(i) * 64}} for i in range(4)]
    contexts = {name: rows for name in ("oracle", "student_cascade", "corrupted")}
    models = {name: {variant: {"items": [{"sample_id": row["sample_id"], "form": "depth",
                                         "depth_pairs": {"correct": correct, "total": 1}} for row in rows]}
                     for variant in contexts} for name, correct in (("joint_feedback", 0), ("S_feedback", 1))}
    env = {"RANK": "64", "ALPHA": "128", "DROPOUT": "0.05"}
    roles = {role: {"budget": {"feedback": {"num_input_tokens_seen": count, "environment": env}}}
             for role, count in (("joint", 1000), ("S", 1400))}
    result = native_specialist_joint_comparisons(contexts, models, roles)["S_feedback_minus_joint_feedback"]
    assert result["all_three_exploratory_lower_bounds_positive"]
    assert result["refresh_budget_observation"]["specialist_minus_joint_input_tokens"] == 400
    assert not result["refresh_budget_observation"]["exact_input_token_counts_equal"]
    assert not result["teacher_eligible"] and not result["publication_claim_eligible"]
    # Initial-only arms have no recorded cumulative training-token budget here.
    initial = {name.replace("feedback", "initial"): value for name, value in models.items()}
    pending = native_specialist_joint_comparisons(contexts, initial, roles)["S_initial_minus_joint_initial"]
    assert not pending["refresh_budget_observation"]["token_counts_available"]


def test_native_code_snapshot_requires_exact_original_bytes_and_cannot_replace_data(tmp_path):
    import hashlib
    from eval.compare_models import native_code_snapshot
    snapshot = tmp_path / "anvil.py"
    original = b"# frozen original implementation\n"
    snapshot.write_bytes(original)
    record = {"path": "/changed/worktree/anvil.py", "sha256": hashlib.sha256(original).hexdigest()}
    assert native_code_snapshot(record, (snapshot,)) == snapshot.resolve()
    with pytest.raises(ValueError, match="exact source snapshot required"):
        native_code_snapshot(record, ())
    snapshot.write_bytes(b"# a different implementation\n")
    with pytest.raises(ValueError, match="exact source snapshot required"):
        native_code_snapshot(record, (snapshot,))
    with pytest.raises(ValueError, match="cannot replace data"):
        native_code_snapshot({**record, "path": "/changed/predictions.jsonl"}, (snapshot,))
