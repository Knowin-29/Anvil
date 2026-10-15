from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import scannet_rgb_depth_v52_dlc as dlc  # noqa: E402
from eval import scannet_rgb_depth_v52_eval as depth  # noqa: E402


def _fake_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fixture_vl",
                "architectures": ["FixtureVLForConditionalGeneration"],
                "vision_config": {"model_type": "fixture_vision"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"fixture-merged-weight-bytes")
    (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    return model


def _synthetic_release(tmp_path: Path) -> depth.RequestRelease:
    images = tmp_path / "release/images"
    images.mkdir(parents=True)
    pairs: list[depth.PairExample] = []
    for index in range(depth.EXPECTED_RECORDS):
        eval_id = f"gsa-sdepth-v52-{index:020x}"
        image = images / f"{eval_id}.png"
        image.write_bytes(f"fixture-image-{index}".encode("ascii"))
        image_sha = depth.file_sha256(image)
        label = {
            "sample_id": f"synthetic-{index}",
            "image_sha256": image_sha,
            "public_interface": {
                "grounding_targets": [
                    {
                        "slot": "A",
                        "category": "chair",
                        "referring_expression": "the only visible chair",
                    },
                    {
                        "slot": "B",
                        "category": "table",
                        "referring_expression": "the only visible table",
                    },
                ]
            },
        }
        pairs.append(
            depth.PairExample(
                eval_id=eval_id,
                sample_id=f"synthetic-{index}",
                scene_id=f"scene{index:04d}_00",
                source_image=image,
                image_sha256=image_sha,
                entity_a=depth.Entity(
                    "chair", "the only visible chair", (100, 100, 300, 500)
                ),
                entity_b=depth.Entity(
                    "table", "the only visible table", (500, 100, 800, 600)
                ),
                nearer_choice="A",
                label=label,
            )
        )
    g_rows = [depth.format_g_request(pair, pair.source_image) for pair in pairs]
    oracle_rows = [
        depth.format_s_request(
            pair,
            pair.source_image,
            track=depth.TRACK_S_ORACLE,
            bbox_a=pair.entity_a.bbox,
            bbox_b=pair.entity_b.bbox,
        )
        for pair in pairs
    ]
    absent_rows = [
        depth.format_s_request(pair, pair.source_image, track=depth.TRACK_S_ABSENT)
        for pair in pairs
    ]
    corrupted_rows = [
        depth.format_corrupted_s_request(pair, pair.source_image) for pair in pairs
    ]
    g_path = tmp_path / "release/requests.g.jsonl"
    oracle_path = tmp_path / "release/requests.s_oracle.jsonl"
    absent_path = tmp_path / "release/requests.s_absent.jsonl"
    corrupted_path = tmp_path / "release/requests.s_corrupted.jsonl"
    depth.write_jsonl(g_path, g_rows)
    depth.write_jsonl(oracle_path, oracle_rows)
    depth.write_jsonl(absent_path, absent_rows)
    depth.write_jsonl(corrupted_path, corrupted_rows)
    formal_source = {"fixture": True, "records": depth.EXPECTED_RECORDS}
    manifest = {
        "identity_sha256": "f" * 64,
        "requests": {
            depth.TRACK_G: {
                "file": dlc.file_binding(g_path),
                "records_sha256": depth.canonical_hash(g_rows),
                "records": depth.EXPECTED_RECORDS,
            },
            depth.TRACK_S_ORACLE: {
                "file": dlc.file_binding(oracle_path),
                "records_sha256": depth.canonical_hash(oracle_rows),
                "records": depth.EXPECTED_RECORDS,
            },
            depth.TRACK_S_ABSENT: {
                "file": dlc.file_binding(absent_path),
                "records_sha256": depth.canonical_hash(absent_rows),
                "records": depth.EXPECTED_RECORDS,
            },
            depth.TRACK_S_CORRUPTED: {
                "file": dlc.file_binding(corrupted_path),
                "records_sha256": depth.canonical_hash(corrupted_rows),
                "records": depth.EXPECTED_RECORDS,
            },
        },
    }
    depth.write_json(tmp_path / "release/release.manifest.json", manifest)
    return depth.RequestRelease(manifest, depth.FormalSource(tuple(pairs), formal_source), {
        depth.TRACK_G: g_rows,
        depth.TRACK_S_ORACLE: oracle_rows,
        depth.TRACK_S_ABSENT: absent_rows,
        depth.TRACK_S_CORRUPTED: corrupted_rows,
    })


def _complete_stage(
    tmp_path: Path,
    receipt: Path,
    binding: Path,
    *,
    alias: str,
    track: str,
    response: str,
    dynamic_manifest: Path | None = None,
) -> tuple[Path, Path, list[Path]]:
    stage = tmp_path / alias
    shards: list[Path] = []

    def fake(public: list[dict], max_tokens: int) -> list[tuple[str, bool]]:
        assert max_tokens == dlc.STAGE_MAX_TOKENS[track]
        assert all(set(item) == {"messages", "images"} for item in public)
        serialized = json.dumps(public).casefold()
        assert "[instance" not in serialized
        assert "scannet-sdepth" not in serialized
        return [(response, False) for _ in public]

    for shard in range(dlc.NUM_SHARDS):
        path = stage / f"shard_{shard}.jsonl"
        generated = dlc.infer_shard(
            receipt_path=receipt,
            binding_path=binding,
            track=track,
            shard=shard,
            num_shards=dlc.NUM_SHARDS,
            batch_size=dlc.BATCH_SIZE,
            max_tokens=dlc.STAGE_MAX_TOKENS[track],
            out=path,
            dynamic_request_manifest=dynamic_manifest,
            infer_public=fake,
        )
        assert generated["status"] == "generated"
        reused = dlc.infer_shard(
            receipt_path=receipt,
            binding_path=binding,
            track=track,
            shard=shard,
            num_shards=dlc.NUM_SHARDS,
            batch_size=dlc.BATCH_SIZE,
            max_tokens=dlc.STAGE_MAX_TOKENS[track],
            out=path,
            dynamic_request_manifest=dynamic_manifest,
            infer_public=fake,
        )
        assert reused["status"] == "reused"
        shards.append(path)
    merged = stage / "predictions.jsonl"
    manifest = stage / "predictions.manifest.json"
    dlc.merge_shards(
        receipt_path=receipt,
        binding_path=binding,
        track=track,
        shard_paths=shards,
        out=merged,
        manifest_out=manifest,
        dynamic_request_manifest=dynamic_manifest,
    )
    return merged, manifest, shards


def test_strict_parser_invalid_g_and_cap_stay_in_all_scene_denominator() -> None:
    pairs: list[depth.PairExample] = []
    for index in range(4):
        pairs.append(
            depth.PairExample(
                eval_id=f"gsa-sdepth-v52-{index:020x}",
                sample_id=f"sample-{index}",
                scene_id=f"scene{index:04d}_00",
                source_image=Path(f"/private/{index}.png"),
                image_sha256=f"{index + 1:064x}",
                entity_a=depth.Entity("chair", "only chair", (100, 100, 300, 500)),
                entity_b=depth.Entity("table", "only table", (500, 100, 800, 600)),
                nearer_choice="A",
                label={},
            )
        )
    good = '{"A":[100,100,300,500],"B":[500,100,800,600]}'
    g = [
        {"eval_id": pairs[0].eval_id, "response": good, "hit_token_cap": False},
        {"eval_id": pairs[1].eval_id, "response": good, "hit_token_cap": False},
        {"eval_id": pairs[2].eval_id, "response": "invalid", "hit_token_cap": False},
        {"eval_id": pairs[3].eval_id, "response": good, "hit_token_cap": True},
    ]
    oracle = [
        {"eval_id": pair.eval_id, "response": "A", "hit_token_cap": False}
        for pair in pairs
    ]
    pred = [
        {"eval_id": pairs[0].eval_id, "response": "A", "hit_token_cap": False},
        {"eval_id": pairs[1].eval_id, "response": "B", "hit_token_cap": False},
    ]
    controls = [
        {"eval_id": pair.eval_id, "response": "B", "hit_token_cap": False}
        for pair in pairs
    ]
    report = depth.score_paired_predictions(pairs, g, oracle, controls, controls, pred)
    assert report["metrics"]["n_scenes"] == 4
    assert report["metrics"]["paired_dependency_gap_oracle_minus_predicted"] == 0.75
    assert report["metrics"]["s_pred_strict_e2e_accuracy"] == 0.25
    assert report["metrics"]["positive_oracle_headroom"] == 0.75
    assert report["token_cap_counts"]["G"] == 1
    assert report["per_item_transitions"][2]["s_pred_request_emitted"] is False
    assert report["per_item_transitions"][3]["s_pred_request_emitted"] is False
    assert report["attribution"]["counts"] == {
        "Attributed(G)": 0,
        "Local(S)": 0,
        "Inconclusive": 4,
    }


def test_dependency_gap_uses_predicted_context_not_strict_end_to_end() -> None:
    pair = depth.PairExample(
        eval_id="gsa-sdepth-v52-00000000000000000000",
        sample_id="semantic-gap-counterexample",
        scene_id="scene0000_00",
        source_image=Path("/private/0.png"),
        image_sha256="1" * 64,
        entity_a=depth.Entity("chair", "only chair", (100, 100, 300, 500)),
        entity_b=depth.Entity("table", "only table", (500, 100, 800, 600)),
        nearer_choice="A",
        label={},
    )
    valid_but_wrong_g = '{"A":[500,100,800,600],"B":[100,100,300,500]}'
    g = [{"eval_id": pair.eval_id, "response": valid_but_wrong_g, "hit_token_cap": False}]
    oracle = [{"eval_id": pair.eval_id, "response": "A", "hit_token_cap": False}]
    controls = [{"eval_id": pair.eval_id, "response": "B", "hit_token_cap": False}]
    predicted = [{"eval_id": pair.eval_id, "response": "A", "hit_token_cap": False}]

    report = depth.score_paired_predictions(
        [pair], g, oracle, controls, controls, predicted
    )
    metrics = report["metrics"]
    assert metrics["s_pred_correct_all_scenes_invalid_g_zero"] == 1.0
    assert metrics["s_pred_strict_e2e_accuracy"] == 0.0
    assert metrics["paired_dependency_gap_oracle_minus_predicted"] == 0.0
    assert report["scene_bootstrap"]["ci"] == [0.0, 0.0]
    assert report["transition_counts"] == {
        "oracle_correct_to_predicted_correct": 1
    }
    assert report["strict_e2e_transition_counts"] == {
        "oracle_correct_to_e2e_wrong": 1
    }


def test_three_state_attribution_withholds_invalid_nonmonotonic_and_insensitive() -> None:
    pairs = [
        depth.PairExample(
            eval_id=f"gsa-sdepth-v52-{index:020x}",
            sample_id=f"sample-{index}",
            scene_id=f"scene{index:04d}_00",
            source_image=Path(f"/private/{index}.png"),
            image_sha256=f"{index + 1:064x}",
            entity_a=depth.Entity("chair", "only chair", (100, 100, 300, 500)),
            entity_b=depth.Entity("table", "only table", (500, 100, 800, 600)),
            nearer_choice="A",
            label={},
        )
        for index in range(6)
    ]
    good = '{"A":[100,100,300,500],"B":[500,100,800,600]}'
    bad = '{"A":[500,100,800,600],"B":[100,100,300,500]}'
    g = [
        {"eval_id": pair.eval_id, "response": bad if index == 0 else good, "hit_token_cap": False}
        for index, pair in enumerate(pairs)
    ]
    oracle_responses = ["A", "A", "B", "A", "A", "B"]
    absent_responses = ["B", "B", "B", "A", "B", "A"]
    corrupt_responses = ["B", "B", "B", "B", "B", "B"]
    oracle = [
        {"eval_id": pair.eval_id, "response": response, "hit_token_cap": False}
        for pair, response in zip(pairs, oracle_responses, strict=True)
    ]
    absent = [
        {"eval_id": pair.eval_id, "response": response, "hit_token_cap": False}
        for pair, response in zip(pairs, absent_responses, strict=True)
    ]
    corrupted = [
        {"eval_id": pair.eval_id, "response": response, "hit_token_cap": False}
        for pair, response in zip(pairs, corrupt_responses, strict=True)
    ]
    predicted = [
        {"eval_id": pair.eval_id, "response": "B", "hit_token_cap": index == 4}
        for index, pair in enumerate(pairs)
    ]
    report = depth.score_paired_predictions(
        pairs, g, oracle, absent, corrupted, predicted
    )
    assert report["attribution"]["counts"] == {
        "Attributed(G)": 1,
        "Local(S)": 1,
        "Inconclusive": 4,
    }
    reasons = report["attribution"]["reason_counts"]
    assert reasons["invalid_or_capped_or_unparsed_output"] == 1
    assert reasons["context_insensitive_or_control_failure"] == 1
    assert reasons["nonmonotonic_control_improvement"] == 1
    assert reasons["oracle_repairs_S_but_G_passes_threshold"] == 1


def test_local_s_requires_failure_under_oracle_context_as_defined_in_method() -> None:
    pairs = [
        depth.PairExample(
            eval_id=f"gsa-sdepth-v52-{index:020x}",
            sample_id=f"method-counterexample-{index}",
            scene_id=f"scene{index:04d}_00",
            source_image=Path(f"/private/{index}.png"),
            image_sha256=f"{index + 1:064x}",
            entity_a=depth.Entity("chair", "only chair", (100, 100, 300, 500)),
            entity_b=depth.Entity("table", "only table", (500, 100, 800, 600)),
            nearer_choice="A",
            label={},
        )
        for index in range(3)
    ]
    good_g = '{"A":[100,100,300,500],"B":[500,100,800,600]}'
    g = [
        {"eval_id": pair.eval_id, "response": good_g, "hit_token_cap": False}
        for pair in pairs
    ]
    oracle = [
        {"eval_id": pair.eval_id, "response": response, "hit_token_cap": False}
        for pair, response in zip(pairs, ["A", "B", "B"], strict=True)
    ]
    controls = [
        {"eval_id": pair.eval_id, "response": "B", "hit_token_cap": False}
        for pair in pairs
    ]
    predicted = [
        {"eval_id": pair.eval_id, "response": response, "hit_token_cap": False}
        for pair, response in zip(pairs, ["B", "B", "A"], strict=True)
    ]

    report = depth.score_paired_predictions(
        pairs, g, oracle, controls, controls, predicted
    )
    rows = report["per_item_transitions"]
    assert (rows[0]["attribution_state"], rows[0]["attribution_reason"]) == (
        "Inconclusive",
        "oracle_repairs_S_but_G_passes_threshold",
    )
    assert (rows[1]["attribution_state"], rows[1]["attribution_reason"]) == (
        "Local(S)",
        "correct_G_but_S_fails_under_oracle_context",
    )
    assert (rows[2]["attribution_state"], rows[2]["attribution_reason"]) == (
        "Inconclusive",
        "oracle_degradation_predicted_context_correct",
    )


def test_cpu_fake_engine_runs_all_five_stride_stages_and_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _synthetic_release(tmp_path)
    monkeypatch.setattr(depth, "validate_request_release", lambda *_args, **_kwargs: release)
    release_manifest = tmp_path / "release/release.manifest.json"
    receipt = tmp_path / "receipt.json"
    dlc.atomic_json(receipt, dlc.build_input_receipt(release_manifest))
    binding = tmp_path / "binding.json"
    dlc.atomic_json(binding, dlc.bind_model("evaluated", _fake_model(tmp_path), receipt))

    good = '{"A":[100,100,300,500],"B":[500,100,800,600]}'
    g_rows, g_manifest, _ = _complete_stage(
        tmp_path, receipt, binding, alias="g", track=depth.TRACK_G, response=good
    )
    oracle_rows, _, _ = _complete_stage(
        tmp_path,
        receipt,
        binding,
        alias="oracle",
        track=depth.TRACK_S_ORACLE,
        response="A",
    )
    absent_rows, _, _ = _complete_stage(
        tmp_path,
        receipt,
        binding,
        alias="absent",
        track=depth.TRACK_S_ABSENT,
        response="B",
    )
    corrupted_rows, _, _ = _complete_stage(
        tmp_path,
        receipt,
        binding,
        alias="corrupted",
        track=depth.TRACK_S_CORRUPTED,
        response="B",
    )
    pred_requests = tmp_path / "pred/requests.jsonl"
    pred_request_manifest = tmp_path / "pred/requests.manifest.json"
    built = dlc.build_predicted_request_release(
        binding_path=binding,
        g_manifest_path=g_manifest,
        g_predictions_path=g_rows,
        out=pred_requests,
        manifest_out=pred_request_manifest,
    )
    assert built["source_records"] == 40
    assert built["valid_grounding_records"] == 40
    pred_rows, pred_manifest, pred_shards = _complete_stage(
        tmp_path,
        receipt,
        binding,
        alias="pred",
        track=depth.TRACK_S_PRED,
        response="A",
        dynamic_manifest=pred_request_manifest,
    )
    result = depth.score_paired_predictions(
        release.source.pairs,
        dlc.load_jsonl(g_rows, "G"),
        dlc.load_jsonl(oracle_rows, "oracle"),
        dlc.load_jsonl(absent_rows, "absent"),
        dlc.load_jsonl(corrupted_rows, "corrupted"),
        dlc.load_jsonl(pred_rows, "pred"),
        provenance={"synthetic_verified_runner": True},
    )
    assert result["status"] == "FORMAL_SCORE_COMPLETE"
    assert result["metrics"]["g_both_box_iou_at_0_5"] == 1.0
    assert result["metrics"]["s_pred_strict_e2e_accuracy"] == 1.0
    assert result["scene_bootstrap"]["paired_scenes"] == 40
    assert result["metrics"]["paired_context_gain_oracle_minus_absent"] == 1.0
    assert result["attribution"]["counts"]["Inconclusive"] == 40

    rows = dlc.load_jsonl(pred_shards[0], "pred shard")
    rows[0]["response"] = "B"
    dlc.atomic_jsonl(pred_shards[0], rows)
    with pytest.raises(
        dlc.RunnerError,
        match="(?:bound )?file (?:size |hash )?drift|differ from verified shards",
    ):
        dlc.verify_prediction_bundle(
            manifest_path=pred_manifest,
            binding_path=binding,
            predictions_path=pred_rows,
            expected_track=depth.TRACK_S_PRED,
            dynamic_request_manifest=pred_request_manifest,
        )


def test_launcher_pending_exit_precedes_any_cuda_query(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python"
    calls = tmp_path / "calls.log"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {calls}\n"
        "if [[ \"$2\" == gate ]]; then\n"
        "  echo '{\"status\":\"BLOCKED_PENDING_HUMAN_REVIEW\",\"gpu_started\":false}'\n"
        "  exit 20\n"
        "fi\n"
        "exit 99\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    out = tmp_path / "out"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHON_BIN": str(fake_python),
            "OUT": str(out),
            "V52_FORMAL_REPORT": str(tmp_path / "formal.json"),
            "PREFLIGHT_ONLY": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 20
    assert "BLOCKED_PENDING_HUMAN_REVIEW" in result.stdout
    assert "gpu_started=false" in result.stdout
    assert "nvidia-smi" not in result.stdout + result.stderr
    assert calls.read_text(encoding="utf-8").count("\n") == 1


def test_launcher_fails_closed_when_nvidia_process_query_fails(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python"
    python_calls = tmp_path / "python-calls.log"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {python_calls}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_calls = tmp_path / "nvidia-calls.log"
    fake_nvidia = fake_bin / "nvidia-smi"
    fake_nvidia.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {nvidia_calls}\n"
        "echo 'fixture NVML query failure' >&2\n"
        "exit 17\n",
        encoding="utf-8",
    )
    fake_nvidia.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "PYTHON_BIN": str(fake_python),
            "OUT": str(tmp_path / "out"),
            "V52_FORMAL_REPORT": str(tmp_path / "formal.json"),
            "PREFLIGHT_ONLY": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 3
    assert "nvidia-smi compute-process query failed" in result.stdout
    assert "refusing to assume GPUs are idle" in result.stdout
    assert "fixture NVML query failure" in result.stdout
    assert "4x physical L20 hardware gate: READY" not in result.stdout
    assert "--query-compute-apps=pid" in nvidia_calls.read_text(encoding="utf-8")
    assert not any(
        line.strip() == "-"
        for line in python_calls.read_text(encoding="utf-8").splitlines()
    )


def test_cli_dry_runs_and_launcher_static_contract() -> None:
    for module in (
        ROOT / "eval/scannet_rgb_depth_v52_eval.py",
        ROOT / "eval/scannet_rgb_depth_v52_dlc.py",
    ):
        result = subprocess.run(
            [sys.executable, str(module), "dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "DRY_RUN_PASS" in result.stdout
    launcher = ROOT / "scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh"
    syntax = subprocess.run(["bash", "-n", str(launcher)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    text = launcher.read_text(encoding="utf-8")
    assert text.index('"$PYTHON_BIN" "$EVALUATOR" "${GATE_ARGS[@]}"') < text.index("nvidia-smi")
    assert "count != 4" in text
    assert '"L20" not in name.upper()' in text
    assert "MIG" in text
    assert "setsid env CUDA_VISIBLE_DEVICES" in text
    assert "for shard in 0 1 2 3" in text
    assert "--num-shards 4" in text
    assert 'wait "$pgid" || rc=1' in text
    assert "run_stage g" in text
    assert "run_stage s_oracle" in text
    assert "run_stage s_absent" in text
    assert "run_stage s_corrupted" in text
    assert "run_stage s_pred" in text
    assert text.index("run_stage g") < text.index("run_stage s_oracle")
    assert text.index("run_stage s_oracle") < text.index("run_stage s_absent")
    assert text.index("run_stage s_absent") < text.index("run_stage s_corrupted")
    assert text.index("run_stage s_corrupted") < text.index("run_stage s_pred")


def test_absent_and_corrupted_requests_follow_preregistered_context_rule(
    tmp_path: Path,
) -> None:
    release = _synthetic_release(tmp_path)
    pair = release.source.pairs[0]
    absent = release.requests[depth.TRACK_S_ABSENT][0]
    corrupted = release.requests[depth.TRACK_S_CORRUPTED][0]
    assert "bbox" not in absent["messages"][0]["content"]
    assert set(absent["pair"]["A"]) == {"category", "referring_expression"}
    assert corrupted["pair"]["A"]["bbox_2d"] == list(pair.entity_b.bbox)
    assert corrupted["pair"]["B"]["bbox_2d"] == list(pair.entity_a.bbox)
    contract = depth.load_intervention_contract()
    assert contract["protocol"].endswith("contract-v3")
    assert contract["metrics"]["strict_end_to_end_is_dependency_gap_endpoint"] is False
    assert contract["contexts"]["corrupted"]["downstream_truth_fields_read"] == []
    assert contract["contexts"]["corrupted"]["per_target_assignment_is_ground_truth"] is False
