from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import scannet_rgb_depth_v51 as depth  # noqa: E402
from eval import scannet_rgb_depth_v51_dlc as dlc  # noqa: E402


RELEASE_MANIFEST = depth.DEFAULT_RELEASE / "release.manifest.json"


def _fake_model(root: Path, *, adapter: bool = False) -> Path:
    model = root / "model"
    model.mkdir()
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
    if adapter:
        (model / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n', encoding="utf-8"
        )
    return model


def _bound_run(tmp_path: Path) -> tuple[Path, Path]:
    receipt_path = tmp_path / "receipt.json"
    dlc.atomic_json(receipt_path, dlc.build_input_receipt(RELEASE_MANIFEST))
    binding_path = tmp_path / "binding.json"
    dlc.atomic_json(
        binding_path,
        dlc.bind_model("evaluated", _fake_model(tmp_path), receipt_path),
    )
    return receipt_path, binding_path


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
    shard_paths: list[Path] = []

    def fake(public: list[dict], max_tokens: int) -> list[tuple[str, bool]]:
        assert max_tokens == dlc.STAGE_MAX_TOKENS[track]
        assert all(set(item) == {"messages", "images"} for item in public)
        serialised = json.dumps(public).casefold()
        assert "[instance" not in serialised
        assert "scannet-sdepth" not in serialised
        assert depth._SCENE_ID.search(serialised) is None
        return [(response, False) for _ in public]

    for shard in range(4):
        path = stage / f"shard_{shard}.jsonl"
        result = dlc.infer_shard(
            receipt_path=receipt,
            binding_path=binding,
            track=track,
            shard=shard,
            num_shards=4,
            batch_size=4,
            max_tokens=dlc.STAGE_MAX_TOKENS[track],
            out=path,
            dynamic_request_manifest=dynamic_manifest,
            infer_public=fake,
        )
        assert result["status"] == "generated"
        shard_paths.append(path)
    merged = stage / "predictions.jsonl"
    manifest = stage / "predictions.manifest.json"
    dlc.merge_shards(
        receipt_path=receipt,
        binding_path=binding,
        track=track,
        shard_paths=shard_paths,
        out=merged,
        manifest_out=manifest,
        dynamic_request_manifest=dynamic_manifest,
    )
    return merged, manifest, shard_paths


def test_cpu_fake_four_way_three_stage_run_and_formal_scorer(tmp_path: Path) -> None:
    receipt, binding = _bound_run(tmp_path)
    g_response = (
        '{"A":{"bbox_2d":[100,100,300,500]},'
        '"B":{"bbox_2d":[500,100,800,600]}}'
    )
    g_predictions, g_manifest, _ = _complete_stage(
        tmp_path,
        receipt,
        binding,
        alias="g",
        track=depth.TRACK_G,
        response=g_response,
    )
    oracle_predictions, oracle_manifest, _ = _complete_stage(
        tmp_path,
        receipt,
        binding,
        alias="oracle",
        track=depth.TRACK_S_ORACLE,
        response="A",
    )
    s_pred_requests = tmp_path / "pred" / "requests.jsonl"
    s_pred_request_manifest = tmp_path / "pred" / "requests.manifest.json"
    built = dlc.build_predicted_request_release(
        binding_path=binding,
        g_manifest_path=g_manifest,
        g_predictions_path=g_predictions,
        out=s_pred_requests,
        manifest_out=s_pred_request_manifest,
    )
    assert built["source_records"] == 40
    assert built["valid_grounding_records"] == 40
    pred_predictions, pred_manifest, _ = _complete_stage(
        tmp_path,
        receipt,
        binding,
        alias="pred",
        track=depth.TRACK_S_PRED,
        response="A",
        dynamic_manifest=s_pred_request_manifest,
    )
    score = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval/scannet_rgb_depth_v51.py"),
            "score",
            "--release-manifest",
            str(RELEASE_MANIFEST),
            "--model-binding",
            str(binding),
            "--g-predictions",
            str(g_predictions),
            "--g-manifest",
            str(g_manifest),
            "--s-oracle-predictions",
            str(oracle_predictions),
            "--s-oracle-manifest",
            str(oracle_manifest),
            "--s-pred-request-manifest",
            str(s_pred_request_manifest),
            "--s-pred-predictions",
            str(pred_predictions),
            "--s-pred-manifest",
            str(pred_manifest),
            "--out",
            str(score),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(score.read_text(encoding="utf-8"))
    assert report["status"] == "FORMAL_SCORE_COMPLETE"
    assert report["metrics"]["n_scenes"] == 40
    assert len(report["per_item_transitions"]) == 40
    assert report["paired_bootstrap"]["paired_scenes"] == 40
    assert report["provenance"]["same_model_binding_all_stages"] is True
    assert report["trust"]["score_available"] is True
    assert report["task_contract"]["training_materialization_eligible"] is False


def test_prediction_shard_and_model_tamper_fail_closed(tmp_path: Path) -> None:
    receipt, binding = _bound_run(tmp_path)
    merged, manifest, shards = _complete_stage(
        tmp_path,
        receipt,
        binding,
        alias="g",
        track=depth.TRACK_G,
        response='{"A":{"bbox_2d":[1,2,100,200]},"B":{"bbox_2d":[300,4,900,800]}}',
    )
    rows = dlc.load_jsonl(shards[0], "fixture shard")
    rows[0]["response"] = "tampered"
    dlc.atomic_jsonl(shards[0], rows)
    with pytest.raises(dlc.RunnerError, match="file (?:size )?drift|differ from verified shards"):
        dlc.verify_prediction_bundle(
            manifest_path=manifest,
            binding_path=binding,
            predictions_path=merged,
            expected_track=depth.TRACK_G,
        )

    bound = json.loads(binding.read_text(encoding="utf-8"))
    weight = Path(bound["model"]["weight_files"][0]["path"])
    original = weight.read_bytes()
    stat = weight.stat()
    weight.write_bytes(bytes(byte ^ 1 for byte in original))
    os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    dlc.verify_model_binding(binding, deep=False)
    with pytest.raises(dlc.RunnerError, match="weight hash drift"):
        dlc.verify_model_binding(binding, deep=True)


def test_unmerged_adapter_and_wrong_stage_budget_are_rejected(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    dlc.atomic_json(receipt_path, dlc.build_input_receipt(RELEASE_MANIFEST))
    with pytest.raises(dlc.RunnerError, match="unmerged adapter"):
        dlc.bind_model("evaluated", _fake_model(tmp_path, adapter=True), receipt_path)

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    binding_path = tmp_path / "clean-binding.json"
    dlc.atomic_json(
        binding_path,
        dlc.bind_model("evaluated", _fake_model(clean_root), receipt_path),
    )
    with pytest.raises(dlc.RunnerError, match="max_tokens=96"):
        dlc.infer_shard(
            receipt_path=receipt_path,
            binding_path=binding_path,
            track=depth.TRACK_G,
            shard=0,
            num_shards=4,
            batch_size=4,
            max_tokens=16,
            out=tmp_path / "wrong-budget.jsonl",
            infer_public=lambda rows, _: [("", False) for _ in rows],
        )


def test_launcher_is_four_l20_gated_and_cpu_preflight_only(tmp_path: Path) -> None:
    launcher = ROOT / "scripts/run_scannet_rgb_depth_v51_eval_l20_4gpu.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(launcher)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    text = launcher.read_text(encoding="utf-8")
    assert "count != 4" in text
    assert '"L20" not in name.upper()' in text
    assert "MIG" in text
    assert "THINKING=off" in text
    assert "G_MAX_TOKENS=96" in text
    assert "S_MAX_TOKENS=16" in text
    assert "run_stage g" in text
    assert "run_stage s_oracle" in text
    assert "run_stage s_pred" in text

    model = _fake_model(tmp_path)
    out = tmp_path / "preflight"
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env={
            **os.environ,
            "MODEL": str(model),
            "OUT": str(out),
            "PREFLIGHT_ONLY": "1",
            "THINKING": "off",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "SCANNET_RGB_DEPTH_V51_PREFLIGHT_READY" in result.stdout
    assert "no GPU query or process was started" in result.stdout
    assert (out / "formal40_inputs.json").is_file()
    assert (out / "model_binding/model.json").is_file()
    assert not (out / "g/shard_0.jsonl").exists()
