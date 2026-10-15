from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import scannet_sensor_pairwise as pair_eval  # noqa: E402
from eval import scannet_sensor_pairwise_dlc as dlc  # noqa: E402


REQUEST_ROOT = ROOT / "artifacts/depth_v50_eval_interface/full_v3"
REQUEST_MANIFEST = REQUEST_ROOT / "s_oracle_requests.manifest.json"
REQUESTS = REQUEST_ROOT / "s_oracle_requests.jsonl"
SOURCE_ROOT = ROOT / "data/depth_v50_scannet_sensor/full_v3"


def _fake_model(root: Path, *, adapter: bool = False) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "vision_config": {"model_type": "qwen3_5"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"fake-merged-weight-bytes")
    (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    if adapter:
        (model / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n', encoding="utf-8"
        )
    return model


def _bound_run(tmp_path: Path) -> tuple[Path, Path]:
    receipt_path = tmp_path / "receipt.json"
    receipt = dlc.build_input_receipt(REQUEST_MANIFEST, REQUESTS)
    dlc.atomic_json(receipt_path, receipt)
    model = _fake_model(tmp_path)
    binding_path = tmp_path / "binding.json"
    binding = dlc.bind_model("base", model, receipt_path)
    dlc.atomic_json(binding_path, binding)
    return receipt_path, binding_path


def _complete_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    receipt_path, binding_path = _bound_run(tmp_path)
    shard_paths = []
    for shard in range(8):
        path = tmp_path / "predictions" / f"shard_{shard}.jsonl"
        dlc.infer_shard(
            receipt_path=receipt_path,
            binding_path=binding_path,
            shard=shard,
            num_shards=8,
            batch_size=4,
            max_tokens=16,
            out=path,
            infer_public=lambda rows, _: [("A", False) for _ in rows],
        )
        shard_paths.append(path)
    merged = tmp_path / "predictions" / "predictions.jsonl"
    manifest = tmp_path / "predictions" / "predictions.manifest.json"
    dlc.merge_shards(
        receipt_path=receipt_path,
        binding_path=binding_path,
        shard_paths=shard_paths,
        out=merged,
        manifest_out=manifest,
    )
    return receipt_path, binding_path, merged, manifest


def _formal_score(
    *,
    predictions: Path,
    manifest: Path | None,
    binding: Path | None,
    out: Path,
    skip_image_verification: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "eval/scannet_sensor_pairwise.py"),
        "score",
        "--source",
        str(SOURCE_ROOT / "labels/S_spatial/depth.jsonl"),
        "--source-report",
        str(SOURCE_ROOT / "generation_report.json"),
        "--track",
        "s-oracle",
        "--predictions",
        str(predictions),
        "--out",
        str(out),
    ]
    if skip_image_verification:
        command.append("--skip-image-verification")
    if manifest is not None:
        command.extend(["--prediction-manifest", str(manifest)])
    if binding is not None:
        command.extend(["--model-binding", str(binding)])
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_formal_request_receipt_is_exactly_pinned_and_blinded(tmp_path: Path) -> None:
    receipt = dlc.build_input_receipt(REQUEST_MANIFEST, REQUESTS)
    release = receipt["request_release"]
    assert receipt["protocol"] == dlc.RUN_PROTOCOL
    assert receipt["evaluation_protocol"] == pair_eval.EVALUATION_PROTOCOL
    assert release["manifest"]["sha256"] == dlc.EXPECTED_REQUEST_MANIFEST_SHA256
    assert release["requests"]["sha256"] == dlc.EXPECTED_REQUESTS_SHA256
    assert release["request_records_sha256"] == dlc.EXPECTED_REQUEST_RECORDS_SHA256
    assert release["n_requests"] == release["n_unique_images"] == 120
    assert release["label_balance"] == {"A": 60, "B": 60}
    assert release["blinded_model_fields"] == ["messages", "images"]
    assert release["private_fields_passed_to_model"] == []
    path = tmp_path / "receipt.json"
    dlc.atomic_json(path, receipt)
    assert dlc.verify_input_receipt(path) == receipt


def test_unmerged_adapter_is_rejected(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    dlc.atomic_json(
        receipt_path, dlc.build_input_receipt(REQUEST_MANIFEST, REQUESTS)
    )
    model = _fake_model(tmp_path, adapter=True)
    with pytest.raises(dlc.RunnerError, match="unmerged adapter"):
        dlc.bind_model("merged", model, receipt_path)


def test_model_weight_hash_binding_detects_same_size_tamper(tmp_path: Path) -> None:
    receipt_path, binding_path = _bound_run(tmp_path)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    weight = Path(binding["model"]["weight_files"][0]["path"])
    original = weight.read_bytes()
    stat = weight.stat()
    replacement = bytes(byte ^ 0x01 for byte in original)
    assert len(replacement) == len(original)
    weight.write_bytes(replacement)
    os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    dlc.verify_model_binding(binding_path, deep=False)
    with pytest.raises(dlc.RunnerError, match="model weight hash drift"):
        dlc.verify_model_binding(binding_path, deep=True)


def test_fake_cpu_eight_way_resume_merge_and_existing_scorer(tmp_path: Path) -> None:
    receipt_path, binding_path = _bound_run(tmp_path)
    seen_inputs: list[list[dict]] = []

    def fake_infer(public_inputs: list[dict], max_tokens: int) -> list[tuple[str, bool]]:
        assert max_tokens == 16
        assert all(set(item) == {"messages", "images"} for item in public_inputs)
        serialised = json.dumps(public_inputs).casefold()
        assert "sensor_measurements" not in serialised
        assert "relative_depth" not in serialised
        assert "[instance" not in serialised
        seen_inputs.append(public_inputs)
        return [("A", False) for _ in public_inputs]

    shard_paths = []
    for shard in range(8):
        path = tmp_path / "predictions" / f"shard_{shard}.jsonl"
        summary = dlc.infer_shard(
            receipt_path=receipt_path,
            binding_path=binding_path,
            shard=shard,
            num_shards=8,
            batch_size=4,
            max_tokens=16,
            out=path,
            infer_public=fake_infer,
        )
        assert summary["status"] == "generated"
        assert summary["records"] == 15
        shard_paths.append(path)
    assert len(seen_inputs) == 8
    assert sum(len(items) for items in seen_inputs) == 120

    def must_not_run(_: list[dict], __: int) -> list[tuple[str, bool]]:
        raise AssertionError("valid resume should not call the inference engine")

    resumed = dlc.infer_shard(
        receipt_path=receipt_path,
        binding_path=binding_path,
        shard=0,
        num_shards=8,
        batch_size=4,
        max_tokens=16,
        out=shard_paths[0],
        infer_public=must_not_run,
    )
    assert resumed["status"] == "reused"

    merged = tmp_path / "predictions" / "predictions.jsonl"
    merged_manifest = tmp_path / "predictions" / "predictions.manifest.json"
    merge_report = dlc.merge_shards(
        receipt_path=receipt_path,
        binding_path=binding_path,
        shard_paths=shard_paths,
        out=merged,
        manifest_out=merged_manifest,
    )
    rows = dlc.load_jsonl(merged, "merged fixture predictions")
    assert merge_report["n_predictions"] == len(rows) == 120
    assert len({row["eval_id"] for row in rows}) == 120
    assert all(row["hit_token_cap"] is False for row in rows)

    score = tmp_path / "score.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval/scannet_sensor_pairwise.py"),
            "score",
            "--source",
            str(SOURCE_ROOT / "labels/S_spatial/depth.jsonl"),
            "--source-report",
            str(SOURCE_ROOT / "generation_report.json"),
            "--track",
            "s-oracle",
            "--predictions",
            str(merged),
            "--prediction-manifest",
            str(merged_manifest),
            "--model-binding",
            str(binding_path),
            "--out",
            str(score),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    scored = json.loads(score.read_text(encoding="utf-8"))
    assert scored["trust"]["score_available"] is True
    assert scored["trust"]["prediction_provenance_verified"] is True
    assert scored["metrics"]["n_predictions"] == 120
    assert scored["metrics"]["pairwise_depth_accuracy_all"] == 0.5
    provenance = scored["provenance"]
    assert provenance["verified"] is True
    assert provenance["prediction_manifest"]["sha256"] == dlc.file_hash(
        merged_manifest
    )
    assert provenance["merged_predictions"]["sha256"] == dlc.file_hash(merged)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert (
        provenance["model_inventory"]["identity_sha256"]
        == binding["model"]["identity_sha256"]
    )


def test_formal_scorer_rejects_missing_provenance(tmp_path: Path) -> None:
    rows = dlc.load_jsonl(REQUESTS, "formal requests")
    forged = tmp_path / "forged.jsonl"
    forged.write_text(
        "".join(
            json.dumps(
                {
                    "eval_id": row["eval_id"],
                    "response": "A",
                    "hit_token_cap": False,
                }
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    result = _formal_score(
        predictions=forged,
        manifest=None,
        binding=None,
        out=tmp_path / "missing-provenance-score.json",
    )
    assert result.returncode == 2
    assert "requires --prediction-manifest and --model-binding" in result.stderr


def test_formal_scorer_forbids_skipping_image_verification(tmp_path: Path) -> None:
    result = _formal_score(
        predictions=REQUESTS,
        manifest=None,
        binding=None,
        out=tmp_path / "skip-images-score.json",
        skip_image_verification=True,
    )
    assert result.returncode == 2
    assert "forbids --skip-image-verification" in result.stderr


def test_formal_scorer_rejects_rehashed_manifest_for_forged_rows(
    tmp_path: Path,
) -> None:
    _, binding_path, _, manifest_path = _complete_bundle(tmp_path)
    requests = dlc.load_jsonl(REQUESTS, "formal requests")
    forged = tmp_path / "forged-minimal.jsonl"
    forged.write_text(
        "".join(
            json.dumps(
                {
                    "eval_id": row["eval_id"],
                    "response": "A",
                    "hit_token_cap": False,
                }
            )
            + "\n"
            for row in requests
        ),
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["merged_predictions"] = dlc.file_binding(forged)
    unsigned = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    manifest["identity_sha256"] = dlc.content_hash(unsigned)
    forged_manifest = tmp_path / "forged.manifest.json"
    forged_manifest.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = _formal_score(
        predictions=forged,
        manifest=forged_manifest,
        binding=binding_path,
        out=tmp_path / "forged-score.json",
    )
    assert result.returncode == 2
    assert "merged prediction 0: schema drift" in result.stderr


def test_formal_scorer_rejects_merged_rows_that_differ_from_shards(
    tmp_path: Path,
) -> None:
    _, binding_path, merged, manifest_path = _complete_bundle(tmp_path)
    rows = dlc.load_jsonl(merged, "merged predictions")
    rows[0]["response"] = "B"
    altered = tmp_path / "altered-predictions.jsonl"
    altered.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["merged_predictions"] = dlc.file_binding(altered)
    unsigned = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    manifest["identity_sha256"] = dlc.content_hash(unsigned)
    altered_manifest = tmp_path / "altered.manifest.json"
    altered_manifest.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = _formal_score(
        predictions=altered,
        manifest=altered_manifest,
        binding=binding_path,
        out=tmp_path / "altered-score.json",
    )
    assert result.returncode == 2
    assert "merged predictions differ from the verified shards" in result.stderr


def test_formal_scorer_rejects_rehashed_empty_model_inventory(
    tmp_path: Path,
) -> None:
    _, binding_path, merged, manifest_path = _complete_bundle(tmp_path)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["model"]["metadata_files"] = {}
    binding["model"]["weight_files"] = []
    model_unsigned = {
        key: value
        for key, value in binding["model"].items()
        if key != "identity_sha256"
    }
    binding["model"]["identity_sha256"] = dlc.content_hash(model_unsigned)
    binding_unsigned = {
        key: value for key, value in binding.items() if key != "identity_sha256"
    }
    binding["identity_sha256"] = dlc.content_hash(binding_unsigned)
    forged_binding = tmp_path / "forged-binding.json"
    forged_binding.write_text(json.dumps(binding) + "\n", encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_binding"] = dlc.file_binding(forged_binding)
    manifest["model_inventory_identity_sha256"] = binding["model"]["identity_sha256"]
    manifest_unsigned = {
        key: value for key, value in manifest.items() if key != "identity_sha256"
    }
    manifest["identity_sha256"] = dlc.content_hash(manifest_unsigned)
    forged_manifest = tmp_path / "forged-model.manifest.json"
    forged_manifest.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = _formal_score(
        predictions=merged,
        manifest=forged_manifest,
        binding=forged_binding,
        out=tmp_path / "forged-model-score.json",
    )
    assert result.returncode == 2
    assert "model metadata inventory coverage drift" in result.stderr


def test_prediction_shard_tampering_fails_closed(tmp_path: Path) -> None:
    receipt_path, binding_path = _bound_run(tmp_path)
    shard_path = tmp_path / "shard_0.jsonl"
    dlc.infer_shard(
        receipt_path=receipt_path,
        binding_path=binding_path,
        shard=0,
        num_shards=8,
        batch_size=4,
        max_tokens=16,
        out=shard_path,
        infer_public=lambda rows, _: [("A", False) for _ in rows],
    )
    rows = dlc.load_jsonl(shard_path, "fixture shard")
    rows[0]["nearer_choice"] = "A"
    shard_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(dlc.RunnerError, match="prediction schema drift"):
        dlc.verify_shard(
            receipt_path=receipt_path,
            binding_path=binding_path,
            shard_path=shard_path,
            expected_shard=0,
        )


def test_l20_launcher_is_syntax_valid_and_hard_gated() -> None:
    launcher = ROOT / "scripts/run_scannet_sensor_pairwise_eval_l20.sh"
    result = subprocess.run(
        ["bash", "-n", str(launcher)], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    text = launcher.read_text(encoding="utf-8")
    assert "count != 8" in text
    assert '"L20" not in name.upper()' in text
    assert "THINKING=off" in text
    assert "--track s-oracle" in text
    assert "adapter checkpoints are rejected" in text.casefold()


def test_l20_launcher_cpu_preflight_stops_before_gpu(tmp_path: Path) -> None:
    model = _fake_model(tmp_path)
    out = tmp_path / "launcher-output"
    env = {
        **os.environ,
        "BASE_MODEL": str(model),
        "OUT": str(out),
        "PREFLIGHT_ONLY": "1",
        "THINKING": "off",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_scannet_sensor_pairwise_eval_l20.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "SCANNET_PAIRWISE_EVAL_PREFLIGHT_READY: no GPU process was started" in result.stdout
    assert (out / "scannet_sensor_pairwise_full_v3_inputs.json").is_file()
    assert (out / "model_bindings/base.json").is_file()
    assert not (out / "predictions/base/shard_0.jsonl").exists()
