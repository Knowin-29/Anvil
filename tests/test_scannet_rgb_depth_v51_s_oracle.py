from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import scannet_rgb_depth_v51 as depth  # noqa: E402
from eval import scannet_rgb_depth_v51_dlc as dlc  # noqa: E402
from eval import scannet_rgb_depth_v51_s_oracle as oracle  # noqa: E402


RELEASE_MANIFEST = depth.DEFAULT_RELEASE / "release.manifest.json"


def _fake_model(root: Path, name: str = "model") -> Path:
    model = root / name
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
    return model


def _bound_oracle_stage(tmp_path: Path) -> tuple[Path, Path, Path, list[Path]]:
    receipt = tmp_path / "receipt.json"
    dlc.atomic_json(receipt, dlc.build_input_receipt(RELEASE_MANIFEST))
    binding = tmp_path / "binding.json"
    dlc.atomic_json(binding, dlc.bind_model("evaluated", _fake_model(tmp_path), receipt))

    stage = tmp_path / "s_oracle"
    shards: list[Path] = []
    for shard in range(4):
        path = stage / f"shard_{shard}.jsonl"
        dlc.infer_shard(
            receipt_path=receipt,
            binding_path=binding,
            track=depth.TRACK_S_ORACLE,
            shard=shard,
            num_shards=4,
            batch_size=4,
            max_tokens=16,
            out=path,
            infer_public=lambda rows, _: [("A", False) for _ in rows],
        )
        shards.append(path)
    predictions = stage / "predictions.jsonl"
    manifest = stage / "predictions.manifest.json"
    dlc.merge_shards(
        receipt_path=receipt,
        binding_path=binding,
        track=depth.TRACK_S_ORACLE,
        shard_paths=shards,
        out=predictions,
        manifest_out=manifest,
    )
    return binding, predictions, manifest, shards


def _score_cli(
    binding: Path, predictions: Path, manifest: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval/scannet_rgb_depth_v51_s_oracle.py"),
            "score-s-oracle",
            "--release-manifest",
            str(RELEASE_MANIFEST),
            "--model-binding",
            str(binding),
            "--s-oracle-predictions",
            str(predictions),
            "--s-oracle-manifest",
            str(manifest),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_oracle_metrics_caps_strata_and_intervals_are_deterministic() -> None:
    pairs = [
        depth.PairExample(f"id-{index}", f"source-{index}", f"scene-{index}", Path("/x"), "0" * 64,
                          depth.Entity("x", (0, 0, 1, 1)), depth.Entity("y", (1, 1, 2, 2)), target)
        for index, target in enumerate(("A", "A", "B", "B"))
    ]
    rows = [
        {"eval_id": "id-0", "response": "A", "hit_token_cap": False},
        {"eval_id": "id-1", "response": "A", "hit_token_cap": True},
        {"eval_id": "id-2", "response": "unparseable", "hit_token_cap": False},
        {"eval_id": "id-3", "response": '{"choice":"B"}', "hit_token_cap": False},
    ]
    first = oracle.score_s_oracle_predictions(
        pairs, rows, parse_choice=depth.parse_s_choice
    )
    second = oracle.score_s_oracle_predictions(
        pairs, rows, parse_choice=depth.parse_s_choice
    )
    assert first == second
    assert first["metrics"]["correct"] == 2
    assert first["metrics"]["accuracy"] == 0.5
    assert first["metrics"]["parsed_responses"] == 3
    assert first["metrics"]["capped_responses"] == 1
    assert first["ab_strata"]["A"]["accuracy"] == 0.5
    assert first["ab_strata"]["B"]["accuracy"] == 0.5
    assert first["confidence_intervals"]["accuracy_scene_bootstrap"]["unit"] == "scene"
    assert first["confidence_intervals"]["accuracy_binomial"]["method"] == "wilson_score_binomial"
    assert first["task_contract"]["training_materialization_eligible"] is False


def test_live_referent_audit_reproduces_7_unique_and_33_nonunique() -> None:
    release = depth.validate_request_release(RELEASE_MANIFEST, verify_images=False)
    audit = oracle.audit_named_referent_uniqueness(depth, release)
    assert audit["status"] == "INTERFACE_BLOCKED_NONUNIQUE_REFERENTS"
    assert audit["records"] == 40
    assert audit["unique_category_referent_pairs"] == 7
    assert audit["nonunique_category_referent_pairs"] == 33
    assert audit["aggregation_files_rehashed"] == 40
    assert audit["instance_projection_members_rehashed"] == 40


def test_complete_bound_bundle_scores_and_marks_g_interface_blocked(tmp_path: Path) -> None:
    binding, predictions, manifest, _ = _bound_oracle_stage(tmp_path)
    output = tmp_path / "result.json"
    completed = _score_cli(binding, predictions, manifest, output)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in report.items() if key != "identity_sha256"}
    assert report["identity_sha256"] == oracle.canonical_hash(unsigned)
    assert report["status"] == "FORMAL_S_ORACLE_SCORE_COMPLETE"
    assert report["metrics"]["n_scenes"] == 40
    assert report["metrics"]["correct"] == 20
    assert report["metrics"]["accuracy"] == 0.5
    assert report["related_tracks"][depth.TRACK_G]["status"] == (
        "INTERFACE_BLOCKED_NONUNIQUE_REFERENTS"
    )
    assert report["provenance"]["verified"] is True
    assert report["trust"]["complete_bound_s_oracle_bundle"] is True
    assert report["trust"]["training_materialization_eligible"] is False


def test_bundle_and_receipt_code_tamper_fail_clean_without_result(tmp_path: Path) -> None:
    binding, predictions, manifest, shards = _bound_oracle_stage(tmp_path)
    rows = dlc.load_jsonl(shards[0], "fixture shard")
    rows[0]["response"] = "B"
    dlc.atomic_jsonl(shards[0], rows)
    output = tmp_path / "must-not-exist.json"
    failed = _score_cli(binding, predictions, manifest, output)
    assert failed.returncode == 2
    assert "contract error:" in failed.stderr
    assert "Traceback" not in failed.stderr
    assert not output.exists()

    receipt = json.loads(Path(json.loads(binding.read_text())["input_receipt"]["path"]).read_text())
    tampered = copy.deepcopy(receipt)
    tampered["evaluation_code"]["files"]["scannet_rgb_depth_v51.py"]["sha256"] = "0" * 64
    live = oracle._frozen_code_bindings()
    try:
        oracle._verify_receipt_code_binding(tampered, live)
    except oracle.OracleScoreError as exc:
        assert "code binding drift" in str(exc)
    else:
        raise AssertionError("tampered receipt code binding was accepted")


def test_wrong_model_binding_fails_closed(tmp_path: Path) -> None:
    binding, predictions, manifest, _ = _bound_oracle_stage(tmp_path)
    receipt_path = Path(json.loads(binding.read_text())["input_receipt"]["path"])
    wrong_binding = tmp_path / "wrong-binding.json"
    dlc.atomic_json(
        wrong_binding,
        dlc.bind_model("evaluated", _fake_model(tmp_path, "other-model"), receipt_path),
    )
    output = tmp_path / "wrong-model-result.json"
    failed = _score_cli(wrong_binding, predictions, manifest, output)
    assert failed.returncode == 2
    assert "model binding drift" in failed.stderr
    assert "Traceback" not in failed.stderr
    assert not output.exists()
