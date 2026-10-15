from __future__ import annotations

import json
import struct
import sys
import zlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.extract_scannet_train_sens import (
    SCANNET_SPLIT_SHA256,
    SourceError,
    build_canary,
    check_document,
    decode_color,
    decode_depth,
    inspect_source,
    load_split_manifests,
    read_sens_frame,
    register_depth_to_color,
)


SPLIT_ROOT = ROOT / "configs/scannet_splits"


def _matrix_bytes(value: np.ndarray) -> bytes:
    return np.asarray(value, dtype="<f4").tobytes()


def _jpeg_bytes(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _png_bytes(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_sens(path: Path, frame_count: int = 2) -> tuple[np.ndarray, np.ndarray]:
    color_height, color_width = 6, 8
    depth_height, depth_width = 3, 4
    intrinsic_color = np.eye(4, dtype=np.float32)
    intrinsic_color[0, 0] = intrinsic_color[1, 1] = 2.0
    intrinsic_color[0, 2] = 3.5
    intrinsic_color[1, 2] = 2.5
    intrinsic_depth = np.eye(4, dtype=np.float32)
    intrinsic_depth[0, 0] = intrinsic_depth[1, 1] = 1.0
    intrinsic_depth[0, 2] = 1.5
    intrinsic_depth[1, 2] = 1.0
    identity = np.eye(4, dtype=np.float32)
    color = np.zeros((color_height, color_width, 3), dtype=np.uint8)
    color[..., 0] = np.arange(color_width, dtype=np.uint8)
    color[..., 1] = 80
    depth = np.array(
        [[1000, 1000, 1000, 1000], [1200, 1200, 1200, 1200], [0, 1400, 1400, 0]],
        dtype=np.uint16,
    )
    sensor_name = b"SyntheticSensor"
    with path.open("wb") as handle:
        handle.write(struct.pack("<I", 4))
        handle.write(struct.pack("<Q", len(sensor_name)))
        handle.write(sensor_name)
        for matrix in (intrinsic_color, identity, intrinsic_depth, identity):
            handle.write(_matrix_bytes(matrix))
        handle.write(struct.pack("<iiIIIIfQ", 2, 1, color_width, color_height, depth_width, depth_height, 1000.0, frame_count))
        for frame_index in range(frame_count):
            color_bytes = _jpeg_bytes(color + frame_index)
            depth_bytes = zlib.compress((depth + frame_index).astype("<u2").tobytes())
            handle.write(_matrix_bytes(identity))
            handle.write(struct.pack("<QQQQ", frame_index, frame_index, len(color_bytes), len(depth_bytes)))
            handle.write(color_bytes)
            handle.write(depth_bytes)
    return color, depth


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    scannet_root = tmp_path / "ScanNet-official/scans"
    scene = scannet_root / "scene0191_00"
    scene.mkdir(parents=True)
    _write_sens(scene / "scene0191_00.sens")
    (scene / "scene0191_00.txt").write_text(
        "\n".join(
            (
                "colorHeight = 6",
                "colorWidth = 8",
                "depthHeight = 3",
                "depthWidth = 4",
                "numColorFrames = 2",
                "numDepthFrames = 2",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (scene / "scene0191_00.aggregation.json").write_text(
        json.dumps({"sceneId": "scannet.scene0191_00", "segGroups": []}) + "\n",
        encoding="utf-8",
    )
    instance = np.zeros((6, 8), dtype=np.uint8)
    instance[1:4, 2:5] = 1
    semantic = instance.astype(np.uint16) * 5
    with ZipFile(scene / "scene0191_00_2d-instance-filt.zip", "w", ZIP_DEFLATED) as archive:
        archive.writestr("instance-filt/1.png", _png_bytes(instance))
    with ZipFile(scene / "scene0191_00_2d-label-filt.zip", "w", ZIP_DEFLATED) as archive:
        archive.writestr("label-filt/1.png", _png_bytes(semantic))
    tasks = scannet_root.parent / "tasks"
    tasks.mkdir()
    label_map = tasks / "scannetv2-labels.combined.tsv"
    label_map.write_text("id\traw_category\tnyu40class\n1\twall\twall\n", encoding="utf-8")
    (scannet_root.parent / "download-scannet.py").write_text(
        "TOS_URL='http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf'\n"
        "print('confirm that you have agreed to the ScanNet terms of use')\n",
        encoding="utf-8",
    )
    return scannet_root, label_map


def test_frozen_split_hashes_and_membership() -> None:
    manifests, scene_to_split = load_split_manifests(SPLIT_ROOT)
    assert {name: doc["sha256"] for name, doc in manifests.items()} == SCANNET_SPLIT_SHA256
    assert scene_to_split["scene0191_00"] == "train"
    assert scene_to_split["scene0700_00"] == "val"


def test_sens_reader_decodes_requested_nonzero_frame(tmp_path: Path) -> None:
    sens = tmp_path / "sample.sens"
    _, depth = _write_sens(sens)
    header, frame = read_sens_frame(sens, 1)
    color_decoded = decode_color(header, frame.color_bytes)
    depth_decoded = decode_depth(header, frame.depth_bytes)
    assert color_decoded.shape == (6, 8, 3)
    assert depth_decoded.dtype == np.uint16
    np.testing.assert_array_equal(depth_decoded, depth + 1)
    assert frame.timestamp_color == 1


def test_registration_is_metric_and_z_buffered(tmp_path: Path) -> None:
    sens = tmp_path / "sample.sens"
    _, depth = _write_sens(sens)
    header, _ = read_sens_frame(sens, 0)
    result = register_depth_to_color(header, depth)
    assert result.native_valid_pixels == int(np.count_nonzero(depth))
    assert result.registered_pixels > 0
    assert result.registered_pixels <= result.projected_pixels
    assert result.aligned_raw.shape == (6, 8)
    assert result.aligned_raw.dtype == np.uint16


def test_official_val_scene_is_rejected_before_source_access(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="not official train"):
        inspect_source(
            scannet_root=tmp_path,
            split_root=SPLIT_ROOT,
            label_map=tmp_path / "labels.tsv",
            scene_id="scene0700_00",
            frame_index=0,
            license_evidence=None,
        )


def test_check_and_canary_are_fail_closed_for_training(tmp_path: Path) -> None:
    scannet_root, label_map = _source_fixture(tmp_path)
    inspected = inspect_source(
        scannet_root=scannet_root,
        split_root=SPLIT_ROOT,
        label_map=label_map,
        scene_id="scene0191_00",
        frame_index=1,
        license_evidence=None,
    )
    check = check_document(inspected)
    assert check["write_count"] == 0
    assert check["official_split"] == "train"
    assert check["verdict"] == "CHECK_PASS_SCALEOUT_BLOCKED_BY_LICENSE_EVIDENCE"

    output = tmp_path / "canary"
    result = build_canary(inspected, output)
    manifest = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
    assert result["training_materialization_eligible"] is False
    assert manifest["use_policy"]["official_train_only"] is True
    assert manifest["use_policy"]["may_enter_held_out_v5_or_schema5"] is False
    assert manifest["use_policy"]["training_materialization_eligible"] is False
    assert manifest["source_files"]["sens"]["sha256"]
    assert manifest["outputs"]["depth_aligned_to_color.png"]["sha256"]
    assert (output / "source_manifest.sha256").is_file()


def test_canary_refuses_overwrite(tmp_path: Path) -> None:
    scannet_root, label_map = _source_fixture(tmp_path)
    inspected = inspect_source(
        scannet_root=scannet_root,
        split_root=SPLIT_ROOT,
        label_map=label_map,
        scene_id="scene0191_00",
        frame_index=1,
        license_evidence=None,
    )
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(SourceError, match="refusing to overwrite"):
        build_canary(inspected, output)


def _anvil_source_fixture(tmp_path: Path):
    import importlib.util
    script = ROOT / "engineering/anvil.py"
    spec = importlib.util.spec_from_file_location("anvil_preparation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources, splits = tmp_path / "sources", tmp_path / "splits"
    shard = sources / "shard_00"
    shard.mkdir(parents=True)
    splits.mkdir()
    scene_ids = [f"scene{group:04d}_{scan:02d}" for group in range(30) for scan in (0, 1)]
    split_ids = {"train": scene_ids, "val": ["scene9998_00"], "test": ["scene9999_00"]}
    for name, values in split_ids.items():
        (splits / f"scannetv2_{name}.txt").write_text("\n".join(values) + "\n")
    scenes = []
    for scene_id in scene_ids:
        image = shard / scene_id / "frame_000000/color.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(scene_id.encode())
        scenes.append({"scene_id": scene_id, "official_split": "train", "frames": [
            {"frame_index": 0, "outputs": {"color.png": {"path": "frame_000000/color.png",
                "size_bytes": image.stat().st_size, "sha256": module._sha256(image)}}}]})
    document = {"protocol": "gsa-scannet-train-sens-cpu-scaleout-v1",
                "verdict": "SCALEOUT_VALIDATED_TRAIN_SPLIT_ONLY",
                "use_policy": {"training_materialization_eligible": True,
                               "official_train_only": True, "official_split": "train"},
                "official_splits": {name: {"sha256": module._sha256(splits / f"scannetv2_{name}.txt")}
                                    for name in split_ids}, "scenes": scenes}
    manifest = shard / "scaleout_manifest.json"
    def freeze():
        manifest.write_text(json.dumps(document))
        manifest.with_suffix(".sha256").write_text(module._sha256(manifest) + "\n")
    freeze()
    args = module.build_parser().parse_args(["prepare-scannet-flywheel", "--source-root", str(sources),
        "--split-root", str(splits), "--output", str(tmp_path / "prepared")])
    return module, args, document, freeze


def test_anvil_preparation_preserves_physical_scene_separation(tmp_path: Path) -> None:
    module, args, _, _ = _anvil_source_fixture(tmp_path)
    assert args.func(args) == 0
    rows = [json.loads(line) for line in (args.output / "images.jsonl").read_text().splitlines()]
    assert len(rows) == 60
    train = {r["scene_group"] for r in rows if r["split"] == "train"}
    dev = {r["scene_group"] for r in rows if r["split"] == "dev"}
    assert train and dev and not train & dev
    assert all(r["label_provenance"] == "unlabelled_source" for r in rows)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["use_policy"]["training_labels_eligible"] is False
    assert manifest["use_policy"]["benchmark_ground_truth"] is False
    with pytest.raises(SystemExit, match="existing preparation"):
        args.func(args)


@pytest.mark.parametrize("failure", ["policy", "image_hash", "heldout_scene", "manifest_hash"])
def test_anvil_preparation_rejects_unbound_or_heldout_sources(tmp_path: Path, failure: str) -> None:
    _, args, document, freeze = _anvil_source_fixture(tmp_path)
    if failure == "policy":
        document["use_policy"]["training_materialization_eligible"] = False
    elif failure == "image_hash":
        document["scenes"][0]["frames"][0]["outputs"]["color.png"]["sha256"] = "0" * 64
    elif failure == "heldout_scene":
        document["scenes"][0]["scene_id"] = "scene9998_00"
    freeze()
    if failure == "manifest_hash":
        (args.source_root / "shard_00/scaleout_manifest.sha256").write_text("0" * 64)
    with pytest.raises(SystemExit):
        args.func(args)
    assert not args.output.exists()


def test_anvil_filter_removes_whole_source_rows_without_rewriting_survivors(tmp_path: Path) -> None:
    module, _, _, _ = _anvil_source_fixture(tmp_path)
    excluded = tmp_path / "datasets/localized_narratives"
    keep = tmp_path / "datasets/localized_narratives_other"
    rows = [
        {"images": [str(excluded / "a.jpg")], "gsa_form": "qa"},
        {"images": [str(keep / "b.jpg"), str(excluded / "c.jpg")], "gsa_form": "qa"},
        {"images": [str(keep / "d.jpg")], "gsa_form": "bbox"},
    ]
    lines = [(json.dumps(row, separators=(", ", ": ")) + "\n").encode() for row in rows]
    source = tmp_path / "corpus.jsonl"
    source.write_bytes(b"".join(lines))
    args = module.build_parser().parse_args(["filter-corpus", "--input", str(source),
        "--output", str(tmp_path / "filtered"), "--expected-rows", "3",
        "--expected-sha256", module._sha256(source), "--exclude-image-root", str(excluded)])
    assert args.func(args) == 0
    assert (args.output / "train.jsonl").read_bytes() == lines[2]
    assert source.read_bytes() == b"".join(lines)
    report = json.loads((args.output / "manifest.json").read_text())
    assert report["exclusions"]["rows"] == 2
    assert report["corpus"]["rows"] == 1
    assert report["status"] == "PENDING_LEAKAGE_REAUDIT"
    args.output = tmp_path / "bad-binding"
    args.expected_sha256 = "0" * 64
    with pytest.raises(SystemExit, match="bound audit"):
        args.func(args)
    assert not args.output.exists()


def _clean_sft_fixture(tmp_path):
    from types import SimpleNamespace
    module, _, _, _ = _anvil_source_fixture(tmp_path)
    corpus, worktree, model = tmp_path / "corpus", tmp_path / "worktree", tmp_path / "model"
    corpus.mkdir()
    (worktree / "scripts").mkdir(parents=True)
    model.mkdir()
    for filename in ("config.json", "tokenizer.json"):
        (model / filename).write_text("{}")
    for filename in ("run_sft_9b_lora_l20_dlc.sh", "run_eval_adapter_dp.sh", "run_general_retention_dp.sh"):
        (worktree / "scripts" / filename).write_text("# fixture\n")
    source = tmp_path / "source.jsonl"
    lines = [json.dumps({"images": [f"image{i}.png"]}) + "\n" for i in range(3)]
    source.write_text("".join(lines))
    (corpus / "train.jsonl").write_text("".join(lines[:2]))
    (corpus / "dev.jsonl").write_text(lines[2])
    manifest = {"source": {"path": str(source), "sha256": module._sha256(source), "rows": 3},
                "checks": {key: True for key in ("row_conservation", "image_disjoint", "all_forms_in_dev")}}
    for split, rows in (("train", 2), ("dev", 1)):
        manifest[split] = {"rows": rows, "sha256": module._sha256(corpus / f"{split}.jsonl")}
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    retention, spatial = tmp_path / "retention.json", tmp_path / "spatial.json"
    retention.write_text(json.dumps({"state": "PASS", "posttraining_corpus_sha256": module._sha256(source),
                                     "posttraining_corpus_rows": 3}))
    spatial.write_text(json.dumps({"state": "BLOCKED", "contamination_verdict": "PASS", "contamination_clean": True,
                                   "posttraining_corpus": {"sha256": module._sha256(source)},
                                   "metadata_completeness_verdict": "INCOMPLETE_EVALUATION_RELEASE_METADATA"}))
    args = SimpleNamespace(corpus=corpus, worktree=worktree, model=model,
                           retention_audit=retention, spatial_audit=spatial)
    return module, args


@pytest.mark.parametrize("failure", ["changed_train", "failed_retention", "wrong_audited_source", "failed_spatial"])
def test_clean_sft_queue_rejects_data_drift_and_wrong_audits(tmp_path, failure):
    module, args = _clean_sft_fixture(tmp_path)
    good = module._clean_sft_inputs(args)
    assert good["spatial_metadata_status"] == "INCOMPLETE_EVALUATION_RELEASE_METADATA"
    if failure == "changed_train":
        with (args.corpus / "train.jsonl").open("a") as handle:
            handle.write("{}\n")
    elif failure in {"failed_retention", "wrong_audited_source"}:
        report = json.loads(args.retention_audit.read_text())
        report["state" if failure == "failed_retention" else "posttraining_corpus_sha256"] = "BLOCKED"
        args.retention_audit.write_text(json.dumps(report))
    else:
        report = json.loads(args.spatial_audit.read_text())
        report["contamination_verdict"] = "BLOCKED"
        args.spatial_audit.write_text(json.dumps(report))
    with pytest.raises(SystemExit):
        module._clean_sft_inputs(args)


def test_clean_sft_lineage_checks_actual_training_inputs_and_selected_checkpoint(tmp_path):
    module, args = _clean_sft_fixture(tmp_path)
    binding = module._clean_sft_inputs(args)
    version = tmp_path / "run/v1"
    selected = version / "checkpoint-10"
    selected.mkdir(parents=True)
    config = {"model": str(args.model), "dataset": [str(args.corpus / "train.jsonl")],
              "val_dataset": [str(args.corpus / "dev.jsonl")], "adapters": [], "resume_from_checkpoint": None,
              "seed": 17, "data_seed": 17, "learning_rate": .0001, "max_steps": 10,
              "tuner_type": "lora", "lora_rank": 64, "lora_alpha": 128, "lora_dropout": .05,
              "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4}
    (version / "args.json").write_text(json.dumps(config))
    state = {"global_step": 10, "max_steps": 10, "best_model_checkpoint": str(selected)}
    (selected / "trainer_state.json").write_text(json.dumps(state))
    (selected / "adapter_config.json").write_text("{}")
    (selected / "adapter_model.safetensors").write_bytes(b"identity fixture only")
    lineage = module._completed_sft_lineage(version, binding, args.model, 10, 17, .0001)
    assert lineage["selected_adapter"] == str(selected)
    durable = tmp_path / "durable"
    durable.mkdir()
    preserved = module._persist_selected_adapter(lineage, durable)
    assert Path(preserved["path"]).is_dir()
    for key, saved in preserved["files"].items():
        assert module._sha256(Path(saved["path"])) == lineage["files"][key]["sha256"]
    with pytest.raises(SystemExit, match="overwrite"):
        module._persist_selected_adapter(lineage, durable)
    config["resume_from_checkpoint"] = "/historical/P4/checkpoint-8000"
    (version / "args.json").write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="actual trainer configuration"):
        module._completed_sft_lineage(version, binding, args.model, 10, 17, .0001)
    changed_output = tmp_path / "changed-output"
    changed_output.mkdir()
    with pytest.raises(SystemExit, match="changed while preserving"):
        module._persist_selected_adapter(lineage, changed_output)
    assert not (changed_output / "selected_adapter").exists()
    config["resume_from_checkpoint"] = None
    (version / "args.json").write_text(json.dumps(config))
    state["best_model_checkpoint"] = str(tmp_path / "foreign/checkpoint-10")
    (selected / "trainer_state.json").write_text(json.dumps(state))
    with pytest.raises(SystemExit, match="escapes"):
        module._completed_sft_lineage(version, binding, args.model, 10, 17, .0001)


@pytest.mark.parametrize("state", ["metadata_only", "truncated", "complete"])
def test_adapter_evaluation_requires_complete_merged_shards(tmp_path, state):
    import os
    import subprocess
    save_file = pytest.importorskip("safetensors.numpy").save_file

    worktree = tmp_path / "worktree"
    (worktree / "scripts").mkdir(parents=True)
    (worktree / "scripts/run_eval_dp.sh").write_text("#!/bin/bash\nprintf 'EVAL_REACHED\\n'\n")
    adapter, merged = tmp_path / "adapter", tmp_path / "merged"
    adapter.mkdir()
    merged.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (merged / "config.json").write_text("{}")
    shard = merged / "model-00001-of-00001.safetensors"
    if state != "metadata_only":
        save_file({"weight": np.ones(4, dtype=np.float32)}, shard)
        (merged / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"weight": shard.name}}))
        if state == "truncated":
            shard.write_bytes(shard.read_bytes()[:-1])
    result = subprocess.run([
        "bash", str(ROOT / "scripts/run_eval_adapter_dp.sh"), "fixture", str(adapter), "cv_bench",
    ], env={**os.environ, "GSA_ROOT": str(worktree), "MERGED_DIR": str(merged),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "ADAPTER_RESOLVE_ONLY": "0"}, capture_output=True, text=True)
    assert (result.returncode == 0) is (state == "complete"), result.stderr
    assert ("EVAL_REACHED" in result.stdout) is (state == "complete")
