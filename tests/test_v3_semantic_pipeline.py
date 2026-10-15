from __future__ import annotations

import argparse
import json

import pytest

from trains.flywheel import prompts_ext
from trains.flywheel import generate_multiform as generator
from trains.flywheel.contracts import context_for
from trains.flywheel.to_viewer_format import adapt
from trains.flywheel.validators import validate


def test_external_teacher_cache_separates_generation_budget_and_retains_accounting(monkeypatch, tmp_path):
    import io
    from trains.flywheel import vlm_client as client
    monkeypatch.setattr(client, "_URL", "http://teacher.invalid/v1/chat/completions")
    monkeypatch.setattr(client, "_CACHE", tmp_path)
    monkeypatch.setattr(client, "_REQUEST_LOG", "")
    monkeypatch.setattr(client, "_RESPONSE_LOG", str(tmp_path / "responses.jsonl"))
    monkeypatch.setattr(client, "_image_data", lambda _: ("aGVsbG8=", "image/png", "a" * 64))
    calls = []

    def respond(request, **kwargs):
        body = json.loads(request.data)
        calls.append((request.full_url, body["max_tokens"]))
        limited = body["max_tokens"] == 1
        return io.StringIO(json.dumps({"model": "teacher", "choices": [{
            "message": {"content": '{"candidate":true}'},
            "finish_reason": "length" if limited else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1 if limited else 8}}))

    monkeypatch.setattr(client, "_open_request", respond)
    assert client.chat_json("image", "prompt", max_tokens=1) is None
    assert client.chat_json("image", "prompt", max_tokens=32) == {"candidate": True}
    assert client.chat_json("image", "prompt", max_tokens=32) == {"candidate": True}
    assert len(calls) == 2
    monkeypatch.setattr(client, "_URL", "http://different.invalid/v1/chat/completions")
    assert client.chat_json("image", "prompt", max_tokens=32) == {"candidate": True}
    assert len(calls) == 3
    records = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(records) == 3
    assert sorted(record["usage"]["completion_tokens"] for record in records) == [1, 8, 8]
    assert next(record for record in records if record["finish_reason"] == "length")["parsed"] is None
    assert client.chat_json("image", "prompt", max_tokens=32, use_cache=False) == {"candidate": True}
    assert len(calls) == 4
    responses = [json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()]
    assert len(responses) == 4 and all(row["event"] == "response" for row in responses)
    assert [row["usage"]["completion_tokens"] for row in responses] == [1, 8, 8, 8]
    assert all(row["content"] == '{"candidate":true}' for row in responses)


@pytest.fixture
def anvil_cli():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "engineering/anvil.py"
    spec = importlib.util.spec_from_file_location("anvil_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("corruption", ["bytes", "missing_variant", "wrong_source", "wrong_path"])
def test_frozen_context_pack_rejects_changed_or_misdirected_inputs(anvil_cli, tmp_path, corruption):
    source = {"path": "/frozen/qualification.jsonl", "sha256": "a" * 64}
    bindings = {}
    for variant in ("oracle", "student_cascade", "corrupted"):
        path = tmp_path / (variant + ".jsonl")
        path.write_text(json.dumps({"context": variant}) + "\n")
        bindings[path.name] = {"path": str(path), "sha256": anvil_cli._sha256(path)}
    manifest = {"status": "COMPLETE", "source": source, "files": bindings}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert anvil_cli._validate_native_context_pack(tmp_path, source) == manifest
    if corruption == "bytes":
        (tmp_path / "student_cascade.jsonl").write_text('{"context":"replaced"}\n')
    elif corruption == "missing_variant":
        del manifest["files"]["corrupted.jsonl"]
    elif corruption == "wrong_source":
        manifest["source"] = {**source, "sha256": "b" * 64}
    else:
        manifest["files"]["corrupted.jsonl"] = dict(manifest["files"]["oracle.jsonl"])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shared context"):
        anvil_cli._validate_native_context_pack(tmp_path, source)


def test_sensor_polygon_preserves_visible_shape_after_coordinate_roundtrip(anvil_cli):
    import cv2
    import numpy as np
    mask = np.zeros((301, 509), dtype=np.uint8)
    cv2.ellipse(mask, (245, 145), (111, 67), 25, 0, 360, 1, -1)
    polygon, evidence = anvil_cli._sensor_mask_polygon(mask)
    assert polygon is not None and 6 <= len(polygon) <= 48
    raster = np.zeros_like(mask)
    pixels = np.rint(np.array(polygon) * [509 / 1000, 301 / 1000]).astype(np.int32)
    cv2.fillPoly(raster, [pixels], 1)
    iou = np.logical_and(raster, mask).sum() / np.logical_or(raster, mask).sum()
    assert iou >= .95 and evidence["polygon_raster_iou"] == pytest.approx(iou)
    assert all(0 <= coordinate < 1000 for point in polygon for coordinate in point)


def test_sensor_polygon_rejects_holes_and_disconnected_instances(anvil_cli):
    import cv2
    import numpy as np
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 70, 1, -1)
    cv2.circle(mask, (100, 100), 35, 0, -1)
    polygon, evidence = anvil_cli._sensor_mask_polygon(mask)
    assert polygon is None and evidence["reason"] == "unrepresentable_holes"
    mask[:] = 0
    cv2.circle(mask, (45, 100), 30, 1, -1)
    cv2.circle(mask, (155, 100), 30, 1, -1)
    polygon, evidence = anvil_cli._sensor_mask_polygon(mask)
    assert polygon is None and evidence["reason"] == "disconnected_instance"


def test_sft_projection_preserves_messages_images_and_rejects_extra_modalities(anvil_cli):
    from copy import deepcopy
    row = {"messages": [{"role": "user", "content": "<image>Locate the cup."},
                        {"role": "assistant", "content": '<GROUND>[{"label":"cup","bbox_2d":[1,2,3,4]}]'}],
           "images": ["bound-image.png"], "gsa_payload": [{"box_2d": [1, 2, 3, 4]}],
           "gsa_context": {"heterogeneous_metadata": {"rank": 1}}}
    before = deepcopy(row)
    result = anvil_cli._sft_message_projection([row])
    assert result == [{"messages": before["messages"], "images": before["images"]}]
    assert row == before
    with pytest.raises(ValueError, match="modality/tool"):
        anvil_cli._sft_message_projection([{**row, "videos": ["clip.mp4"]}])
    with pytest.raises(ValueError, match="complete messages"):
        anvil_cli._sft_message_projection([{**row, "messages": row["messages"][:1]}])


def test_coco_native_masks_keep_train_split_binding_and_reject_cache_drift(anvil_cli, tmp_path):
    import hashlib
    import math
    import zipfile
    from pathlib import Path
    from PIL import Image
    pytest.importorskip("pycocotools.mask")
    image_cache = tmp_path / "images"
    image_cache.mkdir()
    image = image_cache / "000000000001.jpg"
    Image.new("RGB", (200, 200), "white").save(image)
    url = "https://s3.amazonaws.com/images.cocodataset.org/train2017/" + image.name
    (image_cache / "image_bindings.json").write_text(json.dumps({image.name: {
        "url": url, "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}}))
    polygon = [coordinate for i in range(40) for coordinate in (
        100 + 60 * math.cos(2 * math.pi * i / 40), 100 + 40 * math.sin(2 * math.pi * i / 40))]
    source = {"categories": [{"id": 1, "name": "cup"}],
              "images": [{"id": 1, "file_name": image.name, "width": 200, "height": 200, "coco_url": url}],
              "annotations": [{"id": 7, "image_id": 1, "category_id": 1, "iscrowd": 0,
                               "bbox": [40, 60, 120, 80], "area": 7500, "segmentation": [polygon]}]}
    archive = tmp_path / "annotations.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("annotations/instances_train2017.json", json.dumps(source))
        handle.writestr("annotations/instances_val2017.json", "not even parsed by the training builder")
    args = argparse.Namespace(worktree=Path(__file__).resolve().parents[1], annotations_zip=archive,
        expected_annotations_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        image_cache=image_cache, output=tmp_path / "candidate", limit=1, workers=1,
        seed="test-coco", dev_permille=100, direct_download=True)
    assert anvil_cli.prepare_coco_masks(args) == 0
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["records"] == 1 and not manifest["training_labels_eligible"]
    row = json.loads((args.output / "audit_corpus.jsonl").read_text())
    assert row["gsa_meta"]["source_annotation_id"] == 7
    assert row["gsa_meta"]["polygon_measurement"]["polygon_raster_iou"] >= .95
    assert "scene_group" not in row["gsa_meta"]  # No invented physical scene IDs.
    assert "bbox" not in row["messages"][1]["content"]
    first_corpus = args.output / "audit_corpus.jsonl"
    args.output = tmp_path / "higher-fidelity"
    args.mask_min_iou, args.polygon_max_vertices = .985, 96
    args.categories, args.min_box_side_px = ["cup"], 50
    assert anvil_cli.prepare_coco_masks(args) == 0
    refined = json.loads((args.output / "audit_corpus.jsonl").read_text())
    assert refined["gsa_meta"]["polygon_measurement"]["polygon_raster_iou"] >= .985
    assert len(refined["gsa_payload"][0]["polygon"]) <= 96
    args.output = tmp_path / "reserved-image"
    args.exclude_corpus = [first_corpus]
    with pytest.raises(ValueError, match="no representable native COCO masks"):
        anvil_cli.prepare_coco_masks(args)
    assert not args.output.exists()
    args.exclude_corpus = []
    args.categories = ["unsupported category"]
    with pytest.raises(ValueError, match="categories must be"):
        anvil_cli.prepare_coco_masks(args)
    args.categories = ["cup"]
    args.output = tmp_path / "changed-cache"
    Image.new("RGB", (200, 200), "black").save(image)
    with pytest.raises(ValueError, match="cached COCO image binding drift"):
        anvil_cli.prepare_coco_masks(args)
    args.expected_annotations_sha256 = "0" * 64
    with pytest.raises(ValueError, match="annotation archive binding drift"):
        anvil_cli.prepare_coco_masks(args)


def test_native_api_sends_only_public_inputs_and_keeps_cap_metadata(anvil_cli, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    received = []

    def request(**kwargs):
        received.append(kwargs)
        return SimpleNamespace(**kwargs)

    def infer(requests, config, use_tqdm):
        assert len(requests) == 1
        return [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="candidate"), finish_reason="length")],
                                usage=SimpleNamespace(prompt_tokens=31, completion_tokens=512))]

    monkeypatch.setitem(sys.modules, "swift", SimpleNamespace(InferRequest=request))
    engine = SimpleNamespace(infer=infer, args=SimpleNamespace(get_request_config=lambda: "frozen"))
    row = {"sample_id": "fixture", "images": ["public.png"], "gsa_payload": {"private": "reference"},
           "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "public question"},
                        {"role": "assistant", "content": "private answer"}]}
    blocked = {**row, "sample_id": "blocked", "gsa_eval_blocked_upstream": [{"reason": "parent_not_generated"}]}
    output = tmp_path / "predictions.jsonl"
    result = anvil_cli._native_infer_batches(engine, [row, blocked], output)
    assert received == [{"messages": row["messages"][:2], "images": ["public.png"]}]
    assert result[0]["labels"] == "private answer"
    assert result[0]["finish_reason"] == "length" and result[0]["completion_tokens"] == 512
    assert len(output.read_text().splitlines()) == 1
    with pytest.raises(FileExistsError):
        anvil_cli._native_infer_batches(engine, [row], output)


def test_native_inference_clears_lora_rank_before_swift_import(anvil_cli, tmp_path, monkeypatch):
    import os
    import sys
    from types import SimpleNamespace
    for key in anvil_cli._INFERENCE_DISTRIBUTED_ENV:
        monkeypatch.setenv(key, "64")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    model, adapter, output = (tmp_path / name for name in ("model", "adapter", "inference"))
    for path in (model, adapter, output):
        path.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture only")

    def constructor(command):
        # Swift's master-only JSONL writer otherwise suppresses RANK=64 output.
        assert "RANK" not in os.environ and "WORLD_SIZE" not in os.environ
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
        return SimpleNamespace(args=SimpleNamespace(
            model=str(model), adapters=[str(adapter)], infer_backend="transformers",
            max_batch_size=4, max_new_tokens=512, max_length=3072, temperature=0,
            seed=17, enable_thinking=False, stream=False, load_args=False))

    monkeypatch.setitem(sys.modules, "swift.pipelines.infer.infer", SimpleNamespace(SwiftInfer=constructor))
    anvil_cli._native_engine(model, adapter, output)
    assert all(key not in os.environ for key in anvil_cli._INFERENCE_DISTRIBUTED_ENV)
    assert (output / "inference_args.json").is_file()


def test_native_legacy_replay_requires_complete_exact_dependencies(anvil_cli, reviewed_native_pack):
    from trains.flywheel import contracts, validators
    labels = [json.loads(line) for line in (reviewed_native_pack.pack / "annotator_a/labels/all.jsonl").read_text().splitlines()]
    group = {row["task_form"]: [({"messages": [{"role": "assistant", "content": json.dumps(row["payload"])}]}, index)]
             for index, row in enumerate(labels[:8])}
    original = json.dumps(group, sort_keys=True)
    rows, rejected = anvil_cli._native_legacy_group(group, validators, contracts)
    assert len(rows) == 8 and not rejected
    assert json.dumps(group, sort_keys=True) == original
    depth = next(row for row in rows if row["form"] == "depth")
    assert "relative_depth\"" not in depth["instruction"]
    assert "rank\"" not in depth["instruction"]
    assert depth["context"]["spatial"] == {}
    assert len(next(row for row in rows if row["form"] == "action_cot")["upstream_source_lines"]) == 4
    changed = json.loads(group["depth"][0][0]["messages"][-1]["content"])
    changed[1]["label"] = changed[0]["label"]
    group["depth"][0][0]["messages"][-1]["content"] = json.dumps(changed)
    rows, rejected = anvil_cli._native_legacy_group(group, validators, contracts)
    assert {row["form"] for row in rows} == {"bbox", "detection", "mask", "point"}
    assert rejected["depth:nonbijective_depth_entities"] == 1
    assert rejected["action_cot:unaccepted_upstream"] == 1


def test_source_group_keeps_related_camera_views_and_trajectories_together(anvil_cli):
    group = anvil_cli._source_group
    assert group("/dataset/object_ref/abc_cam1_obj1.png") == group("/dataset/object_ref/abc_cam2_obj7.png")
    assert group("/dataset/traj7/images0/im_1.jpg") == group("/dataset/traj7/images2/im_90.jpg")
    assert group("/dataset/FloorPlan31_task1/a.png") == group("/dataset/FloorPlan31_task2/b.png")
    assert group("/unknown/a.jpg") == group("/unknown/b.jpg")
    assert group("/mount1/robopoint/images/object_ref/abc_cam1_obj1.png") == group("/mount2/robopoint/images/region_ref/abc_cam2_obj9.jpg")


def test_native_feedback_freezes_raw_prefixes_and_excludes_reserved_images(anvil_cli, reviewed_native_pack, tmp_path):
    from pathlib import Path
    from trains.flywheel.contracts import FORM_SPECS, image_sha256
    labels = [json.loads(line) for line in (reviewed_native_pack.pack / "annotator_a/labels/all.jsonl").read_text().splitlines()]
    root = tmp_path / "flywheel"
    raw = root / "shard0/raw"
    raw.mkdir(parents=True)
    for form in FORM_SPECS:
        records = []
        for row in labels:
            if row["task_form"] != form:
                continue
            records.append({"form": form, "image_path": row["image_path"],
                            "image_sha256": image_sha256(row["image_path"]), "label_version": "anvil-label-v3.5",
                            "payload": row["payload"], "gsa_context": row["gsa_context"],
                            "_verification": {"verdict": "verified", "form": form},
                            "_semantic_gate": {"scene": {"eligible": True}, "depth": {"valid": True},
                                               "manipulation": {"valid": True}}})
        payload = "".join(json.dumps(row) + "\n" for row in records)
        if form == "detection":
            payload += json.dumps(records[1]) + "\n"  # reject this dependency and its successors
            payload += "malformed historical row\n"
        (raw / f"{form}.jsonl").write_text(payload + "{partial append")
    exclusion = tmp_path / "reserved.jsonl"
    exclusion.write_text(json.dumps({"images": [labels[0]["image_path"]]}) + "\n")
    args = argparse.Namespace(worktree=Path(__file__).resolve().parents[1], output=tmp_path / "feedback",
                              exclude_corpus=[exclusion], raw_root=[root], label_version="anvil-label-v3.5",
                              max_images=20, dev_permille=100, salt="fixture")
    assert anvil_cli.prepare_native_feedback(args) == 0
    result = json.loads((args.output / "manifest.json").read_text())
    records = [json.loads(line) for line in (args.output / "audit_corpus.jsonl").read_text().splitlines()]
    assert len(records) == 11  # 8 valid forms, plus G bbox/mask/point after duplicate detection
    assert result["source_versions"]["invalid_json"] == 1
    assert result["training_eligible"] is False
    assert result["benchmark_ground_truth"] is False
    assert result["rejections"]["detection:duplicate_source"] == 1
    assert all(binding["prefix_bytes"] < binding["snapshot_bytes"] for binding in result["source_prefixes"])
    assert all(row["images"] != [labels[0]["image_path"]] for row in records)
    depth = next(row for row in records if row["gsa_form"] == "depth")
    assert depth["gsa_context"]["spatial"] == {}
    assert '"rank"' not in depth["messages"][1]["content"]
    assert len(depth["gsa_meta"]["upstream_raw_bindings"]) == 1
    original_output = args.output
    for path in raw.glob("*.jsonl"):
        with path.open("a") as handle:
            handle.write('\n{"appended_after_snapshot": true}\n')
    args.snapshot_manifest = original_output / "manifest.json"
    args.output = tmp_path / "replayed-feedback"
    assert anvil_cli.prepare_native_feedback(args) == 0
    assert (args.output / "audit_corpus.jsonl").read_bytes() == (original_output / "audit_corpus.jsonl").read_bytes()


@pytest.fixture
def reviewed_native_pack(tmp_path, monkeypatch):
    """Small historical-review fixture; the compiler's integrity checks have their own suite."""
    from eval import compile_semantic_audit as compiler
    from trains.flywheel.contracts import image_sha256

    pack = tmp_path / "pack"
    pack.mkdir()
    labels, annotations, selected = [], {"a": [], "b": []}, []
    splits = {}
    for index, split in enumerate(("train", "dev", "qualification")):
        image = tmp_path / f"image-{index}.jpg"
        image.write_bytes(f"historical-image-{index}".encode())
        digest = image_sha256(str(image))
        splits[split + "_image_sha256"] = [digest]
        contact = [350, 275]
        payloads = {
            "bbox": [{"label": "mug", "bbox_2d": [100, 100, 400, 500]}],
            "detection": [{"label": g["object_class"], "bbox_2d": g["bbox"]} for g in GROUNDINGS],
            "mask": [{"label": "mug", "polygon": [[110, 220], [150, 120], [300, 110],
                        [390, 220], [380, 400], [300, 480], [150, 470], [105, 400]]}],
            "point": [{"label": "mug", "point_2d": contact}],
            "depth": DEPTH,
            "affordance": [{"label": "handle", "affordance": "grasp", "bbox_2d": [300, 200, 390, 350],
                            "contact_point_2d": contact}],
            "dof": [{"label": "mug", "grasp_point_2d": contact, "jaw_axis_2d": [0.866, 0.5],
                     "approach": "side", "wrist_deg": 30}],
            "action_cot": [{"goal": "Pick up the mug by its handle", "steps": [
                "Approach the visible mug handle", "Grasp the handle", "Lift the mug"],
                "preconditions": ["The handle is visible"], "failure_modes": ["The grasp may slip"],
                "target_point_2d": contact}],
        }
        item = {"image_path": str(image), "image_sha256": digest, "sample_ids": {}, "label_hashes": {}}
        for form, payload in payloads.items():
            spatial = {"depth_order": DEPTH, "affordance": payloads["affordance"], "dof": payloads["dof"]}
            row = {"sample_id": f"{split}-{form}", "task_form": form, "image_path": str(image),
                   "payload": payload, "gsa_context": context_for(form, GROUNDINGS, spatial),
                   "_audit_only": form == "dof"}
            labels.append(row)
            item["sample_ids"][form] = row["sample_id"]
            item["label_hashes"][form] = compiler.content_hash(row)
            for side in annotations:
                annotations[side].append({"correct": True, "label_snapshot": row})
        selected.append(item)
    (pack / "manifest.json").write_text(json.dumps({"schema_version": 3, "selected": selected}))
    label_dir = pack / "annotator_a/labels"
    label_dir.mkdir(parents=True)
    (label_dir / "all.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))
    for side, rows in annotations.items():
        directory = pack / f"annotator_{side}/annotations"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "annotations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"image_split": splits}))
    adjudication = pack / "adjudication.jsonl"
    adjudication.write_text("")
    monkeypatch.setattr(compiler, "compile_report", lambda args: ({
        "schema_version": 3, "verdict": "BLOCKED", "errors": [], "unresolved_sample_ids": []}, 1))
    return argparse.Namespace(pack=pack, out=tmp_path / "native", adjudication=adjudication,
                              preserve_split=split_path, min_accuracy=.8, min_agreement=.8)


def test_native_reuse_preserves_holdouts_and_keeps_candidate_status(reviewed_native_pack):
    from trains.materialize_semantic_audit_v3 import materialize_reviewed_native
    args = reviewed_native_pack
    report = materialize_reviewed_native(args)
    assert report["global_generator_audit_verdict"] == "BLOCKED"
    assert report["training_eligible"] is report["teacher_eligible"] is False
    for split in ("train", "dev", "qualification"):
        rows = [json.loads(line) for line in (args.out / f"{split}.jsonl").read_text().splitlines()]
        assert {row["gsa_form"] for row in rows} == set(generator.FORM_SHARE)
        assert all(row["gsa_meta"]["split"] == split for row in rows)
        for row in rows:
            prefix = {"G": "<GROUND>", "S": "<RELATE>", "A": "<ACT>"}[row["gsa_family"]]
            assert "Return only a JSON array" not in row["messages"][1]["content"]
            assert f"Return {prefix} followed by a JSON array" in row["messages"][1]["content"]
            assert row["messages"][2]["content"].startswith(prefix)
            assert row["gsa_meta"]["linked_causal_trace"] is False
        assert next(row for row in rows if row["gsa_form"] == "dof")["gsa_meta"]["source_audit_only"]
    audited = (args.out / "audit_corpus.jsonl").read_text()
    assert '"split": "qualification"' not in audited


@pytest.mark.parametrize("decision", [False, None])
def test_native_reuse_excludes_false_or_unresolved_upstream_and_descendants(reviewed_native_pack, decision):
    from trains.materialize_semantic_audit_v3 import materialize_reviewed_native
    args = reviewed_native_pack
    for side in ("a", "b"):
        path = args.pack / f"annotator_{side}/annotations/annotations.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["label_snapshot"]["sample_id"] == "train-depth":
                row["correct"] = False if decision is False or side == "b" else True
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    # An entire missing family must fail atomically, not emit downstream A labels.
    with pytest.raises(ValueError, match="no accepted native candidates for train/S"):
        materialize_reviewed_native(args)
    assert not args.out.exists()


def test_native_reuse_rejects_label_drift(reviewed_native_pack):
    from trains.materialize_semantic_audit_v3 import materialize_reviewed_native
    args = reviewed_native_pack
    path = args.pack / "annotator_b/annotations/annotations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["label_snapshot"]["payload"][0]["bbox_2d"][0] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="label binding drift"):
        materialize_reviewed_native(args)
    assert not args.out.exists()


GROUNDINGS = [
    {"object_class": "mug", "bbox": [100, 100, 400, 500]},
    {"object_class": "plate", "bbox": [500, 250, 850, 700]},
]
DEPTH = [
    {"label": "mug", "rank": 1, "relative_depth": 0.1},
    {"label": "plate", "rank": 2, "relative_depth": 0.8},
]


def test_v3_prompts_encode_rejection_and_dependency_rules():
    assert prompts_ext.PROMPT_VERSION == "anvil-flywheel-v3.5"
    assert "multi-panel" in prompts_ext.SCENE_ELIGIBILITY
    assert "pre-drawn boxes" in prompts_ext.SCENE_ELIGIBILITY
    assert "preannotated_overlay" in prompts_ext.SCENE_ELIGIBILITY
    assert "return []" in prompts_ext.depth_order(GROUNDINGS).lower()
    action = prompts_ext.action_cot(
        GROUNDINGS[0],
        {"depth_order": DEPTH},
        [{"label": "handle", "affordance": "grasp"}],
        [{"label": "mug", "grasp_point_2d": [350, 275]}],
    )
    assert "verified manipulation dependencies" in action.lower()
    assert "attached handle" in action.lower()


def test_detection_rejects_duplicate_entity_labels():
    payload = [
        {"label": "green block", "bbox_2d": [100, 100, 300, 400]},
        {"label": "green block", "bbox_2d": [300, 120, 450, 400]},
    ]
    assert validate("detection", payload, None) == (
        False, "duplicate_entity_label")


def test_affordance_requires_part_label_and_action_verb():
    context = context_for("affordance", GROUNDINGS, {"depth_order": DEPTH})
    noun_as_verb = [{
        "label": "handle",
        "affordance": "handle",
        "bbox_2d": [300, 200, 390, 350],
        "contact_point_2d": [350, 275],
    }]
    assert validate("affordance", noun_as_verb, context) == (
        False, "invalid_affordance_verb")

    object_as_part = [{
        "label": "mug",
        "affordance": "grasp",
        "bbox_2d": [150, 150, 350, 400],
        "contact_point_2d": [250, 275],
    }]
    assert validate("affordance", object_as_part, context) == (
        False, "part_label_equals_object")


def test_verified_a_cascade_shares_contact_and_target():
    affordance = [{
        "label": "handle",
        "affordance": "grasp",
        "bbox_2d": [300, 200, 390, 350],
        "contact_point_2d": [350, 275],
    }]
    dof = [{
        "label": "mug",
        "grasp_point_2d": [350, 275],
        "jaw_axis_2d": [0.866, 0.5],
        "approach": "side",
        "wrist_deg": 30,
    }]
    action = [{
        "goal": "Pick up the mug by its handle",
        "steps": [
            "Approach the visible mug handle from the side",
            "Grasp the handle at the verified contact point",
            "Lift the mug clear of its support surface",
        ],
        "preconditions": ["The handle remains visible"],
        "failure_modes": ["The grasp may slip from the handle"],
        "target_point_2d": [350, 275],
    }]

    affordance_context = context_for(
        "affordance", GROUNDINGS, {"depth_order": DEPTH})
    assert validate("affordance", affordance, affordance_context) == (True, "")

    dof_context = context_for(
        "dof", GROUNDINGS,
        {"depth_order": DEPTH, "affordance": affordance})
    assert validate("dof", dof, dof_context) == (True, "")

    action_context = context_for(
        "action_cot", GROUNDINGS,
        {"depth_order": DEPTH, "affordance": affordance, "dof": dof})
    assert validate("action_cot", action, action_context) == (True, "")

    shifted = [dict(action[0], target_point_2d=[200, 200])]
    assert validate("action_cot", shifted, action_context) == (
        False, "target_dof_mismatch")


def test_attached_object_cannot_be_planned_as_detached_pickup():
    groundings = [
        {"object_class": "drawer handle", "bbox": [100, 100, 400, 500]},
        {"object_class": "mug", "bbox": [500, 250, 850, 700]},
    ]
    affordance = [{
        "label": "grip",
        "affordance": "pull",
        "bbox_2d": [180, 220, 330, 300],
        "contact_point_2d": [255, 260],
    }]
    dof = [{
        "label": "drawer handle",
        "grasp_point_2d": [255, 260],
        "jaw_axis_2d": [0.866, 0.5],
        "approach": "front",
        "wrist_deg": 30,
    }]
    action = [{
        "goal": "Pull and pick up the drawer handle",
        "steps": [
            "Grasp the drawer handle at the contact point",
            "Lift off the drawer handle",
        ],
        "preconditions": ["The handle is visible"],
        "failure_modes": ["The grasp may slip"],
        "target_point_2d": [255, 260],
    }]
    context = context_for(
        "action_cot", groundings,
        {"depth_order": DEPTH, "affordance": affordance, "dof": dof})
    assert validate("action_cot", action, context) == (
        False, "attached_object_detachment")


@pytest.mark.parametrize("audit_failure", [None, "affordance", "dof", "action_cot"])
def test_full_v3_cascade_emits_eight_aligned_forms(monkeypatch, tmp_path, audit_failure):
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"offline-test-image")
    responses = [
        {"eligible": True, "reason_code": "single_scene", "confidence": 0.96},
        [
            {"label": "red mug", "bbox_2d": [100, 100, 400, 600]},
            {"label": "blue towel", "bbox_2d": [500, 300, 800, 700]},
        ],
        [
            {"index": 0, "match": True, "confidence": 0.95},
            {"index": 1, "match": True, "confidence": 0.93},
        ],
        [{"label": "red mug", "point_2d": [300, 200]}],
        {"valid": True, "confidence": 0.95, "reason_code": "supported"},
        [{
            "label": "red mug",
            "polygon": [
                [110, 220], [150, 120], [300, 110], [390, 220],
                [380, 500], [300, 580], [150, 570], [105, 400],
            ],
        }],
        [
            {"label": "red mug", "rank": 1, "relative_depth": 0.2},
            {"label": "blue towel", "rank": 2, "relative_depth": 0.7},
        ],
        {"valid": True, "confidence": 0.92, "reason_code": "supported"},
        [{
            "label": "rim",
            "affordance": "grasp",
            "bbox_2d": [200, 150, 350, 250],
            "contact_point_2d": [300, 200],
        }],
        [{
            "label": "red mug",
            "grasp_point_2d": [300, 200],
            "jaw_axis_2d": [0.866, -0.5],
            "approach": "top",
            "wrist_deg": -30,
        }],
        [{
            "goal": "grasp and lift the red mug",
            "steps": [
                "approach the visible mug rim",
                "close the jaws around the rim",
                "lift the mug from the surface",
            ],
            "preconditions": ["the visible rim is unobstructed"],
            "failure_modes": ["the grasp may slip from the rim"],
            "target_point_2d": [300, 200],
        }],
        {"valid": True, "confidence": 0.94, "reason_code": "supported"},
    ]
    monkeypatch.setenv("AUDIT_CANDIDATE_MODE", "1")
    if audit_failure == "affordance":
        responses[8][0]["bbox_2d"] = [100, 100, 400, 600]
    elif audit_failure == "dof":
        responses[9][0]["wrist_deg"] = 60  # Structurally legal, wrong closing axis.
    elif audit_failure == "action_cot":
        responses[10][0]["preconditions"] = ["the robot is calibrated"]
    responses = iter(responses)

    monkeypatch.setattr(generator, "chat_json", lambda *args, **kwargs: next(responses))
    generator.counts.clear()
    generator.rejects.clear()
    quota = {form: 10 for form in generator.FORM_SHARE}
    records = generator.gen_one(str(image), quota, scene="kitchen")

    assert {form for form, _, _ in records} == set(generator.FORM_SHARE)
    swift_by_form = {form: swift for form, _, swift in records}
    if audit_failure is not None:
        action_order = ["affordance", "dof", "action_cot"]
        withheld = set(action_order[action_order.index(audit_failure):])
        assert {form for form, _, swift in records if swift is None} == withheld
        for form, raw, swift in records:
            if form in withheld:
                assert raw["_audit_only"] is True
                assert raw["_verification"]["verdict"] == "audit_only"
                if form != audit_failure:
                    assert raw["_audit_reason"].startswith("upstream_" + audit_failure + ":")
                    assert raw["_verification"]["local_verdict"] == "verified"
            else:
                assert not raw.get("_audit_only", False)
                assert swift is not None
        return
    for swift in swift_by_form.values():
        assert swift["gsa_meta"]["training_instruction_version"] == "anvil-referent-context-v1"
    assert '"red mug"' in swift_by_form["point"]["messages"][0]["content"]
    assert "most prominent" not in swift_by_form["point"]["messages"][0]["content"]
    depth_question = swift_by_form["depth"]["messages"][0]["content"]
    assert '"blue towel"' in depth_question
    assert '"rank": 1' not in depth_question
    plan_question = swift_by_form["action_cot"]["messages"][0]["content"]
    assert '"jaw_axis_2d"' in plan_question
    assert '"steps"' not in plan_question
    raw_by_form = {form: raw for form, raw, _ in records}
    assert raw_by_form["point"]["payload"][0]["point_2d"] == [300, 200]
    assert raw_by_form["point"]["_semantic_gate"]["point"]["valid"] is True
    for form, raw in raw_by_form.items():
        assert raw["form"] == form
        assert raw["label_version"] == prompts_ext.LABEL_VERSION
        assert raw["prompt_version"] == prompts_ext.PROMPT_VERSION
        assert raw["_semantic_gate"]["scene"]["eligible"] is True
    assert raw_by_form["action_cot"]["payload"][0]["target_point_2d"] == (
        raw_by_form["dof"]["payload"][0]["grasp_point_2d"])
    assert raw_by_form["dof"]["payload"][0]["grasp_point_2d"] == (
        raw_by_form["affordance"]["payload"][0]["contact_point_2d"])


@pytest.mark.parametrize("verdict,confidence", [(False, .99), (True, .7), (True, True), (True, float("nan"))])
def test_point_inside_box_requires_visible_surface_confirmation(monkeypatch, tmp_path, verdict, confidence):
    image = tmp_path / "chair.jpg"
    image.write_bytes(b"offline-test-image")
    candidate = [{"label": "chair", "point_2d": [250, 350]}]
    check = {"valid": verdict, "confidence": confidence,
             "reason_code": "supported" if verdict else "opening"}
    responses = iter([
        {"eligible": True, "reason_code": "single_scene", "confidence": .95},
        [{"label": "chair", "bbox_2d": [100, 100, 400, 600]},
         {"label": "mug", "bbox_2d": [500, 300, 800, 700]}],
        [{"index": 0, "match": True, "confidence": .95},
         {"index": 1, "match": True, "confidence": .95}],
        candidate, check, candidate, check,
    ])
    monkeypatch.setattr(generator, "chat_json", lambda *args, **kwargs: next(responses))
    generator.counts.clear()
    generator.rejects.clear()
    records = generator.gen_one(str(image), {"bbox": 10, "point": 10})
    assert [form for form, _, _ in records] == ["bbox"]
    assert generator.rejects["point:visible_surface_unconfirmed"] == 2


def test_v3_request_failure_is_retryable_not_semantic_done(monkeypatch, tmp_path):
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"offline-test-image")
    monkeypatch.setattr(generator, "chat_json", lambda *args, **kwargs: None)
    generator.counts.clear()
    generator.rejects.clear()
    with pytest.raises(RuntimeError, match="scene eligibility request failed"):
        generator.gen_one(
            str(image), {form: 10 for form in generator.FORM_SHARE})


def test_only_local_image_failures_are_permanent():
    assert generator._is_permanent_image_failure(
        ValueError("unsupported image")) is True
    assert generator._is_permanent_image_failure(
        OSError("unreadable image")) is True
    assert generator._is_permanent_image_failure(
        RuntimeError("endpoint unavailable")) is False
    transient = RuntimeError("endpoint unavailable")
    assert generator._is_terminal_image_failure(transient, 2, 3) is False
    assert generator._is_terminal_image_failure(transient, 3, 3) is True


def test_action_rejects_unobservable_robot_state():
    action_context = context_for(
        "action_cot", GROUNDINGS,
        {"depth_order": DEPTH, "affordance": [{
            "label": "handle", "affordance": "grasp",
            "bbox_2d": [150, 150, 300, 300],
            "contact_point_2d": [225, 225],
        }], "dof": [{
            "label": "mug", "grasp_point_2d": [225, 225],
            "jaw_axis_2d": [0.866, 0.5],
            "approach": "top", "wrist_deg": 30,
        }]})
    action = [{
        "goal": "grasp the mug",
        "steps": ["approach the visible handle", "close the jaws"],
        "preconditions": ["the robot is calibrated"],
        "failure_modes": ["collision with a nearby obstacle"],
        "target_point_2d": [225, 225],
    }]
    assert validate("action_cot", action, action_context) == (
        False, "unsupported_assumption")


def test_audit_candidate_mode_keeps_structural_action_only_for_review(monkeypatch):
    context = context_for(
        "action_cot", GROUNDINGS,
        {"depth_order": DEPTH, "affordance": [{
            "label": "handle", "affordance": "grasp",
            "bbox_2d": [150, 150, 300, 300],
            "contact_point_2d": [225, 225],
        }], "dof": [{
            "label": "mug", "grasp_point_2d": [225, 225],
            "jaw_axis_2d": [0.866, -0.5],
            "approach": "top", "wrist_deg": -30,
        }]})
    action = [{
        "goal": "grasp the mug",
        "steps": ["approach the visible handle", "close the jaws"],
        "preconditions": ["the robot is calibrated"],
        "failure_modes": ["collision with a nearby obstacle"],
        "target_point_2d": [225, 225],
    }]
    monkeypatch.delenv("AUDIT_CANDIDATE_MODE", raising=False)
    assert generator._accept_form("action_cot", action, context)[0] is False
    monkeypatch.setenv("AUDIT_CANDIDATE_MODE", "1")
    accepted, reason, audit_only = generator._accept_form(
        "action_cot", action, context)
    assert accepted is True
    assert audit_only is True
    assert reason == "unsupported_assumption"


def test_v3_viewer_ids_are_content_addressed_and_versioned():
    record = {
        "image_path": "/tmp/example.jpg",
        "image_sha256": "a" * 64,
        "source": "test",
        "scene": "tabletop",
        "label_version": prompts_ext.LABEL_VERSION,
        "prompt_version": prompts_ext.PROMPT_VERSION,
        "payload": [{"label": "mug", "bbox_2d": [1, 2, 3, 4]}],
        "gsa_context": {"primary": GROUNDINGS[0], "groundings": GROUNDINGS},
    }
    first = adapt(record, "bbox", 1, "v3")
    second = adapt(record, "bbox", 999, "v3")
    assert first["sample_id"] == second["sample_id"]
    assert first["sample_id"].startswith("v3_bbox_")
    assert first["label_version"] == prompts_ext.LABEL_VERSION


def test_teacher_proxy_bypass_merges_project_and_platform_host_lists(monkeypatch):
    from types import SimpleNamespace
    from trains.flywheel import vlm_client
    import urllib.request
    monkeypatch.setenv("NO_PROXY", "teacher.example")
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"no": "localhost", "http": "http://proxy.example"})
    calls = []
    def direct(handler):
        assert handler.proxies == {}
        return SimpleNamespace(open=lambda req, **kwargs: calls.append(("direct", req.host)) or "direct response")
    monkeypatch.setattr(urllib.request, "build_opener", direct)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, **kwargs: calls.append(("proxy", req.host)) or "proxy response")
    assert vlm_client._open_request(urllib.request.Request("http://teacher.example/v1")) == "direct response"
    assert vlm_client._open_request(urllib.request.Request("http://localhost/v1")) == "direct response"
    assert vlm_client._open_request(urllib.request.Request("http://external.example/v1")) == "proxy response"
    assert calls == [("direct", "teacher.example"), ("direct", "localhost"), ("proxy", "external.example")]


def test_geometry_audit_separates_orientation_frequency_from_invalid_axis(tmp_path):
    import json
    from eval.audit_flywheel import audit
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [{"label_version": "old" if i == 0 else "anvil-label-v3.5",
             "payload": [{"label": "mug", "jaw_axis_2d": axis}]}
            for i, axis in enumerate(([1, 0], [.707, .707], [0, 0], [5, 0]))]
    (raw / "dof.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = audit(tmp_path)["dof"]
    assert report["defects"] == {"zero_axis": 1, "axis_not_unit": 1}
    assert report["diagnostics"]["axis_near_diagonal"] == 1
    assert report["label_versions"] == {"old": 1, "anvil-label-v3.5": 3}
    assert report["semantic_accuracy_measured"] is False
