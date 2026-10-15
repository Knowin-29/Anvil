#!/usr/bin/env python3
"""Map pinned PRISM source annotations to a calibrated gripper-pose extension.

CPU only; no teacher calls, inferred action categories, new grasps or physics
success labels. The source-native extension is deliberately NOT an anvil-gen-v3
training receipt. The --export-sft adapter requires external scene approvals and
held-out exclusion manifests; it never creates them. Run --help for the CLI.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import yaml

CONTRACT = "anvil-prism-gripper-se3-v1"
SOURCE_REVISION = "34cfb7f6d98e019a2251cdc40c8079fbf0cccba1"
RENDERER_REVISION = "73990b5c5121ac5ab5fd402453bfeeefd270d153"
ACRONYM_REVISION = "d02f58fb3a556c3b719976e206cce0dce0103199"
LENGTH_UNIT = "prism_source_length_unit_unscaled"
SPLIT_CONTRACT = "anvil-prism-scene-assignments-v1"
EXCLUSION_CONTRACT = "anvil-prism-training-exclusions-v1"
CONVENTION_SOURCES = {
    "camera_and_grasp_transform": "https://github.com/abhaybd/GraspMolmo/blob/"
        + RENDERER_REVISION + "/graspmolmo/datagen/generate_obs.py#L181-L210",
    "gripper_approach_axis": "https://github.com/abhaybd/GraspMolmo/blob/"
        + RENDERER_REVISION + "/graspmolmo/preprocess_shapenet.py#L83-L102",
    "jaw_axis_geometry": "https://github.com/NVlabs/acronym/blob/"
        + ACRONYM_REVISION + "/acronym_tools/acronym.py",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def matrix(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"invalid {name}")
    return result


def rigid(value):
    pose = matrix(value, (4, 4), "SE3 pose")
    rotation = pose[:3, :3]
    if not (np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            and abs(np.linalg.det(rotation) - 1) <= 1e-5):
        raise ValueError("pose is not a proper rigid SE3 transform")
    return pose


def validate_native_gripper(answer: dict, intrinsics, width: int, height: int):
    """Validate full source-native geometry, NOT robot executability or success.

    Unlike legacy Grasp, this variant has no categorical top/side/front label.
    Translation retains the unscaled source unit: the pinned PRISM documentation
    does not explicitly declare a metric unit. Rotation columns are interpreted
    using the referenced PRISM raycast and ACRONYM gripper marker conventions.
    """
    required = {"contract", "entity_id", "label", "camera_from_gripper",
                "contact_point_camera", "contact_point_px", "approach_axis_camera",
                "jaw_axis_camera", "frame", "length_unit", "pixel_coordinates"}
    if set(answer) != required or answer["contract"] != CONTRACT:
        raise ValueError("unknown source-native gripper contract")
    if answer["frame"] != "camera_x_right_y_down_z_forward":
        raise ValueError("unsupported camera frame")
    if answer["length_unit"] != LENGTH_UNIT or answer["pixel_coordinates"] != "zero_based_xy":
        raise ValueError("unsupported source units")
    if not all(isinstance(answer[k], str) and answer[k].strip() for k in ["label", "entity_id"]):
        raise ValueError("missing entity reference")
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
        raise ValueError("invalid image dimensions")
    pose = rigid(answer["camera_from_gripper"])
    point = matrix(answer["contact_point_camera"], (3,), "contact")
    pixel = matrix(answer["contact_point_px"], (2,), "contact pixel")
    camera = matrix(intrinsics, (3, 3), "intrinsics")
    if not (camera[0, 0] > 0 and camera[1, 1] > 0
            and np.allclose(camera[2], [0, 0, 1], atol=1e-8, rtol=0)):
        raise ValueError("invalid pinhole intrinsics")
    if point[2] <= 0 or not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
        raise ValueError("contact is outside the camera image")
    homogeneous = camera @ point
    if not np.allclose(homogeneous[:2] / homogeneous[2], pixel, atol=.1, rtol=0):
        raise ValueError("contact projection contradicts source calibration")
    for field, axis in [("approach_axis_camera", 2), ("jaw_axis_camera", 0)]:
        vector = matrix(answer[field], (3,), field)
        if not np.allclose(vector, pose[:3, axis], atol=1e-8, rtol=0):
            raise ValueError(f"{field} contradicts the declared gripper convention")


def map_record(view: dict, grasp: dict, task: dict, bbox: list[float]) -> dict:
    """Pure deterministic mapper; caller separately binds all inputs to raw HDF5."""
    keys = ("scene_id", "view_id", "obs_id")
    if any(grasp[k] != task[k] for k in keys) or any(view[k] != grasp[k] for k in keys[:2]):
        raise ValueError("cross-scene/view/observation task join")
    if not re.fullmatch("[a-f0-9]{32}", view["scene_id"]) or not re.fullmatch(r"view_\d+", view["view_id"]) or not re.fullmatch(r"obs_\d+", grasp["obs_id"]):
        raise ValueError("invalid PRISM source key")
    if task.get("scene_path") != view["scene_id"] + ".hdf5":
        raise ValueError("task scene_path disagrees with scene identity")
    annot = yaml.safe_load(grasp["annotation_yaml"])
    if not isinstance(annot, dict) or any(not str(annot.get(k, "")).strip() for k in
            ["object_category", "object_id", "grasp_description", "annotation_id"]):
        raise ValueError("missing source annotation")
    if task["matching_grasp_desc"] != annot["grasp_description"]:
        raise ValueError("task-grasp source description mismatch")
    if not isinstance(task.get("task"), str) or not task["task"].strip():
        raise ValueError("empty annotated task")
    name = str(annot["object_category"]) + "_" + str(annot["object_id"])
    if name not in view["object_names"]:
        raise ValueError("annotated object is absent from the source view")
    bounds = matrix(bbox, (4,), "bbox")
    if not (0 <= bounds[0] < bounds[2] <= 1000 and 0 <= bounds[1] < bounds[3] <= 1000):
        raise ValueError("invalid source instance bounds")
    pose = rigid(grasp["grasp_pose_camera"])
    entity_id = "prism:" + view["scene_id"] + ":" + name
    answer = {"contract": CONTRACT, "entity_id": entity_id, "label": annot["object_category"],
              "camera_from_gripper": pose.tolist(),
              "contact_point_camera": grasp["grasp_point_camera"],
              "contact_point_px": grasp["grasp_point_px"],
              "approach_axis_camera": pose[:3, 2].tolist(), "jaw_axis_camera": pose[:3, 0].tolist(),
              "frame": "camera_x_right_y_down_z_forward", "length_unit": LENGTH_UNIT,
              "pixel_coordinates": "zero_based_xy"}
    validate_native_gripper(answer, view["intrinsics"], view["width"], view["height"])
    source_key = {k: grasp[k] for k in keys}
    identity = {**source_key, "task": task["task"], "contract": CONTRACT}
    source_id = hashlib.sha256(canonical(identity).encode()).hexdigest()
    return {"id": "prism-gripper-" + source_id, "contract": CONTRACT,
            "native_form": "dof", "family": "A", "split": "unassigned_source_only",
            "training_approved": False, "legacy_v3_admitted": False,
            "group_id": "PRISM:" + view["scene_id"],
            "images": [view["scene_id"] + "/" + view["view_id"] + ".png"],
            "input": {"task": task["task"], "intrinsics": view["intrinsics"],
                      "image_size_wh": [view["width"], view["height"]],
                      "entity": {"entity_id": entity_id, "label": annot["object_category"],
                                 "bbox_2d": bounds.tolist()}},
            "answer": answer,
            "provenance": {"dataset": "allenai/PRISM", "dataset_revision": SOURCE_REVISION,
                **source_key, "raw_sha256": view["raw_sha256"],
                "image_sha256": view["image_sha256"], "annotation_id": annot["annotation_id"],
                "annotation_sha256": hashlib.sha256(grasp["annotation_yaml"].encode()).hexdigest(),
                "task_sha256": hashlib.sha256(canonical(task).encode()).hexdigest(),
                "verification": "raw_annotation_equality_and_calibrated_geometry",
                "physics_or_execution_verified": False, "model_calls": 0}}


def read_index(path: Path, keys):
    records = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(row[k] for k in keys)
            if key in records:
                raise ValueError(f"duplicate source key in {path.name}")
            records[key] = row
    return records


def resolve_asset(source: Path, stored_path: str) -> Path:
    """Only read source-contained assets, including relocated raw exports."""
    stored = Path(stored_path)
    candidate = source / stored.parent.name / stored.name
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(source.resolve()):
        raise ValueError("source asset escapes raw root")
    return resolved


def convert(source: Path, output: Path, limit: int | None = None) -> dict:
    import h5py
    from PIL import Image
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    binding = json.loads((source / "manifest.json").read_text())
    for name in ["views.jsonl", "grasps.jsonl", "tasks.jsonl"]:
        if digest(source / name) != binding["files"][name]:
            raise ValueError(f"source manifest checksum mismatch: {name}")
    views = read_index(source / "views.jsonl", ["scene_id", "view_id"])
    grasps = read_index(source / "grasps.jsonl", ["scene_id", "view_id", "obs_id"])
    output.mkdir(parents=True, exist_ok=False)
    counts, rejections = Counter(), Counter()
    used_ids, scenes, images, observations, checked_views = set(), set(), set(), set(), {}
    handles, raw_hashes = {}, {}
    try:
        with (output / "records.jsonl").open("x") as out, (output / "rejections.jsonl").open("x") as rejected, (source / "tasks.jsonl").open() as tasks:
            for line_no, line in enumerate(tasks, 1):
                if limit is not None and counts["tasks_scanned"] >= limit:
                    break
                counts["tasks_scanned"] += 1
                task = json.loads(line)
                key = tuple(task[k] for k in ["scene_id", "view_id", "obs_id"])
                if key not in grasps or key[:2] not in views:
                    raise ValueError("task has no matching materialized observation")
                view, grasp = views[key[:2]], grasps[key]
                raw = resolve_asset(source, view["raw_hdf5"])
                if raw not in handles:
                    raw_hashes[raw] = digest(raw)
                    if raw_hashes[raw] != view["raw_sha256"]:
                        raise ValueError("raw HDF5 checksum mismatch")
                    handles[raw] = h5py.File(raw, "r")
                if grasp["raw_sha256"] != raw_hashes[raw] or view["raw_sha256"] != raw_hashes[raw]:
                    raise ValueError("source grasp/view checksum binding mismatch")
                h5_view = handles[raw][key[1]]
                if key[:2] not in checked_views:
                    image = resolve_asset(source, view["image"])
                    if digest(image) != view["image_sha256"]:
                        raise ValueError("source image checksum mismatch")
                    with Image.open(image) as png:
                        if png.size != (view["width"], view["height"]) or not np.array_equal(np.asarray(png), h5_view["rgb"][:]):
                            raise ValueError("image does not match raw rendered RGB")
                    if not np.array_equal(h5_view["cam_params"][:], view["intrinsics"]):
                        raise ValueError("intrinsics do not match raw HDF5")
                    names = [v.decode() if isinstance(v, bytes) else str(v) for v in h5_view["object_names"][:]]
                    if names != view["object_names"]:
                        raise ValueError("object inventory does not match raw HDF5")
                    seg = h5_view["seg"][:]
                    if seg.shape != (view["height"], view["width"]):
                        raise ValueError("segmentation dimensions mismatch")
                    boxes = {}
                    for index, name in enumerate(names, 1):
                        yy, xx = np.where(seg == index)
                        if len(xx):
                            boxes[name] = [float(xx.min() * 1000 / view["width"]),
                                float(yy.min() * 1000 / view["height"]),
                                float((xx.max() + 1) * 1000 / view["width"]),
                                float((yy.max() + 1) * 1000 / view["height"])]
                    checked_views[key[:2]] = boxes
                obs = h5_view[key[2]]
                for raw_key, mapped_key in [("grasp_pose", "grasp_pose_camera"),
                        ("grasp_point", "grasp_point_camera"), ("grasp_point_px", "grasp_point_px")]:
                    if not np.array_equal(obs[raw_key][:], grasp[mapped_key]):
                        raise ValueError("mapped observation disagrees with raw HDF5")
                annotation = obs["annot"][()]
                if isinstance(annotation, bytes):
                    annotation = annotation.decode()
                if annotation != grasp["annotation_yaml"]:
                    raise ValueError("annotation does not match raw HDF5")
                annot = yaml.safe_load(annotation)
                name = str(annot["object_category"]) + "_" + str(annot["object_id"])
                try:
                    if name not in checked_views[key[:2]]:
                        raise ValueError("target has no visible source instance region")
                    record = map_record(view, grasp, task, checked_views[key[:2]][name])
                    if record["id"] in used_ids:
                        raise ValueError("duplicate task-conditioned observation")
                except ValueError as exc:
                    reason = str(exc)
                    rejections[reason] += 1
                    rejected.write(canonical({"source_task_line": line_no, "key": list(key), "reason": reason}) + "\n")
                    continue
                out.write(canonical(record) + "\n")
                used_ids.add(record["id"])
                scenes.add(key[0]); images.add(key[:2]); observations.add(key)
                counts["accepted_task_records"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    result = {"state": "source_native_extension_materialized", "contract": CONTRACT,
        "counts": {**dict(counts), "unique_scenes": len(scenes), "unique_images": len(images),
                   "unique_grasp_observations": len(observations), "rejected_tasks": sum(rejections.values())},
        "rejections": dict(rejections), "source_manifest_sha256": digest(source / "manifest.json"),
        "source_files": binding["files"], "archive_sha256": binding["archive_sha256"],
        "metadata_sha256": binding["metadata_sha256"], "dataset_revision": SOURCE_REVISION,
        "convention_sources": CONVENTION_SOURCES, "length_unit": LENGTH_UNIT,
        "mapper_sha256": digest(Path(__file__)), "model_calls": 0,
        "legacy_v3_admitted": False, "benchmark_ready": False, "training_approved": False,
        "native_contract_coverage_claim": "source-native DoF extension only; not an 11-form production release",
        "remaining": ["explicit trainer adapter for full-pose variant", "scene-level train/eval exclusion audit",
                      "metric-unit confirmation before metric claims", "robot execution/physics is not verified"],
        "image_resolution": "Resolve relative images against the PRISM raw export root; no images copied.",
        "files": {name: digest(output / name) for name in ["records.jsonl", "rejections.jsonl"]}}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate_source_native_record(record: dict) -> None:
    """Check portable scene/target identity before allowing any stage conversion."""
    if (record.get("contract") != CONTRACT or record.get("native_form") != "dof"
            or record.get("family") != "A" or record.get("split") != "unassigned_source_only"
            or record.get("training_approved") is not False or record.get("legacy_v3_admitted") is not False):
        raise ValueError("not an unassigned calibrated source-native record")
    provenance, prompt = record["provenance"], record["input"]
    scene, view = provenance["scene_id"], provenance["view_id"]
    if not re.fullmatch("[a-f0-9]{32}", scene) or not re.fullmatch(r"view_\d+", view):
        raise ValueError("invalid source scene/view identity")
    if record["group_id"] != "PRISM:" + scene or record["images"] != [scene + "/" + view + ".png"]:
        raise ValueError("scene group or image reference mismatch")
    identity = {k: provenance[k] for k in ["scene_id", "view_id", "obs_id"]}
    identity.update(task=prompt["task"], contract=CONTRACT)
    expected_id = "prism-gripper-" + hashlib.sha256(canonical(identity).encode()).hexdigest()
    if record["id"] != expected_id:
        raise ValueError("source record identity mismatch")
    if (provenance["dataset"] != "allenai/PRISM" or provenance["dataset_revision"] != SOURCE_REVISION
            or provenance["verification"] != "raw_annotation_equality_and_calibrated_geometry"
            or provenance["physics_or_execution_verified"] is not False or provenance["model_calls"] != 0):
        raise ValueError("unexpected source provenance")
    for key in ["raw_sha256", "image_sha256", "annotation_sha256", "task_sha256"]:
        if not re.fullmatch("[a-f0-9]{64}", provenance[key]):
            raise ValueError("invalid source digest")
    entity = prompt["entity"]
    if entity["entity_id"] != record["answer"]["entity_id"] or entity["label"] != record["answer"]["label"]:
        raise ValueError("source target entity mismatch")
    if not entity["entity_id"].startswith("prism:" + scene + ":"):
        raise ValueError("entity belongs to a different physical scene")
    validate_native_gripper(record["answer"], prompt["intrinsics"], *prompt["image_size_wh"])


def load_training_policy(assignments_path: Path, exclusions_path: Path, source_sha: str):
    """Read caller-owned approvals; never derive or invent a data partition.

    Assignment manifest: schema, contract, source_records_sha256, assignments
    (group ID -> {split: train|validation|test|exclude, training_approved: bool}).
    Exclusion manifest: schema, contract, source_records_sha256,
    protected_group_ids, protected_image_sha256, evidence_sha256 (nonempty list
    of checksums for the externally audited held-out registries).
    These are explicit declarations, not proof that every historical dataset was
    audited. Both complete manifest checksums are retained in every SFT record.
    """
    assignments = json.loads(assignments_path.read_text())
    exclusions = json.loads(exclusions_path.read_text())
    for value, schema in [(assignments, SPLIT_CONTRACT), (exclusions, EXCLUSION_CONTRACT)]:
        if value.get("schema") != schema or value.get("contract") != CONTRACT or value.get("source_records_sha256") != source_sha:
            raise ValueError("policy contract or source checksum mismatch")
    if not isinstance(assignments.get("assignments"), dict) or not assignments["assignments"]:
        raise ValueError("explicit scene assignments are required")
    for group, assignment in assignments["assignments"].items():
        if not re.fullmatch(r"PRISM:[a-f0-9]{32}", group) or set(assignment) != {"split", "training_approved"}:
            raise ValueError("invalid physical-scene assignment")
        if assignment["split"] not in {"train", "validation", "test", "exclude"} or type(assignment["training_approved"]) is not bool:
            raise ValueError("invalid scene split or approval")
        if (assignment["split"] == "train") != assignment["training_approved"]:
            raise ValueError("train scenes require explicit approval; held-out scenes cannot be approved")
    for field, pattern in [("protected_group_ids", r"PRISM:[a-f0-9]{32}"),
                           ("protected_image_sha256", r"[a-f0-9]{64}"),
                           ("evidence_sha256", r"[a-f0-9]{64}")]:
        values = exclusions.get(field)
        if not isinstance(values, list) or any(not isinstance(v, str) or not re.fullmatch(pattern, v) for v in values) or len(set(values)) != len(values):
            raise ValueError("invalid exclusion registry")
    if not exclusions["evidence_sha256"]:
        raise ValueError("external exclusion evidence checksums are required")
    bindings = {"source_records_sha256": source_sha,
                "scene_assignments_sha256": digest(assignments_path),
                "training_exclusions_sha256": digest(exclusions_path)}
    return assignments["assignments"], exclusions, bindings


def validate_sft_record(record: dict, source_record: dict, assignment: dict,
                        exclusions: dict, bindings: dict, image_path: Path,
                        allow_heldout: bool = False) -> None:
    """Explicit native extension admission for the existing ms-swift SFT route."""
    validate_source_native_record(source_record)
    group = source_record["group_id"]
    split = assignment.get("split")
    heldout = allow_heldout and split in {"validation", "test"} and assignment.get("training_approved") is False
    if not heldout and assignment != {"split": "train", "training_approved": True}:
        raise ValueError("scene lacks explicit train approval")
    if group in exclusions["protected_group_ids"] or source_record["provenance"]["image_sha256"] in exclusions["protected_image_sha256"]:
        raise ValueError("held-out scene/image cannot enter SFT training")
    if heldout and group in exclusions.get("training_seen_group_ids", []):
        raise ValueError("historical training scene cannot enter the new held-out split")
    meta = record.get("anvil_meta", {})
    if (record.get("id") != source_record["id"] or record.get("images") != [str(image_path)]
            or meta.get("group_id") != group or meta.get("split") != split
            or meta.get("training_approved") is not (not heldout) or meta.get("native_contract") != CONTRACT
            or meta.get("schema") != "anvil-source-native-v1" or meta.get("native_form") != "dof"
            or meta.get("source_provenance") != source_record["provenance"]
            or meta.get("policy_bindings") != bindings or meta.get("legacy_v3_admitted") is not False):
        raise ValueError("SFT scene/image/contract/provenance binding mismatch")
    messages = record.get("messages", [])
    if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
        raise ValueError("expected one SFT prompt and one target")
    if not messages[0]["content"].startswith("<image>\n"):
        raise ValueError("missing SFT image placeholder")
    prompt = json.loads(messages[0]["content"][len("<image>\n"):])
    if (prompt.get("input") != source_record["input"] or prompt.get("contract") != CONTRACT
            or prompt.get("output_fields") != sorted(source_record["answer"])):
        raise ValueError("SFT prompt source mismatch")
    answer = json.loads(messages[1]["content"])
    if answer != source_record["answer"]:
        raise ValueError("SFT answer differs from calibrated source target")
    validate_native_gripper(answer, prompt["input"]["intrinsics"], *prompt["input"]["image_size_wh"])


def export_sft(source: Path, output: Path, image_root: Path, assignments_path: Path,
               exclusions_path: Path, contract: str, include_heldout: bool = False) -> dict:
    """Export locally runnable SFT only after explicit external scene approvals.

    The resulting messages/images JSONL uses the existing anvil-source-native-v1
    route supported by ms-swift --dataset. Absolute local image references make
    this a local training artifact, NOT a ready-to-publish data bundle. It never
    enters the legacy categorical DoF/MOPD admission path.
    """
    if contract != CONTRACT:
        raise ValueError("explicit full-pose contract opt-in is required")
    manifest = json.loads((source / "manifest.json").read_text())
    records_path = source / "records.jsonl"
    source_sha = digest(records_path)
    if (manifest.get("contract") != CONTRACT or manifest.get("state") != "source_native_extension_materialized"
            or manifest["files"]["records.jsonl"] != source_sha):
        raise ValueError("calibrated source manifest mismatch")
    assignments, exclusions, bindings = load_training_policy(assignments_path, exclusions_path, source_sha)
    records = [json.loads(line) for line in records_path.open()]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate source-native ID")
    groups = {r["group_id"] for r in records}
    if groups != set(assignments):
        raise ValueError("scene assignments must cover exactly the calibrated source scenes")
    for r in records:
        validate_source_native_record(r)
    held_out_images = {r["provenance"]["image_sha256"] for r in records
                       if assignments[r["group_id"]]["split"] != "train"}
    # Fail before creating an export when any explicitly approved scene leaks.
    for r in records:
        if assignments[r["group_id"]]["split"] == "train" and (
                r["group_id"] in exclusions["protected_group_ids"]
                or r["provenance"]["image_sha256"] in exclusions["protected_image_sha256"]
                or r["provenance"]["image_sha256"] in held_out_images):
            raise ValueError("held-out scene/image cannot enter SFT training")
    image_root = image_root.resolve(strict=True)
    checked_images, selected, evaluation = {}, [], []
    for source_record in records:
        group = source_record["group_id"]
        split = assignments[group]["split"]
        if split != "train" and (not include_heldout or split not in {"validation", "test"}):
            continue
        image = (image_root / source_record["images"][0]).resolve(strict=True)
        if not image.is_relative_to(image_root):
            raise ValueError("training image escapes source image root")
        if image not in checked_images:
            checked_images[image] = digest(image)
        if checked_images[image] != source_record["provenance"]["image_sha256"]:
            raise ValueError("training image hash differs from calibrated source")
        prompt = {"contract": CONTRACT, "input": source_record["input"],
            "output_fields": sorted(source_record["answer"]),
            "instruction": "Return the calibrated gripper target as one JSON object. Use camera_from_gripper, "
                "contact_point_camera, contact_point_px, approach_axis_camera and jaw_axis_camera. "
                "Keep frame camera_x_right_y_down_z_forward and length_unit prism_source_length_unit_unscaled; "
                "do not invent a categorical approach or physics-success label."}
        record = {"id": source_record["id"], "images": [str(image)],
            "messages": [{"role": "user", "content": "<image>\n" + canonical(prompt)},
                         {"role": "assistant", "content": canonical(source_record["answer"])}],
            "anvil_meta": {"schema": "anvil-source-native-v1", "native_form": "dof",
                "native_contract": CONTRACT, "source": "PRISM", "family": "A", "group_id": group,
                "group_unit": "physical_scene", "split": split, "training_approved": split == "train",
                "approval_basis": "explicit_external_scene_assignment_and_declared_exclusion_registry",
                "legacy_v3_admitted": False, "strong_model_verified": False, "human_reviewed": False,
                "verification": "raw_annotation_equality_and_calibrated_geometry",
                "source_provenance": source_record["provenance"], "policy_bindings": bindings,
                "public_release_eligible": False, "local_path_export": True}}
        validate_sft_record(record, source_record, assignments[group], exclusions, bindings, image, include_heldout)
        (selected if split == "train" else evaluation).append(record)
    if not selected:
        raise ValueError("no explicitly approved training scenes")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "train.jsonl").open("x") as out:
        for record in selected:
            out.write(canonical(record) + "\n")
    if include_heldout:
        with (output / "evaluation.jsonl").open("x") as out:
            for record in evaluation:
                out.write(canonical(record) + "\n")
    result = {"state": "explicit_scene_approved_sft_export", "contract": CONTRACT,
              "counts": {"source_records": len(records), "training_records": len(selected),
                         "training_scenes": len({r["anvil_meta"]["group_id"] for r in selected}),
                         "training_images": len({r["images"][0] for r in selected}), "nontrain_records_excluded": len(records) - len(selected)},
              "policy_bindings": bindings, "source_manifest_sha256": digest(source / "manifest.json"),
              "mapper_sha256": digest(Path(__file__)), "legacy_v3_admitted": False,
              "model_calls": 0, "benchmark_ready": False, "public_release_eligible": False,
              "training_route": "ms-swift messages/images SFT JSONL; not legacy v3 MOPD",
              "exclusion_scope": "caller-supplied explicit physical-scene and image-hash registries only",
              "files": {"train.jsonl": digest(output / "train.jsonl")}}
    if include_heldout:
        result["counts"].update(evaluation_records=len(evaluation),
            evaluation_scenes=len({r["anvil_meta"]["group_id"] for r in evaluation}),
            evaluation_images=len({r["images"][0] for r in evaluation}),
            excluded_records=len(records) - len(selected) - len(evaluation))
        result["files"]["evaluation.jsonl"] = digest(output / "evaluation.jsonl")
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def image_fingerprint(path: Path) -> dict:
    """Same RGB/pHash/dHash definitions as the existing source-scale audit."""
    import cv2
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    gray = image.convert("L")
    low = cv2.dct(np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32))[:8, :8].reshape(-1)
    phash = sum(int(v > np.median(low[1:])) << i for i, v in enumerate(low))
    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    dhash = sum(int(v) << i for i, v in enumerate((small[:, :-1] > small[:, 1:]).reshape(-1)))
    return {"sha256": digest(path), "decoded_rgb_sha256": hashlib.sha256(
        f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()).hexdigest(),
        "phash64": f"{phash:016x}", "dhash64": f"{dhash:016x}"}


class CandidateImageIndex:
    """Index the small candidate set, then stream large historical registries."""
    def __init__(self):
        self.exact = defaultdict(set)
        self.buckets = defaultdict(list)

    def add(self, fingerprint: dict, group: str):
        for key in ("sha256", "decoded_rgb_sha256"):
            self.exact[fingerprint[key]].add(group)
        p, d = int(fingerprint["phash64"], 16), int(fingerprint["dhash64"], 16)
        for i in range(5):
            self.buckets[(i, (p >> (13 * i)) & 8191)].append((p, d, group))

    def match(self, fingerprint: dict) -> set[str]:
        groups = set()
        for key in ("sha256", "decoded_rgb_sha256", "pillow_default_jpeg_sha256"):
            groups.update(self.exact.get(fingerprint.get(key), set()))
        if fingerprint.get("phash64") and fingerprint.get("dhash64"):
            p, d = int(fingerprint["phash64"], 16), int(fingerprint["dhash64"], 16)
            for i in range(5):
                groups.update(group for a, b, group in self.buckets.get((i, (p >> (13 * i)) & 8191), [])
                              if (p ^ a).bit_count() <= 4 and (d ^ b).bit_count() <= 4)
        return groups


def explicit_prism_groups(record: dict) -> set[str]:
    """Do not turn generic scene classes, decoded image IDs or filenames into scenes."""
    groups = set()
    for block in [record, record.get("anvil_meta", {}), record.get("gsa_meta", {}), record.get("provenance", {})]:
        if not isinstance(block, dict):
            continue
        values = [block.get("scene_id"), block.get("group_id"), *block.get("scene_ids", [])]
        for value in values:
            if isinstance(value, str) and re.fullmatch(r"(?:PRISM:)?[a-f0-9]{32}", value):
                groups.add(value if value.startswith("PRISM:") else "PRISM:" + value)
    return groups


def partition_sft(source: Path, output: Path, image_root: Path,
                  protected_records: list[Path], protected_fingerprints: list[Path],
                  training_fingerprints: list[Path], contract: str,
                  seed: str = "Anvil-PRISM-native-SE3-scenes-20261012", heldout_fraction: float = .2) -> dict:
    """Recompute a scene split after real protected-set and historical-pool scans.

    All known views of candidate physical scenes are checked, not just the views
    occurring in task rows. Existing benchmark JSONL images are opened and
    fingerprinted; historical fingerprint registries are streamed and hash-bound.
    Missing physical scene IDs in public benchmarks remain an explicit limitation.
    A match in ANY view excludes the whole scene from training. Historical source
    pool matches disqualify scenes from the new held-out split (not from train).
    """
    if contract != CONTRACT or not seed or not 0 < heldout_fraction < 1:
        raise ValueError("explicit contract, nonempty seed and interior held-out fraction required")
    if not protected_records or not protected_fingerprints or not training_fingerprints:
        raise ValueError("protected JSONL, protected fingerprints and historical training fingerprints are all required")
    if output.exists():
        raise FileExistsError(output)
    records_path = source / "records.jsonl"
    source_manifest = json.loads((source / "manifest.json").read_text())
    source_sha = digest(records_path)
    if source_manifest.get("contract") != CONTRACT or source_manifest["files"]["records.jsonl"] != source_sha:
        raise ValueError("calibrated source checksum mismatch")
    records = [json.loads(line) for line in records_path.open()]
    for record in records:
        validate_source_native_record(record)
    groups = {r["group_id"] for r in records}
    parents = {g: g for g in groups}

    def root(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    def union(a, b):
        x, y = sorted([root(a), root(b)])
        parents[y] = x

    image_root = image_root.resolve(strict=True)
    raw_manifest = json.loads((image_root / "manifest.json").read_text())
    views_path = image_root / "views.jsonl"
    if digest(views_path) != raw_manifest["files"]["views.jsonl"]:
        raise ValueError("raw view index checksum mismatch")
    index, candidate_views, candidate_hashes = CandidateImageIndex(), [], defaultdict(set)
    for line in views_path.open():
        view = json.loads(line)
        group = "PRISM:" + view["scene_id"]
        if group not in groups:
            continue
        image = resolve_asset(image_root, view["image"])
        fp = image_fingerprint(image)
        if fp["sha256"] != view["image_sha256"]:
            raise ValueError("candidate view image checksum mismatch")
        for matched in index.match(fp):
            union(group, matched)
        index.add(fp, group)
        candidate_hashes[group].add(fp["sha256"])
        candidate_views.append({"group_id": group, "view_id": view["view_id"], **fp})
    if {r["group_id"] for r in candidate_views} != groups:
        raise ValueError("not every candidate physical scene has source images")
    evidence, matches, protected, training_seen = [], [], set(), set()
    actual_cache = {}
    for role, paths, actual in [("protected_records", protected_records, True),
                               ("protected_fingerprints", protected_fingerprints, False),
                               ("historical_training_pool", training_fingerprints, False)]:
        for path in paths:
            before = digest(path)
            counts = Counter()
            target = training_seen if role == "historical_training_pool" else protected
            with path.open() as stream:
                for line_no, line in enumerate(stream, 1):
                    row = json.loads(line)
                    counts["rows"] += 1
                    known = explicit_prism_groups(row)
                    counts["rows_with_explicit_scene_ids"] += bool(known)
                    matched = known & groups
                    if actual:
                        images = row.get("images") or ([row["image_path"]] if row.get("image_path") else [])
                        if not images:
                            raise ValueError(f"protected JSONL row has no image: {path.name}:{line_no}")
                        for name in images:
                            asset = Path(name).resolve(strict=True)
                            if asset not in actual_cache:
                                actual_cache[asset] = image_fingerprint(asset)
                                counts["unique_images_read"] += 1
                            fp = actual_cache[asset]
                            if row.get("image_sha256") and row["image_sha256"] != fp["sha256"]:
                                raise ValueError("protected row image checksum mismatch")
                            matched.update(index.match(fp))
                    else:
                        if not row.get("sha256") or not row.get("decoded_rgb_sha256"):
                            raise ValueError("fingerprint registry lacks exact image identities")
                        matched.update(index.match(row))
                    if matched:
                        target.update(matched)
                        counts["matched_rows"] += 1
                        matches.append({"registry_sha256": before, "role": role,
                            "source_line": line_no, "matched_group_ids": sorted(matched),
                            "basis": "explicit_scene_or_file_rgb_phash4_dhash4"})
            if digest(path) != before:
                raise ValueError("protected/training registry changed during audit")
            evidence.append({"name": path.name, "role": role, "sha256": before, **dict(counts)})
    components = defaultdict(set)
    for group in groups:
        components[root(group)].add(group)
    # Similar candidate images stay together even when original source scene IDs differ.
    for members in components.values():
        if members & protected:
            protected.update(members)
        if members & training_seen:
            training_seen.update(members)
    eligible = [key for key, members in components.items() if not members & (protected | training_seen)]
    eligible.sort(key=lambda key: hashlib.sha256((seed + "\0" + key).encode()).hexdigest())
    heldout_count = max(1, int(len(eligible) * heldout_fraction)) if eligible else 0
    heldout = set().union(*(components[key] for key in eligible[:heldout_count])) if heldout_count else set()
    assignments, decisions = {}, []
    for group in sorted(groups):
        split = "exclude" if group in protected else "validation" if group in heldout else "train"
        reasons = (["existing_protected_benchmark_scene_or_image_overlap"] if group in protected else [])
        if group in training_seen:
            reasons.append("historical_source_pool_overlap_disqualifies_new_holdout")
        if split == "validation":
            reasons.append("fixed_seed_scene_component_holdout")
        elif split == "train" and not reasons:
            reasons.append("fixed_seed_scene_component_training")
        assignments[group] = {"split": split, "training_approved": split == "train"}
        decisions.append({"group_id": group, "component": root(group), "split": split, "reasons": reasons})
    if not any(a["split"] == "train" for a in assignments.values()):
        raise ValueError("no train scenes remain after measured exclusions")
    policy_dir = output.parent / (output.name + "_policy")
    policy_dir.mkdir(parents=True, exist_ok=False)
    assignment_path, exclusion_path = policy_dir / "scene_assignments.json", policy_dir / "training_exclusions.json"
    assignment_path.write_text(json.dumps({"schema": SPLIT_CONTRACT, "contract": CONTRACT,
        "source_records_sha256": source_sha, "assignments": assignments,
        "assignment_origin": "programmatic_scene_split_under_user_authorized_data_production",
        "human_reviewed": False, "seed": seed, "heldout_fraction": heldout_fraction}, indent=2) + "\n")
    exclusion_path.write_text(json.dumps({"schema": EXCLUSION_CONTRACT, "contract": CONTRACT,
        "source_records_sha256": source_sha, "protected_group_ids": sorted(protected),
        "protected_image_sha256": sorted(set().union(*(candidate_hashes[g] for g in protected))) if protected else [],
        "training_seen_group_ids": sorted(training_seen),
        "evidence_sha256": sorted({e["sha256"] for e in evidence})}, indent=2) + "\n")
    audit = {"schema": "anvil-prism-programmatic-partition-audit-v1", "source_records_sha256": source_sha,
        "views_index_sha256": digest(views_path), "seed": seed, "heldout_fraction": heldout_fraction,
        "candidate_physical_scenes": len(groups), "candidate_views_checked": len(candidate_views),
        "scene_decisions": decisions, "evidence": evidence,
        "counts": dict(Counter(a["split"] for a in assignments.values())),
        "matches": matches, "new_model_calls": 0, "human_reviewed": False,
        "limitations": ["Unknown physical scene identities in source benchmarks remain unknown.",
            "Historical fingerprint registries are hash-bound snapshots, not re-decoded full historical pools.",
            "File/RGB identity and joint pHash<=4,dHash<=4 cannot prove absence of every near duplicate.",
            "Held-out is local source-native gripper evaluation, not a completed eleven-form benchmark."]}
    (policy_dir / "partition_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    result = export_sft(source, output, image_root, assignment_path, exclusion_path, CONTRACT, include_heldout=True)
    result["partition_audit_sha256"] = digest(policy_dir / "partition_audit.json")
    result["counts"].update(total_source_scenes=len(groups), protected_scenes=len(protected),
                            historical_pool_matched_scenes=len(training_seen), all_source_views_checked=len(candidate_views))
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="PRISM raw export containing manifest/views/grasps/tasks and scene assets")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--limit", type=int, help="Only scan the first N annotated source tasks for a smoke test")
    parser.add_argument("--export-sft", action="store_true", help="Explicit stage conversion of a calibrated output into local SFT JSONL")
    parser.add_argument("--contract", choices=[CONTRACT], help="Required full-pose contract opt-in for --export-sft")
    parser.add_argument("--image-root", type=Path, help="Original PRISM raw image root for SFT")
    parser.add_argument("--scene-assignments", type=Path, help="External source-SHA-bound per-scene split/approval manifest")
    parser.add_argument("--training-exclusions", type=Path, help="External held-out scene/image exclusion manifest")
    parser.add_argument("--partition-sft", action="store_true", help="Scan real protection registries, deterministically assign scenes, and export train/held-out SFT")
    parser.add_argument("--protected-records", type=Path, action="append", default=[], help="Protected benchmark JSONL; every image is read (repeatable)")
    parser.add_argument("--protected-fingerprints", type=Path, action="append", default=[], help="Existing protected fingerprint registry (repeatable)")
    parser.add_argument("--training-fingerprints", type=Path, action="append", default=[], help="Historical training-pool fingerprint registry (repeatable)")
    parser.add_argument("--seed", default="Anvil-PRISM-native-SE3-scenes-20261012")
    parser.add_argument("--heldout-fraction", type=float, default=.2)
    args = parser.parse_args()
    if args.partition_sft:
        if args.export_sft or args.limit is not None or args.contract is None or args.image_root is None or args.scene_assignments or args.training_exclusions:
            parser.error("--partition-sft requires --contract and --image-root and cannot combine with --export-sft, --limit or supplied assignments")
        result = partition_sft(args.source, args.output, args.image_root, args.protected_records,
            args.protected_fingerprints, args.training_fingerprints, args.contract, args.seed, args.heldout_fraction)
    elif args.export_sft:
        if args.limit is not None or any(v is None for v in [args.contract, args.image_root, args.scene_assignments, args.training_exclusions]):
            parser.error("--export-sft requires --contract, --image-root, --scene-assignments and --training-exclusions; --limit is not allowed")
        result = export_sft(args.source, args.output, args.image_root,
                            args.scene_assignments, args.training_exclusions, args.contract)
    else:
        if any(v is not None for v in [args.contract, args.image_root, args.scene_assignments, args.training_exclusions]):
            parser.error("SFT policy arguments require --export-sft")
        result = convert(args.source, args.output, args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
