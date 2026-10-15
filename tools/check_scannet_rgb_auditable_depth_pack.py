#!/usr/bin/env python3
"""Validate and optionally compile the v5.1 RGB-auditable depth calibration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_depth_v50_scannet_sensor as sensor
from tools import build_scannet_rgb_auditable_depth_pack as builder
from tools import extract_scannet_val_frame_bundles as val_extractor


CANARY_PASS_STATUS = "CALIBRATION_PASS_FORMAL_40_BLOCKED"


ALLOWED_ERROR_CODES = {
    "wrong_object",
    "wrong_geometry",
    "wrong_relation_or_order",
    "invalid_action_or_affordance",
    "missing_content",
    "ambiguous_evidence",
    "unsuitable_source",
    "schema_or_format",
    "other",
}


class CheckError(RuntimeError):
    """The frozen pack or review stream violates its evidence contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CheckError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        if allow_missing:
            return []
        raise CheckError(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CheckError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise CheckError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _check_bound_artifact(
    root: Path,
    document: Any,
    expected_relative_path: str,
    section: str,
) -> None:
    if not isinstance(document, dict):
        raise CheckError(f"additional frame-root artifact lacks metadata: {section}")
    relative_path = document.get("path")
    expected_hash = document.get("sha256")
    expected_size = document.get("size_bytes")
    if (
        relative_path != expected_relative_path
        or sensor.SHA256_RE.fullmatch(str(expected_hash or "")) is None
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise CheckError(f"additional frame-root artifact contract drift: {section}")
    candidate_path = root / expected_relative_path
    candidate = candidate_path.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CheckError(
            f"additional frame-root artifact escapes root: {section}"
        ) from exc
    if (
        not candidate.is_file()
        or candidate_path.is_symlink()
        or candidate.stat().st_size != expected_size
        or sensor.file_sha256(candidate) != expected_hash
    ):
        raise CheckError(f"additional frame-root artifact drift: {section}")


def load_additional_frame_root(path: Path) -> dict[str, Any]:
    """Validate every materialized artifact in a manifest-bound val frame root."""
    root = path.resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest_sha256 = sensor.file_sha256(manifest_path)
    checksum_path = root / "manifest.sha256"
    try:
        checksum_text = checksum_path.read_text(encoding="ascii")
    except OSError as exc:
        raise CheckError(f"missing additional frame-root checksum: {checksum_path}") from exc
    if checksum_text != f"{manifest_sha256}  manifest.json\n":
        raise CheckError(f"additional frame-root checksum drift: {checksum_path}")

    scenes = manifest.get("scenes")
    payload = dict(manifest)
    integrity = payload.pop("integrity", None) or {}
    use_policy = manifest.get("use_policy") or {}
    extractor = manifest.get("extractor") or {}
    execution = manifest.get("execution") or {}
    selection = manifest.get("selection") or {}
    if not isinstance(scenes, list) or not scenes or not all(
        isinstance(row, dict) for row in scenes
    ):
        raise CheckError(f"additional frame-root scene list drift: {manifest_path}")
    scene_ids = [row.get("scene_id") for row in scenes]
    if not all(
        isinstance(scene_id, str)
        and val_extractor.sens.SCENE_RE.fullmatch(scene_id) is not None
        for scene_id in scene_ids
    ):
        raise CheckError(f"additional frame-root scene ID drift: {manifest_path}")

    stage = manifest.get("stage")
    expected_verdict = {
        "one_scene_canary": "CANARY_PASS_VAL_AUDIT_ONLY",
        "batch": "BATCH_PASS_VAL_AUDIT_ONLY",
    }.get(stage)
    if (
        manifest.get("schema") != val_extractor.SCHEMA
        or manifest.get("protocol") != val_extractor.PROTOCOL
        or expected_verdict is None
        or manifest.get("verdict") != expected_verdict
        or manifest.get("training_materialization_eligible") is not False
        or use_policy.get("held_out_benchmark_preparation_only") is not True
        or use_policy.get("may_enter_training") is not False
        or use_policy.get("official_split") != "val"
        or use_policy.get("training_materialization_eligible") is not False
        or len(scene_ids) != len(set(scene_ids))
        or integrity.get("scene_ids_sha256")
        != val_extractor.canonical_hash(scene_ids)
        or integrity.get("manifest_payload_sha256")
        != val_extractor.canonical_hash(payload)
        or execution.get("cpu_only") is not True
        or execution.get("gpu_required") is not False
        or execution.get("network_calls") != 0
        or execution.get("scene_count") != len(scenes)
        or selection.get("seed") != val_extractor.SELECTION_SEED
        or selection.get("maximum_frames_per_scene")
        != val_extractor.MAX_FRAMES_PER_SCENE
        or selection.get("selected_only_from_instance_semantic_member_intersection")
        is not True
        or selection.get("scene_unique") is not True
    ):
        raise CheckError(f"additional frame-root contract mismatch: {manifest_path}")

    for path_key, hash_key in (
        ("extractor_path", "extractor_sha256"),
        ("reader_path", "reader_sha256"),
    ):
        source_path = Path(str(extractor.get(path_key, "")))
        if (
            not source_path.is_file()
            or sensor.file_sha256(source_path) != extractor.get(hash_key)
        ):
            raise CheckError(f"additional frame-root source code drift: {path_key}")

    expected_directories: set[str] = set()
    total_frames = 0
    for scene_position, scene in enumerate(scenes):
        scene_id = scene_ids[scene_position]
        ordinal = scene.get("ordinal")
        directory_name = scene.get("output_directory")
        expected_directory = (
            f"{ordinal:06d}-scannet-{scene_id}"
            if isinstance(ordinal, int) and ordinal >= 0
            else None
        )
        frames = scene.get("frames")
        selected_indices = scene.get("selected_frame_indices")
        calibration = scene.get("calibration_outputs")
        if (
            scene.get("official_split") != "val"
            or directory_name != expected_directory
            or not isinstance(frames, list)
            or not frames
            or not all(isinstance(frame, dict) for frame in frames)
            or not isinstance(selected_indices, list)
            or not isinstance(calibration, dict)
            or set(calibration) != set(val_extractor.CALIBRATION_NAMES)
        ):
            raise CheckError(
                f"additional frame-root scene contract drift: {scene_id}"
            )
        frame_indices = [frame.get("frame_index") for frame in frames]
        if (
            frame_indices != selected_indices
            or len(frame_indices) != len(set(frame_indices))
            or not all(isinstance(index, int) and 0 <= index <= 999_999 for index in frame_indices)
            or scene.get("selected_frame_indices_sha256")
            != val_extractor.canonical_hash(selected_indices)
        ):
            raise CheckError(
                f"additional frame-root frame-index drift: {scene_id}"
            )

        expected_directories.add(directory_name)
        expected_files: set[str] = set(val_extractor.CALIBRATION_NAMES)
        for name in val_extractor.CALIBRATION_NAMES:
            relative = f"{directory_name}/{name}"
            _check_bound_artifact(root, calibration[name], relative, f"{scene_id}/{name}")
        for frame in frames:
            frame_index = frame["frame_index"]
            outputs = frame.get("outputs")
            if not isinstance(outputs, dict) or set(outputs) != {"rgb", "depth", "pose"}:
                raise CheckError(
                    f"additional frame-root frame-output drift: {scene_id}/{frame_index}"
                )
            names = {
                "rgb": f"{frame_index:06d}-rgb.png",
                "depth": f"{frame_index:06d}-depth.png",
                "pose": f"{frame_index:06d}.txt",
            }
            for kind, name in names.items():
                expected_files.add(name)
                relative = f"{directory_name}/{name}"
                _check_bound_artifact(
                    root,
                    outputs[kind],
                    relative,
                    f"{scene_id}/{frame_index}/{kind}",
                )
        scene_root = root / directory_name
        observed_files = {child.name for child in scene_root.iterdir() if child.is_file()}
        unexpected_directories = [child.name for child in scene_root.iterdir() if child.is_dir()]
        if observed_files != expected_files or unexpected_directories:
            raise CheckError(f"additional frame-root directory drift: {scene_id}")
        total_frames += len(frames)

    observed_directories = {child.name for child in root.iterdir() if child.is_dir()}
    if observed_directories != expected_directories:
        raise CheckError(f"additional frame-root scene-directory drift: {manifest_path}")
    observed_root_files = {child.name for child in root.iterdir() if child.is_file()}
    if observed_root_files != {"manifest.json", "manifest.sha256"}:
        raise CheckError(f"additional frame-root root-file drift: {manifest_path}")
    if execution.get("frame_count") != total_frames:
        raise CheckError(f"additional frame-root frame count drift: {manifest_path}")

    return {
        "path": str(root),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "manifest_payload_sha256": integrity["manifest_payload_sha256"],
        "scene_count": len(scene_ids),
        "scene_ids": sorted(scene_ids),
        "frame_count": total_frames,
        "stage": stage,
        "training_materialization_eligible": False,
    }


def _labels(pack: Path, annotator: str) -> dict[str, dict[str, Any]]:
    root = pack / f"annotator_{annotator}/labels"
    paths = sorted(root.rglob("*.jsonl")) if root.is_dir() else []
    if not paths:
        raise CheckError(f"annotator_{annotator} has no labels")
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in out:
                raise CheckError(f"invalid or duplicate sample_id in {path}: {sample_id!r}")
            out[sample_id] = row
    return out


def _latest_annotations(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        snapshot = row.get("label_snapshot")
        sample_id = snapshot.get("sample_id") if isinstance(snapshot, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise CheckError(f"annotation lacks label_snapshot.sample_id: {path}")
        latest[sample_id] = row
    return latest


def _check_vote(
    row: dict[str, Any], expected_label_hash: str, sample_id: str, side: str
) -> None:
    if not isinstance(row.get("correct"), bool):
        raise CheckError(f"reviewer {side} has non-Boolean vote: {sample_id}")
    if sensor.canonical_hash(row.get("label_snapshot")) != expected_label_hash:
        raise CheckError(f"reviewer {side} label snapshot drift: {sample_id}")
    error_code = row.get("error_code")
    comment = row.get("comment")
    if row["correct"]:
        if error_code not in (None, ""):
            raise CheckError(f"positive vote has error_code: {side}/{sample_id}")
    elif error_code not in ALLOWED_ERROR_CODES or not isinstance(comment, str) or not comment.strip():
        raise CheckError(
            f"negative vote requires valid error_code and concrete comment: {side}/{sample_id}"
        )


def validate_pack(pack: Path, *, rehash_source_artifacts: bool = True) -> dict[str, Any]:
    manifest_path = pack / "manifest.json"
    manifest = _read_json(manifest_path)
    integrity = manifest.get("integrity") or {}
    if integrity.get("manifest_payload_sha256") != builder.manifest_payload_hash(manifest):
        raise CheckError("manifest payload hash drift")
    if (
        manifest.get("schema_version") != builder.SCHEMA_VERSION
        or manifest.get("protocol_version") != builder.PACK_PROTOCOL
        or manifest.get("audit_round") != builder.AUDIT_ROUND
        or manifest.get("audit_contract") != builder.AUDIT_CONTRACT
        or manifest.get("forms") != ["depth"]
        or manifest.get("stage") not in {"canary", "formal"}
        or (manifest.get("training_materialization") or {}).get("eligible") is not False
    ):
        raise CheckError("manifest protocol contract drift")
    selected = manifest.get("selected")
    if not isinstance(selected, list) or manifest.get("images") != len(selected):
        raise CheckError("selected/image count mismatch")
    if integrity.get("selected_sha256") != sensor.canonical_hash(selected):
        raise CheckError("selected-record fingerprint drift")
    expected_count = (
        builder.CANARY_TARGET
        if manifest["stage"] == "canary"
        else builder.FORMAL_TARGET
    )
    if len(selected) != expected_count:
        raise CheckError(f"{manifest['stage']} requires exactly {expected_count} records")
    if (manifest.get("selection") or {}).get("thresholds") != builder.STRICT_THRESHOLDS:
        raise CheckError("strict threshold contract drift")

    labels_a = _labels(pack, "a")
    labels_b = _labels(pack, "b")
    expected_ids: set[str] = set()
    scene_ids: set[str] = set()
    image_ids: set[str] = set()
    source_ids: set[str] = set()
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            raise CheckError(f"selected[{index}] is not an object")
        sample_ids = item.get("sample_ids")
        if not isinstance(sample_ids, dict) or set(sample_ids) != {"depth"}:
            raise CheckError(f"selected[{index}] sample_ids drift")
        sample_id = sample_ids["depth"]
        if sample_id in expected_ids:
            raise CheckError(f"duplicate sample_id: {sample_id}")
        expected_ids.add(sample_id)
        scene_id = item.get("scene_id")
        image_hash = item.get("image_sha256")
        source_id = item.get("source_sample_id")
        if scene_id in scene_ids or image_hash in image_ids or source_id in source_ids:
            raise CheckError("pack is not scene/image/source-record unique")
        scene_ids.add(scene_id)
        image_ids.add(image_hash)
        source_ids.add(source_id)
        image_path = Path(str(item.get("image_path", "")))
        if not image_path.is_file() or sensor.file_sha256(image_path) != image_hash:
            raise CheckError(f"image missing or hash drift: {image_path}")
        expected_hash = (item.get("label_hashes") or {}).get("depth")
        for side, labels in (("A", labels_a), ("B", labels_b)):
            label = labels.get(sample_id)
            if label is None or sensor.canonical_hash(label) != expected_hash:
                raise CheckError(f"reviewer {side} label hash mismatch: {sample_id}")
            forbidden = {"sensor_measurements", "sensor_provenance", "verified_pair", "visibility_features"}
            if forbidden & set(label):
                raise CheckError(f"reviewer label exposes hidden metric truth: {sample_id}")
            if (
                label.get("image_sha256") != image_hash
                or label.get("scene_id") != scene_id
                or label.get("task_form") != "depth"
                or label.get("depth_scope") != "pairwise"
                or (label.get("training_materialization") or {}).get("eligible") is not False
            ):
                raise CheckError(f"reviewer label contract drift: {sample_id}")

    if set(labels_a) != expected_ids or set(labels_b) != expected_ids:
        raise CheckError("packed labels do not exactly cover the manifest")
    if len(scene_ids) != expected_count or len(image_ids) != expected_count:
        raise CheckError("scene/image uniqueness count mismatch")
    exclusion = manifest.get("exclusion_checks") or {}
    if (
        exclusion.get("old_schema5_scene_overlap") != []
        or exclusion.get("frozen_canary_scene_overlap") != []
        or exclusion.get("old_schema5_and_canary_overlap_count") != 0
    ):
        raise CheckError("selected scenes overlap a frozen prior audit")
    qualification = manifest.get("qualification") or {}
    if manifest["stage"] == "canary" and (
        qualification.get("formal_qualification_eligible") is not False
        or qualification.get("canary_can_establish_formal_pass") is not False
    ):
        raise CheckError("canary incorrectly claims formal qualification")

    provenance_doc = manifest.get("provenance") or {}
    for path_key, hash_key in (
        ("builder_path", "builder_sha256"),
        ("source_builder_path", "source_builder_sha256"),
    ):
        bound_path = Path(str(provenance_doc.get(path_key, "")))
        if (
            not bound_path.is_file()
            or sensor.file_sha256(bound_path) != provenance_doc.get(hash_key)
        ):
            raise CheckError(f"bound builder missing or hash drift: {path_key}")

    if manifest["stage"] == "formal":
        binding = provenance_doc.get("reviewed_rgb_canary_binding")
        inventory_doc = manifest.get("inventory") or {}
        inventory_binding = inventory_doc.get("reviewed_rgb_canary_binding")
        if not isinstance(binding, dict) or inventory_binding != binding:
            raise CheckError("formal pack lacks one exact reviewed-canary binding")
        try:
            live_binding = load_reviewed_canary_binding(
                Path(str(binding.get("pack_path", ""))),
                Path(str(binding.get("report_path", ""))),
                rehash_source_artifacts=False,
            )
        except (CheckError, sensor.SourceError) as exc:
            raise CheckError(f"formal reviewed-canary binding failed: {exc}") from exc
        if live_binding != binding:
            raise CheckError("formal reviewed-canary binding drift")
        formal_builder_path = Path(str(provenance_doc.get("formal_builder_path", "")))
        if (
            not formal_builder_path.is_file()
            or sensor.file_sha256(formal_builder_path)
            != provenance_doc.get("formal_builder_sha256")
        ):
            raise CheckError("formal builder missing or hash drift")
        reviewed_scenes = set(binding.get("scene_ids") or [])
        excluded_scenes = set(
            (manifest.get("inventory") or {}).get("excluded_scene_ids") or []
        )
        if len(reviewed_scenes) != builder.CANARY_TARGET or not reviewed_scenes <= excluded_scenes:
            raise CheckError("reviewed canary scenes are not fully excluded")
        if scene_ids & reviewed_scenes:
            raise CheckError("formal selection overlaps the reviewed RGB canary")
        if (
            exclusion.get("reviewed_rgb_canary_scene_overlap") != []
            or exclusion.get("all_prior_audit_scene_overlap_count") != 0
            or qualification.get("reviewed_rgb_canary_calibration_passed") is not True
            or qualification.get("formal_pack_ready_for_human_audit") is not True
            or qualification.get("formal_qualification_eligible") is not False
        ):
            raise CheckError("formal reviewed-canary gate metadata drift")
        provenance_roots = provenance_doc.get("additional_frame_root_bindings")
        inventory_roots = inventory_doc.get("additional_frame_root_bindings")
        if (
            not isinstance(provenance_roots, list)
            or provenance_roots != inventory_roots
            or not all(isinstance(item, dict) for item in provenance_roots)
        ):
            raise CheckError("formal pack lacks exact additional frame-root bindings")
        for expected_root in provenance_roots:
            live_root = load_additional_frame_root(
                Path(str(expected_root.get("path", "")))
            )
            if live_root != expected_root:
                raise CheckError("formal additional frame-root binding drift")
    provenance_path = pack / str(provenance_doc.get("source_records_path", ""))
    if (
        not provenance_path.is_file()
        or sensor.file_sha256(provenance_path) != provenance_doc.get("source_records_sha256")
    ):
        raise CheckError("source provenance file missing or hash drift")
    rows = _read_jsonl(provenance_path)
    if len(rows) != expected_count:
        raise CheckError("source provenance record count mismatch")
    by_id = {row.get("sample_id"): row for row in rows}
    if set(by_id) != expected_ids:
        raise CheckError("source provenance IDs do not match selected IDs")
    file_hash_cache: dict[str, str] = {}
    member_hash_cache: dict[tuple[str, str], str] = {}
    source_artifact_error_count = 0
    for item in selected:
        sample_id = item["sample_ids"]["depth"]
        document = by_id[sample_id]
        source_record = document.get("source_record")
        if not isinstance(source_record, dict):
            raise CheckError(f"missing source record: {sample_id}")
        source_hash = sensor.canonical_hash(source_record)
        if (
            source_hash != item.get("source_record_sha256")
            or source_hash != document.get("source_record_sha256")
            or document.get("source_sample_id") != item.get("source_sample_id")
        ):
            raise CheckError(f"source record binding mismatch: {sample_id}")
        source_errors = sensor.validate_record(source_record)
        if source_errors:
            raise CheckError(f"source record invalid {sample_id}: {source_errors}")
        features, reasons = builder.candidate_features(source_record)
        if reasons or features != item.get("rgb_auditability_features"):
            raise CheckError(f"strict candidate evidence drift: {sample_id}/{reasons}")
        if rehash_source_artifacts:
            artifact_errors = sensor.source_artifact_errors(
                source_record, file_hash_cache, member_hash_cache
            )
            source_artifact_error_count += len(artifact_errors)
            if artifact_errors:
                raise CheckError(f"source artifact drift {sample_id}: {artifact_errors}")

    return {
        "status": "PACK_VALID",
        "pack": str(pack.resolve()),
        "manifest_sha256": sensor.file_sha256(manifest_path),
        "manifest_payload_sha256": integrity["manifest_payload_sha256"],
        "stage": manifest["stage"],
        "records": expected_count,
        "unique_scenes": len(scene_ids),
        "unique_images": len(image_ids),
        "old_schema5_scene_overlap": 0,
        "frozen_canary_scene_overlap": 0,
        "source_artifacts_rehashed": rehash_source_artifacts,
        "source_artifact_errors": source_artifact_error_count,
        "formal_capacity_available": qualification.get("formal_capacity_available"),
        "formal_qualification_eligible": qualification.get(
            "formal_qualification_eligible"
        ),
    }


def compile_reviews(pack: Path, pack_report: dict[str, Any], adjudication: Path | None) -> dict[str, Any]:
    manifest = _read_json(pack / "manifest.json")
    selected = manifest["selected"]
    labels = _labels(pack, "a")
    expected_ids = {item["sample_ids"]["depth"] for item in selected}
    a = _latest_annotations(pack / "annotator_a/annotations/annotations.jsonl")
    b = _latest_annotations(pack / "annotator_b/annotations/annotations.jsonl")
    if set(a) != expected_ids or set(b) != expected_ids:
        raise CheckError("A/B annotations must exactly cover all calibration samples")
    keys_a = {row.get("annotator_key") for row in a.values()}
    keys_b = {row.get("annotator_key") for row in b.values()}
    if len(keys_a) != 1 or len(keys_b) != 1 or keys_a & keys_b:
        raise CheckError("A/B must use two distinct, non-empty annotator keys")
    label_hashes = {sample_id: sensor.canonical_hash(label) for sample_id, label in labels.items()}
    for sample_id in sorted(expected_ids):
        _check_vote(a[sample_id], label_hashes[sample_id], sample_id, "A")
        _check_vote(b[sample_id], label_hashes[sample_id], sample_id, "B")

    disagreements = sorted(
        sample_id for sample_id in expected_ids if a[sample_id]["correct"] != b[sample_id]["correct"]
    )
    c: dict[str, dict[str, Any]] = {}
    if disagreements:
        if adjudication is None:
            raise CheckError(
                f"{len(disagreements)} disagreements require --adjudication JSONL"
            )
        rows = _read_jsonl(adjudication)
        for row in rows:
            sample_id = row.get("sample_id")
            if sample_id in c:
                raise CheckError(f"duplicate adjudication: {sample_id}")
            c[sample_id] = row
        if set(c) != set(disagreements):
            raise CheckError("adjudication must cover exactly every A/B disagreement")
        reviewer_keys = {row.get("reviewer_key") for row in c.values()}
        if len(reviewer_keys) != 1 or reviewer_keys & (keys_a | keys_b):
            raise CheckError("reviewer C must use one key distinct from A/B")
        for sample_id, row in c.items():
            if (
                not isinstance(row.get("correct"), bool)
                or row.get("label_sha256") != label_hashes[sample_id]
            ):
                raise CheckError(f"invalid adjudication decision: {sample_id}")
            if row["correct"]:
                if row.get("error_code") not in (None, ""):
                    raise CheckError(f"positive adjudication has error_code: {sample_id}")
            elif (
                row.get("error_code") not in ALLOWED_ERROR_CODES
                or not isinstance(row.get("comment"), str)
                or not row["comment"].strip()
            ):
                raise CheckError(f"negative adjudication lacks evidence: {sample_id}")
    elif adjudication is not None and _read_jsonl(adjudication):
        raise CheckError("adjudication supplied but A/B have no disagreements")

    resolved: dict[str, bool] = {}
    for sample_id in expected_ids:
        resolved[sample_id] = (
            bool(a[sample_id]["correct"])
            if a[sample_id]["correct"] == b[sample_id]["correct"]
            else bool(c[sample_id]["correct"])
        )
    agreement = (len(expected_ids) - len(disagreements)) / len(expected_ids)
    correctness = sum(resolved.values()) / len(resolved)
    passed = agreement >= 0.80 and correctness >= 0.80
    formal_passed = passed and manifest["stage"] == "formal"
    return {
        **pack_report,
        "formal_qualification_eligible": formal_passed,
        "status": (
            "CALIBRATION_PASS_FORMAL_40_BLOCKED"
            if passed and manifest["stage"] == "canary"
            else "FORMAL_PASS"
            if passed
            else "BLOCKED"
        ),
        "review": {
            "expected_per_annotator": len(expected_ids),
            "a_complete": len(a),
            "b_complete": len(b),
            "agreement_count": len(expected_ids) - len(disagreements),
            "disagreement_count": len(disagreements),
            "raw_agreement": agreement,
            "resolved_correct_count": sum(resolved.values()),
            "resolved_correctness": correctness,
            "calibration_gate_passed": passed,
            "formal_40_gate_passed": formal_passed,
        },
    }


def _annotation_identity(path: Path) -> tuple[str, str]:
    rows = _read_jsonl(path)
    keys = {row.get("annotator_key") for row in rows}
    if (
        len(rows) != builder.CANARY_TARGET
        or len(keys) != 1
        or not all(isinstance(key, str) and key.strip() for key in keys)
    ):
        raise CheckError(f"canary annotation identity mismatch: {path}")
    return str(next(iter(keys))), sensor.file_sha256(path)


def load_reviewed_canary_binding(
    pack: Path,
    report_path: Path,
    *,
    rehash_source_artifacts: bool = False,
) -> dict[str, Any]:
    """Recompile and bind the frozen canary before formal selection or checking."""
    pack = pack.resolve()
    report_path = report_path.resolve()
    manifest_path = pack / "manifest.json"
    manifest = _read_json(manifest_path)
    persisted = _read_json(report_path)

    pack_report = validate_pack(pack, rehash_source_artifacts=rehash_source_artifacts)
    compiled = compile_reviews(pack, pack_report, adjudication=None)
    comparable_compiled = dict(compiled)
    if not rehash_source_artifacts:
        comparable_compiled["source_artifacts_rehashed"] = True
    if comparable_compiled != persisted:
        raise CheckError(
            "reviewed RGB canary report does not match a fresh compilation"
        )

    selected = manifest.get("selected") or []
    scenes = [item.get("scene_id") for item in selected if isinstance(item, dict)]
    images = [item.get("image_sha256") for item in selected if isinstance(item, dict)]
    sources = [
        item.get("source_sample_id") for item in selected if isinstance(item, dict)
    ]
    selection = manifest.get("selection") or {}
    review = compiled.get("review") or {}
    if (
        manifest.get("schema_version") != builder.SCHEMA_VERSION
        or manifest.get("protocol_version") != builder.PACK_PROTOCOL
        or manifest.get("audit_round") != builder.AUDIT_ROUND
        or manifest.get("audit_contract") != builder.AUDIT_CONTRACT
        or manifest.get("stage") != "canary"
        or manifest.get("forms") != ["depth"]
        or manifest.get("images") != builder.CANARY_TARGET
        or manifest.get("expected_decisions_per_annotator")
        != builder.CANARY_TARGET
        or len(selected) != builder.CANARY_TARGET
        or len(set(scenes)) != builder.CANARY_TARGET
        or len(set(images)) != builder.CANARY_TARGET
        or len(set(sources)) != builder.CANARY_TARGET
        or selection.get("rule")
        != "strict_filter_best_frame_per_scene_then_stable_hash"
        or selection.get("seed") != builder.SELECTION_SEED
        or selection.get("target") != builder.CANARY_TARGET
        or selection.get("unique_scene_required") is not True
        or selection.get("unique_image_required") is not True
        or selection.get("selected_unique_scenes") != builder.CANARY_TARGET
        or selection.get("selected_unique_images") != builder.CANARY_TARGET
        or selection.get("thresholds") != builder.STRICT_THRESHOLDS
        or compiled.get("status") != CANARY_PASS_STATUS
        or compiled.get("stage") != "canary"
        or compiled.get("records") != builder.CANARY_TARGET
        or compiled.get("unique_scenes") != builder.CANARY_TARGET
        or compiled.get("unique_images") != builder.CANARY_TARGET
        or compiled.get("source_artifact_errors") != 0
        or persisted.get("source_artifact_errors") != 0
        or persisted.get("source_artifacts_rehashed") is not True
        or review.get("a_complete") != builder.CANARY_TARGET
        or review.get("b_complete") != builder.CANARY_TARGET
        or review.get("calibration_gate_passed") is not True
        or review.get("formal_40_gate_passed") is not False
        or float(review.get("raw_agreement", -1.0)) < 0.80
        or float(review.get("resolved_correctness", -1.0)) < 0.80
    ):
        raise CheckError("reviewed RGB canary contract mismatch")

    path_a = pack / "annotator_a/annotations/annotations.jsonl"
    path_b = pack / "annotator_b/annotations/annotations.jsonl"
    key_a, hash_a = _annotation_identity(path_a)
    key_b, hash_b = _annotation_identity(path_b)
    if key_a == key_b:
        raise CheckError("canary reviewer keys are not independent")

    return {
        "pack_path": str(pack),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sensor.file_sha256(manifest_path),
        "manifest_payload_sha256": manifest["integrity"][
            "manifest_payload_sha256"
        ],
        "report_path": str(report_path),
        "report_sha256": sensor.file_sha256(report_path),
        "status": CANARY_PASS_STATUS,
        "scene_count": builder.CANARY_TARGET,
        "scene_ids": sorted(scenes),
        "annotation_streams": {
            "a": {
                "path": str(path_a.resolve()),
                "sha256": hash_a,
                "annotator_key": key_a,
            },
            "b": {
                "path": str(path_b.resolve()),
                "sha256": hash_b,
                "annotator_key": key_b,
            },
        },
        "raw_agreement": review["raw_agreement"],
        "resolved_correctness": review["resolved_correctness"],
        "source_artifacts_rehashed": True,
        "source_artifact_errors": 0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=builder.DEFAULT_OUT)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--skip-source-artifact-rehash",
        action="store_true",
        help="development-only structural check; a frozen handoff must not use this flag",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_pack(
            args.pack, rehash_source_artifacts=not args.skip_source_artifact_rehash
        )
        if args.compile:
            report = compile_reviews(args.pack, report, args.adjudication)
        if args.out is not None:
            if args.out.exists():
                raise CheckError(f"refusing to overwrite report: {args.out}")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("status") not in {"BLOCKED"} else 2
    except (CheckError, builder.PackError, sensor.SourceError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
