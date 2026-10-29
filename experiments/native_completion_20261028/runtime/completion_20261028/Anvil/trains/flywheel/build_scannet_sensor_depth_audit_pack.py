#!/usr/bin/env python3
"""Freeze 40 held-out ScanNet sensor-depth cases for blind A/B audit.

This is an audit/benchmark artifact builder, not a training-data builder.  It
accepts only the frozen v5.0 sensor source report and official ScanNet val/test
records, revalidates every referenced artifact, and projects the sensor record
to the public depth target contract.  Metric measurements remain provenance;
they are never part of the assistant target exposed to reviewers.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.build_depth_v50_scannet_sensor import (
    AUDIT_ROUND as SOURCE_AUDIT_ROUND,
    CANARY_TARGET as SOURCE_CANARY_RECORDS,
    FULL_TARGET as SOURCE_MIN_RECORDS,
    LABEL_VERSION,
    MAX_FRAMES_PER_SCENE,
    MAX_RECORDS_PER_SCENE,
    MIN_FULL_SCENES,
    PROTOCOL as SOURCE_PROTOCOL,
    SCANNET_CANARY_SCENES_SHA256,
    SCANNET_LABEL_MAP_SHA256,
    SCANNET_SOURCE_EXCLUSIONS_SHA256,
    SCANNET_SPLIT_REVISION,
    SCANNET_SPLIT_SHA256,
    SOURCE_ARTIFACT_SCHEMA,
    canonical_hash,
    file_sha256,
    validate_record,
    visual_quality_contract,
)
from trains.flywheel.contracts import CONTRACT_VERSION, context_for, grounding
from trains.flywheel.validators import validate
SCHEMA_VERSION = 5
FORM = "depth"
FORMS = [FORM]
DEPTH_SCOPE = "pairwise"
REQUIRED_IMAGES = 40
MIN_SOURCE_SCENES = 60
PACK_PROTOCOL = SOURCE_AUDIT_ROUND
PACK_ROUND = "v5.0-scannet-sensor-depth-heldout-v1"
AUDIT_CONTRACT = "scannet_sensor_pairwise"
DEFAULT_SOURCE_REPORT = (
    ROOT / "data/depth_v50_scannet_sensor/full_v3/generation_report.json"
)
DEFAULT_SOURCE_LABELS = (
    ROOT / "data/depth_v50_scannet_sensor/full_v3/labels/S_spatial/depth.jsonl"
)
DEFAULT_OUT = ROOT / "data/semantic_audit_v5_scannet_sensor_depth"
SOURCE_BUILDER_PATH = ROOT / "tools/build_depth_v50_scannet_sensor.py"
SHA256_CHARS = frozenset("0123456789abcdef")
FULL_SELECTION_RULE = "scan64_visual_top2_per_scene_then_scene_round_robin"
CANARY_SELECTION_RULE = (
    "frozen_canary_scenes_scan64_then_visual_auditability_top1"
)
PACK_SELECTION_RULE = (
    "difficulty_stratified_all_source_rows_scene_unique_"
    "round_robin_official_scannet_val"
)


def expected_generation_parameters(
    *, target: int, max_records_per_scene: int, minimum_scenes: int
) -> dict[str, Any]:
    return {
        "target": target,
        "max_frames_per_scene": MAX_FRAMES_PER_SCENE,
        "max_records_per_scene": max_records_per_scene,
        "minimum_accepted_scenes": minimum_scenes,
        "frozen_canary_scene_manifest_sha256": SCANNET_CANARY_SCENES_SHA256,
        "label_map_sha256": SCANNET_LABEL_MAP_SHA256,
        "visual_quality_contract": visual_quality_contract(),
    }


@dataclass(frozen=True)
class SourceRow:
    line_number: int
    record: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class SourceBundle:
    report_path: Path
    labels_path: Path
    report: dict[str, Any]
    rows: tuple[SourceRow, ...]


def content_hash(value: Any) -> str:
    return canonical_hash(value)


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in SHA256_CHARS for character in value)
    )


def manifest_payload_hash(manifest: dict[str, Any]) -> str:
    payload = copy.deepcopy(manifest)
    payload.pop("integrity", None)
    return content_hash(payload)


def source_inventory_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Expose the frozen OpenEQA/ScanNet split scope without source paths."""
    inventory = report.get("source_inventory") or {}
    discovered = inventory.get("scenes_by_official_split") or {}
    ready = inventory.get("ready_scenes_by_official_split") or {}
    generated = report.get("accepted_by_official_split") or {}

    def all_splits(values: dict[str, Any]) -> dict[str, Any]:
        return {split: values.get(split, 0) for split in ("train", "val", "test")}

    return {
        "openeqa_scenes_by_official_split": all_splits(discovered),
        "ready_source_scenes_by_official_split": all_splits(ready),
        "generated_source_scenes_by_official_split": {
            "train": 0,
            "val": report.get("accepted_scenes"),
            "test": 0,
        },
        "generated_source_records_by_official_split": all_splits(generated),
        "source_inventory_sha256": content_hash(inventory),
        "official_test_used_for_calibration": False,
    }


def _read_jsonl(path: Path, errors: list[str]) -> list[SourceRow]:
    rows: list[SourceRow] = []
    if not path.is_file():
        errors.append(f"source labels missing: {path}")
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path}:{line_number}: source record is not an object")
            continue
        rows.append(SourceRow(line_number, value, content_hash(value)))
    return rows


def _verify_file_document(
    value: Any,
    label: str,
    errors: list[str],
    hash_cache: dict[Path, str] | None = None,
) -> Path | None:
    if not isinstance(value, dict):
        errors.append(f"{label} provenance is not an object")
        return None
    path_value = value.get("path")
    expected = value.get("sha256")
    if not isinstance(path_value, str) or not path_value or not is_sha256(expected):
        errors.append(f"{label} provenance lacks path/sha256")
        return None
    path = Path(path_value).resolve()
    if not path.is_file():
        errors.append(f"{label} artifact missing: {path}")
        return path
    observed = (
        hash_cache.setdefault(path, file_sha256(path))
        if hash_cache is not None and path not in hash_cache
        else hash_cache[path]
        if hash_cache is not None
        else file_sha256(path)
    )
    if observed != expected:
        errors.append(f"{label} artifact hash drift: {path}")
    return path


def _verify_zip_document(
    value: Any,
    label: str,
    errors: list[str],
    hash_cache: dict[Path, str] | None = None,
    member_cache: dict[tuple[Path, str], str] | None = None,
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} provenance is not an object")
        return
    zip_value = value.get("zip_path")
    member = value.get("member")
    member_hash = value.get("sha256")
    zip_hash = value.get("zip_sha256")
    if (
        not isinstance(zip_value, str)
        or not zip_value
        or not isinstance(member, str)
        or not member
        or not is_sha256(member_hash)
        or not is_sha256(zip_hash)
    ):
        errors.append(f"{label} ZIP provenance lacks path/member hashes")
        return
    path = Path(zip_value).resolve()
    if not path.is_file():
        errors.append(f"{label} ZIP missing: {path}")
        return
    observed_zip_hash = (
        hash_cache.setdefault(path, file_sha256(path))
        if hash_cache is not None and path not in hash_cache
        else hash_cache[path]
        if hash_cache is not None
        else file_sha256(path)
    )
    if observed_zip_hash != zip_hash:
        errors.append(f"{label} ZIP hash drift: {path}")
        return
    cache_key = (path, member)
    observed_member_hash = member_cache.get(cache_key) if member_cache is not None else None
    if observed_member_hash is None:
        try:
            with ZipFile(path) as archive:
                payload = archive.read(member)
        except (OSError, BadZipFile, KeyError) as exc:
            errors.append(f"{label} ZIP member unreadable: {path}!{member} ({exc})")
            return
        observed_member_hash = hashlib.sha256(payload).hexdigest()
        if member_cache is not None:
            member_cache[cache_key] = observed_member_hash
    if observed_member_hash != member_hash:
        errors.append(f"{label} ZIP member hash drift: {path}!{member}")


def _load_split_documents(
    report: dict[str, Any],
    errors: list[str],
    hash_cache: dict[Path, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    contract = report.get("source_contract")
    documents = contract.get("official_split_manifests") if isinstance(contract, dict) else None
    if not isinstance(documents, dict) or set(documents) != {"train", "val", "test"}:
        errors.append("source report requires all official ScanNet split manifests")
        return {}, {}
    memberships: dict[str, set[str]] = {}
    seen_scenes: set[str] = set()
    for name in ("train", "val", "test"):
        document = documents.get(name)
        if not isinstance(document, dict):
            errors.append(f"official {name} split document is invalid")
            continue
        if (
            document.get("name") != name
            or document.get("sha256") != SCANNET_SPLIT_SHA256[name]
            or document.get("source_revision") != SCANNET_SPLIT_REVISION
        ):
            errors.append(f"official {name} split identity drift")
        path = _verify_file_document(
            document, f"official {name} split", errors, hash_cache
        )
        if path is None or not path.is_file():
            continue
        scene_ids = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if document.get("scene_count") != len(scene_ids) or not scene_ids:
            errors.append(f"official {name} split count drift")
        overlap = seen_scenes & scene_ids
        if overlap:
            errors.append(f"official ScanNet split overlap: {sorted(overlap)[0]}")
        seen_scenes.update(scene_ids)
        memberships[name] = scene_ids
    return documents, memberships


def _validate_source_exclusions(
    report: dict[str, Any],
    split_memberships: dict[str, set[str]],
    errors: list[str],
    hash_cache: dict[Path, str] | None = None,
) -> set[str]:
    binding = report.get("validated_source_exclusions")
    if not isinstance(binding, dict):
        errors.append("source report lacks validated_source_exclusions")
        return set()
    if binding.get("manifest_sha256") != SCANNET_SOURCE_EXCLUSIONS_SHA256:
        errors.append("source exclusion manifest is not the code-pinned revision")
    manifest_doc = {
        "path": binding.get("manifest_path"),
        "sha256": binding.get("manifest_sha256"),
    }
    manifest_path = _verify_file_document(
        manifest_doc, "source exclusion manifest", errors, hash_cache
    )
    rows = binding.get("exclusions")
    if (
        binding.get("schema") != "gsa-scannet-source-exclusions-v1"
        or binding.get("dataset") != "ScanNet v2"
        or not isinstance(rows, list)
        or binding.get("count") != len(rows)
        or not rows
    ):
        errors.append("source exclusion binding contract drift")
        return set()
    manifest: Any = None
    if manifest_path is not None and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"source exclusion manifest is invalid JSON ({exc})")
    expected_manifest_rows: list[dict[str, Any]] = []
    scene_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"source exclusion row {index} is invalid")
            continue
        scene_id = row.get("scene_id")
        artifact = row.get("artifact")
        artifact_path = row.get("artifact_path")
        if (
            not isinstance(scene_id, str)
            or scene_id in scene_ids
            or scene_id not in split_memberships.get("val", set())
            or row.get("official_split") != "val"
            or not isinstance(artifact, str)
            or not isinstance(artifact_path, str)
            or Path(artifact_path).name != artifact
            or row.get("reason") != "corrupt_zip_archive"
            or row.get("evidence") != "unzip_test_failed"
        ):
            errors.append(f"source exclusion row {index} identity drift")
        excluded_path = _verify_file_document(
            {"path": artifact_path, "sha256": row.get("sha256")},
            f"excluded source artifact {scene_id}",
            errors,
            hash_cache,
        )
        observed_corruption: str | None = None
        if excluded_path is not None and excluded_path.is_file():
            try:
                with ZipFile(excluded_path) as archive:
                    bad_member = archive.testzip()
                if bad_member is not None:
                    observed_corruption = f"crc_or_payload_failure:{bad_member}"
            except (OSError, BadZipFile, zlib.error) as exc:
                observed_corruption = f"archive_read_failure:{type(exc).__name__}"
        if observed_corruption is None:
            errors.append(f"excluded source artifact {scene_id} is not reproducibly corrupt")
        elif row.get("verified_corruption") != observed_corruption:
            errors.append(f"excluded source artifact {scene_id} corruption evidence drift")
        if isinstance(scene_id, str):
            scene_ids.add(scene_id)
        expected_manifest_rows.append({
            key: row.get(key)
            for key in (
                "artifact", "evidence", "official_split", "reason", "scene_id",
                "sha256",
            )
        })
    if manifest != {
        "dataset": "ScanNet v2",
        "exclusions": expected_manifest_rows,
        "schema": "gsa-scannet-source-exclusions-v1",
    }:
        errors.append("source exclusion manifest/binding drift")
    if report.get("excluded_corrupt_source_scene_ids") != sorted(scene_ids):
        errors.append("source report corrupt-scene exclusion list drift")
    return scene_ids


def _validate_frozen_canary_scenes(
    value: Any,
    expected_scene_ids: list[str],
    errors: list[str],
    hash_cache: dict[Path, str] | None = None,
) -> None:
    if not isinstance(value, dict):
        errors.append("frozen canary scene binding is missing")
        return
    if value.get("manifest_sha256") != SCANNET_CANARY_SCENES_SHA256:
        errors.append("frozen canary scene manifest is not the code-pinned revision")
    path = _verify_file_document(
        {"path": value.get("manifest_path"), "sha256": value.get("manifest_sha256")},
        "frozen canary scene manifest",
        errors,
        hash_cache,
    )
    if path is None or not path.is_file():
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"frozen canary scene manifest is invalid JSON ({exc})")
        return
    expected = copy.deepcopy(document)
    if isinstance(expected, dict):
        expected["manifest_path"] = str(path)
        expected["manifest_sha256"] = SCANNET_CANARY_SCENES_SHA256
    if value != expected:
        errors.append("frozen canary scene manifest/binding drift")
    if (
        not isinstance(document, dict)
        or document.get("schema") != "gsa-scannet-sensor-canary-scenes-v1"
        or document.get("protocol") != SOURCE_PROTOCOL
        or document.get("official_split") != "val"
        or document.get("split_revision") != SCANNET_SPLIT_REVISION
        or document.get("quality_contract") != visual_quality_contract()
        or document.get("scene_ids") != expected_scene_ids
        or expected_scene_ids != sorted(expected_scene_ids)
        or len(expected_scene_ids) != SOURCE_CANARY_RECORDS
        or len(set(expected_scene_ids)) != SOURCE_CANARY_RECORDS
    ):
        errors.append("frozen canary scene contract drift")


def _validate_canary_binding(
    report: dict[str, Any],
    rows: list[SourceRow],
    split_documents: dict[str, dict[str, Any]],
    split_memberships: dict[str, set[str]],
    errors: list[str],
    hash_cache: dict[Path, str] | None = None,
    member_cache: dict[tuple[Path, str], str] | None = None,
    source_exclusion_binding: dict[str, Any] | None = None,
) -> None:
    binding = report.get("canary_binding")
    if not isinstance(binding, dict):
        errors.append("full source report lacks canary_binding")
        return
    report_doc = {
        "path": binding.get("report_path"),
        "sha256": binding.get("report_sha256"),
    }
    labels_doc = {
        "path": binding.get("labels_path"),
        "sha256": binding.get("labels_sha256"),
    }
    canary_report_path = _verify_file_document(
        report_doc, "bound canary report", errors, hash_cache
    )
    canary_labels_path = _verify_file_document(
        labels_doc, "bound canary labels", errors, hash_cache
    )
    scene_ids = binding.get("scene_ids")
    image_hashes = binding.get("image_sha256s")
    if (
        not isinstance(scene_ids, list)
        or not scene_ids
        or len(scene_ids) != len(set(scene_ids))
        or not isinstance(image_hashes, list)
        or len(image_hashes) != len(scene_ids)
        or len(image_hashes) != len(set(image_hashes))
        or not all(is_sha256(value) for value in image_hashes)
    ):
        errors.append("canary scene/image binding is invalid")
        return
    _validate_frozen_canary_scenes(
        binding.get("frozen_canary_scenes"), scene_ids, errors, hash_cache
    )
    if binding.get("source_artifact_replay") != {
        "records": SOURCE_CANARY_RECORDS,
        "files_rehashed": 11 * SOURCE_CANARY_RECORDS,
        "zip_members_rehashed": 2 * SOURCE_CANARY_RECORDS,
        "errors": 0,
    }:
        errors.append("canary source artifact replay receipt drift")
    if report.get("frozen_canary_scenes") != binding.get("frozen_canary_scenes"):
        errors.append("full report frozen canary scene snapshot drift")
    if report.get("visual_quality_contract") != visual_quality_contract():
        errors.append("full report visual quality contract drift")
    if report.get("generation_parameters") != expected_generation_parameters(
        target=SOURCE_MIN_RECORDS,
        max_records_per_scene=MAX_RECORDS_PER_SCENE,
        minimum_scenes=MIN_FULL_SCENES,
    ):
        errors.append("full report generation parameters drift")
    report_excluded = report.get("excluded_canary_scene_ids")
    if sorted(scene_ids) != sorted(report_excluded or []):
        errors.append("full report excluded canary scenes drift")
    if report.get("scene_overlap_with_canary") not in ([], None):
        errors.append("full report overlaps its bound canary scenes")
    full_scenes = {row.record.get("scene_id") for row in rows}
    full_images = {row.record.get("image_sha256") for row in rows}
    if full_scenes & set(scene_ids) or full_images & set(image_hashes):
        errors.append("full source labels overlap the bound canary")
    if (
        canary_report_path is None
        or canary_labels_path is None
        or not canary_report_path.is_file()
        or not canary_labels_path.is_file()
    ):
        return
    try:
        canary_report = json.loads(canary_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"bound canary report is invalid JSON ({exc})")
        return
    canary_rows = _read_jsonl(canary_labels_path, errors)
    canary_records = [row.record for row in canary_rows]
    expected_identity = {
        "stage": "canary",
        "protocol": SOURCE_PROTOCOL,
        "label_version": LABEL_VERSION,
        "audit_round": SOURCE_AUDIT_ROUND,
        "source_artifact_schema": SOURCE_ARTIFACT_SCHEMA,
        "builder_sha256": file_sha256(SOURCE_BUILDER_PATH),
        "selection_rule": CANARY_SELECTION_RULE,
        "target": SOURCE_CANARY_RECORDS,
        "accepted": SOURCE_CANARY_RECORDS,
        "accepted_scenes": SOURCE_CANARY_RECORDS,
        "accepted_by_official_split": {"val": SOURCE_CANARY_RECORDS},
        "eligible": True,
    }
    for field, value in expected_identity.items():
        if not isinstance(canary_report, dict) or canary_report.get(field) != value:
            errors.append(f"bound canary report {field} drift")
    canary_system_errors = (
        canary_report.get("system_errors") if isinstance(canary_report, dict) else None
    )
    if (
        not isinstance(canary_system_errors, dict)
        or canary_system_errors.get("total") != 0
        or canary_system_errors.get("items") != []
    ):
        errors.append("bound canary report contains system errors")
    if (
        not isinstance(canary_report, dict)
        or canary_report.get("generation_parameters")
        != expected_generation_parameters(
            target=SOURCE_CANARY_RECORDS,
            max_records_per_scene=1,
            minimum_scenes=SOURCE_CANARY_RECORDS,
        )
        or canary_report.get("visual_quality_contract") != visual_quality_contract()
        or canary_report.get("frozen_canary_scenes")
        != binding.get("frozen_canary_scenes")
        or canary_report.get("source_inventory") != report.get("source_inventory")
        or canary_report.get("labels_sha256") != file_sha256(canary_labels_path)
        or canary_report.get("records_sha256") != content_hash(canary_records)
        or canary_report.get("accepted_scene_ids")
        != sorted(record.get("scene_id") for record in canary_records)
        or canary_report.get("source_contract") != report.get("source_contract")
        or canary_report.get("validated_source_exclusions")
        != source_exclusion_binding
    ):
        errors.append("bound canary report label/scene/source binding drift")
    if sorted(record.get("scene_id") for record in canary_records) != sorted(scene_ids):
        errors.append("bound canary labels scene order drift")
    if [record.get("image_sha256") for record in canary_records] != image_hashes:
        errors.append("bound canary labels image order drift")
    for source in canary_rows:
        record = source.record
        _validate_source_record_artifacts(
            source,
            split_documents,
            split_memberships,
            errors,
            hash_cache,
            member_cache,
        )


def _validate_source_record_artifacts(
    source: SourceRow,
    split_documents: dict[str, dict[str, Any]],
    split_memberships: dict[str, set[str]],
    errors: list[str],
    hash_cache: dict[Path, str] | None = None,
    member_cache: dict[tuple[Path, str], str] | None = None,
) -> None:
    record = source.record
    sample_id = record.get("sample_id") or f"line-{source.line_number}"
    record_errors = validate_record(record)
    if record_errors:
        errors.append(f"source record {sample_id} invalid: {','.join(record_errors)}")
    if record.get("audit_round") != SOURCE_AUDIT_ROUND:
        errors.append(f"source record {sample_id} audit round drift")
    if record.get("protocol_version") != SOURCE_PROTOCOL:
        errors.append(f"source record {sample_id} protocol drift")
    if (
        record.get("_audit_only") is not True
        or record.get("audit_contract") != AUDIT_CONTRACT
        or record.get("task_interface") != "oracle_grounded_pairwise_depth"
    ):
        errors.append(f"source record {sample_id} held-out audit interface drift")
    training = record.get("training_materialization")
    if training != {
        "eligible": False,
        "reason": "official_scannet_held_out_split",
        "allowed_use": "blind_audit_or_benchmark_only",
    }:
        errors.append(f"source record {sample_id} training policy drift")

    split = record.get("official_split")
    scene_id = record.get("scene_id")
    if split not in {"val", "test"} or scene_id not in split_memberships.get(str(split), set()):
        errors.append(f"source record {sample_id} is not in an official held-out split")
    provenance = record.get("sensor_provenance")
    if not isinstance(provenance, dict):
        return
    membership = provenance.get("split_membership")
    if membership != split_documents.get(str(split)):
        errors.append(f"source record {sample_id} split provenance drift")
    if provenance.get("scene_id") != scene_id or provenance.get("frame_index") != record.get("frame_index"):
        errors.append(f"source record {sample_id} scene/frame provenance drift")

    image_path = Path(str(record.get("image_path", ""))).resolve()
    image_hash = record.get("image_sha256")
    if not image_path.is_file():
        errors.append(f"source record {sample_id} image missing: {image_path}")
    elif not is_sha256(image_hash) or file_sha256(image_path) != image_hash:
        errors.append(f"source record {sample_id} image hash drift")
    rgb_path = _verify_file_document(
        provenance.get("rgb"), f"{sample_id} RGB", errors, hash_cache
    )
    if rgb_path != image_path or (provenance.get("rgb") or {}).get("sha256") != image_hash:
        errors.append(f"source record {sample_id} RGB/image identity drift")
    for key in ("depth", "pose", "scene_metadata", "aggregation"):
        _verify_file_document(
            provenance.get(key), f"{sample_id} {key}", errors, hash_cache
        )
    calibration = provenance.get("calibration")
    if not isinstance(calibration, dict) or set(calibration) != {
        "intrinsic_color", "intrinsic_depth", "extrinsic_color", "extrinsic_depth"
    }:
        errors.append(f"source record {sample_id} calibration provenance is incomplete")
    else:
        for key, value in sorted(calibration.items()):
            _verify_file_document(
                value, f"{sample_id} calibration/{key}", errors, hash_cache
            )
    _verify_zip_document(
        provenance.get("instance_projection"),
        f"{sample_id} instance projection",
        errors,
        hash_cache,
        member_cache,
    )
    _verify_zip_document(
        provenance.get("semantic_projection"),
        f"{sample_id} semantic projection",
        errors,
        hash_cache,
        member_cache,
    )
    semantic = provenance.get("semantic_projection")
    aggregation = provenance.get("aggregation")
    if not isinstance(semantic, dict) or semantic.get("encoding") != "scannet_raw_label_id":
        errors.append(f"source record {sample_id} semantic encoding drift")
    if not isinstance(aggregation, dict) or aggregation.get("instance_id_rule") != "objectId_plus_1":
        errors.append(f"source record {sample_id} aggregation identity rule drift")


def load_and_validate_source_bundle(
    report_path: Path,
    labels_path: Path,
) -> tuple[SourceBundle | None, list[str]]:
    """Load and fully revalidate the immutable v5 sensor source bundle."""
    errors: list[str] = []
    hash_cache: dict[Path, str] = {}
    member_cache: dict[tuple[Path, str], str] = {}
    report_path = report_path.resolve()
    labels_path = labels_path.resolve()
    if not report_path.is_file():
        return None, [f"source report missing: {report_path}"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"invalid source report: {report_path} ({exc})"]
    if not isinstance(report, dict):
        return None, ["source report is not an object"]
    rows = _read_jsonl(labels_path, errors)
    records = [row.record for row in rows]

    expected_labels = (report_path.parent / "labels/S_spatial/depth.jsonl").resolve()
    report_labels = report.get("labels_path")
    if report_labels != "labels/S_spatial/depth.jsonl" or labels_path != expected_labels:
        errors.append("source labels are not path-bound to the source report")
    expected_identity = {
        "stage": "full",
        "protocol": SOURCE_PROTOCOL,
        "label_version": LABEL_VERSION,
        "audit_round": SOURCE_AUDIT_ROUND,
        "source_artifact_schema": SOURCE_ARTIFACT_SCHEMA,
        "builder_sha256": file_sha256(SOURCE_BUILDER_PATH),
        "selection_rule": FULL_SELECTION_RULE,
        "eligible": True,
    }
    for field, value in expected_identity.items():
        if report.get(field) != value:
            errors.append(f"source report {field} drift")
    if report.get("generation_parameters") != expected_generation_parameters(
        target=SOURCE_MIN_RECORDS,
        max_records_per_scene=MAX_RECORDS_PER_SCENE,
        minimum_scenes=MIN_FULL_SCENES,
    ):
        errors.append("source report generation parameters drift")
    if report.get("visual_quality_contract") != visual_quality_contract():
        errors.append("source report visual quality contract drift")
    system_errors = report.get("system_errors")
    if not isinstance(system_errors, dict) or system_errors.get("total") != 0 or system_errors.get("items") != []:
        errors.append("source report contains system errors")
    if (
        report.get("accepted") != len(rows)
        or report.get("target") != len(rows)
        or len(rows) < SOURCE_MIN_RECORDS
    ):
        errors.append("source report accepted/target/label count drift")
    scene_ids = sorted({str(row.record.get("scene_id")) for row in rows})
    if (
        report.get("accepted_scenes") != len(scene_ids)
        or report.get("accepted_scene_ids") != scene_ids
        or len(scene_ids) < MIN_FULL_SCENES
    ):
        errors.append("source report accepted scene identity drift")
    split_counts = dict(sorted(Counter(row.record.get("official_split") for row in rows).items()))
    if report.get("accepted_by_official_split") != split_counts or set(split_counts) != {"val"}:
        errors.append("source report official split counts drift")
    gap_counts = dict(sorted(Counter(
        (row.record.get("visibility_features") or {}).get("gap_bucket")
        for row in rows
    ).items()))
    if report.get("accepted_by_gap_bucket") != gap_counts:
        errors.append("source report visibility gap counts drift")
    quartile_separated = sum(bool(
        (row.record.get("visibility_features") or {}).get("quartile_separated")
    ) for row in rows)
    if report.get("quartile_separated") != quartile_separated:
        errors.append("source report quartile separation count drift")
    if labels_path.is_file() and report.get("labels_sha256") != file_sha256(labels_path):
        errors.append("source labels content hash drift")
    if report.get("records_sha256") != content_hash(records):
        errors.append("source records canonical hash drift")

    contract = report.get("source_contract")
    if not isinstance(contract, dict):
        errors.append("source report lacks source_contract")
    else:
        required_contract = {
            "depth_encoding": "png_uint16_millimeter",
            "registration": "depth_to_color_reprojection_zbuffer_v1",
            "instance_projection": "official_2d-instance-filt_zip_member",
            "semantic_projection": "official_2d-label-filt_zip_member_raw_id",
            "generation_split": "val",
            "test_split_excluded_reason": "official_2d_ground_truth_unavailable_locally",
            "intended_use": "held_out_blind_audit_or_benchmark_only",
            "pool_scope": "calibration_candidates_not_full_benchmark_scale",
            "intended_task": "oracle_grounded_pairwise_depth",
            "model_interface_requires_explicit_pair_groundings": True,
            "training_materialization_eligible": False,
            "sensor_only": True,
            "pair_selection_rule": "visual_auditability_then_metric_margin_v1",
            "visual_quality_contract": visual_quality_contract(),
        }
        for field, value in required_contract.items():
            if contract.get(field) != value:
                errors.append(f"source contract {field} drift")
        if contract.get("label_map_sha256") != SCANNET_LABEL_MAP_SHA256:
            errors.append("source contract label map is not the code-pinned revision")
        _verify_file_document(
            {"path": contract.get("label_map_path"), "sha256": contract.get("label_map_sha256")},
            "ScanNet label map",
            errors,
            hash_cache,
        )
    split_documents, split_memberships = _load_split_documents(
        report, errors, hash_cache
    )
    excluded_source_scenes = _validate_source_exclusions(
        report, split_memberships, errors, hash_cache
    )
    inventory = report.get("source_inventory")
    if (
        not isinstance(inventory, dict)
        or inventory.get("scenes_discovered") != 89
        or inventory.get("scenes_by_official_split") != {"test": 10, "val": 79}
        or inventory.get("ready_scenes_by_official_split") != {"val": 78}
        or inventory.get("validated_source_exclusion_count")
        != len(excluded_source_scenes)
        or inventory.get("validated_source_exclusion_scene_ids")
        != sorted(excluded_source_scenes)
    ):
        errors.append("source inventory val/test scope drift")
    _validate_canary_binding(
        report,
        rows,
        split_documents,
        split_memberships,
        errors,
        hash_cache,
        member_cache,
        report.get("validated_source_exclusions"),
    )

    seen_sample_ids: set[str] = set()
    seen_images: set[str] = set()
    for row in rows:
        sample_id = row.record.get("sample_id")
        image_hash = row.record.get("image_sha256")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_sample_ids:
            errors.append(f"duplicate/invalid source sample_id at line {row.line_number}")
        if not is_sha256(image_hash) or image_hash in seen_images:
            errors.append(f"duplicate/invalid source image at line {row.line_number}")
        if isinstance(sample_id, str):
            seen_sample_ids.add(sample_id)
        if isinstance(image_hash, str):
            seen_images.add(image_hash)
        if row.record.get("scene_id") in excluded_source_scenes:
            errors.append(
                f"excluded corrupt source scene entered labels: "
                f"{row.record.get('scene_id')}"
            )
        _validate_source_record_artifacts(
            row,
            split_documents,
            split_memberships,
            errors,
            hash_cache,
            member_cache,
        )

    bundle = SourceBundle(report_path, labels_path, report, tuple(rows))
    return bundle, list(dict.fromkeys(errors))


def _projected_sensor_provenance(source: dict[str, Any]) -> dict[str, Any]:
    provenance = copy.deepcopy(source["sensor_provenance"])
    provenance["measurements"] = copy.deepcopy(source["sensor_measurements"])
    provenance["visibility_features"] = copy.deepcopy(source["visibility_features"])
    return provenance


def project_source_record(
    source: SourceRow,
    source_jsonl_path: Path,
    report_sha256: str,
    seed: int,
) -> dict[str, Any]:
    """Create the exact reviewer-facing audit label from one source record."""
    record = source.record
    payload = copy.deepcopy(record["payload"])
    measurements = record["sensor_measurements"]
    groundings = [
        grounding(item["label"], list(item["bbox_2d"]))
        for item in measurements
    ]
    context = context_for(
        FORM,
        groundings,
        {"relation_mode": DEPTH_SCOPE, "depth_order": copy.deepcopy(payload)},
    )
    valid, reason = validate(FORM, payload, context)
    if not valid:
        raise ValueError(f"source projection violates strict depth contract: {reason}")
    identity = content_hash({
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "source_record_sha256": source.sha256,
        "source_report_sha256": report_sha256,
    })
    semantic_gate = copy.deepcopy(record["semantic_gate"])
    semantic_gate["depth_v50_scannet_sensor"]["verified_pair"] = copy.deepcopy(
        record["verified_pair"]
    )
    return {
        "sample_id": f"v5_scannet_sensor_depth_{identity[:20]}",
        "audit_schema_version": SCHEMA_VERSION,
        "audit_round": PACK_ROUND,
        "audit_contract": AUDIT_CONTRACT,
        "task_interface": "oracle_grounded_pairwise_depth",
        "benchmark_interface": {
            "pair_identity_inputs": ["object_name", "bbox_2d_thousandth"],
            "target": "binary_near_far_order_for_the_supplied_pair",
            "generic_visible_object_order_prompt_allowed": False,
            "required_evaluation_tracks": [
                "S_oracle_grounded_pair",
                "G_to_S_predicted_grounding_cascade",
            ],
        },
        "model_interface": copy.deepcopy(record["model_interface"]),
        "source_sample_id": record["sample_id"],
        "source_label_sha256": source.sha256,
        "source_jsonl_path": str(source_jsonl_path.resolve()),
        "source_jsonl_line": source.line_number,
        "source_report_sha256": report_sha256,
        "source_artifact_schema": SOURCE_ARTIFACT_SCHEMA,
        "task_form": FORM,
        "depth_scope": DEPTH_SCOPE,
        "pipeline": "S_spatial",
        "dimension": "S_spatial",
        "protocol_version": SOURCE_PROTOCOL,
        "prompt_version": SOURCE_PROTOCOL,
        "label_version": LABEL_VERSION,
        "scene_type": record["scene_type"],
        "source_bucket": record["source_bucket"],
        "scene_id": record["scene_id"],
        "official_split": record["official_split"],
        "frame_index": record["frame_index"],
        "image_path": record["image_path"],
        "image_sha256": record["image_sha256"],
        "object_class": payload[0]["label"],
        "payload": payload,
        "gsa_context": context,
        "sensor_provenance": _projected_sensor_provenance(record),
        "semantic_gate": semantic_gate,
        "training_materialization": copy.deepcopy(record["training_materialization"]),
        "_audit_only": True,
        "_audit_reason": "official_scannet_held_out_sensor_calibration",
    }


def source_artifact_binding(record: dict[str, Any]) -> dict[str, Any]:
    provenance = record["sensor_provenance"]
    return {
        "scene_id": record["scene_id"],
        "frame_index": record["frame_index"],
        "official_split": record["official_split"],
        "image_sha256": record["image_sha256"],
        "sensor_provenance_sha256": content_hash(provenance),
    }


def _category(label: Any) -> str:
    value = str(label or "").strip()
    return re.sub(r"\s*\[instance\s+\d+\]\s*$", "", value).strip().casefold()


def difficulty_features(record: dict[str, Any]) -> dict[str, Any]:
    """Project sensor truth to non-metric audit-difficulty bins."""
    visibility = record["visibility_features"]
    gap_bucket = visibility["gap_bucket"]
    near, far = record["sensor_measurements"]
    source_overlap = float(visibility["bbox_intersection_over_min_area"])
    source_edge_distance = float(visibility["bbox_edge_distance_normalized"])
    if source_overlap >= 0.75:
        geometry_bucket = "overlap_at_least_0.75"
    elif source_overlap >= 0.50:
        geometry_bucket = "overlap_0.50_to_0.75"
    elif source_overlap >= 0.10:
        geometry_bucket = "overlap_substantial"
    elif source_overlap > 0:
        geometry_bucket = "overlap_light"
    elif source_edge_distance <= 0.05:
        geometry_bucket = "separate_near_edges"
    else:
        geometry_bucket = "separate_far_edges"
    support_quality = (
        "high"
        if float(visibility["min_valid_fraction"]) >= 0.50
        else "moderate"
    )
    return {
        "gap_bucket": gap_bucket,
        "quartile_separated": bool(visibility["quartile_separated"]),
        "geometry_bucket": geometry_bucket,
        "support_quality_bucket": support_quality,
        "near_category": _category(near["label"]),
        "far_category": _category(far["label"]),
    }


def source_visibility_projection_hash(record: dict[str, Any]) -> str:
    return content_hash(record["visibility_features"])


def _stable_key(seed: int, row: SourceRow) -> str:
    return hashlib.sha256(
        f"{PACK_PROTOCOL}|{seed}|{row.record['scene_id']}|{row.sha256}".encode("utf-8")
    ).hexdigest()


def select_source_rows(rows: tuple[SourceRow, ...], seed: int) -> list[SourceRow]:
    # Stratify the complete source pool before enforcing scene uniqueness. This
    # preserves rare overlap hard cases that a single random scene
    # representative could silently discard.
    strata: dict[tuple[str, str, bool], list[SourceRow]] = {}
    for row in rows:
        # Official ScanNet test scenes remain hidden-test candidates.  They must
        # not influence rubric iteration or checkpoint selection.
        if row.record.get("official_split") != "val":
            continue
        features = difficulty_features(row.record)
        key = (
            features["gap_bucket"],
            features["geometry_bucket"],
            features["quartile_separated"],
        )
        strata.setdefault(key, []).append(row)
    for values in strata.values():
        values.sort(key=lambda row: _stable_key(seed, row))
    selected: list[SourceRow] = []
    selected_scenes: set[str] = set()
    keys = sorted(strata, key=str)
    while len(selected) < REQUIRED_IMAGES and any(strata.values()):
        made_progress = False
        for key in keys:
            while (
                strata[key]
                and str(strata[key][0].record["scene_id"]) in selected_scenes
            ):
                strata[key].pop(0)
            if strata[key] and len(selected) < REQUIRED_IMAGES:
                row = strata[key].pop(0)
                selected.append(row)
                selected_scenes.add(str(row.record["scene_id"]))
                made_progress = True
        if not made_progress:
            break
    if len(selected) != REQUIRED_IMAGES:
        raise ValueError(
            f"only {len(selected)} unique official-val scene/image records are "
            f"available; schema5 requires exactly {REQUIRED_IMAGES} and will not "
            "substitute official-test scenes"
        )
    images = [row.record["image_sha256"] for row in selected]
    if len(images) != len(set(images)):
        raise ValueError("schema5 selection contains duplicate images")
    return selected


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def build_pack(
    report_path: Path,
    labels_path: Path,
    out: Path,
    seed: int,
) -> dict[str, Any]:
    out = out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"refusing to overwrite non-empty output: {out}")
    bundle, errors = load_and_validate_source_bundle(report_path, labels_path)
    if bundle is None or errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"v5 sensor source bundle failed closed:\n{details}")
    if bundle.report.get("accepted_scenes", 0) < MIN_SOURCE_SCENES:
        raise ValueError(
            f"v5 audit requires at least {MIN_SOURCE_SCENES} full-source scenes "
            "so calibration does not exhaust benchmark coverage"
        )
    report_sha256 = file_sha256(bundle.report_path)
    selected_sources = select_source_rows(bundle.rows, seed)
    projected = [
        project_source_record(source, bundle.labels_path, report_sha256, seed)
        for source in selected_sources
    ]

    selected: list[dict[str, Any]] = []
    for source, label in zip(selected_sources, projected):
        binding = source_artifact_binding(source.record)
        difficulty = difficulty_features(source.record)
        selected.append({
            "image_path": label["image_path"],
            "image_sha256": label["image_sha256"],
            "scene_id": label["scene_id"],
            "official_split": label["official_split"],
            "sample_ids": {FORM: label["sample_id"]},
            "source_sample_ids": {FORM: source.record["sample_id"]},
            "source_record_sha256": source.sha256,
            "source_jsonl_line": source.line_number,
            "sensor_provenance_sha256": binding["sensor_provenance_sha256"],
            "sensor_artifact_binding_sha256": content_hash(binding),
            "source_visibility_projection_sha256": (
                source_visibility_projection_hash(source.record)
            ),
            "difficulty_features": difficulty,
            "difficulty_features_sha256": content_hash(difficulty),
            "label_hashes": {FORM: content_hash(label)},
            "label_versions": {FORM: LABEL_VERSION},
            "prompt_versions": {FORM: SOURCE_PROTOCOL},
            "depth_scopes": {FORM: DEPTH_SCOPE},
        })

    source_contract = bundle.report["source_contract"]
    calibration_scene_ids = {row.record["scene_id"] for row in selected_sources}
    reserved_rows = [
        {
            "sample_id": row.record["sample_id"],
            "scene_id": row.record["scene_id"],
            "official_split": row.record["official_split"],
            "source_jsonl_line": row.line_number,
            "source_record_sha256": row.sha256,
            "image_sha256": row.record["image_sha256"],
        }
        for row in bundle.rows
        if row.record["scene_id"] not in calibration_scene_ids
    ]
    reserved_scene_ids = sorted({row["scene_id"] for row in reserved_rows})
    if len(reserved_scene_ids) < MIN_SOURCE_SCENES - REQUIRED_IMAGES:
        raise ValueError("v5 audit leaves fewer than 20 reserved benchmark scenes")
    source_generation = {
        "report_path": str(bundle.report_path),
        "report_sha256": report_sha256,
        "source_labels_path": str(bundle.labels_path),
        "source_labels_sha256": file_sha256(bundle.labels_path),
        "source_records_sha256": bundle.report["records_sha256"],
        "source_contract_sha256": content_hash(source_contract),
        "source_exclusions_sha256": content_hash(
            bundle.report["validated_source_exclusions"]
        ),
        "canary_binding_sha256": content_hash(bundle.report["canary_binding"]),
        "generation_parameters_sha256": content_hash(
            bundle.report["generation_parameters"]
        ),
        "visual_quality_contract_sha256": content_hash(
            bundle.report["visual_quality_contract"]
        ),
        "frozen_canary_scenes_sha256": content_hash(
            bundle.report["frozen_canary_scenes"]
        ),
        "label_map_sha256": SCANNET_LABEL_MAP_SHA256,
        "protocol": SOURCE_PROTOCOL,
        "label_version": LABEL_VERSION,
        "audit_round": SOURCE_AUDIT_ROUND,
        "source_artifact_schema": SOURCE_ARTIFACT_SCHEMA,
        "builder_sha256": bundle.report["builder_sha256"],
        "selection_rule": bundle.report["selection_rule"],
        "generation_split": "val",
        "stage": "full",
        "accepted": bundle.report["accepted"],
        "accepted_scenes": bundle.report["accepted_scenes"],
    }
    source_fingerprint = content_hash(source_generation)
    pairs = [
        {"sample_id": row["sample_id"], "label_sha256": content_hash(row)}
        for row in projected
    ]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PACK_PROTOCOL,
        "round": PACK_ROUND,
        "audit_contract": AUDIT_CONTRACT,
        "task_interface": "oracle_grounded_pairwise_depth",
        "benchmark_interface": {
            "pair_identity_inputs": ["object_name", "bbox_2d_thousandth"],
            "target": "binary_near_far_order_for_the_supplied_pair",
            "generic_visible_object_order_prompt_allowed": False,
            "required_evaluation_tracks": [
                "S_oracle_grounded_pair",
                "G_to_S_predicted_grounding_cascade",
            ],
        },
        "audit_id": f"{PACK_ROUND}:{content_hash(selected)[:16]}",
        "source_fingerprint": source_fingerprint,
        "source_generation": source_generation,
        "source_inventory": source_inventory_summary(bundle.report),
        "source_quality_contract": copy.deepcopy(
            bundle.report["visual_quality_contract"]
        ),
        "source_canary_binding": copy.deepcopy(bundle.report["canary_binding"]),
        "frozen_canary_scenes": copy.deepcopy(
            bundle.report["frozen_canary_scenes"]
        ),
        "source_contract": copy.deepcopy(source_contract),
        "validated_source_exclusions": copy.deepcopy(
            bundle.report["validated_source_exclusions"]
        ),
        "reserved_benchmark": {
            "policy": "source_scenes_disjoint_from_semantic_calibration",
            "minimum_scenes": MIN_SOURCE_SCENES - REQUIRED_IMAGES,
            "scene_ids": reserved_scene_ids,
            "scene_ids_sha256": content_hash(reserved_scene_ids),
            "source_rows": reserved_rows,
            "source_rows_sha256": content_hash(reserved_rows),
            "by_official_split": dict(sorted(Counter(
                row["official_split"] for row in reserved_rows
            ).items())),
            "exposed_to_audit_ui": False,
            "training_materialization_eligible": False,
        },
        "forms": FORMS,
        "depth_scope": DEPTH_SCOPE,
        "images": REQUIRED_IMAGES,
        "records_per_annotator": REQUIRED_IMAGES,
        "selection": {
            "seed": seed,
            "rule": PACK_SELECTION_RULE,
            "unique_scene_required": True,
            "unique_image_required": True,
            "calibration_split": "val",
            "official_test_policy": "reserved_hidden_test_not_selected",
            "source_available_by_official_split": dict(sorted(Counter(
                row.record["official_split"] for row in bundle.rows
            ).items())),
            "selected_by_official_split": dict(sorted(Counter(
                label["official_split"] for label in projected
            ).items())),
            "difficulty_counts": dict(sorted(Counter(
                (
                    item["difficulty_features"]["gap_bucket"] + "|"
                    + item["difficulty_features"]["geometry_bucket"] + "|"
                    + str(item["difficulty_features"]["quartile_separated"])
                )
                for item in selected
            ).items())),
            "near_category_counts": dict(sorted(Counter(
                item["difficulty_features"]["near_category"] for item in selected
            ).items())),
            "far_category_counts": dict(sorted(Counter(
                item["difficulty_features"]["far_category"] for item in selected
            ).items())),
        },
        "selected": selected,
        "per_form_label_hashes": {FORM: content_hash(pairs)},
        "training_materialization": {
            "eligible": False,
            "reason": "held_out_sensor_records_not_trainable",
            "allowed_use": "benchmark_component_calibration_only",
            "required_future_gate": (
                "independent_official_train_source_manifest_and_no_scene_overlap"
            ),
        },
    }
    manifest["integrity"] = {
        "selected_sha256": content_hash(selected),
        "selected_sensor_artifacts_sha256": content_hash([
            item["sensor_artifact_binding_sha256"] for item in selected
        ]),
        "reserved_benchmark_sha256": content_hash(manifest["reserved_benchmark"]),
        "manifest_payload_sha256": manifest_payload_hash(manifest),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=str(out.parent)))
    try:
        for annotator in ("annotator_a", "annotator_b"):
            _write_jsonl(temp / annotator / "labels/S_spatial/depth.jsonl", projected)
            (temp / annotator / "annotations").mkdir(parents=True, exist_ok=True)
        (temp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        if out.exists():
            out.rmdir()
        temp.rename(out)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--source-labels", type=Path, default=DEFAULT_SOURCE_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    try:
        manifest = build_pack(
            args.source_report,
            args.source_labels,
            args.out,
            args.seed,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
