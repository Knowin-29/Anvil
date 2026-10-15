#!/usr/bin/env python3
"""Build the v5.2 uniquely referential ScanNet G->S formal audit pack.

The frozen v5.1 formal40 remains immutable.  Exactly seven v5.1 examples are
eligible for evidence-preserving reuse because both public category names have
one visible instance.  The other 33 slots must come from complete, manifest-
bound v3 frame roots and receive two new independent human decisions.

Inventory never writes a dataset.  Formal materialization fails before any
output write unless at least 33 fresh scene-unique candidates satisfy the v5.1
RGB/sensor filter and the exact-one referent gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import scannet_rgb_depth_v52_interface as public_interface  # noqa: E402
from tools import build_depth_v50_scannet_sensor as sensor  # noqa: E402
from tools import build_scannet_rgb_auditable_depth_pack as v51_filter  # noqa: E402
from tools import extract_scannet_val_frame_bundles as frame_v1  # noqa: E402
from tools import extract_scannet_val_frame_bundles_v3 as frame_v3  # noqa: E402


SCHEMA_VERSION = 1
PACK_PROTOCOL = "gsa-depth-v5.2-scannet-unique-referent-gs-formal-v1"
AUDIT_ROUND = "v5.2-scannet-unique-referent-gs-formal-v1"
AUDIT_CONTRACT = "scannet_sensor_unique_referent_grounding_to_depth"
LABEL_VERSION = "gsa-label-v5.2-scannet-unique-referent-gs-v1"
REFERENT_METHOD = "visible_instance_projection_exact_aggregation_label_count_v1"
SELECTION_SEED = "gsa-depth-v5.2-unique-referent-scene-selection-v1"
SLOT_SEED = "gsa-depth-v5.2-balanced-ab-slots-v1"
FORMAL_TARGET = 40
REUSED_TARGET = 7
NEW_TARGET = 33
EXPECTED_AB_BALANCE = {"A": 20, "B": 20}
SCENE_ID_RE = re.compile(r"^scene\d{4}_\d{2}$")

DEFAULT_V51_PACK = ROOT / "data/semantic_audit_v51_scannet_rgb_auditable_depth_formal"
DEFAULT_V51_REPORT = ROOT / "artifacts/audits/scannet_rgb_depth_v51_formal.json"
DEFAULT_V3_ROOT_PARENT = ROOT / "data/scannet_val_frame_bundles_v3"
DEFAULT_OUTPUT = ROOT / "data/semantic_audit_v52_scannet_unique_referent_gs_formal"

V51_PINNED_SHA256 = {
    "manifest": "6855e6285f00b307d2a230b4460af1d81da518fca77b52a428045103e2639624",
    "report": "2b0c5c3c437d7db6d331f9b65e2d4fd005eb9d745b1a3cfd749926d69cee4198",
    "annotation_a": "3b343da78ab83bd8806d5fba3bb847071c6e04ed9ace9509ac26ae32a7b4dfca",
    "annotation_b": "60ea7bb61bd42f7a190f6d3761385e57d84d15e1e7811b5bc28a444c43e86ed0",
    "labels_a": "f6c65aae16aa1bdc843272d7cd9954bf1edcfd0e58710c51dbdc36d320ce2123",
    "labels_b": "f6c65aae16aa1bdc843272d7cd9954bf1edcfd0e58710c51dbdc36d320ce2123",
    "source_records": "db6b1705f85aebef1964de73f3c7ccbbf4874b1c263fc55545c133d9dafb7921",
}
V51_EXPECTED_MANIFEST_PAYLOAD_SHA256 = (
    "87ceeb9dfb753a0e74455c93687d586c7e414b3f31e3aef81a97bf08a2d42161"
)
V51_EXPECTED_SELECTED_SHA256 = (
    "c08d4e76d01cc4ad002cb22596d5c5c6cb0f4b0a26c7ef47f9642c1d198274a5"
)


class PackError(RuntimeError):
    """The requested v5.2 artifact would violate the evidence contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PackError(f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PackError(f"missing bound file: {path}")
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": file_sha256(path)}


def _read_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PackError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        if allow_missing:
            return []
        raise PackError(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PackError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise PackError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _resolve_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise PackError("source path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise PackError(f"source path is missing: {path}")
    return path


def manifest_payload_hash(manifest: dict[str, Any]) -> str:
    payload = copy.deepcopy(manifest)
    payload.pop("integrity", None)
    return canonical_hash(payload)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _atomic_json(path: Path, value: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _artifact_from_root(
    root: Path, document: Any, expected_relative: str, section: str
) -> Path:
    if not isinstance(document, dict) or set(document) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise PackError(f"v3 artifact binding schema drift: {section}")
    if document.get("path") != expected_relative:
        raise PackError(f"v3 artifact relative path drift: {section}")
    path = (root / expected_relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PackError(f"v3 artifact escapes root: {section}") from exc
    if path.is_symlink() or not path.is_file():
        raise PackError(f"v3 artifact is missing or symlinked: {section}")
    if path.stat().st_size != document.get("size_bytes"):
        raise PackError(f"v3 artifact size drift: {section}")
    if file_sha256(path) != document.get("sha256"):
        raise PackError(f"v3 artifact hash drift: {section}")
    return path


def discover_complete_v3_roots(parent: Path) -> tuple[list[Path], list[str]]:
    """Return only roots with both completion files; never inspect partial roots."""
    parent = parent.expanduser().resolve()
    if not parent.exists():
        return [], []
    if not parent.is_dir():
        raise PackError(f"v3 root parent is not a directory: {parent}")
    complete: list[Path] = []
    ignored: list[str] = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if (child / "manifest.json").is_file() and (child / "manifest.sha256").is_file():
            complete.append(child.resolve())
        else:
            ignored.append(str(child.resolve()))
    return complete, ignored


def load_v3_frame_root(path: Path, *, verify_artifacts: bool = True) -> dict[str, Any]:
    """Validate a complete v3 root and all materialized outputs.

    The manifest/checksum existence gate occurs before directory traversal, so a
    concurrently written temporary or partial root is never consumed.
    """
    root = path.expanduser().resolve()
    manifest_path = root / "manifest.json"
    checksum_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise PackError(f"incomplete v3 frame root (manifest pair absent): {root}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        checksum = checksum_path.read_text(encoding="ascii")
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackError(f"invalid v3 completion manifest: {root}") from exc
    if checksum != f"{manifest_sha256}  manifest.json\n":
        raise PackError(f"v3 manifest checksum drift: {root}")
    if not isinstance(manifest, dict):
        raise PackError(f"v3 manifest must be an object: {root}")
    payload = copy.deepcopy(manifest)
    integrity = payload.pop("integrity", None)
    execution = manifest.get("execution") or {}
    policy = manifest.get("use_policy") or {}
    selection = manifest.get("selection") or {}
    output_contract = manifest.get("output_contract") or {}
    if (
        manifest.get("schema") != frame_v3.SCHEMA
        or manifest.get("protocol") != frame_v3.PROTOCOL
        or manifest.get("stage") not in {"batch", "one_scene_canary"}
        or manifest.get("verdict")
        not in {"BATCH_PASS_VAL_AUDIT_ONLY", "CANARY_PASS_VAL_AUDIT_ONLY"}
        or manifest.get("training_materialization_eligible") is not False
        or policy.get("official_split") != "val"
        or policy.get("held_out_benchmark_preparation_only") is not True
        or policy.get("training_materialization_eligible") is not False
        or policy.get("may_enter_training") is not False
        or execution.get("cpu_only") is not True
        or execution.get("gpu_required") is not False
        or execution.get("network_calls") != 0
        or output_contract.get("root") != str(root)
        or output_contract.get("namespace") != "scannet_val_frame_bundles_v3"
        or selection.get("seed") != frame_v3.SELECTION_SEED
        or selection.get("maximum_frames_per_scene") != frame_v3.MAX_FRAMES_PER_SCENE
        or selection.get("scene_unique") is not True
        or selection.get("selected_indices_strictly_below_sens_num_frames") is not True
        or selection.get("candidate_set")
        != (
            "instance_intersection_semantic_intersection_sens_range_"
            "intersection_finite_homogeneous_pose"
        )
        or not isinstance(integrity, dict)
        or integrity.get("manifest_payload_sha256") != canonical_hash(payload)
    ):
        raise PackError(f"v3 frame-root contract mismatch: {root}")

    extractor = manifest.get("extractor") or {}
    if (
        extractor.get("schema_owner") != frame_v3.SCHEMA
        or extractor.get("protocol_owner") != frame_v3.PROTOCOL
    ):
        raise PackError(f"v3 extractor ownership drift: {root}")
    implementation_files = [
        {
            "path": extractor.get("extractor_path"),
            "sha256": extractor.get("extractor_sha256"),
        }
    ] + list(extractor.get("implementation_dependencies") or [])
    for item in implementation_files:
        if not isinstance(item, dict):
            raise PackError(f"v3 extractor binding malformed: {root}")
        source = Path(str(item.get("path", ""))).expanduser().resolve()
        if not source.is_file() or file_sha256(source) != item.get("sha256"):
            raise PackError(f"v3 extractor dependency drift: {source}")

    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes or not all(
        isinstance(scene, dict) for scene in scenes
    ):
        raise PackError(f"v3 scene list is absent: {root}")
    scene_ids = [scene.get("scene_id") for scene in scenes]
    if (
        len(scene_ids) != len(set(scene_ids))
        or not all(
            isinstance(scene_id, str)
            and SCENE_ID_RE.fullmatch(scene_id) is not None
            for scene_id in scene_ids
        )
        or integrity.get("scene_ids_sha256") != canonical_hash(scene_ids)
        or execution.get("scene_count") != len(scene_ids)
    ):
        raise PackError(f"v3 scene identity drift: {root}")

    expected_directories: set[str] = set()
    frame_count = 0
    for scene in scenes:
        scene_id = scene["scene_id"]
        ordinal = scene.get("ordinal")
        directory = scene.get("output_directory")
        expected_directory = (
            f"{ordinal:06d}-scannet-{scene_id}"
            if isinstance(ordinal, int) and 0 <= ordinal <= 999_999
            else None
        )
        selected = scene.get("selected_frame_indices")
        frames = scene.get("frames")
        calibration = scene.get("calibration_outputs")
        pose = scene.get("pose_validation") or {}
        header = scene.get("sensor_header") or {}
        if (
            scene.get("official_split") != "val"
            or directory != expected_directory
            or not isinstance(selected, list)
            or not selected
            or len(selected) != len(set(selected))
            or not all(isinstance(index, int) and index >= 0 for index in selected)
            or scene.get("selected_frame_indices_sha256") != canonical_hash(selected)
            or not isinstance(frames, list)
            or [item.get("frame_index") for item in frames if isinstance(item, dict)]
            != selected
            or not isinstance(calibration, dict)
            or set(calibration) != set(frame_v3.CALIBRATION_NAMES)
            or pose.get("policy")
            != "finite_and_homogeneous_pose_required_before_stable_selection"
            or pose.get("sens_stream_boundary_exact") is not True
            or pose.get("all_frame_payload_headers_validated") is not True
            or any(index >= int(header.get("num_frames", -1)) for index in selected)
        ):
            raise PackError(f"v3 scene contract drift: {root}/{scene_id}")
        expected_directories.add(directory)
        expected_files = set(frame_v3.CALIBRATION_NAMES)
        if verify_artifacts:
            for name in frame_v3.CALIBRATION_NAMES:
                _artifact_from_root(
                    root, calibration[name], f"{directory}/{name}", f"{scene_id}/{name}"
                )
        for frame in frames:
            if not isinstance(frame, dict):
                raise PackError(f"v3 frame document malformed: {root}/{scene_id}")
            index = frame["frame_index"]
            outputs = frame.get("outputs")
            if not isinstance(outputs, dict) or set(outputs) != {"rgb", "depth", "pose"}:
                raise PackError(f"v3 frame outputs drift: {root}/{scene_id}/{index}")
            names = {
                "rgb": f"{index:06d}-rgb.png",
                "depth": f"{index:06d}-depth.png",
                "pose": f"{index:06d}.txt",
            }
            for kind, name in names.items():
                expected_files.add(name)
                if verify_artifacts:
                    artifact = _artifact_from_root(
                        root,
                        outputs[kind],
                        f"{directory}/{name}",
                        f"{scene_id}/{index}/{kind}",
                    )
                    if kind == "pose" and not sensor.pose_valid(artifact):
                        raise PackError(f"v3 selected pose is invalid: {scene_id}/{index}")
        if verify_artifacts:
            scene_root = root / directory
            observed_files = {item.name for item in scene_root.iterdir() if item.is_file()}
            if observed_files != expected_files or any(item.is_dir() for item in scene_root.iterdir()):
                raise PackError(f"v3 scene directory drift: {root}/{scene_id}")
        frame_count += len(selected)
    if verify_artifacts:
        observed_directories = {item.name for item in root.iterdir() if item.is_dir()}
        observed_files = {item.name for item in root.iterdir() if item.is_file()}
        if observed_directories != expected_directories or observed_files != {
            "manifest.json",
            "manifest.sha256",
        }:
            raise PackError(f"v3 root directory drift: {root}")
    if execution.get("frame_count") != frame_count:
        raise PackError(f"v3 frame count drift: {root}")
    return {
        "path": str(root),
        "manifest": file_binding(manifest_path),
        "manifest_payload_sha256": integrity["manifest_payload_sha256"],
        "scene_count": len(scene_ids),
        "scene_ids": sorted(scene_ids),
        "frame_count": frame_count,
        "artifacts_rehashed": verify_artifacts,
        "training_materialization_eligible": False,
    }


def _latest_annotations(rows: list[dict[str, Any]], context: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        snapshot = row.get("label_snapshot")
        sample_id = snapshot.get("sample_id") if isinstance(snapshot, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise PackError(f"{context}: annotation lacks label_snapshot.sample_id")
        latest[sample_id] = row
    return latest


def _referent_evidence(
    record: dict[str, Any],
    *,
    file_hash_cache: dict[str, str] | None = None,
    member_cache: dict[tuple[str, str], bytes] | None = None,
    aggregation_label_cache: dict[str, dict[int, str]] | None = None,
    archive_cache: dict[str, zipfile.ZipFile] | None = None,
    verify_zip_hash: bool = False,
) -> dict[str, Any]:
    """Recompute the visible exact-category cardinality for one source frame."""
    file_hash_cache = file_hash_cache if file_hash_cache is not None else {}
    member_cache = member_cache if member_cache is not None else {}
    aggregation_label_cache = (
        aggregation_label_cache if aggregation_label_cache is not None else {}
    )
    identity = str(record.get("sample_id") or "<missing>")
    measurements = record.get("sensor_measurements")
    provenance = record.get("sensor_provenance")
    if not isinstance(measurements, list) or len(measurements) != 2:
        raise PackError(f"referent evidence lacks two measurements: {identity}")
    if not isinstance(provenance, dict):
        raise PackError(f"referent evidence lacks sensor provenance: {identity}")
    aggregation = provenance.get("aggregation")
    projection = provenance.get("instance_projection")
    if not isinstance(aggregation, dict) or not isinstance(projection, dict):
        raise PackError(f"referent evidence provenance drift: {identity}")

    aggregation_path = _resolve_path(aggregation.get("path"))
    aggregation_key = str(aggregation_path)
    if aggregation_key not in file_hash_cache:
        file_hash_cache[aggregation_key] = file_sha256(aggregation_path)
    if file_hash_cache[aggregation_key] != aggregation.get("sha256"):
        raise PackError(f"aggregation hash drift: {identity}")
    if aggregation_key not in aggregation_label_cache:
        try:
            aggregation_doc = json.loads(aggregation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PackError(f"invalid aggregation JSON: {identity}") from exc
        groups = (
            aggregation_doc.get("segGroups")
            if isinstance(aggregation_doc, dict)
            else None
        )
        if not isinstance(groups, list):
            raise PackError(f"aggregation groups absent: {identity}")
        labels: dict[int, str] = {}
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("objectId"), int):
                raise PackError(f"aggregation group malformed: {identity}")
            instance_id = group["objectId"] + 1
            label = group.get("label")
            if not isinstance(label, str) or not label.strip() or instance_id in labels:
                raise PackError(f"aggregation label mapping ambiguous: {identity}")
            labels[instance_id] = label
        aggregation_label_cache[aggregation_key] = labels
    labels = aggregation_label_cache[aggregation_key]

    zip_path = _resolve_path(projection.get("zip_path"))
    member = projection.get("member")
    if not isinstance(member, str) or not member:
        raise PackError(f"instance projection member absent: {identity}")
    zip_key = str(zip_path)
    if verify_zip_hash:
        if zip_key not in file_hash_cache:
            file_hash_cache[zip_key] = file_sha256(zip_path)
        if file_hash_cache[zip_key] != projection.get("zip_sha256"):
            raise PackError(f"instance projection ZIP hash drift: {identity}")
    cache_key = (zip_key, member)
    if cache_key not in member_cache:
        try:
            if archive_cache is None:
                with zipfile.ZipFile(zip_path) as archive:
                    member_cache[cache_key] = archive.read(member)
            else:
                if zip_key not in archive_cache:
                    archive_cache[zip_key] = zipfile.ZipFile(zip_path)
                member_cache[cache_key] = archive_cache[zip_key].read(member)
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise PackError(f"cannot read instance projection: {identity}") from exc
    member_bytes = member_cache[cache_key]
    member_sha256 = hashlib.sha256(member_bytes).hexdigest()
    if member_sha256 != projection.get("sha256"):
        raise PackError(f"instance projection member hash drift: {identity}")
    try:
        with Image.open(io.BytesIO(member_bytes)) as image:
            visible_ids = set(image.convert("I").getdata())
    except OSError as exc:
        raise PackError(f"invalid instance projection PNG: {identity}") from exc
    visible_ids.discard(0)
    counts = Counter(labels.get(instance_id) for instance_id in visible_ids)
    categories: list[str] = []
    target_instance_ids: list[int] = []
    for measurement in measurements:
        if not isinstance(measurement, dict):
            raise PackError(f"referent measurement malformed: {identity}")
        category = measurement.get("category")
        instance_id = measurement.get("instance_id")
        if not isinstance(category, str) or not category.strip() or not isinstance(instance_id, int):
            raise PackError(f"referent target identity malformed: {identity}")
        if instance_id not in visible_ids:
            raise PackError(f"target instance is not visible: {identity}/{instance_id}")
        categories.append(category)
        target_instance_ids.append(instance_id)
    target_counts = [counts[category] for category in categories]
    return {
        "method": REFERENT_METHOD,
        "definition": "both public target category names have exactly one visible instance",
        "source_sample_id": identity,
        "scene_id": record.get("scene_id"),
        "frame_index": record.get("frame_index"),
        "target_categories": categories,
        "target_instance_ids": target_instance_ids,
        "target_category_visible_instance_counts": target_counts,
        "both_target_categories_unique": target_counts == [1, 1],
        "visible_instance_count": len(visible_ids),
        "visible_instance_ids_sha256": canonical_hash(sorted(visible_ids)),
        "aggregation": {
            "path": str(aggregation_path),
            "sha256": aggregation.get("sha256"),
        },
        "instance_projection": {
            "zip_path": str(zip_path),
            "zip_sha256": projection.get("zip_sha256"),
            "zip_rehashed": verify_zip_hash,
            "member": member,
            "member_sha256": member_sha256,
        },
    }


def load_v51_reuse_binding(pack: Path, report_path: Path) -> dict[str, Any]:
    """Load the exact frozen v5.1 PASS and derive its seven reusable decisions."""
    pack = pack.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    paths = {
        "manifest": pack / "manifest.json",
        "report": report_path,
        "annotation_a": pack / "annotator_a/annotations/annotations.jsonl",
        "annotation_b": pack / "annotator_b/annotations/annotations.jsonl",
        "labels_a": pack / "annotator_a/labels/S_spatial/depth.jsonl",
        "labels_b": pack / "annotator_b/labels/S_spatial/depth.jsonl",
        "source_records": pack / "provenance/source_records.jsonl",
    }
    bindings: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        binding = file_binding(path)
        if binding["sha256"] != V51_PINNED_SHA256[name]:
            raise PackError(f"frozen v5.1 {name} hash drift")
        bindings[name] = binding

    manifest = _read_json(paths["manifest"])
    report = _read_json(paths["report"])
    integrity = manifest.get("integrity") or {}
    selected = manifest.get("selected")
    review = report.get("review") or {}
    if (
        manifest.get("schema_version") != 6
        or manifest.get("stage") != "formal"
        or manifest.get("images") != FORMAL_TARGET
        or not isinstance(selected, list)
        or len(selected) != FORMAL_TARGET
        or integrity.get("manifest_payload_sha256")
        != V51_EXPECTED_MANIFEST_PAYLOAD_SHA256
        or integrity.get("selected_sha256") != V51_EXPECTED_SELECTED_SHA256
        or v51_filter.manifest_payload_hash(manifest)
        != V51_EXPECTED_MANIFEST_PAYLOAD_SHA256
        or report.get("status") != "FORMAL_PASS"
        or report.get("records") != FORMAL_TARGET
        or report.get("manifest_sha256") != V51_PINNED_SHA256["manifest"]
        or report.get("manifest_payload_sha256")
        != V51_EXPECTED_MANIFEST_PAYLOAD_SHA256
        or report.get("source_artifacts_rehashed") is not True
        or report.get("source_artifact_errors") != 0
        or review.get("a_complete") != FORMAL_TARGET
        or review.get("b_complete") != FORMAL_TARGET
        or review.get("agreement_count") != FORMAL_TARGET
        or review.get("disagreement_count") != 0
        or review.get("resolved_correct_count") != FORMAL_TARGET
        or review.get("formal_40_gate_passed") is not True
    ):
        raise PackError("frozen v5.1 formal PASS contract drift")

    labels_a = _read_jsonl(paths["labels_a"])
    labels_b = _read_jsonl(paths["labels_b"])
    if labels_a != labels_b or len(labels_a) != FORMAL_TARGET:
        raise PackError("v5.1 A/B label projection drift")
    labels = {row.get("sample_id"): row for row in labels_a}
    if len(labels) != FORMAL_TARGET or None in labels:
        raise PackError("v5.1 labels contain missing or duplicate IDs")
    votes_a = _latest_annotations(_read_jsonl(paths["annotation_a"]), "v5.1/A")
    votes_b = _latest_annotations(_read_jsonl(paths["annotation_b"]), "v5.1/B")
    if set(votes_a) != set(labels) or set(votes_b) != set(labels):
        raise PackError("v5.1 A/B vote coverage drift")
    keys_a = {row.get("annotator_key") for row in votes_a.values()}
    keys_b = {row.get("annotator_key") for row in votes_b.values()}
    if len(keys_a) != 1 or len(keys_b) != 1 or None in keys_a | keys_b or keys_a & keys_b:
        raise PackError("v5.1 reviewer independence drift")
    for sample_id, label in labels.items():
        for side, vote in (("A", votes_a[sample_id]), ("B", votes_b[sample_id])):
            if vote.get("case_id") != sample_id or vote.get("correct") is not True:
                raise PackError(f"v5.1 reusable vote is not positive: {side}/{sample_id}")
            if canonical_hash(vote.get("label_snapshot")) != canonical_hash(label):
                raise PackError(f"v5.1 vote snapshot drift: {side}/{sample_id}")

    wrappers = _read_jsonl(paths["source_records"])
    sources: dict[str, dict[str, Any]] = {}
    for wrapper in wrappers:
        sample_id = wrapper.get("sample_id")
        record = wrapper.get("source_record")
        if not isinstance(sample_id, str) or not isinstance(record, dict) or sample_id in sources:
            raise PackError("v5.1 source wrapper identity drift")
        if wrapper.get("source_sample_id") != record.get("sample_id"):
            raise PackError(f"v5.1 source wrapper sample drift: {sample_id}")
        if wrapper.get("source_record_sha256") != canonical_hash(record):
            raise PackError(f"v5.1 source wrapper hash drift: {sample_id}")
        sources[sample_id] = record
    if len(sources) != FORMAL_TARGET:
        raise PackError("v5.1 source wrapper count drift")

    file_cache: dict[str, str] = {}
    member_cache: dict[tuple[str, str], bytes] = {}
    aggregation_label_cache: dict[str, dict[int, str]] = {}
    archive_cache: dict[str, zipfile.ZipFile] = {}
    reusable: list[dict[str, Any]] = []
    nonreusable_scene_ids: list[str] = []
    for item in selected:
        if not isinstance(item, dict):
            raise PackError("v5.1 selected row is malformed")
        sample_id = (item.get("sample_ids") or {}).get("depth")
        record = sources.get(sample_id)
        label = labels.get(sample_id)
        if record is None or label is None:
            raise PackError(f"v5.1 selected/source identity drift: {sample_id}")
        if (
            item.get("source_sample_id") != record.get("sample_id")
            or item.get("source_record_sha256") != canonical_hash(record)
            or item.get("image_sha256") != record.get("image_sha256")
            or item.get("label_hashes", {}).get("depth") != canonical_hash(label)
        ):
            raise PackError(f"v5.1 selected/source hash drift: {sample_id}")
        image = _resolve_path(record.get("image_path"))
        if file_sha256(image) != record.get("image_sha256"):
            raise PackError(f"v5.1 source image drift: {sample_id}")
        evidence = _referent_evidence(
            record,
            file_hash_cache=file_cache,
            member_cache=member_cache,
            aggregation_label_cache=aggregation_label_cache,
            archive_cache=archive_cache,
        )
        if evidence["both_target_categories_unique"]:
            decision_a = votes_a[sample_id]
            decision_b = votes_b[sample_id]
            decision_binding = {
                "v51_sample_id": sample_id,
                "scene_id": record["scene_id"],
                "source_sample_id": record["sample_id"],
                "source_record_sha256": canonical_hash(record),
                "image_sha256": record["image_sha256"],
                "payload_sha256": canonical_hash(record["payload"]),
                "label_sha256": canonical_hash(label),
                "reviewer_a_decision_sha256": canonical_hash(decision_a),
                "reviewer_b_decision_sha256": canonical_hash(decision_b),
            }
            decision_binding["decision_pair_sha256"] = canonical_hash(
                [
                    decision_binding["reviewer_a_decision_sha256"],
                    decision_binding["reviewer_b_decision_sha256"],
                ]
            )
            reusable.append(
                {
                    "lineage": "reused_v51",
                    "record": record,
                    "features": copy.deepcopy(item["rgb_auditability_features"]),
                    "referent_evidence": evidence,
                    "reuse_binding": decision_binding,
                    "reused_decisions": {
                        "reviewer_a": decision_a,
                        "reviewer_b": decision_b,
                    },
                }
            )
        else:
            nonreusable_scene_ids.append(str(record["scene_id"]))
    for archive in archive_cache.values():
        archive.close()
    if len(reusable) != REUSED_TARGET or len(nonreusable_scene_ids) != NEW_TARGET:
        raise PackError(
            f"v5.1 referent-capacity drift: reusable={len(reusable)}, "
            f"nonreusable={len(nonreusable_scene_ids)}"
        )
    reused_bindings = [item["reuse_binding"] for item in reusable]
    return {
        "files": bindings,
        "manifest_payload_sha256": V51_EXPECTED_MANIFEST_PAYLOAD_SHA256,
        "selected_sha256": V51_EXPECTED_SELECTED_SHA256,
        "status": "FORMAL_PASS",
        "reused_count": len(reusable),
        "nonreused_count": len(nonreusable_scene_ids),
        "reused_scene_ids": sorted(item["record"]["scene_id"] for item in reusable),
        "nonreused_scene_ids": sorted(nonreusable_scene_ids),
        "all_selected_scene_ids": sorted(item["scene_id"] for item in selected),
        "reviewer_keys": {"a": next(iter(keys_a)), "b": next(iter(keys_b))},
        "reused_decision_bindings": reused_bindings,
        "reused_decision_bindings_sha256": canonical_hash(reused_bindings),
        "reusable": reusable,
        "manifest": manifest,
    }


def _prior_scene_ids(v51: dict[str, Any]) -> set[str]:
    manifest = v51["manifest"]
    inventory = manifest.get("inventory") or {}
    excluded = inventory.get("excluded_scene_ids")
    if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
        raise PackError("v5.1 prior-audit exclusion inventory is absent")
    required = set(v51["all_selected_scene_ids"])
    required.update(v51["nonreused_scene_ids"])
    required.update(v51["reused_scene_ids"])
    provenance = manifest.get("provenance") or {}
    for key in (
        "old_schema5_audit_binding",
        "frozen_sensor_canary_binding",
        "reviewed_rgb_canary_binding",
    ):
        binding = provenance.get(key) or {}
        scenes = binding.get("scene_ids")
        if not isinstance(scenes, list) or not all(isinstance(item, str) for item in scenes):
            raise PackError(f"v5.1 prior-audit binding lacks scenes: {key}")
        required.update(scenes)
    required.update(excluded)
    return required


def select_new_scene_unique(
    candidates: Iterable[dict[str, Any]], target: int = NEW_TARGET
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        record = candidate["record"]
        by_scene.setdefault(str(record["scene_id"]), []).append(candidate)
    representatives: list[dict[str, Any]] = []
    for scene_id, rows in sorted(by_scene.items()):
        representatives.append(
            max(
                rows,
                key=lambda row: (
                    v51_filter._scene_record_score(row["record"]),
                    canonical_hash(
                        [
                            SELECTION_SEED,
                            scene_id,
                            row["record"]["sample_id"],
                            row["record"]["image_sha256"],
                        ]
                    ),
                ),
            )
        )
    representatives.sort(
        key=lambda row: canonical_hash(
            [
                SELECTION_SEED,
                row["record"]["scene_id"],
                row["record"]["image_sha256"],
            ]
        )
    )
    selected = representatives[:target]
    return selected, {
        "eligible_records": sum(len(rows) for rows in by_scene.values()),
        "eligible_scenes": len(by_scene),
        "selected_records": len(selected),
        "selected_scenes": len({row["record"]["scene_id"] for row in selected}),
        "selected_images": len({row["record"]["image_sha256"] for row in selected}),
    }


def scan_new_candidates(
    args: argparse.Namespace,
    v51: dict[str, Any],
    roots: list[Path],
    ignored_incomplete_roots: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_manifests, scene_to_split = sensor.load_split_manifests(args.split_root)
    exclusions, exclusion_binding = sensor.load_source_exclusions(
        args.source_exclusions, args.scannet_root, scene_to_split
    )
    prior_scenes = _prior_scene_ids(v51) | set(exclusions)
    root_bindings: list[dict[str, Any]] = []
    scenes: list[Any] = []
    scene_to_root: dict[str, str] = {}
    for root in roots:
        binding = load_v3_frame_root(root, verify_artifacts=True)
        discovered = sensor.discover_scenes(root.resolve(), args.scannet_root, scene_to_split)
        discovered_ids = {scene.scene_id for scene in discovered}
        if discovered_ids != set(binding["scene_ids"]):
            raise PackError(f"v3 manifest/discovery scene drift: {root}")
        overlap = set(scene_to_root) & discovered_ids
        if overlap:
            raise PackError(f"duplicate scenes across v3 roots: {sorted(overlap)[:8]}")
        for scene_id in discovered_ids:
            scene_to_root[scene_id] = str(root.resolve())
        scenes.extend(discovered)
        root_bindings.append(binding)

    fresh = [
        scene
        for scene in scenes
        if scene.official_split == "val"
        and scene.scene_id not in prior_scenes
        and sensor.source_ready(scene)
    ]
    label_maps = sensor.load_label_map(args.label_map)
    eligible: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    system_errors: list[str] = []
    sensor_candidates = 0
    strict_rgb_sensor_candidates = 0
    file_cache: dict[str, str] = {}
    member_cache: dict[tuple[str, str], bytes] = {}
    aggregation_label_cache: dict[str, dict[int, str]] = {}
    archive_cache: dict[str, zipfile.ZipFile] = {}
    for scene in fresh:
        rows, source_rejections, instance_rejections, errors = sensor.process_scene(
            scene,
            label_maps,
            split_manifests,
            max_frames=v51_filter.MAX_FRAMES_PER_SCENE,
            max_records=v51_filter.MAX_FRAMES_PER_SCENE,
        )
        system_errors.extend(errors)
        rejected.update(
            {f"source:{key}": count for key, count in source_rejections.items()}
        )
        rejected.update(
            {f"instance:{key}": count for key, count in instance_rejections.items()}
        )
        sensor_candidates += len(rows)
        for record in rows:
            errors = sensor.validate_record(record, split_manifests)
            if errors:
                rejected.update(f"record:{error}" for error in errors)
                continue
            features, reasons = v51_filter.candidate_features(record)
            if reasons:
                rejected.update(f"rgb_sensor:{reason}" for reason in reasons)
                continue
            strict_rgb_sensor_candidates += 1
            evidence = _referent_evidence(
                record,
                file_hash_cache=file_cache,
                member_cache=member_cache,
                aggregation_label_cache=aggregation_label_cache,
                archive_cache=archive_cache,
            )
            counts = evidence["target_category_visible_instance_counts"]
            if counts != [1, 1]:
                rejected[f"referent_counts:{counts[0]}_{counts[1]}"] += 1
                continue
            record_image = _resolve_path(record.get("image_path"))
            root = Path(scene_to_root[record["scene_id"]]).resolve()
            try:
                record_image.relative_to(root)
            except ValueError as exc:
                raise PackError(
                    f"new source image escapes its bound v3 root: {record['sample_id']}"
                ) from exc
            eligible.append(
                {
                    "lineage": "new_v52",
                    "record": record,
                    "features": features,
                    "referent_evidence": evidence,
                    "frame_root_path": str(root),
                }
            )
    for archive in archive_cache.values():
        archive.close()
    if system_errors:
        raise PackError(
            f"v3 source scan produced {len(system_errors)} system errors; "
            f"first={system_errors[0]}"
        )
    new_scenes = {item["record"]["scene_id"] for item in eligible}
    return eligible, {
        "official_split": "val",
        "complete_v3_roots": root_bindings,
        "ignored_incomplete_v3_roots": ignored_incomplete_roots,
        "partial_roots_consumed": 0,
        "discovered_v3_scenes": len(scenes),
        "prior_audit_scene_count": len(prior_scenes),
        "prior_audit_scene_ids_sha256": canonical_hash(sorted(prior_scenes)),
        "v51_nonreuse_33_excluded": sorted(v51["nonreused_scene_ids"]),
        "v51_reused_7_reserved": sorted(v51["reused_scene_ids"]),
        "fresh_source_ready_scenes": len(fresh),
        "sensor_candidate_records": sensor_candidates,
        "strict_rgb_sensor_candidate_records": strict_rgb_sensor_candidates,
        "unique_referent_eligible_records": len(eligible),
        "unique_referent_eligible_scenes": len(new_scenes),
        "unique_referent_eligible_scene_ids": sorted(new_scenes),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "source_exclusions_binding": exclusion_binding,
        "strict_thresholds": copy.deepcopy(v51_filter.STRICT_THRESHOLDS),
        "referent_method": REFERENT_METHOD,
    }


def _slot_assignment(candidates: list[dict[str, Any]]) -> dict[str, str]:
    if len(candidates) != FORMAL_TARGET:
        raise PackError(f"slot assignment requires {FORMAL_TARGET} candidates")
    ordered = sorted(
        candidates,
        key=lambda item: canonical_hash(
            [
                SLOT_SEED,
                item["record"]["sample_id"],
                item["record"]["image_sha256"],
            ]
        ),
    )
    return {
        item["record"]["sample_id"]: ("A" if index < FORMAL_TARGET // 2 else "B")
        for index, item in enumerate(ordered)
    }


def _target_label(candidate: dict[str, Any], nearer_slot: str) -> dict[str, Any]:
    record = candidate["record"]
    measurements = record["sensor_measurements"]
    if nearer_slot == "A":
        by_slot = {"A": measurements[0], "B": measurements[1]}
    elif nearer_slot == "B":
        by_slot = {"A": measurements[1], "B": measurements[0]}
    else:
        raise PackError("nearer slot must be A or B")
    sample_id = "v52_unique_gs_" + canonical_hash(
        [record["sample_id"], record["image_sha256"], PACK_PROTOCOL]
    )[:20]
    public_targets: list[dict[str, Any]] = []
    grounding_targets: list[dict[str, Any]] = []
    for slot in ("A", "B"):
        measurement = by_slot[slot]
        category = str(measurement["category"]).strip()
        if not category or "[" in category or "]" in category or "instance" in category.casefold():
            raise PackError(f"unsafe public category: {record['sample_id']}/{category!r}")
        expression = f"the only visible {category}"
        public_targets.append(
            {
                "slot": slot,
                "category": category,
                "referring_expression": expression,
                "visible_instance_count": 1,
            }
        )
        grounding_targets.append(
            {
                "slot": slot,
                "category": category,
                "referring_expression": expression,
                "bbox": copy.deepcopy(measurement["bbox_2d"]),
                "source_instance_id": measurement["instance_id"],
                "source_label": measurement["label"],
            }
        )
    if public_targets[0]["category"].casefold() == public_targets[1]["category"].casefold():
        raise PackError(f"same public category survived strict filter: {record['sample_id']}")
    return {
        "_audit_only": True,
        "audit_schema_version": SCHEMA_VERSION,
        "audit_contract": AUDIT_CONTRACT,
        "audit_round": AUDIT_ROUND,
        "protocol_version": PACK_PROTOCOL,
        "label_version": LABEL_VERSION,
        "interface_protocol": public_interface.INTERFACE_PROTOCOL,
        "sample_id": sample_id,
        "source_sample_id": record["sample_id"],
        "source_record_sha256": canonical_hash(record),
        "scene_id": record["scene_id"],
        "frame_index": record["frame_index"],
        "official_split": "val",
        "image_path": record["image_path"],
        "image_sha256": record["image_sha256"],
        "lineage": candidate["lineage"],
        "review_required_in_v52": candidate["lineage"] == "new_v52",
        "task_form": "grounding_to_pairwise_depth",
        "dimensions": ["G_grounding", "S_spatial"],
        "public_interface": {
            "grounding_targets": public_targets,
            "grounding_response_grammar": public_interface.GROUNDING_GRAMMAR,
            "spatial_response_grammar": public_interface.SPATIAL_GRAMMAR,
            "predicted_context_policy": (
                "use_only_the_exact_A_B_boxes_returned_by_the_G_parser; "
                "never_oracle_match_or_select_from_multiple_detections"
            ),
        },
        "audit_targets": {
            "grounding": grounding_targets,
            "spatial": {
                "relation": "nearer_to_camera",
                "nearer_slot": nearer_slot,
                "farther_slot": "B" if nearer_slot == "A" else "A",
            },
        },
        "source_payload": copy.deepcopy(record["payload"]),
        "source_payload_sha256": canonical_hash(record["payload"]),
        "referent_gate": copy.deepcopy(candidate["referent_evidence"]),
        "rgb_sensor_gate": {
            "passed": True,
            "thresholds": copy.deepcopy(v51_filter.STRICT_THRESHOLDS),
            "features": copy.deepcopy(candidate["features"]),
        },
        "training_materialization": {
            "eligible": False,
            "allowed_use": "held_out_blind_audit_or_benchmark_only",
            "reason": "official_scannet_val",
        },
    }


def _deep_verify_selected(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    file_cache: dict[str, str] = {}
    member_hash_cache: dict[tuple[str, str], str] = {}
    referent_file_cache: dict[str, str] = {}
    referent_member_cache: dict[tuple[str, str], bytes] = {}
    errors: dict[str, list[str]] = {}
    for candidate in candidates:
        record = candidate["record"]
        source_errors = sensor.source_artifact_errors(
            record, file_cache, member_hash_cache
        )
        if source_errors:
            errors[str(record.get("sample_id"))] = source_errors
            continue
        live = _referent_evidence(
            record,
            file_hash_cache=referent_file_cache,
            member_cache=referent_member_cache,
            verify_zip_hash=True,
        )
        expected = copy.deepcopy(candidate["referent_evidence"])
        expected["instance_projection"]["zip_rehashed"] = True
        if live != expected:
            errors[str(record.get("sample_id"))] = ["referent_evidence_drift"]
    if errors:
        first = next(iter(errors.items()))
        raise PackError(f"selected source artifact rehash failed: {first[0]}={first[1]}")
    return {
        "selected_records_rehashed": len(candidates),
        "unique_files_rehashed": len(file_cache | referent_file_cache),
        "unique_projection_members_rehashed": len(member_hash_cache),
        "source_artifact_errors": 0,
    }


def _relative_binding(root: Path, path: Path) -> dict[str, Any]:
    path = path.resolve()
    relative = str(path.relative_to(root.resolve()))
    return {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def freeze_formal_pack(
    output: Path,
    reusable: list[dict[str, Any]],
    selected_new: list[dict[str, Any]],
    inventory: dict[str, Any],
    v51: dict[str, Any],
    deep_rehash: dict[str, Any],
) -> dict[str, Any]:
    """Atomically freeze exactly 7 reused + 33 pending-review records."""
    output = output.expanduser().resolve()
    if output.exists():
        raise PackError(f"refusing to overwrite existing output: {output}")
    if len(reusable) != REUSED_TARGET or len(selected_new) != NEW_TARGET:
        raise PackError(
            f"formal composition requires {REUSED_TARGET} reused + {NEW_TARGET} new"
        )
    candidates = reusable + selected_new
    scene_ids = [item["record"]["scene_id"] for item in candidates]
    image_ids = [item["record"]["image_sha256"] for item in candidates]
    source_ids = [item["record"]["sample_id"] for item in candidates]
    if (
        len(set(scene_ids)) != FORMAL_TARGET
        or len(set(image_ids)) != FORMAL_TARGET
        or len(set(source_ids)) != FORMAL_TARGET
    ):
        raise PackError("combined v5.2 pack is not scene/image/source unique")
    prior_scenes = _prior_scene_ids(v51)
    new_scene_ids = {item["record"]["scene_id"] for item in selected_new}
    if new_scene_ids & prior_scenes:
        raise PackError("new v5.2 selection overlaps a prior audit or canary")
    if any(
        item["referent_evidence"].get("target_category_visible_instance_counts")
        != [1, 1]
        for item in candidates
    ):
        raise PackError("combined v5.2 selection contains a non-unique referent")

    nearer_slots = _slot_assignment(candidates)
    labeled = [
        (candidate, _target_label(candidate, nearer_slots[candidate["record"]["sample_id"]]))
        for candidate in candidates
    ]
    labeled.sort(key=lambda item: canonical_hash([SELECTION_SEED, item[1]["sample_id"]]))
    ab_counts = Counter(label["audit_targets"]["spatial"]["nearer_slot"] for _, label in labeled)
    if dict(sorted(ab_counts.items())) != EXPECTED_AB_BALANCE:
        raise PackError(f"A/B target balance drift: {dict(ab_counts)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.v52.", dir=output.parent)
    )
    staged = staging_root / "pack"
    try:
        all_labels = [label for _, label in labeled]
        new_labels = [label for candidate, label in labeled if candidate["lineage"] == "new_v52"]
        wrappers: list[dict[str, Any]] = []
        referent_rows: list[dict[str, Any]] = []
        reused_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        for candidate, label in labeled:
            record = candidate["record"]
            label_sha = canonical_hash(label)
            evidence = candidate["referent_evidence"]
            evidence_sha = canonical_hash(evidence)
            wrappers.append(
                {
                    "sample_id": label["sample_id"],
                    "source_sample_id": record["sample_id"],
                    "source_record_sha256": canonical_hash(record),
                    "lineage": candidate["lineage"],
                    "source_record": record,
                }
            )
            referent_rows.append(
                {
                    "sample_id": label["sample_id"],
                    "source_sample_id": record["sample_id"],
                    "evidence_sha256": evidence_sha,
                    "evidence": evidence,
                }
            )
            selected_row = {
                "sample_id": label["sample_id"],
                "scene_id": record["scene_id"],
                "frame_index": record["frame_index"],
                "official_split": "val",
                "image_path": record["image_path"],
                "image_sha256": record["image_sha256"],
                "source_sample_id": record["sample_id"],
                "source_record_sha256": canonical_hash(record),
                "source_payload_sha256": canonical_hash(record["payload"]),
                "label_sha256": label_sha,
                "referent_evidence_sha256": evidence_sha,
                "lineage": candidate["lineage"],
                "nearer_slot": label["audit_targets"]["spatial"]["nearer_slot"],
                "review_required_in_v52": candidate["lineage"] == "new_v52",
            }
            if candidate["lineage"] == "reused_v51":
                selected_row["reuse_binding"] = candidate["reuse_binding"]
                reused_rows.append(
                    {
                        "sample_id": label["sample_id"],
                        "v51_sample_id": candidate["reuse_binding"]["v51_sample_id"],
                        "binding": candidate["reuse_binding"],
                        "reviewer_a": {
                            "decision_sha256": candidate["reuse_binding"][
                                "reviewer_a_decision_sha256"
                            ],
                            "decision": candidate["reused_decisions"]["reviewer_a"],
                        },
                        "reviewer_b": {
                            "decision_sha256": candidate["reuse_binding"][
                                "reviewer_b_decision_sha256"
                            ],
                            "decision": candidate["reused_decisions"]["reviewer_b"],
                        },
                    }
                )
            else:
                selected_row["v3_frame_root_path"] = candidate["frame_root_path"]
            selected_rows.append(selected_row)

        all_labels_path = staged / "labels/GS_dependency/unique_pair.jsonl"
        labels_a_path = staged / "annotator_a/labels/GS_dependency/unique_pair.jsonl"
        labels_b_path = staged / "annotator_b/labels/GS_dependency/unique_pair.jsonl"
        wrappers_path = staged / "provenance/source_records.jsonl"
        evidence_path = staged / "provenance/referent_evidence.jsonl"
        reused_path = staged / "provenance/reused_v51_decisions.jsonl"
        _write_jsonl(all_labels_path, all_labels)
        _write_jsonl(labels_a_path, new_labels)
        _write_jsonl(labels_b_path, new_labels)
        _write_jsonl(wrappers_path, wrappers)
        _write_jsonl(evidence_path, referent_rows)
        _write_jsonl(reused_path, reused_rows)
        (staged / "annotator_a/annotations").mkdir(parents=True)
        (staged / "annotator_b/annotations").mkdir(parents=True)

        v51_public_binding = {
            key: copy.deepcopy(value)
            for key, value in v51.items()
            if key not in {"reusable", "manifest"}
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PACK_PROTOCOL,
            "audit_round": AUDIT_ROUND,
            "audit_contract": AUDIT_CONTRACT,
            "stage": "formal_pending_review",
            "forms": ["grounding_bbox", "pairwise_depth"],
            "records": FORMAL_TARGET,
            "selected": selected_rows,
            "selection": {
                "rule": (
                    "fixed_v51_exact_one_reuse_then_strict_unique_referent_"
                    "best_frame_per_new_scene_then_stable_hash"
                ),
                "seed": SELECTION_SEED,
                "slot_seed": SLOT_SEED,
                "target": FORMAL_TARGET,
                "reused_target": REUSED_TARGET,
                "new_target": NEW_TARGET,
                "selected_reused": REUSED_TARGET,
                "selected_new": NEW_TARGET,
                "selected_unique_scenes": len(set(scene_ids)),
                "selected_unique_images": len(set(image_ids)),
                "selected_unique_source_records": len(set(source_ids)),
                "new_capacity_available": inventory["unique_referent_eligible_scenes"],
                "thresholds": copy.deepcopy(v51_filter.STRICT_THRESHOLDS),
                "referent_method": REFERENT_METHOD,
                "target_category_visible_instance_counts": [1, 1],
                "nearer_slot_balance": dict(sorted(ab_counts.items())),
            },
            "exclusion_checks": {
                "new_vs_all_prior_audits_and_canaries_overlap": sorted(
                    new_scene_ids & prior_scenes
                ),
                "new_vs_v51_nonreuse_33_overlap": sorted(
                    new_scene_ids & set(v51["nonreused_scene_ids"])
                ),
                "new_vs_v51_reused_7_overlap": sorted(
                    new_scene_ids & set(v51["reused_scene_ids"])
                ),
                "combined_scene_duplicates": FORMAL_TARGET - len(set(scene_ids)),
                "partial_v3_roots_consumed": 0,
            },
            "review_contract": {
                "status": "PENDING_HUMAN_REVIEW",
                "reused_v51_decisions_per_side": REUSED_TARGET,
                "new_decisions_required_per_side": NEW_TARGET,
                "new_decisions_required_total": 2 * NEW_TARGET,
                "reviewer_a_expected_new_cases": NEW_TARGET,
                "reviewer_b_expected_new_cases": NEW_TARGET,
                "reviewer_keys_must_be_distinct": True,
                "adjudicate_every_A_B_disagreement": True,
                "reviewer_c_expected_cases": "exactly_the_A_B_disagreement_set",
                "negative_decision_requires_error_code_and_concrete_comment": True,
                "formal_pass_requires_resolved_correct": FORMAL_TARGET,
                "builder_wrote_annotation_decisions": 0,
            },
            "interface_contract": {
                "protocol": public_interface.INTERFACE_PROTOCOL,
                "grounding_response_grammar": public_interface.GROUNDING_GRAMMAR,
                "spatial_response_grammar": public_interface.SPATIAL_GRAMMAR,
                "oracle_matching_of_predicted_detections_allowed": False,
                "multiple_detections_per_slot_allowed": False,
                "category_referent_cardinality": "exactly_one_visible_instance_per_target",
            },
            "provenance": {
                "v51_formal_binding": v51_public_binding,
                "v3_frame_root_bindings": inventory["complete_v3_roots"],
                "source_exclusions_binding": inventory["source_exclusions_binding"],
                "source_records": _relative_binding(staged, wrappers_path),
                "referent_evidence": _relative_binding(staged, evidence_path),
                "reused_v51_decisions": _relative_binding(staged, reused_path),
                "all_labels": _relative_binding(staged, all_labels_path),
                "reviewer_a_labels": _relative_binding(staged, labels_a_path),
                "reviewer_b_labels": _relative_binding(staged, labels_b_path),
                "builder": file_binding(Path(__file__)),
                "checker": file_binding(
                    ROOT / "tools/check_scannet_rgb_depth_v52_unique_referent_formal.py"
                ),
                "public_interface": file_binding(Path(public_interface.__file__)),
                "deep_source_rehash": deep_rehash,
            },
            "inventory": inventory,
            "training_materialization": {
                "eligible": False,
                "allowed_use": "held_out_blind_audit_or_benchmark_only",
                "official_split": "val",
            },
        }
        manifest["integrity"] = {
            "selected_sha256": canonical_hash(selected_rows),
            "manifest_payload_sha256": manifest_payload_hash(manifest),
        }
        _write_json(staged / "manifest.json", manifest)
        staged.rename(output)
        return manifest
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _inventory_document(
    v51: dict[str, Any], scan: dict[str, Any], selection: dict[str, int]
) -> dict[str, Any]:
    capacity = selection["eligible_scenes"]
    return {
        "schema": "gsa-depth-v5.2-unique-referent-inventory-v1",
        "stage": "inventory",
        "write_count": 0,
        "reused_v51_exact_one": REUSED_TARGET,
        "fresh_exact_one_scene_capacity": capacity,
        "fresh_required": NEW_TARGET,
        "formal_total_if_ready": FORMAL_TARGET if capacity >= NEW_TARGET else REUSED_TARGET + capacity,
        "formal_freeze_ready": capacity >= NEW_TARGET,
        "capacity_shortfall": max(0, NEW_TARGET - capacity),
        "human_decisions_after_freeze": {
            "reviewer_a": NEW_TARGET,
            "reviewer_b": NEW_TARGET,
            "total_A_plus_B": 2 * NEW_TARGET,
            "reviewer_c": "one decision per A/B disagreement",
        },
        "selection": selection,
        "scan": scan,
        "v51_binding": {
            "status": v51["status"],
            "files": v51["files"],
            "reused_scene_ids": v51["reused_scene_ids"],
            "nonreused_scene_ids": v51["nonreused_scene_ids"],
            "reused_decision_bindings_sha256": v51[
                "reused_decision_bindings_sha256"
            ],
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inventory", "formal"), default="inventory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--inventory-out",
        type=Path,
        help="optional immutable inventory report path; never a dataset output",
    )
    parser.add_argument("--v51-pack", type=Path, default=DEFAULT_V51_PACK)
    parser.add_argument("--v51-report", type=Path, default=DEFAULT_V51_REPORT)
    parser.add_argument("--v3-root-parent", type=Path, default=DEFAULT_V3_ROOT_PARENT)
    parser.add_argument(
        "--frame-root",
        action="append",
        type=Path,
        help="explicit complete v3 root; otherwise discover complete roots under --v3-root-parent",
    )
    parser.add_argument("--scannet-root", type=Path, default=sensor.DEFAULT_SCANNET_ROOT)
    parser.add_argument("--split-root", type=Path, default=sensor.DEFAULT_SPLIT_ROOT)
    parser.add_argument("--label-map", type=Path, default=sensor.DEFAULT_LABEL_MAP)
    parser.add_argument(
        "--source-exclusions", type=Path, default=sensor.DEFAULT_SOURCE_EXCLUSIONS
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.stage == "formal" and args.output_dir.exists():
            raise PackError(f"refusing to overwrite existing output: {args.output_dir}")
        if args.frame_root:
            roots = [path.expanduser().resolve() for path in args.frame_root]
            ignored: list[str] = []
        else:
            roots, ignored = discover_complete_v3_roots(args.v3_root_parent)
        if len(roots) != len(set(roots)):
            raise PackError("duplicate --frame-root arguments")
        v51 = load_v51_reuse_binding(args.v51_pack, args.v51_report)
        candidates, scan = scan_new_candidates(args, v51, roots, ignored)
        selected_new, selection = select_new_scene_unique(candidates)
        inventory = _inventory_document(v51, scan, selection)
        if args.stage == "inventory":
            if args.inventory_out is not None:
                if args.inventory_out.expanduser().resolve().exists():
                    raise PackError(
                        f"refusing to overwrite inventory report: {args.inventory_out}"
                    )
                _atomic_json(args.inventory_out, inventory)
            print(json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if selection["eligible_scenes"] < NEW_TARGET:
            raise PackError(
                f"CAPACITY_BLOCKED_{NEW_TARGET}: only {selection['eligible_scenes']} "
                "fresh exact-one scene-unique candidates; no output written"
            )
        if len(selected_new) != NEW_TARGET:
            raise PackError("fresh selection count drift; no output written")
        combined = v51["reusable"] + selected_new
        deep_rehash = _deep_verify_selected(combined)
        manifest = freeze_formal_pack(
            args.output_dir,
            v51["reusable"],
            selected_new,
            scan,
            v51,
            deep_rehash,
        )
        print(
            json.dumps(
                {
                    "status": "FORMAL_PACK_FROZEN_PENDING_66_AB_DECISIONS",
                    "output_dir": str(args.output_dir.expanduser().resolve()),
                    "manifest_sha256": file_sha256(
                        args.output_dir.expanduser().resolve() / "manifest.json"
                    ),
                    "manifest_payload_sha256": manifest["integrity"][
                        "manifest_payload_sha256"
                    ],
                    "records": FORMAL_TARGET,
                    "reused": REUSED_TARGET,
                    "new_pending_review": NEW_TARGET,
                    "A_plus_B_decisions_required": 2 * NEW_TARGET,
                    "reviewer_c": "all A/B disagreements",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (PackError, sensor.SourceError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
