#!/usr/bin/env python3
"""Extract builder-compatible frame bundles from official-val ScanNet .sens files.

This is a CPU-only held-out benchmark preparation tool. It never produces
training-eligible data and never downloads source artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from tools import extract_scannet_train_sens as sens  # noqa: E402


SCHEMA = "gsa-scannet-official-val-frame-bundle-v1"
PROTOCOL = "gsa-scannet-val-frame-bundle-extraction-v1"
SELECTION_SEED = "gsa-scannet-val-common-annotation-frames-v1"
DEFAULT_SCENE = "scene0011_00"
DEFAULT_OUTPUT = ROOT / "data/scannet_val_frame_bundles/canary_v1"
DEFAULT_SOURCE_EXCLUSIONS = ROOT / "configs/scannet_source_exclusions_v1.json"
SOURCE_EXCLUSIONS_SHA256 = (
    "34488cc9d3260924ae98c95bfeb6b1c57ab6e116b381a29cad8ae337d69c6cd5"
)
MAX_FRAMES_PER_SCENE = 64
FRAME_BUNDLE_RE = re.compile(r"^\d+-scannet-(scene\d{4}_\d{2})$")
INSTANCE_MEMBER_RE = re.compile(r"^instance-filt/(\d+)\.png$")
SEMANTIC_MEMBER_RE = re.compile(r"^label-filt/(\d+)\.png$")
CALIBRATION_NAMES = (
    "intrinsic_color.txt",
    "intrinsic_depth.txt",
    "extrinsic_color.txt",
    "extrinsic_depth.txt",
)


class ExtractError(RuntimeError):
    """The requested extraction violates the frozen held-out contract."""


@dataclass(frozen=True)
class ScenePaths:
    scene_id: str
    scene_dir: Path
    sens: Path
    metadata: Path
    aggregation: Path
    instance_zip: Path
    semantic_zip: Path


@dataclass(frozen=True)
class SceneInspection:
    paths: ScenePaths
    header: sens.SensHeader
    common_frame_count: int
    requested_max_frames: int
    selected_indices: tuple[int, ...]
    instance_members: dict[int, dict[str, Any]]
    semantic_members: dict[int, dict[str, Any]]
    source_files: dict[str, dict[str, Any]]
    aggregation_object_count: int


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_document(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ExtractError(f"missing or empty source artifact: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sens.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _artifact_document(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ExtractError(f"missing or empty output artifact: {path}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": sens.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _paths(scannet_root: Path, scene_id: str) -> ScenePaths:
    scene_dir = scannet_root / scene_id
    return ScenePaths(
        scene_id=scene_id,
        scene_dir=scene_dir,
        sens=scene_dir / f"{scene_id}.sens",
        metadata=scene_dir / f"{scene_id}.txt",
        aggregation=scene_dir / f"{scene_id}.aggregation.json",
        instance_zip=scene_dir / f"{scene_id}_2d-instance-filt.zip",
        semantic_zip=scene_dir / f"{scene_id}_2d-label-filt.zip",
    )


def _validate_aggregation(path: Path, scene_id: str) -> int:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractError(f"invalid aggregation JSON: {path}") from exc
    if document.get("sceneId") != f"scannet.{scene_id}":
        raise ExtractError(f"aggregation sceneId mismatch: {path}")
    groups = document.get("segGroups")
    if not isinstance(groups, list) or not groups:
        raise ExtractError(f"aggregation has no object groups: {path}")
    return len(groups)


def load_source_exclusions(
    path: Path, scene_to_split: dict[str, str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ExtractError(f"missing or empty source exclusions: {path}")
    observed_sha256 = sens.file_sha256(path)
    if observed_sha256 != SOURCE_EXCLUSIONS_SHA256:
        raise ExtractError(f"source-exclusion manifest hash mismatch: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractError(f"invalid source-exclusion manifest: {path}") from exc
    rows = document.get("exclusions")
    if document.get("schema") != "gsa-scannet-source-exclusions-v1" or not isinstance(
        rows, list
    ):
        raise ExtractError(f"source-exclusion contract mismatch: {path}")
    exclusions: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ExtractError(f"invalid source-exclusion row: {path}")
        scene_id = row.get("scene_id")
        if (
            not isinstance(scene_id, str)
            or sens.SCENE_RE.fullmatch(scene_id) is None
            or scene_id in exclusions
            or row.get("official_split") != "val"
            or scene_to_split.get(scene_id) != "val"
            or not isinstance(row.get("artifact"), str)
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or not isinstance(row.get("reason"), str)
        ):
            raise ExtractError(f"invalid source-exclusion row: {path}/{scene_id}")
        exclusions[scene_id] = row
    return exclusions, {
        "path": str(path.resolve()),
        "sha256": observed_sha256,
        "size_bytes": path.stat().st_size,
        "schema": document["schema"],
        "excluded_scene_ids": sorted(exclusions),
        "excluded_scene_ids_sha256": canonical_hash(sorted(exclusions)),
    }


def _archive_members(
    path: Path, pattern: re.Pattern[str]
) -> dict[int, dict[str, Any]]:
    members: dict[int, dict[str, Any]] = {}
    try:
        with ZipFile(path) as archive:
            for info in archive.infolist():
                match = pattern.fullmatch(info.filename)
                if match is None:
                    continue
                index = int(match.group(1))
                if index in members:
                    raise ExtractError(f"duplicate frame index in {path}: {index}")
                if info.is_dir() or info.file_size <= 0:
                    raise ExtractError(f"empty annotation member: {path}:{info.filename}")
                members[index] = {
                    "member": info.filename,
                    "crc32": f"{info.CRC:08x}",
                    "size_bytes": info.file_size,
                    "compressed_size_bytes": info.compress_size,
                }
    except (OSError, BadZipFile) as exc:
        raise ExtractError(f"invalid annotation archive: {path}") from exc
    if not members:
        raise ExtractError(f"annotation archive has no indexed PNG members: {path}")
    return members


def _validated_member_documents(
    *,
    archive_path: Path,
    member_documents: dict[int, dict[str, Any]],
    selected_indices: tuple[int, ...],
    expected_size: tuple[int, int],
) -> dict[int, dict[str, Any]]:
    validated: dict[int, dict[str, Any]] = {}
    try:
        with ZipFile(archive_path) as archive:
            for index in selected_indices:
                source = member_documents[index]
                payload = archive.read(source["member"])
                with Image.open(BytesIO(payload)) as image:
                    image.load()
                    size = image.size
                    mode = image.mode
                    array = np.asarray(image)
                if size != expected_size or array.ndim != 2:
                    raise ExtractError(
                        f"annotation member shape mismatch: "
                        f"{archive_path}:{source['member']} {size}/{array.shape}"
                    )
                validated[index] = {
                    **source,
                    "sha256": sens.bytes_sha256(payload),
                    "image_width": size[0],
                    "image_height": size[1],
                    "image_mode": mode,
                }
    except (BadZipFile, KeyError, OSError, UnidentifiedImageError, ValueError) as exc:
        raise ExtractError(f"invalid selected annotation member: {archive_path}") from exc
    return validated


def stable_frame_indices(
    scene_id: str, available: Iterable[int], max_frames: int
) -> tuple[int, ...]:
    if not 1 <= max_frames <= MAX_FRAMES_PER_SCENE:
        raise ExtractError(
            f"max_frames must be in [1,{MAX_FRAMES_PER_SCENE}], got {max_frames}"
        )
    ordered = sorted(
        set(available),
        key=lambda index: hashlib.sha256(
            f"{SELECTION_SEED}|{scene_id}|{index}".encode("utf-8")
        ).hexdigest(),
    )
    return tuple(ordered[:max_frames])


def inspect_scene(
    *,
    scannet_root: Path,
    scene_to_split: dict[str, str],
    source_exclusions: dict[str, dict[str, Any]],
    scene_id: str,
    max_frames: int,
) -> SceneInspection:
    if sens.SCENE_RE.fullmatch(scene_id) is None:
        raise ExtractError(f"invalid ScanNet scene id: {scene_id}")
    observed_split = scene_to_split.get(scene_id)
    if observed_split != "val":
        raise ExtractError(
            f"scene {scene_id} is not official val (observed split: "
            f"{observed_split or 'unknown'})"
        )
    if scene_id in source_exclusions:
        row = source_exclusions[scene_id]
        raise ExtractError(
            f"scene is frozen in source exclusions: {scene_id}/"
            f"{row.get('artifact')}/{row.get('reason')}"
        )
    paths = _paths(scannet_root, scene_id)
    for path in (
        paths.sens,
        paths.metadata,
        paths.aggregation,
        paths.instance_zip,
        paths.semantic_zip,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ExtractError(f"missing or empty source artifact: {path}")

    header = sens.read_sens_header(paths.sens)
    if not np.isclose(header.depth_shift, 1000.0, rtol=0.0, atol=1e-6):
        raise ExtractError(
            f"builder-compatible depth requires depth_shift=1000, got "
            f"{header.depth_shift}: {scene_id}"
        )
    sens.validate_metadata(header, sens.parse_scene_metadata(paths.metadata))
    aggregation_object_count = _validate_aggregation(paths.aggregation, scene_id)
    instance_members = _archive_members(paths.instance_zip, INSTANCE_MEMBER_RE)
    semantic_members = _archive_members(paths.semantic_zip, SEMANTIC_MEMBER_RE)
    common = {
        index
        for index in instance_members.keys() & semantic_members.keys()
        if 0 <= index < header.num_frames and index <= 999_999
    }
    selected = stable_frame_indices(scene_id, common, max_frames)
    if not selected:
        raise ExtractError(f"no common instance/semantic frame indices: {scene_id}")

    expected_size = (header.color_width, header.color_height)
    instance_members = _validated_member_documents(
        archive_path=paths.instance_zip,
        member_documents=instance_members,
        selected_indices=selected,
        expected_size=expected_size,
    )
    semantic_members = _validated_member_documents(
        archive_path=paths.semantic_zip,
        member_documents=semantic_members,
        selected_indices=selected,
        expected_size=expected_size,
    )

    source_files = {
        "sens": _source_document(paths.sens),
        "metadata": _source_document(paths.metadata),
        "aggregation": _source_document(paths.aggregation),
        "instance_zip": _source_document(paths.instance_zip),
        "semantic_zip": _source_document(paths.semantic_zip),
    }
    return SceneInspection(
        paths=paths,
        header=header,
        common_frame_count=len(common),
        requested_max_frames=max_frames,
        selected_indices=selected,
        instance_members=instance_members,
        semantic_members=semantic_members,
        source_files=source_files,
        aggregation_object_count=aggregation_object_count,
    )


def _header_matches(left: sens.SensHeader, right: sens.SensHeader) -> bool:
    scalar_fields = (
        "version",
        "sensor_name",
        "color_compression_id",
        "depth_compression_id",
        "color_width",
        "color_height",
        "depth_width",
        "depth_height",
        "depth_shift",
        "num_frames",
        "frame_data_offset",
    )
    if any(getattr(left, name) != getattr(right, name) for name in scalar_fields):
        return False
    return all(
        np.array_equal(getattr(left, name), getattr(right, name))
        for name in (
            "intrinsic_color",
            "extrinsic_color",
            "intrinsic_depth",
            "extrinsic_depth",
        )
    )


def _stream_selected_frames(
    path: Path,
    expected_header: sens.SensHeader,
    selected_indices: tuple[int, ...],
) -> dict[int, sens.FramePayload]:
    selected = set(selected_indices)
    frames: dict[int, sens.FramePayload] = {}
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = sens._read_header(handle)
        if not _header_matches(header, expected_header):
            raise ExtractError(f".sens header changed after inspection: {path}")
        for frame_index in range(header.num_frames):
            pose = sens._matrix(handle, f"frame[{frame_index}].camera_to_world")
            timestamp_color = int(
                sens._scalar(handle, "<Q", f"frame[{frame_index}].timestamp_color")
            )
            timestamp_depth = int(
                sens._scalar(handle, "<Q", f"frame[{frame_index}].timestamp_depth")
            )
            color_size = int(
                sens._scalar(handle, "<Q", f"frame[{frame_index}].color_size")
            )
            depth_size = int(
                sens._scalar(handle, "<Q", f"frame[{frame_index}].depth_size")
            )
            if not 0 < color_size <= sens.MAX_FRAME_PAYLOAD_BYTES:
                raise ExtractError(f"invalid color payload size at frame {frame_index}")
            if not 0 < depth_size <= sens.MAX_FRAME_PAYLOAD_BYTES:
                raise ExtractError(f"invalid depth payload size at frame {frame_index}")
            if handle.tell() + color_size + depth_size > file_size:
                raise ExtractError(f"frame payload exceeds .sens size at {frame_index}")
            if frame_index in selected:
                color_bytes = sens._read_exact(
                    handle, color_size, f"frame[{frame_index}].color"
                )
                depth_bytes = sens._read_exact(
                    handle, depth_size, f"frame[{frame_index}].depth"
                )
                frames[frame_index] = sens.FramePayload(
                    frame_index=frame_index,
                    camera_to_world=pose,
                    timestamp_color=timestamp_color,
                    timestamp_depth=timestamp_depth,
                    color_bytes=color_bytes,
                    depth_bytes=depth_bytes,
                )
            else:
                handle.seek(color_size + depth_size, 1)
            if len(frames) == len(selected):
                break
    if set(frames) != selected:
        missing = sorted(selected - set(frames))
        raise ExtractError(f"selected frames missing from .sens: {missing[:5]}")
    return frames


def _decode_frames(
    inspection: SceneInspection,
) -> dict[int, tuple[sens.FramePayload, np.ndarray, np.ndarray]]:
    frames = _stream_selected_frames(
        inspection.paths.sens, inspection.header, inspection.selected_indices
    )
    decoded: dict[int, tuple[sens.FramePayload, np.ndarray, np.ndarray]] = {}
    for index in inspection.selected_indices:
        frame = frames[index]
        color = sens.decode_color(inspection.header, frame.color_bytes)
        depth = sens.decode_depth(inspection.header, frame.depth_bytes)
        decoded[index] = (frame, color, depth)
    return decoded


def _verify_source_identities(inspection: SceneInspection) -> None:
    paths = inspection.paths
    current = {
        "sens": _source_document(paths.sens),
        "metadata": _source_document(paths.metadata),
        "aggregation": _source_document(paths.aggregation),
        "instance_zip": _source_document(paths.instance_zip),
        "semantic_zip": _source_document(paths.semantic_zip),
    }
    if current != inspection.source_files:
        changed = sorted(
            name
            for name in current
            if current.get(name) != inspection.source_files.get(name)
        )
        raise ExtractError(
            f"source identities changed after inspection: "
            f"{inspection.paths.scene_id}/{','.join(changed)}"
        )


def _write_matrix(path: Path, value: np.ndarray) -> None:
    np.savetxt(path, value, fmt="%.9g")


def _use_policy() -> dict[str, Any]:
    return {
        "official_split": "val",
        "held_out_benchmark_preparation_only": True,
        "training_materialization_eligible": False,
        "may_enter_training": False,
        "redistribution_eligible": False,
    }


def _binding() -> dict[str, Any]:
    extractor = Path(__file__).resolve()
    reader = Path(sens.__file__).resolve()
    return {
        "extractor_path": str(extractor),
        "extractor_sha256": sens.file_sha256(extractor),
        "reader_path": str(reader),
        "reader_sha256": sens.file_sha256(reader),
        "reader_functions": [
            "_read_header",
            "_matrix",
            "_scalar",
            "_read_exact",
            "decode_color",
            "decode_depth",
        ],
    }


def _inspection_document(inspection: SceneInspection) -> dict[str, Any]:
    return {
        "scene_id": inspection.paths.scene_id,
        "official_split": "val",
        "sensor_header": sens.header_document(inspection.header),
        "aggregation_object_count": inspection.aggregation_object_count,
        "common_annotation_frame_count": inspection.common_frame_count,
        "requested_max_frames": inspection.requested_max_frames,
        "selected_frame_indices": list(inspection.selected_indices),
        "selected_frame_indices_sha256": canonical_hash(
            list(inspection.selected_indices)
        ),
        "selected_annotation_members": {
            "instance": [
                inspection.instance_members[index]
                for index in inspection.selected_indices
            ],
            "semantic": [
                inspection.semantic_members[index]
                for index in inspection.selected_indices
            ],
        },
        "source_files": inspection.source_files,
    }


def check_scene(
    inspection: SceneInspection,
    split_documents: dict[str, dict[str, Any]],
    source_exclusions_binding: dict[str, Any],
    frame_root_exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    decoded = _decode_frames(inspection)
    _verify_source_identities(inspection)
    decoded_rows = []
    for index in inspection.selected_indices:
        frame, color, depth = decoded[index]
        decoded_rows.append(
            {
                "frame_index": index,
                "color_shape": list(color.shape),
                "color_dtype": str(color.dtype),
                "depth_shape": list(depth.shape),
                "depth_dtype": str(depth.dtype),
                "pose_finite": bool(np.isfinite(frame.camera_to_world).all()),
            }
        )
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "stage": "check_only",
        "verdict": "CHECK_PASS_VAL_AUDIT_ONLY",
        "cpu_only": True,
        "network_calls": 0,
        "write_count": 0,
        "training_materialization_eligible": False,
        "use_policy": _use_policy(),
        "extractor": _binding(),
        "official_splits": split_documents,
        "source_exclusions": source_exclusions_binding,
        "frame_root_exclusions": frame_root_exclusions,
        "source_identities_reverified_after_decode": True,
        "selection": {
            "seed": SELECTION_SEED,
            "requested_max_frames_per_scene": inspection.requested_max_frames,
            "selected_frame_count": len(inspection.selected_indices),
        },
        "scene": _inspection_document(inspection),
        "decoded_frames": decoded_rows,
    }


def _require_output_empty(output_root: Path) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise ExtractError(f"output exists and is not a directory: {output_root}")
        if any(output_root.iterdir()):
            raise ExtractError(f"refusing nonempty output root: {output_root}")


def materialize_bundles(
    *,
    inspections: list[SceneInspection],
    split_documents: dict[str, dict[str, Any]],
    output_root: Path,
    stage: str,
    start_ordinal: int,
    scene_list_binding: dict[str, Any],
    source_exclusions_binding: dict[str, Any],
    frame_root_exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    if stage not in {"canary", "batch"}:
        raise ExtractError(f"materialization stage must be canary or batch: {stage}")
    if not inspections:
        raise ExtractError("no scenes selected")
    if len({item.paths.scene_id for item in inspections}) != len(inspections):
        raise ExtractError("duplicate scene IDs")
    requested_limits = {item.requested_max_frames for item in inspections}
    if len(requested_limits) != 1:
        raise ExtractError("all scenes must use one requested max_frames value")
    if not frame_root_exclusions:
        raise ExtractError("materialization requires at least one frame-root exclusion")
    if start_ordinal < 0 or start_ordinal + len(inspections) - 1 > 999_999:
        raise ExtractError("output ordinal range must be within [0,999999]")
    _require_output_empty(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        scene_documents: list[dict[str, Any]] = []
        total_frames = 0
        for offset, inspection in enumerate(inspections):
            ordinal = start_ordinal + offset
            directory_name = f"{ordinal:06d}-scannet-{inspection.paths.scene_id}"
            scene_root = temp_root / directory_name
            scene_root.mkdir()
            decoded = _decode_frames(inspection)
            calibration = {
                "intrinsic_color.txt": inspection.header.intrinsic_color,
                "intrinsic_depth.txt": inspection.header.intrinsic_depth,
                "extrinsic_color.txt": inspection.header.extrinsic_color,
                "extrinsic_depth.txt": inspection.header.extrinsic_depth,
            }
            for name, matrix in calibration.items():
                _write_matrix(scene_root / name, matrix)

            frame_documents = []
            for index in inspection.selected_indices:
                frame, color, depth = decoded[index]
                rgb_path = scene_root / f"{index:06d}-rgb.png"
                depth_path = scene_root / f"{index:06d}-depth.png"
                pose_path = scene_root / f"{index:06d}.txt"
                Image.fromarray(color).save(rgb_path)
                Image.fromarray(depth).save(depth_path)
                _write_matrix(pose_path, frame.camera_to_world)
                frame_documents.append(
                    {
                        "frame_index": index,
                        "timestamp_color": frame.timestamp_color,
                        "timestamp_depth": frame.timestamp_depth,
                        "source_payloads": {
                            "color_compressed_sha256": sens.bytes_sha256(
                                frame.color_bytes
                            ),
                            "color_compressed_size_bytes": len(frame.color_bytes),
                            "depth_compressed_sha256": sens.bytes_sha256(
                                frame.depth_bytes
                            ),
                            "depth_compressed_size_bytes": len(frame.depth_bytes),
                        },
                        "outputs": {
                            "rgb": _artifact_document(rgb_path, temp_root),
                            "depth": _artifact_document(depth_path, temp_root),
                            "pose": _artifact_document(pose_path, temp_root),
                        },
                    }
                )
            _verify_source_identities(inspection)
            total_frames += len(frame_documents)
            document = _inspection_document(inspection)
            document.update(
                {
                    "ordinal": ordinal,
                    "output_directory": directory_name,
                    "calibration_outputs": {
                        name: _artifact_document(scene_root / name, temp_root)
                        for name in CALIBRATION_NAMES
                    },
                    "frames": frame_documents,
                    "source_identities_reverified_after_decode": True,
                }
            )
            scene_documents.append(document)

        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "stage": "one_scene_canary" if stage == "canary" else "batch",
            "verdict": (
                "CANARY_PASS_VAL_AUDIT_ONLY"
                if stage == "canary"
                else "BATCH_PASS_VAL_AUDIT_ONLY"
            ),
            "execution": {
                "cpu_only": True,
                "network_calls": 0,
                "gpu_required": False,
                "scene_count": len(scene_documents),
                "frame_count": total_frames,
            },
            "training_materialization_eligible": False,
            "use_policy": _use_policy(),
            "extractor": _binding(),
            "official_splits": split_documents,
            "source_exclusions": source_exclusions_binding,
            "frame_root_exclusions": frame_root_exclusions,
            "scene_list_binding": scene_list_binding,
            "selection": {
                "seed": SELECTION_SEED,
                "maximum_frames_per_scene": MAX_FRAMES_PER_SCENE,
                "requested_max_frames_per_scene": next(iter(requested_limits)),
                "selected_only_from_instance_semantic_member_intersection": True,
                "scene_unique": True,
            },
            "scenes": scene_documents,
        }
        manifest["integrity"] = {
            "manifest_payload_sha256": canonical_hash(manifest),
            "scene_ids_sha256": canonical_hash(
                [item["scene_id"] for item in scene_documents]
            ),
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8")
        (temp_root / "manifest.json").write_bytes(manifest_bytes)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        (temp_root / "manifest.sha256").write_text(
            f"{manifest_sha256}  manifest.json\n", encoding="ascii"
        )
        _verify_frame_root_bindings(frame_root_exclusions)
        if output_root.exists():
            _require_output_empty(output_root)
            output_root.rmdir()
        temp_root.replace(output_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "stage": manifest["stage"],
        "verdict": manifest["verdict"],
        "output_root": str(output_root.resolve()),
        "scene_count": len(scene_documents),
        "frame_count": total_frames,
        "manifest_sha256": manifest_sha256,
        "training_materialization_eligible": False,
    }


def _read_scene_list(path: Path) -> tuple[list[str], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ExtractError(f"missing or empty scene list: {path}")
    scene_ids = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        if sens.SCENE_RE.fullmatch(value) is None:
            raise ExtractError(f"invalid scene id at {path}:{line_number}: {value}")
        scene_ids.append(value)
    if not scene_ids:
        raise ExtractError(f"scene list has no scene IDs: {path}")
    if len(scene_ids) != len(set(scene_ids)):
        raise ExtractError(f"scene list contains duplicates: {path}")
    return scene_ids, {
        "path": str(path.resolve()),
        "sha256": sens.file_sha256(path),
        "size_bytes": path.stat().st_size,
        "scene_count": len(scene_ids),
    }


def _frame_root_binding(path: Path) -> tuple[set[str], dict[str, Any]]:
    if not path.is_dir():
        raise ExtractError(f"exclude frame root is not a directory: {path}")
    result: set[str] = set()
    for child in path.iterdir():
        if not child.is_dir():
            continue
        match = FRAME_BUNDLE_RE.fullmatch(child.name)
        if match is None:
            continue
        scene_id = match.group(1)
        if scene_id in result:
            raise ExtractError(f"duplicate scene in exclude frame root: {scene_id}")
        result.add(scene_id)
    scene_ids = sorted(result)
    return result, {
        "path": str(path.resolve()),
        "scene_count": len(scene_ids),
        "scene_ids": scene_ids,
        "scene_ids_sha256": canonical_hash(scene_ids),
    }


def _verify_frame_root_bindings(bindings: list[dict[str, Any]]) -> None:
    for expected in bindings:
        _, current = _frame_root_binding(Path(expected["path"]))
        if current != expected:
            raise ExtractError(
                f"frame-root exclusion changed during extraction: {expected['path']}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "canary", "batch"), default="check")
    parser.add_argument("--scene-id", action="append")
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--scannet-root", type=Path, default=sens.DEFAULT_SCANNET_ROOT)
    parser.add_argument("--split-root", type=Path, default=sens.DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--source-exclusions", type=Path, default=DEFAULT_SOURCE_EXCLUSIONS
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude-frame-root", type=Path, action="append", default=[])
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--start-ordinal", type=int, default=0)
    return parser.parse_args(argv)


def _requested_scenes(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    if args.stage == "batch":
        if args.scene_id or args.scene_list is None:
            raise ExtractError("batch requires --scene-list and forbids --scene-id")
        return _read_scene_list(args.scene_list)
    if args.scene_list is not None:
        raise ExtractError(f"{args.stage} forbids --scene-list")
    scene_ids = args.scene_id or [DEFAULT_SCENE]
    if len(scene_ids) != 1:
        raise ExtractError(f"{args.stage} requires exactly one --scene-id")
    return scene_ids, {
        "mode": "single_scene",
        "scene_ids_sha256": canonical_hash(scene_ids),
        "scene_count": 1,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not 1 <= args.max_frames <= MAX_FRAMES_PER_SCENE:
            raise ExtractError(
                f"max_frames must be in [1,{MAX_FRAMES_PER_SCENE}], got {args.max_frames}"
            )
        if args.stage in {"canary", "batch"} and not args.exclude_frame_root:
            raise ExtractError(
                "materialization requires at least one --exclude-frame-root"
            )
        scene_ids, scene_list_binding = _requested_scenes(args)
        split_documents, scene_to_split = sens.load_split_manifests(args.split_root)
        source_exclusions, source_exclusions_binding = load_source_exclusions(
            args.source_exclusions, scene_to_split
        )
        excluded: set[str] = set()
        frame_root_exclusions: list[dict[str, Any]] = []
        for root in args.exclude_frame_root:
            root_scenes, root_binding = _frame_root_binding(root)
            excluded.update(root_scenes)
            frame_root_exclusions.append(root_binding)
        overlap = sorted(set(scene_ids) & excluded)
        if overlap:
            raise ExtractError(
                f"requested scenes overlap an excluded frame root: {','.join(overlap[:8])}"
            )
        inspections = [
            inspect_scene(
                scannet_root=args.scannet_root,
                scene_to_split=scene_to_split,
                source_exclusions=source_exclusions,
                scene_id=scene_id,
                max_frames=args.max_frames,
            )
            for scene_id in scene_ids
        ]
        if args.stage == "check":
            result = check_scene(
                inspections[0],
                split_documents,
                source_exclusions_binding,
                frame_root_exclusions,
            )
        else:
            result = materialize_bundles(
                inspections=inspections,
                split_documents=split_documents,
                output_root=args.output_root,
                stage=args.stage,
                start_ordinal=args.start_ordinal,
                scene_list_binding=scene_list_binding,
                source_exclusions_binding=source_exclusions_binding,
                frame_root_exclusions=frame_root_exclusions,
            )
    except (ExtractError, sens.SourceError, OSError, BadZipFile, ValueError) as exc:
        print(
            json.dumps(
                {"schema": SCHEMA, "verdict": "BLOCKED", "error": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
