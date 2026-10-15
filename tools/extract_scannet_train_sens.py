#!/usr/bin/env python3
"""Extract one hash-bound RGB-D frame from an official-train ScanNet .sens.

This is an adapter canary, not a training-set builder.  It refuses validation
and test scenes, performs no network or accelerator work, and keeps training
materialization disabled until local ScanNet Terms-of-Use approval evidence is
provided and a separate scale-out review is completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import tempfile
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "gsa-scannet-official-train-sens-adapter-v1"
PROTOCOL = "gsa-scannet-train-sens-cpu-canary-v1"
SCANNET_SPLIT_REVISION = "3830fce7f8b2e48ef047ef7fd76ea5f62903f51c"
SCANNET_SPLIT_SHA256 = {
    "train": "96acca299b7855f02824c496b19077904d80996e7ced1bb9f0dac98f7dd4d0c8",
    "val": "d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083",
    "test": "0214c6a3b1ee516ad653393b0321e7c0394c7662a4b3702eac1ddd7fbc00f7e0",
}
DEFAULT_SCANNET_ROOT = Path(
    "/path/to/datasets/downloads/ScanNet-official/scans"
)
DEFAULT_LABEL_MAP = DEFAULT_SCANNET_ROOT.parent / "tasks/scannetv2-labels.combined.tsv"
DEFAULT_SPLIT_ROOT = ROOT / "configs/scannet_splits"
DEFAULT_OUTPUT = ROOT / "data/scannet_train_sens_adapter/canary_v1"
SCENE_RE = re.compile(r"^scene\d{4}_\d{2}$")
MAX_STRING_BYTES = 1 << 20
MAX_FRAME_PAYLOAD_BYTES = 1 << 30
COLOR_COMPRESSION = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
DEPTH_COMPRESSION = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class SourceError(RuntimeError):
    """The source or requested operation violates the adapter contract."""


@dataclass(frozen=True)
class SensHeader:
    version: int
    sensor_name: str
    intrinsic_color: np.ndarray
    extrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    extrinsic_depth: np.ndarray
    color_compression_id: int
    depth_compression_id: int
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float
    num_frames: int
    frame_data_offset: int


@dataclass(frozen=True)
class FramePayload:
    frame_index: int
    camera_to_world: np.ndarray
    timestamp_color: int
    timestamp_depth: int
    color_bytes: bytes
    depth_bytes: bytes


@dataclass(frozen=True)
class SourcePaths:
    scene_id: str
    scene_dir: Path
    sens: Path
    metadata: Path
    aggregation: Path
    instance_zip: Path
    semantic_zip: Path
    label_map: Path
    download_script: Path


@dataclass(frozen=True)
class RegistrationResult:
    aligned_raw: np.ndarray
    native_valid_pixels: int
    projected_pixels: int
    registered_pixels: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_exact(handle: BinaryIO, size: int, field: str) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise SourceError(f"truncated .sens while reading {field}")
    return value


def _scalar(handle: BinaryIO, fmt: str, field: str) -> int | float:
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(handle, size, field))[0]


def _matrix(handle: BinaryIO, field: str) -> np.ndarray:
    value = np.frombuffer(_read_exact(handle, 64, field), dtype="<f4").copy().reshape(4, 4)
    if not np.isfinite(value).all():
        raise SourceError(f"non-finite matrix in .sens: {field}")
    if not np.allclose(value[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-4):
        raise SourceError(f"invalid homogeneous matrix in .sens: {field}")
    return value.astype(np.float64)


def _read_header(handle: BinaryIO) -> SensHeader:
    version = int(_scalar(handle, "<I", "version"))
    if version != 4:
        raise SourceError(f"unsupported .sens version: {version}; expected 4")
    name_length = int(_scalar(handle, "<Q", "sensor_name_length"))
    if not 0 < name_length <= MAX_STRING_BYTES:
        raise SourceError(f"invalid .sens sensor-name length: {name_length}")
    try:
        sensor_name = _read_exact(handle, name_length, "sensor_name").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceError("invalid UTF-8 sensor name in .sens") from exc

    intrinsic_color = _matrix(handle, "intrinsic_color")
    extrinsic_color = _matrix(handle, "extrinsic_color")
    intrinsic_depth = _matrix(handle, "intrinsic_depth")
    extrinsic_depth = _matrix(handle, "extrinsic_depth")
    color_compression_id = int(_scalar(handle, "<i", "color_compression"))
    depth_compression_id = int(_scalar(handle, "<i", "depth_compression"))
    color_width = int(_scalar(handle, "<I", "color_width"))
    color_height = int(_scalar(handle, "<I", "color_height"))
    depth_width = int(_scalar(handle, "<I", "depth_width"))
    depth_height = int(_scalar(handle, "<I", "depth_height"))
    depth_shift = float(_scalar(handle, "<f", "depth_shift"))
    num_frames = int(_scalar(handle, "<Q", "num_frames"))

    if color_compression_id not in COLOR_COMPRESSION or COLOR_COMPRESSION[color_compression_id] == "unknown":
        raise SourceError(f"unsupported color compression id: {color_compression_id}")
    if depth_compression_id not in DEPTH_COMPRESSION or DEPTH_COMPRESSION[depth_compression_id] == "unknown":
        raise SourceError(f"unsupported depth compression id: {depth_compression_id}")
    if min(color_width, color_height, depth_width, depth_height, num_frames) <= 0:
        raise SourceError("non-positive image dimension or frame count in .sens")
    if max(color_width, color_height, depth_width, depth_height) > 32768:
        raise SourceError("implausible image dimension in .sens")
    if not math.isfinite(depth_shift) or depth_shift <= 0:
        raise SourceError(f"invalid depth shift in .sens: {depth_shift}")

    return SensHeader(
        version=version,
        sensor_name=sensor_name,
        intrinsic_color=intrinsic_color,
        extrinsic_color=extrinsic_color,
        intrinsic_depth=intrinsic_depth,
        extrinsic_depth=extrinsic_depth,
        color_compression_id=color_compression_id,
        depth_compression_id=depth_compression_id,
        color_width=color_width,
        color_height=color_height,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_shift=depth_shift,
        num_frames=num_frames,
        frame_data_offset=handle.tell(),
    )


def read_sens_header(path: Path) -> SensHeader:
    if not path.is_file():
        raise SourceError(f"missing .sens file: {path}")
    with path.open("rb") as handle:
        return _read_header(handle)


def read_sens_frame(path: Path, frame_index: int) -> tuple[SensHeader, FramePayload]:
    if frame_index < 0:
        raise SourceError("frame index must be non-negative")
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = _read_header(handle)
        if frame_index >= header.num_frames:
            raise SourceError(
                f"frame {frame_index} outside .sens range [0,{header.num_frames - 1}]"
            )
        for current_index in range(frame_index + 1):
            pose = _matrix(handle, f"frame[{current_index}].camera_to_world")
            timestamp_color = int(_scalar(handle, "<Q", f"frame[{current_index}].timestamp_color"))
            timestamp_depth = int(_scalar(handle, "<Q", f"frame[{current_index}].timestamp_depth"))
            color_size = int(_scalar(handle, "<Q", f"frame[{current_index}].color_size"))
            depth_size = int(_scalar(handle, "<Q", f"frame[{current_index}].depth_size"))
            if not 0 < color_size <= MAX_FRAME_PAYLOAD_BYTES:
                raise SourceError(f"invalid color payload size at frame {current_index}: {color_size}")
            if not 0 < depth_size <= MAX_FRAME_PAYLOAD_BYTES:
                raise SourceError(f"invalid depth payload size at frame {current_index}: {depth_size}")
            payload_end = handle.tell() + color_size + depth_size
            if payload_end > file_size:
                raise SourceError(f"frame {current_index} payload exceeds .sens file size")
            if current_index != frame_index:
                handle.seek(color_size + depth_size, 1)
                continue
            color_bytes = _read_exact(handle, color_size, f"frame[{current_index}].color")
            depth_bytes = _read_exact(handle, depth_size, f"frame[{current_index}].depth")
            return header, FramePayload(
                frame_index=current_index,
                camera_to_world=pose,
                timestamp_color=timestamp_color,
                timestamp_depth=timestamp_depth,
                color_bytes=color_bytes,
                depth_bytes=depth_bytes,
            )
    raise SourceError(f"failed to locate frame {frame_index}")


def decode_color(header: SensHeader, payload: bytes) -> np.ndarray:
    compression = COLOR_COMPRESSION[header.color_compression_id]
    try:
        if compression == "raw":
            image = np.frombuffer(payload, dtype=np.uint8).reshape(
                header.color_height, header.color_width, 3
            ).copy()
        elif compression in {"png", "jpeg"}:
            with Image.open(BytesIO(payload)) as handle:
                image = np.asarray(handle.convert("RGB")).copy()
        else:
            raise SourceError(f"unsupported color compression: {compression}")
    except (ValueError, OSError, UnidentifiedImageError) as exc:
        raise SourceError("failed to decode .sens color payload") from exc
    expected = (header.color_height, header.color_width, 3)
    if image.shape != expected or image.dtype != np.uint8:
        raise SourceError(f"decoded color shape/dtype mismatch: {image.shape}/{image.dtype}")
    return image


def decode_depth(header: SensHeader, payload: bytes) -> np.ndarray:
    compression = DEPTH_COMPRESSION[header.depth_compression_id]
    try:
        if compression == "raw_ushort":
            raw = payload
        elif compression == "zlib_ushort":
            raw = zlib.decompress(payload)
        else:
            raise SourceError(f"unsupported depth compression: {compression}")
        depth = np.frombuffer(raw, dtype="<u2").reshape(
            header.depth_height, header.depth_width
        ).copy()
    except (ValueError, zlib.error) as exc:
        raise SourceError("failed to decode .sens depth payload") from exc
    return depth.astype(np.uint16, copy=False)


def register_depth_to_color(header: SensHeader, depth_raw: np.ndarray) -> RegistrationResult:
    expected = (header.depth_height, header.depth_width)
    if depth_raw.shape != expected or depth_raw.dtype != np.uint16:
        raise SourceError(f"native depth shape/dtype mismatch: {depth_raw.shape}/{depth_raw.dtype}")
    valid = depth_raw > 0
    y_depth, x_depth = np.nonzero(valid)
    if len(x_depth) == 0:
        raise SourceError("depth frame has no valid samples")
    z_depth = depth_raw[y_depth, x_depth].astype(np.float64) / header.depth_shift
    points_depth = np.stack(
        (
            (x_depth - header.intrinsic_depth[0, 2])
            * z_depth
            / header.intrinsic_depth[0, 0],
            (y_depth - header.intrinsic_depth[1, 2])
            * z_depth
            / header.intrinsic_depth[1, 1],
            z_depth,
            np.ones_like(z_depth),
        )
    )
    depth_to_color = np.linalg.inv(header.extrinsic_color) @ header.extrinsic_depth
    points_color = depth_to_color @ points_depth
    z_color = points_color[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x_float = (
            header.intrinsic_color[0, 0] * points_color[0] / z_color
            + header.intrinsic_color[0, 2]
        )
        y_float = (
            header.intrinsic_color[1, 1] * points_color[1] / z_color
            + header.intrinsic_color[1, 2]
        )
    projectable = np.isfinite(x_float) & np.isfinite(y_float) & np.isfinite(z_color) & (z_color > 0)
    x_color = np.rint(x_float[projectable]).astype(np.int64)
    y_color = np.rint(y_float[projectable]).astype(np.int64)
    z_color = z_color[projectable]
    inside = (
        (x_color >= 0)
        & (x_color < header.color_width)
        & (y_color >= 0)
        & (y_color < header.color_height)
    )
    x_color = x_color[inside]
    y_color = y_color[inside]
    z_color = z_color[inside]
    if len(x_color) == 0:
        raise SourceError("depth-to-color registration produced no in-frame samples")

    raw_color = np.rint(z_color * header.depth_shift)
    if not np.isfinite(raw_color).all() or raw_color.min() <= 0 or raw_color.max() > np.iinfo(np.uint16).max:
        raise SourceError("registered depth cannot be represented in uint16 source units")
    flat = y_color * header.color_width + x_color
    order = np.argsort(z_color, kind="stable")
    _, first = np.unique(flat[order], return_index=True)
    selected = order[first]
    aligned = np.zeros((header.color_height, header.color_width), dtype=np.uint16)
    aligned[y_color[selected], x_color[selected]] = raw_color[selected].astype(np.uint16)
    return RegistrationResult(
        aligned_raw=aligned,
        native_valid_pixels=int(valid.sum()),
        projected_pixels=int(inside.sum()),
        registered_pixels=int(len(selected)),
    )


def load_split_manifests(split_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    documents: dict[str, dict[str, Any]] = {}
    scene_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = split_root / f"scannetv2_{split}.txt"
        if not path.is_file():
            raise SourceError(f"missing official ScanNet split: {path}")
        observed_hash = file_sha256(path)
        if observed_hash != SCANNET_SPLIT_SHA256[split]:
            raise SourceError(f"official ScanNet {split} split hash mismatch: {path}")
        scene_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(scene_ids) != len(set(scene_ids)) or any(SCENE_RE.fullmatch(value) is None for value in scene_ids):
            raise SourceError(f"invalid or duplicate scene id in official {split} split")
        overlap = set(scene_ids) & set(scene_to_split)
        if overlap:
            raise SourceError(f"official ScanNet split overlap: {sorted(overlap)[0]}")
        for scene_id in scene_ids:
            scene_to_split[scene_id] = split
        documents[split] = {
            "path": str(path.resolve()),
            "sha256": observed_hash,
            "scene_count": len(scene_ids),
            "source_revision": SCANNET_SPLIT_REVISION,
            "source_url": (
                "https://github.com/ScanNet/ScanNet/blob/"
                f"{SCANNET_SPLIT_REVISION}/Tasks/Benchmark/scannetv2_{split}.txt"
            ),
        }
    return documents, scene_to_split


def parse_scene_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SourceError(f"missing ScanNet scene metadata: {path}")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def validate_metadata(header: SensHeader, metadata: dict[str, str]) -> None:
    expected = {
        "colorWidth": header.color_width,
        "colorHeight": header.color_height,
        "depthWidth": header.depth_width,
        "depthHeight": header.depth_height,
        "numColorFrames": header.num_frames,
        "numDepthFrames": header.num_frames,
    }
    for key, value in expected.items():
        try:
            observed = int(metadata[key])
        except (KeyError, ValueError) as exc:
            raise SourceError(f"missing or invalid {key} in ScanNet scene metadata") from exc
        if observed != value:
            raise SourceError(f"scene metadata mismatch for {key}: {observed} != {value}")


def source_paths(scannet_root: Path, label_map: Path, scene_id: str) -> SourcePaths:
    if SCENE_RE.fullmatch(scene_id) is None:
        raise SourceError(f"invalid ScanNet scene id: {scene_id}")
    scene_dir = scannet_root / scene_id
    return SourcePaths(
        scene_id=scene_id,
        scene_dir=scene_dir,
        sens=scene_dir / f"{scene_id}.sens",
        metadata=scene_dir / f"{scene_id}.txt",
        aggregation=scene_dir / f"{scene_id}.aggregation.json",
        instance_zip=scene_dir / f"{scene_id}_2d-instance-filt.zip",
        semantic_zip=scene_dir / f"{scene_id}_2d-label-filt.zip",
        label_map=label_map,
        download_script=scannet_root.parent / "download-scannet.py",
    )


def require_source_files(paths: SourcePaths) -> None:
    for path in (
        paths.scene_dir,
        paths.sens,
        paths.metadata,
        paths.aggregation,
        paths.instance_zip,
        paths.semantic_zip,
        paths.label_map,
        paths.download_script,
    ):
        if not path.exists():
            raise SourceError(f"missing required ScanNet source artifact: {path}")


def read_annotation_member(
    archive_path: Path,
    member: str,
    expected_size: tuple[int, int],
) -> tuple[bytes, dict[str, Any]]:
    try:
        with ZipFile(archive_path) as archive:
            payload = archive.read(member)
        with Image.open(BytesIO(payload)) as image:
            size = image.size
            mode = image.mode
            image.verify()
    except (BadZipFile, KeyError, OSError, UnidentifiedImageError) as exc:
        raise SourceError(f"invalid ScanNet annotation member {archive_path}:{member}") from exc
    if size != expected_size:
        raise SourceError(f"annotation size mismatch for {member}: {size} != {expected_size}")
    return payload, {
        "archive_path": str(archive_path.resolve()),
        "member": member,
        "member_sha256": bytes_sha256(payload),
        "member_size_bytes": len(payload),
        "image_width": size[0],
        "image_height": size[1],
        "image_mode": mode,
    }


def local_license_evidence(paths: SourcePaths, supplied: Path | None) -> dict[str, Any]:
    root = paths.scene_dir.parent.parent
    candidate_names = (
        "ScanNet_TOS.pdf",
        "LICENSE",
        "LICENSE.txt",
        "TERMS",
        "TERMS.txt",
    )
    local_terms = [root / name for name in candidate_names if (root / name).is_file()]
    downloader_text = paths.download_script.read_text(encoding="utf-8", errors="replace")
    supplied_doc: dict[str, Any] | None = None
    if supplied is not None:
        if not supplied.is_file():
            raise SourceError(f"supplied license evidence does not exist: {supplied}")
        supplied_doc = {
            "path": str(supplied.resolve()),
            "sha256": file_sha256(supplied),
            "size_bytes": supplied.stat().st_size,
            "status": "present_unverified",
        }
    return {
        "dataset_terms": "ScanNet Terms of Use",
        "terms_url_declared_by_downloader": "http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf",
        "downloader_requires_terms_confirmation": (
            "confirm that you have agreed to the ScanNet terms of use" in downloader_text
        ),
        "local_terms_files": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in local_terms
        ],
        "supplied_approval_evidence": supplied_doc,
        "approval_status": (
            "LOCAL_EVIDENCE_PRESENT_REQUIRES_HUMAN_VERIFICATION"
            if supplied_doc is not None
            else "BLOCKED_NO_LOCAL_APPROVAL_EVIDENCE"
        ),
    }


def header_document(header: SensHeader) -> dict[str, Any]:
    return {
        "version": header.version,
        "sensor_name": header.sensor_name,
        "color_compression": {
            "id": header.color_compression_id,
            "name": COLOR_COMPRESSION[header.color_compression_id],
        },
        "depth_compression": {
            "id": header.depth_compression_id,
            "name": DEPTH_COMPRESSION[header.depth_compression_id],
        },
        "color_width": header.color_width,
        "color_height": header.color_height,
        "depth_width": header.depth_width,
        "depth_height": header.depth_height,
        "depth_shift_raw_units_per_meter": header.depth_shift,
        "num_frames": header.num_frames,
        "frame_data_offset": header.frame_data_offset,
    }


def inspect_source(
    *,
    scannet_root: Path,
    split_root: Path,
    label_map: Path,
    scene_id: str,
    frame_index: int,
    license_evidence: Path | None,
) -> dict[str, Any]:
    split_documents, scene_to_split = load_split_manifests(split_root)
    observed_split = scene_to_split.get(scene_id)
    if observed_split != "train":
        raise SourceError(
            f"scene {scene_id} is not official train (observed split: {observed_split or 'unknown'})"
        )
    paths = source_paths(scannet_root, label_map, scene_id)
    require_source_files(paths)
    header, frame = read_sens_frame(paths.sens, frame_index)
    metadata = parse_scene_metadata(paths.metadata)
    validate_metadata(header, metadata)
    color = decode_color(header, frame.color_bytes)
    depth = decode_depth(header, frame.depth_bytes)
    registration = register_depth_to_color(header, depth)
    instance_member = f"instance-filt/{frame_index}.png"
    semantic_member = f"label-filt/{frame_index}.png"
    instance_bytes, instance_doc = read_annotation_member(
        paths.instance_zip, instance_member, (header.color_width, header.color_height)
    )
    semantic_bytes, semantic_doc = read_annotation_member(
        paths.semantic_zip, semantic_member, (header.color_width, header.color_height)
    )
    return {
        "paths": paths,
        "split_documents": split_documents,
        "header": header,
        "frame": frame,
        "color": color,
        "depth": depth,
        "registration": registration,
        "instance_bytes": instance_bytes,
        "semantic_bytes": semantic_bytes,
        "instance_doc": instance_doc,
        "semantic_doc": semantic_doc,
        "license": local_license_evidence(paths, license_evidence),
    }


def check_document(inspected: dict[str, Any]) -> dict[str, Any]:
    paths: SourcePaths = inspected["paths"]
    header: SensHeader = inspected["header"]
    frame: FramePayload = inspected["frame"]
    registration: RegistrationResult = inspected["registration"]
    license_doc: dict[str, Any] = inspected["license"]
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "stage": "check_only",
        "verdict": (
            "CHECK_PASS_SCALEOUT_LICENSE_REVIEW_REQUIRED"
            if license_doc["approval_status"] != "BLOCKED_NO_LOCAL_APPROVAL_EVIDENCE"
            else "CHECK_PASS_SCALEOUT_BLOCKED_BY_LICENSE_EVIDENCE"
        ),
        "cpu_only": True,
        "network_calls": 0,
        "scene_id": paths.scene_id,
        "official_split": "train",
        "frame_index": frame.frame_index,
        "header": header_document(header),
        "decoded": {
            "color_shape": list(inspected["color"].shape),
            "color_dtype": str(inspected["color"].dtype),
            "depth_shape": list(inspected["depth"].shape),
            "depth_dtype": str(inspected["depth"].dtype),
            "native_valid_depth_pixels": registration.native_valid_pixels,
            "registered_depth_pixels": registration.registered_pixels,
        },
        "annotation_members": {
            "instance": inspected["instance_doc"],
            "semantic": inspected["semantic_doc"],
        },
        "license": license_doc,
        "write_count": 0,
    }


def _write_matrix(path: Path, value: np.ndarray) -> None:
    np.savetxt(path, value, fmt="%.9g")


def _artifact_document(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _source_document(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def build_canary(inspected: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise SourceError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    paths: SourcePaths = inspected["paths"]
    header: SensHeader = inspected["header"]
    frame: FramePayload = inspected["frame"]
    registration: RegistrationResult = inspected["registration"]

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        Image.fromarray(inspected["color"]).save(temp_dir / "color.png")
        Image.fromarray(inspected["depth"]).save(temp_dir / "depth_native.png")
        Image.fromarray(registration.aligned_raw).save(temp_dir / "depth_aligned_to_color.png")
        (temp_dir / "instance.png").write_bytes(inspected["instance_bytes"])
        (temp_dir / "semantic.png").write_bytes(inspected["semantic_bytes"])
        _write_matrix(temp_dir / "pose_camera_to_world.txt", frame.camera_to_world)
        _write_matrix(temp_dir / "intrinsic_color.txt", header.intrinsic_color)
        _write_matrix(temp_dir / "extrinsic_color.txt", header.extrinsic_color)
        _write_matrix(temp_dir / "intrinsic_depth.txt", header.intrinsic_depth)
        _write_matrix(temp_dir / "extrinsic_depth.txt", header.extrinsic_depth)

        artifact_names = (
            "color.png",
            "depth_native.png",
            "depth_aligned_to_color.png",
            "instance.png",
            "semantic.png",
            "pose_camera_to_world.txt",
            "intrinsic_color.txt",
            "extrinsic_color.txt",
            "intrinsic_depth.txt",
            "extrinsic_depth.txt",
        )
        artifacts = {
            name: _artifact_document(temp_dir / name, temp_dir) for name in artifact_names
        }
        source_files = {
            "sens": _source_document(paths.sens),
            "scene_metadata": _source_document(paths.metadata),
            "aggregation": _source_document(paths.aggregation),
            "instance_zip": _source_document(paths.instance_zip),
            "semantic_zip": _source_document(paths.semantic_zip),
            "label_map": _source_document(paths.label_map),
            "download_script": _source_document(paths.download_script),
        }
        manifest = {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "stage": "one_scene_cpu_canary",
            "verdict": (
                "CANARY_PASS_SCALEOUT_LICENSE_REVIEW_REQUIRED"
                if inspected["license"]["approval_status"]
                != "BLOCKED_NO_LOCAL_APPROVAL_EVIDENCE"
                else "CANARY_PASS_SCALEOUT_BLOCKED_BY_LICENSE_EVIDENCE"
            ),
            "adapter": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
                "reader_implementation": "independent_minimal_scannet_sens_v4_reader",
                "local_official_SensorData_py_found": False,
            },
            "execution": {
                "cpu_only": True,
                "network_calls": 0,
                "gpu_required": False,
                "scene_count": 1,
                "frame_count": 1,
            },
            "use_policy": {
                "official_split": "train",
                "official_train_only": True,
                "may_enter_held_out_v5_or_schema5": False,
                "training_materialization_eligible": False,
                "training_materialization_blockers": [
                    "adapter_canary_only_not_scaleout_approved",
                    "scannet_tos_approval_not_locally_verified",
                ],
                "redistribution_eligible": False,
            },
            "scene": {
                "scene_id": paths.scene_id,
                "frame_index": frame.frame_index,
                "source_directory": str(paths.scene_dir.resolve()),
            },
            "official_splits": inspected["split_documents"],
            "source_files": source_files,
            "source_frame_payloads": {
                "color_compressed_sha256": bytes_sha256(frame.color_bytes),
                "color_compressed_size_bytes": len(frame.color_bytes),
                "depth_compressed_sha256": bytes_sha256(frame.depth_bytes),
                "depth_compressed_size_bytes": len(frame.depth_bytes),
                "instance_member": inspected["instance_doc"],
                "semantic_member": inspected["semantic_doc"],
            },
            "sensor_header": header_document(header),
            "frame_geometry": {
                "pose_convention": "camera_to_world_as_stored_by_scannet_sens_v4",
                "timestamp_color": frame.timestamp_color,
                "timestamp_depth": frame.timestamp_depth,
                "depth_to_color_transform": "inv(extrinsic_color)@extrinsic_depth",
                "projection": "nearest_integer_pixel_then_nearest_depth_z_buffer",
                "aligned_depth_units": "raw_uint16_at_depth_shift_raw_units_per_meter",
                "native_valid_depth_pixels": registration.native_valid_pixels,
                "projected_in_frame_pixels_before_z_buffer": registration.projected_pixels,
                "registered_pixels_after_z_buffer": registration.registered_pixels,
                "registered_color_fraction": (
                    registration.registered_pixels
                    / float(header.color_width * header.color_height)
                ),
            },
            "license": inspected["license"],
            "outputs": artifacts,
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8")
        (temp_dir / "source_manifest.json").write_bytes(manifest_bytes)
        manifest_sha256 = bytes_sha256(manifest_bytes)
        (temp_dir / "source_manifest.sha256").write_text(
            f"{manifest_sha256}  source_manifest.json\n", encoding="ascii"
        )
        temp_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "stage": "one_scene_cpu_canary",
        "verdict": manifest["verdict"],
        "scene_id": paths.scene_id,
        "frame_index": frame.frame_index,
        "output_dir": str(output_dir.resolve()),
        "source_manifest_sha256": manifest_sha256,
        "training_materialization_eligible": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "canary"), default="check")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--scannet-root", type=Path, default=DEFAULT_SCANNET_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--license-evidence", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        inspected = inspect_source(
            scannet_root=args.scannet_root,
            split_root=args.split_root,
            label_map=args.label_map,
            scene_id=args.scene_id,
            frame_index=args.frame_index,
            license_evidence=args.license_evidence,
        )
        result = (
            check_document(inspected)
            if args.stage == "check"
            else build_canary(inspected, args.output_dir)
        )
    except (OSError, SourceError, np.linalg.LinAlgError) as exc:
        print(json.dumps({"schema": SCHEMA, "verdict": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
