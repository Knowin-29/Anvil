#!/usr/bin/env python3
"""Build pairwise depth labels from ScanNet RGB-D sensor truth.

The builder never invokes a vision-language model or monocular depth model.
It joins OpenEQA's extracted ScanNet RGB/depth/pose frames with the matching
official ScanNet 2D instance and semantic projections, registers depth samples
to the color camera, and emits at most one well-separated pair per image.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trains.flywheel.contracts import context_for, grounding
from trains.flywheel.validators import validate

PROTOCOL = "gsa-depth-v5.0-scannet-sensor-pairwise"
LABEL_VERSION = "gsa-label-v5.0-depth-scannet-sensor-pairwise"
AUDIT_ROUND = "depth-v5.0-scannet-sensor-calibration-v1"
SOURCE_ARTIFACT_SCHEMA = "gsa-scannet-sensor-depth-source-v1"
SCANNET_SPLIT_REVISION = "3830fce7f8b2e48ef047ef7fd76ea5f62903f51c"
SCANNET_SPLIT_SHA256 = {
    "train": "96acca299b7855f02824c496b19077904d80996e7ced1bb9f0dac98f7dd4d0c8",
    "val": "d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083",
    "test": "0214c6a3b1ee516ad653393b0321e7c0394c7662a4b3702eac1ddd7fbc00f7e0",
}
SCANNET_SOURCE_EXCLUSIONS_SHA256 = (
    "34488cc9d3260924ae98c95bfeb6b1c57ab6e116b381a29cad8ae337d69c6cd5"
)
SCANNET_CANARY_SCENES_SHA256 = (
    "3f7d4eb134d5a4920b567eb16b98106367e875f215be3ab50570895377e943ce"
)
SCANNET_LABEL_MAP_SHA256 = (
    "68674782476d967c35e8d0776933b70e10d21f1ea8019eba20221d38275139c0"
)

DEFAULT_OPENEQA_ROOT = Path(
    "/path/to/datasets/datasets/transferred_datasets/"
    "EmbodiedAI_benchmark/openeqa/scannet-v0"
)
DEFAULT_SCANNET_ROOT = Path(
    "/path/to/datasets/downloads/"
    "ScanNet-official/scans"
)
DEFAULT_LABEL_MAP = Path(
    "/path/to/datasets/downloads/"
    "ScanNet-official/tasks/scannetv2-labels.combined.tsv"
)
DEFAULT_SPLIT_ROOT = ROOT / "configs/scannet_splits"
DEFAULT_SOURCE_EXCLUSIONS = ROOT / "configs/scannet_source_exclusions_v1.json"
DEFAULT_CANARY_SCENES = ROOT / "configs/scannet_sensor_canary_scenes_v1.json"
DEFAULT_CHECK_OUT = ROOT / "data/depth_v50_scannet_sensor/check_v4"
DEFAULT_CANARY_OUT = ROOT / "data/depth_v50_scannet_sensor/canary_v4"
DEFAULT_FULL_OUT = ROOT / "data/depth_v50_scannet_sensor/full_v3"

CANARY_TARGET = 10
FULL_TARGET = 120
MIN_FULL_SCENES = 60
MIN_DEPTH_GAP_M = 0.25
MAX_DEPTH_M = 12.0
MIN_MASK_PIXELS = 1_250
MIN_MASK_FRACTION = 0.001
MIN_BBOX_SIDE_PX = 12
MIN_VALID_SAMPLES = 32
MIN_VALID_FRACTION = 0.25
MAX_RECORDS_PER_SCENE = 2
MAX_FRAMES_PER_SCENE = 64
MIN_BBOX_AREA_FRACTION = 0.01
MAX_BBOX_AREA_FRACTION = 0.45
EXCLUDED_CATEGORIES = {"", "unknown", "unlabeled", "wall", "floor", "ceiling"}
GENERIC_CATEGORIES = {"object", "objects", "furniture"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCENE_FOLDER_RE = re.compile(r"^\d+-scannet-(scene\d{4}_\d{2})$")
RGB_RE = re.compile(r"^(\d{6})-rgb\.png$")


@dataclass(frozen=True)
class SceneSource:
    folder: Path
    scene_id: str
    official_folder: Path
    instance_zip: Path
    label_zip: Path
    scene_metadata: Path
    aggregation_json: Path
    official_split: str
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class InstanceStat:
    instance_id: int
    aggregation_label: str
    raw_label_id: int
    category: str
    nyu40class: str
    bbox_xyxy_px: tuple[int, int, int, int]
    bbox_2d: tuple[int, int, int, int]
    mask_pixels: int
    bbox_area_fraction: float
    bbox_edge_clearance_px: int
    mask_fill_ratio: float
    valid_samples: int
    valid_fraction: float
    depth_median_m: float
    depth_q1_m: float
    depth_q3_m: float
    depth_iqr_m: float
    depth_mad_m: float
    robust_se_m: float

    @property
    def display_label(self) -> str:
        return f"{self.aggregation_label} [instance {self.instance_id}]"


@dataclass(frozen=True)
class PairChoice:
    nearer: InstanceStat
    farther: InstanceStat
    depth_gap_m: float
    uncertainty_3se_m: float
    required_gap_m: float
    margin_m: float
    quartile_separated: bool
    quartile_gap_m: float


@dataclass(frozen=True)
class SplitManifest:
    name: str
    path: Path
    sha256: str
    source_url: str
    source_revision: str
    scene_ids: frozenset[str]


@dataclass(frozen=True)
class SourceExclusion:
    scene_id: str
    official_split: str
    artifact: str
    artifact_path: Path
    sha256: str
    reason: str
    evidence: str


@dataclass(frozen=True)
class LabelMaps:
    by_id: dict[int, dict[str, str]]
    by_raw_category: dict[str, dict[str, str]]


@dataclass(frozen=True)
class RegisteredDepth:
    x_color: np.ndarray
    y_color: np.ndarray
    z_color_m: np.ndarray
    raw_valid_pixels: int
    registered_pixels: int


class SourceError(RuntimeError):
    """A source artifact is missing, corrupt, or violates the frozen contract."""


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_split_manifests(
    split_root: Path,
) -> tuple[dict[str, SplitManifest], dict[str, str]]:
    manifests: dict[str, SplitManifest] = {}
    scene_to_split: dict[str, str] = {}
    for name in ("train", "val", "test"):
        path = split_root / f"scannetv2_{name}.txt"
        if not path.is_file():
            raise SourceError(f"missing official ScanNet {name} split: {path}")
        observed_sha256 = file_sha256(path)
        if observed_sha256 != SCANNET_SPLIT_SHA256[name]:
            raise SourceError(f"official ScanNet {name} split hash mismatch: {path}")
        scene_ids = frozenset(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not scene_ids or any(not re.fullmatch(r"scene\d{4}_\d{2}", value) for value in scene_ids):
            raise SourceError(f"invalid official ScanNet {name} split: {path}")
        overlap = set(scene_ids) & set(scene_to_split)
        if overlap:
            raise SourceError(
                f"official ScanNet split overlap: {','.join(sorted(overlap)[:5])}"
            )
        for scene_id in scene_ids:
            scene_to_split[scene_id] = name
        manifests[name] = SplitManifest(
            name=name,
            path=path.resolve(),
            sha256=observed_sha256,
            source_url=(
                "https://github.com/ScanNet/ScanNet/blob/"
                f"{SCANNET_SPLIT_REVISION}/Tasks/Benchmark/scannetv2_{name}.txt"
            ),
            source_revision=SCANNET_SPLIT_REVISION,
            scene_ids=scene_ids,
        )
    return manifests, scene_to_split


def split_document(manifest: SplitManifest) -> dict[str, Any]:
    return {
        "name": manifest.name,
        "path": str(manifest.path),
        "sha256": manifest.sha256,
        "source_url": manifest.source_url,
        "source_revision": manifest.source_revision,
        "scene_count": len(manifest.scene_ids),
    }


def load_source_exclusions(
    path: Path,
    scannet_root: Path,
    scene_to_split: dict[str, str],
    expected_manifest_sha256: str = SCANNET_SOURCE_EXCLUSIONS_SHA256,
) -> tuple[dict[str, SourceExclusion], dict[str, Any]]:
    if not path.is_file():
        raise SourceError(f"missing ScanNet source exclusion manifest: {path}")
    observed_manifest_sha256 = file_sha256(path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise SourceError(f"ScanNet source exclusion manifest hash mismatch: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceError(f"invalid ScanNet source exclusion manifest: {path}") from exc
    rows = document.get("exclusions")
    if (
        document.get("schema") != "gsa-scannet-source-exclusions-v1"
        or document.get("dataset") != "ScanNet v2"
        or not isinstance(rows, list)
    ):
        raise SourceError(f"invalid ScanNet source exclusion contract: {path}")
    exclusions: dict[str, SourceExclusion] = {}
    serialized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SourceError(f"invalid ScanNet source exclusion row: {path}")
        scene_id = row.get("scene_id")
        official_split = row.get("official_split")
        artifact = row.get("artifact")
        expected_hash = row.get("sha256")
        reason = row.get("reason")
        evidence = row.get("evidence")
        if (
            not isinstance(scene_id, str)
            or re.fullmatch(r"scene\d{4}_\d{2}", scene_id) is None
            or scene_id in exclusions
            or scene_to_split.get(scene_id) != official_split
            or not isinstance(artifact, str)
            or Path(artifact).name != artifact
            or not artifact.startswith(f"{scene_id}_")
            or not isinstance(expected_hash, str)
            or SHA256_RE.fullmatch(expected_hash) is None
            or reason != "corrupt_zip_archive"
            or evidence != "unzip_test_failed"
        ):
            raise SourceError(f"invalid ScanNet source exclusion row: {path}")
        artifact_path = scannet_root / scene_id / artifact
        if not artifact_path.is_file():
            raise SourceError(f"excluded ScanNet artifact is missing: {artifact_path}")
        observed_hash = file_sha256(artifact_path)
        if observed_hash != expected_hash:
            raise SourceError(
                f"excluded ScanNet artifact hash mismatch: {artifact_path}"
            )
        corruption_evidence: str | None = None
        try:
            with ZipFile(artifact_path) as archive:
                bad_member = archive.testzip()
            if bad_member is not None:
                corruption_evidence = f"crc_or_payload_failure:{bad_member}"
        except (OSError, BadZipFile, zlib.error) as exc:
            corruption_evidence = f"archive_read_failure:{type(exc).__name__}"
        if corruption_evidence is None:
            raise SourceError(
                f"excluded ScanNet artifact no longer reproduces corruption: {artifact_path}"
            )
        exclusion = SourceExclusion(
            scene_id=scene_id,
            official_split=str(official_split),
            artifact=artifact,
            artifact_path=artifact_path.resolve(),
            sha256=observed_hash,
            reason=reason,
            evidence=evidence,
        )
        exclusions[scene_id] = exclusion
        serialized.append({
            "scene_id": exclusion.scene_id,
            "official_split": exclusion.official_split,
            "artifact": exclusion.artifact,
            "artifact_path": str(exclusion.artifact_path),
            "sha256": exclusion.sha256,
            "reason": exclusion.reason,
            "evidence": exclusion.evidence,
            "verified_corruption": corruption_evidence,
        })
    binding = {
        "schema": document["schema"],
        "dataset": document["dataset"],
        "manifest_path": str(path.resolve()),
        "manifest_sha256": observed_manifest_sha256,
        "count": len(serialized),
        "exclusions": sorted(serialized, key=lambda item: item["scene_id"]),
    }
    return exclusions, binding


def visual_quality_contract() -> dict[str, Any]:
    return {
        "max_frames_per_scene": MAX_FRAMES_PER_SCENE,
        "strictly_interior_bbox": True,
        "min_bbox_area_fraction": MIN_BBOX_AREA_FRACTION,
        "max_bbox_area_fraction": MAX_BBOX_AREA_FRACTION,
        "excluded_generic_categories": sorted(GENERIC_CATEGORIES),
        "quartile_separation_required": True,
    }


def load_canary_scene_manifest(
    path: Path,
    scenes: list[SceneSource],
    scene_to_split: dict[str, str],
    source_exclusions: dict[str, SourceExclusion],
) -> dict[str, Any]:
    if not path.is_file():
        raise SourceError(f"missing frozen canary scene manifest: {path}")
    observed_sha256 = file_sha256(path)
    if observed_sha256 != SCANNET_CANARY_SCENES_SHA256:
        raise SourceError(f"frozen canary scene manifest hash mismatch: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceError(f"invalid frozen canary scene manifest: {path}") from exc
    scene_ids = document.get("scene_ids")
    if (
        document.get("schema") != "gsa-scannet-sensor-canary-scenes-v1"
        or document.get("protocol") != PROTOCOL
        or document.get("official_split") != "val"
        or document.get("split_revision") != SCANNET_SPLIT_REVISION
        or document.get("quality_contract") != visual_quality_contract()
        or not isinstance(scene_ids, list)
        or scene_ids != sorted(scene_ids)
        or len(scene_ids) != CANARY_TARGET
        or len(set(scene_ids)) != CANARY_TARGET
    ):
        raise SourceError("frozen canary scene manifest contract mismatch")
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    for scene_id in scene_ids:
        scene = scene_by_id.get(scene_id)
        if (
            scene is None
            or scene_to_split.get(scene_id) != "val"
            or scene.official_split != "val"
            or not source_ready(scene)
            or scene_id in source_exclusions
        ):
            raise SourceError(f"frozen canary scene is not source-ready val: {scene_id}")
    return {
        "schema": document["schema"],
        "protocol": document["protocol"],
        "official_split": document["official_split"],
        "split_revision": document["split_revision"],
        "selection_origin": document.get("selection_origin"),
        "quality_contract": document["quality_contract"],
        "manifest_path": str(path.resolve()),
        "manifest_sha256": observed_sha256,
        "scene_ids": scene_ids,
    }


def stable_order(values: Iterable[Any], namespace: str) -> list[Any]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{PROTOCOL}|{namespace}|{value}".encode("utf-8")
        ).hexdigest(),
    )


def scene_round_robin(
    scene_batches: list[list[dict[str, Any]]], target: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    maximum = max((len(batch) for batch in scene_batches), default=0)
    for record_index in range(maximum):
        for batch in scene_batches:
            if record_index < len(batch):
                selected.append(batch[record_index])
                if len(selected) >= target:
                    return selected
    return selected


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SourceError(f"missing calibration matrix: {path}")
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise SourceError(f"invalid calibration matrix: {path}")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-4):
        raise SourceError(f"invalid homogeneous calibration matrix: {path}")
    return matrix


def pose_valid(path: Path) -> bool:
    try:
        read_matrix(path)
    except SourceError:
        return False
    return True


def load_label_map(
    path: Path, expected_sha256: str | None = SCANNET_LABEL_MAP_SHA256
) -> LabelMaps:
    if not path.is_file():
        raise SourceError(f"missing ScanNet label map: {path}")
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise SourceError(f"ScanNet label map hash mismatch: {path}")
    by_id: dict[int, dict[str, str]] = {}
    raw_categories: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                raw_id = int(row["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SourceError(f"invalid ScanNet label map row in {path}") from exc
            document = {
                "raw_category": str(row.get("raw_category") or ""),
                "category": str(row.get("category") or row.get("raw_category") or ""),
                "nyu40class": str(row.get("nyu40class") or ""),
            }
            raw_category = document["raw_category"].strip().casefold()
            if raw_category:
                raw_categories[raw_category] = document
            by_id[raw_id] = document
    if not by_id or not raw_categories:
        raise SourceError(f"empty ScanNet label map: {path}")
    return LabelMaps(by_id=by_id, by_raw_category=raw_categories)


def load_aggregation(path: Path, scene_id: str, *, allow_bare_scene_id: bool = False) -> tuple[dict[int, str], str]:
    if not path.is_file():
        raise SourceError(f"missing ScanNet aggregation: {path}")
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceError(f"invalid ScanNet aggregation: {path}") from exc
    expected_scene_id = f"scannet.{scene_id}"
    allowed_ids = {expected_scene_id, scene_id} if allow_bare_scene_id else {expected_scene_id}
    if document.get("sceneId") not in allowed_ids:
        raise SourceError(f"aggregation sceneId mismatch: {path}")
    mapping: dict[int, str] = {}
    for group in document.get("segGroups", []):
        try:
            instance_id = int(group["objectId"]) + 1
            label = str(group["label"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceError(f"invalid aggregation group: {path}") from exc
        if instance_id <= 0 or not label or instance_id in mapping:
            raise SourceError(f"invalid aggregation objectId mapping: {path}")
        mapping[instance_id] = label
    if not mapping:
        raise SourceError(f"empty ScanNet aggregation: {path}")
    return mapping, bytes_sha256(payload)


def discover_scenes(
    openeqa_root: Path,
    scannet_root: Path,
    scene_to_split: dict[str, str],
) -> list[SceneSource]:
    if not openeqa_root.is_dir():
        raise SourceError(f"missing OpenEQA ScanNet frame root: {openeqa_root}")
    if not scannet_root.is_dir():
        raise SourceError(f"missing official ScanNet root: {scannet_root}")
    scenes: list[SceneSource] = []
    for folder in sorted(path for path in openeqa_root.iterdir() if path.is_dir()):
        match = SCENE_FOLDER_RE.fullmatch(folder.name)
        if match is None:
            continue
        scene_id = match.group(1)
        official_split = scene_to_split.get(scene_id)
        if official_split is None:
            raise SourceError(f"official ScanNet split is unknown for {scene_id}")
        official = scannet_root / scene_id
        scenes.append(
            SceneSource(
                folder=folder,
                scene_id=scene_id,
                official_folder=official,
                instance_zip=official / f"{scene_id}_2d-instance-filt.zip",
                label_zip=official / f"{scene_id}_2d-label-filt.zip",
                scene_metadata=official / f"{scene_id}.txt",
                aggregation_json=official / f"{scene_id}.aggregation.json",
                official_split=official_split,
                frame_indices=(),
            )
        )
    return sorted(
        scenes,
        key=lambda scene: hashlib.sha256(
            f"{PROTOCOL}|scene|{scene.scene_id}".encode("utf-8")
        ).hexdigest(),
    )


def source_ready(scene: SceneSource) -> bool:
    calibration_names = (
        "intrinsic_color.txt",
        "intrinsic_depth.txt",
        "extrinsic_color.txt",
        "extrinsic_depth.txt",
    )
    return (
        scene.instance_zip.is_file()
        and scene.label_zip.is_file()
        and scene.scene_metadata.is_file()
        and scene.aggregation_json.is_file()
        and all((scene.folder / name).is_file() for name in calibration_names)
    )


def discover_frame_indices(scene: SceneSource) -> tuple[int, ...]:
    if scene.frame_indices:
        return scene.frame_indices
    rgb_indices = {
        int(match.group(1))
        for path in scene.folder.glob("??????-rgb.png")
        for match in [RGB_RE.fullmatch(path.name)]
        if match is not None
    }
    depth_indices = {
        int(path.name[:6])
        for path in scene.folder.glob("??????-depth.png")
        if path.name[:6].isdigit()
    }
    pose_indices = {
        int(path.stem)
        for path in scene.folder.glob("??????.txt")
        if path.stem.isdigit()
    }
    indices = tuple(sorted(rgb_indices & depth_indices & pose_indices))
    if not indices:
        raise SourceError(f"no RGB/depth/pose frame triplets for {scene.scene_id}")
    return indices


def load_png(path: Path) -> tuple[np.ndarray, bytes]:
    try:
        payload = path.read_bytes()
        with Image.open(BytesIO(payload)) as image:
            array = np.asarray(image).copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise SourceError(f"invalid PNG: {path}") from exc
    return array, payload


def load_zip_png(archive: ZipFile, member: str) -> tuple[np.ndarray, bytes]:
    try:
        payload = archive.read(member)
        with Image.open(BytesIO(payload)) as image:
            array = np.asarray(image).copy()
    except (KeyError, OSError, UnidentifiedImageError) as exc:
        raise SourceError(f"invalid or missing ZIP member: {member}") from exc
    return array, payload


def register_depth_to_color(
    depth_mm: np.ndarray,
    intrinsic_depth: np.ndarray,
    extrinsic_depth: np.ndarray,
    intrinsic_color: np.ndarray,
    extrinsic_color: np.ndarray,
    color_shape: tuple[int, int],
) -> RegisteredDepth:
    if depth_mm.ndim != 2 or depth_mm.dtype != np.uint16:
        raise SourceError("depth must be a 16-bit single-channel PNG")
    height_color, width_color = color_shape
    valid = (depth_mm > 0) & (depth_mm <= round(MAX_DEPTH_M * 1000.0))
    y_depth, x_depth = np.nonzero(valid)
    if len(x_depth) == 0:
        raise SourceError("depth frame has no valid metric samples")
    z_depth = depth_mm[y_depth, x_depth].astype(np.float64) * 0.001
    points_depth = np.stack(
        (
            (x_depth - intrinsic_depth[0, 2]) * z_depth / intrinsic_depth[0, 0],
            (y_depth - intrinsic_depth[1, 2]) * z_depth / intrinsic_depth[1, 1],
            z_depth,
            np.ones_like(z_depth),
        )
    )
    depth_to_color = np.linalg.inv(extrinsic_color) @ extrinsic_depth
    points_color = depth_to_color @ points_depth
    z_color = points_color[2]
    x_color = np.rint(
        intrinsic_color[0, 0] * points_color[0] / z_color
        + intrinsic_color[0, 2]
    ).astype(np.int64)
    y_color = np.rint(
        intrinsic_color[1, 1] * points_color[1] / z_color
        + intrinsic_color[1, 2]
    ).astype(np.int64)
    inside = (
        np.isfinite(z_color)
        & (z_color > 0.0)
        & (z_color <= MAX_DEPTH_M)
        & (x_color >= 0)
        & (x_color < width_color)
        & (y_color >= 0)
        & (y_color < height_color)
    )
    x_color = x_color[inside]
    y_color = y_color[inside]
    z_color = z_color[inside]
    if len(x_color) == 0:
        raise SourceError("depth registration produced no in-frame samples")

    flat = y_color * width_color + x_color
    order = np.argsort(z_color, kind="stable")
    _, first = np.unique(flat[order], return_index=True)
    selected = order[first]
    return RegisteredDepth(
        x_color=x_color[selected],
        y_color=y_color[selected],
        z_color_m=z_color[selected],
        raw_valid_pixels=int(valid.sum()),
        registered_pixels=int(len(selected)),
    )


def normalized_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    values = (
        round(1000 * x1 / width),
        round(1000 * y1 / height),
        round(1000 * x2 / width),
        round(1000 * y2 / height),
    )
    return tuple(max(0, min(1000, int(value))) for value in values)


def instance_statistics(
    instance_mask: np.ndarray,
    semantic_mask: np.ndarray,
    registered: RegisteredDepth,
    depth_shape: tuple[int, int],
    label_maps: LabelMaps,
    aggregation_labels: dict[int, str],
) -> tuple[list[InstanceStat], Counter[str]]:
    if instance_mask.ndim != 2 or semantic_mask.ndim != 2:
        raise SourceError("instance and semantic projections must be single-channel")
    if instance_mask.shape != semantic_mask.shape:
        raise SourceError("instance and semantic projection shapes differ")
    height, width = instance_mask.shape
    projected_instances = instance_mask[registered.y_color, registered.x_color]
    density = (depth_shape[0] * depth_shape[1]) / float(height * width)
    minimum_area = max(MIN_MASK_PIXELS, math.ceil(MIN_MASK_FRACTION * height * width))
    accepted: list[InstanceStat] = []
    rejected: Counter[str] = Counter()

    for value in np.unique(instance_mask):
        instance_id = int(value)
        if instance_id == 0:
            continue
        aggregation_label = aggregation_labels.get(instance_id, "").strip()
        aggregation_category = label_maps.by_raw_category.get(
            aggregation_label.casefold()
        )
        if not aggregation_label:
            rejected["missing_aggregation_object"] += 1
            continue
        if aggregation_category is None:
            rejected["unmapped_aggregation_label"] += 1
            continue
        mask = instance_mask == instance_id
        area = int(mask.sum())
        if area < minimum_area:
            rejected["small_mask"] += 1
            continue
        y_pixels, x_pixels = np.nonzero(mask)
        bbox = (
            int(x_pixels.min()),
            int(y_pixels.min()),
            int(x_pixels.max()) + 1,
            int(y_pixels.max()) + 1,
        )
        if bbox[2] - bbox[0] < MIN_BBOX_SIDE_PX or bbox[3] - bbox[1] < MIN_BBOX_SIDE_PX:
            rejected["small_bbox"] += 1
            continue
        if bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= width or bbox[3] >= height:
            rejected["border_touching_bbox"] += 1
            continue
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        bbox_area_fraction = bbox_area / float(width * height)
        if not (
            MIN_BBOX_AREA_FRACTION
            <= bbox_area_fraction
            <= MAX_BBOX_AREA_FRACTION
        ):
            rejected["bbox_area_out_of_range"] += 1
            continue
        bbox_edge_clearance_px = min(
            bbox[0], bbox[1], width - bbox[2], height - bbox[3]
        )
        mask_fill_ratio = area / float(bbox_area)

        semantic_values = semantic_mask[mask].astype(np.int64)
        semantic_values = semantic_values[semantic_values > 0]
        if len(semantic_values) == 0:
            rejected["unknown_semantic"] += 1
            continue
        raw_label_id = int(np.bincount(semantic_values).argmax())
        semantic_category = label_maps.by_id.get(raw_label_id)
        category = str(aggregation_category.get("category") or "").strip()
        if semantic_category is None:
            rejected["unmapped_semantic_label"] += 1
            continue
        if (
            str(semantic_category.get("nyu40class") or "").casefold()
            != str(aggregation_category.get("nyu40class") or "").casefold()
        ):
            rejected["aggregation_semantic_mismatch"] += 1
            continue
        if (
            category.casefold() in EXCLUDED_CATEGORIES
            or aggregation_label.casefold() in EXCLUDED_CATEGORIES
        ):
            rejected["excluded_semantic"] += 1
            continue
        if (
            category.casefold() in GENERIC_CATEGORIES
            or aggregation_label.casefold() in GENERIC_CATEGORIES
        ):
            rejected["generic_category"] += 1
            continue

        values = registered.z_color_m[projected_instances == instance_id]
        values = values[np.isfinite(values) & (values > 0.0) & (values <= MAX_DEPTH_M)]
        if len(values) < MIN_VALID_SAMPLES:
            rejected["too_few_depth_samples"] += 1
            continue
        expected_samples = max(1.0, area * density)
        valid_fraction = min(1.0, float(len(values)) / expected_samples)
        if valid_fraction < MIN_VALID_FRACTION:
            rejected["low_valid_fraction"] += 1
            continue

        q1, median, q3 = np.percentile(values, (25.0, 50.0, 75.0))
        mad = float(np.median(np.abs(values - median)))
        robust_se = 1.4826 * mad / math.sqrt(len(values))
        accepted.append(
            InstanceStat(
                instance_id=instance_id,
                aggregation_label=aggregation_label,
                raw_label_id=raw_label_id,
                category=category,
                nyu40class=str(aggregation_category.get("nyu40class") or ""),
                bbox_xyxy_px=bbox,
                bbox_2d=normalized_bbox(bbox, width, height),
                mask_pixels=area,
                bbox_area_fraction=float(bbox_area_fraction),
                bbox_edge_clearance_px=int(bbox_edge_clearance_px),
                mask_fill_ratio=float(mask_fill_ratio),
                valid_samples=int(len(values)),
                valid_fraction=float(valid_fraction),
                depth_median_m=float(median),
                depth_q1_m=float(q1),
                depth_q3_m=float(q3),
                depth_iqr_m=float(q3 - q1),
                depth_mad_m=mad,
                robust_se_m=float(robust_se),
            )
        )
    return accepted, rejected


def choose_pair(instances: list[InstanceStat]) -> PairChoice | None:
    choices: list[
        tuple[tuple[float, float, float, float, float, float, float, str], PairChoice]
    ] = []
    for left, right in itertools.combinations(instances, 2):
        if left.depth_median_m <= right.depth_median_m:
            nearer, farther = left, right
        else:
            nearer, farther = right, left
        gap = farther.depth_median_m - nearer.depth_median_m
        uncertainty = 3.0 * math.sqrt(
            nearer.robust_se_m**2 + farther.robust_se_m**2
        )
        required = max(MIN_DEPTH_GAP_M, uncertainty)
        if gap < required:
            continue
        margin = gap - required
        quartile_gap = farther.depth_q1_m - nearer.depth_q3_m
        quartile_separated = quartile_gap > 0.0
        if not quartile_separated:
            continue
        a = nearer.bbox_xyxy_px
        b = farther.bbox_xyxy_px
        intersection_width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        intersection_height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        intersection_area = intersection_width * intersection_height
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        intersection_over_min_area = (
            intersection_area / float(min(area_a, area_b))
            if min(area_a, area_b) > 0
            else 1.0
        )
        stable = hashlib.sha256(
            (
                f"{PROTOCOL}|{nearer.instance_id}|{farther.instance_id}|"
                f"{nearer.aggregation_label}|{farther.aggregation_label}"
            ).encode("utf-8")
        ).hexdigest()
        choice = PairChoice(
            nearer=nearer,
            farther=farther,
            depth_gap_m=float(gap),
            uncertainty_3se_m=float(uncertainty),
            required_gap_m=float(required),
            margin_m=float(margin),
            quartile_separated=quartile_separated,
            quartile_gap_m=float(quartile_gap),
        )
        score = (
            min(min(nearer.bbox_area_fraction, farther.bbox_area_fraction) / 0.02, 1.0),
            min(min(nearer.bbox_edge_clearance_px, farther.bbox_edge_clearance_px) / 8.0, 1.0),
            max(0.0, 1.0 - intersection_over_min_area),
            min(min(nearer.mask_fill_ratio, farther.mask_fill_ratio) / 0.25, 1.0),
            min(nearer.valid_fraction, farther.valid_fraction),
            min(margin / 1.0, 1.0),
            gap,
            stable,
        )
        choices.append((score, choice))
    if not choices:
        return None
    return max(choices, key=lambda item: item[0])[1]


def _measurement_document(instance: InstanceStat) -> dict[str, Any]:
    document = asdict(instance)
    document["bbox_xyxy_px"] = list(instance.bbox_xyxy_px)
    document["bbox_2d"] = list(instance.bbox_2d)
    document["label"] = instance.display_label
    return document


def _target_document(instance: InstanceStat, rank: int) -> dict[str, Any]:
    return {
        "label": instance.display_label,
        "rank": rank,
        "relative_depth": float(rank - 1),
    }


def visibility_features(
    pair: PairChoice, color_shape: tuple[int, int]
) -> dict[str, Any]:
    height, width = color_shape
    a = pair.nearer.bbox_xyxy_px
    b = pair.farther.bbox_xyxy_px
    intersection_width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection_area = intersection_width * intersection_height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    edge_distance = math.hypot(dx, dy) / math.hypot(width, height)
    if pair.depth_gap_m < 0.5:
        gap_bucket = "0.25_to_0.50m"
    elif pair.depth_gap_m < 1.0:
        gap_bucket = "0.50_to_1.00m"
    else:
        gap_bucket = "at_least_1.00m"
    return {
        "quartile_separated": pair.quartile_separated,
        "quartile_gap_m": pair.quartile_gap_m,
        "gap_bucket": gap_bucket,
        "bbox_intersection_over_min_area": (
            float(intersection_area / min(area_a, area_b))
            if min(area_a, area_b) > 0
            else 0.0
        ),
        "bbox_edge_distance_normalized": float(edge_distance),
        "min_bbox_area_fraction": float(
            min(pair.nearer.bbox_area_fraction, pair.farther.bbox_area_fraction)
        ),
        "max_bbox_area_fraction": float(
            max(pair.nearer.bbox_area_fraction, pair.farther.bbox_area_fraction)
        ),
        "min_bbox_edge_clearance_px": int(
            min(
                pair.nearer.bbox_edge_clearance_px,
                pair.farther.bbox_edge_clearance_px,
            )
        ),
        "min_mask_fill_ratio": float(
            min(pair.nearer.mask_fill_ratio, pair.farther.mask_fill_ratio)
        ),
        "min_valid_fraction": float(
            min(pair.nearer.valid_fraction, pair.farther.valid_fraction)
        ),
        "pair_selection_rule": "visual_auditability_then_metric_margin_v1",
    }


def record_auditability_score(
    record: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float, str]:
    visibility = record["visibility_features"]
    verified_pair = record["verified_pair"]
    stable = hashlib.sha256(
        (
            f"{PROTOCOL}|frame-pair|{record['scene_id']}|{record['frame_index']}|"
            f"{record['sample_id']}|{record['image_sha256']}"
        ).encode("utf-8")
    ).hexdigest()
    return (
        min(float(visibility["min_bbox_area_fraction"]) / 0.02, 1.0),
        min(float(visibility["min_bbox_edge_clearance_px"]) / 8.0, 1.0),
        max(0.0, 1.0 - float(visibility["bbox_intersection_over_min_area"])),
        min(float(visibility["min_mask_fill_ratio"]) / 0.25, 1.0),
        float(visibility["min_valid_fraction"]),
        min(float(verified_pair["margin_m"]) / 1.0, 1.0),
        float(verified_pair["depth_gap_m"]),
        stable,
    )


def validate_record(
    record: dict[str, Any],
    split_manifests: dict[str, SplitManifest] | None = None,
) -> list[str]:
    errors: list[str] = []
    if record.get("protocol_version") != PROTOCOL:
        errors.append("protocol_version")
    if record.get("label_version") != LABEL_VERSION:
        errors.append("label_version")
    if record.get("source_artifact_schema") != SOURCE_ARTIFACT_SCHEMA:
        errors.append("source_artifact_schema")
    if (
        record.get("_audit_only") is not True
        or record.get("audit_contract") != "scannet_sensor_pairwise"
        or record.get("task_interface") != "oracle_grounded_pairwise_depth"
    ):
        errors.append("held_out_audit_only_contract")
    if record.get("task_form") != "depth" or record.get("depth_scope") != "pairwise":
        errors.append("task_contract")
    payload = record.get("payload")
    if not isinstance(payload, list) or len(payload) != 2:
        errors.append("payload")
        return errors
    if [item.get("rank") for item in payload] != [1, 2]:
        errors.append("ranks")
    if [item.get("relative_depth") for item in payload] != [0.0, 1.0]:
        errors.append("relative_depth_values")
    if any(set(item) != {"label", "rank", "relative_depth"} for item in payload):
        errors.append("payload_exact_keys")
    valid_target, target_reason = validate(
        "depth", payload, record.get("gsa_context")
    )
    if not valid_target:
        errors.append(f"gsa_contract:{target_reason}")
    pair = record.get("verified_pair")
    if not isinstance(pair, dict) or pair.get("nearer") != payload[0].get("label"):
        errors.append("verified_pair")
    gate = record.get("semantic_gate", {}).get("depth_v50_scannet_sensor")
    if (
        not isinstance(gate, dict)
        or gate.get("passed") is not True
        or gate.get("visual_quality_contract") != visual_quality_contract()
        or gate.get("pair_selection_rule")
        != "visual_auditability_then_metric_margin_v1"
    ):
        errors.append("semantic_gate")
    provenance = record.get("sensor_provenance")
    rgb_width: int | None = None
    rgb_height: int | None = None
    if not isinstance(provenance, dict):
        errors.append("sensor_provenance")
    else:
        for section in ("rgb", "depth", "instance_projection", "semantic_projection"):
            item = provenance.get(section)
            if not isinstance(item, dict) or not SHA256_RE.fullmatch(
                str(item.get("sha256", ""))
            ):
                errors.append(f"provenance_{section}")
        rgb_document = provenance.get("rgb")
        if isinstance(rgb_document, dict):
            width_value = rgb_document.get("width")
            height_value = rgb_document.get("height")
            if (
                isinstance(width_value, int)
                and width_value > 0
                and isinstance(height_value, int)
                and height_value > 0
            ):
                rgb_width = width_value
                rgb_height = height_value
            else:
                errors.append("rgb_shape")
        split_membership = provenance.get("split_membership")
        split_name = record.get("official_split")
        expected_url = (
            "https://github.com/ScanNet/ScanNet/blob/"
            f"{SCANNET_SPLIT_REVISION}/Tasks/Benchmark/scannetv2_{split_name}.txt"
        )
        if (
            split_name != "val"
            or not isinstance(split_membership, dict)
            or split_membership.get("name") != split_name
            or split_membership.get("sha256") != SCANNET_SPLIT_SHA256.get(split_name)
            or split_membership.get("source_revision") != SCANNET_SPLIT_REVISION
            or split_membership.get("source_url") != expected_url
        ):
            errors.append("official_split")
        if split_manifests is not None:
            manifest = split_manifests.get(str(split_name))
            if (
                manifest is None
                or record.get("scene_id") not in manifest.scene_ids
                or split_membership != split_document(manifest)
            ):
                errors.append("official_split_membership")
    measurements = record.get("sensor_measurements")
    if not isinstance(measurements, list) or len(measurements) != 2:
        errors.append("sensor_measurements")
        measurements = []
    elif [item.get("label") for item in measurements] != [
        payload[0].get("label"),
        payload[1].get("label"),
    ]:
        errors.append("measurement_labels")
    for item in measurements:
        bbox = item.get("bbox_2d")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, int) and 0 <= value <= 1000 for value in bbox)
            or bbox[0] >= bbox[2]
            or bbox[1] >= bbox[3]
        ):
            errors.append("bbox")
        pixel_bbox = item.get("bbox_xyxy_px")
        if (
            rgb_width is None
            or rgb_height is None
            or not isinstance(pixel_bbox, list)
            or len(pixel_bbox) != 4
            or not all(isinstance(value, int) for value in pixel_bbox)
        ):
            errors.append("pixel_bbox")
            continue
        x1, y1, x2, y2 = pixel_bbox
        if not (0 < x1 < x2 < rgb_width and 0 < y1 < y2 < rgb_height):
            errors.append("strictly_interior_bbox")
            continue
        area_fraction = ((x2 - x1) * (y2 - y1)) / float(
            rgb_width * rgb_height
        )
        edge_clearance = min(x1, y1, rgb_width - x2, rgb_height - y2)
        mask_pixels = item.get("mask_pixels")
        fill_ratio = (
            mask_pixels / float((x2 - x1) * (y2 - y1))
            if isinstance(mask_pixels, int)
            else -1.0
        )
        if not MIN_BBOX_AREA_FRACTION <= area_fraction <= MAX_BBOX_AREA_FRACTION:
            errors.append("bbox_area_fraction_range")
        observed_area_fraction = item.get("bbox_area_fraction")
        if not isinstance(observed_area_fraction, (int, float)) or not math.isclose(
            float(observed_area_fraction), area_fraction, rel_tol=1e-9, abs_tol=1e-12
        ):
            errors.append("bbox_area_fraction")
        if item.get("bbox_edge_clearance_px") != edge_clearance:
            errors.append("bbox_edge_clearance_px")
        observed_fill_ratio = item.get("mask_fill_ratio")
        if not isinstance(observed_fill_ratio, (int, float)) or not math.isclose(
            float(observed_fill_ratio), fill_ratio, rel_tol=1e-9, abs_tol=1e-12
        ):
            errors.append("mask_fill_ratio")
        if (
            str(item.get("aggregation_label", "")).casefold() in GENERIC_CATEGORIES
            or str(item.get("category", "")).casefold() in GENERIC_CATEGORIES
        ):
            errors.append("generic_category")
    training = record.get("training_materialization")
    if not isinstance(training, dict) or training.get("eligible") is not False:
        errors.append("training_materialization")
    visibility = record.get("visibility_features")
    min_area = visibility.get("min_bbox_area_fraction") if isinstance(visibility, dict) else None
    max_area = visibility.get("max_bbox_area_fraction") if isinstance(visibility, dict) else None
    min_edge = visibility.get("min_bbox_edge_clearance_px") if isinstance(visibility, dict) else None
    min_fill = visibility.get("min_mask_fill_ratio") if isinstance(visibility, dict) else None
    if (
        not isinstance(visibility, dict)
        or visibility.get("gap_bucket")
        not in {"0.25_to_0.50m", "0.50_to_1.00m", "at_least_1.00m"}
        or visibility.get("quartile_separated") is not True
        or not isinstance(min_area, (int, float))
        or min_area < MIN_BBOX_AREA_FRACTION
        or not isinstance(max_area, (int, float))
        or max_area > MAX_BBOX_AREA_FRACTION
        or not isinstance(min_edge, int)
        or min_edge < 1
        or not isinstance(min_fill, (int, float))
        or min_fill <= 0.0
        or visibility.get("pair_selection_rule")
        != "visual_auditability_then_metric_margin_v1"
    ):
        errors.append("visibility_features")
    model_interface = record.get("model_interface")
    if (
        not isinstance(model_interface, dict)
        or model_interface.get("intended_task")
        != "oracle_grounded_pairwise_depth"
        or model_interface.get("requires_explicit_pair_groundings") is not True
        or model_interface.get("instance_ids_are_internal_identity_only") is not True
    ):
        errors.append("model_interface")
    return sorted(set(errors))


def analyze_frame(
    scene: SceneSource,
    frame_index: int,
    label_maps: LabelMaps,
    split_manifests: dict[str, SplitManifest],
    instance_archive: ZipFile,
    label_archive: ZipFile,
) -> tuple[dict[str, Any] | None, str, Counter[str]]:
    rgb_path = scene.folder / f"{frame_index:06d}-rgb.png"
    depth_path = scene.folder / f"{frame_index:06d}-depth.png"
    pose_path = scene.folder / f"{frame_index:06d}.txt"
    rgb, rgb_bytes = load_png(rgb_path)
    depth, depth_bytes = load_png(depth_path)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise SourceError(f"RGB must be 8-bit three-channel: {rgb_path}")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise SourceError(f"depth must be 16-bit single-channel: {depth_path}")

    instance_member = f"instance-filt/{frame_index}.png"
    semantic_member = f"label-filt/{frame_index}.png"
    instance_mask, instance_bytes = load_zip_png(instance_archive, instance_member)
    semantic_mask, semantic_bytes = load_zip_png(label_archive, semantic_member)
    if instance_mask.shape != rgb.shape[:2] or semantic_mask.shape != rgb.shape[:2]:
        raise SourceError(
            f"projection/RGB shape mismatch for {scene.scene_id}:{frame_index}"
        )

    calibration_paths = {
        "intrinsic_color": scene.folder / "intrinsic_color.txt",
        "intrinsic_depth": scene.folder / "intrinsic_depth.txt",
        "extrinsic_color": scene.folder / "extrinsic_color.txt",
        "extrinsic_depth": scene.folder / "extrinsic_depth.txt",
    }
    matrices = {name: read_matrix(path) for name, path in calibration_paths.items()}
    registered = register_depth_to_color(
        depth,
        matrices["intrinsic_depth"],
        matrices["extrinsic_depth"],
        matrices["intrinsic_color"],
        matrices["extrinsic_color"],
        rgb.shape[:2],
    )
    aggregation_labels, aggregation_sha256 = load_aggregation(
        scene.aggregation_json, scene.scene_id
    )
    instances, instance_rejections = instance_statistics(
        instance_mask,
        semantic_mask,
        registered,
        depth.shape,
        label_maps,
        aggregation_labels,
    )
    if len(instances) < 2:
        return None, "fewer_than_two_instances", instance_rejections
    pair = choose_pair(instances)
    if pair is None:
        return None, "no_separated_pair", instance_rejections

    near_target = _target_document(pair.nearer, 1)
    far_target = _target_document(pair.farther, 2)
    near_measurement = _measurement_document(pair.nearer)
    far_measurement = _measurement_document(pair.farther)
    sample_id = (
        f"scannet-sdepth-v50-{scene.scene_id}-{frame_index:06d}-"
        f"{pair.nearer.instance_id}-{pair.farther.instance_id}"
    )
    calibration = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in calibration_paths.items()
    }
    payload = [near_target, far_target]
    grounding_items = [
        grounding(pair.nearer.display_label, list(pair.nearer.bbox_2d)),
        grounding(pair.farther.display_label, list(pair.farther.bbox_2d)),
    ]
    gsa_context = context_for(
        "depth",
        grounding_items,
        {"relation_mode": "pairwise", "depth_order": payload},
    )
    valid_target, target_reason = validate("depth", payload, gsa_context)
    if not valid_target:
        raise SourceError(f"canonical depth contract rejected: {target_reason}")
    record = {
        "sample_id": sample_id,
        "source_artifact_schema": SOURCE_ARTIFACT_SCHEMA,
        "_audit_only": True,
        "audit_contract": "scannet_sensor_pairwise",
        "task_interface": "oracle_grounded_pairwise_depth",
        "task_form": "depth",
        "depth_scope": "pairwise",
        "protocol_version": PROTOCOL,
        "label_version": LABEL_VERSION,
        "audit_round": AUDIT_ROUND,
        "scene_type": "indoor_scan",
        "source_bucket": "scannet_sensor",
        "scene_id": scene.scene_id,
        "official_split": scene.official_split,
        "frame_index": frame_index,
        "image_path": str(rgb_path),
        "image_sha256": bytes_sha256(rgb_bytes),
        "payload": payload,
        "gsa_context": gsa_context,
        "sensor_measurements": [near_measurement, far_measurement],
        "visibility_features": visibility_features(pair, rgb.shape[:2]),
        "spatial": {
            "relation_mode": "pairwise",
            "depth_order": [near_target["label"], far_target["label"]],
            "adjacent_depth_evidence": {
                "nearer": near_target["label"],
                "farther": far_target["label"],
                "cue_type": "metric_sensor_depth",
                "evidence": (
                    f"Registered ScanNet sensor medians are "
                    f"{pair.nearer.depth_median_m:.3f} m and "
                    f"{pair.farther.depth_median_m:.3f} m."
                ),
            },
        },
        "verified_pair": {
            "nearer": near_target["label"],
            "farther": far_target["label"],
            "depth_gap_m": pair.depth_gap_m,
            "uncertainty_3se_m": pair.uncertainty_3se_m,
            "required_gap_m": pair.required_gap_m,
            "margin_m": pair.margin_m,
            "quartile_separated": pair.quartile_separated,
            "quartile_gap_m": pair.quartile_gap_m,
            "verification": "registered_metric_sensor_depth",
        },
        "sensor_provenance": {
            "source_dataset": "ScanNet v2",
            "derived_frame_bundle": "OpenEQA scannet-v0",
            "scene_id": scene.scene_id,
            "frame_index": frame_index,
            "split_membership": split_document(
                split_manifests[scene.official_split]
            ),
            "rgb": {
                "path": str(rgb_path),
                "sha256": bytes_sha256(rgb_bytes),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "encoding": "png_uint8_srgb",
            },
            "depth": {
                "path": str(depth_path),
                "sha256": bytes_sha256(depth_bytes),
                "width": int(depth.shape[1]),
                "height": int(depth.shape[0]),
                "encoding": "png_uint16",
                "unit": "millimeter",
                "scale_to_meter": 0.001,
                "valid_pixels": registered.raw_valid_pixels,
            },
            "pose": {
                "path": str(pose_path),
                "sha256": file_sha256(pose_path),
                "valid_finite_homogeneous_transform": pose_valid(pose_path),
                "used_for_depth_registration": False,
            },
            "calibration": calibration,
            "registration": {
                "method": "depth_to_color_reprojection_zbuffer_v1",
                "transform": "inv(extrinsic_color)@extrinsic_depth",
                "registered_pixels": registered.registered_pixels,
            },
            "instance_projection": {
                "zip_path": str(scene.instance_zip),
                "member": instance_member,
                "sha256": bytes_sha256(instance_bytes),
            },
            "semantic_projection": {
                "zip_path": str(scene.label_zip),
                "member": semantic_member,
                "sha256": bytes_sha256(semantic_bytes),
                "encoding": "scannet_raw_label_id",
            },
            "scene_metadata": {
                "path": str(scene.scene_metadata),
                "sha256": file_sha256(scene.scene_metadata),
            },
            "aggregation": {
                "path": str(scene.aggregation_json),
                "sha256": aggregation_sha256,
                "instance_id_rule": "objectId_plus_1",
            },
        },
        "semantic_gate": {
            "depth_v50_scannet_sensor": {
                "passed": True,
                "min_gap_m": MIN_DEPTH_GAP_M,
                "depth_gap_m": pair.depth_gap_m,
                "uncertainty_3se_m": pair.uncertainty_3se_m,
                "required_gap_m": pair.required_gap_m,
                "min_valid_samples": MIN_VALID_SAMPLES,
                "min_valid_fraction": MIN_VALID_FRACTION,
                "visual_quality_contract": visual_quality_contract(),
                "pair_selection_rule": "visual_auditability_then_metric_margin_v1",
                "sensor_only": True,
                "vlm_used": False,
                "monocular_depth_used": False,
            }
        },
        "training_materialization": {
            "eligible": False,
            "reason": "official_scannet_held_out_split",
            "allowed_use": "blind_audit_or_benchmark_only",
        },
        "model_interface": {
            "intended_task": "oracle_grounded_pairwise_depth",
            "requires_explicit_pair_groundings": True,
            "instance_ids_are_internal_identity_only": True,
        },
    }
    errors = validate_record(record, split_manifests)
    if errors:
        raise SourceError(f"record validation failed: {','.join(errors)}")
    return record, "accepted", instance_rejections


def attach_scene_hashes(records: list[dict[str, Any]], scene: SceneSource) -> None:
    if not records:
        return
    instance_zip_sha256 = file_sha256(scene.instance_zip)
    label_zip_sha256 = file_sha256(scene.label_zip)
    for record in records:
        provenance = record["sensor_provenance"]
        provenance["instance_projection"]["zip_sha256"] = instance_zip_sha256
        provenance["semantic_projection"]["zip_sha256"] = label_zip_sha256


def process_scene(
    scene: SceneSource,
    label_maps: LabelMaps,
    split_manifests: dict[str, SplitManifest],
    max_frames: int,
    max_records: int,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], list[str]]:
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    instance_rejections: Counter[str] = Counter()
    system_errors: list[str] = []
    if not source_ready(scene):
        rejected["scene_source_incomplete"] += 1
        return accepted, rejected, instance_rejections, system_errors
    try:
        available_frames = discover_frame_indices(scene)
    except SourceError as exc:
        rejected["scene_frame_inventory_invalid"] += 1
        system_errors.append(f"{scene.scene_id}:frames:{exc}")
        return accepted, rejected, instance_rejections, system_errors
    frame_indices = stable_order(
        available_frames, f"frames|{scene.scene_id}"
    )[:max_frames]
    try:
        with ZipFile(scene.instance_zip) as instance_archive, ZipFile(
            scene.label_zip
        ) as label_archive:
            for frame_index in frame_indices:
                try:
                    record, reason, frame_instance_rejections = analyze_frame(
                        scene,
                        frame_index,
                        label_maps,
                        split_manifests,
                        instance_archive,
                        label_archive,
                    )
                    instance_rejections.update(frame_instance_rejections)
                    if record is None:
                        rejected[reason] += 1
                        continue
                    accepted.append(record)
                except SourceError as exc:
                    rejected["frame_source_rejected"] += 1
                    system_errors.append(f"{scene.scene_id}:{frame_index}:{exc}")
    except (OSError, BadZipFile) as exc:
        rejected["invalid_scene_zip"] += 1
        system_errors.append(f"{scene.scene_id}:zip:{exc}")
    accepted.sort(key=record_auditability_score, reverse=True)
    if len(accepted) > max_records:
        rejected["lower_auditability_pair_not_selected"] += len(accepted) - max_records
        accepted = accepted[:max_records]
    attach_scene_hashes(accepted, scene)
    return accepted, rejected, instance_rejections, system_errors


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def source_artifact_errors(
    record: dict[str, Any],
    file_hash_cache: dict[str, str] | None = None,
    member_hash_cache: dict[tuple[str, str], str] | None = None,
) -> list[str]:
    file_hash_cache = file_hash_cache if file_hash_cache is not None else {}
    member_hash_cache = member_hash_cache if member_hash_cache is not None else {}
    errors: list[str] = []
    provenance = record.get("sensor_provenance")
    if not isinstance(provenance, dict):
        return ["sensor_provenance"]

    def check_file(document: Any, section: str) -> Path | None:
        if not isinstance(document, dict):
            errors.append(f"{section}:document")
            return None
        path = Path(str(document.get("path", "")))
        expected = document.get("sha256")
        if not path.is_file() or not SHA256_RE.fullmatch(str(expected or "")):
            errors.append(f"{section}:identity")
            return None
        resolved = str(path.resolve())
        try:
            if resolved not in file_hash_cache:
                file_hash_cache[resolved] = file_sha256(path)
            observed = file_hash_cache[resolved]
        except OSError:
            errors.append(f"{section}:read")
            return None
        if observed != expected:
            errors.append(f"{section}:sha256")
        return path

    rgb = provenance.get("rgb")
    rgb_path = check_file(rgb, "rgb")
    check_file(provenance.get("depth"), "depth")
    check_file(provenance.get("pose"), "pose")
    check_file(provenance.get("scene_metadata"), "scene_metadata")
    check_file(provenance.get("aggregation"), "aggregation")
    calibration = provenance.get("calibration")
    if not isinstance(calibration, dict):
        errors.append("calibration:document")
    else:
        for name in (
            "intrinsic_color",
            "intrinsic_depth",
            "extrinsic_color",
            "extrinsic_depth",
        ):
            check_file(calibration.get(name), f"calibration:{name}")

    if isinstance(rgb, dict) and rgb_path is not None:
        image_path = Path(str(record.get("image_path", "")))
        if (
            image_path.resolve(strict=False) != rgb_path.resolve()
            or record.get("image_sha256") != rgb.get("sha256")
        ):
            errors.append("image_identity")

    for section in ("instance_projection", "semantic_projection"):
        document = provenance.get(section)
        if not isinstance(document, dict):
            errors.append(f"{section}:document")
            continue
        zip_path = Path(str(document.get("zip_path", "")))
        member = document.get("member")
        expected_zip = document.get("zip_sha256")
        expected_member = document.get("sha256")
        if (
            not zip_path.is_file()
            or not isinstance(member, str)
            or not member
            or not SHA256_RE.fullmatch(str(expected_zip or ""))
            or not SHA256_RE.fullmatch(str(expected_member or ""))
        ):
            errors.append(f"{section}:identity")
            continue
        resolved_zip = str(zip_path.resolve())
        try:
            if resolved_zip not in file_hash_cache:
                file_hash_cache[resolved_zip] = file_sha256(zip_path)
            observed_zip = file_hash_cache[resolved_zip]
            key = (resolved_zip, member)
            if key not in member_hash_cache:
                with ZipFile(zip_path) as archive:
                    member_hash_cache[key] = bytes_sha256(archive.read(member))
        except (OSError, BadZipFile, KeyError):
            errors.append(f"{section}:read")
            continue
        if observed_zip != expected_zip:
            errors.append(f"{section}:zip_sha256")
        if member_hash_cache[key] != expected_member:
            errors.append(f"{section}:member_sha256")

    if (
        provenance.get("scene_id") != record.get("scene_id")
        or provenance.get("frame_index") != record.get("frame_index")
    ):
        errors.append("scene_frame_identity")
    return sorted(set(errors))


def load_canary_binding(
    path: Path,
    split_manifests: dict[str, SplitManifest],
    source_exclusion_binding: dict[str, Any],
    canary_scene_binding: dict[str, Any],
    expected_source_inventory: dict[str, Any],
    label_map_path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"full stage requires canary report: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid canary report: {path}") from exc
    labels_path = path.parent / "labels/S_spatial/depth.jsonl"
    scene_ids = report.get("accepted_scene_ids")
    rows: list[dict[str, Any]] = []
    if labels_path.is_file():
        try:
            rows = [
                json.loads(line)
                for line in labels_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            rows = []
    row_scene_ids = [row.get("scene_id") for row in rows]
    row_image_hashes = [row.get("image_sha256") for row in rows]
    file_hash_cache: dict[str, str] = {}
    member_hash_cache: dict[tuple[str, str], str] = {}
    artifact_errors = {
        str(row.get("sample_id")): source_artifact_errors(
            row, file_hash_cache, member_hash_cache
        )
        for row in rows
    }
    artifact_errors = {
        sample_id: errors for sample_id, errors in artifact_errors.items() if errors
    }
    if (
        report.get("stage") != "canary"
        or report.get("protocol") != PROTOCOL
        or report.get("label_version") != LABEL_VERSION
        or report.get("audit_round") != AUDIT_ROUND
        or report.get("source_artifact_schema") != SOURCE_ARTIFACT_SCHEMA
        or report.get("builder_sha256") != file_sha256(Path(__file__))
        or report.get("selection_rule")
        != "frozen_canary_scenes_scan64_then_visual_auditability_top1"
        or report.get("accepted") != CANARY_TARGET
        or report.get("target") != CANARY_TARGET
        or report.get("accepted_scenes") != CANARY_TARGET
        or report.get("accepted_by_official_split") != {"val": CANARY_TARGET}
        or report.get("quartile_separated") != CANARY_TARGET
        or report.get("eligible") is not True
        or report.get("system_errors", {}).get("total") != 0
        or not isinstance(scene_ids, list)
        or len(scene_ids) != CANARY_TARGET
        or len(set(scene_ids)) != CANARY_TARGET
        or sorted(scene_ids) != canary_scene_binding["scene_ids"]
        or not labels_path.is_file()
        or report.get("labels_sha256") != file_sha256(labels_path)
        or len(rows) != CANARY_TARGET
        or report.get("records_sha256") != canonical_hash(rows)
        or any(validate_record(row, split_manifests) for row in rows)
        or artifact_errors
        or sorted(row_scene_ids) != sorted(scene_ids)
        or len({row.get("sample_id") for row in rows}) != CANARY_TARGET
        or len(set(row_image_hashes)) != CANARY_TARGET
        or report.get("canary_binding") is not None
        or report.get("scene_overlap_with_canary") != []
        or report.get("frozen_canary_scenes") != canary_scene_binding
        or report.get("visual_quality_contract") != visual_quality_contract()
        or report.get("source_inventory") != expected_source_inventory
        or report.get("excluded_corrupt_source_scene_ids")
        != sorted(
            item["scene_id"]
            for item in source_exclusion_binding.get("exclusions", [])
        )
        or report.get("source_contract", {}).get("generation_split") != "val"
        or report.get("source_contract", {}).get("label_map_path")
        != str(label_map_path.resolve())
        or report.get("source_contract", {}).get("label_map_sha256")
        != SCANNET_LABEL_MAP_SHA256
        or report.get("source_contract", {}).get("visual_quality_contract")
        != visual_quality_contract()
        or report.get("source_contract", {}).get(
            "training_materialization_eligible"
        ) is not False
        or report.get("generation_parameters") != {
            "target": CANARY_TARGET,
            "max_frames_per_scene": MAX_FRAMES_PER_SCENE,
            "max_records_per_scene": 1,
            "minimum_accepted_scenes": CANARY_TARGET,
            "visual_quality_contract": visual_quality_contract(),
            "frozen_canary_scene_manifest_sha256": (
                canary_scene_binding["manifest_sha256"]
            ),
            "label_map_sha256": SCANNET_LABEL_MAP_SHA256,
        }
        or report.get("source_contract", {}).get("official_split_manifests")
        != {
            name: split_document(manifest)
            for name, manifest in sorted(split_manifests.items())
        }
        or report.get("validated_source_exclusions") != source_exclusion_binding
    ):
        raise SystemExit("canary report does not satisfy the v5.0 sensor binding")
    return {
        "report_path": str(path.resolve()),
        "report_sha256": file_sha256(path),
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": file_sha256(labels_path),
        "scene_ids": scene_ids,
        "image_sha256s": row_image_hashes,
        "frozen_canary_scenes": canary_scene_binding,
        "source_artifact_replay": {
            "records": len(rows),
            "files_rehashed": len(file_hash_cache),
            "zip_members_rehashed": len(member_hash_cache),
            "errors": 0,
        },
    }


def inventory_report(
    scenes: list[SceneSource],
    openeqa_root: Path,
    scannet_root: Path,
    source_exclusions: dict[str, SourceExclusion] | None = None,
) -> dict[str, Any]:
    excluded = set(source_exclusions or {})
    physically_ready = [scene for scene in scenes if source_ready(scene)]
    ready = [
        scene for scene in physically_ready if scene.scene_id not in excluded
    ]
    return {
        "openeqa_root": str(openeqa_root.resolve()),
        "scannet_root": str(scannet_root.resolve()),
        "scenes_discovered": len(scenes),
        "scenes_with_complete_source_files": len(physically_ready),
        "scenes_ready": len(ready),
        "validated_source_exclusion_count": len(excluded),
        "validated_source_exclusion_scene_ids": sorted(excluded),
        "frame_inventory_mode": "lazy_exact_rgb_depth_pose_intersection_per_scene",
        "paired_frames": None,
        "paired_frames_ready": None,
        "missing_scene_ids": [
            scene.scene_id for scene in scenes if scene not in physically_ready
        ],
        "scenes_by_official_split": dict(
            sorted(Counter(scene.official_split for scene in scenes).items())
        ),
        "ready_scenes_by_official_split": dict(
            sorted(Counter(scene.official_split for scene in ready).items())
        ),
    }


def run_check(
    args: argparse.Namespace,
    scenes: list[SceneSource],
    split_manifests: dict[str, SplitManifest],
    source_exclusions: dict[str, SourceExclusion],
    source_exclusion_binding: dict[str, Any],
    canary_scene_binding: dict[str, Any],
) -> int:
    ensure_empty_output(args.output_dir)
    inventory = inventory_report(
        scenes, args.openeqa_root, args.scannet_root, source_exclusions
    )
    label_maps = load_label_map(args.label_map)
    probes: list[dict[str, Any]] = []
    errors: list[str] = []
    source_rejections: list[str] = []
    ready_scenes = [
        scene
        for scene in scenes
        if (
            scene.official_split == "val"
            and source_ready(scene)
            and scene.scene_id not in source_exclusions
        )
    ]
    for scene in ready_scenes:
        if len(probes) >= args.check_scenes:
            break
        try:
            frame_index = stable_order(
                discover_frame_indices(scene), f"check|{scene.scene_id}"
            )[0]
            with ZipFile(scene.instance_zip) as instance_archive, ZipFile(
                scene.label_zip
            ) as label_archive:
                record, reason, _ = analyze_frame(
                    scene,
                    frame_index,
                    label_maps,
                    split_manifests,
                    instance_archive,
                    label_archive,
                )
            probes.append(
                {
                    "scene_id": scene.scene_id,
                    "frame_index": frame_index,
                    "candidate_pair": record is not None,
                    "reason": reason,
                }
            )
        except SourceError as exc:
            source_rejections.append(f"{scene.scene_id}:{exc}")
        except (OSError, BadZipFile) as exc:
            errors.append(f"{scene.scene_id}:{exc}")
    ready = (
        inventory["scenes_ready"] >= MIN_FULL_SCENES
        and len(probes) == args.check_scenes
        and not errors
        and not source_rejections
    )
    report = {
        "stage": "check",
        "protocol": PROTOCOL,
        "label_version": LABEL_VERSION,
        "audit_round": AUDIT_ROUND,
        "source_artifact_schema": SOURCE_ARTIFACT_SCHEMA,
        "builder_sha256": file_sha256(Path(__file__)),
        "source_inventory": inventory,
        "validated_source_exclusions": source_exclusion_binding,
        "frozen_canary_scenes": canary_scene_binding,
        "visual_quality_contract": visual_quality_contract(),
        "label_map": {
            "path": str(args.label_map.resolve()),
            "sha256": SCANNET_LABEL_MAP_SHA256,
        },
        "probe_count": len(probes),
        "probes": probes,
        "source_rejections": {
            "total": len(source_rejections),
            "items": source_rejections,
        },
        "system_errors": {"total": len(errors), "items": errors},
        "source_ready": ready,
        "official_split_manifests": {
            name: split_document(manifest)
            for name, manifest in sorted(split_manifests.items())
        },
        "license": {
            "frame_extractor_code": "OpenEQA MIT",
            "underlying_data": "ScanNet Terms of Use",
            "terms_reference": (
                "http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf"
            ),
            "redistribution_requires_terms_review": True,
        },
    }
    _write_json(args.output_dir / "generation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ready else 2


def run_generation(
    args: argparse.Namespace,
    scenes: list[SceneSource],
    split_manifests: dict[str, SplitManifest],
    canary_binding: dict[str, Any] | None,
    source_exclusions: dict[str, SourceExclusion],
    source_exclusion_binding: dict[str, Any],
    canary_scene_binding: dict[str, Any],
) -> int:
    ensure_empty_output(args.output_dir)
    label_maps = load_label_map(args.label_map)
    excluded_scenes = set((canary_binding or {}).get("scene_ids", []))
    excluded_source_scenes = set(source_exclusions)
    frozen_canary_scenes = set(canary_scene_binding["scene_ids"])
    stage_scenes = [
        scene
        for scene in scenes
        if (
            scene.official_split == "val"
            and scene.scene_id not in excluded_scenes
            and scene.scene_id not in excluded_source_scenes
            and (
                args.stage != "canary"
                or scene.scene_id in frozen_canary_scenes
            )
        )
    ]
    target = CANARY_TARGET if args.stage == "canary" else args.target
    max_records_per_scene = 1 if args.stage == "canary" else args.max_records_per_scene
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    instance_rejections: Counter[str] = Counter()
    system_errors: list[str] = []
    attempted_scenes = 0

    scene_batches: list[list[dict[str, Any]]] = []
    for scene in stage_scenes:
        if args.stage == "canary" and len(records) >= target:
            break
        attempted_scenes += 1
        scene_records, scene_rejected, scene_instance_rejected, scene_errors = (
            process_scene(
                scene,
                label_maps,
                split_manifests,
                max_frames=args.max_frames_per_scene,
                max_records=max_records_per_scene,
            )
        )
        rejected.update(scene_rejected)
        instance_rejections.update(scene_instance_rejected)
        system_errors.extend(scene_errors)
        if args.stage == "canary":
            records.extend(scene_records[:1])
        else:
            scene_batches.append(scene_records)
    if args.stage == "full":
        records = scene_round_robin(scene_batches, target)

    labels_path = args.output_dir / "labels/S_spatial/depth.jsonl"
    _write_jsonl(labels_path, records)
    accepted_scene_ids = sorted({record["scene_id"] for record in records})
    scene_overlap = sorted(set(accepted_scene_ids) & excluded_scenes)
    canary_scene_set_matches = (
        args.stage != "canary"
        or accepted_scene_ids == canary_scene_binding["scene_ids"]
    )
    eligible = (
        len(records) == target
        and not scene_overlap
        and not system_errors
        and canary_scene_set_matches
        and (
            (args.stage == "canary" and len(accepted_scene_ids) == CANARY_TARGET)
            or len(accepted_scene_ids) >= MIN_FULL_SCENES
        )
    )
    report = {
        "stage": args.stage,
        "protocol": PROTOCOL,
        "label_version": LABEL_VERSION,
        "audit_round": AUDIT_ROUND,
        "source_artifact_schema": SOURCE_ARTIFACT_SCHEMA,
        "builder_sha256": file_sha256(Path(__file__)),
        "selection_rule": (
            "frozen_canary_scenes_scan64_then_visual_auditability_top1"
            if args.stage == "canary"
            else "scan64_visual_top2_per_scene_then_scene_round_robin"
        ),
        "generation_parameters": {
            "target": target,
            "max_frames_per_scene": args.max_frames_per_scene,
            "max_records_per_scene": max_records_per_scene,
            "minimum_accepted_scenes": (
                CANARY_TARGET if args.stage == "canary" else MIN_FULL_SCENES
            ),
            "visual_quality_contract": visual_quality_contract(),
            "frozen_canary_scene_manifest_sha256": (
                canary_scene_binding["manifest_sha256"]
            ),
            "label_map_sha256": SCANNET_LABEL_MAP_SHA256,
        },
        "target": target,
        "accepted": len(records),
        "accepted_scenes": len(accepted_scene_ids),
        "accepted_scene_ids": accepted_scene_ids,
        "attempted_scenes": attempted_scenes,
        "rejected_by_reason": dict(sorted(rejected.items())),
        "instance_rejected_by_reason": dict(sorted(instance_rejections.items())),
        "system_errors": {"total": len(system_errors), "items": system_errors},
        "accepted_by_official_split": dict(
            sorted(Counter(record["official_split"] for record in records).items())
        ),
        "quartile_separated": sum(
            bool(record["verified_pair"]["quartile_separated"])
            for record in records
        ),
        "visual_quality_contract": visual_quality_contract(),
        "accepted_by_gap_bucket": dict(
            sorted(
                Counter(
                    record["visibility_features"]["gap_bucket"]
                    for record in records
                ).items()
            )
        ),
        "source_inventory": inventory_report(
            scenes, args.openeqa_root, args.scannet_root, source_exclusions
        ),
        "validated_source_exclusions": source_exclusion_binding,
        "source_contract": {
            "depth_encoding": "png_uint16_millimeter",
            "registration": "depth_to_color_reprojection_zbuffer_v1",
            "instance_projection": "official_2d-instance-filt_zip_member",
            "semantic_projection": "official_2d-label-filt_zip_member_raw_id",
            "label_map_path": str(args.label_map.resolve()),
            "label_map_sha256": SCANNET_LABEL_MAP_SHA256,
            "min_depth_gap_m": MIN_DEPTH_GAP_M,
            "uncertainty_rule": "max(0.25m,3*sqrt(se_near^2+se_far^2))",
            "depth_statistic": "registered_visible_surface_median",
            "visual_quality_contract": visual_quality_contract(),
            "pair_selection_rule": "visual_auditability_then_metric_margin_v1",
            "official_split_manifests": {
                name: split_document(manifest)
                for name, manifest in sorted(split_manifests.items())
            },
            "generation_split": "val",
            "test_split_excluded_reason": "official_2d_ground_truth_unavailable_locally",
            "intended_use": "held_out_blind_audit_or_benchmark_only",
            "pool_scope": "calibration_candidates_not_full_benchmark_scale",
            "intended_task": "oracle_grounded_pairwise_depth",
            "model_interface_requires_explicit_pair_groundings": True,
            "training_materialization_eligible": False,
            "sensor_only": True,
        },
        "canary_binding": canary_binding,
        "frozen_canary_scenes": canary_scene_binding,
        "excluded_canary_scene_ids": sorted(excluded_scenes),
        "excluded_corrupt_source_scene_ids": sorted(excluded_source_scenes),
        "scene_overlap_with_canary": scene_overlap,
        "labels_path": str(labels_path.relative_to(args.output_dir)),
        "labels_sha256": file_sha256(labels_path),
        "records_sha256": canonical_hash(records),
        "eligible": eligible,
    }
    report_path = args.output_dir / "generation_report.json"
    _write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if eligible else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "canary", "full"), default="check")
    parser.add_argument("--check-only", action="store_true", help="alias for --stage check")
    parser.add_argument("--openeqa-root", type=Path, default=DEFAULT_OPENEQA_ROOT)
    parser.add_argument("--scannet-root", type=Path, default=DEFAULT_SCANNET_ROOT)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--source-exclusions", type=Path, default=DEFAULT_SOURCE_EXCLUSIONS
    )
    parser.add_argument(
        "--canary-scenes", type=Path, default=DEFAULT_CANARY_SCENES
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--canary-report", type=Path)
    parser.add_argument("--target", type=int, default=FULL_TARGET)
    parser.add_argument("--max-records-per-scene", type=int, default=MAX_RECORDS_PER_SCENE)
    parser.add_argument("--max-frames-per-scene", type=int, default=MAX_FRAMES_PER_SCENE)
    parser.add_argument("--check-scenes", type=int, default=10)
    args = parser.parse_args(argv)
    if args.check_only:
        args.stage = "check"
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_CHECK_OUT
            if args.stage == "check"
            else DEFAULT_CANARY_OUT
            if args.stage == "canary"
            else DEFAULT_FULL_OUT
        )
    if args.stage == "full" and args.target != FULL_TARGET:
        parser.error(f"full target is frozen at {FULL_TARGET}")
    if args.stage in {"canary", "full"} and args.max_frames_per_scene != MAX_FRAMES_PER_SCENE:
        parser.error(
            f"formal max-frames-per-scene is frozen at {MAX_FRAMES_PER_SCENE}"
        )
    if args.stage == "full" and args.max_records_per_scene != MAX_RECORDS_PER_SCENE:
        parser.error(
            f"full max-records-per-scene is frozen at {MAX_RECORDS_PER_SCENE}"
        )
    if args.max_records_per_scene < 1 or args.max_frames_per_scene < 1:
        parser.error("scene/frame limits must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        split_manifests, scene_to_split = load_split_manifests(args.split_root)
        source_exclusions, source_exclusion_binding = load_source_exclusions(
            args.source_exclusions, args.scannet_root, scene_to_split
        )
        scenes = discover_scenes(
            args.openeqa_root, args.scannet_root, scene_to_split
        )
        canary_scene_binding = load_canary_scene_manifest(
            args.canary_scenes,
            scenes,
            scene_to_split,
            source_exclusions,
        )
    except SourceError as exc:
        raise SystemExit(str(exc)) from exc
    if args.stage == "check":
        return run_check(
            args,
            scenes,
            split_manifests,
            source_exclusions,
            source_exclusion_binding,
            canary_scene_binding,
        )
    canary_binding = None
    if args.stage == "full":
        if args.canary_report is None:
            raise SystemExit("full stage requires --canary-report")
        canary_binding = load_canary_binding(
            args.canary_report,
            split_manifests,
            source_exclusion_binding,
            canary_scene_binding,
            inventory_report(
                scenes,
                args.openeqa_root,
                args.scannet_root,
                source_exclusions,
            ),
            args.label_map,
        )
    return run_generation(
        args,
        scenes,
        split_manifests,
        canary_binding,
        source_exclusions,
        source_exclusion_binding,
        canary_scene_binding,
    )


if __name__ == "__main__":
    raise SystemExit(main())
