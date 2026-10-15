#!/usr/bin/env python3
"""Extract v2 builder-compatible bundles from official-val ScanNet ``.sens`` files.

V2 differs from the frozen v1 protocol only where official ScanNet metadata can
count a terminal frame that is not present in the sensor stream. Each metadata
frame count may therefore equal ``sens.num_frames`` or ``sens.num_frames + 1``
independently. Dimensions and the 1000 raw-units-per-meter depth shift remain
exact, and selected frames must exist in both annotation streams and in ``.sens``.

This is a CPU-only held-out benchmark preparation tool. It never produces
training-eligible data and never downloads source artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from zipfile import BadZipFile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from tools import extract_scannet_train_sens as sens  # noqa: E402
from tools import extract_scannet_val_frame_bundles as v1  # noqa: E402


SCHEMA = "gsa-scannet-official-val-frame-bundle-v2"
PROTOCOL = "gsa-scannet-val-frame-bundle-extraction-v2"
SELECTION_SEED = v1.SELECTION_SEED
DEFAULT_SCENE = "scene0011_00"
DEFAULT_OUTPUT = ROOT / "data/scannet_val_frame_bundles_v2/canary_v2"
DEFAULT_SOURCE_EXCLUSIONS = v1.DEFAULT_SOURCE_EXCLUSIONS
MAX_FRAMES_PER_SCENE = v1.MAX_FRAMES_PER_SCENE
CALIBRATION_NAMES = v1.CALIBRATION_NAMES

ExtractError = v1.ExtractError
ScenePaths = v1.ScenePaths
canonical_hash = v1.canonical_hash
load_source_exclusions = v1.load_source_exclusions
_frame_root_binding = v1._frame_root_binding


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
    metadata_validation: dict[str, Any]


def _metadata_integer(metadata: dict[str, str], key: str) -> int:
    try:
        return int(metadata[key])
    except (KeyError, ValueError) as exc:
        raise sens.SourceError(
            f"missing or invalid {key} in ScanNet scene metadata"
        ) from exc


def validate_metadata_v2(
    header: sens.SensHeader, metadata: dict[str, str]
) -> dict[str, Any]:
    """Validate exact dimensions and the narrowly allowed terminal-frame offset."""
    expected_dimensions = {
        "colorWidth": header.color_width,
        "colorHeight": header.color_height,
        "depthWidth": header.depth_width,
        "depthHeight": header.depth_height,
    }
    observed_dimensions: dict[str, int] = {}
    for key, expected in expected_dimensions.items():
        observed = _metadata_integer(metadata, key)
        if observed != expected:
            raise sens.SourceError(
                f"scene metadata mismatch for {key}: {observed} != {expected}"
            )
        observed_dimensions[key] = observed

    allowed_frame_counts = (header.num_frames, header.num_frames + 1)
    observed_frame_counts: dict[str, int] = {}
    for key in ("numColorFrames", "numDepthFrames"):
        observed = _metadata_integer(metadata, key)
        if observed not in allowed_frame_counts:
            raise sens.SourceError(
                f"scene metadata mismatch for {key}: {observed} not in "
                f"{{{allowed_frame_counts[0]},{allowed_frame_counts[1]}}}"
            )
        observed_frame_counts[key] = observed

    return {
        "policy": "each_metadata_stream_count_is_sens_or_sens_plus_one",
        "sens_num_frames": header.num_frames,
        "allowed_metadata_frame_counts": list(allowed_frame_counts),
        "observed_frame_counts": observed_frame_counts,
        "dimensions_exact": True,
        "observed_dimensions": observed_dimensions,
        "depth_shift_exact": True,
        "depth_shift_raw_units_per_meter": header.depth_shift,
    }


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

    paths = v1._paths(scannet_root, scene_id)
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
    if header.depth_shift != 1000.0:
        raise ExtractError(
            "builder-compatible depth requires exact depth_shift=1000, got "
            f"{header.depth_shift}: {scene_id}"
        )
    metadata_validation = validate_metadata_v2(
        header, sens.parse_scene_metadata(paths.metadata)
    )
    aggregation_object_count = v1._validate_aggregation(paths.aggregation, scene_id)
    instance_members = v1._archive_members(paths.instance_zip, v1.INSTANCE_MEMBER_RE)
    semantic_members = v1._archive_members(paths.semantic_zip, v1.SEMANTIC_MEMBER_RE)
    common = {
        index
        for index in instance_members.keys() & semantic_members.keys()
        if 0 <= index < header.num_frames and index <= 999_999
    }
    selected = stable_frame_indices(scene_id, common, max_frames)
    if not selected:
        raise ExtractError(f"no common instance/semantic frame indices: {scene_id}")
    if any(
        index not in instance_members
        or index not in semantic_members
        or not 0 <= index < header.num_frames
        for index in selected
    ):
        raise ExtractError(f"selected frame escaped v2 source bounds: {scene_id}")

    expected_size = (header.color_width, header.color_height)
    instance_members = v1._validated_member_documents(
        archive_path=paths.instance_zip,
        member_documents=instance_members,
        selected_indices=selected,
        expected_size=expected_size,
    )
    semantic_members = v1._validated_member_documents(
        archive_path=paths.semantic_zip,
        member_documents=semantic_members,
        selected_indices=selected,
        expected_size=expected_size,
    )
    source_files = {
        "sens": v1._source_document(paths.sens),
        "metadata": v1._source_document(paths.metadata),
        "aggregation": v1._source_document(paths.aggregation),
        "instance_zip": v1._source_document(paths.instance_zip),
        "semantic_zip": v1._source_document(paths.semantic_zip),
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
        metadata_validation=metadata_validation,
    )


def _binding() -> dict[str, Any]:
    extractor = Path(__file__).resolve()
    helper = Path(v1.__file__).resolve()
    reader = Path(sens.__file__).resolve()
    return {
        "extractor_path": str(extractor),
        "extractor_sha256": sens.file_sha256(extractor),
        "schema_owner": SCHEMA,
        "protocol_owner": PROTOCOL,
        "implementation_dependencies": [
            {
                "role": "v1_frame_io_helpers",
                "path": str(helper),
                "sha256": sens.file_sha256(helper),
                "functions": [
                    "_paths",
                    "_validate_aggregation",
                    "canonical_hash",
                    "load_source_exclusions",
                    "_archive_members",
                    "_validated_member_documents",
                    "_source_document",
                    "_decode_frames",
                    "_verify_source_identities",
                    "_use_policy",
                    "_write_matrix",
                    "_artifact_document",
                    "_require_output_empty",
                    "_frame_root_binding",
                    "_verify_frame_root_bindings",
                    "_read_scene_list",
                ],
            },
            {
                "role": "sens_reader",
                "path": str(reader),
                "sha256": sens.file_sha256(reader),
                "functions": [
                    "read_sens_header",
                    "_read_header",
                    "decode_color",
                    "decode_depth",
                    "parse_scene_metadata",
                    "load_split_manifests",
                    "header_document",
                    "file_sha256",
                    "bytes_sha256",
                ],
            },
        ],
    }


def _inspection_document(inspection: SceneInspection) -> dict[str, Any]:
    return {
        "scene_id": inspection.paths.scene_id,
        "official_split": "val",
        "sensor_header": sens.header_document(inspection.header),
        "metadata_validation": inspection.metadata_validation,
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
    decoded = v1._decode_frames(inspection)
    v1._verify_source_identities(inspection)
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
        "use_policy": v1._use_policy(),
        "extractor": _binding(),
        "official_splits": split_documents,
        "source_exclusions": source_exclusions_binding,
        "frame_root_exclusions": frame_root_exclusions,
        "source_identities_reverified_after_decode": True,
        "selection": {
            "seed": SELECTION_SEED,
            "requested_max_frames_per_scene": inspection.requested_max_frames,
            "selected_frame_count": len(inspection.selected_indices),
            "selected_only_from_instance_semantic_member_intersection": True,
            "selected_indices_strictly_below_sens_num_frames": True,
        },
        "scene": _inspection_document(inspection),
        "decoded_frames": decoded_rows,
    }


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

    v1._require_output_empty(output_root)
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
            decoded = v1._decode_frames(inspection)
            calibration = {
                "intrinsic_color.txt": inspection.header.intrinsic_color,
                "intrinsic_depth.txt": inspection.header.intrinsic_depth,
                "extrinsic_color.txt": inspection.header.extrinsic_color,
                "extrinsic_depth.txt": inspection.header.extrinsic_depth,
            }
            for name, matrix in calibration.items():
                v1._write_matrix(scene_root / name, matrix)

            frame_documents = []
            for index in inspection.selected_indices:
                frame, color, depth = decoded[index]
                rgb_path = scene_root / f"{index:06d}-rgb.png"
                depth_path = scene_root / f"{index:06d}-depth.png"
                pose_path = scene_root / f"{index:06d}.txt"
                Image.fromarray(color).save(rgb_path)
                Image.fromarray(depth).save(depth_path)
                v1._write_matrix(pose_path, frame.camera_to_world)
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
                            "rgb": v1._artifact_document(rgb_path, temp_root),
                            "depth": v1._artifact_document(depth_path, temp_root),
                            "pose": v1._artifact_document(pose_path, temp_root),
                        },
                    }
                )
            v1._verify_source_identities(inspection)
            total_frames += len(frame_documents)
            document = _inspection_document(inspection)
            document.update(
                {
                    "ordinal": ordinal,
                    "output_directory": directory_name,
                    "calibration_outputs": {
                        name: v1._artifact_document(scene_root / name, temp_root)
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
            "output_contract": {
                "root": str(output_root.resolve()),
                "namespace": "scannet_val_frame_bundles_v2",
            },
            "training_materialization_eligible": False,
            "use_policy": v1._use_policy(),
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
                "selected_indices_strictly_below_sens_num_frames": True,
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
        v1._verify_frame_root_bindings(frame_root_exclusions)
        if output_root.exists():
            v1._require_output_empty(output_root)
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
        return v1._read_scene_list(args.scene_list)
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
                "requested scenes overlap an excluded frame root: "
                f"{','.join(overlap[:8])}"
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
                {"schema": SCHEMA, "protocol": PROTOCOL, "verdict": "BLOCKED", "error": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
