#!/usr/bin/env python3
"""Bounded scale-out extraction of official-train ScanNet .sens frame bundles.

Reuses the audited minimal v4 reader in tools/extract_scannet_train_sens.py and
adds multi-scene, multi-frame iteration with a per-frame streaming decode so peak
memory stays bounded by one frame payload rather than one scene.

Refuses val and test scenes. CPU only, no network. Requires a project approval
record before writing anything, and records that approval provenance verbatim in
the output manifest so the license state stays auditable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.extract_scannet_train_sens import (  # noqa: E402
    MAX_FRAME_PAYLOAD_BYTES,
    SCENE_RE,
    SensHeader,
    SourceError,
    _matrix,
    _read_exact,
    _read_header,
    _scalar,
    bytes_sha256,
    decode_color,
    decode_depth,
    file_sha256,
    load_split_manifests,
    parse_scene_metadata,
    register_depth_to_color,
    require_source_files,
    source_paths,
    validate_metadata,
)

SCHEMA = "gsa-scannet-official-train-sens-scaleout-v1"
PROTOCOL = "gsa-scannet-train-sens-cpu-scaleout-v1"
DEFAULT_APPROVAL = REPOSITORY_ROOT / "data/scannet_license/approval_record.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data/scannet_train_sens_adapter/scaleout_v1"
# 4x4 float32 pose + 2 uint64 timestamps + 2 uint64 payload sizes.
FRAME_RECORD_HEADER_BYTES = 64 + 16 + 16

ACCEPTED_APPROVAL_STATUS = "APPROVED_BY_PROJECT_OWNER_SIGNED_TERMS_PENDING_ARCHIVAL"


def load_approval(path: Path) -> dict[str, Any]:
    """Fail closed unless a project approval record explicitly clears train use."""
    if not path.is_file():
        raise SourceError(f"missing ScanNet approval record: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    approval = record.get("approval") or {}
    status = str(approval.get("status") or "")
    if status != ACCEPTED_APPROVAL_STATUS:
        raise SourceError(f"approval record does not clear train materialization: {status}")
    if "scannet_tos_approval_not_locally_verified" not in (record.get("released_blockers") or []):
        raise SourceError("approval record does not release the ToS verification blocker")
    split_policy = (record.get("compliance_commitments_for_this_project") or {}).get(
        "split_discipline"
    ) or {}
    for key in ("val_split_materialization", "test_split_materialization"):
        if not str(split_policy.get(key, "")).startswith("FORBIDDEN"):
            raise SourceError(f"approval record must forbid {key}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "record_id": record.get("record_id"),
        "status": status,
        "granted_by": approval.get("granted_by"),
        "granted_on_utc_date": approval.get("granted_on_utc_date"),
        "signed_terms_archival_status": approval.get("signed_terms_archival_status"),
        "released_blockers": record.get("released_blockers"),
        "retained_blockers": record.get("retained_blockers"),
        "provenance_honesty_note": approval.get("provenance_honesty_note"),
    }

def build_frame_index(handle: Any, header: SensHeader) -> list[tuple[int, int, int]]:
    """Return (payload_offset, color_size, depth_size) per frame, skipping payloads.

    Indexing once per scene turns the canary's O(frames^2) rescan into a single
    linear pass, which is what makes multi-frame-per-scene extraction affordable.
    """
    file_size = os.fstat(handle.fileno()).st_size
    handle.seek(header.frame_data_offset)
    index: list[tuple[int, int, int]] = []
    # Some locally mirrored .sens files are truncated or corrupted mid-stream
    # (scene0000_00 diverges at frame 14 under the audited canary reader too).
    # Stop indexing at the first unparseable record instead of discarding the
    # whole scene, and report the shortfall so coverage stays auditable.
    for frame_index in range(header.num_frames):
        try:
            handle.seek(FRAME_RECORD_HEADER_BYTES - 16, 1)  # pose + timestamps
            color_size = int(_scalar(handle, "<Q", f"frame[{frame_index}].color_size"))
            depth_size = int(_scalar(handle, "<Q", f"frame[{frame_index}].depth_size"))
        except (SourceError, OSError, struct.error, ValueError):
            break
        if not 0 < color_size <= MAX_FRAME_PAYLOAD_BYTES:
            break
        if not 0 < depth_size <= MAX_FRAME_PAYLOAD_BYTES:
            break
        payload_offset = handle.tell()
        if payload_offset + color_size + depth_size > file_size:
            break
        index.append((payload_offset, color_size, depth_size))
        try:
            handle.seek(color_size + depth_size, 1)
        except (OSError, ValueError):
            break
    if not index:
        raise SourceError("no parseable frames in .sens")
    return index


def read_indexed_frame(
    handle: Any, entry: tuple[int, int, int], frame_index: int
) -> tuple[np.ndarray, bytes, bytes]:
    payload_offset, color_size, depth_size = entry
    handle.seek(payload_offset - FRAME_RECORD_HEADER_BYTES)
    pose = _matrix(handle, f"frame[{frame_index}].camera_to_world")
    handle.seek(payload_offset)
    color_bytes = _read_exact(handle, color_size, f"frame[{frame_index}].color")
    depth_bytes = _read_exact(handle, depth_size, f"frame[{frame_index}].depth")
    return pose, color_bytes, depth_bytes


def select_frame_indices(num_frames: int, max_frames: int) -> list[int]:
    """Evenly spread frames across the scene so coverage is not front-loaded."""
    if max_frames <= 0:
        raise SourceError("max frames per scene must be positive")
    if num_frames <= max_frames:
        return list(range(num_frames))
    step = num_frames / max_frames
    return sorted({min(num_frames - 1, int(i * step)) for i in range(max_frames)})

def validate_metadata_tolerant(header: SensHeader, metadata: dict[str, str]) -> dict[str, Any]:
    """Enforce geometry strictly; tolerate small declared frame-count drift.

    Resolutions must match exactly. Frame counts are allowed to differ by at
    most one because some official scenes ship a scene .txt whose
    numColorFrames/numDepthFrames is off by one from the .sens header
    (observed on scene0001_00: 1545 declared vs 1544 in the header). Frames are
    always read from the .sens stream, so the header is authoritative; the drift
    is recorded rather than silently accepted.
    """
    for key, expected in (
        ("colorWidth", header.color_width),
        ("colorHeight", header.color_height),
        ("depthWidth", header.depth_width),
        ("depthHeight", header.depth_height),
    ):
        try:
            observed = int(metadata[key])
        except (KeyError, ValueError) as exc:
            raise SourceError(f"missing or invalid {key} in scene metadata") from exc
        if observed != expected:
            raise SourceError(f"scene metadata mismatch for {key}: {observed} != {expected}")

    notes: dict[str, Any] = {"declared_frame_count_drift": {}}
    for key in ("numColorFrames", "numDepthFrames"):
        try:
            observed = int(metadata[key])
        except (KeyError, ValueError) as exc:
            raise SourceError(f"missing or invalid {key} in scene metadata") from exc
        delta = observed - header.num_frames
        if abs(delta) > 1:
            raise SourceError(
                f"scene metadata mismatch for {key}: {observed} != {header.num_frames}"
            )
        if delta:
            notes["declared_frame_count_drift"][key] = {
                "declared": observed,
                "sens_header": header.num_frames,
            }
    return notes


def read_annotation(archive: ZipFile, member: str, expected: tuple[int, int]) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = archive.read(member)
        with Image.open(__import__("io").BytesIO(payload)) as image:
            size, mode = image.size, image.mode
            image.verify()
    except (BadZipFile, KeyError, OSError, UnidentifiedImageError) as exc:
        raise SourceError(f"invalid annotation member {member}") from exc
    if size != expected:
        raise SourceError(f"annotation size mismatch {member}: {size} != {expected}")
    return payload, {
        "member": member,
        "member_sha256": bytes_sha256(payload),
        "member_size_bytes": len(payload),
        "image_width": size[0],
        "image_height": size[1],
        "image_mode": mode,
    }


def write_matrix(path: Path, value: np.ndarray) -> None:
    np.savetxt(path, value, fmt="%.9g")


def artifact_document(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }

def extract_scene(
    *,
    scene_id: str,
    scannet_root: Path,
    label_map: Path,
    max_frames: int,
    scene_out: Path,
) -> dict[str, Any]:
    paths = source_paths(scannet_root, label_map, scene_id)
    require_source_files(paths)
    metadata = parse_scene_metadata(paths.metadata)
    frames: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    started = time.time()

    with paths.sens.open("rb") as handle:
        header = _read_header(handle)
        metadata_notes = validate_metadata_tolerant(header, metadata)
        index = build_frame_index(handle, header)
        selected = select_frame_indices(len(index), max_frames)
        expected_size = (header.color_width, header.color_height)
        with ZipFile(paths.semantic_zip) as semantic_zip, ZipFile(paths.instance_zip) as instance_zip:
            for frame_index in selected:
                # A locally mirrored .sens can carry frames whose payload sizes
                # are plausible but whose bytes do not decode. Record and skip
                # those instead of discarding the whole scene; the manifest keeps
                # the skip list so coverage stays honest.
                try:
                    pose, color_bytes, depth_bytes = read_indexed_frame(
                        handle, index[frame_index], frame_index
                    )
                    color = decode_color(header, color_bytes)
                    depth = decode_depth(header, depth_bytes)
                    registration = register_depth_to_color(header, depth)
                    semantic_bytes, semantic_doc = read_annotation(
                        semantic_zip, f"label-filt/{frame_index}.png", expected_size
                    )
                    instance_bytes, instance_doc = read_annotation(
                        instance_zip, f"instance-filt/{frame_index}.png", expected_size
                    )
                except SourceError as exc:
                    skipped.append({"frame_index": frame_index, "reason": str(exc)})
                    continue
                frame_dir = scene_out / f"frame_{frame_index:06d}"
                frame_dir.mkdir(parents=True, exist_ok=False)
                Image.fromarray(color).save(frame_dir / "color.png")
                Image.fromarray(depth).save(frame_dir / "depth_native.png")
                Image.fromarray(registration.aligned_raw).save(
                    frame_dir / "depth_aligned_to_color.png"
                )
                (frame_dir / "semantic.png").write_bytes(semantic_bytes)
                (frame_dir / "instance.png").write_bytes(instance_bytes)
                write_matrix(frame_dir / "pose_camera_to_world.txt", pose)
                write_matrix(frame_dir / "intrinsic_color.txt", header.intrinsic_color)
                write_matrix(frame_dir / "extrinsic_color.txt", header.extrinsic_color)
                write_matrix(frame_dir / "intrinsic_depth.txt", header.intrinsic_depth)
                write_matrix(frame_dir / "extrinsic_depth.txt", header.extrinsic_depth)
                frames.append({
                    "frame_index": frame_index,
                    "color_compressed_sha256": bytes_sha256(color_bytes),
                    "depth_compressed_sha256": bytes_sha256(depth_bytes),
                    "semantic_member": semantic_doc,
                    "instance_member": instance_doc,
                    "native_valid_depth_pixels": registration.native_valid_pixels,
                    "registered_pixels_after_z_buffer": registration.registered_pixels,
                    "registered_color_fraction": registration.registered_pixels
                    / float(header.color_width * header.color_height),
                    "outputs": {
                        name: artifact_document(frame_dir / name, scene_out)
                        for name in sorted(p.name for p in frame_dir.iterdir())
                    },
                })
                del color, depth, registration, color_bytes, depth_bytes
                del semantic_bytes, instance_bytes

    bytes_written = sum(
        p.stat().st_size for p in scene_out.rglob("*") if p.is_file()
    )
    return {
        "scene_id": scene_id,
        "official_split": "train",
        "sens_bytes": paths.sens.stat().st_size,
        "sens_frames_declared_in_header": header.num_frames,
        "sens_frames_parseable": len(index),
        "sens_stream_truncated_or_corrupt": len(index) < header.num_frames,
        "frames_extracted": len(frames),
        "frames_skipped_undecodable": len(skipped),
        "skipped_frames": skipped,
        "metadata_notes": metadata_notes,
        "depth_shift_raw_units_per_meter": header.depth_shift,
        "sensor_name": header.sensor_name,
        "color_resolution": [header.color_width, header.color_height],
        "depth_resolution": [header.depth_width, header.depth_height],
        "output_bytes": bytes_written,
        "elapsed_seconds": round(time.time() - started, 3),
        "frames": frames,
    }

def peak_rss_mib() -> float:
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-count", type=int, default=24)
    parser.add_argument("--max-frames-per-scene", type=int, default=64)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--approval-record", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--scannet-root", type=Path,
                        default=Path("/path/to/datasets/downloads/ScanNet-official/scans"))
    parser.add_argument("--split-root", type=Path, default=REPOSITORY_ROOT / "configs/scannet_splits")
    parser.add_argument("--label-map", type=Path,
                        default=Path("/path/to/datasets/downloads/ScanNet-official/tasks/scannetv2-labels.combined.tsv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-free-bytes", type=int, default=20 * 1024**3)
    args = parser.parse_args()

    try:
        approval = load_approval(args.approval_record)
        split_documents, scene_to_split = load_split_manifests(args.split_root)
        train_scenes = [
            scene for scene, split in scene_to_split.items() if split == "train"
        ]
        train_scenes.sort()
        window = train_scenes[args.scene_offset : args.scene_offset + args.scene_count]
        if not window:
            raise SourceError("empty scene window")
        for scene_id in window:
            if scene_to_split.get(scene_id) != "train":
                raise SourceError(f"refusing non-train scene: {scene_id}")
            if SCENE_RE.fullmatch(scene_id) is None:
                raise SourceError(f"invalid scene id: {scene_id}")

        if args.output_dir.exists():
            raise SourceError(f"refusing to overwrite existing output: {args.output_dir}")
        usage = shutil.disk_usage(args.output_dir.parent)
        if usage.free < args.min_free_bytes:
            raise SourceError(
                f"insufficient free space: {usage.free} < {args.min_free_bytes}"
            )

        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.tmp-", dir=args.output_dir.parent
        ))
        scenes: list[dict[str, Any]] = []
        failed_scenes: list[dict[str, Any]] = []
        try:
            for position, scene_id in enumerate(window, start=1):
                print(f"[{position}/{len(window)}] {scene_id}", file=sys.stderr, flush=True)
                try:
                    scenes.append(extract_scene(
                        scene_id=scene_id,
                        scannet_root=args.scannet_root,
                        label_map=args.label_map,
                        max_frames=args.max_frames_per_scene,
                        scene_out=temp_dir / scene_id,
                    ))
                except SourceError as exc:
                    print(f"  SKIP {scene_id}: {exc}", file=sys.stderr, flush=True)
                    failed_scenes.append({"scene_id": scene_id, "reason": str(exc)})
                    shutil.rmtree(temp_dir / scene_id, ignore_errors=True)
                free_now = shutil.disk_usage(args.output_dir.parent).free
                if free_now < args.min_free_bytes:
                    raise SourceError(f"free space dropped below floor: {free_now}")

            if not scenes:
                raise SourceError("no scene in the window could be extracted")
            total_frames = sum(s["frames_extracted"] for s in scenes)
            total_bytes = sum(s["output_bytes"] for s in scenes)
            per_frame_bytes = total_bytes / max(total_frames, 1)
            per_frame_seconds = sum(s["elapsed_seconds"] for s in scenes) / max(total_frames, 1)
            full_frames = len(train_scenes) * args.max_frames_per_scene
            manifest = {
                "schema": SCHEMA,
                "protocol": PROTOCOL,
                "stage": "bounded_scaleout_validation",
                "verdict": "SCALEOUT_VALIDATED_TRAIN_SPLIT_ONLY",
                "adapter": {
                    "scaleout_path": str(Path(__file__).resolve()),
                    "scaleout_sha256": file_sha256(Path(__file__).resolve()),
                    "reader_path": str((REPOSITORY_ROOT / "tools/extract_scannet_train_sens.py").resolve()),
                    "reader_sha256": file_sha256(REPOSITORY_ROOT / "tools/extract_scannet_train_sens.py"),
                    "reader_implementation": "independent_minimal_scannet_sens_v4_reader",
                },
                "license_approval": approval,
                "execution": {
                    "cpu_only": True,
                    "gpu_required": False,
                    "network_calls": 0,
                    "scene_count": len(scenes),
                    "frame_count": total_frames,
                    "peak_rss_mib": peak_rss_mib(),
                    "wall_seconds": round(sum(s["elapsed_seconds"] for s in scenes), 1),
                },
                "use_policy": {
                    "official_split": "train",
                    "official_train_only": True,
                    "may_enter_held_out_v5_or_schema5": False,
                    "training_materialization_eligible": True,
                    "training_materialization_blockers": [],
                    "retained_provenance_blockers": approval["retained_blockers"],
                    "redistribution_eligible": False,
                    "val_split_note": "data/scannet_val_frame_bundles_v3 carries may_enter_training=false and is untouched by this extraction",
                },
                "official_splits": split_documents,
                "scene_window": {
                    "offset": args.scene_offset,
                    "requested_scene_count": args.scene_count,
                    "max_frames_per_scene": args.max_frames_per_scene,
                    "scene_ids": window,
                    "scenes_extracted": len(scenes),
                    "scenes_failed": len(failed_scenes),
                    "failed_scenes": failed_scenes,
                },
                "scaling_estimate": {
                    "measured_bytes_per_frame": round(per_frame_bytes, 1),
                    "measured_seconds_per_frame": round(per_frame_seconds, 4),
                    "official_train_scenes": len(train_scenes),
                    "full_run_frames_at_this_cap": full_frames,
                    "full_run_estimated_bytes": int(per_frame_bytes * full_frames),
                    "full_run_estimated_gib": round(per_frame_bytes * full_frames / 1024**3, 1),
                    "full_run_estimated_single_process_hours": round(
                        per_frame_seconds * full_frames / 3600.0, 2
                    ),
                },
                "scenes": scenes,
            }
            payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
            (temp_dir / "scaleout_manifest.json").write_bytes(payload)
            (temp_dir / "scaleout_manifest.sha256").write_text(
                f"{bytes_sha256(payload)}  scaleout_manifest.json\n", encoding="ascii"
            )
            if args.output_dir.exists():
                raise SourceError(
                    f"output directory appeared during extraction: {args.output_dir}"
                )
            os.rename(temp_dir, args.output_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    except (OSError, SourceError, struct.error, np.linalg.LinAlgError) as exc:
        print(json.dumps({"schema": SCHEMA, "verdict": "BLOCKED", "error": str(exc)}), file=sys.stderr)
        return 2

    print(json.dumps({
        "schema": SCHEMA,
        "verdict": manifest["verdict"],
        "scenes": len(scenes),
        "frames": manifest["execution"]["frame_count"],
        "peak_rss_mib": manifest["execution"]["peak_rss_mib"],
        "output_dir": str(args.output_dir.resolve()),
        "scaling_estimate": manifest["scaling_estimate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
