#!/usr/bin/env python3
"""Build the formal v5.1 RGB-auditable depth pack after a reviewed canary.

The canary builder is an immutable audit artifact: its file hash is recorded in
the reviewed pack.  This formal-only wrapper therefore reuses that frozen
selection implementation without modifying it, and adds the missing reviewed-
canary exclusion and provenance gate.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_depth_v50_scannet_sensor as sensor
from tools import build_scannet_rgb_auditable_depth_pack as canary_builder
from tools import check_scannet_rgb_auditable_depth_pack as checker


DEFAULT_OUTPUT = canary_builder.DEFAULT_FORMAL_OUT
DEFAULT_REVIEWED_CANARY_PACK = canary_builder.DEFAULT_OUT
DEFAULT_REVIEWED_CANARY_REPORT = (
    ROOT / "artifacts/audits/scannet_rgb_depth_v51_calibration.json"
)
PASS_STATUS = checker.CANARY_PASS_STATUS


def load_additional_frame_root(path: Path) -> dict[str, Any]:
    try:
        return checker.load_additional_frame_root(path)
    except checker.CheckError as exc:
        raise canary_builder.PackError(str(exc)) from exc


def load_reviewed_canary_binding(
    pack: Path,
    report_path: Path,
    *,
    rehash_source_artifacts: bool = False,
) -> dict[str, Any]:
    try:
        return checker.load_reviewed_canary_binding(
            pack,
            report_path,
            rehash_source_artifacts=rehash_source_artifacts,
        )
    except checker.CheckError as exc:
        raise canary_builder.PackError(str(exc)) from exc


def _scan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reviewed = load_reviewed_canary_binding(
        args.reviewed_canary_pack,
        args.reviewed_canary_report,
        rehash_source_artifacts=False,
    )
    old = canary_builder.load_old_audit_binding(args.old_pack)
    split_manifests, scene_to_split = sensor.load_split_manifests(args.split_root)
    exclusions, exclusion_binding = sensor.load_source_exclusions(
        args.source_exclusions, args.scannet_root, scene_to_split
    )
    scenes = sensor.discover_scenes(args.openeqa_root, args.scannet_root, scene_to_split)
    seen_scene_ids = {scene.scene_id for scene in scenes}
    additional_bindings: list[dict[str, Any]] = []
    for frame_root in args.additional_frame_root:
        binding = load_additional_frame_root(frame_root)
        additional = sensor.discover_scenes(
            frame_root, args.scannet_root, scene_to_split
        )
        discovered_ids = {scene.scene_id for scene in additional}
        if discovered_ids != set(binding["scene_ids"]):
            raise canary_builder.PackError(
                f"additional frame-root discovery drift: {frame_root}"
            )
        overlap = seen_scene_ids & discovered_ids
        if overlap:
            raise canary_builder.PackError(
                f"duplicate scenes across frame roots: {sorted(overlap)[:5]}"
            )
        scenes.extend(additional)
        seen_scene_ids |= discovered_ids
        additional_bindings.append(binding)
    sensor_canary = sensor.load_canary_scene_manifest(
        args.canary_scenes, scenes, scene_to_split, exclusions
    )
    reviewed_scenes = set(reviewed["scene_ids"])
    excluded_scenes = (
        set(old["scene_ids"])
        | set(sensor_canary["scene_ids"])
        | set(exclusions)
        | reviewed_scenes
    )
    fresh = [
        scene
        for scene in scenes
        if scene.official_split == "val"
        and scene.scene_id not in excluded_scenes
        and sensor.source_ready(scene)
    ]
    label_maps = sensor.load_label_map(args.label_map)
    eligible: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    sensor_candidates = 0
    system_errors: list[str] = []
    for scene in fresh:
        rows, _, _, errors = sensor.process_scene(
            scene,
            label_maps,
            split_manifests,
            max_frames=canary_builder.MAX_FRAMES_PER_SCENE,
            max_records=canary_builder.MAX_FRAMES_PER_SCENE,
        )
        system_errors.extend(errors)
        sensor_candidates += len(rows)
        for row in rows:
            source_errors = sensor.validate_record(row, split_manifests)
            if source_errors:
                rejected.update(f"source:{error}" for error in source_errors)
                continue
            _, reasons = canary_builder.candidate_features(row)
            if reasons:
                rejected.update(reasons)
            else:
                eligible.append(row)
    if system_errors:
        raise canary_builder.PackError(
            f"source scan produced {len(system_errors)} system errors; "
            f"first={system_errors[0]}"
        )
    inventory = {
        "official_split": "val",
        "source_ready_scene_count_after_exclusions": len(fresh),
        "source_ready_scene_ids_after_exclusions": sorted(
            scene.scene_id for scene in fresh
        ),
        "sensor_candidate_records": sensor_candidates,
        "strict_eligible_records": len(eligible),
        "strict_eligible_scenes": len({row["scene_id"] for row in eligible}),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "old_schema5_audit_binding": old,
        "frozen_sensor_canary_binding": sensor_canary,
        "reviewed_rgb_canary_binding": reviewed,
        "additional_frame_root_bindings": additional_bindings,
        "validated_source_exclusions": exclusion_binding,
        "excluded_scene_count": len(excluded_scenes),
        "excluded_scene_ids": sorted(excluded_scenes),
        "strict_thresholds": dict(canary_builder.STRICT_THRESHOLDS),
    }
    return eligible, inventory


def _freeze_formal(
    output: Path,
    selected: list[dict[str, Any]],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise canary_builder.PackError(f"refusing to overwrite existing output: {output}")
    binding = inventory.get("reviewed_rgb_canary_binding")
    if not isinstance(binding, dict):
        raise canary_builder.PackError("formal pack lacks reviewed RGB canary binding")
    selected_scenes = {row["scene_id"] for row in selected}
    reviewed_scenes = set(binding.get("scene_ids") or [])
    overlap = sorted(selected_scenes & reviewed_scenes)
    if overlap:
        raise canary_builder.PackError(
            f"formal selection overlaps reviewed RGB canary scenes: {overlap}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.formal.", dir=output.parent))
    staged_pack = staging_root / "pack"
    try:
        manifest = canary_builder._freeze(
            staged_pack, selected, inventory, "formal"
        )
        manifest["exclusion_checks"]["reviewed_rgb_canary_scene_overlap"] = []
        manifest["exclusion_checks"]["all_prior_audit_scene_overlap_count"] = 0
        manifest["qualification"]["formal_qualification_eligible"] = False
        manifest["qualification"]["formal_pack_ready_for_human_audit"] = True
        manifest["qualification"]["reviewed_rgb_canary_calibration_passed"] = True
        manifest["provenance"]["reviewed_rgb_canary_binding"] = binding
        manifest["provenance"]["additional_frame_root_bindings"] = inventory.get(
            "additional_frame_root_bindings", []
        )
        manifest["provenance"]["formal_builder_path"] = str(Path(__file__).resolve())
        manifest["provenance"]["formal_builder_sha256"] = sensor.file_sha256(
            Path(__file__)
        )
        manifest["integrity"] = {
            "selected_sha256": sensor.canonical_hash(manifest["selected"]),
            "manifest_payload_sha256": canary_builder.manifest_payload_hash(manifest),
        }
        canary_builder._write_json(staged_pack / "manifest.json", manifest)
        staged_pack.rename(output)
        return manifest
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inventory", "formal"), default="inventory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reviewed-canary-pack", type=Path, default=DEFAULT_REVIEWED_CANARY_PACK
    )
    parser.add_argument(
        "--reviewed-canary-report", type=Path, default=DEFAULT_REVIEWED_CANARY_REPORT
    )
    parser.add_argument("--old-pack", type=Path, default=canary_builder.DEFAULT_OLD_PACK)
    parser.add_argument("--openeqa-root", type=Path, default=sensor.DEFAULT_OPENEQA_ROOT)
    parser.add_argument(
        "--additional-frame-root",
        action="append",
        type=Path,
        default=[],
        help="append a manifest-bound official-val frame-bundle root",
    )
    parser.add_argument("--scannet-root", type=Path, default=sensor.DEFAULT_SCANNET_ROOT)
    parser.add_argument("--label-map", type=Path, default=sensor.DEFAULT_LABEL_MAP)
    parser.add_argument("--split-root", type=Path, default=sensor.DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--source-exclusions", type=Path, default=sensor.DEFAULT_SOURCE_EXCLUSIONS
    )
    parser.add_argument("--canary-scenes", type=Path, default=sensor.DEFAULT_CANARY_SCENES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        eligible, inventory = _scan(args)
        selected, stats = canary_builder.select_scene_unique(
            eligible, canary_builder.FORMAL_TARGET
        )
        if args.stage == "inventory":
            print(
                json.dumps(
                    {"stage": "inventory", "selection": stats, "inventory": inventory},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if inventory["strict_eligible_scenes"] < canary_builder.FORMAL_TARGET:
            raise canary_builder.PackError(
                f"CAPACITY_BLOCKED_{canary_builder.FORMAL_TARGET}: only "
                f"{inventory['strict_eligible_scenes']} strict scene-unique candidates "
                "after reviewed-canary exclusion; no output written"
            )
        manifest = _freeze_formal(args.output_dir, selected, inventory)
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir.resolve()),
                    "manifest_sha256": sensor.file_sha256(
                        args.output_dir / "manifest.json"
                    ),
                    "manifest_payload_sha256": manifest["integrity"][
                        "manifest_payload_sha256"
                    ],
                    "images": manifest["images"],
                    "unique_scenes": manifest["selection"]["selected_unique_scenes"],
                    "reviewed_canary_scene_overlap": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (canary_builder.PackError, checker.CheckError, sensor.SourceError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
