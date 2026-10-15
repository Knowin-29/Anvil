#!/usr/bin/env python3
"""Build a scene-disjoint ScanNet depth calibration pack with visible RGB cues.

This builder is deliberately narrower than the v5.0 sensor-depth source.  It
keeps exact registered RGB-D truth, but admits a pair only when the RGB image
also contains a simple, inspectable perspective cue.  The output is audit-only
and can never qualify the formal 40-scene gate when the eligible pool is short.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_depth_v50_scannet_sensor as sensor


SCHEMA_VERSION = 6
PACK_PROTOCOL = "gsa-depth-v5.1-scannet-rgb-auditable-calibration-v1"
AUDIT_ROUND = "v5.1-scannet-rgb-auditable-canary-v1"
AUDIT_CONTRACT = "scannet_sensor_pairwise_rgb_auditable"
PROMPT_VERSION = "gsa-depth-v5.1-scannet-rgb-auditable-pairwise"
LABEL_VERSION = "gsa-label-v5.1-depth-scannet-rgb-auditable-pairwise"
CANARY_TARGET = 12
FORMAL_TARGET = 40
MAX_FRAMES_PER_SCENE = 64
SELECTION_SEED = "gsa-depth-v5.1-rgb-auditable-scene-unique-v1"

DEFAULT_OLD_PACK = ROOT / "data/semantic_audit_v5_scannet_sensor_depth"
DEFAULT_OUT = ROOT / "data/semantic_audit_v51_scannet_rgb_auditable_depth_canary"
DEFAULT_FORMAL_OUT = ROOT / "data/semantic_audit_v51_scannet_rgb_auditable_depth_formal"

STRICT_THRESHOLDS: dict[str, Any] = {
    "different_normalized_categories": True,
    "max_bbox_intersection_over_min_area": 0.0,
    "min_sensor_depth_gap_m": 0.40,
    "min_sensor_quartile_gap_m": 0.15,
    "min_bbox_area_fraction": 0.02,
    "min_mask_fill_ratio": 0.30,
    "min_valid_fraction": 0.50,
    "min_bbox_edge_clearance_px": 16,
    "max_bbox_edge_distance_normalized": 0.35,
    "min_near_minus_far_bbox_bottom_fraction": 0.05,
    "min_near_minus_far_bbox_center_y_fraction": 0.05,
    "bbox_overlap_allowed": False,
    "same_category_allowed": False,
}


class PackError(RuntimeError):
    """The requested pack cannot be frozen without violating its contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PackError(f"expected JSON object: {path}")
    return value


def manifest_payload_hash(manifest: dict[str, Any]) -> str:
    payload = copy.deepcopy(manifest)
    payload.pop("integrity", None)
    return sensor.canonical_hash(payload)


def load_old_audit_binding(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    manifest = _read_json(manifest_path)
    integrity = manifest.get("integrity") or {}
    scenes = [item.get("scene_id") for item in manifest.get("selected") or []]
    expected = integrity.get("manifest_payload_sha256")
    if (
        manifest.get("schema_version") != 5
        or manifest.get("audit_contract") != "scannet_sensor_pairwise"
        or manifest.get("images") != FORMAL_TARGET
        or len(scenes) != FORMAL_TARGET
        or len(set(scenes)) != FORMAL_TARGET
        or not all(isinstance(scene, str) for scene in scenes)
        or expected != manifest_payload_hash(manifest)
    ):
        raise PackError(f"old schema5 audit pack contract mismatch: {manifest_path}")
    return {
        "path": str(path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sensor.file_sha256(manifest_path),
        "manifest_payload_sha256": expected,
        "scene_count": len(scenes),
        "scene_ids": sorted(scenes),
    }


def _bbox_cues(record: dict[str, Any]) -> dict[str, float]:
    measurements = record["sensor_measurements"]
    rgb = record["sensor_provenance"]["rgb"]
    height = float(rgb["height"])
    if height <= 0 or len(measurements) != 2:
        raise ValueError("invalid RGB height or measurement count")
    near = measurements[0]["bbox_xyxy_px"]
    far = measurements[1]["bbox_xyxy_px"]
    near_center_y = (float(near[1]) + float(near[3])) / 2.0
    far_center_y = (float(far[1]) + float(far[3])) / 2.0
    return {
        "near_minus_far_bbox_bottom_fraction":
            (float(near[3]) - float(far[3])) / height,
        "near_minus_far_bbox_center_y_fraction":
            (near_center_y - far_center_y) / height,
    }


def candidate_features(record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return frozen evidence features and every strict rejection reason."""
    errors: list[str] = []
    try:
        measurements = record["sensor_measurements"]
        visibility = record["visibility_features"]
        pair = record["verified_pair"]
        if len(measurements) != 2:
            raise ValueError("measurement count")
        near_category = str(measurements[0]["category"]).strip().casefold()
        far_category = str(measurements[1]["category"]).strip().casefold()
        cues = _bbox_cues(record)
        features = {
            "near_category": near_category,
            "far_category": far_category,
            "sensor_depth_gap_m": float(pair["depth_gap_m"]),
            "sensor_quartile_gap_m": float(visibility["quartile_gap_m"]),
            "bbox_intersection_over_min_area": float(
                visibility["bbox_intersection_over_min_area"]
            ),
            "bbox_edge_distance_normalized": float(
                visibility["bbox_edge_distance_normalized"]
            ),
            "min_bbox_area_fraction": float(visibility["min_bbox_area_fraction"]),
            "min_bbox_edge_clearance_px": int(
                visibility["min_bbox_edge_clearance_px"]
            ),
            "min_mask_fill_ratio": float(visibility["min_mask_fill_ratio"]),
            "min_valid_fraction": float(visibility["min_valid_fraction"]),
            **cues,
        }
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {}, [f"malformed_record:{type(exc).__name__}"]

    if near_category == far_category:
        errors.append("same_normalized_category")
    if features["bbox_intersection_over_min_area"] > 0.0:
        errors.append("bbox_overlap")
    comparisons = (
        ("sensor_depth_gap", features["sensor_depth_gap_m"], 0.40, "min"),
        ("sensor_quartile_gap", features["sensor_quartile_gap_m"], 0.15, "min"),
        ("bbox_area", features["min_bbox_area_fraction"], 0.02, "min"),
        ("mask_fill", features["min_mask_fill_ratio"], 0.30, "min"),
        ("valid_fraction", features["min_valid_fraction"], 0.50, "min"),
        ("bbox_edge_clearance", features["min_bbox_edge_clearance_px"], 16, "min"),
        (
            "bbox_edge_distance",
            features["bbox_edge_distance_normalized"],
            0.35,
            "max",
        ),
        (
            "rgb_bottom_perspective_cue",
            features["near_minus_far_bbox_bottom_fraction"],
            0.05,
            "min",
        ),
        (
            "rgb_center_perspective_cue",
            features["near_minus_far_bbox_center_y_fraction"],
            0.05,
            "min",
        ),
    )
    for name, observed, threshold, direction in comparisons:
        if (direction == "min" and observed < threshold) or (
            direction == "max" and observed > threshold
        ):
            errors.append(name)
    return features, errors


def _scene_record_score(record: dict[str, Any]) -> tuple[float, ...]:
    features, errors = candidate_features(record)
    if errors:
        raise PackError("attempted to score an ineligible candidate")
    return (
        min(
            features["near_minus_far_bbox_bottom_fraction"],
            features["near_minus_far_bbox_center_y_fraction"],
        ),
        features["sensor_quartile_gap_m"],
        features["sensor_depth_gap_m"],
        features["min_bbox_area_fraction"],
        features["min_mask_fill_ratio"],
        features["min_valid_fraction"],
    )


def select_scene_unique(
    records: Iterable[dict[str, Any]], target: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pick one mechanically strongest frame per scene, then stable-hash scenes."""
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_scene.setdefault(str(record["scene_id"]), []).append(record)
    representatives = [
        max(rows, key=lambda row: (_scene_record_score(row), row["sample_id"]))
        for _, rows in sorted(by_scene.items())
    ]
    representatives.sort(
        key=lambda row: sensor.canonical_hash(
            [SELECTION_SEED, row["scene_id"], row["image_sha256"]]
        )
    )
    selected = representatives[:target]
    stats = {
        "eligible_records": sum(len(rows) for rows in by_scene.values()),
        "eligible_scenes": len(by_scene),
        "selected_records": len(selected),
        "selected_scenes": len({row["scene_id"] for row in selected}),
        "selected_images": len({row["image_sha256"] for row in selected}),
    }
    return selected, stats


def _safe_label(record: dict[str, Any], sample_id: str, features: dict[str, Any]) -> dict[str, Any]:
    """Project only reviewer-visible evidence; metric truth stays in provenance/."""
    return {
        "_audit_only": True,
        "_audit_reason": "rgb_auditable_sensor_depth_calibration_candidate",
        "audit_schema_version": SCHEMA_VERSION,
        "audit_contract": AUDIT_CONTRACT,
        "audit_round": AUDIT_ROUND,
        "protocol_version": PACK_PROTOCOL,
        "prompt_version": PROMPT_VERSION,
        "label_version": LABEL_VERSION,
        "sample_id": sample_id,
        "source_sample_id": record["sample_id"],
        "source_record_sha256": sensor.canonical_hash(record),
        "scene_id": record["scene_id"],
        "frame_index": record["frame_index"],
        "official_split": "val",
        "scene_type": "indoor_scan",
        "source_bucket": "scannet_sensor_rgb_auditable",
        "task_form": "depth",
        "dimension": "S_spatial",
        "pipeline": "S_spatial",
        "depth_scope": "pairwise",
        "task_interface": "oracle_grounded_pairwise_depth",
        "object_class": record["payload"][0]["label"],
        "image_path": record["image_path"],
        "image_sha256": record["image_sha256"],
        "payload": copy.deepcopy(record["payload"]),
        "gsa_context": copy.deepcopy(record["gsa_context"]),
        "model_interface": {
            "intended_task": "oracle_grounded_pairwise_depth",
            "requires_explicit_pair_groundings": True,
            "instance_ids_are_internal_identity_only": True,
        },
        "semantic_gate": {
            "protocol": PACK_PROTOCOL,
            "passed_mechanical_candidate_filter": True,
            "thresholds": copy.deepcopy(STRICT_THRESHOLDS),
            "rgb_auditability_features": copy.deepcopy(features),
            "sensor_truth_hidden_from_audit_ui": True,
            "human_semantic_confirmation_required": True,
        },
        "training_materialization": {
            "eligible": False,
            "allowed_use": "calibration_candidate_audit_only",
            "reason": "official_val_held_out_and_formal_40_scene_gate_unmet",
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _scan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old = load_old_audit_binding(args.old_pack)
    split_manifests, scene_to_split = sensor.load_split_manifests(args.split_root)
    exclusions, exclusion_binding = sensor.load_source_exclusions(
        args.source_exclusions, args.scannet_root, scene_to_split
    )
    scenes = sensor.discover_scenes(args.openeqa_root, args.scannet_root, scene_to_split)
    canary = sensor.load_canary_scene_manifest(
        args.canary_scenes, scenes, scene_to_split, exclusions
    )
    excluded_scenes = set(old["scene_ids"]) | set(canary["scene_ids"]) | set(exclusions)
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
            max_frames=MAX_FRAMES_PER_SCENE,
            max_records=MAX_FRAMES_PER_SCENE,
        )
        system_errors.extend(errors)
        sensor_candidates += len(rows)
        for row in rows:
            source_errors = sensor.validate_record(row, split_manifests)
            if source_errors:
                rejected.update(f"source:{error}" for error in source_errors)
                continue
            _, reasons = candidate_features(row)
            if reasons:
                rejected.update(reasons)
            else:
                eligible.append(row)
    if system_errors:
        raise PackError(
            f"source scan produced {len(system_errors)} system errors; first={system_errors[0]}"
        )
    inventory = {
        "official_split": "val",
        "source_ready_scene_count_after_exclusions": len(fresh),
        "source_ready_scene_ids_after_exclusions": sorted(scene.scene_id for scene in fresh),
        "sensor_candidate_records": sensor_candidates,
        "strict_eligible_records": len(eligible),
        "strict_eligible_scenes": len({row["scene_id"] for row in eligible}),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "old_schema5_audit_binding": old,
        "frozen_sensor_canary_binding": canary,
        "validated_source_exclusions": exclusion_binding,
        "excluded_scene_count": len(excluded_scenes),
        "excluded_scene_ids": sorted(excluded_scenes),
        "strict_thresholds": copy.deepcopy(STRICT_THRESHOLDS),
    }
    return eligible, inventory


def _freeze(
    output: Path,
    selected: list[dict[str, Any]],
    inventory: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    target = CANARY_TARGET if stage == "canary" else FORMAL_TARGET
    if len(selected) != target:
        raise PackError(f"selected {len(selected)} records, expected exactly {target}")
    if output.exists():
        raise PackError(f"refusing to overwrite existing output: {output}")

    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        labels: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        selected_manifest: list[dict[str, Any]] = []
        for record in selected:
            features, reasons = candidate_features(record)
            if reasons:
                raise PackError(f"selected record no longer eligible: {record['sample_id']}")
            sample_id = "v51_rgb_depth_" + sensor.canonical_hash(
                [record["sample_id"], record["image_sha256"], PACK_PROTOCOL]
            )[:20]
            label = _safe_label(record, sample_id, features)
            label_hash = sensor.canonical_hash(label)
            labels.append(label)
            provenance.append(
                {
                    "sample_id": sample_id,
                    "source_sample_id": record["sample_id"],
                    "source_record_sha256": sensor.canonical_hash(record),
                    "source_record": record,
                }
            )
            selected_manifest.append(
                {
                    "scene_id": record["scene_id"],
                    "frame_index": record["frame_index"],
                    "official_split": "val",
                    "image_path": record["image_path"],
                    "image_sha256": record["image_sha256"],
                    "sample_ids": {"depth": sample_id},
                    "label_hashes": {"depth": label_hash},
                    "source_sample_id": record["sample_id"],
                    "source_record_sha256": sensor.canonical_hash(record),
                    "rgb_auditability_features": features,
                    "rgb_auditability_features_sha256": sensor.canonical_hash(features),
                }
            )

        provenance_path = temp / "provenance/source_records.jsonl"
        _write_jsonl(provenance_path, provenance)
        for annotator in ("a", "b"):
            label_path = temp / f"annotator_{annotator}/labels/S_spatial/depth.jsonl"
            _write_jsonl(label_path, labels)
            (temp / f"annotator_{annotator}/annotations").mkdir(parents=True)

        scene_ids = {item["scene_id"] for item in selected_manifest}
        image_ids = {item["image_sha256"] for item in selected_manifest}
        old_scenes = set(inventory["old_schema5_audit_binding"]["scene_ids"])
        canary_scenes = set(inventory["frozen_sensor_canary_binding"]["scene_ids"])
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PACK_PROTOCOL,
            "audit_round": AUDIT_ROUND,
            "audit_contract": AUDIT_CONTRACT,
            "stage": stage,
            "forms": ["depth"],
            "images": len(selected_manifest),
            "expected_decisions_per_annotator": len(selected_manifest),
            "selected": selected_manifest,
            "selection": {
                "rule": "strict_filter_best_frame_per_scene_then_stable_hash",
                "seed": SELECTION_SEED,
                "target": target,
                "unique_scene_required": True,
                "unique_image_required": True,
                "selected_unique_scenes": len(scene_ids),
                "selected_unique_images": len(image_ids),
                "strict_eligible_records": inventory["strict_eligible_records"],
                "strict_eligible_scenes": inventory["strict_eligible_scenes"],
                "thresholds": copy.deepcopy(STRICT_THRESHOLDS),
            },
            "exclusion_checks": {
                "old_schema5_scene_overlap": sorted(scene_ids & old_scenes),
                "frozen_canary_scene_overlap": sorted(scene_ids & canary_scenes),
                "old_schema5_and_canary_overlap_count": len(
                    scene_ids & (old_scenes | canary_scenes)
                ),
            },
            "qualification": {
                "formal_target_scene_unique": FORMAL_TARGET,
                "formal_capacity_available": inventory["strict_eligible_scenes"],
                "formal_qualification_eligible": stage == "formal",
                "canary_can_establish_formal_pass": False,
                "allowed_use": "calibration_candidate_audit_only",
            },
            "human_gate": {
                "independent_annotators": 2,
                "minimum_raw_agreement": 0.80,
                "minimum_resolved_correctness": 0.80,
                "adjudicate_all_disagreements": True,
                "negative_vote_requires_error_code_and_comment": True,
            },
            "provenance": {
                "source_records_path": "provenance/source_records.jsonl",
                "source_records_sha256": sensor.file_sha256(provenance_path),
                "old_schema5_audit_binding": inventory["old_schema5_audit_binding"],
                "frozen_sensor_canary_binding": inventory[
                    "frozen_sensor_canary_binding"
                ],
                "validated_source_exclusions": inventory[
                    "validated_source_exclusions"
                ],
                "builder_path": str(Path(__file__).resolve()),
                "builder_sha256": sensor.file_sha256(Path(__file__)),
                "source_builder_path": str(Path(sensor.__file__).resolve()),
                "source_builder_sha256": sensor.file_sha256(Path(sensor.__file__)),
            },
            "inventory": inventory,
            "training_materialization": {
                "eligible": False,
                "allowed_use": "calibration_candidate_audit_only",
            },
        }
        manifest["integrity"] = {
            "selected_sha256": sensor.canonical_hash(selected_manifest),
            "manifest_payload_sha256": manifest_payload_hash(manifest),
        }
        _write_json(temp / "manifest.json", manifest)
        temp.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inventory", "canary", "formal"), default="inventory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--old-pack", type=Path, default=DEFAULT_OLD_PACK)
    parser.add_argument("--openeqa-root", type=Path, default=sensor.DEFAULT_OPENEQA_ROOT)
    parser.add_argument("--scannet-root", type=Path, default=sensor.DEFAULT_SCANNET_ROOT)
    parser.add_argument("--label-map", type=Path, default=sensor.DEFAULT_LABEL_MAP)
    parser.add_argument("--split-root", type=Path, default=sensor.DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--source-exclusions", type=Path, default=sensor.DEFAULT_SOURCE_EXCLUSIONS
    )
    parser.add_argument("--canary-scenes", type=Path, default=sensor.DEFAULT_CANARY_SCENES)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = DEFAULT_FORMAL_OUT if args.stage == "formal" else DEFAULT_OUT
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        eligible, inventory = _scan(args)
        target = CANARY_TARGET if args.stage == "canary" else FORMAL_TARGET
        selected, stats = select_scene_unique(eligible, target)
        report = {"stage": args.stage, "selection": stats, "inventory": inventory}
        if args.stage == "inventory":
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if inventory["strict_eligible_scenes"] < target:
            raise PackError(
                f"CAPACITY_BLOCKED_{target}: only "
                f"{inventory['strict_eligible_scenes']} strict scene-unique candidates; "
                "no output written"
            )
        manifest = _freeze(args.output_dir, selected, inventory, args.stage)
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir.resolve()),
                    "manifest_sha256": sensor.file_sha256(args.output_dir / "manifest.json"),
                    "manifest_payload_sha256": manifest["integrity"][
                        "manifest_payload_sha256"
                    ],
                    "images": manifest["images"],
                    "unique_scenes": manifest["selection"]["selected_unique_scenes"],
                    "unique_images": manifest["selection"]["selected_unique_images"],
                },
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
