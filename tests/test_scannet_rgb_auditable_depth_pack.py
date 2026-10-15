from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import build_depth_v50_scannet_sensor as sensor
from tools import build_scannet_rgb_auditable_depth_pack as builder
from tools import check_scannet_rgb_auditable_depth_pack as checker
from tools.quick_semantic_audit import QuickAuditStore


def _record(tmp_path: Path, index: int, *, scene_id: str | None = None) -> dict:
    image_path = tmp_path / f"image-{index:02d}.png"
    image_path.write_bytes(b"fake-rgb-" + str(index).encode("ascii"))
    scene_id = scene_id or f"scene9{index:03d}_00"
    near_label = f"bucket [instance {index * 2 + 1}]"
    far_label = f"broom [instance {index * 2 + 2}]"
    return {
        "protocol_version": sensor.PROTOCOL,
        "label_version": sensor.LABEL_VERSION,
        "source_artifact_schema": sensor.SOURCE_ARTIFACT_SCHEMA,
        "_audit_only": True,
        "audit_contract": "scannet_sensor_pairwise",
        "task_interface": "oracle_grounded_pairwise_depth",
        "task_form": "depth",
        "depth_scope": "pairwise",
        "sample_id": f"source-{scene_id}-{index}",
        "scene_id": scene_id,
        "frame_index": index,
        "official_split": "val",
        "image_path": str(image_path.resolve()),
        "image_sha256": sensor.file_sha256(image_path),
        "payload": [
            {"label": near_label, "rank": 1, "relative_depth": 0.0},
            {"label": far_label, "rank": 2, "relative_depth": 1.0},
        ],
        "gsa_context": {
            "contract_version": "gsa-contract-v2",
            "groundings": [
                {"object_class": near_label, "bbox": [100, 300, 300, 800]},
                {"object_class": far_label, "bbox": [500, 100, 700, 500]},
            ],
            "primary": {"object_class": near_label, "bbox": [100, 300, 300, 800]},
            "spatial": {
                "relation_mode": "pairwise",
                "depth_order": [
                    {"label": near_label, "rank": 1, "relative_depth": 0.0},
                    {"label": far_label, "rank": 2, "relative_depth": 1.0},
                ],
            },
        },
        "sensor_measurements": [
            {"category": "bucket", "bbox_xyxy_px": [100, 300, 300, 800]},
            {"category": "broom", "bbox_xyxy_px": [500, 100, 700, 500]},
        ],
        "sensor_provenance": {"rgb": {"height": 1000, "width": 1000}},
        "verified_pair": {"depth_gap_m": 0.80},
        "visibility_features": {
            "quartile_gap_m": 0.50,
            "bbox_intersection_over_min_area": 0.0,
            "bbox_edge_distance_normalized": 0.20,
            "min_bbox_area_fraction": 0.05,
            "min_bbox_edge_clearance_px": 30,
            "min_mask_fill_ratio": 0.60,
            "min_valid_fraction": 0.90,
        },
    }


def _inventory() -> dict:
    return {
        "strict_eligible_records": builder.CANARY_TARGET,
        "strict_eligible_scenes": builder.CANARY_TARGET,
        "old_schema5_audit_binding": {
            "scene_ids": [f"scene0{i:03d}_00" for i in range(builder.FORMAL_TARGET)]
        },
        "frozen_sensor_canary_binding": {
            "scene_ids": [f"scene1{i:03d}_00" for i in range(10)]
        },
        "validated_source_exclusions": {"count": 0, "exclusions": []},
    }


def test_candidate_filter_rejects_same_category_and_overlap(tmp_path: Path) -> None:
    record = _record(tmp_path, 0)
    _, errors = builder.candidate_features(record)
    assert errors == []

    bad = copy.deepcopy(record)
    bad["sensor_measurements"][1]["category"] = "BUCKET"
    bad["visibility_features"]["bbox_intersection_over_min_area"] = 0.01
    _, errors = builder.candidate_features(bad)
    assert "same_normalized_category" in errors
    assert "bbox_overlap" in errors


def test_selection_is_scene_unique_and_deterministic(tmp_path: Path) -> None:
    first = _record(tmp_path, 1, scene_id="scene9001_00")
    second = _record(tmp_path, 2, scene_id="scene9001_00")
    second["verified_pair"]["depth_gap_m"] = 1.2
    third = _record(tmp_path, 3, scene_id="scene9002_00")
    chosen_a, stats_a = builder.select_scene_unique([first, second, third], 2)
    chosen_b, stats_b = builder.select_scene_unique([third, first, second], 2)
    assert [row["sample_id"] for row in chosen_a] == [row["sample_id"] for row in chosen_b]
    assert second["sample_id"] in {row["sample_id"] for row in chosen_a}
    assert stats_a == stats_b
    assert stats_a["selected_scenes"] == 2


def test_schema6_pack_loads_in_existing_quick_audit_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [_record(tmp_path, index) for index in range(builder.CANARY_TARGET)]
    output = tmp_path / "pack"
    manifest = builder._freeze(output, records, _inventory(), "canary")
    assert manifest["images"] == builder.CANARY_TARGET

    store = QuickAuditStore(output, "a")
    assert store.image_count == builder.CANARY_TARGET
    case = store.case(0)
    assert case["forms"][0]["audit_contract"] == builder.AUDIT_CONTRACT
    assert "sensor_measurements" not in case["forms"][0]
    assert "sensor_provenance" not in case["forms"][0]

    monkeypatch.setattr(sensor, "validate_record", lambda record: [])
    report = checker.validate_pack(output, rehash_source_artifacts=False)
    assert report["status"] == "PACK_VALID"
    assert report["unique_scenes"] == builder.CANARY_TARGET
    assert report["old_schema5_scene_overlap"] == 0


def test_formal_mode_fails_before_writing_when_capacity_is_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [_record(tmp_path, index) for index in range(builder.CANARY_TARGET)]
    inventory = _inventory()
    monkeypatch.setattr(builder, "_scan", lambda args: (records, inventory))
    output = tmp_path / "must-not-exist"
    assert builder.main(["--stage", "formal", "--output-dir", str(output)]) == 2
    assert not output.exists()
