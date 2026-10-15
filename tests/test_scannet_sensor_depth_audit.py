from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval import compile_semantic_audit as compiler  # noqa: E402
from eval import compile_semantic_composite_gate as composite  # noqa: E402
from tools import build_depth_v50_scannet_sensor as sensor  # noqa: E402
from tools.quick_semantic_audit import QuickAuditStore  # noqa: E402
from semantic_adjudication import (  # noqa: E402
    AdjudicationStore,
    build_pack as build_adjudication_pack,
)
from trains import materialize_semantic_audit_composite as materializer  # noqa: E402
from trains.flywheel import build_scannet_sensor_depth_audit_pack as packer  # noqa: E402
from trains.flywheel.contracts import context_for, grounding  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Schema5Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.source_root = root / "source"
        self.labels = self.source_root / "labels/S_spatial/depth.jsonl"
        self.report = self.source_root / "generation_report.json"
        historical_report = (
            ROOT / "data/depth_v50_scannet_sensor/canary_v4/generation_report.json"
        )
        self.canary_report = json.loads(historical_report.read_text())
        self.source_exclusions = copy.deepcopy(
            self.canary_report["validated_source_exclusions"]
        )
        self.canary_rows = [
            self._record(1000 + index, scene_id, 0)
            for index, scene_id in enumerate(
                self.canary_report["frozen_canary_scenes"]["scene_ids"]
            )
        ]
        canary_root = root / "canary"
        canary_labels = canary_root / "labels/S_spatial/depth.jsonl"
        canary_labels.parent.mkdir(parents=True)
        canary_labels.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.canary_rows),
            encoding="utf-8",
        )
        self.canary_report.update({
            "builder_sha256": sha256(ROOT / "tools/build_depth_v50_scannet_sensor.py"),
            "validated_source_exclusions": self.source_exclusions,
            "excluded_corrupt_source_scene_ids": ["scene0015_00"],
            "labels_sha256": sha256(canary_labels),
            "records_sha256": packer.content_hash(self.canary_rows),
            "accepted_scene_ids": sorted(
                row["scene_id"] for row in self.canary_rows
            ),
            "accepted_by_gap_bucket": dict(sorted(Counter(
                row["visibility_features"]["gap_bucket"]
                for row in self.canary_rows
            ).items())),
        })
        canary_report = canary_root / "generation_report.json"
        canary_report.write_text(json.dumps(self.canary_report), encoding="utf-8")
        self.canary_binding = {
            "report_path": str(canary_report.resolve()),
            "report_sha256": sha256(canary_report),
            "labels_path": str(canary_labels.resolve()),
            "labels_sha256": sha256(canary_labels),
            "scene_ids": self.canary_report["accepted_scene_ids"],
            "image_sha256s": [row["image_sha256"] for row in self.canary_rows],
            "frozen_canary_scenes": copy.deepcopy(
                self.canary_report["frozen_canary_scenes"]
            ),
            "source_artifact_replay": {
                "records": sensor.CANARY_TARGET,
                "files_rehashed": 11 * sensor.CANARY_TARGET,
                "zip_members_rehashed": 2 * sensor.CANARY_TARGET,
                "errors": 0,
            },
        }
        val_scenes = [
            line.strip()
            for line in (ROOT / "configs/scannet_splits/scannetv2_val.txt")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        excluded = set(self.canary_binding["scene_ids"]) | {"scene0015_00"}
        self.scene_ids = [scene for scene in val_scenes if scene not in excluded][:65]
        assert len(self.scene_ids) == 65
        self.rows = [self._record(index, scene, 0) for index, scene in enumerate(self.scene_ids)]
        self.rows.extend(
            self._record(65 + index, scene, 1)
            for index, scene in enumerate(self.scene_ids[:55])
        )
        self.labels.parent.mkdir(parents=True, exist_ok=True)
        self.labels.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows),
            encoding="utf-8",
        )
        self._write_report()

    def _make_source_exclusions(self) -> dict:
        artifact = self.root / "excluded" / "scene0015_00_2d-label-filt.zip"
        write(artifact, b"known-corrupt-fixture")
        document = {
            "dataset": "ScanNet v2",
            "exclusions": [{
                "artifact": artifact.name,
                "evidence": "unzip_test_failed",
                "official_split": "val",
                "reason": "corrupt_zip_archive",
                "scene_id": "scene0015_00",
                "sha256": sha256(artifact),
            }],
            "schema": "gsa-scannet-source-exclusions-v1",
        }
        manifest = self.root / "excluded" / "source_exclusions.json"
        manifest.write_text(json.dumps(document), encoding="utf-8")
        return {
            "schema": document["schema"],
            "dataset": document["dataset"],
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": sha256(manifest),
            "count": 1,
            "exclusions": [{
                **document["exclusions"][0],
                "artifact_path": str(artifact.resolve()),
            }],
        }

    @property
    def split_documents(self) -> dict:
        return copy.deepcopy(
            self.canary_report["source_contract"]["official_split_manifests"]
        )

    def _record(self, index: int, scene_id: str, frame_index: int) -> dict:
        artifact = self.root / "artifacts" / f"row-{index:03d}"
        rgb = write(artifact / "rgb.png", f"rgb-{index}".encode())
        rgb.update({"width": 1000, "height": 1000})
        depth = write(artifact / "depth.png", f"depth-{index}".encode())
        pose = write(artifact / "pose.txt", b"pose")
        metadata = write(artifact / "scene.txt", b"metadata")
        aggregation = write(artifact / "aggregation.json", b"{}")
        calibration = {
            key: write(artifact / f"{key}.txt", key.encode())
            for key in (
                "intrinsic_color", "intrinsic_depth",
                "extrinsic_color", "extrinsic_depth",
            )
        }
        instance_zip = artifact / "instance.zip"
        semantic_zip = artifact / "semantic.zip"
        instance_member = f"instance-filt/{frame_index}.png"
        semantic_member = f"label-filt/{frame_index}.png"
        instance_bytes = f"instance-{index}".encode()
        semantic_bytes = f"semantic-{index}".encode()
        with ZipFile(instance_zip, "w") as archive:
            archive.writestr(instance_member, instance_bytes)
        with ZipFile(semantic_zip, "w") as archive:
            archive.writestr(semantic_member, semantic_bytes)

        near_label = f"chair-{index % 7} [instance 1]"
        far_label = f"table-{index % 5} [instance 2]"
        payload = [
            {"label": near_label, "rank": 1, "relative_depth": 0.0},
            {"label": far_label, "rank": 2, "relative_depth": 1.0},
        ]
        if index % 4 == 0:
            boxes = ([100, 100, 430, 430], [350, 300, 680, 650])
            overlap, edge = 0.12, 0.0
        elif index % 4 == 1:
            boxes = ([80, 80, 320, 320], [300, 300, 560, 560])
            overlap, edge = 0.02, 0.0
        elif index % 4 == 2:
            boxes = ([80, 80, 300, 300], [320, 320, 540, 540])
            overlap, edge = 0.0, 0.03
        else:
            boxes = ([50, 50, 260, 260], [650, 650, 900, 900])
            overlap, edge = 0.0, 0.45
        measurements = [
            {
                "label": near_label,
                "bbox_2d": list(boxes[0]),
                "mask_pixels": 2000,
                "valid_samples": 400,
                "valid_fraction": 0.7,
                "depth_median_m": 1.0,
                "depth_iqr_m": 0.1,
                "depth_mad_m": 0.04,
                "robust_se_m": 0.01,
            },
            {
                "label": far_label,
                "bbox_2d": list(boxes[1]),
                "mask_pixels": 1800,
                "valid_samples": 380,
                "valid_fraction": 0.65,
                "depth_median_m": 1.35 + 0.4 * (index % 3),
                "depth_iqr_m": 0.12,
                "depth_mad_m": 0.05,
                "robust_se_m": 0.01,
            },
        ]
        for measurement, bbox in zip(measurements, boxes):
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            measurement.update({
                "bbox_xyxy_px": list(bbox),
                "bbox_area_fraction": area / 1_000_000.0,
                "bbox_edge_clearance_px": min(
                    bbox[0], bbox[1], 1000 - bbox[2], 1000 - bbox[3]
                ),
                "mask_fill_ratio": measurement["mask_pixels"] / float(area),
                "aggregation_label": measurement["label"].split(" [instance", 1)[0],
                "category": measurement["label"].split(" [instance", 1)[0],
            })
        groundings = [
            grounding(near_label, list(boxes[0])),
            grounding(far_label, list(boxes[1])),
        ]
        gap_buckets = (
            "0.25_to_0.50m", "0.50_to_1.00m", "at_least_1.00m"
        )
        gap_values = (0.35, 0.70, 1.20)
        gap = gap_values[index % 3]
        split = self.split_documents["val"]
        return {
            "sample_id": f"source-v50-{index:03d}",
            "source_artifact_schema": sensor.SOURCE_ARTIFACT_SCHEMA,
            "task_form": "depth",
            "depth_scope": "pairwise",
            "protocol_version": sensor.PROTOCOL,
            "label_version": sensor.LABEL_VERSION,
            "audit_round": sensor.AUDIT_ROUND,
            "_audit_only": True,
            "audit_contract": "scannet_sensor_pairwise",
            "task_interface": "oracle_grounded_pairwise_depth",
            "scene_type": "indoor_scan",
            "source_bucket": "scannet_sensor",
            "scene_id": scene_id,
            "official_split": "val",
            "frame_index": frame_index,
            "image_path": rgb["path"],
            "image_sha256": rgb["sha256"],
            "payload": payload,
            "gsa_context": context_for(
                "depth", groundings,
                {"relation_mode": "pairwise", "depth_order": payload},
            ),
            "sensor_measurements": measurements,
            "visibility_features": {
                "quartile_separated": True,
                "quartile_gap_m": 0.05,
                "gap_bucket": gap_buckets[index % 3],
                "bbox_intersection_over_min_area": overlap,
                "bbox_edge_distance_normalized": edge,
                "min_bbox_area_fraction": min(
                    item["bbox_area_fraction"] for item in measurements
                ),
                "max_bbox_area_fraction": max(
                    item["bbox_area_fraction"] for item in measurements
                ),
                "min_bbox_edge_clearance_px": min(
                    item["bbox_edge_clearance_px"] for item in measurements
                ),
                "min_mask_fill_ratio": min(
                    item["mask_fill_ratio"] for item in measurements
                ),
                "min_valid_fraction": 0.65,
                "pair_selection_rule": "visual_auditability_then_metric_margin_v1",
            },
            "spatial": {
                "relation_mode": "pairwise",
                "depth_order": [near_label, far_label],
            },
            "verified_pair": {
                "nearer": near_label,
                "farther": far_label,
                "depth_gap_m": gap,
                "uncertainty_3se_m": 0.05,
                "required_gap_m": 0.25,
                "margin_m": gap - 0.25,
                "quartile_separated": True,
                "quartile_gap_m": 0.05,
                "verification": "registered_metric_sensor_depth",
            },
            "sensor_provenance": {
                "source_dataset": "ScanNet v2",
                "derived_frame_bundle": "OpenEQA scannet-v0",
                "scene_id": scene_id,
                "frame_index": frame_index,
                "split_membership": split,
                "rgb": rgb,
                "depth": depth,
                "pose": pose,
                "calibration": calibration,
                "instance_projection": {
                    "zip_path": str(instance_zip.resolve()),
                    "zip_sha256": sha256(instance_zip),
                    "member": instance_member,
                    "sha256": hashlib.sha256(instance_bytes).hexdigest(),
                },
                "semantic_projection": {
                    "zip_path": str(semantic_zip.resolve()),
                    "zip_sha256": sha256(semantic_zip),
                    "member": semantic_member,
                    "sha256": hashlib.sha256(semantic_bytes).hexdigest(),
                    "encoding": "scannet_raw_label_id",
                },
                "scene_metadata": metadata,
                "aggregation": {
                    **aggregation,
                    "instance_id_rule": "objectId_plus_1",
                },
            },
            "semantic_gate": {
                "depth_v50_scannet_sensor": {
                    "passed": True,
                    "sensor_only": True,
                    "vlm_used": False,
                    "monocular_depth_used": False,
                    "visual_quality_contract": sensor.visual_quality_contract(),
                    "pair_selection_rule": "visual_auditability_then_metric_margin_v1",
                },
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

    def _write_report(self) -> None:
        contract = copy.deepcopy(self.canary_report["source_contract"])
        report = {
            "stage": "full",
            "protocol": sensor.PROTOCOL,
            "label_version": sensor.LABEL_VERSION,
            "audit_round": sensor.AUDIT_ROUND,
            "source_artifact_schema": sensor.SOURCE_ARTIFACT_SCHEMA,
            "builder_sha256": sha256(ROOT / "tools/build_depth_v50_scannet_sensor.py"),
            "selection_rule": packer.FULL_SELECTION_RULE,
            "generation_parameters": packer.expected_generation_parameters(
                target=len(self.rows),
                max_records_per_scene=sensor.MAX_RECORDS_PER_SCENE,
                minimum_scenes=sensor.MIN_FULL_SCENES,
            ),
            "target": len(self.rows),
            "accepted": len(self.rows),
            "accepted_scenes": len(self.scene_ids),
            "accepted_scene_ids": sorted(self.scene_ids),
            "attempted_scenes": len(self.scene_ids),
            "system_errors": {"total": 0, "items": []},
            "accepted_by_official_split": {"val": len(self.rows)},
            "accepted_by_gap_bucket": dict(sorted(Counter(
                row["visibility_features"]["gap_bucket"] for row in self.rows
            ).items())),
            "quartile_separated": sum(
                row["visibility_features"]["quartile_separated"] for row in self.rows
            ),
            "source_inventory": copy.deepcopy(self.canary_report["source_inventory"]),
            "validated_source_exclusions": self.source_exclusions,
            "visual_quality_contract": sensor.visual_quality_contract(),
            "frozen_canary_scenes": copy.deepcopy(
                self.canary_report["frozen_canary_scenes"]
            ),
            "source_contract": contract,
            "canary_binding": self.canary_binding,
            "excluded_canary_scene_ids": sorted(self.canary_binding["scene_ids"]),
            "excluded_corrupt_source_scene_ids": ["scene0015_00"],
            "scene_overlap_with_canary": [],
            "labels_path": "labels/S_spatial/depth.jsonl",
            "labels_sha256": sha256(self.labels),
            "records_sha256": packer.content_hash(self.rows),
            "eligible": True,
        }
        self.report.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    def build(self) -> Path:
        out = self.root / "pack"
        packer.build_pack(self.report, self.labels, out, seed=20260820)
        return out


def write_votes(pack: Path, disagreements: set[str] | None = None) -> None:
    disagreements = disagreements or set()
    for side, key in (("a", "reviewer-a"), ("b", "reviewer-b")):
        labels = read_jsonl(pack / f"annotator_{side}/labels/S_spatial/depth.jsonl")
        rows = []
        for label in labels:
            correct = not (side == "b" and label["sample_id"] in disagreements)
            rows.append({
                "annotator_key": key,
                "correct": correct,
                "error_code": None if correct else "ambiguous_evidence",
                "label_snapshot": label,
            })
        path = pack / f"annotator_{side}/annotations/annotations.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class ScanNetSensorAuditTests(unittest.TestCase):
    def test_source_audit_only_interface_is_required_after_source_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Schema5Fixture(Path(directory))
            rows = read_jsonl(fixture.labels)
            rows[0].pop("_audit_only")
            fixture.labels.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = json.loads(fixture.report.read_text())
            report["labels_sha256"] = sha256(fixture.labels)
            report["records_sha256"] = packer.content_hash(rows)
            fixture.report.write_text(json.dumps(report, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "held-out audit interface drift"):
                packer.build_pack(
                    fixture.report,
                    fixture.labels,
                    Path(directory) / "should-not-exist",
                    seed=20260820,
                )

    def test_pack_is_val_only_reserved_and_ui_hides_sensor_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Schema5Fixture(Path(directory))
            pack = fixture.build()
            manifest = json.loads((pack / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 5)
            self.assertEqual(manifest["task_interface"], "oracle_grounded_pairwise_depth")
            self.assertEqual(manifest["selection"]["selected_by_official_split"], {"val": 40})
            self.assertEqual(
                manifest["source_inventory"]["openeqa_scenes_by_official_split"],
                {"train": 0, "val": 79, "test": 10},
            )
            self.assertEqual(
                manifest["source_inventory"]["ready_source_scenes_by_official_split"],
                {"train": 0, "val": 78, "test": 0},
            )
            self.assertEqual(
                manifest["source_inventory"]["generated_source_records_by_official_split"],
                {"train": 0, "val": 120, "test": 0},
            )
            self.assertFalse(
                manifest["source_inventory"]["official_test_used_for_calibration"]
            )
            self.assertEqual(len({item["scene_id"] for item in manifest["selected"]}), 40)
            self.assertGreaterEqual(len(manifest["reserved_benchmark"]["scene_ids"]), 20)
            self.assertFalse(manifest["training_materialization"]["eligible"])
            labels = read_jsonl(pack / "annotator_a/labels/S_spatial/depth.jsonl")
            self.assertEqual(len(labels), 40)
            self.assertTrue(all(row["_audit_only"] for row in labels))
            self.assertTrue(all(set(item) == {"label", "rank", "relative_depth"}
                                for row in labels for item in row["payload"]))
            store = QuickAuditStore(pack, "a")
            case = store.case(0)
            serialized = json.dumps(case)
            self.assertIn('"audit_contract": "scannet_sensor_pairwise"', serialized)
            for secret in (
                "sensor_measurements", "verified_pair", "depth_gap_m",
                "depth_median_m", "visibility_features", "robust_se_m",
            ):
                self.assertNotIn(secret, serialized)

    def test_compiler_pass_is_benchmark_only_and_source_tamper_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Schema5Fixture(Path(directory))
            pack = fixture.build()
            write_votes(pack)
            args = argparse.Namespace(
                pack=pack, adjudication=None,
                min_accuracy=0.80, min_agreement=0.80,
            )
            report, rc = compiler.compile_report(args)
            self.assertEqual(rc, 0, report["errors"])
            self.assertEqual(report["verdict"], "PASS")
            self.assertEqual(
                report["qualification_scope"],
                "held_out_benchmark_component_calibration",
            )
            self.assertFalse(report["training_materialization"]["eligible"])
            first_image = Path(fixture.rows[0]["image_path"])
            first_image.write_bytes(b"tampered")
            blocked, rc = compiler.compile_report(args)
            self.assertEqual(rc, 2)
            self.assertTrue(any("image hash drift" in error for error in blocked["errors"]))

    def test_source_inventory_manifest_tamper_blocks_after_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Schema5Fixture(Path(directory))
            pack = fixture.build()
            write_votes(pack)
            manifest_path = pack / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["source_inventory"]["openeqa_scenes_by_official_split"]["test"] = 9
            manifest["integrity"]["manifest_payload_sha256"] = (
                packer.manifest_payload_hash(manifest)
            )
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            args = argparse.Namespace(
                pack=pack, adjudication=None,
                min_accuracy=0.80, min_agreement=0.80,
            )
            blocked, rc = compiler.compile_report(args)
            self.assertEqual(rc, 2)
            self.assertIn("v5 source_inventory snapshot drift", blocked["errors"])

    def test_reviewer_c_identity_and_negative_error_code_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Schema5Fixture(Path(directory))
            pack = fixture.build()
            label = read_jsonl(pack / "annotator_a/labels/S_spatial/depth.jsonl")[0]
            sample_id = label["sample_id"]
            write_votes(pack, {sample_id})
            manifest = json.loads((pack / "manifest.json").read_text())
            c_pack = Path(directory) / "reviewer-c-pack"
            build_adjudication_pack(
                pack,
                c_pack,
                excluded_forms=set(),
                expected_count=1,
                reviewer_key="reviewer-c",
            )
            c_case = AdjudicationStore(c_pack).case(0)
            self.assertEqual(
                c_case["forms"][0]["audit_contract"],
                "scannet_sensor_pairwise",
            )
            c_payload = json.dumps(c_case)
            self.assertNotIn("sensor_measurements", c_payload)
            self.assertNotIn("depth_gap_m", c_payload)
            adjudication = Path(directory) / "adjudication.jsonl"
            row = {
                "sample_id": sample_id,
                "correct": True,
                "error_code": None,
                "reviewer_key": "reviewer-c",
                "source_audit_id": manifest["audit_id"],
                "adjudication_id": "schema5-c-v1",
                "label_sha256": packer.content_hash(label),
            }
            adjudication.write_text(json.dumps(row) + "\n")
            args = argparse.Namespace(
                pack=pack, adjudication=adjudication,
                min_accuracy=0.80, min_agreement=0.80,
            )
            report, rc = compiler.compile_report(args)
            self.assertEqual(rc, 0, report["errors"])
            row["reviewer_key"] = "reviewer-a"
            adjudication.write_text(json.dumps(row) + "\n")
            report, rc = compiler.compile_report(args)
            self.assertEqual(rc, 2)
            self.assertIn("v5 reviewer C must be independent", "\n".join(report["errors"]))

            path = pack / "annotator_a/annotations/annotations.jsonl"
            votes = read_jsonl(path)
            votes[1]["correct"] = False
            votes[1]["error_code"] = None
            path.write_text("".join(json.dumps(value) + "\n" for value in votes))
            report, rc = compiler.compile_report(args)
            self.assertEqual(rc, 2)
            self.assertIn("lacks error_code", "\n".join(report["errors"]))

    def test_nonempty_output_and_heldout_materializer_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = Schema5Fixture(root)
            pack = fixture.build()
            with self.assertRaisesRegex(ValueError, "non-empty output"):
                packer.build_pack(fixture.report, fixture.labels, pack, seed=20260820)

            v3_adjudication = root / "v3-adjudication.jsonl"
            depth_adjudication = root / "depth-adjudication.jsonl"
            v3_adjudication.write_text("{}\n")
            depth_adjudication.write_text("{}\n")
            report_path = root / "composite.json"
            report_path.write_text("{}")
            heldout_report = {
                "training_materialization": {
                    "eligible": False,
                    "reason": "held_out_sensor_records_not_trainable",
                },
                "depth_component": {"key": "v5_scannet_sensor_depth"},
            }
            args = argparse.Namespace(
                v3_pack=root / "v3",
                depth_pack=pack,
                v4_pack=None,
                v3_adjudication=v3_adjudication,
                depth_adjudication=depth_adjudication,
                v4_adjudication=None,
                audit=report_path,
                out=root / "must-not-exist",
                seed=1,
                val_fraction=0.2,
            )
            with patch.object(
                materializer,
                "_fresh_composite_report",
                return_value=(heldout_report, "a" * 64),
            ):
                with self.assertRaisesRegex(
                    ValueError, "held_out_sensor_records_not_trainable"
                ):
                    materializer.materialize(args)
            self.assertFalse(args.out.exists())

    def test_composite_uses_explicit_schema5_name_and_scope(self) -> None:
        row = {
            "expected": 40, "annotator_a": 40, "annotator_b": 40,
            "paired": 40, "agreed": 40, "adjudicated": 0,
            "unresolved": 0, "raw_agreement": 1.0,
            "resolved_accuracy": 1.0, "eligible": True,
        }
        v3 = {
            "schema_version": 3, "errors": [], "manifest": "/absent/v3.json",
            "by_form": {form: row for form in composite.NON_DEPTH_FORMS},
            "unresolved_sample_ids": [],
        }
        depth = {
            "schema_version": 5,
            "protocol": packer.PACK_PROTOCOL,
            "depth_scope": "pairwise",
            "errors": [],
            "manifest": "/absent/v5.json",
            "by_form": {"depth": row},
            "unresolved_sample_ids": [],
            "training_materialization": {"eligible": False},
        }
        args = argparse.Namespace(
            v3_pack=Path("/v3"), v3_adjudication=None,
            depth_pack=Path("/v5"), depth_adjudication=None,
            v4_pack=None, v4_adjudication=None,
        )
        with patch.object(composite, "compile_component", side_effect=[v3, depth]):
            report, rc = composite.compile_composite(args)
        self.assertEqual(rc, 0)
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["training_release_verdict"], "BLOCKED")
        self.assertEqual(
            report["qualification_scope"],
            "held_out_benchmark_component_calibration",
        )
        self.assertEqual(
            report["depth_component"]["corpus_version"],
            "gsa-benchmark-v50-scannet-sensor-calibration-heldout-v1",
        )
        self.assertNotIn("v42", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
