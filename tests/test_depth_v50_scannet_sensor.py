from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import build_depth_v50_scannet_sensor as sensor


def png_bytes(array: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def identity_matrix(path: Path) -> None:
    np.savetxt(path, np.eye(4), fmt="%.8f")


def make_stat(
    instance_id: int,
    median: float,
    q1: float,
    q3: float,
    robust_se: float = 0.01,
    *,
    bbox: tuple[int, int, int, int] = (10, 10, 30, 30),
    bbox_area_fraction: float = 0.04,
    bbox_edge_clearance_px: int = 10,
    mask_fill_ratio: float = 0.8,
) -> sensor.InstanceStat:
    return sensor.InstanceStat(
        instance_id=instance_id,
        aggregation_label=f"object-{instance_id}",
        raw_label_id=instance_id,
        category=f"object-{instance_id}",
        nyu40class="otherprop",
        bbox_xyxy_px=bbox,
        bbox_2d=bbox,
        mask_pixels=400,
        bbox_area_fraction=bbox_area_fraction,
        bbox_edge_clearance_px=bbox_edge_clearance_px,
        mask_fill_ratio=mask_fill_ratio,
        valid_samples=200,
        valid_fraction=0.8,
        depth_median_m=median,
        depth_q1_m=q1,
        depth_q3_m=q3,
        depth_iqr_m=q3 - q1,
        depth_mad_m=robust_se,
        robust_se_m=robust_se,
    )


class ScanNetSensorDepthTests(unittest.TestCase):
    def test_bare_training_scene_id_is_opt_in_and_cannot_match_another_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene0009_00.aggregation.json"
            payload = {"sceneId": "scene0009_00", "segGroups": [{"objectId": 0, "label": "chair"}]}
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(sensor.SourceError, "sceneId mismatch"):
                sensor.load_aggregation(path, "scene0009_00")
            labels, digest = sensor.load_aggregation(path, "scene0009_00", allow_bare_scene_id=True)
            self.assertEqual(labels, {1: "chair"})
            self.assertEqual(digest, sensor.file_sha256(path))
            with self.assertRaisesRegex(sensor.SourceError, "sceneId mismatch"):
                sensor.load_aggregation(path, "scene0010_00", allow_bare_scene_id=True)

    def test_frozen_canary_scene_manifest_is_pinned(self) -> None:
        manifest = ROOT / "configs/scannet_sensor_canary_scenes_v1.json"
        document = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            sensor.file_sha256(manifest), sensor.SCANNET_CANARY_SCENES_SHA256
        )
        self.assertEqual(document["quality_contract"], sensor.visual_quality_contract())
        self.assertEqual(len(document["scene_ids"]), sensor.CANARY_TARGET)
        self.assertEqual(document["scene_ids"], sorted(set(document["scene_ids"])))
        self.assertNotIn("scene0077_00", document["scene_ids"])
        self.assertIn("scene0300_01", document["scene_ids"])

    def test_canonical_source_exclusion_manifest_is_pinned(self) -> None:
        manifest = ROOT / "configs/scannet_source_exclusions_v1.json"
        self.assertEqual(
            sensor.file_sha256(manifest),
            sensor.SCANNET_SOURCE_EXCLUSIONS_SHA256,
        )

    def test_source_exclusion_is_hash_bound_and_split_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "scene0015_00" / "scene0015_00_2d-label-filt.zip"
            artifact.parent.mkdir()
            artifact.write_bytes(b"known-corrupt-archive")
            manifest = root / "exclusions.json"
            manifest.write_text(json.dumps({
                "schema": "gsa-scannet-source-exclusions-v1",
                "dataset": "ScanNet v2",
                "exclusions": [{
                    "scene_id": "scene0015_00",
                    "official_split": "val",
                    "artifact": artifact.name,
                    "sha256": sensor.file_sha256(artifact),
                    "reason": "corrupt_zip_archive",
                    "evidence": "unzip_test_failed",
                }],
            }), encoding="utf-8")
            exclusions, binding = sensor.load_source_exclusions(
                manifest,
                root,
                {"scene0015_00": "val"},
                sensor.file_sha256(manifest),
            )
            self.assertEqual(set(exclusions), {"scene0015_00"})
            self.assertEqual(binding["count"], 1)
            artifact.write_bytes(b"different-bytes")
            with self.assertRaises(sensor.SourceError):
                sensor.load_source_exclusions(
                    manifest,
                    root,
                    {"scene0015_00": "val"},
                    sensor.file_sha256(manifest),
                )

    def test_source_exclusion_rejects_unknown_split_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "scene0015_00" / "scene0015_00_2d-label-filt.zip"
            artifact.parent.mkdir()
            artifact.write_bytes(b"known-corrupt-archive")
            manifest = root / "exclusions.json"
            manifest.write_text(json.dumps({
                "schema": "gsa-scannet-source-exclusions-v1",
                "dataset": "ScanNet v2",
                "exclusions": [{
                    "scene_id": "scene0015_00",
                    "official_split": "val",
                    "artifact": artifact.name,
                    "sha256": sensor.file_sha256(artifact),
                    "reason": "corrupt_zip_archive",
                    "evidence": "unzip_test_failed",
                }],
            }), encoding="utf-8")
            with self.assertRaises(sensor.SourceError):
                sensor.load_source_exclusions(
                    manifest, root, {}, sensor.file_sha256(manifest)
                )

    def test_source_exclusion_rejects_a_healthy_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "scene0015_00" / "scene0015_00_2d-label-filt.zip"
            artifact.parent.mkdir()
            with ZipFile(artifact, "w") as archive:
                archive.writestr("label-filt/0.png", png_bytes(np.zeros((2, 2), dtype=np.uint8)))
            manifest = root / "exclusions.json"
            manifest.write_text(json.dumps({
                "schema": "gsa-scannet-source-exclusions-v1",
                "dataset": "ScanNet v2",
                "exclusions": [{
                    "scene_id": "scene0015_00",
                    "official_split": "val",
                    "artifact": artifact.name,
                    "sha256": sensor.file_sha256(artifact),
                    "reason": "corrupt_zip_archive",
                    "evidence": "unzip_test_failed",
                }],
            }), encoding="utf-8")
            with self.assertRaises(sensor.SourceError):
                sensor.load_source_exclusions(
                    manifest,
                    root,
                    {"scene0015_00": "val"},
                    sensor.file_sha256(manifest),
                )

    def test_official_split_manifests_are_pinned_and_disjoint(self) -> None:
        manifests, lookup = sensor.load_split_manifests(
            ROOT / "configs/scannet_splits"
        )
        self.assertEqual({name: len(item.scene_ids) for name, item in manifests.items()}, {
            "train": 1201,
            "val": 312,
            "test": 100,
        })
        self.assertEqual(lookup["scene0700_01"], "val")
        self.assertEqual(
            manifests["train"].sha256,
            sensor.SCANNET_SPLIT_SHA256["train"],
        )

    def test_official_split_hash_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split_root = Path(directory) / "splits"
            shutil.copytree(ROOT / "configs/scannet_splits", split_root)
            with (split_root / "scannetv2_val.txt").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("scene9999_99\n")
            with self.assertRaises(sensor.SourceError):
                sensor.load_split_manifests(split_root)

    def test_label_projection_uses_raw_id_not_nyu40id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.tsv"
            path.write_text(
                "id\traw_category\tcategory\tnyu40id\tnyu40class\n"
                "5\tdoor\tdoor\t8\tdoor\n"
                "10\toffice chair\toffice chair\t5\tchair\n"
                "101\ttelephone\ttelephone\t40\totherprop\n",
                encoding="utf-8",
            )
            labels = sensor.load_label_map(path, expected_sha256=None)
        self.assertEqual(labels.by_id[5]["category"], "door")
        self.assertEqual(labels.by_id[10]["category"], "office chair")
        self.assertEqual(labels.by_id[101]["category"], "telephone")

    def test_official_label_map_is_hash_pinned(self) -> None:
        self.assertEqual(
            sensor.file_sha256(sensor.DEFAULT_LABEL_MAP),
            sensor.SCANNET_LABEL_MAP_SHA256,
        )
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "labels.tsv"
            changed.write_text(
                "id\traw_category\tcategory\tnyu40class\n"
                "1\tchair\tchair\tchair\n",
                encoding="utf-8",
            )
            with self.assertRaises(sensor.SourceError):
                sensor.load_label_map(changed)

    def test_identity_registration_preserves_metric_samples(self) -> None:
        depth = np.array([[1000, 2000], [3000, 0]], dtype=np.uint16)
        registered = sensor.register_depth_to_color(
            depth,
            np.eye(4),
            np.eye(4),
            np.eye(4),
            np.eye(4),
            (2, 2),
        )
        self.assertEqual(registered.raw_valid_pixels, 3)
        self.assertEqual(registered.registered_pixels, 3)
        self.assertEqual(sorted(registered.z_color_m.tolist()), [1.0, 2.0, 3.0])

    def test_instance_mapping_and_robust_statistics(self) -> None:
        instance = np.zeros((100, 100), dtype=np.uint8)
        instance[10:90, 5:45] = 1
        instance[10:90, 55:95] = 2
        semantic = np.zeros((100, 100), dtype=np.uint16)
        semantic[instance == 1] = 10
        semantic[instance == 2] = 101
        y, x = np.nonzero(instance)
        depths = np.where(instance[y, x] == 1, 1.0, 2.0)
        registered = sensor.RegisteredDepth(x, y, depths, len(x), len(x))
        labels = sensor.LabelMaps(
            by_id={
                10: {"category": "office chair", "nyu40class": "chair"},
                101: {"category": "telephone", "nyu40class": "otherprop"},
            },
            by_raw_category={
                "office chair": {
                    "category": "office chair",
                    "nyu40class": "chair",
                },
                "telephone": {
                    "category": "telephone",
                    "nyu40class": "otherprop",
                },
            },
        )
        stats, rejected = sensor.instance_statistics(
            instance,
            semantic,
            registered,
            instance.shape,
            labels,
            {1: "office chair", 2: "telephone"},
        )
        self.assertFalse(rejected)
        self.assertEqual([item.display_label for item in stats], [
            "office chair [instance 1]",
            "telephone [instance 2]",
        ])
        self.assertEqual([item.raw_label_id for item in stats], [10, 101])
        self.assertEqual([item.depth_median_m for item in stats], [1.0, 2.0])
        self.assertEqual([item.valid_fraction for item in stats], [1.0, 1.0])
        self.assertEqual([item.bbox_edge_clearance_px for item in stats], [5, 5])
        self.assertEqual([item.bbox_area_fraction for item in stats], [0.32, 0.32])
        self.assertEqual([item.mask_fill_ratio for item in stats], [1.0, 1.0])

    def test_pair_requires_metric_and_uncertainty_margin(self) -> None:
        near = make_stat(1, 1.0, 0.95, 1.05)
        too_close = make_stat(2, 1.20, 1.15, 1.25)
        far = make_stat(3, 1.60, 1.50, 1.70)
        self.assertIsNone(sensor.choose_pair([near, too_close]))
        selected = sensor.choose_pair([near, far])
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.nearer.instance_id, 1)
        self.assertEqual(selected.farther.instance_id, 3)
        self.assertTrue(selected.quartile_separated)
        self.assertGreaterEqual(selected.depth_gap_m, selected.required_gap_m)

    def test_pair_requires_strict_quartile_separation(self) -> None:
        near = make_stat(1, 1.0, 0.8, 1.20)
        far = make_stat(2, 1.6, 1.15, 1.85)
        self.assertIsNone(sensor.choose_pair([near, far]))

    def test_pair_selection_prioritizes_visual_auditability(self) -> None:
        clear_near = make_stat(
            1, 1.0, 0.9, 1.1, bbox=(10, 10, 30, 30)
        )
        clear_far = make_stat(
            2, 1.6, 1.5, 1.7, bbox=(60, 60, 80, 80)
        )
        large_gap_poor_box = make_stat(
            3,
            3.5,
            3.4,
            3.6,
            bbox=(1, 1, 12, 12),
            bbox_area_fraction=0.0121,
            bbox_edge_clearance_px=1,
        )
        selected = sensor.choose_pair([clear_near, clear_far, large_gap_poor_box])
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(
            {selected.nearer.instance_id, selected.farther.instance_id}, {1, 2}
        )

    def test_cross_frame_selection_uses_the_same_visual_priority(self) -> None:
        def record(
            sample_id: str,
            min_area: float,
            edge: int,
            overlap: float,
            margin: float,
            gap: float,
        ) -> dict:
            return {
                "sample_id": sample_id,
                "scene_id": "scene0000_00",
                "frame_index": int(sample_id[-1]),
                "image_sha256": f"{int(sample_id[-1]):064x}",
                "visibility_features": {
                    "min_bbox_area_fraction": min_area,
                    "min_bbox_edge_clearance_px": edge,
                    "bbox_intersection_over_min_area": overlap,
                    "min_mask_fill_ratio": 0.8,
                    "min_valid_fraction": 0.9,
                },
                "verified_pair": {"margin_m": margin, "depth_gap_m": gap},
            }

        clear = record("pair1", 0.04, 12, 0.0, 0.30, 0.55)
        cropped_large_gap = record("pair2", 0.011, 1, 0.8, 2.0, 2.5)
        self.assertGreater(
            sensor.record_auditability_score(clear),
            sensor.record_auditability_score(cropped_large_gap),
        )

    def test_instance_quality_filters_border_generic_and_area(self) -> None:
        def classify(mask: np.ndarray, label: str) -> dict[str, int]:
            semantic = np.zeros(mask.shape, dtype=np.uint16)
            semantic[mask == 1] = 10
            y, x = np.nonzero(mask)
            registered = sensor.RegisteredDepth(
                x, y, np.ones(len(x), dtype=np.float64), len(x), len(x)
            )
            maps = sensor.LabelMaps(
                by_id={10: {"category": label, "nyu40class": "otherprop"}},
                by_raw_category={
                    label.casefold(): {
                        "category": label,
                        "nyu40class": "otherprop",
                    }
                },
            )
            stats, rejected = sensor.instance_statistics(
                mask,
                semantic,
                registered,
                mask.shape,
                maps,
                {1: label},
            )
            self.assertEqual(stats, [])
            return dict(rejected)

        border = np.zeros((100, 100), dtype=np.uint8)
        border[0:80, 10:50] = 1
        self.assertEqual(classify(border, "office chair"), {"border_touching_bbox": 1})

        generic = np.zeros((100, 100), dtype=np.uint8)
        generic[10:50, 10:50] = 1
        self.assertEqual(classify(generic, "object"), {"generic_category": 1})

        oversized = np.zeros((100, 100), dtype=np.uint8)
        oversized[5:95, 5:95] = 1
        self.assertEqual(
            classify(oversized, "office chair"), {"bbox_area_out_of_range": 1}
        )

    def test_canonical_depth_context_rejects_label_drift(self) -> None:
        payload = [
            {"label": "chair", "rank": 1, "relative_depth": 0.0},
            {"label": "table", "rank": 2, "relative_depth": 1.0},
        ]
        context = sensor.context_for(
            "depth",
            [
                sensor.grounding("chair", [0, 0, 100, 100]),
                sensor.grounding("lamp", [200, 0, 300, 100]),
            ],
            {"relation_mode": "pairwise", "depth_order": payload},
        )
        self.assertEqual(sensor.validate("depth", payload, context)[0], False)

    def test_full_selection_covers_scenes_before_second_records(self) -> None:
        batches = [
            [{"sample_id": "a1"}, {"sample_id": "a2"}],
            [{"sample_id": "b1"}, {"sample_id": "b2"}],
            [{"sample_id": "c1"}, {"sample_id": "c2"}],
        ]
        selected = sensor.scene_round_robin(batches, 4)
        self.assertEqual(
            [record["sample_id"] for record in selected],
            ["a1", "b1", "c1", "a2"],
        )

    def test_analyze_frame_reads_zip_members_and_emits_source_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_root = root / "001-scannet-scene0000_00"
            official = root / "scene0000_00"
            frame_root.mkdir()
            official.mkdir()
            rgb = np.zeros((100, 100, 3), dtype=np.uint8)
            depth = np.zeros((100, 100), dtype=np.uint16)
            (frame_root / "000000-rgb.png").write_bytes(png_bytes(rgb))
            for name in (
                "intrinsic_color.txt",
                "intrinsic_depth.txt",
                "extrinsic_color.txt",
                "extrinsic_depth.txt",
                "000000.txt",
            ):
                identity_matrix(frame_root / name)
            instance = np.zeros((100, 100), dtype=np.uint8)
            instance[10:90, 5:45] = 1
            instance[10:90, 55:95] = 2
            depth[instance == 1] = 1000
            depth[instance == 2] = 1800
            (frame_root / "000000-depth.png").write_bytes(png_bytes(depth))
            semantic = np.zeros((100, 100), dtype=np.uint16)
            semantic[instance == 1] = 10
            semantic[instance == 2] = 101
            instance_zip = official / "scene0000_00_2d-instance-filt.zip"
            label_zip = official / "scene0000_00_2d-label-filt.zip"
            with ZipFile(instance_zip, "w") as archive:
                archive.writestr("instance-filt/0.png", png_bytes(instance))
            with ZipFile(label_zip, "w") as archive:
                archive.writestr("label-filt/0.png", png_bytes(semantic))
            metadata = official / "scene0000_00.txt"
            metadata.write_text("axisAlignment = 1 0 0 0\n", encoding="utf-8")
            aggregation = official / "scene0000_00.aggregation.json"
            aggregation.write_text(
                json.dumps({
                    "sceneId": "scannet.scene0000_00",
                    "segGroups": [
                        {"objectId": 0, "label": "office chair"},
                        {"objectId": 1, "label": "telephone"},
                    ],
                }),
                encoding="utf-8",
            )
            scene = sensor.SceneSource(
                folder=frame_root,
                scene_id="scene0000_00",
                official_folder=official,
                instance_zip=instance_zip,
                label_zip=label_zip,
                scene_metadata=metadata,
                aggregation_json=aggregation,
                official_split="val",
                frame_indices=(0,),
            )
            labels = sensor.LabelMaps(
                by_id={
                    10: {"category": "office chair", "nyu40class": "chair"},
                    101: {"category": "telephone", "nyu40class": "otherprop"},
                },
                by_raw_category={
                    "office chair": {
                        "category": "office chair",
                        "nyu40class": "chair",
                    },
                    "telephone": {
                        "category": "telephone",
                        "nyu40class": "otherprop",
                    },
                },
            )
            split = sensor.SplitManifest(
                name="val",
                path=root / "split.txt",
                sha256=sensor.SCANNET_SPLIT_SHA256["val"],
                source_url=(
                    "https://github.com/ScanNet/ScanNet/blob/"
                    f"{sensor.SCANNET_SPLIT_REVISION}/Tasks/Benchmark/"
                    "scannetv2_val.txt"
                ),
                source_revision=sensor.SCANNET_SPLIT_REVISION,
                scene_ids=frozenset({"scene0000_00"}),
            )
            with ZipFile(instance_zip) as instance_archive, ZipFile(
                label_zip
            ) as label_archive:
                record, reason, rejected = sensor.analyze_frame(
                    scene,
                    0,
                    labels,
                    {"val": split},
                    instance_archive,
                    label_archive,
                )
        self.assertEqual(reason, "accepted")
        self.assertFalse(rejected)
        assert record is not None
        self.assertEqual(record["payload"], [
            {
                "label": "office chair [instance 1]",
                "rank": 1,
                "relative_depth": 0.0,
            },
            {
                "label": "telephone [instance 2]",
                "rank": 2,
                "relative_depth": 1.0,
            },
        ])
        self.assertEqual(record["official_split"], "val")
        self.assertFalse(record["training_materialization"]["eligible"])
        self.assertEqual(
            record["sensor_provenance"]["aggregation"]["instance_id_rule"],
            "objectId_plus_1",
        )
        self.assertEqual(sensor.validate_record(record), [])


if __name__ == "__main__":
    unittest.main()
