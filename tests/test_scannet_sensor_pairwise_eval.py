from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import scannet_sensor_pairwise as depth_eval  # noqa: E402


class ScanNetSensorPairwiseEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def image(self, index: int) -> Path:
        path = self.root / f"image-{index}.png"
        path.write_bytes(b"fixture-image-" + bytes([index]))
        return path

    def record(
        self,
        index: int,
        *,
        audit_projection: bool = False,
        same_class: bool = False,
    ) -> dict:
        image = self.image(index)
        near_label = f"chair [instance {index * 2 + 1}]"
        far_class = "chair" if same_class else "table"
        far_label = f"{far_class} [instance {index * 2 + 2}]"
        near_box = [40, 80, 360, 820]
        far_box = [590, 130, 960, 910]
        measurements = [
            {"label": near_label, "category": "chair", "bbox_2d": near_box},
            {"label": far_label, "category": far_class, "bbox_2d": far_box},
        ]
        record = {
            "sample_id": f"source-sample-{index}",
            "protocol_version": depth_eval.SOURCE_PROTOCOL,
            "label_version": depth_eval.SOURCE_LABEL_VERSION,
            "official_split": "val",
            "image_path": str(image),
            "image_sha256": depth_eval.file_sha256(image),
            "payload": [
                {"label": near_label, "rank": 1, "relative_depth": 0.0},
                {"label": far_label, "rank": 2, "relative_depth": 1.0},
            ],
            "sensor_measurements": measurements,
            "gsa_context": {
                "contract_version": "gsa-contract-v2",
                "groundings": [
                    {"object_class": near_label, "bbox": near_box},
                    {"object_class": far_label, "bbox": far_box},
                ],
            },
            "model_interface": {
                "intended_task": "oracle_grounded_pairwise_depth",
                "requires_explicit_pair_groundings": True,
                "instance_ids_are_internal_identity_only": True,
            },
            "training_materialization": {
                "eligible": False,
                "allowed_use": "blind_audit_or_benchmark_only",
            },
            "visibility_features": {"gap_bucket": "0.50_to_0.99m"},
        }
        if audit_projection:
            record["source_sample_id"] = record["sample_id"]
            record["sample_id"] = f"audit-projection-{index}"
            record["audit_contract"] = "scannet_sensor_pairwise"
            record["task_interface"] = "oracle_grounded_pairwise_depth"
            record["_audit_only"] = True
            record["sensor_provenance"] = {
                "measurements": record.pop("sensor_measurements"),
                "visibility_features": record.pop("visibility_features"),
            }
        return record

    def write_jsonl(self, name: str, rows: list[dict]) -> Path:
        path = self.root / name
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def test_request_explicitly_names_pair_and_leaks_no_internal_identity(self) -> None:
        pair = depth_eval.build_pair_example(self.record(0))
        request = depth_eval.format_requests(
            [pair], track=depth_eval.TRACK_S_ORACLE
        )[0]
        prompt = request["messages"][0]["content"]
        serialised = json.dumps(request).casefold()
        self.assertIn("A: ", prompt)
        self.assertIn("B: ", prompt)
        self.assertIn("0-1000", prompt)
        self.assertIn("compare only the two supplied objects", prompt)
        self.assertNotIn("order visible objects", prompt.casefold())
        self.assertNotIn("[instance", serialised)
        self.assertNotIn("depth_m", serialised)
        self.assertNotIn(pair.source_identity, serialised)
        self.assertEqual(
            request["evaluation_protocol"],
            "gsa-depth-v5.0-oracle-pairwise-eval-v3",
        )

    def test_blinding_validator_fails_closed_on_private_fields_or_identity(self) -> None:
        pair = depth_eval.build_pair_example(self.record(0))
        clean = depth_eval.format_requests(
            [pair], track=depth_eval.TRACK_S_ORACLE
        )[0]
        cases = []
        leaked_answer = copy.deepcopy(clean)
        leaked_answer["answer"] = pair.nearer_choice
        cases.append(leaked_answer)
        leaked_sensor_value = copy.deepcopy(clean)
        leaked_sensor_value["pair"]["A"]["depth_m"] = 1.5
        cases.append(leaked_sensor_value)
        leaked_identity = copy.deepcopy(clean)
        leaked_identity["pair"]["A"]["name"] = pair.source_identity
        cases.append(leaked_identity)
        for request in cases:
            with self.subTest(keys=sorted(request)):
                with self.assertRaises(depth_eval.ContractError):
                    depth_eval._assert_blinded_request(request, pair)

    def test_ab_permutation_is_stable_across_source_and_audit_projection(self) -> None:
        source = self.record(1)
        audit = copy.deepcopy(source)
        audit["source_sample_id"] = audit["sample_id"]
        audit["sample_id"] = "different-audit-id"
        audit["audit_contract"] = "scannet_sensor_pairwise"
        audit["task_interface"] = "oracle_grounded_pairwise_depth"
        audit["_audit_only"] = True
        audit["sensor_provenance"] = {
            "measurements": audit.pop("sensor_measurements"),
            "visibility_features": audit.pop("visibility_features"),
        }
        source_pair = depth_eval.build_pair_example(source)
        audit_pair = depth_eval.build_pair_example(audit)
        self.assertEqual(source_pair.eval_id, audit_pair.eval_id)
        self.assertEqual(source_pair.nearer_choice, audit_pair.nearer_choice)
        self.assertEqual(source_pair.entity_a.internal_label, audit_pair.entity_a.internal_label)

    def test_parser_accepts_choice_json_relation_and_unique_label(self) -> None:
        cases = {
            "A": "A",
            "Answer: b.": "B",
            '{"nearer": "A"}': "A",
            "Object B is nearer than object A.": "B",
            "B is farther than A.": "A",
            "chair": "A",
            "<think>compare the boxes</think> B": "B",
        }
        for response, expected in cases.items():
            with self.subTest(response=response):
                parsed = depth_eval.parse_nearer_choice(response, "chair", "table")
                self.assertEqual(parsed.choice, expected)

    def test_parser_refuses_ambiguous_same_class_label_and_unclosed_reasoning(self) -> None:
        self.assertIsNone(
            depth_eval.parse_nearer_choice("chair", "chair", "chair").choice
        )
        self.assertIsNone(
            depth_eval.parse_nearer_choice("A or B", "chair", "table").choice
        )
        self.assertIsNone(
            depth_eval.parse_nearer_choice("<think>A seems closer", "chair", "table").choice
        )

    def test_label_relation_does_not_confuse_nested_class_names(self) -> None:
        self.assertEqual(
            depth_eval.parse_nearer_choice(
                "office chair is nearer than chair", "chair", "office chair"
            ).choice,
            "B",
        )

    def test_oracle_scorer_uses_all_rows_as_denominator(self) -> None:
        pairs = [
            depth_eval.build_pair_example(self.record(0)),
            depth_eval.build_pair_example(self.record(1)),
        ]
        predictions = {
            pairs[0].eval_id: {
                "response": pairs[0].nearer_choice,
                "hit_token_cap": False,
            },
            pairs[1].eval_id: {"response": "not an answer", "hit_token_cap": False},
        }
        report = depth_eval.score_predictions(
            pairs, predictions, track=depth_eval.TRACK_S_ORACLE
        )
        self.assertEqual(report["metrics"]["pairwise_depth_accuracy_all"], 0.5)
        self.assertEqual(report["metrics"]["parse_coverage_all"], 0.5)
        self.assertEqual(report["metrics"]["pairwise_depth_accuracy_given_parsed"], 1.0)
        self.assertTrue(report["trust"]["score_available"])

    def test_missing_prediction_withholds_primary_score(self) -> None:
        pairs = [
            depth_eval.build_pair_example(self.record(0)),
            depth_eval.build_pair_example(self.record(1)),
        ]
        predictions = {
            pairs[0].eval_id: {
                "response": pairs[0].nearer_choice,
                "hit_token_cap": False,
            }
        }
        report = depth_eval.score_predictions(
            pairs, predictions, track=depth_eval.TRACK_S_ORACLE
        )
        self.assertIsNone(report["metrics"]["pairwise_depth_accuracy_all"])
        self.assertEqual(report["metrics"]["score_if_untrusted"], 0.5)
        self.assertFalse(report["trust"]["score_available"])

    def test_formal_oracle_without_prediction_provenance_is_untrusted(self) -> None:
        pairs = [
            depth_eval.build_pair_example(self.record(0)),
            depth_eval.build_pair_example(self.record(1)),
        ]
        predictions = {
            pair.eval_id: {
                "response": pair.nearer_choice,
                "hit_token_cap": False,
            }
            for pair in pairs
        }
        report = depth_eval.score_predictions(
            pairs,
            predictions,
            track=depth_eval.TRACK_S_ORACLE,
            source_metadata={"formal_source_ready": True},
        )
        self.assertFalse(report["trust"]["score_available"])
        self.assertFalse(report["trust"]["prediction_provenance_verified"])
        self.assertIn("prediction-manifest", " ".join(report["trust"]["reasons"]))

    def test_cascade_reports_grounding_and_strict_end_to_end_metrics(self) -> None:
        pairs = [
            depth_eval.build_pair_example(self.record(0)),
            depth_eval.build_pair_example(self.record(1)),
        ]
        groundings = {
            pairs[0].eval_id: depth_eval.GroundingRow(
                pairs[0].entity_a.bbox, pairs[0].entity_b.bbox
            ),
            pairs[1].eval_id: depth_eval.GroundingRow(
                (0, 0, 20, 20), (20, 20, 40, 40)
            ),
        }
        predictions = {
            pair.eval_id: {
                "response": pair.nearer_choice,
                "hit_token_cap": False,
            }
            for pair in pairs
        }
        report = depth_eval.score_predictions(
            pairs,
            predictions,
            track=depth_eval.TRACK_G_TO_S,
            groundings=groundings,
        )
        self.assertEqual(report["metrics"]["pairwise_depth_accuracy_all"], 1.0)
        self.assertEqual(
            report["cascade_metrics"]["grounding_pair_success_iou_at_0_5"], 0.5
        )
        self.assertEqual(
            report["cascade_metrics"]["end_to_end_accuracy_iou_at_0_5"], 0.5
        )
        self.assertEqual(
            report["task_contract"]["primary_metric"],
            "end_to_end_accuracy_iou_at_0_5",
        )

    def test_missing_grounding_is_retained_as_cascade_failure(self) -> None:
        pairs = [
            depth_eval.build_pair_example(self.record(0)),
            depth_eval.build_pair_example(self.record(1)),
        ]
        path = self.write_jsonl(
            "groundings.jsonl",
            [
                {
                    "eval_id": pairs[0].eval_id,
                    "A": {"bbox_2d": list(pairs[0].entity_a.bbox)},
                    "B": {"bbox_2d": list(pairs[0].entity_b.bbox)},
                }
            ],
        )
        valid, invalid = depth_eval.load_groundings(path, pairs)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(invalid), 1)
        requests = depth_eval.format_requests(
            pairs, track=depth_eval.TRACK_G_TO_S, groundings=valid
        )
        predictions = {
            pairs[0].eval_id: {
                "response": pairs[0].nearer_choice,
                "hit_token_cap": False,
            }
        }
        report = depth_eval.score_predictions(
            pairs,
            predictions,
            track=depth_eval.TRACK_G_TO_S,
            groundings=valid,
            invalid_groundings=invalid,
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(report["metrics"]["no_request_due_to_grounding"], 1)
        self.assertEqual(
            report["cascade_metrics"]["end_to_end_accuracy_iou_at_0_5"], 0.5
        )

    def test_empty_grounding_and_prediction_files_are_measured_not_crashes(self) -> None:
        pairs = [depth_eval.build_pair_example(self.record(0))]
        grounding_path = self.root / "empty-groundings.jsonl"
        prediction_path = self.root / "empty-predictions.jsonl"
        grounding_path.write_text("", encoding="utf-8")
        prediction_path.write_text("", encoding="utf-8")
        groundings, invalid = depth_eval.load_groundings(grounding_path, pairs)
        predictions = depth_eval.load_predictions(prediction_path, set())
        report = depth_eval.score_predictions(
            pairs,
            predictions,
            track=depth_eval.TRACK_G_TO_S,
            groundings=groundings,
            invalid_groundings=invalid,
        )
        self.assertEqual(report["cascade_metrics"]["valid_grounding_pair_coverage"], 0.0)
        self.assertEqual(report["cascade_metrics"]["end_to_end_accuracy_iou_at_0_5"], 0.0)

    def test_training_eligible_source_is_rejected(self) -> None:
        record = self.record(0)
        record["training_materialization"]["eligible"] = True
        with self.assertRaisesRegex(depth_eval.ContractError, "training prohibition"):
            depth_eval.build_pair_example(record)

    def test_source_report_binds_records_and_label_bytes(self) -> None:
        records = [self.record(0), self.record(1)]
        labels = self.write_jsonl("source.jsonl", records)
        report = {
            "stage": "full",
            "protocol": depth_eval.SOURCE_PROTOCOL,
            "label_version": depth_eval.SOURCE_LABEL_VERSION,
            "eligible": True,
            "accepted": 2,
            "target": 2,
            "system_errors": {"total": 0, "items": []},
            "labels_sha256": depth_eval.file_sha256(labels),
            "records_sha256": depth_eval.canonical_hash(records),
            "source_contract": {
                "intended_task": "oracle_grounded_pairwise_depth",
                "training_materialization_eligible": False,
                "generation_split": "val",
                "sensor_only": True,
            },
        }
        report_path = self.root / "generation_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        pairs, metadata = depth_eval.load_pairs(
            labels, report_path=report_path, verify_images=True
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual(metadata["source_stage"], "full")
        self.assertFalse(metadata["formal_source_identity_match"])
        self.assertFalse(metadata["formal_source_ready"])
        records[0]["payload"][0]["rank"] = 9
        labels.write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
        with self.assertRaisesRegex(depth_eval.ContractError, "labels_sha256 drift"):
            depth_eval.load_pairs(labels, report_path=report_path, verify_images=True)

    def test_frozen_full_v3_is_pinned_exactly_balanced_and_blinded(self) -> None:
        source_root = ROOT / "data" / "depth_v50_scannet_sensor" / "full_v3"
        labels = source_root / "labels" / "S_spatial" / "depth.jsonl"
        report = source_root / "generation_report.json"
        self.assertTrue(labels.is_file(), "pinned full_v3 labels are absent")
        self.assertTrue(report.is_file(), "pinned full_v3 report is absent")
        pairs, metadata = depth_eval.load_pairs(
            labels, report_path=report, verify_images=False
        )
        self.assertTrue(metadata["formal_source_identity_match"])
        self.assertTrue(metadata["formal_source_ready"])
        self.assertEqual(len(pairs), 120)
        self.assertEqual(
            dict(sorted(Counter(pair.nearer_choice for pair in pairs).items())),
            depth_eval.FORMAL_LABEL_BALANCE,
        )
        requests = depth_eval.format_requests(
            pairs, track=depth_eval.TRACK_S_ORACLE
        )
        self.assertEqual(len(requests), 120)
        serialised = json.dumps(requests).casefold()
        self.assertNotIn("[instance", serialised)
        for key in depth_eval._FORBIDDEN_PUBLIC_KEYS:
            self.assertNotIn(f'"{key.casefold()}"', serialised)


if __name__ == "__main__":
    unittest.main()
