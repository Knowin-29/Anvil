from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trains.flywheel.validators import validate_depth_v4_record
from eval.compile_semantic_audit import content_hash, manifest_payload_hash
from trains.flywheel.build_depth_semantic_audit_pack import generation_run_evidence_hash

BUILDER = ROOT / "trains/flywheel/build_depth_semantic_audit_pack.py"
COMPILER = ROOT / "eval/compile_semantic_audit.py"


def pairwise_record(image: Path, index: int) -> dict:
    nearer = f"front-cup-{index}"
    farther = f"rear-bowl-{index}"
    cue = ("occlusion", "contact_layering", "support_plane_perspective")[index % 3]
    groundings = [
        {"object_class": nearer, "bbox": [100, 120, 360, 480]},
        {"object_class": farther, "bbox": [300, 220, 620, 560]},
    ]
    payload = [
        {"label": nearer, "rank": 1, "relative_depth": 0.0},
        {"label": farther, "rank": 2, "relative_depth": 1.0},
    ]
    evidence = {
        "nearer": nearer,
        "farther": farther,
        "cue_type": cue,
        "evidence": f"{nearer} and {farther} have a directly visible {cue} depth cue.",
    }
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    sorted_labels = sorted((nearer, farther))
    selection_value = (
        f"gsa-depth-v4.2-pairwise|{image_sha}|{sorted_labels[0]}|"
        f"{sorted_labels[1]}|{nearer}>{farther}"
    )
    stable_hash = hashlib.sha256(selection_value.encode("utf-8")).hexdigest()
    return {
        "sample_id": f"generated-v42-{index:03d}",
        "pipeline": "S_spatial",
        "dimension": "S_spatial",
        "task_form": "depth",
        "depth_scope": "pairwise",
        "image_path": str(image),
        "scene_type": f"scene-{index % 2}",
        "source_bucket": "clean",
        "object_class": nearer,
        "confidence": 0.95,
        "label_version": "gsa-label-v4-depth-pairwise",
        "prompt_version": "gsa-depth-v4.2-pairwise",
        "payload": payload,
        "gsa_context": {
            "contract_version": "gsa-contract-v2",
            "primary": groundings[0],
            "groundings": groundings,
            "spatial": {
                "relation_mode": "pairwise",
                "depth_order": payload,
                "adjacent_depth_evidence": [evidence],
            },
        },
        "_v4_sources": {"image_sha256": image_sha},
        "_semantic_gate": {
            "depth_v4_pairwise": {
                "role_order_counterbalanced": True,
                "selection": {
                    "rule": "stable_hash_after_threshold",
                    "pair_index": 0,
                    "pairs_evaluated": 1,
                    "stable_hash": "0" * 64,
                    "gate_confidence": 0.95,
                },
                "generator": {
                    "orderable": True,
                    "reason_code": "supported_pair",
                    "confidence": 0.95,
                    "selected_pair": {**evidence, "confidence": 0.95},
                },
                "falsifier": {
                    "valid": True,
                    "reason_code": "independent_support",
                    "confidence": 0.95,
                    "pair_check": {
                        "nearer": nearer,
                        "farther": farther,
                        "cue_type": cue,
                        "evidence": (
                            f"Independent inspection confirms {nearer} lies before "
                            f"{farther} through the visible {cue} cue."
                        ),
                        "supported": True,
                        "confidence": 0.95,
                        "counterevidence": "",
                    },
                },
                "role_order_counterbalanced": True,
                "selection": {
                    "rule": "stable_hash_after_threshold",
                    "pair_index": 0,
                    "pairs_evaluated": 1,
                    "stable_hash": stable_hash,
                    "gate_confidence": 0.95,
                },
            },
        },
        "_verification": {"verdict": "verified_pair"},
    }


class DepthV42PairwiseValidatorTests(unittest.TestCase):
    def test_valid_record_and_fail_closed_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "image.jpg"
            image.write_bytes(b"image")
            base = pairwise_record(image, 0)
            generator_evidence = base["_semantic_gate"]["depth_v4_pairwise"][
                "generator"
            ]["selected_pair"]["evidence"]
            verifier_evidence = base["_semantic_gate"]["depth_v4_pairwise"][
                "falsifier"
            ]["pair_check"]["evidence"]
            self.assertNotEqual(generator_evidence, verifier_evidence)
            self.assertEqual(validate_depth_v4_record(base), (True, ""))

            mutations = [
                (lambda row: row.update(depth_scope="total_order"), "v4_pair_wrong_scope"),
                (
                    lambda row: row["gsa_context"]["groundings"].append({
                        "object_class": "third-object", "bbox": [700, 100, 800, 200]
                    }),
                    "v4_pair_entity_cardinality",
                ),
                (
                    lambda row: row["gsa_context"]["groundings"][1].update(
                        bbox=[0, 0, 1001, 10]
                    ),
                    "v4_pair_grounding_bbox",
                ),
                (
                    lambda row: row["gsa_context"]["spatial"].update(
                        relation_mode="total_order"
                    ),
                    "v4_pair_spatial_scope",
                ),
                (
                    lambda row: row["_semantic_gate"]["depth_v4_pairwise"][
                        "generator"
                    ].update(confidence=0.84),
                    "v4_pair_generator_gate",
                ),
                (
                    lambda row: row["_semantic_gate"]["depth_v4_pairwise"].update(
                        role_order_counterbalanced=False
                    ),
                    "v4_pair_role_order_not_counterbalanced",
                ),
                (
                    lambda row: row["_semantic_gate"]["depth_v4_pairwise"][
                        "selection"
                    ].update(rule="highest_confidence"),
                    "v4_pair_selection_gate",
                ),
                (
                    lambda row: row["_semantic_gate"]["depth_v4_pairwise"][
                        "selection"
                    ].update(stable_hash="b" * 64),
                    "v4_pair_selection_gate",
                ),
                (
                    lambda row: row["_semantic_gate"]["depth_v4_pairwise"][
                        "selection"
                    ].update(gate_confidence=0.84),
                    "v4_pair_selection_gate",
                ),
                (
                    lambda row: row["_semantic_gate"]["depth_v4_pairwise"][
                        "selection"
                    ].update(pairs_evaluated=0),
                    "v4_pair_selection_gate",
                ),
                (
                    lambda row: row["_semantic_gate"]["depth_v4_pairwise"][
                        "selection"
                    ].update(pair_index=1),
                    "v4_pair_selection_gate",
                ),
                (
                    lambda row: row["_v4_sources"].update(image_sha256="b" * 64),
                    "v4_pair_selection_gate",
                ),
                (
                    lambda row: row["_semantic_gate"]["depth_v4_pairwise"][
                        "falsifier"
                    ]["pair_check"].update(counterevidence="direction is ambiguous"),
                    "v4_pair_verifier_gate",
                ),
                (
                    lambda row: row["_semantic_gate"]["depth_v4_pairwise"][
                        "falsifier"
                    ].update(reason_code="no_counterexample"),
                    "v4_pair_verifier_gate",
                ),
                (
                    lambda row: row["_verification"].update(verdict="verified"),
                    "v4_pair_verification",
                ),
            ]
            for mutate, reason in mutations:
                row = copy.deepcopy(base)
                mutate(row)
                self.assertEqual(validate_depth_v4_record(row)[1], reason)

            unsupported = copy.deepcopy(base)
            text = "front-cup-0 is larger than rear-bowl-0 and therefore nearer."
            evidence = unsupported["gsa_context"]["spatial"][
                "adjacent_depth_evidence"
            ][0]
            evidence["evidence"] = text
            unsupported["_semantic_gate"]["depth_v4_pairwise"]["generator"][
                "selected_pair"
            ]["evidence"] = text
            self.assertEqual(
                validate_depth_v4_record(unsupported)[1],
                "v4_pair_unsupported_heuristic",
            )

            verifier_unsupported = copy.deepcopy(base)
            verifier_unsupported["_semantic_gate"]["depth_v4_pairwise"][
                "falsifier"
            ]["pair_check"]["evidence"] = (
                "front-cup-0 is larger than rear-bowl-0 and therefore nearer."
            )
            self.assertEqual(
                validate_depth_v4_record(verifier_unsupported)[1],
                "v4_pair_verifier_unsupported_heuristic",
            )

            verifier_reversed = copy.deepcopy(base)
            check = verifier_reversed["_semantic_gate"]["depth_v4_pairwise"][
                "falsifier"
            ]["pair_check"]
            check["nearer"], check["farther"] = check["farther"], check["nearer"]
            self.assertEqual(
                validate_depth_v4_record(verifier_reversed)[1],
                "v4_pair_verifier_gate",
            )


class DepthV42PairwisePackTests(unittest.TestCase):
    def _write_generation(
        self,
        root: Path,
        records: list[dict],
        overrides: dict | None = None,
    ) -> tuple[Path, Path]:
        generation = root / "full-generation"
        source = generation / "labels/S_spatial/depth.jsonl"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("".join(json.dumps(row) + "\n" for row in records))
        accepted_pool = []
        accepted_by_cue: dict[str, int] = {}
        accepted_by_scene: dict[str, int] = {}
        accepted_by_source: dict[str, int] = {}
        candidate_hashes = []
        for row in records:
            cue = row["gsa_context"]["spatial"]["adjacent_depth_evidence"][0][
                "cue_type"
            ]
            image_sha = row["_v4_sources"]["image_sha256"]
            candidate_hashes.append(image_sha)
            accepted_by_cue[cue] = accepted_by_cue.get(cue, 0) + 1
            scene = str(row.get("scene_type") or "unknown")
            source_bucket = str(row.get("source_bucket") or "unknown")
            accepted_by_scene[scene] = accepted_by_scene.get(scene, 0) + 1
            accepted_by_source[source_bucket] = (
                accepted_by_source.get(source_bucket, 0) + 1
            )
            accepted_pool.append({
                "image_path": row["image_path"],
                "image_sha256": image_sha,
                "sample_id": row["sample_id"],
                "label_sha256": content_hash(row),
                "selected_pair": [item["label"] for item in row["payload"]],
                "selection_score": row["confidence"],
                "cue_type": cue,
            })
        cue_pass = all(
            accepted_by_cue.get(cue, 0) >= 8
            for cue in ("occlusion", "contact_layering")
        )
        report = {
            "schema_version": 1,
            "protocol": "depth-v4.2-pairwise-generation",
            "depth_scope": "pairwise",
            "prompt_version": "gsa-depth-v4.2-pairwise",
            "label_version": "gsa-label-v4-depth-pairwise",
            "stage": "full",
            "protocol_artifact": True,
            "artifact_kind": "generation_pool",
            "candidate_limit_images": None,
            "minimum_audit_images": 40,
            "candidate_images": len(candidate_hashes),
            "candidate_image_sha256s": candidate_hashes,
            "candidate_set_sha256": content_hash(candidate_hashes),
            "attempted_images": len(candidate_hashes),
            "accepted_pool_images": len(records),
            "emitted_records": len(records),
            "pairs_attempted": len(records),
            "pairs_accepted": len(records),
            "pair_selection_rule": "stable_hash_after_threshold",
            "role_order_counterbalanced": True,
            "audit_pool_eligible": len(records) >= 40 and cue_pass,
            "system_errors": {
                "parser_rejections": 0,
                "runtime_or_record_rejections": 0,
                "total": 0,
            },
            "rejections": {},
            "pair_rejections": {},
            "accepted_by_scene": dict(sorted(accepted_by_scene.items())),
            "accepted_by_source": dict(sorted(accepted_by_source.items())),
            "accepted_by_cue": dict(sorted(accepted_by_cue.items())),
            "cue_gate": {
                "minimum_images": {"contact_layering": 8, "occlusion": 8},
                "passed": cue_pass,
            },
            "source_provenance": {},
            "prompt_hashes": {
                "generator_template_sha256": (
                    "b42e4a696d363b5f634842ec0c1a0cbceffe52ff642e60c3a18ca79922afa751"
                ),
                "falsifier_protocol_sha256": (
                    "3fbf3161547157cc3db1fbd543e2180b4d3024573a9ef14831f5b323893d170e"
                ),
            },
            "accepted_pool": accepted_pool,
            "labels_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "accepted_pool_sha256": content_hash(accepted_pool),
        }
        if overrides:
            report.update(overrides)
        report_path = generation / "generation_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        return source, report_path

    def _write_exclusions(self, root: Path) -> list[str]:
        args: list[str] = []
        for schema in (1, 3):
            path = root / f"exclude-{schema}.json"
            path.write_text(json.dumps({"schema_version": schema, "selected": []}))
            args.extend(["--exclude-manifest", str(path)])
        return args

    def _build(
        self,
        root: Path,
        records: list[dict],
        extra: list[str] | None = None,
        report_overrides: dict | None = None,
        include_report: bool = True,
    ):
        source, report = self._write_generation(root, records, report_overrides)
        pack = root / "pack"
        report_args = ["--generation-report", str(report)] if include_report else []
        result = subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--source",
                str(source),
                *report_args,
                "--out",
                str(pack),
                *self._write_exclusions(root),
                *(extra or []),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return pack, result

    def test_pairwise_manifest_and_compiler_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index in range(40):
                image = root / "images" / f"image-{index:02d}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"image-{index}".encode())
                records.append(pairwise_record(image, index))
            pack, built = self._build(root, records)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            manifest = json.loads((pack / "manifest.json").read_text())
            self.assertEqual(manifest["protocol"], "depth-pairwise-calibration-v1")
            self.assertEqual(manifest["round"], "v4-depth-pairwise")
            self.assertEqual(manifest["depth_scope"], "pairwise")
            self.assertEqual(
                manifest["generation_run"]["report_path"],
                str((root / "full-generation/generation_report.json").resolve()),
            )
            self.assertEqual(len(manifest["generation_run"]["report_sha256"]), 64)
            self.assertEqual(
                len(manifest["generation_run"]["run_evidence_sha256"]), 64
            )
            self.assertGreaterEqual(manifest["cue_counts"]["occlusion"], 8)
            self.assertGreaterEqual(manifest["cue_counts"]["contact_layering"], 8)
            self.assertEqual(sum(manifest["cue_counts"].values()), 40)
            self.assertEqual(sum(manifest["candidate_cue_counts"].values()), 40)
            self.assertTrue(all(
                item["depth_scopes"] == {"depth": "pairwise"}
                for item in manifest["selected"]
            ))

            for root_name, key in (("annotator_a", "a"), ("annotator_b", "b")):
                labels = [
                    json.loads(line)
                    for line in (
                        pack / root_name / "labels/S_spatial/depth.jsonl"
                    ).read_text().splitlines()
                ]
                annotation_path = pack / root_name / "annotations/annotations.jsonl"
                annotation_path.write_text("".join(json.dumps({
                    "correct": True,
                    "annotator_key": key,
                    "label_snapshot": row,
                }) + "\n" for row in labels))
            report = root / "report.json"
            compiled = subprocess.run(
                [sys.executable, str(COMPILER), "--pack", str(pack), "--out", str(report)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            payload = json.loads(report.read_text())
            self.assertEqual(payload["verdict"], "PASS")
            self.assertEqual(payload["depth_scope"], "pairwise")

    def test_compiler_recomputes_and_enforces_pairwise_cue_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index in range(40):
                image = root / "images" / f"image-{index:02d}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"image-{index}".encode())
                records.append(pairwise_record(image, index))
            pack, built = self._build(root, records)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)

            rows_by_id: dict[str, dict] = {}
            for root_name in ("annotator_a", "annotator_b"):
                label_path = pack / root_name / "labels/S_spatial/depth.jsonl"
                labels = [json.loads(line) for line in label_path.read_text().splitlines()]
                for row in labels:
                    spatial = row["gsa_context"]["spatial"]
                    generator = row["_semantic_gate"]["depth_v4_pairwise"]["generator"]
                    verifier = row["_semantic_gate"]["depth_v4_pairwise"]["falsifier"]
                    if spatial["adjacent_depth_evidence"][0]["cue_type"] == "contact_layering":
                        spatial["adjacent_depth_evidence"][0]["cue_type"] = "occlusion"
                        generator["selected_pair"]["cue_type"] = "occlusion"
                        verifier["pair_check"]["cue_type"] = "occlusion"
                    self.assertEqual(validate_depth_v4_record(row), (True, ""))
                    rows_by_id[row["sample_id"]] = row
                label_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in labels)
                )

                annotation_path = pack / root_name / "annotations/annotations.jsonl"
                annotation_path.write_text(
                    "".join(json.dumps({
                        "correct": True,
                        "annotator_key": root_name,
                        "label_snapshot": row,
                    }) + "\n" for row in labels)
                )

            manifest_path = pack / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            cue_counts: dict[str, int] = {}
            pairs = []
            for item in manifest["selected"]:
                sample_id = item["sample_ids"]["depth"]
                row = rows_by_id[sample_id]
                cue = row["gsa_context"]["spatial"]["adjacent_depth_evidence"][0][
                    "cue_type"
                ]
                digest = content_hash(row)
                item["label_hashes"]["depth"] = digest
                item["depth_cues"] = {"depth": cue}
                cue_counts[cue] = cue_counts.get(cue, 0) + 1
                pairs.append({"sample_id": sample_id, "label_sha256": digest})
            manifest["per_form_label_hashes"]["depth"] = content_hash(pairs)
            manifest["cue_counts"] = dict(sorted(cue_counts.items()))
            manifest["candidate_cue_counts"] = dict(sorted(cue_counts.items()))
            manifest["integrity"]["selected_sha256"] = content_hash(manifest["selected"])
            manifest["integrity"]["manifest_payload_sha256"] = manifest_payload_hash(
                manifest
            )
            manifest_path.write_text(json.dumps(manifest, indent=2))

            report = root / "quota-report.json"
            compiled = subprocess.run(
                [sys.executable, str(COMPILER), "--pack", str(pack), "--out", str(report)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(compiled.returncode, 2, compiled.stdout + compiled.stderr)
            payload = json.loads(report.read_text())
            self.assertIn(
                "v4 pairwise audit requires at least 8 contact_layering cases",
                payload["errors"],
            )
            self.assertTrue(any(
                "packed label payload is not derived from source" in error
                for error in payload["errors"]
            ))

    def test_builder_requires_an_eligible_full_generation_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index in range(40):
                image = root / f"image-{index}.jpg"
                image.write_bytes(f"image-{index}".encode())
                records.append(pairwise_record(image, index))
            _, missing = self._build(root, records, include_report=False)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("requires --generation-report", missing.stderr)

        mutations = (
            ({"stage": "canary"}, "stage='canary'"),
            ({"protocol_artifact": False}, "protocol_artifact=False"),
            ({"audit_pool_eligible": False}, "audit_pool_eligible=False"),
            ({
                "system_errors": {
                    "parser_rejections": 1,
                    "runtime_or_record_rejections": 0,
                    "total": 1,
                },
            }, "zero internally consistent system_errors"),
        )
        for overrides, expected in mutations:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                records = []
                for index in range(40):
                    image = root / f"image-{index}.jpg"
                    image.write_bytes(f"image-{index}".encode())
                    records.append(pairwise_record(image, index))
                _, result = self._build(root, records, report_overrides=overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_compiler_rejects_self_consistent_failed_run_rebinding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index in range(40):
                image = root / f"image-{index}.jpg"
                image.write_bytes(f"image-{index}".encode())
                records.append(pairwise_record(image, index))
            pack, built = self._build(root, records)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)

            manifest_path = pack / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            report_path = Path(manifest["generation_run"]["report_path"])
            generation_report = json.loads(report_path.read_text())
            generation_report["audit_pool_eligible"] = False
            report_path.write_text(json.dumps(generation_report, indent=2))
            manifest["generation_run"]["report_sha256"] = hashlib.sha256(
                report_path.read_bytes()
            ).hexdigest()
            manifest["generation_run"]["run_evidence_sha256"] = (
                generation_run_evidence_hash(generation_report)
            )
            manifest["integrity"]["manifest_payload_sha256"] = manifest_payload_hash(
                manifest
            )
            manifest_path.write_text(json.dumps(manifest, indent=2))

            report = root / "compile-report.json"
            compiled = subprocess.run(
                [sys.executable, str(COMPILER), "--pack", str(pack), "--out", str(report)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(compiled.returncode, 2, compiled.stdout + compiled.stderr)
            payload = json.loads(report.read_text())
            self.assertTrue(any(
                "audit_pool_eligible=False, expected True" in error
                for error in payload["errors"]
            ))

    def test_builder_rejects_protocol_mix_and_wrong_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index in range(40):
                image = root / f"pair-{index}.jpg"
                image.write_bytes(f"pair-{index}".encode())
                records.append(pairwise_record(image, index))
            legacy = copy.deepcopy(records[0])
            legacy["sample_id"] = "legacy-valid-record"
            legacy["depth_scope"] = "total_order"
            legacy["label_version"] = "gsa-label-v4-depth"
            legacy["prompt_version"] = "gsa-depth-v4.1"
            legacy["gsa_context"]["spatial"].pop("relation_mode")
            legacy["_semantic_gate"] = {
                "depth_v4": {
                    "generator": {
                        "orderable": True,
                        "reason_code": "supported",
                        "confidence": 0.95,
                        "order": legacy["payload"],
                        "adjacent_evidence": legacy["gsa_context"]["spatial"][
                            "adjacent_depth_evidence"
                        ],
                    },
                    "falsifier": {
                        "valid": True,
                        "reason_code": "no_counterexample",
                        "confidence": 0.95,
                        "pair_checks": [{
                            **legacy["_semantic_gate"]["depth_v4_pairwise"][
                                "falsifier"
                            ]["pair_check"],
                        }],
                    },
                },
            }
            legacy["_verification"] = {"verdict": "verified"}
            mixed_root = root / "mixed"
            mixed_root.mkdir()
            _, mixed = self._build(mixed_root, records + [legacy])
            self.assertNotEqual(mixed.returncode, 0)
            self.assertIn("mixes total-order and pairwise", mixed.stderr)

            round_root = root / "round"
            round_root.mkdir()
            _, wrong_round = self._build(
                round_root, records, ["--round", "v4-depth"]
            )
            self.assertNotEqual(wrong_round.returncode, 0)
            self.assertIn("requires --round v4-depth-pairwise", wrong_round.stderr)

    def test_builder_rejects_insufficient_pairwise_cue_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index in range(40):
                image = root / f"image-{index}.jpg"
                image.write_bytes(f"image-{index}".encode())
                row = pairwise_record(image, index)
                evidence = row["gsa_context"]["spatial"]["adjacent_depth_evidence"][0]
                evidence["cue_type"] = "occlusion"
                row["_semantic_gate"]["depth_v4_pairwise"]["generator"][
                    "selected_pair"
                ]["cue_type"] = "occlusion"
                records.append(row)
            _, result = self._build(root, records)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "generation source requires at least 8 contact_layering records",
                result.stderr,
            )


if __name__ == "__main__":
    unittest.main()
