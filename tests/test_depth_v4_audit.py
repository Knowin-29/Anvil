from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import compile_semantic_composite_gate as composite
from tools import generate_depth_v4 as generator
from trains.flywheel.validators import validate_depth_v4_record

BUILDER = ROOT / "trains/flywheel/build_depth_semantic_audit_pack.py"
COMPILER = ROOT / "eval/compile_semantic_audit.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def v4_record(image: Path, index: int) -> dict:
    near = f"near-{index}"
    far = f"far-{index}"
    entities = [
        {"object_class": near, "bbox": [100, 100, 300, 300]},
        {"object_class": far, "bbox": [400, 400, 600, 600]},
    ]
    order = [
        {"label": near, "rank": 1, "relative_depth": 0.0},
        {"label": far, "rank": 2, "relative_depth": 1.0},
    ]
    evidence = [{
        "nearer": near,
        "farther": far,
        "cue_type": "occlusion",
        "evidence": f"{near} visibly occludes part of the {far} boundary.",
    }]
    generator_gate = {
        "orderable": True,
        "reason_code": "supported",
        "confidence": 0.95,
        "order": order,
        "adjacent_evidence": evidence,
    }
    falsifier_gate = {
        "valid": True,
        "reason_code": "no_counterexample",
        "confidence": 0.95,
        "pair_checks": [{
            "nearer": near,
            "farther": far,
            "supported": True,
            "confidence": 0.95,
            "counterevidence": "",
        }],
    }
    return {
        "sample_id": f"generated-v4-{index:03d}",
        "pipeline": "S_spatial",
        "dimension": "S_spatial",
        "task_form": "depth",
        "image_path": str(image),
        "scene_type": f"scene-{index % 2}",
        "source_bucket": "clean",
        "object_class": near,
        "label_version": "gsa-label-v4-depth",
        "prompt_version": "gsa-depth-v4.1",
        "payload": order,
        "gsa_context": {
            "contract_version": "gsa-contract-v2",
            "primary": entities[0],
            "groundings": entities,
            "spatial": {
                "depth_order": order,
                "adjacent_depth_evidence": evidence,
            },
        },
        "_semantic_gate": {
            "depth_v4": {
                "generator": generator_gate,
                "falsifier": falsifier_gate,
            }
        },
        "_verification": {"verdict": "verified"},
    }


class DepthV4GeneratorTests(unittest.TestCase):
    def test_strict_generator_and_falsifier_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "image.jpg"
            image.write_bytes(b"image")
            record = v4_record(image, 0)
            entities = [
                {"label": item["object_class"], "bbox_2d": item["bbox"]}
                for item in record["gsa_context"]["groundings"]
            ]
            primary = record["_semantic_gate"]["depth_v4"]["generator"]
            falsifier = record["_semantic_gate"]["depth_v4"]["falsifier"]
            self.assertEqual(generator.validate_generator_response(primary, entities), (True, ""))
            self.assertEqual(
                generator.validate_falsifier_response(falsifier, primary["order"]),
                (True, ""),
            )
            self.assertEqual(validate_depth_v4_record(record), (True, ""))
            primary["adjacent_evidence"][0]["evidence"] = (
                "near-0 is larger than far-0, so it must be nearer."
            )
            self.assertEqual(
                generator.validate_generator_response(primary, entities)[1],
                "unsupported_depth_heuristic",
            )
            self.assertEqual(validate_depth_v4_record(record)[1], "v4_unsupported_heuristic")

    def test_full_pool_canary_and_minimum_audit_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = []
            for index in range(45):
                image = root / f"image-{index}.jpg"
                image.write_bytes(f"image-{index}".encode())
                record = v4_record(image, index)
                entities = [
                    {"label": item["object_class"], "bbox_2d": item["bbox"]}
                    for item in record["gsa_context"]["groundings"]
                ]
                detection = {
                    "sample_id": f"d-{index}",
                    "task_form": "detection",
                    "image_path": str(image),
                    "source_bucket": "clean",
                    "payload": [
                        {"label": item["label"], "bbox_2d": item["bbox_2d"]}
                        for item in entities
                    ],
                }
                candidates.append({
                    "image_path": str(image),
                    "image_sha256": sha256(image),
                    "entities": entities,
                    "detection": detection,
                    "bbox": {
                        "sample_id": f"b-{index}",
                        "task_form": "bbox",
                        "image_path": str(image),
                        "payload": [detection["payload"][0]],
                    },
                    "scene_type": "clean",
                })

            def fake_client(_image, prompt, **_kwargs):
                index = int(Path(_image).stem.split("-")[1])
                record = v4_record(Path(_image), index)
                gate = record["_semantic_gate"]["depth_v4"]
                return gate["falsifier"] if prompt.startswith("Act as an independent") else gate["generator"]

            out = root / "out"
            report, rc = generator.run_generation(
                candidates, {}, out, workers=8, client=fake_client
            )
            self.assertEqual(rc, 0)
            self.assertEqual(report["attempted_images"], 45)
            self.assertEqual(report["accepted_pool_images"], 45)
            self.assertEqual(report["emitted_records"], 45)
            self.assertIsNone(report["candidate_limit_images"])
            self.assertTrue(report["audit_pool_eligible"])
            labels = (out / "labels/S_spatial/depth.jsonl").read_text().splitlines()
            self.assertEqual(len(labels), 45)
            self.assertIn("image-40.jpg", (out / "generation_report.json").read_text())

            short_out = root / "short-out"
            short_report, short_rc = generator.run_generation(
                candidates[:39], {}, short_out, workers=8, client=fake_client
            )
            self.assertEqual(short_rc, 2)
            self.assertEqual(short_report["attempted_images"], 39)
            self.assertEqual(short_report["accepted_pool_images"], 39)
            self.assertFalse(short_report["audit_pool_eligible"])
            self.assertEqual(
                len((short_out / "labels/S_spatial/depth.jsonl").read_text().splitlines()),
                39,
            )
            self.assertTrue((short_out / "generation_report.json").is_file())

            canary_out = root / "canary-out"
            canary_report, canary_rc = generator.run_generation(
                candidates[:8],
                {},
                canary_out,
                workers=8,
                client=fake_client,
                protocol_artifact=False,
                stage="canary",
            )
            self.assertEqual(canary_rc, 0)
            self.assertEqual(canary_report["attempted_images"], 8)
            self.assertEqual(canary_report["accepted_pool_images"], 8)
            self.assertEqual(canary_report["candidate_limit_images"], 8)
            self.assertFalse(canary_report["protocol_artifact"])

    def test_check_only_never_imports_client_or_creates_api_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "index.txt"
            lines = []
            for number in range(40):
                image = root / f"fresh-{number}.jpg"
                image.write_bytes(f"fresh-{number}".encode())
                lines.append(f"{image}\tclean\n")
            index.write_text("".join(lines), encoding="utf-8")
            v2 = root / "v2.json"
            v3 = root / "v3.json"
            v2.write_text(json.dumps({"schema_version": 1, "selected": []}))
            v3.write_text(json.dumps({"schema_version": 3, "selected": []}))
            out = root / "out"
            env = {
                **os.environ,
                "GSA_VLM_URL": "http://127.0.0.1:1/never-called",
                "GSA_VLM_CACHE": str(root / "cache"),
                "GSA_VLM_REQUEST_LOG": str(root / "requests.jsonl"),
            }
            result = subprocess.run([
                sys.executable,
                str(ROOT / "tools/generate_depth_v4.py"),
                "--index", str(index),
                "--exclude-manifest", str(v2),
                "--exclude-manifest", str(v3),
                "--out", str(out),
                "--check-only",
            ], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["eligible_images"], 40)
            self.assertEqual(plan["groundings_to_generate"], 40)
            self.assertFalse(out.exists())
            self.assertFalse((root / "cache").exists())
            self.assertFalse((root / "requests.jsonl").exists())


class DepthV4PackCompilerTests(unittest.TestCase):
    def make_pack(self, root: Path) -> Path:
        source = root / "source.jsonl"
        records = []
        images = []
        for index in range(42):
            image = root / "images" / f"image-{index:02d}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image-{index}".encode())
            images.append(image)
            records.append(v4_record(image, index))
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
        v2 = root / "v2.json"
        v3 = root / "v3.json"
        v2.write_text(json.dumps({
            "schema_version": 1,
            "selected": [{
                "image_path": str(images[0]),
                "sample_ids": {"depth": "old-v2-depth"},
            }],
        }))
        v3.write_text(json.dumps({
            "schema_version": 3,
            "selected": [{
                "image_path": str(images[1]),
                "image_sha256": sha256(images[1]),
                "sample_ids": {"depth": "old-v3-depth"},
            }],
        }))
        pack = root / "pack"
        result = subprocess.run([
            sys.executable, str(BUILDER),
            "--source", str(source),
            "--out", str(pack),
            "--exclude-manifest", str(v2),
            "--exclude-manifest", str(v3),
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for root_name, key in (("annotator_a", "reviewer-a"), ("annotator_b", "reviewer-b")):
            labels = [
                json.loads(line)
                for line in (
                    pack / root_name / "labels/S_spatial/depth.jsonl"
                ).read_text().splitlines()
            ]
            annotations = pack / root_name / "annotations/annotations.jsonl"
            annotations.write_text(
                "".join(json.dumps({
                    "correct": True,
                    "annotator_key": key,
                    "label_snapshot": row,
                }) + "\n" for row in labels),
                encoding="utf-8",
            )
        return pack

    def compile(self, pack: Path, out: Path, extra: list[str] | None = None):
        return subprocess.run([
            sys.executable, str(COMPILER),
            "--pack", str(pack),
            "--out", str(out),
            *(extra or []),
        ], cwd=ROOT, capture_output=True, text=True, check=False)

    def test_builder_exclusion_integrity_and_compiler_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self.make_pack(root)
            manifest = json.loads((pack / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 4)
            self.assertEqual(manifest["forms"], ["depth"])
            self.assertEqual(len(manifest["selected"]), 40)
            self.assertTrue(all(
                item["sample_ids"]["depth"].startswith("v4_depth_")
                and item["sample_ids"]["depth"] != item["source_sample_ids"]["depth"]
                for item in manifest["selected"]
            ))
            result = self.compile(pack, root / "report.json")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["verdict"], "PASS")
            self.assertEqual(report["by_form"]["depth"]["paired"], 40)

    def test_compiler_rejects_threshold_lowering_and_annotator_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self.make_pack(root)
            lowered = self.compile(
                pack, root / "lowered.json",
                ["--min-accuracy", "0.7", "--min-agreement", "0.7"],
            )
            self.assertEqual(lowered.returncode, 2)
            self.assertIn("thresholds cannot be lower", lowered.stdout)
            path = pack / "annotator_b/annotations/annotations.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row["annotator_key"] = "reviewer-a"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            aliased = self.compile(pack, root / "aliased.json")
            self.assertEqual(aliased.returncode, 2)
            self.assertIn("must use different annotator_key", aliased.stdout)

    def test_builder_rejects_audit_only_or_legacy_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.jpg"
            image.write_bytes(b"image")
            row = v4_record(image, 0)
            row["_audit_only"] = True
            source = root / "source.jsonl"
            source.write_text(json.dumps(row) + "\n")
            manifests = []
            for schema in (1, 3):
                path = root / f"exclude-{schema}.json"
                path.write_text(json.dumps({"schema_version": schema, "selected": []}))
                manifests.extend(["--exclude-manifest", str(path)])
            result = subprocess.run([
                sys.executable, str(BUILDER),
                "--source", str(source),
                "--out", str(root / "pack"),
                *manifests,
            ], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no structurally valid depth records", result.stderr)


class CompositeGateTests(unittest.TestCase):
    @staticmethod
    def row(eligible: bool = True, unresolved: int = 0) -> dict:
        return {
            "expected": 40,
            "annotator_a": 40,
            "annotator_b": 40,
            "paired": 40,
            "agreed": 36,
            "adjudicated": 4 if unresolved == 0 else 0,
            "unresolved": unresolved,
            "raw_agreement": 0.9,
            "resolved_accuracy": 0.9,
            "eligible": eligible,
        }

    def test_v3_failed_depth_is_replaced_but_unresolved_nondepth_blocks(self):
        v3 = {
            "schema_version": 3,
            "verdict": "BLOCKED",
            "errors": [],
            "manifest": "/absent/v3.json",
            "by_form": {form: self.row() for form in composite.NON_DEPTH_FORMS},
            "unresolved_sample_ids": [],
        }
        v3["by_form"]["depth"] = {
            **self.row(eligible=False, unresolved=14),
            "raw_agreement": 0.65,
        }
        v4 = {
            "schema_version": 4,
            "verdict": "PASS",
            "errors": [],
            "manifest": "/absent/v4.json",
            "by_form": {"depth": self.row()},
            "unresolved_sample_ids": [],
        }
        args = argparse.Namespace(
            v3_pack=Path("/v3"),
            v3_adjudication=None,
            v4_pack=Path("/v4"),
            v4_adjudication=None,
        )
        with patch.object(composite, "compile_component", side_effect=[v3, v4]):
            report, rc = composite.compile_composite(args)
        self.assertEqual(rc, 0)
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["by_form"]["depth"]["source"], "v4")
        self.assertEqual(set(report["component_report_hashes"]), {
            "v3_report_sha256", "v4_report_sha256",
        })

        v3["by_form"]["action_cot"] = self.row(eligible=False, unresolved=1)
        with patch.object(composite, "compile_component", side_effect=[v3, v4]):
            report, rc = composite.compile_composite(args)
        self.assertEqual(rc, 2)
        self.assertTrue(any(
            blocker.startswith("v3:action_cot:unresolved=1")
            for blocker in report["blockers"]
        ))


if __name__ == "__main__":
    unittest.main()
