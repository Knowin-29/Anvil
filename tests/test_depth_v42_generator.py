from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import generate_depth_v42_pairwise as pairwise
from trains.flywheel.validators import validate_depth_v4_record


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_candidate(root: Path, index: int, entity_count: int = 2) -> dict[str, Any]:
    image = root / f"kit-{index:03d}.jpg"
    image.write_bytes(f"image-{index}".encode())
    entities = [
        {
            "label": f"entity-{index}-{entity_index}",
            "bbox_2d": [
                50 + 100 * entity_index,
                50 + 100 * entity_index,
                130 + 100 * entity_index,
                130 + 100 * entity_index,
            ],
        }
        for entity_index in range(entity_count)
    ]
    payload = [
        {"label": entity["label"], "bbox_2d": entity["bbox_2d"]}
        for entity in entities
    ]
    detection = {
        "sample_id": f"d-{index}",
        "task_form": "detection",
        "image_path": str(image),
        "scene_type": f"scene-{index % 2}",
        "source_bucket": None,
        "payload": payload,
    }
    return {
        "image_path": str(image),
        "image_sha256": sha256(image),
        "entities": entities,
        "detection": detection,
        "bbox": {
            "sample_id": f"b-{index}",
            "task_form": "bbox",
            "image_path": str(image),
            "payload": [payload[0]],
        },
        "scene_type": f"scene-{index % 2}",
    }


def prompt_entities(prompt: str, marker: str, suffix: str) -> list[dict[str, Any]]:
    return json.loads(prompt.split(marker, 1)[1].split(suffix, 1)[0])


def generator_response(
    nearer: str, farther: str, cue: str, confidence: float = 0.95
) -> dict[str, Any]:
    return {
        "orderable": True,
        "reason_code": "supported_pair",
        "confidence": confidence,
        "selected_pair": {
            "nearer": nearer,
            "farther": farther,
            "cue_type": cue,
            "evidence": (
                f"{nearer} visibly overlaps a boundary of {farther} with direct {cue} support."
            ),
            "confidence": confidence,
        },
    }


def verifier_response(
    nearer: str, farther: str, cue: str, confidence: float = 0.94
) -> dict[str, Any]:
    return {
        "valid": True,
        "reason_code": "independent_support",
        "confidence": confidence,
        "pair_check": {
            "nearer": nearer,
            "farther": farther,
            "cue_type": cue,
            "evidence": (
                f"{nearer} independently overlaps the visible boundary of {farther} via {cue}."
            ),
            "supported": True,
            "confidence": confidence,
            "counterevidence": "",
        },
    }


class FakePairClient:
    def __init__(self) -> None:
        self.generator_pairs: list[list[str]] = []
        self.verifier_pairs: list[list[str]] = []
        self.verifier_prompts: list[str] = []

    def __call__(self, image: str, prompt: str, **_kwargs: Any) -> dict[str, Any]:
        index = int(Path(image).stem.split("-")[1])
        cue = (
            "occlusion",
            "contact_layering",
            "support_plane_perspective",
        )[index % 3]
        if prompt.startswith("You are the primary pairwise"):
            entities = prompt_entities(prompt, "Pair entities: ", ". Decide whether")
            labels = [entity["label"] for entity in entities]
            self.generator_pairs.append(labels)
            nearer, farther = sorted(labels)
            return generator_response(nearer, farther, cue)
        if prompt.startswith("Act as a direction-blind pairwise"):
            entities = prompt_entities(prompt, "Frozen pair: ", ". Accept only")
            labels = [entity["label"] for entity in entities]
            self.verifier_pairs.append(labels)
            self.verifier_prompts.append(prompt)
            nearer, farther = sorted(labels)
            return verifier_response(nearer, farther, cue)
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


class DepthV42GeneratorTests(unittest.TestCase):
    def test_all_pairs_counterbalanced_and_stably_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = make_candidate(Path(tmp), 0, entity_count=3)
            client = FakePairClient()
            record, stats = pairwise.generate_one(candidate, client)
            self.assertIsNotNone(record)
            self.assertEqual(stats["pairs_attempted"], 3)
            self.assertEqual(stats["pairs_accepted"], 3)
            self.assertTrue(stats["multiple_verified_pairs"])
            self.assertTrue(stats["uniform_confidence"])
            self.assertEqual(len(client.generator_pairs), 3)
            self.assertEqual(
                client.verifier_pairs,
                [list(reversed(labels)) for labels in client.generator_pairs],
            )
            expected_keys = []
            for labels in client.generator_pairs:
                response = generator_response(
                    min(labels), max(labels), "occlusion"
                )
                expected_keys.append(pairwise._selection_key(candidate, response))
            expected_index = min(range(3), key=lambda index: expected_keys[index])
            self.assertEqual(stats["selected_pair_index"], expected_index)
            selection = record["_semantic_gate"]["depth_v4_pairwise"]["selection"]
            self.assertEqual(selection["rule"], "stable_hash_after_threshold")
            self.assertEqual(selection["pair_index"], expected_index)
            self.assertTrue(
                record["_semantic_gate"]["depth_v4_pairwise"][
                    "role_order_counterbalanced"
                ]
            )
            self.assertEqual(record["source_bucket"], "kitchen")
            self.assertEqual(validate_depth_v4_record(record), (True, ""))

    def test_direction_blind_verifier_has_own_strict_evidence(self):
        entities = [
            {"label": "near", "bbox_2d": [10, 10, 100, 100]},
            {"label": "far", "bbox_2d": [200, 200, 300, 300]},
        ]
        response = verifier_response("near", "far", "occlusion")
        self.assertEqual(
            pairwise.validate_falsifier_response(response, entities), (True, "")
        )
        response["pair_check"]["evidence"] = (
            "near looks larger than far, so near must be in front."
        )
        self.assertEqual(
            pairwise.validate_falsifier_response(response, entities)[1],
            "falsifier_unsupported_depth_heuristic",
        )
        prompt = pairwise.falsifier_prompt(list(reversed(entities)))
        self.assertNotIn("Proposed direction", prompt)
        self.assertNotIn("selected_pair", prompt)

    def test_generation_report_and_cue_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [make_candidate(root, index) for index in range(40)]
            report, rc = pairwise.run_generation(
                candidates,
                {"fixture": True},
                root / "full",
                workers=8,
                client=FakePairClient(),
                minimum_audit_images=40,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(report["attempted_images"], 40)
            self.assertEqual(report["accepted_pool_images"], 40)
            self.assertEqual(report["pairs_attempted"], 40)
            self.assertEqual(report["pairs_accepted"], 40)
            self.assertEqual(report["accepted_by_scene"], {"scene-0": 20, "scene-1": 20})
            self.assertEqual(report["accepted_by_source"], {"kitchen": 40})
            self.assertGreaterEqual(report["accepted_by_cue"]["occlusion"], 8)
            self.assertGreaterEqual(
                report["accepted_by_cue"]["contact_layering"], 8
            )
            self.assertTrue(report["cue_gate"]["passed"])
            self.assertEqual(report["system_errors"]["total"], 0)
            self.assertTrue(report["audit_pool_eligible"])
            self.assertEqual(report["pair_selection_rule"], "stable_hash_after_threshold")
            self.assertTrue(report["role_order_counterbalanced"])

    def test_full_gate_rejects_missing_cue_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [make_candidate(root, index) for index in range(40)]

            class OcclusionOnly(FakePairClient):
                def __call__(self, image: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
                    response = super().__call__(image, prompt, **kwargs)
                    if "selected_pair" in response:
                        response["selected_pair"]["cue_type"] = "occlusion"
                        labels = response["selected_pair"]
                        labels["evidence"] = (
                            f"{labels['nearer']} visibly occludes the boundary of {labels['farther']}."
                        )
                    else:
                        check = response["pair_check"]
                        check["cue_type"] = "occlusion"
                        check["evidence"] = (
                            f"{check['nearer']} independently occludes the boundary of {check['farther']}."
                        )
                    return response

            report, rc = pairwise.run_generation(
                candidates,
                {},
                root / "full",
                workers=8,
                client=OcclusionOnly(),
                minimum_audit_images=40,
            )
            self.assertEqual(rc, 2)
            self.assertEqual(report["accepted_pool_images"], 40)
            self.assertFalse(report["cue_gate"]["passed"])
            self.assertFalse(report["audit_pool_eligible"])

    def test_verifier_retries_only_a_transient_parse_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = make_candidate(Path(tmp), 7)
            base = FakePairClient()
            verifier_calls: list[bool] = []

            def client(image: str, prompt: str, **kwargs: Any) -> dict[str, Any] | None:
                if prompt.startswith("Act as a direction-blind pairwise"):
                    verifier_calls.append(bool(kwargs["use_cache"]))
                    if len(verifier_calls) == 1:
                        return None
                return base(image, prompt, **kwargs)

            record, stats = pairwise.generate_one(candidate, client)
            self.assertIsNotNone(record)
            self.assertEqual(stats["pairs_accepted"], 1)
            self.assertEqual(verifier_calls, [True, False])

    def test_canary_blocks_parser_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [make_candidate(root, index) for index in range(8)]

            def invalid_client(_image: str, _prompt: str, **_kwargs: Any) -> None:
                return None

            report, rc = pairwise.run_generation(
                candidates,
                {},
                root / "canary",
                workers=4,
                client=invalid_client,
                protocol_artifact=False,
                stage="canary",
            )
            self.assertEqual(rc, 2)
            self.assertEqual(report["system_errors"]["parser_rejections"], 8)
            self.assertFalse(report["canary_gate"]["passed"])

    def test_canary_binding_checks_prompts_labels_and_candidate_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [make_candidate(root, index) for index in range(8)]
            provenance = {"fixture": "frozen"}
            out = root / "canary"
            report, rc = pairwise.run_generation(
                candidates,
                provenance,
                out,
                workers=4,
                client=FakePairClient(),
                protocol_artifact=False,
                stage="canary",
            )
            self.assertEqual(rc, 0)
            binding = pairwise.validate_canary_binding(
                out / "generation_report.json", candidates, provenance
            )
            self.assertEqual(binding["candidate_set_sha256"], report["candidate_set_sha256"])

            report_path = out / "generation_report.json"
            tampered = json.loads(report_path.read_text())
            tampered["prompt_hashes"]["generator_template_sha256"] = "0" * 64
            report_path.write_text(json.dumps(tampered))
            with self.assertRaises(SystemExit):
                pairwise.validate_canary_binding(report_path, candidates, provenance)

    def test_guarded_check_only_creates_no_network_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            cache = root / "cache"
            request_log = root / "requests.jsonl"
            env = {
                **os.environ,
                "CHECK_ONLY": "1",
                "CANARY": "8",
                "OUT": str(out),
                "V42_CACHE": str(cache),
                "V42_REQUEST_LOG": str(request_log),
            }
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/run_depth_v42_pairwise.sh")],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            plan = json.loads(result.stdout)
            self.assertTrue(plan["authorized_candidate_set"])
            self.assertEqual(plan["eligible_images"], 8)
            self.assertFalse(out.exists())
            self.assertFalse(cache.exists())
            self.assertFalse(request_log.exists())


if __name__ == "__main__":
    unittest.main()
