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

from tools import generate_depth_v43_pairwise as pairwise
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
                50 + 20 * entity_index,
                50 + 20 * entity_index,
                150 + 20 * entity_index,
                150 + 20 * entity_index,
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
    relation, geometry_claim = pairwise.OBSERVATION_BY_CUE[cue]
    statement = f"{nearer} is nearer to the camera than {farther}."
    cue_evidence = {
        "occlusion": f"{nearer} occludes the boundary of {farther}",
        "contact_layering": f"{nearer} contacts or layers over {farther}",
        "support_plane_perspective": (
            f"support-plane perspective places {nearer} before {farther}"
        ),
    }[cue]
    return {
        "orderable": True,
        "reason_code": "supported_pair",
        "confidence": confidence,
        "selected_pair": {
            "nearer": nearer,
            "farther": farther,
            "cue_type": cue,
            "evidence": f"{statement} {cue_evidence}.",
            "observation": {
                "nearer": nearer,
                "farther": farther,
                "relation": relation,
                "geometry_claim": geometry_claim,
                "direction_statement": statement,
            },
            "confidence": confidence,
        },
    }


def verifier_response(
    nearer: str, farther: str, cue: str, confidence: float = 0.94
) -> dict[str, Any]:
    relation, geometry_claim = pairwise.OBSERVATION_BY_CUE[cue]
    statement = f"{nearer} is nearer to the camera than {farther}."
    cue_evidence = {
        "occlusion": f"{nearer} independently occludes {farther}",
        "contact_layering": f"{nearer} independently layers over {farther}",
        "support_plane_perspective": (
            f"support-plane perspective places {nearer} before {farther}"
        ),
    }[cue]
    return {
        "valid": True,
        "reason_code": "independent_support",
        "confidence": confidence,
        "pair_check": {
            "nearer": nearer,
            "farther": farther,
            "cue_type": cue,
            "evidence": f"{statement} {cue_evidence}.",
            "observation": {
                "nearer": nearer,
                "farther": farther,
                "relation": relation,
                "geometry_claim": geometry_claim,
                "direction_statement": statement,
            },
            "supported": True,
            "confidence": confidence,
            "counterevidence": "",
        },
    }


def set_response_cue(response: dict[str, Any], cue: str) -> None:
    pair = response.get("selected_pair") or response["pair_check"]
    nearer = pair["nearer"]
    farther = pair["farther"]
    replacement = (
        generator_response(nearer, farther, cue)
        if "selected_pair" in response
        else verifier_response(nearer, farther, cue)
    )
    response.clear()
    response.update(replacement)


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


def attach_fixture_run_audit(report: dict[str, Any], out: Path) -> Path:
    run_id = "fixture-run-v43"
    response_rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = [
        {"event": "stage_start", "run_id": run_id}
    ]
    for call_index in range(report["pairs_attempted"]):
        prompt = f"fixture-prompt-{call_index}"
        response = {"fixture_response": call_index}
        response_sha = pairwise.content_hash(response)
        common = {
            "call_index": call_index,
            "image_sha256": "a" * 64,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "model": "fixture-model",
            "temperature": 0.0,
            "max_tokens": 600,
            "use_cache": False,
            "cache_source": "uncached_response",
            "cache_snapshot": None,
            "cache_snapshot_sha256": None,
            "response_sha256": response_sha,
            "error_type": "",
        }
        response_rows.append({
            **common,
            "image_path_local_only": "/fixture/image.jpg",
            "prompt": prompt,
            "response": response,
        })
        entries.append(common)
        request_rows.append({
            "image_sha256": "a" * 64,
            "prompt": prompt,
            "model": "fixture-model",
            "temperature": 0.0,
            "max_tokens": 600,
        })
    request_rows.append({"event": "stage_end", "run_id": run_id})
    request_log = out / "fixture_requests.jsonl"
    request_log.write_text(
        "".join(json.dumps(row) + "\n" for row in request_rows), encoding="utf-8"
    )
    response_path = out / "run_response_audit.jsonl"
    response_path.write_text(
        "".join(json.dumps(row) + "\n" for row in response_rows), encoding="utf-8"
    )
    cache_document = {
        "schema_version": 1,
        "protocol": pairwise.PROTOCOL,
        "run_id": run_id,
        "entries": entries,
        "entries_sha256": pairwise.content_hash(entries),
    }
    cache_path = out / "run_cache_manifest.json"
    cache_path.write_text(json.dumps(cache_document), encoding="utf-8")
    request_bytes = request_log.read_bytes()
    report["run_audit"] = {
        "run_id": run_id,
        "request_log_slice": {
            "path": str(request_log.resolve()),
            "start_byte": 0,
            "end_byte_exclusive": len(request_bytes),
            "start_row": 1,
            "end_row": len(request_rows),
            "row_count": len(request_rows),
            "sha256": hashlib.sha256(request_bytes).hexdigest(),
        },
        "response_audit": {
            "path": response_path.name,
            "row_count": len(response_rows),
            "sha256": sha256(response_path),
        },
        "cache_manifest": {
            "path": cache_path.name,
            "entry_count": len(entries),
            "entries_sha256": pairwise.content_hash(entries),
            "sha256": sha256(cache_path),
        },
    }
    report_path = out / "generation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def write_human_review(
    report_path: Path, labels: list[dict[str, Any]], correct_count: int = 6
) -> Path:
    report = json.loads(report_path.read_text())
    rows = []
    for index, label in enumerate(labels):
        correct = index < correct_count
        rows.append({
            "sample_id": label["sample_id"],
            "label_sha256": pairwise.content_hash(label),
            "image_sha256": label["_v4_sources"]["image_sha256"],
            "direction_correct": correct,
            "cue_consistent": True,
            "evidence_entails_direction": True,
            "geometry_supported": True,
            "correct": correct,
            "critical_error": not correct,
            "error_code": "" if correct else "direction_incorrect",
            "notes": "fixture review",
        })
    audit = report["run_audit"]
    review = {
        "schema_version": 1,
        "protocol": pairwise.HUMAN_REVIEW_PROTOCOL,
        "generation_protocol": pairwise.PROTOCOL,
        "prompt_version": pairwise.PROMPT_VERSION,
        "label_version": pairwise.LABEL_VERSION,
        "reviewer_id": "fixture-reviewer",
        "reviewed_at_utc": "2026-08-20T00:00:00Z",
        "canary_report_sha256": sha256(report_path),
        "labels_sha256": report["labels_sha256"],
        "candidate_set_sha256": report["candidate_set_sha256"],
        "request_log_slice_sha256": audit["request_log_slice"]["sha256"],
        "response_audit_sha256": audit["response_audit"]["sha256"],
        "cache_manifest_sha256": audit["cache_manifest"]["sha256"],
        "reviews": rows,
    }
    path = report_path.parent / "human_review.json"
    path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    return path


def full_provenance() -> dict[str, Any]:
    return {
        "fixture": True,
        "canary_binding": {"human_review": {"passed": True}},
    }


class DepthV43GeneratorTests(unittest.TestCase):
    def test_v43_protocol_is_cache_isolated_from_v42(self):
        from tools import generate_depth_v42_pairwise as v42

        self.assertNotEqual(pairwise.PROMPT_VERSION, v42.PROMPT_VERSION)
        self.assertNotEqual(pairwise.LABEL_VERSION, v42.LABEL_VERSION)
        self.assertNotEqual(pairwise.PROTOCOL, v42.PROTOCOL)
        self.assertNotEqual(pairwise.protocol_prompt_hashes(), v42.protocol_prompt_hashes())

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
            selection = record["_semantic_gate"]["depth_v43_pairwise"]["selection"]
            self.assertEqual(selection["rule"], "stable_hash_after_threshold")
            self.assertEqual(selection["pair_index"], expected_index)
            self.assertTrue(
                record["_semantic_gate"]["depth_v43_pairwise"][
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
            pairwise.validate_verifier_response(response, entities), (True, "")
        )
        response["pair_check"]["evidence"] = (
            "near is nearer to the camera than far. near looks larger than far."
        )
        self.assertEqual(
            pairwise.validate_verifier_response(response, entities)[1],
            "verifier_unsupported_depth_heuristic",
        )
        prompt = pairwise.verifier_prompt(list(reversed(entities)))
        self.assertNotIn("Proposed direction", prompt)
        self.assertNotIn("selected_pair", prompt)

    def test_static_direction_and_frame_heuristic_rejections(self):
        entities = [
            {"label": "bread", "bbox_2d": [10, 10, 100, 100]},
            {"label": "knife", "bbox_2d": [50, 50, 150, 150]},
        ]
        response = generator_response("bread", "knife", "occlusion")
        response["selected_pair"]["evidence"] += (
            " Yet bread is farther than knife and knife is nearer than bread."
        )
        self.assertEqual(
            pairwise.validate_generator_response(response, entities)[1],
            "generator_evidence_direction_contradiction",
        )
        for phrase in ("lower in frame", "higher in the frame", "bottom edge", "foreground"):
            response = generator_response("bread", "knife", "occlusion")
            response["selected_pair"]["evidence"] += f" It is inferred from {phrase}."
            self.assertEqual(
                pairwise.validate_generator_response(response, entities)[1],
                "generator_unsupported_depth_heuristic",
            )
        response = generator_response("bread", "knife", "contact_layering")
        response["selected_pair"]["evidence"] = (
            "bread is nearer to the camera than knife. bread occludes knife."
        )
        self.assertEqual(
            pairwise.validate_generator_response(response, entities)[1],
            "generator_evidence_cue_mismatch",
        )

    def test_geometry_gate_is_conservative(self):
        overlap = [
            {"label": "a", "bbox_2d": [0, 0, 100, 100]},
            {"label": "b", "bbox_2d": [90, 90, 190, 190]},
        ]
        passed, _, geometry = pairwise.geometry_gate(overlap, "occlusion")
        self.assertTrue(passed)
        self.assertEqual(geometry["intersection_over_min_box_area"], 0.01)
        tiny_overlap = [
            {"label": "a", "bbox_2d": [0, 0, 100, 100]},
            {"label": "b", "bbox_2d": [99, 99, 199, 199]},
        ]
        passed, reason, _ = pairwise.geometry_gate(tiny_overlap, "occlusion")
        self.assertFalse(passed)
        self.assertEqual(reason, "geometry:occlusion_insufficient_2d_intersection")
        ten_gap = [
            {"label": "a", "bbox_2d": [0, 0, 100, 100]},
            {"label": "b", "bbox_2d": [110, 0, 210, 100]},
        ]
        self.assertTrue(pairwise.geometry_gate(ten_gap, "contact_layering")[0])
        eleven_gap = [
            {"label": "a", "bbox_2d": [0, 0, 100, 100]},
            {"label": "b", "bbox_2d": [111, 0, 211, 100]},
        ]
        self.assertFalse(pairwise.geometry_gate(eleven_gap, "contact_layering")[0])

    def test_generator_and_verifier_must_agree_on_exact_cue(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = make_candidate(Path(tmp), 0)

            class CueMismatch(FakePairClient):
                def __call__(self, image: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
                    response = super().__call__(image, prompt, **kwargs)
                    if "pair_check" in response:
                        set_response_cue(response, "contact_layering")
                    return response

            record, stats = pairwise.generate_one(candidate, CueMismatch())
            self.assertIsNone(record)
            self.assertEqual(
                stats["pair_rejections"], {"verifier_cue_disagreement": 1}
            )

    def test_generation_report_and_cue_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [make_candidate(root, index) for index in range(40)]
            report, rc = pairwise.run_generation(
                candidates,
                full_provenance(),
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
            self.assertTrue(report["human_canary_gate"]["passed"])
            self.assertEqual(report["pair_selection_rule"], "stable_hash_after_threshold")
            self.assertTrue(report["role_order_counterbalanced"])

    def test_full_gate_rejects_missing_cue_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [make_candidate(root, index) for index in range(40)]

            class OcclusionOnly(FakePairClient):
                def __call__(self, image: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
                    response = super().__call__(image, prompt, **kwargs)
                    set_response_cue(response, "occlusion")
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

    def test_run_audit_replays_uncached_retry_and_snapshots_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            cache_dir = root / "cache"
            cache_dir.mkdir()
            request_log = root / "requests.jsonl"
            image = root / "image.jpg"
            image.write_bytes(b"fixture-image")
            old_log = os.environ.get("GSA_VLM_REQUEST_LOG")
            old_model = os.environ.get("GSA_VLM_MODEL")
            os.environ["GSA_VLM_REQUEST_LOG"] = str(request_log)
            os.environ["GSA_VLM_MODEL"] = "fixture-model"
            try:
                tracker = pairwise.RunAuditTracker(out, request_log, cache_dir, "canary")
                tracker.begin()

                def inner(image_path: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
                    response = {"prompt": prompt, "uncached": not kwargs["use_cache"]}
                    with request_log.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "image_sha256": sha256(Path(image_path)),
                            "prompt": prompt,
                            "model": "fixture-model",
                            "temperature": kwargs["temperature"],
                            "max_tokens": kwargs["max_tokens"],
                        }) + "\n")
                    if kwargs["use_cache"]:
                        locator = hashlib.sha256(
                            (
                                f"fixture-model|{kwargs['temperature']}|{prompt}|"
                                f"{sha256(Path(image_path))}"
                            ).encode()
                        ).hexdigest()
                        (cache_dir / f"{locator}.json").write_text(
                            json.dumps({"content": "fixture", "parsed": response})
                        )
                    return response

                client = tracker.wrap(inner)
                client(
                    str(image), "cached prompt", temperature=0.0,
                    max_tokens=600, use_cache=True
                )
                retry_response = client(
                    str(image), "retry prompt", temperature=0.0,
                    max_tokens=600, use_cache=False
                )
                report = {
                    "attempted_images": 1,
                    "accepted_pool_images": 1,
                    "pairs_attempted": 1,
                    "pairs_accepted": 1,
                    "audit_pool_eligible": False,
                    "canary_gate": {"passed": True},
                    "system_errors": {"total": 0},
                }
                report["run_audit"] = tracker.finalize(report, 0)
                report_path = out / "generation_report.json"
                report_path.write_text(json.dumps(report))
                binding = pairwise._validate_run_audit(report, report_path)
                self.assertEqual(binding["run_id"], report["run_audit"]["run_id"])
                rows = [
                    json.loads(line)
                    for line in (out / "run_response_audit.jsonl").read_text().splitlines()
                ]
                self.assertEqual(rows[1]["response"], retry_response)
                self.assertIsNone(rows[1]["cache_snapshot"])
                self.assertIsNotNone(rows[0]["cache_snapshot"])
            finally:
                if old_log is None:
                    os.environ.pop("GSA_VLM_REQUEST_LOG", None)
                else:
                    os.environ["GSA_VLM_REQUEST_LOG"] = old_log
                if old_model is None:
                    os.environ.pop("GSA_VLM_MODEL", None)
                else:
                    os.environ["GSA_VLM_MODEL"] = old_model

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

    def test_canary_requires_six_machine_accepted_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [make_candidate(root, index) for index in range(8)]

            class BoundedAcceptance(FakePairClient):
                def __init__(self, accepted: int):
                    super().__init__()
                    self.accepted = accepted

                def __call__(self, image: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
                    index = int(Path(image).stem.split("-")[1])
                    if index >= self.accepted and prompt.startswith("You are the primary"):
                        return {
                            "orderable": False,
                            "reason_code": "ambiguous_pair",
                            "confidence": 0.0,
                            "selected_pair": None,
                        }
                    return super().__call__(image, prompt, **kwargs)

            report, rc = pairwise.run_generation(
                candidates,
                {},
                root / "five",
                workers=4,
                client=BoundedAcceptance(5),
                protocol_artifact=False,
                stage="canary",
            )
            self.assertEqual(report["accepted_pool_images"], 5)
            self.assertEqual(rc, 2)
            report, rc = pairwise.run_generation(
                candidates,
                {},
                root / "six",
                workers=4,
                client=BoundedAcceptance(6),
                protocol_artifact=False,
                stage="canary",
            )
            self.assertEqual(report["accepted_pool_images"], 6)
            self.assertEqual(rc, 0)

    def test_canary_binding_requires_complete_human_review_and_run_hashes(self):
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
            report_path = attach_fixture_run_audit(report, out)
            label_path = out / "labels/S_spatial/depth.jsonl"
            labels = [json.loads(line) for line in label_path.read_text().splitlines()]
            review_path = write_human_review(report_path, labels, correct_count=8)
            binding = pairwise.validate_canary_binding(
                report_path, review_path, candidates, provenance
            )
            self.assertEqual(binding["candidate_set_sha256"], report["candidate_set_sha256"])
            self.assertEqual(binding["human_review"]["correct_records"], 8)

            tampered = json.loads(report_path.read_text())
            tampered["prompt_hashes"]["generator_template_sha256"] = "0" * 64
            report_path.write_text(json.dumps(tampered))
            with self.assertRaises(SystemExit):
                pairwise.validate_canary_binding(
                    report_path, review_path, candidates, provenance
                )

    def test_human_review_rejects_critical_error(self):
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
            report_path = attach_fixture_run_audit(report, out)
            labels = [
                json.loads(line)
                for line in (out / "labels/S_spatial/depth.jsonl").read_text().splitlines()
            ]
            review_path = write_human_review(report_path, labels, correct_count=8)
            review = json.loads(review_path.read_text())
            row = review["reviews"][0]
            row["direction_correct"] = False
            row["correct"] = False
            row["critical_error"] = True
            row["error_code"] = "reversed_direction"
            review_path.write_text(json.dumps(review))
            with self.assertRaises(SystemExit):
                pairwise.validate_canary_binding(
                    report_path, review_path, candidates, provenance
                )

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
                "V43_CACHE": str(cache),
                "V43_REQUEST_LOG": str(request_log),
            }
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/run_depth_v43_pairwise.sh")],
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

    def test_full_launcher_refuses_missing_human_review_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "generation_report.json"
            report.write_text("{}", encoding="utf-8")
            env = {
                **os.environ,
                "CANARY": "0",
                "ALLOW_EXTERNAL_UPLOAD": "1",
                "CANARY_APPROVED": "1",
                "CANARY_REPORT": str(report),
                "HUMAN_REVIEW": str(root / "missing_human_review.json"),
                "OUT": str(root / "full"),
            }
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/run_depth_v43_pairwise.sh")],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("hash-bound human canary review", result.stderr)

    def test_v42_full_launcher_is_retired(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "CANARY": "0",
                "CHECK_ONLY": "0",
                "ALLOW_EXTERNAL_UPLOAD": "1",
                "OUT": str(Path(tmp) / "retired_full"),
            }
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/run_depth_v42_pairwise.sh")],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("v4.2 full is retired", result.stderr)


if __name__ == "__main__":
    unittest.main()
