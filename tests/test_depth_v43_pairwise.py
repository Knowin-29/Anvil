from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_depth_v4_audit import v4_record
from tests.test_depth_v42_pairwise import pairwise_record
from tools import generate_depth_v43_pairwise as generator_tool
from trains.flywheel.validators import validate_depth_v4_record

PROMPT_VERSION = "gsa-depth-v4.3-pairwise"
LABEL_VERSION = "gsa-label-v4.3-depth-pairwise"
RELATIONS = {
    "occlusion": ("occludes", "positive_2d_intersection"),
    "contact_layering": ("contacts_or_layers_over", "overlap_or_near_touch"),
    "support_plane_perspective": ("support_plane_nearer", "support_plane_depth"),
}


def geometry_metrics(boxes: list[list[int]]) -> dict[str, float | bool | str]:
    first, second = boxes
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    area = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return {
        "intersection_width": width,
        "intersection_height": height,
        "intersection_area": area,
        "intersection_over_min_box_area": area / min(first_area, second_area),
        "min_intersection_over_min_box_area": 0.01,
        "normalized_edge_distance": math.hypot(dx, dy) / (1000.0 * math.sqrt(2.0)),
        "max_normalized_edge_distance": 0.01,
        "max_axis_edge_distance": max(dx, dy),
        "max_axis_edge_distance_limit": 10.0,
    }


def v43_record(image: Path, cue: str = "occlusion", index: int = 0) -> dict:
    nearer = f"near-object-{index}"
    farther = f"far-object-{index}"
    if cue == "occlusion":
        boxes = [[100, 100, 400, 400], [300, 250, 600, 550]]
    elif cue == "contact_layering":
        boxes = [[100, 100, 300, 300], [308, 100, 508, 300]]
    else:
        boxes = [[100, 550, 300, 750], [600, 100, 800, 300]]
    entities = [
        {"object_class": nearer, "bbox": boxes[0]},
        {"object_class": farther, "bbox": boxes[1]},
    ]
    payload = [
        {"label": nearer, "rank": 1, "relative_depth": 0.0},
        {"label": farther, "rank": 2, "relative_depth": 1.0},
    ]
    relation, geometry_claim = RELATIONS[cue]
    observation = {
        "nearer": nearer,
        "farther": farther,
        "relation": relation,
        "geometry_claim": geometry_claim,
        "direction_statement": f"{nearer} is nearer to the camera than {farther}.",
    }
    cue_sentence = (
        f"{nearer} visibly occludes part of {farther}."
        if cue == "occlusion"
        else f"{nearer} is in contact with and layers over {farther}."
        if cue == "contact_layering"
        else f"Support-plane perspective places {nearer} before {farther}."
    )
    generator_evidence = (
        f"{nearer} is nearer to the camera than {farther}. {cue_sentence}"
    )
    verifier_evidence = (
        f"{nearer} is nearer to the camera than {farther}. Independent inspection: "
        f"{cue_sentence}"
    )
    evidence = {
        "nearer": nearer,
        "farther": farther,
        "cue_type": cue,
        "evidence": generator_evidence,
        "observation": observation,
    }
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    sorted_labels = sorted((nearer, farther))
    selection_value = (
        f"{PROMPT_VERSION}|{image_sha}|{sorted_labels[0]}|{sorted_labels[1]}|"
        f"{nearer}>{farther}"
    )
    stable_hash = hashlib.sha256(selection_value.encode()).hexdigest()
    geometry = {
        "cue_type": cue,
        "passed": True,
        **geometry_metrics(boxes),
    }
    return {
        "sample_id": f"v43_depth_{index:020d}",
        "pipeline": "S_spatial",
        "dimension": "S_spatial",
        "task_form": "depth",
        "depth_scope": "pairwise",
        "image_path": str(image),
        "scene_type": "test",
        "source_bucket": "clean",
        "object_class": nearer,
        "confidence": 0.93,
        "label_version": LABEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "payload": payload,
        "gsa_context": {
            "contract_version": "gsa-contract-v2",
            "primary": entities[0],
            "groundings": entities,
            "spatial": {
                "relation_mode": "pairwise",
                "depth_order": payload,
                "adjacent_depth_evidence": [evidence],
            },
        },
        "bbox": boxes[0],
        "_v4_sources": {"image_sha256": image_sha},
        "_semantic_gate": {
            "depth_v43_pairwise": {
                "generator": {
                    "orderable": True,
                    "reason_code": "supported_pair",
                    "confidence": 0.95,
                    "selected_pair": {
                        **evidence,
                        "observation": observation,
                        "confidence": 0.94,
                    },
                },
                "verifier": {
                    "valid": True,
                    "reason_code": "independent_support",
                    "confidence": 0.93,
                    "pair_check": {
                        "nearer": nearer,
                        "farther": farther,
                        "cue_type": cue,
                        "evidence": verifier_evidence,
                        "observation": observation,
                        "supported": True,
                        "confidence": 0.96,
                        "counterevidence": "",
                    },
                },
                "role_order_counterbalanced": True,
                "cue_agreement": True,
                "geometry": geometry,
                "selection": {
                    "rule": "stable_hash_after_threshold",
                    "pair_index": 0,
                    "pairs_evaluated": 1,
                    "stable_hash": stable_hash,
                    "gate_confidence": 0.93,
                },
            },
        },
        "_verification": {
            "signals": {
                "direction_blind_verifier": 1.0,
                "exact_cue_agreement": 1.0,
                "frozen_bbox_geometry": 1.0,
            },
            "verdict": "verified_pair",
        },
    }


def test_v43_accepts_all_cues_and_preserves_distinct_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "image.jpg"
        image.write_bytes(b"image")
        for index, cue in enumerate(RELATIONS):
            record = v43_record(image, cue, index)
            gate = record["_semantic_gate"]["depth_v43_pairwise"]
            assert (
                gate["generator"]["selected_pair"]["evidence"]
                != gate["verifier"]["pair_check"]["evidence"]
            )
            assert validate_depth_v4_record(record) == (True, "")


def test_v43_fail_closed_semantic_and_provenance_mutations():
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "image.jpg"
        image.write_bytes(b"image")
        base = v43_record(image)
        mutations = [
            (
                lambda row: row.update(prompt_version="gsa-depth-v4.3-pairwise-tampered"),
                "wrong_prompt_version",
            ),
            (
                lambda row: row["_semantic_gate"]["depth_v43_pairwise"].update(
                    cue_agreement=False
                ),
                "v43_pair_cue_agreement",
            ),
            (
                lambda row: row["_semantic_gate"]["depth_v43_pairwise"][
                    "verifier"
                ]["pair_check"].update(cue_type="contact_layering"),
                "v43_pair_verifier_claim_schema",
            ),
            (
                lambda row: row["_semantic_gate"]["depth_v43_pairwise"][
                    "generator"
                ]["selected_pair"]["observation"].update(
                    direction_statement="far-object-0 is nearer to the camera than near-object-0."
                ),
                    "v43_pair_spatial_observation_entailment",
            ),
            (
                lambda row: row["_semantic_gate"]["depth_v43_pairwise"][
                    "geometry"
                ].update(intersection_area=1.0),
                "v43_pair_geometry_provenance",
            ),
            (
                lambda row: row["_semantic_gate"]["depth_v43_pairwise"][
                    "selection"
                ].update(stable_hash="a" * 64),
                "v43_pair_selection_gate",
            ),
            (
                lambda row: row["_verification"]["signals"].update(
                    exact_cue_agreement=0.0
                ),
                "v43_pair_verification",
            ),
        ]
        for mutate, reason in mutations:
            record = copy.deepcopy(base)
            mutate(record)
            assert validate_depth_v4_record(record)[1] == reason


def test_v43_rejects_reverse_and_frame_heuristic_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "image.jpg"
        image.write_bytes(b"image")
        reverse = v43_record(image)
        reverse_text = (
            "near-object-0 is further back, while far-object-0 is nearer to the camera. "
            "near-object-0 visibly occludes far-object-0."
        )
        gate = reverse["_semantic_gate"]["depth_v43_pairwise"]
        gate["generator"]["selected_pair"]["evidence"] = reverse_text
        reverse["gsa_context"]["spatial"]["adjacent_depth_evidence"][0][
            "evidence"
        ] = reverse_text
        assert validate_depth_v4_record(reverse)[1] == "v43_pair_spatial_reverse_direction"

        frame = v43_record(image)
        frame_text = (
            "near-object-0 is at the bottom edge of the image before far-object-0."
        )
        check = frame["_semantic_gate"]["depth_v43_pairwise"]["verifier"][
            "pair_check"
        ]
        check["evidence"] = frame_text
        assert validate_depth_v4_record(frame)[1] == "v43_pair_verifier_unsupported_heuristic"

        cue_mismatch = v43_record(image, "contact_layering")
        mismatch_text = (
            "near-object-0 is nearer to the camera than far-object-0. "
            "near-object-0 visibly occludes far-object-0."
        )
        selected = cue_mismatch["_semantic_gate"]["depth_v43_pairwise"][
            "generator"
        ]["selected_pair"]
        selected["evidence"] = mismatch_text
        cue_mismatch["gsa_context"]["spatial"]["adjacent_depth_evidence"][0][
            "evidence"
        ] = mismatch_text
        assert (
            validate_depth_v4_record(cue_mismatch)[1]
            == "v43_pair_spatial_evidence_cue_mismatch"
        )


def test_v43_geometry_thresholds_reject_bbox_noise():
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "image.jpg"
        image.write_bytes(b"image")

        tiny_overlap = v43_record(image, "occlusion")
        entities = tiny_overlap["gsa_context"]["groundings"]
        entities[1]["bbox"] = [399, 399, 600, 600]
        geometry = tiny_overlap["_semantic_gate"]["depth_v43_pairwise"]["geometry"]
        geometry.update(geometry_metrics([item["bbox"] for item in entities]))
        assert validate_depth_v4_record(tiny_overlap)[1] == "v43_pair_geometry_not_supported"

        wide_gap = v43_record(image, "contact_layering")
        entities = wide_gap["gsa_context"]["groundings"]
        entities[1]["bbox"] = [326, 100, 526, 300]
        geometry = wide_gap["_semantic_gate"]["depth_v43_pairwise"]["geometry"]
        geometry.update(geometry_metrics([item["bbox"] for item in entities]))
        assert validate_depth_v4_record(wide_gap)[1] == "v43_pair_geometry_not_supported"


def test_v43_does_not_reinterpret_v41_or_v42_records():
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "image.jpg"
        image.write_bytes(b"image")
        assert validate_depth_v4_record(v4_record(image, 0)) == (True, "")
        assert validate_depth_v4_record(pairwise_record(image, 0)) == (True, "")


def test_v43_actual_generator_record_matches_validator_contract():
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "image.jpg"
        image.write_bytes(b"image")
        fixture = v43_record(image)
        entities = [
            {"label": item["object_class"], "bbox_2d": item["bbox"]}
            for item in fixture["gsa_context"]["groundings"]
        ]
        candidate = {
            "image_path": str(image),
            "image_sha256": fixture["_v4_sources"]["image_sha256"],
            "entities": entities,
            "detection": {"sample_id": "detection", "source_bucket": "clean"},
            "bbox": {"sample_id": "bbox"},
            "scene_type": "test",
        }
        gate = fixture["_semantic_gate"]["depth_v43_pairwise"]
        passed, reason, geometry = generator_tool.geometry_gate(entities, "occlusion")
        assert passed, reason
        record = generator_tool._build_record(
            candidate,
            gate["generator"],
            gate["verifier"],
            0,
            1,
            generator_tool._selection_key(candidate, gate["generator"]),
            geometry,
        )
        assert validate_depth_v4_record(record) == (True, "")
