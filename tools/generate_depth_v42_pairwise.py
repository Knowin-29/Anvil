#!/usr/bin/env python3
"""Generate strict pairwise camera-depth labels from fresh audit images.

Every unordered pair among the frozen image entities is evaluated in a
separate generator call that sees only those two entities. A direction-blind
verifier independently judges the pair with reversed entity order. At most one
verified pair is emitted per image, chosen by a stable hash after strict gates.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_depth_v4 import (  # noqa: E402
    _source_bucket,
    ground_candidate,
    load_candidates,
)
from trains.flywheel.build_depth_semantic_audit_pack import (  # noqa: E402
    content_hash,
    ensure_empty_output,
    file_hash,
)
from trains.flywheel.validators import validate_depth_v4_record  # noqa: E402

PROMPT_VERSION = "gsa-depth-v4.2-pairwise"
LABEL_VERSION = "gsa-label-v4-depth-pairwise"
PROTOCOL = "depth-v4.2-pairwise-generation"
AUTHORIZED_ENDPOINT = os.environ.get("AUTHORIZED_ENDPOINT", "")
TARGET = 40
FROZEN_SEED = 20260821
CANARY_IMAGES = 8
MIN_CANARY_ACCEPTED = 4
MIN_CONFIDENCE = 0.85
MIN_CUE_IMAGES = {"occlusion": 8, "contact_layering": 8}
FROZEN_CANARY_SHA256S = (
    "4aa5717afe34fa5c89111e478c370fdbce10d29b3774ec0cb9aafab802ed7a11",
    "60c91ddfc30ad2f56633b59be92b617c4ebb37a9a9b0c1055d7de9a667c4458d",
    "1378b6f83858f24925dd556f72bc40665181bc490e148cdf759b9eda538acca2",
    "d78b64050c9ee028ee4c1d579d0d4a4831b800129d75d96af9eaf5567d02bbdf",
    "992d3716a80823ba3e27bc84fb8754be524ca59fd2ad6d6a92b758b124406152",
    "d195e4ae11d8fbc2752f4344a2ec92ca9f377d11ffe46e6d4b4b1620667b2d71",
    "6f430ea7a537bdc035a531c7b7eb7ea2c10e6df2d6c5c8f6c1ff2503858d93e3",
    "602db552890d81da239b3b3fe95f541cc2da70eb13070449377bbb64f0522eaf",
)
FROZEN_CANARY_SET_SHA256 = (
    "51f1a9f013a48d7444e5499b2d63ec52e7b476a86aaaf41779bacb1753645ea2"
)
FROZEN_ALL_CANDIDATE_SET_SHA256 = (
    "06f38fd2b9171d15cdde4b94e0153d5a74279a8fd36f72d0498058462f2c3a01"
)
FROZEN_REMAINDER_SET_SHA256 = (
    "998058359943ff6e6f3ac9733df7801ea0d51e6b66ad793dc7c9f0c19d7d8e89"
)
FROZEN_SOURCE_SHA256S = {
    "index_sha256": "88927843f40af3074ee9dbe246f1fe495ee7caabbd7c14331bf5eb303ce2acb0",
    "detection_sha256": "004f937c80cc90d13160021b9f4ebffe0867bf37db889f4fda8e2c9c18702d64",
    "bbox_sha256": "72bddda5b01aa3ec618b118d20786aa479300e6b51c0f00afaf633cd770cd6bd",
}
FROZEN_EXCLUSION_SHA256S = (
    "11e5358f84e4a7bf0db00ce3792d1988ea388975e51e5208f84dd9f1a314cdcf",
    "b6cb238cfdca993c6540a16abb6e9b23a46c277526dfaaf70bf09ce0b6a83b9a",
)
ALLOWED_CUES = {"occlusion", "contact_layering", "support_plane_perspective"}
BANNED_EVIDENCE = (
    "left of",
    "right of",
    "higher in the image",
    "lower in the image",
    "larger",
    "smaller",
    "image height",
    "looks bigger",
    "looks smaller",
)


def _confidence(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and MIN_CONFIDENCE <= float(value) <= 1.0
    )


def _candidate_set_sha256(candidates: list[dict[str, Any]]) -> str:
    return content_hash([candidate["image_sha256"] for candidate in candidates])


def _frozen_source_matches(provenance: dict[str, Any]) -> bool:
    exclusions = provenance.get("exclusion_manifests")
    exclusion_hashes = (
        tuple(item.get("manifest_sha256") for item in exclusions)
        if isinstance(exclusions, list)
        else ()
    )
    return (
        all(
            provenance.get(key) == expected
            for key, expected in FROZEN_SOURCE_SHA256S.items()
        )
        and exclusion_hashes == FROZEN_EXCLUSION_SHA256S
    )


def validate_canary_binding(
    report_path: Path,
    candidates: list[dict[str, Any]],
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    if not report_path.is_file():
        raise SystemExit(f"missing v4.2 canary report: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid v4.2 canary report: {report_path}: {exc}") from exc
    label_path = report_path.parent / "labels/S_spatial/depth.jsonl"
    try:
        label_lines = label_path.read_text(encoding="utf-8").splitlines()
        labels = [json.loads(line) for line in label_lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid v4.2 canary labels: {label_path}: {exc}") from exc
    labels_valid = all(validate_depth_v4_record(label)[0] for label in labels)
    reported_pool = report.get("accepted_pool")
    accepted_pool_valid = (
        isinstance(reported_pool, list)
        and len(reported_pool) == len(labels)
        and all(
            isinstance(item, dict)
            and item.get("sample_id") == label.get("sample_id")
            and item.get("label_sha256") == content_hash(label)
            and item.get("image_path") == label.get("image_path")
            and item.get("image_sha256")
            == label.get("_v4_sources", {}).get("image_sha256")
            for item, label in zip(reported_pool, labels)
        )
    )
    expected_hashes = [
        candidate["image_sha256"] for candidate in candidates[:CANARY_IMAGES]
    ]
    checks = (
        report.get("protocol") == PROTOCOL,
        report.get("prompt_version") == PROMPT_VERSION,
        report.get("label_version") == LABEL_VERSION,
        report.get("stage") == "canary",
        report.get("protocol_artifact") is False,
        report.get("candidate_limit_images") == CANARY_IMAGES,
        report.get("attempted_images") == CANARY_IMAGES,
        report.get("accepted_pool_images", -1) >= MIN_CANARY_ACCEPTED,
        report.get("canary_gate", {}).get("passed") is True,
        report.get("system_errors", {}).get("total") == 0,
        report.get("role_order_counterbalanced") is True,
        report.get("pair_selection_rule") == "stable_hash_after_threshold",
        report.get("prompt_hashes") == protocol_prompt_hashes(),
        label_path.is_file(),
        report.get("labels_sha256") == file_hash(label_path),
        len(labels) == report.get("emitted_records"),
        len(labels) == report.get("accepted_pool_images"),
        labels_valid,
        accepted_pool_valid,
        report.get("candidate_image_sha256s") == expected_hashes,
        report.get("candidate_set_sha256")
        == content_hash(expected_hashes),
        content_hash(report.get("source_provenance"))
        == content_hash(source_provenance),
    )
    if not all(checks):
        raise SystemExit(
            "v4.2 full generation refused: canary report does not bind to "
            "the first eight current candidates and frozen source provenance"
        )
    return {
        "path": str(report_path.resolve()),
        "report_sha256": file_hash(report_path),
        "candidate_set_sha256": report["candidate_set_sha256"],
    }


def _frozen_pair(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"label": entity["label"], "bbox_2d": list(entity["bbox_2d"])}
        for entity in entities
    ]


def generator_prompt(entities: list[dict[str, Any]]) -> str:
    if len(entities) != 2:
        raise ValueError("pairwise generator requires exactly two entities")
    return (
        "You are the primary pairwise camera-depth annotator. Judge only the two "
        "frozen entities below and do not add, rename, merge, or split either one. "
        f"Pair entities: {json.dumps(_frozen_pair(entities), ensure_ascii=False)}. "
        "Decide whether one entity is physically nearer to the camera than the "
        "other using a direct visible cue: occlusion, contact/layering, or "
        "support-plane perspective. Apparent size, image height, and left/right "
        "position alone are invalid. A tie or ambiguity is unsupported. If no "
        "direction is directly supported, return exactly "
        '{"orderable":false,"reason_code":"ambiguous_pair","confidence":0.0,'
        '"selected_pair":null}. If supported, return exactly one JSON object: '
        '{"orderable":true,"reason_code":"supported_pair","confidence":0.95,'
        '"selected_pair":{"nearer":"<exact label>","farther":"<exact label>",'
        '"cue_type":"occlusion|contact_layering|support_plane_perspective",'
        '"evidence":"a concrete visible observation naming both exact labels",'
        '"confidence":0.95}}. Use confidence in [0.85,1.0] for a supported '
        "pair. Return JSON only."
    )

def falsifier_prompt(entities: list[dict[str, Any]]) -> str:
    if len(entities) != 2:
        raise ValueError("pairwise verifier requires exactly two entities")
    return (
        "Act as a direction-blind pairwise camera-depth verifier. Judge only the "
        "two frozen entities below; no proposed direction, cue, or rationale is "
        "provided. Independently determine whether one entity is physically nearer "
        "to the camera than the other. "
        f"Frozen pair: {json.dumps(_frozen_pair(entities), ensure_ascii=False)}. "
        "Accept only direct occlusion, contact/layering, or support-plane "
        "perspective. Reject apparent size, image height, left/right position, a "
        "tie, ambiguous direction, or unsupported evidence. If independently "
        "supported, return exactly "
        '{"valid":true,"reason_code":"independent_support","confidence":0.95,'
        '"pair_check":{"nearer":"<exact label>","farther":"<exact label>",'
        '"cue_type":"occlusion|contact_layering|support_plane_perspective",'
        '"evidence":"a concrete visible observation naming both exact labels",'
        '"supported":true,"confidence":0.95,"counterevidence":""}}. '
        "If no direction has direct support, return valid=false with a specific "
        "reason_code and counterevidence. Use confidence in [0.85,1.0] for a valid "
        "result and its pair_check. Return JSON only."
    )


def protocol_prompt_hashes() -> dict[str, str]:
    placeholder = [
        {"label": "<entity-a>", "bbox_2d": [0, 0, 1, 1]},
        {"label": "<entity-b>", "bbox_2d": [1, 1, 2, 2]},
    ]
    return {
        "generator_template_sha256": hashlib.sha256(
            generator_prompt(placeholder).encode("utf-8")
        ).hexdigest(),
        "falsifier_protocol_sha256": hashlib.sha256(
            falsifier_prompt(list(reversed(placeholder))).encode("utf-8")
        ).hexdigest(),
    }


def validate_generator_response(
    response: Any, entities: list[dict[str, Any]]
) -> tuple[bool, str]:
    if not isinstance(response, dict):
        return False, "generator_not_object"
    if response.get("orderable") is not True:
        reason = response.get("reason_code")
        if isinstance(reason, str) and reason:
            return False, f"not_orderable:{reason}"
        return False, "generator_missing_orderability"
    if (
        response.get("reason_code") != "supported_pair"
        or not _confidence(response.get("confidence"))
    ):
        return False, "generator_gate"
    selected = response.get("selected_pair")
    if not isinstance(selected, dict):
        return False, "selected_pair_not_object"
    expected = {entity["label"] for entity in entities}
    nearer = selected.get("nearer")
    farther = selected.get("farther")
    evidence = selected.get("evidence")
    pair_confidence = selected.get("confidence", response.get("confidence"))
    if (
        not isinstance(nearer, str)
        or not isinstance(farther, str)
        or nearer == farther
        or {nearer, farther} != expected
        or selected.get("cue_type") not in ALLOWED_CUES
        or not isinstance(evidence, str)
        or len(evidence.strip()) < 20
        or not _confidence(pair_confidence)
    ):
        return False, "selected_pair_schema"
    folded = evidence.casefold()
    if nearer.casefold() not in folded or farther.casefold() not in folded:
        return False, "evidence_missing_exact_labels"
    if any(term in folded for term in BANNED_EVIDENCE):
        return False, "unsupported_depth_heuristic"
    return True, ""


def validate_falsifier_response(
    response: Any, entities: list[dict[str, Any]]
) -> tuple[bool, str]:
    if not isinstance(response, dict):
        return False, "falsifier_not_object"
    reason_code = response.get("reason_code", "invalid")
    if (
        response.get("valid") is not True
        or reason_code != "independent_support"
        or not _confidence(response.get("confidence"))
    ):
        return False, f"falsifier_rejected:{reason_code}"
    check = response.get("pair_check")
    if not isinstance(check, dict):
        return False, "falsifier_pair"
    nearer = check.get("nearer")
    farther = check.get("farther")
    evidence = check.get("evidence")
    expected = {entity["label"] for entity in entities}
    if (
        not isinstance(nearer, str)
        or not isinstance(farther, str)
        or nearer == farther
        or {nearer, farther} != expected
        or check.get("cue_type") not in ALLOWED_CUES
        or not isinstance(evidence, str)
        or len(evidence.strip()) < 20
        or check.get("supported") is not True
        or not _confidence(check.get("confidence"))
        or check.get("counterevidence") not in ("", None)
    ):
        return False, "falsifier_pair"
    folded = evidence.casefold()
    if nearer.casefold() not in folded or farther.casefold() not in folded:
        return False, "falsifier_evidence_missing_exact_labels"
    if any(term in folded for term in BANNED_EVIDENCE):
        return False, "falsifier_unsupported_depth_heuristic"
    return True, ""


def _evidence(selected_pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "nearer": selected_pair["nearer"],
        "farther": selected_pair["farther"],
        "cue_type": selected_pair["cue_type"],
        "evidence": selected_pair["evidence"],
    }


def _proposal_score(
    generator: dict[str, Any], falsifier: dict[str, Any]
) -> float:
    selected = generator["selected_pair"]
    check = falsifier["pair_check"]
    return min(
        float(generator["confidence"]),
        float(selected.get("confidence", generator["confidence"])),
        float(falsifier["confidence"]),
        float(check["confidence"]),
    )


def _selection_key(
    candidate: dict[str, Any], generator: dict[str, Any]
) -> str:
    selected = generator["selected_pair"]
    nearer = selected["nearer"]
    farther = selected["farther"]
    labels = sorted((nearer, farther))
    image_sha256 = candidate["image_sha256"]
    value = (
        f"{PROMPT_VERSION}|{image_sha256}|{labels[0]}|{labels[1]}|"
        f"{nearer}>{farther}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_record(
    candidate: dict[str, Any],
    generator: dict[str, Any],
    falsifier: dict[str, Any],
    pair_index: int,
    pair_count: int,
    selection_key: str,
) -> dict[str, Any]:
    selected = generator["selected_pair"]
    entity_by_label = {entity["label"]: entity for entity in candidate["entities"]}
    ordered_entities = [
        entity_by_label[selected["nearer"]],
        entity_by_label[selected["farther"]],
    ]
    groundings = [
        {"object_class": entity["label"], "bbox": list(entity["bbox_2d"])}
        for entity in ordered_entities
    ]
    payload = [
        {"label": selected["nearer"], "rank": 1, "relative_depth": 0.0},
        {"label": selected["farther"], "rank": 2, "relative_depth": 1.0},
    ]
    adjacent_evidence = [_evidence(selected)]
    identity = content_hash({
        "image_sha256": candidate["image_sha256"],
        "payload": payload,
        "prompt_version": PROMPT_VERSION,
    })
    detection = candidate["detection"]
    bbox = candidate["bbox"]
    score = _proposal_score(generator, falsifier)
    context = {
        "contract_version": "gsa-contract-v2",
        "primary": groundings[0],
        "groundings": groundings,
        "spatial": {
            "relation_mode": "pairwise",
            "depth_order": payload,
            "adjacent_depth_evidence": adjacent_evidence,
        },
    }
    return {
        "sample_id": f"v42_depth_{identity[:20]}",
        "pipeline": "S_spatial",
        "dimension": "S_spatial",
        "task_form": "depth",
        "depth_scope": "pairwise",
        "image_path": candidate["image_path"],
        "source": detection.get("source", "upload_review_500_v2"),
        "dataset": detection.get("dataset", "upload_review_500_v2"),
        "scene_type": (
            detection.get("scene_type") or candidate.get("scene_type") or "unknown"
        ),
        "source_bucket": (
            detection.get("source_bucket") or _source_bucket(candidate["image_path"])
        ),
        "object_class": groundings[0]["object_class"],
        "confidence": score,
        "label_version": LABEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "payload": payload,
        "gsa_context": context,
        "bbox": groundings[0]["bbox"],
        "_v4_sources": {
            "detection_sample_id": detection.get("sample_id"),
            "detection_label_sha256": content_hash(detection),
            "bbox_sample_id": bbox.get("sample_id"),
            "bbox_label_sha256": content_hash(bbox),
            "image_sha256": candidate["image_sha256"],
        },
        "_semantic_gate": {
            "depth_v4_pairwise": {
                "generator": generator,
                "falsifier": falsifier,
                "role_order_counterbalanced": True,
                "selection": {
                    "rule": "stable_hash_after_threshold",
                    "pair_index": pair_index,
                    "pairs_evaluated": pair_count,
                    "stable_hash": selection_key,
                    "gate_confidence": score,
                },
            }
        },
        "_verification": {
            "contract_version": "gsa-contract-v2",
            "form": "depth",
            "family": "S",
            "signals": {
                "exact_pair_identity": 1.0,
                "direct_visible_evidence": 1.0,
                "direction_blind_verifier": 1.0,
            },
            "verdict": "verified_pair",
        },
    }


def generate_pair(
    candidate: dict[str, Any],
    entities: list[dict[str, Any]],
    client: Callable[..., Any],
) -> tuple[tuple[dict[str, Any], dict[str, Any]] | None, str]:
    prompt = generator_prompt(entities)
    generator: dict[str, Any] | None = None
    last_reason = "generator_failed"
    for attempt in range(2):
        attempt_prompt = prompt
        if attempt:
            attempt_prompt += (
                f" Your previous response failed deterministic validation ({last_reason}). "
                "Return a corrected response without weakening the evidence rule."
            )
        response = client(
            candidate["image_path"],
            attempt_prompt,
            max_tokens=600,
            temperature=0.0 if attempt == 0 else 0.1,
            use_cache=(attempt == 0),
        )
        valid, reason = validate_generator_response(response, entities)
        if valid:
            generator = response
            break
        last_reason = reason
        if reason.startswith("not_orderable:"):
            return None, reason
    if generator is None:
        return None, last_reason

    verifier_entities = list(reversed(entities))
    verifier_instruction = falsifier_prompt(verifier_entities)
    falsifier: dict[str, Any] | None = None
    reason = "falsifier_not_object"
    for attempt in range(2):
        response = client(
            candidate["image_path"],
            verifier_instruction,
            max_tokens=600,
            temperature=0.0,
            use_cache=(attempt == 0),
        )
        valid, reason = validate_falsifier_response(response, verifier_entities)
        if valid:
            falsifier = response
            break
        if reason != "falsifier_not_object":
            return None, reason
    if falsifier is None:
        return None, reason
    selected = generator["selected_pair"]
    pair_check = falsifier["pair_check"]
    if (
        pair_check["nearer"] != selected["nearer"]
        or pair_check["farther"] != selected["farther"]
    ):
        return None, "verifier_direction_disagreement"
    return (generator, falsifier), ""


def generate_one(
    candidate: dict[str, Any], client: Callable[..., Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    grounded, grounding_reason = ground_candidate(candidate, client)
    if grounded is None:
        return None, {
            "pairs_attempted": 0,
            "pairs_accepted": 0,
            "pair_rejections": {},
            "candidate_rejection": grounding_reason,
            "multiple_verified_pairs": False,
            "uniform_confidence": False,
            "selected_pair_index": None,
        }
    candidate = grounded
    indexed_pairs = list(enumerate(itertools.combinations(candidate["entities"], 2)))
    proposals: list[
        tuple[str, int, dict[str, Any], dict[str, Any]]
    ] = []
    pair_rejections: Counter[str] = Counter()
    for pair_index, pair_tuple in indexed_pairs:
        pair = list(pair_tuple)
        proposal, reason = generate_pair(candidate, pair, client)
        if proposal is None:
            pair_rejections[reason or "unknown"] += 1
            continue
        generator, falsifier = proposal
        proposals.append((
            _selection_key(candidate, generator),
            pair_index,
            generator,
            falsifier,
        ))
    confidences = [
        _proposal_score(generator, falsifier)
        for _, _, generator, falsifier in proposals
    ]
    stats = {
        "pairs_attempted": len(indexed_pairs),
        "pairs_accepted": len(proposals),
        "pair_rejections": dict(sorted(pair_rejections.items())),
        "candidate_rejection": "" if proposals else "no_verified_pair",
        "multiple_verified_pairs": len(proposals) > 1,
        "uniform_confidence": (
            len(proposals) > 1 and len(set(confidences)) == 1
        ),
        "selected_pair_index": None,
    }
    if not proposals:
        return None, stats
    proposals.sort(key=lambda item: item[0])
    selection_key, pair_index, generator, falsifier = proposals[0]
    stats["selected_pair_index"] = pair_index
    record = _build_record(
        candidate,
        generator,
        falsifier,
        pair_index,
        len(indexed_pairs),
        selection_key,
    )
    valid, reason = validate_depth_v4_record(record)
    if not valid:
        stats["candidate_rejection"] = f"record:{reason}"
        return None, stats
    return record, stats


def run_generation(
    candidates: list[dict[str, Any]],
    provenance: dict[str, Any],
    out: Path,
    workers: int,
    client: Callable[..., Any],
    minimum_audit_images: int = TARGET,
    minimum_canary_images: int = MIN_CANARY_ACCEPTED,
    protocol_artifact: bool = True,
    stage: str = "full",
) -> tuple[dict[str, Any], int]:
    accepted_by_sha: dict[str, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    pair_rejections: Counter[str] = Counter()
    scene_attempts: Counter[str] = Counter()
    scene_accepts: Counter[str] = Counter()
    scene_pair_attempts: Counter[str] = Counter()
    scene_pair_accepts: Counter[str] = Counter()
    pairs_attempted = 0
    pairs_accepted = 0
    multiple_verified_pair_images = 0
    uniform_confidence_images = 0
    selected_pair_indices: Counter[int] = Counter()
    attempted = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(generate_one, candidate, client): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            candidate = futures[future]
            attempted += 1
            scene = str(candidate.get("scene_type") or "unknown")
            scene_attempts[scene] += 1
            try:
                record, stats = future.result()
            except Exception as exc:
                rejected[f"runtime:{type(exc).__name__}"] += 1
                continue
            candidate_pairs_attempted = int(stats["pairs_attempted"])
            candidate_pairs_accepted = int(stats["pairs_accepted"])
            pairs_attempted += candidate_pairs_attempted
            pairs_accepted += candidate_pairs_accepted
            scene_pair_attempts[scene] += candidate_pairs_attempted
            scene_pair_accepts[scene] += candidate_pairs_accepted
            pair_rejections.update(stats["pair_rejections"])
            multiple_verified_pair_images += int(stats["multiple_verified_pairs"])
            uniform_confidence_images += int(stats["uniform_confidence"])
            selected_pair_index = stats["selected_pair_index"]
            if selected_pair_index is not None:
                selected_pair_indices[int(selected_pair_index)] += 1
            if record is None:
                rejected[stats["candidate_rejection"] or "unknown"] += 1
                continue
            accepted_by_sha[candidate["image_sha256"]] = record
            scene_accepts[scene] += 1

    accepted_pool = [
        accepted_by_sha[candidate["image_sha256"]]
        for candidate in candidates
        if candidate["image_sha256"] in accepted_by_sha
    ]
    accepted_by_scene = Counter(
        str(record.get("scene_type") or "unknown") for record in accepted_pool
    )
    accepted_by_source = Counter(
        str(record.get("source_bucket") or "unknown") for record in accepted_pool
    )
    accepted_by_cue = Counter(
        str(record["gsa_context"]["spatial"]["adjacent_depth_evidence"][0]["cue_type"])
        for record in accepted_pool
    )
    cue_gate_pass = all(
        accepted_by_cue[cue] >= minimum for cue, minimum in MIN_CUE_IMAGES.items()
    )
    parser_rejections = sum(
        pair_rejections[reason]
        for reason in ("generator_not_object", "falsifier_not_object")
    )
    runtime_or_record_rejections = sum(
        count
        for reason, count in rejected.items()
        if reason.startswith(("runtime:", "record:"))
    )
    system_error_count = parser_rejections + runtime_or_record_rejections
    canary_pass = stage != "canary" or (
        len(accepted_pool) >= minimum_canary_images and system_error_count == 0
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "depth_scope": "pairwise",
        "prompt_version": PROMPT_VERSION,
        "label_version": LABEL_VERSION,
        "stage": stage,
        "protocol_artifact": protocol_artifact,
        "artifact_kind": "generation_pool" if protocol_artifact else "canary_preview",
        "candidate_limit_images": CANARY_IMAGES if stage == "canary" else None,
        "minimum_audit_images": minimum_audit_images,
        "minimum_canary_accepted_images": minimum_canary_images,
        "candidate_images": len(candidates),
        "candidate_image_sha256s": [
            candidate["image_sha256"] for candidate in candidates
        ],
        "candidate_set_sha256": _candidate_set_sha256(candidates),
        "excluded_canary_image_sha256s": provenance.get(
            "excluded_canary_image_sha256s", []
        ),
        "excluded_canary_candidate_set_sha256": provenance.get(
            "excluded_canary_candidate_set_sha256"
        ),
        "attempted_images": attempted,
        "accepted_pool_images": len(accepted_pool),
        "emitted_records": len(accepted_pool),
        "pairs_attempted": pairs_attempted,
        "pairs_accepted": pairs_accepted,
        "pair_selection_rule": "stable_hash_after_threshold",
        "role_order_counterbalanced": True,
        "multiple_verified_pair_images": multiple_verified_pair_images,
        "uniform_confidence_images": uniform_confidence_images,
        "selected_pair_index_counts": {
            str(index): count for index, count in sorted(selected_pair_indices.items())
        },
        "audit_pool_eligible": (
            protocol_artifact
            and len(accepted_pool) >= minimum_audit_images
            and system_error_count == 0
            and cue_gate_pass
        ),
        "canary_gate": {
            "minimum_accepted_images": minimum_canary_images,
            "requires_zero_system_errors": True,
            "passed": canary_pass,
        },
        "system_errors": {
            "parser_rejections": parser_rejections,
            "runtime_or_record_rejections": runtime_or_record_rejections,
            "total": system_error_count,
        },
        "rejections": dict(sorted(rejected.items())),
        "pair_rejections": dict(sorted(pair_rejections.items())),
        "accepted_by_scene": dict(sorted(accepted_by_scene.items())),
        "accepted_by_source": dict(sorted(accepted_by_source.items())),
        "accepted_by_cue": dict(sorted(accepted_by_cue.items())),
        "cue_gate": {
            "minimum_images": dict(sorted(MIN_CUE_IMAGES.items())),
            "passed": cue_gate_pass,
        },
        "scene_strata": {
            scene: {
                "attempted_images": scene_attempts[scene],
                "accepted_images": scene_accepts[scene],
                "pairs_attempted": scene_pair_attempts[scene],
                "pairs_accepted": scene_pair_accepts[scene],
            }
            for scene in sorted(scene_attempts)
        },
        "source_provenance": provenance,
        "prompt_hashes": protocol_prompt_hashes(),
        "accepted_pool": [
            {
                "image_path": record["image_path"],
                "image_sha256": record["_v4_sources"]["image_sha256"],
                "sample_id": record["sample_id"],
                "label_sha256": content_hash(record),
                "selected_pair": [item["label"] for item in record["payload"]],
                "selection_score": record["confidence"],
                "cue_type": record["gsa_context"]["spatial"][
                    "adjacent_depth_evidence"
                ][0]["cue_type"],
            }
            for record in accepted_pool
        ],
    }
    label_path = out / "labels/S_spatial/depth.jsonl"
    label_path.parent.mkdir(parents=True, exist_ok=False)
    label_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in accepted_pool
        ),
        encoding="utf-8",
    )
    report["labels_sha256"] = file_hash(label_path)
    report["accepted_pool_sha256"] = content_hash(report["accepted_pool"])
    (out / "generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if stage == "canary":
        return report, 0 if canary_pass else 2
    return report, 0 if report["audit_pool_eligible"] else 2


def append_stage_event(stage: str, event: str, **fields: Any) -> None:
    request_log = os.environ.get("GSA_VLM_REQUEST_LOG")
    if not request_log:
        return
    path = Path(request_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "depth_v42_stage": stage,
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--detection", type=Path)
    parser.add_argument("--bbox", type=Path)
    parser.add_argument("--exclude-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("data/depth_v42_pairwise/generated")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--canary", type=int, choices=(0, CANARY_IMAGES), default=0)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target != TARGET:
        raise SystemExit(f"depth-v4.2 protocol requires --target {TARGET}")
    if args.seed != FROZEN_SEED:
        raise SystemExit(f"depth-v4.2 protocol requires --seed {FROZEN_SEED}")
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be in [1,16]")
    if (args.detection is None) != (args.bbox is None):
        raise SystemExit("--detection and --bbox must be supplied together")
    stage = "canary" if args.canary == CANARY_IMAGES else "full"
    if stage == "full" and not args.check_only:
        raise SystemExit(
            "depth-v4.2 full generation is retired after failed human canary review; "
            "use the guarded depth-v4.3 pairwise protocol"
        )
    if not args.check_only:
        endpoint = os.environ.get("GSA_VLM_URL", "").rstrip("/")
        if not endpoint or not AUTHORIZED_ENDPOINT or endpoint != AUTHORIZED_ENDPOINT.rstrip("/"):
            raise SystemExit(
                "external upload endpoint is not the explicitly authorized endpoint"
            )
        if os.environ.get("GSA_DEPTH_V42_EXTERNAL_UPLOAD_AUTHORIZED") != "1":
            raise SystemExit(
                "external upload is disabled; use the guarded depth-v4.2 launcher"
            )
        if (
            stage == "full"
            and os.environ.get("GSA_DEPTH_V42_CANARY_APPROVED") != "1"
        ):
            raise SystemExit("full depth-v4.2 generation requires canary approval")
    candidates, provenance = load_candidates(
        args.detection, args.bbox, args.index, args.exclude_manifest, args.seed
    )
    provenance = {**provenance, "pairwise_selection_seed": args.seed}
    frozen_canary_hashes = tuple(
        candidate["image_sha256"] for candidate in candidates[:CANARY_IMAGES]
    )
    frozen_canary_match = (
        frozen_canary_hashes == FROZEN_CANARY_SHA256S
        and content_hash(list(frozen_canary_hashes)) == FROZEN_CANARY_SET_SHA256
    )
    all_candidates_match = (
        _candidate_set_sha256(candidates) == FROZEN_ALL_CANDIDATE_SET_SHA256
    )
    remainder_candidates = candidates[CANARY_IMAGES:]
    remainder_match = (
        _candidate_set_sha256(remainder_candidates) == FROZEN_REMAINDER_SET_SHA256
    )
    frozen_source_match = _frozen_source_matches(provenance)
    authorized_candidate_set = (
        frozen_canary_match
        and all_candidates_match
        and remainder_match
        and frozen_source_match
    )
    if not args.check_only and not authorized_candidate_set:
        raise SystemExit(
            "external upload refused: candidates or source artifacts do not match "
            "the frozen, explicitly authorized v4 image set"
        )
    if not args.check_only and stage == "full":
        report_value = os.environ.get("GSA_DEPTH_V42_CANARY_REPORT", "")
        if not report_value:
            raise SystemExit("full depth-v4.2 generation requires a canary report path")
        provenance["canary_binding"] = validate_canary_binding(
            Path(report_value), candidates, provenance
        )
    if stage == "full":
        provenance["excluded_canary_image_sha256s"] = list(frozen_canary_hashes)
        provenance["excluded_canary_candidate_set_sha256"] = (
            FROZEN_CANARY_SET_SHA256
        )
        candidates = remainder_candidates
    else:
        candidates = candidates[:CANARY_IMAGES]
    plan = {
        "protocol": PROTOCOL,
        "depth_scope": "pairwise",
        "stage": stage,
        "protocol_artifact": args.canary == 0,
        "artifact_kind": "generation_pool" if args.canary == 0 else "canary_preview",
        "candidate_limit_images": args.canary or None,
        "minimum_audit_images": args.target,
        "minimum_canary_accepted_images": MIN_CANARY_ACCEPTED,
        "eligible_images": len(candidates),
        "candidate_image_sha256s": [
            candidate["image_sha256"] for candidate in candidates
        ],
        "candidate_set_sha256": _candidate_set_sha256(candidates),
        "frozen_canary_match": frozen_canary_match,
        "frozen_all_candidates_match": all_candidates_match,
        "frozen_remainder_match": remainder_match,
        "frozen_source_match": frozen_source_match,
        "authorized_candidate_set": authorized_candidate_set,
        "excluded_canary_image_sha256s": (
            list(frozen_canary_hashes) if stage == "full" else []
        ),
        "excluded_canary_candidate_set_sha256": (
            FROZEN_CANARY_SET_SHA256 if stage == "full" else None
        ),
        "cached_groundings": sum(
            candidate["entities"] is not None for candidate in candidates
        ),
        "groundings_to_generate": sum(
            candidate["entities"] is None for candidate in candidates
        ),
        "known_pairs_to_evaluate": sum(
            len(list(itertools.combinations(candidate["entities"], 2)))
            for candidate in candidates
            if candidate["entities"] is not None
        ),
        "image_paths": [candidate["image_path"] for candidate in candidates],
        "generator_prompt_example": generator_prompt(candidates[0]["entities"][:2])
        if candidates and candidates[0]["entities"]
        else None,
        "falsifier_prompt_protocol": falsifier_prompt([
            {"label": "<entity-b>", "bbox_2d": [1, 1, 2, 2]},
            {"label": "<entity-a>", "bbox_2d": [0, 0, 1, 1]},
        ]),
        "role_order_counterbalanced": True,
        "source_provenance": provenance,
    }
    if args.check_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        minimum_candidates = CANARY_IMAGES if args.canary else args.target
        return (
            0
            if authorized_candidate_set and len(candidates) >= minimum_candidates
            else 2
        )
    ensure_empty_output(args.out)
    from trains.flywheel.vlm_client import chat_json

    append_stage_event(
        stage,
        "stage_start",
        minimum_audit_images=args.target,
        candidates=len(candidates),
    )
    report, return_code = run_generation(
        candidates,
        provenance,
        args.out,
        args.workers,
        chat_json,
        minimum_audit_images=args.target,
        protocol_artifact=args.canary == 0,
        stage=stage,
    )
    append_stage_event(
        stage,
        "stage_end",
        return_code=return_code,
        attempted_images=report["attempted_images"],
        accepted_pool_images=report["accepted_pool_images"],
        pairs_attempted=report["pairs_attempted"],
        pairs_accepted=report["pairs_accepted"],
        audit_pool_eligible=report["audit_pool_eligible"],
        canary_passed=report["canary_gate"]["passed"],
        system_errors=report["system_errors"]["total"],
    )
    print(json.dumps(report, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
