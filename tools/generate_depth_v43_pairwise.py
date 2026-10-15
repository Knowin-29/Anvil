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
import math
import os
import re
import shutil
import sys
import threading
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

PROMPT_VERSION = "gsa-depth-v4.3-pairwise"
LABEL_VERSION = "gsa-label-v4.3-depth-pairwise"
PROTOCOL = "depth-v4.3-pairwise-generation"
HUMAN_REVIEW_PROTOCOL = "depth-v4.3-pairwise-human-canary-review"
AUTHORIZED_ENDPOINT = os.environ.get("AUTHORIZED_ENDPOINT", "")
TARGET = 40
FROZEN_SEED = 20260821
CANARY_IMAGES = 8
MIN_CANARY_ACCEPTED = 6
MIN_CONFIDENCE = 0.85
MIN_CUE_IMAGES = {"occlusion": 8, "contact_layering": 8}
CONTACT_MAX_NORMALIZED_EDGE_DISTANCE = 0.01
CONTACT_MAX_AXIS_EDGE_DISTANCE = 10.0
OCCLUSION_MIN_INTERSECTION_OVER_MIN_BOX_AREA = 0.01
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
    "higher in frame",
    "lower in frame",
    "higher in the frame",
    "lower in the frame",
    "top edge",
    "bottom edge",
    "top of the frame",
    "bottom of the frame",
    "foreground",
    "background",
    "larger",
    "smaller",
    "image height",
    "looks bigger",
    "looks smaller",
)
OBSERVATION_BY_CUE = {
    "occlusion": ("occludes", "positive_2d_intersection"),
    "contact_layering": ("contacts_or_layers_over", "overlap_or_near_touch"),
    "support_plane_perspective": ("support_plane_nearer", "support_plane_depth"),
}
GENERATOR_KEYS = {"orderable", "reason_code", "confidence", "selected_pair"}
SELECTED_PAIR_KEYS = {
    "nearer",
    "farther",
    "cue_type",
    "evidence",
    "observation",
    "confidence",
}
VERIFIER_KEYS = {"valid", "reason_code", "confidence", "pair_check"}
PAIR_CHECK_KEYS = SELECTED_PAIR_KEYS | {"supported", "counterevidence"}
OBSERVATION_KEYS = {
    "nearer",
    "farther",
    "relation",
    "geometry_claim",
    "direction_statement",
}


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


def _load_json_file(path: Path, kind: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {kind}: {path}: {exc}") from exc


def _contains_forbidden_audit_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key.casefold() in {"key", "api_key", "base64", "image_base64"}
            or _contains_forbidden_audit_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_audit_key(item) for item in value)
    return False


def _local_run_artifact(report_path: Path, value: Any) -> Path:
    base = report_path.parent.resolve()
    path = (base / str(value or "")).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise SystemExit("v4.3 run artifact escapes the canary output") from exc
    return path


def _validate_run_audit(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    audit = report.get("run_audit")
    if not isinstance(audit, dict) or _contains_forbidden_audit_key(audit):
        raise SystemExit("v4.3 canary run audit is missing or contains forbidden keys")
    request = audit.get("request_log_slice")
    response = audit.get("response_audit")
    cache = audit.get("cache_manifest")
    if not all(isinstance(item, dict) for item in (request, response, cache)):
        raise SystemExit("v4.3 canary run audit artifacts are incomplete")

    request_path = Path(str(request.get("path", "")))
    start_byte = request.get("start_byte")
    end_byte = request.get("end_byte_exclusive")
    if (
        not request_path.is_file()
        or not isinstance(start_byte, int)
        or not isinstance(end_byte, int)
        or start_byte < 0
        or end_byte <= start_byte
    ):
        raise SystemExit("invalid v4.3 request-log run slice")
    all_request_bytes = request_path.read_bytes()
    if (
        end_byte > len(all_request_bytes)
        or (start_byte and all_request_bytes[start_byte - 1:start_byte] != b"\n")
        or all_request_bytes[end_byte - 1:end_byte] != b"\n"
    ):
        raise SystemExit("v4.3 request-log byte range is not line aligned")
    request_bytes = all_request_bytes[start_byte:end_byte]
    request_lines = [line for line in request_bytes.splitlines() if line.strip()]
    try:
        request_rows = [json.loads(line) for line in request_lines]
    except json.JSONDecodeError as exc:
        raise SystemExit("invalid JSON inside v4.3 request-log run slice") from exc
    run_id = audit.get("run_id")
    lines_before = len([
        line for line in all_request_bytes[:start_byte].splitlines() if line.strip()
    ])
    if (
        request.get("sha256") != hashlib.sha256(request_bytes).hexdigest()
        or request.get("row_count") != len(request_rows)
        or request.get("start_row") != lines_before + 1
        or request.get("end_row") != lines_before + len(request_rows)
        or not request_rows
        or request_rows[0].get("event") != "stage_start"
        or request_rows[-1].get("event") != "stage_end"
        or request_rows[0].get("run_id") != run_id
        or request_rows[-1].get("run_id") != run_id
        or _contains_forbidden_audit_key(request_rows)
    ):
        raise SystemExit("v4.3 request-log run slice does not bind to this run")

    response_path = _local_run_artifact(report_path, response.get("path"))
    if not response_path.is_file() or response.get("sha256") != file_hash(response_path):
        raise SystemExit("v4.3 response audit hash mismatch")
    try:
        response_rows = [
            json.loads(line)
            for line in response_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        raise SystemExit("invalid v4.3 response audit JSONL") from exc
    if response.get("row_count") != len(response_rows):
        raise SystemExit("v4.3 response audit row count mismatch")
    if (
        len(response_rows) < int(report.get("pairs_attempted", -1))
        or _contains_forbidden_audit_key(response_rows)
    ):
        raise SystemExit("v4.3 response audit cannot account for attempted pairs")

    cache_path = _local_run_artifact(report_path, cache.get("path"))
    if not cache_path.is_file() or cache.get("sha256") != file_hash(cache_path):
        raise SystemExit("v4.3 cache manifest hash mismatch")
    cache_document = _load_json_file(cache_path, "v4.3 cache manifest")
    entries = cache_document.get("entries") if isinstance(cache_document, dict) else None
    if (
        not isinstance(entries, list)
        or cache_document.get("protocol") != PROTOCOL
        or cache_document.get("run_id") != run_id
        or cache_document.get("entries_sha256") != content_hash(entries)
        or cache.get("entries_sha256") != content_hash(entries)
        or cache.get("entry_count") != len(entries)
        or _contains_forbidden_audit_key(cache_document)
    ):
        raise SystemExit("v4.3 cache manifest contract mismatch")
    rows_by_call = {row.get("call_index"): row for row in response_rows}
    if len(rows_by_call) != len(response_rows) or len(entries) != len(response_rows):
        raise SystemExit("v4.3 call audit is not one-to-one")
    request_signatures = Counter(
        (
            row.get("image_sha256"),
            hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest(),
            row.get("model"),
            row.get("temperature"),
            row.get("max_tokens"),
        )
        for row in request_rows
        if "event" not in row
    )
    required_network_signatures: Counter[tuple[Any, ...]] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            raise SystemExit("v4.3 cache manifest contains a non-object entry")
        row = rows_by_call.get(entry.get("call_index"))
        if (
            not isinstance(row, dict)
            or row.get("response_sha256") != entry.get("response_sha256")
            or content_hash(row.get("response")) != entry.get("response_sha256")
            or row.get("error_type") != ""
        ):
            raise SystemExit("v4.3 response/cache call binding mismatch")
        snapshot = entry.get("cache_snapshot")
        snapshot_sha = entry.get("cache_snapshot_sha256")
        if snapshot is None:
            if (
                entry.get("use_cache") is not False
                or entry.get("cache_source") != "uncached_response"
                or snapshot_sha is not None
            ):
                raise SystemExit("v4.3 cache-less call is not an uncached retry")
        else:
            if (
                entry.get("use_cache") is not True
                or entry.get("cache_source") not in {"cache_hit", "network_cached"}
            ):
                raise SystemExit("v4.3 cache snapshot has an invalid source")
            snapshot_path = _local_run_artifact(report_path, snapshot)
            if not snapshot_path.is_file() or file_hash(snapshot_path) != snapshot_sha:
                raise SystemExit("v4.3 cache snapshot hash mismatch")
            snapshot_value = _load_json_file(snapshot_path, "v4.3 cache snapshot")
            if (
                not isinstance(snapshot_value, dict)
                or content_hash(snapshot_value.get("parsed"))
                != entry.get("response_sha256")
            ):
                raise SystemExit("v4.3 cache snapshot response mismatch")
        if entry.get("cache_source") != "cache_hit":
            required_network_signatures[(
                entry.get("image_sha256"),
                entry.get("prompt_sha256"),
                entry.get("model"),
                entry.get("temperature"),
                entry.get("max_tokens"),
            )] += 1
    if any(
        request_signatures[signature] < count
        for signature, count in required_network_signatures.items()
    ):
        raise SystemExit("v4.3 request-log slice misses a networked call")
    return {
        "run_id": run_id,
        "request_log_slice_sha256": request["sha256"],
        "response_audit_sha256": response["sha256"],
        "cache_manifest_sha256": cache["sha256"],
    }


def _validate_human_canary_review(
    review_path: Path,
    report_path: Path,
    report: dict[str, Any],
    labels: list[dict[str, Any]],
    run_binding: dict[str, Any],
) -> dict[str, Any]:
    if not review_path.is_file():
        raise SystemExit(f"missing v4.3 human canary review: {review_path}")
    review = _load_json_file(review_path, "v4.3 human canary review")
    required_top = {
        "schema_version",
        "protocol",
        "generation_protocol",
        "prompt_version",
        "label_version",
        "reviewer_id",
        "reviewed_at_utc",
        "canary_report_sha256",
        "labels_sha256",
        "candidate_set_sha256",
        "request_log_slice_sha256",
        "response_audit_sha256",
        "cache_manifest_sha256",
        "reviews",
    }
    if not isinstance(review, dict) or set(review) != required_top:
        raise SystemExit("v4.3 human canary review has an invalid top-level schema")
    binding_checks = (
        review.get("schema_version") == 1,
        review.get("protocol") == HUMAN_REVIEW_PROTOCOL,
        review.get("generation_protocol") == PROTOCOL,
        review.get("prompt_version") == PROMPT_VERSION,
        review.get("label_version") == LABEL_VERSION,
        isinstance(review.get("reviewer_id"), str) and bool(review["reviewer_id"].strip()),
        isinstance(review.get("reviewed_at_utc"), str)
        and bool(review["reviewed_at_utc"].strip()),
        review.get("canary_report_sha256") == file_hash(report_path),
        review.get("labels_sha256") == report.get("labels_sha256"),
        review.get("candidate_set_sha256") == report.get("candidate_set_sha256"),
        review.get("request_log_slice_sha256")
        == run_binding["request_log_slice_sha256"],
        review.get("response_audit_sha256") == run_binding["response_audit_sha256"],
        review.get("cache_manifest_sha256") == run_binding["cache_manifest_sha256"],
    )
    if not all(binding_checks):
        raise SystemExit("v4.3 human canary review does not bind to current artifacts")
    rows = review.get("reviews")
    row_keys = {
        "sample_id",
        "label_sha256",
        "image_sha256",
        "direction_correct",
        "cue_consistent",
        "evidence_entails_direction",
        "geometry_supported",
        "correct",
        "critical_error",
        "error_code",
        "notes",
    }
    if not isinstance(rows, list) or len(rows) != len(labels):
        raise SystemExit("v4.3 human review must cover every emitted canary record")
    expected = {
        label["sample_id"]: {
            "label_sha256": content_hash(label),
            "image_sha256": label["_v4_sources"]["image_sha256"],
        }
        for label in labels
    }
    seen: set[str] = set()
    correct_count = 0
    critical_count = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != row_keys:
            raise SystemExit("v4.3 human review contains an invalid record row")
        sample_id = row.get("sample_id")
        if sample_id not in expected or sample_id in seen:
            raise SystemExit("v4.3 human review has a missing/duplicate sample")
        seen.add(sample_id)
        if (
            row.get("label_sha256") != expected[sample_id]["label_sha256"]
            or row.get("image_sha256") != expected[sample_id]["image_sha256"]
        ):
            raise SystemExit("v4.3 human review row hash mismatch")
        votes = [
            row.get("direction_correct"),
            row.get("cue_consistent"),
            row.get("evidence_entails_direction"),
            row.get("geometry_supported"),
        ]
        if any(not isinstance(vote, bool) for vote in votes):
            raise SystemExit("v4.3 human review votes must be booleans")
        correct = all(votes)
        critical = row.get("critical_error")
        error_code = row.get("error_code")
        notes = row.get("notes")
        if (
            not isinstance(critical, bool)
            or not isinstance(error_code, str)
            or not isinstance(notes, str)
            or row.get("correct") is not correct
            or critical != (not correct)
            or (correct and error_code != "")
            or (not correct and not error_code.strip())
        ):
            raise SystemExit("v4.3 human review row verdict is inconsistent")
        correct_count += int(correct)
        critical_count += int(critical)
    if set(expected) != seen or correct_count < MIN_CANARY_ACCEPTED or critical_count:
        raise SystemExit(
            "v4.3 human canary review failed: require >=6 correct and zero critical errors"
        )
    return {
        "path": str(review_path.resolve()),
        "sha256": file_hash(review_path),
        "protocol": HUMAN_REVIEW_PROTOCOL,
        "reviewed_records": len(rows),
        "correct_records": correct_count,
        "critical_errors": critical_count,
        "passed": True,
    }


def validate_canary_binding(
    report_path: Path,
    human_review_path: Path,
    candidates: list[dict[str, Any]],
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    if not report_path.is_file():
        raise SystemExit(f"missing v4.3 canary report: {report_path}")
    report = _load_json_file(report_path, "v4.3 canary report")
    label_path = report_path.parent / "labels/S_spatial/depth.jsonl"
    try:
        label_lines = label_path.read_text(encoding="utf-8").splitlines()
        labels = [json.loads(line) for line in label_lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid v4.3 canary labels: {label_path}: {exc}") from exc
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
    run_binding = _validate_run_audit(report, report_path)
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
        report.get("cue_agreement_required") is True,
        report.get("geometry_gate", {}).get("required") is True,
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
            "v4.3 full generation refused: canary report does not bind to "
            "the first eight current candidates and frozen source provenance"
        )
    human_binding = _validate_human_canary_review(
        human_review_path, report_path, report, labels, run_binding
    )
    return {
        "path": str(report_path.resolve()),
        "report_sha256": file_hash(report_path),
        "candidate_set_sha256": report["candidate_set_sha256"],
        "labels_sha256": report["labels_sha256"],
        "run_audit": run_binding,
        "human_review": human_binding,
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
        "other using exactly one direct visible cue: occlusion, contact/layering, "
        "or support-plane perspective. For occlusion, the frozen boxes must have "
        "a material positive 2D intersection (at least 1% of the smaller box). "
        "For contact/layering, their boxes must overlap or be within 10 units on "
        "the frozen 0-1000 scale. Apparent size, left/right position, higher/lower frame "
        "position, top/bottom edges, or calling an entity foreground/background "
        "are invalid. Evidence must literally say 'occludes' for occlusion, "
        "'contacts' or 'layers' for contact/layering, or 'support-plane perspective' "
        "for that cue. A tie or ambiguity is unsupported. If no "
        "direction is directly supported, return exactly "
        '{"orderable":false,"reason_code":"ambiguous_pair","confidence":0.0,'
        '"selected_pair":null}. If supported, return exactly one JSON object: '
        '{"orderable":true,"reason_code":"supported_pair","confidence":0.95,'
        '"selected_pair":{"nearer":"<exact label>","farther":"<exact label>",'
        '"cue_type":"occlusion|contact_layering|support_plane_perspective",'
        '"evidence":"begin with: <nearer> is nearer to the camera than <farther>. '
        'Then state the direct visible cue naming both labels",'
        '"observation":{"nearer":"<exact nearer label>",'
        '"farther":"<exact farther label>",'
        '"relation":"occludes|contacts_or_layers_over|support_plane_nearer",'
        '"geometry_claim":"positive_2d_intersection|overlap_or_near_touch|'
        'support_plane_depth","direction_statement":"<nearer> is nearer to the '
        'camera than <farther>."},'
        '"confidence":0.95}}. Use confidence in [0.85,1.0] for a supported '
        "pair. Return JSON only."
    )

def verifier_prompt(entities: list[dict[str, Any]]) -> str:
    if len(entities) != 2:
        raise ValueError("pairwise verifier requires exactly two entities")
    return (
        "Act as a direction-blind pairwise camera-depth verifier. Judge only the "
        "two frozen entities below; no proposed direction, cue, or rationale is "
        "provided. Independently determine whether one entity is physically nearer "
        "to the camera than the other. "
        f"Frozen pair: {json.dumps(_frozen_pair(entities), ensure_ascii=False)}. "
        "Accept only direct occlusion, contact/layering, or support-plane "
        "perspective. Occlusion requires intersection of at least 1% of the smaller "
        "frozen box; contact/layering requires overlap or at most 10 units of edge "
        "separation on the 0-1000 scale. Reject apparent size, "
        "left/right position, higher/lower frame position, top/bottom edges, "
        "foreground/background alone. Evidence must literally say 'occludes' for "
        "occlusion, 'contacts' or 'layers' for contact/layering, or 'support-plane "
        "perspective' for that cue. Reject a "
        "tie, ambiguous direction, or unsupported evidence. If independently "
        "supported, return exactly "
        '{"valid":true,"reason_code":"independent_support","confidence":0.95,'
        '"pair_check":{"nearer":"<exact label>","farther":"<exact label>",'
        '"cue_type":"occlusion|contact_layering|support_plane_perspective",'
        '"evidence":"begin with: <nearer> is nearer to the camera than <farther>. '
        'Then state the direct visible cue naming both labels",'
        '"observation":{"nearer":"<exact nearer label>",'
        '"farther":"<exact farther label>",'
        '"relation":"occludes|contacts_or_layers_over|support_plane_nearer",'
        '"geometry_claim":"positive_2d_intersection|overlap_or_near_touch|'
        'support_plane_depth","direction_statement":"<nearer> is nearer to the '
        'camera than <farther>."},'
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
        "verifier_protocol_sha256": hashlib.sha256(
            verifier_prompt(list(reversed(placeholder))).encode("utf-8")
        ).hexdigest(),
    }


def _direction_statement(nearer: str, farther: str) -> str:
    return f"{nearer} is nearer to the camera than {farther}."


def _has_direction_contradiction(text: str, nearer: str, farther: str) -> bool:
    folded = text.casefold()
    near = re.escape(nearer.casefold())
    far = re.escape(farther.casefold())
    middle = r".{0,80}"
    reverse_patterns = (
        rf"\b{near}\b{middle}\b(?:farther|further|behind)\b{middle}\b{far}\b",
        rf"\b{far}\b{middle}\b(?:nearer|closer|in front of)\b{middle}\b{near}\b",
    )
    return any(re.search(pattern, folded) for pattern in reverse_patterns)


def _evidence_entails_cue(text: str, cue: str) -> bool:
    folded = text.casefold()
    if cue == "occlusion":
        return "occlud" in folded
    if cue == "contact_layering":
        return "occlud" not in folded and any(
            term in folded for term in ("contact", "layer")
        )
    if cue == "support_plane_perspective":
        return "occlud" not in folded and any(
            term in folded for term in ("support plane", "support-plane", "perspective")
        )
    return False


def _validate_pair_content(
    pair: Any,
    entities: list[dict[str, Any]],
    expected_keys: set[str],
    reason_prefix: str,
) -> tuple[bool, str]:
    if not isinstance(pair, dict) or set(pair) != expected_keys:
        return False, f"{reason_prefix}_schema"
    expected = {entity["label"] for entity in entities}
    nearer = pair.get("nearer")
    farther = pair.get("farther")
    cue = pair.get("cue_type")
    evidence = pair.get("evidence")
    if (
        not isinstance(nearer, str)
        or not isinstance(farther, str)
        or nearer == farther
        or {nearer, farther} != expected
        or cue not in ALLOWED_CUES
        or not isinstance(evidence, str)
        or len(evidence.strip()) < 30
        or not _confidence(pair.get("confidence"))
    ):
        return False, f"{reason_prefix}_schema"
    folded = evidence.casefold()
    if nearer.casefold() not in folded or farther.casefold() not in folded:
        return False, f"{reason_prefix}_evidence_missing_exact_labels"
    if any(term in folded for term in BANNED_EVIDENCE):
        return False, f"{reason_prefix}_unsupported_depth_heuristic"
    if not _evidence_entails_cue(evidence, cue):
        return False, f"{reason_prefix}_evidence_cue_mismatch"
    statement = _direction_statement(nearer, farther)
    if statement.casefold() not in folded:
        return False, f"{reason_prefix}_evidence_missing_direction_statement"
    if _has_direction_contradiction(evidence, nearer, farther):
        return False, f"{reason_prefix}_evidence_direction_contradiction"
    observation = pair.get("observation")
    relation, geometry_claim = OBSERVATION_BY_CUE[cue]
    if (
        not isinstance(observation, dict)
        or set(observation) != OBSERVATION_KEYS
        or observation.get("nearer") != nearer
        or observation.get("farther") != farther
        or observation.get("relation") != relation
        or observation.get("geometry_claim") != geometry_claim
        or observation.get("direction_statement") != statement
    ):
        return False, f"{reason_prefix}_observation"
    return True, ""


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
        set(response) != GENERATOR_KEYS
        or
        response.get("reason_code") != "supported_pair"
        or not _confidence(response.get("confidence"))
    ):
        return False, "generator_gate"
    return _validate_pair_content(
        response.get("selected_pair"), entities, SELECTED_PAIR_KEYS, "generator"
    )


def validate_verifier_response(
    response: Any, entities: list[dict[str, Any]]
) -> tuple[bool, str]:
    if not isinstance(response, dict):
        return False, "verifier_not_object"
    reason_code = response.get("reason_code", "invalid")
    if (
        set(response) != VERIFIER_KEYS
        or
        response.get("valid") is not True
        or reason_code != "independent_support"
        or not _confidence(response.get("confidence"))
    ):
        return False, f"verifier_rejected:{reason_code}"
    check = response.get("pair_check")
    if not isinstance(check, dict):
        return False, "verifier_pair"
    if check.get("supported") is not True or check.get("counterevidence") != "":
        return False, "verifier_pair"
    return _validate_pair_content(
        check, entities, PAIR_CHECK_KEYS, "verifier"
    )


def _evidence(selected_pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "nearer": selected_pair["nearer"],
        "farther": selected_pair["farther"],
        "cue_type": selected_pair["cue_type"],
        "evidence": selected_pair["evidence"],
        "observation": selected_pair["observation"],
    }


def _proposal_score(
    generator: dict[str, Any], verifier: dict[str, Any]
) -> float:
    selected = generator["selected_pair"]
    check = verifier["pair_check"]
    return min(
        float(generator["confidence"]),
        float(selected.get("confidence", generator["confidence"])),
        float(verifier["confidence"]),
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


def geometry_gate(
    entities: list[dict[str, Any]], cue_type: str
) -> tuple[bool, str, dict[str, Any]]:
    if len(entities) != 2:
        raise ValueError("geometry gate requires exactly two entities")
    boxes: list[list[float]] = []
    for entity in entities:
        bbox = entity.get("bbox_2d")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in bbox
            )
        ):
            return False, "geometry:invalid_frozen_bbox", {
                "cue_type": cue_type,
                "passed": False,
                "intersection_width": None,
                "intersection_height": None,
                "intersection_area": None,
                "intersection_over_min_box_area": None,
                "min_intersection_over_min_box_area": (
                    OCCLUSION_MIN_INTERSECTION_OVER_MIN_BOX_AREA
                ),
                "normalized_edge_distance": None,
                "max_normalized_edge_distance": CONTACT_MAX_NORMALIZED_EDGE_DISTANCE,
                "max_axis_edge_distance": None,
                "max_axis_edge_distance_limit": CONTACT_MAX_AXIS_EDGE_DISTANCE,
            }
        values = [float(value) for value in bbox]
        if values[0] >= values[2] or values[1] >= values[3]:
            return False, "geometry:invalid_frozen_bbox", {
                "cue_type": cue_type,
                "passed": False,
                "intersection_width": None,
                "intersection_height": None,
                "intersection_area": None,
                "intersection_over_min_box_area": None,
                "min_intersection_over_min_box_area": (
                    OCCLUSION_MIN_INTERSECTION_OVER_MIN_BOX_AREA
                ),
                "normalized_edge_distance": None,
                "max_normalized_edge_distance": CONTACT_MAX_NORMALIZED_EDGE_DISTANCE,
                "max_axis_edge_distance": None,
                "max_axis_edge_distance_limit": CONTACT_MAX_AXIS_EDGE_DISTANCE,
            }
        boxes.append(values)
    a, b = boxes
    intersection_width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection_area = intersection_width * intersection_height
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    intersection_over_min_box_area = intersection_area / min(a_area, b_area)
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    normalized_edge_distance = math.hypot(dx, dy) / (1000.0 * math.sqrt(2.0))
    max_axis_edge_distance = max(dx, dy)
    if cue_type == "occlusion":
        passed = (
            intersection_width > 0.0
            and intersection_height > 0.0
            and intersection_over_min_box_area
            >= OCCLUSION_MIN_INTERSECTION_OVER_MIN_BOX_AREA
        )
        reason = "" if passed else "geometry:occlusion_insufficient_2d_intersection"
    elif cue_type == "contact_layering":
        passed = (
            intersection_area > 0.0
            or (
                normalized_edge_distance <= CONTACT_MAX_NORMALIZED_EDGE_DISTANCE
                and max_axis_edge_distance <= CONTACT_MAX_AXIS_EDGE_DISTANCE
            )
        )
        reason = "" if passed else "geometry:contact_not_overlap_or_near_touch"
    elif cue_type == "support_plane_perspective":
        passed = True
        reason = ""
    else:
        passed = False
        reason = "geometry:unsupported_cue"
    return passed, reason, {
        "cue_type": cue_type,
        "passed": passed,
        "intersection_width": intersection_width,
        "intersection_height": intersection_height,
        "intersection_area": intersection_area,
        "intersection_over_min_box_area": intersection_over_min_box_area,
        "min_intersection_over_min_box_area": (
            OCCLUSION_MIN_INTERSECTION_OVER_MIN_BOX_AREA
        ),
        "normalized_edge_distance": normalized_edge_distance,
        "max_normalized_edge_distance": CONTACT_MAX_NORMALIZED_EDGE_DISTANCE,
        "max_axis_edge_distance": max_axis_edge_distance,
        "max_axis_edge_distance_limit": CONTACT_MAX_AXIS_EDGE_DISTANCE,
    }


def _build_record(
    candidate: dict[str, Any],
    generator: dict[str, Any],
    verifier: dict[str, Any],
    pair_index: int,
    pair_count: int,
    selection_key: str,
    geometry: dict[str, Any],
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
    score = _proposal_score(generator, verifier)
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
        "sample_id": f"v43_depth_{identity[:20]}",
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
            "depth_v43_pairwise": {
                "generator": generator,
                "verifier": verifier,
                "role_order_counterbalanced": True,
                "cue_agreement": True,
                "geometry": geometry,
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
                "exact_cue_agreement": 1.0,
                "frozen_bbox_geometry": 1.0,
            },
            "verdict": "verified_pair",
        },
    }


def generate_pair(
    candidate: dict[str, Any],
    entities: list[dict[str, Any]],
    client: Callable[..., Any],
) -> tuple[
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None, str
]:
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
    verifier_instruction = verifier_prompt(verifier_entities)
    verifier: dict[str, Any] | None = None
    reason = "verifier_not_object"
    for attempt in range(2):
        response = client(
            candidate["image_path"],
            verifier_instruction,
            max_tokens=600,
            temperature=0.0,
            use_cache=(attempt == 0),
        )
        valid, reason = validate_verifier_response(response, verifier_entities)
        if valid:
            verifier = response
            break
        if reason != "verifier_not_object":
            return None, reason
    if verifier is None:
        return None, reason
    selected = generator["selected_pair"]
    pair_check = verifier["pair_check"]
    if (
        pair_check["nearer"] != selected["nearer"]
        or pair_check["farther"] != selected["farther"]
    ):
        return None, "verifier_direction_disagreement"
    if pair_check["cue_type"] != selected["cue_type"]:
        return None, "verifier_cue_disagreement"
    geometry_passed, geometry_reason, geometry = geometry_gate(
        entities, selected["cue_type"]
    )
    if not geometry_passed:
        return None, geometry_reason
    return (generator, verifier, geometry), ""


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
        tuple[str, int, dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    pair_rejections: Counter[str] = Counter()
    for pair_index, pair_tuple in indexed_pairs:
        pair = list(pair_tuple)
        proposal, reason = generate_pair(candidate, pair, client)
        if proposal is None:
            pair_rejections[reason or "unknown"] += 1
            continue
        generator, verifier, geometry = proposal
        proposals.append((
            _selection_key(candidate, generator),
            pair_index,
            generator,
            verifier,
            geometry,
        ))
    confidences = [
        _proposal_score(generator, verifier)
        for _, _, generator, verifier, _ in proposals
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
    selection_key, pair_index, generator, verifier, geometry = proposals[0]
    stats["selected_pair_index"] = pair_index
    record = _build_record(
        candidate,
        generator,
        verifier,
        pair_index,
        len(indexed_pairs),
        selection_key,
        geometry,
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
        for reason in ("generator_not_object", "verifier_not_object")
    )
    runtime_or_record_rejections = sum(
        count
        for reason, count in rejected.items()
        if reason.startswith(("runtime:", "record:"))
    )
    system_error_count = parser_rejections + runtime_or_record_rejections
    geometry_rejections = {
        reason: count
        for reason, count in sorted(pair_rejections.items())
        if reason.startswith("geometry:")
    }
    canary_pass = stage != "canary" or (
        len(accepted_pool) >= minimum_canary_images and system_error_count == 0
    )
    canary_binding = provenance.get("canary_binding")
    human_canary_pass = (
        stage != "full"
        or isinstance(canary_binding, dict)
        and canary_binding.get("human_review", {}).get("passed") is True
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
        "canary_binding": canary_binding,
        "attempted_images": attempted,
        "accepted_pool_images": len(accepted_pool),
        "emitted_records": len(accepted_pool),
        "pairs_attempted": pairs_attempted,
        "pairs_accepted": pairs_accepted,
        "pair_selection_rule": "stable_hash_after_threshold",
        "role_order_counterbalanced": True,
        "cue_agreement_required": True,
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
            and human_canary_pass
        ),
        "canary_gate": {
            "minimum_accepted_images": minimum_canary_images,
            "requires_zero_system_errors": True,
            "passed": canary_pass,
        },
        "human_canary_gate": {
            "required_for_full": True,
            "protocol": HUMAN_REVIEW_PROTOCOL,
            "minimum_correct_records": MIN_CANARY_ACCEPTED,
            "requires_every_emitted_record_reviewed": True,
            "requires_zero_critical_errors": True,
            "passed": human_canary_pass if stage == "full" else None,
        },
        "system_errors": {
            "parser_rejections": parser_rejections,
            "runtime_or_record_rejections": runtime_or_record_rejections,
            "total": system_error_count,
        },
        "rejections": dict(sorted(rejected.items())),
        "pair_rejections": dict(sorted(pair_rejections.items())),
        "geometry_rejections": geometry_rejections,
        "accepted_by_scene": dict(sorted(accepted_by_scene.items())),
        "accepted_by_source": dict(sorted(accepted_by_source.items())),
        "accepted_by_cue": dict(sorted(accepted_by_cue.items())),
        "cue_gate": {
            "minimum_images": dict(sorted(MIN_CUE_IMAGES.items())),
            "passed": cue_gate_pass,
        },
        "geometry_gate": {
            "required": True,
            "bbox_coordinate_scale": 1000,
            "occlusion_min_intersection_over_min_box_area": (
                OCCLUSION_MIN_INTERSECTION_OVER_MIN_BOX_AREA
            ),
            "contact_max_normalized_edge_distance": (
                CONTACT_MAX_NORMALIZED_EDGE_DISTANCE
            ),
            "contact_max_axis_edge_distance": CONTACT_MAX_AXIS_EDGE_DISTANCE,
            "rejections": geometry_rejections,
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


class RunAuditTracker:
    """Bind one run's request slice, parsed responses, and cache snapshots."""

    def __init__(self, out: Path, request_log: Path, cache_dir: Path, stage: str):
        self.out = out
        self.request_log = request_log.resolve()
        self.cache_dir = cache_dir.resolve()
        self.stage = stage
        self.run_id = hashlib.sha256(
            f"{PROTOCOL}|{stage}|{os.getpid()}|{time.time_ns()}".encode("utf-8")
        ).hexdigest()[:24]
        self._lock = threading.Lock()
        self._next_call = 0
        self._response_rows: list[dict[str, Any]] = []
        self._manifest_entries: list[dict[str, Any]] = []
        self._start_byte = 0
        self._start_row = 1
        self.snapshot_dir = out / "cache_snapshot"

    def begin(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=False)
        self.request_log.parent.mkdir(parents=True, exist_ok=True)
        if self.request_log.is_file():
            data = self.request_log.read_bytes()
            if data and not data.endswith(b"\n"):
                raise SystemExit("v4.3 request log must end on a complete JSONL row")
            self._start_byte = len(data)
            self._start_row = len([line for line in data.splitlines() if line.strip()]) + 1
        append_stage_event(self.stage, "stage_start", run_id=self.run_id)

    def wrap(self, client: Callable[..., Any]) -> Callable[..., Any]:
        def audited_client(
            image_path: str, prompt: str, **kwargs: Any
        ) -> Any:
            with self._lock:
                call_index = self._next_call
                self._next_call += 1
            image_sha256 = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
            model = kwargs.get("model") or os.environ.get(
                "GSA_VLM_MODEL", "Qwen3.5-397B-A17B"
            )
            temperature = kwargs.get("temperature", 0.2)
            max_tokens = kwargs.get("max_tokens", 512)
            use_cache = kwargs.get("use_cache", True)
            cache_locator = hashlib.sha256(
                f"{model}|{temperature}|{prompt}|{image_sha256}".encode("utf-8")
            ).hexdigest()
            cache_file = self.cache_dir / f"{cache_locator}.json"
            cache_existed_before = bool(use_cache and cache_file.is_file())
            response: Any = None
            error_type = ""
            try:
                response = client(image_path, prompt, **kwargs)
            except Exception as exc:
                error_type = type(exc).__name__
                raise
            finally:
                snapshot_relative: str | None = None
                snapshot_sha256: str | None = None
                if use_cache and cache_file.is_file():
                    snapshot_path = self.snapshot_dir / f"{call_index:06d}.json"
                    shutil.copyfile(cache_file, snapshot_path)
                    snapshot_relative = str(snapshot_path.relative_to(self.out))
                    snapshot_sha256 = file_hash(snapshot_path)
                response_sha256 = content_hash(response)
                common = {
                    "call_index": call_index,
                    "image_sha256": image_sha256,
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "use_cache": bool(use_cache),
                    "cache_source": (
                        "cache_hit"
                        if cache_existed_before
                        else "network_cached"
                        if snapshot_relative is not None
                        else "uncached_response"
                    ),
                    "cache_snapshot": snapshot_relative,
                    "cache_snapshot_sha256": snapshot_sha256,
                    "response_sha256": response_sha256,
                    "error_type": error_type,
                }
                response_row = {
                    **common,
                    "image_path_local_only": image_path,
                    "prompt": prompt,
                    "response": response,
                }
                with self._lock:
                    self._response_rows.append(response_row)
                    self._manifest_entries.append(common)
            return response

        return audited_client

    def finalize(self, report: dict[str, Any], return_code: int) -> dict[str, Any]:
        append_stage_event(
            self.stage,
            "stage_end",
            run_id=self.run_id,
            return_code=return_code,
            attempted_images=report["attempted_images"],
            accepted_pool_images=report["accepted_pool_images"],
            pairs_attempted=report["pairs_attempted"],
            pairs_accepted=report["pairs_accepted"],
            audit_pool_eligible=report["audit_pool_eligible"],
            canary_passed=report["canary_gate"]["passed"],
            system_errors=report["system_errors"]["total"],
        )
        request_bytes = self.request_log.read_bytes()
        end_byte = len(request_bytes)
        run_slice = request_bytes[self._start_byte:end_byte]
        request_rows = [line for line in run_slice.splitlines() if line.strip()]

        response_rows = sorted(self._response_rows, key=lambda row: row["call_index"])
        response_path = self.out / "run_response_audit.jsonl"
        response_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in response_rows
            ),
            encoding="utf-8",
        )
        entries = sorted(
            self._manifest_entries, key=lambda entry: entry["call_index"]
        )
        cache_document = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "run_id": self.run_id,
            "entries": entries,
            "entries_sha256": content_hash(entries),
        }
        cache_path = self.out / "run_cache_manifest.json"
        cache_path.write_text(
            json.dumps(cache_document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "run_id": self.run_id,
            "request_log_slice": {
                "path": str(self.request_log),
                "start_byte": self._start_byte,
                "end_byte_exclusive": end_byte,
                "start_row": self._start_row,
                "end_row": self._start_row + len(request_rows) - 1,
                "row_count": len(request_rows),
                "sha256": hashlib.sha256(run_slice).hexdigest(),
            },
            "response_audit": {
                "path": str(response_path.relative_to(self.out)),
                "row_count": len(response_rows),
                "sha256": file_hash(response_path),
            },
            "cache_manifest": {
                "path": str(cache_path.relative_to(self.out)),
                "entry_count": len(entries),
                "entries_sha256": content_hash(entries),
                "sha256": file_hash(cache_path),
            },
        }


def append_stage_event(stage: str, event: str, **fields: Any) -> None:
    request_log = os.environ.get("GSA_VLM_REQUEST_LOG")
    if not request_log:
        return
    path = Path(request_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "depth_v43_stage": stage,
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
        "--out", type=Path, default=Path("data/depth_v43_pairwise/generated")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--canary", type=int, choices=(0, CANARY_IMAGES), default=0)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--canary-report", type=Path)
    parser.add_argument("--human-review", type=Path)
    parser.add_argument("--validate-canary-binding-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target != TARGET:
        raise SystemExit(f"depth-v4.3 protocol requires --target {TARGET}")
    if args.seed != FROZEN_SEED:
        raise SystemExit(f"depth-v4.3 protocol requires --seed {FROZEN_SEED}")
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be in [1,16]")
    if (args.detection is None) != (args.bbox is None):
        raise SystemExit("--detection and --bbox must be supplied together")
    stage = "canary" if args.canary == CANARY_IMAGES else "full"
    offline_mode = args.check_only or args.validate_canary_binding_only
    if args.validate_canary_binding_only and stage != "full":
        raise SystemExit("canary binding preflight is only valid for the full stage")
    if not offline_mode:
        endpoint = os.environ.get("GSA_VLM_URL", "").rstrip("/")
        if not endpoint or not AUTHORIZED_ENDPOINT or endpoint != AUTHORIZED_ENDPOINT.rstrip("/"):
            raise SystemExit(
                "external upload endpoint is not the explicitly authorized endpoint"
            )
        if os.environ.get("GSA_DEPTH_V43_EXTERNAL_UPLOAD_AUTHORIZED") != "1":
            raise SystemExit(
                "external upload is disabled; use the guarded depth-v4.3 launcher"
            )
        if (
            stage == "full"
            and os.environ.get("GSA_DEPTH_V43_CANARY_APPROVED") != "1"
        ):
            raise SystemExit(
                "full depth-v4.3 generation requires explicit canary approval "
                "plus a hash-bound human review"
            )
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
    if stage == "full" and (args.validate_canary_binding_only or not offline_mode):
        report_path = args.canary_report
        review_path = args.human_review
        if report_path is None or review_path is None:
            raise SystemExit(
                "full depth-v4.3 generation requires canary report and human review paths"
            )
        provenance["canary_binding"] = validate_canary_binding(
            report_path, review_path, candidates, provenance
        )
        if args.validate_canary_binding_only:
            print(json.dumps(provenance["canary_binding"], indent=2))
            return 0
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
        "human_canary_review_protocol": HUMAN_REVIEW_PROTOCOL,
        "full_requires_hash_bound_human_review": True,
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
        "verifier_prompt_protocol": verifier_prompt([
            {"label": "<entity-b>", "bbox_2d": [1, 1, 2, 2]},
            {"label": "<entity-a>", "bbox_2d": [0, 0, 1, 1]},
        ]),
        "role_order_counterbalanced": True,
        "cue_agreement_required": True,
        "geometry_gate": {
            "occlusion_min_intersection_over_min_box_area": (
                OCCLUSION_MIN_INTERSECTION_OVER_MIN_BOX_AREA
            ),
            "contact_max_normalized_edge_distance": (
                CONTACT_MAX_NORMALIZED_EDGE_DISTANCE
            ),
            "contact_max_axis_edge_distance": CONTACT_MAX_AXIS_EDGE_DISTANCE,
        },
        "prompt_hashes": protocol_prompt_hashes(),
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
    request_log_value = os.environ.get("GSA_VLM_REQUEST_LOG", "")
    cache_value = os.environ.get("GSA_VLM_CACHE", "")
    if not request_log_value or not cache_value:
        raise SystemExit("v4.3 requires explicit request-log and cache paths")
    from trains.flywheel.vlm_client import chat_json

    tracker = RunAuditTracker(
        args.out, Path(request_log_value), Path(cache_value), stage
    )
    tracker.begin()
    report, return_code = run_generation(
        candidates,
        provenance,
        args.out,
        args.workers,
        tracker.wrap(chat_json),
        minimum_audit_images=args.target,
        protocol_artifact=args.canary == 0,
        stage=stage,
    )
    report["run_audit"] = tracker.finalize(report, return_code)
    (args.out / "generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
