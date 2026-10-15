#!/usr/bin/env python3
"""Validate and compile the v5.2 uniquely referential ScanNet G->S audit."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import scannet_rgb_depth_v52_interface as public_interface  # noqa: E402
from tools import build_depth_v50_scannet_sensor as sensor  # noqa: E402
from tools import build_scannet_rgb_auditable_depth_pack as v51_filter  # noqa: E402
from tools import build_scannet_rgb_depth_v52_unique_referent_formal as builder  # noqa: E402


PACK_VALID = "PACK_VALID_PENDING_HUMAN_REVIEW"
PENDING_REVIEW = "PENDING_HUMAN_REVIEW"
PENDING_ADJUDICATION = "PENDING_REVIEWER_C_ADJUDICATION"
FULL_REHASH_REQUIRED = "REVIEW_COMPLETE_FULL_SOURCE_REHASH_REQUIRED"
FORMAL_FAILED = "FORMAL_REVIEW_FAILED_REBUILD_REQUIRED"
FORMAL_PASS = "FORMAL_PASS"

ALLOWED_ERROR_CODES = {
    "wrong_object",
    "bbox_misaligned",
    "identity_ambiguous",
    "ambiguous_depth",
    "wrong_geometry",
    "wrong_relation_or_order",
    "occlusion_or_truncation",
    "missing_content",
    "unsuitable_source",
    "schema_or_format",
    "other",
}


class CheckError(RuntimeError):
    """A pack, vote, or adjudication violates the v5.2 evidence contract."""


def _translate(exc: Exception) -> CheckError:
    return CheckError(str(exc))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return builder._read_json(path)
    except builder.PackError as exc:
        raise _translate(exc) from exc


def _read_jsonl(path: Path, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    try:
        return builder._read_jsonl(path, allow_missing=allow_missing)
    except builder.PackError as exc:
        raise _translate(exc) from exc


def _verify_relative_binding(pack: Path, binding: Any, section: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
        raise CheckError(f"relative artifact binding schema drift: {section}")
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise CheckError(f"relative artifact path drift: {section}")
    path = (pack / relative).resolve()
    try:
        path.relative_to(pack)
    except ValueError as exc:
        raise CheckError(f"relative artifact escapes pack: {section}") from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != binding.get("bytes")
        or builder.file_sha256(path) != binding.get("sha256")
    ):
        raise CheckError(f"relative artifact drift: {section}")
    return path


def _verify_absolute_binding(binding: Any, section: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
        raise CheckError(f"absolute artifact binding schema drift: {section}")
    path = Path(str(binding.get("path", ""))).expanduser().resolve()
    if (
        str(path) != binding.get("path")
        or not path.is_file()
        or path.stat().st_size != binding.get("bytes")
        or builder.file_sha256(path) != binding.get("sha256")
    ):
        raise CheckError(f"absolute artifact drift: {section}")
    return path


def _key_rows(rows: list[dict[str, Any]], key: str, context: str) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = row.get(key)
        if not isinstance(identity, str) or not identity or identity in keyed:
            raise CheckError(f"{context} has a missing or duplicate {key}: {identity!r}")
        keyed[identity] = row
    return keyed


def _latest_votes(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path, allow_missing=True):
        snapshot = row.get("label_snapshot")
        sample_id = snapshot.get("sample_id") if isinstance(snapshot, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise CheckError(f"vote lacks label_snapshot.sample_id: {path}")
        latest[sample_id] = row
    return latest


def _validate_vote(
    row: dict[str, Any], label: dict[str, Any], side: str
) -> None:
    sample_id = label["sample_id"]
    if row.get("case_id") != sample_id:
        raise CheckError(f"reviewer {side} case identity drift: {sample_id}")
    if builder.canonical_hash(row.get("label_snapshot")) != builder.canonical_hash(label):
        raise CheckError(f"reviewer {side} label snapshot drift: {sample_id}")
    for field in ("annotation_id", "annotator_key", "timestamp"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise CheckError(f"reviewer {side} lacks {field}: {sample_id}")
    correct = row.get("correct")
    if not isinstance(correct, bool):
        raise CheckError(f"reviewer {side} has non-Boolean decision: {sample_id}")
    error_code = row.get("error_code")
    comment = row.get("comment")
    if correct:
        if error_code not in (None, ""):
            raise CheckError(f"positive decision has an error code: {side}/{sample_id}")
    elif (
        error_code not in ALLOWED_ERROR_CODES
        or not isinstance(comment, str)
        or not comment.strip()
    ):
        raise CheckError(
            f"negative decision needs an error code and concrete comment: {side}/{sample_id}"
        )


def _single_reviewer_key(
    votes: dict[str, dict[str, Any]], side: str
) -> str | None:
    keys = {row.get("annotator_key") for row in votes.values()}
    if not keys:
        return None
    if len(keys) != 1 or None in keys:
        raise CheckError(f"reviewer {side} stream contains multiple reviewer keys")
    key = next(iter(keys))
    if not isinstance(key, str) or not key:
        raise CheckError(f"reviewer {side} key is invalid")
    return key


def _compare_v3_binding(
    expected: dict[str, Any], *, verify_artifacts: bool
) -> None:
    try:
        live = builder.load_v3_frame_root(
            Path(str(expected.get("path", ""))), verify_artifacts=verify_artifacts
        )
    except builder.PackError as exc:
        raise CheckError(f"bound v3 frame root failed: {exc}") from exc
    if not verify_artifacts:
        live["artifacts_rehashed"] = expected.get("artifacts_rehashed")
    if live != expected:
        raise CheckError(f"bound v3 frame-root binding drift: {expected.get('path')}")


def validate_pack(
    pack: Path,
    *,
    rehash_source_artifacts: bool = True,
    rehash_v3_outputs: bool = True,
) -> dict[str, Any]:
    pack = pack.expanduser().resolve()
    manifest_path = pack / "manifest.json"
    manifest = _read_json(manifest_path)
    integrity = manifest.get("integrity") or {}
    if integrity.get("manifest_payload_sha256") != builder.manifest_payload_hash(manifest):
        raise CheckError("v5.2 manifest payload hash drift")
    selected = manifest.get("selected")
    selection = manifest.get("selection") or {}
    review_contract = manifest.get("review_contract") or {}
    exclusions = manifest.get("exclusion_checks") or {}
    interface = manifest.get("interface_contract") or {}
    if (
        manifest.get("schema_version") != builder.SCHEMA_VERSION
        or manifest.get("protocol_version") != builder.PACK_PROTOCOL
        or manifest.get("audit_round") != builder.AUDIT_ROUND
        or manifest.get("audit_contract") != builder.AUDIT_CONTRACT
        or manifest.get("stage") != "formal_pending_review"
        or manifest.get("records") != builder.FORMAL_TARGET
        or not isinstance(selected, list)
        or len(selected) != builder.FORMAL_TARGET
        or integrity.get("selected_sha256") != builder.canonical_hash(selected)
        or selection.get("reused_target") != builder.REUSED_TARGET
        or selection.get("new_target") != builder.NEW_TARGET
        or selection.get("selected_reused") != builder.REUSED_TARGET
        or selection.get("selected_new") != builder.NEW_TARGET
        or selection.get("thresholds") != v51_filter.STRICT_THRESHOLDS
        or selection.get("referent_method") != builder.REFERENT_METHOD
        or selection.get("target_category_visible_instance_counts") != [1, 1]
        or selection.get("nearer_slot_balance") != builder.EXPECTED_AB_BALANCE
        or review_contract.get("new_decisions_required_total")
        != 2 * builder.NEW_TARGET
        or review_contract.get("builder_wrote_annotation_decisions") != 0
        or interface.get("protocol") != public_interface.INTERFACE_PROTOCOL
        or interface.get("oracle_matching_of_predicted_detections_allowed") is not False
        or interface.get("multiple_detections_per_slot_allowed") is not False
        or (manifest.get("training_materialization") or {}).get("eligible") is not False
    ):
        raise CheckError("v5.2 manifest contract drift")
    if (
        exclusions.get("new_vs_all_prior_audits_and_canaries_overlap") != []
        or exclusions.get("new_vs_v51_nonreuse_33_overlap") != []
        or exclusions.get("new_vs_v51_reused_7_overlap") != []
        or exclusions.get("combined_scene_duplicates") != 0
        or exclusions.get("partial_v3_roots_consumed") != 0
    ):
        raise CheckError("v5.2 manifest contains a forbidden overlap or partial root")

    provenance = manifest.get("provenance") or {}
    for name in ("builder", "checker", "public_interface"):
        _verify_absolute_binding(provenance.get(name), f"implementation/{name}")
    paths = {
        name: _verify_relative_binding(pack, provenance.get(name), name)
        for name in (
            "source_records",
            "referent_evidence",
            "reused_v51_decisions",
            "all_labels",
            "reviewer_a_labels",
            "reviewer_b_labels",
        )
    }

    expected_v51 = provenance.get("v51_formal_binding")
    if not isinstance(expected_v51, dict):
        raise CheckError("v5.2 manifest lacks a v5.1 formal binding")
    try:
        live_v51 = builder.load_v51_reuse_binding(
            Path(expected_v51["files"]["manifest"]["path"]).parent,
            Path(expected_v51["files"]["report"]["path"]),
        )
    except (KeyError, builder.PackError) as exc:
        raise CheckError(f"v5.1 formal reuse binding failed: {exc}") from exc
    live_v51_public = {
        key: copy.deepcopy(value)
        for key, value in live_v51.items()
        if key not in {"reusable", "manifest"}
    }
    if live_v51_public != expected_v51:
        raise CheckError("v5.1 formal reuse binding drift")

    root_bindings = provenance.get("v3_frame_root_bindings")
    if not isinstance(root_bindings, list) or not all(
        isinstance(item, dict) for item in root_bindings
    ):
        raise CheckError("v5.2 v3 frame-root bindings are malformed")
    for binding in root_bindings:
        _compare_v3_binding(binding, verify_artifacts=rehash_v3_outputs)
    bound_root_paths = {str(Path(item["path"]).resolve()) for item in root_bindings}

    all_labels_rows = _read_jsonl(paths["all_labels"])
    labels_a_rows = _read_jsonl(paths["reviewer_a_labels"])
    labels_b_rows = _read_jsonl(paths["reviewer_b_labels"])
    if labels_a_rows != labels_b_rows or len(labels_a_rows) != builder.NEW_TARGET:
        raise CheckError("A/B v5.2 review queues must be identical 33-row projections")
    all_labels = _key_rows(all_labels_rows, "sample_id", "all labels")
    labels_a = _key_rows(labels_a_rows, "sample_id", "reviewer A labels")
    labels_b = _key_rows(labels_b_rows, "sample_id", "reviewer B labels")
    wrappers = _key_rows(_read_jsonl(paths["source_records"]), "sample_id", "source wrappers")
    evidence_rows = _key_rows(
        _read_jsonl(paths["referent_evidence"]), "sample_id", "referent evidence"
    )
    reused_rows = _key_rows(
        _read_jsonl(paths["reused_v51_decisions"]), "sample_id", "reused decisions"
    )
    selected_by_id = _key_rows(selected, "sample_id", "selected manifest")
    expected_ids = set(selected_by_id)
    if (
        len(all_labels) != builder.FORMAL_TARGET
        or set(all_labels) != expected_ids
        or set(wrappers) != expected_ids
        or set(evidence_rows) != expected_ids
        or len(reused_rows) != builder.REUSED_TARGET
    ):
        raise CheckError("v5.2 manifest/label/provenance coverage drift")

    reused_ids = {
        sample_id
        for sample_id, item in selected_by_id.items()
        if item.get("lineage") == "reused_v51"
    }
    new_ids = {
        sample_id
        for sample_id, item in selected_by_id.items()
        if item.get("lineage") == "new_v52"
    }
    if (
        len(reused_ids) != builder.REUSED_TARGET
        or len(new_ids) != builder.NEW_TARGET
        or reused_ids | new_ids != expected_ids
        or set(labels_a) != new_ids
        or set(labels_b) != new_ids
        or set(reused_rows) != reused_ids
    ):
        raise CheckError("v5.2 reused/new review partition drift")

    scenes: set[str] = set()
    images: set[str] = set()
    sources: set[str] = set()
    nearer = Counter()
    file_cache: dict[str, str] = {}
    projection_cache: dict[tuple[str, str], bytes] = {}
    source_file_cache: dict[str, str] = {}
    source_member_cache: dict[tuple[str, str], str] = {}
    source_errors: dict[str, list[str]] = {}
    live_v51_reused = {
        item["reuse_binding"]["v51_sample_id"]: item
        for item in live_v51["reusable"]
    }
    prior_scenes = builder._prior_scene_ids(live_v51)

    for sample_id, item in selected_by_id.items():
        label = all_labels[sample_id]
        wrapper = wrappers[sample_id]
        evidence_wrapper = evidence_rows[sample_id]
        record = wrapper.get("source_record")
        if not isinstance(record, dict):
            raise CheckError(f"source wrapper lacks a record: {sample_id}")
        scene_id = item.get("scene_id")
        image_sha = item.get("image_sha256")
        source_id = item.get("source_sample_id")
        if scene_id in scenes or image_sha in images or source_id in sources:
            raise CheckError("v5.2 pack is not scene/image/source unique")
        scenes.add(scene_id)
        images.add(image_sha)
        sources.add(source_id)
        if (
            wrapper.get("source_sample_id") != source_id
            or wrapper.get("source_record_sha256") != builder.canonical_hash(record)
            or item.get("source_record_sha256") != builder.canonical_hash(record)
            or item.get("source_payload_sha256")
            != builder.canonical_hash(record.get("payload"))
            or item.get("label_sha256") != builder.canonical_hash(label)
            or label.get("source_record_sha256") != builder.canonical_hash(record)
            or label.get("source_payload_sha256")
            != builder.canonical_hash(record.get("payload"))
            or label.get("lineage") != item.get("lineage")
            or label.get("review_required_in_v52") is not (sample_id in new_ids)
        ):
            raise CheckError(f"v5.2 source/label identity drift: {sample_id}")
        structural_errors = sensor.validate_record(record)
        features, feature_errors = v51_filter.candidate_features(record)
        if structural_errors or feature_errors:
            raise CheckError(
                f"v5.2 source no longer passes RGB/sensor filters: {sample_id} "
                f"{structural_errors + feature_errors}"
            )
        try:
            live_evidence = builder._referent_evidence(
                record,
                file_hash_cache=file_cache,
                member_cache=projection_cache,
                verify_zip_hash=False,
            )
        except builder.PackError as exc:
            raise CheckError(f"v5.2 referent evidence failed: {exc}") from exc
        evidence = evidence_wrapper.get("evidence")
        if (
            evidence_wrapper.get("source_sample_id") != source_id
            or evidence_wrapper.get("evidence_sha256") != builder.canonical_hash(evidence)
            or item.get("referent_evidence_sha256") != builder.canonical_hash(evidence)
            or live_evidence != evidence
            or live_evidence.get("target_category_visible_instance_counts") != [1, 1]
            or live_evidence.get("both_target_categories_unique") is not True
        ):
            raise CheckError(f"v5.2 exact-one referent evidence drift: {sample_id}")
        image = builder._resolve_path(record.get("image_path"))
        if builder.file_sha256(image) != image_sha:
            raise CheckError(f"v5.2 image hash drift: {sample_id}")
        candidate = {
            "lineage": item["lineage"],
            "record": record,
            "features": features,
            "referent_evidence": evidence,
        }
        rebuilt_label = builder._target_label(candidate, item.get("nearer_slot"))
        if rebuilt_label != label:
            raise CheckError(f"v5.2 canonical label drift: {sample_id}")
        request = public_interface.format_grounding_request(label, image)
        public_target_keys = {
            key
            for target in request.get("targets", [])
            if isinstance(target, dict)
            for key in target
        }
        if (
            public_target_keys != {"slot", "category", "referring_expression"}
            or "boxes" in request
            or "nearer_slot" in request
            or "audit_targets" in request
        ):
            raise CheckError(f"v5.2 public G request leaks private truth: {sample_id}")
        nearer[item.get("nearer_slot")] += 1

        if sample_id in new_ids:
            root_path = str(Path(str(item.get("v3_frame_root_path", ""))).resolve())
            if root_path not in bound_root_paths or scene_id in prior_scenes:
                raise CheckError(f"new v5.2 source is not from an allowed v3 root: {sample_id}")
            try:
                image.relative_to(Path(root_path))
            except ValueError as exc:
                raise CheckError(f"new v5.2 image escapes its v3 root: {sample_id}") from exc
        else:
            reuse = item.get("reuse_binding")
            if not isinstance(reuse, dict):
                raise CheckError(f"reused v5.2 row lacks lineage: {sample_id}")
            live_candidate = live_v51_reused.get(reuse.get("v51_sample_id"))
            if live_candidate is None or live_candidate["reuse_binding"] != reuse:
                raise CheckError(f"reused v5.1 decision lineage drift: {sample_id}")
            row = reused_rows[sample_id]
            if row.get("binding") != reuse:
                raise CheckError(f"reused decision row binding drift: {sample_id}")
            for side in ("reviewer_a", "reviewer_b"):
                document = row.get(side)
                if (
                    not isinstance(document, dict)
                    or document.get("decision_sha256")
                    != builder.canonical_hash(document.get("decision"))
                    or document.get("decision_sha256")
                    != reuse[f"{side}_decision_sha256"]
                ):
                    raise CheckError(f"reused decision hash drift: {side}/{sample_id}")

        if rehash_source_artifacts:
            errors = sensor.source_artifact_errors(
                record, source_file_cache, source_member_cache
            )
            if errors:
                source_errors[sample_id] = errors
            try:
                deep_evidence = builder._referent_evidence(
                    record,
                    file_hash_cache=source_file_cache,
                    member_cache=projection_cache,
                    verify_zip_hash=True,
                )
            except builder.PackError as exc:
                source_errors.setdefault(sample_id, []).append(str(exc))
            else:
                expected_deep = copy.deepcopy(evidence)
                expected_deep["instance_projection"]["zip_rehashed"] = True
                if deep_evidence != expected_deep:
                    source_errors.setdefault(sample_id, []).append(
                        "deep_referent_evidence_drift"
                    )

    if source_errors:
        first = next(iter(source_errors.items()))
        raise CheckError(f"v5.2 source artifact drift: {first[0]}={first[1]}")
    if (
        len(scenes) != builder.FORMAL_TARGET
        or len(images) != builder.FORMAL_TARGET
        or len(sources) != builder.FORMAL_TARGET
        or dict(sorted(nearer.items())) != builder.EXPECTED_AB_BALANCE
    ):
        raise CheckError("v5.2 uniqueness or A/B balance drift")

    return {
        "schema": "gsa-depth-v5.2-unique-referent-pack-check-v1",
        "status": PACK_VALID,
        "pack": str(pack),
        "manifest_sha256": builder.file_sha256(manifest_path),
        "manifest_payload_sha256": integrity["manifest_payload_sha256"],
        "records": builder.FORMAL_TARGET,
        "unique_scenes": len(scenes),
        "unique_images": len(images),
        "reused_v51": len(reused_ids),
        "new_pending_review": len(new_ids),
        "A_plus_B_decisions_required": 2 * builder.NEW_TARGET,
        "referent_gate": {
            "method": builder.REFERENT_METHOD,
            "exact_one_pairs": builder.FORMAL_TARGET,
            "nonunique_pairs": 0,
        },
        "source_artifacts_rehashed": rehash_source_artifacts,
        "source_artifact_errors": 0,
        "v3_outputs_rehashed": rehash_v3_outputs,
        "training_materialization_eligible": False,
    }


def compile_reviews(
    pack: Path,
    pack_report: dict[str, Any],
    adjudication_path: Path | None = None,
) -> dict[str, Any]:
    pack = pack.expanduser().resolve()
    manifest = _read_json(pack / "manifest.json")
    all_labels_path = _verify_relative_binding(
        pack, manifest["provenance"]["all_labels"], "all_labels"
    )
    labels = _key_rows(_read_jsonl(all_labels_path), "sample_id", "all labels")
    selected = _key_rows(manifest["selected"], "sample_id", "selected manifest")
    new_ids = {
        sample_id for sample_id, item in selected.items() if item.get("lineage") == "new_v52"
    }
    reused_ids = set(selected) - new_ids
    path_a = pack / "annotator_a/annotations/annotations.jsonl"
    path_b = pack / "annotator_b/annotations/annotations.jsonl"
    votes_a = _latest_votes(path_a)
    votes_b = _latest_votes(path_b)
    if not set(votes_a) <= new_ids or not set(votes_b) <= new_ids:
        raise CheckError("v5.2 annotation streams may contain only the 33 new cases")
    for sample_id, vote in votes_a.items():
        _validate_vote(vote, labels[sample_id], "A")
    for sample_id, vote in votes_b.items():
        _validate_vote(vote, labels[sample_id], "B")
    key_a = _single_reviewer_key(votes_a, "A")
    key_b = _single_reviewer_key(votes_b, "B")
    if key_a is not None and key_b is not None and key_a == key_b:
        raise CheckError("reviewer A and B keys must be independent")

    complete_both = set(votes_a) & set(votes_b)
    agreements = {
        sample_id
        for sample_id in complete_both
        if votes_a[sample_id]["correct"] == votes_b[sample_id]["correct"]
    }
    disagreements = complete_both - agreements
    votes_c: dict[str, dict[str, Any]] = {}
    if adjudication_path is not None:
        votes_c = _latest_votes(adjudication_path)
        if set(votes_c) != disagreements:
            raise CheckError(
                "reviewer C stream must cover exactly the current A/B disagreement set"
            )
        for sample_id, vote in votes_c.items():
            _validate_vote(vote, labels[sample_id], "C")
    key_c = _single_reviewer_key(votes_c, "C")
    if key_c is not None and key_c in {key_a, key_b}:
        raise CheckError("reviewer C key must differ from reviewer A and B")

    resolved: dict[str, bool] = {sample_id: True for sample_id in reused_ids}
    for sample_id in agreements:
        resolved[sample_id] = bool(votes_a[sample_id]["correct"])
    for sample_id in disagreements:
        if sample_id in votes_c:
            resolved[sample_id] = bool(votes_c[sample_id]["correct"])
    a_complete = len(votes_a)
    b_complete = len(votes_b)
    complete = a_complete == builder.NEW_TARGET and b_complete == builder.NEW_TARGET
    unresolved = set(selected) - set(resolved)
    raw_agreement = (
        round(len(agreements) / builder.NEW_TARGET, 6) if complete else None
    )
    if not complete:
        status = PENDING_REVIEW
    elif disagreements - set(votes_c):
        status = PENDING_ADJUDICATION
    elif len(resolved) != builder.FORMAL_TARGET or not all(resolved.values()):
        status = FORMAL_FAILED
    elif not pack_report.get("source_artifacts_rehashed"):
        status = FULL_REHASH_REQUIRED
    else:
        status = FORMAL_PASS

    report = copy.deepcopy(pack_report)
    report.update(
        {
            "schema": "gsa-depth-v5.2-unique-referent-formal-review-v1",
            "status": status,
            "formal_qualification_eligible": status == FORMAL_PASS,
            "review": {
                "reused_positive_per_side": builder.REUSED_TARGET,
                "expected_new_per_annotator": builder.NEW_TARGET,
                "a_complete": a_complete,
                "b_complete": b_complete,
                "both_complete": len(complete_both),
                "agreement_count": len(agreements),
                "disagreement_count": len(disagreements),
                "raw_agreement": raw_agreement,
                "c_complete": len(votes_c),
                "resolved_count": len(resolved),
                "resolved_correct_count": sum(int(value) for value in resolved.values()),
                "unresolved_count": len(unresolved),
                "unresolved_sample_ids": sorted(unresolved),
                "reviewer_keys": {"a": key_a, "b": key_b, "c": key_c},
                "required_human_decisions_A_plus_B": 2 * builder.NEW_TARGET,
                "required_reviewer_c_decisions": len(disagreements),
            },
            "annotation_streams": {
                "a": (
                    builder.file_binding(path_a) if path_a.is_file() else None
                ),
                "b": (
                    builder.file_binding(path_b) if path_b.is_file() else None
                ),
                "c": (
                    builder.file_binding(adjudication_path)
                    if adjudication_path is not None and adjudication_path.is_file()
                    else None
                ),
            },
        }
    )
    report["identity_sha256"] = builder.canonical_hash(report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=builder.DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--compile", action="store_true")
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--skip-source-artifact-rehash",
        action="store_true",
        help="structural handoff only; cannot produce FORMAL_PASS",
    )
    parser.add_argument(
        "--skip-v3-output-rehash",
        action="store_true",
        help="development-only manifest check for already bound v3 roots",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_pack(
            args.pack,
            rehash_source_artifacts=not args.skip_source_artifact_rehash,
            rehash_v3_outputs=not args.skip_v3_output_rehash,
        )
        if args.compile:
            report = compile_reviews(args.pack, report, args.adjudication)
        elif args.adjudication is not None:
            raise CheckError("--adjudication requires --compile")
        if args.out is not None:
            output = args.out.expanduser().resolve()
            if output.exists():
                raise CheckError(f"refusing to overwrite report: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (CheckError, builder.PackError, public_interface.InterfaceError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
