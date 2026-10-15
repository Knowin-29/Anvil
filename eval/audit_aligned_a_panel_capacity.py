#!/usr/bin/env python3
"""Audit capacity for a paired, aligned FirstFault A panel.

This is an inventory/check-only tool.  It never materializes benchmark rows and it
does not promote model-generated payloads to human ground truth.  By default the
report is printed to stdout; ``--out`` is the only write path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trains.flywheel.validators import validate  # noqa: E402


A_FORMS = ("affordance", "dof", "action_cot")
BASE_DEPENDENCIES = ("bbox", "detection", "depth")
FORM_DISPLAY = {
    "affordance": "affordance",
    "dof": "DoF",
    "action_cot": "action_plan",
}
DEPTH_INSTANCE_RE = re.compile(
    r"^(.*?)@\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]$"
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_no}")
        rows.append(row)
    return rows


def _grounding_payload(payload: Any) -> list[dict[str, Any]] | None:
    if not isinstance(payload, list):
        return None
    result: list[dict[str, Any]] = []
    for value in payload:
        if not isinstance(value, dict):
            return None
        box = value.get("bbox_2d") if "bbox_2d" in value else value.get("bbox")
        result.append({"object_class": value.get("label"), "bbox": box})
    return result


def context_matches_dependency(
    row: dict[str, Any], dependency_form: str, payload: Any
) -> bool:
    context = row.get("gsa_context")
    if not isinstance(context, dict):
        return False
    spatial = context.get("spatial")
    if not isinstance(spatial, dict):
        spatial = {}
    if dependency_form == "bbox":
        groundings = _grounding_payload(payload)
        return bool(groundings) and context.get("primary") == groundings[0]
    if dependency_form == "detection":
        return context.get("groundings") == _grounding_payload(payload)
    if dependency_form == "depth":
        return spatial.get("depth_order") == payload
    if dependency_form in {"affordance", "dof"}:
        return spatial.get(dependency_form) == payload
    raise KeyError(f"unsupported dependency form: {dependency_form}")


def _as_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def depth_referents_are_declared_one_to_one(row: dict[str, Any]) -> bool:
    """Check identity inside declared G/S state, not visible-instance uniqueness."""
    context = row.get("gsa_context") or {}
    groundings = context.get("groundings") or []
    payload = row.get("payload") or []
    if not isinstance(groundings, list) or not isinstance(payload, list) or not payload:
        return False
    for entry in payload:
        if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
            return False
        label = entry["label"]
        match = DEPTH_INSTANCE_RE.match(label)
        if match:
            object_class = match.group(1)
            box = tuple(float(match.group(index)) for index in range(2, 6))
            candidates = [
                grounding
                for grounding in groundings
                if isinstance(grounding, dict)
                and grounding.get("object_class") == object_class
                and _as_box(grounding.get("bbox")) == box
            ]
        else:
            candidates = [
                grounding
                for grounding in groundings
                if isinstance(grounding, dict)
                and grounding.get("object_class") == label
            ]
        if len(candidates) != 1:
            return False
    return True


def source_episode_key(source_path: str) -> str:
    """Conservative source episode key for the two v3 source families."""
    path = Path(source_path)
    for part in path.parts:
        if part.startswith("FloorPlan"):
            return part
    return str(path.parent)


def _load_labels(pack: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for path in sorted((pack / "annotator_a/labels").rglob("*.jsonl")):
        for row in read_jsonl(path):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in labels:
                raise ValueError(f"invalid or duplicate packed sample_id: {sample_id!r}")
            labels[sample_id] = row
    return labels


def _load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        snapshot = row.get("label_snapshot")
        sample_id = snapshot.get("sample_id") if isinstance(snapshot, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"annotation without label_snapshot.sample_id: {path}")
        if sample_id in annotations:
            raise ValueError(f"duplicate annotation for {sample_id}: {path}")
        if row.get("case_id") != sample_id:
            raise ValueError(f"annotation case identity mismatch for {sample_id}: {path}")
        if not isinstance(row.get("correct"), bool):
            raise ValueError(f"annotation lacks Boolean decision for {sample_id}: {path}")
        annotations[sample_id] = row
    return annotations


def _load_split_hashes(paths: Iterable[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            value = (row.get("gsa_meta") or {}).get("image_sha256")
            if isinstance(value, str) and value:
                hashes.add(value)
    return hashes


def _load_raw_source_clean(
    raw_dir: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    rows_by_image: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if not raw_dir.is_dir():
        return rows_by_image
    for path in sorted(raw_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            image_sha = row.get("image_sha256")
            if isinstance(image_sha, str) and image_sha:
                rows_by_image[image_sha][path.stem] = row
    return rows_by_image


def _count_raw_source_clean(
    rows_by_image: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not rows_by_image:
        return {"available": False}
    return {
        "available": True,
        "images_with_any_generated_record": len(rows_by_image),
        "images_with_all_eight_generated_forms": sum(
            len(forms) == 8 for forms in rows_by_image.values()
        ),
        "images_with_all_three_generated_A_forms": sum(
            all(form in forms for form in A_FORMS)
            for forms in rows_by_image.values()
        ),
        "images_with_all_three_non_audit_generated_A_forms": sum(
            all(
                form in forms and forms[form].get("_audit_only") is not True
                for form in A_FORMS
            )
            for forms in rows_by_image.values()
        ),
        "generated_records_by_form": dict(sorted(Counter(
            form for forms in rows_by_image.values() for form in forms
        ).items())),
    }


def audit_capacity(args: argparse.Namespace) -> dict[str, Any]:
    pack = args.pack.resolve()
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = _load_labels(pack)
    annotations_a_path = pack / "annotator_a/annotations/annotations.jsonl"
    annotations_b_path = pack / "annotator_b/annotations/annotations.jsonl"
    annotations_a = _load_annotations(annotations_a_path)
    annotations_b = _load_annotations(annotations_b_path)
    adjudication_path = args.adjudication.resolve()
    adjudication = {
        row["sample_id"]: row for row in read_jsonl(adjudication_path)
    }
    source_manifest_path = args.source_manifest.resolve()
    source_rows = {
        row["upload_sha256"]: row for row in read_jsonl(source_manifest_path)
    }

    split_sets = {
        "pilot_train": _load_split_hashes([args.pilot_train.resolve()]),
        "pilot_val": _load_split_hashes([args.pilot_val.resolve()]),
        "g_train": _load_split_hashes([args.g_train.resolve()]),
        "g_dev": _load_split_hashes([args.g_dev.resolve()]),
        "g_qualification": _load_split_hashes([args.g_qualification.resolve()]),
    }
    pilot_exposed = split_sets["pilot_train"] | split_sets["pilot_val"]

    integrity = Counter()
    identity = Counter()
    votes = {form: Counter() for form in A_FORMS}
    items: list[dict[str, Any]] = []
    legacy_by_form = Counter()
    strict_by_form = Counter()
    legacy_images: set[str] = set()
    strict_images: set[str] = set()
    triple_images: set[str] = set()
    primary_declared_unique = 0
    depth_declared_unique = 0
    all_a_same_primary = 0

    for item in manifest.get("selected", []):
        image_path = Path(item["image_path"])
        image_sha = item["image_sha256"]
        identity["image_hash_expected"] += 1
        if not image_path.is_file() or file_hash(image_path) != image_sha:
            integrity["image_hash_or_presence_failure"] += 1
        else:
            identity["image_hash_passed"] += 1
        source_row = source_rows.get(image_sha)
        identity["source_manifest_binding_expected"] += 1
        if source_row is None:
            integrity["source_manifest_identity_missing"] += 1
            source_episode = None
        else:
            identity["source_manifest_binding_passed"] += 1
            source_episode = source_episode_key(str(source_row.get("source_path", "")))

        states: dict[str, dict[str, Any]] = {}
        for form, sample_id in (item.get("sample_ids") or {}).items():
            row = labels.get(sample_id)
            if row is None:
                raise ValueError(f"manifest label missing from pack: {sample_id}")
            expected_label_hash = (item.get("label_hashes") or {}).get(form)
            identity["label_content_hash_expected"] += 1
            if content_hash(row) != expected_label_hash:
                integrity["label_hash_failure"] += 1
            else:
                identity["label_content_hash_passed"] += 1
            if row.get("task_form") != form or row.get("image_path") != item["image_path"]:
                integrity["packed_label_identity_failure"] += 1
            ann_a = annotations_a.get(sample_id)
            ann_b = annotations_b.get(sample_id)
            if ann_a is None or ann_b is None:
                raise ValueError(f"missing A/B vote for {sample_id}")
            identity["annotation_snapshot_expected"] += 2
            if canonical_json(ann_a.get("label_snapshot")) != canonical_json(row):
                integrity["annotator_a_snapshot_failure"] += 1
            else:
                identity["annotation_snapshot_passed"] += 1
            if canonical_json(ann_b.get("label_snapshot")) != canonical_json(row):
                integrity["annotator_b_snapshot_failure"] += 1
            else:
                identity["annotation_snapshot_passed"] += 1
            decision_a = ann_a["correct"]
            decision_b = ann_b["correct"]
            if decision_a == decision_b:
                decision = decision_a
                if sample_id in adjudication:
                    integrity["spurious_adjudication"] += 1
            else:
                adjudicated = adjudication.get(sample_id)
                decision = adjudicated.get("correct") if adjudicated else None
                if adjudicated is None:
                    if form in A_FORMS:
                        integrity["unresolved_A_disagreement"] += 1
                elif adjudicated.get("label_sha256") != expected_label_hash:
                    integrity["adjudication_label_hash_failure"] += 1
            valid, reason = validate(form, row.get("payload"), row.get("gsa_context"))
            states[form] = {
                "sample_id": sample_id,
                "row": row,
                "decision_a": decision_a,
                "decision_b": decision_b,
                "decision": decision,
                "strict_valid": valid,
                "strict_reason": reason,
                "audit_only": row.get("_audit_only") is True,
            }
            if form in A_FORMS:
                votes[form]["paired"] += 1
                votes[form]["agreed" if decision_a == decision_b else "disagreed"] += 1
                votes[form][f"final_{str(decision).lower()}"] += 1
                if row.get("_audit_only") is True:
                    votes[form]["audit_only"] += 1
                if valid:
                    votes[form]["strict_valid"] += 1

        base_ok = all(
            states[form]["decision"] is True and states[form]["strict_valid"]
            for form in BASE_DEPENDENCIES
        )
        for dependency_form in BASE_DEPENDENCIES:
            identity["base_self_context_expected"] += 1
            dependency_row = states[dependency_form]["row"]
            if context_matches_dependency(
                dependency_row, dependency_form, dependency_row.get("payload")
            ):
                identity["base_self_context_passed"] += 1
            else:
                integrity["base_self_context_failure"] += 1
        primary = (states["bbox"]["row"].get("gsa_context") or {}).get("primary")
        groundings = (
            states["detection"]["row"].get("gsa_context") or {}
        ).get("groundings") or []
        primary_matches = [
            grounding for grounding in groundings
            if isinstance(grounding, dict) and grounding == primary
        ]
        primary_class = primary.get("object_class") if isinstance(primary, dict) else None
        primary_class_matches = [
            grounding for grounding in groundings
            if isinstance(grounding, dict)
            and grounding.get("object_class") == primary_class
        ]
        declared_primary_unique = (
            len(primary_matches) == 1 and len(primary_class_matches) == 1
        )
        primary_declared_unique += int(declared_primary_unique)
        declared_depth_unique = depth_referents_are_declared_one_to_one(
            states["depth"]["row"]
        )
        depth_declared_unique += int(declared_depth_unique)
        a_primaries = [
            (states[form]["row"].get("gsa_context") or {}).get("primary")
            for form in A_FORMS
        ]
        same_primary = all(value == primary for value in a_primaries)
        all_a_same_primary += int(same_primary)

        form_rows: dict[str, Any] = {}
        released: set[str] = set()
        for form in A_FORMS:
            state = states[form]
            row = state["row"]
            base_context_results = []
            for dependency_form in BASE_DEPENDENCIES:
                identity["A_base_context_expected"] += 1
                matched = context_matches_dependency(
                    row, dependency_form, states[dependency_form]["row"].get("payload")
                )
                base_context_results.append(matched)
                if matched:
                    identity["A_base_context_passed"] += 1
                else:
                    integrity["A_base_context_failure"] += 1
            base_context_ok = all(base_context_results)
            legacy_eligible = bool(
                base_ok
                and state["decision"] is True
                and state["strict_valid"]
                and base_context_ok
            )
            if legacy_eligible:
                legacy_by_form[form] += 1
                legacy_images.add(image_sha)

            prerequisites = []
            if form in {"dof", "action_cot"}:
                prerequisites.append("affordance")
            if form == "action_cot":
                prerequisites.append("dof")
            upstream_a_results = []
            for dependency in prerequisites:
                identity["A_chain_context_expected"] += 1
                matched = context_matches_dependency(
                    row, dependency, states[dependency]["row"].get("payload")
                )
                if matched:
                    identity["A_chain_context_passed"] += 1
                else:
                    integrity["A_chain_context_failure"] += 1
                upstream_a_results.append(dependency in released and matched)
            upstream_a_ok = all(upstream_a_results)
            strict_release = bool(
                legacy_eligible
                and state["audit_only"] is False
                and upstream_a_ok
            )
            if strict_release:
                released.add(form)
                strict_by_form[form] += 1
                strict_images.add(image_sha)
            form_rows[FORM_DISPLAY[form]] = {
                "sample_id": state["sample_id"],
                "A_vote": state["decision_a"],
                "B_vote": state["decision_b"],
                "final_teacher_payload_correctness": state["decision"],
                "strict_payload_valid": state["strict_valid"],
                "strict_payload_reason": state["strict_reason"],
                "audit_only": state["audit_only"],
                "base_GS_dependency_candidate": legacy_eligible,
                "strict_sequential_release_candidate": strict_release,
                "formal_human_GT": False,
            }
        if released == set(A_FORMS):
            triple_images.add(image_sha)

        exposure = [name for name, hashes in split_sets.items() if image_sha in hashes]
        items.append({
            "image_path": str(image_path),
            "image_sha256": image_sha,
            "source_episode_key": source_episode,
            "primary": primary,
            "declared_primary_category_unique_in_model_groundings": (
                declared_primary_unique
            ),
            "declared_depth_entities_one_to_one_with_model_groundings": (
                declared_depth_unique
            ),
            "human_verified_visible_instance_uniqueness": None,
            "all_A_forms_share_primary_identity": same_primary,
            "exposure": exposure,
            "forms": form_rows,
        })

    if any(integrity.values()):
        integrity_verdict = "FAIL"
    else:
        integrity_verdict = "PASS"

    source_clean_manifest_path = args.source_clean_manifest.resolve()
    source_clean = json.loads(source_clean_manifest_path.read_text(encoding="utf-8"))
    source_clean_rows = source_clean.get("selected") or []
    raw_source_clean = _load_raw_source_clean(args.source_clean_raw.resolve())
    pilot_scene_keys = {
        source_episode_key(str(source_rows[image_sha].get("source_path", "")))
        for image_sha in pilot_exposed
        if image_sha in source_rows
    }
    all_registered_exposure = set().union(*split_sets.values())
    all_registered_scene_keys = {
        source_episode_key(str(source_rows[image_sha].get("source_path", "")))
        for image_sha in all_registered_exposure
        if image_sha in source_rows
    }
    source_clean_image_disjoint = 0
    source_clean_scene_disjoint = 0
    source_clean_fully_unexposed: list[dict[str, Any]] = []
    for row in source_clean_rows:
        image_sha = row.get("image_sha256")
        if image_sha not in pilot_exposed:
            source_clean_image_disjoint += 1
        source_row = source_rows.get(image_sha)
        if source_row is not None:
            episode_key = source_episode_key(str(source_row.get("source_path", "")))
            if episode_key not in pilot_scene_keys:
                source_clean_scene_disjoint += 1
            if episode_key not in all_registered_scene_keys:
                generated_forms = raw_source_clean.get(image_sha, {})
                source_clean_fully_unexposed.append({
                    "image_path": row.get("image_path"),
                    "image_sha256": image_sha,
                    "scene": row.get("scene"),
                    "source_episode_key": episode_key,
                    "generated_seed_forms": sorted(generated_forms),
                    "all_three_generated_A_forms": all(
                        form in generated_forms for form in A_FORMS
                    ),
                    "all_three_non_audit_generated_A_forms": all(
                        form in generated_forms
                        and generated_forms[form].get("_audit_only") is not True
                        for form in A_FORMS
                    ),
                })

    vote_rows: dict[str, Any] = {}
    total_a_disagreements = 0
    for form in A_FORMS:
        counter = votes[form]
        total_a_disagreements += counter["disagreed"]
        vote_rows[FORM_DISPLAY[form]] = dict(counter)

    formal_capacity = 0
    report = {
        "schema_version": 1,
        "protocol": "firstfault-aligned-a-capacity-audit-v1",
        "verdict": "FORMAL_PANEL_BLOCKED_NO_HUMAN_GT",
        "integrity_verdict": integrity_verdict,
        "integrity_failures": dict(sorted(integrity.items())),
        "identity_checks": dict(sorted(identity.items())),
        "inputs": {
            "v3_manifest": {
                "path": str(manifest_path), "sha256": file_hash(manifest_path)
            },
            "annotator_a": {
                "path": str(annotations_a_path), "sha256": file_hash(annotations_a_path)
            },
            "annotator_b": {
                "path": str(annotations_b_path), "sha256": file_hash(annotations_b_path)
            },
            "adjudication": {
                "path": str(adjudication_path), "sha256": file_hash(adjudication_path)
            },
            "source_manifest": {
                "path": str(source_manifest_path), "sha256": file_hash(source_manifest_path)
            },
            "source_clean_manifest": {
                "path": str(source_clean_manifest_path),
                "sha256": file_hash(source_clean_manifest_path),
            },
        },
        "provenance": {
            "payload_origin": "Qwen3.5-397B-A17B model generation",
            "payload_label_versions": sorted({
                str(row.get("label_version")) for row in labels.values()
            }),
            "human_evidence": (
                "A/B Boolean correctness votes plus C decisions for disagreements; "
                "reviewers did not independently author replacement payloads"
            ),
            "formal_GT_policy": (
                "model-generated payloads remain non-GT even when both reviewers "
                "mark them correct"
            ),
        },
        "v3_inventory": {
            "images": len(items),
            "A_units": len(items) * len(A_FORMS),
            "vote_evidence": vote_rows,
            "A_disagreements_adjudicated": total_a_disagreements,
            "declared_primary_category_unique_in_model_groundings": (
                primary_declared_unique
            ),
            "declared_depth_entities_one_to_one_with_model_groundings": (
                depth_declared_unique
            ),
            "all_A_forms_share_primary_identity": all_a_same_primary,
            "human_verified_visible_instance_unique": 0,
            "visible_instance_uniqueness_unverified": len(items),
            "human_GT_A_units": 0,
        },
        "capacity": {
            "legacy_22_definition": (
                "base bbox/detection/v3-depth final-true and strict, plus each A form "
                "independently final-true and strict; audit_only is not excluded and "
                "affordance->DoF->action dependencies are not required"
            ),
            "legacy_base_GS_candidate_images": len(legacy_images),
            "legacy_base_GS_candidates_by_form": {
                FORM_DISPLAY[key]: value for key, value in legacy_by_form.items()
            },
            "strict_sequential_release_definition": (
                "final-true, strict, non-audit payloads with exact embedded G/S state "
                "and affordance->DoF->action dependency closure"
            ),
            "strict_sequential_release_images": len(strict_images),
            "strict_sequential_release_by_form": {
                FORM_DISPLAY[key]: value for key, value in strict_by_form.items()
            },
            "strict_complete_A_triples": len(triple_images),
            "formal_human_GT_scenes": formal_capacity,
            "formal_human_GT_by_form": {
                "affordance": 0, "DoF": 0, "action_plan": 0
            },
            "paired_oracle_predicted_A_scenes": 0,
            "paired_oracle_predicted_corrupted_A_scenes": 0,
            "matching_v3_predicted_GS_context_artifacts": 0,
            "matching_v3_corrupted_context_artifacts": 0,
            "matching_v3_A_branch_output_artifacts": 0,
            "target_scenes": args.target_scenes,
            "formal_scene_shortfall": max(args.target_scenes - formal_capacity, 0),
        },
        "leakage": {
            "scene_key_policy": (
                "AI2-THOR FloorPlan episode directory or Bridge trajectory image parent"
            ),
            "v3_exact_image_exposure": {
                name: sum(item["image_sha256"] in hashes for item in items)
                for name, hashes in split_sets.items()
            },
            "legacy_22_exact_image_exposure": {
                name: len(legacy_images & hashes) for name, hashes in split_sets.items()
            },
            "strict_release_exact_image_exposure": {
                name: len(strict_images & hashes) for name, hashes in split_sets.items()
            },
        },
        "source_clean_candidate_inventory": {
            "images": len(source_clean_rows),
            "image_disjoint_from_pilot_train_val": source_clean_image_disjoint,
            "source_episode_disjoint_from_pilot_train_val": source_clean_scene_disjoint,
            "source_episode_disjoint_candidate_shortfall": max(
                args.target_scenes - source_clean_scene_disjoint, 0
            ),
            "source_episode_disjoint_from_all_registered_pilot_G_exposure": len(
                source_clean_fully_unexposed
            ),
            "fully_unexposed_candidate_shortfall": max(
                args.target_scenes - len(source_clean_fully_unexposed), 0
            ),
            "fully_unexposed_with_all_three_generated_A_forms": sum(
                item["all_three_generated_A_forms"]
                for item in source_clean_fully_unexposed
            ),
            "fully_unexposed_with_all_three_non_audit_generated_A_forms": sum(
                item["all_three_non_audit_generated_A_forms"]
                for item in source_clean_fully_unexposed
            ),
            "fully_unexposed_candidates": source_clean_fully_unexposed,
            "formal_human_GT_scenes": 0,
            "generated_seed_labels_are_human_GT": False,
            "raw_generation": _count_raw_source_clean(raw_source_clean),
        },
        "human_packet": {
            "selection": (
                "freeze 40 new source-episode-disjoint images before model generation; "
                "verify a non-answer-bearing unique referring expression for each target"
            ),
            "incremental_A_scored_units": args.target_scenes * len(A_FORMS),
            "incremental_A_or_B_records": args.target_scenes * len(A_FORMS) * 2,
            "units_by_form_per_reviewer": {
                "affordance": args.target_scenes,
                "DoF": args.target_scenes,
                "action_plan": args.target_scenes,
            },
            "unit_contracts": {
                "affordance": [
                    "target_id", "action_type", "part_label", "part_region",
                    "contact_point", "feasibility", "equivalent_valid_contacts",
                ],
                "DoF": [
                    "target_id", "grasp_point", "jaw_axis", "approach",
                    "wrist_angle", "coordinate_frame", "units", "score_tolerances",
                    "physical_or_expert_provenance",
                ],
                "action_plan": [
                    "target_id", "goal", "ordered_atomic_steps", "preconditions",
                    "failure_modes", "target_point", "valid_alternative_plan_set",
                    "rubric_or_execution_evidence",
                ],
            },
            "review_protocol": (
                "A and B independently author or correct structured payloads without "
                "seeing the teacher proposal or each other's answer; C sees both only "
                "after submission"
            ),
            "C_trigger": (
                "non-equivalent normalized payload hashes, semantic validity conflict, "
                "or metric-tolerance conflict"
            ),
            "C_adjudication_min": 0,
            "C_adjudication_max": args.target_scenes * len(A_FORMS),
            "C_operational_reserve_from_v3_binary_disagreement_rate": round(
                args.target_scenes * len(A_FORMS)
                * total_a_disagreements / (len(items) * len(A_FORMS))
            ),
            "existing_vote_hash_reuse": (
                "allowed only for diagnostics of the identical immutable teacher payload"
            ),
            "existing_vote_reuse_as_human_GT": False,
            "upstream_requirement": (
                "each selected scene also needs human-final oracle G and S payloads; "
                "the existing ScanNet S40 is not aligned to these v3 images"
            ),
        },
        "paired_panel_required_fields": [
            "panel_id and split",
            "source_id, source_episode_id, license, image_path, image_sha256",
            "target_id, human-verified referring expression, target category",
            "A form and human-authored GT payload/hash/metric version",
            "oracle G payload/hash and oracle S payload/hash with human provenance",
            "predicted G/S payloads, response hashes, checkpoint identity, prompt and decoding identity",
            "corruption operator/id/seed, parent-context hash, changed field, validity result",
            "A outputs under oracle/predicted/corrupted contexts with response hashes and native scores",
            "transition tuple, oracle-headroom recovery, corruption-direction check",
            "first-fault state in {G,S,A,recovered,unattributed} and attribution-rule version",
            "route label provenance, eligible teacher, teacher regret, dependency-advantage token bindings",
        ],
        "paired_intervention_schema": {
            "controlled_constants": [
                "image_sha256", "target_id", "A form", "A GT hash",
                "checkpoint identity", "prompt identity", "decoding identity",
            ],
            "required_context_variants": [
                "predicted_G + predicted_S_from_predicted_G",
                "oracle_G + predicted_S_recomputed_from_oracle_G",
                "oracle_G + oracle_S",
                "one-field-corrupted_G + downstream_S_recomputed",
                "oracle_G + one-field-corrupted_S",
            ],
            "required_transition_fields": [
                "parsed", "native_A_score", "A_correct", "context_contract_valid",
                "transition_type", "oracle_headroom_recovery",
                "corruption_effect_direction", "scene_bootstrap_group",
            ],
            "first_fault_rule": {
                "G": "repairing G and recomputing S first repairs A",
                "S": "G is oracle-correct and repairing S first repairs A",
                "A": "A remains wrong under oracle G and oracle S",
                "recovered": "an upstream state is wrong but A remains correct",
                "unattributed": "the nested interventions do not isolate one stage",
            },
        },
        "items": items,
        "no_formal_pack_materialized": True,
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=ROOT / "data/semantic_audit_v3")
    parser.add_argument(
        "--adjudication", type=Path,
        default=ROOT / "data/semantic_audit_v3/adjudication.jsonl",
    )
    parser.add_argument(
        "--source-manifest", type=Path,
        default=ROOT / "artifacts/upload_review_500_v2/manifest.jsonl",
    )
    parser.add_argument(
        "--source-clean-manifest", type=Path,
        default=ROOT / "artifacts/manifests/v3_generation_manifest.json",
    )
    parser.add_argument(
        "--source-clean-raw", type=Path,
        default=ROOT / "data/flywheel_v3/unseen_clean_audit_v34/raw",
    )
    parser.add_argument(
        "--pilot-train", type=Path,
        default=ROOT / "trains/gsa_bench_v2_pilot_clean/train.jsonl",
    )
    parser.add_argument(
        "--pilot-val", type=Path,
        default=ROOT / "trains/gsa_bench_v2_pilot_clean/val.jsonl",
    )
    parser.add_argument(
        "--g-train", type=Path,
        default=ROOT / "trains/gsa_bench_v3_g_family_scoped_v2/train_G_only.jsonl",
    )
    parser.add_argument(
        "--g-dev", type=Path,
        default=ROOT / "trains/gsa_bench_v3_g_family_scoped_v2/dev_G_only.jsonl",
    )
    parser.add_argument(
        "--g-qualification", type=Path,
        default=(
            ROOT
            / "trains/gsa_bench_v3_g_family_scoped_v2/qualification_G_only.jsonl"
        ),
    )
    parser.add_argument("--target-scenes", type=int, default=40)
    parser.add_argument(
        "--require-formal-scenes", type=int,
        help="return status 2 unless at least this many formal human-GT scenes exist",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.target_scenes <= 0:
        parser.error("--target-scenes must be positive")
    if args.require_formal_scenes is not None and args.require_formal_scenes < 0:
        parser.error("--require-formal-scenes must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_capacity(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if report["integrity_verdict"] != "PASS":
        return 1
    if (
        args.require_formal_scenes is not None
        and report["capacity"]["formal_human_GT_scenes"]
        < args.require_formal_scenes
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
