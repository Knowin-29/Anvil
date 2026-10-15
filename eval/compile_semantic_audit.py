#!/usr/bin/env python3
"""Compile independent semantic-audit streams into a scoped verdict.

Schema v1/v2 packs retain the historical reader. Schema v3 fails closed on
manifest, label, image, reviewer-identity, and negative-vote provenance drift.
Schema v5 additionally replays the held-out ScanNet sensor source binding; its
PASS means benchmark-component calibration only and never authorizes training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trains.flywheel.validators import validate_depth_v4_record  # noqa: E402
from trains.flywheel.build_depth_semantic_audit_pack import (  # noqa: E402
    generation_run_evidence_hash,
    is_sha256,
    pairwise_generation_report_errors,
)
from trains.flywheel.build_scannet_sensor_depth_audit_pack import (  # noqa: E402
    AUDIT_CONTRACT as V5_AUDIT_CONTRACT,
    DEPTH_SCOPE as V5_DEPTH_SCOPE,
    FORM as V5_FORM,
    FORMS as V5_FORMS,
    FULL_SELECTION_RULE as V5_FULL_SELECTION_RULE,
    LABEL_VERSION as V5_LABEL_VERSION,
    MIN_SOURCE_SCENES as V5_MIN_SOURCE_SCENES,
    PACK_PROTOCOL as V5_PACK_PROTOCOL,
    PACK_ROUND as V5_PACK_ROUND,
    PACK_SELECTION_RULE as V5_PACK_SELECTION_RULE,
    REQUIRED_IMAGES as V5_REQUIRED_IMAGES,
    SCHEMA_VERSION as V5_SCHEMA_VERSION,
    SOURCE_ARTIFACT_SCHEMA as V5_SOURCE_ARTIFACT_SCHEMA,
    SOURCE_AUDIT_ROUND as V5_SOURCE_AUDIT_ROUND,
    SOURCE_PROTOCOL as V5_SOURCE_PROTOCOL,
    SourceRow as V5SourceRow,
    difficulty_features as v5_difficulty_features,
    load_and_validate_source_bundle,
    manifest_payload_hash as v5_manifest_payload_hash,
    project_source_record as project_v5_source_record,
    select_source_rows as select_v5_source_rows,
    source_artifact_binding as v5_source_artifact_binding,
    source_inventory_summary as v5_source_inventory_summary,
    source_visibility_projection_hash as v5_visibility_projection_hash,
)
from trains.flywheel.validators import validate  # noqa: E402

DEFAULT_MIN_ACCURACY = 0.80
DEFAULT_MIN_AGREEMENT = 0.80
V3_FORMS = [
    "action_cot", "affordance", "bbox", "depth",
    "detection", "dof", "mask", "point",
]
V4_FORMS = ["depth"]
SECURE_SCHEMAS = {3, 4, 5}
V4_DEPTH_PROTOCOLS = {
    "depth-only-calibration": {
        "round": "v4-depth",
        "depth_scope": "total_order",
        "prompt_version": "gsa-depth-v4.1",
        "label_version": "gsa-label-v4-depth",
    },
    "depth-pairwise-calibration-v1": {
        "round": "v4-depth-pairwise",
        "depth_scope": "pairwise",
        "prompt_version": "gsa-depth-v4.2-pairwise",
        "label_version": "gsa-label-v4-depth-pairwise",
    },
    "depth-v4.3-pairwise-semantic-audit-v1": {
        "round": "v4.3-depth-pairwise",
        "depth_scope": "pairwise",
        "prompt_version": "gsa-depth-v4.3-pairwise",
        "label_version": "gsa-label-v4.3-depth-pairwise",
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_path(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def manifest_payload_hash(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("integrity", None)
    return content_hash(payload)


def has_error_code(row: dict[str, Any]) -> bool:
    value = row.get("error_code")
    return isinstance(value, str) and bool(value.strip())


def load_latest(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    latest: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    if not path.is_file():
        return latest, errors
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON ({exc.msg})")
            continue
        snapshot = row.get("label_snapshot") if isinstance(row, dict) else None
        sample_id = snapshot.get("sample_id") if isinstance(snapshot, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"{path}:{line_number}: missing label_snapshot.sample_id")
            continue
        latest[sample_id] = row
    return latest, errors


def load_adjudication(path: Path | None, schema_version: int) -> tuple[dict[str, bool], list[str]]:
    if path is None:
        return {}, []
    if not path.is_file():
        return {}, [f"adjudication file does not exist: {path}"]
    out: dict[str, bool] = {}
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON ({exc.msg})")
            continue
        sample_id = row.get("sample_id") if isinstance(row, dict) else None
        value = row.get("correct") if isinstance(row, dict) else None
        if not isinstance(sample_id, str) or not sample_id or not isinstance(value, bool):
            errors.append(f"{path}:{line_number}: sample_id and Boolean correct are required")
            continue
        if sample_id in out:
            errors.append(f"{path}:{line_number}: duplicate adjudication for {sample_id}")
            continue
        out[sample_id] = value
        if schema_version in SECURE_SCHEMAS and value is False and not has_error_code(row):
            errors.append(f"{path}:{line_number}: negative adjudication {sample_id} lacks error_code")
    return out, errors


def manifest_index(
    manifest: dict[str, Any], errors: list[str]
) -> tuple[dict[str, str], dict[str, tuple[str, dict[str, Any]]]]:
    selected = manifest.get("selected")
    forms = manifest.get("forms")
    images = manifest.get("images")
    if not isinstance(selected, list) or not isinstance(forms, list) or not isinstance(images, int):
        errors.append("manifest requires selected, forms, and integer images")
        return {}, {}
    if len(selected) != images:
        errors.append(f"manifest images={images} but selected has {len(selected)} entries")
    if len(set(forms)) != len(forms) or not all(isinstance(form, str) for form in forms):
        errors.append("manifest forms must be a unique list of strings")
    expected: dict[str, str] = {}
    sample_items: dict[str, tuple[str, dict[str, Any]]] = {}
    seen_images: set[str] = set()
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            errors.append(f"manifest selected[{index}] is not an object")
            continue
        image = item.get("image_path")
        sample_ids = item.get("sample_ids")
        if not isinstance(image, str) or not image or not isinstance(sample_ids, dict):
            errors.append(f"manifest selected[{index}] requires image_path and sample_ids")
            continue
        if image in seen_images:
            errors.append(f"duplicate manifest image_path: {image}")
        seen_images.add(image)
        for form, sample_id in sample_ids.items():
            if form not in forms or not isinstance(sample_id, str) or not sample_id:
                errors.append(f"invalid sample mapping for {image}: {form}")
                continue
            if sample_id in expected:
                errors.append(f"duplicate manifest sample_id: {sample_id}")
            expected[sample_id] = form
            sample_items[sample_id] = (image, item)
    return expected, sample_items


def validate_v3_manifest(manifest: dict[str, Any], errors: list[str]) -> None:
    if manifest.get("schema_version") != 3:
        return
    for field in ("round", "audit_id", "source_fingerprint"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            errors.append(f"v3 manifest missing valid {field}")
    if not isinstance(manifest.get("per_form_label_hashes"), dict):
        errors.append("v3 manifest missing valid per_form_label_hashes")
    if manifest.get("images") != 40:
        errors.append("v3 protocol requires exactly 40 images")
    if manifest.get("forms") != V3_FORMS:
        errors.append("v3 protocol requires the fixed eight-form order")
    fingerprint = manifest.get("source_fingerprint")
    if isinstance(fingerprint, str) and (
        len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        errors.append("v3 source_fingerprint is not a SHA-256 hex digest")


def validate_v4_manifest(manifest: dict[str, Any], errors: list[str]) -> None:
    if manifest.get("schema_version") != 4:
        return
    for field in ("round", "audit_id", "source_fingerprint"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            errors.append(f"v4 manifest missing valid {field}")
    protocol = manifest.get("protocol")
    protocol_contract = V4_DEPTH_PROTOCOLS.get(protocol)
    if protocol_contract is None:
        errors.append(
            "v4 manifest requires a registered total-order or pairwise depth protocol"
        )
    else:
        expected_round = protocol_contract["round"]
        expected_scope = protocol_contract["depth_scope"]
        if manifest.get("round") != expected_round:
            errors.append(f"v4 {protocol} protocol requires round={expected_round}")
        scope = manifest.get("depth_scope", "total_order")
        if scope != expected_scope:
            errors.append(
                f"v4 {protocol} protocol requires depth_scope={expected_scope}"
            )
    if manifest.get("images") != 40:
        errors.append("v4 depth protocol requires exactly 40 images")
    if manifest.get("forms") != V4_FORMS:
        errors.append("v4 depth protocol requires forms=[depth]")
    if manifest.get("records_per_annotator") != 40:
        errors.append("v4 depth protocol requires 40 records per annotator")
    fingerprint = manifest.get("source_fingerprint")
    if isinstance(fingerprint, str) and (
        len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        errors.append("v4 source_fingerprint is not a SHA-256 hex digest")
    sources = manifest.get("source_files")
    if not isinstance(sources, list) or not sources:
        errors.append("v4 manifest requires source_files provenance")
    else:
        for item in sources:
            path = item.get("path") if isinstance(item, dict) else None
            digest = item.get("sha256") if isinstance(item, dict) else None
            if not isinstance(path, str) or not isinstance(digest, str):
                errors.append("v4 source_files entry is invalid")
            elif not Path(path).is_file():
                errors.append(f"v4 source file missing: {path}")
            elif file_hash(Path(path)) != digest:
                errors.append(f"v4 source file hash drift: {path}")
        if (
            isinstance(fingerprint, str)
            and content_hash(sources) != fingerprint
        ):
            errors.append("v4 source_fingerprint drift")
    if manifest.get("depth_scope", "total_order") == "pairwise":
        generation_run = manifest.get("generation_run")
        if not isinstance(generation_run, dict):
            errors.append("v4 pairwise manifest requires generation_run provenance")
        else:
            for field in (
                "report_path",
                "report_sha256",
                "run_evidence_sha256",
                "source_labels_path",
                "source_labels_sha256",
                "accepted_pool_sha256",
                "protocol",
                "depth_scope",
                "prompt_version",
                "label_version",
                "stage",
                "prompt_hashes",
                "accepted_pool_images",
            ):
                if field not in generation_run:
                    errors.append(f"v4 pairwise generation_run missing {field}")
            for field in (
                "report_sha256",
                "run_evidence_sha256",
                "source_labels_sha256",
                "accepted_pool_sha256",
            ):
                if field in generation_run and not is_sha256(generation_run[field]):
                    errors.append(f"v4 pairwise generation_run {field} is invalid")
            if manifest.get("protocol") == "depth-v4.3-pairwise-semantic-audit-v1":
                for field in ("run_audit_sha256", "canary_binding_sha256"):
                    if not is_sha256(generation_run.get(field)):
                        errors.append(f"v4.3 generation_run {field} is invalid")
    exclusions = manifest.get("exclusion_manifests")
    if not isinstance(exclusions, list) or len(exclusions) < 2:
        errors.append("v4 manifest requires at least v2 and v3 exclusion provenance")
    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict):
        errors.append("v4 manifest requires integrity hashes")
    else:
        if content_hash(manifest.get("selected")) != integrity.get("selected_sha256"):
            errors.append("v4 selected integrity hash drift")
        if manifest_payload_hash(manifest) != integrity.get("manifest_payload_sha256"):
            errors.append("v4 manifest payload hash drift")


def validate_v5_manifest(manifest: dict[str, Any], errors: list[str]) -> None:
    if manifest.get("schema_version") != V5_SCHEMA_VERSION:
        return
    expected = {
        "protocol": V5_PACK_PROTOCOL,
        "round": V5_PACK_ROUND,
        "audit_contract": V5_AUDIT_CONTRACT,
        "task_interface": "oracle_grounded_pairwise_depth",
        "forms": V5_FORMS,
        "depth_scope": V5_DEPTH_SCOPE,
        "images": V5_REQUIRED_IMAGES,
        "records_per_annotator": V5_REQUIRED_IMAGES,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            errors.append(f"v5 manifest {field} drift")
    if manifest.get("benchmark_interface") != {
        "pair_identity_inputs": ["object_name", "bbox_2d_thousandth"],
        "target": "binary_near_far_order_for_the_supplied_pair",
        "generic_visible_object_order_prompt_allowed": False,
        "required_evaluation_tracks": [
            "S_oracle_grounded_pair",
            "G_to_S_predicted_grounding_cascade",
        ],
    }:
        errors.append("v5 oracle-grounded benchmark interface drift")
    for field in ("audit_id", "source_fingerprint"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"v5 manifest missing valid {field}")
    if not is_sha256(manifest.get("source_fingerprint")):
        errors.append("v5 source_fingerprint is not a SHA-256 digest")

    generation = manifest.get("source_generation")
    required_generation = {
        "report_path", "report_sha256", "source_labels_path",
        "source_labels_sha256", "source_records_sha256",
        "source_contract_sha256", "canary_binding_sha256", "protocol",
        "source_exclusions_sha256",
        "generation_parameters_sha256", "visual_quality_contract_sha256",
        "frozen_canary_scenes_sha256", "label_map_sha256",
        "label_version", "audit_round", "source_artifact_schema", "stage",
        "builder_sha256", "selection_rule", "generation_split", "accepted",
        "accepted_scenes",
    }
    if not isinstance(generation, dict) or not required_generation.issubset(generation):
        errors.append("v5 manifest source_generation binding is incomplete")
    else:
        for field in (
            "report_sha256", "source_labels_sha256", "source_records_sha256",
            "source_contract_sha256", "canary_binding_sha256", "builder_sha256",
            "source_exclusions_sha256", "generation_parameters_sha256",
            "visual_quality_contract_sha256", "frozen_canary_scenes_sha256",
            "label_map_sha256",
        ):
            if not is_sha256(generation.get(field)):
                errors.append(f"v5 source_generation {field} is invalid")
        identity = {
            "protocol": V5_SOURCE_PROTOCOL,
            "label_version": V5_LABEL_VERSION,
            "audit_round": V5_SOURCE_AUDIT_ROUND,
            "source_artifact_schema": V5_SOURCE_ARTIFACT_SCHEMA,
            "selection_rule": (
                V5_FULL_SELECTION_RULE
            ),
            "generation_split": "val",
            "stage": "full",
        }
        for field, value in identity.items():
            if generation.get(field) != value:
                errors.append(f"v5 source_generation {field} drift")
        if content_hash(generation) != manifest.get("source_fingerprint"):
            errors.append("v5 source_fingerprint drift")

    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        errors.append("v5 manifest lacks selection contract")
    else:
        if (
            type(selection.get("seed")) is not int
            or selection.get("rule") != V5_PACK_SELECTION_RULE
            or selection.get("unique_scene_required") is not True
            or selection.get("unique_image_required") is not True
            or selection.get("calibration_split") != "val"
            or selection.get("official_test_policy")
            != "reserved_hidden_test_not_selected"
            or selection.get("selected_by_official_split") != {"val": 40}
        ):
            errors.append("v5 official-val selection contract drift")
        available = selection.get("source_available_by_official_split")
        if (
            not isinstance(available, dict)
            or set(available) != {"val"}
            or not all(type(value) is int and value >= 0 for value in available.values())
            or available.get("val", 0) < V5_REQUIRED_IMAGES
        ):
            errors.append("v5 source split inventory is invalid")

    training = manifest.get("training_materialization")
    if training != {
        "eligible": False,
        "reason": "held_out_sensor_records_not_trainable",
        "allowed_use": "benchmark_component_calibration_only",
        "required_future_gate": (
            "independent_official_train_source_manifest_and_no_scene_overlap"
        ),
    }:
        errors.append("v5 held-out training-materialization policy drift")
    if not isinstance(manifest.get("source_contract"), dict):
        errors.append("v5 manifest lacks a source_contract snapshot")
    if not isinstance(manifest.get("source_quality_contract"), dict):
        errors.append("v5 manifest lacks source_quality_contract")
    if not isinstance(manifest.get("source_canary_binding"), dict):
        errors.append("v5 manifest lacks source_canary_binding")
    if not isinstance(manifest.get("frozen_canary_scenes"), dict):
        errors.append("v5 manifest lacks frozen_canary_scenes")
    inventory = manifest.get("source_inventory")
    expected_split_keys = {"train", "val", "test"}
    if not isinstance(inventory, dict):
        errors.append("v5 manifest lacks source_inventory")
    else:
        for field in (
            "openeqa_scenes_by_official_split",
            "ready_source_scenes_by_official_split",
            "generated_source_scenes_by_official_split",
            "generated_source_records_by_official_split",
        ):
            counts = inventory.get(field)
            if (
                not isinstance(counts, dict)
                or set(counts) != expected_split_keys
                or not all(type(value) is int and value >= 0 for value in counts.values())
            ):
                errors.append(f"v5 source_inventory {field} is invalid")
        if not is_sha256(inventory.get("source_inventory_sha256")):
            errors.append("v5 source_inventory hash is invalid")
        if inventory.get("official_test_used_for_calibration") is not False:
            errors.append("v5 official test must not enter calibration")
    if not isinstance(manifest.get("validated_source_exclusions"), dict):
        errors.append("v5 manifest lacks validated source exclusions")
    if not isinstance(manifest.get("per_form_label_hashes"), dict):
        errors.append("v5 manifest lacks per_form_label_hashes")

    selected = manifest.get("selected")
    selected_scene_ids: list[Any] = []
    if isinstance(selected, list):
        scene_ids = [item.get("scene_id") for item in selected if isinstance(item, dict)]
        selected_scene_ids = scene_ids
        image_hashes = [item.get("image_sha256") for item in selected if isinstance(item, dict)]
        scene_identity_valid = all(
            isinstance(value, str) and bool(value) for value in scene_ids
        )
        image_identity_valid = all(is_sha256(value) for value in image_hashes)
        if (
            len(scene_ids) != V5_REQUIRED_IMAGES
            or not scene_identity_valid
            or len(set(value for value in scene_ids if isinstance(value, str)))
            != V5_REQUIRED_IMAGES
            or any(item.get("official_split") != "val" for item in selected if isinstance(item, dict))
        ):
            errors.append("v5 selection must contain 40 unique official-val scenes")
        if (
            len(image_hashes) != V5_REQUIRED_IMAGES
            or not image_identity_valid
            or len(set(value for value in image_hashes if isinstance(value, str)))
            != V5_REQUIRED_IMAGES
        ):
            errors.append("v5 selection must contain 40 unique images")

    reserved = manifest.get("reserved_benchmark")
    if not isinstance(reserved, dict):
        errors.append("v5 manifest lacks reserved_benchmark binding")
    else:
        reserved_scene_ids = reserved.get("scene_ids")
        reserved_rows = reserved.get("source_rows")
        valid_reserved_ids = (
            isinstance(reserved_scene_ids, list)
            and all(isinstance(value, str) and value for value in reserved_scene_ids)
        )
        if (
            reserved.get("policy")
            != "source_scenes_disjoint_from_semantic_calibration"
            or reserved.get("minimum_scenes")
            != V5_MIN_SOURCE_SCENES - V5_REQUIRED_IMAGES
            or not valid_reserved_ids
            or len(reserved_scene_ids) < V5_MIN_SOURCE_SCENES - V5_REQUIRED_IMAGES
            or len(reserved_scene_ids) != len(set(reserved_scene_ids))
            or set(reserved_scene_ids) & {
                value for value in selected_scene_ids if isinstance(value, str)
            }
            or not isinstance(reserved_rows, list)
            or reserved.get("exposed_to_audit_ui") is not False
            or reserved.get("training_materialization_eligible") is not False
        ):
            errors.append("v5 reserved benchmark scope is invalid")
        if content_hash(reserved_scene_ids) != reserved.get("scene_ids_sha256"):
            errors.append("v5 reserved benchmark scene hash drift")
        if content_hash(reserved_rows) != reserved.get("source_rows_sha256"):
            errors.append("v5 reserved benchmark source-row hash drift")

    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict):
        errors.append("v5 manifest lacks integrity hashes")
    else:
        if content_hash(selected) != integrity.get("selected_sha256"):
            errors.append("v5 selected integrity hash drift")
        artifact_hashes = [
            item.get("sensor_artifact_binding_sha256")
            for item in selected or []
            if isinstance(item, dict)
        ]
        if content_hash(artifact_hashes) != integrity.get("selected_sensor_artifacts_sha256"):
            errors.append("v5 sensor artifact integrity hash drift")
        if content_hash(reserved) != integrity.get("reserved_benchmark_sha256"):
            errors.append("v5 reserved benchmark integrity hash drift")
        if v5_manifest_payload_hash(manifest) != integrity.get("manifest_payload_sha256"):
            errors.append("v5 manifest payload hash drift")


def load_generation_source_rows(
    path: Path,
    errors: list[str],
) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    if not path.is_file():
        errors.append(f"v4 pairwise generation source labels missing: {path}")
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(
                f"{path}:{line_number}: invalid generation label JSON ({exc.msg})"
            )
            continue
        if not isinstance(row, dict):
            errors.append(f"{path}:{line_number}: generation label is not an object")
            continue
        rows.append((line_number, row))
    return rows


def validate_v4_generation_run(
    manifest: dict[str, Any],
    errors: list[str],
) -> dict[str, tuple[dict[str, Any], str, int, str]]:
    """Revalidate the full generation gate and index immutable source labels."""
    if (
        manifest.get("schema_version") != 4
        or manifest.get("depth_scope", "total_order") != "pairwise"
    ):
        return {}
    run = manifest.get("generation_run")
    if not isinstance(run, dict):
        return {}
    report_value = run.get("report_path")
    source_value = run.get("source_labels_path")
    if not isinstance(report_value, str) or not isinstance(source_value, str):
        errors.append("v4 pairwise generation_run requires report/source label paths")
        return {}
    report_path = Path(report_value).resolve()
    source_path = Path(source_value).resolve()
    if source_path != (report_path.parent / "labels/S_spatial/depth.jsonl").resolve():
        errors.append("v4 pairwise generation source path is not report-bound")
    if not report_path.is_file():
        errors.append(f"v4 pairwise generation report missing: {report_path}")
        return {}
    if file_hash(report_path) != run.get("report_sha256"):
        errors.append("v4 pairwise generation report content hash drift")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"v4 pairwise generation report invalid: {report_path} ({exc})")
        return {}
    if not isinstance(report, dict):
        errors.append("v4 pairwise generation report is not an object")
        return {}

    rows_with_lines = load_generation_source_rows(source_path, errors)
    source_records = [row for _, row in rows_with_lines]
    if source_path.is_file() and file_hash(source_path) != run.get("source_labels_sha256"):
        errors.append("v4 pairwise generation source label hash drift")
    source_files = manifest.get("source_files")
    expected_source_files = [{
        "path": str(source_path),
        "sha256": run.get("source_labels_sha256"),
    }]
    if source_files != expected_source_files:
        errors.append("v4 pairwise source_files do not bind the generation labels")

    if generation_run_evidence_hash(report) != run.get("run_evidence_sha256"):
        errors.append("v4 pairwise generation run evidence hash drift")
    report_snapshot = {
        "accepted_pool_sha256": report.get("accepted_pool_sha256"),
        "accepted_pool_images": report.get("accepted_pool_images"),
        "protocol": report.get("protocol"),
        "depth_scope": report.get("depth_scope"),
        "prompt_version": report.get("prompt_version"),
        "label_version": report.get("label_version"),
        "prompt_hashes": report.get("prompt_hashes"),
        "stage": report.get("stage"),
    }
    for field, value in report_snapshot.items():
        if run.get(field) != value:
            errors.append(f"v4 pairwise generation_run {field} snapshot drift")
    if report.get("run_audit") is not None and (
        run.get("run_audit_sha256") != content_hash(report["run_audit"])
    ):
        errors.append("v4 pairwise generation run_audit snapshot drift")
    provenance = report.get("source_provenance")
    canary_binding = (
        provenance.get("canary_binding") if isinstance(provenance, dict) else None
    )
    if canary_binding is not None and (
        run.get("canary_binding_sha256") != content_hash(canary_binding)
    ):
        errors.append("v4 pairwise generation canary_binding snapshot drift")

    protocol = manifest.get("protocol")
    contract = V4_DEPTH_PROTOCOLS.get(protocol)
    if contract is None:
        errors.append("v4 pairwise audit protocol has no generation contract")
    else:
        protocol_spec = {
            "prompt_version": contract["prompt_version"],
            "label_version": contract["label_version"],
            "depth_scope": contract["depth_scope"],
        }
        report_errors = pairwise_generation_report_errors(
            report,
            report_path,
            [source_path],
            source_records,
            protocol_spec,
        )
        errors.extend(
            f"v4 pairwise generation binding: {error}" for error in report_errors
        )

    indexed: dict[str, tuple[dict[str, Any], str, int, str]] = {}
    for line_number, record in rows_with_lines:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"{source_path}:{line_number}: generation label lacks sample_id")
            continue
        if sample_id in indexed:
            errors.append(f"v4 pairwise generation source duplicates {sample_id}")
            continue
        valid, reason = validate_depth_v4_record(record)
        if not valid:
            errors.append(
                f"v4 pairwise generation source {sample_id} is invalid ({reason})"
            )
        if contract is not None and (
            record.get("prompt_version") != contract["prompt_version"]
            or record.get("label_version") != contract["label_version"]
            or record.get("depth_scope") != "pairwise"
        ):
            errors.append(f"v4 pairwise generation source version drift for {sample_id}")
        indexed[sample_id] = (
            record,
            str(source_path),
            line_number,
            content_hash(record),
        )
    return indexed


def validate_v5_source_generation(
    manifest: dict[str, Any],
    errors: list[str],
) -> dict[str, V5SourceRow]:
    """Recompute the complete held-out sensor source and selection binding."""
    if manifest.get("schema_version") != V5_SCHEMA_VERSION:
        return {}
    generation = manifest.get("source_generation")
    if not isinstance(generation, dict):
        return {}
    report_value = generation.get("report_path")
    labels_value = generation.get("source_labels_path")
    if not isinstance(report_value, str) or not isinstance(labels_value, str):
        errors.append("v5 source_generation requires report/source label paths")
        return {}
    report_path = Path(report_value).resolve()
    labels_path = Path(labels_value).resolve()
    if labels_path != (report_path.parent / "labels/S_spatial/depth.jsonl").resolve():
        errors.append("v5 source labels are not report-bound")
    bundle, source_errors = load_and_validate_source_bundle(report_path, labels_path)
    errors.extend(f"v5 sensor source binding: {error}" for error in source_errors)
    if bundle is None:
        return {}

    report = bundle.report
    if report.get("accepted_scenes", 0) < V5_MIN_SOURCE_SCENES:
        errors.append(
            f"v5 source has fewer than {V5_MIN_SOURCE_SCENES} unique scenes"
        )
    observed = {
        "report_sha256": file_hash(report_path),
        "source_labels_sha256": file_hash(labels_path),
        "source_records_sha256": content_hash([row.record for row in bundle.rows]),
        "source_contract_sha256": content_hash(report.get("source_contract")),
        "source_exclusions_sha256": content_hash(
            report.get("validated_source_exclusions")
        ),
        "canary_binding_sha256": content_hash(report.get("canary_binding")),
        "generation_parameters_sha256": content_hash(
            report.get("generation_parameters")
        ),
        "visual_quality_contract_sha256": content_hash(
            report.get("visual_quality_contract")
        ),
        "frozen_canary_scenes_sha256": content_hash(
            report.get("frozen_canary_scenes")
        ),
        "label_map_sha256": (report.get("source_contract") or {}).get(
            "label_map_sha256"
        ),
        "protocol": report.get("protocol"),
        "label_version": report.get("label_version"),
        "audit_round": report.get("audit_round"),
        "source_artifact_schema": report.get("source_artifact_schema"),
        "builder_sha256": report.get("builder_sha256"),
        "selection_rule": report.get("selection_rule"),
        "generation_split": (report.get("source_contract") or {}).get(
            "generation_split"
        ),
        "stage": report.get("stage"),
        "accepted": report.get("accepted"),
        "accepted_scenes": report.get("accepted_scenes"),
    }
    for field, value in observed.items():
        if generation.get(field) != value:
            errors.append(f"v5 source_generation {field} snapshot drift")
    if manifest.get("source_contract") != report.get("source_contract"):
        errors.append("v5 source_contract snapshot drift")
    if manifest.get("source_quality_contract") != report.get("visual_quality_contract"):
        errors.append("v5 source quality contract snapshot drift")
    if manifest.get("source_canary_binding") != report.get("canary_binding"):
        errors.append("v5 source canary binding snapshot drift")
    if manifest.get("frozen_canary_scenes") != report.get("frozen_canary_scenes"):
        errors.append("v5 frozen canary scene snapshot drift")
    if manifest.get("source_inventory") != v5_source_inventory_summary(report):
        errors.append("v5 source_inventory snapshot drift")
    if (
        manifest.get("validated_source_exclusions")
        != report.get("validated_source_exclusions")
    ):
        errors.append("v5 source exclusion snapshot drift")
    split_counts = dict(sorted(Counter(
        row.record.get("official_split") for row in bundle.rows
    ).items()))
    selection = manifest.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    if selection.get("source_available_by_official_split") != split_counts:
        errors.append("v5 source split inventory drift")
    seed = selection.get("seed")
    expected_selected: list[V5SourceRow] = []
    if type(seed) is int:
        try:
            expected_selected = select_v5_source_rows(bundle.rows, seed)
        except ValueError as exc:
            errors.append(f"v5 selection cannot be recomputed: {exc}")
    actual_source_ids = [
        (item.get("source_sample_ids") or {}).get(V5_FORM)
        for item in manifest.get("selected") or []
        if isinstance(item, dict)
    ]
    expected_source_ids = [row.record.get("sample_id") for row in expected_selected]
    if actual_source_ids != expected_source_ids:
        errors.append("v5 selected source identities/order drift")
    if expected_selected:
        features = [v5_difficulty_features(row.record) for row in expected_selected]
        expected_selection_stats = {
            "selected_by_official_split": {"val": V5_REQUIRED_IMAGES},
            "difficulty_counts": dict(sorted(Counter(
                feature["gap_bucket"] + "|" + feature["geometry_bucket"] + "|"
                + str(feature["quartile_separated"])
                for feature in features
            ).items())),
            "near_category_counts": dict(sorted(Counter(
                feature["near_category"] for feature in features
            ).items())),
            "far_category_counts": dict(sorted(Counter(
                feature["far_category"] for feature in features
            ).items())),
        }
        for field, value in expected_selection_stats.items():
            if selection.get(field) != value:
                errors.append(f"v5 selection {field} drift")
    selected_scene_ids = {
        row.record.get("scene_id") for row in expected_selected
    }
    reserved_rows = [
        {
            "sample_id": row.record.get("sample_id"),
            "scene_id": row.record.get("scene_id"),
            "official_split": row.record.get("official_split"),
            "source_jsonl_line": row.line_number,
            "source_record_sha256": row.sha256,
            "image_sha256": row.record.get("image_sha256"),
        }
        for row in bundle.rows
        if row.record.get("scene_id") not in selected_scene_ids
    ]
    reserved_scene_ids = sorted({row["scene_id"] for row in reserved_rows})
    expected_reserved = {
        "policy": "source_scenes_disjoint_from_semantic_calibration",
        "minimum_scenes": V5_MIN_SOURCE_SCENES - V5_REQUIRED_IMAGES,
        "scene_ids": reserved_scene_ids,
        "scene_ids_sha256": content_hash(reserved_scene_ids),
        "source_rows": reserved_rows,
        "source_rows_sha256": content_hash(reserved_rows),
        "by_official_split": dict(sorted(Counter(
            row["official_split"] for row in reserved_rows
        ).items())),
        "exposed_to_audit_ui": False,
        "training_materialization_eligible": False,
    }
    if manifest.get("reserved_benchmark") != expected_reserved:
        errors.append("v5 reserved benchmark binding drift")

    indexed: dict[str, V5SourceRow] = {}
    for row in bundle.rows:
        sample_id = row.record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            continue
        if sample_id in indexed:
            errors.append(f"v5 source labels duplicate {sample_id}")
            continue
        indexed[sample_id] = row
    return indexed


def load_packed_labels(pack: Path, root_name: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    label_root = pack / root_name / "labels"
    paths = sorted(label_root.rglob("*.jsonl")) if label_root.is_dir() else []
    if not paths:
        errors.append(f"{root_name}: no packed labels found")
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid label JSON ({exc.msg})")
                continue
            sample_id = record.get("sample_id") if isinstance(record, dict) else None
            if not isinstance(sample_id, str) or not sample_id:
                errors.append(f"{path}:{line_number}: packed label lacks sample_id")
                continue
            if sample_id in records:
                errors.append(f"{root_name}: duplicate packed label {sample_id}")
            records[sample_id] = record
    return records


def validate_pairwise_label_provenance(
    root_name: str,
    record: dict[str, Any],
    item: dict[str, Any],
    manifest: dict[str, Any],
    generation_sources: dict[str, tuple[dict[str, Any], str, int, str]],
    errors: list[str],
) -> None:
    sample_id = record.get("sample_id")
    source_sample_id = record.get("source_sample_id")
    manifest_source_id = item.get("source_sample_ids", {}).get("depth")
    if source_sample_id != manifest_source_id:
        errors.append(f"{root_name}: source sample mapping drift for {sample_id}")
    if record.get("audit_schema_version") != 4:
        errors.append(f"{root_name}: audit schema provenance drift for {sample_id}")
    if record.get("audit_round") != manifest.get("round"):
        errors.append(f"{root_name}: audit round provenance drift for {sample_id}")
    source = (
        generation_sources.get(source_sample_id)
        if isinstance(source_sample_id, str)
        else None
    )
    if source is None:
        errors.append(
            f"{root_name}: packed label {sample_id} is absent from the accepted generation pool"
        )
        return
    source_record, source_path, source_line, source_hash = source
    if record.get("source_label_sha256") != source_hash:
        errors.append(f"{root_name}: source label hash drift for {sample_id}")
    if record.get("source_jsonl_path") != source_path:
        errors.append(f"{root_name}: source label path drift for {sample_id}")
    if record.get("source_jsonl_line") != source_line:
        errors.append(f"{root_name}: source label line drift for {sample_id}")

    reconstructed = copy.deepcopy(record)
    reconstructed["sample_id"] = source_sample_id
    for field in (
        "audit_schema_version",
        "audit_round",
        "source_sample_id",
        "source_label_sha256",
        "source_jsonl_path",
        "source_jsonl_line",
    ):
        reconstructed.pop(field, None)
    if reconstructed != source_record:
        errors.append(
            f"{root_name}: packed label payload is not derived from source {source_sample_id}"
        )


def validate_v3_labels(
    pack: Path,
    manifest: dict[str, Any],
    expected: dict[str, str],
    sample_items: dict[str, tuple[str, dict[str, Any]]],
    generation_sources: dict[str, tuple[dict[str, Any], str, int, str]],
    v5_sources: dict[str, V5SourceRow],
    errors: list[str],
) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version not in SECURE_SCHEMAS:
        return
    forms = (
        V3_FORMS
        if schema_version == 3
        else V4_FORMS
        if schema_version == 4
        else V5_FORMS
    )
    reference_records: dict[str, dict[str, Any]] = {}
    for root_name in ("annotator_a", "annotator_b"):
        records = load_packed_labels(pack, root_name, errors)
        if root_name == "annotator_a":
            reference_records = records
        for sample_id, form in expected.items():
            record = records.get(sample_id)
            if record is None:
                errors.append(f"{root_name}: missing packed label {sample_id}")
                continue
            image, item = sample_items[sample_id]
            expected_hash = item.get("label_hashes", {}).get(form)
            if not isinstance(expected_hash, str) or content_hash(record) != expected_hash:
                errors.append(f"{root_name}: packed label hash drift for {sample_id}")
            if record.get("task_form") != form or record.get("image_path") != image:
                errors.append(f"{root_name}: packed label identity drift for {sample_id}")
            if not all(isinstance(record.get(field), str) and record[field].strip()
                       for field in ("label_version", "prompt_version")):
                errors.append(f"{root_name}: label version metadata missing for {sample_id}")
            if schema_version == 4:
                valid, reason = validate_depth_v4_record(record)
                if not valid:
                    errors.append(f"{root_name}: invalid v4 depth label {sample_id} ({reason})")
                record_scope = record.get("depth_scope") or "total_order"
                if record_scope != manifest.get("depth_scope", "total_order"):
                    errors.append(
                        f"{root_name}: v4 depth scope drift for {sample_id}"
                    )
                contract = V4_DEPTH_PROTOCOLS.get(manifest.get("protocol"))
                if contract is not None and (
                    record.get("prompt_version") != contract["prompt_version"]
                    or record.get("label_version") != contract["label_version"]
                ):
                    errors.append(
                        f"{root_name}: v4 protocol version drift for {sample_id}"
                    )
                if manifest.get("depth_scope", "total_order") == "pairwise":
                    validate_pairwise_label_provenance(
                        root_name,
                        record,
                        item,
                        manifest,
                        generation_sources,
                        errors,
                    )
            elif schema_version == V5_SCHEMA_VERSION:
                source_sample_id = record.get("source_sample_id")
                source = (
                    v5_sources.get(source_sample_id)
                    if isinstance(source_sample_id, str)
                    else None
                )
                manifest_source_id = (item.get("source_sample_ids") or {}).get(V5_FORM)
                if source_sample_id != manifest_source_id or source is None:
                    errors.append(f"{root_name}: v5 source identity drift for {sample_id}")
                else:
                    generation = manifest.get("source_generation") or {}
                    selection = manifest.get("selection") or {}
                    try:
                        projected = project_v5_source_record(
                            source,
                            Path(generation["source_labels_path"]),
                            generation["report_sha256"],
                            selection["seed"],
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        errors.append(
                            f"{root_name}: cannot recompute v5 projection for "
                            f"{sample_id} ({exc})"
                        )
                    else:
                        if record != projected:
                            errors.append(
                                f"{root_name}: v5 packed label is not the exact "
                                f"source projection for {sample_id}"
                            )
                    binding = v5_source_artifact_binding(source.record)
                    difficulty = v5_difficulty_features(source.record)
                    expected_item = {
                        "source_record_sha256": source.sha256,
                        "source_jsonl_line": source.line_number,
                        "sensor_provenance_sha256": binding[
                            "sensor_provenance_sha256"
                        ],
                        "sensor_artifact_binding_sha256": content_hash(binding),
                        "source_visibility_projection_sha256": (
                            v5_visibility_projection_hash(source.record)
                        ),
                        "difficulty_features": difficulty,
                        "difficulty_features_sha256": content_hash(difficulty),
                    }
                    for field, value in expected_item.items():
                        if item.get(field) != value:
                            errors.append(
                                f"{root_name}: v5 manifest {field} drift for "
                                f"{sample_id}"
                            )
                valid, reason = validate(
                    V5_FORM, record.get("payload"), record.get("gsa_context")
                )
                if not valid:
                    errors.append(
                        f"{root_name}: invalid v5 depth projection {sample_id} "
                        f"({reason})"
                    )
                if (
                    record.get("audit_schema_version") != V5_SCHEMA_VERSION
                    or record.get("audit_round") != V5_PACK_ROUND
                    or record.get("audit_contract") != V5_AUDIT_CONTRACT
                    or record.get("depth_scope") != V5_DEPTH_SCOPE
                    or record.get("protocol_version") != V5_SOURCE_PROTOCOL
                    or record.get("prompt_version") != V5_SOURCE_PROTOCOL
                    or record.get("label_version") != V5_LABEL_VERSION
                    or record.get("official_split") != "val"
                    or record.get("_audit_only") is not True
                    or (record.get("training_materialization") or {}).get("eligible")
                    is not False
                ):
                    errors.append(f"{root_name}: v5 audit/held-out contract drift for {sample_id}")
                if any(
                    field in record
                    for field in ("sensor_measurements", "verified_pair", "spatial")
                ):
                    errors.append(
                        f"{root_name}: v5 metric truth escaped the provenance "
                        f"envelope for {sample_id}"
                    )
        for sample_id in records:
            if sample_id not in expected:
                errors.append(f"{root_name}: unexpected packed label {sample_id}")

    selected = manifest.get("selected", [])
    selected_depth_cues: Counter[str] = Counter()
    for item in selected:
        image = item.get("image_path")
        digest = item.get("image_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"v{schema_version} manifest missing image_sha256 for {image}")
        elif not Path(image).is_file():
            errors.append(f"image file missing for {image}")
        elif file_hash(Path(image)) != digest:
            errors.append(f"image content drift for {image}")
        hashes = item.get("label_hashes")
        versions = item.get("label_versions")
        prompts = item.get("prompt_versions")
        for field_name, values in (("label_hashes", hashes), ("label_versions", versions), ("prompt_versions", prompts)):
            if not isinstance(values, dict) or set(values) != set(forms):
                errors.append(f"v{schema_version} manifest {field_name} incomplete for {image}")
        if schema_version == 4:
            depth_scopes = item.get("depth_scopes")
            if (
                not isinstance(depth_scopes, dict)
                or depth_scopes != {"depth": manifest.get("depth_scope", "total_order")}
            ):
                errors.append(f"v4 manifest depth_scopes drift for {image}")
            sample_id = item.get("sample_ids", {}).get("depth")
            if manifest.get("depth_scope", "total_order") == "pairwise":
                depth_cues = item.get("depth_cues")
                record = reference_records.get(sample_id) if isinstance(sample_id, str) else None
                evidence = (
                    ((record or {}).get("gsa_context") or {}).get("spatial") or {}
                ).get("adjacent_depth_evidence")
                actual_cue = (
                    evidence[0].get("cue_type")
                    if isinstance(evidence, list)
                    and len(evidence) == 1
                    and isinstance(evidence[0], dict)
                    else None
                )
                if depth_cues != {"depth": actual_cue}:
                    errors.append(f"v4 manifest depth_cues drift for {image}")
                if isinstance(actual_cue, str):
                    selected_depth_cues[actual_cue] += 1
            source_sample_id = item.get("source_sample_ids", {}).get("depth")
            if (
                not isinstance(sample_id, str)
                or not sample_id.startswith("v4_depth_")
                or not isinstance(source_sample_id, str)
                or source_sample_id == sample_id
            ):
                errors.append(f"v4 sample/source identity invalid for {image}")
        elif schema_version == V5_SCHEMA_VERSION:
            depth_scopes = item.get("depth_scopes")
            sample_id = item.get("sample_ids", {}).get(V5_FORM)
            source_sample_id = item.get("source_sample_ids", {}).get(V5_FORM)
            if depth_scopes != {V5_FORM: V5_DEPTH_SCOPE}:
                errors.append(f"v5 manifest depth_scopes drift for {image}")
            if (
                not isinstance(sample_id, str)
                or not sample_id.startswith("v5_scannet_sensor_depth_")
                or not isinstance(source_sample_id, str)
                or source_sample_id == sample_id
                or item.get("official_split") != "val"
            ):
                errors.append(f"v5 sample/source identity invalid for {image}")

    if schema_version == 4 and manifest.get("depth_scope", "total_order") == "pairwise":
        if dict(sorted(selected_depth_cues.items())) != manifest.get("cue_counts"):
            errors.append("v4 pairwise cue_counts drift")
        for cue in ("occlusion", "contact_layering"):
            if selected_depth_cues.get(cue, 0) < 8:
                errors.append(f"v4 pairwise audit requires at least 8 {cue} cases")
        candidate_counts = manifest.get("candidate_cue_counts")
        if (
            not isinstance(candidate_counts, dict)
            or not set(selected_depth_cues).issubset(candidate_counts)
            or any(
                not isinstance(value, int) or value < selected_depth_cues.get(cue, 0)
                for cue, value in candidate_counts.items()
            )
        ):
            errors.append("v4 pairwise candidate_cue_counts are invalid")

    per_form = manifest.get("per_form_label_hashes", {})
    for form in forms:
        pairs = []
        for item in selected:
            sample_id = item.get("sample_ids", {}).get(form)
            label_hash = item.get("label_hashes", {}).get(form)
            if sample_id is not None and label_hash is not None:
                pairs.append({"sample_id": sample_id, "label_sha256": label_hash})
        if content_hash(pairs) != per_form.get(form):
            errors.append(f"v{schema_version} per-form label hash drift for {form}")


def validate_v4_exclusions(manifest: dict[str, Any], errors: list[str]) -> None:
    if manifest.get("schema_version") != 4:
        return
    prior_paths: set[str] = set()
    prior_hashes: set[str] = set()
    prior_sample_ids: set[str] = set()
    schemas: set[int] = set()
    for provenance in manifest.get("exclusion_manifests", []):
        if not isinstance(provenance, dict):
            errors.append("v4 exclusion provenance entry is invalid")
            continue
        path = provenance.get("path")
        expected_hash = provenance.get("manifest_sha256")
        if not isinstance(path, str) or not isinstance(expected_hash, str):
            errors.append("v4 exclusion provenance lacks path/hash")
            continue
        manifest_path = Path(path)
        if not manifest_path.is_file():
            errors.append(f"v4 exclusion manifest missing: {path}")
            continue
        if file_hash(manifest_path) != expected_hash:
            errors.append(f"v4 exclusion manifest hash drift: {path}")
            continue
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"v4 exclusion manifest invalid: {path} ({exc})")
            continue
        schema = int(prior.get("schema_version", 1))
        schemas.add(schema)
        selected = prior.get("selected")
        if not isinstance(selected, list):
            errors.append(f"v4 exclusion manifest lacks selected list: {path}")
            continue
        for item in selected:
            image = item.get("image_path") if isinstance(item, dict) else None
            if not isinstance(image, str) or not image:
                errors.append(f"v4 exclusion manifest has invalid image: {path}")
                continue
            prior_paths.add(normalized_path(image))
            digest = item.get("image_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                image_path = Path(image)
                if not image_path.is_file():
                    errors.append(f"cannot verify excluded image: {image}")
                    continue
                digest = file_hash(image_path)
            prior_hashes.add(digest)
            ids = item.get("sample_ids", {})
            if isinstance(ids, dict):
                prior_sample_ids.update(
                    value for value in ids.values() if isinstance(value, str)
                )
    if not {1, 3}.issubset(schemas):
        errors.append("v4 exclusions must include the v2(schema 1) and v3(schema 3) packs")

    for item in manifest.get("selected", []):
        image = item.get("image_path")
        digest = item.get("image_sha256")
        sample_id = item.get("sample_ids", {}).get("depth")
        source_sample_id = item.get("source_sample_ids", {}).get("depth")
        if normalized_path(str(image)) in prior_paths:
            errors.append(f"v4 image path overlaps prior audit: {image}")
        if digest in prior_hashes:
            errors.append(f"v4 image hash overlaps prior audit: {image}")
        if sample_id in prior_sample_ids or source_sample_id in prior_sample_ids:
            errors.append(f"v4 sample identity overlaps prior audit: {image}")


def validate_annotations(
    label: str,
    latest: dict[str, dict[str, Any]],
    expected: dict[str, str],
    sample_items: dict[str, tuple[str, dict[str, Any]]],
    schema_version: int,
    errors: list[str],
) -> set[str]:
    keys: set[str] = set()
    for sample_id, row in latest.items():
        if sample_id not in expected:
            errors.append(f"{label}: unexpected annotation {sample_id}")
            continue
        snapshot = row.get("label_snapshot")
        if not isinstance(snapshot, dict):
            errors.append(f"{label}: {sample_id} has no label_snapshot")
            continue
        form = expected[sample_id]
        if schema_version in SECURE_SCHEMAS:
            image, item = sample_items[sample_id]
            if snapshot.get("task_form") != form or snapshot.get("image_path") != image:
                errors.append(f"{label}: label snapshot identity drift for {sample_id}")
            if content_hash(snapshot) != item.get("label_hashes", {}).get(form):
                errors.append(f"{label}: label snapshot hash drift for {sample_id}")
            if not all(isinstance(snapshot.get(field), str) and snapshot[field].strip()
                       for field in ("label_version", "prompt_version")):
                errors.append(f"{label}: version metadata missing for {sample_id}")
        value = row.get("correct")
        if schema_version in SECURE_SCHEMAS and not isinstance(value, bool):
            errors.append(f"{label}: {sample_id} has non-Boolean correct")
        elif schema_version in SECURE_SCHEMAS and value is False and not has_error_code(row):
            errors.append(f"{label}: negative vote {sample_id} lacks error_code")
        key = row.get("annotator_key")
        if isinstance(key, str) and key.strip():
            keys.add(key)
        elif schema_version in SECURE_SCHEMAS:
            errors.append(f"{label}: {sample_id} lacks annotator_key")
    return keys


def validate_v5_adjudication_identity(
    path: Path | None,
    adjudication: dict[str, bool],
    expected: dict[str, str],
    sample_items: dict[str, tuple[str, dict[str, Any]]],
    manifest: dict[str, Any],
    annotator_keys: set[str],
    errors: list[str],
) -> None:
    if manifest.get("schema_version") != V5_SCHEMA_VERSION or not adjudication:
        return
    if path is None or not path.is_file():
        errors.append("v5 reviewer-C adjudication source is missing")
        return
    reviewer_keys: set[str] = set()
    adjudication_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        sample_id = row.get("sample_id")
        reviewer_key = row.get("reviewer_key")
        if not isinstance(reviewer_key, str) or not reviewer_key.strip():
            errors.append(f"{path}:{line_number}: v5 adjudication lacks reviewer_key")
        else:
            reviewer_keys.add(reviewer_key)
        adjudication_id = row.get("adjudication_id")
        if not isinstance(adjudication_id, str) or not adjudication_id.strip():
            errors.append(f"{path}:{line_number}: v5 adjudication lacks adjudication_id")
        else:
            adjudication_ids.add(adjudication_id)
        if row.get("source_audit_id") != manifest.get("audit_id"):
            errors.append(f"{path}:{line_number}: v5 adjudication audit identity drift")
        item = sample_items.get(str(sample_id), (None, {}))[1]
        form = expected.get(str(sample_id))
        label_hash = (item.get("label_hashes") or {}).get(form)
        if row.get("label_sha256") != label_hash:
            errors.append(f"{path}:{line_number}: v5 adjudication label hash drift")
    if len(reviewer_keys) != 1:
        errors.append("v5 adjudication must use exactly one reviewer-C key")
    if reviewer_keys & annotator_keys:
        errors.append("v5 reviewer C must be independent from annotators A/B")
    if len(adjudication_ids) != 1:
        errors.append("v5 adjudication must bind exactly one adjudication_id")


def compile_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    errors: list[str] = []
    manifest_path = args.pack / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest is not an object")
        schema_version = int(manifest.get("schema_version", 1))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"schema_version": 1, "verdict": "BLOCKED", "errors": [f"invalid manifest: {exc}"]}, 2

    validate_v3_manifest(manifest, errors)
    validate_v4_manifest(manifest, errors)
    validate_v5_manifest(manifest, errors)
    generation_sources = validate_v4_generation_run(manifest, errors)
    v5_sources = validate_v5_source_generation(manifest, errors)
    if schema_version in {4, 5} and (
        args.min_accuracy < DEFAULT_MIN_ACCURACY
        or args.min_agreement < DEFAULT_MIN_AGREEMENT
    ):
        errors.append(
            f"v{schema_version} thresholds cannot be lower than "
            "accuracy=0.80 and agreement=0.80"
        )
    expected, sample_items = manifest_index(manifest, errors)
    validate_v3_labels(
        args.pack,
        manifest,
        expected,
        sample_items,
        generation_sources,
        v5_sources,
        errors,
    )
    validate_v4_exclusions(manifest, errors)
    a, a_errors = load_latest(args.pack / "annotator_a/annotations/annotations.jsonl")
    b, b_errors = load_latest(args.pack / "annotator_b/annotations/annotations.jsonl")
    errors.extend(a_errors + b_errors)
    keys_a = validate_annotations("annotator_a", a, expected, sample_items, schema_version, errors)
    keys_b = validate_annotations("annotator_b", b, expected, sample_items, schema_version, errors)
    if schema_version in SECURE_SCHEMAS:
        if len(keys_a) != 1 or len(keys_b) != 1:
            errors.append(
                f"each v{schema_version} stream must use exactly one non-empty annotator_key"
            )
        if keys_a & keys_b:
            errors.append(
                f"v{schema_version} annotator_a and annotator_b must use different annotator_key values"
            )

    adjudication, adjudication_errors = load_adjudication(args.adjudication, schema_version)
    errors.extend(adjudication_errors)
    validate_v5_adjudication_identity(
        args.adjudication,
        adjudication,
        expected,
        sample_items,
        manifest,
        keys_a | keys_b,
        errors,
    )
    by_form: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unresolved: list[str] = []
    resolved_values: dict[str, bool] = {}
    for sample_id, form in expected.items():
        va = a.get(sample_id, {}).get("correct")
        vb = b.get(sample_id, {}).get("correct")
        if isinstance(va, bool):
            by_form[form]["annotator_a"] += 1
        if isinstance(vb, bool):
            by_form[form]["annotator_b"] += 1
        if not (isinstance(va, bool) and isinstance(vb, bool)):
            continue
        by_form[form]["paired"] += 1
        if va == vb:
            by_form[form]["agreed"] += 1
            resolved_values[sample_id] = va
        elif sample_id in adjudication:
            by_form[form]["adjudicated"] += 1
            resolved_values[sample_id] = adjudication[sample_id]
        else:
            by_form[form]["unresolved"] += 1
            unresolved.append(sample_id)
    if schema_version in SECURE_SCHEMAS:
        for sample_id in adjudication:
            if sample_id not in expected:
                errors.append(f"adjudication contains unexpected sample_id {sample_id}")
            elif sample_id in a and sample_id in b and a[sample_id].get("correct") == b[sample_id].get("correct"):
                errors.append(f"adjudication supplied for already-agreed sample {sample_id}")

    rows: dict[str, Any] = {}
    pass_forms = not errors
    images = int(manifest.get("images", 0))
    for form in manifest.get("forms", []):
        counts = by_form[form]
        paired = counts["paired"]
        agreed = counts["agreed"]
        resolved = [value for sample_id, value in resolved_values.items() if expected.get(sample_id) == form]
        accuracy = sum(resolved) / len(resolved) if resolved else None
        agreement = agreed / paired if paired else None
        eligible = (
            paired == images
            and counts["unresolved"] == 0
            and agreement is not None and agreement >= args.min_agreement
            and accuracy is not None and accuracy >= args.min_accuracy
            and not errors
        )
        pass_forms = pass_forms and eligible
        rows[form] = {
            "expected": images,
            "annotator_a": counts["annotator_a"],
            "annotator_b": counts["annotator_b"],
            "paired": paired,
            "agreed": agreed,
            "adjudicated": counts["adjudicated"],
            "unresolved": counts["unresolved"],
            "raw_agreement": agreement,
            "resolved_accuracy": accuracy,
            "eligible": eligible,
        }
    verdict = "PASS" if pass_forms else "BLOCKED"
    report = {
        "schema_version": schema_version,
        "protocol": manifest.get("protocol"),
        "depth_scope": (
            manifest.get("depth_scope") if schema_version in {4, 5} else None
        ),
        "qualification_scope": (
            "held_out_benchmark_component_calibration"
            if schema_version == V5_SCHEMA_VERSION
            else "pilot_qualification"
        ),
        "training_materialization": (
            copy.deepcopy(manifest.get("training_materialization"))
            if schema_version == V5_SCHEMA_VERSION
            else None
        ),
        "verdict": verdict,
        "thresholds": {"min_accuracy": args.min_accuracy, "min_agreement": args.min_agreement},
        "manifest": str(manifest_path),
        "by_form": rows,
        "unresolved_sample_ids": unresolved,
        "errors": errors,
        "annotator_keys": {"a": sorted(keys_a), "b": sorted(keys_b)},
        "notes": (
            "Schema v5 PASS means held-out benchmark-component calibration only. "
            "Its official ScanNet val/test sensor records are audit-only and cannot "
            "authorize joint, S-family, or MOPD training; an independent official-train "
            "source manifest plus a no-scene-overlap gate is required."
            if schema_version == V5_SCHEMA_VERSION
            else
            "PASS authorizes pilot qualification only; it does not convert teacher "
            "labels into benchmark GT. Schema v4 covers depth only and must be combined "
            "with the resolved v3 non-depth forms by the composite gate."
        ),
    }
    return report, 0 if verdict == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=Path("data/semantic_audit_v2"))
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/audits/pilot_semantic_audit.json"))
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY)
    parser.add_argument("--min-agreement", type=float, default=DEFAULT_MIN_AGREEMENT)
    args = parser.parse_args()
    report, return_code = compile_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
