#!/usr/bin/env python3
"""Build and score the held-out ScanNet RGB-depth v5.1 formal40 release.

This module is intentionally separate from the frozen v5.0 evaluator.  It binds
the completed RGB-auditable human audit, derives exactly forty private source
records, emits blinded G and S-oracle requests, and scores the paired G -> S
experiment only after the DLC runner verifies the complete inference bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_PACK = ROOT / "data/semantic_audit_v51_scannet_rgb_auditable_depth_formal"
DEFAULT_AUDIT_REPORT = ROOT / "artifacts/audits/scannet_rgb_depth_v51_formal.json"
DEFAULT_RELEASE = ROOT / "artifacts/depth_v51_eval_interface/formal40"

EVALUATION_PROTOCOL = "gsa-depth-v5.1-formal40-paired-gs-eval-v1"
RELEASE_PROTOCOL = "gsa-depth-v5.1-formal40-request-release-v1"
RESULT_PROTOCOL = "gsa-depth-v5.1-formal40-paired-gs-result-v1"
SOURCE_PROTOCOL = "gsa-depth-v5.0-scannet-sensor-pairwise"
SOURCE_LABEL_VERSION = "gsa-label-v5.0-depth-scannet-sensor-pairwise"
TRACK_G = "G_named_pair_bbox"
TRACK_S_ORACLE = "S_oracle_grounded_pair"
TRACK_S_PRED = "G_to_S_predicted_grounding"
EXPECTED_RECORDS = 40
AB_PERMUTATION_SALT = "ab-balance-v51-formal40-27"
EXPECTED_LABEL_BALANCE = {"A": 20, "B": 20}
BOOTSTRAP_SEED = 51040
BOOTSTRAP_SAMPLES = 10_000

EXPECTED_MANIFEST_SHA256 = (
    "6855e6285f00b307d2a230b4460af1d81da518fca77b52a428045103e2639624"
)
EXPECTED_MANIFEST_PAYLOAD_SHA256 = (
    "87ceeb9dfb753a0e74455c93687d586c7e414b3f31e3aef81a97bf08a2d42161"
)
EXPECTED_SELECTED_SHA256 = (
    "c08d4e76d01cc4ad002cb22596d5c5c6cb0f4b0a26c7ef47f9642c1d198274a5"
)
EXPECTED_AUDIT_REPORT_SHA256 = (
    "2b0c5c3c437d7db6d331f9b65e2d4fd005eb9d745b1a3cfd749926d69cee4198"
)
EXPECTED_ANNOTATION_A_SHA256 = (
    "3b343da78ab83bd8806d5fba3bb847071c6e04ed9ace9509ac26ae32a7b4dfca"
)
EXPECTED_ANNOTATION_B_SHA256 = (
    "60ea7bb61bd42f7a190f6d3761385e57d84d15e1e7811b5bc28a444c43e86ed0"
)
EXPECTED_LABELS_SHA256 = (
    "f6c65aae16aa1bdc843272d7cd9954bf1edcfd0e58710c51dbdc36d320ce2123"
)
EXPECTED_SOURCE_WRAPPERS_SHA256 = (
    "db6b1705f85aebef1964de73f3c7ccbbf4874b1c263fc55545c133d9dafb7921"
)
EXPECTED_SOURCE_RECORDS_CANONICAL_SHA256 = (
    "c143424ecbd4431d2743192e93e1cb1ec0ce38cb50a2637bacebcd1cbb7474e9"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVAL_ID = re.compile(r"^gsa-sdepth-v51-[0-9a-f]{20}$")
_INSTANCE_SUFFIX = re.compile(r"\s*\[instance\s+\d+\]\s*$", re.IGNORECASE)
_SCENE_ID = re.compile(r"scene\d{4}_\d{2}", re.IGNORECASE)
_FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "depth_m",
    "farther_choice",
    "instance_id",
    "label",
    "median_depth_m",
    "nearer_choice",
    "payload",
    "rank",
    "relative_depth",
    "scene_id",
    "sensor_measurements",
    "sensor_provenance",
    "source_id",
    "source_sample_id",
    "target",
}
_ALLOWED_ERROR_CODES = {
    "ambiguous_depth",
    "bbox_misaligned",
    "identity_ambiguous",
    "occlusion_or_truncation",
    "other",
    "wrong_relation_or_order",
}


class ContractError(ValueError):
    """Raised when a release or result cannot be trusted."""


@dataclass(frozen=True)
class Entity:
    name: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PairExample:
    eval_id: str
    source_identity: str
    scene_id: str
    source_image: Path
    image_sha256: str
    entity_a: Entity
    entity_b: Entity
    nearer_choice: str


@dataclass(frozen=True)
class GroundingParse:
    bbox_a: tuple[int, int, int, int] | None
    bbox_b: tuple[int, int, int, int] | None
    status: str

    @property
    def valid(self) -> bool:
        return self.bbox_a is not None and self.bbox_b is not None


@dataclass(frozen=True)
class ChoiceParse:
    choice: str | None
    status: str


@dataclass(frozen=True)
class FormalSource:
    pairs: tuple[PairExample, ...]
    binding: dict[str, Any]


@dataclass(frozen=True)
class RequestRelease:
    manifest: dict[str, Any]
    source: FormalSource
    requests: dict[str, list[dict[str, Any]]]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ContractError(f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ContractError(f"missing file: {path}")
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": file_sha256(path)}


def _future_binding(actual: Path, final: Path) -> dict[str, Any]:
    stat = actual.stat()
    return {
        "path": str(final.expanduser().resolve()),
        "bytes": stat.st_size,
        "sha256": file_sha256(actual),
    }


def verify_file_binding(binding: Any, *, deep: bool = True) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
        raise ContractError("file binding schema drift")
    path = Path(str(binding.get("path"))).expanduser().resolve()
    if str(path) != binding.get("path") or not path.is_file():
        raise ContractError(f"bound file is missing or non-canonical: {path}")
    if path.stat().st_size != binding.get("bytes"):
        raise ContractError(f"bound file size drift: {path}")
    if not isinstance(binding.get("sha256"), str) or not _SHA256.fullmatch(binding["sha256"]):
        raise ContractError(f"bound file hash is invalid: {path}")
    if deep and file_sha256(path) != binding["sha256"]:
        raise ContractError(f"bound file hash drift: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ContractError(f"missing JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON must be an object: {path}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ContractError(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    if not rows and not allow_empty:
        raise ContractError(f"JSONL has no records: {path}")
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


def _assert_hash(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ContractError(f"{label} hash drift: expected {expected}, got {actual}")


def _bbox(value: Any, context: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ContractError(f"{context}: bbox must contain four integer coordinates")
    box = tuple(value)
    if any(item < 0 or item > 1000 for item in box):
        raise ContractError(f"{context}: bbox is outside 0-1000")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ContractError(f"{context}: bbox has non-positive area")
    return box  # type: ignore[return-value]


def _public_name(measurement: dict[str, Any], internal_label: str) -> str:
    for key in ("category", "aggregation_label"):
        value = measurement.get(key)
        if isinstance(value, str) and value.strip():
            name = _INSTANCE_SUFFIX.sub("", value).strip()
            if name:
                return name
    name = _INSTANCE_SUFFIX.sub("", internal_label).strip()
    if not name:
        raise ContractError("entity has no public class name")
    return name


def _resolve_source_image(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError("source record lacks image_path")
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ContractError(f"source image is missing or escapes the repository: {path}")
    return path


def _iter_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _iter_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_keys(item)


def _validate_source_record(record: dict[str, Any], *, verify_image: bool) -> None:
    identity = record.get("sample_id")
    prefix = f"source record {identity or '<missing>'}"
    if not isinstance(identity, str) or not identity:
        raise ContractError("source record lacks sample_id")
    if record.get("protocol_version") != SOURCE_PROTOCOL:
        raise ContractError(f"{prefix}: protocol drift")
    if record.get("label_version") != SOURCE_LABEL_VERSION:
        raise ContractError(f"{prefix}: label version drift")
    if record.get("official_split") != "val":
        raise ContractError(f"{prefix}: only official ScanNet val is accepted")
    training = record.get("training_materialization")
    if not isinstance(training, dict) or training.get("eligible") is not False:
        raise ContractError(f"{prefix}: held-out training prohibition is absent")
    interface = record.get("model_interface")
    if (
        not isinstance(interface, dict)
        or interface.get("intended_task") != "oracle_grounded_pairwise_depth"
        or interface.get("requires_explicit_pair_groundings") is not True
        or interface.get("instance_ids_are_internal_identity_only") is not True
    ):
        raise ContractError(f"{prefix}: model interface drift")
    if record.get("task_interface") not in (None, "oracle_grounded_pairwise_depth"):
        raise ContractError(f"{prefix}: task interface drift")
    scene_id = record.get("scene_id")
    if not isinstance(scene_id, str) or not _SCENE_ID.fullmatch(scene_id):
        raise ContractError(f"{prefix}: invalid scene identity")

    payload = record.get("payload")
    if not isinstance(payload, list) or len(payload) != 2 or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ContractError(f"{prefix}: payload must contain two entities")
    ordered = sorted(payload, key=lambda item: item.get("rank", 999))
    if [item.get("rank") for item in ordered] != [1, 2]:
        raise ContractError(f"{prefix}: ranks must be [1, 2]")
    if [item.get("relative_depth") for item in ordered] != [0.0, 1.0]:
        raise ContractError(f"{prefix}: relative depths must be [0.0, 1.0]")
    labels = [item.get("label") for item in ordered]
    if not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != 2:
        raise ContractError(f"{prefix}: internal labels are invalid")

    measurements = record.get("sensor_measurements")
    if not isinstance(measurements, list) or len(measurements) != 2:
        raise ContractError(f"{prefix}: sensor measurements must contain two entities")
    by_label: dict[str, dict[str, Any]] = {}
    for item in measurements:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise ContractError(f"{prefix}: invalid sensor measurement")
        label = item["label"]
        if label in by_label:
            raise ContractError(f"{prefix}: duplicate sensor measurement")
        _bbox(item.get("bbox_2d"), f"{prefix}/{label}")
        by_label[label] = item
    if set(by_label) != set(labels):
        raise ContractError(f"{prefix}: payload/measurement identity mismatch")

    context = record.get("gsa_context")
    groundings = context.get("groundings") if isinstance(context, dict) else None
    if not isinstance(groundings, list) or len(groundings) != 2:
        raise ContractError(f"{prefix}: private source groundings are absent")
    grounding_map = {
        item.get("object_class"): item.get("bbox")
        for item in groundings
        if isinstance(item, dict)
    }
    for label, measurement in by_label.items():
        if grounding_map.get(label) != measurement.get("bbox_2d"):
            raise ContractError(f"{prefix}: context/measurement bbox drift")

    image_hash = record.get("image_sha256")
    if not isinstance(image_hash, str) or not _SHA256.fullmatch(image_hash):
        raise ContractError(f"{prefix}: invalid image hash")
    image = _resolve_source_image(record.get("image_path"))
    if verify_image and file_sha256(image) != image_hash:
        raise ContractError(f"{prefix}: image hash drift")


def _latest_annotations(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        snapshot = row.get("label_snapshot")
        sample_id = snapshot.get("sample_id") if isinstance(snapshot, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ContractError(f"annotation lacks label_snapshot.sample_id: {path}")
        latest[sample_id] = row
    return latest


def _validate_vote(row: dict[str, Any], label: dict[str, Any], side: str) -> None:
    sample_id = label["sample_id"]
    if row.get("case_id") != sample_id:
        raise ContractError(f"reviewer {side} case identity drift: {sample_id}")
    if canonical_hash(row.get("label_snapshot")) != canonical_hash(label):
        raise ContractError(f"reviewer {side} label snapshot drift: {sample_id}")
    correct = row.get("correct")
    if not isinstance(correct, bool):
        raise ContractError(f"reviewer {side} has non-Boolean vote: {sample_id}")
    error_code = row.get("error_code")
    comment = row.get("comment")
    if correct and error_code not in (None, ""):
        raise ContractError(f"reviewer {side} positive vote has error_code: {sample_id}")
    if not correct and (
        error_code not in _ALLOWED_ERROR_CODES
        or not isinstance(comment, str)
        or not comment.strip()
    ):
        raise ContractError(f"reviewer {side} negative vote lacks concrete evidence: {sample_id}")


def _pair_from_record(record: dict[str, Any], *, verify_image: bool) -> PairExample:
    _validate_source_record(record, verify_image=verify_image)
    identity = record["sample_id"]
    payload = sorted(record["payload"], key=lambda item: item["rank"])
    measurements = {item["label"]: item for item in record["sensor_measurements"]}
    near_label, far_label = payload[0]["label"], payload[1]["label"]
    near = Entity(
        _public_name(measurements[near_label], near_label),
        _bbox(measurements[near_label]["bbox_2d"], f"{identity}/near"),
    )
    far = Entity(
        _public_name(measurements[far_label], far_label),
        _bbox(measurements[far_label]["bbox_2d"], f"{identity}/far"),
    )
    if near.name.casefold() == far.name.casefold():
        raise ContractError(f"{identity}: formal40 requires two distinct public names")
    bit = hashlib.sha256(
        f"{EVALUATION_PROTOCOL}\0{AB_PERMUTATION_SALT}\0{identity}".encode("utf-8")
    ).digest()[0] & 1
    if bit:
        entity_a, entity_b, nearer_choice = far, near, "B"
    else:
        entity_a, entity_b, nearer_choice = near, far, "A"
    image_hash = record["image_sha256"]
    eval_id = "gsa-sdepth-v51-" + hashlib.sha256(
        f"{EVALUATION_PROTOCOL}\0{identity}\0{image_hash}".encode("utf-8")
    ).hexdigest()[:20]
    return PairExample(
        eval_id=eval_id,
        source_identity=identity,
        scene_id=record["scene_id"],
        source_image=_resolve_source_image(record["image_path"]),
        image_sha256=image_hash,
        entity_a=entity_a,
        entity_b=entity_b,
        nearer_choice=nearer_choice,
    )


def load_formal_source(
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_AUDIT_REPORT,
    *,
    verify_images: bool = True,
) -> FormalSource:
    pack = pack.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    manifest_path = pack / "manifest.json"
    annotation_a = pack / "annotator_a/annotations/annotations.jsonl"
    annotation_b = pack / "annotator_b/annotations/annotations.jsonl"
    labels_a_path = pack / "annotator_a/labels/S_spatial/depth.jsonl"
    labels_b_path = pack / "annotator_b/labels/S_spatial/depth.jsonl"
    wrappers_path = pack / "provenance/source_records.jsonl"

    expected_hashes = (
        (manifest_path, EXPECTED_MANIFEST_SHA256, "formal manifest"),
        (report_path, EXPECTED_AUDIT_REPORT_SHA256, "formal audit report"),
        (annotation_a, EXPECTED_ANNOTATION_A_SHA256, "reviewer A stream"),
        (annotation_b, EXPECTED_ANNOTATION_B_SHA256, "reviewer B stream"),
        (labels_a_path, EXPECTED_LABELS_SHA256, "reviewer A labels"),
        (labels_b_path, EXPECTED_LABELS_SHA256, "reviewer B labels"),
        (wrappers_path, EXPECTED_SOURCE_WRAPPERS_SHA256, "source wrappers"),
    )
    for path, expected, label in expected_hashes:
        _assert_hash(path, expected, label)

    manifest = _read_json(manifest_path)
    integrity = manifest.get("integrity") or {}
    if (
        manifest.get("schema_version") != 6
        or manifest.get("stage") != "formal"
        or manifest.get("images") != EXPECTED_RECORDS
        or manifest.get("expected_decisions_per_annotator") != EXPECTED_RECORDS
        or integrity.get("manifest_payload_sha256") != EXPECTED_MANIFEST_PAYLOAD_SHA256
        or integrity.get("selected_sha256") != EXPECTED_SELECTED_SHA256
        or (manifest.get("training_materialization") or {}).get("eligible") is not False
    ):
        raise ContractError("formal manifest contract drift")
    exclusions = manifest.get("exclusion_checks") or {}
    if any(
        exclusions.get(key) not in ([], 0)
        for key in (
            "old_schema5_scene_overlap",
            "frozen_canary_scene_overlap",
            "reviewed_rgb_canary_scene_overlap",
            "all_prior_audit_scene_overlap_count",
            "old_schema5_and_canary_overlap_count",
        )
    ):
        raise ContractError("formal manifest contains prior-audit scene overlap")

    report = _read_json(report_path)
    review = report.get("review") or {}
    if (
        report.get("status") != "FORMAL_PASS"
        or report.get("stage") != "formal"
        or report.get("records") != EXPECTED_RECORDS
        or report.get("unique_images") != EXPECTED_RECORDS
        or report.get("unique_scenes") != EXPECTED_RECORDS
        or report.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or report.get("manifest_payload_sha256") != EXPECTED_MANIFEST_PAYLOAD_SHA256
        or report.get("source_artifacts_rehashed") is not True
        or report.get("source_artifact_errors") != 0
        or report.get("formal_qualification_eligible") is not True
        or review.get("a_complete") != EXPECTED_RECORDS
        or review.get("b_complete") != EXPECTED_RECORDS
        or review.get("agreement_count") != EXPECTED_RECORDS
        or review.get("disagreement_count") != 0
        or review.get("resolved_correct_count") != EXPECTED_RECORDS
        or review.get("formal_40_gate_passed") is not True
    ):
        raise ContractError("formal audit report contract drift")

    labels_a_rows = read_jsonl(labels_a_path)
    labels_b_rows = read_jsonl(labels_b_path)
    if labels_a_rows != labels_b_rows or len(labels_a_rows) != EXPECTED_RECORDS:
        raise ContractError("A/B labels are not identical formal40 projections")
    labels = {row.get("sample_id"): row for row in labels_a_rows}
    if len(labels) != EXPECTED_RECORDS or None in labels:
        raise ContractError("formal labels contain missing or duplicate sample IDs")
    votes_a = _latest_annotations(annotation_a)
    votes_b = _latest_annotations(annotation_b)
    if set(votes_a) != set(labels) or set(votes_b) != set(labels):
        raise ContractError("A/B votes do not exactly cover formal labels")
    keys_a = {row.get("annotator_key") for row in votes_a.values()}
    keys_b = {row.get("annotator_key") for row in votes_b.values()}
    if len(keys_a) != 1 or len(keys_b) != 1 or None in keys_a | keys_b or keys_a & keys_b:
        raise ContractError("formal audit lacks two distinct reviewer keys")
    disagreements = 0
    resolved_correct = 0
    for sample_id, label in labels.items():
        _validate_vote(votes_a[sample_id], label, "A")
        _validate_vote(votes_b[sample_id], label, "B")
        if votes_a[sample_id]["correct"] != votes_b[sample_id]["correct"]:
            disagreements += 1
        elif votes_a[sample_id]["correct"]:
            resolved_correct += 1
    if disagreements != 0 or resolved_correct != EXPECTED_RECORDS:
        raise ContractError("formal audit decisions no longer match FORMAL_PASS")

    selected = manifest.get("selected")
    wrappers = read_jsonl(wrappers_path)
    if not isinstance(selected, list) or len(selected) != EXPECTED_RECORDS:
        raise ContractError("formal selected set is not exactly forty")
    if len(wrappers) != EXPECTED_RECORDS:
        raise ContractError("formal source wrapper count drift")
    wrapper_by_id: dict[str, dict[str, Any]] = {}
    for wrapper in wrappers:
        sample_id = wrapper.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in wrapper_by_id:
            raise ContractError("formal source wrapper identity drift")
        wrapper_by_id[sample_id] = wrapper
    selected_ids = [item.get("sample_ids", {}).get("depth") for item in selected]
    if len(set(selected_ids)) != EXPECTED_RECORDS or set(selected_ids) != set(labels):
        raise ContractError("formal selected/label identity mismatch")

    records: list[dict[str, Any]] = []
    pairs: list[PairExample] = []
    for item, sample_id in zip(selected, selected_ids):
        wrapper = wrapper_by_id.get(sample_id)
        if wrapper is None:
            raise ContractError(f"missing source wrapper: {sample_id}")
        record = wrapper.get("source_record")
        if not isinstance(record, dict):
            raise ContractError(f"source wrapper lacks record: {sample_id}")
        record_hash = canonical_hash(record)
        if (
            wrapper.get("source_record_sha256") != record_hash
            or item.get("source_record_sha256") != record_hash
            or wrapper.get("source_sample_id") != record.get("sample_id")
            or item.get("source_sample_id") != record.get("sample_id")
            or item.get("scene_id") != record.get("scene_id")
            or item.get("image_sha256") != record.get("image_sha256")
        ):
            raise ContractError(f"selected/source binding drift: {sample_id}")
        pair = _pair_from_record(record, verify_image=verify_images)
        records.append(record)
        pairs.append(pair)

    if canonical_hash(records) != EXPECTED_SOURCE_RECORDS_CANONICAL_SHA256:
        raise ContractError("embedded source record canonical hash drift")
    if (
        len({pair.eval_id for pair in pairs}) != EXPECTED_RECORDS
        or len({pair.scene_id for pair in pairs}) != EXPECTED_RECORDS
        or len({pair.image_sha256 for pair in pairs}) != EXPECTED_RECORDS
        or Counter(pair.nearer_choice for pair in pairs) != Counter(EXPECTED_LABEL_BALANCE)
    ):
        raise ContractError("formal40 uniqueness or A/B balance drift")

    binding: dict[str, Any] = {
        "release": "scannet_rgb_depth_v51_formal40",
        "pack_manifest": file_binding(manifest_path),
        "pack_manifest_payload_sha256": EXPECTED_MANIFEST_PAYLOAD_SHA256,
        "selected_sha256": EXPECTED_SELECTED_SHA256,
        "audit_report": file_binding(report_path),
        "annotation_a": file_binding(annotation_a),
        "annotation_b": file_binding(annotation_b),
        "labels_a": file_binding(labels_a_path),
        "labels_b": file_binding(labels_b_path),
        "source_wrappers": file_binding(wrappers_path),
        "source_records_canonical_sha256": EXPECTED_SOURCE_RECORDS_CANONICAL_SHA256,
        "records": EXPECTED_RECORDS,
        "unique_scenes": EXPECTED_RECORDS,
        "unique_images": EXPECTED_RECORDS,
        "official_split": "val",
        "human_audit_status": "FORMAL_PASS",
        "training_materialization_eligible": False,
    }
    binding["identity_sha256"] = canonical_hash(binding)
    return FormalSource(tuple(pairs), binding)


def _g_prompt(pair: PairExample) -> str:
    return (
        "Inspect the RGB image and locate only the two named objects. "
        "Return one bounding box for each object in image-normalized 0-1000 "
        "[x_min, y_min, x_max, y_max] coordinates.\n"
        f"A: {pair.entity_a.name}\n"
        f"B: {pair.entity_b.name}\n"
        'Return exactly JSON: {"A":{"bbox_2d":[x1,y1,x2,y2]},'
        '"B":{"bbox_2d":[x1,y1,x2,y2]}}'
    )


def _s_prompt(name_a: str, bbox_a: tuple[int, ...], name_b: str, bbox_b: tuple[int, ...]) -> str:
    return (
        "Inspect the RGB image and compare only the two supplied objects. "
        "Bounding boxes use image-normalized 0-1000 [x_min, y_min, x_max, y_max] coordinates.\n"
        f"A: {name_a}; bbox {list(bbox_a)}\n"
        f"B: {name_b}; bbox {list(bbox_b)}\n"
        "Which supplied object is nearer to the camera? Answer only A or B."
    )


def format_g_request(pair: PairExample, public_image: Path) -> dict[str, Any]:
    request = {
        "eval_id": pair.eval_id,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "track": TRACK_G,
        "messages": [{"role": "user", "content": _g_prompt(pair)}],
        "images": [str(public_image.expanduser().resolve())],
        "image_sha256": pair.image_sha256,
        "pair": {
            "A": {"name": pair.entity_a.name},
            "B": {"name": pair.entity_b.name},
            "bbox_coordinate_space": "image_normalized_0_1000_xyxy",
        },
    }
    validate_public_request(request, expected_track=TRACK_G)
    return request


def format_s_request(
    pair: PairExample,
    public_image: Path,
    *,
    track: str,
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> dict[str, Any]:
    if track not in {TRACK_S_ORACLE, TRACK_S_PRED}:
        raise ContractError(f"invalid S track: {track}")
    request = {
        "eval_id": pair.eval_id,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "track": track,
        "messages": [
            {
                "role": "user",
                "content": _s_prompt(pair.entity_a.name, bbox_a, pair.entity_b.name, bbox_b),
            }
        ],
        "images": [str(public_image.expanduser().resolve())],
        "image_sha256": pair.image_sha256,
        "pair": {
            "A": {"name": pair.entity_a.name, "bbox_2d": list(bbox_a)},
            "B": {"name": pair.entity_b.name, "bbox_2d": list(bbox_b)},
            "bbox_coordinate_space": "image_normalized_0_1000_xyxy",
        },
    }
    validate_public_request(request, expected_track=track)
    return request


def format_predicted_s_request(
    g_request: dict[str, Any], grounding: GroundingParse
) -> dict[str, Any]:
    if not grounding.valid:
        raise ContractError("cannot format S-pred request from invalid grounding")
    pair = g_request["pair"]
    request = {
        "eval_id": g_request["eval_id"],
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "track": TRACK_S_PRED,
        "messages": [
            {
                "role": "user",
                "content": _s_prompt(
                    pair["A"]["name"],
                    grounding.bbox_a or (0, 0, 1, 1),
                    pair["B"]["name"],
                    grounding.bbox_b or (0, 0, 1, 1),
                ),
            }
        ],
        "images": list(g_request["images"]),
        "image_sha256": g_request["image_sha256"],
        "pair": {
            "A": {"name": pair["A"]["name"], "bbox_2d": list(grounding.bbox_a or ())},
            "B": {"name": pair["B"]["name"], "bbox_2d": list(grounding.bbox_b or ())},
            "bbox_coordinate_space": "image_normalized_0_1000_xyxy",
        },
    }
    validate_public_request(request, expected_track=TRACK_S_PRED)
    return request


def validate_public_request(row: dict[str, Any], *, expected_track: str | None = None) -> None:
    required = {
        "eval_id",
        "evaluation_protocol",
        "track",
        "messages",
        "images",
        "image_sha256",
        "pair",
    }
    if set(row) != required:
        raise ContractError("public request schema drift")
    eval_id = row.get("eval_id")
    if not isinstance(eval_id, str) or not _EVAL_ID.fullmatch(eval_id):
        raise ContractError("public request has invalid blinded eval_id")
    if row.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        raise ContractError(f"{eval_id}: evaluation protocol drift")
    track = row.get("track")
    if track not in {TRACK_G, TRACK_S_ORACLE, TRACK_S_PRED}:
        raise ContractError(f"{eval_id}: request track drift")
    if expected_track is not None and track != expected_track:
        raise ContractError(f"{eval_id}: expected track {expected_track}, got {track}")
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or set(messages[0]) != {"role", "content"}
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise ContractError(f"{eval_id}: message schema drift")
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ContractError(f"{eval_id}: exactly one blinded image path is required")
    image_path = Path(images[0]).expanduser().resolve()
    if image_path.name != f"{eval_id}.png" or image_path.parent.name != "images":
        raise ContractError(f"{eval_id}: image path is not blinded by eval_id")
    image_hash = row.get("image_sha256")
    if not isinstance(image_hash, str) or not _SHA256.fullmatch(image_hash):
        raise ContractError(f"{eval_id}: invalid image hash")
    pair = row.get("pair")
    if not isinstance(pair, dict) or set(pair) != {"A", "B", "bbox_coordinate_space"}:
        raise ContractError(f"{eval_id}: pair schema drift")
    if pair.get("bbox_coordinate_space") != "image_normalized_0_1000_xyxy":
        raise ContractError(f"{eval_id}: bbox coordinate contract drift")
    names: list[str] = []
    for choice in ("A", "B"):
        entity = pair.get(choice)
        expected_entity_keys = {"name"} if track == TRACK_G else {"name", "bbox_2d"}
        if not isinstance(entity, dict) or set(entity) != expected_entity_keys:
            raise ContractError(f"{eval_id}/{choice}: public entity schema drift")
        name = entity.get("name")
        if not isinstance(name, str) or not name.strip() or _INSTANCE_SUFFIX.search(name):
            raise ContractError(f"{eval_id}/{choice}: public name is invalid")
        names.append(name)
        if track != TRACK_G:
            _bbox(entity.get("bbox_2d"), f"{eval_id}/{choice}")
    if names[0].casefold() == names[1].casefold():
        raise ContractError(f"{eval_id}: pair names are ambiguous")
    keys = {key.casefold() for key in _iter_keys(row)}
    leaked_keys = keys & _FORBIDDEN_PUBLIC_KEYS
    if leaked_keys:
        raise ContractError(f"{eval_id}: private keys leaked: {sorted(leaked_keys)}")
    serialised = json.dumps(row, ensure_ascii=False).casefold()
    if "[instance" in serialised or "scannet-sdepth" in serialised or _SCENE_ID.search(serialised):
        raise ContractError(f"{eval_id}: private source identity leaked")


def _copy_blinded_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def build_request_release(
    *,
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_AUDIT_REPORT,
    out: Path = DEFAULT_RELEASE,
) -> dict[str, Any]:
    out = out.expanduser().resolve()
    if out.exists():
        release = validate_request_release(out / "release.manifest.json", verify_images=True)
        return {**release.manifest, "build_status": "reused_verified"}
    out.parent.mkdir(parents=True, exist_ok=True)
    source = load_formal_source(pack, report_path, verify_images=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.building.", dir=out.parent))
    try:
        g_rows: list[dict[str, Any]] = []
        oracle_rows: list[dict[str, Any]] = []
        public_images: list[dict[str, Any]] = []
        for pair in source.pairs:
            actual_image = temporary / "images" / f"{pair.eval_id}.png"
            final_image = out / "images" / f"{pair.eval_id}.png"
            _copy_blinded_image(pair.source_image, actual_image)
            if file_sha256(actual_image) != pair.image_sha256:
                raise ContractError(f"blinded image copy hash drift: {pair.eval_id}")
            public_images.append({"eval_id": pair.eval_id, **_future_binding(actual_image, final_image)})
            g_rows.append(format_g_request(pair, final_image))
            oracle_rows.append(
                format_s_request(
                    pair,
                    final_image,
                    track=TRACK_S_ORACLE,
                    bbox_a=pair.entity_a.bbox,
                    bbox_b=pair.entity_b.bbox,
                )
            )

        actual_g = temporary / "g_requests.jsonl"
        actual_oracle = temporary / "s_oracle_requests.jsonl"
        write_jsonl(actual_g, g_rows)
        write_jsonl(actual_oracle, oracle_rows)
        final_g = out / actual_g.name
        final_oracle = out / actual_oracle.name
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "protocol": RELEASE_PROTOCOL,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "release": "formal40",
            "formal_source": source.binding,
            "ab_assignment": {
                "rule": "sha256_protocol_salt_source_identity_low_bit",
                "salt": AB_PERMUTATION_SALT,
                "label_balance": EXPECTED_LABEL_BALANCE,
            },
            "requests": {
                TRACK_G: {
                    "file": _future_binding(actual_g, final_g),
                    "records_sha256": canonical_hash(g_rows),
                    "records": len(g_rows),
                },
                TRACK_S_ORACLE: {
                    "file": _future_binding(actual_oracle, final_oracle),
                    "records_sha256": canonical_hash(oracle_rows),
                    "records": len(oracle_rows),
                },
            },
            "public_images": public_images,
            "blinding": {
                "model_fields": ["messages", "images"],
                "private_fields_passed_to_model": [],
                "instance_ids_exposed": False,
                "sensor_depths_exposed": False,
                "target_direction_or_ranks_exposed": False,
                "source_or_scene_ids_exposed": False,
            },
            "builder": file_binding(Path(__file__)),
            "training_materialization": {
                "eligible": False,
                "allowed_use": "held_out_evaluation_only",
            },
        }
        manifest["identity_sha256"] = canonical_hash(manifest)
        write_json(temporary / "release.manifest.json", manifest)
        temporary.replace(out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**validate_request_release(out / "release.manifest.json", verify_images=True).manifest, "build_status": "generated"}


def validate_request_release(
    manifest_path: Path,
    *,
    verify_images: bool = True,
) -> RequestRelease:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path)
    identity = manifest.get("identity_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    if identity != canonical_hash(unsigned):
        raise ContractError("request release identity drift")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol") != RELEASE_PROTOCOL
        or manifest.get("evaluation_protocol") != EVALUATION_PROTOCOL
        or manifest.get("release") != "formal40"
        or manifest.get("builder") != file_binding(Path(__file__))
        or (manifest.get("training_materialization") or {}).get("eligible") is not False
    ):
        raise ContractError("request release contract drift")

    source_binding = manifest.get("formal_source")
    if not isinstance(source_binding, dict):
        raise ContractError("request release lacks formal source binding")
    pack_manifest = verify_file_binding(source_binding.get("pack_manifest"), deep=True)
    audit_report = verify_file_binding(source_binding.get("audit_report"), deep=True)
    source = load_formal_source(pack_manifest.parent, audit_report, verify_images=verify_images)
    if source.binding != source_binding:
        raise ContractError("request release formal source binding drift")

    assignment = manifest.get("ab_assignment") or {}
    if (
        assignment.get("rule") != "sha256_protocol_salt_source_identity_low_bit"
        or assignment.get("salt") != AB_PERMUTATION_SALT
        or assignment.get("label_balance") != EXPECTED_LABEL_BALANCE
    ):
        raise ContractError("request release A/B assignment drift")
    blinding = manifest.get("blinding") or {}
    if (
        blinding.get("model_fields") != ["messages", "images"]
        or blinding.get("private_fields_passed_to_model") != []
        or any(
            blinding.get(key) is not False
            for key in (
                "instance_ids_exposed",
                "sensor_depths_exposed",
                "target_direction_or_ranks_exposed",
                "source_or_scene_ids_exposed",
            )
        )
    ):
        raise ContractError("request release blinding contract drift")

    public_images = manifest.get("public_images")
    if not isinstance(public_images, list) or len(public_images) != EXPECTED_RECORDS:
        raise ContractError("public image release count drift")
    image_binding_by_id: dict[str, dict[str, Any]] = {}
    pair_by_id = {pair.eval_id: pair for pair in source.pairs}
    for item in public_images:
        if not isinstance(item, dict) or set(item) != {"eval_id", "path", "bytes", "sha256"}:
            raise ContractError("public image binding schema drift")
        eval_id = item["eval_id"]
        if eval_id in image_binding_by_id or eval_id not in pair_by_id:
            raise ContractError("public image eval_id coverage drift")
        path = verify_file_binding(
            {key: item[key] for key in ("path", "bytes", "sha256")}, deep=verify_images
        )
        if path.name != f"{eval_id}.png" or path.parent.name != "images":
            raise ContractError(f"public image path is not blinded: {eval_id}")
        if item["sha256"] != pair_by_id[eval_id].image_sha256:
            raise ContractError(f"public/source image hash mismatch: {eval_id}")
        image_binding_by_id[eval_id] = item
    if set(image_binding_by_id) != set(pair_by_id):
        raise ContractError("public image release does not cover formal40")

    request_manifest = manifest.get("requests")
    if not isinstance(request_manifest, dict) or set(request_manifest) != {TRACK_G, TRACK_S_ORACLE}:
        raise ContractError("static request release track coverage drift")
    rows_by_track: dict[str, list[dict[str, Any]]] = {}
    for track in (TRACK_G, TRACK_S_ORACLE):
        binding = request_manifest.get(track)
        if not isinstance(binding, dict) or set(binding) != {"file", "records_sha256", "records"}:
            raise ContractError(f"{track}: request file binding schema drift")
        request_path = verify_file_binding(binding["file"], deep=True)
        rows = read_jsonl(request_path)
        if binding.get("records") != EXPECTED_RECORDS or canonical_hash(rows) != binding.get(
            "records_sha256"
        ):
            raise ContractError(f"{track}: request count or canonical hash drift")
        for row in rows:
            validate_public_request(row, expected_track=track)
        expected: list[dict[str, Any]] = []
        for pair in source.pairs:
            public_image = Path(image_binding_by_id[pair.eval_id]["path"])
            if track == TRACK_G:
                expected.append(format_g_request(pair, public_image))
            else:
                expected.append(
                    format_s_request(
                        pair,
                        public_image,
                        track=TRACK_S_ORACLE,
                        bbox_a=pair.entity_a.bbox,
                        bbox_b=pair.entity_b.bbox,
                    )
                )
        if rows != expected:
            raise ContractError(f"{track}: requests do not exactly derive from formal40")
        rows_by_track[track] = rows
    return RequestRelease(manifest, source, rows_by_track)


def _strip_reasoning(text: Any) -> tuple[str, str | None]:
    if not isinstance(text, str):
        return "", "response_not_string"
    value = text.strip()
    while "<think>" in value.casefold():
        lower = value.casefold()
        start = lower.find("<think>")
        end = lower.find("</think>", start + 7)
        if end < 0:
            return "", "unclosed_reasoning"
        value = (value[:start] + value[end + len("</think>") :]).strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value).strip()
    return value, None


def parse_grounding_response(text: Any) -> GroundingParse:
    value, error = _strip_reasoning(text)
    if error:
        return GroundingParse(None, None, error)
    if not value:
        return GroundingParse(None, None, "empty_response")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        if start < 0:
            return GroundingParse(None, None, "no_json_object")
        try:
            payload, end = json.JSONDecoder().raw_decode(value[start:])
        except json.JSONDecodeError:
            return GroundingParse(None, None, "invalid_json")
        trailing = value[start + end :].strip(" \t\r\n.;")
        if trailing:
            return GroundingParse(None, None, "trailing_non_json_content")
    if not isinstance(payload, dict) or set(payload) != {"A", "B"}:
        return GroundingParse(None, None, "grounding_schema")
    boxes: list[tuple[int, int, int, int]] = []
    for choice in ("A", "B"):
        entity = payload.get(choice)
        if not isinstance(entity, dict) or set(entity) != {"bbox_2d"}:
            return GroundingParse(None, None, f"grounding_schema_{choice}")
        try:
            boxes.append(_bbox(entity.get("bbox_2d"), f"G response/{choice}"))
        except ContractError:
            return GroundingParse(None, None, f"invalid_bbox_{choice}")
    return GroundingParse(boxes[0], boxes[1], "parsed_strict_json")


def parse_s_choice(text: Any) -> ChoiceParse:
    value, error = _strip_reasoning(text)
    if error:
        return ChoiceParse(None, error)
    if not value:
        return ChoiceParse(None, "empty_response")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("nearer", "answer", "choice"):
            choice = payload.get(key)
            if isinstance(choice, str) and choice.strip().upper() in {"A", "B"}:
                return ChoiceParse(choice.strip().upper(), "structured")
    normal = " ".join(re.sub(r"[^A-Za-z]+", " ", value).split())
    match = re.fullmatch(r"(?:(?:answer|choice|option|object)\s+)?([ABab])", normal)
    if match:
        return ChoiceParse(match.group(1).upper(), "direct")
    relation = re.search(
        r"\b(?:object\s+)?([ABab])\s+(?:is\s+)?(?:the\s+)?(?:nearer|closer)\b",
        value,
    )
    if relation:
        return ChoiceParse(relation.group(1).upper(), "relation_phrase")
    return ChoiceParse(None, "ambiguous_or_unrecognised")


def iou(box_a: tuple[int, ...], box_b: tuple[int, ...]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return round(float(numerator) / denominator, 6) if denominator else None


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ContractError("cannot compute percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(
    deltas: list[int],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if not deltas:
        raise ContractError("paired bootstrap requires at least one scene")
    rng = random.Random(seed)
    n = len(deltas)
    means = [sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return {
        "unit": "scene",
        "paired_scenes": n,
        "samples": samples,
        "seed": seed,
        "confidence": 0.95,
        "ci": [round(_percentile(means, 0.025), 6), round(_percentile(means, 0.975), 6)],
    }


def _key_predictions(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        eval_id = row.get("eval_id")
        if not isinstance(eval_id, str) or eval_id in keyed:
            raise ContractError(f"{label}: missing or duplicate eval_id")
        if not isinstance(row.get("response"), str):
            raise ContractError(f"{label}/{eval_id}: response is not a string")
        if not isinstance(row.get("hit_token_cap"), bool):
            raise ContractError(f"{label}/{eval_id}: token-cap metadata is absent")
        if row["hit_token_cap"]:
            raise ContractError(f"{label}/{eval_id}: response hit the frozen token cap")
        keyed[eval_id] = row
    return keyed


def score_paired_predictions(
    pairs: Iterable[PairExample],
    g_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    predicted_rows: list[dict[str, Any]],
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pair_list = list(pairs)
    if not pair_list:
        raise ContractError("paired score requires source pairs")
    pair_ids = [pair.eval_id for pair in pair_list]
    if len(set(pair_ids)) != len(pair_ids):
        raise ContractError("paired source has duplicate eval IDs")
    expected = set(pair_ids)
    g = _key_predictions(g_rows, "G predictions")
    oracle = _key_predictions(oracle_rows, "S-oracle predictions")
    predicted = _key_predictions(predicted_rows, "S-pred predictions")
    if set(g) != expected or set(oracle) != expected:
        raise ContractError("G and S-oracle predictions must cover every formal scene")

    parsed_groundings = {eval_id: parse_grounding_response(row["response"]) for eval_id, row in g.items()}
    valid_grounding_ids = {eval_id for eval_id, parsed in parsed_groundings.items() if parsed.valid}
    if set(predicted) != valid_grounding_ids:
        raise ContractError("S-pred predictions must exactly cover valid parsed G requests")

    g_parse_status: Counter[str] = Counter()
    oracle_parse_status: Counter[str] = Counter()
    predicted_parse_status: Counter[str] = Counter()
    transitions: list[dict[str, Any]] = []
    g_success = 0
    g_valid = 0
    min_ious: list[float] = []
    oracle_correct_count = 0
    oracle_parsed = 0
    predicted_correct_count = 0
    predicted_parsed = 0
    predicted_correct_given_g_success = 0
    end_to_end_correct = 0
    paired_deltas: list[int] = []

    for pair in pair_list:
        grounding = parsed_groundings[pair.eval_id]
        g_parse_status[grounding.status] += 1
        iou_a: float | None = None
        iou_b: float | None = None
        both_box_success = False
        if grounding.valid:
            g_valid += 1
            iou_a = iou(grounding.bbox_a or (0, 0, 1, 1), pair.entity_a.bbox)
            iou_b = iou(grounding.bbox_b or (0, 0, 1, 1), pair.entity_b.bbox)
            min_ious.append(min(iou_a, iou_b))
            both_box_success = iou_a >= 0.5 and iou_b >= 0.5
            g_success += int(both_box_success)
        else:
            min_ious.append(0.0)

        oracle_choice = parse_s_choice(oracle[pair.eval_id]["response"])
        oracle_parse_status[oracle_choice.status] += 1
        oracle_is_correct = oracle_choice.choice == pair.nearer_choice
        oracle_parsed += int(oracle_choice.choice is not None)
        oracle_correct_count += int(oracle_is_correct)

        predicted_choice = ChoiceParse(None, "no_request_due_to_invalid_g")
        predicted_is_correct = False
        if grounding.valid:
            predicted_choice = parse_s_choice(predicted[pair.eval_id]["response"])
            predicted_parse_status[predicted_choice.status] += 1
            predicted_parsed += int(predicted_choice.choice is not None)
            predicted_is_correct = predicted_choice.choice == pair.nearer_choice
            predicted_correct_count += int(predicted_is_correct)
        else:
            predicted_parse_status[predicted_choice.status] += 1
        strict_correct = both_box_success and predicted_is_correct
        end_to_end_correct += int(strict_correct)
        if both_box_success and predicted_is_correct:
            predicted_correct_given_g_success += 1
        paired_deltas.append(int(oracle_is_correct) - int(strict_correct))

        transition = (
            "oracle_correct_to_predicted_correct"
            if oracle_is_correct and strict_correct
            else "oracle_correct_to_predicted_wrong"
            if oracle_is_correct
            else "oracle_wrong_to_predicted_correct"
            if strict_correct
            else "oracle_wrong_to_predicted_wrong"
        )
        transitions.append(
            {
                "eval_id": pair.eval_id,
                "g_parse_status": grounding.status,
                "g_parse_valid": grounding.valid,
                "g_iou_a": round(iou_a, 6) if iou_a is not None else None,
                "g_iou_b": round(iou_b, 6) if iou_b is not None else None,
                "g_both_box_iou_at_0_5": both_box_success,
                "s_oracle_parse_status": oracle_choice.status,
                "s_oracle_correct": oracle_is_correct,
                "s_pred_request_emitted": grounding.valid,
                "s_pred_parse_status": predicted_choice.status,
                "s_pred_correct": predicted_is_correct,
                "s_pred_end_to_end_correct": strict_correct,
                "paired_delta_oracle_minus_predicted_e2e": int(oracle_is_correct)
                - int(strict_correct),
                "transition": transition,
            }
        )

    n = len(pair_list)
    oracle_accuracy = _ratio(oracle_correct_count, n)
    predicted_e2e = _ratio(end_to_end_correct, n)
    dependency_gap = round((oracle_accuracy or 0.0) - (predicted_e2e or 0.0), 6)
    transition_counts = dict(sorted(Counter(item["transition"] for item in transitions).items()))
    report: dict[str, Any] = {
        "result_schema_version": 1,
        "protocol": RESULT_PROTOCOL,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "release": "formal40",
        "status": "FORMAL_SCORE_COMPLETE" if provenance is not None else "SYNTHETIC_SCORE_COMPLETE",
        "task_contract": {
            "stages": [TRACK_G, TRACK_S_ORACLE, TRACK_S_PRED],
            "same_model_and_deterministic_decoding_required": True,
            "missing_or_invalid_g_end_to_end_credit": 0,
            "primary_unit": "scene",
            "training_materialization_eligible": False,
        },
        "metrics": {
            "n_scenes": n,
            "g_valid_pair_coverage": _ratio(g_valid, n),
            "g_both_box_iou_at_0_5": _ratio(g_success, n),
            "g_mean_min_pair_iou": round(sum(min_ious) / n, 6),
            "s_oracle_accuracy": oracle_accuracy,
            "s_oracle_parse_coverage": _ratio(oracle_parsed, n),
            "s_predicted_context_accuracy_all": _ratio(predicted_correct_count, n),
            "s_predicted_context_parse_coverage_all": _ratio(predicted_parsed, n),
            "s_predicted_context_end_to_end_accuracy": predicted_e2e,
            "s_accuracy_given_g_success": _ratio(predicted_correct_given_g_success, g_success),
            "dependency_gap_oracle_minus_predicted_e2e": dependency_gap,
        },
        "parse_status": {
            "G": dict(sorted(g_parse_status.items())),
            "S_oracle": dict(sorted(oracle_parse_status.items())),
            "S_pred": dict(sorted(predicted_parse_status.items())),
        },
        "paired_bootstrap": {
            "metric": "dependency_gap_oracle_minus_predicted_e2e",
            **paired_bootstrap_ci(paired_deltas),
        },
        "transition_counts": transition_counts,
        "per_item_transitions": transitions,
        "provenance": provenance or {"synthetic_fixture_only": True},
        "trust": {
            "score_available": provenance is not None,
            "all_three_prediction_stages_complete": True,
            "all_token_cap_metadata_complete": True,
            "zero_capped_responses": True,
            "held_out_evaluation_only": True,
        },
    }
    return report


def score_command(args: argparse.Namespace) -> int:
    release = validate_request_release(args.release_manifest, verify_images=True)
    if len(release.source.pairs) != EXPECTED_RECORDS:
        raise ContractError("formal score source count drift")
    from eval import scannet_rgb_depth_v51_dlc as dlc

    try:
        binding = dlc.verify_model_binding(args.model_binding, deep=True)
        g_bundle = dlc.verify_prediction_bundle(
            manifest_path=args.g_manifest,
            binding_path=args.model_binding,
            predictions_path=args.g_predictions,
            expected_track=TRACK_G,
            deep_model=False,
        )
        oracle_bundle = dlc.verify_prediction_bundle(
            manifest_path=args.s_oracle_manifest,
            binding_path=args.model_binding,
            predictions_path=args.s_oracle_predictions,
            expected_track=TRACK_S_ORACLE,
            deep_model=False,
        )
        predicted_request = dlc.verify_predicted_request_release(
            manifest_path=args.s_pred_request_manifest,
            binding_path=args.model_binding,
            g_manifest_path=args.g_manifest,
            g_predictions_path=args.g_predictions,
            verify_images=True,
        )
        predicted_bundle = dlc.verify_prediction_bundle(
            manifest_path=args.s_pred_manifest,
            binding_path=args.model_binding,
            predictions_path=args.s_pred_predictions,
            expected_track=TRACK_S_PRED,
            dynamic_request_manifest=args.s_pred_request_manifest,
            deep_model=False,
        )
    except dlc.RunnerError as exc:
        raise ContractError(f"DLC provenance failed: {exc}") from exc

    identities = {
        bundle["provenance"]["model_binding_identity_sha256"]
        for bundle in (g_bundle, oracle_bundle, predicted_bundle)
    }
    if identities != {binding["identity_sha256"]}:
        raise ContractError("three stages do not use the same bound model")
    receipt_identities = {
        bundle["receipt"]["identity_sha256"]
        for bundle in (g_bundle, oracle_bundle, predicted_bundle)
    }
    if len(receipt_identities) != 1:
        raise ContractError("three stages do not use the same code/runtime receipt")
    if predicted_request["manifest"]["g_prediction_manifest"] != file_binding(
        args.g_manifest
    ):
        raise ContractError("S-pred requests are not bound to the scored G predictions")

    provenance: dict[str, Any] = {
        "verified": True,
        "request_release": file_binding(args.release_manifest),
        "request_release_identity_sha256": release.manifest["identity_sha256"],
        "formal_source": release.source.binding,
        "input_receipt": g_bundle["receipt"],
        "model_binding": file_binding(args.model_binding),
        "model_binding_identity_sha256": binding["identity_sha256"],
        "model_inventory": binding["model"],
        "stages": {
            TRACK_G: g_bundle["provenance"],
            TRACK_S_ORACLE: oracle_bundle["provenance"],
            TRACK_S_PRED: predicted_bundle["provenance"],
        },
        "predicted_request_release": file_binding(args.s_pred_request_manifest),
        "same_model_binding_all_stages": True,
        "training_materialization_eligible": False,
    }
    provenance["identity_sha256"] = canonical_hash(provenance)
    report = score_paired_predictions(
        release.source.pairs,
        g_bundle["rows"],
        oracle_bundle["rows"],
        predicted_bundle["rows"],
        provenance=provenance,
    )
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def dry_run_command(_: argparse.Namespace) -> int:
    pairs: list[PairExample] = []
    for index in range(4):
        pairs.append(
            PairExample(
                eval_id=f"gsa-sdepth-v51-{index:020x}",
                source_identity=f"synthetic-private-{index}",
                scene_id=f"scene{index:04d}_00",
                source_image=Path(f"/synthetic/{index}.png"),
                image_sha256=f"{index + 1:064x}",
                entity_a=Entity("chair", (100, 100, 300, 500)),
                entity_b=Entity("table", (500, 100, 800, 600)),
                nearer_choice="A" if index % 2 == 0 else "B",
            )
        )
    good = '{"A":{"bbox_2d":[100,100,300,500]},"B":{"bbox_2d":[500,100,800,600]}}'
    bad_iou = '{"A":{"bbox_2d":[0,0,50,50]},"B":{"bbox_2d":[50,50,100,100]}}'
    g_rows = [
        {"eval_id": pairs[0].eval_id, "response": good, "hit_token_cap": False},
        {"eval_id": pairs[1].eval_id, "response": good, "hit_token_cap": False},
        {"eval_id": pairs[2].eval_id, "response": "not-json", "hit_token_cap": False},
        {"eval_id": pairs[3].eval_id, "response": bad_iou, "hit_token_cap": False},
    ]
    oracle_rows = [
        {"eval_id": pair.eval_id, "response": pair.nearer_choice, "hit_token_cap": False}
        for pair in pairs
    ]
    predicted_rows = [
        {"eval_id": pairs[0].eval_id, "response": "A", "hit_token_cap": False},
        {"eval_id": pairs[1].eval_id, "response": "A", "hit_token_cap": False},
        {"eval_id": pairs[3].eval_id, "response": "B", "hit_token_cap": False},
    ]
    report = score_paired_predictions(pairs, g_rows, oracle_rows, predicted_rows)
    passed = (
        report["metrics"]["g_both_box_iou_at_0_5"] == 0.5
        and report["metrics"]["s_oracle_accuracy"] == 1.0
        and report["metrics"]["s_predicted_context_end_to_end_accuracy"] == 0.25
        and report["metrics"]["s_accuracy_given_g_success"] == 0.5
        and report["metrics"]["dependency_gap_oracle_minus_predicted_e2e"] == 0.75
    )
    output = {
        "status": "DRY_RUN_PASS" if passed else "DRY_RUN_FAIL",
        "synthetic_fixture_only": True,
        "metrics": report["metrics"],
        "transition_counts": report["transition_counts"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if passed else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-release", help="derive the pinned blinded formal40 requests")
    build.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    build.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    build.add_argument("--out", type=Path, default=DEFAULT_RELEASE)
    build.set_defaults(func=lambda args: _build_command(args))

    verify = sub.add_parser("verify-release", help="rehash and verify a blinded request release")
    verify.add_argument("--manifest", type=Path, default=DEFAULT_RELEASE / "release.manifest.json")
    verify.set_defaults(func=lambda args: _verify_command(args))

    score = sub.add_parser("score", help="score a fully bound three-stage paired run")
    score.add_argument("--release-manifest", type=Path, required=True)
    score.add_argument("--model-binding", type=Path, required=True)
    score.add_argument("--g-predictions", type=Path, required=True)
    score.add_argument("--g-manifest", type=Path, required=True)
    score.add_argument("--s-oracle-predictions", type=Path, required=True)
    score.add_argument("--s-oracle-manifest", type=Path, required=True)
    score.add_argument("--s-pred-request-manifest", type=Path, required=True)
    score.add_argument("--s-pred-predictions", type=Path, required=True)
    score.add_argument("--s-pred-manifest", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.set_defaults(func=score_command)

    dry = sub.add_parser("dry-run", help="run a CPU-only synthetic parser/scorer fixture")
    dry.set_defaults(func=dry_run_command)
    return parser.parse_args(argv)


def _build_command(args: argparse.Namespace) -> int:
    result = build_request_release(pack=args.pack, report_path=args.audit_report, out=args.out)
    summary = {
        "status": "REQUEST_RELEASE_READY",
        "build_status": result["build_status"],
        "manifest": str((args.out / "release.manifest.json").resolve()),
        "identity_sha256": result["identity_sha256"],
        "records": result["formal_source"]["records"],
        "training_materialization_eligible": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    release = validate_request_release(args.manifest, verify_images=True)
    summary = {
        "status": "REQUEST_RELEASE_VALID",
        "manifest_sha256": file_sha256(args.manifest),
        "identity_sha256": release.manifest["identity_sha256"],
        "records": len(release.source.pairs),
        "tracks": {key: len(value) for key, value in release.requests.items()},
        "label_balance": dict(sorted(Counter(pair.nearer_choice for pair in release.source.pairs).items())),
        "training_materialization_eligible": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except ContractError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
