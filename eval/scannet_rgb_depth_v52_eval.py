#!/usr/bin/env python3
"""Formal v5.2 unique-referent G -> S intervention builder and scorer.

The evaluator keeps the forty private ScanNet labels outside model inputs.  It
accepts a run only after the live v5.2 checker reproduces ``FORMAL_PASS`` and
binds every annotation stream used by that decision.  Grounding predictions
are parsed by the frozen public interface and are never oracle matched.
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

from eval import scannet_rgb_depth_v52_interface as interface  # noqa: E402
from tools import build_scannet_rgb_depth_v52_unique_referent_formal as builder  # noqa: E402
from tools import check_scannet_rgb_depth_v52_unique_referent_formal as checker  # noqa: E402


DEFAULT_PACK = builder.DEFAULT_OUTPUT
DEFAULT_FORMAL_REPORT = ROOT / "artifacts/audits/scannet_rgb_depth_v52_formal_live.json"
DEFAULT_RELEASE = ROOT / "artifacts/depth_v52_eval_interface/formal40"
INTERVENTION_CONTRACT_PATH = (
    ROOT / "experiments/scannet_rgb_depth_v52_gs_intervention_contract.json"
)

EVALUATION_PROTOCOL = "gsa-depth-v5.2-unique-referent-formal40-gs-intervention-eval-v4"
RELEASE_PROTOCOL = "gsa-depth-v5.2-unique-referent-formal40-intervention-release-v4"
RESULT_PROTOCOL = "gsa-depth-v5.2-unique-referent-formal40-gs-intervention-result-v4"
TRACK_G = "G_unique_named_pair_bbox"
TRACK_S_ORACLE = "S_oracle_unique_grounding"
TRACK_S_PRED = "G_to_S_predicted_unique_grounding"
TRACK_S_ABSENT = "S_absent_grounding"
TRACK_S_CORRUPTED = "S_corrupted_grounding"
STATIC_TRACKS = (TRACK_G, TRACK_S_ORACLE, TRACK_S_ABSENT, TRACK_S_CORRUPTED)
ALL_TRACKS = (*STATIC_TRACKS, TRACK_S_PRED)
ATTRIBUTION_STATES = ("Attributed(G)", "Local(S)", "Inconclusive")
EXPECTED_RECORDS = 40
EXPECTED_LABEL_BALANCE = {"A": 20, "B": 20}
BOOTSTRAP_SEED = 52040
BOOTSTRAP_SAMPLES = 10_000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVAL_ID = re.compile(r"^gsa-sdepth-v52-[0-9a-f]{20}$")
_SCENE_ID = re.compile(r"scene\d{4}_\d{2}", re.IGNORECASE)
_FORBIDDEN_PUBLIC_TEXT = (
    "[instance",
    "source_sample_id",
    "source_record",
    "sensor_measurement",
    "relative_depth",
    "nearer_slot",
    "farther_slot",
)


class ContractError(ValueError):
    """An input, release, prediction, or receipt violates the formal contract."""


class FormalGateBlocked(ContractError):
    """Human review has not produced a live FORMAL_PASS decision."""

    def __init__(self, checker_status: str, report: dict[str, Any]):
        super().__init__(f"BLOCKED_PENDING_HUMAN_REVIEW: checker_status={checker_status}")
        self.checker_status = checker_status
        self.report = report


@dataclass(frozen=True)
class Entity:
    category: str
    referring_expression: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PairExample:
    eval_id: str
    sample_id: str
    scene_id: str
    source_image: Path
    image_sha256: str
    entity_a: Entity
    entity_b: Entity
    nearer_choice: str
    label: dict[str, Any]


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

    @property
    def valid(self) -> bool:
        return self.choice is not None


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
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def verify_file_binding(binding: Any, *, deep: bool = True) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
        raise ContractError("file binding schema drift")
    path = Path(str(binding.get("path", ""))).expanduser().resolve()
    if str(path) != binding.get("path") or not path.is_file():
        raise ContractError(f"bound file is missing or non-canonical: {path}")
    if path.stat().st_size != binding.get("bytes"):
        raise ContractError(f"bound file size drift: {path}")
    digest = binding.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ContractError(f"bound file hash is invalid: {path}")
    if deep and file_sha256(path) != digest:
        raise ContractError(f"bound file hash drift: {path}")
    return path


def _future_binding(actual: Path, final: Path) -> dict[str, Any]:
    return {
        "path": str(final.expanduser().resolve()),
        "bytes": actual.stat().st_size,
        "sha256": file_sha256(actual),
    }


def read_json(path: Path) -> dict[str, Any]:
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


def load_intervention_contract() -> dict[str, Any]:
    contract = read_json(INTERVENTION_CONTRACT_PATH)
    if (
        contract.get("schema_version") != 1
        or contract.get("protocol")
        != "gsa-depth-v5.2-gs-context-intervention-contract-v3"
        or (contract.get("scope") or {}).get("records") != EXPECTED_RECORDS
        or (contract.get("attribution") or {}).get("states") != list(ATTRIBUTION_STATES)
        or (contract.get("attribution") or {}).get("Local(S)_interpretation")
        != "the_target_stage_fails_under_oracle_context_while_G_passes_the_frozen_correctness_threshold"
        or ((contract.get("attribution") or {}).get("decision_rule") or {}).get(
            "Attributed(G)"
        )
        != "all_outputs_healthy_and_S_pred_wrong_and_G_incorrect_and_S_oracle_correct_and_S_absent_wrong_and_S_corrupted_wrong"
        or ((contract.get("attribution") or {}).get("decision_rule") or {}).get(
            "Local(S)"
        )
        != "all_outputs_healthy_and_G_correct_and_S_pred_wrong_and_S_oracle_wrong_and_S_absent_wrong_and_S_corrupted_wrong"
        or ((contract.get("contexts") or {}).get("corrupted") or {}).get("transform")
        != {"A_bbox": "oracle_B_bbox", "B_bbox": "oracle_A_bbox"}
        or ((contract.get("contexts") or {}).get("corrupted") or {}).get(
            "downstream_truth_fields_read"
        )
        != []
        or ((contract.get("contexts") or {}).get("corrupted") or {}).get(
            "selection_rule"
        )
        != "apply_to_every_item_without_branching"
        or ((contract.get("contexts") or {}).get("corrupted") or {}).get(
            "per_target_assignment_is_ground_truth"
        )
        is not False
        or ((contract.get("contexts") or {}).get("corrupted") or {}).get(
            "per_target_non_gt_rule"
        )
        != "A_receives_distinct_B_oracle_bbox_and_B_receives_distinct_A_oracle_bbox"
        or contract.get("metrics")
        != {
            "dependency_gap": "S_oracle_accuracy_minus_S_predicted_context_accuracy_on_all_40_scenes",
            "invalid_or_capped_G_predicted_context_credit": 0,
            "strict_end_to_end": "G_both_boxes_IoU_at_least_0.5_and_S_predicted_context_correct",
            "strict_end_to_end_is_dependency_gap_endpoint": False,
            "primary_bootstrap_unit": "paired_scene",
            "predicted_context_and_strict_end_to_end_transitions_reported_separately": True,
        }
    ):
        raise ContractError("G-to-S intervention contract drift")
    return contract


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
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows),
    )


def _standard_adjudication(pack: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    for relative in (
        "reviewer_c/annotations/annotations.jsonl",
        "annotator_c/annotations/annotations.jsonl",
    ):
        candidate = pack / relative
        if candidate.is_file():
            return candidate
    return None


def compute_live_formal_report(pack: Path, adjudication: Path | None = None) -> dict[str, Any]:
    """Run a cheap pending check, then full source rehash only for a complete review."""
    pack = pack.expanduser().resolve()
    adjudication = _standard_adjudication(pack, adjudication)
    shallow = checker.validate_pack(
        pack, rehash_source_artifacts=False, rehash_v3_outputs=False
    )
    review = checker.compile_reviews(pack, shallow, adjudication)
    if review.get("status") not in {checker.FULL_REHASH_REQUIRED, checker.FORMAL_PASS}:
        return review
    deep = checker.validate_pack(pack, rehash_source_artifacts=True, rehash_v3_outputs=True)
    return checker.compile_reviews(pack, deep, adjudication)


def refresh_formal_report(
    pack: Path, report_path: Path, adjudication: Path | None = None
) -> dict[str, Any]:
    report = compute_live_formal_report(pack, adjudication)
    write_json(report_path, report)
    return report


def _validate_report_identity(report: dict[str, Any]) -> None:
    identity = report.get("identity_sha256")
    unsigned = {key: value for key, value in report.items() if key != "identity_sha256"}
    if identity != canonical_hash(unsigned):
        raise ContractError("formal checker report identity drift")


def validate_formal_report(pack: Path, report_path: Path) -> dict[str, Any]:
    pack = pack.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    report = read_json(report_path)
    _validate_report_identity(report)
    if report.get("status") != checker.FORMAL_PASS:
        raise FormalGateBlocked(str(report.get("status", "MISSING_STATUS")), report)
    if (
        report.get("formal_qualification_eligible") is not True
        or Path(str(report.get("pack", ""))).expanduser().resolve() != pack
        or report.get("records") != EXPECTED_RECORDS
        or report.get("unique_scenes") != EXPECTED_RECORDS
        or report.get("unique_images") != EXPECTED_RECORDS
        or report.get("source_artifacts_rehashed") is not True
        or report.get("v3_outputs_rehashed") is not True
    ):
        raise ContractError("formal checker report contract drift")
    streams = report.get("annotation_streams")
    review = report.get("review") or {}
    if not isinstance(streams, dict) or set(streams) != {"a", "b", "c"}:
        raise ContractError("formal report lacks the A/B/C annotation stream contract")
    for side in ("a", "b"):
        verify_file_binding(streams.get(side), deep=True)
    c_binding = streams.get("c")
    c_required = int(review.get("required_reviewer_c_decisions", -1)) > 0
    if c_required:
        verify_file_binding(c_binding, deep=True)
    elif c_binding is not None:
        raise ContractError("unexpected reviewer C stream when no adjudication is required")
    if (
        review.get("a_complete") != builder.NEW_TARGET
        or review.get("b_complete") != builder.NEW_TARGET
        or review.get("resolved_count") != EXPECTED_RECORDS
        or review.get("resolved_correct_count") != EXPECTED_RECORDS
        or review.get("unresolved_count") != 0
    ):
        raise ContractError("FORMAL_PASS report does not resolve forty positive scenes")
    manifest = read_json(pack / "manifest.json")
    if (
        report.get("manifest_sha256") != file_sha256(pack / "manifest.json")
        or report.get("manifest_payload_sha256")
        != (manifest.get("integrity") or {}).get("manifest_payload_sha256")
    ):
        raise ContractError("formal report/pack manifest binding drift")
    return report


def _relative_pack_file(pack: Path, binding: Any, label: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
        raise ContractError(f"{label}: relative file binding schema drift")
    relative = binding.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ContractError(f"{label}: expected a relative pack path")
    path = (pack / relative).resolve()
    try:
        path.relative_to(pack)
    except ValueError as exc:
        raise ContractError(f"{label}: path escapes pack") from exc
    expected = {"path": str(path), "bytes": binding["bytes"], "sha256": binding["sha256"]}
    verify_file_binding(expected, deep=True)
    return path


def _bbox(value: Any, context: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ContractError(f"{context}: bbox must contain four integers")
    box = tuple(value)
    if any(item < 0 or item > 1000 for item in box) or box[0] >= box[2] or box[1] >= box[3]:
        raise ContractError(f"{context}: invalid normalized bbox")
    return box  # type: ignore[return-value]


def _pair_from_label(label: dict[str, Any], *, verify_image: bool) -> PairExample:
    sample_id = label.get("sample_id")
    scene_id = label.get("scene_id")
    image_sha = label.get("image_sha256")
    image = Path(str(label.get("image_path", ""))).expanduser().resolve()
    if (
        not isinstance(sample_id, str)
        or not isinstance(scene_id, str)
        or not _SCENE_ID.fullmatch(scene_id)
        or not isinstance(image_sha, str)
        or not _SHA256.fullmatch(image_sha)
        or not image.is_file()
    ):
        raise ContractError("v5.2 label identity or source image is invalid")
    if verify_image and file_sha256(image) != image_sha:
        raise ContractError(f"source image hash drift: {sample_id}")
    targets = label.get("audit_targets") or {}
    grounding = targets.get("grounding") if isinstance(targets, dict) else None
    spatial = targets.get("spatial") if isinstance(targets, dict) else None
    public = label.get("public_interface") or {}
    public_targets = public.get("grounding_targets") if isinstance(public, dict) else None
    if (
        not isinstance(grounding, list)
        or len(grounding) != 2
        or not isinstance(public_targets, list)
        or len(public_targets) != 2
        or not isinstance(spatial, dict)
    ):
        raise ContractError(f"v5.2 label target schema drift: {sample_id}")
    entities: list[Entity] = []
    for index, slot in enumerate(("A", "B")):
        private = grounding[index]
        projected = public_targets[index]
        if (
            not isinstance(private, dict)
            or not isinstance(projected, dict)
            or private.get("slot") != slot
            or projected.get("slot") != slot
            or private.get("category") != projected.get("category")
            or private.get("referring_expression") != projected.get("referring_expression")
            or projected.get("visible_instance_count") != 1
        ):
            raise ContractError(f"v5.2 public/private target drift: {sample_id}/{slot}")
        entities.append(
            Entity(
                category=str(projected["category"]),
                referring_expression=str(projected["referring_expression"]),
                bbox=_bbox(private.get("bbox"), f"{sample_id}/{slot}"),
            )
        )
    nearer = spatial.get("nearer_slot")
    if nearer not in {"A", "B"}:
        raise ContractError(f"v5.2 nearer slot drift: {sample_id}")
    if entities[0].bbox == entities[1].bbox:
        raise ContractError(f"v5.2 pair cannot support deterministic slot-swap corruption: {sample_id}")
    eval_id = "gsa-sdepth-v52-" + canonical_hash(
        [EVALUATION_PROTOCOL, sample_id, image_sha]
    )[:20]
    return PairExample(
        eval_id=eval_id,
        sample_id=sample_id,
        scene_id=scene_id,
        source_image=image,
        image_sha256=image_sha,
        entity_a=entities[0],
        entity_b=entities[1],
        nearer_choice=nearer,
        label=label,
    )


def load_formal_source(
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_FORMAL_REPORT,
    *,
    verify_images: bool = True,
) -> FormalSource:
    pack = pack.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    report = validate_formal_report(pack, report_path)
    manifest_path = pack / "manifest.json"
    manifest = read_json(manifest_path)
    provenance = manifest.get("provenance") or {}
    labels_path = _relative_pack_file(pack, provenance.get("all_labels"), "all_labels")
    wrappers_path = _relative_pack_file(pack, provenance.get("source_records"), "source_records")
    labels = read_jsonl(labels_path)
    wrappers = read_jsonl(wrappers_path)
    if len(labels) != EXPECTED_RECORDS or len(wrappers) != EXPECTED_RECORDS:
        raise ContractError("v5.2 formal source must contain exactly forty rows")
    label_by_id = {row.get("sample_id"): row for row in labels}
    wrapper_by_id = {row.get("sample_id"): row for row in wrappers}
    if (
        None in label_by_id
        or None in wrapper_by_id
        or len(label_by_id) != EXPECTED_RECORDS
        or len(wrapper_by_id) != EXPECTED_RECORDS
        or set(label_by_id) != set(wrapper_by_id)
    ):
        raise ContractError("v5.2 label/source identities are missing or duplicated")
    selected = manifest.get("selected")
    if not isinstance(selected, list) or len(selected) != EXPECTED_RECORDS:
        raise ContractError("v5.2 selected manifest is not exact40")
    selected_ids = [item.get("sample_id") for item in selected if isinstance(item, dict)]
    if len(selected_ids) != EXPECTED_RECORDS or set(selected_ids) != set(label_by_id):
        raise ContractError("v5.2 selected/label identity mismatch")
    pairs: list[PairExample] = []
    for sample_id in selected_ids:
        label = label_by_id[sample_id]
        wrapper = wrapper_by_id[sample_id]
        source_record = wrapper.get("source_record")
        if (
            not isinstance(source_record, dict)
            or wrapper.get("source_record_sha256") != canonical_hash(source_record)
            or label.get("source_record_sha256") != canonical_hash(source_record)
            or label.get("source_sample_id") != wrapper.get("source_sample_id")
        ):
            raise ContractError(f"v5.2 source wrapper drift: {sample_id}")
        pairs.append(_pair_from_label(label, verify_image=verify_images))
    if (
        len({pair.eval_id for pair in pairs}) != EXPECTED_RECORDS
        or len({pair.sample_id for pair in pairs}) != EXPECTED_RECORDS
        or len({pair.scene_id for pair in pairs}) != EXPECTED_RECORDS
        or len({pair.image_sha256 for pair in pairs}) != EXPECTED_RECORDS
        or Counter(pair.nearer_choice for pair in pairs) != Counter(EXPECTED_LABEL_BALANCE)
    ):
        raise ContractError("v5.2 exact40 scene/image identity or A/B balance drift")
    streams = report["annotation_streams"]
    stream_hashes = {
        side: (binding.get("sha256") if isinstance(binding, dict) else None)
        for side, binding in streams.items()
    }
    binding: dict[str, Any] = {
        "release": "scannet_rgb_depth_v52_unique_referent_formal40",
        "pack_manifest": file_binding(manifest_path),
        "pack_manifest_payload_sha256": report["manifest_payload_sha256"],
        "selected_sha256": (manifest.get("integrity") or {}).get("selected_sha256"),
        "live_checker_report": file_binding(report_path),
        "live_checker_report_identity_sha256": report["identity_sha256"],
        "live_checker_status": checker.FORMAL_PASS,
        "annotation_streams": streams,
        "annotation_stream_sha256": stream_hashes,
        "reviewer_keys": report["review"]["reviewer_keys"],
        "labels": file_binding(labels_path),
        "source_wrappers": file_binding(wrappers_path),
        "records": EXPECTED_RECORDS,
        "unique_scenes": EXPECTED_RECORDS,
        "unique_images": EXPECTED_RECORDS,
        "official_split": "val",
        "training_materialization_eligible": False,
    }
    binding["identity_sha256"] = canonical_hash(binding)
    return FormalSource(tuple(pairs), binding)


def parse_grounding_response(value: Any) -> GroundingParse:
    parsed = interface.parse_grounding_response(value)
    boxes = parsed.boxes or {}
    return GroundingParse(boxes.get("A"), boxes.get("B"), parsed.status)


def parse_s_choice(value: Any) -> ChoiceParse:
    parsed = interface.parse_spatial_response(value)
    return ChoiceParse(parsed.choice, parsed.status)


def _pair_projection(pair: PairExample, *, boxes: dict[str, tuple[int, int, int, int]] | None) -> dict[str, Any]:
    projected: dict[str, Any] = {
        "A": {
            "category": pair.entity_a.category,
            "referring_expression": pair.entity_a.referring_expression,
        },
        "B": {
            "category": pair.entity_b.category,
            "referring_expression": pair.entity_b.referring_expression,
        },
        "bbox_coordinate_space": "image_normalized_0_1000_xyxy",
    }
    if boxes is not None:
        for slot in ("A", "B"):
            projected[slot]["bbox_2d"] = list(boxes[slot])
    return projected


def format_g_request(pair: PairExample, public_image: Path) -> dict[str, Any]:
    projected = interface.format_grounding_request(pair.label, public_image)
    row = {
        "eval_id": pair.eval_id,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "track": TRACK_G,
        "messages": [{"role": "user", "content": projected["prompt"]}],
        "images": [str(public_image.expanduser().resolve())],
        "image_sha256": pair.image_sha256,
        "pair": _pair_projection(pair, boxes=None),
    }
    validate_public_request(row, expected_track=TRACK_G)
    return row


def format_s_request(
    pair: PairExample,
    public_image: Path,
    *,
    track: str,
    bbox_a: tuple[int, int, int, int] | None = None,
    bbox_b: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    context_by_track = {
        TRACK_S_ORACLE: "oracle_grounding",
        TRACK_S_PRED: "predicted_grounding",
        TRACK_S_ABSENT: "absent_grounding",
        TRACK_S_CORRUPTED: "corrupted_grounding",
    }
    if track not in context_by_track:
        raise ContractError(f"invalid S track: {track}")
    context = context_by_track[track]
    if track == TRACK_S_ABSENT:
        if bbox_a is not None or bbox_b is not None:
            raise ContractError("S-absent must not receive grounding boxes")
        boxes = None
    else:
        if bbox_a is None or bbox_b is None:
            raise ContractError(f"{track}: both grounding boxes are required")
        boxes = {"A": bbox_a, "B": bbox_b}
    projected = interface.format_spatial_request(
        pair.label, public_image, boxes, context=context
    )
    row = {
        "eval_id": pair.eval_id,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "track": track,
        "messages": [{"role": "user", "content": projected["prompt"]}],
        "images": [str(public_image.expanduser().resolve())],
        "image_sha256": pair.image_sha256,
        "pair": _pair_projection(pair, boxes=boxes),
    }
    validate_public_request(row, expected_track=track)
    return row


def format_corrupted_s_request(pair: PairExample, public_image: Path) -> dict[str, Any]:
    """Apply the preregistered unconditional A/B oracle-box assignment swap."""
    load_intervention_contract()
    return format_s_request(
        pair,
        public_image,
        track=TRACK_S_CORRUPTED,
        bbox_a=pair.entity_b.bbox,
        bbox_b=pair.entity_a.bbox,
    )


def format_predicted_s_request(
    g_request: dict[str, Any], grounding: GroundingParse
) -> dict[str, Any]:
    if not grounding.valid:
        raise ContractError("cannot emit S-pred request for invalid G")
    pair = g_request.get("pair") or {}
    pseudo_label = {
        "sample_id": g_request["eval_id"],
        "image_sha256": g_request["image_sha256"],
        "public_interface": {
            "grounding_targets": [
                {"slot": slot, **pair[slot]} for slot in ("A", "B")
            ]
        },
    }
    boxes = {"A": grounding.bbox_a, "B": grounding.bbox_b}
    projected = interface.format_spatial_request(
        pseudo_label,
        Path(g_request["images"][0]),
        boxes,  # type: ignore[arg-type]
        context="predicted_grounding",
    )
    row = {
        "eval_id": g_request["eval_id"],
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "track": TRACK_S_PRED,
        "messages": [{"role": "user", "content": projected["prompt"]}],
        "images": list(g_request["images"]),
        "image_sha256": g_request["image_sha256"],
        "pair": {
            "A": {**pair["A"], "bbox_2d": list(grounding.bbox_a or ())},
            "B": {**pair["B"], "bbox_2d": list(grounding.bbox_b or ())},
            "bbox_coordinate_space": "image_normalized_0_1000_xyxy",
        },
    }
    validate_public_request(row, expected_track=TRACK_S_PRED)
    return row


def validate_public_request(row: dict[str, Any], *, expected_track: str | None = None) -> None:
    required = {
        "eval_id", "evaluation_protocol", "track", "messages", "images",
        "image_sha256", "pair",
    }
    if set(row) != required:
        raise ContractError("public request schema drift")
    eval_id = row.get("eval_id")
    track = row.get("track")
    if not isinstance(eval_id, str) or not _EVAL_ID.fullmatch(eval_id):
        raise ContractError("public request has invalid blinded eval_id")
    if row.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        raise ContractError(f"{eval_id}: evaluation protocol drift")
    if track not in set(ALL_TRACKS) or (
        expected_track is not None and track != expected_track
    ):
        raise ContractError(f"{eval_id}: request track drift")
    messages = row.get("messages")
    images = row.get("images")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or set(messages[0]) != {"role", "content"}
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
        or not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
    ):
        raise ContractError(f"{eval_id}: public model-input schema drift")
    image = Path(images[0]).expanduser().resolve()
    if image.name != f"{eval_id}.png" or image.parent.name != "images":
        raise ContractError(f"{eval_id}: public image path is not blinded")
    if not isinstance(row.get("image_sha256"), str) or not _SHA256.fullmatch(row["image_sha256"]):
        raise ContractError(f"{eval_id}: image hash drift")
    pair = row.get("pair")
    if not isinstance(pair, dict) or set(pair) != {"A", "B", "bbox_coordinate_space"}:
        raise ContractError(f"{eval_id}: public pair schema drift")
    for slot in ("A", "B"):
        item = pair.get(slot)
        expected = {"category", "referring_expression"}
        if track not in {TRACK_G, TRACK_S_ABSENT}:
            expected.add("bbox_2d")
        if not isinstance(item, dict) or set(item) != expected:
            raise ContractError(f"{eval_id}: public pair slot drift")
        if track not in {TRACK_G, TRACK_S_ABSENT}:
            _bbox(item.get("bbox_2d"), f"{eval_id}/{slot}")
    public_text = canonical_json({"messages": messages, "pair": pair}).casefold()
    if _SCENE_ID.search(public_text) or any(value in public_text for value in _FORBIDDEN_PUBLIC_TEXT):
        raise ContractError(f"{eval_id}: private source identity or truth leaked")


def _copy_blinded_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1 << 20)


def build_request_release(
    *,
    pack: Path = DEFAULT_PACK,
    report_path: Path = DEFAULT_FORMAL_REPORT,
    out: Path = DEFAULT_RELEASE,
) -> dict[str, Any]:
    out = out.expanduser().resolve()
    if out.exists():
        verified = validate_request_release(out / "release.manifest.json", verify_images=True)
        return {**verified.manifest, "build_status": "reused_verified"}
    intervention_contract = load_intervention_contract()
    source = load_formal_source(pack, report_path, verify_images=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        images_dir = temporary / "images"
        images_dir.mkdir()
        image_bindings: list[dict[str, Any]] = []
        actual_image_paths: dict[str, Path] = {}
        final_image_paths: dict[str, Path] = {}
        for pair in source.pairs:
            actual = images_dir / f"{pair.eval_id}.png"
            final = out / "images" / actual.name
            _copy_blinded_image(pair.source_image, actual)
            if file_sha256(actual) != pair.image_sha256:
                raise ContractError(f"blinded image copy drift: {pair.eval_id}")
            actual_image_paths[pair.eval_id] = actual
            final_image_paths[pair.eval_id] = final
            image_bindings.append({"eval_id": pair.eval_id, **_future_binding(actual, final)})
        g_rows: list[dict[str, Any]] = []
        oracle_rows: list[dict[str, Any]] = []
        absent_rows: list[dict[str, Any]] = []
        corrupted_rows: list[dict[str, Any]] = []
        for pair in source.pairs:
            g_row = format_g_request(pair, actual_image_paths[pair.eval_id])
            oracle_row = format_s_request(
                pair,
                actual_image_paths[pair.eval_id],
                track=TRACK_S_ORACLE,
                bbox_a=pair.entity_a.bbox,
                bbox_b=pair.entity_b.bbox,
            )
            absent_row = format_s_request(
                pair,
                actual_image_paths[pair.eval_id],
                track=TRACK_S_ABSENT,
            )
            corrupted_row = format_corrupted_s_request(
                pair, actual_image_paths[pair.eval_id]
            )
            # The request is built against an existing temporary copy, then
            # rebound to the identical final blinded path recorded below.
            for row in (g_row, oracle_row, absent_row, corrupted_row):
                row["images"] = [str(final_image_paths[pair.eval_id])]
            validate_public_request(g_row, expected_track=TRACK_G)
            validate_public_request(oracle_row, expected_track=TRACK_S_ORACLE)
            validate_public_request(absent_row, expected_track=TRACK_S_ABSENT)
            validate_public_request(corrupted_row, expected_track=TRACK_S_CORRUPTED)
            g_rows.append(g_row)
            oracle_rows.append(oracle_row)
            absent_rows.append(absent_row)
            corrupted_rows.append(corrupted_row)
        actual_g = temporary / "requests.g.jsonl"
        actual_oracle = temporary / "requests.s_oracle.jsonl"
        actual_absent = temporary / "requests.s_absent.jsonl"
        actual_corrupted = temporary / "requests.s_corrupted.jsonl"
        write_jsonl(actual_g, g_rows)
        write_jsonl(actual_oracle, oracle_rows)
        write_jsonl(actual_absent, absent_rows)
        write_jsonl(actual_corrupted, corrupted_rows)
        final_g = out / actual_g.name
        final_oracle = out / actual_oracle.name
        final_absent = out / actual_absent.name
        final_corrupted = out / actual_corrupted.name
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "protocol": RELEASE_PROTOCOL,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "release": "formal40",
            "formal_source": source.binding,
            "requests": {
                TRACK_G: {
                    "file": _future_binding(actual_g, final_g),
                    "records_sha256": canonical_hash(g_rows),
                    "records": EXPECTED_RECORDS,
                },
                TRACK_S_ORACLE: {
                    "file": _future_binding(actual_oracle, final_oracle),
                    "records_sha256": canonical_hash(oracle_rows),
                    "records": EXPECTED_RECORDS,
                },
                TRACK_S_ABSENT: {
                    "file": _future_binding(actual_absent, final_absent),
                    "records_sha256": canonical_hash(absent_rows),
                    "records": EXPECTED_RECORDS,
                },
                TRACK_S_CORRUPTED: {
                    "file": _future_binding(actual_corrupted, final_corrupted),
                    "records_sha256": canonical_hash(corrupted_rows),
                    "records": EXPECTED_RECORDS,
                },
            },
            "public_images": image_bindings,
            "blinding": {
                "model_fields": ["messages", "images"],
                "private_fields_passed_to_model": [],
                "oracle_matching_of_predicted_boxes": False,
                "invalid_g_emits_s_pred_request": False,
            },
            "implementation": {
                "evaluator": file_binding(Path(__file__)),
                "public_interface": file_binding(Path(interface.__file__)),
                "intervention_contract": file_binding(INTERVENTION_CONTRACT_PATH),
                "intervention_contract_protocol": intervention_contract["protocol"],
            },
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
    verified = validate_request_release(out / "release.manifest.json", verify_images=True)
    return {**verified.manifest, "build_status": "generated"}


def validate_request_release(
    manifest_path: Path, *, verify_images: bool = True
) -> RequestRelease:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = read_json(manifest_path)
    identity = manifest.get("identity_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    if identity != canonical_hash(unsigned):
        raise ContractError("request release identity drift")
    intervention_contract = load_intervention_contract()
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol") != RELEASE_PROTOCOL
        or manifest.get("evaluation_protocol") != EVALUATION_PROTOCOL
        or manifest.get("release") != "formal40"
        or (manifest.get("training_materialization") or {}).get("eligible") is not False
        or manifest.get("implementation")
        != {
            "evaluator": file_binding(Path(__file__)),
            "public_interface": file_binding(Path(interface.__file__)),
            "intervention_contract": file_binding(INTERVENTION_CONTRACT_PATH),
            "intervention_contract_protocol": intervention_contract["protocol"],
        }
    ):
        raise ContractError("request release contract or implementation drift")
    formal = manifest.get("formal_source")
    if not isinstance(formal, dict):
        raise ContractError("request release lacks formal source binding")
    pack_manifest = verify_file_binding(formal.get("pack_manifest"), deep=True)
    report_path = verify_file_binding(formal.get("live_checker_report"), deep=True)
    source = load_formal_source(pack_manifest.parent, report_path, verify_images=verify_images)
    if source.binding != formal:
        raise ContractError("request release formal source binding drift")
    blinding = manifest.get("blinding") or {}
    if blinding != {
        "model_fields": ["messages", "images"],
        "private_fields_passed_to_model": [],
        "oracle_matching_of_predicted_boxes": False,
        "invalid_g_emits_s_pred_request": False,
    }:
        raise ContractError("request release blinding contract drift")
    public_images = manifest.get("public_images")
    if not isinstance(public_images, list) or len(public_images) != EXPECTED_RECORDS:
        raise ContractError("public image release is not exact40")
    images_by_id: dict[str, dict[str, Any]] = {}
    pairs_by_id = {pair.eval_id: pair for pair in source.pairs}
    for item in public_images:
        if not isinstance(item, dict) or set(item) != {"eval_id", "path", "bytes", "sha256"}:
            raise ContractError("public image binding schema drift")
        eval_id = item["eval_id"]
        path = verify_file_binding(
            {key: item[key] for key in ("path", "bytes", "sha256")}, deep=verify_images
        )
        if (
            eval_id in images_by_id
            or eval_id not in pairs_by_id
            or path.name != f"{eval_id}.png"
            or path.parent.name != "images"
            or item["sha256"] != pairs_by_id[eval_id].image_sha256
        ):
            raise ContractError("public image identity or blinding drift")
        images_by_id[eval_id] = item
    if set(images_by_id) != set(pairs_by_id):
        raise ContractError("public images do not cover exact40")
    request_bindings = manifest.get("requests")
    if not isinstance(request_bindings, dict) or set(request_bindings) != set(STATIC_TRACKS):
        raise ContractError("static request track coverage drift")
    rows_by_track: dict[str, list[dict[str, Any]]] = {}
    for track in STATIC_TRACKS:
        item = request_bindings[track]
        if not isinstance(item, dict) or set(item) != {"file", "records_sha256", "records"}:
            raise ContractError(f"{track}: request binding schema drift")
        request_path = verify_file_binding(item["file"], deep=True)
        rows = read_jsonl(request_path)
        if item["records"] != EXPECTED_RECORDS or canonical_hash(rows) != item["records_sha256"]:
            raise ContractError(f"{track}: request count or hash drift")
        expected: list[dict[str, Any]] = []
        for pair in source.pairs:
            image = Path(images_by_id[pair.eval_id]["path"])
            if track == TRACK_G:
                row = format_g_request(pair, image)
            elif track == TRACK_S_ORACLE:
                row = format_s_request(
                    pair,
                    image,
                    track=track,
                    bbox_a=pair.entity_a.bbox,
                    bbox_b=pair.entity_b.bbox,
                )
            elif track == TRACK_S_ABSENT:
                row = format_s_request(pair, image, track=track)
            else:
                row = format_corrupted_s_request(pair, image)
            expected.append(row)
        if rows != expected:
            raise ContractError(f"{track}: public requests do not derive exactly from formal source")
        rows_by_track[track] = rows
    return RequestRelease(manifest, source, rows_by_track)


def iou(box_a: tuple[int, ...], box_b: tuple[int, ...]) -> float:
    left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return round(float(numerator) / denominator, 6) if denominator else None


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def scene_bootstrap_ci(
    values: list[int], *, seed: int = BOOTSTRAP_SEED, samples: int = BOOTSTRAP_SAMPLES
) -> dict[str, Any]:
    if not values:
        raise ContractError("scene bootstrap requires at least one scene")
    rng = random.Random(seed)
    n = len(values)
    estimates = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return {
        "unit": "scene",
        "paired_scenes": n,
        "samples": samples,
        "seed": seed,
        "confidence": 0.95,
        "ci": [round(_percentile(estimates, 0.025), 6), round(_percentile(estimates, 0.975), 6)],
    }


def _key_predictions(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        eval_id = row.get("eval_id")
        if not isinstance(eval_id, str) or eval_id in keyed:
            raise ContractError(f"{label}: missing or duplicate eval_id")
        if not isinstance(row.get("response"), str) or not isinstance(row.get("hit_token_cap"), bool):
            raise ContractError(f"{label}/{eval_id}: response/cap metadata drift")
        keyed[eval_id] = row
    return keyed


def score_paired_predictions(
    pairs: Iterable[PairExample],
    g_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    absent_rows: list[dict[str, Any]],
    corrupted_rows: list[dict[str, Any]],
    predicted_rows: list[dict[str, Any]],
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pair_list = list(pairs)
    if not pair_list or len({pair.scene_id for pair in pair_list}) != len(pair_list):
        raise ContractError("paired score requires one unique record per scene")
    expected = {pair.eval_id for pair in pair_list}
    g = _key_predictions(g_rows, "G predictions")
    oracle = _key_predictions(oracle_rows, "S-oracle predictions")
    absent = _key_predictions(absent_rows, "S-absent predictions")
    corrupted = _key_predictions(corrupted_rows, "S-corrupted predictions")
    predicted = _key_predictions(predicted_rows, "S-pred predictions")
    if any(set(rows) != expected for rows in (g, oracle, absent, corrupted)):
        raise ContractError("G and all static S interventions must cover all formal scenes")
    parsed_g = {
        eval_id: (
            GroundingParse(None, None, "hit_token_cap")
            if row["hit_token_cap"]
            else parse_grounding_response(row["response"])
        )
        for eval_id, row in g.items()
    }
    valid_g_ids = {eval_id for eval_id, parsed in parsed_g.items() if parsed.valid}
    if set(predicted) != valid_g_ids:
        raise ContractError("S-pred must cover exactly valid, uncapped G outputs")

    parse_status = {
        "G": Counter(),
        "S_oracle": Counter(),
        "S_absent": Counter(),
        "S_corrupted": Counter(),
        "S_pred": Counter(),
    }
    cap_counts = {key: 0 for key in parse_status}
    transitions: list[dict[str, Any]] = []
    g_valid = g_success = oracle_correct = oracle_parsed = 0
    absent_correct = absent_parsed = corrupted_correct = corrupted_parsed = 0
    pred_correct_emitted = pred_parsed = strict_e2e = 0
    min_ious: list[float] = []
    dependency_deltas: list[int] = []
    absent_deltas: list[int] = []
    corrupted_deltas: list[int] = []
    attribution_counts: Counter[str] = Counter()
    attribution_reasons: Counter[str] = Counter()

    def parse_choice_row(row: dict[str, Any]) -> ChoiceParse:
        return (
            ChoiceParse(None, "hit_token_cap")
            if row["hit_token_cap"]
            else parse_s_choice(row["response"])
        )

    for pair in pair_list:
        grounding = parsed_g[pair.eval_id]
        cap_counts["G"] += int(g[pair.eval_id]["hit_token_cap"])
        parse_status["G"][grounding.status] += 1
        iou_a = iou_b = None
        both_box = False
        if grounding.valid:
            g_valid += 1
            iou_a = iou(grounding.bbox_a or (0, 0, 1, 1), pair.entity_a.bbox)
            iou_b = iou(grounding.bbox_b or (0, 0, 1, 1), pair.entity_b.bbox)
            both_box = iou_a >= 0.5 and iou_b >= 0.5
            g_success += int(both_box)
            min_ious.append(min(iou_a, iou_b))
        else:
            min_ious.append(0.0)

        oracle_row = oracle[pair.eval_id]
        cap_counts["S_oracle"] += int(oracle_row["hit_token_cap"])
        oracle_choice = parse_choice_row(oracle_row)
        parse_status["S_oracle"][oracle_choice.status] += 1
        oracle_ok = oracle_choice.choice == pair.nearer_choice
        oracle_parsed += int(oracle_choice.valid)
        oracle_correct += int(oracle_ok)

        absent_row = absent[pair.eval_id]
        cap_counts["S_absent"] += int(absent_row["hit_token_cap"])
        absent_choice = parse_choice_row(absent_row)
        parse_status["S_absent"][absent_choice.status] += 1
        absent_ok = absent_choice.choice == pair.nearer_choice
        absent_parsed += int(absent_choice.valid)
        absent_correct += int(absent_ok)

        corrupted_row = corrupted[pair.eval_id]
        cap_counts["S_corrupted"] += int(corrupted_row["hit_token_cap"])
        corrupted_choice = parse_choice_row(corrupted_row)
        parse_status["S_corrupted"][corrupted_choice.status] += 1
        corrupted_ok = corrupted_choice.choice == pair.nearer_choice
        corrupted_parsed += int(corrupted_choice.valid)
        corrupted_correct += int(corrupted_ok)

        pred_choice = ChoiceParse(None, "no_request_due_to_invalid_g")
        pred_ok = False
        if grounding.valid:
            pred_row = predicted[pair.eval_id]
            cap_counts["S_pred"] += int(pred_row["hit_token_cap"])
            pred_choice = parse_choice_row(pred_row)
            pred_ok = pred_choice.choice == pair.nearer_choice
            pred_correct_emitted += int(pred_ok)
            pred_parsed += int(pred_choice.valid)
        parse_status["S_pred"][pred_choice.status] += 1
        e2e_ok = both_box and pred_ok
        strict_e2e += int(e2e_ok)
        dependency_delta = int(oracle_ok) - int(pred_ok)
        dependency_deltas.append(dependency_delta)
        absent_deltas.append(int(oracle_ok) - int(absent_ok))
        corrupted_deltas.append(int(oracle_ok) - int(corrupted_ok))
        predicted_transition = (
            "oracle_correct_to_predicted_correct" if oracle_ok and pred_ok
            else "oracle_correct_to_predicted_wrong" if oracle_ok
            else "oracle_wrong_to_predicted_correct" if pred_ok
            else "oracle_wrong_to_predicted_wrong"
        )
        strict_e2e_transition = (
            "oracle_correct_to_e2e_correct" if oracle_ok and e2e_ok
            else "oracle_correct_to_e2e_wrong" if oracle_ok
            else "oracle_wrong_to_e2e_correct" if e2e_ok
            else "oracle_wrong_to_e2e_wrong"
        )
        health = {
            "G": grounding.valid,
            "S_oracle": oracle_choice.valid,
            "S_absent": absent_choice.valid,
            "S_corrupted": corrupted_choice.valid,
            "S_pred": pred_choice.valid,
        }
        if not all(health.values()):
            attribution_state = "Inconclusive"
            attribution_reason = "invalid_or_capped_or_unparsed_output"
        elif pred_ok:
            attribution_state = "Inconclusive"
            attribution_reason = (
                "oracle_degradation_predicted_context_correct"
                if not oracle_ok
                else "no_downstream_failure"
            )
        elif not oracle_ok:
            if absent_ok or corrupted_ok:
                attribution_state = "Inconclusive"
                attribution_reason = "nonmonotonic_control_improvement"
            elif both_box:
                attribution_state = "Local(S)"
                attribution_reason = "correct_G_but_S_fails_under_oracle_context"
            else:
                attribution_state = "Inconclusive"
                attribution_reason = "multiple_unresolved_faults_context_insensitive"
        elif absent_ok or corrupted_ok:
            attribution_state = "Inconclusive"
            attribution_reason = "context_insensitive_or_control_failure"
        elif not both_box:
            attribution_state = "Attributed(G)"
            attribution_reason = "failed_G_and_oracle_repairs_S"
        else:
            attribution_state = "Inconclusive"
            attribution_reason = "oracle_repairs_S_but_G_passes_threshold"
        attribution_counts[attribution_state] += 1
        attribution_reasons[attribution_reason] += 1

        transitions.append(
            {
                "eval_id": pair.eval_id,
                "scene_id": pair.scene_id,
                "g_parse_status": grounding.status,
                "g_parse_valid": grounding.valid,
                "g_iou_a": round(iou_a, 6) if iou_a is not None else None,
                "g_iou_b": round(iou_b, 6) if iou_b is not None else None,
                "g_both_box_iou_at_0_5": both_box,
                "s_oracle_parse_status": oracle_choice.status,
                "s_oracle_correct": oracle_ok,
                "s_absent_parse_status": absent_choice.status,
                "s_absent_correct": absent_ok,
                "s_corrupted_parse_status": corrupted_choice.status,
                "s_corrupted_correct": corrupted_ok,
                "s_pred_request_emitted": grounding.valid,
                "s_pred_parse_status": pred_choice.status,
                "s_pred_correct": pred_ok,
                "s_pred_e2e_correct": e2e_ok,
                "paired_dependency_delta": dependency_delta,
                "predicted_context_transition": predicted_transition,
                "strict_e2e_transition": strict_e2e_transition,
                "attribution_state": attribution_state,
                "attribution_reason": attribution_reason,
            }
        )
    n = len(pair_list)
    oracle_accuracy = float(oracle_correct) / n
    e2e_accuracy = float(strict_e2e) / n
    predicted_all_scene_accuracy = float(pred_correct_emitted) / n
    signed_gap = oracle_accuracy - predicted_all_scene_accuracy
    positive_headroom = max(0.0, signed_gap)
    bounded_recovery = (
        min(1.0, max(0.0, predicted_all_scene_accuracy / oracle_accuracy))
        if oracle_accuracy
        else None
    )
    absent_accuracy = float(absent_correct) / n
    corrupted_accuracy = float(corrupted_correct) / n
    report: dict[str, Any] = {
        "result_schema_version": 1,
        "protocol": RESULT_PROTOCOL,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "release": "formal40",
        "status": "FORMAL_SCORE_COMPLETE" if provenance is not None else "SYNTHETIC_SCORE_COMPLETE",
        "task_contract": {
            "stages": list(ALL_TRACKS),
            "same_model_and_deterministic_decoding_required": True,
            "strict_grounding_parser": interface.INTERFACE_PROTOCOL,
            "oracle_matching_of_predicted_boxes": False,
            "invalid_or_capped_g_emits_s_pred_request": False,
            "all_primary_metrics_denominator": n,
            "primary_unit": "scene",
            "dependency_gap_endpoint": "S_pred_correct_all_scenes_invalid_g_zero",
            "strict_e2e_is_reported_separately": True,
            "training_materialization_eligible": False,
            "intervention_contract": file_binding(INTERVENTION_CONTRACT_PATH),
            "corrupted_context_rule": "unconditional_A_B_oracle_bbox_assignment_swap",
            "corrupted_context_is_per_target_non_gt": True,
            "corruption_reads_downstream_truth": False,
        },
        "metrics": {
            "n_scenes": n,
            "g_valid_pair_coverage": _ratio(g_valid, n),
            "g_both_box_iou_at_0_5": _ratio(g_success, n),
            "g_mean_min_pair_iou": round(sum(min_ious) / n, 6),
            "s_oracle_accuracy": round(oracle_accuracy, 6),
            "s_oracle_parse_coverage": _ratio(oracle_parsed, n),
            "s_absent_accuracy": round(absent_accuracy, 6),
            "s_absent_parse_coverage": _ratio(absent_parsed, n),
            "s_corrupted_accuracy": round(corrupted_accuracy, 6),
            "s_corrupted_parse_coverage": _ratio(corrupted_parsed, n),
            "s_pred_accuracy_given_emitted_request": _ratio(pred_correct_emitted, len(predicted)),
            "s_pred_parse_coverage_all_scenes": _ratio(pred_parsed, n),
            "s_pred_correct_all_scenes_invalid_g_zero": round(
                predicted_all_scene_accuracy, 6
            ),
            "s_pred_strict_e2e_accuracy": round(e2e_accuracy, 6),
            "paired_dependency_gap_oracle_minus_predicted": round(signed_gap, 6),
            "paired_context_gain_oracle_minus_absent": round(
                oracle_accuracy - absent_accuracy, 6
            ),
            "paired_context_gain_oracle_minus_corrupted": round(
                oracle_accuracy - corrupted_accuracy, 6
            ),
            "positive_oracle_headroom": round(positive_headroom, 6),
            "bounded_oracle_recovery": (
                round(bounded_recovery, 6) if bounded_recovery is not None else None
            ),
        },
        "metric_definitions": {
            "positive_oracle_headroom": "max(0, S_oracle_accuracy - S_pred_correct_all_scenes_invalid_g_zero)",
            "bounded_oracle_recovery": "clip(S_pred_correct_all_scenes_invalid_g_zero / S_oracle_accuracy, 0, 1)",
            "s_pred_strict_e2e_accuracy": "G both boxes IoU>=0.5 AND S-pred correct; all scenes remain in denominator",
            "paired_dependency_gap_oracle_minus_predicted": "S-oracle accuracy minus S-pred accuracy on all scenes; invalid or capped G receives zero S-pred credit",
            "paired_context_gain_oracle_minus_absent": "S-oracle accuracy minus same-item S accuracy with no boxes",
            "paired_context_gain_oracle_minus_corrupted": "S-oracle accuracy minus same-item S accuracy after unconditional A/B box-assignment swap",
        },
        "parse_status": {key: dict(sorted(value.items())) for key, value in parse_status.items()},
        "token_cap_counts": cap_counts,
        "scene_bootstrap": {
            "metric": "paired_dependency_gap_oracle_minus_predicted",
            **scene_bootstrap_ci(dependency_deltas),
        },
        "context_control_bootstrap": {
            "oracle_minus_absent": {
                "metric": "paired_context_gain_oracle_minus_absent",
                **scene_bootstrap_ci(absent_deltas),
            },
            "oracle_minus_corrupted": {
                "metric": "paired_context_gain_oracle_minus_corrupted",
                **scene_bootstrap_ci(corrupted_deltas),
            },
        },
        "transition_counts": dict(
            sorted(Counter(row["predicted_context_transition"] for row in transitions).items())
        ),
        "strict_e2e_transition_counts": dict(
            sorted(Counter(row["strict_e2e_transition"] for row in transitions).items())
        ),
        "attribution": {
            "states": list(ATTRIBUTION_STATES),
            "counts": {state: attribution_counts[state] for state in ATTRIBUTION_STATES},
            "reason_counts": dict(sorted(attribution_reasons.items())),
            "conclusive_failures": attribution_counts["Attributed(G)"]
            + attribution_counts["Local(S)"],
            "policy": file_binding(INTERVENTION_CONTRACT_PATH),
        },
        "per_item_transitions": transitions,
        "provenance": provenance or {"synthetic_fixture_only": True},
        "trust": {
            "score_available": provenance is not None,
            "all_static_scenes_complete": all(
                len(rows) == n for rows in (g, oracle, absent, corrupted)
            ),
            "s_pred_exact_valid_g_coverage": set(predicted) == valid_g_ids,
            "all_token_cap_metadata_complete": True,
            "held_out_evaluation_only": True,
        },
    }
    return report


def score_command(args: argparse.Namespace) -> int:
    release = validate_request_release(args.release_manifest, verify_images=True)
    if len(release.source.pairs) != EXPECTED_RECORDS:
        raise ContractError("formal scoring source is not exact40")
    from eval import scannet_rgb_depth_v52_dlc as dlc

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
        absent_bundle = dlc.verify_prediction_bundle(
            manifest_path=args.s_absent_manifest,
            binding_path=args.model_binding,
            predictions_path=args.s_absent_predictions,
            expected_track=TRACK_S_ABSENT,
            deep_model=False,
        )
        corrupted_bundle = dlc.verify_prediction_bundle(
            manifest_path=args.s_corrupted_manifest,
            binding_path=args.model_binding,
            predictions_path=args.s_corrupted_predictions,
            expected_track=TRACK_S_CORRUPTED,
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
    bundles = (g_bundle, oracle_bundle, absent_bundle, corrupted_bundle, predicted_bundle)
    identities = {bundle["provenance"]["model_binding_identity_sha256"] for bundle in bundles}
    receipt_ids = {bundle["receipt"]["identity_sha256"] for bundle in bundles}
    if identities != {binding["identity_sha256"]} or len(receipt_ids) != 1:
        raise ContractError("three stages do not share one model/code/runtime receipt")
    if predicted_request["manifest"]["g_prediction_manifest"] != file_binding(args.g_manifest):
        raise ContractError("S-pred requests are not bound to the scored G outputs")
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
            TRACK_S_ABSENT: absent_bundle["provenance"],
            TRACK_S_CORRUPTED: corrupted_bundle["provenance"],
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
        absent_bundle["rows"],
        corrupted_bundle["rows"],
        predicted_bundle["rows"],
        provenance=provenance,
    )
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def dry_run_command() -> int:
    pairs: list[PairExample] = []
    for index in range(4):
        image = Path(f"/synthetic/{index}.png")
        label = {
            "sample_id": f"synthetic-{index}",
            "image_sha256": f"{index + 1:064x}",
            "public_interface": {
                "grounding_targets": [
                    {"slot": "A", "category": "chair", "referring_expression": "the only visible chair"},
                    {"slot": "B", "category": "table", "referring_expression": "the only visible table"},
                ]
            },
        }
        pairs.append(
            PairExample(
                eval_id=f"gsa-sdepth-v52-{index:020x}",
                sample_id=f"synthetic-{index}",
                scene_id=f"scene{index:04d}_00",
                source_image=image,
                image_sha256=f"{index + 1:064x}",
                entity_a=Entity("chair", "the only visible chair", (100, 100, 300, 500)),
                entity_b=Entity("table", "the only visible table", (500, 100, 800, 600)),
                nearer_choice="A" if index % 2 == 0 else "B",
                label=label,
            )
        )
    good = '{"A":[100,100,300,500],"B":[500,100,800,600]}'
    g = [{"eval_id": pair.eval_id, "response": good, "hit_token_cap": False} for pair in pairs]
    oracle = [
        {"eval_id": pair.eval_id, "response": pair.nearer_choice, "hit_token_cap": False}
        for pair in pairs
    ]
    absent = [
        {"eval_id": pair.eval_id, "response": "B" if pair.nearer_choice == "A" else "A", "hit_token_cap": False}
        for pair in pairs
    ]
    corrupted = list(absent)
    result = score_paired_predictions(pairs, g, oracle, absent, corrupted, list(oracle))
    passed = (
        result["metrics"]["g_both_box_iou_at_0_5"] == 1.0
        and result["metrics"]["s_oracle_accuracy"] == 1.0
        and result["metrics"]["s_pred_strict_e2e_accuracy"] == 1.0
        and result["metrics"]["bounded_oracle_recovery"] == 1.0
    )
    payload = {
        "status": "DRY_RUN_PASS" if passed else "DRY_RUN_FAIL",
        "synthetic_fixture_only": True,
        "metrics": result["metrics"],
        "scene_bootstrap": result["scene_bootstrap"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    gate.add_argument("--report", type=Path, default=DEFAULT_FORMAL_REPORT)
    gate.add_argument("--adjudication", type=Path)
    build = sub.add_parser("build-release")
    build.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    build.add_argument("--report", type=Path, default=DEFAULT_FORMAL_REPORT)
    build.add_argument("--out", type=Path, default=DEFAULT_RELEASE)
    verify = sub.add_parser("verify-release")
    verify.add_argument("--release-manifest", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--release-manifest", type=Path, required=True)
    score.add_argument("--model-binding", type=Path, required=True)
    score.add_argument("--g-predictions", type=Path, required=True)
    score.add_argument("--g-manifest", type=Path, required=True)
    score.add_argument("--s-oracle-predictions", type=Path, required=True)
    score.add_argument("--s-oracle-manifest", type=Path, required=True)
    score.add_argument("--s-absent-predictions", type=Path, required=True)
    score.add_argument("--s-absent-manifest", type=Path, required=True)
    score.add_argument("--s-corrupted-predictions", type=Path, required=True)
    score.add_argument("--s-corrupted-manifest", type=Path, required=True)
    score.add_argument("--s-pred-request-manifest", type=Path, required=True)
    score.add_argument("--s-pred-predictions", type=Path, required=True)
    score.add_argument("--s-pred-manifest", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    sub.add_parser("dry-run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "gate":
            report = refresh_formal_report(args.pack, args.report, args.adjudication)
            if report.get("status") != checker.FORMAL_PASS:
                payload = {
                    "status": "BLOCKED_PENDING_HUMAN_REVIEW",
                    "checker_status": report.get("status"),
                    "report": str(args.report.expanduser().resolve()),
                    "review": report.get("review"),
                    "gpu_started": False,
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 20
            payload = {
                "status": "FORMAL_PASS",
                "report": file_binding(args.report),
                "annotation_stream_sha256": {
                    side: (binding.get("sha256") if isinstance(binding, dict) else None)
                    for side, binding in report["annotation_streams"].items()
                },
                "gpu_started": False,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "build-release":
            payload = build_request_release(pack=args.pack, report_path=args.report, out=args.out)
        elif args.command == "verify-release":
            payload = validate_request_release(args.release_manifest, verify_images=True).manifest
        elif args.command == "score":
            return score_command(args)
        else:
            return dry_run_command()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except FormalGateBlocked as exc:
        print(str(exc), file=sys.stderr)
        return 20
    except (ContractError, checker.CheckError, builder.PackError, interface.InterfaceError) as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
