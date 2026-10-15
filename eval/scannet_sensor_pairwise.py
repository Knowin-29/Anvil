#!/usr/bin/env python3
"""Format and score the held-out ScanNet sensor pairwise-depth task.

This evaluator is intentionally separate from the generic GSA structured-output
runner.  Every request identifies exactly two objects, A and B, by a public class
name and a 0--1000 bounding box.  The model predicts which supplied object is
nearer; it is never asked to discover or order every visible object.

The module does not run a model.  ``prepare`` writes blinded requests for any
inference engine, and ``score`` consumes keyed response JSONL.  That separation
keeps the sensor labels and internal ScanNet instance identities out of model
requests and makes missing predictions observable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gsa_eval.coords import iou, to_unit_box  # noqa: E402
from eval.gsa_eval.parsing import parse_json_items, strip_reasoning  # noqa: E402


SOURCE_PROTOCOL = "gsa-depth-v5.0-scannet-sensor-pairwise"
SOURCE_LABEL_VERSION = "gsa-label-v5.0-depth-scannet-sensor-pairwise"
EVALUATION_PROTOCOL = "gsa-depth-v5.0-oracle-pairwise-eval-v3"
RESULT_SCHEMA_VERSION = 1
MIN_FORMAL_RECORDS = 120
MIN_FORMAL_SCENES = 60
# Fixed before any model evaluation. This identity-only salt gives the pinned
# 120-record source-replayed full_v3 release an exact 60/60 A/B target balance.
# The v1/full_v2 and pre-source-replay v2 artifacts remain immutable history.
AB_PERMUTATION_SALT = "ab-balance-full-v3-source-replay-9"
FORMAL_SOURCE_RELEASE = "full_v3"
FORMAL_SOURCE_REPORT_SHA256 = (
    "2ab25e258c11fc022343bcdf7ed587a0f1d8f57ca385b4ebddf9c5964b3c13e4"
)
FORMAL_SOURCE_LABELS_SHA256 = (
    "ae91cb86af9042c12e2bd0b3ed797387b0cd19eea66a31f7b4055ae490201e59"
)
FORMAL_SOURCE_RECORDS_SHA256 = (
    "ab4de02044e59a32e86b24d3e8460a421849c466617e1c3f6be8abe1784a12c8"
)
FORMAL_LABEL_BALANCE = {"A": 60, "B": 60}
TRACK_S_ORACLE = "S_oracle_grounded_pair"
TRACK_G_TO_S = "G_to_S_predicted_grounding_cascade"
TRACK_ALIASES = {
    "s-oracle": TRACK_S_ORACLE,
    "s_oracle": TRACK_S_ORACLE,
    TRACK_S_ORACLE: TRACK_S_ORACLE,
    "g-to-s": TRACK_G_TO_S,
    "g_to_s": TRACK_G_TO_S,
    TRACK_G_TO_S: TRACK_G_TO_S,
}
_INSTANCE_SUFFIX = re.compile(r"\s*\[\s*instance\s+\d+\s*\]\s*$", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_REQUEST_KEYS = {
    "eval_id",
    "evaluation_protocol",
    "track",
    "messages",
    "images",
    "image_sha256",
    "pair",
}
_PUBLIC_PAIR_KEYS = {"A", "B", "bbox_coordinate_space"}
_PUBLIC_ENTITY_KEYS = {"name", "bbox_2d"}
_FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "depth_gap_m",
    "depth_m",
    "far_label",
    "instance_id",
    "internal_label",
    "median_depth_m",
    "near_label",
    "nearer_choice",
    "payload",
    "q1_depth_m",
    "q3_depth_m",
    "rank",
    "relative_depth",
    "sample_id",
    "sensor_measurements",
    "sensor_provenance",
    "source_identity",
    "source_sample_id",
    "target",
}


class ContractError(ValueError):
    """Raised when an input could make the benchmark identity or score ambiguous."""


@dataclass(frozen=True)
class Entity:
    internal_label: str
    public_name: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PairExample:
    eval_id: str
    source_identity: str
    image_path: str
    image_sha256: str
    entity_a: Entity
    entity_b: Entity
    nearer_choice: str
    gap_bucket: str
    official_split: str


@dataclass(frozen=True)
class ParsedChoice:
    choice: str | None
    status: str


@dataclass(frozen=True)
class GroundingRow:
    bbox_a: tuple[int, int, int, int]
    bbox_b: tuple[int, int, int, int]


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ContractError(f"JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(value)
    if not rows and not allow_empty:
        raise ContractError(f"JSONL has no records: {path}")
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _normalise_track(value: str) -> str:
    try:
        return TRACK_ALIASES[value]
    except KeyError as exc:
        raise ContractError(f"unknown track {value!r}; choose s-oracle or g-to-s") from exc


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


def _bbox(value: Any, *, context: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ContractError(f"{context}: bbox must contain four integer coordinates")
    box = tuple(value)
    if any(item < 0 or item > 1000 for item in box):
        raise ContractError(f"{context}: bbox is outside the declared 0-1000 space")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ContractError(f"{context}: bbox has non-positive area")
    return box  # type: ignore[return-value]


def _record_measurements(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("sensor_measurements")
    if value is None:
        provenance = record.get("sensor_provenance")
        value = provenance.get("measurements") if isinstance(provenance, dict) else None
    if not isinstance(value, list) or len(value) != 2 or not all(
        isinstance(item, dict) for item in value
    ):
        raise ContractError("record must contain exactly two sensor measurements")
    return value


def _source_identity(record: dict[str, Any]) -> str:
    # Audit projections receive a new sample_id.  Binding to source_sample_id keeps
    # the A/B permutation identical between the source and audited projections.
    value = record.get("source_sample_id") or record.get("sample_id")
    if not isinstance(value, str) or not value:
        raise ContractError("record lacks a stable sample identity")
    return value


def _visibility(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("visibility_features")
    if isinstance(value, dict):
        return value
    provenance = record.get("sensor_provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("visibility_features"), dict):
        return provenance["visibility_features"]
    return {}


def validate_source_record(record: dict[str, Any], *, verify_image: bool = True) -> None:
    identity = record.get("source_sample_id") or record.get("sample_id") or "<unknown>"
    prefix = f"source record {identity}"
    if record.get("protocol_version") != SOURCE_PROTOCOL:
        raise ContractError(f"{prefix}: protocol_version drift")
    if record.get("label_version") != SOURCE_LABEL_VERSION:
        raise ContractError(f"{prefix}: label_version drift")
    if record.get("official_split") != "val":
        raise ContractError(f"{prefix}: only the official ScanNet val split is accepted")
    training = record.get("training_materialization")
    if not isinstance(training, dict) or training.get("eligible") is not False:
        raise ContractError(f"{prefix}: held-out training prohibition is absent")
    model_interface = record.get("model_interface")
    if (
        not isinstance(model_interface, dict)
        or model_interface.get("intended_task") != "oracle_grounded_pairwise_depth"
        or model_interface.get("requires_explicit_pair_groundings") is not True
        or model_interface.get("instance_ids_are_internal_identity_only") is not True
    ):
        raise ContractError(f"{prefix}: model_interface drift")
    if record.get("_audit_only") not in (None, True):
        raise ContractError(f"{prefix}: invalid audit-only marker")
    if record.get("_audit_only") is True:
        if (
            record.get("audit_contract") != "scannet_sensor_pairwise"
            or record.get("task_interface") != "oracle_grounded_pairwise_depth"
        ):
            raise ContractError(f"{prefix}: audit projection contract drift")

    payload = record.get("payload")
    if not isinstance(payload, list) or len(payload) != 2:
        raise ContractError(f"{prefix}: payload must contain exactly two entities")
    ordered = sorted(payload, key=lambda item: item.get("rank", 999) if isinstance(item, dict) else 999)
    if not all(isinstance(item, dict) for item in ordered):
        raise ContractError(f"{prefix}: payload entries must be objects")
    if [item.get("rank") for item in ordered] != [1, 2]:
        raise ContractError(f"{prefix}: pair ranks must be [1, 2]")
    if [item.get("relative_depth") for item in ordered] != [0.0, 1.0]:
        raise ContractError(f"{prefix}: pair relative depths must be [0.0, 1.0]")
    labels = [item.get("label") for item in ordered]
    if not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != 2:
        raise ContractError(f"{prefix}: payload labels must be two distinct internal identities")

    measurements = _record_measurements(record)
    measurement_by_label: dict[str, dict[str, Any]] = {}
    for item in measurements:
        label = item.get("label")
        if not isinstance(label, str) or label in measurement_by_label:
            raise ContractError(f"{prefix}: sensor measurement labels are invalid")
        _bbox(item.get("bbox_2d"), context=f"{prefix}/{label}")
        measurement_by_label[label] = item
    if set(measurement_by_label) != set(labels):
        raise ContractError(f"{prefix}: payload/measurement identities do not match")

    context = record.get("gsa_context")
    groundings = context.get("groundings") if isinstance(context, dict) else None
    if not isinstance(groundings, list) or len(groundings) != 2:
        raise ContractError(f"{prefix}: explicit pair groundings are absent")
    grounding_map = {
        item.get("object_class"): item.get("bbox")
        for item in groundings
        if isinstance(item, dict)
    }
    for label, measurement in measurement_by_label.items():
        if grounding_map.get(label) != measurement.get("bbox_2d"):
            raise ContractError(f"{prefix}: context/measurement grounding drift for {label}")

    image_path = record.get("image_path")
    image_hash = record.get("image_sha256")
    if not isinstance(image_path, str) or not image_path:
        raise ContractError(f"{prefix}: image_path is absent")
    if not isinstance(image_hash, str) or not _SHA256.fullmatch(image_hash):
        raise ContractError(f"{prefix}: image_sha256 is invalid")
    if verify_image:
        path = Path(image_path)
        if not path.is_file():
            raise ContractError(f"{prefix}: image is missing: {path}")
        if file_sha256(path) != image_hash:
            raise ContractError(f"{prefix}: image hash drift")


def build_pair_example(record: dict[str, Any], *, verify_image: bool = True) -> PairExample:
    validate_source_record(record, verify_image=verify_image)
    identity = _source_identity(record)
    payload = sorted(record["payload"], key=lambda item: item["rank"])
    measurements = {item["label"]: item for item in _record_measurements(record)}

    near_measurement = measurements[payload[0]["label"]]
    far_measurement = measurements[payload[1]["label"]]
    near = Entity(
        internal_label=payload[0]["label"],
        public_name=_public_name(near_measurement, payload[0]["label"]),
        bbox=_bbox(near_measurement["bbox_2d"], context=f"{identity}/near"),
    )
    far = Entity(
        internal_label=payload[1]["label"],
        public_name=_public_name(far_measurement, payload[1]["label"]),
        bbox=_bbox(far_measurement["bbox_2d"], context=f"{identity}/far"),
    )

    permutation = hashlib.sha256(
        f"{EVALUATION_PROTOCOL}\0{AB_PERMUTATION_SALT}\0{identity}".encode("utf-8")
    ).digest()[0] & 1
    if permutation:
        entity_a, entity_b, nearer_choice = far, near, "B"
    else:
        entity_a, entity_b, nearer_choice = near, far, "A"
    image_sha = record["image_sha256"]
    eval_id = "gsa-sdepth-v50-" + hashlib.sha256(
        f"{EVALUATION_PROTOCOL}\0{identity}\0{image_sha}".encode("utf-8")
    ).hexdigest()[:20]
    gap_bucket = str(_visibility(record).get("gap_bucket") or "unspecified")
    return PairExample(
        eval_id=eval_id,
        source_identity=identity,
        image_path=record["image_path"],
        image_sha256=image_sha,
        entity_a=entity_a,
        entity_b=entity_b,
        nearer_choice=nearer_choice,
        gap_bucket=gap_bucket,
        official_split=record["official_split"],
    )


def _validate_source_report(report_path: Path, labels_path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not report_path.is_file():
        raise ContractError(f"source report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ContractError("source report must be a JSON object")
    if report.get("protocol") != SOURCE_PROTOCOL:
        raise ContractError("source report protocol drift")
    if report.get("label_version") != SOURCE_LABEL_VERSION:
        raise ContractError("source report label version drift")
    if report.get("eligible") is not True:
        raise ContractError("source report is not eligible")
    system_errors = report.get("system_errors")
    if not isinstance(system_errors, dict) or system_errors.get("total") != 0:
        raise ContractError("source report has system errors")
    if report.get("accepted") != len(records) or report.get("target") != len(records):
        raise ContractError("source report accepted/target count does not bind the labels")
    if report.get("labels_sha256") != file_sha256(labels_path):
        raise ContractError("source report labels_sha256 drift")
    if report.get("records_sha256") != canonical_hash(records):
        raise ContractError("source report records_sha256 drift")
    contract = report.get("source_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("intended_task") != "oracle_grounded_pairwise_depth"
        or contract.get("training_materialization_eligible") is not False
        or contract.get("generation_split") != "val"
        or contract.get("sensor_only") is not True
    ):
        raise ContractError("source report held-out task contract drift")
    return report


def load_pairs(
    labels_path: Path,
    *,
    report_path: Path | None = None,
    verify_images: bool = True,
) -> tuple[list[PairExample], dict[str, Any]]:
    records = _read_jsonl(labels_path)
    report = (
        _validate_source_report(report_path, labels_path, records)
        if report_path is not None
        else None
    )
    pairs = [build_pair_example(record, verify_image=verify_images) for record in records]
    eval_ids = [pair.eval_id for pair in pairs]
    identities = [pair.source_identity for pair in pairs]
    image_hashes = [pair.image_sha256 for pair in pairs]
    if len(set(eval_ids)) != len(eval_ids):
        raise ContractError("duplicate blinded eval_id")
    if len(set(identities)) != len(identities):
        raise ContractError("duplicate source sample identity")
    if len(set(image_hashes)) != len(image_hashes):
        raise ContractError("duplicate held-out image")
    labels_sha256 = file_sha256(labels_path)
    records_sha256 = canonical_hash(records)
    report_sha256 = file_sha256(report_path) if report_path else None
    formal_identity_match = bool(
        report
        and labels_sha256 == FORMAL_SOURCE_LABELS_SHA256
        and records_sha256 == FORMAL_SOURCE_RECORDS_SHA256
        and report_sha256 == FORMAL_SOURCE_REPORT_SHA256
    )
    label_balance = dict(sorted(Counter(pair.nearer_choice for pair in pairs).items()))
    if formal_identity_match and label_balance != FORMAL_LABEL_BALANCE:
        raise ContractError(
            "formal full_v3 A/B assignment drift: "
            f"expected {FORMAL_LABEL_BALANCE}, got {label_balance}"
        )
    metadata = {
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": labels_sha256,
        "records_sha256": records_sha256,
        "source_report_path": str(report_path.resolve()) if report_path else None,
        "source_report_sha256": report_sha256,
        "source_stage": report.get("stage") if report else "audit_projection_or_unbound_source",
        "formal_source_release": FORMAL_SOURCE_RELEASE,
        "formal_source_identity_match": formal_identity_match,
        "formal_source_ready": bool(
            formal_identity_match
            and report.get("stage") == "full"
            and len(records) >= MIN_FORMAL_RECORDS
            and isinstance(report.get("accepted_scenes"), int)
            and report["accepted_scenes"] >= MIN_FORMAL_SCENES
        ),
    }
    return pairs, metadata


def _extract_grounding_bbox(row: dict[str, Any], choice: str) -> tuple[int, int, int, int]:
    candidates: list[Any] = []
    for key in (choice, choice.lower()):
        value = row.get(key)
        if isinstance(value, dict):
            candidates.extend([value.get("bbox_2d"), value.get("bbox")])
    entities = row.get("entities")
    if isinstance(entities, dict):
        value = entities.get(choice) or entities.get(choice.lower())
        if isinstance(value, dict):
            candidates.extend([value.get("bbox_2d"), value.get("bbox")])
    candidates.extend([row.get(f"bbox_{choice.lower()}"), row.get(f"{choice.lower()}_bbox")])
    for candidate in candidates:
        if candidate is not None:
            return _bbox(candidate, context=f"grounding {row.get('eval_id')}/{choice}")
    raise ContractError(f"grounding {row.get('eval_id')}/{choice}: bbox is absent")


def load_groundings(
    path: Path,
    pairs: list[PairExample],
) -> tuple[dict[str, GroundingRow], dict[str, str]]:
    known = {pair.eval_id for pair in pairs}
    valid: dict[str, GroundingRow] = {}
    invalid: dict[str, str] = {}
    seen: set[str] = set()
    for row in _read_jsonl(path, allow_empty=True):
        eval_id = row.get("eval_id")
        if not isinstance(eval_id, str) or not eval_id:
            raise ContractError("grounding row lacks eval_id")
        if eval_id in seen:
            raise ContractError(f"duplicate grounding row: {eval_id}")
        if eval_id not in known:
            raise ContractError(f"grounding row is outside the source set: {eval_id}")
        seen.add(eval_id)
        try:
            valid[eval_id] = GroundingRow(
                bbox_a=_extract_grounding_bbox(row, "A"),
                bbox_b=_extract_grounding_bbox(row, "B"),
            )
        except ContractError as exc:
            invalid[eval_id] = str(exc)
    for eval_id in sorted(known - seen):
        invalid[eval_id] = "missing upstream grounding row"
    return valid, invalid


def prompt_for(pair: PairExample, *, bbox_a: tuple[int, ...], bbox_b: tuple[int, ...]) -> str:
    return (
        "Inspect the RGB image and compare only the two supplied objects. "
        "Bounding boxes are [x_min, y_min, x_max, y_max] in an image-normalized "
        "0-1000 coordinate system.\n"
        f"A: {pair.entity_a.public_name}; bbox {list(bbox_a)}\n"
        f"B: {pair.entity_b.public_name}; bbox {list(bbox_b)}\n"
        "Which supplied object is nearer to the camera? Answer only A or B."
    )


def _iter_mapping_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _iter_mapping_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_mapping_keys(item)


def _assert_blinded_request(request: dict[str, Any], pair: PairExample) -> None:
    """Fail closed if a future formatter adds private labels or source identity."""
    if set(request) != _PUBLIC_REQUEST_KEYS:
        raise ContractError(f"public request schema drift for {pair.eval_id}")
    public_pair = request.get("pair")
    if not isinstance(public_pair, dict) or set(public_pair) != _PUBLIC_PAIR_KEYS:
        raise ContractError(f"public pair schema drift for {pair.eval_id}")
    for choice in ("A", "B"):
        entity = public_pair.get(choice)
        if not isinstance(entity, dict) or set(entity) != _PUBLIC_ENTITY_KEYS:
            raise ContractError(f"public entity schema drift for {pair.eval_id}/{choice}")
    messages = request.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or set(messages[0]) != {"role", "content"}
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise ContractError(f"public message schema drift for {pair.eval_id}")
    leaked_keys = set(_iter_mapping_keys(request)) & _FORBIDDEN_PUBLIC_KEYS
    if leaked_keys:
        raise ContractError(
            f"private key leaked into {pair.eval_id}: {sorted(leaked_keys)}"
        )
    public_text = json.dumps(request, ensure_ascii=False)
    if pair.source_identity in public_text:
        raise ContractError(f"source identity leaked into {pair.eval_id}")
    if _INSTANCE_SUFFIX.search(public_text) or re.search(
        r"\[\s*instance\s+\d+\s*\]", public_text, re.IGNORECASE
    ):
        raise ContractError(f"internal instance identity leaked into {pair.eval_id}")
    for entity in (pair.entity_a, pair.entity_b):
        if (
            entity.internal_label != entity.public_name
            and entity.internal_label in public_text
        ):
            raise ContractError(f"internal entity label leaked into {pair.eval_id}")


def format_requests(
    pairs: list[PairExample],
    *,
    track: str,
    groundings: dict[str, GroundingRow] | None = None,
) -> list[dict[str, Any]]:
    track = _normalise_track(track)
    if track == TRACK_G_TO_S and groundings is None:
        raise ContractError("g-to-s preparation requires upstream grounding rows")
    requests: list[dict[str, Any]] = []
    for pair in pairs:
        if track == TRACK_S_ORACLE:
            bbox_a, bbox_b = pair.entity_a.bbox, pair.entity_b.bbox
        else:
            grounding = (groundings or {}).get(pair.eval_id)
            if grounding is None:
                continue
            bbox_a, bbox_b = grounding.bbox_a, grounding.bbox_b
        prompt = prompt_for(pair, bbox_a=bbox_a, bbox_b=bbox_b)
        request = {
            "eval_id": pair.eval_id,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "track": track,
            "messages": [{"role": "user", "content": prompt}],
            "images": [pair.image_path],
            "image_sha256": pair.image_sha256,
            "pair": {
                "A": {"name": pair.entity_a.public_name, "bbox_2d": list(bbox_a)},
                "B": {"name": pair.entity_b.public_name, "bbox_2d": list(bbox_b)},
                "bbox_coordinate_space": "image_normalized_0_1000_xyxy",
            },
        }
        # This is an executable privacy/contract assertion, not merely documentation.
        _assert_blinded_request(request, pair)
        requests.append(request)
    return requests


def _normalise_answer(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _choice_from_value(value: Any, name_a: str, name_b: str) -> str | None:
    if not isinstance(value, str):
        return None
    normal = _normalise_answer(value)
    match = re.fullmatch(r"(?:answer|choice|option|object|entity)?\s*([ab])", normal)
    if match:
        return match.group(1).upper()
    name_map: dict[str, set[str]] = defaultdict(set)
    name_map[_normalise_answer(name_a)].add("A")
    name_map[_normalise_answer(name_b)].add("B")
    choices = name_map.get(normal, set())
    return next(iter(choices)) if len(choices) == 1 else None


def _relation_choice(answer: str, name_a: str, name_b: str) -> str | None:
    letter_patterns = (
        (r"\b(?:object|entity|option)?\s*([ab])\s+(?:is\s+)?(?:the\s+)?(?:nearer|closer)\b", False),
        (r"\b(?:the\s+)?(?:nearer|closer)(?:\s+object|\s+entity|\s+option)?\s+(?:is\s+)?([ab])\b", False),
        (r"\b(?:object|entity|option)?\s*([ab])\s+(?:is\s+)?(?:the\s+)?(?:farther|further)\b", True),
    )
    for pattern, invert in letter_patterns:
        hits = re.findall(pattern, answer, re.IGNORECASE)
        if hits:
            choices = {hit.upper() for hit in hits}
            if len(choices) != 1:
                return None
            choice = next(iter(choices))
            return ({"A": "B", "B": "A"}[choice] if invert else choice)

    # Visible labels are accepted only when they uniquely identify one member of the
    # pair.  Same-class pairs therefore require A/B and cannot be guessed from "chair".
    normal_answer = _normalise_answer(answer)
    normal_names = {"A": _normalise_answer(name_a), "B": _normalise_answer(name_b)}
    if normal_names["A"] == normal_names["B"]:
        return None
    for choice, name in normal_names.items():
        escaped = re.escape(name)
        patterns = (
            rf"^(?:the\s+)?{escaped}\s+(?:is\s+)?(?:the\s+)?(?:nearer|closer)\b",
            rf"^(?:the\s+)?(?:nearer|closer)(?:\s+object|\s+entity)?\s+(?:is\s+)?(?:the\s+)?{escaped}(?:\s|$)",
        )
        if any(re.search(pattern, normal_answer) for pattern in patterns):
            return choice
    return None


def parse_nearer_choice(text: Any, name_a: str, name_b: str) -> ParsedChoice:
    if not isinstance(text, str):
        return ParsedChoice(None, "response_not_string")
    answer = strip_reasoning(text)
    if not answer:
        return ParsedChoice(None, "empty_or_unclosed_reasoning")

    for item in parse_json_items(answer):
        if not isinstance(item, dict):
            continue
        for key in ("nearer", "answer", "choice", "object", "entity"):
            if key in item:
                choice = _choice_from_value(item[key], name_a, name_b)
                if choice:
                    return ParsedChoice(choice, "structured")

    relation = _relation_choice(answer, name_a, name_b)
    if relation:
        return ParsedChoice(relation, "relation_phrase")
    direct = _choice_from_value(answer, name_a, name_b)
    if direct:
        return ParsedChoice(direct, "direct")
    return ParsedChoice(None, "ambiguous_or_unrecognised")


def load_predictions(
    path: Path, expected_eval_ids: set[str]
) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path, allow_empty=True):
        eval_id = row.get("eval_id") or row.get("uid")
        if not isinstance(eval_id, str) or not eval_id:
            raise ContractError("prediction row lacks eval_id")
        if eval_id in predictions:
            raise ContractError(f"duplicate prediction row: {eval_id}")
        if eval_id not in expected_eval_ids:
            raise ContractError(f"prediction row is outside the prepared request set: {eval_id}")
        response = None
        for key in ("response", "text", "output", "prediction"):
            if key in row:
                response = row[key]
                break
        if response is None and "choice" in row:
            response = row["choice"]
        predictions[eval_id] = {
            "response": response,
            "hit_token_cap": row.get("hit_token_cap"),
        }
    return predictions


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return round(float(numerator) / denominator, 6) if denominator else None


def score_predictions(
    pairs: list[PairExample],
    predictions: dict[str, dict[str, Any]],
    *,
    track: str,
    groundings: dict[str, GroundingRow] | None = None,
    invalid_groundings: dict[str, str] | None = None,
    source_metadata: dict[str, Any] | None = None,
    prediction_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    track = _normalise_track(track)
    invalid_groundings = invalid_groundings or {}
    requests = format_requests(pairs, track=track, groundings=groundings)
    expected_prediction_ids = {request["eval_id"] for request in requests}
    unexpected = set(predictions) - expected_prediction_ids
    if unexpected:
        raise ContractError(f"unexpected prediction ids: {sorted(unexpected)[:3]}")

    n_total = len(pairs)
    n_parsed = 0
    correct = 0
    response_parse_failures = 0
    parse_status = Counter()
    label_balance = Counter(pair.nearer_choice for pair in pairs)
    breakdown: dict[str, Counter] = defaultdict(Counter)
    capped = 0
    cap_metadata = 0

    grounding_valid = 0
    grounding_success = 0
    min_pair_ious: list[float] = []
    correct_with_grounding_success = 0

    for pair in pairs:
        group = breakdown[pair.gap_bucket]
        group["n"] += 1
        grounding_ok = True
        grounding_iou_ok = True
        if track == TRACK_G_TO_S:
            grounding = (groundings or {}).get(pair.eval_id)
            grounding_ok = grounding is not None
            grounding_iou_ok = False
            if grounding is not None:
                grounding_valid += 1
                iou_a = iou(
                    to_unit_box(grounding.bbox_a, 1, 1, declared="thousandth"),
                    to_unit_box(pair.entity_a.bbox, 1, 1, declared="thousandth"),
                )
                iou_b = iou(
                    to_unit_box(grounding.bbox_b, 1, 1, declared="thousandth"),
                    to_unit_box(pair.entity_b.bbox, 1, 1, declared="thousandth"),
                )
                min_iou = min(iou_a, iou_b)
                min_pair_ious.append(min_iou)
                grounding_iou_ok = iou_a >= 0.5 and iou_b >= 0.5
                if grounding_iou_ok:
                    grounding_success += 1
            else:
                min_pair_ious.append(0.0)

        row = predictions.get(pair.eval_id) if grounding_ok else None
        if row is None:
            status = "missing_prediction" if grounding_ok else "no_request_due_to_grounding"
            parse_status[status] += 1
            group["parse_failures"] += 1
            continue
        if isinstance(row.get("hit_token_cap"), bool):
            cap_metadata += 1
            capped += int(row["hit_token_cap"])
        parsed = parse_nearer_choice(
            row.get("response"), pair.entity_a.public_name, pair.entity_b.public_name
        )
        parse_status[parsed.status] += 1
        if parsed.choice is None:
            response_parse_failures += 1
            group["parse_failures"] += 1
            continue
        n_parsed += 1
        group["parsed"] += 1
        if parsed.choice == pair.nearer_choice:
            correct += 1
            group["correct"] += 1
            if track == TRACK_G_TO_S and grounding_iou_ok:
                correct_with_grounding_success += 1

    missing_predictions = len(expected_prediction_ids - set(predictions))
    generation_complete = missing_predictions == 0
    cap_metadata_complete = cap_metadata == len(predictions)
    source_ready = (
        True
        if source_metadata is None
        else source_metadata.get("formal_source_ready") is True
    )
    formal_s_oracle = bool(
        track == TRACK_S_ORACLE
        and source_metadata is not None
        and source_metadata.get("formal_source_ready") is True
    )
    provenance_verified = bool(
        not formal_s_oracle
        or (
            isinstance(prediction_provenance, dict)
            and prediction_provenance.get("verified") is True
            and prediction_provenance.get("track") == TRACK_S_ORACLE
            and prediction_provenance.get("evaluation_protocol")
            == EVALUATION_PROTOCOL
        )
    )
    score_trusted = (
        generation_complete
        and cap_metadata_complete
        and capped == 0
        and source_ready
        and provenance_verified
    )
    diagnostic_accuracy = _ratio(correct, n_total)
    primary_score = diagnostic_accuracy if score_trusted else None

    per_gap_bucket: dict[str, Any] = {}
    for name, counts in sorted(breakdown.items()):
        per_gap_bucket[name] = {
            "n": counts["n"],
            "parsed": counts["parsed"],
            "correct": counts["correct"],
            "parse_coverage": _ratio(counts["parsed"], counts["n"]),
            "pairwise_depth_accuracy_all": _ratio(counts["correct"], counts["n"]),
        }

    trust_reasons: list[str] = []
    if missing_predictions:
        trust_reasons.append(f"missing {missing_predictions} prepared prediction rows")
    if not cap_metadata_complete:
        trust_reasons.append("hit_token_cap metadata is incomplete")
    if capped:
        trust_reasons.append(f"{capped} responses hit the generation token cap")
    if not source_ready:
        trust_reasons.append(
            f"formal source requires the pinned {FORMAL_SOURCE_RELEASE} report/labels "
            f"identity, >= {MIN_FORMAL_RECORDS} records, and >= {MIN_FORMAL_SCENES} scenes"
        )
    if not provenance_verified:
        trust_reasons.append(
            "formal S-oracle scoring requires verified prediction-manifest, model, "
            "receipt, request, and source provenance"
        )
    primary_metric_name = (
        "pairwise_depth_accuracy_all"
        if track == TRACK_S_ORACLE
        else "end_to_end_accuracy_iou_at_0_5"
    )
    report: dict[str, Any] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "benchmark": "GSA-Depth ScanNet sensor pairwise",
        "track": track,
        "task_contract": {
            "input": "RGB plus exactly two named 0-1000 xyxy boxes",
            "target": "which supplied object is nearer: A or B",
            "generic_visible_object_order_prompt_allowed": False,
            "instance_ids_exposed_to_model": False,
            "training_materialization_eligible": False,
            "primary_metric": primary_metric_name,
        },
        "source": source_metadata or {},
        "provenance": prediction_provenance or {},
        "request_records_sha256": canonical_hash(requests),
        "prediction_file_rows": len(predictions),
        "label_balance": dict(sorted(label_balance.items())),
        "ab_assignment": {
            "rule": "sha256_protocol_salt_source_identity_low_bit",
            "salt": AB_PERMUTATION_SALT,
        },
        "metrics": {
            "n_total": n_total,
            "n_depth_requests": len(requests),
            "n_predictions": len(predictions),
            "n_parsed": n_parsed,
            "n_correct": correct,
            "response_parse_failures": response_parse_failures,
            "no_request_due_to_grounding": n_total - len(requests),
            "missing_prepared_predictions": missing_predictions,
            "parse_coverage_all": _ratio(n_parsed, n_total),
            "pairwise_depth_accuracy_all": primary_score,
            "score_if_untrusted": diagnostic_accuracy if not score_trusted else None,
            "pairwise_depth_accuracy_given_parsed": _ratio(correct, n_parsed),
            "hit_token_cap": capped,
            "hit_token_cap_metadata_coverage": _ratio(cap_metadata, len(predictions)),
        },
        "parse_status": dict(sorted(parse_status.items())),
        "by_gap_bucket": per_gap_bucket,
        "trust": {
            "score_available": score_trusted,
            "generation_complete": generation_complete,
            "token_cap_metadata_complete": cap_metadata_complete,
            "formal_source_ready": source_ready,
            "prediction_provenance_verified": provenance_verified,
            "reasons": trust_reasons,
            "publication_gate": "requires an independently compiled schema-5 human audit PASS",
        },
    }
    if track == TRACK_G_TO_S:
        cascade_diagnostic = _ratio(correct_with_grounding_success, n_total)
        report["cascade_metrics"] = {
            "valid_grounding_pair_coverage": _ratio(grounding_valid, n_total),
            "grounding_pair_success_iou_at_0_5": _ratio(grounding_success, n_total),
            "mean_min_pair_iou": (
                round(sum(min_pair_ious) / n_total, 6) if n_total else None
            ),
            "pairwise_depth_accuracy_given_grounding_success": _ratio(
                correct_with_grounding_success, grounding_success
            ),
            "end_to_end_accuracy_iou_at_0_5": (
                cascade_diagnostic if score_trusted else None
            ),
            "score_if_untrusted": cascade_diagnostic if not score_trusted else None,
            "invalid_or_missing_grounding_pairs": len(invalid_groundings),
        }
    return report


def prepare_command(args: argparse.Namespace) -> int:
    pairs, source_metadata = load_pairs(
        args.source,
        report_path=args.source_report,
        verify_images=not args.skip_image_verification,
    )
    track = _normalise_track(args.track)
    groundings: dict[str, GroundingRow] | None = None
    invalid: dict[str, str] = {}
    if track == TRACK_G_TO_S:
        if args.groundings is None:
            raise ContractError("--groundings is required for g-to-s")
        groundings, invalid = load_groundings(args.groundings, pairs)
    requests = format_requests(pairs, track=track, groundings=groundings)
    _write_jsonl(args.out, requests)
    report = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "track": track,
        "source": source_metadata,
        "n_source": len(pairs),
        "n_requests": len(requests),
        "n_grounding_invalid_or_missing": len(invalid),
        "label_balance": dict(sorted(Counter(pair.nearer_choice for pair in pairs).items())),
        "ab_assignment": {
            "rule": "sha256_protocol_salt_source_identity_low_bit",
            "salt": AB_PERMUTATION_SALT,
        },
        "request_records_sha256": canonical_hash(requests),
        "requests_path": str(args.out.resolve()),
        "requests_file_sha256": file_sha256(args.out),
        "blinding": {
            "internal_instance_ids_exposed": False,
            "sensor_measurements_exposed": False,
            "answers_exposed": False,
            "enforcement": "closed_request_schema_and_private_key_denylist",
        },
    }
    report_path = args.report_out or args.out.with_suffix(args.out.suffix + ".manifest.json")
    _write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def score_command(args: argparse.Namespace) -> int:
    pairs, source_metadata = load_pairs(
        args.source,
        report_path=args.source_report,
        verify_images=not args.skip_image_verification,
    )
    track = _normalise_track(args.track)
    formal_s_oracle = bool(
        track == TRACK_S_ORACLE
        and source_metadata.get("formal_source_ready") is True
    )
    prediction_provenance: dict[str, Any] | None = None
    if formal_s_oracle:
        if args.skip_image_verification:
            raise ContractError(
                "formal S-oracle scoring forbids --skip-image-verification"
            )
        if args.prediction_manifest is None or args.model_binding is None:
            raise ContractError(
                "formal S-oracle scoring requires --prediction-manifest and "
                "--model-binding"
            )
        from eval import scannet_sensor_pairwise_dlc as dlc_runner

        try:
            prediction_provenance = dlc_runner.verify_scoring_bundle(
                manifest_path=args.prediction_manifest,
                binding_path=args.model_binding,
                predictions_path=args.predictions,
            )
        except dlc_runner.RunnerError as exc:
            raise ContractError(f"formal prediction provenance failed: {exc}") from exc
    elif args.prediction_manifest is not None or args.model_binding is not None:
        raise ContractError(
            "prediction provenance arguments are reserved for the formal S-oracle release"
        )
    groundings: dict[str, GroundingRow] | None = None
    invalid: dict[str, str] = {}
    if track == TRACK_G_TO_S:
        if args.groundings is None:
            raise ContractError("--groundings is required for g-to-s")
        groundings, invalid = load_groundings(args.groundings, pairs)
    requests = format_requests(pairs, track=track, groundings=groundings)
    predictions = load_predictions(args.predictions, {row["eval_id"] for row in requests})
    source_metadata["predictions_path"] = str(args.predictions.resolve())
    source_metadata["predictions_sha256"] = file_sha256(args.predictions)
    if (
        prediction_provenance is not None
        and prediction_provenance["merged_predictions"]["sha256"]
        != source_metadata["predictions_sha256"]
    ):
        raise ContractError("prediction file drifted after provenance verification")
    if args.groundings:
        source_metadata["groundings_path"] = str(args.groundings.resolve())
        source_metadata["groundings_sha256"] = file_sha256(args.groundings)
    report = score_predictions(
        pairs,
        predictions,
        track=track,
        groundings=groundings,
        invalid_groundings=invalid,
        source_metadata=source_metadata,
        prediction_provenance=prediction_provenance,
    )
    _write_json(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["trust"]["score_available"] else 2


def _fixture_record(image: Path, index: int) -> dict[str, Any]:
    near_label = f"chair [instance {index * 2 + 1}]"
    far_label = f"table [instance {index * 2 + 2}]"
    groundings = [
        {"object_class": near_label, "bbox": [50, 100, 350, 800]},
        {"object_class": far_label, "bbox": [600, 150, 950, 900]},
    ]
    measurements = [
        {"label": near_label, "category": "chair", "bbox_2d": groundings[0]["bbox"]},
        {"label": far_label, "category": "table", "bbox_2d": groundings[1]["bbox"]},
    ]
    return {
        "sample_id": f"synthetic-depth-{index}",
        "protocol_version": SOURCE_PROTOCOL,
        "label_version": SOURCE_LABEL_VERSION,
        "official_split": "val",
        "image_path": str(image),
        "image_sha256": file_sha256(image),
        "payload": [
            {"label": near_label, "rank": 1, "relative_depth": 0.0},
            {"label": far_label, "rank": 2, "relative_depth": 1.0},
        ],
        "sensor_measurements": measurements,
        "gsa_context": {"contract_version": "gsa-contract-v2", "groundings": groundings},
        "model_interface": {
            "intended_task": "oracle_grounded_pairwise_depth",
            "requires_explicit_pair_groundings": True,
            "instance_ids_are_internal_identity_only": True,
        },
        "training_materialization": {
            "eligible": False,
            "allowed_use": "blind_audit_or_benchmark_only",
        },
        "visibility_features": {"gap_bucket": "synthetic_fixture"},
    }


def dry_run_command(args: argparse.Namespace) -> int:
    def run(directory: Path) -> dict[str, Any]:
        # A valid 1x1 PNG; fixture scores are contract checks, never evidence.
        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63606060f80f0001040100f51c0c020000000049454e44ae426082"
        )
        images = [directory / "fixture-0.png", directory / "fixture-1.png"]
        # PNG readers permit trailing bytes; the suffix makes the two fixture assets
        # independently hash-bound without adding an imaging dependency here.
        for index, image in enumerate(images):
            image.write_bytes(png + bytes([index]))
        source_path = directory / "source.jsonl"
        records = [_fixture_record(images[index], index) for index in range(2)]
        _write_jsonl(source_path, records)
        pairs, _ = load_pairs(source_path, verify_images=True)
        requests = format_requests(pairs, track=TRACK_S_ORACLE)
        predictions = {
            pair.eval_id: {"response": pair.nearer_choice, "hit_token_cap": False}
            for pair in pairs
        }
        report = score_predictions(
            pairs,
            predictions,
            track=TRACK_S_ORACLE,
        )
        public_payload = json.dumps(requests, ensure_ascii=False).casefold()
        passed = (
            report["metrics"]["pairwise_depth_accuracy_all"] == 1.0
            and "[instance" not in public_payload
            and "depth_m" not in public_payload
            and all("A:" in row["messages"][0]["content"] for row in requests)
            and all("B:" in row["messages"][0]["content"] for row in requests)
        )
        return {
            "status": "PASS" if passed else "FAIL",
            "synthetic_fixture_only": True,
            "n": len(pairs),
            "oracle_accuracy": report["metrics"]["pairwise_depth_accuracy_all"],
            "public_request_instance_id_leak": "[instance" in public_payload,
            "public_request_sensor_measurement_leak": "depth_m" in public_payload,
        }

    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        result = run(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="gsa-sdepth-eval-") as tmp:
            result = run(Path(tmp))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--source", type=Path, required=True, help="v5 source/audit JSONL")
        subparser.add_argument("--source-report", type=Path, help="optional frozen source generation report")
        subparser.add_argument("--track", required=True, choices=sorted(TRACK_ALIASES))
        subparser.add_argument("--groundings", type=Path, help="G-stage bbox JSONL; required by g-to-s")
        subparser.add_argument(
            "--skip-image-verification",
            action="store_true",
            help="diagnostic only; formal preparation/scoring verifies image hashes",
        )

    prepare = subparsers.add_parser("prepare", help="write blinded model requests")
    common(prepare)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--report-out", type=Path)
    prepare.set_defaults(func=prepare_command)

    score = subparsers.add_parser("score", help="score keyed model responses")
    common(score)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument(
        "--prediction-manifest",
        type=Path,
        help="formal DLC prediction manifest; required for formal S-oracle scoring",
    )
    score.add_argument(
        "--model-binding",
        type=Path,
        help="deep-hashed model binding; required for formal S-oracle scoring",
    )
    score.add_argument("--out", type=Path, required=True)
    score.set_defaults(func=score_command)

    dry_run = subparsers.add_parser("dry-run", help="run synthetic contract checks only")
    dry_run.add_argument("--work-dir", type=Path)
    dry_run.set_defaults(func=dry_run_command)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except ContractError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
