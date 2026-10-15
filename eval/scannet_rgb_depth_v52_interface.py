#!/usr/bin/env python3
"""Strict public request/response interface for the ScanNet v5.2 G->S panel.

This module contains no model runner.  It only defines the public, non-answer-
bearing request projection and the parsers that a future frozen evaluator must
use.  In particular, a grounding response is never matched to one of several
same-category detections with oracle IoU.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INTERFACE_PROTOCOL = "gsa-depth-v5.2-unique-referent-gs-intervention-interface-v2"
GROUNDING_GRAMMAR = '{"A":[x1,y1,x2,y2],"B":[x1,y1,x2,y2]}'
SPATIAL_GRAMMAR = "A|B"


class InterfaceError(ValueError):
    """A private label cannot be projected into the frozen public interface."""


@dataclass(frozen=True)
class GroundingParse:
    boxes: dict[str, tuple[int, int, int, int]] | None
    status: str

    @property
    def valid(self) -> bool:
        return self.boxes is not None


@dataclass(frozen=True)
class SpatialParse:
    choice: str | None
    status: str

    @property
    def valid(self) -> bool:
        return self.choice is not None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _bbox(value: Any) -> tuple[int, int, int, int] | None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        return None
    box = tuple(value)
    if any(item < 0 or item > 1000 for item in box):
        return None
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return box  # type: ignore[return-value]


def parse_grounding_response(value: Any) -> GroundingParse:
    """Accept exactly one JSON object containing one box for each public slot."""
    if not isinstance(value, str):
        return GroundingParse(None, "not_string")
    try:
        document = json.loads(value)
    except json.JSONDecodeError:
        return GroundingParse(None, "invalid_json")
    if not isinstance(document, dict):
        return GroundingParse(None, "not_object")
    if set(document) != {"A", "B"}:
        return GroundingParse(None, "exact_keys_required")
    box_a = _bbox(document["A"])
    box_b = _bbox(document["B"])
    if box_a is None or box_b is None:
        return GroundingParse(None, "invalid_bbox")
    return GroundingParse({"A": box_a, "B": box_b}, "parsed_exact")


def parse_spatial_response(value: Any) -> SpatialParse:
    """Accept one uppercase slot and no prose or punctuation."""
    if not isinstance(value, str):
        return SpatialParse(None, "not_string")
    stripped = value.strip()
    if stripped not in {"A", "B"}:
        return SpatialParse(None, "not_exact_A_or_B")
    return SpatialParse(stripped, "parsed_exact")


def _public_targets(label: dict[str, Any]) -> list[dict[str, str]]:
    interface = label.get("public_interface")
    targets = interface.get("grounding_targets") if isinstance(interface, dict) else None
    if not isinstance(targets, list) or len(targets) != 2:
        raise InterfaceError("label lacks two public grounding targets")
    projected: list[dict[str, str]] = []
    for expected_slot, item in zip(("A", "B"), targets):
        if not isinstance(item, dict) or item.get("slot") != expected_slot:
            raise InterfaceError("public target slot order must be A then B")
        expression = item.get("referring_expression")
        category = item.get("category")
        if (
            not isinstance(expression, str)
            or not expression.strip()
            or not isinstance(category, str)
            or not category.strip()
        ):
            raise InterfaceError("public target text is absent")
        if "instance" in expression.casefold() or "[" in expression or "]" in expression:
            raise InterfaceError("public target exposes an internal instance identifier")
        projected.append(
            {
                "slot": expected_slot,
                "category": category,
                "referring_expression": expression,
            }
        )
    if projected[0]["category"].casefold() == projected[1]["category"].casefold():
        raise InterfaceError("public target categories must differ")
    return projected


def _request_id(label: dict[str, Any], track: str) -> str:
    sample_id = label.get("sample_id")
    image_sha256 = label.get("image_sha256")
    if not isinstance(sample_id, str) or not isinstance(image_sha256, str):
        raise InterfaceError("label identity is incomplete")
    return "gsa-sdepth-v52-" + canonical_hash(
        [INTERFACE_PROTOCOL, track, sample_id, image_sha256]
    )[:20]


def format_grounding_request(label: dict[str, Any], public_image: Path) -> dict[str, Any]:
    """Project a private label into a blind G request without boxes or S truth."""
    targets = _public_targets(label)
    public_image = public_image.expanduser().resolve()
    if not public_image.is_file():
        raise InterfaceError(f"public image is missing: {public_image}")
    prompt = (
        f"A is {targets[0]['referring_expression']}. "
        f"B is {targets[1]['referring_expression']}. "
        "Return exactly one 0-1000 bbox for each target as "
        f"{GROUNDING_GRAMMAR}. Use no other keys or text."
    )
    return {
        "request_id": _request_id(label, "G"),
        "track": "G_unique_named_pair_bbox",
        "image": str(public_image),
        "prompt": prompt,
        "response_grammar": GROUNDING_GRAMMAR,
        "targets": targets,
    }


def format_spatial_request(
    label: dict[str, Any],
    public_image: Path,
    boxes: dict[str, tuple[int, int, int, int] | list[int]] | None,
    *,
    context: str,
) -> dict[str, Any]:
    """Format one frozen S context intervention.

    The caller must state the context. This function never repairs or oracle-
    matches predicted boxes. Corrupted coordinates are constructed upstream by
    the preregistered evaluator rule, not selected from model behavior or S GT.
    """
    contexts = {
        "oracle_grounding": "S_oracle_unique_grounding",
        "predicted_grounding": "G_to_S_predicted_unique_grounding",
        "absent_grounding": "S_absent_grounding",
        "corrupted_grounding": "S_corrupted_grounding",
    }
    if context not in contexts:
        raise InterfaceError("unknown S context intervention")
    targets = _public_targets(label)
    normalized: dict[str, list[int]] = {}
    if context == "absent_grounding":
        if boxes is not None:
            raise InterfaceError("absent S context must not contain boxes")
    else:
        if not isinstance(boxes, dict) or set(boxes) != {"A", "B"}:
            raise InterfaceError("S context must contain exactly A/B boxes")
        for slot in ("A", "B"):
            candidate = list(boxes[slot]) if isinstance(boxes[slot], tuple) else boxes[slot]
            parsed = _bbox(candidate)
            if parsed is None:
                raise InterfaceError(f"invalid S context box: {slot}")
            normalized[slot] = list(parsed)
    public_image = public_image.expanduser().resolve()
    if not public_image.is_file():
        raise InterfaceError(f"public image is missing: {public_image}")
    if context == "absent_grounding":
        prompt = (
            f"A is {targets[0]['referring_expression']}; "
            f"B is {targets[1]['referring_expression']}. "
            f"Which target is nearer to the camera? Return exactly {SPATIAL_GRAMMAR}."
        )
    else:
        prompt = (
            f"A ({targets[0]['category']}) has bbox {normalized['A']}; "
            f"B ({targets[1]['category']}) has bbox {normalized['B']}. "
            f"Which target is nearer to the camera? Return exactly {SPATIAL_GRAMMAR}."
        )
    return {
        "request_id": _request_id(label, f"S:{context}"),
        "track": contexts[context],
        "context": context,
        "image": str(public_image),
        "prompt": prompt,
        "response_grammar": SPATIAL_GRAMMAR,
        "targets": targets,
        "boxes": normalized if context != "absent_grounding" else None,
    }
