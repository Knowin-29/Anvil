"""A_action schema — Dataclass and validation for A_action labels.

Implements the A_action v0 contract: per-action required fields,
validation logic, and serialization to JSONL.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .taxonomy import (
    ActionType,
    Bucket,
    FallbackAction,
    LabelType,
    SceneFamily,
    SourceProvenance,
    VerifierStatus,
)


@dataclass
class AActionLabel:
    """A single A_action label row."""

    sample_id: str
    scene_family: str
    scenario_id: str
    object_id: str
    action_type: str
    label_type: str
    verifier_status: str
    source_provenance: str
    bucket: str = "A_action"
    s_evidence_ids: List[str] = field(default_factory=list)

    # Pick fields
    grasp_point_2d: Optional[List[float]] = None
    jaw_axis_2d: Optional[List[float]] = None
    can_pick_now: Optional[bool] = None
    blocked_by: Optional[List[str]] = None
    need_pre_push: Optional[bool] = None

    # Push fields
    push_contact_point_2d: Optional[List[float]] = None
    push_direction_2d: Optional[List[float]] = None
    can_push_now: Optional[bool] = None

    # Place fields
    place_region_2d: Optional[Dict[str, Any]] = None
    can_place_now: Optional[bool] = None

    # Optional / deferred
    affordance_region: Optional[Any] = None
    entry_clearance_score: Optional[float] = None
    stability_margin: Optional[float] = None
    fallback_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# Required fields per action type
PICK_REQUIRED = {"grasp_point_2d", "jaw_axis_2d", "can_pick_now", "blocked_by", "need_pre_push"}
PUSH_REQUIRED = {"push_contact_point_2d", "push_direction_2d", "can_push_now"}
PLACE_REQUIRED = {"place_region_2d", "can_place_now"}

ACTION_REQUIRED_FIELDS: Dict[str, set] = {
    ActionType.PICK.value: PICK_REQUIRED,
    ActionType.PUSH.value: PUSH_REQUIRED,
    ActionType.PLACE.value: PLACE_REQUIRED,
}


@dataclass
class ValidationResult:
    valid: bool
    missing_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def validate_a_action_label(label: AActionLabel) -> ValidationResult:
    """Validate an A_action label against the v0 contract."""
    errors: List[str] = []
    missing: List[str] = []

    # Basic field checks
    if not label.sample_id:
        errors.append("sample_id is empty")
    if not label.object_id:
        errors.append("object_id is empty")

    # Action type validation
    try:
        ActionType(label.action_type)
    except ValueError:
        errors.append(f"invalid action_type: {label.action_type}")
        return ValidationResult(valid=False, missing_fields=missing, errors=errors)

    # Scene family validation
    if label.scene_family not in {sf.value for sf in SceneFamily}:
        errors.append(f"invalid scene_family: {label.scene_family}")

    # Verifier status validation
    try:
        VerifierStatus(label.verifier_status)
    except ValueError:
        errors.append(f"invalid verifier_status: {label.verifier_status}")

    # Per-action required fields
    required = ACTION_REQUIRED_FIELDS.get(label.action_type, set())
    label_dict = asdict(label)
    for f in required:
        if label_dict.get(f) is None:
            missing.append(f)

    # Point validation
    if label.grasp_point_2d is not None:
        if not _is_valid_point(label.grasp_point_2d):
            errors.append("grasp_point_2d must be [x, y] with numeric values")

    if label.push_contact_point_2d is not None:
        if not _is_valid_point(label.push_contact_point_2d):
            errors.append("push_contact_point_2d must be [x, y] with numeric values")

    # Vector validation
    if label.jaw_axis_2d is not None:
        if not _is_valid_vector(label.jaw_axis_2d):
            errors.append("jaw_axis_2d must be non-zero [dx, dy]")

    if label.push_direction_2d is not None:
        if not _is_valid_vector(label.push_direction_2d):
            errors.append("push_direction_2d must be non-zero [dx, dy]")

    # Place region validation
    if label.place_region_2d is not None:
        if not _is_valid_region(label.place_region_2d):
            errors.append("place_region_2d must have type and points (>=3)")

    # Fallback action validation
    if label.fallback_action is not None:
        try:
            FallbackAction(label.fallback_action)
        except ValueError:
            errors.append(f"invalid fallback_action: {label.fallback_action}")

    valid = len(errors) == 0 and len(missing) == 0
    return ValidationResult(valid=valid, missing_fields=missing, errors=errors)


def _is_valid_point(pt: Any) -> bool:
    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
        return False
    return all(isinstance(v, (int, float)) for v in pt)


def _is_valid_vector(vec: Any) -> bool:
    if not _is_valid_point(vec):
        return False
    return not (vec[0] == 0 and vec[1] == 0)


def _is_valid_region(region: Any) -> bool:
    if not isinstance(region, dict):
        return False
    if "type" not in region or "points" not in region:
        return False
    points = region["points"]
    if not isinstance(points, list) or len(points) < 3:
        return False
    return all(_is_valid_point(p) for p in points)
