"""A Pipeline Schemas — Dataclass definitions for the 5 fine-grained A pipelines.

Aligned with the fusion classification doc (Section 6):
- A_identity_attr: object identity + static attributes
- A_obj_state: object physical state
- A_robot_state: gripper / end-effector state
- A_action_precondition: action feasibility
- A_action_interface: 2D geometric action output
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .taxonomy import (
    ActionType,
    AActionInterfaceSubtype,
    AActionPreconditionSubtype,
    AIdentityAttrSubtype,
    AObjStateSubtype,
    ARobotStateSubtype,
    Bucket,
    CameraView,
    EndEffector,
    FailureReason,
    FallbackAction,
    KNOWN_COLORS,
    KNOWN_MATERIALS,
    KNOWN_SHAPES,
    LabelType,
    PreAction,
    SourceProvenance,
    STATE_TYPE_VALUES,
    VerifierStatus,
)
from .a_action_interface_contract import (
    ACTION_INTERFACE_SCHEMA_VERSION,
    LABEL_STATUS_CANDIDATE,
)


@dataclass
class _BaseLabelMixin:
    sample_id: str
    scene_family: str = ""
    scenario_id: str = ""
    object_id: str = ""
    verifier_status: str = VerifierStatus.AUTO_VERIFIABLE.value
    source_provenance: str = SourceProvenance.VLM.value
    g_context: bool = False
    s_context: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class AIdentityAttrLabel(_BaseLabelMixin):
    """Object identity and static attribute label."""
    bucket: str = Bucket.A_IDENTITY_ATTR.value
    subtype: str = AIdentityAttrSubtype.OBJECT_CLASS.value
    object_class: str = ""
    bbox: Optional[List[float]] = None
    mask: Optional[Any] = None
    attributes: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "color": None, "material": None, "shape": None, "size": None,
    })
    confidence: Optional[float] = None


@dataclass
class AIdentityAttrValidation:
    valid: bool
    errors: List[str] = field(default_factory=list)


def validate_identity_attr(label: AIdentityAttrLabel) -> AIdentityAttrValidation:
    errors: List[str] = []
    if not label.sample_id:
        errors.append("sample_id is empty")
    if not label.object_class:
        errors.append("object_class is empty")
    attrs = label.attributes
    if attrs.get("color") and attrs["color"] not in KNOWN_COLORS:
        errors.append(f"unknown color: {attrs['color']}")
    if attrs.get("material") and attrs["material"] not in KNOWN_MATERIALS:
        errors.append(f"unknown material: {attrs['material']}")
    if attrs.get("shape") and attrs["shape"] not in KNOWN_SHAPES:
        errors.append(f"unknown shape: {attrs['shape']}")
    if label.bbox:
        if len(label.bbox) != 4:
            errors.append("bbox must have 4 elements")
        elif label.bbox[2] <= label.bbox[0] or label.bbox[3] <= label.bbox[1]:
            errors.append("bbox has invalid dimensions")
    return AIdentityAttrValidation(valid=len(errors) == 0, errors=errors)


@dataclass
class AObjStateLabel(_BaseLabelMixin):
    """Object physical state label."""
    bucket: str = Bucket.A_OBJ_STATE.value
    subtype: str = AObjStateSubtype.POSE.value
    object_class: str = ""
    bbox: Optional[List[float]] = None
    state_type: str = ""
    state_value: str = ""
    evidence_region: Optional[Any] = None
    confidence: Optional[float] = None


@dataclass
class AObjStateValidation:
    valid: bool
    errors: List[str] = field(default_factory=list)


def validate_obj_state(label: AObjStateLabel) -> AObjStateValidation:
    errors: List[str] = []
    if not label.sample_id:
        errors.append("sample_id is empty")
    if not label.state_type:
        errors.append("state_type is empty")
    if not label.state_value:
        errors.append("state_value is empty")
    if label.state_type and label.state_type not in STATE_TYPE_VALUES:
        errors.append(f"invalid state_type: {label.state_type}")
    elif label.state_type and label.state_value:
        allowed = STATE_TYPE_VALUES.get(label.state_type, [])
        if allowed and label.state_value not in allowed:
            errors.append(
                f"state_value '{label.state_value}' not valid for "
                f"state_type '{label.state_type}'"
            )
    return AObjStateValidation(valid=len(errors) == 0, errors=errors)


@dataclass
class ARobotStateLabel(_BaseLabelMixin):
    """Robot end-effector / gripper state label."""
    bucket: str = Bucket.A_ROBOT_STATE.value
    subtype: str = ARobotStateSubtype.GRIPPER_EMPTY.value
    camera_view: str = CameraView.WRIST.value
    end_effector: str = EndEffector.PARALLEL_JAW.value
    state_type: str = ""
    state_value: str = ""
    target_object_class: Optional[str] = None
    evidence_region: Optional[Any] = None
    visual_evidence: Optional[str] = None
    confidence: Optional[float] = None
    verifier_result: Optional[str] = None
    verifier_reason: Optional[str] = None
    image_path: Optional[str] = None
    source_dataset: Optional[str] = None
    episode_id: Optional[str] = None
    step_index: Optional[int] = None
    episode_phase: Optional[str] = None
    task_instruction: Optional[str] = None
    task_type: Optional[str] = None
    task_target_object_class: Optional[str] = None
    instruction_role: Optional[str] = None
    label_status: Optional[str] = None
    stable_training_label: bool = False
    quality_flags: List[str] = field(default_factory=list)


@dataclass
class ARobotStateValidation:
    valid: bool
    errors: List[str] = field(default_factory=list)


def validate_robot_state(label: ARobotStateLabel) -> ARobotStateValidation:
    errors: List[str] = []
    if not label.sample_id:
        errors.append("sample_id is empty")
    if not label.camera_view:
        errors.append("camera_view is required")
    elif label.camera_view not in {v.value for v in CameraView}:
        errors.append(f"invalid camera_view: {label.camera_view}")
    if not label.state_type:
        errors.append("state_type is empty")
    elif label.state_type not in {s.value for s in ARobotStateSubtype}:
        errors.append(f"invalid state_type: {label.state_type}")
    if not label.state_value:
        errors.append("state_value is empty")
    elif label.state_value not in {"yes", "no", "unknown"}:
        errors.append(f"invalid state_value: {label.state_value}")
    return ARobotStateValidation(valid=len(errors) == 0, errors=errors)


@dataclass
class AActionPreconditionLabel(_BaseLabelMixin):
    """Action feasibility / precondition label."""
    bucket: str = Bucket.A_ACTION_PRECONDITION.value
    subtype: str = AActionPreconditionSubtype.CAN_PICK.value
    action_type: str = ActionType.PICK.value
    can_do_now: bool = True
    blocked_by: List[str] = field(default_factory=list)
    need_pre_action: str = PreAction.NONE.value
    failure_reason: Optional[str] = None
    fallback_action: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class AActionPreconditionValidation:
    valid: bool
    errors: List[str] = field(default_factory=list)


def validate_action_precondition(
    label: AActionPreconditionLabel,
) -> AActionPreconditionValidation:
    errors: List[str] = []
    if not label.sample_id:
        errors.append("sample_id is empty")
    if label.action_type not in {a.value for a in ActionType}:
        errors.append(f"invalid action_type: {label.action_type}")
    if not label.can_do_now:
        if not label.failure_reason:
            errors.append("failure_reason required when can_do_now=false")
        elif label.failure_reason not in {r.value for r in FailureReason}:
            errors.append(f"invalid failure_reason: {label.failure_reason}")
    if label.need_pre_action not in {p.value for p in PreAction}:
        errors.append(f"invalid need_pre_action: {label.need_pre_action}")
    if label.fallback_action is not None:
        if label.fallback_action not in {f.value for f in FallbackAction}:
            errors.append(f"invalid fallback_action: {label.fallback_action}")
    return AActionPreconditionValidation(valid=len(errors) == 0, errors=errors)


@dataclass
class AActionInterfaceLabel(_BaseLabelMixin):
    """2D geometric action interface label."""
    label_id: Optional[str] = None
    schema_version: str = ACTION_INTERFACE_SCHEMA_VERSION
    label_status: str = LABEL_STATUS_CANDIDATE
    stable_training_label: bool = False
    bucket: str = Bucket.A_ACTION_INTERFACE.value
    subtype: str = AActionInterfaceSubtype.GRASP_POINT.value
    action_type: str = ActionType.PICK.value
    label_type: str = LabelType.POINT.value
    image_path: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    object_name: Optional[str] = None
    bbox: Optional[List[float]] = None
    affordance_region: Optional[Any] = None
    grasp_point_2d: Optional[List[float]] = None
    jaw_axis_2d: Optional[List[float]] = None
    push_contact_point_2d: Optional[List[float]] = None
    push_direction_2d: Optional[List[float]] = None
    place_region_2d: Optional[Any] = None
    s_evidence_ids: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    promotion_blockers: List[str] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None


@dataclass
class AActionInterfaceValidation:
    valid: bool
    missing_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


_INTERFACE_REQUIRED = {
    ActionType.PICK.value: {"grasp_point_2d", "jaw_axis_2d"},
    ActionType.PUSH.value: {"push_contact_point_2d", "push_direction_2d"},
    ActionType.PLACE.value: {"place_region_2d"},
    ActionType.REORIENT.value: {"grasp_point_2d"},
}


def validate_action_interface(
    label: AActionInterfaceLabel,
) -> AActionInterfaceValidation:
    errors: List[str] = []
    missing: List[str] = []
    if not label.sample_id:
        errors.append("sample_id is empty")
    if not label.object_id:
        errors.append("object_id is empty")
    if label.action_type not in {a.value for a in ActionType}:
        errors.append(f"invalid action_type: {label.action_type}")
    required = _INTERFACE_REQUIRED.get(label.action_type, set())
    label_dict = asdict(label)
    for f in required:
        if label_dict.get(f) is None:
            missing.append(f)
    if label.grasp_point_2d is not None:
        if not _is_valid_point(label.grasp_point_2d):
            errors.append("grasp_point_2d must be [x, y]")
    if label.push_contact_point_2d is not None:
        if not _is_valid_point(label.push_contact_point_2d):
            errors.append("push_contact_point_2d must be [x, y]")
    if label.jaw_axis_2d is not None:
        if not _is_valid_vector(label.jaw_axis_2d):
            errors.append("jaw_axis_2d must be non-zero [dx, dy]")
    if label.push_direction_2d is not None:
        if not _is_valid_vector(label.push_direction_2d):
            errors.append("push_direction_2d must be non-zero [dx, dy]")
    return AActionInterfaceValidation(
        valid=len(errors) == 0 and len(missing) == 0,
        missing_fields=missing, errors=errors,
    )


def _is_valid_point(pt: Any) -> bool:
    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
        return False
    return all(isinstance(v, (int, float)) for v in pt)


def _is_valid_vector(vec: Any) -> bool:
    if not _is_valid_point(vec):
        return False
    return not (vec[0] == 0 and vec[1] == 0)
