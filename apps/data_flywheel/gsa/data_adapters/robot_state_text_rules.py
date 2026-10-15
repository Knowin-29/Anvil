"""Shared text/action rules for A_robot_state dataset adapters."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from ..taxonomy import ARobotStateSubtype, CameraView

_HOLDING_RE = re.compile(
    r"\b(grasped|holding|picked up|carrying|gripper holds|with the .+ in (?:hand|gripper))\b",
    re.IGNORECASE,
)
_EMPTY_RE = re.compile(
    r"\b(release|released|put down|place the .+ down|empty gripper|open gripper)\b",
    re.IGNORECASE,
)
_REACH_RE = re.compile(r"\b(reach for|approach|move toward|extend toward)\b", re.IGNORECASE)
_CONTACT_RE = re.compile(r"\b(contact|touch|touching|in contact)\b", re.IGNORECASE)
_BETWEEN_RE = re.compile(r"\b(between (?:the )?fingers|pinch|pinching|between finger)\b", re.IGNORECASE)
_GRASP_RE = re.compile(r"\b(grasp|pick up|pick the|grab)\b", re.IGNORECASE)


def infer_state_from_text(text: str) -> Optional[Tuple[str, str]]:
    """Map free text to (state_type, state_value)."""
    if not text or not text.strip():
        return None
    if _BETWEEN_RE.search(text):
        return ARobotStateSubtype.OBJECT_BETWEEN_FINGERS.value, "yes"
    if _CONTACT_RE.search(text):
        return ARobotStateSubtype.CONTACT_STATE.value, "yes"
    if _HOLDING_RE.search(text):
        return ARobotStateSubtype.HOLDING_OBJECT.value, "yes"
    if _EMPTY_RE.search(text):
        return ARobotStateSubtype.GRIPPER_EMPTY.value, "yes"
    if _REACH_RE.search(text):
        return ARobotStateSubtype.GRIPPER_EMPTY.value, "yes"
    if _GRASP_RE.search(text):
        return ARobotStateSubtype.HOLDING_OBJECT.value, "unknown"
    return None


def infer_state_from_instruction(instruction: str) -> Optional[Tuple[str, str]]:
    mapped = infer_state_from_text(instruction)
    if mapped:
        return mapped
    if instruction and instruction.strip():
        return ARobotStateSubtype.CONTACT_STATE.value, "unknown"
    return None


def infer_state_from_er_action(action: str) -> Optional[Tuple[str, str]]:
    """Map Embodied-Reasoner key_actions.action to robot state."""
    if not action:
        return None
    act = action.strip().lower()
    if act in ("end",):
        return None
    if act == "pickup":
        return ARobotStateSubtype.HOLDING_OBJECT.value, "yes"
    if act in ("put in", "drop", "release", "put"):
        return ARobotStateSubtype.GRIPPER_EMPTY.value, "yes"
    if act in ("open", "close", "toggle"):
        return ARobotStateSubtype.CONTACT_STATE.value, "yes"
    if act == "navigate to":
        return ARobotStateSubtype.GRIPPER_EMPTY.value, "yes"
    return None


def infer_camera_view(image_path: str, meta: Optional[dict] = None) -> str:
    meta = meta or {}
    original = str(meta.get("original_dataset", "")).lower()
    path_lower = (image_path or "").lower()
    if any(k in path_lower or k in original for k in ("wrist", "hand_camera", "bridgedata", "bridge", "episode_")):
        return CameraView.WRIST.value
    if "head" in path_lower:
        return CameraView.HEAD.value
    return CameraView.EXTERNAL.value
