"""RobotStateWorker — sidecar A_robot_state annotation for robot/tool runtime state."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.vlm_client import VLMClient

logger = logging.getLogger("flywheel.robot_state.worker")

ROBOT_STATE_PROMPT = """Look at this single robot-view image and infer the current robot/tool runtime state.

Return JSON only, no markdown, no explanation outside JSON. The output must describe A_robot_state, not generic object attributes.

Use this exact JSON object shape:
{
  "pipeline_type": "A_robot_state",
  "robot_part": "gripper",
  "tool_name": "parallel_jaw_gripper",
  "state_type": "holding_state",
  "state_value": "holding_object",
  "gripper_empty": false,
  "holding_object": "sponge",
  "contact_state": "in_contact",
  "object_between_fingers": true,
  "evidence_bbox": [120, 80, 300, 260],
  "confidence": 0.82,
  "confidence_reason": "Object is visibly centered between the two gripper fingers."
}

Allowed state_type examples: gripper_state, holding_state, contact_state, tool_state, wrist_view_state, uncertain.
Allowed state_value examples: empty, holding_object, in_contact, no_contact, object_between_fingers, tool_in_hand, uncertain_or_occluded.
If the state is unclear, set state_type and state_value to "uncertain" and explain briefly in confidence_reason.
"""


def _as_bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes", "1"}:
            return True
        if lower in {"false", "no", "0"}:
            return False
    return None


def _as_bbox(value: Any) -> List[float]:
    if not isinstance(value, list) or len(value) < 4:
        return []
    try:
        return [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return []


def _as_confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, score))


class RobotStateWorker:
    """Single-frame A_robot_state worker kept outside the existing capability registry."""

    def __init__(self, vlm_client: Optional[VLMClient] = None):
        self._vlm = vlm_client

    def process(self, sample: Sample) -> List[Dict[str, Any]]:
        if not self._vlm or not sample.image_path:
            return []

        try:
            result = self._vlm.chat_json(
                ROBOT_STATE_PROMPT,
                image_path=sample.image_path,
                max_tokens=2048,
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning(f"VLM robot_state failed: {exc}")
            return []

        items = result if isinstance(result, list) else [result] if isinstance(result, dict) else []
        annotations: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            state_type = str(item.get("state_type") or "uncertain")
            state_value = str(item.get("state_value") or "uncertain")
            if state_type == "uncertain" and state_value == "uncertain" and not item.get("robot_part"):
                continue

            annotations.append({
                "sample_id": sample.sample_id,
                "pipeline_type": "A_robot_state",
                "image_path": sample.image_path,
                "robot_part": str(item.get("robot_part") or "unknown"),
                "tool_name": str(item.get("tool_name") or "unknown"),
                "state_type": state_type,
                "state_value": state_value,
                "gripper_empty": _as_bool_or_none(item.get("gripper_empty")),
                "holding_object": item.get("holding_object"),
                "contact_state": str(item.get("contact_state") or "unknown"),
                "object_between_fingers": _as_bool_or_none(item.get("object_between_fingers")),
                "evidence_bbox": _as_bbox(item.get("evidence_bbox")),
                "confidence": _as_confidence(item.get("confidence")),
                "confidence_reason": str(item.get("confidence_reason") or ""),
            })

        return annotations
