"""Pick-and-Place Schema — Application-aligned output format.

Defines the complete JSON structure that the trained model must output,
directly aligned with the pick-and-place application system.

Coordinates are normalized to [0, 1000] relative to image dimensions.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# Valid enum values (aligned with application)
OBJECT_TYPES = ("box", "cylinder", "frustum", "bowl", "plate", "slender", "thin", "irregular", "unknown")
POSE_HINTS = ("upright", "horizontal", "unknown")
GRASP_HINTS = ("rim", "side", "middle", "edge", "handle", "body", "unknown")
CONTAINER_OPENINGS = ("open_visible", "closed_or_lidded", "unknown")
APPROACH_DIRECTIONS = ("horizontal", "top_down", "diagonal_down", "unknown")


@dataclass
class PickPlaceLabel:
    """Complete pick-and-place label aligned with application output format."""

    sample_id: str
    image_path: str
    query: str
    object_id: str = ""
    object_name: str = ""

    # G: Grounding — bbox normalized [0, 1000]
    bbox: List[int] = field(default_factory=list)

    # A_affordance: Affordance Point
    affordance_point: List[int] = field(default_factory=list)
    affordance_valid: bool = True
    affordance_reason: str = ""

    # A_state: Object Classification
    object_type: str = "unknown"
    object_type_confidence: float = 0.0
    pose_hint: str = "unknown"
    pose_hint_confidence: float = 0.0
    pose_hint_reason: str = ""
    grasp_hint: str = "unknown"
    object_pose: str = ""

    # A_state: Container State
    container_opening: str = "unknown"
    container_opening_confidence: float = 0.0
    container_opening_reason: str = ""

    # A_action: Approach Planning
    preferred_approach: str = "unknown"
    preferred_approach_confidence: float = 0.0
    preferred_approach_reason: str = ""
    fallback_approaches: List[str] = field(default_factory=list)

    # RynnBrain extension: Grasp Pose (4-corner oriented rectangle)
    grasp_rect_corners: Optional[List[List[float]]] = None
    grasp_center: Optional[List[float]] = None
    grasp_width: Optional[float] = None
    grasp_height: Optional[float] = None
    grasp_angle: Optional[float] = None

    def to_app_json(self) -> Dict[str, Any]:
        """Export in the exact application output format."""
        return {
            "bbox": self.bbox,
            "affordance_point": self.affordance_point,
            "affordance_valid": self.affordance_valid,
            "affordance_reason": self.affordance_reason,
            "object_type": self.object_type,
            "object_type_confidence": self.object_type_confidence,
            "pose_hint": self.pose_hint,
            "pose_hint_confidence": self.pose_hint_confidence,
            "pose_hint_reason": self.pose_hint_reason,
            "grasp_hint": self.grasp_hint,
            "object_pose": self.object_pose,
            "container_opening": self.container_opening,
            "container_opening_confidence": self.container_opening_confidence,
            "container_opening_reason": self.container_opening_reason,
            "preferred_approach": self.preferred_approach,
            "preferred_approach_confidence": self.preferred_approach_confidence,
            "preferred_approach_reason": self.preferred_approach_reason,
            "fallback_approaches": self.fallback_approaches,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Export all fields including RynnBrain extensions."""
        d = self.to_app_json()
        d.update({
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "query": self.query,
            "object_id": self.object_id,
            "object_name": self.object_name,
        })
        if self.grasp_rect_corners:
            d["grasp_rect_corners"] = self.grasp_rect_corners
        if self.grasp_center:
            d["grasp_center"] = self.grasp_center
        if self.grasp_width is not None:
            d["grasp_width"] = self.grasp_width
        if self.grasp_height is not None:
            d["grasp_height"] = self.grasp_height
        if self.grasp_angle is not None:
            d["grasp_angle"] = self.grasp_angle
        return d

    def to_jsonl(self) -> str:
        return json.dumps(self.to_full_dict(), ensure_ascii=False)


def center_wh_angle_to_corners(
    cx: float, cy: float, w: float, h: float, angle: float
) -> List[List[float]]:
    """Convert center + width + height + angle to 4 ordered corner points.

    Args:
        cx, cy: center of grasp rectangle (0-1000 normalized)
        w: gripper opening width
        h: gripper finger length
        angle: rotation angle in radians (0 = horizontal)

    Returns:
        4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = w / 2
    dy = h / 2
    corners = [
        [cx + cos_a * dx - sin_a * dy, cy + sin_a * dx + cos_a * dy],
        [cx - cos_a * dx - sin_a * dy, cy - sin_a * dx + cos_a * dy],
        [cx - cos_a * dx + sin_a * dy, cy - sin_a * dx - cos_a * dy],
        [cx + cos_a * dx + sin_a * dy, cy + sin_a * dx - cos_a * dy],
    ]
    return [[round(x, 1), round(y, 1)] for x, y in corners]


def corners_to_center_wh_angle(
    corners: List[List[float]],
) -> tuple:
    """Convert 4 corner points back to center + w + h + angle."""
    cx = sum(c[0] for c in corners) / 4
    cy = sum(c[1] for c in corners) / 4
    edge1_dx = corners[1][0] - corners[0][0]
    edge1_dy = corners[1][1] - corners[0][1]
    edge2_dx = corners[2][0] - corners[1][0]
    edge2_dy = corners[2][1] - corners[1][1]
    w = math.sqrt(edge1_dx**2 + edge1_dy**2)
    h = math.sqrt(edge2_dx**2 + edge2_dy**2)
    angle = math.atan2(edge1_dy, edge1_dx)
    return cx, cy, w, h, angle


def validate_pick_place_label(label: PickPlaceLabel) -> Dict[str, Any]:
    """Validate a PickPlaceLabel against application requirements."""
    issues = []

    if not label.bbox or len(label.bbox) != 4:
        issues.append("bbox must be [x1, y1, x2, y2]")
    elif label.bbox[0] >= label.bbox[2] or label.bbox[1] >= label.bbox[3]:
        issues.append("bbox invalid: x1 >= x2 or y1 >= y2")

    if not label.affordance_point or len(label.affordance_point) != 2:
        issues.append("affordance_point must be [x, y]")
    elif label.bbox and len(label.bbox) == 4:
        ax, ay = label.affordance_point
        if not (label.bbox[0] <= ax <= label.bbox[2] and label.bbox[1] <= ay <= label.bbox[3]):
            issues.append("affordance_point must lie inside bbox")

    if label.object_type not in OBJECT_TYPES:
        issues.append(f"object_type must be one of {OBJECT_TYPES}")

    if label.pose_hint not in POSE_HINTS:
        issues.append(f"pose_hint must be one of {POSE_HINTS}")

    if label.grasp_hint not in GRASP_HINTS:
        issues.append(f"grasp_hint must be one of {GRASP_HINTS}")

    if label.container_opening not in CONTAINER_OPENINGS:
        issues.append(f"container_opening must be one of {CONTAINER_OPENINGS}")

    if label.preferred_approach not in APPROACH_DIRECTIONS:
        issues.append(f"preferred_approach must be one of {APPROACH_DIRECTIONS}")

    for fa in label.fallback_approaches:
        if fa not in APPROACH_DIRECTIONS:
            issues.append(f"fallback_approach '{fa}' invalid")

    return {"valid": len(issues) == 0, "issues": issues}
