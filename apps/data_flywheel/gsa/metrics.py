"""GSA Metrics — Offline evaluation metrics for GSA labels.

Implements: bbox_iou, point_in_bbox, point_in_polygon, point_in_mask,
direction_cosine, exact_match, blocked_by_consistency, required_fields_score.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def point_in_bbox(point: List[float], bbox: List[float]) -> bool:
    """Check if a 2D point is inside a bbox [x1, y1, x2, y2]."""
    if len(point) != 2 or len(bbox) != 4:
        return False
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    if len(point) != 2 or len(polygon) < 3:
        return False

    x, y = point[0], point[1]
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def direction_cosine(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two 2D direction vectors."""
    if len(vec_a) != 2 or len(vec_b) != 2:
        return 0.0

    dot = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]
    mag_a = math.sqrt(vec_a[0] ** 2 + vec_a[1] ** 2)
    mag_b = math.sqrt(vec_b[0] ** 2 + vec_b[1] ** 2)

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def exact_match(pred: Any, gold: Any) -> bool:
    """Exact match for class names, state values, yes/no answers."""
    if isinstance(pred, str) and isinstance(gold, str):
        return pred.strip().lower() == gold.strip().lower()
    return pred == gold


def blocked_by_consistency(
    label_blocked_by: List[str],
    spatial_facts: List[Dict[str, Any]],
    object_id: str,
) -> float:
    """Check if blocked_by list is consistent with spatial facts.

    Returns 1.0 if consistent, 0.0 if contradicted, 0.5 if no evidence.
    """
    if not label_blocked_by:
        # No blockers claimed — check if spatial facts agree
        blocking_facts = [f for f in spatial_facts
                          if f.get("object_id") == object_id
                          and f.get("predicate") in ("occluding", "blocking", "on_top_of")]
        if blocking_facts:
            return 0.0  # Spatial says blocked but label says not
        return 1.0

    # Has blockers — verify they exist in spatial facts
    confirmed = 0
    for blocker in label_blocked_by:
        matching = [f for f in spatial_facts
                    if f.get("subject_id") == blocker
                    and f.get("object_id") == object_id
                    and f.get("predicate") in ("occluding", "blocking", "on_top_of")]
        if matching:
            confirmed += 1

    if not spatial_facts:
        return 0.5  # No evidence to confirm or deny

    return confirmed / len(label_blocked_by) if label_blocked_by else 1.0


def required_fields_score(label: Dict[str, Any], required: set) -> float:
    """Fraction of required fields that are present and non-None."""
    if not required:
        return 1.0
    present = sum(1 for f in required if label.get(f) is not None)
    return present / len(required)


def verifiable_rate(labels: List[Dict[str, Any]]) -> float:
    """Fraction of labels that are auto_verifiable."""
    if not labels:
        return 0.0
    auto = sum(1 for l in labels if l.get("verifier_status") == "auto_verifiable")
    return auto / len(labels)


def point_distance(pt_a: List[float], pt_b: List[float]) -> float:
    """Euclidean distance between two 2D points."""
    if len(pt_a) != 2 or len(pt_b) != 2:
        return float("inf")
    return math.sqrt((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2)
