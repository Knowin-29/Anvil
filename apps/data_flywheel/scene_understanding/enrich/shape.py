"""Size and shape hint extraction helpers."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from apps.data_flywheel.scene_understanding.thresholds import SceneGraphThresholds


def extract_size(
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    thresholds: SceneGraphThresholds,
) -> Optional[str]:
    width, height = image_size
    bbox_w = max(0.0, float(bbox[2]) - float(bbox[0]))
    bbox_h = max(0.0, float(bbox[3]) - float(bbox[1]))
    ratio = (bbox_w * bbox_h) / max(1.0, float(width) * float(height))
    if ratio < thresholds.size_small_threshold:
        return "small"
    if ratio < thresholds.size_medium_threshold:
        return "medium"
    return "large"


def extract_shape_hint(bbox: Sequence[float]) -> Optional[str]:
    bbox_w = float(bbox[2]) - float(bbox[0])
    bbox_h = float(bbox[3]) - float(bbox[1])
    if bbox_w <= 0 or bbox_h <= 0:
        return None
    ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
    if ratio < 1.3:
        return "square"
    if bbox_w > bbox_h:
        return "wide"
    return "tall"
