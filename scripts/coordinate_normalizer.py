#!/usr/bin/env python3
"""Coordinate normalizer — unify diverse coordinate systems to thousandth [0, 1000).

Addresses the systemic issue identified in 高优先级数据集说明.html:
  pixel xywh (COCO) / normalized 0-1 (OIv7/RefSpatial) / percent 0-100 (PixMo/Molmo)
  / thousandth 0-1000 (Qwen/InternVL target) — must unify at pipeline entry.

Usage:
    from scripts.coordinate_normalizer import CoordinateNormalizer
    norm = CoordinateNormalizer(source_system="pixel_xywh", image_wh=(1024, 768))
    bbox_1000 = norm.normalize_bbox([100, 200, 300, 400])  # → [97, 260, 292, 520]
    point_1000 = norm.normalize_point([512, 384])           # → [500, 500]

CLI:
    python scripts/coordinate_normalizer.py --test
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

logger = logging.getLogger("coord_normalizer")

TARGET_RANGE = 1000


class CoordSystem(str, Enum):
    PIXEL_XYWH = "pixel_xywh"       # [x, y, w, h] in pixels (COCO convention)
    PIXEL_XYXY = "pixel_xyxy"       # [x1, y1, x2, y2] in pixels
    NORM_01 = "norm_01"             # [0.0, 1.0] normalized
    PERCENT_100 = "percent_100"     # [0, 100] percentage
    THOUSANDTH_1000 = "thousandth_1000"  # [0, 1000) target format
    NORM_01_XMIN_XMAX = "norm_01_xmin_xmax"  # OIv7: [XMin, XMax, YMin, YMax] in [0,1]
    TEXT_EMBEDDED = "text_embedded"  # Coordinates in text, no geometric normalization


@dataclass
class CoordinateNormalizer:
    """Normalize coordinates from various systems to thousandth [0, 1000)."""

    source_system: str
    image_wh: Optional[Tuple[int, int]] = None  # (width, height), required for pixel systems

    def __post_init__(self):
        if self.source_system in (CoordSystem.PIXEL_XYWH, CoordSystem.PIXEL_XYXY):
            if self.image_wh is None:
                raise ValueError(f"image_wh required for {self.source_system}")

    def normalize_bbox(self, bbox: List[Union[int, float]]) -> Optional[List[int]]:
        """Convert bbox to [x1, y1, x2, y2] in [0, 1000).

        Input formats:
          pixel_xywh: [x, y, w, h] in pixels
          pixel_xyxy: [x1, y1, x2, y2] in pixels
          norm_01: [x1, y1, x2, y2] in [0, 1]
          percent_100: [x1, y1, x2, y2] in [0, 100]
          norm_01_xmin_xmax: [XMin, XMax, YMin, YMax] in [0, 1] (OIv7)
          thousandth_1000: already in target (passthrough)
        """
        if not bbox or len(bbox) < 4:
            return None

        sys = CoordSystem(self.source_system)

        if sys == CoordSystem.PIXEL_XYWH:
            x, y, w, h = bbox[:4]
            img_w, img_h = self.image_wh
            x1 = int(round(x * TARGET_RANGE / img_w))
            y1 = int(round(y * TARGET_RANGE / img_h))
            x2 = int(round((x + w) * TARGET_RANGE / img_w))
            y2 = int(round((y + h) * TARGET_RANGE / img_h))

        elif sys == CoordSystem.PIXEL_XYXY:
            x1_px, y1_px, x2_px, y2_px = bbox[:4]
            img_w, img_h = self.image_wh
            x1 = int(round(x1_px * TARGET_RANGE / img_w))
            y1 = int(round(y1_px * TARGET_RANGE / img_h))
            x2 = int(round(x2_px * TARGET_RANGE / img_w))
            y2 = int(round(y2_px * TARGET_RANGE / img_h))

        elif sys == CoordSystem.NORM_01:
            x1 = int(round(bbox[0] * TARGET_RANGE))
            y1 = int(round(bbox[1] * TARGET_RANGE))
            x2 = int(round(bbox[2] * TARGET_RANGE))
            y2 = int(round(bbox[3] * TARGET_RANGE))

        elif sys == CoordSystem.NORM_01_XMIN_XMAX:
            # OIv7 format: [XMin, XMax, YMin, YMax]
            xmin, xmax, ymin, ymax = bbox[:4]
            x1 = int(round(xmin * TARGET_RANGE))
            y1 = int(round(ymin * TARGET_RANGE))
            x2 = int(round(xmax * TARGET_RANGE))
            y2 = int(round(ymax * TARGET_RANGE))

        elif sys == CoordSystem.PERCENT_100:
            x1 = int(round(bbox[0] * TARGET_RANGE / 100))
            y1 = int(round(bbox[1] * TARGET_RANGE / 100))
            x2 = int(round(bbox[2] * TARGET_RANGE / 100))
            y2 = int(round(bbox[3] * TARGET_RANGE / 100))

        elif sys == CoordSystem.THOUSANDTH_1000:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        else:
            return None

        # Clamp and validate
        x1 = max(0, min(x1, TARGET_RANGE - 1))
        y1 = max(0, min(y1, TARGET_RANGE - 1))
        x2 = max(0, min(x2, TARGET_RANGE - 1))
        y2 = max(0, min(y2, TARGET_RANGE - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]

    def normalize_point(self, point: List[Union[int, float]]) -> Optional[List[int]]:
        """Convert point to [x, y] in [0, 1000)."""
        if not point or len(point) < 2:
            return None

        sys = CoordSystem(self.source_system)
        x_raw, y_raw = point[0], point[1]

        if sys in (CoordSystem.PIXEL_XYWH, CoordSystem.PIXEL_XYXY):
            img_w, img_h = self.image_wh
            x = int(round(x_raw * TARGET_RANGE / img_w))
            y = int(round(y_raw * TARGET_RANGE / img_h))

        elif sys == CoordSystem.NORM_01:
            x = int(round(x_raw * TARGET_RANGE))
            y = int(round(y_raw * TARGET_RANGE))

        elif sys == CoordSystem.PERCENT_100:
            x = int(round(x_raw * TARGET_RANGE / 100))
            y = int(round(y_raw * TARGET_RANGE / 100))

        elif sys == CoordSystem.THOUSANDTH_1000:
            x, y = int(x_raw), int(y_raw)

        else:
            return None

        x = max(0, min(x, TARGET_RANGE - 1))
        y = max(0, min(y, TARGET_RANGE - 1))
        return [x, y]

    def normalize_polygon(self, polygon: List[Union[int, float]]) -> Optional[List[int]]:
        """Convert flat polygon [x1,y1,x2,y2,...] to thousandth."""
        if not polygon or len(polygon) < 4 or len(polygon) % 2 != 0:
            return None
        result = []
        for i in range(0, len(polygon), 2):
            pt = self.normalize_point([polygon[i], polygon[i + 1]])
            if pt is None:
                return None
            result.extend(pt)
        return result


def detect_coord_system(values: List[float], image_wh: Optional[Tuple[int, int]] = None) -> CoordSystem:
    """Heuristic detection of coordinate system from sample values.

    Rules:
      - All values in [0, 1.0] and max < 1.05 → norm_01
      - All values in [0, 100] and max < 105 → percent_100
      - All values in [0, 1000] and max < 1005 → thousandth_1000
      - Otherwise (max > 1000 or matches image dims) → pixel
    """
    if not values:
        return CoordSystem.THOUSANDTH_1000

    max_val = max(abs(v) for v in values)
    min_val = min(values)

    if min_val < 0:
        # Negative values suggest pixel coords with possible errors
        return CoordSystem.PIXEL_XYXY

    if max_val <= 1.05:
        return CoordSystem.NORM_01
    elif max_val <= 105:
        return CoordSystem.PERCENT_100
    elif max_val <= 1005:
        return CoordSystem.THOUSANDTH_1000
    else:
        return CoordSystem.PIXEL_XYXY


def batch_normalize(
    annotations: List[dict],
    source_system: str,
    image_wh_key: str = "image_size",
    bbox_key: str = "bbox",
    point_key: str = "grasp_point_2d",
) -> List[dict]:
    """Batch normalize a list of annotation dicts.

    Modifies annotations in-place, adding _normalized suffix fields.
    """
    stats = {"total": 0, "bbox_ok": 0, "point_ok": 0, "failed": 0}

    for ann in annotations:
        stats["total"] += 1
        image_wh = ann.get(image_wh_key)
        if isinstance(image_wh, (list, tuple)) and len(image_wh) >= 2:
            wh = (image_wh[0], image_wh[1])
        else:
            wh = None

        try:
            norm = CoordinateNormalizer(source_system=source_system, image_wh=wh)
        except ValueError:
            stats["failed"] += 1
            continue

        # Normalize bbox
        bbox = ann.get(bbox_key)
        if bbox:
            norm_bbox = norm.normalize_bbox(bbox)
            if norm_bbox:
                ann[f"{bbox_key}_1000"] = norm_bbox
                stats["bbox_ok"] += 1

        # Normalize point
        point = ann.get(point_key)
        if point:
            norm_pt = norm.normalize_point(point)
            if norm_pt:
                ann[f"{point_key}_1000"] = norm_pt
                stats["point_ok"] += 1

    return annotations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Coordinate normalizer test")
    parser.add_argument("--test", action="store_true", help="Run self-tests")
    args = parser.parse_args()

    if args.test:
        print("Running coordinate normalizer tests...")

        # Test pixel_xywh → thousandth
        norm = CoordinateNormalizer("pixel_xywh", image_wh=(1024, 768))
        result = norm.normalize_bbox([100, 200, 300, 400])
        assert result == [98, 260, 391, 781], f"pixel_xywh bbox failed: {result}"
        print(f"  pixel_xywh bbox [100,200,300,400] @ 1024x768 → {result}")

        # Test pixel_xyxy
        norm2 = CoordinateNormalizer("pixel_xyxy", image_wh=(1920, 1080))
        result2 = norm2.normalize_bbox([960, 540, 1440, 810])
        assert result2 == [500, 500, 750, 750], f"pixel_xyxy failed: {result2}"
        print(f"  pixel_xyxy bbox [960,540,1440,810] @ 1920x1080 → {result2}")

        # Test norm_01
        norm3 = CoordinateNormalizer("norm_01")
        result3 = norm3.normalize_bbox([0.1, 0.2, 0.5, 0.8])
        assert result3 == [100, 200, 500, 800], f"norm_01 failed: {result3}"
        print(f"  norm_01 bbox [0.1,0.2,0.5,0.8] → {result3}")

        # Test percent_100
        norm4 = CoordinateNormalizer("percent_100")
        result4 = norm4.normalize_point([50.0, 75.0])
        assert result4 == [500, 750], f"percent_100 point failed: {result4}"
        print(f"  percent_100 point [50,75] → {result4}")

        # Test OIv7 format
        norm5 = CoordinateNormalizer("norm_01_xmin_xmax")
        result5 = norm5.normalize_bbox([0.2, 0.8, 0.3, 0.9])  # XMin,XMax,YMin,YMax
        assert result5 == [200, 300, 800, 900], f"OIv7 failed: {result5}"
        print(f"  norm_01_xmin_xmax [0.2,0.8,0.3,0.9] → {result5}")

        # Test detection
        detected = detect_coord_system([0.1, 0.5, 0.3, 0.8])
        assert detected == CoordSystem.NORM_01
        detected2 = detect_coord_system([50, 30, 80, 90])
        assert detected2 == CoordSystem.PERCENT_100
        detected3 = detect_coord_system([100, 200, 500, 800])
        assert detected3 == CoordSystem.THOUSANDTH_1000
        detected4 = detect_coord_system([1920, 540, 1440, 1080])
        assert detected4 == CoordSystem.PIXEL_XYXY
        print(f"  detect_coord_system: all heuristics pass")

        print("\nAll tests passed!")
