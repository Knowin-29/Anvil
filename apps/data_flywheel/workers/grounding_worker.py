"""GroundingWorker — 视觉定位能力 (tag G)，VLM 驱动物体检测。

通过 VLM 看图检测所有可见物体，输出名称+bbox。
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.base import CapabilityAnnotation, CapabilityTag, CapabilityWorker
from apps.data_flywheel.workers.vlm_client import VLMClient

logger = logging.getLogger("flywheel.workers.grounding")

GROUNDING_PROMPT = """You are an expert object detector. Analyze this image carefully and detect ALL visible objects.

For each object, provide:
- "name": the object name in English (be specific, e.g. "rotary knob" not just "object")
- "bbox": bounding box as [x1, y1, x2, y2] in pixel coordinates
- "confidence": your confidence score (0.0-1.0)

The image size is {width}x{height} pixels. All coordinates must be within these bounds.

Output ONLY a JSON array, no other text:
```json
[
  {{"name": "object_name", "bbox": [x1, y1, x2, y2], "confidence": 0.9}},
  ...
]
```

Detect at least 3 objects if visible. Be precise with bounding boxes."""


class GroundingWorker(CapabilityWorker):
    """VLM 驱动的物体检测 Worker。"""

    tag = CapabilityTag.G

    def __init__(self, vlm_client: Optional[VLMClient] = None, config=None, **kwargs):
        self._vlm = vlm_client
        self._config = config

    def process(self, sample: Sample, **kwargs) -> List[CapabilityAnnotation]:
        if not self._vlm:
            logger.warning("No VLM client, skipping grounding")
            return []

        image_path = sample.image_path
        if not image_path:
            return []

        try:
            w, h = self._vlm.get_image_size(image_path)
        except Exception:
            w, h = 640, 480

        prompt = GROUNDING_PROMPT.format(width=w, height=h)

        try:
            detections = self._vlm.chat_json(prompt, image_path=image_path, max_tokens=1024)
        except Exception as e:
            logger.warning(f"VLM grounding failed: {e}")
            return []

        if not isinstance(detections, list):
            return []

        annotations: List[CapabilityAnnotation] = []
        num_objects = len(detections)

        for det in detections:
            if not isinstance(det, dict):
                continue
            name = det.get("name", "")
            bbox = det.get("bbox", [])
            conf = det.get("confidence", 0.7)

            if not name or not bbox or len(bbox) < 4:
                continue

            bbox = [int(c) for c in bbox]
            bbox[0] = max(0, min(w, bbox[0]))
            bbox[1] = max(0, min(h, bbox[1]))
            bbox[2] = max(0, min(w, bbox[2]))
            bbox[3] = max(0, min(h, bbox[3]))

            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (w * h)
            difficulty = self._compute_difficulty(area_ratio, num_objects)

            sub_tag = "grounding_single" if num_objects <= 2 else "grounding_multi"

            annotations.append(CapabilityAnnotation(
                sample_id=sample.sample_id,
                capability_tag=CapabilityTag.G,
                sub_tag=sub_tag,
                content={
                    "phrase": name,
                    "name": name,
                    "boxes": [bbox],
                    "bbox": bbox,
                    "image_path": image_path,
                    "image_size": [w, h],
                },
                confidence=round(float(conf), 3),
                provenance="vlm",
                difficulty=difficulty,
            ))

        return annotations

    @staticmethod
    def _compute_difficulty(area_ratio: float, num_objects: int) -> int:
        base = 2
        if area_ratio < 0.01:
            base += 2
        elif area_ratio < 0.05:
            base += 1
        if num_objects > 5:
            base += 1
        return max(1, min(5, base + random.randint(-1, 0)))
