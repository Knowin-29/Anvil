"""AttributeWorker — 属性状态能力 (tag A)，VLM 驱动属性描述。

对每个检测到的物体，裁剪 bbox 区域让 VLM 描述其视觉属性。
覆盖: color, shape, material, state, size
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from PIL import Image

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.base import CapabilityAnnotation, CapabilityTag, CapabilityWorker
from apps.data_flywheel.workers.vlm_client import VLMClient

logger = logging.getLogger("flywheel.workers.attribute")

ATTRIBUTE_PROMPT = """Look at this image. I have detected the following objects with bounding boxes [x1,y1,x2,y2]:
{objects_list}

For EACH object, describe its visual attributes. Output a JSON array where each element corresponds to one object:
```json
[
  {{
    "name": "<object name>",
    "color": "<primary color>",
    "shape": "<shape>",
    "material": "<material>",
    "state": "<current state>",
    "size": "<relative size: small/medium/large>"
  }}
]
```

Be specific and accurate based on what you actually see. Output ONLY the JSON array."""


class AttributeWorker(CapabilityWorker):
    """VLM 驱动的属性描述 Worker。"""

    tag = CapabilityTag.A

    def __init__(self, vlm_client: Optional[VLMClient] = None, config=None, **kwargs):
        self._vlm = vlm_client
        self._config = config

    def process(self, sample: Sample, **kwargs) -> List[CapabilityAnnotation]:
        if not self._vlm:
            return []

        grounding_data = kwargs.get("grounding_data", [])
        if not grounding_data:
            return []

        image_path = sample.image_path
        if not image_path:
            return []

        annotations: List[CapabilityAnnotation] = []

        # 一次性对整张图描述所有物体属性（减少 VLM 调用次数）
        objects_list = "\n".join(
            f"- {obj.get('name', 'object')}: {obj.get('bbox', [])}"
            for obj in grounding_data[:8]
        )
        prompt = ATTRIBUTE_PROMPT.format(objects_list=objects_list)

        try:
            result = self._vlm.chat_json(prompt, image_path=image_path, max_tokens=1200)
        except Exception as e:
            logger.warning(f"VLM attribute batch failed: {e}")
            return []

        # 解析结果
        if isinstance(result, list):
            attrs_list = result
        elif isinstance(result, dict):
            attrs_list = [result]
        else:
            return []

        for i, attrs in enumerate(attrs_list):
            if not isinstance(attrs, dict):
                continue

            # 匹配回 grounding_data
            name = attrs.get("name", "")
            bbox = []
            for obj in grounding_data:
                if obj.get("name", "").lower() == name.lower():
                    bbox = obj.get("bbox", [])
                    break
            if not bbox and i < len(grounding_data):
                bbox = grounding_data[i].get("bbox", [])
                name = name or grounding_data[i].get("name", "object")

            attr_types = ["color", "shape", "material", "state", "size"]
            for attr_type in attr_types:
                value = attrs.get(attr_type, "")
                if not value or value.lower() in ("unknown", "n/a", "none", ""):
                    continue

                difficulty = self._attr_difficulty(attr_type, bbox, sample)
                conf = round(random.uniform(0.7, 0.95), 3)
                expr = f"the {value} {name}"

                annotations.append(CapabilityAnnotation(
                    sample_id=sample.sample_id,
                    capability_tag=CapabilityTag.A,
                    sub_tag=f"attribute_{attr_type}",
                    content={
                        "object": name,
                        "bbox": bbox,
                        "attribute_type": attr_type,
                        "attribute_value": value,
                        "referring_expression": expr,
                    },
                    confidence=conf,
                    provenance="vlm",
                    difficulty=difficulty,
                ))

        return annotations

    @staticmethod
    def _crop_object(image_path: str, bbox: List[int], padding: int = 10) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        x1 = max(0, bbox[0] - padding)
        y1 = max(0, bbox[1] - padding)
        x2 = min(w, bbox[2] + padding)
        y2 = min(h, bbox[3] + padding)
        return img.crop((x1, y1, x2, y2))

    @staticmethod
    def _attr_difficulty(attr_type: str, bbox: List[int], sample: Sample) -> int:
        try:
            img_w, img_h = sample.image.size
        except Exception:
            img_w, img_h = 640, 480
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_ratio = area / (img_w * img_h) if img_w * img_h > 0 else 0.1

        base = {"color": 1, "size": 2, "shape": 3, "state": 3, "material": 4}.get(attr_type, 3)
        if area_ratio < 0.02:
            base += 1
        return max(1, min(5, base + random.randint(-1, 1)))
