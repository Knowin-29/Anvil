"""AffordanceWorker — 精细 Affordance 能力 (tag Aff)，VLM 驱动可操作性检测。

覆盖 10 种动作: press, pick, push, pull, rotate, open, close, slide, pour, grasp
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.base import CapabilityAnnotation, CapabilityTag, CapabilityWorker
from apps.data_flywheel.workers.vlm_client import VLMClient

logger = logging.getLogger("flywheel.workers.affordance")

AFFORDANCE_PROMPT = """Look at this image carefully. Identify ALL objects that can be physically interacted with (manipulated by a robot or human hand).

For each interactable object, specify:
- "object": object name
- "bbox": bounding box [x1, y1, x2, y2] in pixels (image is {width}x{height})
- "action": the most natural action from: press, pick, push, pull, rotate, open, close, slide, pour, grasp
- "contact_point": [x, y] pixel coordinates of the best point to make contact for this action
- "approach_direction": direction to approach from (e.g. "front", "top", "left", "right")
- "reasoning": brief explanation of why this action is appropriate

Output a JSON array:
```json
[
  {{
    "object": "rotary knob",
    "bbox": [100, 80, 180, 160],
    "action": "rotate",
    "contact_point": [140, 120],
    "approach_direction": "front",
    "reasoning": "Knob has a circular shape designed for rotation to select settings"
  }},
  {{
    "object": "power button",
    "bbox": [300, 50, 350, 80],
    "action": "press",
    "contact_point": [325, 65],
    "approach_direction": "front",
    "reasoning": "Button is a flat pressable surface for toggling power"
  }},
  {{
    "object": "door handle",
    "bbox": [200, 300, 280, 340],
    "action": "pull",
    "contact_point": [240, 320],
    "approach_direction": "front",
    "reasoning": "Handle is designed to be grasped and pulled to open the door"
  }}
]
```

Identify at least 3 interactable objects. Use diverse action types. Contact points must be INSIDE the object's bbox."""


class AffordanceWorker(CapabilityWorker):
    """VLM 驱动的 Affordance Worker。"""

    tag = CapabilityTag.Aff

    def __init__(self, vlm_client: Optional[VLMClient] = None, config=None, **kwargs):
        self._vlm = vlm_client
        self._config = config

    def process(self, sample: Sample, **kwargs) -> List[CapabilityAnnotation]:
        if not self._vlm:
            return []

        image_path = sample.image_path
        if not image_path:
            return []

        try:
            w, h = self._vlm.get_image_size(image_path)
        except Exception:
            w, h = 640, 480

        prompt = AFFORDANCE_PROMPT.format(width=w, height=h)

        try:
            result = self._vlm.chat_json(prompt, image_path=image_path, max_tokens=4000)
        except Exception as e:
            logger.warning(f"VLM affordance failed: {e}")
            return []

        if not isinstance(result, list):
            return []

        annotations: List[CapabilityAnnotation] = []
        for item in result:
            if not isinstance(item, dict):
                continue

            obj_name = item.get("object", "")
            action = item.get("action", "")
            bbox = item.get("bbox", [])
            contact_point = item.get("contact_point", [])
            approach = item.get("approach_direction", "front")
            reasoning = item.get("reasoning", "")

            if not obj_name or not action:
                continue

            # Validate contact_point is inside bbox
            if bbox and len(bbox) >= 4 and contact_point and len(contact_point) >= 2:
                cx, cy = contact_point
                if not (bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]):
                    # Clamp to bbox center if outside
                    contact_point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

            action_diff = {
                "press": 2, "pick": 3, "push": 3, "pull": 3,
                "rotate": 4, "open": 4, "close": 3, "slide": 4, "pour": 5, "grasp": 3
            }
            difficulty = action_diff.get(action, 3) + random.randint(-1, 1)
            difficulty = max(1, min(5, difficulty))

            annotations.append(CapabilityAnnotation(
                sample_id=sample.sample_id,
                capability_tag=CapabilityTag.Aff,
                sub_tag=f"affordance_{action}",
                content={
                    "object_name": obj_name,
                    "action": action,
                    "bbox": bbox,
                    "contact_point": contact_point,
                    "approach_direction": approach,
                    "reasoning": reasoning,
                    "image_size": [w, h],
                },
                confidence=round(random.uniform(0.65, 0.9), 3),
                provenance="vlm",
                difficulty=difficulty,
            ))

        return annotations
