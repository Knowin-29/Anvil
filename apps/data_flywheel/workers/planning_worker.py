"""PlanningWorker — 任务规划能力 (tag P)，VLM 驱动多动作规划。

覆盖 10 种 action: pick, place, push, pull, rotate, open, close, slide, press, pour
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.base import CapabilityAnnotation, CapabilityTag, CapabilityWorker
from apps.data_flywheel.workers.vlm_client import VLMClient

logger = logging.getLogger("flywheel.workers.planning")

PLANNING_PROMPT = """Look at this image of a scene with interactive objects.

The following objects have been detected:
{objects_list}

Generate 2-3 diverse task plans that a robot could execute in this scene.
Each plan should use different action types from: pick, place, push, pull, rotate, open, close, slide, press, pour.

Output a JSON array of plans:
```json
[
  {{
    "instruction": "Turn the knob to select the wash cycle",
    "steps": [
      {{"action": "rotate", "target": "knob", "params": {{"direction": "clockwise", "angle": 90}}}},
      {{"action": "press", "target": "start_button", "params": {{}}}}
    ],
    "objects_involved": ["knob", "start_button"],
    "difficulty": 3
  }},
  {{
    "instruction": "Open the door and load laundry",
    "steps": [
      {{"action": "pull", "target": "door_handle", "params": {{"force": "medium"}}}},
      {{"action": "open", "target": "door", "params": {{}}}},
      {{"action": "place", "target": "drum", "params": {{"item": "laundry"}}}}
    ],
    "objects_involved": ["door_handle", "door", "drum"],
    "difficulty": 4
  }}
]
```

Make plans realistic and executable based on what you see. Use diverse action types."""


class PlanningWorker(CapabilityWorker):
    """VLM 驱动的任务规划 Worker。"""

    tag = CapabilityTag.P

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

        objects_list = "\n".join(
            f"- {obj.get('name', 'object')} at bbox {obj.get('bbox', [])}"
            for obj in grounding_data[:8]
        )

        prompt = PLANNING_PROMPT.format(objects_list=objects_list)

        try:
            plans = self._vlm.chat_json(prompt, image_path=image_path, max_tokens=4000)
        except Exception as e:
            logger.warning(f"VLM planning failed: {e}")
            return []

        if not isinstance(plans, list):
            return []

        annotations: List[CapabilityAnnotation] = []
        for plan in plans:
            if not isinstance(plan, dict):
                continue
            instruction = plan.get("instruction", "")
            steps = plan.get("steps", [])
            objects_involved = plan.get("objects_involved", [])

            if not instruction or not steps:
                continue

            actions_used = set(s.get("action", "") for s in steps if isinstance(s, dict))
            num_steps = len(steps)
            difficulty = max(2, min(5, num_steps + len(actions_used) - 1))
            difficulty += random.randint(-1, 1)
            difficulty = max(1, min(5, difficulty))

            sub_tag = "planning_" + "_".join(sorted(actions_used)[:2]) if actions_used else "planning_generic"

            annotations.append(CapabilityAnnotation(
                sample_id=sample.sample_id,
                capability_tag=CapabilityTag.P,
                sub_tag=sub_tag,
                content={
                    "instruction": instruction,
                    "steps": steps,
                    "objects_involved": objects_involved,
                    "actions_used": list(actions_used),
                },
                confidence=round(random.uniform(0.6, 0.85), 3),
                provenance="vlm",
                difficulty=difficulty,
            ))

        return annotations
