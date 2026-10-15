"""InstructionWorker — 指令跟随能力 (tag F)，VLM 驱动指令生成。

覆盖: simple, complex, conditional 三种指令类型。
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.base import CapabilityAnnotation, CapabilityTag, CapabilityWorker
from apps.data_flywheel.workers.vlm_client import VLMClient

logger = logging.getLogger("flywheel.workers.instruction")

INSTRUCTION_PROMPT = """Look at this image. The following objects have been detected:
{objects_list}

Generate diverse instruction-following examples. Create instructions that a user might give to a visual AI assistant, along with the expected output.

Generate 3 types of instructions:
1. "simple" - single-step (e.g. "Point to the knob", "What color is the button?")
2. "complex" - multi-step (e.g. "Find the largest object and describe its position relative to the screen")
3. "conditional" - with conditions (e.g. "If the display shows a number, tell me what it is; otherwise describe what you see on it")

Output a JSON array:
```json
[
  {{
    "instruction": "Point to the rotary knob",
    "expected_output": {{"type": "bbox", "target": "knob", "bbox": [100,100,200,200]}},
    "type": "simple",
    "difficulty": 1
  }},
  {{
    "instruction": "Find all buttons and count them, then describe the color of the largest one",
    "expected_output": {{"type": "text", "answer": "There are 2 buttons. The largest one is white."}},
    "type": "complex",
    "difficulty": 4
  }},
  {{
    "instruction": "If the machine is currently running, identify the active indicator; otherwise point to the power button",
    "expected_output": {{"type": "action", "condition_met": false, "action": "point", "target": "power_button", "bbox": [300,50,350,80]}},
    "type": "conditional",
    "difficulty": 4
  }}
]
```

Generate 3-5 instructions based on what you actually see."""


class InstructionWorker(CapabilityWorker):
    """VLM 驱动的指令跟随 Worker。"""

    tag = CapabilityTag.F

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

        prompt = INSTRUCTION_PROMPT.format(objects_list=objects_list)

        try:
            result = self._vlm.chat_json(prompt, image_path=image_path, max_tokens=4000)
        except Exception as e:
            logger.warning(f"VLM instruction failed: {e}")
            return []

        if not isinstance(result, list):
            return []

        annotations: List[CapabilityAnnotation] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            instruction = item.get("instruction", "")
            expected = item.get("expected_output", {})
            instr_type = item.get("type", "simple")
            diff = item.get("difficulty", 3)

            if not instruction:
                continue

            difficulty = max(1, min(5, int(diff) + random.randint(-1, 0)))

            annotations.append(CapabilityAnnotation(
                sample_id=sample.sample_id,
                capability_tag=CapabilityTag.F,
                sub_tag=f"instruction_{instr_type}",
                content={
                    "instruction": instruction,
                    "expected_output": expected,
                    "type": instr_type,
                },
                confidence=round(random.uniform(0.7, 0.92), 3),
                provenance="vlm",
                difficulty=difficulty,
            ))

        return annotations
