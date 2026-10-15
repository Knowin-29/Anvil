"""MultiturnWorker — 多轮对话一致性能力 (tag E)，VLM 驱动多轮对话生成。

覆盖: consistency, correction, followup 三种对话类型。
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.base import CapabilityAnnotation, CapabilityTag, CapabilityWorker
from apps.data_flywheel.workers.vlm_client import VLMClient

logger = logging.getLogger("flywheel.workers.multiturn")

MULTITURN_PROMPT = """Look at this image. Generate multi-turn dialogue examples for training a visual AI assistant.

The following objects are in the image:
{objects_list}

Generate 2-3 dialogues of different types:
1. "consistency" - User asks about the same object in different ways; assistant must give consistent answers
2. "correction" - User points out an error in assistant's response; assistant corrects itself
3. "followup" - User asks follow-up questions that build on previous answers

Output a JSON array:
```json
[
  {{
    "turns": [
      {{"role": "user", "content": "What is the round object on the left side?"}},
      {{"role": "assistant", "content": "That is a rotary knob used for selecting wash cycles."}},
      {{"role": "user", "content": "What color is it?"}},
      {{"role": "assistant", "content": "The rotary knob is white/silver colored."}}
    ],
    "type": "consistency",
    "key_object": "rotary knob"
  }},
  {{
    "turns": [
      {{"role": "user", "content": "Is that a red button on the right?"}},
      {{"role": "assistant", "content": "I apologize, but that is not a red button. It appears to be a white/gray power button."}},
      {{"role": "user", "content": "OK, what does it do?"}},
      {{"role": "assistant", "content": "The power button is used to turn the washing machine on and off."}}
    ],
    "type": "correction",
    "key_object": "power button"
  }},
  {{
    "turns": [
      {{"role": "user", "content": "Describe what you see in this image."}},
      {{"role": "assistant", "content": "I see a washing machine control panel with a rotary knob, display, and several buttons."}},
      {{"role": "user", "content": "How many buttons are there?"}},
      {{"role": "assistant", "content": "I can count 3 buttons on the control panel."}},
      {{"role": "user", "content": "Which one is the start button?"}},
      {{"role": "assistant", "content": "The start button is the one on the far right, typically marked with a play/triangle symbol."}}
    ],
    "type": "followup",
    "key_object": "control panel"
  }}
]
```

Make dialogues natural and grounded in what you actually see."""


class MultiturnWorker(CapabilityWorker):
    """VLM 驱动的多轮对话 Worker。"""

    tag = CapabilityTag.E

    def __init__(self, vlm_client: Optional[VLMClient] = None, config=None, **kwargs):
        self._vlm = vlm_client
        self._config = config

    def process(self, sample: Sample, **kwargs) -> List[CapabilityAnnotation]:
        if not self._vlm:
            return []

        grounding_data = kwargs.get("grounding_data", [])
        image_path = sample.image_path
        if not image_path:
            return []

        objects_list = "\n".join(
            f"- {obj.get('name', 'object')}" for obj in grounding_data[:6]
        ) if grounding_data else "- (describe what you see)"

        prompt = MULTITURN_PROMPT.format(objects_list=objects_list)

        try:
            result = self._vlm.chat_json(prompt, image_path=image_path, max_tokens=4000)
        except Exception as e:
            logger.warning(f"VLM multiturn failed: {e}")
            return []

        if not isinstance(result, list):
            return []

        annotations: List[CapabilityAnnotation] = []
        for dialogue in result:
            if not isinstance(dialogue, dict):
                continue
            turns = dialogue.get("turns", [])
            d_type = dialogue.get("type", "followup")
            key_obj = dialogue.get("key_object", "")

            if not turns or len(turns) < 3:
                continue

            diff_map = {"consistency": 3, "correction": 4, "followup": 3}
            num_turns = len(turns)
            difficulty = diff_map.get(d_type, 3) + (1 if num_turns > 4 else 0)
            difficulty = max(1, min(5, difficulty + random.randint(-1, 0)))

            annotations.append(CapabilityAnnotation(
                sample_id=sample.sample_id,
                capability_tag=CapabilityTag.E,
                sub_tag=f"multiturn_{d_type}",
                content={
                    "turns": turns,
                    "type": d_type,
                    "key_object": key_obj,
                    "num_turns": num_turns,
                },
                confidence=round(random.uniform(0.7, 0.9), 3),
                provenance="vlm",
                difficulty=difficulty,
            ))

        return annotations
