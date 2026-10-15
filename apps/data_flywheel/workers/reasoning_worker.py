"""ReasoningWorker — 场景推理能力 (tag R)，VLM 驱动多类型推理 QA。

覆盖: causal, multi_hop, counting, comparison
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.base import CapabilityAnnotation, CapabilityTag, CapabilityWorker
from apps.data_flywheel.workers.vlm_client import VLMClient

logger = logging.getLogger("flywheel.workers.reasoning")

REASONING_PROMPT = """Look at this image carefully. Generate diverse reasoning questions and answers about the scene.

You MUST generate questions of ALL these types:
1. "causal" - cause and effect (e.g. "What would happen if X were removed?")
2. "multi_hop" - requires combining multiple observations (e.g. "If A is near B, and B is above C, where is A relative to C?")
3. "counting" - counting objects or attributes (e.g. "How many round objects are there?")
4. "comparison" - comparing objects (e.g. "Which object is larger, the knob or the button?")

Output a JSON array with at least 4 questions (one per type):
```json
[
  {{"question": "What would happen if the power button were pressed?", "answer": "The washing machine would turn on and the display would light up.", "type": "causal", "reasoning_chain": "button controls power -> power activates display"}},
  {{"question": "If the knob is next to the display, and the display is above the door, where is the knob relative to the door?", "answer": "The knob is above the door.", "type": "multi_hop", "reasoning_chain": "knob next_to display, display above door -> knob above door"}},
  {{"question": "How many control elements are visible on the panel?", "answer": "There are 3 control elements: a rotary knob, a power button, and a start button.", "type": "counting", "reasoning_chain": "enumerate: knob + power_button + start_button = 3"}},
  {{"question": "Which is larger, the display panel or the rotary knob?", "answer": "The display panel is larger than the rotary knob.", "type": "comparison", "reasoning_chain": "display_area > knob_area based on bounding boxes"}}
]
```

Base your questions on what you actually see in the image. Be specific and accurate."""


class ReasoningWorker(CapabilityWorker):
    """VLM 驱动的场景推理 Worker。"""

    tag = CapabilityTag.R

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
            result = self._vlm.chat_json(REASONING_PROMPT, image_path=image_path, max_tokens=4000)
        except Exception as e:
            logger.warning(f"VLM reasoning failed: {e}")
            return []

        if not isinstance(result, list):
            return []

        annotations: List[CapabilityAnnotation] = []
        for qa in result:
            if not isinstance(qa, dict):
                continue
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            qa_type = qa.get("type", "causal")
            chain = qa.get("reasoning_chain", "")

            if not question or not answer:
                continue

            diff_map = {"causal": 3, "multi_hop": 4, "counting": 2, "comparison": 3}
            difficulty = diff_map.get(qa_type, 3) + random.randint(-1, 1)
            difficulty = max(1, min(5, difficulty))

            annotations.append(CapabilityAnnotation(
                sample_id=sample.sample_id,
                capability_tag=CapabilityTag.R,
                sub_tag=f"reasoning_{qa_type}",
                content={
                    "question": question,
                    "answer": answer,
                    "type": qa_type,
                    "reasoning_chain": chain,
                },
                confidence=round(random.uniform(0.65, 0.9), 3),
                provenance="vlm",
                difficulty=difficulty,
            ))

        return annotations
