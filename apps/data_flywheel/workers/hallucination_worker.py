"""HallucinationWorker — 幻觉控制能力 (tag H)，VLM 驱动否定+反事实样本。

让 VLM 看图后生成:
1. negative: 确认某物体不存在
2. counterfactual: 对存在物体做错误描述
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.base import CapabilityAnnotation, CapabilityTag, CapabilityWorker
from apps.data_flywheel.workers.vlm_client import VLMClient

logger = logging.getLogger("flywheel.workers.hallucination")

HALLUCINATION_PROMPT = """Look at this image carefully. I need you to generate hallucination-control training data.

Generate two types of examples:
1. "negative" - Ask about objects that are clearly NOT in the image. The answer should correctly deny their presence.
2. "counterfactual" - Make a FALSE statement about an object that IS in the image, then provide the correction.

The following objects ARE in the image:
{objects_list}

Output a JSON array with 3-5 examples:
```json
[
  {{
    "question": "Is there an elephant in this image?",
    "answer": "No, there is no elephant in this image. The image shows a washing machine control panel.",
    "type": "negative",
    "absent_object": "elephant"
  }},
  {{
    "question": "Is the rotary knob colored bright red?",
    "answer": "No, the rotary knob is not red. It appears to be white/silver colored.",
    "type": "counterfactual",
    "target_object": "rotary knob",
    "false_claim": "bright red color",
    "true_description": "white/silver colored"
  }},
  {{
    "question": "Can you see a bicycle in the background?",
    "answer": "No, there is no bicycle visible in this image.",
    "type": "negative",
    "absent_object": "bicycle"
  }}
]
```

Choose absent objects that are plausible distractors but clearly not present. For counterfactuals, make claims that are obviously wrong based on what you see."""


class HallucinationWorker(CapabilityWorker):
    """VLM 驱动的幻觉控制 Worker。"""

    tag = CapabilityTag.H

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
        ) if grounding_data else "- (detect objects yourself)"

        prompt = HALLUCINATION_PROMPT.format(objects_list=objects_list)

        try:
            result = self._vlm.chat_json(prompt, image_path=image_path, max_tokens=4000)
        except Exception as e:
            logger.warning(f"VLM hallucination failed: {e}")
            return []

        if not isinstance(result, list):
            return []

        annotations: List[CapabilityAnnotation] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            question = item.get("question", "")
            answer = item.get("answer", "")
            h_type = item.get("type", "negative")

            if not question or not answer:
                continue

            diff_map = {"negative": 2, "counterfactual": 4}
            difficulty = diff_map.get(h_type, 3) + random.randint(-1, 1)
            difficulty = max(1, min(5, difficulty))

            content = {
                "question": question,
                "answer": answer,
                "type": h_type,
            }
            if h_type == "negative":
                content["absent_object"] = item.get("absent_object", "")
            else:
                content["target_object"] = item.get("target_object", "")
                content["false_claim"] = item.get("false_claim", "")
                content["true_description"] = item.get("true_description", "")

            annotations.append(CapabilityAnnotation(
                sample_id=sample.sample_id,
                capability_tag=CapabilityTag.H,
                sub_tag=f"hallucination_{h_type}",
                content=content,
                confidence=round(random.uniform(0.8, 0.95), 3),
                provenance="vlm",
                difficulty=difficulty,
            ))

        return annotations
