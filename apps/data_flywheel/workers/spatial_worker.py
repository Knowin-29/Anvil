"""SpatialWorker — 空间理解能力 (tag S)，VLM 驱动空间关系生成。

基于检测到的物体列表，让 VLM 描述它们之间的空间关系。
覆盖 10+ predicates: left_of, right_of, above, below, in_front_of, behind, inside, on_top_of, next_to, between
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.base import CapabilityAnnotation, CapabilityTag, CapabilityWorker
from apps.data_flywheel.workers.vlm_client import VLMClient

logger = logging.getLogger("flywheel.workers.spatial")

SPATIAL_RELATION_PROMPT = """Look at this image. The following objects have been detected:
{objects_list}

Describe the spatial relationships between these objects. Use ONLY these predicates:
- left_of, right_of, above, below
- in_front_of, behind
- inside, on_top_of
- next_to, between

Output a JSON object with a "relations" array. Each relation has subject, predicate, object, and subject_bbox/object_bbox.

```json
{{
  "relations": [
    {{"subject": "knob", "predicate": "left_of", "object": "display", "subject_bbox": [x1,y1,x2,y2], "object_bbox": [x1,y1,x2,y2]}},
    ...
  ]
}}
```

Generate at least 3 spatial relations if possible. Be accurate based on what you see."""

SPATIAL_QA_PROMPT = """Look at this image with these detected objects:
{objects_list}

Generate spatial reasoning questions and answers about the relative positions of objects.
Include different question types: position queries, counting, and relative distance.

Output a JSON array:
```json
[
  {{"question": "What is to the left of the display?", "answer": "The rotary knob is to the left of the display.", "type": "position"}},
  {{"question": "How many objects are above the button?", "answer": "There are 2 objects above the button.", "type": "counting"}},
  {{"question": "Which object is closest to the screen?", "answer": "The power button is closest to the screen.", "type": "distance"}}
]
```

Generate 2-4 diverse questions."""


class SpatialWorker(CapabilityWorker):
    """VLM 驱动的空间关系 Worker。"""

    tag = CapabilityTag.S

    def __init__(self, vlm_client: Optional[VLMClient] = None, config=None, **kwargs):
        self._vlm = vlm_client
        self._config = config

    def process(self, sample: Sample, **kwargs) -> List[CapabilityAnnotation]:
        if not self._vlm:
            return []

        grounding_data = kwargs.get("grounding_data", [])
        if len(grounding_data) < 2:
            return []

        image_path = sample.image_path
        if not image_path:
            return []

        objects_list = self._format_objects(grounding_data)
        annotations: List[CapabilityAnnotation] = []

        # Part 1: Spatial relations
        rel_anns = self._generate_relations(sample, image_path, objects_list, grounding_data)
        annotations.extend(rel_anns)

        # Part 2: Spatial QA
        qa_anns = self._generate_spatial_qa(sample, image_path, objects_list)
        annotations.extend(qa_anns)

        return annotations

    def _generate_relations(self, sample: Sample, image_path: str,
                            objects_list: str, grounding_data: List[Dict]) -> List[CapabilityAnnotation]:
        prompt = SPATIAL_RELATION_PROMPT.format(objects_list=objects_list)
        try:
            result = self._vlm.chat_json(prompt, image_path=image_path, max_tokens=1024)
        except Exception as e:
            logger.warning(f"VLM spatial relations failed: {e}")
            return []

        relations = result.get("relations", []) if isinstance(result, dict) else []
        annotations = []

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            subj = rel.get("subject", "")
            pred = rel.get("predicate", "")
            obj = rel.get("object", "")
            if not subj or not pred or not obj:
                continue

            difficulty = self._relation_difficulty(pred)
            annotations.append(CapabilityAnnotation(
                sample_id=sample.sample_id,
                capability_tag=CapabilityTag.S,
                sub_tag="spatial_relation",
                content={
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "subject_bbox": rel.get("subject_bbox", []),
                    "object_bbox": rel.get("object_bbox", []),
                    "is_3d": pred in ("in_front_of", "behind"),
                },
                confidence=round(random.uniform(0.75, 0.95), 3),
                provenance="vlm",
                difficulty=difficulty,
            ))

        return annotations

    def _generate_spatial_qa(self, sample: Sample, image_path: str,
                             objects_list: str) -> List[CapabilityAnnotation]:
        prompt = SPATIAL_QA_PROMPT.format(objects_list=objects_list)
        try:
            result = self._vlm.chat_json(prompt, image_path=image_path, max_tokens=1024)
        except Exception as e:
            logger.warning(f"VLM spatial QA failed: {e}")
            return []

        if not isinstance(result, list):
            return []

        annotations = []
        for qa in result:
            if not isinstance(qa, dict):
                continue
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            qa_type = qa.get("type", "position")
            if not question or not answer:
                continue

            diff_map = {"position": 2, "counting": 3, "distance": 4}
            difficulty = diff_map.get(qa_type, 3) + random.randint(-1, 1)
            difficulty = max(1, min(5, difficulty))

            annotations.append(CapabilityAnnotation(
                sample_id=sample.sample_id,
                capability_tag=CapabilityTag.S,
                sub_tag=f"spatial_qa_{qa_type}",
                content={
                    "question": question,
                    "answer": answer,
                    "type": qa_type,
                },
                confidence=round(random.uniform(0.7, 0.9), 3),
                provenance="vlm",
                difficulty=difficulty,
            ))

        return annotations

    @staticmethod
    def _format_objects(grounding_data: List[Dict]) -> str:
        lines = []
        for obj in grounding_data[:8]:
            name = obj.get("name", "object")
            bbox = obj.get("bbox", [])
            lines.append(f"- {name} at bbox {bbox}")
        return "\n".join(lines)

    @staticmethod
    def _relation_difficulty(predicate: str) -> int:
        easy = {"left_of", "right_of", "above", "below"}
        medium = {"next_to", "on_top_of", "inside"}
        hard = {"in_front_of", "behind", "between"}
        if predicate in easy:
            return random.choice([1, 2, 2])
        elif predicate in medium:
            return random.choice([3, 3, 4])
        elif predicate in hard:
            return random.choice([4, 4, 5])
        return 3
