"""VLM-driven referring expression generation."""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from PIL import Image

from apps.data_flywheel.scene_understanding.vlm.backend import VLMBackend
from apps.data_flywheel.types import SceneNode


REFERRING_PROMPT_TEMPLATE = """
Please describe the object at bbox {bbox} in the image in 3 different ways:
1. An English short phrase (<= 15 words)
2. A Chinese short phrase (中文,不超过 15 字)
3. An English phrase that references a nearby object

Context: The object's canonical name is "{hierarchical_name}".
Other objects in the image: {other_objects}

Return a JSON object with keys "en", "zh", "en_contextual":
{{"en": "...", "zh": "...", "en_contextual": "..."}}
""".strip()


_LOGGER = logging.getLogger("scene_understanding.referring")


def _normalize(value: str) -> str:
    return value.strip().lower()


def _fallback(node: SceneNode) -> List[str]:
    if node.hierarchical_name:
        return [node.hierarchical_name]
    return []


def generate_referring_expressions(
    node: SceneNode,
    all_nodes: List[SceneNode],
    image: Image.Image,
    vlm: Optional[VLMBackend],
) -> List[str]:
    if vlm is None:
        return []

    other_objects = [n.hierarchical_name or n.class_name for n in all_nodes if n.id != node.id][:10]
    prompt = REFERRING_PROMPT_TEMPLATE.format(
        bbox=node.bbox,
        hierarchical_name=node.hierarchical_name or node.class_name,
        other_objects=", ".join(other_objects),
    )

    try:
        response = vlm.ask(image, prompt)
        data = json.loads(response)
    except Exception as exc:
        _LOGGER.warning("VLM failed for node %s: %s", node.id, exc)
        return _fallback(node)

    if not isinstance(data, dict):
        return _fallback(node)

    values = []
    for key in ("en", "zh", "en_contextual"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())

    deduped: List[str] = []
    seen = set()
    for value in values:
        normalized = _normalize(value)
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(value)

    return deduped or _fallback(node)
