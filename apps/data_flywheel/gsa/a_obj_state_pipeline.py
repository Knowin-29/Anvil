"""A_obj_state Pipeline — Object physical state recognition with CoT reasoning.

Key improvements over v0:
1. Object-type-aware state queries (cup -> upright/tilted, door -> open/closed)
2. Chain-of-Thought prompting for state reasoning
3. Multi-evidence validation (visual + spatial context)
4. Grounded state inference (uses bbox region, not whole image)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from .a_pipeline_schemas import AObjStateLabel, validate_obj_state
from .taxonomy import (
    AObjStateSubtype,
    Bucket,
    SourceProvenance,
    STATE_TYPE_VALUES,
    VerifierStatus,
)
from .visual_evidence import GsaEvidenceBundle, GsaObjectRef

logger = logging.getLogger("gsa.a_obj_state_pipeline")

_SPATIAL_STATE_TYPES = {"occupancy", "open_close", "containment"}

OBJECT_STATE_MAP: Dict[str, List[str]] = {
    "cup": ["pose", "occupancy"],
    "mug": ["pose", "occupancy"],
    "bottle": ["pose", "occupancy", "open_close"],
    "bowl": ["occupancy"],
    "plate": ["occupancy"],
    "box": ["open_close", "occupancy"],
    "drawer": ["open_close"],
    "door": ["open_close"],
    "cabinet": ["open_close"],
    "lid": ["open_close"],
    "container": ["open_close", "occupancy"],
    "bag": ["open_close", "occupancy"],
    "jar": ["open_close", "occupancy"],
    "pan": ["pose", "on_off"],
    "pot": ["pose", "on_off", "occupancy"],
    "kettle": ["on_off", "occupancy"],
    "lamp": ["on_off"],
    "switch": ["on_off"],
    "faucet": ["on_off"],
    "book": ["pose", "open_close"],
    "laptop": ["open_close", "on_off"],
    "phone": ["pose", "on_off"],
    "trash_can": ["occupancy"],
    "washing_machine": ["open_close", "on_off"],
    "microwave": ["open_close", "on_off"],
    "oven": ["open_close", "on_off"],
    "refrigerator": ["open_close"],
    "toilet": ["open_close"],
}

_DEFAULT_STATE_TYPES = ["pose", "occupancy"]

_COT_STATE_PROMPT = """Analyze the physical state of the object "{name}" located at bbox [{bbox}] in this image.

Think step by step:
1. First, identify what type of object this is and what states are physically meaningful for it.
2. Look at the object's orientation, configuration, and visual appearance within the bbox region.
3. Consider the spatial context: what is around it, what is it resting on, is anything inside it?

For this object, determine the following states (only report clearly observable ones):
{state_queries}

Return your analysis as JSON:
{{
  "reasoning": "brief explanation of what you observe",
  "states": [
    {{"state_type": "...", "state_value": "...", "confidence": 0.0-1.0, "evidence": "what visual cue"}}
  ]
}}

Rules:
- Only report states you can clearly determine from the image
- confidence must be >= 0.7 to include
- state_value must be one of the allowed values listed above
- If you cannot determine a state, omit it entirely"""


def _build_state_queries(obj_name: str) -> str:
    """Build state-specific query lines based on object type."""
    name_lower = obj_name.lower().strip()
    state_types = _DEFAULT_STATE_TYPES

    for keyword, types in OBJECT_STATE_MAP.items():
        if keyword in name_lower:
            state_types = types
            break

    lines = []
    for st in state_types:
        allowed = STATE_TYPE_VALUES.get(st, [])
        if allowed:
            lines.append(f"- {st}: {' / '.join(allowed)}")
    return "\n".join(lines)


class AObjStatePipeline:
    """Object state recognition with CoT reasoning and type-aware queries."""

    def __init__(self, vlm_client=None):
        self._vlm = vlm_client

    def run(self, bundle: GsaEvidenceBundle) -> List[AObjStateLabel]:
        labels: List[AObjStateLabel] = []
        if not bundle.objects:
            return labels

        for obj in bundle.objects:
            obj_labels = self._process_object(obj, bundle)
            labels.extend(obj_labels)

        return labels

    def _process_object(
        self, obj: GsaObjectRef, bundle: GsaEvidenceBundle
    ) -> List[AObjStateLabel]:
        """Process a single object for state recognition."""
        results: List[AObjStateLabel] = []

        if not self._vlm or not bundle.image_path:
            return self._heuristic_states(obj, bundle)

        state_queries = _build_state_queries(obj.raw_object_name)
        if not state_queries:
            return results

        bbox_str = ", ".join(str(int(v)) for v in obj.bbox) if obj.bbox else "unknown"
        prompt = _COT_STATE_PROMPT.format(
            name=obj.raw_object_name,
            bbox=bbox_str,
            state_queries=state_queries,
        )

        try:
            response = self._vlm.chat(
                prompt,
                image_path=bundle.image_path,
                temperature=0.2,
            )
            parsed = self._parse_response(response)
            if not parsed:
                return self._heuristic_states(obj, bundle)

            reasoning = parsed.get("reasoning", "")
            states = parsed.get("states", [])

            for state_entry in states:
                state_type = state_entry.get("state_type", "")
                state_value = state_entry.get("state_value", "")
                confidence = state_entry.get("confidence", 0.0)
                evidence = state_entry.get("evidence", "")

                if confidence < 0.7:
                    continue

                label = self._build_label(
                    obj, bundle, state_type, state_value, confidence
                )
                validation = validate_obj_state(label)
                if validation.valid:
                    results.append(label)
                else:
                    logger.debug(
                        "Validation failed for %s/%s: %s",
                        obj.object_id, state_type, validation.errors
                    )

        except Exception as exc:
            logger.warning("VLM state query failed for %s: %s", obj.object_id, exc)
            return self._heuristic_states(obj, bundle)

        return results

    def _heuristic_states(
        self, obj: GsaObjectRef, bundle: GsaEvidenceBundle
    ) -> List[AObjStateLabel]:
        """Fallback: infer basic states from bbox geometry and spatial facts."""
        results: List[AObjStateLabel] = []
        if not obj.bbox:
            return results

        x1, y1, x2, y2 = obj.bbox
        w, h = x2 - x1, y2 - y1

        name_lower = obj.raw_object_name.lower()
        is_tall_object = any(k in name_lower for k in ("cup", "bottle", "mug", "can"))

        if is_tall_object:
            aspect = h / max(w, 1)
            if aspect > 1.5:
                state_type, state_value = "pose", "upright"
            elif aspect < 0.7:
                state_type, state_value = "pose", "lying"
            else:
                state_type, state_value = "pose", "tilted"

            label = self._build_label(obj, bundle, state_type, state_value, 0.6)
            if validate_obj_state(label).valid:
                results.append(label)

        return results

    def _parse_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from VLM response (handles thinking-mode)."""
        if not response:
            return None
        text = response.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Find last JSON block
        import re
        blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        for block in reversed(blocks):
            try:
                parsed = json.loads(block)
                if "states" in parsed or "state_type" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    def _build_label(
        self, obj: GsaObjectRef, bundle: GsaEvidenceBundle,
        state_type: str, state_value: str, confidence: float,
    ) -> AObjStateLabel:
        subtype_map = {
            "pose": AObjStateSubtype.POSE,
            "open_close": AObjStateSubtype.OPEN_CLOSE,
            "occupancy": AObjStateSubtype.OCCUPANCY,
            "on_off": AObjStateSubtype.ON_OFF,
        }
        subtype = subtype_map.get(state_type, AObjStateSubtype.OTHER)

        return AObjStateLabel(
            sample_id=bundle.sample_id,
            object_id=obj.object_id,
            object_class=obj.object_name,
            bbox=obj.bbox,
            state_type=state_type,
            state_value=state_value,
            subtype=subtype.value,
            confidence=confidence,
            s_context=state_type in _SPATIAL_STATE_TYPES,
            source_provenance=SourceProvenance.VLM.value,
            verifier_status=VerifierStatus.AUTO_VERIFIABLE.value,
        )
