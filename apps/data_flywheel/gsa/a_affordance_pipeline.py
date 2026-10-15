"""A_Affordance Pipeline — Object interactability analysis with CoT reasoning.

Upstream of A_action_interface: determines WHAT actions can be performed on
each object, WHERE the best contact point is, and from which direction to
approach. Its output feeds into A_action_interface for precise geometric
generation (grasp point, jaw axis, push vector, etc.).

CoT flow per object:
1. Shape analysis — describe geometry relevant to manipulation
2. Action selection — pick the most natural action type
3. Contact point — identify the optimal interaction hotspot
4. Approach direction — determine how to reach the contact point
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .a_pipeline_schemas import (
    AAffordanceLabel,
    KNOWN_AFFORDANCE_ACTIONS,
    KNOWN_APPROACH_DIRECTIONS,
    validate_affordance,
)
from .taxonomy import (
    AAffordanceSubtype,
    Bucket,
    SourceProvenance,
    VerifierStatus,
)
from .visual_evidence import GsaEvidenceBundle, GsaObjectRef

logger = logging.getLogger("gsa.a_affordance_pipeline")

_AFFORDANCE_COT_PROMPT = """You are a robot manipulation expert. Given the object "{name}" at bbox [{bbox}] in the image, analyze its affordance for robotic interaction.

Think step by step:
1. SHAPE ANALYSIS: Describe the object's shape, size, and physical properties relevant to manipulation.
2. ACTION SELECTION: What is the most natural robotic action for this object? Choose from: pick, push, press, pull, rotate, open, close, slide, pour, grasp, place.
3. REASON: Why is this action appropriate given the object's shape and function?
4. CONTACT POINT: Where exactly should the robot make contact? Must be INSIDE the bbox [{bbox}].
5. APPROACH DIRECTION: From which direction should the robot approach? Choose from: top, front, left, right, bottom, diagonal_down.

{spatial_context}

Return ONLY a JSON object (no markdown, no explanation outside JSON):
{{
  "shape_description": "brief shape description",
  "action": "one action from the list above",
  "reason": "why this action suits this object",
  "contact_point": [x, y],
  "approach_direction": "top|front|left|right|bottom|diagonal_down",
  "confidence": 0.0-1.0
}}

CRITICAL: contact_point [x, y] must be pixel coordinates INSIDE the bbox [{bbox}]."""

_REPAIR_PROMPT = """The following text is a robot affordance analysis that needs to be converted to strict JSON.

Original response:
{raw_response}

Target object: "{name}" at bbox [{bbox}].

Extract the affordance information and return ONLY this JSON (no other text):
{{
  "shape_description": "string",
  "action": "pick|push|press|pull|rotate|open|close|slide|pour|grasp|place",
  "reason": "string",
  "contact_point": [x, y],
  "approach_direction": "top|front|left|right|bottom|diagonal_down",
  "confidence": 0.0-1.0
}}"""


# Objects that are typically non-interactable (background/fixed)
_SKIP_OBJECTS = {
    "wall", "floor", "ceiling", "table", "desk", "counter",
    "robot", "arm", "gripper", "microwave", "faucet", "sink",
    "window", "door_frame", "cabinet_body",
}


class AAffordancePipeline:
    """Generates affordance labels for each object in the scene.

    For every grounded object, determines:
    - what action can be performed
    - where the best contact point is
    - from which direction to approach
    """

    def __init__(self, vlm_client=None):
        self._vlm = vlm_client

    def run(self, bundle: GsaEvidenceBundle) -> List[AAffordanceLabel]:
        """Run affordance analysis on all objects in the bundle."""
        labels: List[AAffordanceLabel] = []
        if not bundle.objects:
            return labels

        spatial_ctx = self._format_spatial_context(bundle)

        for obj in bundle.objects:
            if self._should_skip(obj):
                continue
            label = self._generate_affordance(obj, bundle, spatial_ctx)
            if label:
                labels.append(label)

        return labels

    def _should_skip(self, obj: GsaObjectRef) -> bool:
        name_lower = (obj.raw_object_name or "").lower()
        return any(skip in name_lower for skip in _SKIP_OBJECTS)

    def _generate_affordance(
        self, obj: GsaObjectRef, bundle: GsaEvidenceBundle, spatial_ctx: str
    ) -> Optional[AAffordanceLabel]:
        """Generate affordance for a single object via VLM + fallback."""
        bbox_str = self._bbox_str(obj)

        if self._vlm and bundle.image_path:
            label = self._vlm_generate(obj, bundle, bbox_str, spatial_ctx)
            if label:
                return label

        return self._heuristic_fallback(obj, bundle)

    def _vlm_generate(
        self, obj: GsaObjectRef, bundle: GsaEvidenceBundle,
        bbox_str: str, spatial_ctx: str,
    ) -> Optional[AAffordanceLabel]:
        """Call VLM with CoT prompt, with JSON repair fallback."""
        ctx_text = f"Spatial context:\n{spatial_ctx}" if spatial_ctx else ""
        prompt = _AFFORDANCE_COT_PROMPT.format(
            name=obj.raw_object_name or "unknown",
            bbox=bbox_str,
            spatial_context=ctx_text,
        )

        try:
            response = self._vlm.chat(
                prompt, image_path=bundle.image_path, temperature=0.1,
            )
        except Exception as exc:
            logger.warning("VLM affordance failed for %s: %s", obj.object_id, exc)
            return None

        parsed = self._parse_json(response)

        # JSON repair: if first attempt didn't yield valid JSON
        if not parsed and response:
            parsed = self._repair_json(response, obj, bbox_str)

        if not parsed:
            return None

        return self._build_label_from_vlm(parsed, obj, bundle)

    def _repair_json(
        self, raw_response: str, obj: GsaObjectRef, bbox_str: str
    ) -> Optional[Dict]:
        """Lightweight repair call without image."""
        if not self._vlm:
            return None
        repair_prompt = _REPAIR_PROMPT.format(
            raw_response=raw_response[:1500],
            name=obj.raw_object_name or "unknown",
            bbox=bbox_str,
        )
        try:
            repaired = self._vlm.chat(repair_prompt, temperature=0.0)
            return self._parse_json(repaired)
        except Exception:
            return None

    def _build_label_from_vlm(
        self, parsed: Dict, obj: GsaObjectRef, bundle: GsaEvidenceBundle
    ) -> Optional[AAffordanceLabel]:
        """Convert VLM JSON output to AAffordanceLabel."""
        action = parsed.get("action", "")
        contact = parsed.get("contact_point")
        approach = parsed.get("approach_direction", "front")
        shape_desc = parsed.get("shape_description", "")
        reason = parsed.get("reason", "")
        confidence = parsed.get("confidence", 0.7)

        if not action:
            return None

        # Validate and clamp contact point
        if contact and obj.bbox:
            contact = self._clamp_to_bbox(contact, obj.bbox)

        # Determine subtype from action
        subtype = self._action_to_subtype(action, obj)

        label = AAffordanceLabel(
            sample_id=bundle.sample_id,
            object_id=obj.object_id,
            object_name=obj.raw_object_name or "",
            bbox=obj.bbox,
            shape_description=shape_desc,
            action=action,
            reason=reason,
            contact_point=contact,
            approach_direction=approach,
            image_size=None,
            confidence=confidence if isinstance(confidence, float) else 0.7,
            scene_family=bundle.scene_family,
            scenario_id=bundle.scenario_id,
            subtype=subtype,
            source_provenance=SourceProvenance.VLM.value,
            verifier_status=VerifierStatus.REVIEW_REQUIRED.value,
            g_context=True,
            s_context=bool(bundle.spatial_facts),
        )

        validation = validate_affordance(label)
        if validation.valid:
            return label

        logger.debug("Affordance validation failed for %s: %s",
                     obj.object_id, validation.errors)
        return None

    def _heuristic_fallback(
        self, obj: GsaObjectRef, bundle: GsaEvidenceBundle
    ) -> Optional[AAffordanceLabel]:
        """Rule-based fallback when VLM is unavailable."""
        if not obj.bbox:
            return None

        x1, y1, x2, y2 = obj.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1

        name_lower = (obj.raw_object_name or "").lower()

        # Heuristic action selection
        action, approach, contact, reason = self._infer_action(
            name_lower, cx, cy, w, h, x1, y1, x2, y2
        )

        subtype = self._action_to_subtype(action, obj)

        label = AAffordanceLabel(
            sample_id=bundle.sample_id,
            object_id=obj.object_id,
            object_name=obj.raw_object_name or "",
            bbox=obj.bbox,
            shape_description=f"bbox {int(w)}x{int(h)} pixels",
            action=action,
            reason=reason,
            contact_point=contact,
            approach_direction=approach,
            image_size=None,
            confidence=0.5,
            scene_family=bundle.scene_family,
            scenario_id=bundle.scenario_id,
            subtype=subtype,
            source_provenance=SourceProvenance.ADAPTER.value,
            verifier_status=VerifierStatus.HEURISTIC_ONLY.value,
            g_context=True,
            s_context=False,
        )

        validation = validate_affordance(label)
        if validation.valid:
            return label
        return None

    def _infer_action(self, name: str, cx, cy, w, h, x1, y1, x2, y2):
        """Heuristic action inference from object name and shape."""
        # Button-like objects
        if any(kw in name for kw in ("button", "switch", "key")):
            return "press", "front", [cx, cy], "button-shaped object designed for pressing"

        # Knob/dial
        if any(kw in name for kw in ("knob", "dial", "wheel")):
            return "rotate", "front", [cx, cy], "circular control designed for rotation"

        # Handle/lever
        if any(kw in name for kw in ("handle", "lever", "latch")):
            return "pull", "front", [cx, cy], "handle designed for pulling"

        # Door/lid
        if any(kw in name for kw in ("door", "lid", "cover", "cap")):
            return "open", "front", [cx, cy], "closure that can be opened"

        # Thin/flat objects
        if w > h * 2.5 or h > w * 2.5:
            contact = [x1 + w * 0.2, cy] if w > h else [cx, y1 + h * 0.2]
            return "push", "front", contact, "thin/flat object better suited for pushing"

        # Default: pick from center
        return "pick", "top", [cx, cy - h * 0.1], "general object suitable for grasping"

    def _action_to_subtype(self, action: str, obj: GsaObjectRef) -> str:
        """Map action to AAffordanceSubtype."""
        name_lower = (obj.raw_object_name or "").lower()
        mapping = {
            "press": AAffordanceSubtype.BUTTON_POINT.value,
            "rotate": AAffordanceSubtype.KNOB_POINT.value,
            "pull": AAffordanceSubtype.HANDLE_POINT.value,
            "open": AAffordanceSubtype.HANDLE_POINT.value,
            "close": AAffordanceSubtype.HANDLE_POINT.value,
        }
        if action in mapping:
            return mapping[action]
        if any(kw in name_lower for kw in ("edge", "rim", "lip")):
            return AAffordanceSubtype.RIM_POINT.value
        return AAffordanceSubtype.INTERACTION_HOTSPOT.value

    def _format_spatial_context(self, bundle: GsaEvidenceBundle) -> str:
        if not bundle.spatial_facts:
            return ""
        lines = []
        for fact in bundle.spatial_facts[:10]:
            if isinstance(fact, dict):
                subj = fact.get("subject_id", "?")
                pred = fact.get("predicate", "?")
                obj = fact.get("object_id", "?")
                lines.append(f"- {subj} is {pred} {obj}")
            elif isinstance(fact, str):
                lines.append(f"- {fact}")
        return "\n".join(lines)

    def _clamp_to_bbox(self, point: List[float], bbox: List[float]) -> List[float]:
        if not point or len(point) != 2 or not bbox or len(bbox) < 4:
            return point
        x1, y1, x2, y2 = bbox[:4]
        px = max(x1, min(x2, point[0]))
        py = max(y1, min(y2, point[1]))
        return [px, py]

    def _bbox_str(self, obj: GsaObjectRef) -> str:
        if obj.bbox:
            return ", ".join(str(int(v)) for v in obj.bbox)
        return "unknown"

    def _parse_json(self, response: str) -> Optional[Dict]:
        if not response:
            return None
        text = response.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        for block in reversed(blocks):
            try:
                parsed = json.loads(block)
                if any(k in parsed for k in ("action", "contact_point", "shape_description")):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None
