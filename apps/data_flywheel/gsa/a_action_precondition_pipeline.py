"""A_action_precondition pipeline — action feasibility with spatial reasoning.

Key improvements:
1. Uses S-stage spatial facts (occlusion, support, containment) for precondition inference
2. CoT prompt asks VLM to reason about physical constraints
3. Produces actionable blocked_by references and pre-action suggestions
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from .a_pipeline_schemas import AActionPreconditionLabel, validate_action_precondition
from .taxonomy import (
    AActionPreconditionSubtype,
    ActionType,
    Bucket,
    FailureReason,
    PreAction,
    SourceProvenance,
    VerifierStatus,
)
from .visual_evidence import GsaEvidenceBundle, GsaObjectRef

logger = logging.getLogger("gsa.a_action_precondition_pipeline")

_COT_PRECONDITION_PROMPT = """You are a robot manipulation planner. Determine if the object "{name}" at bbox [{bbox}] can be picked up right now.

Think step by step:
1. VISIBILITY: Is the object fully visible or partially occluded by other objects?
2. ACCESSIBILITY: Can a gripper reach it from above or the side without collision?
3. STABILITY: Is the object in a stable pose that allows grasping?
4. CLEARANCE: Is there enough space around it for gripper fingers?
5. PHYSICAL CONSTRAINTS: Is it too heavy, too large, or fixed in place?

Known spatial relationships in this scene:
{spatial_facts}

Objects detected in scene: {object_list}

Return JSON:
{{
  "reasoning": "step-by-step analysis of each constraint",
  "can_do_now": true/false,
  "failure_reason": "occluded|unstable|unreachable|no_clearance|too_heavy|too_large|fixed|none",
  "blocked_by": ["list of object IDs blocking access"],
  "need_pre_action": "none|push_first|reorient_first|remove_obstacle",
  "confidence": 0.0-1.0
}}"""


class AActionPreconditionPipeline:
    """Determines action feasibility using spatial facts + VLM reasoning."""

    def __init__(self, vlm_client=None):
        self._vlm = vlm_client

    def run(self, bundle: GsaEvidenceBundle) -> List[AActionPreconditionLabel]:
        labels: List[AActionPreconditionLabel] = []
        if not bundle.objects:
            return labels

        spatial_text = self._format_spatial_facts(bundle)
        object_list = ", ".join(o.object_id for o in bundle.objects)
        occlusion_map = self._build_occlusion_map(bundle)

        for obj in bundle.objects:
            label = self._assess_object(obj, bundle, spatial_text, object_list, occlusion_map)
            if label:
                labels.append(label)

        return labels

    def _assess_object(
        self, obj: GsaObjectRef, bundle: GsaEvidenceBundle,
        spatial_text: str, object_list: str,
        occlusion_map: Dict[str, Set[str]],
    ) -> Optional[AActionPreconditionLabel]:
        """Assess pick feasibility for a single object."""

        # Step 1: Heuristic pre-check from spatial facts
        blockers = occlusion_map.get(obj.object_id, set())
        heuristic_can_do = len(blockers) == 0

        # Step 2: VLM CoT assessment (if available)
        vlm_result = None
        if self._vlm and bundle.image_path:
            vlm_result = self._vlm_assess(obj, bundle, spatial_text, object_list)

        # Step 3: Merge heuristic + VLM
        if vlm_result:
            can_do = vlm_result.get("can_do_now", heuristic_can_do)
            failure_reason = vlm_result.get("failure_reason", "none")
            vlm_blockers = vlm_result.get("blocked_by", [])
            need_pre = vlm_result.get("need_pre_action", "none")
            confidence = vlm_result.get("confidence", 0.8)

            all_blockers = list(blockers | set(vlm_blockers))
            if all_blockers and can_do:
                can_do = False
                failure_reason = "occluded"
        else:
            can_do = heuristic_can_do
            failure_reason = "occluded" if blockers else "none"
            all_blockers = list(blockers)
            need_pre = "push_first" if blockers else "none"
            confidence = 0.6

        # Map failure_reason to enum
        fr_map = {
            "occluded": FailureReason.OCCLUDED.value,
            "unstable": FailureReason.UNSTABLE.value,
            "unreachable": FailureReason.UNREACHABLE.value,
            "no_clearance": FailureReason.NO_CLEARANCE.value,
            "too_heavy": FailureReason.TOO_HEAVY.value,
            "too_large": FailureReason.TOO_LARGE.value,
            "fixed": FailureReason.FIXED.value,
            "none": None,
        }
        mapped_fr = fr_map.get(failure_reason) if not can_do else None

        # Map pre_action
        pre_map = {
            "none": PreAction.NONE.value,
            "push_first": PreAction.PUSH_FIRST.value,
            "reorient_first": PreAction.REORIENT_FIRST.value,
            "remove_obstacle": PreAction.PUSH_FIRST.value,
        }
        mapped_pre = pre_map.get(need_pre, PreAction.NONE.value)

        label = AActionPreconditionLabel(
            sample_id=bundle.sample_id,
            object_id=obj.object_id,
            scene_family=bundle.scene_family,
            scenario_id=bundle.scenario_id,
            action_type=ActionType.PICK.value,
            can_do_now=can_do,
            blocked_by=all_blockers,
            failure_reason=mapped_fr,
            need_pre_action=mapped_pre,
            confidence=confidence,
            s_context=True,
            source_provenance=SourceProvenance.VLM.value,
            verifier_status=VerifierStatus.AUTO_VERIFIABLE.value,
        )

        validation = validate_action_precondition(label)
        if validation.valid:
            return label
        logger.debug("Precondition validation failed: %s", validation.errors)
        return None

    def _vlm_assess(
        self, obj: GsaObjectRef, bundle: GsaEvidenceBundle,
        spatial_text: str, object_list: str,
    ) -> Optional[Dict]:
        """Query VLM for precondition assessment."""
        bbox_str = ", ".join(str(int(v)) for v in obj.bbox) if obj.bbox else "unknown"
        prompt = _COT_PRECONDITION_PROMPT.format(
            name=obj.raw_object_name,
            bbox=bbox_str,
            spatial_facts=spatial_text,
            object_list=object_list,
        )
        try:
            response = self._vlm.chat(
                prompt, image_path=bundle.image_path, temperature=0.2,
            )
            return self._parse_json(response)
        except Exception as exc:
            logger.warning("VLM precondition query failed for %s: %s", obj.object_id, exc)
            return None

    def _build_occlusion_map(self, bundle: GsaEvidenceBundle) -> Dict[str, Set[str]]:
        """Extract occlusion relationships from spatial facts."""
        occ_map: Dict[str, Set[str]] = {}
        if not bundle.spatial_facts:
            return occ_map

        for fact in bundle.spatial_facts:
            if isinstance(fact, dict):
                pred = fact.get("predicate", "")
                subj = fact.get("subject_id", "")
                obj_id = fact.get("object_id", "")
                # "A occludes B" means B is blocked by A
                if "occlu" in pred.lower() or "on_top" in pred.lower() or "covers" in pred.lower():
                    if obj_id not in occ_map:
                        occ_map[obj_id] = set()
                    occ_map[obj_id].add(subj)
                # "B is_under A" means B is blocked by A
                if "under" in pred.lower() or "below" in pred.lower():
                    if subj not in occ_map:
                        occ_map[subj] = set()
                    occ_map[subj].add(obj_id)
        return occ_map

    def _format_spatial_facts(self, bundle: GsaEvidenceBundle) -> str:
        if not bundle.spatial_facts:
            return "No spatial relationships detected."
        lines = []
        for fact in bundle.spatial_facts[:15]:
            if isinstance(fact, dict):
                subj = fact.get("subject_id", "?")
                pred = fact.get("predicate", "?")
                obj_id = fact.get("object_id", "?")
                lines.append(f"- {subj} {pred} {obj_id}")
        return "\n".join(lines) if lines else "No spatial relationships detected."

    def _parse_json(self, response: str) -> Optional[Dict]:
        if not response:
            return None
        text = response.strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        for block in reversed(blocks):
            try:
                parsed = json.loads(block)
                if "can_do_now" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        return None
