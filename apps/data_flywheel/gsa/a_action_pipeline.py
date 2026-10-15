"""A_action Pipeline — 2D execution interface for pick/push/place.

Produces stable 2D labels for downstream geometry/depth/grasp optimizers.
Does NOT regenerate spatial facts — references S evidence instead.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .action_schema import AActionLabel, validate_a_action_label, ValidationResult
from .taxonomy import (
    ActionType,
    AActionSubtype,
    Bucket,
    FallbackAction,
    LabelType,
    SceneFamily,
    SourceProvenance,
    VerifierStatus,
)
from .visual_evidence import (
    GsaEvidenceBundle,
    GsaObjectRef,
    GsaPlacementRegion,
    GsaSpatialFact,
)

logger = logging.getLogger("gsa.a_action_pipeline")


@dataclass
class ReviewQueueRow:
    """A row in the review queue for incomplete labels."""

    sample_id: str
    bucket: str = Bucket.A_ACTION.value
    object_id: str = ""
    action_type: str = ""
    reason: str = ""
    suggested_reviewer_task: str = ""
    source_provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AActionPipeline:
    """Pipeline to produce A_action labels from evidence bundles.

    Three builders:
    1. build_pick_label() — grasp_point_2d, jaw_axis_2d, can_pick_now, blocked_by, need_pre_push
    2. build_push_label() — push_contact_point_2d, push_direction_2d, can_push_now
    3. build_place_label() — place_region_2d, can_place_now
    """

    def __init__(self, vlm_client=None):
        self.vlm_client = vlm_client

    def run(
        self, bundle: GsaEvidenceBundle
    ) -> Tuple[List[AActionLabel], List[ReviewQueueRow]]:
        """Process an evidence bundle, return (stable_labels, review_queue)."""
        labels: List[AActionLabel] = []
        review_queue: List[ReviewQueueRow] = []

        for obj in bundle.objects:
            # Determine candidate actions
            candidates = self._generate_candidates(bundle, obj)

            for action_type in candidates:
                if action_type == ActionType.PICK.value:
                    label = self._build_pick_label(bundle, obj)
                elif action_type == ActionType.PUSH.value:
                    label = self._build_push_label(bundle, obj)
                elif action_type == ActionType.PLACE.value:
                    label = self._build_place_label(bundle, obj)
                else:
                    continue

                if label is None:
                    continue

                result = validate_a_action_label(label)
                if result.valid:
                    labels.append(label)
                else:
                    review_queue.append(ReviewQueueRow(
                        sample_id=bundle.sample_id,
                        object_id=obj.object_id,
                        action_type=action_type,
                        reason=f"missing_{','.join(result.missing_fields)}" if result.missing_fields else "; ".join(result.errors),
                        suggested_reviewer_task=self._suggest_task(action_type, result),
                        source_provenance=obj.source_provenance,
                    ))

        return labels, review_queue

    def _generate_candidates(
        self, bundle: GsaEvidenceBundle, obj: GsaObjectRef
    ) -> List[str]:
        """Determine which action types are candidates for this object."""
        candidates = []

        # Pick: most objects can be picked unless push-only
        is_push_only = self._is_push_only(obj)
        if not is_push_only:
            candidates.append(ActionType.PICK.value)

        # Push: thin/flat objects, or objects blocking others
        if is_push_only or obj.object_id in self._get_blockers(bundle):
            candidates.append(ActionType.PUSH.value)

        # Place: if placement regions exist
        if bundle.placement_regions:
            candidates.append(ActionType.PLACE.value)

        return candidates or [ActionType.PICK.value]

    def _build_pick_label(
        self, bundle: GsaEvidenceBundle, obj: GsaObjectRef
    ) -> Optional[AActionLabel]:
        """Build a pick label using VLM or heuristics."""
        grasp_point = None
        jaw_axis = None
        can_pick = True
        blocked_by = obj.blocked_by.copy()
        need_pre_push = len(blocked_by) > 0

        if can_pick and blocked_by:
            can_pick = False

        # Use VLM to get grasp point if available
        if self.vlm_client and bundle.image_path and obj.bbox:
            vlm_result = self._vlm_pick_query(bundle, obj)
            if vlm_result and isinstance(vlm_result, dict):
                grasp_point = vlm_result.get("grasp_point_2d")
                jaw_axis = vlm_result.get("jaw_axis_2d")
                if vlm_result.get("can_pick_now") is not None:
                    can_pick = vlm_result["can_pick_now"]

        # Always fallback to heuristic if VLM didn't provide valid data
        if obj.bbox and not grasp_point:
            # Heuristic: bbox center as grasp point
            cx = (obj.bbox[0] + obj.bbox[2]) / 2
            cy = (obj.bbox[1] + obj.bbox[3]) / 2
            grasp_point = [cx, cy]
            # Heuristic: short axis of bbox as jaw axis
            w = obj.bbox[2] - obj.bbox[0]
            h = obj.bbox[3] - obj.bbox[1]
            if w < h:
                jaw_axis = [1.0, 0.0]
            else:
                jaw_axis = [0.0, 1.0]

        s_evidence = [f.fact_id for f in bundle.spatial_facts
                      if f.subject_id == obj.object_id or f.object_id == obj.object_id]

        verifier = VerifierStatus.AUTO_VERIFIABLE.value
        if not grasp_point:
            verifier = VerifierStatus.REVIEW_REQUIRED.value

        return AActionLabel(
            sample_id=bundle.sample_id,
            scene_family=bundle.scene_family,
            scenario_id=bundle.scenario_id,
            object_id=obj.object_id,
            action_type=ActionType.PICK.value,
            label_type=LabelType.POINT.value,
            verifier_status=verifier,
            source_provenance=obj.source_provenance,
            s_evidence_ids=s_evidence,
            grasp_point_2d=grasp_point,
            jaw_axis_2d=jaw_axis,
            can_pick_now=can_pick,
            blocked_by=blocked_by,
            need_pre_push=need_pre_push,
        )

    def _build_push_label(
        self, bundle: GsaEvidenceBundle, obj: GsaObjectRef
    ) -> Optional[AActionLabel]:
        """Build a push label."""
        contact_point = None
        direction = None
        can_push = True

        if self.vlm_client and bundle.image_path and obj.bbox:
            vlm_result = self._vlm_push_query(bundle, obj)
            if vlm_result and isinstance(vlm_result, dict):
                contact_point = vlm_result.get("push_contact_point_2d")
                direction = vlm_result.get("push_direction_2d")
                if vlm_result.get("can_push_now") is not None:
                    can_push = vlm_result["can_push_now"]

        # Always fallback to heuristic
        if obj.bbox and not contact_point:
            cx = (obj.bbox[0] + obj.bbox[2]) / 2
            cy = (obj.bbox[1] + obj.bbox[3]) / 2
            contact_point = [obj.bbox[0], cy]  # left edge midpoint
            direction = [-1.0, 0.0]  # push left

        s_evidence = [f.fact_id for f in bundle.spatial_facts
                      if f.subject_id == obj.object_id]

        verifier = VerifierStatus.AUTO_VERIFIABLE.value
        if not contact_point:
            verifier = VerifierStatus.REVIEW_REQUIRED.value

        return AActionLabel(
            sample_id=bundle.sample_id,
            scene_family=bundle.scene_family,
            scenario_id=bundle.scenario_id,
            object_id=obj.object_id,
            action_type=ActionType.PUSH.value,
            label_type=LabelType.VECTOR.value,
            verifier_status=verifier,
            source_provenance=obj.source_provenance,
            s_evidence_ids=s_evidence,
            push_contact_point_2d=contact_point,
            push_direction_2d=direction,
            can_push_now=can_push,
        )

    def _build_place_label(
        self, bundle: GsaEvidenceBundle, obj: GsaObjectRef
    ) -> Optional[AActionLabel]:
        """Build a place label from placement regions."""
        if not bundle.placement_regions:
            return None

        # Find best placement region
        region = bundle.placement_regions[0]
        place_region = {
            "type": "polygon",
            "points": region.polygon,
        }

        can_place = len(region.polygon) >= 3

        return AActionLabel(
            sample_id=bundle.sample_id,
            scene_family=bundle.scene_family,
            scenario_id=bundle.scenario_id,
            object_id=obj.object_id,
            action_type=ActionType.PLACE.value,
            label_type=LabelType.REGION.value,
            verifier_status=VerifierStatus.AUTO_VERIFIABLE.value if can_place else VerifierStatus.REVIEW_REQUIRED.value,
            source_provenance=obj.source_provenance,
            s_evidence_ids=[],
            place_region_2d=place_region,
            can_place_now=can_place,
        )

    def _vlm_pick_query(self, bundle: GsaEvidenceBundle, obj: GsaObjectRef) -> Optional[Dict]:
        """Ask VLM for grasp point and jaw axis."""
        prompt = f"""Look at the object "{obj.raw_object_name}" at bbox [{obj.bbox[0]:.0f},{obj.bbox[1]:.0f},{obj.bbox[2]:.0f},{obj.bbox[3]:.0f}].

Determine the best grasp point (where a parallel-jaw gripper should close) and jaw axis (gripper closing direction).

Return JSON:
{{"grasp_point_2d": [x, y], "jaw_axis_2d": [dx, dy], "can_pick_now": true/false}}

- grasp_point_2d: pixel coordinates of the optimal grasp center
- jaw_axis_2d: unit vector of gripper closing direction
- can_pick_now: false if object is blocked, too large, or not graspable"""

        try:
            result = self.vlm_client.chat_json(
                prompt=prompt,
                image_path=bundle.image_path,
                max_tokens=4000,
                temperature=0.2,
                system_prompt="You are a robotic manipulation expert. Respond in JSON only.",
            )
            if isinstance(result, dict):
                # Denormalize grasp point from 0-1000 to pixel coords
                gp = result.get("grasp_point_2d")
                if gp and len(gp) == 2 and all(0 <= v <= 1000 for v in gp):
                    from PIL import Image as PILImage
                    try:
                        img = PILImage.open(bundle.image_path)
                        from apps.data_flywheel.workers.vlm_client import VLMClient
                        result["grasp_point_2d"] = VLMClient.denormalize_point(gp, img.size[0], img.size[1])
                        img.close()
                    except Exception:
                        pass
                return result
            if isinstance(result, list) and result:
                return result[0] if isinstance(result[0], dict) else None
            return None
        except Exception as e:
            logger.warning(f"VLM pick query failed: {e}")
            return None

    def _vlm_push_query(self, bundle: GsaEvidenceBundle, obj: GsaObjectRef) -> Optional[Dict]:
        """Ask VLM for push contact point and direction."""
        prompt = f"""Look at the object "{obj.raw_object_name}" at bbox [{obj.bbox[0]:.0f},{obj.bbox[1]:.0f},{obj.bbox[2]:.0f},{obj.bbox[3]:.0f}].

Determine the best push contact point and push direction to clear this object.

Return JSON:
{{"push_contact_point_2d": [x, y], "push_direction_2d": [dx, dy], "can_push_now": true/false}}

- push_contact_point_2d: pixel coordinates where to contact the object
- push_direction_2d: unit vector of push direction
- can_push_now: false if object cannot be pushed"""

        try:
            result = self.vlm_client.chat_json(
                prompt=prompt,
                image_path=bundle.image_path,
                max_tokens=512,
                temperature=0.2,
                system_prompt="You are a robotic manipulation expert. Respond in JSON only.",
            )
            if isinstance(result, dict):
                return result
            if isinstance(result, list) and result:
                return result[0] if isinstance(result[0], dict) else None
            return None
        except Exception as e:
            logger.warning(f"VLM push query failed: {e}")
            return None

    def _is_push_only(self, obj: GsaObjectRef) -> bool:
        """Determine if object is push-only (thin, flat, card-like)."""
        push_only_keywords = {"card", "paper", "sheet", "thin", "flat", "coin", "ticket"}
        return any(kw in obj.object_name for kw in push_only_keywords)

    def _get_blockers(self, bundle: GsaEvidenceBundle) -> set:
        """Get set of object_ids that block other objects."""
        blockers = set()
        for obj in bundle.objects:
            for b in obj.blocked_by:
                blockers.add(b)
        return blockers

    def _suggest_task(self, action_type: str, result: ValidationResult) -> str:
        if result.missing_fields:
            return f"mark {', '.join(result.missing_fields)} for {action_type}"
        return f"fix {action_type} label errors"
