"""Pick-Place Pipeline — Unified pipeline outputting complete application JSON.

One VLM call per object to produce the full pick-and-place label,
covering: bbox, affordance_point, object_type, pose, approach, grasp_rect.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from .pick_place_schema import (
    PickPlaceLabel,
    center_wh_angle_to_corners,
    validate_pick_place_label,
    OBJECT_TYPES,
    POSE_HINTS,
    GRASP_HINTS,
    CONTAINER_OPENINGS,
    APPROACH_DIRECTIONS,
)
from .prompt_templates import generate_query, PICK_PLACE_SYSTEM_PROMPT
from .visual_evidence import GsaEvidenceBundle, GsaObjectRef

logger = logging.getLogger("gsa.pick_place_pipeline")


class PickPlacePipeline:
    """Unified pipeline: one VLM call -> complete pick-place label."""

    def __init__(self, vlm_client, use_judge: bool = True):
        self.vlm = vlm_client
        self.use_judge = use_judge

    def run(self, bundle: GsaEvidenceBundle) -> List[PickPlaceLabel]:
        """Generate pick-place labels for all objects in the bundle."""
        labels = []
        for obj in bundle.objects:
            if not obj.bbox:
                continue
            label = self._generate_for_object(bundle, obj)
            if label:
                labels.append(label)
        return labels

    def _generate_for_object(
        self, bundle: GsaEvidenceBundle, obj: GsaObjectRef
    ) -> Optional[PickPlaceLabel]:
        """Generate a complete pick-place label for one object."""
        if not bundle.image_path or not obj.bbox:
            return None

        # Get image dimensions for coordinate handling
        try:
            from PIL import Image
            img = Image.open(bundle.image_path)
            img_w, img_h = img.size
            img.close()
        except Exception:
            img_w, img_h = 1000, 1000

        # Normalize bbox to 0-1000
        norm_bbox = [
            int(obj.bbox[0] / img_w * 1000),
            int(obj.bbox[1] / img_h * 1000),
            int(obj.bbox[2] / img_w * 1000),
            int(obj.bbox[3] / img_h * 1000),
        ]

        # Generate diverse query
        query = generate_query(
            object_name=obj.raw_object_name or obj.object_name,
            scene_family=bundle.scene_family,
        )

        # Build VLM prompt
        user_prompt = self._build_prompt(obj, norm_bbox, query)

        # Call VLM
        try:
            result = self.vlm.chat_json(
                prompt=user_prompt,
                image_path=bundle.image_path,
                max_tokens=4000,
                temperature=0.2,
                system_prompt="You are a robotic manipulation expert. Return strict JSON only.",
            )
        except Exception as e:
            logger.warning(f"VLM call failed for {obj.object_id}: {e}")
            result = {}

        if not isinstance(result, dict):
            result = {}

        # Build label from VLM result + fallbacks
        label = self._build_label(
            bundle, obj, result, norm_bbox, query, img_w, img_h
        )

        # Validate
        validation = validate_pick_place_label(label)
        if not validation["valid"]:
            logger.debug(f"Validation issues for {obj.object_id}: {validation['issues']}")

        return label

    def _build_prompt(self, obj: GsaObjectRef, norm_bbox: List[int], query: str) -> str:
        """Build the user prompt for VLM."""
        obj_name = obj.raw_object_name or obj.object_name
        return (
            f'{query}\n\n'
            f'The target object "{obj_name}" is at bbox [{norm_bbox[0]}, {norm_bbox[1]}, {norm_bbox[2]}, {norm_bbox[3]}].\n\n'
            f'Return a JSON object with ALL fields:\n'
            f'{{"bbox": [{norm_bbox[0]}, {norm_bbox[1]}, {norm_bbox[2]}, {norm_bbox[3]}], '
            f'"affordance_point": [x, y], "affordance_valid": true, '
            f'"affordance_reason": "why this point", '
            f'"object_type": "box|cylinder|frustum|bowl|plate|slender|thin|irregular|unknown", '
            f'"object_type_confidence": 0.9, '
            f'"pose_hint": "upright|horizontal|unknown", '
            f'"pose_hint_confidence": 0.9, "pose_hint_reason": "reason", '
            f'"grasp_hint": "rim|side|middle|edge|handle|body|unknown", '
            f'"object_pose": "short pose", '
            f'"container_opening": "open_visible|closed_or_lidded|unknown", '
            f'"container_opening_confidence": 0.9, "container_opening_reason": "reason", '
            f'"preferred_approach": "horizontal|top_down|diagonal_down|unknown", '
            f'"preferred_approach_confidence": 0.9, "preferred_approach_reason": "reason", '
            f'"fallback_approaches": ["horizontal", "top_down"]}}\n\n'
            f'affordance_point rules: must be ON the object (not background). '
            f'For handles->on handle, bottles->side wall, bowls->rim, thin objects->edge. '
            f'Coords in [0,1000]. Return ONLY JSON.'
        )

    def _build_label(
        self,
        bundle: GsaEvidenceBundle,
        obj: GsaObjectRef,
        vlm_result: Dict[str, Any],
        norm_bbox: List[int],
        query: str,
        img_w: int,
        img_h: int,
    ) -> PickPlaceLabel:
        """Build PickPlaceLabel from VLM result with fallbacks."""

        # bbox: use VLM result or fallback to detected bbox
        bbox = vlm_result.get("bbox", norm_bbox)
        if not bbox or len(bbox) != 4:
            bbox = norm_bbox

        # affordance_point: VLM or fallback to bbox center
        aff_pt = vlm_result.get("affordance_point")
        if not aff_pt or len(aff_pt) != 2:
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            aff_pt = [cx, cy]
        # Ensure affordance_point is inside bbox
        aff_pt[0] = max(bbox[0], min(bbox[2], aff_pt[0]))
        aff_pt[1] = max(bbox[1], min(bbox[3], aff_pt[1]))

        # object_type
        obj_type = vlm_result.get("object_type", "unknown")
        if obj_type not in OBJECT_TYPES:
            obj_type = "unknown"

        # pose_hint
        pose = vlm_result.get("pose_hint", "unknown")
        if pose not in POSE_HINTS:
            pose = "unknown"

        # grasp_hint
        grasp_h = vlm_result.get("grasp_hint", "unknown")
        if grasp_h not in GRASP_HINTS:
            grasp_h = "unknown"

        # container_opening
        container = vlm_result.get("container_opening", "unknown")
        if container not in CONTAINER_OPENINGS:
            container = "unknown"

        # preferred_approach
        approach = vlm_result.get("preferred_approach", "unknown")
        if approach not in APPROACH_DIRECTIONS:
            approach = "unknown"

        # fallback_approaches
        fallbacks = vlm_result.get("fallback_approaches", [])
        if not isinstance(fallbacks, list):
            fallbacks = []
        fallbacks = [f for f in fallbacks if f in APPROACH_DIRECTIONS]

        # Grasp rect: compute from affordance_point + object geometry
        grasp_center = aff_pt.copy()
        obj_w = bbox[2] - bbox[0]
        obj_h = bbox[3] - bbox[1]
        grasp_width = vlm_result.get("grasp_width") or min(obj_w * 0.6, 80)
        grasp_height = vlm_result.get("grasp_height") or min(obj_h * 0.3, 40)
        grasp_angle = vlm_result.get("grasp_angle") or 0.0

        # Infer angle from approach direction if not provided
        if grasp_angle == 0.0 and approach == "horizontal":
            grasp_angle = 0.0
        elif grasp_angle == 0.0 and approach == "top_down":
            grasp_angle = math.pi / 2

        grasp_corners = center_wh_angle_to_corners(
            grasp_center[0], grasp_center[1],
            grasp_width, grasp_height, grasp_angle
        )

        return PickPlaceLabel(
            sample_id=bundle.sample_id,
            image_path=bundle.image_path or "",
            query=query,
            object_id=obj.object_id,
            object_name=obj.raw_object_name or obj.object_name,
            bbox=[int(b) for b in bbox],
            affordance_point=[int(p) for p in aff_pt],
            affordance_valid=vlm_result.get("affordance_valid", True),
            affordance_reason=vlm_result.get("affordance_reason", "bbox center fallback"),
            object_type=obj_type,
            object_type_confidence=float(vlm_result.get("object_type_confidence", 0.5)),
            pose_hint=pose,
            pose_hint_confidence=float(vlm_result.get("pose_hint_confidence", 0.5)),
            pose_hint_reason=vlm_result.get("pose_hint_reason", ""),
            grasp_hint=grasp_h,
            object_pose=vlm_result.get("object_pose", ""),
            container_opening=container,
            container_opening_confidence=float(vlm_result.get("container_opening_confidence", 0.0)),
            container_opening_reason=vlm_result.get("container_opening_reason", ""),
            preferred_approach=approach,
            preferred_approach_confidence=float(vlm_result.get("preferred_approach_confidence", 0.5)),
            preferred_approach_reason=vlm_result.get("preferred_approach_reason", ""),
            fallback_approaches=fallbacks,
            grasp_rect_corners=grasp_corners,
            grasp_center=[float(c) for c in grasp_center],
            grasp_width=float(grasp_width),
            grasp_height=float(grasp_height),
            grasp_angle=float(grasp_angle),
        )
