"""A_action Adapter — Convert old Aff/scene graph outputs to A_action labels.

Maps existing AffordanceWorker annotations, SceneNode blocking info,
and PlacementRegion data into the new A_action contract.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .action_schema import AActionLabel, validate_a_action_label
from .a_action_pipeline import ReviewQueueRow
from .taxonomy import (
    ActionType,
    LabelType,
    SceneFamily,
    SourceProvenance,
    VerifierStatus,
)
from .visual_evidence import GsaEvidenceBundle, GsaObjectRef

logger = logging.getLogger("gsa.a_action_adapter")

# Mapping from old Aff action names to new ActionType
AFF_ACTION_MAP: Dict[str, str] = {
    "press": ActionType.PICK.value,
    "grasp": ActionType.PICK.value,
    "pick": ActionType.PICK.value,
    "pull": ActionType.PICK.value,
    "push": ActionType.PUSH.value,
    "slide": ActionType.PUSH.value,
    "place": ActionType.PLACE.value,
    "put": ActionType.PLACE.value,
    "open": ActionType.PICK.value,
    "close": ActionType.PICK.value,
    "rotate": ActionType.PICK.value,
    "turn": ActionType.PICK.value,
}


def adapt_aff_annotation(
    aff_ann: Dict[str, Any],
    bundle: GsaEvidenceBundle,
) -> Tuple[Optional[AActionLabel], Optional[ReviewQueueRow]]:
    """Convert a single old Aff annotation to an A_action label.

    Returns (label, None) if valid, or (None, review_row) if incomplete.
    """
    content = aff_ann.get("content", {})
    action_name = content.get("action", "").lower()
    action_type = AFF_ACTION_MAP.get(action_name, ActionType.PICK.value)

    object_name = content.get("object_name", "unknown")
    contact_point = content.get("contact_point")
    bbox = content.get("bbox")

    # Find matching object in bundle
    obj_id = _find_object_id(object_name, bundle)

    label = AActionLabel(
        sample_id=aff_ann.get("sample_id", bundle.sample_id),
        scene_family=bundle.scene_family,
        scenario_id=bundle.scenario_id,
        object_id=obj_id or f"{object_name}_0",
        action_type=action_type,
        label_type=LabelType.POINT.value,
        verifier_status=VerifierStatus.HEURISTIC_ONLY.value,
        source_provenance=SourceProvenance.ADAPTER.value,
        s_evidence_ids=[],
    )

    # Map contact_point to appropriate field
    if action_type == ActionType.PICK.value:
        label.grasp_point_2d = contact_point
        label.jaw_axis_2d = [1.0, 0.0]  # default horizontal
        label.can_pick_now = True
        label.blocked_by = []
        label.need_pre_push = False
    elif action_type == ActionType.PUSH.value:
        label.push_contact_point_2d = contact_point
        label.push_direction_2d = [1.0, 0.0]  # default right
        label.can_push_now = True
    elif action_type == ActionType.PLACE.value:
        if bbox:
            label.place_region_2d = {
                "type": "polygon",
                "points": [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                ],
            }
        label.can_place_now = bbox is not None

    result = validate_a_action_label(label)
    if result.valid:
        return label, None
    else:
        review = ReviewQueueRow(
            sample_id=label.sample_id,
            object_id=label.object_id,
            action_type=action_type,
            reason=f"adapter_missing_{','.join(result.missing_fields)}",
            suggested_reviewer_task=f"complete {action_type} fields from Aff annotation",
            source_provenance=SourceProvenance.ADAPTER.value,
        )
        return None, review


def adapt_scene_graph_blocking(
    scene_nodes: List[Dict[str, Any]],
    bundle: GsaEvidenceBundle,
) -> List[AActionLabel]:
    """Extract blocked_by / can_pick_now from scene graph nodes."""
    labels: List[AActionLabel] = []

    for node in scene_nodes:
        obj_id = node.get("object_id", node.get("canonical_name", ""))
        blocked_by = node.get("blocked_by", [])
        directly_reachable = node.get("directly_reachable", True)

        if not obj_id:
            continue

        # Only create labels for objects that have blocking info
        if blocked_by or not directly_reachable:
            label = AActionLabel(
                sample_id=bundle.sample_id,
                scene_family=bundle.scene_family,
                scenario_id=bundle.scenario_id,
                object_id=obj_id,
                action_type=ActionType.PICK.value,
                label_type=LabelType.YES_NO.value,
                verifier_status=VerifierStatus.AUTO_VERIFIABLE.value,
                source_provenance=SourceProvenance.ADAPTER.value,
                s_evidence_ids=[f"S:blocked_by:{obj_id}"],
                can_pick_now=directly_reachable and len(blocked_by) == 0,
                blocked_by=blocked_by,
                need_pre_push=len(blocked_by) > 0,
            )
            labels.append(label)

    return labels


def adapt_placement_regions(
    regions: List[Dict[str, Any]],
    bundle: GsaEvidenceBundle,
) -> List[AActionLabel]:
    """Convert PlacementRegion data to A_action place labels."""
    labels: List[AActionLabel] = []

    for region in regions:
        polygon = region.get("polygon", region.get("points", []))
        region_type = region.get("region_type", "free-space")
        parent_id = region.get("parent_object_id", "surface_0")

        if len(polygon) < 3:
            continue

        label = AActionLabel(
            sample_id=bundle.sample_id,
            scene_family=bundle.scene_family,
            scenario_id=bundle.scenario_id,
            object_id=parent_id,
            action_type=ActionType.PLACE.value,
            label_type=LabelType.REGION.value,
            verifier_status=VerifierStatus.AUTO_VERIFIABLE.value,
            source_provenance=SourceProvenance.ADAPTER.value,
            s_evidence_ids=[],
            place_region_2d={"type": "polygon", "points": polygon},
            can_place_now=True,
        )
        labels.append(label)

    return labels


def _find_object_id(name: str, bundle: GsaEvidenceBundle) -> Optional[str]:
    """Find matching object_id in bundle by name."""
    name_lower = name.lower()
    for obj in bundle.objects:
        if obj.object_name == name_lower or name_lower in obj.raw_object_name.lower():
            return obj.object_id
    return None
