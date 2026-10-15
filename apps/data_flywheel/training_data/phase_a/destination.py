from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.scene_understanding.placement import select_default_destination
from apps.data_flywheel.types import PlacementRegion, SceneGraph, SceneNode

from .resolver import resolve_reference


@dataclass
class DestinationResolution:
    status: str
    destination_id: Optional[str] = None
    region: Optional[PlacementRegion] = None
    placement_reason: Optional[str] = None
    matched_rule: str = ""
    candidates: List[str] = field(default_factory=list)
    note: Optional[str] = None


def destination_type_for_region(region: PlacementRegion) -> str:
    if region.region_type == "inside-container":
        return "container"
    return "free-space"


def _regions_for_support(scene_graph: SceneGraph, support_id: str) -> List[PlacementRegion]:
    return [
        region
        for region in scene_graph.placement_regions
        if region.support_id == support_id
    ]


def resolve_explicit_destination(
    scene_graph: SceneGraph,
    destination_phrase: str,
    *,
    language: str,
) -> DestinationResolution:
    result = resolve_reference(destination_phrase, scene_graph.nodes, language=language)
    if result.status != "resolved":
        return DestinationResolution(
            status=result.status,
            matched_rule=result.matched_rule,
            candidates=list(result.candidates),
            note=f"Destination phrase '{destination_phrase}' did not resolve cleanly",
        )

    regions = _regions_for_support(scene_graph, result.node_id or "")
    if not regions:
        return DestinationResolution(
            status="not_found",
            destination_id=result.node_id,
            matched_rule=result.matched_rule,
            note=f"Destination '{result.node_id}' has no placement region",
        )

    preferred = next(
        (region for region in regions if region.region_type == "inside-container"),
        None,
    )
    if preferred is None:
        return DestinationResolution(
            status="not_found",
            destination_id=result.node_id,
            matched_rule=result.matched_rule,
            note=f"Destination '{result.node_id}' is not a container placement target",
        )
    return DestinationResolution(
        status="resolved",
        destination_id=result.node_id,
        region=preferred,
        placement_reason="explicit_destination",
        matched_rule=result.matched_rule,
    )


def resolve_cleanup_destination(
    target_node: SceneNode,
    scene_graph: SceneGraph,
    ontology: OntologyStore,
) -> DestinationResolution:
    del ontology

    container_regions = [
        region
        for region in scene_graph.placement_regions
        if region.region_type == "inside-container" and region.support_id != target_node.id
    ]
    if container_regions:
        preferred_container_ids = [region.support_id for region in container_regions]
        region = select_default_destination(
            target_node.class_name,
            scene_graph.placement_regions,
            preferred_container_ids=preferred_container_ids,
        )
        return DestinationResolution(
            status="resolved",
            destination_id=region.support_id,
            region=region,
            placement_reason="preferred_container",
        )

    same_support_free_regions = [
        region
        for region in scene_graph.placement_regions
        if region.region_type == "free-space" and region.support_id == target_node.support_parent
    ]
    fallback_regions = same_support_free_regions or [
        region
        for region in scene_graph.placement_regions
        if region.region_type == "free-space" and region.support_id != target_node.id
    ]
    if fallback_regions:
        region = select_default_destination(
            target_node.class_name,
            fallback_regions,
            preferred_container_ids=[],
        )
        return DestinationResolution(
            status="resolved",
            destination_id=region.support_id,
            region=region,
            placement_reason="fallback_free_space",
        )

    return DestinationResolution(
        status="not_found",
        note=f"No placement destination available for '{target_node.id}'",
    )
