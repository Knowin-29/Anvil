"""Placement-region derivation for tabletop/indoor scenes."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from apps.data_flywheel.scene_understanding.pointcloud import rasterize_polygon
from apps.data_flywheel.types import PlacementRegion, SceneNode


def _bbox_polygon(bbox: Sequence[float]) -> List[List[float]]:
    return [
        [float(bbox[0]), float(bbox[1])],
        [float(bbox[2]), float(bbox[1])],
        [float(bbox[2]), float(bbox[3])],
        [float(bbox[0]), float(bbox[3])],
    ]


def _has_role(node: SceneNode, role_id: str) -> bool:
    return role_id in set(node.role_ids or [])


def _polygon_or_bbox(node: SceneNode) -> List[List[float]]:
    if node.mask_polygon and len(node.mask_polygon) >= 3 and bool(getattr(node, "has_true_mask", False)):
        return [list(point) for point in node.mask_polygon]
    return _bbox_polygon(node.bbox)


def _mask_for_node(node: SceneNode, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    return rasterize_polygon(_polygon_or_bbox(node), width=width, height=height).astype(bool)


def _point_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    distances = (xs.astype(np.float32) - centroid_x) ** 2 + (ys.astype(np.float32) - centroid_y) ** 2
    best = int(np.argmin(distances))
    return [float(xs[best]) + 0.5, float(ys[best]) + 0.5]


def _children_of_support(support_id: str, nodes: Iterable[SceneNode]) -> List[SceneNode]:
    return [node for node in nodes if node.support_parent == support_id]


def _free_space_mask(support_node: SceneNode, nodes: List[SceneNode], image_size: Tuple[int, int]) -> np.ndarray:
    support_mask = _mask_for_node(support_node, image_size)
    occupied_mask = np.zeros_like(support_mask, dtype=bool)

    for child in _children_of_support(support_node.id, nodes):
        occupied_mask |= _mask_for_node(child, image_size)

    free_mask = np.logical_and(support_mask, np.logical_not(occupied_mask))
    if np.any(free_mask):
        return free_mask
    return support_mask


def compute_placement_regions(
    nodes: List[SceneNode],
    image_size: Tuple[int, int],
) -> List[PlacementRegion]:
    regions: List[PlacementRegion] = []

    for node in nodes:
        node_polygon = _polygon_or_bbox(node)
        node_mask = _mask_for_node(node, image_size)

        if _has_role(node, "container"):
            regions.append(
                PlacementRegion(
                    region_id=f"inside_{node.id}",
                    region_type="inside-container",
                    support_id=node.id,
                    anchor_ids=[node.id],
                    polygon_2d=node_polygon,
                    point_2d=_point_from_mask(node_mask),
                    reason="visible_container",
                )
            )

        if _has_role(node, "support_surface"):
            free_mask = _free_space_mask(node, nodes, image_size)
            regions.append(
                PlacementRegion(
                    region_id=f"free_{node.id}",
                    region_type="free-space",
                    support_id=node.id,
                    anchor_ids=[node.id],
                    polygon_2d=node_polygon,
                    point_2d=_point_from_mask(free_mask),
                    reason="support_surface_free_space",
                )
            )

    return regions


def select_default_destination(
    class_id: str,
    regions: List[PlacementRegion],
    preferred_container_ids: List[str],
) -> PlacementRegion:
    del class_id

    for container_id in preferred_container_ids:
        for region in regions:
            if region.support_id == container_id and region.region_type == "inside-container":
                return region

    for region in regions:
        if region.region_type == "inside-container":
            return region

    for region in regions:
        if region.region_type == "free-space":
            return region

    raise ValueError("No placement region available")
