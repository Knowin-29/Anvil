"""Support graph derivation for tabletop/indoor scenes."""

from __future__ import annotations

from typing import List, Optional, Tuple

from apps.data_flywheel.geometry import bbox_area, bbox_contains, bbox_center, bbox_h_overlap_ratio, bbox_iou
from apps.data_flywheel.scene_understanding.pointcloud import polygon_bbox
from apps.data_flywheel.types import SceneNode, SupportRelation


_MIN_SUPPORT_OVERLAP = 0.35
_MAX_VERTICAL_MARGIN_PX = 24.0
_MIN_INSIDE_CONTAINMENT = 0.75
_DEPTH_TOLERANCE = 0.05
_MIN_PARENT_AREA_RATIO = 1.1
_AMBIGUOUS_SUPPORT_IOU = 0.9
_AMBIGUOUS_SUPPORT_AREA_RATIO = 0.9


def _node_depth(node: SceneNode) -> Optional[float]:
    if node.point_cloud_sample:
        z_values = [float(point[2]) for point in node.point_cloud_sample if len(point) >= 3]
        if z_values:
            return sum(z_values) / len(z_values)
    if node.mean_depth is not None:
        return float(node.mean_depth)
    return None


def _has_role(node: SceneNode, role_id: str) -> bool:
    return role_id in set(node.role_ids or [])


def _can_support(node: SceneNode) -> bool:
    return _has_role(node, "support_surface") or _has_role(node, "container")


def _geometry_bbox(node: SceneNode) -> List[float]:
    polygon_box = polygon_bbox(node.mask_polygon)
    if polygon_box is not None and node.has_true_mask:
        return polygon_box
    return list(node.bbox)


def _geometry_center(node: SceneNode) -> Tuple[float, float]:
    return bbox_center(_geometry_bbox(node))


def _geometry_area(node: SceneNode) -> float:
    return bbox_area(_geometry_bbox(node))


def _geometry_overlap_ratio(child: SceneNode, parent: SceneNode) -> float:
    return bbox_h_overlap_ratio(_geometry_bbox(child), _geometry_bbox(parent))


def _geometry_iou(child: SceneNode, parent: SceneNode) -> float:
    return bbox_iou(_geometry_bbox(child), _geometry_bbox(parent))


def _area_similarity(a: SceneNode, b: SceneNode) -> float:
    area_a = _geometry_area(a)
    area_b = _geometry_area(b)
    larger = max(area_a, area_b, 1e-6)
    smaller = min(area_a, area_b)
    return smaller / larger


def _is_ambiguous_same_scale_support_pair(a: SceneNode, b: SceneNode) -> bool:
    if _geometry_iou(a, b) < _AMBIGUOUS_SUPPORT_IOU:
        return False
    if _area_similarity(a, b) < _AMBIGUOUS_SUPPORT_AREA_RATIO:
        return False

    depth_a = _node_depth(a)
    depth_b = _node_depth(b)
    if depth_a is None or depth_b is None:
        return True
    return abs(depth_a - depth_b) <= _DEPTH_TOLERANCE


def _supports_inside(child: SceneNode, parent: SceneNode) -> bool:
    if not _has_role(parent, "container"):
        return False
    parent_bbox = _geometry_bbox(parent)
    child_bbox = _geometry_bbox(child)
    if _geometry_area(parent) < _geometry_area(child) * _MIN_PARENT_AREA_RATIO:
        return False
    if bbox_contains(parent_bbox, child_bbox, thresh=_MIN_INSIDE_CONTAINMENT):
        return True
    center_x, center_y = _geometry_center(child)
    return (
        parent_bbox[0] <= center_x <= parent_bbox[2]
        and parent_bbox[1] <= center_y <= parent_bbox[3]
    )


def _supports_on_top(child: SceneNode, parent: SceneNode) -> bool:
    if not _can_support(parent):
        return False
    if _is_ambiguous_same_scale_support_pair(child, parent):
        return False
    child_bbox = _geometry_bbox(child)
    parent_bbox = _geometry_bbox(parent)
    if _geometry_overlap_ratio(child, parent) < _MIN_SUPPORT_OVERLAP:
        return False

    child_bottom = float(child_bbox[3])
    parent_top = float(parent_bbox[1])
    parent_bottom = float(parent_bbox[3])
    if child_bottom < parent_top - _MAX_VERTICAL_MARGIN_PX:
        return False
    if child_bottom > parent_bottom + _MAX_VERTICAL_MARGIN_PX:
        return False

    child_depth = _node_depth(child)
    parent_depth = _node_depth(parent)
    if (
        child_depth is not None
        and parent_depth is not None
        and parent_depth + _DEPTH_TOLERANCE < child_depth
    ):
        return False
    return True


def _clear_support(node: SceneNode) -> None:
    node.support_parent = None
    node.support_type = None
    node.support_excluded = False
    node.excluded_from_action_graph = _has_role(node, "base_support_root")


def _assign_support(node: SceneNode, parent: SceneNode, relation: str) -> None:
    node.support_parent = parent.id
    node.support_type = relation
    node.support_excluded = _has_role(parent, "base_support_root")
    node.excluded_from_action_graph = _has_role(node, "base_support_root")


def _preferred_cycle_parent(node_a: SceneNode, node_b: SceneNode) -> Optional[SceneNode]:
    if _has_role(node_a, "base_support_root") and not _has_role(node_b, "base_support_root"):
        return node_a
    if _has_role(node_b, "base_support_root") and not _has_role(node_a, "base_support_root"):
        return node_b
    if _is_ambiguous_same_scale_support_pair(node_a, node_b):
        return None

    area_a = _geometry_area(node_a)
    area_b = _geometry_area(node_b)
    if area_a >= area_b * _MIN_PARENT_AREA_RATIO:
        return node_a
    if area_b >= area_a * _MIN_PARENT_AREA_RATIO:
        return node_b

    depth_a = _node_depth(node_a)
    depth_b = _node_depth(node_b)
    if depth_a is not None and depth_b is not None:
        if depth_a >= depth_b + _DEPTH_TOLERANCE:
            return node_a
        if depth_b >= depth_a + _DEPTH_TOLERANCE:
            return node_b

    return None


def _best_support_parent(child: SceneNode, nodes: List[SceneNode]) -> tuple[Optional[SceneNode], Optional[str]]:
    inside_candidates: List[SceneNode] = []
    on_top_candidates: List[SceneNode] = []

    for candidate in nodes:
        if candidate.id == child.id:
            continue
        if _supports_inside(child, candidate):
            inside_candidates.append(candidate)
            continue
        if _supports_on_top(child, candidate):
            on_top_candidates.append(candidate)

    if inside_candidates:
        parent = min(inside_candidates, key=lambda node: (_geometry_area(node), _geometry_center(node)[1]))
        return parent, "inside"
    if on_top_candidates:
        non_base_roots = [node for node in on_top_candidates if not _has_role(node, "base_support_root")]
        pool = non_base_roots or on_top_candidates
        parent = min(pool, key=lambda node: (_geometry_area(node), _geometry_center(node)[1]))
        return parent, "on-top-of"
    return None, None


def compute_support_relations(nodes: List[SceneNode]) -> List[SupportRelation]:
    for node in nodes:
        parent, relation = _best_support_parent(node, nodes)
        if parent is None or relation is None:
            _clear_support(node)
            continue

        _assign_support(node, parent, relation)

    node_by_id = {node.id: node for node in nodes}
    resolved_pairs: set[tuple[str, str]] = set()
    for node in nodes:
        parent_id = node.support_parent
        if parent_id is None:
            continue
        pair_key = tuple(sorted((node.id, parent_id)))
        if pair_key in resolved_pairs:
            continue
        parent = node_by_id.get(parent_id)
        if parent is None or parent.support_parent != node.id:
            continue

        preferred_parent = _preferred_cycle_parent(node, parent)
        if preferred_parent is None:
            _clear_support(node)
            _clear_support(parent)
        elif preferred_parent.id == parent.id:
            _clear_support(parent)
        else:
            _clear_support(node)
        resolved_pairs.add(pair_key)

    relations: List[SupportRelation] = []
    for node in nodes:
        if node.support_parent is None or node.support_type is None:
            continue
        relations.append(
            SupportRelation(
                subject=node.id,
                object=node.support_parent,
                relation=node.support_type,
                confidence=0.9,
            )
        )

    return relations
