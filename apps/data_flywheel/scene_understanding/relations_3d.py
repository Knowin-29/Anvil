"""3D binary relation computation for scene understanding."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from apps.data_flywheel.geometry import (
    bbox_area,
    bbox_h_overlap_ratio,
    bbox_intersection,
    depth_diff_3d,
    is_above_3d,
    is_inside_3d,
    vertical_gap_3d,
)
from apps.data_flywheel.scene_understanding.pointcloud import (
    point_cloud_pose_from_sample,
    rasterize_polygon,
)
from apps.data_flywheel.scene_understanding.thresholds import SceneGraphThresholds
from apps.data_flywheel.types import SceneEdge, SceneNode


def _resolve_pose(node: SceneNode) -> Optional[Tuple[List[float], List[float], float]]:
    legacy_pose: Optional[Tuple[List[float], List[float], float]] = None
    if (
        node.valid_3d
        and node.center_3d is not None
        and node.size_3d is not None
        and node.mean_depth is not None
        and node.mean_depth > 0
    ):
        legacy_pose = list(node.center_3d), list(node.size_3d), float(node.mean_depth)

    sample_pose = point_cloud_pose_from_sample(list(node.point_cloud_sample or []))
    if sample_pose is not None:
        _, sample_size, _ = sample_pose
        reliable = len(list(node.point_cloud_sample or [])) >= 3 and max(sample_size) > 1e-4
        if reliable or legacy_pose is None:
            return sample_pose
    return legacy_pose


def _has_valid_3d(node: SceneNode) -> bool:
    return _resolve_pose(node) is not None


def _object_is_container(node: SceneNode) -> bool:
    return "container" in set(node.role_ids or []) or bool(node.is_container)


def _object_is_base_support_root(node: SceneNode) -> bool:
    return "base_support_root" in set(node.role_ids or [])


def _has_planar_or_stackable_2d(a: SceneNode, b: SceneNode) -> bool:
    tags = set(a.extra_tags) | set(b.extra_tags)
    return bool(tags & {"planar", "stackable_2d"})


def _stacking_2d_only(a: SceneNode, b: SceneNode, thresholds: SceneGraphThresholds) -> List[SceneEdge]:
    if bbox_h_overlap_ratio(a.bbox, b.bbox) <= thresholds.stacking_h_overlap:
        return []
    y_gap_ab = abs(float(a.bbox[3]) - float(b.bbox[1]))
    y_gap_ba = abs(float(b.bbox[3]) - float(a.bbox[1]))
    if y_gap_ab <= 5.0:
        upper, lower = a, b
    elif y_gap_ba <= 5.0:
        upper, lower = b, a
    else:
        return []
    return [
        SceneEdge(
            subject=upper.id,
            object=lower.id,
            predicate="stacked-on",
            source="2d",
            confidence=0.7,
            frame="world",
        ),
        SceneEdge(
            subject=lower.id,
            object=upper.id,
            predicate="supporting",
            source="2d",
            confidence=0.7,
            frame="world",
        ),
    ]


def _needs_sparse_stacking_fallback(edges: List[SceneEdge]) -> bool:
    return not any(edge.predicate == "stacked-on" for edge in edges)


def _range_overlap_ratio(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    inter = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    denom = max(min(a_max - a_min, b_max - b_min), 1e-6)
    return inter / denom


def _pose_bounds(node: SceneNode) -> Optional[Tuple[List[float], List[float], List[float]]]:
    pose = _resolve_pose(node)
    if pose is None:
        return None
    center, size, _ = pose
    half = [max(float(value) / 2.0, 1e-6) for value in size]
    mins = [float(center[idx]) - half[idx] for idx in range(3)]
    maxs = [float(center[idx]) + half[idx] for idx in range(3)]
    return center, mins, maxs


def _horizontal_overlap_ratio_3d(a: SceneNode, b: SceneNode) -> float:
    bounds_a = _pose_bounds(a)
    bounds_b = _pose_bounds(b)
    if bounds_a is None or bounds_b is None:
        return 0.0
    _, mins_a, maxs_a = bounds_a
    _, mins_b, maxs_b = bounds_b
    overlap_x = _range_overlap_ratio(mins_a[0], maxs_a[0], mins_b[0], maxs_b[0])
    overlap_z = _range_overlap_ratio(mins_a[2], maxs_a[2], mins_b[2], maxs_b[2])
    overlap = min(overlap_x, overlap_z)
    if overlap <= 1e-6:
        return bbox_h_overlap_ratio(a.bbox, b.bbox)
    return overlap


def _mask_overlap_ratio(
    a: SceneNode,
    b: SceneNode,
    image_size: Tuple[int, int],
) -> float:
    width, height = image_size
    if a.mask_polygon and b.mask_polygon and a.has_true_mask and b.has_true_mask:
        mask_a = rasterize_polygon(a.mask_polygon, width=width, height=height).astype(bool)
        mask_b = rasterize_polygon(b.mask_polygon, width=width, height=height).astype(bool)
        union = float(np.logical_or(mask_a, mask_b).sum())
        if union <= 0:
            return 0.0
        return float(np.logical_and(mask_a, mask_b).sum()) / union

    inter = bbox_intersection(a.bbox, b.bbox)
    area_a = bbox_area(a.bbox)
    area_b = bbox_area(b.bbox)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return max(inter / area_a, inter / area_b)


def _support_edges_from_nodes(nodes: List[SceneNode]) -> List[SceneEdge]:
    node_by_id: Dict[str, SceneNode] = {node.id: node for node in nodes}
    edges: List[SceneEdge] = []
    for node in nodes:
        if not node.support_parent or not node.support_type:
            continue
        parent = node_by_id.get(node.support_parent)
        if parent is None or _object_is_base_support_root(parent):
            continue
        if node.support_type == "inside":
            edges.append(
                SceneEdge(
                    subject=node.id,
                    object=parent.id,
                    predicate="inside",
                    source="3d",
                    confidence=0.92,
                    frame="world",
                )
            )
            continue

        edges.append(
            SceneEdge(
                subject=node.id,
                object=parent.id,
                predicate="on-top-of",
                source="3d",
                confidence=0.9,
                frame="world",
            )
        )
        edges.append(
            SceneEdge(
                subject=parent.id,
                object=node.id,
                predicate="under",
                source="3d",
                confidence=0.9,
                frame="world",
            )
        )
        if _horizontal_overlap_ratio_3d(node, parent) >= 0.35:
            edges.append(
                SceneEdge(
                    subject=node.id,
                    object=parent.id,
                    predicate="stacked-on",
                    source="3d",
                    confidence=0.88,
                    frame="world",
                )
            )
            edges.append(
                SceneEdge(
                    subject=parent.id,
                    object=node.id,
                    predicate="supporting",
                    source="3d",
                    confidence=0.88,
                    frame="world",
                )
            )
    return edges


def _pairwise_edges(
    a: SceneNode,
    b: SceneNode,
    image_size: Tuple[int, int],
    thresholds: SceneGraphThresholds,
) -> List[SceneEdge]:
    if _object_is_base_support_root(a) or _object_is_base_support_root(b):
        return []

    pose_a = _resolve_pose(a)
    pose_b = _resolve_pose(b)
    if pose_a is None or pose_b is None:
        return []

    center_a, size_a, depth_a = pose_a
    center_b, size_b, depth_b = pose_b
    bounds_a = _pose_bounds(a)
    bounds_b = _pose_bounds(b)
    if bounds_a is None or bounds_b is None:
        return []
    _, mins_a, maxs_a = bounds_a
    _, mins_b, maxs_b = bounds_b

    edges: List[SceneEdge] = []
    if _has_planar_or_stackable_2d(a, b):
        edges.extend(_stacking_2d_only(a, b, thresholds))
    overlap_xz = _horizontal_overlap_ratio_3d(a, b)
    overlap_mask = _mask_overlap_ratio(a, b, image_size)
    upper_fallback, lower_fallback = (
        (a, b)
        if (float(a.bbox[1]) + float(a.bbox[3])) / 2.0 <= (float(b.bbox[1]) + float(b.bbox[3])) / 2.0
        else (b, a)
    )

    vertical_delta = float(center_a[1]) - float(center_b[1])
    vertical_gap = vertical_gap_3d(center_a, size_a, center_b, size_b)
    if abs(vertical_delta) > thresholds.on_top_height_thresh:
        upper, lower = (a, b) if vertical_delta < 0 else (b, a)
        edges.append(
            SceneEdge(
                subject=upper.id,
                object=lower.id,
                predicate="above-world",
                source="3d",
                confidence=0.8,
                frame="world",
            )
        )
        edges.append(
            SceneEdge(
                subject=lower.id,
                object=upper.id,
                predicate="below-world",
                source="3d",
                confidence=0.8,
                frame="world",
            )
        )
        if is_above_3d(center_a, size_a, center_b, size_b, thresholds.on_top_height_thresh, thresholds.on_top_h_dist_ratio):
            edges.append(
                SceneEdge(
                    subject=a.id,
                    object=b.id,
                    predicate="on-top-of",
                    source="3d",
                    confidence=0.9,
                    frame="world",
                )
            )
            edges.append(
                SceneEdge(
                    subject=b.id,
                    object=a.id,
                    predicate="under",
                    source="3d",
                    confidence=0.9,
                    frame="world",
                )
            )
        elif is_above_3d(center_b, size_b, center_a, size_a, thresholds.under_height_thresh, thresholds.on_top_h_dist_ratio):
            edges.append(
                SceneEdge(
                    subject=b.id,
                    object=a.id,
                    predicate="on-top-of",
                    source="3d",
                    confidence=0.9,
                    frame="world",
                )
            )
            edges.append(
                SceneEdge(
                    subject=a.id,
                    object=b.id,
                    predicate="under",
                    source="3d",
                    confidence=0.9,
                    frame="world",
                )
            )
        if overlap_xz >= thresholds.stacking_h_overlap and abs(vertical_gap) <= thresholds.stacking_gap_3d:
            edges.append(
                SceneEdge(
                    subject=upper.id,
                    object=lower.id,
                    predicate="stacked-on",
                    source="3d",
                    confidence=0.88,
                    frame="world",
                )
            )
            edges.append(
                SceneEdge(
                    subject=lower.id,
                    object=upper.id,
                    predicate="supporting",
                    source="3d",
                    confidence=0.88,
                    frame="world",
                )
            )

    if _needs_sparse_stacking_fallback(edges):
        sparse_edges = _stacking_2d_only(a, b, thresholds)
        if sparse_edges:
            edges.extend(sparse_edges)
        elif bbox_h_overlap_ratio(a.bbox, b.bbox) >= thresholds.stacking_h_overlap:
            if upper_fallback is a:
                edges.append(
                    SceneEdge(
                        subject=a.id,
                        object=b.id,
                        predicate="on-top-of",
                        source="3d",
                        confidence=0.82,
                        frame="world",
                    )
                )
                edges.append(
                    SceneEdge(
                        subject=b.id,
                        object=a.id,
                        predicate="under",
                        source="3d",
                        confidence=0.82,
                        frame="world",
                    )
                )
            else:
                edges.append(
                    SceneEdge(
                        subject=b.id,
                        object=a.id,
                        predicate="on-top-of",
                        source="3d",
                        confidence=0.82,
                        frame="world",
                    )
                )
                edges.append(
                    SceneEdge(
                        subject=a.id,
                        object=b.id,
                        predicate="under",
                        source="3d",
                        confidence=0.82,
                        frame="world",
                    )
                )

    depth_delta = depth_diff_3d(center_a, center_b)
    if abs(depth_delta) > thresholds.front_depth_thresh:
        closer, farther = (a, b) if depth_delta > 0 else (b, a)
        edges.append(
            SceneEdge(
                subject=closer.id,
                object=farther.id,
                predicate="front",
                source="3d",
                confidence=0.85,
                frame="world",
            )
        )
        edges.append(
            SceneEdge(
                subject=farther.id,
                object=closer.id,
                predicate="behind",
                source="3d",
                confidence=0.85,
                frame="world",
            )
        )

    if _object_is_container(b):
        inside = is_inside_3d(center_a, size_a, center_b, size_b, thresholds.inside_margin)
        if inside:
            edges.append(
                SceneEdge(
                    subject=a.id,
                    object=b.id,
                    predicate="inside",
                    source="3d",
                    confidence=0.88,
                    frame="world",
                )
            )
    if _object_is_container(a):
        inside = is_inside_3d(center_b, size_b, center_a, size_a, thresholds.inside_margin)
        if inside:
            edges.append(
                SceneEdge(
                    subject=b.id,
                    object=a.id,
                    predicate="inside",
                    source="3d",
                    confidence=0.88,
                    frame="world",
                )
            )

    if overlap_mask > thresholds.occlusion_2d_overlap and abs(depth_delta) > thresholds.occlusion_depth_thresh:
        closer, farther = (a, b) if depth_delta > 0 else (b, a)
        edges.append(
            SceneEdge(
                subject=closer.id,
                object=farther.id,
                predicate="occluding",
                source="3d",
                confidence=0.85,
                frame="world",
            )
        )
        edges.append(
            SceneEdge(
                subject=farther.id,
                object=closer.id,
                predicate="occluded-by",
                source="3d",
                confidence=0.85,
                frame="world",
            )
        )

    euclidean = float(np.linalg.norm(np.asarray(center_a, dtype=np.float32) - np.asarray(center_b, dtype=np.float32)))
    if euclidean > max(thresholds.far_ratio, 0.2):
        edges.append(
            SceneEdge(
                subject=a.id,
                object=b.id,
                predicate="far",
                source="3d",
                confidence=0.7,
                frame="world",
            )
        )

    return edges


def _dedupe_edges(edges: List[SceneEdge]) -> List[SceneEdge]:
    keyed: Dict[tuple[str, str, str, str], SceneEdge] = {}
    for edge in edges:
        key = (edge.subject, edge.object, edge.predicate, edge.frame)
        current = keyed.get(key)
        if current is None or edge.confidence > current.confidence:
            keyed[key] = edge
    return list(keyed.values())


def compute_3d_relations(
    nodes: List[SceneNode],
    image_size: Tuple[int, int],
    thresholds: SceneGraphThresholds,
) -> List[SceneEdge]:
    """Compute 3D relations with support-first, point-cloud-first evidence."""

    edges: List[SceneEdge] = []
    edges.extend(_support_edges_from_nodes(nodes))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            if not (_has_valid_3d(a) and _has_valid_3d(b)):
                continue
            edges.extend(_pairwise_edges(a, b, image_size, thresholds))
    return _dedupe_edges(edges)
