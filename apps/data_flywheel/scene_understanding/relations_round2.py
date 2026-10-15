"""Round 2 relation computation for scene understanding."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Set, Tuple

from apps.data_flywheel.geometry import bbox_center
from apps.data_flywheel.scene_understanding.thresholds import SceneGraphThresholds
from apps.data_flywheel.types import SceneEdge, SceneNode, TernaryRelation


def _node_pose(node: SceneNode) -> Tuple[float, float, float] | None:
    if node.point_cloud_sample:
        valid_points = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in node.point_cloud_sample
            if len(point) >= 3 and point[2] is not None
        ]
        if valid_points:
            xs = [point[0] for point in valid_points]
            ys = [point[1] for point in valid_points]
            zs = [point[2] for point in valid_points]
            return (
                sum(xs) / len(xs),
                sum(ys) / len(ys),
                sum(zs) / len(zs),
            )
    if node.valid_3d and node.center_3d and node.size_3d and node.mean_depth is not None:
        return float(node.center_3d[0]), float(node.center_3d[1]), float(node.mean_depth)
    return None


def _node_size(node: SceneNode) -> Tuple[float, float, float] | None:
    if node.point_cloud_sample:
        valid_points = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in node.point_cloud_sample
            if len(point) >= 3 and point[2] is not None
        ]
        if valid_points:
            xs = [point[0] for point in valid_points]
            ys = [point[1] for point in valid_points]
            zs = [point[2] for point in valid_points]
            return (
                max(xs) - min(xs),
                max(ys) - min(ys),
                max(zs) - min(zs),
            )
    if node.valid_3d and node.size_3d:
        return float(node.size_3d[0]), float(node.size_3d[1]), float(node.size_3d[2])
    return None


def _min_bbox_edge_distance(bbox_a: List[float], bbox_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0.0, max(ay1, by1) - min(ay2, by2))
    return math.hypot(dx, dy)


def _bbox_center_tuple(bbox: List[float]) -> Tuple[float, float]:
    cx, cy = bbox_center(bbox)
    return float(cx), float(cy)


def _points_nearly_collinear(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    tolerance_deg: float,
) -> bool:
    abx, aby = b[0] - a[0], b[1] - a[1]
    acx, acy = c[0] - a[0], c[1] - a[1]
    ab_len = math.hypot(abx, aby)
    ac_len = math.hypot(acx, acy)
    if ab_len < 1e-6 or ac_len < 1e-6:
        return False
    cross = abs(abx * acy - aby * acx)
    sin_theta = cross / (ab_len * ac_len)
    tolerance = math.sin(math.radians(max(0.0, tolerance_deg)))
    return sin_theta <= tolerance


def _is_between(a: SceneNode, b: SceneNode, c: SceneNode, t: SceneGraphThresholds) -> bool:
    acx, acy = _bbox_center_tuple(a.bbox)
    bcx, bcy = _bbox_center_tuple(b.bbox)
    ccx, ccy = _bbox_center_tuple(c.bbox)

    bc_len = math.hypot(ccx - bcx, ccy - bcy)
    if bc_len < 1.0:
        return False

    proj = ((acx - bcx) * (ccx - bcx) + (acy - bcy) * (ccy - bcy)) / (bc_len ** 2)
    if not (0.0 < proj < 1.0):
        return False

    cross = abs((ccx - bcx) * (bcy - acy) - (bcx - acx) * (ccy - bcy)) / bc_len
    return cross < t.between_perpendicular_dist_ratio * bc_len


def _not_inside(a: SceneNode, b: SceneNode) -> bool:
    pose_a = _node_pose(a)
    pose_b = _node_pose(b)
    size_b = _node_size(b)
    if pose_a is None or pose_b is None or size_b is None:
        ax_c, ay_c = _bbox_center_tuple(a.bbox)
        bx1, by1, bx2, by2 = b.bbox
        return not (bx1 <= ax_c <= bx2 and by1 <= ay_c <= by2)

    ac = [pose_a[0], pose_a[1], pose_a[2]]
    bc = [pose_b[0], pose_b[1], pose_b[2]]
    bs = [size_b[0], size_b[1], size_b[2]]
    for i in range(3):
        if abs(ac[i] - bc[i]) > bs[i] / 2:
            return True
    return False


def _has_planar_or_stackable_2d(a: SceneNode, b: SceneNode) -> bool:
    tags = set(a.extra_tags) | set(b.extra_tags)
    return bool(tags & {"planar", "stackable_2d"})


def compute_round2_binary_relations(
    nodes: List[SceneNode],
    image_size: Tuple[int, int],
    thresholds: SceneGraphThresholds,
) -> List[SceneEdge]:
    edges: List[SceneEdge] = []

    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            edge_dist = _min_bbox_edge_distance(a.bbox, b.bbox)

            if edge_dist < thresholds.touching_edge_dist_px:
                source = "2d"
                depth_ok = True
                pose_a = _node_pose(a)
                pose_b = _node_pose(b)
                if pose_a is not None and pose_b is not None:
                    depth_diff = abs(pose_a[2] - pose_b[2])
                    depth_ok = depth_diff < thresholds.touching_depth_diff
                    source = "2d+3d"
                if depth_ok:
                    edges.append(
                        SceneEdge(
                            subject=a.id,
                            object=b.id,
                            predicate="touching",
                            source=source,
                            confidence=0.85,
                            frame="image",
                        )
                    )
            elif edge_dist > thresholds.separated_min_gap_px:
                edges.append(
                    SceneEdge(
                        subject=a.id,
                        object=b.id,
                        predicate="separated",
                        source="2d",
                        confidence=0.7,
                        frame="image",
                    )
                )

    by_class: defaultdict[str, List[SceneNode]] = defaultdict(list)
    for node in nodes:
        by_class[node.class_name].append(node)

    for group in by_class.values():
        if len(group) < 3:
            continue
        for i, a in enumerate(group):
            for j in range(i + 1, len(group)):
                b = group[j]
                if any(
                    _points_nearly_collinear(
                        _bbox_center_tuple(a.bbox),
                        _bbox_center_tuple(b.bbox),
                        _bbox_center_tuple(c.bbox),
                        thresholds.aligned_angle_tolerance_deg,
                    )
                    for k, c in enumerate(group)
                    if k != i and k != j
                ):
                    edges.append(
                        SceneEdge(
                            subject=a.id,
                            object=b.id,
                            predicate="aligned-with",
                            source="2d",
                            confidence=0.8,
                            frame="image",
                        )
                    )

    for a in nodes:
        for b in nodes:
            if a.id == b.id or not b.is_container:
                continue
            if _not_inside(a, b):
                edges.append(
                    SceneEdge(
                        subject=a.id,
                        object=b.id,
                        predicate="outside",
                        source="derived",
                        confidence=0.75,
                        frame="world" if _node_pose(a) is not None and _node_pose(b) is not None and _node_size(b) is not None else "image",
                    )
                )

    return edges


def compute_between_relations(
    nodes: List[SceneNode],
    thresholds: SceneGraphThresholds,
) -> List[TernaryRelation]:
    relations: List[TernaryRelation] = []
    seen: Set[Tuple[str, str, str]] = set()

    for a in nodes:
        others = [node for node in nodes if node.id != a.id]
        for i, b in enumerate(others):
            for j in range(i + 1, len(others)):
                c = others[j]
                reference_a, reference_b = sorted((b.id, c.id))
                key = (a.id, reference_a, reference_b)
                if key in seen:
                    continue
                if _is_between(a, b, c, thresholds):
                    seen.add(key)
                    relations.append(
                        TernaryRelation(
                            subject=a.id,
                            reference_a=reference_a,
                            reference_b=reference_b,
                            predicate="between",
                            confidence=0.75,
                        )
                    )

    return relations
