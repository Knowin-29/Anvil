"""2D binary relation computation for scene understanding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from apps.data_flywheel.geometry import (
    bbox_area,
    bbox_center,
    bbox_distance,
    bbox_iou,
    image_diagonal,
)
from apps.data_flywheel.scene_understanding.pointcloud import rasterize_polygon
from apps.data_flywheel.scene_understanding.thresholds import SceneGraphThresholds
from apps.data_flywheel.types import SceneEdge, SceneNode


@dataclass
class _Node2DGeometry:
    bbox: List[float]
    center: Tuple[float, float]
    area: float
    mask: Optional[np.ndarray]


def _mask_geometry(node: SceneNode, image_size: Tuple[int, int]) -> Optional[_Node2DGeometry]:
    if not node.mask_polygon or len(node.mask_polygon) < 3 or not node.has_true_mask:
        return None
    width, height = image_size
    mask = rasterize_polygon(node.mask_polygon, width=width, height=height)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    center = (float(xs.mean()), float(ys.mean()))
    return _Node2DGeometry(
        bbox=bbox,
        center=center,
        area=float(mask.sum()),
        mask=mask.astype(bool),
    )


def _node_geometry(node: SceneNode, image_size: Tuple[int, int]) -> _Node2DGeometry:
    mask_geometry = _mask_geometry(node, image_size)
    if mask_geometry is not None:
        return mask_geometry
    return _Node2DGeometry(
        bbox=list(node.bbox),
        center=bbox_center(node.bbox),
        area=bbox_area(node.bbox),
        mask=None,
    )


def _mask_iou(a: _Node2DGeometry, b: _Node2DGeometry) -> float:
    if a.mask is None or b.mask is None:
        return bbox_iou(a.bbox, b.bbox)
    inter = float(np.logical_and(a.mask, b.mask).sum())
    union = float(np.logical_or(a.mask, b.mask).sum())
    return inter / union if union > 0 else 0.0


def _infer_2d_position(
    a: _Node2DGeometry,
    b: _Node2DGeometry,
    image_size: Tuple[int, int],
    thresholds: SceneGraphThresholds,
) -> Optional[str]:
    cx_a, cy_a = a.center
    cx_b, cy_b = b.center
    dx, dy = cx_a - cx_b, cy_a - cy_b

    width, height = image_size
    b_width = b.bbox[2] - b.bbox[0]
    b_height = b.bbox[3] - b.bbox[1]
    min_dx = max(b_width * thresholds.position_min_ratio, width * thresholds.position_min_img_ratio)
    min_dy = max(b_height * thresholds.position_min_ratio, height * thresholds.position_min_img_ratio)

    if abs(dx) < min_dx and abs(dy) < min_dy:
        return None
    if abs(dx) > abs(dy):
        return "right-of" if dx > 0 else "left-of"
    return "below-image" if dy > 0 else "above-image"


def _infer_next_to(
    a: _Node2DGeometry,
    b: _Node2DGeometry,
    image_size: Tuple[int, int],
    thresholds: SceneGraphThresholds,
) -> bool:
    width, height = image_size
    img_diag = image_diagonal(width, height)
    h_gap = max(0.0, max(a.bbox[0], b.bbox[0]) - min(a.bbox[2], b.bbox[2]))
    v_gap = max(0.0, max(a.bbox[1], b.bbox[1]) - min(a.bbox[3], b.bbox[3]))
    edge_dist = (h_gap ** 2 + v_gap ** 2) ** 0.5
    return edge_dist < img_diag * thresholds.next_to_max_gap


def _infer_far(
    a: _Node2DGeometry,
    b: _Node2DGeometry,
    image_size: Tuple[int, int],
    thresholds: SceneGraphThresholds,
) -> bool:
    width, height = image_size
    img_diag = image_diagonal(width, height)
    return bbox_distance(a.bbox, b.bbox) > img_diag * thresholds.far_ratio


def _infer_size_relation(a: _Node2DGeometry, b: _Node2DGeometry, thresholds: SceneGraphThresholds) -> Optional[str]:
    area_a = a.area
    area_b = b.area
    if area_a == 0 or area_b == 0:
        return None
    ratio = area_a / area_b
    if ratio > thresholds.size_ratio:
        return "larger-than"
    if ratio < (1.0 / thresholds.size_ratio):
        return "smaller-than"
    return None


def _containment_ratio(container: _Node2DGeometry, child: _Node2DGeometry) -> float:
    if child.area <= 0:
        return 0.0
    if container.mask is not None and child.mask is not None:
        inter = float(np.logical_and(container.mask, child.mask).sum())
        return inter / max(child.area, 1.0)

    child_x1, child_y1, child_x2, child_y2 = child.bbox
    cont_x1, cont_y1, cont_x2, cont_y2 = container.bbox
    inter_w = max(0.0, min(child_x2, cont_x2) - max(child_x1, cont_x1))
    inter_h = max(0.0, min(child_y2, cont_y2) - max(child_y1, cont_y1))
    return (inter_w * inter_h) / max(child.area, 1.0)


def _is_container(node: SceneNode) -> bool:
    return "container" in set(node.role_ids or []) or bool(node.is_container)


def _skip_pair(node_a: SceneNode, node_b: SceneNode) -> bool:
    if node_a.id == node_b.id:
        return True
    return "base_support_root" in set(node_a.role_ids or []) or "base_support_root" in set(node_b.role_ids or [])


def compute_2d_relations(
    nodes: List[SceneNode],
    image_size: Tuple[int, int],
    thresholds: SceneGraphThresholds,
) -> List[SceneEdge]:
    """Compute 2D relations using the legacy scene_graph heuristics."""

    width, height = image_size
    img_diag = image_diagonal(width, height)
    edges: List[SceneEdge] = []
    node_geometry = {node.id: _node_geometry(node, image_size) for node in nodes}

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_a = nodes[i]
            node_b = nodes[j]
            if _skip_pair(node_a, node_b):
                continue
            geom_a = node_geometry[node_a.id]
            geom_b = node_geometry[node_b.id]

            position_predicate = _infer_2d_position(geom_a, geom_b, image_size, thresholds)
            if position_predicate:
                edges.append(
                    SceneEdge(
                        subject=node_a.id,
                        object=node_b.id,
                        predicate=position_predicate,
                        source="2d",
                        confidence=0.9,
                        frame="image",
                    )
                )

            if _infer_next_to(geom_a, geom_b, image_size, thresholds):
                edges.append(
                    SceneEdge(
                        subject=node_a.id,
                        object=node_b.id,
                        predicate="next-to",
                        source="2d",
                        confidence=0.85,
                        frame="image",
                    )
                )

            if _infer_far(geom_a, geom_b, image_size, thresholds):
                edges.append(
                    SceneEdge(
                        subject=node_a.id,
                        object=node_b.id,
                        predicate="far",
                        source="2d",
                        confidence=0.75,
                        frame="image",
                    )
                )

            size_predicate = _infer_size_relation(geom_a, geom_b, thresholds)
            if size_predicate:
                edges.append(
                    SceneEdge(
                        subject=node_a.id,
                        object=node_b.id,
                        predicate=size_predicate,
                        source="2d",
                        confidence=0.9,
                        frame="image",
                    )
                )

            iou = _mask_iou(geom_a, geom_b)
            if iou > thresholds.overlap_iou:
                edges.append(
                    SceneEdge(
                        subject=node_a.id,
                        object=node_b.id,
                        predicate="overlapping",
                        source="2d",
                        confidence=round(min(1.0, iou * 3), 2),
                        frame="image",
                    )
                )

            if bbox_distance(geom_a.bbox, geom_b.bbox) < img_diag * thresholds.near_ratio:
                edges.append(
                    SceneEdge(
                        subject=node_a.id,
                        object=node_b.id,
                        predicate="near",
                        source="2d",
                        confidence=0.8,
                        frame="image",
                    )
                )

            if _is_container(node_b):
                containment = _containment_ratio(geom_b, geom_a)
                if containment >= 0.75:
                    edges.append(
                        SceneEdge(
                            subject=node_a.id,
                            object=node_b.id,
                            predicate="inside",
                            source="2d",
                            confidence=round(min(0.95, containment), 2),
                            frame="image",
                        )
                    )
            if _is_container(node_a):
                containment = _containment_ratio(geom_a, geom_b)
                if containment >= 0.75:
                    edges.append(
                        SceneEdge(
                            subject=node_b.id,
                            object=node_a.id,
                            predicate="inside",
                            source="2d",
                            confidence=round(min(0.95, containment), 2),
                            frame="image",
                        )
                    )

    return edges
