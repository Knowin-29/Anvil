"""公共几何工具函数 — 所有模块共享，避免重复实现。

包含 2D bbox 工具和 3D 空间关系工具。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


# =====================================================================
# 2D bbox 工具
# =====================================================================

def bbox_center(b: List[float]) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def bbox_area(b: List[float]) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def bbox_intersection(a: List[float], b: List[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(a: List[float], b: List[float]) -> float:
    inter = bbox_intersection(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_distance(a: List[float], b: List[float]) -> float:
    cx1, cy1 = bbox_center(a)
    cx2, cy2 = bbox_center(b)
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def image_diagonal(w: int, h: int) -> float:
    """图片对角线长度。"""
    return (w ** 2 + h ** 2) ** 0.5


def bbox_contains(outer: List[float], inner: List[float], thresh: float = 0.6) -> bool:
    """判断 outer 是否包含 inner（inner 面积的 thresh 比例在 outer 内）。"""
    inner_area = bbox_area(inner)
    if inner_area == 0:
        return False
    inter = bbox_intersection(outer, inner)
    return (inter / inner_area) >= thresh


def bbox_h_overlap_ratio(a: List[float], b: List[float]) -> float:
    """两个 bbox 在水平方向上的重叠比例（相对于较小宽度）。"""
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    min_w = min(a[2] - a[0], b[2] - b[0])
    return overlap / min_w if min_w > 0 else 0.0


def bbox_aspect_ratio(b: List[float]) -> float:
    """bbox 宽高比 (w/h)。"""
    w = max(1e-6, b[2] - b[0])
    h = max(1e-6, b[3] - b[1])
    return w / h


# =====================================================================
# 3D 空间关系工具
# =====================================================================

def point3d_distance(a: List[float], b: List[float]) -> float:
    """两个 3D 点之间的欧氏距离。"""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a[:3], b[:3])))


def horizontal_distance_3d(a: List[float], b: List[float]) -> float:
    """两个 3D 点在水平面 (XZ) 上的距离。"""
    dx = a[0] - b[0]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dz * dz)


def is_above_3d(
    a_center: List[float], a_size: List[float],
    b_center: List[float], b_size: List[float],
    height_thresh: float = 0.02,
    h_dist_ratio: float = 1.5,
) -> bool:
    """判断 A 是否在 B 的正上方（3D 空间）。

    条件：A 的 Y 中心高于 B 的 Y 中心（注意相机坐标系 Y 轴向下），
    且水平距离不超过 B 尺寸的 h_dist_ratio 倍。
    """
    # 相机坐标系中 Y 轴向下，所以 "上方" 意味着 Y 值更小
    y_diff = b_center[1] - a_center[1]  # 正值 = A 在 B 上方
    if y_diff < height_thresh:
        return False
    h_dist = horizontal_distance_3d(a_center, b_center)
    max_h_dist = max(b_size[0], b_size[2]) * h_dist_ratio if b_size[0] > 0 else float("inf")
    return h_dist < max_h_dist


def is_inside_3d(
    inner_center: List[float], inner_size: List[float],
    outer_center: List[float], outer_size: List[float],
    margin: float = 0.5,
) -> bool:
    """判断 inner 的 3D 中心是否在 outer 的 3D 包围盒内（带 margin 容差）。"""
    for i in range(3):
        half = outer_size[i] * (0.5 + margin)
        if abs(inner_center[i] - outer_center[i]) > half:
            return False
    return True


def is_in_front_3d(
    a_center: List[float], b_center: List[float],
    depth_thresh: float = 0.05,
) -> bool:
    """判断 A 是否在 B 的前方（深度更小 = 离相机更近）。"""
    return (b_center[2] - a_center[2]) > depth_thresh


def depth_diff_3d(a_center: List[float], b_center: List[float]) -> float:
    """A 和 B 的深度差（正值 = A 更近）。"""
    return b_center[2] - a_center[2]


def vertical_gap_3d(
    a_center: List[float], a_size: List[float],
    b_center: List[float], b_size: List[float],
) -> float:
    """A 底面到 B 顶面的垂直间距（相机坐标系 Y 向下）。

    返回正值表示 A 在 B 上方且有间距，负值表示重叠。
    """
    # A 底面 = a_center[1] + a_size[1]/2
    # B 顶面 = b_center[1] - b_size[1]/2
    a_bottom = a_center[1] + a_size[1] / 2
    b_top = b_center[1] - b_size[1] / 2
    return b_top - a_bottom  # 负值 = A 底面低于 B 顶面 = 接触/重叠

