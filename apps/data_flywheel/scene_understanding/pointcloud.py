"""Mask-aligned point-cloud extraction helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from apps.data_flywheel.perception.mask_io import load_mask_png


def polygon_bbox(mask_polygon: Optional[List[List[float]]]) -> Optional[List[float]]:
    if not mask_polygon:
        return None
    xs = [float(point[0]) for point in mask_polygon if len(point) >= 2]
    ys = [float(point[1]) for point in mask_polygon if len(point) >= 2]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def rasterize_polygon(
    mask_polygon: Optional[List[List[float]]],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not mask_polygon or len(mask_polygon) < 3 or width <= 0 or height <= 0:
        return mask

    import cv2

    pts = np.array(mask_polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _remap_polygon_to_raster(
    mask_polygon: Optional[List[List[float]]],
    *,
    source_image_size: Optional[Any],
    target_width: int,
    target_height: int,
) -> Optional[List[List[float]]]:
    if not mask_polygon:
        return mask_polygon
    if (
        not isinstance(source_image_size, (list, tuple))
        or len(source_image_size) != 2
    ):
        return mask_polygon

    source_width = float(source_image_size[0])
    source_height = float(source_image_size[1])
    if source_width <= 0 or source_height <= 0:
        return mask_polygon
    if int(round(source_width)) == target_width and int(round(source_height)) == target_height:
        return mask_polygon

    scale_x = float(target_width) / source_width
    scale_y = float(target_height) / source_height
    return [
        [float(point[0]) * scale_x, float(point[1]) * scale_y]
        for point in mask_polygon
    ]


def _sample_points(points: List[List[float]], max_points: int) -> List[List[float]]:
    if max_points <= 0 or not points:
        return []
    if len(points) <= max_points:
        return points

    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
    return [points[idx] for idx in indices.tolist()]


def point_cloud_pose_from_sample(
    point_cloud_sample: List[List[float]],
) -> Optional[tuple[List[float], List[float], float]]:
    if not point_cloud_sample:
        return None
    points = np.asarray(point_cloud_sample, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    if valid.sum() == 0:
        return None
    points = points[valid]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = ((mins + maxs) / 2.0).tolist()
    size = np.maximum(maxs - mins, 1e-6).tolist()
    mean_depth = float(points[:, 2].mean())
    return [float(v) for v in center], [float(v) for v in size], mean_depth


def polygon_from_mask_file(mask_file: Optional[str]) -> Optional[List[List[float]]]:
    if not isinstance(mask_file, str) or not mask_file or not os.path.exists(mask_file):
        return None

    try:
        mask = load_mask_png(mask_file)
    except Exception:
        return None
    if mask.ndim != 2:
        return None

    import cv2

    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None
    epsilon = 0.002 * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    polygon = [[float(point[0][0]), float(point[0][1])] for point in simplified]
    return polygon or None


def extract_masked_point_cloud_sample(
    *,
    depth_map: Any,
    mask_polygon: Optional[List[List[float]]],
    intrinsics: Dict[str, float],
    source_image_size: Optional[Any] = None,
    max_points: int = 256,
) -> List[List[float]]:
    depth_array = np.asarray(depth_map, dtype=np.float32)
    if depth_array.ndim != 2:
        return []

    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    if fx <= 0 or fy <= 0:
        return []

    remapped_polygon = _remap_polygon_to_raster(
        mask_polygon,
        source_image_size=source_image_size,
        target_width=depth_array.shape[1],
        target_height=depth_array.shape[0],
    )
    mask = rasterize_polygon(
        remapped_polygon,
        width=depth_array.shape[1],
        height=depth_array.shape[0],
    )
    ys, xs = np.nonzero(mask)

    points: List[List[float]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        z = float(depth_array[y, x])
        if not np.isfinite(z) or z <= 0:
            continue
        x3 = (float(x) - cx) * z / fx
        y3 = (float(y) - cy) * z / fy
        points.append([x3, y3, z])
    return _sample_points(points, max_points)


def extract_masked_xyz_point_cloud_sample(
    *,
    xyz_map: Any,
    mask_polygon: Optional[List[List[float]]],
    source_image_size: Optional[Any] = None,
    max_points: int = 256,
) -> List[List[float]]:
    xyz_array = np.asarray(xyz_map, dtype=np.float32)
    if xyz_array.ndim != 3 or xyz_array.shape[2] != 3:
        return []

    remapped_polygon = _remap_polygon_to_raster(
        mask_polygon,
        source_image_size=source_image_size,
        target_width=xyz_array.shape[1],
        target_height=xyz_array.shape[0],
    )
    mask = rasterize_polygon(
        remapped_polygon,
        width=xyz_array.shape[1],
        height=xyz_array.shape[0],
    )
    ys, xs = np.nonzero(mask)

    points: List[List[float]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        point = xyz_array[y, x]
        if not np.all(np.isfinite(point)) or float(point[2]) <= 0:
            continue
        points.append([float(point[0]), float(point[1]), float(point[2])])
    return _sample_points(points, max_points)


def load_xyz_map(depth_info: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not depth_info:
        return None

    xyz_map = depth_info.get("xyz_map")
    if xyz_map is not None:
        xyz_array = np.asarray(xyz_map, dtype=np.float32)
        if xyz_array.ndim == 3 and xyz_array.shape[2] == 3:
            return xyz_array

    xyz_map_path = depth_info.get("xyz_map_path")
    if not isinstance(xyz_map_path, str) or not xyz_map_path or not os.path.exists(xyz_map_path):
        return None

    try:
        xyz_array = np.load(xyz_map_path)
    except Exception:
        return None
    if xyz_array.ndim != 3 or xyz_array.shape[2] != 3:
        return None
    return xyz_array.astype(np.float32)
