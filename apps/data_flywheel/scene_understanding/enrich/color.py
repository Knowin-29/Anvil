"""HSV color extraction helpers."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PIL import Image


COLOR_BUCKETS = [
    ("red", [(0, 8), (170, 179)]),
    ("orange", [(9, 22)]),
    ("yellow", [(23, 33)]),
    ("green", [(34, 78)]),
    ("cyan", [(79, 99)]),
    ("blue", [(100, 130)]),
    ("purple", [(131, 155)]),
    ("pink", [(156, 169)]),
]


def hsv_to_color_name(hsv: np.ndarray | List[float]) -> Optional[str]:
    h, s, v = [float(x) for x in hsv]

    if v < 50:
        return "black"
    if s < 30 and v > 200:
        return "white"
    if s < 30:
        return "gray"
    if 10 <= h <= 25 and v < 150:
        return "brown"

    for name, ranges in COLOR_BUCKETS:
        for h_min, h_max in ranges:
            if h_min <= h <= h_max:
                return name
    return None


def _polygon_mask(image: Image.Image, mask_polygon: Optional[List[List[float]]]) -> Optional[np.ndarray]:
    if not mask_polygon or len(mask_polygon) < 3:
        return None

    import cv2

    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]
    mask_img = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(mask_polygon, dtype=np.int32)
    cv2.fillPoly(mask_img, [pts], 255)
    return mask_img


def _kmeans_centers(hsv_pixels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import KMeans
    except Exception:
        # Equal-brightness colors must not be averaged into a fictitious hue.
        # Deterministic farthest-point initialization plus Lloyd updates avoids
        # depending on scikit-learn for this small clustering problem.
        points = hsv_pixels.astype(np.float32)
        centers = [points[0]]
        for _ in range(1, k):
            distances = np.min(np.sum((points[:, None] - np.asarray(centers))**2, axis=2), axis=1)
            farthest = int(np.argmax(distances))
            if distances[farthest] == 0:
                break
            centers.append(points[farthest])
        centers = np.asarray(centers)
        for _ in range(30):
            labels = np.argmin(np.sum((points[:, None] - centers)**2, axis=2), axis=1)
            updated = np.asarray([points[labels == i].mean(axis=0) if np.any(labels == i)
                                  else centers[i] for i in range(len(centers))])
            if np.allclose(updated, centers):
                break
            centers = updated
        return labels.astype(np.int32), centers

    km = KMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(hsv_pixels)
    return km.labels_, km.cluster_centers_


def extract_color_attributes(
    image: Image.Image,
    mask_polygon: Optional[List[List[float]]],
    n_colors: int = 2,
    secondary_min_ratio: float = 0.15,
) -> Dict[str, Optional[object]]:
    if not mask_polygon or len(mask_polygon) < 3:
        return {"color_primary": None, "color_primary_hsv": None, "color_secondary": None}

    mask_img = _polygon_mask(image, mask_polygon)
    if mask_img is None:
        return {"color_primary": None, "color_primary_hsv": None, "color_secondary": None}

    import cv2

    img_array = np.array(image.convert("RGB"))
    masked_pixels = img_array[mask_img > 0]
    if len(masked_pixels) < 10:
        return {"color_primary": None, "color_primary_hsv": None, "color_secondary": None}

    hsv_pixels = cv2.cvtColor(masked_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    k = max(1, min(int(n_colors), len(hsv_pixels)))
    labels, centers = _kmeans_centers(hsv_pixels, k)

    unique, counts = np.unique(labels, return_counts=True)
    order = unique[np.argsort(-counts)]
    primary_idx = int(order[0])
    primary_center = centers[primary_idx]

    secondary_name = None
    if len(order) > 1:
        secondary_idx = int(order[1])
        secondary_ratio = counts[np.where(unique == secondary_idx)[0][0]] / len(labels)
        if secondary_ratio >= secondary_min_ratio:
            secondary_name = hsv_to_color_name(centers[secondary_idx])

    return {
        "color_primary": hsv_to_color_name(primary_center),
        "color_primary_hsv": [float(x) for x in primary_center],
        "color_secondary": secondary_name,
    }
