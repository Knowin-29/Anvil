from __future__ import annotations

from typing import List

import cv2
import numpy as np


def grasp_point_from_mask(mask_polygon: List[List[float]]) -> List[float]:
    if not mask_polygon:
        return [0.0, 0.0]

    if len(mask_polygon) < 3:
        xs = [point[0] for point in mask_polygon]
        ys = [point[1] for point in mask_polygon]
        return [float(sum(xs) / len(xs)), float(sum(ys) / len(ys))]

    points = np.asarray(mask_polygon, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    center_x, center_y = rect[0]
    center = (float(center_x), float(center_y))

    if cv2.pointPolygonTest(points.astype(np.int32), center, False) >= 0:
        return [float(center_x), float(center_y)]

    contour = points.astype(np.int32)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return [float(center_x), float(center_y)]

    centroid_x = moments["m10"] / moments["m00"]
    centroid_y = moments["m01"] / moments["m00"]
    return [float(centroid_x), float(centroid_y)]
