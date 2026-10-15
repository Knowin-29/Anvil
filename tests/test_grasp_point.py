import cv2
import numpy as np

from apps.data_flywheel.perception.grasp_point import grasp_point_from_mask


def test_grasp_point_of_rectangle_is_center():
    assert grasp_point_from_mask([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]) == [50.0, 25.0]


def test_grasp_point_of_concave_uses_centroid():
    mask_polygon = [
        [0.0, 0.0],
        [6.0, 0.0],
        [6.0, 2.0],
        [2.0, 2.0],
        [2.0, 6.0],
        [0.0, 6.0],
    ]

    assert grasp_point_from_mask(mask_polygon) == [2.2, 2.2]


def test_grasp_point_empty_returns_origin():
    assert grasp_point_from_mask([]) == [0.0, 0.0]


def test_grasp_point_single_point():
    assert grasp_point_from_mask([[50.0, 50.0]]) == [50.0, 50.0]


def test_grasp_point_always_inside_convex_polygon():
    polygons = [
        [[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]],
        [[0.0, 0.0], [12.0, 1.0], [14.0, 8.0], [7.0, 15.0], [1.0, 10.0]],
        [[3.0, 1.0], [9.0, 2.0], [12.0, 7.0], [8.0, 14.0], [2.0, 11.0], [0.0, 5.0]],
    ]

    for polygon in polygons:
        point = grasp_point_from_mask(polygon)
        contour = np.array(polygon, dtype=np.float32)
        assert cv2.pointPolygonTest(contour.astype(np.int32), (point[0], point[1]), False) >= 0
