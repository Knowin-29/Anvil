import numpy as np
from PIL import Image, ImageDraw

from apps.data_flywheel.scene_understanding.pointcloud import (
    extract_masked_point_cloud_sample,
    point_cloud_pose_from_sample,
    polygon_from_mask_file,
)


def test_extract_masked_point_cloud_uses_mask_not_bbox():
    depth_map = np.array(
        [
            [1.0, 1.0, 4.0],
            [1.0, 1.0, 4.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    mask_polygon = [[0, 0], [1.9, 0], [1.9, 1.9], [0, 1.9]]

    sample = extract_masked_point_cloud_sample(
        depth_map=depth_map,
        mask_polygon=mask_polygon,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        max_points=16,
    )

    assert sample
    assert all(point[2] < 2.0 for point in sample)


def test_extract_masked_point_cloud_filters_invalid_depths_and_caps_points():
    depth_map = np.array(
        [
            [1.0, 0.0, 2.0],
            [np.nan, 3.0, 4.0],
            [5.0, np.inf, 6.0],
        ],
        dtype=np.float32,
    )

    sample = extract_masked_point_cloud_sample(
        depth_map=depth_map,
        mask_polygon=[[0, 0], [2.9, 0], [2.9, 2.9], [0, 2.9]],
        intrinsics={"fx": 2.0, "fy": 2.0, "cx": 0.0, "cy": 0.0},
        max_points=3,
    )

    assert len(sample) == 3
    assert all(np.isfinite(point).all() for point in np.asarray(sample))
    assert all(point[2] > 0 for point in sample)


def test_extract_masked_point_cloud_rescales_polygon_to_depth_raster():
    depth_map = np.array(
        [
            [1.0, 4.0],
            [4.0, 4.0],
        ],
        dtype=np.float32,
    )

    sample = extract_masked_point_cloud_sample(
        depth_map=depth_map,
        mask_polygon=[[0, 0], [1.9, 0], [1.9, 1.9], [0, 1.9]],
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        source_image_size=(4, 4),
        max_points=16,
    )

    assert sample
    assert all(point[2] < 2.0 for point in sample)


def test_polygon_from_mask_file_restores_true_mask_outline(tmp_path):
    mask_path = tmp_path / "mask.png"
    image = Image.new("L", (20, 20), 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 12, 14), fill=255)
    image.save(mask_path)

    polygon = polygon_from_mask_file(str(mask_path))

    assert polygon is not None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    assert min(xs) <= 4.0
    assert max(xs) >= 12.0
    assert min(ys) <= 4.0
    assert max(ys) >= 14.0


def test_point_cloud_pose_from_sample_derives_center_size_and_depth():
    pose = point_cloud_pose_from_sample(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 4.0, 3.0],
            [2.0, 4.0, 3.0],
        ]
    )

    assert pose is not None
    center, size, mean_depth = pose
    assert center == [1.0, 2.0, 2.0]
    assert size == [2.0, 4.0, 2.0]
    assert mean_depth == 2.0
