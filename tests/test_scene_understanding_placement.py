from __future__ import annotations

from apps.data_flywheel.scene_understanding.placement import (
    compute_placement_regions,
    select_default_destination,
)
from apps.data_flywheel.types import SceneNode


def test_select_default_placement_prefers_container_over_free_space():
    bowl = SceneNode(
        id="bowl_0",
        class_name="bowl",
        bbox=[200, 200, 320, 320],
        role_ids=["container", "support_surface"],
        point_cloud_sample=[[0.0, 0.0, 0.8]],
        mask_polygon=[[200, 200], [320, 200], [320, 320], [200, 320]],
    )
    table = SceneNode(
        id="table_0",
        class_name="table",
        bbox=[0, 180, 640, 480],
        role_ids=["support_surface", "base_support_root"],
        point_cloud_sample=[[0.0, 0.0, 1.0]],
        mask_polygon=[[0, 180], [640, 180], [640, 480], [0, 480]],
    )

    regions = compute_placement_regions([table, bowl], image_size=(640, 480))
    selected = select_default_destination("pen", regions, preferred_container_ids=["bowl_0"])

    assert selected.region_type == "inside-container"
    assert selected.support_id == "bowl_0"


def test_free_space_region_avoids_masked_occupied_children():
    table = SceneNode(
        id="table_0",
        class_name="table",
        bbox=[0, 0, 100, 100],
        role_ids=["support_surface", "base_support_root"],
        mask_polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
    )
    pen = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[40, 40, 60, 60],
        support_parent="table_0",
        support_type="on-top-of",
        mask_polygon=[[40, 40], [60, 40], [60, 60], [40, 60]],
    )

    regions = compute_placement_regions([table, pen], image_size=(100, 100))
    free_region = next(region for region in regions if region.region_type == "free-space")

    assert free_region.support_id == "table_0"
    assert free_region.point_2d is not None
    point_x, point_y = free_region.point_2d
    assert not (40 <= point_x <= 60 and 40 <= point_y <= 60)


def test_placement_ignores_mask_polygon_when_mask_is_not_true():
    table = SceneNode(
        id="table_0",
        class_name="table",
        bbox=[0, 0, 100, 100],
        role_ids=["support_surface", "base_support_root"],
        mask_polygon=[[10, 10], [30, 10], [30, 30], [10, 30]],
        has_true_mask=False,
    )

    regions = compute_placement_regions([table], image_size=(100, 100))
    free_region = next(region for region in regions if region.region_type == "free-space")

    assert free_region.polygon_2d == [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
