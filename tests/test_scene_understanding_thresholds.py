from dataclasses import fields

from apps.data_flywheel.scene_understanding import SceneGraphThresholds


def test_default_values():
    thresholds = SceneGraphThresholds()

    assert thresholds.position_min_ratio == 0.3
    assert thresholds.position_min_img_ratio == 0.05
    assert thresholds.next_to_max_gap == 0.08
    assert thresholds.size_ratio == 2.0
    assert thresholds.overlap_iou == 0.1
    assert thresholds.near_ratio == 0.15
    assert thresholds.on_top_height_thresh == 0.02
    assert thresholds.on_top_h_dist_ratio == 1.5
    assert thresholds.under_height_thresh == 0.02
    assert thresholds.front_depth_thresh == 0.05
    assert thresholds.inside_margin == 0.5
    assert thresholds.occlusion_2d_overlap == 0.15
    assert thresholds.occlusion_depth_thresh == 0.03
    assert thresholds.stacking_h_overlap == 0.4
    assert thresholds.stacking_gap_3d == 0.05
    assert thresholds.fusion_strategy == "higher_confidence"
    assert thresholds.touching_edge_dist_px == 5
    assert thresholds.touching_depth_diff == 0.01
    assert thresholds.separated_min_gap_px == 15
    assert thresholds.between_perpendicular_dist_ratio == 0.15
    assert thresholds.aligned_angle_tolerance_deg == 10.0
    assert thresholds.planar_3d_stacking_disabled is True
    assert thresholds.planar_use_2d_fallback is True
    assert thresholds.hsv_cluster_k == 2
    assert thresholds.hsv_secondary_min_ratio == 0.15
    assert thresholds.size_small_threshold == 0.02
    assert thresholds.size_medium_threshold == 0.10


def test_fusion_strategy_enum_values():
    assert SceneGraphThresholds().fusion_strategy == "higher_confidence"
    assert SceneGraphThresholds(fusion_strategy="3d_wins").fusion_strategy == "3d_wins"
    assert SceneGraphThresholds(fusion_strategy="2d_wins").fusion_strategy == "2d_wins"


def test_all_numeric_fields_positive():
    thresholds = SceneGraphThresholds()

    for field_info in fields(SceneGraphThresholds):
        value = getattr(thresholds, field_info.name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            assert value > 0, f"{field_info.name} must be positive"
