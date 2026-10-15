from dataclasses import dataclass


@dataclass
class SceneGraphThresholds:
    # 2D relation thresholds
    position_min_ratio: float = 0.3
    position_min_img_ratio: float = 0.05
    next_to_max_gap: float = 0.08
    size_ratio: float = 2.0
    overlap_iou: float = 0.1
    near_ratio: float = 0.15
    far_ratio: float = 0.35

    # 3D relation thresholds
    on_top_height_thresh: float = 0.02
    on_top_h_dist_ratio: float = 1.5
    under_height_thresh: float = 0.02
    front_depth_thresh: float = 0.05
    inside_margin: float = 0.5
    occlusion_2d_overlap: float = 0.15
    occlusion_depth_thresh: float = 0.03
    stacking_h_overlap: float = 0.4
    stacking_gap_3d: float = 0.05

    # Round 2 thresholds
    fusion_strategy: str = "higher_confidence"
    touching_edge_dist_px: int = 5
    touching_depth_diff: float = 0.01
    separated_min_gap_px: int = 15
    between_perpendicular_dist_ratio: float = 0.15
    aligned_angle_tolerance_deg: float = 10.0
    planar_3d_stacking_disabled: bool = True
    planar_use_2d_fallback: bool = True

    # Phase B thresholds
    hsv_cluster_k: int = 2
    hsv_secondary_min_ratio: float = 0.15
    size_small_threshold: float = 0.02
    size_medium_threshold: float = 0.10
