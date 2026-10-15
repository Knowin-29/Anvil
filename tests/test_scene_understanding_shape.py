from apps.data_flywheel.scene_understanding.enrich.shape import (
    extract_shape_hint,
    extract_size,
)
from apps.data_flywheel.scene_understanding.thresholds import SceneGraphThresholds


def test_size_small_medium_large_thresholds():
    thresholds = SceneGraphThresholds()

    assert extract_size([0, 0, 1, 1], (100, 100), thresholds) == "small"
    assert extract_size([0, 0, 20, 20], (100, 100), thresholds) == "medium"
    assert extract_size([0, 0, 50, 50], (100, 100), thresholds) == "large"


def test_shape_square_aspect_ratio():
    assert extract_shape_hint([0, 0, 20, 20]) == "square"


def test_shape_tall_vs_wide():
    assert extract_shape_hint([0, 0, 10, 30]) == "tall"
    assert extract_shape_hint([0, 0, 30, 10]) == "wide"


def test_shape_hint_degenerate_bbox():
    assert extract_shape_hint([0, 0, 0, 10]) is None
