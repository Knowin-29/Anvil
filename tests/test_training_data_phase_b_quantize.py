from apps.data_flywheel.training_data.phase_b.quantize import quantize_bbox, quantize_coord


def test_quantize_zero():
    assert quantize_coord(0, 1280) == 0


def test_quantize_max():
    assert quantize_coord(1280, 1280) == 999


def test_quantize_center_returns_500():
    assert quantize_coord(640, 1280) == 500


def test_quantize_bbox_preserves_order():
    quantized = quantize_bbox([100, 200, 300, 400], (1000, 1000))
    assert quantized[0] < quantized[2]
    assert quantized[1] < quantized[3]


def test_quantize_negative_clips_to_zero():
    assert quantize_coord(-10, 1280) == 0


def test_quantize_above_max_clips_to_999():
    assert quantize_coord(2000, 1280) == 999
