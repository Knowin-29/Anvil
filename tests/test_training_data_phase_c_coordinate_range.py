from apps.data_flywheel.training_data.phase_c.validators.coordinate_range import CoordinateRangeValidator
from apps.data_flywheel.types import Action, SceneObjectSample, ThinkingTrace, TrainingSample


def _sample(*, bbox_quantized: list[int], grasp_point_quantized: list[int] | None = None, action: Action | None = None) -> TrainingSample:
    return TrainingSample(
        sample_id="sample-1",
        image_id="img-1",
        image_path="img-1.jpg",
        image_size=(100, 100),
        data_source="real",
        instruction="pick pen",
        instruction_template_id="tpl",
        instruction_language="en",
        scene_objects=[
            SceneObjectSample(
                id="pen_0",
                class_name="pen",
                bbox_pixel=[10.0, 10.0, 30.0, 30.0],
                bbox_quantized=bbox_quantized,
                grasp_point_quantized=grasp_point_quantized,
            )
        ],
        blocking_dag=[],
        thinking=ThinkingTrace(steps=[]),
        actions=[] if action is None else [action],
        rex_omni_text="",
    )


def test_all_coords_in_range():
    sample = _sample(
        bbox_quantized=[10, 20, 30, 40],
        grasp_point_quantized=[50, 60],
        action=Action(
            type="pick",
            target="pen_0",
            grasp_point_pixel=[10.0, 10.0],
            grasp_point_quantized=[70, 80],
        ),
    )

    result = CoordinateRangeValidator().check(sample, {})

    assert result.passed is True


def test_negative_coord_fails():
    sample = _sample(bbox_quantized=[-1, 20, 30, 40])

    result = CoordinateRangeValidator().check(sample, {})

    assert result.passed is False
    assert "out-of-range" in result.reason


def test_bbox_order_fails():
    sample = _sample(bbox_quantized=[40, 20, 30, 10])

    result = CoordinateRangeValidator().check(sample, {})

    assert result.passed is False
    assert "x1 >= x2" in result.reason
    assert "y1 >= y2" in result.reason
