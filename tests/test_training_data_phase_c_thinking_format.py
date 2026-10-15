from apps.data_flywheel.training_data.phase_c.validators.thinking_format import ThinkingFormatValidator
from apps.data_flywheel.types import ThinkingStep, ThinkingTrace, TrainingSample, SceneObjectSample


def _sample(steps: list[ThinkingStep]) -> TrainingSample:
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
                bbox_quantized=[10, 10, 30, 30],
            )
        ],
        blocking_dag=[],
        thinking=ThinkingTrace(steps=steps),
        actions=[],
        rex_omni_text="",
    )


def test_valid_trace():
    sample = _sample(
        [
            ThinkingStep(perception_type="ReferenceResolution", target="pen_0", content="resolved", output={"node_id": "pen_0"}),
            ThinkingStep(perception_type="Affordance", target="pen_0", content="operation=pick", output={"operation": "pick"}),
            ThinkingStep(perception_type="PlanOrder", target="", content="pick pen_0", output=["pick:pen_0"]),
        ]
    )

    result = ThinkingFormatValidator().check(sample, {})

    assert result.passed is True


def test_unknown_perception_type_fails():
    sample = _sample(
        [
            ThinkingStep(perception_type="ReferenceResolution", target="pen_0", content="resolved"),
            ThinkingStep(perception_type="Foobar", target="pen_0", content="oops"),
            ThinkingStep(perception_type="PlanOrder", target="", content="pick pen_0", output=["pick:pen_0"]),
        ]
    )

    result = ThinkingFormatValidator().check(sample, {})

    assert result.passed is False
    assert "unknown perception type" in result.reason


def test_missing_required_types_fails():
    sample = _sample(
        [
            ThinkingStep(perception_type="ReferenceResolution", target="pen_0", content="resolved"),
            ThinkingStep(perception_type="PlanOrder", target="", content="pick pen_0", output=["pick:pen_0"]),
        ]
    )

    result = ThinkingFormatValidator().check(sample, {})

    assert result.passed is False
    assert "missing required perception types" in result.reason


def test_empty_trace_fails():
    sample = _sample([])

    result = ThinkingFormatValidator().check(sample, {})

    assert result.passed is False
    assert "missing required perception types" in result.reason


def test_plan_order_with_empty_target_passes():
    sample = _sample(
        [
            ThinkingStep(perception_type="ReferenceResolution", target="pen_0", content="resolved"),
            ThinkingStep(perception_type="Affordance", target="pen_0", content="operation=pick"),
            ThinkingStep(perception_type="PlanOrder", target="", content="pick pen_0", output=["pick:pen_0"]),
        ]
    )

    result = ThinkingFormatValidator().check(sample, {})

    assert result.passed is True
