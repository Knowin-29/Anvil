from apps.data_flywheel.training_data.phase_c.validators.image_exists import ImageExistsValidator
from apps.data_flywheel.types import Action, SceneObjectSample, ThinkingStep, ThinkingTrace, TrainingSample


def _sample(image_path: str) -> TrainingSample:
    obj = SceneObjectSample(
        id="pen",
        class_name="pen",
        bbox_pixel=[0, 0, 10, 10],
        bbox_quantized=[0, 0, 10, 10],
    )
    return TrainingSample(
        sample_id="sample-image",
        image_id="image-image",
        image_path=image_path,
        image_size=(100, 100),
        data_source="unit-test",
        instruction="pick the pen",
        instruction_template_id="tmpl",
        instruction_language="en",
        scene_objects=[obj],
        blocking_dag=[],
        thinking=ThinkingTrace(steps=[ThinkingStep(perception_type="PlanOrder", target="", content="plan")]),
        actions=[Action(type="pick", target="pen", grasp_point_pixel=[1, 1], grasp_point_quantized=[10, 10])],
        rex_omni_text="",
        metadata={},
    )


def test_image_exists_passes_for_existing_file(tmp_path):
    image_path = tmp_path / "scene.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    result = ImageExistsValidator().check(_sample(str(image_path)), {})

    assert result.passed
    assert result.reason is None


def test_image_exists_fails_for_missing_file(tmp_path):
    image_path = tmp_path / "missing.jpg"

    result = ImageExistsValidator().check(_sample(str(image_path)), {})

    assert not result.passed
    assert "does not exist" in result.reason
