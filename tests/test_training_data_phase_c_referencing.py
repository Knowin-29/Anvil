from apps.data_flywheel.training_data.phase_c.validators.referencing_uniqueness import ReferencingUniquenessValidator
from apps.data_flywheel.types import Action, SceneObjectSample, ThinkingStep, ThinkingTrace, TrainingSample


def _sample() -> TrainingSample:
    obj_a = SceneObjectSample(
        id="dup",
        class_name="pen",
        bbox_pixel=[0, 0, 10, 10],
        bbox_quantized=[0, 0, 10, 10],
        hierarchical_name="the red pen",
    )
    obj_b = SceneObjectSample(
        id="dup",
        class_name="pen",
        bbox_pixel=[10, 0, 20, 10],
        bbox_quantized=[10, 0, 20, 10],
        hierarchical_name="the red pen",
    )
    return TrainingSample(
        sample_id="sample-ref",
        image_id="image-ref",
        image_path="/tmp/ref.jpg",
        image_size=(100, 100),
        data_source="unit-test",
        instruction="pick the red pen",
        instruction_template_id="tmpl",
        instruction_language="en",
        scene_objects=[obj_a, obj_b],
        blocking_dag=[],
        thinking=ThinkingTrace(steps=[ThinkingStep(perception_type="PlanOrder", target="", content="plan")]),
        actions=[Action(type="pick", target="dup", grasp_point_pixel=[1, 1], grasp_point_quantized=[10, 10])],
        rex_omni_text="",
        metadata={},
    )


def test_referencing_uniqueness_detects_duplicates():
    result = ReferencingUniquenessValidator().check(_sample(), {})

    assert not result.passed
    assert result.severity == "warning"
    assert "duplicate scene_object ids" in result.reason
    assert "duplicate hierarchical_name" in result.reason
