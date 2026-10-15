from apps.data_flywheel.training_data.phase_c.validators.instruction_language import InstructionLanguageValidator
from apps.data_flywheel.types import Action, SceneObjectSample, ThinkingStep, ThinkingTrace, TrainingSample


def _sample(instruction: str, instruction_language: str) -> TrainingSample:
    obj = SceneObjectSample(
        id="pen",
        class_name="pen",
        bbox_pixel=[0, 0, 10, 10],
        bbox_quantized=[0, 0, 10, 10],
    )
    return TrainingSample(
        sample_id="sample-lang",
        image_id="image-lang",
        image_path="/tmp/lang.jpg",
        image_size=(100, 100),
        data_source="unit-test",
        instruction=instruction,
        instruction_template_id="tmpl",
        instruction_language=instruction_language,
        scene_objects=[obj],
        blocking_dag=[],
        thinking=ThinkingTrace(steps=[ThinkingStep(perception_type="PlanOrder", target="", content="plan")]),
        actions=[Action(type="pick", target="pen", grasp_point_pixel=[1, 1], grasp_point_quantized=[10, 10])],
        rex_omni_text="",
        metadata={},
    )


def test_instruction_language_validator_passes_for_matching_en():
    result = InstructionLanguageValidator().check(_sample("pick the pen", "en"), {})

    assert result.passed
    assert result.reason is None


def test_instruction_language_validator_allows_mixed_zh():
    result = InstructionLanguageValidator().check(_sample("拿 red 笔", "zh"), {})

    assert result.passed
    assert result.reason is None


def test_instruction_language_validator_flags_mismatch():
    result = InstructionLanguageValidator().check(_sample("pick the pen", "zh"), {})

    assert not result.passed
    assert "declared=zh, detected=en" in result.reason
