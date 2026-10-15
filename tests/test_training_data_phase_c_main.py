from __future__ import annotations

from apps.data_flywheel.training_data.phase_c.base import ValidationResult
from apps.data_flywheel.training_data.phase_c.main import default_validators, validate_samples
from apps.data_flywheel.types import Action, SceneObjectSample, ThinkingStep, ThinkingTrace, TrainingSample


def _sample(sample_id: str, *, instruction: str = "pick the pen", instruction_language: str = "en") -> TrainingSample:
    scene_object = SceneObjectSample(
        id=f"{sample_id}_pen",
        class_name="pen",
        bbox_pixel=[0, 0, 10, 10],
        bbox_quantized=[0, 0, 10, 10],
        grasp_point_pixel=[5, 5],
        grasp_point_quantized=[500, 500],
        mask_file=None,
        operation="pick",
        hierarchical_name="the pen",
        color_primary="red",
    )
    thinking = ThinkingTrace(
        steps=[
            ThinkingStep(perception_type="ReferenceResolution", target=scene_object.id, content="resolved target"),
            ThinkingStep(perception_type="Affordance", target=scene_object.id, content="operation=pick"),
            ThinkingStep(perception_type="PlanOrder", target="", content="pick sequence", output=["pick"]),
        ]
    )
    return TrainingSample(
        sample_id=sample_id,
        image_id=f"image-{sample_id}",
        image_path=f"/tmp/{sample_id}.jpg",
        image_size=(100, 100),
        data_source="unit-test",
        instruction=instruction,
        instruction_template_id="tmpl-1",
        instruction_language=instruction_language,
        scene_objects=[scene_object],
        blocking_dag=[],
        thinking=thinking,
        actions=[Action(type="pick", target=scene_object.id, grasp_point_pixel=[5, 5], grasp_point_quantized=[500, 500])],
        rex_omni_text="",
        metadata={},
    )


class _CountingValidator:
    def __init__(self, name: str, severity: str, outcomes):
        self.name = name
        self.severity = severity
        self._outcomes = dict(outcomes)
        self.calls = []

    def check(self, sample, context):
        self.calls.append(sample.sample_id)
        outcome = self._outcomes[sample.sample_id]
        if outcome == "crash":
            raise RuntimeError(f"{self.name} boom")
        return ValidationResult(
            validator_name=self.name,
            passed=bool(outcome),
            severity=self.severity,
            reason=None if outcome else f"{self.name} failed",
        )


def test_validate_samples_splits_passed_and_rejected():
    good = _sample("good")
    bad = _sample("bad")
    validators = [
        _CountingValidator("critical_gate", "critical", {"good": True, "bad": False}),
        _CountingValidator("warning_gate", "warning", {"good": False, "bad": False}),
    ]

    passed, rejected = validate_samples([good, bad], validators=validators)

    assert [sample.sample_id for sample in passed] == ["good"]
    assert [entry.sample_id for entry in rejected] == ["bad"]
    assert validators[0].calls == ["good", "bad"]
    assert validators[1].calls == ["good", "bad"]


def test_all_8_validators_registered():
    validators = default_validators()

    assert len(validators) == 8
    assert [validator.name for validator in validators] == [
        "point_in_mask",
        "coordinate_range",
        "thinking_format",
        "action_consistency",
        "ontology_consistency",
        "referencing_uniqueness",
        "image_exists",
        "instruction_language",
    ]
    assert [validator.severity for validator in validators] == [
        "critical",
        "critical",
        "critical",
        "critical",
        "critical",
        "warning",
        "warning",
        "warning",
    ]


def test_critical_fail_rejects_sample():
    sample = _sample("reject-me")
    validators = [
        _CountingValidator("critical_fail", "critical", {"reject-me": False}),
        _CountingValidator("warning_ok", "warning", {"reject-me": True}),
    ]

    passed, rejected = validate_samples([sample], validators=validators)

    assert passed == []
    assert len(rejected) == 1
    assert rejected[0].critical_fails == ["critical_fail"]
    assert rejected[0].warning_fails == []


def test_warning_fail_does_not_reject():
    sample = _sample("warn-only")
    validators = [
        _CountingValidator("critical_ok", "critical", {"warn-only": True}),
        _CountingValidator("warning_fail", "warning", {"warn-only": False}),
    ]

    passed, rejected = validate_samples([sample], validators=validators)

    assert [item.sample_id for item in passed] == ["warn-only"]
    assert rejected == []


def test_validator_crash_becomes_fail():
    sample = _sample("boom")
    validators = [
        _CountingValidator("critical_crash", "critical", {"boom": "crash"}),
    ]

    passed, rejected = validate_samples([sample], validators=validators)

    assert passed == []
    assert len(rejected) == 1
    assert rejected[0].reasons[0].startswith("validator crashed:")
    assert rejected[0].critical_fails == ["critical_crash"]
