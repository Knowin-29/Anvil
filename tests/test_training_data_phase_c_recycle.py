from __future__ import annotations

import json

from apps.data_flywheel.training_data.phase_c.base import RejectedSampleEntry, ValidationReport, ValidationResult
from apps.data_flywheel.training_data.phase_c.main import _build_rejected_entry, recycle_rejected_to_review
from apps.data_flywheel.types import Action, SceneObjectSample, ThinkingStep, ThinkingTrace, TrainingSample


def _sample(sample_id: str = "sample-1") -> TrainingSample:
    obj = SceneObjectSample(
        id="pen",
        class_name="pen",
        bbox_pixel=[0, 0, 10, 10],
        bbox_quantized=[0, 0, 10, 10],
    )
    return TrainingSample(
        sample_id=sample_id,
        image_id="image-1",
        image_path="/tmp/image-1.jpg",
        image_size=(100, 100),
        data_source="unit-test",
        instruction="pick the pen",
        instruction_template_id="tmpl",
        instruction_language="en",
        scene_objects=[obj],
        blocking_dag=[],
        thinking=ThinkingTrace(
            steps=[
                ThinkingStep(perception_type="ReferenceResolution", target="pen", content="resolved"),
                ThinkingStep(perception_type="Affordance", target="pen", content="operation=pick"),
                ThinkingStep(perception_type="PlanOrder", target="", content="pick sequence", output=["pick:pen"]),
            ]
        ),
        actions=[Action(type="pick", target="pen", grasp_point_pixel=[1, 1], grasp_point_quantized=[10, 10])],
        rex_omni_text="",
        metadata={},
    )


def _rejected_entry() -> RejectedSampleEntry:
    sample = _sample()
    report = ValidationReport(
        sample_id=sample.sample_id,
        overall_passed=False,
        results=[
            ValidationResult(validator_name="critical_fail", passed=False, severity="critical", reason="boom"),
            ValidationResult(validator_name="warning_fail", passed=False, severity="warning", reason="warn"),
        ],
    )
    return _build_rejected_entry(sample, report)


def test_recycle_writes_jsonl(tmp_path):
    output = tmp_path / "review" / "rejected_samples.jsonl"
    entry = _rejected_entry()

    count = recycle_rejected_to_review([entry], review_pool_path=str(output))

    assert count == 1
    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["sample_id"] == entry.sample_id
    assert row["rejected_at"].endswith("Z")
    assert "thinking_preview" in row
    assert "actions_summary" in row


def test_recycle_appends_to_existing_file(tmp_path):
    output = tmp_path / "rejected_samples.jsonl"
    output.write_text(json.dumps({"sample_id": "existing"}) + "\n", encoding="utf-8")

    count = recycle_rejected_to_review([_rejected_entry()], review_pool_path=str(output))

    assert count == 1
    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    assert json.loads(rows[0])["sample_id"] == "existing"


def test_recycle_creates_parent_dir(tmp_path):
    output = tmp_path / "nested" / "deeper" / "rejected_samples.jsonl"

    count = recycle_rejected_to_review([_rejected_entry()], review_pool_path=str(output))

    assert count == 1
    assert output.exists()
    assert output.parent.exists()
