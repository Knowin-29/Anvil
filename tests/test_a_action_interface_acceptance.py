from pathlib import Path

from apps.data_flywheel.gsa.a_action_interface_acceptance import (
    AActionInterfaceVLMJudge,
    build_verifier_eval_set,
    evaluate_verifier_predictions,
    parse_vlm_judge_response,
    prepare_acceptance_review_set,
    summarize_human_review,
)
from apps.data_flywheel.gsa.a_action_interface_review import (
    promote_action_interface_candidates,
    read_jsonl,
)


def _candidate(label_id, action_type="pick", source="vlm", object_name="cup"):
    row = {
        "label_id": label_id,
        "schema_version": "a_action_interface_candidate_v1",
        "sample_id": "sample_001",
        "image_path": "tests/fixtures/desk_sample.jpg",
        "image_width": 160,
        "image_height": 120,
        "object_id": f"{object_name}_0",
        "object_name": object_name,
        "bbox": [0.0, 0.0, 20.0, 20.0],
        "bucket": "A_action_interface",
        "action_type": action_type,
        "source_provenance": source,
        "verifier_status": "review_required" if source == "vlm" else "heuristic_only",
        "label_status": "teacher_generated" if source == "vlm" else "candidate",
        "stable_training_label": False,
        "risk_flags": [],
    }
    if action_type == "push":
        row.update(
            {
                "subtype": "push_contact",
                "label_type": "vector",
                "push_contact_point_2d": [10.0, 10.0],
                "push_direction_2d": [1.0, 0.0],
            }
        )
    else:
        row.update(
            {
                "subtype": "grasp_point",
                "label_type": "point",
                "grasp_point_2d": [10.0, 10.0],
                "jaw_axis_2d": [1.0, 0.0],
            }
        )
    return row


def test_human_review_summary_counts_uncertain_in_pass_rate_denominator():
    candidates = [
        _candidate("aif_pick_ok", "pick"),
        _candidate("aif_pick_bad", "pick"),
        _candidate("aif_push_uncertain", "push"),
        _candidate("aif_push_pending", "push"),
    ]
    decisions = {
        "aif_pick_ok": {"decision": "approve"},
        "aif_pick_bad": {"decision": "reject"},
        "aif_push_uncertain": {"decision": "uncertain"},
    }

    summary = summarize_human_review(candidates, decisions)

    overall = summary["overall"]
    assert overall["reviewed_count"] == 3
    assert overall["approve_count"] == 1
    assert overall["uncertain_count"] == 1
    assert overall["pending_count"] == 1
    assert overall["pass_rate"] == 1 / 3
    assert summary["by_action_type"]["pick"]["pass_rate"] == 0.5
    assert summary["by_action_type"]["push"]["pass_rate"] == 0.0


def test_verifier_eval_compares_vlm_judge_predictions_to_human_labels():
    eval_set = [
        {**_candidate("aif_ok"), "human_decision": "approve"},
        {**_candidate("aif_bad"), "human_decision": "reject"},
        {**_candidate("aif_unsure"), "human_decision": "uncertain"},
    ]
    predictions = [
        {"label_id": "aif_ok", "verifier_decision": "approve", "verifier_reason": "good"},
        {"label_id": "aif_bad", "verifier_decision": "approve", "verifier_reason": "mistake"},
        {"label_id": "aif_unsure", "verifier_decision": "uncertain", "verifier_reason": "unclear"},
    ]

    report = evaluate_verifier_predictions(eval_set, predictions)

    assert report["compared_count"] == 3
    assert report["correct_count"] == 2
    assert report["verifier_accuracy"] == 2 / 3
    assert report["confusion_matrix"]["reject"]["approve"] == 1
    assert len(report["false_positive"]) == 1


def test_build_verifier_eval_set_uses_only_reviewed_human_decisions():
    candidates = [_candidate("aif_ok"), _candidate("aif_pending")]
    decisions = {"aif_ok": {"decision": "approve"}, "aif_pending": {"decision": "pending"}}

    rows = build_verifier_eval_set(candidates, decisions)

    assert [row["label_id"] for row in rows] == ["aif_ok"]
    assert rows[0]["human_decision"] == "approve"


def test_vlm_judge_parses_json_and_normalizes_invalid_decisions_to_uncertain():
    assert parse_vlm_judge_response('{"decision":"approve","reason":"looks ok"}') == {
        "decision": "approve",
        "reason": "looks ok",
    }
    assert parse_vlm_judge_response('FINAL {"decision":"maybe","reason":"bad value"}') == {
        "decision": "uncertain",
        "reason": "bad value",
    }


class _FakeJudgeVLM:
    def chat(self, **kwargs):
        return '{"decision":"reject","reason":"point is on background"}'


def test_vlm_judge_returns_label_id_decision_and_reason():
    candidate = _candidate("aif_judge")

    result = AActionInterfaceVLMJudge(_FakeJudgeVLM()).judge(candidate)

    assert result == {
        "label_id": "aif_judge",
        "verifier_decision": "reject",
        "verifier_reason": "point is on background",
    }


def test_uncertain_review_decision_cannot_promote_to_stable_label():
    candidate = _candidate("aif_uncertain")

    result = promote_action_interface_candidates(
        [candidate],
        {"aif_uncertain": {"decision": "uncertain"}},
    )

    assert result.stable_labels == []
    assert "review_decision_uncertain" in result.candidates[0]["promotion_blockers"]


def test_prepare_acceptance_review_set_writes_100_sample_package(tmp_path):
    candidates_path = tmp_path / "review_priority_candidates.jsonl"
    with candidates_path.open("w", encoding="utf-8") as f:
        for idx in range(120):
            action_type = "push" if idx % 2 else "pick"
            f.write(
                __import__("json").dumps(
                    _candidate(f"aif_{idx:03d}", action_type=action_type, object_name="cup"),
                    ensure_ascii=False,
                )
                + "\n"
            )

    output_dir = tmp_path / "acceptance"
    summary = prepare_acceptance_review_set(
        candidates_path=candidates_path,
        output_dir=output_dir,
        sample_count=100,
        max_per_object_name=0,
        max_per_image=0,
    )

    assert summary["selected_count"] == 100
    assert len(read_jsonl(output_dir / "acceptance_review_candidates.jsonl")) == 100
    assert len(read_jsonl(output_dir / "review_decisions.template.jsonl")) == 100
    assert (output_dir / "acceptance_review.html").exists()
