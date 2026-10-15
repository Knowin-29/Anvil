"""Tests for GSA metrics and eval harness."""

import pytest
from apps.data_flywheel.gsa.metrics import (
    bbox_iou, point_in_bbox, point_in_polygon, direction_cosine,
    exact_match, blocked_by_consistency, required_fields_score,
    verifiable_rate, point_distance,
)
from apps.data_flywheel.gsa.eval_harness import GSAEvalHarness, EvalResults


class TestMetrics:
    def test_bbox_iou_perfect(self):
        assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_bbox_iou_no_overlap(self):
        assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_bbox_iou_partial(self):
        iou = bbox_iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.1 < iou < 0.2  # 25/175 ≈ 0.143

    def test_point_in_bbox_inside(self):
        assert point_in_bbox([5, 5], [0, 0, 10, 10])

    def test_point_in_bbox_outside(self):
        assert not point_in_bbox([15, 5], [0, 0, 10, 10])

    def test_point_in_bbox_edge(self):
        assert point_in_bbox([0, 0], [0, 0, 10, 10])

    def test_point_in_polygon_inside(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert point_in_polygon([5, 5], polygon)

    def test_point_in_polygon_outside(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert not point_in_polygon([15, 5], polygon)

    def test_point_in_polygon_triangle(self):
        polygon = [[0, 0], [10, 0], [5, 10]]
        assert point_in_polygon([5, 3], polygon)
        assert not point_in_polygon([0, 10], polygon)

    def test_direction_cosine_same(self):
        assert direction_cosine([1, 0], [1, 0]) == pytest.approx(1.0)

    def test_direction_cosine_opposite(self):
        assert direction_cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_direction_cosine_perpendicular(self):
        assert direction_cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_exact_match_strings(self):
        assert exact_match("cup", "Cup")
        assert exact_match("  hello ", "hello")
        assert not exact_match("cup", "mug")

    def test_exact_match_values(self):
        assert exact_match(True, True)
        assert not exact_match(True, False)
        assert exact_match(5, 5)

    def test_blocked_by_consistency_no_blockers(self):
        score = blocked_by_consistency([], [], "cup_0")
        assert score == 1.0

    def test_blocked_by_consistency_confirmed(self):
        facts = [{"subject_id": "plate_0", "object_id": "cup_0", "predicate": "blocking"}]
        score = blocked_by_consistency(["plate_0"], facts, "cup_0")
        assert score == 1.0

    def test_blocked_by_consistency_contradicted(self):
        facts = [{"subject_id": "plate_0", "object_id": "cup_0", "predicate": "on_top_of"}]
        score = blocked_by_consistency([], facts, "cup_0")
        assert score == 0.0

    def test_required_fields_score_all_present(self):
        label = {"a": 1, "b": 2, "c": 3}
        assert required_fields_score(label, {"a", "b", "c"}) == 1.0

    def test_required_fields_score_partial(self):
        label = {"a": 1, "b": None, "c": 3}
        assert required_fields_score(label, {"a", "b", "c"}) == pytest.approx(2/3)

    def test_verifiable_rate(self):
        labels = [
            {"verifier_status": "auto_verifiable"},
            {"verifier_status": "auto_verifiable"},
            {"verifier_status": "review_required"},
        ]
        assert verifiable_rate(labels) == pytest.approx(2/3)

    def test_point_distance(self):
        assert point_distance([0, 0], [3, 4]) == pytest.approx(5.0)


class TestEvalHarness:
    def test_evaluate_empty(self):
        harness = GSAEvalHarness()
        results = harness.evaluate([])
        assert results.total_labels == 0
        assert results.overall_verifiable_rate == 0.0

    def test_evaluate_g_labels(self):
        labels = [
            {"bucket": "G", "subtype": "simple_grounding",
             "verifier_status": "auto_verifiable",
             "answer": {"bbox": [0, 0, 10, 10], "object_name": "cup"}},
            {"bucket": "G", "subtype": "counting",
             "verifier_status": "auto_verifiable",
             "answer": {"count": 3, "object_class": "cup"}},
        ]
        harness = GSAEvalHarness()
        results = harness.evaluate(labels)
        assert results.total_labels == 2
        assert results.overall_verifiable_rate == 1.0
        assert "G" in results.bucket_scores
        assert results.bucket_scores["G"].avg_score == 1.0

    def test_evaluate_a_action_labels(self):
        labels = [
            {"bucket": "A_action", "subtype": "grasp_point_2d",
             "verifier_status": "auto_verifiable",
             "action_type": "pick",
             "grasp_point_2d": [100, 200],
             "jaw_axis_2d": [1, 0],
             "can_pick_now": True,
             "blocked_by": [],
             "need_pre_push": False},
        ]
        harness = GSAEvalHarness()
        results = harness.evaluate(labels)
        assert results.total_labels == 1
        assert results.bucket_scores["A_action"].avg_score > 0.5

    def test_review_queue_populated(self):
        labels = [
            {"bucket": "G", "subtype": "simple_grounding",
             "verifier_status": "review_required",
             "sample_id": "s1", "object_id": "cup_0",
             "answer": {"bbox": [0, 0, 10, 10], "object_name": "cup"}},
        ]
        harness = GSAEvalHarness()
        results = harness.evaluate(labels)
        assert len(results.review_queue) == 1
        assert results.review_queue[0]["bucket"] == "G"
