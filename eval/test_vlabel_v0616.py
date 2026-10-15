#!/usr/bin/env python3
"""Lightweight regression tests for v0616 typed IR parsing."""

from vlabel_ir_v0616 import (
    build_typed_ir,
    classify_answer_type,
    greedy_match_f1,
    manifest_entry,
    parse_boxes_points,
)


def test_bbox_formats():
    text = '{"bbox_1000":[10,20,110,220],"bboxes":[[300,300,400,420]]}<box>[[1,2],[3,4]]</box>'
    boxes, points, frame = parse_boxes_points(text)
    assert len(boxes) >= 3
    assert not points
    assert frame == "norm_1000"


def test_finevision_cxcywh():
    text = "Normalized bounding box (center_x, center_y, width, height): [0.5, 0.5, 0.2, 0.4]"
    boxes, _, _ = parse_boxes_points(text)
    assert boxes[0] == [400.0, 300.0, 600.0, 700.0]


def test_gui_point():
    _, points, _ = parse_boxes_points("click(x=0.4174, y=0.4735)")
    assert points == [[417.4, 473.5]]


def test_routing_prediction_blind_manifest():
    m1 = manifest_entry(
        sample_id="a",
        source="unit",
        task_family_raw="bbox_grounding_single_or_few",
        prompt="<image>\nfind cup",
        gt_answer='{"bbox":[1,2,3,4]}',
    )
    m2 = manifest_entry(
        sample_id="a",
        source="unit",
        task_family_raw="bbox_grounding_single_or_few",
        prompt="<image>\nfind cup",
        gt_answer='{"bbox":[1,2,3,4]}',
    )
    assert m1["manifest_hash"] == m2["manifest_hash"]
    assert m1["answer_type"] == "bbox_region"
    assert m1["evaluable"] is True


def test_choice_count_relation():
    assert classify_answer_type("是否有杯子", '{"answer":"yes"}', "closed_choice_token")[0] == "choice"
    assert build_typed_ir("How many cups?", '{"count": 3}')["count"] == 3
    assert build_typed_ir("pose?", '{"pose_family":"standing"}')["relation"] == "standing"


def test_multi_box_f1():
    f1, tp, fp, fn, best = greedy_match_f1(
        [[0, 0, 100, 100], [300, 300, 400, 400]],
        [[0, 0, 100, 100], [305, 305, 405, 405]],
    )
    assert tp == 2
    assert fp == 0
    assert fn == 0
    assert f1 == 1.0
    assert best == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
