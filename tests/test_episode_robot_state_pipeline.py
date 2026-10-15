"""Tests for episode-level A_robot_state multi-frame pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from apps.data_flywheel.gsa.episode_robot_state_pipeline import (
    ARobotStateEpisodePipeline,
    EpisodeBatch,
    EpisodeFrame,
    discover_episode_batches,
    filter_dynamic_capable_batches,
    group_manifest_rows,
    is_episode_capable_row,
)


def test_droid_rows_group_by_episode_and_sort_steps():
    rows = [
        {"episode_id": "ep2", "step_index": 4, "image_path": "/tmp/ep2_4.jpg"},
        {"episode_id": "ep1", "step_index": 8, "image_path": "/tmp/ep1_8.jpg"},
        {"episode_id": "ep1", "step_index": 2, "image_path": "/tmp/ep1_2.jpg"},
    ]

    batches = group_manifest_rows("droid_100", rows, frames_per_episode=6)

    assert [batch.episode_id for batch in batches] == ["ep1", "ep2"]
    assert [frame.step_index for frame in batches[0].frames] == [2, 8]


def test_fmb_rows_group_by_trajectory_and_sample_frames():
    rows = [
        {"trajectory_id": "traj_b", "step_index": i, "image_path": f"/tmp/b_{i}.jpg"}
        for i in range(10)
    ]

    batches = group_manifest_rows("fmb", rows, frames_per_episode=3)

    assert len(batches) == 1
    assert batches[0].episode_id == "traj_b"
    assert [frame.step_index for frame in batches[0].frames] == [0, 4, 9]


class FakeVLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(self.payload)


def _batch() -> EpisodeBatch:
    return EpisodeBatch(
        source_dataset="droid_100",
        episode_id="ep001",
        frames=[
            EpisodeFrame("droid_100", "ep001", 0, 0, "/tmp/0.jpg", {"language_instruction": "pick the block"}),
            EpisodeFrame("droid_100", "ep001", 1, 5, "/tmp/5.jpg", {}),
        ],
    )


def test_mock_vlm_output_generates_frame_labels_and_summary():
    payload = {
        "frames": [
            {
                "frame_index": 0,
                "step_index": 0,
                "states": {
                    "gripper_empty": {
                        "value": "yes",
                        "confidence": 0.91,
                        "target_object_class": None,
                        "visual_evidence": "frame 0 shows open empty gripper jaws",
                    },
                    "holding_object": {
                        "value": "no",
                        "confidence": 0.9,
                        "target_object_class": None,
                        "visual_evidence": "frame 0 has no object between jaws",
                    },
                    "object_between_fingers": {
                        "value": "no",
                        "confidence": 0.9,
                        "target_object_class": None,
                        "visual_evidence": "frame 0 fingers are separated and empty",
                    },
                    "contact_state": {
                        "value": "no",
                        "confidence": 0.85,
                        "target_object_class": None,
                        "visual_evidence": "frame 0 gripper is not touching a surface",
                    },
                    "tool_ready": {
                        "value": "yes",
                        "confidence": 0.84,
                        "target_object_class": None,
                        "visual_evidence": "frame 0 gripper is visible and unobstructed",
                    },
                },
            },
            {
                "frame_index": 1,
                "step_index": 5,
                "states": {
                    "holding_object": {
                        "value": "yes",
                        "confidence": 0.93,
                        "target_object_class": "block",
                        "visual_evidence": "frame 1 shows a block held between the gripper jaws",
                    }
                },
            },
        ],
        "episode_summary": {
            "episode_outcome": "success",
            "state_timeline": ["empty gripper", "block grasped"],
            "summary_evidence": "the gripper changes from empty to holding the block across frames",
            "confidence": 0.88,
        },
    }
    pipeline = ARobotStateEpisodePipeline(FakeVLM(payload), auto_pass_confidence=0.8)

    labels, summary = pipeline.run_episode(_batch())

    assert len(labels) == 10
    assert labels[0].state_type == "gripper_empty"
    assert labels[0].verifier_status == "auto_verifiable"
    holding_labels = [l for l in labels if l.state_type == "holding_object" and l.image_path == "/tmp/5.jpg"]
    assert holding_labels[0].state_value == "yes"
    assert holding_labels[0].target_object_class == "block"
    assert summary["episode_id"] == "ep001"
    assert summary["episode_outcome"] == "success"
    assert summary["sampled_steps"] == [0, 5]


def test_missing_frame_and_state_are_filled_unknown():
    payload = {
        "frames": [
            {
                "frame_index": 0,
                "step_index": 0,
                "states": {
                    "gripper_empty": {
                        "value": "yes",
                        "confidence": 0.95,
                        "visual_evidence": "frame 0 shows the gripper jaws are empty",
                    }
                },
            }
        ],
        "episode_summary": {
            "episode_outcome": "unknown",
            "state_timeline": [],
            "summary_evidence": "not enough visible evidence across frames",
            "confidence": 0.4,
        },
    }
    pipeline = ARobotStateEpisodePipeline(FakeVLM(payload), auto_pass_confidence=0.8)

    labels, summary = pipeline.run_episode(_batch())

    missing_frame = [l for l in labels if l.image_path == "/tmp/5.jpg"]
    assert all(l.state_value == "unknown" for l in missing_frame)
    assert all("missing_episode_frame" in l.quality_flags for l in missing_frame)
    missing_state = [l for l in labels if l.image_path == "/tmp/0.jpg" and l.state_type == "holding_object"][0]
    assert missing_state.state_value == "unknown"
    assert "missing_episode_state" in missing_state.quality_flags
    assert summary["quality_flags"] == ["low_confidence"]



def test_dynamic_capability_requires_episode_step_and_task_context():
    assert is_episode_capable_row({
        "episode_id": "ep1",
        "step_index": 0,
        "language_instruction": "pick the block",
    })
    assert not is_episode_capable_row({"episode_id": "ep1", "language_instruction": "pick"})
    assert not is_episode_capable_row({"episode_id": "ep1", "step_index": 0})


def test_filter_dynamic_capable_batches_skips_static_only_or_short_batches():
    static_batch = EpisodeBatch(
        source_dataset="bridge_data_v2",
        episode_id="single",
        frames=[EpisodeFrame("bridge_data_v2", "single", 0, 0, "/tmp/a.jpg", {})],
    )
    capable_batch = EpisodeBatch(
        source_dataset="droid_100",
        episode_id="ep_dynamic",
        frames=[
            EpisodeFrame("droid_100", "ep_dynamic", 0, 0, "/tmp/0.jpg", {"step_index": 0, "language_instruction": "pick"}),
            EpisodeFrame("droid_100", "ep_dynamic", 1, 5, "/tmp/5.jpg", {"step_index": 5, "language_instruction": "pick"}),
            EpisodeFrame("droid_100", "ep_dynamic", 2, 9, "/tmp/9.jpg", {"step_index": 9, "language_instruction": "pick"}),
        ],
    )

    capable, skipped = filter_dynamic_capable_batches([static_batch, capable_batch], min_frames=3)

    assert [batch.episode_id for batch in capable] == ["ep_dynamic"]
    assert skipped[0]["skip_reason"] == "too_few_frames"


def test_dynamic_annotation_generated_when_enabled():
    payload = {
        "frames": [
            {
                "frame_index": 0,
                "states": {
                    "holding_object": {"value": "no", "confidence": 0.9, "visual_evidence": "empty gripper visible before grasp"},
                    "contact_state": {"value": "no", "confidence": 0.9, "visual_evidence": "gripper is away from the block"},
                },
            },
            {
                "frame_index": 1,
                "states": {
                    "holding_object": {"value": "yes", "confidence": 0.92, "target_object_class": "block", "visual_evidence": "block is held between the gripper jaws"},
                    "contact_state": {"value": "yes", "confidence": 0.91, "target_object_class": "block", "visual_evidence": "gripper jaws contact the block"},
                },
            },
        ],
        "episode_summary": {
            "episode_outcome": "success",
            "state_timeline": ["empty", "holding"],
            "summary_evidence": "the block becomes held by the gripper across frames",
            "confidence": 0.9,
        },
        "dynamic_annotation": {
            "task_parse": {
                "instruction": "pick the block",
                "task_type": "pick",
                "target_object_class": "block",
                "expected_events": ["approach_target", "contact_target", "close_gripper"],
            },
            "sampled_frames": [
                {"frame_index": 0, "step_index": 0, "phase": "pre", "phase_confidence": 0.9},
                {"frame_index": 1, "step_index": 5, "phase": "interaction", "phase_confidence": 0.9},
            ],
            "phase_segments": [
                {"phase": "pre", "start_step": 0, "end_step": 2, "state_summary": "empty before grasp", "evidence": "frame 0 shows empty gripper", "confidence": 0.9},
                {"phase": "interaction", "start_step": 3, "end_step": 5, "state_summary": "grasp acquired", "evidence": "frame 1 shows block held between jaws", "confidence": 0.9},
            ],
            "state_transitions": [
                {"transition_type": "grasp_acquired", "from_step": 0, "to_step": 5, "before_state": {"holding_object": "no"}, "after_state": {"holding_object": "yes"}, "target_object_class": "block", "evidence": "block moves from outside to between the gripper jaws", "confidence": 0.9}
            ],
            "episode_outcome": {"value": "success", "evidence": "late frame shows the block held by the gripper", "confidence": 0.9},
        },
    }
    pipeline = ARobotStateEpisodePipeline(FakeVLM(payload), auto_pass_confidence=0.8)

    labels, summary, dynamic = pipeline.run_episode(_batch(), include_dynamic=True, dynamic_cfg={"dynamic_min_confidence": 0.8})

    assert len(labels) == 10
    assert summary["episode_outcome"] == "success"
    assert dynamic["bucket"] == "A_robot_state_dynamic"
    assert dynamic["task_parse"]["instruction_role"] == "prior_only"
    assert dynamic["quality"]["verifier_status"] == "auto_verifiable"


def test_dynamic_annotation_low_confidence_and_instruction_conflict_review_required():
    payload = {
        "frames": [{"frame_index": 0, "states": {"holding_object": {"value": "yes", "confidence": 0.9, "target_object_class": "block", "visual_evidence": "block is held between gripper jaws"}}}],
        "episode_summary": {"episode_outcome": "success", "state_timeline": [], "summary_evidence": "block is held", "confidence": 0.9},
        "dynamic_annotation": {
            "task_parse": {"instruction": "push the block", "task_type": "push", "target_object_class": "block"},
            "phase_segments": [{"phase": "interaction", "start_step": 0, "end_step": 5, "state_summary": "grasp", "evidence": "gripper holds the block", "confidence": 0.4}],
            "state_transitions": [{"transition_type": "grasp_acquired", "from_step": 0, "to_step": 5, "evidence": "gripper grasps the block", "confidence": 0.4}],
            "episode_outcome": {"value": "success", "evidence": "block is held", "confidence": 0.9},
        },
    }
    pipeline = ARobotStateEpisodePipeline(FakeVLM(payload), auto_pass_confidence=0.8)

    _, _, dynamic = pipeline.run_episode(_batch(), include_dynamic=True, dynamic_cfg={"dynamic_min_confidence": 0.8})

    flags = dynamic["quality"]["quality_flags"]
    assert dynamic["quality"]["verifier_status"] == "review_required"
    assert "low_transition_confidence" in flags
    assert "instruction_transition_conflict" in flags



def test_generic_episode_source_with_manifest_can_be_discovered(tmp_path: Path):
    root = tmp_path / "generic"
    root.mkdir()
    manifest = root / "manifest.jsonl"
    rows = [
        {"episode_id": "g1", "step_index": 0, "image_path": "/tmp/g0.jpg", "language_instruction": "move the cup"},
        {"episode_id": "g1", "step_index": 2, "image_path": "/tmp/g2.jpg", "language_instruction": "move the cup"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    batches = discover_episode_batches(
        ["generic_robot"],
        {"generic_robot": {"root": str(root), "manifest": "manifest.jsonl"}},
        frames_per_episode=6,
    )

    assert len(batches) == 1
    assert batches[0].source_dataset == "generic_robot"
    assert [frame.step_index for frame in batches[0].frames] == [0, 2]
