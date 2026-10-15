"""Tests for robot state ingest and data_prep conversation conversion."""

from __future__ import annotations

import json
from pathlib import Path

from apps.teacher_training.data_prep import gsa_label_to_conversation, prepare_training_data


def test_a_robot_state_to_conversation():
    label = {
        "bucket": "A_robot_state",
        "subtype": "holding_object",
        "state_type": "holding_object",
        "state_value": "yes",
        "image_path": "/tmp/frame.png",
        "source_dataset": "sharerobot",
        "target_object_class": "mug",
        "label_status": "human_verified",
        "stable_training_label": True,
        "quality_flags": [],
    }
    conv = gsa_label_to_conversation(label)
    assert conv["image"] == "/tmp/frame.png"
    assert conv["conversations"][0]["role"] == "user"
    assert "holding" in conv["conversations"][1]["content"].lower()
    assert conv["metadata"]["bucket"] == "A_robot_state"
    assert conv["metadata"]["stable_training_label"] is True


def test_prepare_training_data_exports_stable_robot_state(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    robot_dir = labels_dir / "A_robot_state"
    robot_dir.mkdir(parents=True)
    label = {
        "bucket": "A_robot_state",
        "subtype": "holding_object",
        "state_type": "holding_object",
        "state_value": "yes",
        "image_path": "/tmp/frame.png",
        "source_dataset": "bridge_data_v2",
        "target_object_class": "block",
        "verifier_status": "auto_verifiable",
        "label_status": "human_verified",
        "stable_training_label": True,
        "quality_flags": [],
    }
    (robot_dir / "A_robot_state.jsonl").write_text(
        json.dumps(label) + "\n", encoding="utf-8"
    )
    out = tmp_path / "out.jsonl"

    count = prepare_training_data(str(labels_dir), str(out))

    assert count == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["metadata"]["stable_training_label"] is True


def test_prepare_training_data_blocks_non_stable_robot_state(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    robot_dir = labels_dir / "A_robot_state"
    robot_dir.mkdir(parents=True)
    label = {
        "bucket": "A_robot_state",
        "subtype": "holding_object",
        "state_type": "holding_object",
        "state_value": "yes",
        "image_path": "/tmp/frame.png",
        "source_dataset": "vlm",
        "target_object_class": "block",
        "verifier_status": "auto_verifiable",
        "label_status": "teacher_generated",
        "stable_training_label": False,
        "quality_flags": [],
    }
    (robot_dir / "A_robot_state.jsonl").write_text(
        json.dumps(label) + "\n", encoding="utf-8"
    )
    out = tmp_path / "out.jsonl"

    count = prepare_training_data(str(labels_dir), str(out))

    assert count == 0
    assert out.read_text(encoding="utf-8") == ""
