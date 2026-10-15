"""Tests for ShareRobot -> A_robot_state adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.data_flywheel.gsa.a_pipeline_schemas import validate_robot_state
from apps.data_flywheel.gsa.data_adapters.robot_state_ingest import (
    RobotStateIngestRegistry,
    build_robot_state_registry,
)
from apps.data_flywheel.gsa.data_adapters.robot_state_text_rules import infer_state_from_text
from apps.data_flywheel.gsa.data_adapters.sharerobot_adapter import ShareRobotRobotStateAdapter
from apps.data_flywheel.gsa.taxonomy import VerifierStatus


class TestInferStateFromText:
    def test_holding_grasped(self):
        text = "After <move the grasped banana towards the mug>, what's next?"
        result = infer_state_from_text(text)
        assert result == ("holding_object", "yes")

    def test_empty_reach(self):
        text = "reach for the carrot"
        result = infer_state_from_text(text)
        assert result == ("gripper_empty", "yes")

    def test_contact(self):
        text = "the gripper is in contact with the table"
        result = infer_state_from_text(text)
        assert result == ("contact_state", "yes")


@pytest.fixture
def mini_sharerobot_root(tmp_path: Path) -> Path:
    root = tmp_path / "sharerobot"
    train_dir = root / "planning" / "train"
    train_dir.mkdir(parents=True)
    images_dir = root / "images" / "rt_frames_success" / "ep1"
    images_dir.mkdir(parents=True)
    frame = images_dir / "frame_0.png"
    frame.write_bytes(b"fake")

    planning = [
        {
            "id": "ep1",
            "task": "Future_Prediction_Task",
            "selected_step": 0,
            "conversations": [
                {
                    "from": "human",
                    "value": "After <move the grasped mug>, what's next?",
                },
                {"from": "gpt", "value": "<place the mug on the table>"},
            ],
            "image": ["rt_frames_success/ep1/frame_0.png"],
            "meta_data": {"original_dataset": "bridge"},
        }
    ]
    (train_dir / "future_prediction_task.json").write_text(
        json.dumps(planning), encoding="utf-8"
    )

    trajectory = [
        {
            "id": 1,
            "image_path": "rt_frames_success/ep1/frame_0.png",
            "instruction": "reach for the cloth",
            "trajectory": [[1, 1], [2, 2], [3, 3]],
            "meta_data": {"original_dataset": "bridge"},
        }
    ]
    (root / "trajectory").mkdir(parents=True, exist_ok=True)
    (root / "trajectory" / "trajectory.json").write_text(
        json.dumps(trajectory), encoding="utf-8"
    )
    return root


class TestShareRobotAdapter:
    def test_discover_and_convert(self, mini_sharerobot_root: Path):
        adapter = ShareRobotRobotStateAdapter()
        config = {
            "root": str(mini_sharerobot_root),
            "planning_dirs": ["planning/train"],
            "planning_tasks_for_robot_state": ["future_prediction_task"],
            "trajectory_json": "trajectory/trajectory.json",
            "images_root": "images",
        }
        samples = adapter.discover(mini_sharerobot_root, max_samples=10, config=config)
        assert len(samples) >= 2

        labels = adapter.convert(samples, mini_sharerobot_root, config)
        assert len(labels) >= 2
        for label in labels:
            assert label.subtype == label.state_type
            assert label.source_dataset == "sharerobot"
            assert validate_robot_state(label).valid

        with_image = [l for l in labels if l.image_path and Path(l.image_path).exists()]
        assert with_image
        assert with_image[0].verifier_status == VerifierStatus.AUTO_VERIFIABLE.value

    def test_registry_ingest(self, mini_sharerobot_root: Path):
        registry = RobotStateIngestRegistry()
        registry.register(ShareRobotRobotStateAdapter())
        labels = registry.ingest_one(
            "sharerobot",
            {
                "root": str(mini_sharerobot_root),
                "planning_dirs": ["planning/train"],
                "planning_tasks_for_robot_state": ["future_prediction_task"],
                "trajectory_json": "trajectory/trajectory.json",
                "images_root": "images",
            },
            max_samples=10,
        )
        assert labels
        assert {l.state_type for l in labels} >= {"holding_object", "gripper_empty"}

    def test_build_registry_has_all_sources(self):
        registry = build_robot_state_registry()
        assert registry.get("sharerobot") is not None
        assert registry.get("embodied_reasoner") is not None
        assert registry.get("robo2vlm") is not None
        assert registry.get("bridge_data_v2") is not None
