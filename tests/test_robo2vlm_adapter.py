"""Tests for Robo2VLM -> A_robot_state adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from apps.data_flywheel.gsa.a_pipeline_schemas import validate_robot_state
from apps.data_flywheel.gsa.data_adapters.robo2vlm_adapter import Robo2VLMAdapter
from apps.data_flywheel.gsa.taxonomy import CameraView


@pytest.fixture
def mini_robo2vlm_root(tmp_path: Path) -> Path:
    root = tmp_path / "robo2vlm"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)

    table = pa.table(
        {
            "question": [
                "The robot picked up the mug. Is the gripper empty?",
                "Has the robot successfully completed the task?",
            ],
            "choices": [["yes", "no"], ["yes", "no"]],
            "correct_answer": [1, 0],
            "category": ["gripper_state", "task_success_state"],
            "image": [
                {"bytes": b"fake-jpg-1", "path": None},
                {"bytes": b"fake-jpg-2", "path": None},
            ],
        }
    )
    pq.write_table(table, data_dir / "train-00000-of-00001.parquet")
    return root


class TestRobo2VLMAdapter:
    def test_discover_and_convert(self, mini_robo2vlm_root: Path):
        adapter = Robo2VLMAdapter()
        config = {"parquet_glob": "data/*.parquet", "extracted_images_dir": "extracted_images"}
        samples = adapter.discover(mini_robo2vlm_root, max_samples=10, config=config)
        assert len(samples) == 2

        labels = adapter.convert(samples, mini_robo2vlm_root, config)
        assert len(labels) >= 1
        label = labels[0]
        assert label.source_dataset == "robo2vlm"
        assert label.camera_view == CameraView.EXTERNAL.value
        assert label.image_path and Path(label.image_path).exists()
        assert validate_robot_state(label).valid
