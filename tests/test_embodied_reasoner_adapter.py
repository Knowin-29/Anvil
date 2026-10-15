"""Tests for Embodied-Reasoner -> A_robot_state adapter."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from apps.data_flywheel.gsa.a_pipeline_schemas import validate_robot_state
from apps.data_flywheel.gsa.data_adapters.embodied_reasoner_adapter import (
    EmbodiedReasonerRobotStateAdapter,
    discover_er_step_samples,
)
from apps.data_flywheel.gsa.data_adapters.robot_state_text_rules import infer_state_from_er_action
from apps.data_flywheel.gsa.taxonomy import CameraView, VerifierStatus


class TestInferStateFromErAction:
    def test_pickup(self):
        assert infer_state_from_er_action("pickup") == ("holding_object", "yes")

    def test_put_in(self):
        assert infer_state_from_er_action("put in") == ("gripper_empty", "yes")

    def test_open(self):
        assert infer_state_from_er_action("open") == ("contact_state", "yes")

    def test_navigate(self):
        assert infer_state_from_er_action("navigate to") == ("gripper_empty", "yes")

    def test_end_skipped(self):
        assert infer_state_from_er_action("end") is None


@pytest.fixture
def mini_er_root(tmp_path: Path) -> Path:
    root = tmp_path / "embodied_reasoner"
    images_dir = root / "data" / "images" / "bundle_a"
    images_dir.mkdir(parents=True)
    image_file = images_dir / "step.png"
    image_file.write_bytes(b"png")

    trajectories = [
        {
            "id": 7,
            "scene": "FloorPlan1",
            "key_actions": [
                {"action": "pickup", "objectType": "Mug"},
                {"action": "end", "objectType": ""},
            ],
            "images": [
                "./data/images/bundle_a/step.png",
                "./data/images/bundle_a/step.png",
            ],
        }
    ]
    (root / "data_engine_origin_trajectory_9390.json").write_text(
        json.dumps(trajectories), encoding="utf-8"
    )
    return root


class TestEmbodiedReasonerAdapter:
    def test_discover_and_convert(self, mini_er_root: Path):
        config = {
            "trajectory_json": "data_engine_origin_trajectory_9390.json",
        }
        samples = discover_er_step_samples(mini_er_root, max_samples=10, config=config)
        assert len(samples) == 2

        adapter = EmbodiedReasonerRobotStateAdapter()
        labels = adapter.convert(samples, mini_er_root, config)
        assert len(labels) == 1
        label = labels[0]
        assert label.state_type == "holding_object"
        assert label.camera_view == CameraView.EXTERNAL.value
        assert label.target_object_class == "Mug"
        assert label.sample_id.startswith("er_7_0_")
        assert label.image_path and Path(label.image_path).exists()
        assert label.verifier_status == VerifierStatus.AUTO_VERIFIABLE.value
        assert validate_robot_state(label).valid

    def test_zip_fallback(self, tmp_path: Path):
        root = tmp_path / "embodied_reasoner"
        bundle = "bundle_b"
        inner = f"{bundle}/step.png"
        zip_path = root / "data" / "images" / f"{bundle}.zip"
        zip_path.parent.mkdir(parents=True)
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(inner, b"zip-png")

        trajectories = [
            {
                "id": 9,
                "key_actions": [{"action": "open", "objectType": "Drawer"}],
                "images": [f"./data/images/{bundle}/step.png"],
            }
        ]
        (root / "data_engine_origin_trajectory_9390.json").write_text(
            json.dumps(trajectories), encoding="utf-8"
        )

        adapter = EmbodiedReasonerRobotStateAdapter()
        samples = adapter.discover(root, max_samples=5, config={})
        labels = adapter.convert(samples, root, config={})
        assert len(labels) == 1
        assert labels[0].state_type == "contact_state"
        assert labels[0].image_path and Path(labels[0].image_path).exists()
