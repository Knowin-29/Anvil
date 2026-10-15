"""Tests for BridgeDataV2 -> A_robot_state adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.data_flywheel.gsa.a_pipeline_schemas import validate_robot_state
from apps.data_flywheel.gsa.data_adapters.bridge_robot_state_adapter import BridgeRobotStateAdapter
from apps.data_flywheel.gsa.taxonomy import CameraView, VerifierStatus


@pytest.fixture
def mini_bridge_root(tmp_path: Path) -> Path:
    root = tmp_path / "bridge_data_v2"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "episode_000001.png").write_bytes(b"png")
    (images_dir / "episode_000002.png").write_bytes(b"png")
    return root


class TestBridgeRobotStateAdapter:
    def test_discover_and_convert(self, mini_bridge_root: Path):
        adapter = BridgeRobotStateAdapter()
        config = {"images_root": "images"}
        samples = adapter.discover(mini_bridge_root, max_samples=10, config=config)
        assert len(samples) == 2

        labels = adapter.convert(samples, mini_bridge_root, config)
        assert len(labels) == 2
        for label in labels:
            assert label.camera_view == CameraView.WRIST.value
            assert label.state_value == "unknown"
            assert label.verifier_status == VerifierStatus.REVIEW_REQUIRED.value
            assert label.source_dataset == "bridge_data_v2"
            assert validate_robot_state(label).valid
