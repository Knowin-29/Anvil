"""Tests for GSA runner stage gating."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from apps.data_flywheel.gsa.a_pipeline_schemas import ARobotStateLabel
from apps.data_flywheel.gsa.runner import GSAv0Runner


def _write_config(tmp_path: Path) -> Path:
    cfg = {
        "vlm": {
            "base_url": "http://localhost:10301/v1",
            "model": "dummy",
            "api_key": "",
            "max_image_size": 1280,
            "timeout": 1,
            "max_retries": 0,
        },
        "input": {"image_dirs": [], "max_images": 1, "discover_external_sources": False},
        "pipeline": {"enabled_stages": ["G", "S", "A_robot_state"], "concurrency": 1},
        "output": {"root": str(tmp_path / "out")},
        "scene_families": {"tabletop_tidy": {"keywords": ["bridge"], "scenario": "SC.2"}},
        "quotas": {"G": 100, "S": 100, "A_robot_state": 500},
        "robot_state_datasets": {"enabled": [], "ingest_only": False, "max_samples_per_source": 0},
        "robot_state_episode": {"enabled": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_runner_stage_gating_keeps_g_s_robot_state_only(tmp_path: Path):
    runner = GSAv0Runner(str(_write_config(tmp_path)))
    runner.g_worker.generate = MagicMock(return_value=[{
        "bucket": "G",
        "subtype": "simple_grounding",
        "answer": {"object_name": "cube", "bbox": [1, 2, 3, 4]},
    }])
    runner.s_worker.generate = MagicMock(return_value=[{"bucket": "S", "subtype": "left_of"}])
    runner.s_worker.extract_spatial_facts = MagicMock(return_value=[{"relation": "left_of"}])
    runner.a_robot_state.run = MagicMock(return_value=[ARobotStateLabel(
        sample_id="sample_1",
        camera_view="wrist",
        state_type="gripper_empty",
        subtype="gripper_empty",
        state_value="yes",
        image_path="/data/bridge_data_v2/images/episode_001.png",
        source_dataset="bridge_data_v2",
        source_provenance="vlm",
        verifier_status="auto_verifiable",
    )])

    for pipe in (
        runner.a_state_pipeline,
        runner.a_identity_attr,
        runner.a_obj_state,
        runner.a_action_precondition,
        runner.a_action_interface,
    ):
        pipe.run = MagicMock(return_value=[])
    runner.a_action_pipeline.run = MagicMock(return_value=([], []))

    labels, reviews = runner._process_image(
        "/data/bridge_data_v2/images/episode_001.png",
        "sample_1",
        "tabletop_tidy",
        "SC.2",
    )

    assert [label["bucket"] for label in labels] == ["G", "S", "A_robot_state"]
    assert reviews == []
    runner.g_worker.generate.assert_called_once()
    runner.s_worker.generate.assert_called_once()
    runner.a_robot_state.run.assert_called_once()
    runner.a_state_pipeline.run.assert_not_called()
    runner.a_action_pipeline.run.assert_not_called()
    runner.a_identity_attr.run.assert_not_called()
    runner.a_obj_state.run.assert_not_called()
    runner.a_action_precondition.run.assert_not_called()
    runner.a_action_interface.run.assert_not_called()


def test_runner_discovers_recursive_image_globs_and_balances_sources(tmp_path: Path):
    bridge = tmp_path / "bridge_data_v2" / "images"
    er = tmp_path / "embodied_reasoner" / ".image_cache" / "task"
    robo = tmp_path / "robo2vlm" / "extracted_images"
    share = tmp_path / "sharerobot" / "images"
    for directory in (bridge, er, robo, share):
        directory.mkdir(parents=True)
    for i in range(3):
        (bridge / f"episode_{i:06d}.png").write_bytes(b"")
        (er / f"{i}_observe.png").write_bytes(b"")
        (robo / f"{i}.jpg").write_bytes(b"")
        (share / f"{i}.png").write_bytes(b"")

    cfg = {
        "vlm": {
            "base_url": "http://localhost:10301/v1",
            "model": "dummy",
            "api_key": "",
            "max_image_size": 1280,
            "timeout": 1,
            "max_retries": 0,
        },
        "input": {
            "image_dirs": [str(bridge), str(robo), str(share)],
            "image_globs": [str(er / "**" / "*.png")],
            "image_source_priority": ["bridge_data_v2", "embodied_reasoner", "robo2vlm", "sharerobot"],
            "balance_image_sources": True,
            "max_images": 4,
            "discover_external_sources": False,
        },
        "pipeline": {"enabled_stages": ["G", "S", "A_robot_state"], "concurrency": 1},
        "output": {"root": str(tmp_path / "out")},
        "scene_families": {"tabletop_tidy": {"keywords": ["bridge"], "scenario": "SC.2"}},
        "quotas": {"G": 100, "S": 100, "A_robot_state": 500},
        "robot_state_datasets": {"enabled": [], "ingest_only": False, "max_samples_per_source": 0},
        "robot_state_episode": {"enabled": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    runner = GSAv0Runner(str(path))

    images = runner._discover_images()

    assert len(images) == 4
    assert any("bridge_data_v2" in image for image in images)
    assert any("embodied_reasoner" in image for image in images)
    assert any("robo2vlm" in image for image in images)
    assert any("sharerobot" in image for image in images)


def test_runner_allows_non_wrist_robot_state_vlm_sources(tmp_path: Path):
    runner = GSAv0Runner(str(_write_config(tmp_path)))
    runner.g_worker.generate = MagicMock(return_value=[])
    runner.s_worker.generate = MagicMock(return_value=[])
    runner.s_worker.extract_spatial_facts = MagicMock(return_value=[])
    runner.a_robot_state.run = MagicMock(return_value=[ARobotStateLabel(
        sample_id="sample_1",
        camera_view="external",
        state_type="gripper_empty",
        subtype="gripper_empty",
        state_value="unknown",
        image_path="/data/robo2vlm/extracted_images/10.jpg",
        source_dataset="robo2vlm",
        source_provenance="vlm",
        verifier_status="review_required",
    )])

    labels, reviews = runner._process_image(
        "/data/robo2vlm/extracted_images/10.jpg",
        "sample_1",
        "tabletop_tidy",
        "SC.2",
    )

    assert [label["bucket"] for label in labels] == ["A_robot_state"]
    assert reviews[0]["source_dataset"] == "robo2vlm"
    runner.a_robot_state.run.assert_called_once()


def test_multi_vllm_config_loads_with_expected_sources():
    runner = GSAv0Runner("scripts/configs/gsa_robot_state_multi_vllm_local.yaml")

    assert runner._enabled_stages() == {"G", "S", "A_robot_state"}
    assert runner.config["input"]["balance_image_sources"] is True
    assert runner.config["input"]["force_robot_state_vlm"] is True
    assert runner.config["robot_state_datasets"]["enabled"] == [
        "sharerobot",
        "embodied_reasoner",
        "robo2vlm",
        "fmb",
        "droid_100",
    ]
    assert runner.config["robot_state_episode"] == {
        "enabled": False,
        "sources": ["droid_100", "fmb"],
        "frames_per_episode": 6,
        "max_episodes_per_source": 5,
        "output_episode_summary": True,
        "output_dynamic_labels": False,
        "dynamic_label_version": "v1",
        "dynamic_min_confidence": 0.8,
        "min_frames_per_dynamic_episode": 3,
        "dynamic_sources": [],
    }


def test_runner_saves_episode_robot_state_outputs(tmp_path: Path):
    cfg_path = _write_config(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["input"]["max_images"] = 0
    cfg["robot_state_episode"] = {
        "enabled": True,
        "sources": ["droid_100", "fmb"],
        "frames_per_episode": 6,
        "max_episodes_per_source": 1,
        "output_episode_summary": True,
    }
    cfg_path.write_text(yaml.safe_dump(cfg))

    runner = GSAv0Runner(str(cfg_path))
    runner.a_robot_state_episode.run_from_config = MagicMock(return_value=([
        ARobotStateLabel(
            sample_id="droid_100_episode_ep001_step00000_gripper_empty",
            camera_view="wrist",
            state_type="gripper_empty",
            subtype="gripper_empty",
            state_value="unknown",
            image_path="/tmp/ep001_0.jpg",
            source_dataset="droid_100",
            source_provenance="vlm",
            verifier_status="review_required",
            quality_flags=["unknown_state_value"],
        )
    ], [
        {
            "bucket": "A_robot_state_episode",
            "episode_id": "ep001",
            "source_dataset": "droid_100",
            "frame_count": 1,
            "sampled_steps": [0],
            "task_instruction": "pick",
            "state_timeline": [],
            "episode_outcome": "unknown",
            "summary_evidence": None,
            "confidence": None,
            "quality_flags": ["missing_confidence"],
        }
    ]))

    summary = runner.run(image_paths=[])

    assert summary["episode_summaries"] == 1
    labels_path = runner.output_root / "labels" / "A_robot_state" / "A_robot_state.jsonl"
    episode_path = runner.output_root / "labels" / "A_robot_state_episode" / "episode_summaries.jsonl"
    review_path = runner.output_root / "review_queue" / "A_robot_state.review.jsonl"
    assert labels_path.exists()
    assert episode_path.exists()
    assert review_path.exists()
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    episodes = [json.loads(line) for line in episode_path.read_text().splitlines()]
    assert labels[0]["bucket"] == "A_robot_state"
    assert episodes[0]["episode_id"] == "ep001"



def test_runner_saves_optional_dynamic_episode_labels(tmp_path: Path):
    cfg_path = _write_config(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["input"]["max_images"] = 0
    cfg["robot_state_episode"] = {
        "enabled": True,
        "sources": ["droid_100"],
        "frames_per_episode": 6,
        "max_episodes_per_source": 1,
        "output_episode_summary": True,
        "output_dynamic_labels": True,
        "dynamic_label_version": "v1",
        "dynamic_min_confidence": 0.8,
        "min_frames_per_dynamic_episode": 3,
        "dynamic_sources": [],
    }
    cfg_path.write_text(yaml.safe_dump(cfg))

    runner = GSAv0Runner(str(cfg_path))
    dynamic_label = {
        "bucket": "A_robot_state_dynamic",
        "label_version": "v1",
        "source_dataset": "droid_100",
        "episode_id": "ep001",
        "task_parse": {"instruction": "pick", "task_type": "pick", "instruction_role": "prior_only"},
        "sampled_frames": [],
        "phase_segments": [],
        "state_transitions": [],
        "episode_outcome": {"value": "unknown", "confidence": None, "evidence": None},
        "quality": {"verifier_status": "review_required", "quality_flags": ["missing_phase_segments"], "skip_reason": None},
    }
    runner.a_robot_state_episode.run_from_config = MagicMock(return_value=([], [], [dynamic_label]))

    summary = runner.run(image_paths=[])

    dynamic_path = runner.output_root / "labels" / "A_robot_state_dynamic" / "A_robot_state_dynamic.jsonl"
    assert summary["episode_dynamic_labels"] == 1
    assert dynamic_path.exists()
    saved = json.loads(dynamic_path.read_text().strip())
    assert saved["bucket"] == "A_robot_state_dynamic"
    assert not (runner.output_root / "labels" / "A_robot_state" / "A_robot_state.jsonl").exists()
    runner.a_robot_state_episode.run_from_config.assert_called_once()
