"""Tests for Bridge manual labeling scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.apply_bridge_review import apply_decision, load_decisions


def test_apply_decision_approve():
    label = {
        "sample_id": "bridge_episode_000001",
        "source_dataset": "bridge_data_v2",
        "camera_view": "wrist",
        "end_effector": "parallel_jaw",
        "state_type": "gripper_empty",
        "state_value": "unknown",
        "verifier_status": "review_required",
        "quality_flags": ["low_confidence"],
        "stable_training_label": False,
    }
    decision = {
        "review_decision": "approve",
        "state_type": "holding_object",
        "state_value": "yes",
        "target_object_class": "block",
        "reviewer": "tester",
    }
    updated, changed = apply_decision(label, decision)
    assert changed
    assert updated["state_type"] == "holding_object"
    assert updated["state_value"] == "yes"
    assert updated["stable_training_label"] is True
    assert updated["verifier_status"] == "auto_verifiable"
    assert updated["label_status"] == "human_verified"
    assert updated["quality_flags"] == []


def test_apply_decision_skip(tmp_path: Path):
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps({"sample_id": "bridge_a", "review_decision": "skip"}) + "\n",
        encoding="utf-8",
    )
    decisions = load_decisions(path)
    assert decisions["bridge_a"]["review_decision"] == "skip"
    label = {"sample_id": "x", "stable_training_label": False}
    updated, changed = apply_decision(label, decisions["bridge_a"])
    assert not changed
    assert updated["stable_training_label"] is False


def test_apply_decision_reject_does_not_promote():
    label = {"sample_id": "bridge_b", "stable_training_label": False}
    decision = {"review_decision": "reject", "state_type": "holding_object", "state_value": "yes"}
    updated, changed = apply_decision(label, decision)
    assert not changed
    assert updated["stable_training_label"] is False
