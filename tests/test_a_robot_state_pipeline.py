"""Tests for staged A_robot_state VLM pipeline helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from apps.data_flywheel.gsa.a_robot_state_pipeline import ARobotStatePipeline
from apps.data_flywheel.gsa.taxonomy import VerifierStatus
from apps.data_flywheel.gsa.visual_evidence import GsaEvidenceBundle


def _bundle(sample_id="test_1"):
    return GsaEvidenceBundle(
        sample_id=sample_id,
        image_path="/data/bridgedatav2/wrist/ep001.png",
        scene_family="tabletop_tidy",
        scenario_id="SC.2",
        objects=[],
        spatial_facts=[],
    )


def _visibility(clear=True):
    clear_text = "true" if clear else "false"
    return f'''{{
        "visibility": {{
            "gripper_visible": true,
            "gripper_clear": {clear_text},
            "tool_visible": true,
            "tool_clear": {clear_text},
            "suitable_for_robot_state": {clear_text},
            "visual_evidence": "parallel jaw gripper is visible near the center crop"
        }}
    }}'''


def _grasp_states(evidence=True):
    ev = '"open gripper fingers are visible with no object between the jaws"' if evidence else 'null'
    return f'''{{
        "states": {{
            "gripper_empty": {{"value": "yes", "confidence": 0.95, "visual_evidence": {ev}}},
            "holding_object": {{"value": "no", "confidence": 0.92, "target_object_class": null, "visual_evidence": "no object is enclosed by the visible gripper fingers"}},
            "object_between_fingers": {{"value": "no", "confidence": 0.91, "target_object_class": null, "visual_evidence": "the space between the gripper fingers appears empty"}}
        }}
    }}'''


def _contact_tool_states(tool_ready="yes", contact_evidence="the gripper jaws do not touch any visible object"):
    return f'''{{
        "states": {{
            "contact_state": {{"value": "no", "confidence": 0.9, "target_object_class": null, "visual_evidence": "{contact_evidence}"}},
            "tool_ready": {{"value": "{tool_ready}", "confidence": 0.93, "target_object_class": null, "visual_evidence": "the gripper tool is visible, unobstructed, and ready for use"}}
        }}
    }}'''


def _verifier(result="supported", state_type=None):
    states = [
        "gripper_empty",
        "holding_object",
        "object_between_fingers",
        "contact_state",
        "tool_ready",
    ]
    entries = []
    for state in states:
        value = result if state_type in {None, state} else "supported"
        entries.append(f'"{state}": {{"result": "{value}", "reason": "candidate matches visible gripper evidence"}}')
    return '{"verifications": {' + ",".join(entries) + '}}'


def test_staged_response_generates_five_robot_state_labels():
    vlm = MagicMock()
    vlm.query.side_effect = [
        _visibility(clear=True),
        _grasp_states(evidence=True),
        _contact_tool_states(),
        _verifier("supported"),
    ]
    pipe = ARobotStatePipeline(vlm, auto_pass_confidence=0.8)

    labels = pipe.run(_bundle())

    assert len(labels) == 5
    by_type = {label.state_type: label for label in labels}
    assert set(by_type) == {
        "gripper_empty",
        "holding_object",
        "object_between_fingers",
        "contact_state",
        "tool_ready",
    }
    assert by_type["gripper_empty"].subtype == "gripper_empty"
    assert by_type["gripper_empty"].visual_evidence
    assert by_type["gripper_empty"].verifier_result == "supported"
    assert by_type["gripper_empty"].verifier_status == VerifierStatus.AUTO_VERIFIABLE.value
    assert vlm.query.call_count == 4


def test_missing_visual_evidence_requires_review():
    vlm = MagicMock()
    vlm.query.side_effect = [
        _visibility(clear=True),
        _grasp_states(evidence=False),
        _contact_tool_states(),
        _verifier("supported"),
    ]
    pipe = ARobotStatePipeline(vlm, auto_pass_confidence=0.8)

    labels = pipe.run(_bundle())

    by_type = {label.state_type: label for label in labels}
    assert by_type["gripper_empty"].verifier_status == VerifierStatus.REVIEW_REQUIRED.value
    assert "missing_visual_evidence" in by_type["gripper_empty"].quality_flags


def test_contact_state_table_only_evidence_requires_review():
    vlm = MagicMock()
    vlm.query.side_effect = [
        _visibility(clear=True),
        _grasp_states(evidence=True),
        '''{"states": {
            "contact_state": {"value": "yes", "confidence": 0.95, "target_object_class": "table", "visual_evidence": "the object is resting on the table surface"},
            "tool_ready": {"value": "yes", "confidence": 0.93, "visual_evidence": "the gripper tool is visible, unobstructed, and ready for use"}
        }}''',
        _verifier("supported"),
    ]
    pipe = ARobotStatePipeline(vlm, auto_pass_confidence=0.8)

    labels = pipe.run(_bundle())

    by_type = {label.state_type: label for label in labels}
    assert by_type["contact_state"].verifier_status == VerifierStatus.REVIEW_REQUIRED.value
    assert "contact_evidence_not_end_effector" in by_type["contact_state"].quality_flags


def test_tool_ready_yes_becomes_unknown_when_tool_not_clear():
    vlm = MagicMock()
    vlm.query.side_effect = [
        _visibility(clear=False),
        _grasp_states(evidence=True),
        _contact_tool_states(tool_ready="yes"),
        _verifier("supported"),
    ]
    pipe = ARobotStatePipeline(vlm, auto_pass_confidence=0.8)

    labels = pipe.run(_bundle())

    by_type = {label.state_type: label for label in labels}
    assert by_type["tool_ready"].state_value == "unknown"
    assert by_type["tool_ready"].verifier_status == VerifierStatus.REVIEW_REQUIRED.value
    assert "tool_not_clearly_visible" in by_type["tool_ready"].quality_flags


def test_verifier_unsupported_requires_review():
    vlm = MagicMock()
    vlm.query.side_effect = [
        _visibility(clear=True),
        _grasp_states(evidence=True),
        _contact_tool_states(),
        _verifier("unsupported", state_type="tool_ready"),
    ]
    pipe = ARobotStatePipeline(vlm, auto_pass_confidence=0.8)

    labels = pipe.run(_bundle())

    by_type = {label.state_type: label for label in labels}
    assert by_type["tool_ready"].verifier_result == "unsupported"
    assert by_type["tool_ready"].verifier_status == VerifierStatus.REVIEW_REQUIRED.value
    assert "verifier_unsupported" in by_type["tool_ready"].quality_flags


def test_legacy_single_state_response_still_supported():
    vlm = MagicMock()
    vlm.query.return_value = (
        '{"state_type": "contact_state", "state_value": "yes", '
        '"target_object_class": null, "confidence": 0.95, '
        '"visual_evidence": "the gripper finger visibly touches the object edge"}'
    )
    pipe = ARobotStatePipeline(vlm, auto_pass_confidence=0.8)

    labels = pipe.run(_bundle())

    assert len(labels) == 1
    assert labels[0].state_type == "contact_state"
    assert labels[0].subtype == labels[0].state_type
    assert labels[0].label_status == "teacher_generated"
    assert labels[0].stable_training_label is False
    assert labels[0].visual_evidence


def test_non_robot_camera_skipped():
    vlm = MagicMock()
    pipe = ARobotStatePipeline(vlm)
    bundle = GsaEvidenceBundle(
        sample_id="test_3",
        image_path="/data/indoor/kitchen_scene.jpg",
        scene_family="tabletop_tidy",
        scenario_id="SC.2",
        objects=[],
        spatial_facts=[],
    )
    assert pipe.run(bundle) == []
    vlm.query.assert_not_called()


def test_bridge_image_gets_wrist_camera_and_bridge_source_dataset():
    vlm = MagicMock()
    vlm.query.side_effect = [
        _visibility(clear=True),
        _grasp_states(evidence=True),
        _contact_tool_states(),
        _verifier("supported"),
    ]
    pipe = ARobotStatePipeline(vlm, auto_pass_confidence=0.8)
    bundle = GsaEvidenceBundle(
        sample_id="bridge_1",
        image_path="/path/to/datasets/downloads/bridge_data_v2/images/episode_001.png",
        scene_family="tabletop_tidy",
        scenario_id="SC.2",
        objects=[],
        spatial_facts=[],
    )

    labels = pipe.run(bundle)

    assert len(labels) == 5
    assert labels[0].camera_view == "wrist"
    assert labels[0].source_dataset == "bridge_data_v2"
    assert labels[0].source_provenance == "vlm"


def test_multisource_vlm_images_are_accepted_and_source_dataset_inferred():
    cases = [
        (
            "/path/to/datasets/downloads/embodied_reasoner/.image_cache/task/0_init_observe.png",
            "embodied_reasoner",
            "head",
        ),
        (
            "/path/to/datasets/downloads/robo2vlm/extracted_images/10.jpg",
            "robo2vlm",
            "external",
        ),
        (
            "/path/to/datasets/downloads/sharerobot/images/demo.png",
            "sharerobot",
            "external",
        ),
    ]

    for path, source_dataset, camera_view in cases:
        vlm = MagicMock()
        vlm.query.side_effect = [
            _visibility(clear=True),
            _grasp_states(evidence=True),
            _contact_tool_states(),
            _verifier("supported"),
        ]
        pipe = ARobotStatePipeline(vlm, auto_pass_confidence=0.8)
        bundle = GsaEvidenceBundle(
            sample_id=f"sample_{source_dataset}",
            image_path=path,
            scene_family="tabletop_tidy",
            scenario_id="SC.2",
            objects=[],
            spatial_facts=[],
        )

        labels = pipe.run(bundle)

        assert len(labels) == 5
        assert labels[0].source_dataset == source_dataset
        assert labels[0].camera_view == camera_view
        assert labels[0].label_status == "teacher_generated"
        assert labels[0].stable_training_label is False


def test_robot_state_vlm_roots_allow_non_keyword_images():
    vlm = MagicMock()
    vlm.query.side_effect = [
        _visibility(clear=True),
        _grasp_states(evidence=True),
        _contact_tool_states(),
        _verifier("supported"),
    ]
    pipe = ARobotStatePipeline(
        vlm,
        auto_pass_confidence=0.8,
        robot_state_vlm_roots=["/tmp/custom_robot_state_images"],
    )
    bundle = GsaEvidenceBundle(
        sample_id="custom_1",
        image_path="/tmp/custom_robot_state_images/frame_001.png",
        scene_family="tabletop_tidy",
        scenario_id="SC.2",
        objects=[],
        spatial_facts=[],
    )

    labels = pipe.run(bundle)

    assert len(labels) == 5
    assert labels[0].source_dataset == "vlm"
