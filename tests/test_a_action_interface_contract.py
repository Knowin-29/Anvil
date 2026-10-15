from apps.data_flywheel.gsa.a_action_interface_contract import (
    ACTION_INTERFACE_SCHEMA_VERSION,
    LABEL_STATUS_CANDIDATE,
    LABEL_STATUS_HUMAN_VERIFIED,
    LABEL_STATUS_STABLE_TRAINING_LABEL,
    action_interface_schema,
    make_action_interface_label_id,
    verify_action_interface_candidate,
)


def _base_record():
    return {
        "schema_version": ACTION_INTERFACE_SCHEMA_VERSION,
        "source_dataset": "unit",
        "sample_id": "unit_001",
        "candidate_kind": "unit",
        "action_type": "pick",
        "label_type": "point",
        "source_provenance": "human_anchor",
        "verifier_status": "auto_verifiable",
        "label_status": LABEL_STATUS_HUMAN_VERIFIED,
        "stable_training_label": True,
        "grasp_point_2d": [5.0, 5.0],
        "jaw_axis_2d": [1.0, 0.0],
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "risk_flags": [],
    }


def test_action_interface_schema_lists_required_pick_fields():
    schema = action_interface_schema()

    assert schema["schema_version"] == ACTION_INTERFACE_SCHEMA_VERSION
    assert schema["required_by_action_type"]["pick"] == ["grasp_point_2d", "jaw_axis_2d"]
    assert LABEL_STATUS_STABLE_TRAINING_LABEL in schema["label_status_values"]


def test_verified_record_with_required_geometry_is_promotion_eligible():
    result = verify_action_interface_candidate(_base_record())

    assert result.errors == []
    assert result.promotion_blockers == []
    assert result.promotion_eligible


def test_label_id_generation_is_stable():
    record = _base_record()

    assert make_action_interface_label_id(record) == make_action_interface_label_id(dict(record))


def test_candidate_record_cannot_be_stable_training_label():
    record = _base_record()
    record["label_status"] = LABEL_STATUS_CANDIDATE

    result = verify_action_interface_candidate(record)

    assert not result.promotion_eligible
    assert "candidate/teacher_generated records cannot be stable_training_label" in result.errors


def test_missing_required_geometry_blocks_promotion_without_schema_error():
    record = _base_record()
    record.pop("jaw_axis_2d")

    result = verify_action_interface_candidate(record)

    assert result.errors == []
    assert "missing_jaw_axis_2d" in result.promotion_blockers
    assert not result.promotion_eligible


def test_grasp_point_outside_bbox_is_geometry_error():
    record = _base_record()
    record["grasp_point_2d"] = [50.0, 50.0]

    result = verify_action_interface_candidate(record)

    assert "grasp_point_2d outside bbox" in result.errors
    assert not result.geometry_valid
    assert not result.promotion_eligible


def test_non_unit_vector_blocks_stable_promotion():
    record = _base_record()
    record["jaw_axis_2d"] = [2.0, 0.0]

    result = verify_action_interface_candidate(record)

    assert "invalid vector jaw_axis_2d: not unit length" in result.errors
    assert not result.geometry_valid
    assert not result.promotion_eligible
