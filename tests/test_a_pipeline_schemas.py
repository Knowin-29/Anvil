"""Tests for A pipeline schemas and routing logic."""

import pytest
import sys
sys.path.insert(0, ".")

from apps.data_flywheel.gsa.taxonomy import (
    Bucket,
    route_to_pipeline,
    needs_g_context,
    needs_s_context,
    AIdentityAttrSubtype,
    AObjStateSubtype,
    ARobotStateSubtype,
    AActionPreconditionSubtype,
    AActionInterfaceSubtype,
    BUCKET_SUBTYPES,
    validate_bucket_subtype,
)
from apps.data_flywheel.gsa.a_pipeline_schemas import (
    AIdentityAttrLabel,
    AObjStateLabel,
    ARobotStateLabel,
    AActionPreconditionLabel,
    AActionInterfaceLabel,
    validate_identity_attr,
    validate_obj_state,
    validate_robot_state,
    validate_action_precondition,
    validate_action_interface,
)


class TestTaxonomyRouting:
    """Test pipeline routing logic."""

    def test_route_gripper_view(self):
        result = route_to_pipeline(has_gripper_view=True)
        assert result == Bucket.A_ROBOT_STATE

    def test_route_a_action_bucket(self):
        result = route_to_pipeline(gsa_bucket="A_action", has_action_output=True)
        assert result == Bucket.A_ACTION_INTERFACE

    def test_route_a_action_no_output(self):
        result = route_to_pipeline(gsa_bucket="A_action", has_action_output=False)
        assert result == Bucket.A_ACTION_PRECONDITION

    def test_route_a_state_with_state(self):
        result = route_to_pipeline(gsa_bucket="A_state", has_state_output=True)
        assert result == Bucket.A_OBJ_STATE

    def test_route_a_state_no_state(self):
        result = route_to_pipeline(gsa_bucket="A_state", has_state_output=False)
        assert result == Bucket.A_IDENTITY_ATTR

    def test_route_html_a3(self):
        result = route_to_pipeline(html_tags=["A.3"])
        assert result == Bucket.A_OBJ_STATE

    def test_route_html_a1(self):
        result = route_to_pipeline(html_tags=["A.1"])
        assert result == Bucket.A_IDENTITY_ATTR

    def test_route_default(self):
        result = route_to_pipeline()
        assert result == Bucket.A_IDENTITY_ATTR

    def test_bucket_subtypes_complete(self):
        assert Bucket.A_IDENTITY_ATTR in BUCKET_SUBTYPES
        assert Bucket.A_OBJ_STATE in BUCKET_SUBTYPES
        assert Bucket.A_ROBOT_STATE in BUCKET_SUBTYPES
        assert Bucket.A_ACTION_PRECONDITION in BUCKET_SUBTYPES
        assert Bucket.A_ACTION_INTERFACE in BUCKET_SUBTYPES

    def test_validate_new_subtypes(self):
        assert validate_bucket_subtype("A_identity_attr", "color")
        assert validate_bucket_subtype("A_obj_state", "pose")
        assert validate_bucket_subtype("A_robot_state", "gripper_empty")
        assert validate_bucket_subtype("A_action_precondition", "can_pick")
        assert validate_bucket_subtype("A_action_interface", "grasp_point")
        assert not validate_bucket_subtype("A_identity_attr", "gripper_empty")


class TestIdentityAttrSchema:
    """Test A_identity_attr schema validation."""

    def test_valid_label(self):
        label = AIdentityAttrLabel(
            sample_id="test_001",
            object_id="cup_0",
            object_class="cup",
            bbox=[100, 200, 300, 400],
            attributes={"color": "red", "material": "ceramic",
                        "shape": None, "size": None},
        )
        result = validate_identity_attr(label)
        assert result.valid
        assert len(result.errors) == 0

    def test_missing_object_class(self):
        label = AIdentityAttrLabel(sample_id="test_001", object_class="")
        result = validate_identity_attr(label)
        assert not result.valid
        assert "object_class is empty" in result.errors

    def test_unknown_color(self):
        label = AIdentityAttrLabel(
            sample_id="test_001",
            object_class="cup",
            attributes={"color": "neon_sparkle", "material": None,
                        "shape": None, "size": None},
        )
        result = validate_identity_attr(label)
        assert not result.valid
        assert any("unknown color" in e for e in result.errors)

    def test_invalid_bbox(self):
        label = AIdentityAttrLabel(
            sample_id="test_001",
            object_class="cup",
            bbox=[300, 200, 100, 400],
        )
        result = validate_identity_attr(label)
        assert not result.valid


class TestObjStateSchema:
    """Test A_obj_state schema validation."""

    def test_valid_label(self):
        label = AObjStateLabel(
            sample_id="test_001",
            object_id="door_0",
            state_type="open_close",
            state_value="open",
        )
        result = validate_obj_state(label)
        assert result.valid

    def test_invalid_state_type(self):
        label = AObjStateLabel(
            sample_id="test_001",
            state_type="nonexistent",
            state_value="yes",
        )
        result = validate_obj_state(label)
        assert not result.valid

    def test_invalid_state_value(self):
        label = AObjStateLabel(
            sample_id="test_001",
            state_type="pose",
            state_value="flying",
        )
        result = validate_obj_state(label)
        assert not result.valid


class TestRobotStateSchema:
    """Test A_robot_state schema validation."""

    def test_valid_label(self):
        label = ARobotStateLabel(
            sample_id="test_001",
            object_id="gripper_0",
            camera_view="wrist",
            state_type="gripper_empty",
            state_value="yes",
        )
        result = validate_robot_state(label)
        assert result.valid

    def test_invalid_camera_view(self):
        label = ARobotStateLabel(
            sample_id="test_001",
            camera_view="satellite",
            state_type="gripper_empty",
            state_value="yes",
        )
        result = validate_robot_state(label)
        assert not result.valid

    def test_missing_state_type(self):
        label = ARobotStateLabel(
            sample_id="test_001",
            camera_view="wrist",
            state_type="",
            state_value="yes",
        )
        result = validate_robot_state(label)
        assert not result.valid


class TestActionPreconditionSchema:
    """Test A_action_precondition schema validation."""

    def test_valid_can_do(self):
        label = AActionPreconditionLabel(
            sample_id="test_001",
            object_id="cup_0",
            action_type="pick",
            can_do_now=True,
        )
        result = validate_action_precondition(label)
        assert result.valid

    def test_cannot_do_needs_reason(self):
        label = AActionPreconditionLabel(
            sample_id="test_001",
            object_id="cup_0",
            action_type="pick",
            can_do_now=False,
            failure_reason=None,
        )
        result = validate_action_precondition(label)
        assert not result.valid
        assert any("failure_reason" in e for e in result.errors)

    def test_cannot_do_with_reason(self):
        label = AActionPreconditionLabel(
            sample_id="test_001",
            object_id="cup_0",
            action_type="pick",
            can_do_now=False,
            failure_reason="occluded",
        )
        result = validate_action_precondition(label)
        assert result.valid


class TestActionInterfaceSchema:
    """Test A_action_interface schema validation."""

    def test_valid_pick(self):
        label = AActionInterfaceLabel(
            sample_id="test_001",
            object_id="cup_0",
            action_type="pick",
            grasp_point_2d=[500, 300],
            jaw_axis_2d=[1.0, 0.0],
        )
        result = validate_action_interface(label)
        assert result.valid

    def test_missing_grasp_point(self):
        label = AActionInterfaceLabel(
            sample_id="test_001",
            object_id="cup_0",
            action_type="pick",
            grasp_point_2d=None,
            jaw_axis_2d=[1.0, 0.0],
        )
        result = validate_action_interface(label)
        assert not result.valid
        assert "grasp_point_2d" in result.missing_fields

    def test_invalid_vector(self):
        label = AActionInterfaceLabel(
            sample_id="test_001",
            object_id="cup_0",
            action_type="pick",
            grasp_point_2d=[500, 300],
            jaw_axis_2d=[0, 0],
        )
        result = validate_action_interface(label)
        assert not result.valid

    def test_valid_push(self):
        label = AActionInterfaceLabel(
            sample_id="test_001",
            object_id="card_0",
            action_type="push",
            push_contact_point_2d=[100, 200],
            push_direction_2d=[-1.0, 0.0],
        )
        result = validate_action_interface(label)
        assert result.valid


class TestContextFlags:
    """Test G_context and S_context detection."""

    def test_needs_g_context_with_bbox(self):
        assert needs_g_context({"bbox": [1, 2, 3, 4]})

    def test_needs_g_context_without(self):
        assert not needs_g_context({"text": "hello"})

    def test_needs_s_context_with_blocked(self):
        assert needs_s_context({"blocked_by": ["obj_1"]})

    def test_needs_s_context_without(self):
        assert not needs_s_context({"text": "hello"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
