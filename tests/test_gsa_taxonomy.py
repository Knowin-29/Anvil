"""Tests for GSA taxonomy and action schema."""

import pytest
from apps.data_flywheel.gsa.taxonomy import (
    Bucket, GSubtype, SSubtype, AStateSubtype, AActionSubtype,
    SceneFamily, Scenario, ActionType, VerifierStatus, FallbackAction,
    BUCKET_SUBTYPES, STATE_KEYS,
    validate_bucket_subtype, validate_scene_family, validate_scenario,
    get_bucket_for_subtype,
)
from apps.data_flywheel.gsa.action_schema import (
    AActionLabel, validate_a_action_label, ValidationResult,
)


class TestTaxonomy:
    def test_bucket_values(self):
        assert Bucket.G.value == "G"
        assert Bucket.S.value == "S"
        assert Bucket.A_STATE.value == "A_state"
        assert Bucket.A_ACTION.value == "A_action"

    def test_g_subtypes(self):
        assert len(GSubtype) == 4
        assert "simple_grounding" in [s.value for s in GSubtype]
        assert "counting" in [s.value for s in GSubtype]

    def test_s_subtypes(self):
        assert len(SSubtype) == 7
        assert "occlusion_relation" in [s.value for s in SSubtype]
        assert "stacking_relation" in [s.value for s in SSubtype]

    def test_a_action_subtypes(self):
        assert len(AActionSubtype) == 12
        assert "grasp_point_2d" in [s.value for s in AActionSubtype]
        assert "blocked_by" in [s.value for s in AActionSubtype]

    def test_validate_bucket_subtype_valid(self):
        assert validate_bucket_subtype("G", "simple_grounding")
        assert validate_bucket_subtype("S", "occlusion_relation")
        assert validate_bucket_subtype("A_state", "object_recognition")
        assert validate_bucket_subtype("A_action", "grasp_point_2d")

    def test_validate_bucket_subtype_invalid(self):
        assert not validate_bucket_subtype("G", "occlusion_relation")
        assert not validate_bucket_subtype("S", "simple_grounding")
        assert not validate_bucket_subtype("INVALID", "anything")

    def test_validate_scene_family(self):
        assert validate_scene_family("delivery")
        assert validate_scene_family("tabletop_tidy")
        assert validate_scene_family("storage")
        assert not validate_scene_family("kitchen")

    def test_get_bucket_for_subtype(self):
        assert get_bucket_for_subtype("simple_grounding") == Bucket.G
        assert get_bucket_for_subtype("occlusion_relation") == Bucket.S
        assert get_bucket_for_subtype("object_recognition") == Bucket.A_STATE
        assert get_bucket_for_subtype("grasp_point_2d") == Bucket.A_ACTION
        assert get_bucket_for_subtype("nonexistent") is None

    def test_state_keys(self):
        assert "open_closed" in STATE_KEYS
        assert "empty_filled" in STATE_KEYS
        assert "visible_occluded" in STATE_KEYS
        assert len(STATE_KEYS) == 8


class TestActionSchema:
    def test_valid_pick_label(self):
        label = AActionLabel(
            sample_id="test_001",
            scene_family="delivery",
            scenario_id="SC.1",
            object_id="cup_0",
            action_type="pick",
            label_type="point",
            verifier_status="auto_verifiable",
            source_provenance="vlm",
            grasp_point_2d=[100.0, 200.0],
            jaw_axis_2d=[1.0, 0.0],
            can_pick_now=True,
            blocked_by=[],
            need_pre_push=False,
        )
        result = validate_a_action_label(label)
        assert result.valid
        assert len(result.missing_fields) == 0

    def test_valid_push_label(self):
        label = AActionLabel(
            sample_id="test_002",
            scene_family="tabletop_tidy",
            scenario_id="SC.2",
            object_id="card_0",
            action_type="push",
            label_type="vector",
            verifier_status="auto_verifiable",
            source_provenance="vlm",
            push_contact_point_2d=[50.0, 100.0],
            push_direction_2d=[-1.0, 0.0],
            can_push_now=True,
        )
        result = validate_a_action_label(label)
        assert result.valid

    def test_valid_place_label(self):
        label = AActionLabel(
            sample_id="test_003",
            scene_family="storage",
            scenario_id="SC.3",
            object_id="bowl_0",
            action_type="place",
            label_type="region",
            verifier_status="auto_verifiable",
            source_provenance="vlm",
            place_region_2d={
                "type": "polygon",
                "points": [[10, 10], [100, 10], [100, 100], [10, 100]],
            },
            can_place_now=True,
        )
        result = validate_a_action_label(label)
        assert result.valid

    def test_missing_pick_fields(self):
        label = AActionLabel(
            sample_id="test_004",
            scene_family="delivery",
            scenario_id="SC.1",
            object_id="cup_0",
            action_type="pick",
            label_type="point",
            verifier_status="review_required",
            source_provenance="vlm",
            # Missing: grasp_point_2d, jaw_axis_2d, can_pick_now, blocked_by, need_pre_push
        )
        result = validate_a_action_label(label)
        assert not result.valid
        assert "grasp_point_2d" in result.missing_fields
        assert "jaw_axis_2d" in result.missing_fields

    def test_invalid_action_type(self):
        label = AActionLabel(
            sample_id="test_005",
            scene_family="delivery",
            scenario_id="SC.1",
            object_id="cup_0",
            action_type="fly",  # invalid
            label_type="point",
            verifier_status="auto_verifiable",
            source_provenance="vlm",
        )
        result = validate_a_action_label(label)
        assert not result.valid
        assert any("invalid action_type" in e for e in result.errors)

    def test_invalid_vector_zero(self):
        label = AActionLabel(
            sample_id="test_006",
            scene_family="delivery",
            scenario_id="SC.1",
            object_id="cup_0",
            action_type="pick",
            label_type="point",
            verifier_status="auto_verifiable",
            source_provenance="vlm",
            grasp_point_2d=[100.0, 200.0],
            jaw_axis_2d=[0.0, 0.0],  # zero vector invalid
            can_pick_now=True,
            blocked_by=[],
            need_pre_push=False,
        )
        result = validate_a_action_label(label)
        assert not result.valid
        assert any("jaw_axis_2d" in e for e in result.errors)

    def test_invalid_region_too_few_points(self):
        label = AActionLabel(
            sample_id="test_007",
            scene_family="storage",
            scenario_id="SC.3",
            object_id="bowl_0",
            action_type="place",
            label_type="region",
            verifier_status="auto_verifiable",
            source_provenance="vlm",
            place_region_2d={"type": "polygon", "points": [[10, 10], [20, 20]]},
            can_place_now=True,
        )
        result = validate_a_action_label(label)
        assert not result.valid

    def test_serialization(self):
        label = AActionLabel(
            sample_id="test_008",
            scene_family="delivery",
            scenario_id="SC.1",
            object_id="cup_0",
            action_type="pick",
            label_type="point",
            verifier_status="auto_verifiable",
            source_provenance="vlm",
            grasp_point_2d=[100.0, 200.0],
            jaw_axis_2d=[1.0, 0.0],
            can_pick_now=True,
            blocked_by=[],
            need_pre_push=False,
        )
        jsonl = label.to_jsonl()
        import json
        parsed = json.loads(jsonl)
        assert parsed["sample_id"] == "test_008"
        assert parsed["action_type"] == "pick"
        assert parsed["grasp_point_2d"] == [100.0, 200.0]
