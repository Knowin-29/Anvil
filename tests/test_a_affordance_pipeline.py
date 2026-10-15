"""Tests for A_Affordance pipeline — schema validation, heuristic fallback, and integration."""

import pytest
import sys
sys.path.insert(0, ".")

from apps.data_flywheel.gsa.taxonomy import (
    AAffordanceSubtype,
    Bucket,
    BUCKET_SUBTYPES,
    validate_bucket_subtype,
)
from apps.data_flywheel.gsa.a_pipeline_schemas import (
    AAffordanceLabel,
    KNOWN_AFFORDANCE_ACTIONS,
    KNOWN_APPROACH_DIRECTIONS,
    validate_affordance,
)
from apps.data_flywheel.gsa.a_affordance_pipeline import AAffordancePipeline
from apps.data_flywheel.gsa.visual_evidence import GsaEvidenceBundle, GsaObjectRef


def _make_obj(object_id="obj_001", name="red cup", bbox=None):
    return GsaObjectRef(
        object_id=object_id,
        object_name=name,
        raw_object_name=name,
        bbox=bbox or [100, 100, 200, 200],
        source_provenance="vlm",
    )


def _make_bundle(objects=None, spatial_facts=None):
    if objects is None:
        objects = [_make_obj()]
    return GsaEvidenceBundle(
        sample_id="test_sample_001",
        image_path="/tmp/test_image.jpg",
        scene_family="tabletop_tidy",
        scenario_id="SC.2",
        objects=objects,
        spatial_facts=spatial_facts or [],
    )


# ============================================================
# Taxonomy registration tests
# ============================================================

class TestAffordanceTaxonomy:
    def test_bucket_exists(self):
        assert Bucket.A_AFFORDANCE.value == "A_affordance"

    def test_subtypes_registered(self):
        subtypes = BUCKET_SUBTYPES[Bucket.A_AFFORDANCE]
        assert "handle_point" in subtypes
        assert "button_point" in subtypes
        assert "knob_point" in subtypes
        assert "interaction_hotspot" in subtypes

    def test_validate_bucket_subtype(self):
        assert validate_bucket_subtype("A_affordance", "handle_point")
        assert validate_bucket_subtype("A_affordance", "interaction_hotspot")
        assert not validate_bucket_subtype("A_affordance", "grasp_point")


# ============================================================
# Schema validation tests
# ============================================================

class TestAffordanceSchema:
    def test_valid_label(self):
        label = AAffordanceLabel(
            sample_id="s_001",
            object_id="obj_001",
            object_name="red cup",
            bbox=[100, 100, 200, 200],
            shape_description="cylindrical cup",
            action="pick",
            reason="cup is graspable around the body",
            contact_point=[150, 150],
            approach_direction="top",
            confidence=0.85,
        )
        result = validate_affordance(label)
        assert result.valid
        assert result.errors == []

    def test_invalid_action(self):
        label = AAffordanceLabel(
            sample_id="s_001",
            object_id="obj_001",
            object_name="thing",
            action="fly",
        )
        result = validate_affordance(label)
        assert not result.valid
        assert any("unknown action" in e for e in result.errors)

    def test_missing_action(self):
        label = AAffordanceLabel(
            sample_id="s_001",
            object_id="obj_001",
            object_name="thing",
            action="",
        )
        result = validate_affordance(label)
        assert not result.valid
        assert any("action is empty" in e for e in result.errors)

    def test_missing_identity(self):
        label = AAffordanceLabel(
            sample_id="s_001",
            object_name="",
            object_id="",
            action="pick",
        )
        result = validate_affordance(label)
        assert not result.valid
        assert any("object_name or object_id required" in e for e in result.errors)

    def test_contact_point_outside_bbox(self):
        label = AAffordanceLabel(
            sample_id="s_001",
            object_id="obj_001",
            object_name="cup",
            bbox=[100, 100, 200, 200],
            action="pick",
            contact_point=[300, 300],
        )
        result = validate_affordance(label)
        assert not result.valid
        assert any("outside bbox" in e for e in result.errors)

    def test_contact_point_inside_bbox(self):
        label = AAffordanceLabel(
            sample_id="s_001",
            object_id="obj_001",
            object_name="cup",
            bbox=[100, 100, 200, 200],
            action="pick",
            contact_point=[150, 150],
        )
        result = validate_affordance(label)
        assert result.valid

    def test_invalid_approach_direction(self):
        label = AAffordanceLabel(
            sample_id="s_001",
            object_id="obj_001",
            object_name="cup",
            action="pick",
            approach_direction="sideways",
        )
        result = validate_affordance(label)
        assert not result.valid
        assert any("unknown approach_direction" in e for e in result.errors)

    def test_invalid_bbox(self):
        label = AAffordanceLabel(
            sample_id="s_001",
            object_id="obj_001",
            object_name="cup",
            action="pick",
            bbox=[200, 200, 100, 100],
        )
        result = validate_affordance(label)
        assert not result.valid
        assert any("invalid dimensions" in e for e in result.errors)

    def test_to_dict(self):
        label = AAffordanceLabel(
            sample_id="s_001",
            object_id="obj_001",
            object_name="cup",
            action="pick",
            contact_point=[150, 150],
            approach_direction="top",
        )
        d = label.to_dict()
        assert d["bucket"] == "A_affordance"
        assert d["action"] == "pick"
        assert d["contact_point"] == [150, 150]

    def test_all_known_actions(self):
        for action in KNOWN_AFFORDANCE_ACTIONS:
            label = AAffordanceLabel(
                sample_id="s_001", object_id="obj_001",
                object_name="thing", action=action,
            )
            result = validate_affordance(label)
            assert result.valid, f"action '{action}' should be valid"

    def test_all_known_directions(self):
        for direction in KNOWN_APPROACH_DIRECTIONS:
            label = AAffordanceLabel(
                sample_id="s_001", object_id="obj_001",
                object_name="thing", action="pick",
                approach_direction=direction,
            )
            result = validate_affordance(label)
            assert result.valid, f"direction '{direction}' should be valid"


# ============================================================
# Pipeline heuristic tests (no VLM)
# ============================================================

class TestAffordancePipelineHeuristic:
    def setup_method(self):
        self.pipeline = AAffordancePipeline(vlm_client=None)

    def test_basic_object(self):
        bundle = _make_bundle()
        labels = self.pipeline.run(bundle)
        assert len(labels) == 1
        label = labels[0]
        assert label.action in KNOWN_AFFORDANCE_ACTIONS
        assert label.contact_point is not None
        assert label.source_provenance == "adapter"
        assert label.verifier_status == "heuristic_only"

    def test_button_object(self):
        obj = _make_obj(name="power button", bbox=[50, 50, 100, 80])
        bundle = _make_bundle(objects=[obj])
        labels = self.pipeline.run(bundle)
        assert len(labels) == 1
        assert labels[0].action == "press"
        assert labels[0].subtype == AAffordanceSubtype.BUTTON_POINT.value

    def test_handle_object(self):
        obj = _make_obj(name="door handle", bbox=[200, 300, 280, 340])
        bundle = _make_bundle(objects=[obj])
        labels = self.pipeline.run(bundle)
        assert len(labels) == 1
        assert labels[0].action == "pull"
        assert labels[0].subtype == AAffordanceSubtype.HANDLE_POINT.value

    def test_knob_object(self):
        obj = _make_obj(name="rotary knob", bbox=[80, 80, 160, 160])
        bundle = _make_bundle(objects=[obj])
        labels = self.pipeline.run(bundle)
        assert len(labels) == 1
        assert labels[0].action == "rotate"
        assert labels[0].subtype == AAffordanceSubtype.KNOB_POINT.value

    def test_thin_flat_object(self):
        obj = _make_obj(name="paper sheet", bbox=[50, 100, 400, 120])
        bundle = _make_bundle(objects=[obj])
        labels = self.pipeline.run(bundle)
        assert len(labels) == 1
        assert labels[0].action == "push"

    def test_skip_background(self):
        obj = _make_obj(name="wall", bbox=[0, 0, 640, 480])
        bundle = _make_bundle(objects=[obj])
        labels = self.pipeline.run(bundle)
        assert len(labels) == 0

    def test_skip_robot(self):
        obj = _make_obj(name="robot arm", bbox=[100, 100, 300, 300])
        bundle = _make_bundle(objects=[obj])
        labels = self.pipeline.run(bundle)
        assert len(labels) == 0

    def test_multiple_objects(self):
        objects = [
            _make_obj("obj_1", "red cup", [100, 100, 200, 200]),
            _make_obj("obj_2", "blue plate", [300, 100, 500, 200]),
            _make_obj("obj_3", "wall", [0, 0, 640, 480]),
        ]
        bundle = _make_bundle(objects=objects)
        labels = self.pipeline.run(bundle)
        assert len(labels) == 2  # wall is skipped

    def test_empty_objects(self):
        bundle = _make_bundle(objects=[])
        labels = self.pipeline.run(bundle)
        assert len(labels) == 0

    def test_contact_point_in_bbox(self):
        obj = _make_obj(name="cup", bbox=[100, 100, 200, 200])
        bundle = _make_bundle(objects=[obj])
        labels = self.pipeline.run(bundle)
        assert len(labels) == 1
        cp = labels[0].contact_point
        bbox = labels[0].bbox
        assert bbox[0] <= cp[0] <= bbox[2]
        assert bbox[1] <= cp[1] <= bbox[3]

    def test_label_has_correct_bucket(self):
        bundle = _make_bundle()
        labels = self.pipeline.run(bundle)
        assert all(l.bucket == "A_affordance" for l in labels)


# ============================================================
# Integration with action_interface pipeline
# ============================================================

class TestAffordanceActionInterfaceIntegration:
    def test_action_interface_accepts_affordance_hints(self):
        from apps.data_flywheel.gsa.a_action_interface_pipeline import AActionInterfacePipeline

        pipeline = AActionInterfacePipeline(vlm_client=None)
        bundle = _make_bundle()

        # Without hints
        labels_no_hint = pipeline.run(bundle, affordance_hints=None)

        # With hints
        aff_label = AAffordanceLabel(
            sample_id="test_sample_001",
            object_id="obj_001",
            object_name="red cup",
            bbox=[100, 100, 200, 200],
            action="push",
            contact_point=[100, 150],
            approach_direction="front",
        )
        labels_with_hint = pipeline.run(bundle, affordance_hints=[aff_label])

        # With push hint, action_interface should generate push (not pick)
        assert any(l.action_type == "push" for l in labels_with_hint)
