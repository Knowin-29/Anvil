import json
import os

import pytest

from apps.data_flywheel.ontology.schema import (
    BaseAffordance,
    KNOWN_EXTRA_TAGS,
    ObjectClass,
    PLACEMENT_POLICIES,
)
from apps.data_flywheel.ontology.matcher import RuntimeInstanceContext
from apps.data_flywheel.ontology.seed_data import build_seed_store
from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.review.observation import CandidateCluster
from apps.data_flywheel.review.promotion import PromotionAction, apply_review_action


def test_known_extra_tags_whitelist():
    assert KNOWN_EXTRA_TAGS == frozenset(
        {
            "planar",
            "push_only",
            "stackable_2d",
            "fragile",
        }
    )


def test_placement_policies_whitelist():
    assert PLACEMENT_POLICIES == frozenset(
        {
            "must_place",
            "leave_in_place",
            "push_only",
        }
    )


def test_placement_policy_validation_rejects_unknown():
    with pytest.raises(ValueError):
        ObjectClass(
            class_id="bad",
            name="bad",
            affordance=BaseAffordance(),
            placement_policy="nonsense_policy",
        )


def test_seed_store_contains_expected_seed_classes():
    store = build_seed_store()
    expected = {
        "card",
        "pen",
        "cup",
        "book",
        "notebook",
        "plate",
        "bowl",
        "tray",
        "pen_holder",
        "keyboard",
        "table",
        "spoon",
        "chopstick",
        "marker",
        "box",
        "tape",
        "scissors",
        "stapler",
        "mouse",
        "paper",
        "container",
        "napkin",
        "chair",
    }
    assert set(store.classes.keys()) == expected


def test_seed_store_resolves_desktop_debug_aliases():
    store = build_seed_store()

    table = store.resolve_observation_name("a table")
    chopsticks = store.resolve_observation_name("chopsticks")
    sticky_notes = store.resolve_observation_name("sticky notes")

    assert table.object_class_id == "table"
    assert table.operation == "non_operable"
    assert chopsticks.object_class_id == "chopstick"
    assert chopsticks.operation == "pick"
    assert sticky_notes.object_class_id == "paper"
    assert sticky_notes.operation == "push"


def test_card_operation_is_push():
    store = build_seed_store()
    label = store.resolve_observation_name("卡片")
    assert label.operation == "push"
    assert label.placement_policy == "push_only"
    assert "push_only" in label.extra_tags
    assert "planar" in label.extra_tags
    assert "stackable_2d" in label.extra_tags


def test_card_has_no_orientation_tag():
    store = build_seed_store()
    label = store.resolve_observation_name("卡片")
    assert "orientation_sensitive" not in label.extra_tags
    assert "ocr_identified" not in label.extra_tags


def test_seed_store_round_trip(tmp_path):
    store = build_seed_store()
    path = str(tmp_path / "seed.json")
    store.save(path)
    loaded = OntologyStore.load(path)
    assert loaded.resolve_observation_name("卡片").operation == "push"
    assert loaded.classes["card"].placement_policy == "push_only"


def test_get_operation_shortcut():
    store = build_seed_store()
    assert store.get_operation("卡片") == "push"
    assert store.get_operation("笔") == "pick"
    assert store.get_operation("托盘") == "non_operable"
    assert store.get_operation("键盘") == "non_operable"
    assert store.get_operation("未知物体") is None


def test_new_class_from_review_has_none_operation(tmp_path):
    store = OntologyStore()
    store_path = str(tmp_path / "store.json")
    store.save(store_path)

    cluster = CandidateCluster(
        cluster_id="cluster-cup",
        representative_label="杯子",
        normalized_label="杯子",
        labels={"杯子": 2},
        observation_ids=["obs-1", "obs-2"],
        sample_images=["scene.jpg"],
        sample_crops=[],
        count=2,
        avg_confidence=0.91,
    )
    clusters_path = str(tmp_path / "clusters.jsonl")
    with open(clusters_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(cluster.to_dict(), ensure_ascii=False) + "\n")

    audit_path = str(tmp_path / "audit.jsonl")
    apply_review_action(
        store_path,
        clusters_path,
        audit_path,
        PromotionAction(cluster_id="cluster-cup", action_type="create_class", new_name="杯子"),
    )

    loaded = OntologyStore.load(store_path)
    created = loaded.get_class("杯子")
    assert created is not None
    assert created.affordance.operation is None
    assert created.metadata.status == "draft"


def test_backward_compat_old_placement_policy_default():
    old_data = {
        "class_id": "pen",
        "name": "笔",
        "affordance": {"operation": "pick"},
    }
    obj_class = ObjectClass.from_dict(old_data)
    assert obj_class.placement_policy == "must_place"


def test_card_context_does_not_change_operation():
    store = build_seed_store()
    context = RuntimeInstanceContext(
        bbox=[0, 0, 100, 200],
        image_width=1280,
        image_height=800,
        estimated_max_dimension_cm=25.0,
        mean_depth=0.5,
    )
    label = store.resolve_observation_name("卡片", context=context)
    assert label.operation == "push"
    assert label.triggered_constraints == []


def test_ontology_package_exports_stable_surface():
    import apps.data_flywheel.ontology as ontology

    assert ontology.OntologyStore is OntologyStore
    assert ontology.RuntimeInstanceContext is RuntimeInstanceContext
    assert ontology.KNOWN_EXTRA_TAGS == KNOWN_EXTRA_TAGS
    assert ontology.PLACEMENT_POLICIES == PLACEMENT_POLICIES
    assert hasattr(ontology, "ResolvedRuntimeLabel")
