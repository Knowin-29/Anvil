import unittest

from apps.data_flywheel.ontology.matcher import (
    RuntimeInstanceContext,
    build_runtime_label,
    resolve_observation_name,
)
from apps.data_flywheel.ontology.schema import (
    AliasEntry,
    BaseAffordance,
    ConstraintRule,
    IntrinsicActionProfile,
    GovernanceMetadata,
    ObjectClass,
    ObjectSubtype,
)
from apps.data_flywheel.ontology.seed_data import build_seed_store
from apps.data_flywheel.ontology.store import OntologyStore


class TestBuildRuntimeLabelNewSchema(unittest.TestCase):
    def _build_store(self) -> OntologyStore:
        store = OntologyStore()
        store.add_class(
            ObjectClass(
                class_id="spoon",
                name="勺子",
                affordance=BaseAffordance(
                    operation="pick",
                    is_container=False,
                    container_placeability="not_applicable",
                ),
                placement_policy="must_place",
                constraints=[
                    ConstraintRule(
                        rule_id="spoon-large",
                        source_field="estimated_max_dimension_cm",
                        operator="gt",
                        value=15,
                        target_field="operation",
                        effect_value="push",
                        reason="超过 15cm 改为推动",
                        priority=0,
                    ),
                ],
                metadata=GovernanceMetadata(status="approved"),
            )
        )
        store.add_subtype(
            ObjectSubtype(
                subtype_id="plastic_spoon",
                class_id="spoon",
                name="塑料勺",
                affordance_override=BaseAffordance(extra_tags=["disposable"]),
                metadata=GovernanceMetadata(status="approved"),
            )
        )
        store.add_alias(
            AliasEntry(
                alias="汤匙",
                class_id="spoon",
                subtype_id="plastic_spoon",
                status="approved",
            )
        )
        return store

    def test_resolved_label_has_operation_field(self):
        store = self._build_store()
        label = build_runtime_label("汤匙", store)
        self.assertEqual(label.operation, "pick")
        self.assertFalse(label.is_container)
        self.assertEqual(label.container_placeability, "not_applicable")
        self.assertEqual(label.placement_policy, "must_place")
        self.assertIn("disposable", label.extra_tags)

    def test_resolved_label_no_old_affordance_dict(self):
        store = self._build_store()
        label = build_runtime_label("汤匙", store)
        self.assertFalse(hasattr(label, "affordance"))
        self.assertFalse(hasattr(label, "placement"))

    def test_constraint_overrides_operation(self):
        store = self._build_store()
        ctx = RuntimeInstanceContext(
            bbox=[0, 0, 100, 30],
            image_width=120,
            image_height=80,
            estimated_max_dimension_cm=20,
        )
        label = build_runtime_label("汤匙", store, context=ctx)
        self.assertEqual(label.operation, "push")
        self.assertIn("超过 15cm 改为推动", label.triggered_constraints)

    def test_constraint_priority_order(self):
        store = OntologyStore()
        store.add_class(
            ObjectClass(
                class_id="item",
                name="物品",
                affordance=BaseAffordance(
                    operation="pick",
                    is_container=False,
                    container_placeability="not_applicable",
                ),
                constraints=[
                    ConstraintRule(
                        rule_id="low",
                        source_field="estimated_max_dimension_cm",
                        operator="gt",
                        value=5,
                        target_field="operation",
                        effect_value="push",
                        priority=0,
                    ),
                    ConstraintRule(
                        rule_id="high",
                        source_field="estimated_max_dimension_cm",
                        operator="gt",
                        value=5,
                        target_field="operation",
                        effect_value="non_operable",
                        priority=10,
                    ),
                ],
            )
        )
        store.add_alias(AliasEntry(alias="物品", class_id="item", status="approved"))
        ctx = RuntimeInstanceContext(
            bbox=[0, 0, 50, 50],
            image_width=100,
            image_height=100,
            estimated_max_dimension_cm=10,
        )
        label = build_runtime_label("物品", store, context=ctx)
        self.assertEqual(label.operation, "non_operable")

    def test_unknown_observation(self):
        store = self._build_store()
        label = build_runtime_label("未知物品", store)
        self.assertEqual(label.ontology_status, "unknown")
        self.assertIsNone(label.operation)

    def test_to_dict_output(self):
        store = self._build_store()
        label = build_runtime_label("汤匙", store)
        d = label.to_dict()
        self.assertEqual(d["operation"], "pick")
        self.assertEqual(d["placement_policy"], "must_place")
        self.assertNotIn("affordance", d)

    def test_resolve_observation_name_symbol(self):
        store = self._build_store()
        label = resolve_observation_name("汤匙", store)
        self.assertEqual(label.operation, "pick")

    def test_legacy_container_fallback_from_affordance(self):
        store = OntologyStore()
        store.add_class(
            ObjectClass(
                class_id="tray",
                name="托盘",
                affordance=BaseAffordance(is_container=True),
            )
        )

        label = build_runtime_label("托盘", store)

        self.assertEqual(label.role_ids, ["container"])
        self.assertTrue(label.is_container)

    def test_legacy_operation_fallback_from_affordance(self):
        store = OntologyStore()
        store.add_class(
            ObjectClass(
                class_id="pen",
                name="笔",
                affordance=BaseAffordance(operation="pick"),
            )
        )

        label = build_runtime_label("笔", store)

        self.assertEqual(label.operation, "pick")
        self.assertEqual(label.allowed_operations, ["pick"])

    def test_allowed_operations_only_uses_first_allowed_as_runtime_operation(self):
        store = OntologyStore()
        store.add_class(
            ObjectClass(
                class_id="tool",
                name="工具",
                intrinsic_action_profile=IntrinsicActionProfile(
                    allowed_operations=["pick", "push"]
                ),
            )
        )

        label = build_runtime_label("工具", store)

        self.assertEqual(label.allowed_operations, ["pick", "push"])
        self.assertEqual(label.operation, "pick")
        self.assertEqual(label.preferred_operation, "pick")

    def test_conflicting_container_state_is_rejected_at_runtime(self):
        store = OntologyStore()
        tray = ObjectClass(class_id="tray", name="托盘")
        tray.role_ids = ["container"]
        tray.affordance.is_container = False
        store.add_class(tray)

        with self.assertRaises(ValueError):
            build_runtime_label("托盘", store)

    def test_conflicting_operation_state_is_rejected_at_runtime(self):
        store = OntologyStore()
        spoon = ObjectClass(class_id="spoon", name="勺子")
        spoon.affordance.operation = "pick"
        spoon.intrinsic_action_profile.preferred_operation = "push"
        store.add_class(spoon)

        with self.assertRaises(ValueError):
            build_runtime_label("勺子", store)

    def test_resolve_observation_name_normalizes_leading_article(self):
        store = OntologyStore()
        store.add_class(
            ObjectClass(
                class_id="table",
                name="桌子",
                affordance=BaseAffordance(
                    operation="non_operable",
                    is_container=False,
                    container_placeability="not_applicable",
                ),
                placement_policy="leave_in_place",
                metadata=GovernanceMetadata(status="approved"),
            )
        )
        store.add_alias(AliasEntry(alias="table", class_id="table", status="approved"))

        label = build_runtime_label("a table", store)

        self.assertEqual(label.ontology_status, "resolved")
        self.assertEqual(label.object_class_id, "table")
        self.assertEqual(label.operation, "non_operable")

    def test_resolve_observation_name_normalizes_simple_plural(self):
        store = OntologyStore()
        store.add_class(
            ObjectClass(
                class_id="chopstick",
                name="筷子",
                affordance=BaseAffordance(
                    operation="pick",
                    is_container=False,
                    container_placeability="not_applicable",
                ),
                placement_policy="must_place",
                metadata=GovernanceMetadata(status="approved"),
            )
        )
        store.add_alias(AliasEntry(alias="chopstick", class_id="chopstick", status="approved"))

        label = build_runtime_label("chopsticks", store)

        self.assertEqual(label.ontology_status, "resolved")
        self.assertEqual(label.object_class_id, "chopstick")
        self.assertEqual(label.operation, "pick")


def test_seed_store_table_resolves_as_base_support_root_context():
    store = build_seed_store()

    label = store.resolve_observation_name("a table")

    assert label.object_class_id == "table"
    assert label.role_ids == ["support_surface", "base_support_root"]
    assert label.is_container is False
    assert label.allowed_operations == ["non_operable"]
    assert label.preferred_operation == "non_operable"


def test_seed_store_plate_resolves_as_support_surface_not_container():
    store = build_seed_store()

    label = store.resolve_observation_name("plate")

    assert label.object_class_id == "plate"
    assert label.role_ids == ["support_surface"]
    assert label.is_container is False


def test_seed_store_container_alias_prefers_concrete_box_identity():
    store = build_seed_store()

    label = store.resolve_observation_name("container")

    assert label.object_class_id == "box"
    assert "container" in label.role_ids
    assert label.is_container is True


def test_resolve_observation_name_derives_container_and_preferred_operation_from_role_ids():
    store = OntologyStore()
    store.add_class(
        ObjectClass(
            class_id="tray",
            name="tray",
            role_ids=["container", "support_surface"],
            intrinsic_action_profile=IntrinsicActionProfile(
                allowed_operations=["pick", "push"],
                preferred_operation="pick",
            ),
        )
    )

    label = resolve_observation_name("tray", store)

    assert label.role_ids == ["container", "support_surface"]
    assert label.is_container is True
    assert label.operation == "pick"
    assert label.allowed_operations == ["pick", "push"]


if __name__ == "__main__":
    unittest.main()
