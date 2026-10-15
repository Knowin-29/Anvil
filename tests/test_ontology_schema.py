import unittest

from apps.data_flywheel.ontology.schema import (
    BaseAffordance,
    ContainerProfile,
    DescriptorSchema,
    ConstraintRule,
    GovernanceMetadata,
    IntrinsicActionProfile,
    ObjectClass,
    ObjectSubtype,
    PlacementPreference,
)


class TestBaseAffordance(unittest.TestCase):
    def test_default_values_are_none(self):
        aff = BaseAffordance()
        self.assertIsNone(aff.operation)
        self.assertIsNone(aff.is_container)
        self.assertIsNone(aff.container_placeability)
        self.assertEqual(aff.extra_tags, [])

    def test_explicit_values(self):
        aff = BaseAffordance(
            operation="push",
            is_container=True,
            container_placeability="placeable",
        )
        self.assertEqual(aff.operation, "push")
        self.assertTrue(aff.is_container)
        self.assertEqual(aff.container_placeability, "placeable")

    def test_merged_with_none_override_inherits_parent(self):
        parent = BaseAffordance(
            operation="non_operable",
            is_container=True,
            container_placeability="placeable",
        )
        override = BaseAffordance()
        merged = parent.merged_with(override)
        self.assertEqual(merged.operation, "non_operable")
        self.assertTrue(merged.is_container)
        self.assertEqual(merged.container_placeability, "placeable")

    def test_merged_with_partial_override(self):
        parent = BaseAffordance(
            operation="pick",
            is_container=False,
            container_placeability="not_applicable",
        )
        override = BaseAffordance(operation="push")
        merged = parent.merged_with(override)
        self.assertEqual(merged.operation, "push")
        self.assertFalse(merged.is_container)

    def test_merged_with_extra_tags_union(self):
        parent = BaseAffordance(extra_tags=["stackable"])
        override = BaseAffordance(extra_tags=["fragile", "stackable"])
        merged = parent.merged_with(override)
        self.assertEqual(merged.extra_tags, ["stackable", "fragile"])

    def test_to_dict_from_dict_roundtrip(self):
        aff = BaseAffordance(
            operation="pick",
            is_container=False,
            container_placeability="not_applicable",
            extra_tags=["flat"],
        )
        restored = BaseAffordance.from_dict(aff.to_dict())
        self.assertEqual(restored.operation, aff.operation)
        self.assertEqual(restored.is_container, aff.is_container)
        self.assertEqual(restored.extra_tags, aff.extra_tags)

    def test_no_old_boolean_fields(self):
        aff = BaseAffordance()
        self.assertFalse(hasattr(aff, "can_pick"))
        self.assertFalse(hasattr(aff, "can_push"))
        self.assertFalse(hasattr(aff, "can_stack"))
        self.assertFalse(hasattr(aff, "is_surface"))


class TestContainerProfile(unittest.TestCase):
    def test_create_with_required_fields(self):
        cp = ContainerProfile(container_id=3)
        self.assertEqual(cp.container_id, 3)
        self.assertTrue(cp.enabled)
        self.assertEqual(cp.container_category, "")

    def test_to_dict_from_dict_roundtrip(self):
        cp = ContainerProfile(
            container_id=7,
            container_category="tray",
            enabled=False,
            capacity_hint="large",
        )
        restored = ContainerProfile.from_dict(cp.to_dict())
        self.assertEqual(restored.container_id, 7)
        self.assertEqual(restored.container_category, "tray")
        self.assertFalse(restored.enabled)


class TestPlacementPreference(unittest.TestCase):
    def test_fields(self):
        pref = PlacementPreference(object_class_id="pen", container_id=3, rank=1)
        self.assertEqual(pref.object_class_id, "pen")
        self.assertEqual(pref.container_id, 3)
        self.assertEqual(pref.rank, 1)

    def test_to_dict_from_dict_roundtrip(self):
        pref = PlacementPreference(object_class_id="pen", container_id=7, rank=2)
        restored = PlacementPreference.from_dict(pref.to_dict())
        self.assertEqual(restored.object_class_id, "pen")
        self.assertEqual(restored.rank, 2)


class TestConstraintRulePriority(unittest.TestCase):
    def test_default_priority_is_zero(self):
        rule = ConstraintRule(
            rule_id="r1",
            source_field="size",
            operator="gt",
            value=10,
            target_field="operation",
            effect_value="push",
        )
        self.assertEqual(rule.priority, 0)

    def test_explicit_priority(self):
        rule = ConstraintRule(
            rule_id="r2",
            source_field="size",
            operator="gt",
            value=10,
            target_field="operation",
            effect_value="push",
            priority=5,
        )
        self.assertEqual(rule.priority, 5)

    def test_to_dict_includes_priority(self):
        rule = ConstraintRule(
            rule_id="r1",
            source_field="s",
            operator="gt",
            value=1,
            target_field="t",
            effect_value="v",
            priority=3,
        )
        self.assertEqual(rule.to_dict()["priority"], 3)


class TestObjectClassNewFields(unittest.TestCase):
    def test_default_placement_policy(self):
        oc = ObjectClass(class_id="pen", name="笔")
        self.assertEqual(oc.placement_policy, "must_place")

    def test_container_profile_default_none(self):
        oc = ObjectClass(class_id="pen", name="笔")
        self.assertIsNone(oc.container_profile)

    def test_container_with_profile(self):
        oc = ObjectClass(
            class_id="tray",
            name="托盘",
            affordance=BaseAffordance(
                operation="non_operable",
                is_container=True,
                container_placeability="placeable",
            ),
            container_profile=ContainerProfile(
                container_id=1,
                container_category="tray",
            ),
            placement_policy="leave_in_place",
        )
        self.assertEqual(oc.container_profile.container_id, 1)
        self.assertEqual(oc.placement_policy, "leave_in_place")

    def test_no_placement_profile_field(self):
        oc = ObjectClass(class_id="pen", name="笔")
        self.assertFalse(hasattr(oc, "placement"))

    def test_to_dict_from_dict_roundtrip(self):
        oc = ObjectClass(
            class_id="tray",
            name="托盘",
            affordance=BaseAffordance(
                operation="non_operable",
                is_container=True,
                container_placeability="placeable",
            ),
            container_profile=ContainerProfile(
                container_id=1,
                container_category="tray",
            ),
            placement_policy="leave_in_place",
        )
        data = oc.to_dict()
        restored = ObjectClass.from_dict(data)
        self.assertEqual(restored.container_profile.container_id, 1)
        self.assertEqual(restored.placement_policy, "leave_in_place")
        self.assertEqual(restored.affordance.operation, "non_operable")

    def test_legacy_operation_falls_back_into_intrinsic_profile(self):
        oc = ObjectClass(
            class_id="pen",
            name="笔",
            affordance=BaseAffordance(operation="pick"),
        )
        self.assertEqual(oc.intrinsic_action_profile.preferred_operation, "pick")
        self.assertEqual(oc.intrinsic_action_profile.allowed_operations, ["pick"])

    def test_allowed_operations_are_deduped(self):
        oc = ObjectClass(
            class_id="tool",
            name="工具",
            affordance=BaseAffordance(operation="pick"),
            intrinsic_action_profile=IntrinsicActionProfile(
                allowed_operations=["pick", "pick", "push"],
                preferred_operation="pick",
            ),
        )
        self.assertEqual(oc.intrinsic_action_profile.allowed_operations, ["pick", "push"])
        self.assertEqual(oc.intrinsic_action_profile.preferred_operation, "pick")

    def test_allowed_operations_only_sets_legacy_operation(self):
        oc = ObjectClass(
            class_id="tool",
            name="工具",
            intrinsic_action_profile=IntrinsicActionProfile(
                allowed_operations=["pick", "push"]
            ),
        )
        self.assertEqual(oc.intrinsic_action_profile.preferred_operation, "pick")
        self.assertEqual(oc.affordance.operation, "pick")
        self.assertEqual(oc.intrinsic_action_profile.allowed_operations, ["pick", "push"])

    def test_conflicting_container_state_is_rejected(self):
        with self.assertRaises(ValueError):
            ObjectClass(
                class_id="tray",
                name="托盘",
                role_ids=["container"],
                affordance=BaseAffordance(is_container=False),
            )

    def test_conflicting_operation_state_is_rejected(self):
        with self.assertRaises(ValueError):
            ObjectClass(
                class_id="spoon",
                name="勺子",
                affordance=BaseAffordance(operation="pick"),
                intrinsic_action_profile=IntrinsicActionProfile(
                    preferred_operation="push"
                ),
            )


class TestObjectSubtypeNewFields(unittest.TestCase):
    def test_no_placement_override_field(self):
        st = ObjectSubtype(subtype_id="green_pen", class_id="pen", name="绿色笔")
        self.assertFalse(hasattr(st, "placement_override"))

    def test_container_profile_override(self):
        st = ObjectSubtype(
            subtype_id="small_tray",
            class_id="tray",
            name="小托盘",
            container_profile_override=ContainerProfile(
                container_id=2,
                capacity_hint="small",
            ),
        )
        self.assertEqual(st.container_profile_override.container_id, 2)


def test_object_class_serializes_role_ids_and_intrinsic_action_profile():
    cls = ObjectClass(
        class_id="bowl",
        name="bowl",
        role_ids=["container", "support_surface"],
        intrinsic_action_profile=IntrinsicActionProfile(
            allowed_operations=["pick", "push"],
            preferred_operation="pick",
        ),
        descriptor_schema=DescriptorSchema(
            priority=["color", "contents", "relative_position"]
        ),
    )

    data = cls.to_dict()

    assert data["role_ids"] == ["container", "support_surface"]
    assert data["intrinsic_action_profile"]["preferred_operation"] == "pick"
    assert data["descriptor_schema"]["priority"][0] == "color"


if __name__ == "__main__":
    unittest.main()
