import json
import os
import tempfile
import unittest

from apps.data_flywheel.ontology.schema import (
    AliasEntry,
    BaseAffordance,
    ContainerProfile,
    GovernanceMetadata,
    ObjectClass,
    ObjectSubtype,
    PlacementPreference,
)
from apps.data_flywheel.ontology.store import OntologyStore, QueryEntry


class TestOntologyStorePreferences(unittest.TestCase):
    def _build_store(self) -> OntologyStore:
        store = OntologyStore()
        store.add_class(
            ObjectClass(
                class_id="pen",
                name="笔",
                affordance=BaseAffordance(
                    operation="pick",
                    is_container=False,
                    container_placeability="not_applicable",
                ),
            )
        )
        store.add_class(
            ObjectClass(
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
        )
        store.add_class(
            ObjectClass(
                class_id="bowl",
                name="碗",
                affordance=BaseAffordance(
                    operation="non_operable",
                    is_container=True,
                    container_placeability="placeable",
                ),
                container_profile=ContainerProfile(
                    container_id=2,
                    container_category="bowl",
                ),
                placement_policy="leave_in_place",
            )
        )
        store.preferences = [
            PlacementPreference(object_class_id="pen", container_id=1, rank=1),
            PlacementPreference(object_class_id="pen", container_id=2, rank=2),
        ]
        return store

    def test_save_load_roundtrip_with_preferences(self):
        store = self._build_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "store.json")
            store.save(path)
            loaded = OntologyStore.load(path)
            self.assertEqual(len(loaded.preferences), 2)
            self.assertEqual(loaded.preferences[0].object_class_id, "pen")
            self.assertEqual(loaded.preferences[0].rank, 1)

    def test_save_load_roundtrip_with_container_profile(self):
        store = self._build_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "store.json")
            store.save(path)
            loaded = OntologyStore.load(path)
            tray = loaded.get_class("tray")
            self.assertIsNotNone(tray.container_profile)
            self.assertEqual(tray.container_profile.container_id, 1)
            self.assertEqual(tray.placement_policy, "leave_in_place")

    def test_backward_compat_load_old_format(self):
        old_data = {
            "classes": {
                "pen": {
                    "class_id": "pen",
                    "name": "笔",
                    "affordance": {
                        "can_pick": True,
                        "can_push": False,
                        "can_stack": False,
                        "is_container": False,
                        "is_surface": False,
                        "extra_tags": [],
                    },
                    "placement": {
                        "default_container_types": ["笔筒"],
                        "placement_modes": ["竖放"],
                        "preferred_supports": [],
                        "notes": "",
                    },
                    "perception": {
                        "needs_size_estimate": False,
                        "needs_pose_estimate": False,
                        "needs_container_state": False,
                        "needs_depth_validation": False,
                        "extra_checks": [],
                    },
                    "constraints": [
                        {
                            "rule_id": "r1",
                            "source_field": "s",
                            "operator": "gt",
                            "value": 1,
                            "target_field": "t",
                            "effect_value": "v",
                            "reason": "",
                            "severity": "block",
                        }
                    ],
                    "metadata": {
                        "status": "approved",
                        "source": "manual",
                        "confidence": 1.0,
                        "version": 1,
                        "reviewer": "",
                        "notes": "",
                    },
                    "alias_auto_approve": True,
                }
            },
            "subtypes": {},
            "aliases": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old_store.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(old_data, f, ensure_ascii=False)
            loaded = OntologyStore.load(path)
            pen = loaded.get_class("pen")
            self.assertIsNone(pen.affordance.operation)
            self.assertIsNone(pen.container_profile)
            self.assertEqual(pen.placement_policy, "must_place")
            self.assertEqual(pen.constraints[0].priority, 0)
            self.assertEqual(len(loaded.preferences), 0)


class TestExportQueryList(unittest.TestCase):
    def _build_store(self) -> OntologyStore:
        store = OntologyStore()
        store.add_class(
            ObjectClass(
                class_id="pen",
                name="笔",
                affordance=BaseAffordance(
                    operation="pick",
                    is_container=False,
                    container_placeability="not_applicable",
                ),
                metadata=GovernanceMetadata(status="approved"),
            )
        )
        store.add_subtype(
            ObjectSubtype(
                subtype_id="green_pen",
                class_id="pen",
                name="绿色笔",
                metadata=GovernanceMetadata(status="approved"),
            )
        )
        store.add_alias(AliasEntry(alias="钢笔", class_id="pen", status="approved"))
        store.add_alias(AliasEntry(alias="铅笔", class_id="pen", status="draft"))
        return store

    def test_class_level(self):
        store = self._build_store()
        queries = store.export_query_list(level="class")
        names = [q.query for q in queries]
        self.assertIn("笔", names)
        self.assertNotIn("绿色笔", names)
        self.assertNotIn("钢笔", names)

    def test_subtype_level(self):
        store = self._build_store()
        queries = store.export_query_list(level="subtype")
        names = [q.query for q in queries]
        self.assertIn("笔", names)
        self.assertIn("绿色笔", names)
        self.assertNotIn("钢笔", names)

    def test_alias_level(self):
        store = self._build_store()
        queries = store.export_query_list(level="alias")
        names = [q.query for q in queries]
        self.assertIn("笔", names)
        self.assertIn("绿色笔", names)
        self.assertIn("钢笔", names)
        self.assertNotIn("铅笔", names)

    def test_query_entry_has_level(self):
        store = self._build_store()
        queries = store.export_query_list(level="all")
        class_entry = next(q for q in queries if q.query == "笔")
        self.assertIsInstance(class_entry, QueryEntry)
        self.assertEqual(class_entry.level, "class")
        subtype_entry = next(q for q in queries if q.query == "绿色笔")
        self.assertEqual(subtype_entry.level, "subtype")
        alias_entry = next(q for q in queries if q.query == "钢笔")
        self.assertEqual(alias_entry.level, "alias")


class TestGetPreferredContainer(unittest.TestCase):
    def _build_store(self) -> OntologyStore:
        store = OntologyStore()
        store.add_class(
            ObjectClass(
                class_id="pen",
                name="笔",
                affordance=BaseAffordance(
                    operation="pick",
                    is_container=False,
                    container_placeability="not_applicable",
                ),
            )
        )
        store.add_class(
            ObjectClass(
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
            )
        )
        store.add_class(
            ObjectClass(
                class_id="bowl",
                name="碗",
                affordance=BaseAffordance(
                    operation="non_operable",
                    is_container=True,
                    container_placeability="placeable",
                ),
                container_profile=ContainerProfile(
                    container_id=2,
                    container_category="bowl",
                ),
            )
        )
        store.add_class(
            ObjectClass(
                class_id="box",
                name="盒子",
                affordance=BaseAffordance(
                    operation="non_operable",
                    is_container=True,
                    container_placeability="placeable",
                ),
                container_profile=ContainerProfile(
                    container_id=3,
                    container_category="box",
                    enabled=False,
                ),
            )
        )
        store.preferences = [
            PlacementPreference(object_class_id="pen", container_id=3, rank=1),
            PlacementPreference(object_class_id="pen", container_id=1, rank=2),
            PlacementPreference(object_class_id="pen", container_id=2, rank=3),
        ]
        return store

    def test_returns_highest_rank_visible_enabled(self):
        store = self._build_store()
        result = store.get_preferred_container("pen", visible_container_ids={1, 2, 3})
        self.assertEqual(result, 1)

    def test_returns_none_when_no_match(self):
        store = self._build_store()
        result = store.get_preferred_container("pen", visible_container_ids={99})
        self.assertIsNone(result)

    def test_returns_none_for_unknown_class(self):
        store = self._build_store()
        result = store.get_preferred_container("unknown_class", visible_container_ids={1, 2})
        self.assertIsNone(result)

    def test_skips_invisible_containers(self):
        store = self._build_store()
        result = store.get_preferred_container("pen", visible_container_ids={2})
        self.assertEqual(result, 2)


class TestContainerQueries(unittest.TestCase):
    def _build_store(self) -> OntologyStore:
        store = OntologyStore()
        store.add_class(ObjectClass(class_id="pen", name="笔", affordance=BaseAffordance(operation="pick")))
        store.add_class(
            ObjectClass(
                class_id="tray",
                name="托盘",
                affordance=BaseAffordance(
                    operation="non_operable",
                    is_container=True,
                    container_placeability="placeable",
                ),
                container_profile=ContainerProfile(container_id=1, enabled=True),
            )
        )
        store.add_class(
            ObjectClass(
                class_id="box",
                name="盒子",
                affordance=BaseAffordance(
                    operation="non_operable",
                    is_container=True,
                    container_placeability="placeable",
                ),
                container_profile=ContainerProfile(container_id=2, enabled=False),
            )
        )
        return store

    def test_list_containers_enabled_only(self):
        store = self._build_store()
        containers = store.list_containers(enabled_only=True)
        ids = [c.class_id for c in containers]
        self.assertIn("tray", ids)
        self.assertNotIn("box", ids)
        self.assertNotIn("pen", ids)

    def test_list_containers_all(self):
        store = self._build_store()
        containers = store.list_containers(enabled_only=False)
        self.assertEqual(len(containers), 2)

    def test_get_container_by_id(self):
        store = self._build_store()
        tray = store.get_container_by_id(1)
        self.assertEqual(tray.class_id, "tray")

    def test_get_container_by_id_not_found(self):
        store = self._build_store()
        self.assertIsNone(store.get_container_by_id(99))


if __name__ == "__main__":
    unittest.main()
