"""Seed data generator for the ontology store."""

from __future__ import annotations

import os

from apps.data_flywheel.ontology.schema import (
    AliasEntry,
    BaseAffordance,
    DescriptorSchema,
    GovernanceMetadata,
    IntrinsicActionProfile,
    ObjectClass,
)
from apps.data_flywheel.ontology.store import OntologyStore


def _approved_metadata(source: str = "seed", notes: str = "") -> GovernanceMetadata:
    return GovernanceMetadata(status="approved", source=source, notes=notes)


def _intrinsic_profile(operation: str) -> IntrinsicActionProfile:
    return IntrinsicActionProfile(
        allowed_operations=[operation],
        preferred_operation=operation,
    )


def _descriptor_schema(priority: list[str], alias_hints: list[str] | None = None) -> DescriptorSchema:
    return DescriptorSchema(priority=priority, alias_hints=alias_hints or [])


def build_seed_store() -> OntologyStore:
    store = OntologyStore()

    store.add_class(
        ObjectClass(
            class_id="card",
            name="卡片",
            description="扁平的卡片类物体(银行卡/会员卡/名片/身份证)",
            intrinsic_action_profile=_intrinsic_profile("push"),
            descriptor_schema=_descriptor_schema(["color", "relative_position"]),
            affordance=BaseAffordance(
                operation="push",
                is_container=False,
                container_placeability="not_applicable",
                extra_tags=["planar", "push_only", "stackable_2d"],
            ),
            placement_policy="push_only",
            metadata=_approved_metadata(
                source="round2_requirement",
                notes="Round 2 决策:卡片 push-only,不识别朝向,不识别内容",
            ),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="pen",
            name="笔",
            description="书写工具,可拿起放入笔筒",
            intrinsic_action_profile=_intrinsic_profile("pick"),
            descriptor_schema=_descriptor_schema(["color", "relative_position"]),
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="cup",
            name="杯子",
            description="饮用容器,可拿起",
            role_ids=["container"],
            intrinsic_action_profile=_intrinsic_profile("pick"),
            descriptor_schema=_descriptor_schema(["color", "contents", "relative_position"]),
            affordance=BaseAffordance(
                operation="pick",
                is_container=True,
                container_placeability="placeable",
                extra_tags=["fragile"],
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="book",
            name="书",
            description="书本,可拿起",
            intrinsic_action_profile=_intrinsic_profile("pick"),
            descriptor_schema=_descriptor_schema(["color", "relative_position"]),
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="notebook",
            name="笔记本",
            description="纸质笔记本,可拿起",
            intrinsic_action_profile=_intrinsic_profile("pick"),
            descriptor_schema=_descriptor_schema(["color", "relative_position"]),
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="plate",
            name="盘子",
            description="扁平承载面,可拿起",
            role_ids=["support_surface"],
            intrinsic_action_profile=_intrinsic_profile("pick"),
            descriptor_schema=_descriptor_schema(["color", "contents", "relative_position"]),
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
                extra_tags=["fragile"],
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="bowl",
            name="碗",
            description="容器,可拿起",
            role_ids=["container"],
            intrinsic_action_profile=_intrinsic_profile("pick"),
            descriptor_schema=_descriptor_schema(["color", "contents", "relative_position"]),
            affordance=BaseAffordance(
                operation="pick",
                is_container=True,
                container_placeability="placeable",
                extra_tags=["fragile"],
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="tray",
            name="托盘",
            description="桌面上的大托盘,自己不移动,可容纳其他物体",
            role_ids=["container", "support_surface"],
            intrinsic_action_profile=_intrinsic_profile("non_operable"),
            descriptor_schema=_descriptor_schema(["contents", "relative_position"]),
            affordance=BaseAffordance(
                operation="non_operable",
                is_container=True,
                container_placeability="placeable",
            ),
            placement_policy="leave_in_place",
            metadata=_approved_metadata(),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="pen_holder",
            name="笔筒",
            description="容纳笔的容器",
            role_ids=["container"],
            intrinsic_action_profile=_intrinsic_profile("non_operable"),
            descriptor_schema=_descriptor_schema(["contents", "relative_position"]),
            affordance=BaseAffordance(
                operation="non_operable",
                is_container=True,
                container_placeability="placeable",
            ),
            placement_policy="leave_in_place",
            metadata=_approved_metadata(),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="keyboard",
            name="键盘",
            description="桌面键盘,不移动",
            role_ids=["support_surface"],
            intrinsic_action_profile=_intrinsic_profile("non_operable"),
            descriptor_schema=_descriptor_schema(["relative_position"]),
            affordance=BaseAffordance(
                operation="non_operable",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="leave_in_place",
            metadata=_approved_metadata(),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="table",
            name="桌子",
            description="桌面或台面,作为支撑面,默认不移动",
            role_ids=["support_surface", "base_support_root"],
            intrinsic_action_profile=_intrinsic_profile("non_operable"),
            descriptor_schema=_descriptor_schema(["relative_position"]),
            affordance=BaseAffordance(
                operation="non_operable",
                is_container=False,
                container_placeability="not_applicable",
                extra_tags=["planar"],
            ),
            placement_policy="leave_in_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="spoon",
            name="勺子",
            description="小型餐具,可拿起",
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="chopstick",
            name="筷子",
            description="细长餐具,可拿起",
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="marker",
            name="马克笔",
            description="可拿起的书写工具",
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="box",
            name="盒子",
            description="小型盒状物体,可拿起",
            role_ids=["container"],
            affordance=BaseAffordance(
                operation="pick",
                is_container=True,
                container_placeability="placeable",
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="tape",
            name="胶带",
            description="小型胶带卷,可拿起",
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="scissors",
            name="剪刀",
            description="小型工具,可拿起",
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="stapler",
            name="订书机",
            description="桌面工具,可拿起",
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="mouse",
            name="鼠标",
            description="桌面输入设备,可拿起",
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="paper",
            name="纸张",
            description="轻薄平面物体,默认以推动为主",
            affordance=BaseAffordance(
                operation="push",
                is_container=False,
                container_placeability="not_applicable",
                extra_tags=["planar", "push_only", "stackable_2d"],
            ),
            placement_policy="push_only",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="container",
            name="容器",
            description="通用小容器,默认不移动",
            affordance=BaseAffordance(
                operation="non_operable",
                is_container=True,
                container_placeability="placeable",
            ),
            placement_policy="leave_in_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="napkin",
            name="餐巾纸",
            description="轻薄可抓取物体",
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
                extra_tags=["planar"],
            ),
            placement_policy="must_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="chair",
            name="椅子",
            description="大型家具,默认不移动",
            affordance=BaseAffordance(
                operation="non_operable",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="leave_in_place",
            metadata=_approved_metadata(source="desktop_debug"),
        )
    )

    aliases = [
        ("card", "card"),
        ("pen", "pen"),
        ("cup", "cup"),
        ("book", "book"),
        ("notebook", "notebook"),
        ("plate", "plate"),
        ("bowl", "bowl"),
        ("tray", "tray"),
        ("pen holder", "pen_holder"),
        ("keyboard", "keyboard"),
        ("table", "table"),
        ("spoon", "spoon"),
        ("chopstick", "chopstick"),
        ("chopsticks", "chopstick"),
        ("marker", "marker"),
        ("box", "box"),
        ("tape", "tape"),
        ("scissors", "scissors"),
        ("stapler", "stapler"),
        ("mouse", "mouse"),
        ("paper", "paper"),
        ("note", "paper"),
        ("notes", "paper"),
        ("sticky note", "paper"),
        ("sticky notes", "paper"),
        ("container", "box"),
        ("napkin", "napkin"),
        ("chair", "chair"),
    ]
    for alias, class_id in aliases:
        store.add_alias(
            AliasEntry(
                alias=alias,
                class_id=class_id,
                status="approved",
                source="seed",
            )
        )

    return store


def main() -> None:
    store = build_seed_store()
    out_path = os.environ.get("ONTOLOGY_STORE_PATH", "data/ontology/ontology_store.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    store.save(out_path)
    print(f"Seed store written to {out_path}")


if __name__ == "__main__":
    main()
