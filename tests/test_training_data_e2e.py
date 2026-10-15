from __future__ import annotations

import json
from pathlib import Path

from apps.data_flywheel.ontology.schema import BaseAffordance, ObjectClass
from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.training_data.phase_a.core import plan
from apps.data_flywheel.training_data.phase_b.sample import synthesize_samples
from apps.data_flywheel.training_data.phase_c.main import validate_samples
from apps.data_flywheel.types import (
    InstanceAttributes,
    PlacementRegion,
    SceneEdge,
    SupportRelation,
    SceneGraph,
    SceneNode,
)


FIXTURE_PATH = Path("tests/fixtures/scenes/scene_03_card_occludes_pen.json")


def _ontology() -> OntologyStore:
    store = OntologyStore()
    for name, operation in (
        ("pen", "pick"),
        ("card", "push"),
        ("book", "pick"),
        ("box", "non_operable"),
        ("bowl", "pick"),
        ("table", "push"),
    ):
        store.add_class(
            ObjectClass(
                class_id=name,
                name=name,
                affordance=BaseAffordance(operation=operation),
                placement_policy="push_only" if operation == "push" else "must_place",
            )
        )
    return store



def _node_from_dict(data: dict) -> SceneNode:
    return SceneNode(
        id=data["id"],
        class_name=data["class_name"],
        bbox=list(data["bbox"]),
        mask_polygon=data.get("mask_polygon"),
        mask_file=data.get("mask_file"),
        grasp_point=data.get("grasp_point"),
        center_3d=data.get("center_3d"),
        size_3d=data.get("size_3d"),
        mean_depth=data.get("mean_depth"),
        valid_3d=data.get("valid_3d", False),
        operation=data.get("operation"),
        is_container=data.get("is_container", False),
        container_placeability=data.get("container_placeability"),
        placement_policy=data.get("placement_policy"),
        extra_tags=list(data.get("extra_tags", [])),
        canonical_name=data.get("canonical_name"),
        object_class_id=data.get("object_class_id"),
        ontology_status=data.get("ontology_status", "unknown"),
        triggered_constraints=list(data.get("triggered_constraints", [])),
        hierarchical_name=data.get("hierarchical_name"),
        referring_expressions=list(data.get("referring_expressions", [])),
        attributes=InstanceAttributes(**(data.get("attributes") or {})),
    )



def _edge_from_dict(data: dict) -> SceneEdge:
    return SceneEdge(
        subject=data["subject"],
        object=data["object"],
        predicate=data["predicate"],
        source=data.get("source", "derived"),
        confidence=data.get("confidence", 1.0),
    )



def _load_scene_graph(path: Path) -> SceneGraph:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    return SceneGraph(
        image_id=data["image_id"],
        image_path=data["image_path"],
        image_size=tuple(data["image_size"]),
        nodes=[_node_from_dict(node) for node in data["nodes"]],
        edges=[_edge_from_dict(edge) for edge in data["edges"]],
        ternary_relations=[],
        blocking_dag=[tuple(edge) for edge in data["blocking_dag"]],
        support_relations=[
            SupportRelation(
                subject=relation["subject"],
                object=relation["object"],
                relation=relation["relation"],
                confidence=float(relation.get("confidence", 1.0)),
            )
            for relation in data.get("support_relations", [])
        ],
        placement_regions=[
            PlacementRegion(
                region_id=region["region_id"],
                region_type=region["region_type"],
                support_id=region["support_id"],
                anchor_ids=list(region.get("anchor_ids", [])),
                polygon_2d=[list(point) for point in region.get("polygon_2d", [])],
                point_2d=list(region["point_2d"]) if region.get("point_2d") else None,
                reason=region.get("reason", ""),
            )
            for region in data.get("placement_regions", [])
        ],
    )



def _all_quantized_coords(samples):
    for sample in samples:
        for obj in sample.scene_objects:
            yield from obj.bbox_quantized
            if obj.grasp_point_quantized is not None:
                yield from obj.grasp_point_quantized
        for action in sample.actions:
            if action.grasp_point_quantized is not None:
                yield from action.grasp_point_quantized
            if action.place_point_quantized is not None:
                yield from action.place_point_quantized



def test_e2e_pen_occluded_by_card_produces_valid_sample():
    scene_graph = _load_scene_graph(FIXTURE_PATH)

    plan_result = plan(scene_graph, "拿被卡片压住的笔", _ontology())

    assert plan_result.status == "success"
    assert [action.type for action in plan_result.actions] == ["push", "pick"]
    assert [action.target for action in plan_result.actions] == ["card_0", "pen_0"]

    thinking_types = {step.perception_type for step in plan_result.thinking.steps}
    assert {"ReferenceResolution", "Affordance", "BlockingDependency", "PlanOrder"} <= thinking_types

    samples = synthesize_samples(
        image_path=scene_graph.image_path,
        image_size=scene_graph.image_size,
        scene_graph=scene_graph,
        plan_result=plan_result,
    )

    assert len(samples) == 3
    assert all("<|thinking_start|>" in sample.rex_omni_text for sample in samples)
    assert all("<|action_start|>" in sample.rex_omni_text for sample in samples)

    passed, rejected = validate_samples(samples)

    assert len(passed) == 3
    assert rejected == []
    assert all(0 <= coord <= 999 for coord in _all_quantized_coords(samples))


def test_e2e_pick_place_plan_produces_valid_samples():
    scene_graph = SceneGraph(
        image_id="scene_place",
        image_path="tests/fixtures/images/scene_place.png",
        image_size=(100, 100),
        nodes=[
            SceneNode(
                id="pen_0",
                class_name="pen",
                bbox=[10, 10, 20, 20],
                grasp_point=[15, 15],
                operation="pick",
                hierarchical_name="red pen",
                referring_expressions=["红笔", "笔"],
                attributes=InstanceAttributes(color_primary="red"),
            ),
            SceneNode(
                id="bowl_0",
                class_name="bowl",
                bbox=[60, 60, 90, 90],
                operation="pick",
                is_container=True,
                hierarchical_name="blue bowl",
                referring_expressions=["蓝碗", "碗"],
                attributes=InstanceAttributes(color_primary="blue"),
            ),
        ],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
        placement_regions=[
            PlacementRegion(
                region_id="inside_bowl_0",
                region_type="inside-container",
                support_id="bowl_0",
                anchor_ids=["bowl_0"],
                point_2d=[75.0, 75.0],
                reason="visible_container",
            )
        ],
    )

    plan_result = plan(scene_graph, "把红笔放到碗里", _ontology())

    assert plan_result.status == "success"
    assert [action.type for action in plan_result.actions] == ["pick", "place"]

    samples = synthesize_samples(
        image_path=scene_graph.image_path,
        image_size=scene_graph.image_size,
        scene_graph=scene_graph,
        plan_result=plan_result,
        variants_per_plan=1,
    )

    assert len(samples) == 1
    assert any(action.type == "place" for action in samples[0].actions)
    passed, rejected = validate_samples(samples)
    assert len(passed) == 1
    assert rejected == []


def test_tabletop_pick_place_fixture_produces_support_relations_and_place_action():
    scene_graph = _load_scene_graph(Path("tests/fixtures/scenes/scene_11_tabletop_pick_place.json"))
    plan_result = plan(scene_graph, "把那支笔放到碗里", _ontology())
    samples = synthesize_samples(
        image_path=scene_graph.image_path,
        image_size=scene_graph.image_size,
        scene_graph=scene_graph,
        plan_result=plan_result,
        variants_per_plan=1,
    )

    assert scene_graph.support_relations
    assert scene_graph.placement_regions
    assert plan_result.status == "success"
    assert samples[0].actions[-1].type == "place"
