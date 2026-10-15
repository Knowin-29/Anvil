from __future__ import annotations

from apps.data_flywheel.ontology.schema import BaseAffordance, ObjectClass
from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.training_data.phase_a.core import plan
from apps.data_flywheel.types import (
    Action,
    InstanceAttributes,
    PlacementRegion,
    SceneEdge,
    SceneGraph,
    SceneNode,
    THINKING_LINE_REGEX,
)


def _node(
    node_id: str,
    class_name: str,
    bbox,
    *,
    operation: str | None = None,
    grasp_point=None,
    extra_tags=None,
    hierarchical_name=None,
    referring_expressions=None,
    color=None,
):
    return SceneNode(
        id=node_id,
        class_name=class_name,
        bbox=list(bbox),
        operation=operation,
        grasp_point=list(grasp_point) if grasp_point is not None else None,
        extra_tags=list(extra_tags or []),
        hierarchical_name=hierarchical_name,
        referring_expressions=list(referring_expressions or []),
        attributes=InstanceAttributes(color_primary=color),
    )


def _edge(subject: str, object_: str, predicate: str) -> SceneEdge:
    return SceneEdge(subject=subject, object=object_, predicate=predicate, source="derived", confidence=1.0)


def _graph(nodes: list[SceneNode], edges=None, blocking_dag=None) -> SceneGraph:
    return SceneGraph(
        image_id="scene-a",
        image_path="tests/fixtures/images/scene-a.png",
        image_size=(1000, 1000),
        nodes=nodes,
        edges=list(edges or []),
        blocking_dag=list(blocking_dag or []),
    )


def _ontology() -> OntologyStore:
    store = OntologyStore()
    for name, operation in (
        ("pen", "pick"),
        ("card", "push"),
        ("book", "pick"),
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


def _graph_with_pen_and_bowl() -> SceneGraph:
    table = SceneNode(
        id="table_0",
        class_name="table",
        bbox=[0, 700, 1000, 1000],
        operation="push",
        role_ids=["support_surface", "base_support_root"],
        hierarchical_name="the table",
        referring_expressions=["桌子"],
    )
    bowl = SceneNode(
        id="bowl_0",
        class_name="bowl",
        bbox=[700, 650, 900, 900],
        operation="pick",
        is_container=True,
        role_ids=["container", "support_surface"],
        support_parent="table_0",
        support_type="on-top-of",
        hierarchical_name="the bowl",
        referring_expressions=["碗"],
    )
    pen = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[120, 720, 280, 840],
        operation="pick",
        grasp_point=[200, 780],
        support_parent="table_0",
        support_type="on-top-of",
        hierarchical_name="the pen",
        referring_expressions=["那支笔", "笔"],
    )
    return SceneGraph(
        image_id="scene-place",
        image_path="tests/fixtures/images/scene-place.png",
        image_size=(1000, 1000),
        nodes=[table, bowl, pen],
        edges=[],
        blocking_dag=[],
        placement_regions=[
            PlacementRegion(
                region_id="inside_bowl_0",
                region_type="inside-container",
                support_id="bowl_0",
                anchor_ids=["bowl_0"],
                point_2d=[800.0, 780.0],
                reason="visible_container",
            ),
            PlacementRegion(
                region_id="free_table_0",
                region_type="free-space",
                support_id="table_0",
                anchor_ids=["table_0"],
                point_2d=[520.0, 860.0],
                reason="support_surface_free_space",
            ),
        ],
    )


def _graph_with_card_and_bowl() -> SceneGraph:
    bowl_graph = _graph_with_pen_and_bowl()
    card = SceneNode(
        id="card_0",
        class_name="card",
        bbox=[120, 720, 280, 840],
        operation="push",
        grasp_point=[200, 780],
        extra_tags=["push_only"],
        support_parent="table_0",
        support_type="on-top-of",
        hierarchical_name="the card",
        referring_expressions=["卡片"],
    )
    return SceneGraph(
        image_id=bowl_graph.image_id,
        image_path=bowl_graph.image_path,
        image_size=bowl_graph.image_size,
        nodes=[bowl_graph.nodes[0], bowl_graph.nodes[1], card],
        edges=[],
        blocking_dag=[],
        placement_regions=list(bowl_graph.placement_regions),
    )


def _graph_with_pen_and_table_free_space() -> SceneGraph:
    table = SceneNode(
        id="table_0",
        class_name="table",
        bbox=[0, 700, 1000, 1000],
        operation="push",
        role_ids=["support_surface", "base_support_root"],
        hierarchical_name="the table",
        referring_expressions=["桌子"],
    )
    pen = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[120, 720, 280, 840],
        operation="pick",
        grasp_point=[200, 780],
        support_parent="table_0",
        support_type="on-top-of",
        hierarchical_name="the pen",
        referring_expressions=["那支笔", "笔"],
    )
    return SceneGraph(
        image_id="scene-table",
        image_path="tests/fixtures/images/scene-table.png",
        image_size=(1000, 1000),
        nodes=[table, pen],
        edges=[],
        blocking_dag=[],
        placement_regions=[
            PlacementRegion(
                region_id="free_table_0",
                region_type="free-space",
                support_id="table_0",
                anchor_ids=["table_0"],
                point_2d=[520.0, 860.0],
                reason="support_surface_free_space",
            )
        ],
    )


def test_plan_simple_pick_of_uncluttered_pen():
    pen = _node(
        "pen_0",
        "pen",
        [100, 100, 200, 200],
        operation="pick",
        grasp_point=[150, 150],
        hierarchical_name="the pen",
        referring_expressions=["那支笔"],
    )

    result = plan(_graph([pen]), "拿那支笔", _ontology())

    assert result.status == "success"
    assert result.resolved_target_id == "pen_0"
    assert result.actions == [
        Action(
            type="pick",
            target="pen_0",
            grasp_point_pixel=[150, 150],
            grasp_point_quantized=[150, 150],
            reason="main target",
        )
    ]


def test_plan_push_card():
    card = _node(
        "card_0",
        "card",
        [200, 200, 300, 300],
        operation="push",
        grasp_point=[250, 250],
        extra_tags=["push_only"],
        hierarchical_name="the card",
        referring_expressions=["那张卡片"],
    )

    result = plan(_graph([card]), "推那张卡片", _ontology())

    assert result.status == "success"
    assert [action.type for action in result.actions] == ["push"]
    assert [action.target for action in result.actions] == ["card_0"]


def test_plan_pick_pen_under_book():
    pen = _node(
        "pen_0",
        "pen",
        [500, 200, 600, 300],
        operation="pick",
        grasp_point=[550, 250],
        hierarchical_name="the pen",
        referring_expressions=["笔"],
    )
    book = _node(
        "book_0",
        "book",
        [450, 150, 650, 350],
        operation="pick",
        grasp_point=[520, 210],
        hierarchical_name="the book",
        referring_expressions=["书"],
    )

    scene_graph = _graph(
        [pen, book],
        edges=[_edge("book_0", "pen_0", "stacked-on")],
        blocking_dag=[("book_0", "pen_0")],
    )

    result = plan(scene_graph, "拿被书压住的笔", _ontology())

    assert result.status == "success"
    assert [action.type for action in result.actions] == ["pick", "pick"]
    assert [action.target for action in result.actions] == ["book_0", "pen_0"]


def test_plan_pick_pen_under_card_uses_push():
    pen = _node(
        "pen_0",
        "pen",
        [500, 200, 600, 300],
        operation="pick",
        grasp_point=[550, 250],
        hierarchical_name="the pen",
        referring_expressions=["笔"],
    )
    card = _node(
        "card_0",
        "card",
        [480, 180, 620, 320],
        operation="push",
        grasp_point=[540, 220],
        extra_tags=["push_only"],
        hierarchical_name="the card",
        referring_expressions=["卡片"],
    )

    scene_graph = _graph(
        [pen, card],
        edges=[_edge("card_0", "pen_0", "occluding")],
        blocking_dag=[("card_0", "pen_0")],
    )

    result = plan(scene_graph, "拿被卡片压住的笔", _ontology())

    assert result.status == "success"
    assert [action.type for action in result.actions] == ["push", "pick"]
    assert [action.target for action in result.actions] == ["card_0", "pen_0"]


def test_plan_put_pen_in_bowl_emits_pick_then_place():
    result = plan(_graph_with_pen_and_bowl(), "把那支笔放到碗里", _ontology())

    assert result.status == "success"
    assert [action.type for action in result.actions] == ["pick", "place"]
    assert result.actions[1].destination == "bowl_0"
    assert result.actions[1].destination_type == "container"
    assert result.actions[1].destination_anchor_ids == ["bowl_0"]


def test_plan_tidy_desk_uses_container_first_then_free_space():
    result = plan(_graph_with_pen_and_bowl(), "收拾桌面", _ontology())

    assert result.status == "success"
    assert any(action.type == "place" for action in result.actions)
    assert result.actions[-1].placement_reason in {"preferred_container", "fallback_free_space"}


def test_plan_place_requires_pickable_target():
    result = plan(_graph_with_card_and_bowl(), "把卡片放到碗里", _ontology())

    assert result.status == "blocked_unresolvable"
    assert result.actions == []


def test_plan_place_with_unknown_destination_returns_no_actions():
    result = plan(_graph_with_pen_and_bowl(), "把那支笔放到盒子里", _ontology())

    assert result.status == "unresolved"
    assert result.actions == []


def test_plan_place_requires_container_region():
    result = plan(_graph_with_pen_and_table_free_space(), "把那支笔放到桌子里", _ontology())

    assert result.status == "unresolved"
    assert result.actions == []


def test_plan_pick_push_only_card_fails():
    card = _node(
        "card_0",
        "card",
        [200, 200, 300, 300],
        operation="push",
        grasp_point=[250, 250],
        extra_tags=["push_only"],
        hierarchical_name="the card",
        referring_expressions=["那张卡片"],
    )

    result = plan(_graph([card]), "拿那张卡片", _ontology())

    assert result.status == "blocked_unresolvable"
    assert "push_only" in result.notes[0]


def test_thinking_contains_required_types():
    pen = _node(
        "pen_0",
        "pen",
        [100, 100, 200, 200],
        operation="pick",
        grasp_point=[150, 150],
        hierarchical_name="the pen",
        referring_expressions=["那支笔"],
    )

    result = plan(_graph([pen]), "拿那支笔", _ontology())

    types = {step.perception_type for step in result.thinking.steps}
    assert {"ReferenceResolution", "Affordance", "PlanOrder"} <= types


def test_thinking_line_format():
    pen = _node(
        "pen_0",
        "pen",
        [100, 100, 200, 200],
        operation="pick",
        grasp_point=[150, 150],
        hierarchical_name="the pen",
        referring_expressions=["那支笔"],
    )

    result = plan(_graph([pen]), "拿那支笔", _ontology())

    for line in result.thinking.to_text().split("\n"):
        if line.strip():
            assert THINKING_LINE_REGEX.match(line), f"bad line: {line}"
