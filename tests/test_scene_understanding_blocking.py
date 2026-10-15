import logging

from apps.data_flywheel.scene_understanding.blocking import derive_blocking_dag, populate_node_actionability
from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations
from apps.data_flywheel.scene_understanding.thresholds import SceneGraphThresholds
from apps.data_flywheel.types import SceneEdge
from apps.data_flywheel.types import SceneNode


def _node(node_id: str, bbox, *, valid_3d=True, center_3d=None, size_3d=None, mean_depth=None, extra_tags=None, is_container=False, operation=None):
    return SceneNode(
        id=node_id,
        class_name="obj",
        bbox=bbox,
        valid_3d=valid_3d,
        center_3d=center_3d,
        size_3d=size_3d,
        mean_depth=mean_depth,
        extra_tags=list(extra_tags or []),
        is_container=is_container,
        operation=operation,
    )


def _find_edge(edges, predicate, subject=None, object_=None):
    for edge in edges:
        if edge.predicate != predicate:
            continue
        if subject is not None and edge.subject != subject:
            continue
        if object_ is not None and edge.object != object_:
            continue
        return edge
    return None


def _edge(subject, object_, predicate, source="3d"):
    return SceneEdge(subject=subject, object=object_, predicate=predicate, source=source, confidence=0.9)


def test_planar_tag_uses_2d_stacking():
    nodes = [
        _node("a", [0, 0, 20, 10], center_3d=[0, 0.0, 0.1], size_3d=[1, 0.1, 1], mean_depth=0.1, extra_tags=["planar"]),
        _node("b", [0, 11, 20, 21], center_3d=[0, 0.2, 0.1], size_3d=[1, 0.1, 1], mean_depth=0.2),
    ]

    edges = compute_3d_relations(nodes, (100, 100), SceneGraphThresholds())

    edge = _find_edge(edges, "stacked-on", "a", "b")
    assert edge is not None
    assert edge.source == "2d"


def test_stackable_2d_tag_uses_2d_stacking():
    nodes = [
        _node("a", [0, 0, 20, 10], center_3d=[0, 0.0, 0.1], size_3d=[1, 0.1, 1], mean_depth=0.1, extra_tags=["stackable_2d"]),
        _node("b", [0, 11, 20, 21], center_3d=[0, 0.2, 0.1], size_3d=[1, 0.1, 1], mean_depth=0.2),
    ]

    edges = compute_3d_relations(nodes, (100, 100), SceneGraphThresholds())

    edge = _find_edge(edges, "stacked-on", "a", "b")
    assert edge is not None
    assert edge.source == "2d"


def test_blocking_from_stacked_on():
    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0], size_3d=[1, 1, 1], mean_depth=0.1),
        _node("b", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0.1, 0], size_3d=[1, 1, 1], mean_depth=0.2),
    ]

    dag = derive_blocking_dag(nodes, [_edge("a", "b", "stacked-on")])

    assert dag == [("a", "b")]


def test_blocking_from_occluding_pickable():
    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0.1], size_3d=[1, 1, 1], mean_depth=0.1, operation="pick"),
        _node("b", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0.3], size_3d=[1, 1, 1], mean_depth=0.3),
    ]

    dag = derive_blocking_dag(nodes, [_edge("a", "b", "occluding")])

    assert dag == [("a", "b")]


def test_blocking_from_occluding_non_operable_skipped():
    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0.1], size_3d=[1, 1, 1], mean_depth=0.1, operation="non_operable"),
        _node("b", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0.3], size_3d=[1, 1, 1], mean_depth=0.3),
    ]

    dag = derive_blocking_dag(nodes, [_edge("a", "b", "occluding")])

    assert dag == []


def test_blocking_cascade_not_transitively_closed():
    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0], size_3d=[1, 1, 1], mean_depth=0.1),
        _node("b", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0.1, 0], size_3d=[1, 1, 1], mean_depth=0.2),
        _node("c", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0.2, 0], size_3d=[1, 1, 1], mean_depth=0.3),
    ]

    dag = derive_blocking_dag(nodes, [_edge("a", "b", "stacked-on"), _edge("b", "c", "stacked-on")])

    assert ("a", "b") in dag
    assert ("b", "c") in dag
    assert ("a", "c") not in dag


def test_blocking_dag_no_cycle_in_normal_case():
    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0], size_3d=[1, 1, 1], mean_depth=0.1),
        _node("b", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0.1, 0], size_3d=[1, 1, 1], mean_depth=0.2),
    ]

    dag = derive_blocking_dag(nodes, [_edge("a", "b", "stacked-on")])

    assert dag == [("a", "b")]


def test_blocking_dag_cycle_warning_and_break(caplog):
    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0], size_3d=[1, 1, 1], mean_depth=0.1),
        _node("b", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0.1, 0], size_3d=[1, 1, 1], mean_depth=0.2),
    ]

    with caplog.at_level(logging.WARNING):
        dag = derive_blocking_dag(nodes, [_edge("a", "b", "stacked-on"), _edge("b", "a", "stacked-on")])

    assert len(dag) == 1
    assert "cycle" in caplog.text.lower()


def test_base_support_root_never_enters_blocking_dag():
    nodes = [
        _node(
            "table_0",
            [0, 0, 20, 20],
            valid_3d=True,
            center_3d=[0, 0, 0.4],
            size_3d=[1, 1, 1],
            mean_depth=0.4,
            operation="non_operable",
        ),
        _node(
            "mug_0",
            [0, 0, 20, 20],
            valid_3d=True,
            center_3d=[0, 0, 0.2],
            size_3d=[1, 1, 1],
            mean_depth=0.2,
            operation="pick",
        ),
    ]
    nodes[0].role_ids = ["support_surface", "base_support_root"]

    dag = derive_blocking_dag(nodes, [_edge("table_0", "mug_0", "occluding")])

    assert dag == []


def test_actionable_support_surface_can_still_block_when_not_base_root():
    nodes = [
        _node(
            "tray_0",
            [0, 0, 20, 20],
            valid_3d=True,
            center_3d=[0, 0, 0.2],
            size_3d=[1, 1, 1],
            mean_depth=0.2,
            operation="pick",
        ),
        _node(
            "mug_0",
            [0, 0, 20, 20],
            valid_3d=True,
            center_3d=[0, 0, 0.4],
            size_3d=[1, 1, 1],
            mean_depth=0.4,
            operation="pick",
        ),
    ]
    nodes[0].role_ids = ["container", "support_surface"]
    nodes[0].preferred_operation = "pick"

    dag = derive_blocking_dag(nodes, [_edge("tray_0", "mug_0", "occluding")])

    assert dag == [("tray_0", "mug_0")]


def test_populate_node_actionability_keeps_intrinsic_operation_and_sets_runtime_state():
    pen = _node("pen_0", [0, 0, 10, 10], operation="pick")
    card = _node("card_0", [0, 0, 10, 10], operation="push")

    dag = [("card_0", "pen_0")]
    populate_node_actionability([pen, card], dag)

    assert pen.operation == "pick"
    assert pen.blocked_by == ["card_0"]
    assert pen.directly_reachable is False
    assert card.operation == "push"
    assert card.blocking == ["pen_0"]
    assert card.directly_reachable is True
