from apps.data_flywheel.training_data.phase_a.blocking import _topological_sort_blockers
from apps.data_flywheel.types import SceneGraph, SceneNode


def _node(
    node_id: str,
    *,
    class_name: str = "obj",
    x_center: float = 50.0,
    operation: str | None = None,
) -> SceneNode:
    half_width = 5.0
    return SceneNode(
        id=node_id,
        class_name=class_name,
        bbox=[x_center - half_width, 0.0, x_center + half_width, 10.0],
        operation=operation,
    )


def _graph(nodes: list[SceneNode], blocking_dag: list[tuple[str, str]]) -> SceneGraph:
    return SceneGraph(
        image_id="img-1",
        image_path="img-1.jpg",
        image_size=(200, 100),
        nodes=nodes,
        edges=[],
        blocking_dag=blocking_dag,
    )


def _node_map(nodes: list[SceneNode]) -> dict[str, SceneNode]:
    return {node.id: node for node in nodes}


def test_single_push_blocker():
    nodes = [
        _node("target_0", class_name="pen", x_center=80.0, operation="pick"),
        _node("box_0", class_name="box", x_center=30.0, operation="push"),
    ]
    scene_graph = _graph(nodes, [("box_0", "target_0")])

    ordered = _topological_sort_blockers(["box_0"], scene_graph, set())

    assert ordered == ["box_0"]
    assert _node_map(nodes)[ordered[0]].operation == "push"


def test_single_pick_blocker():
    nodes = [
        _node("target_0", class_name="pen", x_center=80.0, operation="pick"),
        _node("book_0", class_name="book", x_center=30.0, operation="pick"),
    ]
    scene_graph = _graph(nodes, [("book_0", "target_0")])

    ordered = _topological_sort_blockers(["book_0"], scene_graph, set())

    assert ordered == ["book_0"]
    assert _node_map(nodes)[ordered[0]].operation == "pick"


def test_cascade_stacked_three_layers():
    nodes = [
        _node("target_0", class_name="pen", x_center=80.0, operation="pick"),
        _node("book_0", class_name="book", x_center=50.0, operation="pick"),
        _node("card_0", class_name="card", x_center=20.0, operation="push"),
    ]
    scene_graph = _graph(
        nodes,
        [
            ("book_0", "target_0"),
            ("card_0", "book_0"),
        ],
    )

    ordered = _topological_sort_blockers(["book_0"], scene_graph, set())

    assert ordered == ["card_0", "book_0"]


def test_non_operable_blocker_fails():
    nodes = [
        _node("target_0", class_name="pen", x_center=80.0, operation="pick"),
        _node("sealed_box_0", class_name="box", x_center=30.0, operation="non_operable"),
    ]
    scene_graph = _graph(nodes, [("sealed_box_0", "target_0")])

    ordered = _topological_sort_blockers(["sealed_box_0"], scene_graph, set())

    assert ordered == ["sealed_box_0"]
    assert _node_map(nodes)[ordered[0]].operation == "non_operable"


def test_card_blocker_uses_push():
    nodes = [
        _node("target_0", class_name="pen", x_center=80.0, operation="pick"),
        _node("card_0", class_name="card", x_center=30.0, operation="push"),
    ]
    scene_graph = _graph(nodes, [("card_0", "target_0")])

    ordered = _topological_sort_blockers(["card_0"], scene_graph, set())

    assert ordered == ["card_0"]
    assert _node_map(nodes)[ordered[0]].operation == "push"


def test_multiple_blockers_order_by_x():
    nodes = [
        _node("target_0", class_name="pen", x_center=80.0, operation="pick"),
        _node("right_box_0", class_name="box", x_center=70.0, operation="pick"),
        _node("left_card_0", class_name="card", x_center=20.0, operation="push"),
    ]
    scene_graph = _graph(
        nodes,
        [
            ("left_card_0", "target_0"),
            ("right_box_0", "target_0"),
        ],
    )

    ordered = _topological_sort_blockers(["right_box_0", "left_card_0"], scene_graph, set())

    assert ordered == ["left_card_0", "right_box_0"]
