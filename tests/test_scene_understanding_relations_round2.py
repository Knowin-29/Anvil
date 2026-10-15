from apps.data_flywheel.scene_understanding.relations_round2 import (
    compute_between_relations,
    compute_round2_binary_relations,
)
from apps.data_flywheel.scene_understanding.thresholds import SceneGraphThresholds
from apps.data_flywheel.types import SceneNode


def _node(
    node_id: str,
    class_name: str,
    bbox,
    *,
    valid_3d: bool = False,
    center_3d=None,
    size_3d=None,
    mean_depth=None,
    is_container: bool = False,
) -> SceneNode:
    return SceneNode(
        id=node_id,
        class_name=class_name,
        bbox=bbox,
        valid_3d=valid_3d,
        center_3d=center_3d,
        size_3d=size_3d,
        mean_depth=mean_depth,
        is_container=is_container,
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


def _find_ternary(relations, subject, reference_a, reference_b):
    for rel in relations:
        if (
            rel.subject == subject
            and rel.reference_a == reference_a
            and rel.reference_b == reference_b
        ):
            return rel
    return None


def test_touching_adjacent_books():
    nodes = [
        _node("book_a", "book", [0, 0, 20, 20]),
        _node("book_b", "book", [21, 0, 41, 20]),
    ]

    edges = compute_round2_binary_relations(nodes, (100, 100), SceneGraphThresholds())

    edge = _find_edge(edges, "touching", "book_a", "book_b")
    assert edge is not None
    assert edge.source == "2d"
    assert edge.confidence == 0.85
    assert edge.frame == "image"


def test_separated_with_gap():
    nodes = [
        _node("book_a", "book", [0, 0, 20, 20]),
        _node("book_b", "book", [50, 0, 70, 20]),
    ]

    edges = compute_round2_binary_relations(nodes, (100, 100), SceneGraphThresholds())

    edge = _find_edge(edges, "separated", "book_a", "book_b")
    assert edge is not None
    assert edge.source == "2d"
    assert edge.confidence == 0.7
    assert edge.frame == "image"


def test_aligned_with_three_pens_in_row():
    nodes = [
        _node("pen_a", "pen", [0, 0, 10, 10]),
        _node("pen_b", "pen", [20, 0, 30, 10]),
        _node("pen_c", "pen", [40, 0, 50, 10]),
    ]

    edges = compute_round2_binary_relations(nodes, (100, 100), SceneGraphThresholds())

    assert _find_edge(edges, "aligned-with", "pen_a", "pen_b") is not None
    assert _find_edge(edges, "aligned-with", "pen_a", "pen_c") is not None
    assert _find_edge(edges, "aligned-with", "pen_b", "pen_c") is not None


def test_aligned_with_requires_same_class():
    nodes = [
        _node("pen_a", "pen", [0, 0, 10, 10]),
        _node("book_b", "book", [20, 0, 30, 10]),
        _node("pen_c", "pen", [40, 0, 50, 10]),
    ]

    edges = compute_round2_binary_relations(nodes, (100, 100), SceneGraphThresholds())

    assert _find_edge(edges, "aligned-with", "pen_a", "book_b") is None
    assert _find_edge(edges, "aligned-with", "book_b", "pen_c") is None


def test_outside_requires_is_container():
    nodes = [
        _node("cup_a", "cup", [0, 0, 30, 30], valid_3d=True, center_3d=[0, 0, 0], size_3d=[10, 10, 10], mean_depth=1.0, is_container=True),
        _node("pen_b", "pen", [100, 100, 120, 120], valid_3d=True, center_3d=[100, 100, 100], size_3d=[5, 5, 5], mean_depth=2.0),
    ]

    edges = compute_round2_binary_relations(nodes, (200, 200), SceneGraphThresholds())

    edge = _find_edge(edges, "outside", "pen_b", "cup_a")
    assert edge is not None
    assert edge.source == "derived"
    assert edge.confidence == 0.75
    assert edge.frame == "world"


def test_between_three_nodes_inline():
    nodes = [
        _node("left", "pen", [0, 0, 10, 10]),
        _node("mid", "pen", [20, 0, 30, 10]),
        _node("right", "pen", [40, 0, 50, 10]),
    ]

    relations = compute_between_relations(nodes, SceneGraphThresholds())

    rel = _find_ternary(relations, "mid", "left", "right")
    assert rel is not None
    assert rel.predicate == "between"
    assert rel.confidence == 0.75


def test_between_deduplication():
    nodes = [
        _node("right", "pen", [40, 0, 50, 10]),
        _node("mid", "pen", [20, 0, 30, 10]),
        _node("left", "pen", [0, 0, 10, 10]),
    ]

    relations = compute_between_relations(nodes, SceneGraphThresholds())

    assert len(relations) == 1
    assert _find_ternary(relations, "mid", "left", "right") is not None


def test_between_ignores_reverse_order():
    nodes = [
        _node("right", "pen", [40, 0, 50, 10]),
        _node("mid", "pen", [20, 0, 30, 10]),
        _node("left", "pen", [0, 0, 10, 10]),
    ]

    relations = compute_between_relations(nodes, SceneGraphThresholds())

    assert _find_ternary(relations, "mid", "right", "left") is None


def test_touching_uses_point_cloud_only_depth_validation():
    nodes = [
        _node("book_a", "book", [0, 0, 20, 20], mean_depth=None),
        _node("book_b", "book", [21, 0, 41, 20], mean_depth=None),
    ]
    nodes[0].point_cloud_sample = [[0.0, 0.0, 0.5]]
    nodes[1].point_cloud_sample = [[0.0, 0.0, 0.505]]

    edges = compute_round2_binary_relations(nodes, (100, 100), SceneGraphThresholds())

    edge = _find_edge(edges, "touching", "book_a", "book_b")
    assert edge is not None
    assert edge.source == "2d+3d"
    assert edge.frame == "image"


def test_outside_uses_point_cloud_only_world_frame():
    nodes = [
        _node("cup_a", "cup", [0, 0, 30, 30], is_container=True),
        _node("pen_b", "pen", [100, 100, 120, 120]),
    ]
    nodes[0].point_cloud_sample = [[-1.0, -1.0, 0.9], [1.0, 1.0, 1.1]]
    nodes[1].point_cloud_sample = [[5.0, 5.0, 2.0]]

    edges = compute_round2_binary_relations(nodes, (200, 200), SceneGraphThresholds())

    edge = _find_edge(edges, "outside", "pen_b", "cup_a")
    assert edge is not None
    assert edge.frame == "world"
