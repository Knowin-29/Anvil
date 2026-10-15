from apps.data_flywheel.scene_understanding.relations_2d import compute_2d_relations
from apps.data_flywheel.scene_understanding.thresholds import SceneGraphThresholds
from apps.data_flywheel.types import SceneNode


def _node(node_id: str, bbox):
    return SceneNode(id=node_id, class_name="obj", bbox=bbox)


def _find_edge(edges, predicate):
    for edge in edges:
        if edge.predicate == predicate:
            return edge
    return None


def test_left_of_relation():
    nodes = [
        _node("a", [0, 10, 20, 30]),
        _node("b", [60, 10, 80, 30]),
    ]

    edges = compute_2d_relations(nodes, (100, 100), SceneGraphThresholds())

    left = _find_edge(edges, "left-of")
    assert left is not None
    assert left.subject == "a"
    assert left.object == "b"
    assert left.source == "2d"
    assert left.confidence == 0.9
    assert left.frame == "image"
    assert _find_edge(edges, "right-of") is None


def test_right_of_relation():
    nodes = [
        _node("a", [60, 10, 80, 30]),
        _node("b", [0, 10, 20, 30]),
    ]

    edges = compute_2d_relations(nodes, (100, 100), SceneGraphThresholds())

    right = _find_edge(edges, "right-of")
    assert right is not None
    assert right.subject == "a"
    assert right.object == "b"
    assert right.source == "2d"
    assert right.confidence == 0.9
    assert right.frame == "image"
    assert _find_edge(edges, "left-of") is None


def test_above_below_relation():
    above_nodes = [
        _node("a", [10, 0, 30, 20]),
        _node("b", [10, 60, 30, 80]),
    ]
    below_nodes = [
        _node("a", [10, 60, 30, 80]),
        _node("b", [10, 0, 30, 20]),
    ]

    above_edges = compute_2d_relations(above_nodes, (100, 100), SceneGraphThresholds())
    below_edges = compute_2d_relations(below_nodes, (100, 100), SceneGraphThresholds())

    above = _find_edge(above_edges, "above-image")
    assert above is not None
    assert above.subject == "a"
    assert above.object == "b"
    assert above.source == "2d"
    assert above.frame == "image"

    below = _find_edge(below_edges, "below-image")
    assert below is not None
    assert below.subject == "a"
    assert below.object == "b"
    assert below.source == "2d"
    assert below.frame == "image"


def test_next_to_adjacent_objects():
    nodes = [
        _node("a", [10, 10, 30, 30]),
        _node("b", [31, 10, 51, 30]),
    ]

    edges = compute_2d_relations(nodes, (100, 100), SceneGraphThresholds())

    edge = _find_edge(edges, "next-to")
    assert edge is not None
    assert edge.subject == "a"
    assert edge.object == "b"
    assert edge.source == "2d"
    assert edge.confidence == 0.85
    assert edge.frame == "image"


def test_near_within_ratio():
    nodes = [
        _node("a", [10, 10, 20, 20]),
        _node("b", [45, 10, 55, 20]),
    ]

    edges = compute_2d_relations(nodes, (200, 200), SceneGraphThresholds())

    edge = _find_edge(edges, "near")
    assert edge is not None
    assert edge.subject == "a"
    assert edge.object == "b"
    assert edge.source == "2d"
    assert edge.confidence == 0.8
    assert edge.frame == "image"


def test_larger_smaller_than():
    larger_nodes = [
        _node("a", [0, 0, 20, 20]),
        _node("b", [40, 0, 50, 10]),
    ]
    smaller_nodes = [
        _node("a", [40, 0, 50, 10]),
        _node("b", [0, 0, 20, 20]),
    ]

    larger_edges = compute_2d_relations(larger_nodes, (100, 100), SceneGraphThresholds())
    smaller_edges = compute_2d_relations(smaller_nodes, (100, 100), SceneGraphThresholds())

    larger = _find_edge(larger_edges, "larger-than")
    assert larger is not None
    assert larger.subject == "a"
    assert larger.object == "b"
    assert larger.source == "2d"
    assert larger.confidence == 0.9
    assert larger.frame == "image"

    smaller = _find_edge(smaller_edges, "smaller-than")
    assert smaller is not None
    assert smaller.subject == "a"
    assert smaller.object == "b"
    assert smaller.source == "2d"
    assert smaller.confidence == 0.9
    assert smaller.frame == "image"


def test_overlapping_iou_threshold():
    nodes = [
        _node("a", [0, 0, 20, 20]),
        _node("b", [10, 10, 30, 30]),
    ]

    edges = compute_2d_relations(nodes, (100, 100), SceneGraphThresholds())

    edge = _find_edge(edges, "overlapping")
    assert edge is not None
    assert edge.subject == "a"
    assert edge.object == "b"
    assert edge.source == "2d"
    assert edge.confidence == 0.43
    assert edge.frame == "image"


def test_no_relation_when_center_distance_below_min():
    nodes = [
        _node("a", [10, 10, 30, 30]),
        _node("b", [14, 12, 34, 32]),
    ]

    edges = compute_2d_relations(nodes, (100, 100), SceneGraphThresholds())
    predicates = {edge.predicate for edge in edges}

    assert "left-of" not in predicates
    assert "right-of" not in predicates
    assert "above-image" not in predicates
    assert "below-image" not in predicates


def test_image_above_and_world_stacked_are_distinct():
    top_card = SceneNode(
        id="card_0",
        class_name="card",
        bbox=[10, 10, 40, 25],
        point_cloud_sample=[[0.0, 0.02, 0.5]],
    )
    lower_box = SceneNode(
        id="box_0",
        class_name="box",
        bbox=[10, 30, 50, 60],
        role_ids=["support_surface"],
        point_cloud_sample=[[0.0, 0.0, 0.52]],
    )

    edges = compute_2d_relations([top_card, lower_box], (100, 100), SceneGraphThresholds())

    assert any(edge.predicate == "above-image" and edge.frame == "image" for edge in edges)


def test_inside_relation_prefers_mask_topology_for_container():
    apple = SceneNode(
        id="apple_0",
        class_name="apple",
        bbox=[20, 20, 40, 40],
        mask_polygon=[[22, 22], [38, 22], [38, 38], [22, 38]],
        mask_source="mask_polygon",
        has_true_mask=True,
    )
    bowl = SceneNode(
        id="bowl_0",
        class_name="bowl",
        bbox=[10, 10, 50, 50],
        role_ids=["container"],
        mask_polygon=[[10, 10], [50, 10], [50, 50], [10, 50]],
        mask_source="mask_polygon",
        has_true_mask=True,
    )

    edges = compute_2d_relations([apple, bowl], (100, 100), SceneGraphThresholds())

    inside = _find_edge(edges, "inside")
    assert inside is not None
    assert inside.subject == "apple_0"
    assert inside.object == "bowl_0"


def test_base_support_root_is_excluded_from_pairwise_2d_exports():
    table = SceneNode(
        id="table_0",
        class_name="table",
        bbox=[0, 70, 100, 100],
        role_ids=["support_surface", "base_support_root"],
    )
    pen = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[10, 50, 30, 65],
    )

    edges = compute_2d_relations([table, pen], (100, 100), SceneGraphThresholds())

    assert edges == []
