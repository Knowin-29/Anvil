from apps.data_flywheel.scene_understanding.thresholds import SceneGraphThresholds
from apps.data_flywheel.types import SceneNode


def _node(node_id: str, bbox, *, valid_3d=False, center_3d=None, size_3d=None, mean_depth=None, is_container=False, operation=None):
    return SceneNode(
        id=node_id,
        class_name="obj",
        bbox=bbox,
        valid_3d=valid_3d,
        center_3d=center_3d,
        size_3d=size_3d,
        mean_depth=mean_depth,
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


def test_on_top_of_3d():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0.0, 0], size_3d=[1, 1, 1], mean_depth=0.2),
        _node("b", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0.2, 0], size_3d=[1, 1, 1], mean_depth=0.4),
    ]

    edges = compute_3d_relations(nodes, (100, 100), SceneGraphThresholds())

    edge = _find_edge(edges, "on-top-of", "a", "b")
    assert edge is not None
    assert edge.source == "3d"
    assert edge.frame == "world"


def test_under_3d():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0.2, 0], size_3d=[1, 1, 1], mean_depth=0.4),
        _node("b", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0.0, 0], size_3d=[1, 1, 1], mean_depth=0.2),
    ]

    edges = compute_3d_relations(nodes, (100, 100), SceneGraphThresholds())

    edge = _find_edge(edges, "under", "a", "b")
    assert edge is not None
    assert edge.source == "3d"
    assert edge.frame == "world"


def test_inside_container_3d():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0], size_3d=[0.2, 0.2, 0.2], mean_depth=0.2),
        _node("b", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0], size_3d=[1, 1, 1], mean_depth=0.3, is_container=True),
    ]

    edges = compute_3d_relations(nodes, (100, 100), SceneGraphThresholds())

    edge = _find_edge(edges, "inside", "a", "b")
    assert edge is not None
    assert edge.source == "3d"
    assert edge.frame == "world"


def test_front_behind_depth():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0.1], size_3d=[1, 1, 1], mean_depth=0.1),
        _node("b", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0, 0.2], size_3d=[1, 1, 1], mean_depth=0.2),
    ]

    edges = compute_3d_relations(nodes, (100, 100), SceneGraphThresholds())

    assert _find_edge(edges, "front", "a", "b") is not None
    assert _find_edge(edges, "behind", "b", "a") is not None
    assert _find_edge(edges, "front", "a", "b").frame == "world"


def test_above_world_and_below_world_exported_separately():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0.0, 0.1], size_3d=[1, 1, 1], mean_depth=0.1),
        _node("b", [0, 0, 10, 10], valid_3d=True, center_3d=[0, 0.3, 0.1], size_3d=[1, 1, 1], mean_depth=0.1),
    ]

    edges = compute_3d_relations(nodes, (100, 100), SceneGraphThresholds())

    assert _find_edge(edges, "above-world", "a", "b") is not None
    assert _find_edge(edges, "below-world", "b", "a") is not None


def test_occluding_by_depth_diff():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    nodes = [
        _node("a", [0, 0, 20, 20], valid_3d=True, center_3d=[0, 0, 0.1], size_3d=[1, 1, 1], mean_depth=0.1),
        _node("b", [10, 10, 30, 30], valid_3d=True, center_3d=[0, 0, 0.4], size_3d=[1, 1, 1], mean_depth=0.4),
    ]

    edges = compute_3d_relations(nodes, (100, 100), SceneGraphThresholds())

    assert _find_edge(edges, "occluding", "a", "b") is not None
    assert _find_edge(edges, "occluded-by", "b", "a") is not None
    assert _find_edge(edges, "occluding", "a", "b").frame == "world"


def test_stacked_on_3d():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    top = SceneNode(
        id="a",
        class_name="obj",
        bbox=[0, 0, 20, 20],
        support_parent="b",
        support_type="on-top-of",
        point_cloud_sample=[[0.0, 0.0, 0.0], [0.2, 0.02, 0.2]],
    )
    bottom = SceneNode(
        id="b",
        class_name="obj",
        bbox=[0, 0, 20, 20],
        role_ids=["support_surface"],
        point_cloud_sample=[[0.0, 0.08, 0.0], [0.2, 0.1, 0.2]],
    )

    edges = compute_3d_relations([top, bottom], (100, 100), SceneGraphThresholds())

    assert _find_edge(edges, "stacked-on", "a", "b") is not None
    assert _find_edge(edges, "stacked-on", "a", "b").frame == "world"


def test_supporting_is_inverse_of_stacked_on():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    top = SceneNode(
        id="a",
        class_name="obj",
        bbox=[0, 0, 20, 20],
        support_parent="b",
        support_type="on-top-of",
        point_cloud_sample=[[0.0, 0.0, 0.0], [0.2, 0.02, 0.2]],
    )
    bottom = SceneNode(
        id="b",
        class_name="obj",
        bbox=[0, 0, 20, 20],
        role_ids=["support_surface"],
        point_cloud_sample=[[0.0, 0.08, 0.0], [0.2, 0.1, 0.2]],
    )

    edges = compute_3d_relations([top, bottom], (100, 100), SceneGraphThresholds())

    assert _find_edge(edges, "supporting", "b", "a") is not None
    assert _find_edge(edges, "supporting", "b", "a").frame == "world"


def test_no_3d_relation_when_valid_3d_false():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    nodes = [
        _node("a", [0, 0, 10, 10], valid_3d=False),
        _node("b", [0, 0, 10, 10], valid_3d=False),
    ]

    edges = compute_3d_relations(nodes, (100, 100), SceneGraphThresholds())

    assert edges == []


def test_image_above_and_world_stacked_are_distinct():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

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

    edges = compute_3d_relations([top_card, lower_box], (100, 100), SceneGraphThresholds())

    assert any(edge.predicate == "stacked-on" and edge.frame == "world" for edge in edges)


def test_degenerate_point_cloud_falls_back_to_legacy_pose():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    top = SceneNode(
        id="top",
        class_name="obj",
        bbox=[0, 0, 20, 20],
        valid_3d=True,
        center_3d=[0.0, 0.0, 0.2],
        size_3d=[1.0, 0.1, 1.0],
        mean_depth=0.2,
        point_cloud_sample=[[0.0, 0.3, 1.0]],
    )
    bottom = SceneNode(
        id="bottom",
        class_name="obj",
        bbox=[0, 0, 20, 20],
        valid_3d=True,
        center_3d=[0.0, 0.09, 0.3],
        size_3d=[1.0, 0.1, 1.0],
        mean_depth=0.3,
        point_cloud_sample=[[0.0, 0.0, 0.1]],
    )

    edges = compute_3d_relations([top, bottom], (100, 100), SceneGraphThresholds())

    assert _find_edge(edges, "stacked-on", "top", "bottom") is not None
    assert _find_edge(edges, "stacked-on", "bottom", "top") is None


def test_base_support_root_is_excluded_from_pairwise_3d_exports():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    table = SceneNode(
        id="table_0",
        class_name="table",
        bbox=[0, 70, 100, 100],
        role_ids=["support_surface", "base_support_root"],
        point_cloud_sample=[[0.0, 0.1, 0.6], [1.0, 0.1, 0.6]],
    )
    pen = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[10, 50, 20, 60],
        point_cloud_sample=[[0.1, 0.0, 0.5], [0.2, 0.0, 0.5]],
    )

    edges = compute_3d_relations([table, pen], (100, 100), SceneGraphThresholds())

    assert edges == []


def test_bbox_overlap_does_not_invent_world_support_without_vertical_3d_evidence():
    from apps.data_flywheel.scene_understanding.relations_3d import compute_3d_relations

    upper_in_image = _node(
        "a",
        [10, 10, 40, 30],
        valid_3d=True,
        center_3d=[0.0, 0.0, 0.3],
        size_3d=[1.0, 0.1, 1.0],
        mean_depth=0.3,
    )
    lower_in_image = _node(
        "b",
        [10, 35, 40, 55],
        valid_3d=True,
        center_3d=[0.0, 0.0, 0.31],
        size_3d=[1.0, 0.1, 1.0],
        mean_depth=0.31,
    )

    edges = compute_3d_relations([upper_in_image, lower_in_image], (100, 100), SceneGraphThresholds())

    assert _find_edge(edges, "on-top-of", "a", "b") is None
    assert _find_edge(edges, "under", "b", "a") is None
