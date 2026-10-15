from apps.data_flywheel.scene_understanding.support import compute_support_relations
from apps.data_flywheel.types import SceneNode


def _node(node_id: str, bbox, *, class_name="obj", role_ids=None, preferred_operation=None, point_cloud_sample=None):
    return SceneNode(
        id=node_id,
        class_name=class_name,
        bbox=bbox,
        role_ids=list(role_ids or []),
        preferred_operation=preferred_operation,
        point_cloud_sample=[list(point) for point in (point_cloud_sample or [])],
    )


def test_support_graph_marks_table_as_base_support_root_and_keeps_it_out_of_blocking():
    table = _node(
        "table_0",
        [0, 200, 400, 400],
        class_name="table",
        role_ids=["support_surface", "base_support_root"],
        preferred_operation="non_operable",
        point_cloud_sample=[[0.0, 0.0, 1.0]],
    )
    tray = _node(
        "tray_0",
        [100, 180, 220, 260],
        class_name="tray",
        role_ids=["container", "support_surface"],
        preferred_operation="pick",
        point_cloud_sample=[[0.0, 0.0, 0.8]],
    )
    mug = _node(
        "mug_0",
        [120, 160, 170, 220],
        class_name="mug",
        role_ids=[],
        preferred_operation="pick",
        point_cloud_sample=[[0.0, 0.0, 0.7]],
    )

    support_relations = compute_support_relations([table, tray, mug])

    assert any(rel.subject == "tray_0" and rel.object == "table_0" for rel in support_relations)
    assert any(rel.subject == "mug_0" and rel.object == "tray_0" for rel in support_relations)
    assert tray.support_parent == "table_0"
    assert tray.support_type == "on-top-of"
    assert tray.support_excluded is True
    assert table.excluded_from_action_graph is True
    assert mug.support_parent == "tray_0"
    assert mug.support_type == "inside"
    assert mug.support_excluded is False


def test_support_graph_does_not_create_mutual_inside_cycle_for_overlapping_containers():
    outer_box = _node(
        "box_0",
        [10, 10, 80, 80],
        class_name="box",
        role_ids=["container"],
        preferred_operation="pick",
        point_cloud_sample=[[0.0, 0.0, 0.8]],
    )
    inner_cup = _node(
        "cup_0",
        [25, 25, 55, 55],
        class_name="cup",
        role_ids=["container"],
        preferred_operation="pick",
        point_cloud_sample=[[0.0, 0.0, 0.7]],
    )

    support_relations = compute_support_relations([outer_box, inner_cup])

    assert any(rel.subject == "cup_0" and rel.object == "box_0" and rel.relation == "inside" for rel in support_relations)
    assert not any(rel.subject == "box_0" and rel.object == "cup_0" for rel in support_relations)
    assert outer_box.support_parent is None
    assert inner_cup.support_parent == "box_0"


def test_support_graph_rejects_ambiguous_duplicate_container_support():
    box = _node(
        "box_0",
        [10, 10, 80, 80],
        class_name="box",
        role_ids=["container"],
        preferred_operation="pick",
        point_cloud_sample=[[0.0, 0.0, 0.8]],
    )
    alias_box = _node(
        "box_1",
        [10, 10, 80, 80],
        class_name="box",
        role_ids=["container"],
        preferred_operation="pick",
        point_cloud_sample=[[0.0, 0.0, 0.8]],
    )

    support_relations = compute_support_relations([box, alias_box])

    assert support_relations == []
    assert box.support_parent is None
    assert box.support_type is None
    assert alias_box.support_parent is None
    assert alias_box.support_type is None
