from apps.data_flywheel.types import (
    InstanceAttributes,
    SceneEdge,
    SceneGraph,
    SceneNode,
    SupportRelation,
)


def _make_node(node_id, cx, cy, cz, sx, sy, sz, op="pick", name=None):
    return SceneNode(
        id=node_id,
        class_name=node_id.split("_")[0],
        bbox=[0.0, 0.0, 100.0, 100.0],
        center_3d=[cx, cy, cz],
        size_3d=[sx, sy, sz],
        mean_depth=cz,
        valid_3d=True,
        operation=op,
        hierarchical_name=name or f"the {node_id}",
        attributes=InstanceAttributes(),
    )


def test_render_three_scene_contains_threejs_code():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    scene_graph = SceneGraph(
        image_id="test",
        image_path="test.jpg",
        image_size=(640, 480),
        nodes=[_make_node("pen_0", 0.1, 0.2, 0.5, 0.05, 0.01, 0.15)],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_three_scene(scene_graph)

    assert "THREE.BoxGeometry" in html or "BoxGeometry" in html
    assert "OrbitControls" in html
    assert "pen_0" in html
    assert "&quot;" not in html
    assert "if (!container) return;" not in html


def test_render_three_scene_skips_invalid_3d():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    node = SceneNode(
        id="cup_0",
        class_name="cup",
        bbox=[0.0, 0.0, 50.0, 50.0],
        valid_3d=False,
        attributes=InstanceAttributes(),
    )
    scene_graph = SceneGraph(
        image_id="t",
        image_path="t.jpg",
        image_size=(640, 480),
        nodes=[node],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_three_scene(scene_graph)

    assert "No 3D data" in html or "无 3D 数据" in html


def test_render_three_scene_relation_lines():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    node_a = _make_node("pen_0", 0.1, 0.1, 0.5, 0.05, 0.01, 0.15)
    node_b = _make_node("card_0", 0.1, 0.1, 0.3, 0.1, 0.001, 0.08, op="push")
    edge = SceneEdge(
        subject="card_0",
        object="pen_0",
        predicate="stacked-on",
        source="3d",
        confidence=0.9,
    )
    scene_graph = SceneGraph(
        image_id="t",
        image_path="t.jpg",
        image_size=(640, 480),
        nodes=[node_a, node_b],
        edges=[edge],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_three_scene(scene_graph)

    assert "LineBasicMaterial" in html or "new THREE.Line" in html


def test_render_three_scene_blocking_dag_arrows():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    node_a = _make_node("pen_0", 0.1, 0.1, 0.5, 0.05, 0.01, 0.15)
    node_b = _make_node("card_0", 0.1, 0.1, 0.3, 0.1, 0.001, 0.08, op="push")
    scene_graph = SceneGraph(
        image_id="t",
        image_path="t.jpg",
        image_size=(640, 480),
        nodes=[node_a, node_b],
        edges=[],
        ternary_relations=[],
        blocking_dag=[("card_0", "pen_0")],
    )

    html = render_three_scene(scene_graph)

    assert "blocking" in html.lower() or "ArrowHelper" in html or "arrow" in html.lower()


def test_render_three_scene_has_hover_and_details_interaction():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    scene_graph = SceneGraph(
        image_id="interactive",
        image_path="interactive.jpg",
        image_size=(640, 480),
        nodes=[_make_node("pen_0", 0.1, 0.2, 0.5, 0.05, 0.01, 0.15)],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_three_scene(scene_graph)

    assert "Raycaster" in html
    assert "mousemove" in html
    assert "click" in html
    assert "details-panel" in html
    assert "tooltip" in html


def test_render_three_scene_normalizes_large_coordinates_and_autoframes():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    scene_graph = SceneGraph(
        image_id="big_scene",
        image_path="big.jpg",
        image_size=(1280, 720),
        nodes=[
            _make_node("table_0", 40.0, -271.0, 4454.0, 12708.0, 4959.0, 11780.0, op="non_operable", name="table"),
            _make_node("bowl_0", -860.0, -1628.0, 4840.0, 2940.0, 6595.0, 19520.0, op="pick", name="bowl"),
        ],
        edges=[
            SceneEdge(subject="bowl_0", object="table_0", predicate="on-top-of", source="3d", confidence=0.9),
        ],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_three_scene(scene_graph)

    assert "render_center" in html
    assert "render_size" in html
    assert "sceneRadius" in html
    assert "cameraDistance" in html


def test_render_three_scene_includes_point_cloud_and_box_toggle_when_available():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    node = SceneNode(
        id="mug_0",
        class_name="mug",
        bbox=[0.0, 0.0, 100.0, 100.0],
        center_3d=[0.1, 0.2, 0.5],
        size_3d=[0.08, 0.08, 0.12],
        mean_depth=0.5,
        valid_3d=True,
        operation="pick",
        hierarchical_name="the mug",
        point_cloud_sample=[[0.1, 0.2, 0.5], [0.11, 0.2, 0.51]],
        attributes=InstanceAttributes(),
    )
    scene_graph = SceneGraph(
        image_id="pc",
        image_path="pc.jpg",
        image_size=(640, 480),
        nodes=[node],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_three_scene(scene_graph)

    assert "Show boxes" in html
    assert "point_cloud_sample" in html
    assert "PointsMaterial" in html or "THREE.Points" in html


def test_render_three_scene_emphasizes_point_cloud_first_layout():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    node = SceneNode(
        id="mug_0",
        class_name="mug",
        bbox=[0.0, 0.0, 100.0, 100.0],
        center_3d=[0.1, 0.2, 0.5],
        size_3d=[0.08, 0.08, 0.12],
        mean_depth=0.5,
        valid_3d=True,
        operation="pick",
        hierarchical_name="the mug",
        point_cloud_sample=[[0.1, 0.2, 0.5], [0.11, 0.2, 0.51], [0.09, 0.19, 0.49]],
        attributes=InstanceAttributes(),
    )
    scene_graph = SceneGraph(
        image_id="pc-layout",
        image_path="pc-layout.jpg",
        image_size=(640, 480),
        nodes=[node],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_three_scene(scene_graph)

    assert "Point-Cloud-First" in html
    assert "Show global cloud" in html
    assert "Global scene cloud" in html
    assert "Boxes are debug overlays only" in html


def test_render_three_scene_accepts_point_cloud_only_nodes():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    node = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[0.0, 0.0, 100.0, 100.0],
        valid_3d=False,
        operation="pick",
        hierarchical_name="the pen",
        point_cloud_sample=[[0.0, 0.0, 0.4], [0.02, 0.0, 0.42], [0.01, 0.01, 0.41]],
        attributes=InstanceAttributes(),
    )
    scene_graph = SceneGraph(
        image_id="pc-only",
        image_path="pc-only.jpg",
        image_size=(640, 480),
        nodes=[node],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_three_scene(scene_graph)

    assert "No 3D data" not in html
    assert "point_cloud_sample" in html
    assert "PointsMaterial" in html or "THREE.Points" in html


def test_render_three_scene_selection_payload_includes_support_and_blocking_neighbors():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    table = _make_node("table_0", 0.0, 0.0, 1.0, 1.0, 1.0, 0.1, op="push", name="the table")
    pen = _make_node("pen_0", 0.1, 0.1, 0.4, 0.05, 0.01, 0.12, op="pick", name="the pen")
    pen.support_parent = "table_0"
    pen.support_type = "on-top-of"
    card = _make_node("card_0", 0.1, 0.1, 0.3, 0.1, 0.001, 0.08, op="push", name="the card")
    scene_graph = SceneGraph(
        image_id="neighbors",
        image_path="neighbors.jpg",
        image_size=(640, 480),
        nodes=[table, pen, card],
        edges=[],
        ternary_relations=[],
        blocking_dag=[("card_0", "pen_0")],
        support_relations=[
            SupportRelation(
                subject="pen_0",
                object="table_0",
                relation="on-top-of",
                confidence=0.9,
            )
        ],
    )

    html = render_three_scene(scene_graph)

    assert "support_parent" in html
    assert "support_relations" in html
    assert "neighborIds" in html or "supportNeighbors" in html


def test_render_three_scene_point_cloud_labels_attach_to_visible_anchor():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    node = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[0.0, 0.0, 100.0, 100.0],
        valid_3d=False,
        operation="pick",
        hierarchical_name="the pen",
        point_cloud_sample=[[0.0, 0.0, 0.4], [0.02, 0.0, 0.42], [0.01, 0.01, 0.41]],
        attributes=InstanceAttributes(),
    )
    scene_graph = SceneGraph(
        image_id="pc-label",
        image_path="pc-label.jpg",
        image_size=(640, 480),
        nodes=[node],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_three_scene(scene_graph)

    assert "box.add(label)" not in html
    assert "labelAnchor = points" not in html
    assert "new THREE.Object3D()" in html
    assert "labelAnchor.position.set(center[0], center[1], center[2])" in html
    assert "labelAnchor.add(label)" in html


def test_render_three_scene_uses_normalized_point_cloud_payload():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    node = SceneNode(
        id="mug_0",
        class_name="mug",
        bbox=[0.0, 0.0, 100.0, 100.0],
        center_3d=[1200.0, -800.0, 4200.0],
        size_3d=[400.0, 500.0, 600.0],
        mean_depth=4200.0,
        valid_3d=True,
        operation="pick",
        hierarchical_name="the mug",
        point_cloud_sample=[[1180.0, -810.0, 4180.0], [1220.0, -790.0, 4220.0]],
        attributes=InstanceAttributes(),
    )
    scene_graph = SceneGraph(
        image_id="pc-normalized",
        image_path="pc-normalized.jpg",
        image_size=(640, 480),
        nodes=[node],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_three_scene(scene_graph)

    assert "render_point_cloud_sample" in html
    assert "node.render_point_cloud_sample" in html
    assert "node.point_cloud_sample.flat()" not in html


def test_render_three_scene_includes_muted_global_point_cloud_layer():
    from apps.data_flywheel.visualization.three_scene import render_three_scene

    node_a = SceneNode(
        id="cup_0",
        class_name="cup",
        bbox=[0.0, 0.0, 100.0, 100.0],
        center_3d=[0.0, 0.0, 1.0],
        size_3d=[0.2, 0.2, 0.2],
        mean_depth=1.0,
        valid_3d=True,
        operation="pick",
        hierarchical_name="the cup",
        point_cloud_sample=[[0.0, 0.0, 1.0], [0.05, 0.02, 1.01]],
        attributes=InstanceAttributes(),
    )
    node_b = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[0.0, 0.0, 100.0, 100.0],
        center_3d=[0.3, 0.1, 0.9],
        size_3d=[0.15, 0.03, 0.03],
        mean_depth=0.9,
        valid_3d=True,
        operation="pick",
        hierarchical_name="the pen",
        point_cloud_sample=[[0.3, 0.1, 0.9], [0.28, 0.11, 0.92]],
        attributes=InstanceAttributes(),
    )
    scene_graph = SceneGraph(
        image_id="global-pc",
        image_path="global-pc.jpg",
        image_size=(640, 480),
        nodes=[node_a, node_b],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_three_scene(scene_graph)

    assert "global_point_cloud_sample" in html
    assert "globalPointsMaterial" in html
