import pytest

from apps.data_flywheel.types import (
    Detection,
    InstanceAttributes,
    PlacementRegion,
    SceneEdge,
    SceneGraph,
    SceneNode,
    Segmentation,
    SupportRelation,
    TernaryRelation,
)


def test_detection_dataclass_fields():
    detection = Detection(
        class_name="pen",
        bbox=[10.0, 20.0, 30.0, 40.0],
        confidence=0.9,
        source="detvlm",
    )

    assert detection.class_name == "pen"
    assert detection.bbox == [10.0, 20.0, 30.0, 40.0]
    assert isinstance(detection.bbox, list)
    assert detection.confidence == 0.9
    assert detection.source == "detvlm"


def test_segmentation_requires_mask_file():
    detection = Detection(
        class_name="cup",
        bbox=[1.0, 2.0, 3.0, 4.0],
        confidence=0.8,
        source="sam3",
    )

    with pytest.raises(TypeError):
        Segmentation(
            detection=detection,
            mask_polygon=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            grasp_point=[2.0, 3.0],
        )


def test_segmentation_mask_polygon_is_list_of_list():
    detection = Detection(
        class_name="box",
        bbox=[1.0, 2.0, 3.0, 4.0],
        confidence=0.95,
        source="sam3",
    )
    segmentation = Segmentation(
        detection=detection,
        mask_polygon=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        mask_file="data/output/masks/scene_box_0.png",
        grasp_point=[2.0, 3.0],
    )

    assert isinstance(segmentation.mask_polygon, list)
    assert all(isinstance(point, list) for point in segmentation.mask_polygon)
    assert segmentation.mask_source == "mask_polygon"


def test_instance_attributes_defaults_are_optional_none():
    attributes = InstanceAttributes()

    assert attributes.color_primary is None
    assert attributes.color_primary_hsv is None
    assert attributes.color_secondary is None
    assert attributes.size is None
    assert attributes.shape is None


def test_scene_node_defaults_follow_shared_contract():
    node = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[10.0, 20.0, 30.0, 40.0],
    )

    assert node.id == "pen_0"
    assert node.class_name == "pen"
    assert node.bbox == [10.0, 20.0, 30.0, 40.0]
    assert node.mask_polygon is None
    assert node.mask_file is None
    assert node.mask_source is None
    assert node.has_true_mask is False
    assert node.grasp_point is None
    assert node.center_3d is None
    assert node.size_3d is None
    assert node.mean_depth is None
    assert node.valid_3d is False
    assert node.operation is None
    assert node.is_container is False
    assert node.container_placeability is None
    assert node.placement_policy is None
    assert node.extra_tags == []
    assert node.canonical_name is None
    assert node.object_class_id is None
    assert node.ontology_status == "unknown"
    assert node.triggered_constraints == []
    assert node.hierarchical_name is None
    assert node.referring_expressions == []
    assert node.role_ids == []
    assert node.allowed_operations == []
    assert node.preferred_operation is None
    assert node.descriptor_axes == []
    assert node.support_parent is None
    assert node.support_type is None
    assert node.support_excluded is False
    assert node.directly_reachable is None
    assert node.blocked_by == []
    assert node.blocking == []
    assert node.excluded_from_action_graph is False
    assert node.point_cloud_sample == []
    assert isinstance(node.attributes, InstanceAttributes)


def test_scene_node_actionability_state_fields_are_available():
    node = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[10.0, 20.0, 30.0, 40.0],
        mask_source="bbox_fallback",
        has_true_mask=False,
        directly_reachable=False,
        blocked_by=["card_0"],
        blocking=["block_0"],
        excluded_from_action_graph=True,
    )

    assert node.mask_source == "bbox_fallback"
    assert node.has_true_mask is False
    assert node.directly_reachable is False
    assert node.blocked_by == ["card_0"]
    assert node.blocking == ["block_0"]
    assert node.excluded_from_action_graph is True


def test_scene_node_mutable_defaults_are_not_shared():
    first = SceneNode(id="cup_0", class_name="cup", bbox=[0.0, 0.0, 1.0, 1.0])
    second = SceneNode(id="cup_1", class_name="cup", bbox=[1.0, 1.0, 2.0, 2.0])

    first.extra_tags.append("fragile")
    first.triggered_constraints.append("size override")
    first.referring_expressions.append("the fragile cup")
    first.role_ids.append("movable")
    first.allowed_operations.append("pick")
    first.descriptor_axes.append("left-right")
    first.blocked_by.append("card_0")
    first.blocking.append("block_0")
    first.point_cloud_sample.append([0.1, 0.2, 0.3])
    first.attributes.color_primary = "red"

    assert second.extra_tags == []
    assert second.triggered_constraints == []
    assert second.referring_expressions == []
    assert second.role_ids == []
    assert second.allowed_operations == []
    assert second.descriptor_axes == []
    assert second.blocked_by == []
    assert second.blocking == []
    assert second.point_cloud_sample == []
    assert second.attributes.color_primary is None


def test_scene_edge_and_ternary_relation_fields():
    edge = SceneEdge(
        subject="pen_0",
        object="tray_0",
        predicate="on",
        source="2d+3d",
        confidence=0.9,
        frame="image",
    )
    ternary = TernaryRelation(
        subject="card_0",
        reference_a="pen_0",
        reference_b="cup_0",
        predicate="between",
        confidence=0.75,
    )

    assert edge.subject == "pen_0"
    assert edge.object == "tray_0"
    assert edge.predicate == "on"
    assert edge.source == "2d+3d"
    assert edge.confidence == 0.9
    assert edge.frame == "image"
    assert ternary.reference_a == "pen_0"
    assert ternary.reference_b == "cup_0"
    assert ternary.predicate == "between"


def test_support_relation_fields():
    relation = SupportRelation(
        subject="mug_0",
        object="table_0",
        relation="on-top-of",
        confidence=0.9,
    )

    assert relation.subject == "mug_0"
    assert relation.object == "table_0"
    assert relation.relation == "on-top-of"
    assert relation.confidence == 0.9


def test_placement_region_defaults_and_fields():
    region = PlacementRegion(
        region_id="container_bowl_0",
        region_type="inside-container",
        support_id="bowl_0",
        anchor_ids=["bowl_0"],
        point_2d=[320.0, 240.0],
        reason="preferred_container",
    )

    assert region.region_id == "container_bowl_0"
    assert region.region_type == "inside-container"
    assert region.support_id == "bowl_0"
    assert region.anchor_ids == ["bowl_0"]
    assert region.polygon_2d == []
    assert region.point_2d == [320.0, 240.0]
    assert region.reason == "preferred_container"


def test_scene_graph_defaults_follow_shared_contract():
    node = SceneNode(id="book_0", class_name="book", bbox=[1.0, 2.0, 3.0, 4.0])
    edge = SceneEdge(
        subject="book_0",
        object="desk_0",
        predicate="on",
        source="derived",
        confidence=1.0,
    )
    graph = SceneGraph(
        image_id="img-1",
        image_path="data/input/images/img-1.png",
        image_size=(640, 480),
        nodes=[node],
        edges=[edge],
    )

    assert graph.image_id == "img-1"
    assert graph.image_path == "data/input/images/img-1.png"
    assert graph.image_size == (640, 480)
    assert graph.nodes == [node]
    assert graph.edges == [edge]
    assert graph.edges[0].frame == "untyped"
    assert graph.ternary_relations == []
    assert graph.blocking_dag == []
    assert graph.support_relations == []
    assert graph.placement_regions == []


def test_scene_graph_support_and_placement_fields_round_trip():
    region = PlacementRegion(
        region_id="container_bowl_0",
        region_type="inside-container",
        support_id="bowl_0",
        anchor_ids=["bowl_0"],
        point_2d=[320.0, 240.0],
        reason="preferred_container",
    )
    graph = SceneGraph(
        image_id="scene",
        image_path="scene.png",
        image_size=(640, 480),
        nodes=[],
        edges=[
            SceneEdge(
                subject="a",
                object="b",
                predicate="above",
                source="2d",
                confidence=0.7,
                frame="image",
            )
        ],
        support_relations=[
            SupportRelation(
                subject="mug_0",
                object="table_0",
                relation="on-top-of",
                confidence=0.9,
            )
        ],
        placement_regions=[region],
    )

    assert graph.edges[0].frame == "image"
    assert graph.support_relations[0].relation == "on-top-of"
    assert graph.placement_regions[0].region_type == "inside-container"
