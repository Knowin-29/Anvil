from apps.data_flywheel.ontology.schema import (
    AliasEntry,
    BaseAffordance,
    ConstraintRule,
    GovernanceMetadata,
    ObjectClass,
)
from apps.data_flywheel.ontology.seed_data import build_seed_store
from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.scene_understanding.builder import _build_nodes
from apps.data_flywheel.types import Detection, InstanceAttributes, Segmentation


def _seg(class_name: str, bbox: list[float], idx: int) -> Segmentation:
    return Segmentation(
        detection=Detection(
            class_name=class_name,
            bbox=bbox,
            confidence=0.9,
            source="detvlm",
        ),
        mask_polygon=[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]]],
        mask_file=f"data/output/masks/test_{idx}.png",
        grasp_point=[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
    )


def test_build_nodes_assigns_ids_per_class():
    store = build_seed_store()
    segmentations = [
        _seg("pen", [10, 10, 40, 80], 0),
        _seg("card", [100, 10, 150, 60], 1),
        _seg("pen", [20, 100, 60, 170], 2),
        _seg("pen", [80, 120, 120, 190], 3),
        _seg("card", [200, 20, 250, 70], 4),
    ]

    nodes = _build_nodes(
        image_id="img-1",
        segmentations=segmentations,
        ontology=store,
        depth_info=None,
    )

    assert [node.id for node in nodes] == [
        "pen_0",
        "card_0",
        "pen_1",
        "pen_2",
        "card_1",
    ]


def test_build_nodes_injects_ontology_affordance():
    store = build_seed_store()
    segmentations = [_seg("card", [100, 80, 180, 140], 0)]

    nodes = _build_nodes(
        image_id="img-2",
        segmentations=segmentations,
        ontology=store,
        depth_info=None,
    )

    assert len(nodes) == 1
    card = nodes[0]
    assert card.operation == "push"
    assert card.is_container is False
    assert card.class_name == "card"
    assert card.container_placeability == "not_applicable"
    assert card.placement_policy == "push_only"
    assert "planar" in card.extra_tags
    assert card.descriptor_axes == ["color", "relative_position"]
    assert card.canonical_name == "卡片"
    assert card.object_class_id == "card"
    assert card.ontology_status == "resolved"
    assert card.triggered_constraints == []
    assert card.mask_source == "mask_polygon"
    assert card.has_true_mask is True
    assert card.hierarchical_name is None
    assert card.referring_expressions == []
    assert isinstance(card.attributes, InstanceAttributes)


def test_build_nodes_matches_depth_info_by_bbox():
    store = build_seed_store()
    segmentations = [
        _seg("pen", [10, 10, 110, 110], 0),
        _seg("cup", [200, 200, 260, 260], 1),
        _seg("book", [320, 320, 380, 380], 2),
    ]
    depth_info = {
        "depth_valid": True,
        "objects": [
            {
                "bbox": [12, 12, 108, 108],
                "center_3d": [0.1, 0.2, 0.5],
                "size_3d": [0.05, 0.12, 0.04],
                "mean_depth": 0.5,
                "valid": True,
            },
            {
                "bbox": [202, 202, 258, 258],
                "center_3d": [0.4, 0.3, 0.8],
                "size_3d": [0.06, 0.1, 0.06],
                "mean_depth": 0.8,
                "valid": True,
            },
        ],
    }

    nodes = _build_nodes(
        image_id="img-3",
        segmentations=segmentations,
        ontology=store,
        depth_info=depth_info,
    )

    assert len(nodes) == 3
    assert nodes[0].valid_3d is True
    assert nodes[0].center_3d == [0.1, 0.2, 0.5]
    assert nodes[0].size_3d == [0.05, 0.12, 0.04]
    assert nodes[0].mean_depth == 0.5

    assert nodes[1].valid_3d is True
    assert nodes[1].center_3d == [0.4, 0.3, 0.8]
    assert nodes[1].size_3d == [0.06, 0.1, 0.06]
    assert nodes[1].mean_depth == 0.8

    assert nodes[2].valid_3d is False
    assert nodes[2].center_3d is None
    assert nodes[2].size_3d is None
    assert nodes[2].mean_depth is None


def test_build_nodes_no_depth_info_all_2d():
    store = build_seed_store()
    segmentations = [
        _seg("pen", [10, 10, 30, 60], 0),
        _seg("tray", [100, 100, 200, 180], 1),
    ]

    nodes = _build_nodes(
        image_id="img-4",
        segmentations=segmentations,
        ontology=store,
        depth_info=None,
    )

    assert len(nodes) == 2
    assert all(node.valid_3d is False for node in nodes)
    assert all(node.center_3d is None for node in nodes)
    assert all(node.size_3d is None for node in nodes)
    assert all(node.mean_depth is None for node in nodes)


def test_build_nodes_partial_depth_payload_treated_as_invalid_3d():
    store = build_seed_store()
    segmentations = [_seg("pen", [10, 10, 110, 110], 0)]
    depth_info = {
        "depth_valid": True,
        "objects": [
            {
                "bbox": [12, 12, 108, 108],
                "center_3d": None,
                "size_3d": [0.05, 0.12, 0.04],
                "mean_depth": 0.5,
                "valid": True,
            }
        ],
    }

    nodes = _build_nodes(
        image_id="img-5",
        segmentations=segmentations,
        ontology=store,
        depth_info=depth_info,
    )

    assert len(nodes) == 1
    node = nodes[0]
    assert node.valid_3d is False
    assert node.center_3d is None
    assert node.size_3d is None
    assert node.mean_depth is None


def test_build_nodes_prefers_usable_depth_match_over_invalid_higher_iou():
    store = build_seed_store()
    segmentations = [_seg("pen", [10, 10, 110, 110], 0)]
    depth_info = {
        "depth_valid": True,
        "objects": [
            {
                "bbox": [11, 11, 109, 109],
                "center_3d": None,
                "size_3d": [0.05, 0.12, 0.04],
                "mean_depth": 0.5,
                "valid": True,
            },
            {
                "bbox": [13, 13, 107, 107],
                "center_3d": [0.2, 0.1, 0.7],
                "size_3d": [0.06, 0.09, 0.03],
                "mean_depth": 0.7,
                "valid": True,
            },
        ],
    }

    nodes = _build_nodes(
        image_id="img-5b",
        segmentations=segmentations,
        ontology=store,
        depth_info=depth_info,
    )

    assert len(nodes) == 1
    node = nodes[0]
    assert node.valid_3d is True
    assert node.center_3d == [0.2, 0.1, 0.7]
    assert node.size_3d == [0.06, 0.09, 0.03]
    assert node.mean_depth == 0.7


def test_build_nodes_passes_runtime_context_to_ontology_constraints():
    store = OntologyStore()
    store.add_class(
        ObjectClass(
            class_id="widget",
            name="widget",
            affordance=BaseAffordance(
                operation="pick",
                is_container=False,
                container_placeability="not_applicable",
            ),
            placement_policy="must_place",
            constraints=[
                ConstraintRule(
                    rule_id="wide-image-push",
                    source_field="image_width",
                    operator="gt",
                    value=100,
                    target_field="operation",
                    effect_value="push",
                    reason="wide image override",
                    priority=0,
                )
            ],
            metadata=GovernanceMetadata(status="approved"),
        )
    )
    store.add_alias(
        AliasEntry(alias="widget", class_id="widget", status="approved")
    )

    nodes = _build_nodes(
        image_id="img-6",
        segmentations=[_seg("widget", [20, 20, 60, 60], 0)],
        ontology=store,
        depth_info=None,
        image_size=(200, 100),
    )

    assert len(nodes) == 1
    node = nodes[0]
    assert node.operation == "push"
    assert "wide image override" in node.triggered_constraints
