import json
from pathlib import Path

import numpy as np
from PIL import Image

from apps.data_flywheel.ontology.schema import IntrinsicActionProfile, ObjectClass
from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.ontology.seed_data import build_seed_store
from apps.data_flywheel.scene_understanding.builder import (
    _segmentation_from_box,
    build_phase_a,
    build_scene_graph,
    enrich_phase_b,
)
from apps.data_flywheel.types import (
    Detection,
    InstanceAttributes,
    PlacementRegion,
    SceneEdge,
    SceneGraph,
    SceneNode,
    Segmentation,
    SupportRelation,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "desk_sample.jpg"


def _seg(class_name: str, bbox: list[float], idx: int) -> Segmentation:
    return Segmentation(
        detection=Detection(
            class_name=class_name,
            bbox=bbox,
            confidence=0.95,
            source="mock",
        ),
        mask_polygon=[
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ],
        mask_file=f"data/output/masks/test_{idx}.png",
        grasp_point=[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
    )


def _scene_fixture():
    with Image.open(FIXTURE_PATH) as raw:
        image = raw.convert("RGB")
    image.filename = str(FIXTURE_PATH)

    objects = [
        ("book", [18, 54, 62, 82]),
        ("book", [68, 50, 112, 78]),
        ("book", [114, 56, 158, 84]),
    ]
    segmentations = []
    for idx, (class_name, bbox) in enumerate(objects):
        segmentations.append(_seg(class_name, bbox, idx))
    return image, segmentations


def _phase_a_fixture():
    image, segmentations = _scene_fixture()
    store = build_seed_store()
    scene_graph = build_phase_a(
        image_id="img-100",
        image_size=image.size,
        segmentations=segmentations,
        ontology=store,
    )
    return image, scene_graph


def _load_fixture_scene_graph(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = []
    for raw in data["nodes"]:
        nodes.append(
            SceneNode(
                id=raw["id"],
                class_name=raw["class_name"],
                bbox=list(raw["bbox"]),
                mask_file=raw.get("mask_file"),
                grasp_point=list(raw["grasp_point"]) if raw.get("grasp_point") else None,
                center_3d=raw.get("center_3d"),
                size_3d=raw.get("size_3d"),
                mean_depth=raw.get("mean_depth"),
                valid_3d=raw.get("valid_3d", False),
                operation=raw.get("operation"),
                is_container=raw.get("is_container", False),
                role_ids=list(raw.get("role_ids", [])),
                support_parent=raw.get("support_parent"),
                support_type=raw.get("support_type"),
                hierarchical_name=raw.get("hierarchical_name"),
                referring_expressions=list(raw.get("referring_expressions", [])),
                attributes=InstanceAttributes(**(raw.get("attributes") or {})),
            )
        )
    edges = [
        SceneEdge(
            subject=raw["subject"],
            object=raw["object"],
            predicate=raw["predicate"],
            source=raw.get("source", "derived"),
            confidence=float(raw.get("confidence", 1.0)),
        )
        for raw in data.get("edges", [])
    ]
    support_relations = [
        SupportRelation(
            subject=raw["subject"],
            object=raw["object"],
            relation=raw["relation"],
            confidence=float(raw.get("confidence", 1.0)),
        )
        for raw in data.get("support_relations", [])
    ]
    placement_regions = [
        PlacementRegion(
            region_id=raw["region_id"],
            region_type=raw["region_type"],
            support_id=raw["support_id"],
            anchor_ids=list(raw.get("anchor_ids", [])),
            polygon_2d=[list(point) for point in raw.get("polygon_2d", [])],
            point_2d=list(raw["point_2d"]) if raw.get("point_2d") else None,
            reason=raw.get("reason", ""),
        )
        for raw in data.get("placement_regions", [])
    ]
    return SceneGraph(
        image_id=data["image_id"],
        image_path=data["image_path"],
        image_size=tuple(data["image_size"]),
        nodes=nodes,
        edges=edges,
        ternary_relations=[],
        blocking_dag=[tuple(edge) for edge in data.get("blocking_dag", [])],
        support_relations=support_relations,
        placement_regions=placement_regions,
    )


def test_build_phase_a_assembles_nodes_relations_and_blocking_defaults():
    image, scene_graph = _phase_a_fixture()

    assert scene_graph.image_id == "img-100"
    assert scene_graph.image_path == "img-100"
    assert scene_graph.image_size == image.size
    assert len(scene_graph.nodes) == 3
    assert scene_graph.edges
    assert len(scene_graph.ternary_relations) == 1
    assert scene_graph.blocking_dag == []

    assert [node.id for node in scene_graph.nodes] == ["book_0", "book_1", "book_2"]
    assert all(node.hierarchical_name is None for node in scene_graph.nodes)
    assert all(node.referring_expressions == [] for node in scene_graph.nodes)
    assert all(node.attributes.color_primary is None for node in scene_graph.nodes)
    assert all(node.attributes.size is None for node in scene_graph.nodes)
    assert all(node.attributes.shape is None for node in scene_graph.nodes)
    assert all(node.valid_3d is False for node in scene_graph.nodes)


def test_enrich_phase_b_populates_attributes_and_referring_expressions_without_mutating_input():
    image, phase_a = _phase_a_fixture()
    original_snapshot = [
        (node.hierarchical_name, list(node.referring_expressions), node.attributes.color_primary)
        for node in phase_a.nodes
    ]

    class RecordingVLM:
        name = "recording"

        def __init__(self):
            self.calls = []

        def ask(self, passed_image, question, max_tokens=512):
            marker = 'canonical name is "'
            start = question.index(marker) + len(marker)
            end = question.index('"', start)
            canonical_name = question[start:end]
            self.calls.append((passed_image.size, question, max_tokens, canonical_name))
            return json.dumps(
                {
                    "en": canonical_name,
                    "zh": f"zh:{canonical_name}",
                    "en_contextual": f"{canonical_name} near the desk",
                }
            )

    backend = RecordingVLM()
    enriched = enrich_phase_b(phase_a, image, backend)
    no_vlm = enrich_phase_b(phase_a, image, None)

    assert [
        (node.hierarchical_name, list(node.referring_expressions), node.attributes.color_primary)
        for node in phase_a.nodes
    ] == original_snapshot
    assert enriched is not phase_a

    assert [node.attributes.color_primary for node in enriched.nodes] == ["red", "green", "blue"]
    assert [node.attributes.size for node in enriched.nodes] == ["medium", "medium", "medium"]
    assert [node.attributes.shape for node in enriched.nodes] == ["wide", "wide", "wide"]
    assert [node.hierarchical_name for node in enriched.nodes] == [
        "the red book",
        "the green book",
        "the blue book",
    ]
    assert [node.referring_expressions for node in enriched.nodes] == [
        ["the red book", "zh:the red book", "the red book near the desk"],
        ["the green book", "zh:the green book", "the green book near the desk"],
        ["the blue book", "zh:the blue book", "the blue book near the desk"],
    ]
    assert [call[0] for call in backend.calls] == [image.size, image.size, image.size]
    assert [call[2] for call in backend.calls] == [512, 512, 512]
    assert [call[3] for call in backend.calls] == [
        "the red book",
        "the green book",
        "the blue book",
    ]

    assert all(node.referring_expressions == [] for node in no_vlm.nodes)
    assert [node.hierarchical_name for node in no_vlm.nodes] == [
        "the red book",
        "the green book",
        "the blue book",
    ]


def test_build_scene_graph_end_to_end_with_fixture_and_real_referring_path():
    image, segmentations = _scene_fixture()
    store = build_seed_store()

    class RecordingVLM:
        name = "recording"

        def __init__(self):
            self.calls = []

        def ask(self, passed_image, question, max_tokens=512):
            marker = 'canonical name is "'
            start = question.index(marker) + len(marker)
            end = question.index('"', start)
            canonical_name = question[start:end]
            self.calls.append((passed_image.size, question, max_tokens, canonical_name))
            return json.dumps(
                {
                    "en": canonical_name,
                    "zh": f"zh:{canonical_name}",
                    "en_contextual": f"{canonical_name} near the desk",
                }
            )

    backend = RecordingVLM()
    graph = build_scene_graph(
        image_id="img-200",
        image=image,
        image_size=image.size,
        segmentations=segmentations,
        ontology=store,
        vlm_backend=backend,
    )

    assert graph.image_id == "img-200"
    assert graph.image_path == str(FIXTURE_PATH)
    assert graph.image_size == image.size
    assert len(graph.nodes) == 3
    assert graph.edges
    assert len(graph.ternary_relations) == 1
    assert graph.blocking_dag == []

    assert [node.attributes.color_primary for node in graph.nodes] == ["red", "green", "blue"]
    assert [node.attributes.size for node in graph.nodes] == ["medium", "medium", "medium"]
    assert [node.attributes.shape for node in graph.nodes] == ["wide", "wide", "wide"]
    assert [node.hierarchical_name for node in graph.nodes] == [
        "the red book",
        "the green book",
        "the blue book",
    ]
    assert [node.referring_expressions for node in graph.nodes] == [
        ["the red book", "zh:the red book", "the red book near the desk"],
        ["the green book", "zh:the green book", "the green book near the desk"],
        ["the blue book", "zh:the blue book", "the blue book near the desk"],
    ]
    assert [call[0] for call in backend.calls] == [image.size, image.size, image.size]
    assert [call[2] for call in backend.calls] == [512, 512, 512]
    assert [call[3] for call in backend.calls] == [
        "the red book",
        "the green book",
        "the blue book",
    ]


def test_build_scene_graph_rejects_non_pil_image():
    _, segmentations = _scene_fixture()
    store = build_seed_store()

    try:
        build_scene_graph(
            image_id="img-201",
            image="not-an-image",
            image_size=(160, 120),
            segmentations=segmentations,
            ontology=store,
        )
    except TypeError as exc:
        assert "PIL.Image.Image" in str(exc)
    else:
        raise AssertionError("expected TypeError for non-PIL image input")


def test_build_phase_a_populates_mask_aligned_point_cloud_sample_from_xyz_map(tmp_path):
    store = build_seed_store()
    segmentations = [
        Segmentation(
            detection=Detection(
                class_name="book",
                bbox=[0, 0, 2, 2],
                confidence=0.95,
                source="mock",
            ),
            mask_polygon=[[0, 0], [1.9, 0], [1.9, 1.9], [0, 1.9]],
            mask_file="data/output/masks/test_pointcloud.png",
            grasp_point=[1.0, 1.0],
        )
    ]

    xyz_map = np.full((3, 3, 3), 5.0, dtype=np.float32)
    xyz_map[:, :, 0] = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    xyz_map[:, :, 1] = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    xyz_map[0:2, 0:2, 2] = 1.0

    xyz_map_path = tmp_path / "scene_xyz.npy"
    np.save(xyz_map_path, xyz_map)

    graph = build_phase_a(
        image_id="img-pointcloud",
        image_size=(3, 3),
        segmentations=segmentations,
        ontology=store,
        depth_info={
            "depth_valid": True,
            "xyz_map_path": str(xyz_map_path),
            "objects": [
                {
                    "bbox": [0, 0, 2, 2],
                    "center_3d": [0.5, 0.5, 1.0],
                    "size_3d": [1.0, 1.0, 0.1],
                    "mean_depth": 1.0,
                    "valid": True,
                }
            ],
        },
    )

    node = graph.nodes[0]
    assert node.valid_3d is True
    assert node.center_3d == [0.5, 0.5, 1.0]
    assert node.point_cloud_sample
    assert all(point[2] < 2.0 for point in node.point_cloud_sample)
    assert node.has_true_mask is True
    assert node.mask_source == "mask_polygon"


def test_build_phase_a_populates_support_and_placement_regions():
    store = OntologyStore()
    store.add_class(
        ObjectClass(
            class_id="table",
            name="table",
            role_ids=["support_surface", "base_support_root"],
            intrinsic_action_profile=IntrinsicActionProfile(
                allowed_operations=["push"],
                preferred_operation="push",
            ),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="bowl",
            name="bowl",
            role_ids=["container", "support_surface"],
            intrinsic_action_profile=IntrinsicActionProfile(
                allowed_operations=["pick", "push"],
                preferred_operation="pick",
            ),
        )
    )
    store.add_class(
        ObjectClass(
            class_id="pen",
            name="pen",
            role_ids=[],
            intrinsic_action_profile=IntrinsicActionProfile(
                allowed_operations=["pick"],
                preferred_operation="pick",
            ),
        )
    )

    segmentations = [
        _seg("table", [0, 70, 100, 100], 0),
        _seg("bowl", [10, 60, 40, 90], 1),
        _seg("pen", [15, 65, 25, 80], 2),
    ]

    graph = build_phase_a(
        image_id="img-placement",
        image_size=(100, 100),
        segmentations=segmentations,
        ontology=store,
    )

    support_pairs = {(rel.subject, rel.object, rel.relation) for rel in graph.support_relations}
    assert ("bowl_0", "table_0", "on-top-of") in support_pairs
    assert ("pen_0", "bowl_0", "inside") in support_pairs

    regions = {(region.region_type, region.support_id) for region in graph.placement_regions}
    assert ("inside-container", "bowl_0") in regions
    assert ("free-space", "table_0") in regions


def test_segmentation_from_box_marks_bbox_fallback_when_no_true_mask():
    segmentation = _segmentation_from_box(
        "pen",
        {
            "bbox": [1, 2, 3, 4],
            "confidence": 0.9,
        },
    )

    assert segmentation.mask_source == "bbox_fallback"
    assert segmentation.mask_polygon == [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]


def test_segmentation_from_box_prefers_explicit_polygon():
    segmentation = _segmentation_from_box(
        "pen",
        {
            "bbox": [1, 2, 3, 4],
            "confidence": 0.9,
            "mask_polygon": [[1, 2], [3, 2], [2, 4]],
        },
    )

    assert segmentation.mask_source == "mask_polygon"
    assert segmentation.mask_polygon == [[1.0, 2.0], [3.0, 2.0], [2.0, 4.0]]


def test_build_scene_graphs_from_grounding_uses_sam_masks_when_enabled(tmp_path, monkeypatch):
    from apps.data_flywheel.scene_understanding.builder import build_scene_graphs_from_grounding

    image_path = tmp_path / "scene.png"
    Image.new("RGB", (64, 64), color="white").save(image_path)

    grounding_path = tmp_path / "grounding.jsonl"
    grounding_record = {
        "image": str(image_path),
        "width": 64,
        "height": 64,
        "phrases": [
            {
                "phrase": "pen",
                "boxes": [
                    {
                        "bbox": [10, 20, 30, 40],
                        "confidence": 0.9,
                    }
                ],
            }
        ],
    }
    grounding_path.write_text(json.dumps(grounding_record, ensure_ascii=False) + "\n", encoding="utf-8")
    output_path = tmp_path / "scene_graph.jsonl"

    class FakeSAM3Backend:
        def __init__(self, base_url="http://sam3", mask_output_dir="data/output/masks"):
            self.base_url = base_url
            self.mask_output_dir = mask_output_dir

        def segment(self, image, *, bbox=None, text_prompt=None, **kwargs):
            return [
                Segmentation(
                    detection=Detection(
                        class_name=kwargs["class_name"],
                        bbox=list(bbox or [10.0, 20.0, 30.0, 40.0]),
                        confidence=1.0,
                        source="sam3",
                    ),
                    mask_polygon=[[12.0, 22.0], [28.0, 22.0], [28.0, 38.0], [12.0, 38.0]],
                    mask_file=f'{self.mask_output_dir}/scene_pen_{kwargs["det_idx"]}.png',
                    grasp_point=[20.0, 30.0],
                )
            ]

    monkeypatch.setattr("apps.data_flywheel.perception.backends.sam3.SAM3Backend", FakeSAM3Backend)

    build_scene_graphs_from_grounding(
        grounding_jsonl=str(grounding_path),
        output_jsonl=str(output_path),
        enable_3d=False,
        enable_mask_segmentation=True,
        sam3_url="http://sam3",
        mask_output_dir="data/output/masks",
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    obj = records[0]["objects"][0]
    assert obj["has_true_mask"] is True
    assert obj["mask_source"] == "mask_polygon"
    assert obj["mask_file"] == "data/output/masks/scene_pen_0.png"
    assert obj["mask_polygon"] == [[12.0, 22.0], [28.0, 22.0], [28.0, 38.0], [12.0, 38.0]]


def test_build_phase_a_populates_runtime_actionability_without_mutating_intrinsic_affordances():
    store = build_seed_store()
    segmentations = [
        _seg("pen", [20, 50, 50, 80], 0),
        _seg("card", [18, 48, 52, 82], 1),
    ]
    graph = build_phase_a(
        image_id="img-actionability",
        image_size=(100, 100),
        segmentations=segmentations,
        ontology=store,
        depth_info={
            "depth_valid": True,
            "objects": [
                {
                    "bbox": [20, 50, 50, 80],
                    "center_3d": [0.0, 0.0, 0.45],
                    "size_3d": [0.1, 0.1, 0.05],
                    "mean_depth": 0.45,
                    "valid": True,
                },
                {
                    "bbox": [18, 48, 52, 82],
                    "center_3d": [0.0, 0.0, 0.35],
                    "size_3d": [0.12, 0.12, 0.04],
                    "mean_depth": 0.35,
                    "valid": True,
                },
            ],
        },
    )

    node_by_id = {node.id: node for node in graph.nodes}
    pen = node_by_id["pen_0"]
    card = node_by_id["card_0"]

    assert pen.preferred_operation == "pick"
    assert pen.blocked_by == ["card_0"]
    assert pen.directly_reachable is False
    assert card.blocking == ["pen_0"]
    assert card.preferred_operation == "push"


def test_tabletop_pick_place_fixture_contains_support_and_placement_regions():
    graph = _load_fixture_scene_graph(
        Path("tests/fixtures/scenes/scene_11_tabletop_pick_place.json")
    )

    assert graph.support_relations
    assert graph.placement_regions
    assert any(region.region_type == "inside-container" for region in graph.placement_regions)
