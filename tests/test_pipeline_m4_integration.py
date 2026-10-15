from __future__ import annotations

import json
from pathlib import Path

from apps.data_flywheel.config import FlywheelConfig
from apps.data_flywheel.ontology.seed_data import build_seed_store
from apps.data_flywheel.ontology.schema import BaseAffordance, ObjectClass
from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.pipeline import _generate_round_reports, _legacy_scene_record_to_scene_graph
from apps.data_flywheel.scene_understanding.builder import scene_graph_to_legacy_record
from apps.data_flywheel.types import (
    InstanceAttributes,
    PlacementRegion,
    SceneEdge,
    SceneGraph,
    SceneNode,
    SupportRelation,
)


def _load_fixture_scene(path: str) -> SceneGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = []
    for raw in data["nodes"]:
        nodes.append(
            SceneNode(
                id=raw["id"],
                class_name=raw["class_name"],
                bbox=list(raw["bbox"]),
                mask_file=raw.get("mask_file"),
                grasp_point=list(raw["grasp_point"]) if raw.get("grasp_point") else None,
                center_3d=list(raw["center_3d"]) if raw.get("center_3d") else None,
                size_3d=list(raw["size_3d"]) if raw.get("size_3d") else None,
                mean_depth=raw.get("mean_depth"),
                valid_3d=bool(raw.get("valid_3d", False)),
                operation=raw.get("operation"),
                is_container=raw.get("is_container"),
                role_ids=list(raw.get("role_ids", [])),
                placement_policy=raw.get("placement_policy"),
                support_parent=raw.get("support_parent"),
                support_type=raw.get("support_type"),
                point_cloud_sample=[list(point) for point in raw.get("point_cloud_sample", [])],
                extra_tags=list(raw.get("extra_tags", [])),
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


def _ontology() -> OntologyStore:
    store = OntologyStore()
    for name, operation in (("pen", "pick"), ("card", "push"), ("book", "pick"), ("box", "non_operable")):
        store.add_class(
            ObjectClass(
                class_id=name,
                name=name,
                affordance=BaseAffordance(operation=operation),
                placement_policy="push_only" if operation == "push" else "must_place",
            )
        )
    return store


def test_legacy_scene_round_trip_preserves_m4_fields():
    scene_graph = SceneGraph(
        image_id="desk_001",
        image_path="tests/fixtures/images/scene_03_card_occludes_pen.png",
        image_size=(100, 100),
        nodes=[
            SceneNode(
                id="pen_0",
                class_name="pen",
                bbox=[40.0, 50.0, 70.0, 80.0],
                mask_file="tests/fixtures/masks/scene_03_card_occludes_pen_pen_0.png",
                grasp_point=[55.0, 65.0],
                operation="pick",
                hierarchical_name="the pen",
                referring_expressions=["笔", "那支笔"],
                attributes=InstanceAttributes(color_primary="red", size="small", shape="wide"),
            )
        ],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    record = scene_graph_to_legacy_record(scene_graph)
    restored = _legacy_scene_record_to_scene_graph(record)
    restored_node = restored.nodes[0]

    assert restored_node.mask_file == "tests/fixtures/masks/scene_03_card_occludes_pen_pen_0.png"
    assert restored_node.grasp_point == [55.0, 65.0]
    assert restored_node.hierarchical_name == "the pen"
    assert restored_node.referring_expressions == ["笔", "那支笔"]
    assert restored_node.attributes.color_primary == "red"
    assert restored_node.attributes.size == "small"
    assert restored_node.attributes.shape == "wide"


def test_legacy_scene_round_trip_preserves_support_and_runtime_metadata():
    scene_graph = SceneGraph(
        image_id="desk_002",
        image_path="tests/fixtures/images/scene_03_card_occludes_pen.png",
        image_size=(100, 100),
        nodes=[
            SceneNode(
                id="tray_0",
                class_name="tray",
                bbox=[10.0, 40.0, 80.0, 90.0],
                role_ids=["container", "support_surface"],
                allowed_operations=["pick", "push"],
                preferred_operation="pick",
                descriptor_axes=["contents", "relative_position"],
                point_cloud_sample=[[0.0, 0.0, 0.8]],
            ),
            SceneNode(
                id="mug_0",
                class_name="mug",
                bbox=[20.0, 45.0, 40.0, 70.0],
                role_ids=[],
                allowed_operations=["pick"],
                preferred_operation="pick",
                support_parent="tray_0",
                support_type="inside",
                support_excluded=False,
                point_cloud_sample=[[0.0, 0.0, 0.7]],
            ),
        ],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
        support_relations=[
            SupportRelation(
                subject="mug_0",
                object="tray_0",
                relation="inside",
                confidence=0.9,
            )
        ],
    )

    record = scene_graph_to_legacy_record(scene_graph)
    restored = _legacy_scene_record_to_scene_graph(record)

    assert record["objects"][0]["role_ids"] == ["container", "support_surface"]
    assert record["objects"][0]["allowed_operations"] == ["pick", "push"]
    assert record["objects"][0]["preferred_operation"] == "pick"
    assert record["objects"][1]["support_parent"] == "tray_0"
    assert record["objects"][1]["support_type"] == "inside"
    assert record["support_relations"][0]["relation"] == "inside"

    tray = restored.nodes[0]
    mug = restored.nodes[1]
    assert tray.role_ids == ["container", "support_surface"]
    assert tray.allowed_operations == ["pick", "push"]
    assert tray.preferred_operation == "pick"
    assert mug.support_parent == "tray_0"
    assert mug.support_type == "inside"
    assert mug.support_excluded is False
    assert mug.point_cloud_sample == [[0.0, 0.0, 0.7]]
    assert restored.support_relations[0].subject == "mug_0"
    assert restored.support_relations[0].object == "tray_0"


def test_legacy_scene_round_trip_preserves_placement_regions():
    scene_graph = SceneGraph(
        image_id="desk_003",
        image_path="tests/fixtures/images/scene_03_card_occludes_pen.png",
        image_size=(100, 100),
        nodes=[
            SceneNode(
                id="table_0",
                class_name="table",
                bbox=[0.0, 60.0, 100.0, 100.0],
                role_ids=["support_surface", "base_support_root"],
            ),
            SceneNode(
                id="bowl_0",
                class_name="bowl",
                bbox=[10.0, 50.0, 40.0, 80.0],
                role_ids=["container", "support_surface"],
            ),
        ],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
        placement_regions=[
            PlacementRegion(
                region_id="inside_bowl_0",
                region_type="inside-container",
                support_id="bowl_0",
                anchor_ids=["bowl_0"],
                polygon_2d=[[10.0, 50.0], [40.0, 50.0], [40.0, 80.0], [10.0, 80.0]],
                point_2d=[25.0, 65.0],
                reason="visible_container",
            ),
            PlacementRegion(
                region_id="free_table_0",
                region_type="free-space",
                support_id="table_0",
                anchor_ids=["table_0"],
                point_2d=[80.0, 80.0],
                reason="support_surface_free_space",
            ),
        ],
    )

    record = scene_graph_to_legacy_record(scene_graph)
    restored = _legacy_scene_record_to_scene_graph(record)

    assert record["placement_regions"][0]["region_type"] == "inside-container"
    assert record["placement_regions"][1]["support_id"] == "table_0"
    assert restored.placement_regions[0].support_id == "bowl_0"
    assert restored.placement_regions[1].point_2d == [80.0, 80.0]


def test_generate_candidate_instructions_prefers_blocker_pick_and_push():
    from apps.data_flywheel.training_data.runtime import generate_candidate_instructions

    scene_graph = _load_fixture_scene("tests/fixtures/scenes/scene_03_card_occludes_pen.json")

    assert generate_candidate_instructions(scene_graph) == [
        "拿被卡片压住的笔",
        "推卡片",
    ]


def test_legacy_scene_loader_infers_true_mask_from_legacy_mask_metadata():
    record = {
        "image": "tests/fixtures/images/scene_03_card_occludes_pen.png",
        "image_size": [100, 100],
        "objects": [
            {
                "id": 0,
                "node_id": "pen_0",
                "name": "pen",
                "bbox": [10.0, 10.0, 30.0, 30.0],
                "mask_polygon": [[12.0, 12.0], [28.0, 12.0], [28.0, 28.0], [12.0, 28.0]],
                "mask_file": "data/output/masks/pen_0.png",
                "grasp_point": [20.0, 20.0],
                "operation": "pick",
                "role_ids": [],
                "allowed_operations": ["pick"],
                "preferred_operation": "pick",
                "valid": False,
            }
        ],
        "relations": [],
    }

    graph = _legacy_scene_record_to_scene_graph(record)

    node = graph.nodes[0]
    assert node.has_true_mask is True
    assert node.mask_source == "mask_file"


def test_generate_candidate_instructions_adds_place_and_cleanup_for_tabletop_fixture():
    from apps.data_flywheel.training_data.runtime import generate_candidate_instructions

    scene_graph = _load_fixture_scene("tests/fixtures/scenes/scene_11_tabletop_pick_place.json")

    instructions = generate_candidate_instructions(scene_graph)

    assert "拿红笔" in instructions
    assert "把红笔放到蓝碗里" in instructions
    assert "收拾桌面" in instructions
    assert all("把蓝碗放到" not in instruction for instruction in instructions)


def test_run_training_generation_materializes_missing_masks_and_accepts_samples(tmp_path):
    from apps.data_flywheel.training_data.runtime import run_training_generation

    scene_graph = _load_fixture_scene("tests/fixtures/scenes/scene_03_card_occludes_pen.json")
    for node in scene_graph.nodes:
        node.mask_file = ""

    result = run_training_generation(
        scene_graph=scene_graph,
        ontology=_ontology(),
        variants_per_plan=2,
        max_instructions=2,
        mask_output_dir=str(tmp_path / "masks"),
    )

    assert result["instructions"] == ["拿被卡片压住的笔", "推卡片"]
    assert [plan.status for plan in result["plan_results"]] == ["success", "success"]
    assert len(result["training_samples"]) == 4
    assert result["rejected_entries"] == []
    assert all(Path(obj.mask_file).exists() for sample in result["training_samples"] for obj in sample.scene_objects)


def test_run_training_generation_emits_place_samples_for_tabletop_fixture(tmp_path):
    from apps.data_flywheel.training_data.runtime import run_training_generation

    scene_graph = _load_fixture_scene("tests/fixtures/scenes/scene_11_tabletop_pick_place.json")

    result = run_training_generation(
        scene_graph=scene_graph,
        ontology=build_seed_store(),
        variants_per_plan=1,
        max_instructions=0,
        mask_output_dir=str(tmp_path / "masks"),
    )

    assert "把红笔放到蓝碗里" in result["instructions"]
    assert "收拾桌面" in result["instructions"]
    assert any(any(action.type == "place" for action in plan.actions) for plan in result["plan_results"])
    assert any(any(action.type == "place" for action in sample.actions) for sample in result["training_samples"])


def test_build_training_artifacts_writes_round_outputs(tmp_path):
    from apps.data_flywheel.training_data.runtime import build_training_artifacts

    scene_graph = _load_fixture_scene("tests/fixtures/scenes/scene_03_card_occludes_pen.json")
    result = build_training_artifacts(
        [scene_graph],
        ontology=_ontology(),
        output_dir=str(tmp_path),
        variants_per_plan=2,
        max_instructions=2,
    )

    assert Path(result["plan_results_path"]).exists()
    assert Path(result["training_samples_path"]).exists()
    assert Path(result["rejected_samples_path"]).exists()
    assert Path(result["validation_metrics_path"]).exists()
    assert result["accepted_samples"] == 4
    assert result["rejected_samples"] == 0
    assert len(result["plan_results_by_image"][scene_graph.image_path]) == 2
    assert len(result["training_samples_by_image"][scene_graph.image_path]) == 4


def test_generate_round_reports_includes_m4_training_section(tmp_path):
    from apps.data_flywheel.training_data.runtime import build_training_artifacts

    scene_graph = _load_fixture_scene("tests/fixtures/scenes/scene_03_card_occludes_pen.json")
    legacy_record = scene_graph_to_legacy_record(scene_graph, image_path=scene_graph.image_path)
    scene_graph_jsonl = tmp_path / "scene_graph.jsonl"
    scene_graph_jsonl.write_text(json.dumps(legacy_record, ensure_ascii=False) + "\n", encoding="utf-8")

    grounding_record = {
        "image": scene_graph.image_path,
        "width": scene_graph.image_size[0],
        "height": scene_graph.image_size[1],
        "phrases": [
            {
                "phrase": node.class_name,
                "boxes": [
                    {
                        "bbox": list(node.bbox),
                        "confidence": 1.0,
                    }
                ],
            }
            for node in scene_graph.nodes
        ],
    }
    grounding_jsonl = tmp_path / "grounding.jsonl"
    grounding_jsonl.write_text(json.dumps(grounding_record, ensure_ascii=False) + "\n", encoding="utf-8")

    training = build_training_artifacts(
        [scene_graph],
        ontology=_ontology(),
        output_dir=str(tmp_path / "training"),
        variants_per_plan=2,
        max_instructions=2,
    )
    report_paths = _generate_round_reports(
        str(grounding_jsonl),
        str(scene_graph_jsonl),
        training_by_image={
            scene_graph.image_path: {
                "plan_results": training["plan_results_by_image"][scene_graph.image_path],
                "training_samples": training["training_samples_by_image"][scene_graph.image_path],
                "rejected_entries": training["rejected_entries_by_image"][scene_graph.image_path],
            }
        },
    )

    assert len(report_paths) == 1
    content = Path(report_paths[0]).read_text(encoding="utf-8")
    assert "M4 - Training Data" in content
    assert "拿被卡片压住的笔" in content


def test_flywheel_config_default_ontology_store_path_exists():
    cfg = FlywheelConfig()

    assert Path(cfg.ontology_store_path).exists()


def test_run_training_generation_recovers_operation_from_ontology_for_legacy_unknown_scene(tmp_path):
    from apps.data_flywheel.training_data.runtime import run_training_generation

    image_path = Path("tests/fixtures/images/scene_03_card_occludes_pen.png")
    scene_graph = SceneGraph(
        image_id="legacy_plate",
        image_path=str(image_path),
        image_size=(100, 100),
        nodes=[
            SceneNode(
                id="plate_0",
                class_name="plate",
                bbox=[10.0, 10.0, 70.0, 70.0],
                operation=None,
                ontology_status="unknown",
                attributes=InstanceAttributes(),
            )
        ],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    result = run_training_generation(
        scene_graph=scene_graph,
        ontology=build_seed_store(),
        variants_per_plan=1,
        max_instructions=1,
        mask_output_dir=str(tmp_path / "masks"),
    )

    hydrated_node = result["scene_graph"].nodes[0]
    assert hydrated_node.operation == "pick"
    assert hydrated_node.object_class_id == "plate"
    assert hydrated_node.ontology_status == "resolved"
    assert len(result["instructions"]) == 1
    assert result["instructions"][0].startswith("拿")
    assert len(result["plan_results"]) == 1
    assert len(result["training_samples"]) == 1
