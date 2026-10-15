"""End-to-end tests for pipeline visualization report generation."""

import json
import os
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

from apps.data_flywheel.types import (
    InstanceAttributes,
    PlacementRegion,
    SceneEdge,
    SceneGraph,
    SceneNode,
    SupportRelation,
    ThinkingStep,
    ThinkingTrace,
)
from apps.data_flywheel.visualization.report import generate_report


def test_e2e_full_report_with_all_sections(tmp_path):
    image = Image.new("RGB", (640, 480), (180, 180, 200))
    image_path = str(tmp_path / "desk.jpg")
    image.save(image_path)

    pen = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[100.0, 200.0, 200.0, 220.0],
        center_3d=[0.15, 0.1, 0.5],
        size_3d=[0.08, 0.01, 0.01],
        mean_depth=0.5,
        valid_3d=True,
        operation="pick",
        hierarchical_name="the red pen",
        referring_expressions=["红色的笔", "the red pen"],
        attributes=InstanceAttributes(color_primary="red", size="small", shape="tall"),
    )
    card = SceneNode(
        id="card_0",
        class_name="card",
        bbox=[90.0, 195.0, 210.0, 225.0],
        center_3d=[0.15, 0.1, 0.45],
        size_3d=[0.1, 0.001, 0.06],
        mean_depth=0.45,
        valid_3d=True,
        operation="push",
        extra_tags=["planar", "push_only", "stackable_2d"],
        hierarchical_name="the card",
        attributes=InstanceAttributes(color_primary="white", size="medium", shape="wide"),
    )
    edge = SceneEdge(subject="card_0", object="pen_0", predicate="stacked-on", source="3d", confidence=0.92)
    scene_graph = SceneGraph(
        image_id="desk_001",
        image_path=image_path,
        image_size=(640, 480),
        nodes=[pen, card],
        edges=[edge],
        ternary_relations=[],
        blocking_dag=[("card_0", "pen_0")],
    )

    labels = [
        {
            "observation_name": "pen",
            "canonical_name": "pen",
            "operation": "pick",
            "extra_tags": [],
            "ontology_status": "resolved",
        },
        {
            "observation_name": "card",
            "canonical_name": "card",
            "operation": "push",
            "extra_tags": ["planar", "push_only"],
            "ontology_status": "resolved",
        },
    ]
    detections = [
        {"class_name": "pen", "bbox": [100.0, 200.0, 200.0, 220.0], "confidence": 0.95, "source": "detvlm"},
        {"class_name": "card", "bbox": [90.0, 195.0, 210.0, 225.0], "confidence": 0.91, "source": "sam3"},
    ]
    thinking = ThinkingTrace(
        steps=[
            ThinkingStep(
                perception_type="ReferenceResolution",
                target="pen_0",
                content="the red pen",
                output={"node_id": "pen_0"},
            ),
            ThinkingStep(
                perception_type="Affordance",
                target="pen_0",
                content="operation=pick",
                output={"operation": "pick"},
            ),
            ThinkingStep(
                perception_type="BlockingDependency",
                target="pen_0",
                content="blocked by card_0",
                output={"blocker": "card_0"},
            ),
            ThinkingStep(
                perception_type="PlanOrder",
                target="",
                content="push card_0 -> pick pen_0",
                output=["push:card_0", "pick:pen_0"],
            ),
        ]
    )
    plan = {
        "instruction": "拿被卡片压住的红笔",
        "status": "success",
        "resolved_target_id": "pen_0",
        "thinking": thinking,
        "actions": [
            {"type": "push", "target": "card_0"},
            {"type": "pick", "target": "pen_0"},
        ],
        "notes": [],
    }
    sample = {
        "sample_id": "s_001",
        "image_id": "desk_001",
        "instruction": "拿被卡片压住的红笔",
        "instruction_language": "zh",
        "rex_omni_text": "<|user|>拿被卡片压住的红笔<|end|><|thinking_start|>[ReferenceResolution] pen_0: the red pen\n[Affordance] pen_0: pick<|thinking_end|><|action_start|>push card_0\npick pen_0<|action_end|>",
        "image_path": image_path,
    }

    report_path = generate_report(
        image_path=image_path,
        image_size=(640, 480),
        ontology_labels=labels,
        detections=detections,
        segmentations=[],
        scene_graph=scene_graph,
        plan_results=[plan],
        training_samples=[sample],
        validation_report=[],
        output_dir=str(tmp_path / "reports"),
    )

    assert os.path.exists(report_path)
    content = Path(report_path).read_text(encoding="utf-8")

    assert "<!DOCTYPE html>" in content
    assert "desk_001" in content
    assert "badge-pick" in content
    assert "badge-push" in content
    assert "<svg" in content or "detection" in content.lower()
    assert "vis.Network" in content or "vis-network" in content.lower()
    assert "BoxGeometry" in content or "THREE" in content
    assert "card_0" in content and "pen_0" in content
    assert "Support Graph" in content
    assert "Placement Regions" in content
    assert "ReferenceResolution" in content
    assert "Affordance" in content
    assert "token-user" in content or "token-thinking" in content

    file_size = os.path.getsize(report_path)
    assert file_size < 5 * 1024 * 1024, f"Report too large: {file_size / 1024 / 1024:.1f}MB"


def test_e2e_report_opens_without_errors(tmp_path):
    image = Image.new("RGB", (640, 480))
    image_path = str(tmp_path / "test.jpg")
    image.save(image_path)
    scene_graph = SceneGraph(
        image_id="t",
        image_path=image_path,
        image_size=(640, 480),
        nodes=[],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    report_path = generate_report(
        image_path=image_path,
        image_size=(640, 480),
        ontology_labels=[],
        detections=[],
        segmentations=[],
        scene_graph=scene_graph,
        output_dir=str(tmp_path / "reports"),
    )
    content = Path(report_path).read_text(encoding="utf-8")

    errors = []

    class StrictParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            return None

    parser = StrictParser()
    try:
        parser.feed(content)
    except Exception as exc:  # pragma: no cover - exercised only on malformed HTML
        errors.append(str(exc))

    assert not errors, f"HTML parse errors: {errors}"


def test_e2e_pick_place_fixture_report_shows_placement_overlay(tmp_path):
    data = json.loads(Path("tests/fixtures/scenes/scene_11_tabletop_pick_place.json").read_text(encoding="utf-8"))
    scene_graph = SceneGraph(
        image_id=data["image_id"],
        image_path=data["image_path"],
        image_size=tuple(data["image_size"]),
        nodes=[
            SceneNode(
                id=node["id"],
                class_name=node["class_name"],
                bbox=list(node["bbox"]),
                mask_file=node.get("mask_file"),
                grasp_point=node.get("grasp_point"),
                center_3d=node.get("center_3d"),
                size_3d=node.get("size_3d"),
                mean_depth=node.get("mean_depth"),
                valid_3d=node.get("valid_3d", False),
                operation=node.get("operation"),
                is_container=node.get("is_container", False),
                role_ids=list(node.get("role_ids", [])),
                support_parent=node.get("support_parent"),
                support_type=node.get("support_type"),
                hierarchical_name=node.get("hierarchical_name"),
                referring_expressions=list(node.get("referring_expressions", [])),
                attributes=InstanceAttributes(**(node.get("attributes") or {})),
            )
            for node in data["nodes"]
        ],
        edges=[
            SceneEdge(
                subject=edge["subject"],
                object=edge["object"],
                predicate=edge["predicate"],
                source=edge.get("source", "derived"),
                confidence=float(edge.get("confidence", 1.0)),
            )
            for edge in data.get("edges", [])
        ],
        ternary_relations=[],
        blocking_dag=[tuple(edge) for edge in data.get("blocking_dag", [])],
        support_relations=[
            SupportRelation(
                subject=relation["subject"],
                object=relation["object"],
                relation=relation["relation"],
                confidence=float(relation.get("confidence", 1.0)),
            )
            for relation in data.get("support_relations", [])
        ],
        placement_regions=[
            PlacementRegion(
                region_id=region["region_id"],
                region_type=region["region_type"],
                support_id=region["support_id"],
                anchor_ids=list(region.get("anchor_ids", [])),
                polygon_2d=[list(point) for point in region.get("polygon_2d", [])],
                point_2d=list(region["point_2d"]) if region.get("point_2d") else None,
                reason=region.get("reason", ""),
            )
            for region in data.get("placement_regions", [])
        ],
    )

    report_path = generate_report(
        image_path=scene_graph.image_path,
        image_size=scene_graph.image_size,
        ontology_labels=[],
        detections=[],
        segmentations=[],
        scene_graph=scene_graph,
        training_samples=[
            {
                "sample_id": "fixture_place_1",
                "instruction": "把那支笔放到碗里",
                "image_path": scene_graph.image_path,
                "image_size": scene_graph.image_size,
                "actions": [
                    {
                        "type": "place",
                        "target": "pen_0",
                        "destination": "bowl_0",
                        "place_point_quantized": [750, 630],
                    }
                ],
                "rex_omni_text": "<|user|>把那支笔放到碗里<|end|>",
            }
        ],
        output_dir=str(tmp_path / "reports"),
    )
    content = Path(report_path).read_text(encoding="utf-8")

    assert "A. 2D Image-Anchored Relation View" in content
    assert "B. 3D Point-Cloud-First View" in content
    assert "C. Support / Blocking Structure View" in content
    assert "D. Placement View" in content
    assert "Placement Regions" in content
    assert "Selected place point" in content
