from apps.data_flywheel.ontology.matcher import ResolvedRuntimeLabel
from apps.data_flywheel.types import (
    Detection,
    InstanceAttributes,
    PlacementRegion,
    SceneEdge,
    SceneGraph,
    SceneNode,
    Segmentation,
    SupportRelation,
    ThinkingStep,
    ThinkingTrace,
)


def test_cdn_urls_are_https():
    from apps.data_flywheel.visualization.assets import (
        CDN_THREEJS,
        CDN_THREEJS_CSS2D,
        CDN_THREEJS_ORBIT,
        CDN_VISJS,
        CDN_VISJS_CSS,
    )

    for url in [CDN_THREEJS, CDN_THREEJS_CSS2D, CDN_THREEJS_ORBIT, CDN_VISJS, CDN_VISJS_CSS]:
        assert url.startswith("https://"), f"CDN URL must be HTTPS: {url}"


def test_operation_colors_has_all_types():
    from apps.data_flywheel.visualization.assets import OPERATION_COLORS

    for op in ["pick", "push", "non_operable", None]:
        assert op in OPERATION_COLORS


def test_build_html_skeleton_has_required_tags():
    from apps.data_flywheel.visualization.assets import build_html_skeleton

    html = build_html_skeleton(
        title="Test Report",
        body_sections="<p>hello</p>",
        inline_js="console.log(1);",
    )
    assert "<!DOCTYPE html>" in html
    assert "<title>Test Report</title>" in html
    assert "three" in html.lower() or "three.min.js" in html
    assert "vis-network" in html.lower() or "vis-network.min.js" in html
    assert "<p>hello</p>" in html
    assert "console.log(1);" in html


def test_build_html_skeleton_loads_dependencies_before_body_sections():
    from apps.data_flywheel.visualization.assets import CDN_VISJS, build_html_skeleton

    marker = '<div id="marker">body</div>'
    html = build_html_skeleton(
        title="Dependency Order",
        body_sections=marker,
        inline_js="",
    )

    assert html.index(CDN_VISJS) < html.index(marker)
    assert html.index('type="importmap"') < html.index(marker)


def test_legacy_scene_record_to_ontology_labels_preserves_roles_and_intrinsic_ops():
    from apps.data_flywheel.pipeline import _legacy_scene_record_to_ontology_labels

    labels = _legacy_scene_record_to_ontology_labels(
        {
            "objects": [
                {
                    "name": "plate",
                    "canonical_name": "plate",
                    "object_class_id": "plate",
                    "object_subtype_id": "ceramic_plate",
                    "matched_alias": "dish",
                    "operation": "pick",
                    "role_ids": ["support_surface"],
                    "allowed_operations": ["pick", "push"],
                    "preferred_operation": "pick",
                    "ontology_status": "resolved",
                    "perception": {"extra_tags": ["flat"]},
                }
            ]
        }
    )

    assert labels[0]["object_class_id"] == "plate"
    assert labels[0]["object_subtype_id"] == "ceramic_plate"
    assert labels[0]["matched_alias"] == "dish"
    assert labels[0]["role_ids"] == ["support_surface"]
    assert labels[0]["allowed_operations"] == ["pick", "push"]
    assert labels[0]["preferred_operation"] == "pick"


def test_ontology_section_renders_table():
    from apps.data_flywheel.visualization.section_ontology import render_ontology_section

    labels = [
        {
            "observation_name": "pen",
            "canonical_name": "pen",
            "operation": "pick",
            "extra_tags": ["fragile"],
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

    html = render_ontology_section(labels)

    assert "<table" in html
    assert "pen" in html
    assert "card" in html
    assert "badge-pick" in html
    assert "badge-push" in html


def test_ontology_section_accepts_runtime_labels():
    from apps.data_flywheel.visualization.section_ontology import render_ontology_section

    labels = [
        ResolvedRuntimeLabel(
            observation_name="mug",
            normalized_name="mug",
            ontology_status="resolved",
            canonical_name="mug",
            object_class_id="mug",
            object_subtype_id=None,
            matched_alias=None,
            operation="non_operable",
            is_container=True,
            container_placeability="open_top",
            placement_policy="place_inside",
            extra_tags=["container"],
            triggered_constraints=[],
            perception_notes=[],
        )
    ]

    html = render_ontology_section(labels)

    assert "mug" in html
    assert "badge-non_operable" in html


def test_ontology_section_surfaces_layered_naming_contract_and_runtime_bridge():
    from apps.data_flywheel.visualization.section_ontology import render_ontology_section

    labels = [
        ResolvedRuntimeLabel(
            observation_name="dish",
            normalized_name="dish",
            ontology_status="resolved",
            canonical_name="plate",
            object_class_id="plate",
            object_subtype_id="ceramic_plate",
            matched_alias="dish",
            operation="pick",
            is_container=False,
            container_placeability=None,
            placement_policy="place_on_support",
            extra_tags=["flat"],
            triggered_constraints=[],
            perception_notes=[],
            role_ids=["support_surface"],
            allowed_operations=["pick", "push"],
            preferred_operation="pick",
        )
    ]
    scene_graph = SceneGraph(
        image_id="ontology_bridge",
        image_path="ontology_bridge.jpg",
        image_size=(640, 480),
        nodes=[
            SceneNode(
                id="plate_0",
                class_name="plate",
                bbox=[0.0, 0.0, 100.0, 100.0],
                operation="pick",
                object_class_id="plate",
                hierarchical_name="left white plate under scissors",
                referring_expressions=["the left white plate", "左边白色盘子"],
                descriptor_axes=["color", "relative_position", "near_anchor"],
                attributes=InstanceAttributes(color_primary="white"),
            )
        ],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_ontology_section(labels, scene_graph=scene_graph)

    assert "Layer 1" in html
    assert "Layer 1.5" in html
    assert "Layer 2" in html
    assert "Layer 3" in html
    assert "ceramic_plate" in html
    assert "dish" in html
    assert "support_surface" in html
    assert "pick, push" in html
    assert "left white plate under scissors" in html
    assert "the left white plate" in html
    assert "descriptor axes" in html.lower()


def test_ontology_section_empty_labels():
    from apps.data_flywheel.visualization.section_ontology import render_ontology_section

    html = render_ontology_section([])
    assert "No ontology data" in html or "无数据" in html


def test_perception_section_renders_detections():
    from apps.data_flywheel.visualization.section_perception import render_perception_section

    detections = [
        {
            "class_name": "pen",
            "bbox": [10, 20, 100, 60],
            "confidence": 0.95,
            "source": "detvlm",
        },
        {
            "class_name": "cup",
            "bbox": [200, 100, 300, 250],
            "confidence": 0.88,
            "source": "sam3",
        },
    ]

    html = render_perception_section(
        image_base64="data:image/png;base64,iVBOR",
        image_size=(640, 480),
        detections=detections,
        segmentations=[],
    )

    assert "<svg" in html
    assert "pen" in html
    assert "0.95" in html


def test_perception_section_accepts_dataclasses():
    from apps.data_flywheel.visualization.section_perception import render_perception_section

    detection = Detection(
        class_name="pen",
        bbox=[10.0, 20.0, 100.0, 60.0],
        confidence=0.95,
        source="detvlm",
    )
    segmentation = Segmentation(
        detection=detection,
        mask_polygon=[[10.0, 20.0], [20.0, 20.0], [20.0, 30.0]],
        mask_file="mask.png",
        grasp_point=[15.0, 25.0],
    )

    html = render_perception_section(
        image_base64="data:image/png;base64,iVBOR",
        image_size=(640, 480),
        detections=[detection],
        segmentations=[segmentation],
    )

    assert "<polygon" in html
    assert "<circle" in html
    assert "detvlm" in html


def test_perception_section_renders_backend_comparison():
    from apps.data_flywheel.visualization.section_perception import render_perception_section

    detections = [
        {"class_name": "pen", "bbox": [10.0, 20.0, 100.0, 60.0], "confidence": 0.95, "source": "detvlm"},
        {"class_name": "card", "bbox": [20.0, 30.0, 120.0, 90.0], "confidence": 0.87, "source": "detvlm"},
        {"class_name": "cup", "bbox": [200.0, 100.0, 300.0, 250.0], "confidence": 0.88, "source": "sam3"},
    ]
    segmentations = [
        {
            "class_name": "pen",
            "detection": detections[0],
            "mask_polygon": [[10.0, 20.0], [20.0, 20.0], [20.0, 30.0]],
            "grasp_point": [15.0, 25.0],
        }
    ]

    html = render_perception_section(
        image_base64="data:image/png;base64,iVBOR",
        image_size=(640, 480),
        detections=detections,
        segmentations=segmentations,
    )

    assert "Backend Comparison" in html
    assert "Detection Count" in html
    assert "Segmentation Count" in html
    assert "detvlm" in html
    assert "sam3" in html


def test_perception_section_empty():
    from apps.data_flywheel.visualization.section_perception import render_perception_section

    html = render_perception_section(
        image_base64="",
        image_size=(640, 480),
        detections=[],
        segmentations=[],
    )
    assert "无数据" in html or "No detections" in html


def test_scene_section_has_all_subsections():
    from apps.data_flywheel.visualization.section_scene import render_scene_section

    node = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[10.0, 20.0, 100.0, 60.0],
        center_3d=[0.1, 0.2, 0.5],
        size_3d=[0.05, 0.01, 0.15],
        mean_depth=0.5,
        valid_3d=True,
        operation="pick",
        hierarchical_name="the pen",
        attributes=InstanceAttributes(color_primary="red", size="small", shape="tall"),
    )
    edge = SceneEdge(subject="pen_0", object="pen_0", predicate="self", source="2d", confidence=1.0)
    scene_graph = SceneGraph(
        image_id="test",
        image_path="test.jpg",
        image_size=(640, 480),
        nodes=[node],
        edges=[edge],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_scene_section(scene_graph)

    assert (
        "A. 2D Image-Anchored Relation View" in html
        or "vis-network" in html.lower()
        or "vis.Network" in html
    )
    assert "3D Scene" in html or "THREE" in html


def test_scene_section_surfaces_diagnostics_and_secondary_relation_toggle():
    from apps.data_flywheel.visualization.section_scene import render_scene_section

    node_a = SceneNode(
        id="cup_0",
        class_name="cup",
        bbox=[10.0, 20.0, 100.0, 80.0],
        operation="pick",
        ontology_status="resolved",
        hierarchical_name="the cup",
        has_true_mask=False,
        mask_source="bbox",
        valid_3d=False,
        attributes=InstanceAttributes(),
    )
    node_b = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[120.0, 20.0, 220.0, 60.0],
        operation="pick",
        ontology_status="unknown",
        hierarchical_name="the pen",
        attributes=InstanceAttributes(),
    )
    edge = SceneEdge(subject="cup_0", object="pen_0", predicate="left-of", source="2d", confidence=0.9)
    scene_graph = SceneGraph(
        image_id="scene-diag",
        image_path="scene.jpg",
        image_size=(640, 480),
        nodes=[node_a, node_b],
        edges=[edge],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_scene_section(scene_graph)

    assert "Diagnostics" in html
    assert "mask fallback" in html
    assert "class unresolved" in html
    assert "Show secondary relations" in html


def test_placement_section_surfaces_destination_provenance_and_anchor_labels():
    from apps.data_flywheel.visualization.section_placement import render_placement_section

    pen = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[20.0, 20.0, 60.0, 40.0],
        operation="pick",
        hierarchical_name="the red pen",
        attributes=InstanceAttributes(),
    )
    bowl = SceneNode(
        id="bowl_0",
        class_name="bowl",
        bbox=[80.0, 20.0, 140.0, 90.0],
        operation="pick",
        hierarchical_name="the yellow bowl",
        attributes=InstanceAttributes(),
    )
    scene_graph = SceneGraph(
        image_id="placement",
        image_path="placement.jpg",
        image_size=(320, 240),
        nodes=[pen, bowl],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
        placement_regions=[
            PlacementRegion(
                region_id="inside_bowl_0",
                region_type="inside-container",
                support_id="bowl_0",
                anchor_ids=["bowl_0"],
                polygon_2d=[[80.0, 20.0], [140.0, 20.0], [140.0, 90.0], [80.0, 90.0]],
                point_2d=[110.0, 55.0],
                reason="container interior",
            )
        ],
    )

    html = render_placement_section(
        scene_graph,
        training_samples=[
            {
                "actions": [
                    {
                        "type": "place",
                        "target": "pen_0",
                        "destination": "bowl_0",
                        "place_point_pixel": [110.0, 55.0],
                        "destination_type": "container",
                        "destination_anchor_ids": ["bowl_0"],
                        "placement_reason": "preferred_container",
                    }
                ]
            }
        ],
        image_base64="",
    )

    assert "preferred_container" in html
    assert "container" in html
    assert "the yellow bowl" in html


def test_scene_section_empty_graph():
    from apps.data_flywheel.visualization.section_scene import render_scene_section

    scene_graph = SceneGraph(
        image_id="t",
        image_path="t.jpg",
        image_size=(640, 480),
        nodes=[],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_scene_section(scene_graph)

    assert "无数据" in html or "No nodes" in html


def test_scene_section_renders_image_anchored_key_relations():
    from apps.data_flywheel.visualization.section_scene import render_scene_section

    scene_graph = SceneGraph(
        image_id="layout_test",
        image_path="layout.jpg",
        image_size=(640, 480),
        nodes=[
            SceneNode(
                id="marker_0",
                class_name="marker",
                bbox=[10.0, 20.0, 80.0, 120.0],
                operation="pick",
                attributes=InstanceAttributes(),
            ),
            SceneNode(
                id="marker_1",
                class_name="marker",
                bbox=[420.0, 260.0, 500.0, 360.0],
                operation="pick",
                attributes=InstanceAttributes(),
            ),
        ],
        edges=[
            SceneEdge(subject="marker_0", object="marker_1", predicate="left-of", source="2d", confidence=0.9),
            SceneEdge(subject="marker_1", object="marker_0", predicate="right-of", source="2d", confidence=0.9),
        ],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_scene_section(scene_graph, image_base64="data:image/jpeg;base64,AAAA")

    assert 'aria-label="2D relation overlay"' in html
    assert "Key Relations" in html
    assert "marker_0" in html
    assert "marker_1" in html
    assert "left-of" in html


def test_scene_section_summarizes_support_plane_relations():
    from apps.data_flywheel.visualization.section_scene import render_scene_section

    scene_graph = SceneGraph(
        image_id="support_test",
        image_path="support.jpg",
        image_size=(640, 480),
        nodes=[
            SceneNode(
                id="table_0",
                class_name="table",
                bbox=[0.0, 40.0, 640.0, 460.0],
                operation="non_operable",
                attributes=InstanceAttributes(),
            ),
            SceneNode(
                id="bowl_0",
                class_name="bowl",
                bbox=[220.0, 120.0, 320.0, 260.0],
                operation="pick",
                attributes=InstanceAttributes(),
            ),
        ],
        edges=[
            SceneEdge(subject="bowl_0", object="table_0", predicate="on-top-of", source="2d+3d", confidence=0.95),
        ],
        ternary_relations=[],
        blocking_dag=[],
    )

    html = render_scene_section(scene_graph, image_base64="data:image/jpeg;base64,BBBB")

    assert "Support Context" in html
    assert "on-top-of" in html
    assert "table_0" in html
    assert "bowl_0" in html


def test_scene_section_centers_blocking_dag_in_viewport():
    from apps.data_flywheel.visualization.section_scene import render_scene_section

    scene_graph = SceneGraph(
        image_id="dag_test",
        image_path="dag.jpg",
        image_size=(640, 480),
        nodes=[
            SceneNode(
                id="paper_0",
                class_name="paper",
                bbox=[10.0, 20.0, 80.0, 120.0],
                operation="push",
                attributes=InstanceAttributes(),
            ),
            SceneNode(
                id="bowl_0",
                class_name="bowl",
                bbox=[220.0, 120.0, 320.0, 260.0],
                operation="pick",
                attributes=InstanceAttributes(),
            ),
        ],
        edges=[],
        ternary_relations=[],
        blocking_dag=[("paper_0", "bowl_0")],
    )

    html = render_scene_section(scene_graph)

    assert "layout:{hierarchical:{enabled:true,direction:'LR',sortMethod:'directed'}}" in html
    assert "const fitDag=()=>network.fit({animation:false});" in html


def test_training_section_thinking_trace():
    from apps.data_flywheel.visualization.section_training import render_training_section

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
                perception_type="PlanOrder",
                target="",
                content="1. pick pen_0",
                output=["pick:pen_0"],
            ),
        ]
    )
    plan = {
        "instruction": "拿红色的笔",
        "status": "success",
        "thinking": thinking,
        "actions": [{"type": "pick", "target": "pen_0"}],
        "notes": [],
    }

    html = render_training_section(
        plan_results=[plan],
        training_samples=[],
        rejected_entries=[],
    )

    assert "ReferenceResolution" in html
    assert "Affordance" in html
    assert "success" in html.lower()


def test_training_section_renders_actions_and_status_palette():
    from apps.data_flywheel.visualization.section_training import render_training_section

    plan = {
        "instruction": "pick the blocked pen",
        "status": "ambiguous",
        "thinking": ThinkingTrace(
            steps=[
                ThinkingStep(
                    perception_type="ReferenceResolution",
                    target="pen_0",
                    content="multiple red pens",
                    output={"candidates": ["pen_0", "pen_1"]},
                )
            ]
        ),
        "actions": [
            {"type": "push", "target": "card_0"},
            {"type": "pick", "target": "pen_0"},
        ],
        "notes": [],
    }

    html = render_training_section(
        plan_results=[plan],
        training_samples=[],
        rejected_entries=[],
    )

    assert "Action: push" in html
    assert "Action: pick" in html
    assert "badge-ambiguous" in html
    assert "badge-blocked_unresolvable" in html


def test_training_section_validator_heatmap():
    from apps.data_flywheel.visualization.section_training import render_training_section

    html = render_training_section(
        plan_results=[],
        training_samples=[],
        rejected_entries=[
            {
                "sample_id": "s1",
                "instruction": "拿笔",
                "critical_fails": ["CoordinateRange"],
                "warning_fails": ["ImageExists"],
                "reasons": ["coord out of range"],
            }
        ],
    )

    assert "CoordinateRange" in html
    assert "ActionConsistency" in html
    assert "coord out of range" in html
    assert "heatmap" in html.lower() or "validator" in html.lower()


def test_training_section_validator_detail_panel_and_stats():
    from apps.data_flywheel.visualization.section_training import render_training_section

    accepted_sample = {
        "sample_id": "ok_1",
        "instruction": "pick the pen",
        "rex_omni_text": "<|user|>pick the pen<|end|>",
    }
    rejected_entry = {
        "sample_id": "bad_1",
        "instruction": "pick the invalid pen",
        "critical_fails": ["CoordinateRange"],
        "warning_fails": ["ImageExists"],
        "reasons": ["coord out of range"],
    }

    html = render_training_section(
        plan_results=[],
        training_samples=[accepted_sample],
        rejected_entries=[rejected_entry],
    )

    assert "validator-detail" in html
    assert "Pass rate" in html
    assert "coord out of range" in html
    assert "data-reason=" in html


def test_training_section_rex_omni_syntax_highlight():
    from apps.data_flywheel.visualization.section_training import render_training_section

    sample = {
        "sample_id": "s1",
        "instruction": "pick the pen",
        "rex_omni_text": "<|user|>pick the pen<|end|><|thinking_start|>[Affordance] pen_0: pick<|thinking_end|><|action_start|>pick<|action_end|>",
        "image_path": "test.jpg",
    }

    html = render_training_section(
        plan_results=[],
        training_samples=[sample],
        rejected_entries=[],
    )

    assert "token-user" in html or "token-thinking" in html


def test_training_section_sample_preview_renders_thumbnail_overlay(tmp_path):
    from PIL import Image

    from apps.data_flywheel.visualization.section_training import render_training_section

    image_path = tmp_path / "preview.jpg"
    Image.new("RGB", (640, 480), (220, 220, 220)).save(image_path)
    sample = {
        "sample_id": "s2",
        "instruction": "pick the pen",
        "image_path": str(image_path),
        "image_size": (640, 480),
        "scene_objects": [
            {"id": "pen_0", "grasp_point_quantized": [500, 400]},
        ],
        "actions": [
            {"type": "pick", "target": "pen_0", "grasp_point_quantized": [500, 400]},
        ],
        "rex_omni_text": "<|user|>pick the pen<|end|>",
    }

    html = render_training_section(
        plan_results=[],
        training_samples=[sample],
        rejected_entries=[],
    )

    assert "sample-thumb" in html
    assert "quantized-overlay" in html
    assert "pen_0" in html


def test_generate_report_creates_html_file(tmp_path):
    from PIL import Image

    from apps.data_flywheel.visualization.report import generate_report

    node = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[10.0, 20.0, 100.0, 60.0],
        center_3d=[0.1, 0.2, 0.5],
        size_3d=[0.05, 0.01, 0.15],
        mean_depth=0.5,
        valid_3d=True,
        operation="pick",
        hierarchical_name="the pen",
        attributes=InstanceAttributes(color_primary="red"),
    )
    scene_graph = SceneGraph(
        image_id="test_001",
        image_path="test.jpg",
        image_size=(640, 480),
        nodes=[node],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    image = Image.new("RGB", (640, 480), (200, 200, 200))
    image_path = str(tmp_path / "test.jpg")
    image.save(image_path)

    report_path = generate_report(
        image_path=image_path,
        image_size=(640, 480),
        ontology_labels=[
            {
                "observation_name": "pen",
                "canonical_name": "pen",
                "operation": "pick",
                "extra_tags": [],
                "ontology_status": "resolved",
            }
        ],
        detections=[
            {
                "class_name": "pen",
                "bbox": [10.0, 20.0, 100.0, 60.0],
                "confidence": 0.95,
                "source": "detvlm",
            }
        ],
        segmentations=[],
        scene_graph=scene_graph,
        output_dir=str(tmp_path / "reports"),
    )

    assert report_path.endswith(".html")
    assert (tmp_path / "reports").exists()


def test_report_includes_support_blocking_and_placement_sections(tmp_path):
    from pathlib import Path

    from PIL import Image

    from apps.data_flywheel.visualization.report import generate_report

    image_path = tmp_path / "placement.jpg"
    Image.new("RGB", (640, 480), (210, 210, 220)).save(image_path)

    table = SceneNode(
        id="table_0",
        class_name="table",
        bbox=[0.0, 300.0, 640.0, 480.0],
        center_3d=[0.0, 0.0, 1.0],
        size_3d=[1.0, 1.0, 0.1],
        mean_depth=1.0,
        valid_3d=True,
        operation="push",
        role_ids=["support_surface", "base_support_root"],
        hierarchical_name="the table",
        attributes=InstanceAttributes(),
    )
    bowl = SceneNode(
        id="bowl_0",
        class_name="bowl",
        bbox=[420.0, 260.0, 540.0, 380.0],
        center_3d=[0.4, 0.0, 0.8],
        size_3d=[0.2, 0.2, 0.12],
        mean_depth=0.8,
        valid_3d=True,
        operation="pick",
        is_container=True,
        support_parent="table_0",
        support_type="on-top-of",
        hierarchical_name="the bowl",
        attributes=InstanceAttributes(),
    )
    pen = SceneNode(
        id="pen_0",
        class_name="pen",
        bbox=[120.0, 320.0, 240.0, 360.0],
        center_3d=[-0.2, 0.0, 0.75],
        size_3d=[0.16, 0.02, 0.02],
        mean_depth=0.75,
        valid_3d=True,
        operation="pick",
        support_parent="table_0",
        support_type="on-top-of",
        hierarchical_name="the pen",
        attributes=InstanceAttributes(color_primary="red"),
    )
    scene_graph = SceneGraph(
        image_id="placement_report",
        image_path=str(image_path),
        image_size=(640, 480),
        nodes=[table, bowl, pen],
        edges=[
            SceneEdge(
                subject="bowl_0",
                object="pen_0",
                predicate="occluding",
                source="2d",
                confidence=0.8,
            )
        ],
        ternary_relations=[],
        blocking_dag=[("bowl_0", "pen_0")],
        support_relations=[
            SupportRelation(subject="bowl_0", object="table_0", relation="on-top-of", confidence=0.9),
            SupportRelation(subject="pen_0", object="table_0", relation="on-top-of", confidence=0.9),
        ],
        placement_regions=[
            PlacementRegion(
                region_id="inside_bowl_0",
                region_type="inside-container",
                support_id="bowl_0",
                anchor_ids=["bowl_0"],
                polygon_2d=[[420.0, 260.0], [540.0, 260.0], [540.0, 380.0], [420.0, 380.0]],
                point_2d=[480.0, 320.0],
                reason="visible_container",
            ),
            PlacementRegion(
                region_id="free_table_0",
                region_type="free-space",
                support_id="table_0",
                anchor_ids=["table_0"],
                point_2d=[320.0, 400.0],
                reason="support_surface_free_space",
            ),
        ],
    )

    report_path = generate_report(
        image_path=str(image_path),
        image_size=(640, 480),
        ontology_labels=[],
        detections=[],
        segmentations=[],
        scene_graph=scene_graph,
        training_samples=[
            {
                "sample_id": "sample_1",
                "instruction": "把笔放到碗里",
                "image_path": str(image_path),
                "image_size": (640, 480),
                "actions": [
                    {
                        "type": "place",
                        "target": "pen_0",
                        "destination": "bowl_0",
                        "place_point_quantized": [749, 666],
                    }
                ],
                "rex_omni_text": "<|user|>把笔放到碗里<|end|>",
            }
        ],
        output_dir=str(tmp_path / "reports"),
    )

    html = Path(report_path).read_text(encoding="utf-8")

    assert "Support Graph" in html
    assert "Blocking DAG" in html
    assert "Placement Regions" in html
    assert "place point" in html.lower()

    with open(report_path, encoding="utf-8") as handle:
        content = handle.read()

    assert "<!DOCTYPE html>" in content
    assert "M1" in content
    assert "M2" in content
    assert "M3" in content
    assert "placement_report" in content


def test_generate_report_graceful_without_m4(tmp_path):
    from PIL import Image

    from apps.data_flywheel.visualization.report import generate_report

    scene_graph = SceneGraph(
        image_id="t",
        image_path="t.jpg",
        image_size=(640, 480),
        nodes=[],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )
    image = Image.new("RGB", (640, 480))
    image_path = str(tmp_path / "t.jpg")
    image.save(image_path)

    report_path = generate_report(
        image_path=image_path,
        image_size=(640, 480),
        ontology_labels=[],
        detections=[],
        segmentations=[],
        scene_graph=scene_graph,
        plan_results=None,
        training_samples=None,
        validation_report=None,
        output_dir=str(tmp_path / "reports"),
    )

    with open(report_path, encoding="utf-8") as handle:
        content = handle.read()

    assert "M4" in content
    assert "No data" in content or "无数据" in content


def test_generate_report_importable_from_visualization():
    from apps.data_flywheel.visualization import generate_report

    assert callable(generate_report)
