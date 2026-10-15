import json
from pathlib import Path

import numpy as np
from PIL import Image

from apps.data_flywheel.types import InstanceAttributes, SceneGraph, SceneNode


def test_generate_pointcloud_debug_bundle_outputs_expected_files_and_sections(tmp_path):
    from apps.data_flywheel.visualization.pointcloud_debug import generate_pointcloud_debug_bundle

    image_path = tmp_path / "scene.png"
    Image.new("RGB", (5, 5), color=(220, 220, 220)).save(image_path)

    xyz_map = np.zeros((5, 5, 3), dtype=np.float32)
    for y in range(1, 4):
        for x in range(1, 4):
            xyz_map[y, x] = [float(x), float(y), 1.0 + 0.1 * x + 0.01 * y]
    xyz_map_path = tmp_path / "scene_xyz.npy"
    np.save(xyz_map_path, xyz_map)

    node = SceneNode(
        id="plate_0",
        class_name="plate",
        bbox=[1.0, 1.0, 3.0, 3.0],
        mask_polygon=[[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]],
        mask_source="mask_polygon",
        has_true_mask=True,
        center_3d=[2.0, 2.0, 1.22],
        size_3d=[2.0, 2.0, 0.2],
        mean_depth=1.22,
        valid_3d=True,
        operation="pick",
        hierarchical_name="the plate",
        point_cloud_sample=[[1.0, 1.0, 1.11], [3.0, 3.0, 1.33]],
        attributes=InstanceAttributes(color_primary="white"),
    )
    scene_graph = SceneGraph(
        image_id="scene",
        image_path=str(image_path),
        image_size=(5, 5),
        nodes=[node],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )
    depth_info = {
        "image": str(image_path),
        "image_size": [5, 5],
        "depth_valid": True,
        "xyz_map_path": str(xyz_map_path),
    }

    bundle_dir = tmp_path / "debug_bundle"
    result = generate_pointcloud_debug_bundle(
        image_path=str(image_path),
        scene_graph=scene_graph,
        depth_info=depth_info,
        output_dir=str(bundle_dir),
    )

    html_path = Path(result["html_path"])
    summary_path = Path(result["summary_path"])
    heatmap_path = Path(result["depth_heatmap_path"])
    assert html_path.exists()
    assert summary_path.exists()
    assert heatmap_path.exists()

    html = html_path.read_text(encoding="utf-8")
    assert "Point-Cloud Debug" in html
    assert "Raw XYZ / Depth" in html
    assert "Mask -> Valid XYZ" in html
    assert "Sampled scene-graph point cloud" in html
    assert "plate_0" in html
    assert "raw_scene_points" in html
    assert "valid_points_preview" in html

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["image_id"] == "scene"
    assert summary["raw_xyz_shape"] == [5, 5, 3]
    assert summary["node_count"] == 1
    node_summary = summary["nodes"][0]
    assert node_summary["node_id"] == "plate_0"
    assert node_summary["sampled_point_count"] == 2
    assert node_summary["valid_point_count"] >= node_summary["sampled_point_count"]
    assert node_summary["mask_pixel_count"] >= node_summary["valid_point_count"]


def test_generate_pointcloud_debug_bundle_reports_missing_xyz_map(tmp_path):
    from apps.data_flywheel.visualization.pointcloud_debug import generate_pointcloud_debug_bundle

    image_path = tmp_path / "scene.png"
    Image.new("RGB", (8, 8), color=(200, 200, 200)).save(image_path)

    node = SceneNode(
        id="cup_0",
        class_name="cup",
        bbox=[1.0, 1.0, 6.0, 6.0],
        operation="pick",
        hierarchical_name="the cup",
        attributes=InstanceAttributes(),
    )
    scene_graph = SceneGraph(
        image_id="missing_xyz",
        image_path=str(image_path),
        image_size=(8, 8),
        nodes=[node],
        edges=[],
        ternary_relations=[],
        blocking_dag=[],
    )

    bundle_dir = tmp_path / "missing_bundle"
    result = generate_pointcloud_debug_bundle(
        image_path=str(image_path),
        scene_graph=scene_graph,
        depth_info={"image": str(image_path), "depth_valid": False, "xyz_map_path": str(tmp_path / "missing.npy")},
        output_dir=str(bundle_dir),
    )

    html = Path(result["html_path"]).read_text(encoding="utf-8")
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))

    assert "XYZ map unavailable" in html
    assert summary["raw_xyz_shape"] is None
    assert summary["nodes"][0]["valid_point_count"] == 0
