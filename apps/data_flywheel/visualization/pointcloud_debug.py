"""Dedicated point-cloud debug bundle generation."""

from __future__ import annotations

import base64
import io
import json
import math
import os
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from apps.data_flywheel.scene_understanding.pointcloud import load_xyz_map, rasterize_polygon
from apps.data_flywheel.types import SceneGraph, SceneNode
from apps.data_flywheel.visualizer import draw_depth_map

from .assets import build_html_skeleton


def _slug(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in (value or "scene"))
    token = token.strip("-")
    return token or "scene"


def _sample_rows(rows: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or rows.size == 0:
        return rows[:0]
    if len(rows) <= max_points:
        return rows
    indices = np.linspace(0, len(rows) - 1, num=max_points, dtype=int)
    return rows[indices]


def _sample_list(points: Sequence[Sequence[float]], max_points: int) -> List[List[float]]:
    if not points:
        return []
    array = np.asarray(points, dtype=np.float32)
    sampled = _sample_rows(array, max_points=max_points)
    return [[float(value) for value in row.tolist()] for row in sampled]


def _remap_polygon_to_raster(
    mask_polygon: Optional[List[List[float]]],
    *,
    source_image_size: Optional[Tuple[int, int]],
    target_width: int,
    target_height: int,
) -> Optional[List[List[float]]]:
    if not mask_polygon:
        return mask_polygon
    if not source_image_size:
        return mask_polygon
    source_width = float(source_image_size[0])
    source_height = float(source_image_size[1])
    if source_width <= 0 or source_height <= 0:
        return mask_polygon
    if int(round(source_width)) == target_width and int(round(source_height)) == target_height:
        return mask_polygon
    scale_x = float(target_width) / source_width
    scale_y = float(target_height) / source_height
    return [
        [float(point[0]) * scale_x, float(point[1]) * scale_y]
        for point in mask_polygon
    ]


def _image_to_data_uri(image: Image.Image, *, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{payload}"


def _save_image(image: Image.Image, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return str(output_path)


def _extract_raw_scene_points(xyz_map: Optional[np.ndarray], *, max_points: int = 12000) -> List[List[float]]:
    if xyz_map is None:
        return []
    points = np.asarray(xyz_map, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    sampled = _sample_rows(points[valid], max_points=max_points)
    return [[float(value) for value in row.tolist()] for row in sampled]


def _extract_node_debug(
    node: SceneNode,
    *,
    xyz_map: Optional[np.ndarray],
    source_image_size: Tuple[int, int],
    valid_preview_points: int = 2048,
    valid_preview_pixels: int = 900,
) -> Dict[str, Any]:
    sampled_points = _sample_list(node.point_cloud_sample or [], max_points=2048)
    base = {
        "node_id": node.id,
        "class_name": node.class_name,
        "hierarchical_name": node.hierarchical_name or node.class_name or node.id,
        "mask_source": node.mask_source or "unknown",
        "has_true_mask": bool(node.has_true_mask),
        "bbox": [float(value) for value in node.bbox],
        "sampled_point_count": len(node.point_cloud_sample or []),
        "sampled_points": sampled_points,
        "mask_pixel_count": 0,
        "valid_point_count": 0,
        "invalid_point_count": 0,
        "valid_ratio": 0.0,
        "valid_pixels_preview": [],
        "invalid_pixels_preview": [],
        "valid_points_preview": [],
        "debug_note": "",
    }
    if xyz_map is None:
        base["debug_note"] = "XYZ map unavailable"
        return base
    if not node.mask_polygon:
        base["debug_note"] = "Mask polygon unavailable"
        return base

    remapped_polygon = _remap_polygon_to_raster(
        node.mask_polygon,
        source_image_size=source_image_size,
        target_width=int(xyz_map.shape[1]),
        target_height=int(xyz_map.shape[0]),
    )
    mask = rasterize_polygon(
        remapped_polygon,
        width=int(xyz_map.shape[1]),
        height=int(xyz_map.shape[0]),
    )
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        base["debug_note"] = "Rasterized mask is empty"
        return base

    points = xyz_map[ys, xs]
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    valid_points = points[valid]
    valid_pixels = np.stack([xs[valid], ys[valid]], axis=1) if valid.any() else np.zeros((0, 2), dtype=np.int32)
    invalid_pixels = np.stack([xs[~valid], ys[~valid]], axis=1) if (~valid).any() else np.zeros((0, 2), dtype=np.int32)

    mask_pixel_count = int(len(xs))
    valid_point_count = int(len(valid_points))
    invalid_point_count = int(mask_pixel_count - valid_point_count)
    base.update(
        {
            "mask_pixel_count": mask_pixel_count,
            "valid_point_count": valid_point_count,
            "invalid_point_count": invalid_point_count,
            "valid_ratio": round(valid_point_count / max(mask_pixel_count, 1), 6),
            "valid_pixels_preview": [
                [int(value) for value in row.tolist()]
                for row in _sample_rows(valid_pixels.astype(np.int32), max_points=valid_preview_pixels)
            ],
            "invalid_pixels_preview": [
                [int(value) for value in row.tolist()]
                for row in _sample_rows(invalid_pixels.astype(np.int32), max_points=valid_preview_pixels)
            ],
            "valid_points_preview": [
                [float(value) for value in row.tolist()]
                for row in _sample_rows(valid_points.astype(np.float32), max_points=valid_preview_points)
            ],
            "debug_note": "ok" if valid_point_count else "No valid XYZ points inside mask",
        }
    )
    return base


def _node_overlay_image(image: Image.Image, node: SceneNode, node_debug: Dict[str, Any]) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")

    if node.mask_polygon:
        polygon = [(float(point[0]), float(point[1])) for point in node.mask_polygon]
        draw.polygon(polygon, fill=(59, 130, 246, 45), outline=(37, 99, 235, 220))

    bbox = [float(value) for value in node.bbox]
    draw.rectangle(bbox, outline=(250, 204, 21, 255), width=2)

    for x, y in node_debug.get("invalid_pixels_preview", []):
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(239, 68, 68, 180))
    for x, y in node_debug.get("valid_pixels_preview", []):
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(34, 197, 94, 220))

    label = (
        f"{node.id} | mask={node_debug['mask_pixel_count']} | "
        f"valid={node_debug['valid_point_count']} | sampled={node_debug['sampled_point_count']}"
    )
    draw.rectangle([8, 8, 8 + len(label) * 7, 28], fill=(15, 23, 42, 220))
    draw.text((12, 11), label, fill=(255, 255, 255, 255))
    return canvas.convert("RGB")


def _normalize_debug_payload(
    raw_scene_points: List[List[float]],
    node_debugs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    all_points: List[List[float]] = []
    all_points.extend(raw_scene_points)
    for node_debug in node_debugs:
        all_points.extend(node_debug.get("valid_points_preview", []))
        all_points.extend(node_debug.get("sampled_points", []))
    if not all_points:
        return {"raw_scene_points": [], "nodes": [], "scene": {"scene_radius": 1.0}}

    coords = np.asarray(all_points, dtype=np.float32)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2.0
    extents = np.maximum(maxs - mins, 1e-6)
    max_extent = float(np.max(extents))
    scale = 2.4 / max_extent if max_extent > 0 else 1.0

    def _transform(points: Iterable[Sequence[float]]) -> List[List[float]]:
        output: List[List[float]] = []
        for point in points:
            output.append(
                [
                    round((float(point[0]) - float(center[0])) * scale, 6),
                    round((float(point[1]) - float(center[1])) * scale, 6),
                    round((float(point[2]) - float(center[2])) * scale, 6),
                ]
            )
        return output

    return {
        "raw_scene_points": _transform(raw_scene_points),
        "nodes": [
            {
                "node_id": node_debug["node_id"],
                "label": node_debug["hierarchical_name"],
                "valid_points_preview": _transform(node_debug.get("valid_points_preview", [])),
                "sampled_points": _transform(node_debug.get("sampled_points", [])),
                "valid_point_count": node_debug["valid_point_count"],
                "sampled_point_count": node_debug["sampled_point_count"],
                "debug_note": node_debug["debug_note"],
            }
            for node_debug in node_debugs
        ],
        "scene": {
            "center_3d": [round(float(value), 6) for value in center.tolist()],
            "extents_3d": [round(float(value), 6) for value in extents.tolist()],
            "scale": round(float(scale), 8),
            "scene_radius": round(max(math.sqrt(sum(((float(extent) * scale) / 2.0) ** 2 for extent in extents)), 1.0), 6),
        },
    }


def _render_overview_card(summary: Dict[str, Any], heatmap_relpath: Optional[str]) -> str:
    rows = [
        ("Image ID", summary["image_id"]),
        ("Image Path", summary["image_path"]),
        ("XYZ Map", summary.get("xyz_map_path") or "—"),
        ("Raw XYZ Shape", str(summary.get("raw_xyz_shape"))),
        ("Raw Scene Sampled Points", str(summary.get("raw_scene_sampled_points", 0))),
        ("Node Count", str(summary.get("node_count", 0))),
    ]
    table_rows = "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(str(value))}</td></tr>" for key, value in rows
    )
    preview = ""
    if heatmap_relpath:
        preview = (
            '<div class="card">'
            "<h3>Raw XYZ / Depth</h3>"
            '<p style="margin:0 0 12px 0;color:#475569;font-size:13px;">'
            "This is the raw Z heatmap from the saved XYZ map before any mask filtering or point-cloud sampling."
            "</p>"
            f'<img src="{escape(heatmap_relpath)}" alt="depth heatmap" style="width:100%;max-width:900px;border:1px solid #d1d5db;border-radius:8px;" />'
            "</div>"
        )
    else:
        preview = (
            '<div class="card"><h3>Raw XYZ / Depth</h3>'
            '<p style="margin:0;color:#b91c1c;font-size:13px;">XYZ map unavailable.</p></div>'
        )
    return (
        '<div class="card">'
        "<h2>Point-Cloud Debug</h2>"
        '<p style="margin:0 0 12px 0;color:#475569;font-size:13px;">'
        "This bundle is for debugging the point-cloud path specifically: raw XYZ map -> rasterized mask -> valid XYZ pixels inside the mask -> sampled scene-graph point cloud."
        "</p>"
        "<table>"
        f"{table_rows}"
        "</table>"
        "</div>"
        f"{preview}"
    )


def _render_node_cards(node_debugs: List[Dict[str, Any]]) -> str:
    if not node_debugs:
        return '<div class="card"><h3>Mask -> Valid XYZ</h3><p>No nodes.</p></div>'
    cards: List[str] = []
    for node_debug in node_debugs:
        overlay_path = node_debug.get("overlay_relpath")
        metrics = (
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px 14px;font-size:13px;">'
            f'<div><strong>Mask pixels</strong><br>{node_debug["mask_pixel_count"]}</div>'
            f'<div><strong>Valid XYZ</strong><br>{node_debug["valid_point_count"]}</div>'
            f'<div><strong>Invalid masked pixels</strong><br>{node_debug["invalid_point_count"]}</div>'
            f'<div><strong>Sampled scene-graph point cloud</strong><br>{node_debug["sampled_point_count"]}</div>'
            f'<div><strong>Valid ratio</strong><br>{node_debug["valid_ratio"]:.3f}</div>'
            f'<div><strong>Mask source</strong><br>{escape(str(node_debug["mask_source"]))}</div>'
            "</div>"
        )
        cards.append(
            '<div class="card">'
            f'<h3>{escape(node_debug["node_id"])} — {escape(node_debug["hierarchical_name"])}</h3>'
            f'<p style="margin:0 0 10px 0;color:#475569;font-size:13px;">{escape(node_debug["debug_note"])}</p>'
            f"{metrics}"
            + (
                f'<img src="{escape(str(overlay_path))}" alt="{escape(node_debug["node_id"])} overlay" '
                'style="width:100%;max-width:920px;margin-top:12px;border:1px solid #d1d5db;border-radius:8px;" />'
                if overlay_path
                else ""
            )
            + "</div>"
        )
    return '<div class="card"><h3>Mask -> Valid XYZ</h3><p style="margin:0;color:#475569;font-size:13px;">Green dots are masked pixels with valid XYZ values. Red dots are masked pixels with invalid / non-positive depth. The overlay is before final point sampling.</p></div>' + "".join(cards)


def _render_3d_debug_card(payload: Dict[str, Any], dom_prefix: str) -> str:
    data_id = f"{dom_prefix}-pointcloud-debug-data"
    container_id = f"{dom_prefix}-pointcloud-debug-view"
    select_id = f"{dom_prefix}-pointcloud-debug-select"
    raw_toggle_id = f"{dom_prefix}-pointcloud-debug-toggle-raw"
    valid_toggle_id = f"{dom_prefix}-pointcloud-debug-toggle-valid"
    sampled_toggle_id = f"{dom_prefix}-pointcloud-debug-toggle-sampled"
    details_id = f"{dom_prefix}-pointcloud-debug-details"
    payload_json = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")

    options = "".join(
        f'<option value="{escape(str(item["node_id"]))}">{escape(str(item["label"]))}</option>'
        for item in payload.get("nodes", [])
    )
    return f"""
<div class="card">
  <h3>3D Debug View</h3>
  <p style="margin:0 0 12px 0;color:#475569;font-size:13px;">Gray = raw XYZ scene sample from the saved XYZ map. Cyan = valid XYZ points inside the rasterized mask before downsampling. Orange = the final sampled scene-graph point cloud.</p>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:12px;">
    <label>Node
      <select id="{select_id}">
        {options}
      </select>
    </label>
    <label><input type="checkbox" checked id="{raw_toggle_id}"> Show raw scene sample</label>
    <label><input type="checkbox" checked id="{valid_toggle_id}"> Show valid mask cloud</label>
    <label><input type="checkbox" checked id="{sampled_toggle_id}"> Show sampled scene-graph cloud</label>
  </div>
  <div id="{details_id}" style="margin:0 0 12px 0;color:#475569;font-size:13px;"></div>
  <div id="{container_id}" style="width:100%;height:540px;border-radius:10px;overflow:hidden;background:#0f172a;"></div>
  <script type="application/json" id="{data_id}">{payload_json}</script>
  <script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

const data = JSON.parse(document.getElementById({json.dumps(data_id)}).textContent);
const container = document.getElementById({json.dumps(container_id)});
const select = document.getElementById({json.dumps(select_id)});
const rawToggle = document.getElementById({json.dumps(raw_toggle_id)});
const validToggle = document.getElementById({json.dumps(valid_toggle_id)});
const sampledToggle = document.getElementById({json.dumps(sampled_toggle_id)});
const details = document.getElementById({json.dumps(details_id)});
if (container) {{
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#0f172a");
  const radius = Math.max(data.scene?.scene_radius || 1, 1);
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 100);
  camera.position.set(radius * 1.8, -radius * 1.25, radius * 3.0);
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);

  scene.add(new THREE.AxesHelper(1.2));
  scene.add(new THREE.GridHelper(radius * 4, 8, 0x64748b, 0x1e293b));
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));
  const light = new THREE.DirectionalLight(0xffffff, 0.9);
  light.position.set(2.0, 2.0, 3.5);
  scene.add(light);

  function makePoints(points, color, size, opacity) {{
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(points.flat()), 3));
    const material = new THREE.PointsMaterial({{ color, size, sizeAttenuation: true, transparent: true, opacity }});
    return new THREE.Points(geometry, material);
  }}

  let rawScene = null;
  if (Array.isArray(data.raw_scene_points) && data.raw_scene_points.length) {{
    rawScene = makePoints(data.raw_scene_points, "#cbd5e1", 0.012, 0.42);
    scene.add(rawScene);
  }}

  const nodeState = new Map();
  for (const node of data.nodes || []) {{
    const validCloud = Array.isArray(node.valid_points_preview) && node.valid_points_preview.length
      ? makePoints(node.valid_points_preview, "#22d3ee", 0.02, 0.88)
      : null;
    const sampledCloud = Array.isArray(node.sampled_points) && node.sampled_points.length
      ? makePoints(node.sampled_points, "#f59e0b", 0.045, 0.98)
      : null;
    if (validCloud) scene.add(validCloud);
    if (sampledCloud) scene.add(sampledCloud);
    nodeState.set(node.node_id, {{ node, validCloud, sampledCloud }});
  }}

  function applyState() {{
    const selectedId = select && select.value ? select.value : (data.nodes[0] ? data.nodes[0].node_id : "");
    for (const [nodeId, state] of nodeState.entries()) {{
      const visible = nodeId === selectedId;
      if (state.validCloud) state.validCloud.visible = visible && !!(validToggle && validToggle.checked);
      if (state.sampledCloud) state.sampledCloud.visible = visible && !!(sampledToggle && sampledToggle.checked);
      if (visible && details) {{
        details.innerHTML =
          `<strong>${{state.node.label}}</strong> · valid mask XYZ ${{state.node.valid_point_count}} · sampled scene-graph points ${{state.node.sampled_point_count}} · note: ${{state.node.debug_note}}`;
      }}
    }}
    if (rawScene) rawScene.visible = !!(rawToggle && rawToggle.checked);
  }}

  function resize() {{
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }}

  if (select) select.addEventListener("change", applyState);
  if (rawToggle) rawToggle.addEventListener("change", applyState);
  if (validToggle) validToggle.addEventListener("change", applyState);
  if (sampledToggle) sampledToggle.addEventListener("change", applyState);
  window.addEventListener("resize", resize);
  resize();
  applyState();

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }}
  animate();
}}
  </script>
</div>
"""


def generate_pointcloud_debug_bundle(
    *,
    image_path: str,
    scene_graph: SceneGraph,
    depth_info: Optional[Dict[str, Any]],
    output_dir: str = "data/output/pointcloud_debug",
) -> Dict[str, str]:
    """Generate a dedicated point-cloud debug bundle for one scene."""

    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")

    xyz_map = load_xyz_map(depth_info)
    raw_scene_points = _extract_raw_scene_points(xyz_map, max_points=12000)

    heatmap_path = ""
    heatmap_relpath: Optional[str] = None
    if xyz_map is not None:
        heatmap = draw_depth_map(xyz_map, image_path=image_path, alpha=0.55)
        heatmap_path = _save_image(heatmap, bundle_dir / "depth_heatmap.png")
        heatmap_relpath = os.path.relpath(heatmap_path, bundle_dir)

    node_debugs: List[Dict[str, Any]] = []
    objects_dir = bundle_dir / "objects"
    for node in scene_graph.nodes:
        node_debug = _extract_node_debug(
            node,
            xyz_map=xyz_map,
            source_image_size=scene_graph.image_size,
        )
        overlay_path = _save_image(
            _node_overlay_image(image, node, node_debug),
            objects_dir / f"{node.id}_mask_debug.png",
        )
        node_debug["overlay_path"] = overlay_path
        node_debug["overlay_relpath"] = os.path.relpath(overlay_path, bundle_dir)
        node_debugs.append(node_debug)

    summary = {
        "image_id": scene_graph.image_id,
        "image_path": image_path,
        "xyz_map_path": depth_info.get("xyz_map_path") if depth_info else None,
        "raw_xyz_shape": list(xyz_map.shape) if xyz_map is not None else None,
        "raw_scene_sampled_points": len(raw_scene_points),
        "node_count": len(scene_graph.nodes),
        "nodes": [
            {
                "node_id": node_debug["node_id"],
                "class_name": node_debug["class_name"],
                "hierarchical_name": node_debug["hierarchical_name"],
                "mask_source": node_debug["mask_source"],
                "has_true_mask": node_debug["has_true_mask"],
                "mask_pixel_count": node_debug["mask_pixel_count"],
                "valid_point_count": node_debug["valid_point_count"],
                "invalid_point_count": node_debug["invalid_point_count"],
                "sampled_point_count": node_debug["sampled_point_count"],
                "valid_ratio": node_debug["valid_ratio"],
                "debug_note": node_debug["debug_note"],
            }
            for node_debug in node_debugs
        ],
    }
    summary_path = bundle_dir / "debug_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = _normalize_debug_payload(raw_scene_points=raw_scene_points, node_debugs=node_debugs)
    overview_html = _render_overview_card(summary, heatmap_relpath)
    nodes_html = _render_node_cards(node_debugs)
    debug_3d_html = _render_3d_debug_card(payload, dom_prefix=_slug(scene_graph.image_id))
    body = overview_html + nodes_html + debug_3d_html
    html = build_html_skeleton(
        title=f"Point-Cloud Debug: {scene_graph.image_id}",
        body_sections=body,
    )
    html_path = bundle_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")

    return {
        "html_path": str(html_path),
        "summary_path": str(summary_path),
        "depth_heatmap_path": heatmap_path,
    }
