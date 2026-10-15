"""M3 Three.js 3D scene renderer."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional

from apps.data_flywheel.types import SceneGraph

from .assets import OPERATION_COLORS, RELATION_COLORS


def _slug(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", value or "scene")
    token = token.strip("-")
    return token or "scene"


def _node_to_payload(node: Any) -> Dict[str, Any]:
    point_cloud = list(getattr(node, "point_cloud_sample", []) or [])
    center_3d, size_3d = _resolve_render_pose(node, point_cloud)
    return {
        "id": node.id,
        "class_name": node.class_name,
        "center_3d": center_3d,
        "size_3d": size_3d,
        "operation": node.operation,
        "hierarchical_name": node.hierarchical_name or node.id,
        "mean_depth": node.mean_depth,
        "canonical_name": getattr(node, "canonical_name", None),
        "extra_tags": list(getattr(node, "extra_tags", []) or []),
        "referring_expressions": list(getattr(node, "referring_expressions", []) or []),
        "point_cloud_sample": point_cloud,
        "support_parent": getattr(node, "support_parent", None),
        "support_type": getattr(node, "support_type", None),
        "support_excluded": bool(getattr(node, "support_excluded", False)),
    }


def _edge_to_payload(edge: Any) -> Dict[str, Any]:
    return {
        "subject": edge.subject,
        "object": edge.object,
        "predicate": edge.predicate,
        "source": edge.source,
        "confidence": edge.confidence,
    }


def _blocking_to_payload(blocking_dag: List[tuple[str, str]]) -> List[Dict[str, str]]:
    return [{"blocker": blocker, "blocked": blocked} for blocker, blocked in blocking_dag]


def _support_to_payload(support_relations: List[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "subject": rel.subject,
            "object": rel.object,
            "relation": rel.relation,
            "confidence": rel.confidence,
        }
        for rel in support_relations
    ]


def _derive_pose_from_points(point_cloud: List[List[float]]) -> Optional[tuple[List[float], List[float]]]:
    if not point_cloud:
        return None
    coords = [point for point in point_cloud if isinstance(point, list) and len(point) >= 3]
    if not coords:
        return None
    xs = [float(point[0]) for point in coords]
    ys = [float(point[1]) for point in coords]
    zs = [float(point[2]) for point in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    center = [
        (min_x + max_x) / 2.0,
        (min_y + max_y) / 2.0,
        (min_z + max_z) / 2.0,
    ]
    size = [
        max(max_x - min_x, 0.02),
        max(max_y - min_y, 0.02),
        max(max_z - min_z, 0.02),
    ]
    return center, size


def _resolve_render_pose(node: Any, point_cloud: List[List[float]]) -> tuple[Optional[List[float]], Optional[List[float]]]:
    if node.center_3d and node.size_3d:
        return list(node.center_3d), list(node.size_3d)
    derived = _derive_pose_from_points(point_cloud)
    if derived is not None:
        return derived
    return None, None


def _json_script_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")


_EDGE_PRIORITY = {
    "blocking": 0,
    "inside": 1,
    "on-top-of": 2,
    "stacked-on": 2,
    "supporting": 2,
    "occluding": 3,
    "occluded-by": 3,
    "overlapping": 4,
    "front": 5,
    "behind": 5,
    "above": 6,
    "below": 6,
}


def _select_3d_edges(scene_graph: SceneGraph) -> List[Any]:
    grouped: Dict[tuple[str, str], tuple[int, float, Any]] = {}
    for edge in scene_graph.edges:
        if edge.source not in {"3d", "2d+3d"}:
            continue
        priority = _EDGE_PRIORITY.get(edge.predicate)
        if priority is None or edge.subject == edge.object:
            continue
        key = tuple(sorted((edge.subject, edge.object)))
        score = (priority, -float(edge.confidence))
        current = grouped.get(key)
        if current is None or score < (current[0], current[1]):
            grouped[key] = (priority, -float(edge.confidence), edge)

    limit = max(10, min(24, len(scene_graph.nodes) * 2))
    selected = [item[2] for item in sorted(grouped.values(), key=lambda item: (item[0], item[1], item[2].predicate))]
    return selected[:limit]


def _normalized_scene_payload(nodes_3d: List[Any]) -> Dict[str, Any]:
    min_corner = [float("inf"), float("inf"), float("inf")]
    max_corner = [float("-inf"), float("-inf"), float("-inf")]
    raw_nodes = [_node_to_payload(node) for node in nodes_3d]

    for item in raw_nodes:
        center = item["center_3d"]
        size = item["size_3d"]
        for idx in range(3):
            half = float(size[idx]) / 2.0
            min_corner[idx] = min(min_corner[idx], float(center[idx]) - half)
            max_corner[idx] = max(max_corner[idx], float(center[idx]) + half)

    scene_center = [(mn + mx) / 2.0 for mn, mx in zip(min_corner, max_corner)]
    extents = [max(mx - mn, 1.0) for mn, mx in zip(min_corner, max_corner)]
    max_extent = max(extents)
    scale = 2.4 / max_extent if max_extent > 0 else 1.0

    for item in raw_nodes:
        item["render_center"] = [
            round((float(value) - scene_center[idx]) * scale, 6)
            for idx, value in enumerate(item["center_3d"])
        ]
        item["render_size"] = [
            max(round(float(value) * scale, 6), 0.04)
            for value in item["size_3d"]
        ]
        item["render_point_cloud_sample"] = [
            [
                round((float(point[idx]) - scene_center[idx]) * scale, 6)
                for idx in range(3)
            ]
            for point in item["point_cloud_sample"]
            if isinstance(point, list) and len(point) >= 3
        ]

    global_point_cloud_sample: List[List[float]] = []
    for item in raw_nodes:
        global_point_cloud_sample.extend(item["render_point_cloud_sample"])
    if len(global_point_cloud_sample) > 4000:
        step = max(1, math.ceil(len(global_point_cloud_sample) / 4000))
        global_point_cloud_sample = global_point_cloud_sample[::step]

    scene_radius = max(
        math.sqrt(sum((extent * scale / 2.0) ** 2 for extent in extents)),
        0.75,
    )
    return {
        "nodes": raw_nodes,
        "global_point_cloud_sample": global_point_cloud_sample,
        "scene": {
            "center_3d": [round(value, 6) for value in scene_center],
            "extents_3d": [round(value, 6) for value in extents],
            "scale": round(scale, 8),
            "scene_radius": round(scene_radius, 6),
        },
    }


def render_three_scene(scene_graph: SceneGraph) -> str:
    """Render a self-contained HTML fragment for the 3D scene section."""

    nodes_3d = []
    for node in scene_graph.nodes:
        point_cloud = list(getattr(node, "point_cloud_sample", []) or [])
        center_3d, size_3d = _resolve_render_pose(node, point_cloud)
        if center_3d is None or size_3d is None:
            continue
        nodes_3d.append(node)
    if not nodes_3d:
        return '<div class="card"><h3>3D Scene</h3><p>No 3D data</p></div>'

    payload = {
        **_normalized_scene_payload(nodes_3d),
        "edges": [
            _edge_to_payload(edge)
            for edge in _select_3d_edges(scene_graph)
        ],
        "blocking_dag": _blocking_to_payload(scene_graph.blocking_dag),
        "support_relations": _support_to_payload(scene_graph.support_relations),
    }

    prefix = _slug(getattr(scene_graph, "image_id", "scene"))
    container_id = f"{prefix}-scene-3d-container"
    data_id = f"{prefix}-scene-3d-data"
    relation_toggle_id = f"{prefix}-toggle-relations"
    label_toggle_id = f"{prefix}-toggle-labels"
    blocking_toggle_id = f"{prefix}-toggle-blocking"
    global_toggle_id = f"{prefix}-toggle-global-cloud"
    box_toggle_id = f"{prefix}-toggle-boxes"
    tooltip_id = f"{prefix}-scene-3d-tooltip"
    details_id = f"{prefix}-scene-3d-details-panel"

    payload_json = _json_script_payload(payload)
    op_colors = json.dumps(
        {
            "pick": OPERATION_COLORS["pick"],
            "push": OPERATION_COLORS["push"],
            "non_operable": OPERATION_COLORS["non_operable"],
            "unknown": OPERATION_COLORS[None],
        },
        ensure_ascii=True,
    )
    relation_colors = json.dumps(
        {
            "stacked-on": RELATION_COLORS["stacked-on"],
            "supporting": RELATION_COLORS["supporting"],
            "occluding": RELATION_COLORS["occluding"],
            "occluded-by": RELATION_COLORS["occluded-by"],
            "inside": RELATION_COLORS["inside"],
            "outside": RELATION_COLORS["outside"],
            "blocking": RELATION_COLORS["blocking"],
        },
        ensure_ascii=True,
    )

    return f"""
<div class="card">
  <h3>B. 3D Point-Cloud-First View</h3>
  <p style="margin:0 0 12px 0;color:#475569;font-size:13px;">Point-cloud-first review view: the muted global scene cloud stays visible by default, instance point clouds are individually colored, and boxes are debug overlays only.</p>
  <div id="{container_id}" style="width:100%;height:500px;position:relative;background:linear-gradient(180deg,#0f172a 0%,#111827 100%);border-radius:8px;overflow:hidden;">
    <div class="three-controls" style="position:absolute;top:8px;right:8px;z-index:10;background:rgba(15,23,42,0.78);padding:10px 12px;border-radius:8px;border:1px solid rgba(255,255,255,0.12);">
      <div style="color:#e2e8f0;font-size:12px;font-weight:700;margin-bottom:8px;">Point-Cloud-First Controls</div>
      <label style="color:#fff;font-size:12px;display:block;margin:4px 0;cursor:pointer;">
        <input type="checkbox" checked id="{relation_toggle_id}"> Show relations
      </label>
      <label style="color:#fff;font-size:12px;display:block;margin:4px 0;cursor:pointer;">
        <input type="checkbox" checked id="{label_toggle_id}"> Show labels
      </label>
      <label style="color:#fff;font-size:12px;display:block;margin:4px 0;cursor:pointer;">
        <input type="checkbox" checked id="{blocking_toggle_id}"> Show blocking
      </label>
      <label style="color:#fff;font-size:12px;display:block;margin:4px 0;cursor:pointer;">
        <input type="checkbox" checked id="{global_toggle_id}"> Show global cloud
      </label>
      <label style="color:#fff;font-size:12px;display:block;margin:4px 0;cursor:pointer;">
        <input type="checkbox" id="{box_toggle_id}"> Show boxes
      </label>
    </div>
    <div style="position:absolute;left:260px;top:8px;z-index:10;display:flex;gap:8px;flex-wrap:wrap;max-width:calc(100% - 520px);">
      <span style="display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;background:rgba(15,23,42,0.74);color:#e2e8f0;border:1px solid rgba(255,255,255,0.12);font-size:12px;"><span style="width:10px;height:10px;border-radius:999px;background:#cbd5e1;display:inline-block;"></span> Global scene cloud</span>
      <span style="display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;background:rgba(15,23,42,0.74);color:#e2e8f0;border:1px solid rgba(255,255,255,0.12);font-size:12px;"><span style="width:10px;height:10px;border-radius:999px;background:#38bdf8;display:inline-block;"></span> Object point clouds</span>
      <span style="display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;background:rgba(15,23,42,0.74);color:#e2e8f0;border:1px solid rgba(255,255,255,0.12);font-size:12px;">Boxes are debug overlays only</span>
    </div>
    <div id="{tooltip_id}" class="tooltip" style="display:none;position:absolute;z-index:20;padding:6px 8px;background:rgba(15,23,42,0.92);color:#fff;border:1px solid rgba(255,255,255,0.12);border-radius:6px;font-size:12px;pointer-events:none;max-width:220px;"></div>
    <div id="{details_id}" class="details-panel" style="position:absolute;left:8px;top:8px;z-index:10;width:240px;max-height:calc(100% - 16px);overflow:auto;background:rgba(15,23,42,0.82);color:#e5e7eb;padding:12px;border-radius:8px;border:1px solid rgba(255,255,255,0.12);font-size:12px;">
      <strong>Selection</strong>
      <p style="margin:8px 0 0 0;color:#94a3b8;">Click an object to inspect its 3D node details. Global scene cloud and object point clouds are enabled by default.</p>
    </div>
    <script type="application/json" id="{data_id}">{payload_json}</script>
    <script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
import {{ CSS2DRenderer, CSS2DObject }} from "three/addons/renderers/CSS2DRenderer.js";

const container = document.getElementById({json.dumps(container_id)});
const dataEl = document.getElementById({json.dumps(data_id)});
if (container && dataEl) {{
  const data = JSON.parse(dataEl.textContent);
  const operationColors = {op_colors};
  const relationColors = {relation_colors};
  const tooltip = document.getElementById({json.dumps(tooltip_id)});
  const detailsPanel = document.getElementById({json.dumps(details_id)});

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#0f172a");

  const sceneRadius = Math.max(data.scene?.scene_radius || 1, 1);
  const cameraDistance = Math.max(sceneRadius * 3.2, 3.5);
  const globalPointSize = Math.max(sceneRadius * 0.014, 0.02);
  const defaultPointSize = Math.max(sceneRadius * 0.02, 0.032);
  const hoverPointSize = defaultPointSize * 1.25;
  const neighborPointSize = defaultPointSize * 1.12;
  const selectedPointSize = defaultPointSize * 1.4;
  const camera = new THREE.PerspectiveCamera(
    45,
    container.clientWidth / container.clientHeight,
    Math.max(cameraDistance / 100, 0.01),
    Math.max(cameraDistance * 20, 20),
  );
  camera.position.set(sceneRadius * 1.8, -sceneRadius * 1.25, cameraDistance);

  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(container.clientWidth, container.clientHeight);
  labelRenderer.domElement.style.position = "absolute";
  labelRenderer.domElement.style.top = "0";
  labelRenderer.domElement.style.pointerEvents = "none";
  container.appendChild(labelRenderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);

  scene.add(new THREE.AxesHelper(1.2));
  scene.add(new THREE.GridHelper(sceneRadius * 4, 8, 0x64748b, 0x1e293b));
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));
  const directional = new THREE.DirectionalLight(0xffffff, 0.9);
  directional.position.set(2.5, 2.0, 3.5);
  scene.add(directional);

  let globalPointCloud = null;
  let globalPointsMaterial = null;
  if (Array.isArray(data.global_point_cloud_sample) && data.global_point_cloud_sample.length) {{
    const globalGeometry = new THREE.BufferGeometry();
    const globalPositions = new Float32Array(data.global_point_cloud_sample.flat());
    globalGeometry.setAttribute("position", new THREE.BufferAttribute(globalPositions, 3));
    globalPointsMaterial = new THREE.PointsMaterial({{ color: "#cbd5e1", size: globalPointSize, sizeAttenuation: true, transparent: true, opacity: 0.48 }});
    globalPointCloud = new THREE.Points(globalGeometry, globalPointsMaterial);
    scene.add(globalPointCloud);
  }}

  const nodeCenters = new Map();
  const relationLines = [];
  const blockingArrows = [];
  const nodeBoxes = [];
  const pointClouds = [];
  const labels = [];
  const interactiveBoxes = [];
  const interactiveTargets = [];
  const raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.06;
  const pointer = new THREE.Vector2();
  let hoveredItem = null;
  let selectedItem = null;

  function setItemState(item, state) {{
    const baseColor = new THREE.Color(operationColors[item.data.operation] || operationColors.unknown);
    const pointColor = item.points && item.points.material && item.points.material.color;
    const pointMaterial = item.points && item.points.material;
    if (state === "selected") {{
      item.box.material.opacity = 0.55;
      item.outline.material.color.set("#ffffff");
      if (pointColor) pointColor.set("#ffffff");
      if (pointMaterial) pointMaterial.size = selectedPointSize;
      return;
    }}
    if (state === "neighbor") {{
      item.box.material.opacity = 0.38;
      item.outline.material.color.set("#60a5fa");
      if (pointColor) pointColor.set("#60a5fa");
      if (pointMaterial) pointMaterial.size = neighborPointSize;
      return;
    }}
    if (state === "hover") {{
      item.box.material.opacity = 0.45;
      item.outline.material.color.set("#fde68a");
      if (pointColor) pointColor.set("#fde68a");
      if (pointMaterial) pointMaterial.size = hoverPointSize;
      return;
    }}
    item.box.material.opacity = 0.3;
    item.outline.material.color.copy(baseColor);
    if (pointColor) pointColor.copy(baseColor);
    if (pointMaterial) pointMaterial.size = defaultPointSize;
  }}

  function renderDetails(node) {{
    if (!detailsPanel) return;
    const expressions = Array.isArray(node.referring_expressions) && node.referring_expressions.length
      ? node.referring_expressions.join(", ")
      : "-";
    const tags = Array.isArray(node.extra_tags) && node.extra_tags.length ? node.extra_tags.join(", ") : "-";
    const points = Array.isArray(node.point_cloud_sample) ? node.point_cloud_sample.length : 0;
    const depth = node.mean_depth == null ? "-" : Number(node.mean_depth).toFixed(3);
    const support = node.support_parent
      ? `${{node.support_type || "supported-by"}} -> ${{node.support_parent}}`
      : "-";
    detailsPanel.innerHTML =
      `<strong>${{node.hierarchical_name || node.id}}</strong>` +
      `<div style="margin-top:8px;display:grid;grid-template-columns:auto 1fr;gap:4px 8px;">` +
      `<span style="color:#94a3b8;">ID</span><span>${{node.id}}</span>` +
      `<span style="color:#94a3b8;">Class</span><span>${{node.class_name}}</span>` +
      `<span style="color:#94a3b8;">Operation</span><span>${{node.operation || "unknown"}}</span>` +
      `<span style="color:#94a3b8;">Depth</span><span>${{depth}}</span>` +
      `<span style="color:#94a3b8;">Size</span><span>${{node.size_3d.join(", ")}}</span>` +
      `<span style="color:#94a3b8;">Center</span><span>${{node.center_3d.join(", ")}}</span>` +
      `<span style="color:#94a3b8;">Point cloud points</span><span>${{points}}</span>` +
      `<span style="color:#94a3b8;">Support</span><span>${{support}}</span>` +
      `<span style="color:#94a3b8;">Canonical</span><span>${{node.canonical_name || "-"}}</span>` +
      `<span style="color:#94a3b8;">Tags</span><span>${{tags}}</span>` +
      `<span style="color:#94a3b8;">Refs</span><span>${{expressions}}</span>` +
      `</div>`;
  }}

  function computeNeighborIds(nodeId) {{
    const neighborIds = new Set();
    for (const rel of data.support_relations || []) {{
      if (rel.subject === nodeId) neighborIds.add(rel.object);
      if (rel.object === nodeId) neighborIds.add(rel.subject);
    }}
    for (const item of data.blocking_dag || []) {{
      if (item.blocker === nodeId) neighborIds.add(item.blocked);
      if (item.blocked === nodeId) neighborIds.add(item.blocker);
    }}
    return neighborIds;
  }}

  function applySelectionState() {{
    const neighborIds = selectedItem ? computeNeighborIds(selectedItem.data.id) : new Set();
    for (const item of interactiveBoxes) {{
      if (!selectedItem) {{
        setItemState(item, hoveredItem === item ? "hover" : "default");
        continue;
      }}
      if (item === selectedItem) {{
        setItemState(item, "selected");
      }} else if (neighborIds.has(item.data.id)) {{
        setItemState(item, "neighbor");
      }} else if (hoveredItem === item) {{
        setItemState(item, "hover");
      }} else {{
        setItemState(item, "default");
      }}
    }}
  }}

  function pickInteractive(event) {{
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const intersections = raycaster.intersectObjects(interactiveTargets, false);
    if (!intersections.length) return {{ item: null, rect }};
    const hit = intersections[0].object;
    return {{
      item: interactiveBoxes.find((candidate) => candidate.box === hit || candidate.points === hit) || null,
      rect,
    }};
  }}

  for (const node of data.nodes) {{
    const center = node.render_center;
    const size = node.render_size;
    const color = new THREE.Color(operationColors[node.operation] || operationColors.unknown);
    const pointCloudPayload = Array.isArray(node.render_point_cloud_sample) && node.render_point_cloud_sample.length
      ? node.render_point_cloud_sample
      : node.point_cloud_sample;
    const hasPointCloud = Array.isArray(pointCloudPayload) && pointCloudPayload.length > 0;
    let labelAnchor = null;

    const geometry = new THREE.BoxGeometry(size[0], size[1], size[2]);
    const material = new THREE.MeshBasicMaterial({{ color, transparent: true, opacity: 0.3 }});
    const box = new THREE.Mesh(geometry, material);
    box.position.set(center[0], center[1], center[2]);
    box.visible = !hasPointCloud;
    scene.add(box);
    nodeBoxes.push(box);
    labelAnchor = box;

    nodeCenters.set(node.id, new THREE.Vector3(center[0], center[1], center[2]));

    const edges = new THREE.EdgesGeometry(geometry);
    const outline = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color }}));
    outline.position.copy(box.position);
    outline.visible = box.visible;
    scene.add(outline);
    const interactiveItem = {{ box, outline, points: null, data: node }};
    interactiveBoxes.push(interactiveItem);
    interactiveTargets.push(box);

    if (hasPointCloud) {{
      const pointGeometry = new THREE.BufferGeometry();
      const pointPositions = new Float32Array(pointCloudPayload.flat());
      pointGeometry.setAttribute("position", new THREE.BufferAttribute(pointPositions, 3));
      const pointMaterial = new THREE.PointsMaterial({{ color, size: defaultPointSize, sizeAttenuation: true, transparent: true, opacity: 0.96 }});
      const points = new THREE.Points(pointGeometry, pointMaterial);
      points.visible = true;
      scene.add(points);
      interactiveItem.points = points;
      interactiveTargets.push(points);
      labelAnchor = new THREE.Object3D();
      labelAnchor.position.set(center[0], center[1], center[2]);
      scene.add(labelAnchor);
      pointClouds.push({{ points, box, outline }});
    }}

    const labelDiv = document.createElement("div");
    labelDiv.textContent = node.hierarchical_name || node.id;
    labelDiv.style.color = "#fff";
    labelDiv.style.fontSize = "12px";
    labelDiv.style.whiteSpace = "nowrap";
    const label = new CSS2DObject(labelDiv);
    label.position.set(0, Math.max(size[1], size[2]) / 2 + 0.06, 0);
    labelAnchor.add(label);
    labels.push(label);
  }}

  for (const edge of data.edges) {{
    const from = nodeCenters.get(edge.subject);
    const to = nodeCenters.get(edge.object);
    if (!from || !to) continue;

    const geometry = new THREE.BufferGeometry().setFromPoints([from.clone(), to.clone()]);
    const color = new THREE.Color(relationColors[edge.predicate] || "#95a5a6");
    const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({{ color }}));
    scene.add(line);
    relationLines.push(line);
  }}

  for (const item of data.blocking_dag) {{
    const from = nodeCenters.get(item.blocker);
    const to = nodeCenters.get(item.blocked);
    if (!from || !to) continue;

    const direction = new THREE.Vector3().subVectors(to, from);
    const length = direction.length();
    if (!length) continue;

    direction.normalize();
    const origin = from.clone().add(direction.clone().multiplyScalar(length * 0.5));
    const arrow = new THREE.ArrowHelper(direction, origin, Math.max(length * 0.4, 0.08), 0xFF5722, 0.06, 0.03);
    scene.add(arrow);
    blockingArrows.push(arrow);
  }}

  const relationToggle = document.getElementById({json.dumps(relation_toggle_id)});
  const labelToggle = document.getElementById({json.dumps(label_toggle_id)});
  const blockingToggle = document.getElementById({json.dumps(blocking_toggle_id)});
  const globalToggle = document.getElementById({json.dumps(global_toggle_id)});
  const boxToggle = document.getElementById({json.dumps(box_toggle_id)});

  if (relationToggle) {{
    relationToggle.addEventListener("change", () => {{
      for (const line of relationLines) line.visible = relationToggle.checked;
    }});
  }}
  if (labelToggle) {{
    labelToggle.addEventListener("change", () => {{
      for (const label of labels) label.visible = labelToggle.checked;
    }});
  }}
  if (blockingToggle) {{
    blockingToggle.addEventListener("change", () => {{
      for (const arrow of blockingArrows) arrow.visible = blockingToggle.checked;
    }});
  }}
  if (globalToggle) {{
    globalToggle.addEventListener("change", () => {{
      if (globalPointCloud) globalPointCloud.visible = globalToggle.checked;
    }});
  }}
  if (boxToggle) {{
    boxToggle.addEventListener("change", () => {{
      for (const item of interactiveBoxes) {{
        const hasPointCloud = Array.isArray(item.data.point_cloud_sample) && item.data.point_cloud_sample.length > 0;
        item.box.visible = boxToggle.checked || !hasPointCloud;
        item.outline.visible = boxToggle.checked || !hasPointCloud;
      }}
    }});
  }}

  renderer.domElement.addEventListener("mousemove", (event) => {{
    const result = pickInteractive(event);
    const item = result.item;
    hoveredItem = item;
    renderer.domElement.style.cursor = item ? "pointer" : "grab";
    applySelectionState();
    if (!tooltip) return;
    if (!item) {{
      tooltip.style.display = "none";
      return;
    }}
    tooltip.style.display = "block";
    tooltip.style.left = `${{event.clientX - result.rect.left + 12}}px`;
    tooltip.style.top = `${{event.clientY - result.rect.top + 12}}px`;
    tooltip.innerHTML = `<strong>${{item.data.hierarchical_name || item.data.id}}</strong><br>${{item.data.class_name}} · depth ${{item.data.mean_depth == null ? "-" : Number(item.data.mean_depth).toFixed(3)}}`;
  }});

  renderer.domElement.addEventListener("mouseleave", () => {{
    hoveredItem = null;
    renderer.domElement.style.cursor = "grab";
    applySelectionState();
    if (tooltip) {{
      tooltip.style.display = "none";
    }}
  }});

  renderer.domElement.addEventListener("click", (event) => {{
    const result = pickInteractive(event);
    const item = result.item;
    selectedItem = item;
    if (!item) {{
      if (detailsPanel) {{
        detailsPanel.innerHTML = "<strong>Selection</strong><p style='margin:8px 0 0 0;color:#94a3b8;'>Click an object to inspect its 3D node details.</p>";
      }}
      applySelectionState();
      return;
    }}
    applySelectionState();
    renderDetails(item.data);
  }});

  function resize() {{
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    labelRenderer.setSize(width, height);
  }}

  window.addEventListener("resize", resize);
  resize();
  renderDetails(data.nodes[0]);
  if (relationToggle) for (const line of relationLines) line.visible = relationToggle.checked;
  if (blockingToggle) for (const arrow of blockingArrows) arrow.visible = blockingToggle.checked;
  if (globalToggle && globalPointCloud) globalPointCloud.visible = globalToggle.checked;
  applySelectionState();

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }}

  animate();
}}
    </script>
  </div>
</div>
"""
