"""Placement-region visualization section."""

from __future__ import annotations

from html import escape
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from apps.data_flywheel.types import PlacementRegion, SceneGraph


def _mapping(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        data = value.to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _svg_background(image_base64: str, width: int, height: int) -> str:
    if image_base64:
        return (
            f'<image href="{escape(image_base64)}" x="0" y="0" width="{width}" height="{height}" '
            'preserveAspectRatio="none" />'
        )
    return f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" />'


def _polygon_points(points: List[List[float]]) -> str:
    return " ".join(f"{float(x):.1f},{float(y):.1f}" for x, y in points)


def _selected_place_action(training_samples: Iterable[Any]) -> dict:
    for sample in training_samples:
        data = _mapping(sample)
        for action in data.get("actions") or []:
            action_data = _mapping(action)
            if action_data.get("type") == "place":
                return action_data
    return {}


def _node_label_map(scene_graph: SceneGraph) -> dict[str, str]:
    return {
        node.id: (node.hierarchical_name or node.class_name or node.id)
        for node in scene_graph.nodes
    }


def _label_for(node_labels: Mapping[str, str], node_id: str) -> str:
    return str(node_labels.get(node_id) or node_id)


def _selected_place_point(training_samples: Iterable[Any], image_size: Tuple[int, int]) -> tuple[Optional[List[float]], Optional[str]]:
    width, height = image_size
    for sample in training_samples:
        data = _mapping(sample)
        for action in data.get("actions") or []:
            action_data = _mapping(action)
            if action_data.get("type") != "place":
                continue
            point_pixel = action_data.get("place_point_pixel")
            if isinstance(point_pixel, (list, tuple)) and len(point_pixel) >= 2:
                return [float(point_pixel[0]), float(point_pixel[1])], str(action_data.get("destination") or "")
            point_quantized = action_data.get("place_point_quantized")
            if isinstance(point_quantized, (list, tuple)) and len(point_quantized) >= 2:
                return [
                    float(point_quantized[0]) / 999.0 * width,
                    float(point_quantized[1]) / 999.0 * height,
                ], str(action_data.get("destination") or "")
    return None, None


def _region_summary(region: PlacementRegion) -> str:
    return f"{region.region_id} ({region.region_type}) -> {region.support_id}"


def render_placement_section(
    scene_graph: SceneGraph,
    training_samples: Iterable[Any] = (),
    image_base64: str = "",
) -> str:
    if not scene_graph.placement_regions:
        return (
            '<div class="card"><h3>D. Placement View</h3>'
            '<p style="margin:0 0 8px 0;color:#475569;font-size:13px;">Placement Regions are unavailable for this scene.</p>'
            '<p>No placement regions</p></div>'
        )

    width, height = scene_graph.image_size
    node_labels = _node_label_map(scene_graph)
    selected_point, destination_id = _selected_place_point(training_samples, scene_graph.image_size)
    selected_action = _selected_place_action(training_samples)

    region_shapes: List[str] = []
    rows: List[str] = []
    for region in scene_graph.placement_regions:
        color = "#2563eb" if region.region_type == "inside-container" else "#16a34a"
        if region.polygon_2d:
            region_shapes.append(
                f'<polygon points="{_polygon_points(region.polygon_2d)}" fill="{color}" fill-opacity="0.18" '
                f'stroke="{color}" stroke-width="3" stroke-dasharray="8 4" />'
            )
        if region.point_2d is not None:
            region_shapes.append(
                f'<circle cx="{float(region.point_2d[0]):.1f}" cy="{float(region.point_2d[1]):.1f}" r="8" '
                f'fill="{color}" stroke="#ffffff" stroke-width="3" />'
            )
        rows.append(
            "<tr>"
            f"<td>{escape(region.region_id)}</td>"
            f"<td>{escape(region.region_type)}</td>"
            f"<td>{escape(_label_for(node_labels, region.support_id))}</td>"
            f"<td>{escape(', '.join(_label_for(node_labels, anchor_id) for anchor_id in region.anchor_ids) or '-')}</td>"
            f"<td>{escape(region.reason)}</td>"
            "</tr>"
        )

    selected_marker = ""
    selected_text = "No selected place point"
    selected_meta = ""
    if selected_point is not None:
        selected_marker = (
            f'<circle cx="{selected_point[0]:.1f}" cy="{selected_point[1]:.1f}" r="12" '
            'fill="#f97316" fill-opacity="0.75" stroke="#ffffff" stroke-width="4" />'
        )
        selected_text = f"Selected place point: {selected_point[0]:.1f}, {selected_point[1]:.1f}"
        if destination_id:
            selected_text += f" -> {_label_for(node_labels, destination_id)}"
        destination_type = selected_action.get("destination_type")
        placement_reason = selected_action.get("placement_reason")
        anchor_ids = selected_action.get("destination_anchor_ids") or []
        target_id = selected_action.get("target")
        selected_bits = []
        if target_id:
            selected_bits.append(f"target={_label_for(node_labels, str(target_id))}")
        if destination_type:
            selected_bits.append(f"destination_type={destination_type}")
        if anchor_ids:
            selected_bits.append(
                "anchors=" + ", ".join(_label_for(node_labels, str(anchor_id)) for anchor_id in anchor_ids)
            )
        if placement_reason:
            selected_bits.append(f"placement_reason={placement_reason}")
        if selected_bits:
            selected_meta = (
                '<div style="margin:0 0 12px 0;color:#475569;font-size:13px;">'
                f"{escape(' | '.join(selected_bits))}"
                '</div>'
            )

    return (
        '<div class="card">'
        '<h3>D. Placement View</h3>'
        '<p style="margin:0 0 8px 0;color:#475569;font-size:13px;">Placement view shows candidate container / free-space regions together with the selected final place point and placement provenance.</p>'
        f'<p style="margin:0 0 12px 0;color:#475569;">{escape(selected_text)}</p>'
        f"{selected_meta}"
        '<div class="placement-overlay" style="border:1px solid #d1d5db;border-radius:12px;overflow:hidden;background:#fff;">'
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img" aria-label="Placement regions">'
        f"{_svg_background(image_base64, width, height)}"
        f"{''.join(region_shapes)}"
        f"{selected_marker}"
        '</svg>'
        '</div>'
        '<h4 style="margin:14px 0 8px 0;">Placement Regions</h4>'
        '<table style="margin-top:12px;">'
        '<thead><tr><th>Region</th><th>Type</th><th>Support</th><th>Anchors</th><th>Reason</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody>"
        '</table>'
        '</div>'
    )
