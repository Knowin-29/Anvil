"""M2 perception section renderer.

This module intentionally stays self-contained so it can be implemented
before the shared assets package exists.
"""

from __future__ import annotations

from collections import defaultdict
from html import escape
from typing import Any, Iterable, List, Mapping, Optional, Tuple


SOURCE_PALETTE = [
    "#ef4444",
    "#3b82f6",
    "#10b981",
    "#f59e0b",
    "#8b5cf6",
    "#ec4899",
    "#14b8a6",
    "#f97316",
]


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return converted
    if hasattr(value, "__dict__"):
        return value.__dict__
    return {}


def _normalize_box(box: Any) -> List[float]:
    raw = list(box or [])
    if len(raw) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]


def _normalize_point(point: Any) -> Optional[Tuple[float, float]]:
    if not point or len(point) < 2:
        return None
    return float(point[0]), float(point[1])


def _normalize_polygon_points(mask_polygon: Any) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for point in mask_polygon or []:
        if not point or len(point) < 2:
            continue
        points.append((float(point[0]), float(point[1])))
    return points


def _normalize_detection(detection: Any) -> Mapping[str, Any]:
    data = dict(_to_mapping(detection))
    bbox = data.get("bbox")
    if bbox is None and "detection" in data:
        bbox = _to_mapping(data["detection"]).get("bbox")
    data["bbox"] = _normalize_box(bbox)
    data["class_name"] = data.get("class_name") or data.get("name") or "object"
    data["confidence"] = data.get("confidence")
    data["source"] = data.get("source") or "unknown"
    return data


def _normalize_segmentation(segmentation: Any) -> Mapping[str, Any]:
    data = dict(_to_mapping(segmentation))
    detection = data.get("detection")
    detection_data = _normalize_detection(detection) if detection is not None else {}
    data["detection"] = detection_data
    data["mask_polygon"] = data.get("mask_polygon")
    data["grasp_point"] = data.get("grasp_point")
    if not data.get("class_name"):
        data["class_name"] = detection_data.get("class_name", "object")
    return data


def _format_confidence(confidence: Any) -> str:
    try:
        if confidence is None:
            return ""
        return f"{float(confidence):.2f}"
    except (TypeError, ValueError):
        return escape(str(confidence))


def _svg_point(point: Tuple[float, float]) -> str:
    return f"{point[0]:.1f},{point[1]:.1f}"


def _source_color(source: Any) -> str:
    label = str(source or "unknown")
    palette_index = sum(ord(ch) for ch in label) % len(SOURCE_PALETTE)
    return SOURCE_PALETTE[palette_index]


def _render_detection_rect(index: int, detection: Mapping[str, Any]) -> str:
    x1, y1, x2, y2 = detection["bbox"]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    label = escape(str(detection.get("class_name", "object")))
    confidence = _format_confidence(detection.get("confidence"))
    source = escape(str(detection.get("source", "unknown")))
    title = f"{label}"
    if confidence:
        title += f" {confidence}"
    title += f" [{source}]"
    color = _source_color(source)
    return (
        f'<g class="detection" data-source="{source}">'
        f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="3" />'
        f'<rect x="{x1:.1f}" y="{max(0.0, y1 - 18):.1f}" width="{min(240.0, max(70.0, width)):.1f}" '
        f'height="18" fill="{color}" opacity="0.92" rx="3" />'
        f'<text x="{x1 + 4:.1f}" y="{max(12.0, y1 - 5):.1f}" fill="#ffffff" font-size="12" '
        f'font-family="sans-serif">{title}</text>'
        f"</g>"
    )


def _render_segmentation_overlay(index: int, segmentation: Mapping[str, Any]) -> str:
    detection = segmentation.get("detection", {})
    bbox = detection.get("bbox", [0.0, 0.0, 0.0, 0.0])
    x1, y1, x2, y2 = bbox
    source = detection.get("source", segmentation.get("source", "unknown"))
    color = _source_color(source)
    class_name = escape(str(segmentation.get("class_name", detection.get("class_name", "object"))))
    parts = [f'<g class="segmentation" data-class="{class_name}">']

    mask_polygon = segmentation.get("mask_polygon")
    if mask_polygon:
        points = " ".join(_svg_point(point) for point in _normalize_polygon_points(mask_polygon))
        parts.append(
            f'<polygon points="{points}" fill="{color}" fill-opacity="0.22" '
            f'stroke="{color}" stroke-width="2" />'
        )
    else:
        parts.append(
            f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{max(0.0, x2 - x1):.1f}" '
            f'height="{max(0.0, y2 - y1):.1f}" fill="{color}" fill-opacity="0.12" '
            f'stroke="{color}" stroke-width="2" stroke-dasharray="6 4" />'
        )

    grasp_point = _normalize_point(segmentation.get("grasp_point"))
    if grasp_point is None:
        grasp_point = _normalize_point(detection.get("grasp_point"))
    if grasp_point is not None:
        gx, gy = grasp_point
        parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="2" />')

    confidence = _format_confidence(detection.get("confidence"))
    source = escape(str(source))
    label = class_name
    if confidence:
        label += f" {confidence}"
    label += f" [{source}]"
    parts.append(
        f'<text x="{x1 + 4:.1f}" y="{max(14.0, y1 - 6):.1f}" fill="#111111" font-size="12" '
        f'font-family="sans-serif" paint-order="stroke" stroke="#ffffff" stroke-width="3">{escape(label)}</text>'
    )
    parts.append("</g>")
    return "".join(parts)


def _render_backend_comparison(
    detections: Iterable[Mapping[str, Any]],
    segmentations: Iterable[Mapping[str, Any]],
) -> str:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"detections": 0, "segmentations": 0})
    for det in detections:
        counts[str(det.get("source", "unknown"))]["detections"] += 1
    for seg in segmentations:
        detection = seg.get("detection", {})
        source = str(detection.get("source") or seg.get("source") or "unknown")
        counts[source]["segmentations"] += 1

    rows = []
    for source in sorted(counts):
        detection_count = counts[source]["detections"]
        segmentation_count = counts[source]["segmentations"]
        color = _source_color(source)
        rows.append(
            "<tr>"
            f'<td><span class="color-swatch" style="background:{color}" title="{escape(source)}"></span>{escape(source)}</td>'
            f"<td>{detection_count}</td>"
            f"<td>{segmentation_count}</td>"
            f"<td>{detection_count + segmentation_count}</td>"
            "</tr>"
        )

    if not rows:
        return "<p>No backend statistics</p>"

    return (
        "<h4 style=\"margin:16px 0 8px 0\">Backend Comparison</h4>"
        "<table style=\"margin-top:12px\">"
        "<thead><tr><th>Backend</th><th>Detection Count</th><th>Segmentation Count</th><th>Total</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_perception_section(
    image_base64: str,
    image_size: Tuple[int, int],
    detections: Iterable[Any],
    segmentations: Iterable[Any],
) -> str:
    """Render the M2 perception section as a self-contained HTML fragment."""

    detections_list = [_normalize_detection(item) for item in detections or []]
    segmentations_list = [_normalize_segmentation(item) for item in segmentations or []]

    if not detections_list and not segmentations_list:
        return '<div class="card"><h3>M2 - Perception</h3><p>No detections</p></div>'

    width = int(image_size[0]) if image_size and image_size[0] else 1
    height = int(image_size[1]) if image_size and image_size[1] else 1
    image_src = escape(image_base64 or "")

    svg_layers = [
        f'<image href="{image_src}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none" />'
    ]
    svg_layers.extend(_render_segmentation_overlay(i, seg) for i, seg in enumerate(segmentations_list))
    svg_layers.extend(_render_detection_rect(i, det) for i, det in enumerate(detections_list))

    detections_rows = []
    for item in detections_list:
        conf = _format_confidence(item.get("confidence"))
        detections_rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('class_name', 'object')))}</td>"
            f"<td>{escape(conf or '—')}</td>"
            f"<td>{escape(str(item.get('source', 'unknown')))}</td>"
            "</tr>"
        )

    segmentations_rows = []
    for item in segmentations_list:
        det = item.get("detection", {})
        seg_label = escape(str(item.get("class_name", det.get("class_name", "object"))))
        has_mask = "yes" if item.get("mask_polygon") else "no"
        has_grasp = "yes" if _normalize_point(item.get("grasp_point") or det.get("grasp_point")) else "no"
        segmentations_rows.append(
            "<tr>"
            f"<td>{seg_label}</td>"
            f"<td>{escape(has_mask)}</td>"
            f"<td>{escape(has_grasp)}</td>"
            f"<td>{escape(str(det.get('source', 'unknown')))}</td>"
            "</tr>"
        )

    return (
        '<div class="card">'
        '<h3>M2 - Perception</h3>'
        '<div class="img-container">'
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img" '
        f'aria-label="Perception overlay">{"".join(svg_layers)}</svg>'
        "</div>"
        '<table>'
        "<thead><tr><th>Detection</th><th>Confidence</th><th>Source</th></tr></thead>"
        f"<tbody>{''.join(detections_rows)}</tbody>"
        "</table>"
        '<table style="margin-top:12px">'
        "<thead><tr><th>Segmentation</th><th>Mask</th><th>Grasp</th><th>Source</th></tr></thead>"
        f"<tbody>{''.join(segmentations_rows)}</tbody>"
        "</table>"
        f"{_render_backend_comparison(detections_list, segmentations_list)}"
        "</div>"
    )
