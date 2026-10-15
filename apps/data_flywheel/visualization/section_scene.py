"""M3 scene understanding section renderer."""

from __future__ import annotations

import json
import re
from collections import Counter
from html import escape
from typing import Any, Dict, List, Tuple

from apps.data_flywheel.types import SceneGraph, SceneNode

from .assets import EDGE_SOURCE_COLORS, OPERATION_COLORS, RELATION_COLORS
from .three_scene import render_three_scene


def _slug(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", value or "scene")
    token = token.strip("-")
    return token or "scene"


def _node_label(node: SceneNode) -> str:
    return node.hierarchical_name or node.class_name or node.id


def _node_meta_label(node: SceneNode) -> str:
    label = node.class_name or node.id
    if node.id and node.id != label:
        return f"{label} [{node.id}]"
    return label


def _display_node_label(node: SceneNode, duplicate_counts: Counter[str]) -> str:
    base = _node_label(node)
    if duplicate_counts.get(base, 0) > 1 or node.hierarchical_name is None:
        return f"{base} [{node.id}]"
    return base


def _node_title(node: SceneNode) -> str:
    expressions = ", ".join(node.referring_expressions) if node.referring_expressions else "-"
    tags = ", ".join(node.extra_tags) if node.extra_tags else "-"
    depth = "-" if node.mean_depth is None else f"{node.mean_depth:.3f}"
    support = (
        f"{node.support_type or 'supported-by'} -> {node.support_parent}"
        if node.support_parent
        else "-"
    )
    return "\n".join(
        [
            f"id: {node.id}",
            f"class: {node.class_name}",
            f"operation: {node.operation or 'unknown'}",
            f"depth: {depth}",
            f"support: {support}",
            f"tags: {tags}",
            f"refs: {expressions}",
        ]
    )


_EDGE_PRIORITY = {
    "blocking": 0,
    "inside": 1,
    "on-top-of": 2,
    "stacked-on": 2,
    "supporting": 2,
    "occluding": 3,
    "occluded-by": 3,
    "overlapping": 4,
    "left-of": 5,
    "right-of": 5,
    "above": 6,
    "below": 6,
    "next-to": 7,
    "near": 8,
}


_RELATION_COLORS_2D = {
    "blocking": "#f97316",
    "inside": "#2563eb",
    "on-top-of": "#7c3aed",
    "stacked-on": "#7c3aed",
    "supporting": "#dc2626",
    "occluding": "#eab308",
    "occluded-by": "#f59e0b",
    "overlapping": "#0ea5e9",
    "left-of": "#475569",
    "right-of": "#475569",
    "above": "#475569",
    "below": "#475569",
    "next-to": "#64748b",
    "near": "#64748b",
}

_SUPPORT_CONTEXT_PREDICATES = {"on-top-of", "inside", "supporting", "stacked-on"}
_CORE_RELATION_PREDICATES = {"stacked-on", "inside", "occluding", "occluded-by"}
_DIRECTIONAL_RELATION_PREDICATES = {"left-of", "right-of", "above", "below", "front", "behind"}
_PROXIMITY_RELATION_PREDICATES = {"near", "next-to", "overlapping"}


def _node_center(node: SceneNode) -> Tuple[float, float]:
    return (
        (float(node.bbox[0]) + float(node.bbox[2])) / 2.0,
        (float(node.bbox[1]) + float(node.bbox[3])) / 2.0,
    )


def _bbox_area(node: SceneNode) -> float:
    return max(0.0, float(node.bbox[2]) - float(node.bbox[0])) * max(0.0, float(node.bbox[3]) - float(node.bbox[1]))


def _dominant_support_node_ids(scene_graph: SceneGraph) -> set[str]:
    explicit_roots = {
        node.id
        for node in scene_graph.nodes
        if "base_support_root" in (node.role_ids or [])
    }
    if explicit_roots:
        return explicit_roots
    width, height = scene_graph.image_size
    image_area = max(float(width * height), 1.0)
    return {
        node.id
        for node in scene_graph.nodes
        if node.operation == "non_operable" and _bbox_area(node) / image_area >= 0.18
    }


def _chip_width(label: str) -> int:
    return max(54, min(220, 16 + len(label) * 7))


def _clamp_label_x(x: float, label: str, image_width: int) -> float:
    width = _chip_width(label)
    return max(4.0, min(float(image_width) - width - 4.0, x))


def _label_y(y: float) -> float:
    return max(18.0, y)


def _predicate_color(predicate: str) -> str:
    return _RELATION_COLORS_2D.get(predicate, "#64748b")


def _select_key_relations(
    scene_graph: SceneGraph,
    support_ids: set[str],
    *,
    allowed_predicates: set[str] | None = None,
    limit_override: int | None = None,
) -> List[Any]:
    grouped: Dict[tuple[str, str], tuple[int, float, Any]] = {}
    for edge in scene_graph.edges:
        if edge.source not in {"2d", "2d+3d", "derived"}:
            continue
        priority = _EDGE_PRIORITY.get(edge.predicate)
        if priority is None or edge.subject == edge.object:
            continue
        if allowed_predicates is not None and edge.predicate not in allowed_predicates:
            continue
        if {edge.subject, edge.object} & support_ids and edge.predicate in _SUPPORT_CONTEXT_PREDICATES:
            continue
        if {edge.subject, edge.object} & support_ids and priority >= 5:
            continue
        key = tuple(sorted((edge.subject, edge.object)))
        score = (priority, -float(edge.confidence))
        current = grouped.get(key)
        if current is None or score < (current[0], current[1]):
            grouped[key] = (priority, -float(edge.confidence), edge)

    limit = limit_override if limit_override is not None else max(6, min(12, len(scene_graph.nodes)))
    candidates = [item[2] for item in sorted(grouped.values(), key=lambda item: (item[0], item[1], item[2].predicate))]
    selected: List[Any] = []
    subject_counts: Counter[str] = Counter()
    subject_predicate_counts: Counter[tuple[str, str]] = Counter()
    directional_count = 0

    for edge in candidates:
        priority = _EDGE_PRIORITY.get(edge.predicate, 99)
        is_directional = priority >= 5
        if subject_counts[edge.subject] >= 3:
            continue
        if subject_predicate_counts[(edge.subject, edge.predicate)] >= 2:
            continue
        if is_directional and directional_count >= 6:
            continue
        selected.append(edge)
        subject_counts[edge.subject] += 1
        subject_predicate_counts[(edge.subject, edge.predicate)] += 1
        if is_directional:
            directional_count += 1
        if len(selected) >= limit:
            break

    return selected


def _diagnostic_reasons(node: SceneNode, duplicate_class_counts: Counter[str]) -> List[str]:
    reasons: List[str] = []
    if (node.ontology_status or "unknown") != "resolved" or (node.class_name or "").strip().lower() in {"unknown", "container"}:
        reasons.append("class unresolved")
    if node.is_container and "container" not in (node.role_ids or []):
        reasons.append("role unresolved")
    label = (node.hierarchical_name or "").strip().lower()
    generic_label = (node.class_name or "").strip().lower()
    if not node.hierarchical_name:
        reasons.append("descriptor missing")
    elif label in {generic_label, f"the {generic_label}"} and duplicate_class_counts[generic_label] > 1:
        reasons.append("descriptor insufficient")
    if not node.has_true_mask:
        reasons.append("mask fallback")
    if not node.point_cloud_sample and not node.valid_3d:
        reasons.append("point-cloud missing")
    return reasons


def _render_diagnostics_card(scene_graph: SceneGraph) -> str:
    duplicate_class_counts = Counter((node.class_name or "").strip().lower() for node in scene_graph.nodes)
    rows: List[str] = []
    for node in scene_graph.nodes:
        reasons = _diagnostic_reasons(node, duplicate_class_counts)
        if not reasons:
            continue
        chips = "".join(
            f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;background:#e2e8f0;color:#0f172a;font-size:11px;margin:2px 6px 2px 0;">{escape(reason)}</span>'
            for reason in reasons
        )
        rows.append(
            '<tr>'
            f'<td><div><strong>{escape(_node_label(node))}</strong></div><div style="color:#64748b;font-size:12px;">{escape(_node_meta_label(node))}</div></td>'
            f'<td>{chips}</td>'
            '</tr>'
        )

    if not rows:
        body = '<p style="margin:0;color:#475569;">No unresolved class, descriptor, mask, or point-cloud diagnostics in this scene.</p>'
    else:
        body = (
            '<p style="margin:0 0 12px 0;color:#475569;font-size:13px;">'
            'This card breaks ambiguous or degraded nodes into explicit buckets instead of collapsing everything into a single unknown label.'
            '</p>'
            '<table>'
            '<thead><tr><th>Node</th><th>Issues</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody>"
            '</table>'
        )

    return '<div class="card"><h3>Diagnostics</h3>' + body + '</div>'


def _support_context_rows(scene_graph: SceneGraph, support_ids: set[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[str]] = {}
    for edge in scene_graph.edges:
        if edge.source not in {"2d", "2d+3d", "derived"}:
            continue
        if edge.predicate not in _SUPPORT_CONTEXT_PREDICATES:
            continue
        if edge.subject in support_ids and edge.object not in support_ids:
            support_id = edge.subject
            other_id = edge.object
        elif edge.object in support_ids and edge.subject not in support_ids:
            support_id = edge.object
            other_id = edge.subject
        else:
            continue
        grouped.setdefault((support_id, edge.predicate), [])
        if other_id not in grouped[(support_id, edge.predicate)]:
            grouped[(support_id, edge.predicate)].append(other_id)

    rows = []
    for (support_id, predicate), members in sorted(grouped.items(), key=lambda item: (_EDGE_PRIORITY.get(item[0][1], 99), item[0][0])):
        rows.append(
            {
                "support_id": support_id,
                "predicate": predicate,
                "members": sorted(members),
            }
        )
    return rows


def _render_node_overlay(scene_graph: SceneGraph, support_ids: set[str]) -> str:
    width, _ = scene_graph.image_size
    duplicate_counts = Counter(_node_label(node) for node in scene_graph.nodes)
    parts: List[str] = []
    for node in scene_graph.nodes:
        color = OPERATION_COLORS.get(node.operation, OPERATION_COLORS[None])
        x1, y1, x2, y2 = [float(value) for value in node.bbox]
        label = _display_node_label(node, duplicate_counts).replace("[", "(").replace("]", ")")
        label_x = _clamp_label_x(x1 + 4.0, label, width)
        label_y = _label_y(y1 - 6.0)
        parts.append(
            f'<rect x="{label_x:.1f}" y="{label_y - 16.0:.1f}" width="{_chip_width(label):.1f}" height="18" '
            f'fill="{color}" fill-opacity="0.95" rx="4" />'
        )
        parts.append(
            f'<text x="{label_x + 6.0:.1f}" y="{label_y - 3.0:.1f}" fill="#ffffff" font-size="11" '
            f'font-family="sans-serif">{escape(label)}</text>'
        )
        if node.id in support_ids:
            continue
        parts.append(
            f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{max(0.0, x2 - x1):.1f}" height="{max(0.0, y2 - y1):.1f}" '
            f'fill="{color}" fill-opacity="0.08" stroke="{color}" stroke-width="2" rx="4" />'
        )
    return "".join(parts)


def _render_relation_overlay(scene_graph: SceneGraph, image_base64: str) -> str:
    width, height = scene_graph.image_size
    prefix = _slug(getattr(scene_graph, "image_id", "scene"))
    support_ids = _dominant_support_node_ids(scene_graph)
    visible_relations = _select_key_relations(
        scene_graph,
        support_ids,
        allowed_predicates=_CORE_RELATION_PREDICATES,
        limit_override=max(4, min(10, len(scene_graph.nodes))),
    )
    directional_relations = _select_key_relations(
        scene_graph,
        support_ids,
        allowed_predicates=_DIRECTIONAL_RELATION_PREDICATES,
        limit_override=max(4, min(8, len(scene_graph.nodes))),
    )
    proximity_relations = _select_key_relations(
        scene_graph,
        support_ids,
        allowed_predicates=_PROXIMITY_RELATION_PREDICATES,
        limit_override=max(4, min(8, len(scene_graph.nodes))),
    )
    support_rows = _support_context_rows(scene_graph, support_ids)
    nodes_by_id = {node.id: node for node in scene_graph.nodes}
    directional_toggle_id = f"{prefix}-toggle-directional"
    proximity_toggle_id = f"{prefix}-toggle-proximity"
    directional_group_id = f"{prefix}-directional-layer"
    proximity_group_id = f"{prefix}-proximity-layer"

    svg_parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" />',
    ]
    if image_base64:
        svg_parts.append(
            f'<image href="{escape(image_base64)}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none" />'
        )
    svg_parts.append(_render_node_overlay(scene_graph, support_ids))

    relation_rows: List[str] = []

    def _relation_row(edge: Any, *, index: str, subdued: bool = False) -> str:
        subject = nodes_by_id.get(edge.subject)
        object_node = nodes_by_id.get(edge.object)
        if subject is None or object_node is None:
            return ""
        color = _predicate_color(edge.predicate)
        badge_style = (
            f'display:inline-flex;align-items:center;justify-content:center;min-width:24px;height:24px;'
            f'border-radius:999px;background:{color};color:#fff;font-weight:700;font-size:12px;'
        )
        if subdued:
            badge_style += 'opacity:0.75;'
        return (
            '<div style="display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-bottom:1px solid #e5e7eb;">'
            f'<span style="{badge_style}">{escape(index)}</span>'
            '<div style="font-size:13px;line-height:1.45;">'
            f'<div><strong>{escape(_node_label(subject))}</strong> '
            f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;background:{color};color:#fff;font-size:11px;">{escape(edge.predicate)}</span> '
            f'<strong>{escape(_node_label(object_node))}</strong></div>'
            f'<div style="color:#6b7280;">{escape(_node_meta_label(subject))} -> {escape(_node_meta_label(object_node))}, source={escape(edge.source)}, conf={edge.confidence:.2f}</div>'
            "</div>"
            "</div>"
        )

    for index, edge in enumerate(visible_relations, start=1):
        subject = nodes_by_id.get(edge.subject)
        object_node = nodes_by_id.get(edge.object)
        if subject is None or object_node is None:
            continue
        sx, sy = _node_center(subject)
        ox, oy = _node_center(object_node)
        mx = (sx + ox) / 2.0
        my = (sy + oy) / 2.0
        color = _predicate_color(edge.predicate)
        svg_parts.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ox:.1f}" y2="{oy:.1f}" '
            f'stroke="{color}" stroke-width="3" stroke-opacity="0.9" />'
        )
        svg_parts.append(
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="12" fill="{color}" stroke="#ffffff" stroke-width="2" />'
        )
        svg_parts.append(
            f'<text x="{mx:.1f}" y="{my + 4.0:.1f}" text-anchor="middle" fill="#ffffff" font-size="12" '
            f'font-family="sans-serif" font-weight="700">{index}</text>'
        )
        relation_rows.append(_relation_row(edge, index=str(index)))

    def _secondary_svg_group(edges: List[Any], group_id: str, dasharray: str) -> str:
        group_parts: List[str] = []
        for edge in edges:
            subject = nodes_by_id.get(edge.subject)
            object_node = nodes_by_id.get(edge.object)
            if subject is None or object_node is None:
                continue
            sx, sy = _node_center(subject)
            ox, oy = _node_center(object_node)
            color = _predicate_color(edge.predicate)
            group_parts.append(
                f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ox:.1f}" y2="{oy:.1f}" '
                f'stroke="{color}" stroke-width="2.5" stroke-opacity="0.72" stroke-dasharray="{dasharray}" />'
            )
        if not group_parts:
            return ""
        return f'<g id="{group_id}" style="display:none;">{"".join(group_parts)}</g>'

    svg_parts.append(_secondary_svg_group(directional_relations, directional_group_id, "8 6"))
    svg_parts.append(_secondary_svg_group(proximity_relations, proximity_group_id, "3 6"))

    support_html = ""
    if support_rows:
        support_cards = []
        for row in support_rows:
            members = ", ".join(
                escape(_node_label(nodes_by_id[member])) if member in nodes_by_id else escape(member)
                for member in row["members"]
            )
            support_cards.append(
                '<div style="padding:8px 0;border-bottom:1px solid #e5e7eb;font-size:13px;line-height:1.45;">'
                f'<div><span style="display:inline-block;padding:2px 8px;border-radius:999px;background:{_predicate_color(row["predicate"])};color:#fff;font-size:11px;">{escape(row["predicate"])}</span> '
                f'<strong>{escape(_node_label(nodes_by_id[row["support_id"]])) if row["support_id"] in nodes_by_id else escape(row["support_id"])}</strong></div>'
                f'<div style="color:#6b7280;">{members}</div>'
                "</div>"
            )
        support_html = (
            '<div style="margin-bottom:16px;">'
            '<h4 style="margin:0 0 8px 0;">Support Context</h4>'
            '<p style="color:#6b7280;font-size:12px;margin:0 0 8px 0;">Large support-plane relations are summarized here instead of drawing every edge on the image.</p>'
            f'{"".join(support_cards)}'
            "</div>"
        )

    secondary_rows: List[str] = []
    if directional_relations:
        secondary_rows.append(
            '<div style="margin-top:14px;">'
            '<h4 style="margin:0 0 8px 0;">Directional Relations</h4>'
            f'{"".join(_relation_row(edge, index="D", subdued=True) for edge in directional_relations)}'
            '</div>'
        )
    if proximity_relations:
        secondary_rows.append(
            '<div style="margin-top:14px;">'
            '<h4 style="margin:0 0 8px 0;">Proximity Relations</h4>'
            f'{"".join(_relation_row(edge, index="P", subdued=True) for edge in proximity_relations)}'
            '</div>'
        )
    relations_html = "".join(relation_rows) if relation_rows else '<p style="font-size:13px;color:#6b7280;">No core tabletop relations selected for overlay.</p>'
    note = (
        f'<p style="color:#6b7280;font-size:12px;margin:0 0 8px 0;">'
        f'Image-anchored overlay with {len(visible_relations)} core tabletop relations. '
        f'Stacked / occlusion / containment stay on by default; secondary directional and proximity edges can be toggled on demand.'
        f"</p>"
    )

    return (
        '<div class="card">'
        "<h3>A. 2D Image-Anchored Relation View</h3>"
        f"{note}"
        '<div style="display:flex;gap:16px;flex-wrap:wrap;margin:0 0 12px 0;font-size:13px;">'
        '<span style="display:inline-flex;align-items:center;gap:8px;color:#0f172a;"><span style="width:12px;height:12px;border-radius:999px;background:#7c3aed;display:inline-block;"></span> Core relations</span>'
        f'<label style="display:inline-flex;align-items:center;gap:8px;cursor:pointer;color:#0f172a;"><input type="checkbox" id="{escape(directional_toggle_id)}"> Show secondary relations</label>'
        f'<label style="display:inline-flex;align-items:center;gap:8px;cursor:pointer;color:#0f172a;"><input type="checkbox" id="{escape(proximity_toggle_id)}"> Show proximity relations</label>'
        '</div>'
        '<div style="display:grid;grid-template-columns:minmax(0,2fr) minmax(300px,1fr);gap:16px;align-items:start;">'
        '<div class="img-container" style="border:1px solid #d1d5db;border-radius:8px;overflow:hidden;background:#fff;">'
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img" aria-label="2D relation overlay">{"".join(svg_parts)}</svg>'
        "</div>"
        '<div style="border:1px solid #e5e7eb;border-radius:8px;padding:14px;background:#f8fafc;">'
        f"{support_html}"
        '<h4 style="margin:0 0 8px 0;">Key Relations</h4>'
        f"{relations_html}"
        f'{"".join(secondary_rows)}'
        "</div>"
        "</div>"
        "<script>"
        "(function(){"
        f"const directionalToggle=document.getElementById({json.dumps(directional_toggle_id)});"
        f"const proximityToggle=document.getElementById({json.dumps(proximity_toggle_id)});"
        f"const directionalLayer=document.getElementById({json.dumps(directional_group_id)});"
        f"const proximityLayer=document.getElementById({json.dumps(proximity_group_id)});"
        "const apply=(toggle,layer)=>{if(!toggle||!layer){return;}layer.style.display=toggle.checked?'':'none';};"
        "if(directionalToggle){directionalToggle.addEventListener('change',()=>apply(directionalToggle,directionalLayer));apply(directionalToggle,directionalLayer);}"
        "if(proximityToggle){proximityToggle.addEventListener('change',()=>apply(proximityToggle,proximityLayer));apply(proximityToggle,proximityLayer);}"
        "})();"
        "</script>"
        "</div>"
    )


def _render_support_graph_card(scene_graph: SceneGraph) -> str:
    if not scene_graph.support_relations:
        return '<div class="card"><h3>C1. Support Graph</h3><p>No support relationships</p></div>'

    nodes_by_id = {node.id: node for node in scene_graph.nodes}
    base_root_rows: List[str] = []
    regular_rows: List[str] = []
    for relation in scene_graph.support_relations:
        parent = nodes_by_id.get(relation.object)
        parent_label = _node_label(parent) if parent is not None else relation.object
        child = nodes_by_id.get(relation.subject)
        child_label = _node_label(child) if child is not None else relation.subject
        row = (
            "<tr>"
            f"<td>{escape(child_label)}</td>"
            f"<td>{escape(relation.relation)}</td>"
            f"<td>{escape(parent_label)}</td>"
            f"<td>{relation.confidence:.2f}</td>"
            "</tr>"
        )
        if parent is not None and "base_support_root" in (parent.role_ids or []):
            base_root_rows.append(row)
        else:
            regular_rows.append(row)

    summary = ""
    if base_root_rows:
        summary = (
            '<p style="margin:0 0 12px 0;color:#475569;">'
            "Base support root context is summarized here instead of drawing every tabletop support edge into the main 2D overlay."
            "</p>"
        )

    table_rows = regular_rows or base_root_rows
    return (
        '<div class="card">'
        "<h3>C1. Support Graph</h3>"
        f"{summary}"
        "<table>"
        "<thead><tr><th>Subject</th><th>Relation</th><th>Support</th><th>Confidence</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _build_blocking_dag_card(scene_graph: SceneGraph, dom_id: str) -> str:
    if not scene_graph.blocking_dag:
        return '<div class="card"><h3>C2. Blocking DAG</h3><p>No blocking relationships</p></div>'

    nodes_by_id = {node.id: node for node in scene_graph.nodes}
    node_ids = sorted({item for pair in scene_graph.blocking_dag for item in pair})
    nodes_payload: List[Dict[str, Any]] = []
    for node_id in node_ids:
        node = nodes_by_id.get(node_id)
        operation = node.operation if node else None
        color = OPERATION_COLORS.get(operation, OPERATION_COLORS[None])
        label = _node_label(node) if node else node_id
        nodes_payload.append(
            {
                "id": node_id,
                "label": label,
                "title": _node_title(node) if node else node_id,
                "color": {"background": color, "border": color},
                "font": {"color": "#ffffff", "size": 12},
                "shape": "box",
                "margin": 8,
            }
        )

    edges_payload = []
    for blocker, blocked in scene_graph.blocking_dag:
        edges_payload.append(
            {
                "from": blocker,
                "to": blocked,
                "label": "blocking",
                "title": f"{blocker} -> {blocked}",
                "arrows": "to",
                "color": {"color": RELATION_COLORS["blocking"]},
                "width": 2,
                "font": {"size": 10, "align": "middle"},
            }
        )

    return (
        '<div class="card">'
        "<h3>C2. Blocking DAG</h3>"
        f'<div id="{escape(dom_id)}" style="width:100%;height:320px;border:1px solid #d1d5db;border-radius:8px;"></div>'
        "<script>"
        "(function(){"
        f"const container=document.getElementById({json.dumps(dom_id)});"
        "if(!container||typeof vis==='undefined'){return;}"
        f"const nodes=new vis.DataSet({json.dumps(nodes_payload, ensure_ascii=True)});"
        f"const edges=new vis.DataSet({json.dumps(edges_payload, ensure_ascii=True)});"
        "const network=new vis.Network(container,{nodes:nodes,edges:edges},{"
        "layout:{hierarchical:{enabled:true,direction:'LR',sortMethod:'directed'}},"
        "physics:false,"
        "interaction:{hover:true,tooltipDelay:100}"
        "});"
        "const fitDag=()=>network.fit({animation:false});"
        "network.once('afterDrawing',fitDag);"
        "window.addEventListener('resize',fitDag);"
        "})();"
        "</script>"
        "</div>"
    )


def _render_attributes_table(scene_graph: SceneGraph) -> str:
    rows = []
    for node in scene_graph.nodes:
        attrs = node.attributes
        color_name = attrs.color_primary or "-"
        expressions = ", ".join(node.referring_expressions) if node.referring_expressions else "-"
        swatch = ""
        if attrs.color_primary:
            swatch = (
                '<span class="color-swatch" '
                f'style="background:{escape(attrs.color_primary)}" '
                f'title="{escape(attrs.color_primary)}"></span>'
            )
        rows.append(
            "<tr>"
            f"<td><div><strong>{escape(_node_label(node))}</strong></div><div style=\"color:#64748b;font-size:12px;\">{escape(_node_meta_label(node))}</div></td>"
            f"<td>{escape(node.class_name)}</td>"
            f"<td>{escape(node.operation or 'unknown')}</td>"
            f"<td>{swatch}{escape(color_name)}</td>"
            f"<td>{escape(attrs.size or '-')}</td>"
            f"<td>{escape(attrs.shape or '-')}</td>"
            f"<td>{escape(node.hierarchical_name or '-')}</td>"
            f"<td>{escape(expressions)}</td>"
            "</tr>"
        )

    return (
        '<div class="card">'
        "<h3>Attributes</h3>"
        "<table>"
        "<thead><tr><th>ID</th><th>Class</th><th>Operation</th><th>Color</th><th>Size</th><th>Shape</th><th>Hierarchical Name</th><th>Referring Expressions</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def render_scene_section(scene_graph: SceneGraph, image_base64: str = "") -> str:
    """Render the M3 scene section as a set of cards."""

    if not scene_graph.nodes:
        return '<div class="card"><h3>M3 - Scene Understanding</h3><p>No nodes</p></div>'

    prefix = _slug(getattr(scene_graph, "image_id", "scene"))
    dag_dom_id = f"{prefix}-blocking-dag"

    return (
        '<div class="scene-section">'
        "<h2>M3 - Scene Understanding</h2>"
        f"{_render_relation_overlay(scene_graph, image_base64)}"
        f"{render_three_scene(scene_graph)}"
        '<div class="card"><h3>C. Support / Blocking Structure View</h3><p style="margin:0;color:#475569;font-size:13px;">Support hierarchy stays separate from the actionable blocking DAG so base support roots remain context instead of becoming central blocker hubs.</p></div>'
        f"{_render_support_graph_card(scene_graph)}"
        f"{_build_blocking_dag_card(scene_graph, dag_dom_id)}"
        f"{_render_diagnostics_card(scene_graph)}"
        f"{_render_attributes_table(scene_graph)}"
        "</div>"
    )
