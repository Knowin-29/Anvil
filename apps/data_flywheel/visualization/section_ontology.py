"""M1 ontology section renderer."""

from __future__ import annotations

from collections import Counter
from html import escape
from typing import Any, Dict, Iterable, List, Optional

from apps.data_flywheel.types import SceneGraph, SceneNode


def _normalize_label(label: Any) -> Dict[str, Any]:
    data: Dict[str, Any]
    if isinstance(label, dict):
        data = dict(label)
    elif hasattr(label, "to_dict"):
        payload = label.to_dict()
        data = dict(payload) if isinstance(payload, dict) else {}
    elif hasattr(label, "__dict__"):
        data = dict(vars(label))
    else:
        data = {}

    observation_name = str(
        data.get("observation_name")
        or data.get("normalized_name")
        or data.get("canonical_name")
        or label
    )
    canonical_name = str(data.get("canonical_name") or observation_name)
    operation = data.get("operation") or data.get("preferred_operation") or "unknown"

    return {
        "observation_name": observation_name,
        "canonical_name": canonical_name,
        "object_class_id": data.get("object_class_id") or canonical_name,
        "object_subtype_id": data.get("object_subtype_id"),
        "matched_alias": data.get("matched_alias"),
        "operation": operation,
        "role_ids": list(data.get("role_ids") or []),
        "allowed_operations": list(data.get("allowed_operations") or []),
        "preferred_operation": data.get("preferred_operation") or operation,
        "extra_tags": list(data.get("extra_tags") or []),
        "ontology_status": data.get("ontology_status") or "unknown",
    }


def _operation_badge_class(operation: Any) -> str:
    if operation in {"pick", "push", "non_operable"}:
        return f"badge-{operation}"
    return "badge-unknown"


def _dedupe(values: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    normalized: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _format_inline_list(values: Iterable[Any]) -> str:
    normalized = _dedupe(values)
    if not normalized:
        return "—"
    return ", ".join(escape(value) for value in normalized)


def _format_chips(values: Iterable[Any], *, empty: str = "—", tone: str = "neutral") -> str:
    normalized = _dedupe(values)
    if not normalized:
        return empty
    return "".join(
        f'<span class="mini-chip mini-chip-{tone}">{escape(value)}</span>' for value in normalized
    )


def _format_badge(label: str, value: Any) -> str:
    return f'<span class="badge {label}">{escape(str(value))}</span>'


def _summary_badges(labels: List[Dict[str, Any]]) -> tuple[str, str]:
    op_counts = Counter(
        (item.get("preferred_operation") or item.get("operation") or "unknown") for item in labels
    )
    role_counts = Counter(
        role
        for item in labels
        for role in _dedupe(item.get("role_ids") or [])
    )
    op_badges = " ".join(
        _format_badge(_operation_badge_class(operation), f"{operation}: {count}")
        for operation, count in sorted(op_counts.items(), key=lambda item: str(item[0]))
    )
    role_badges = (
        " ".join(
            f'<span class="mini-chip mini-chip-neutral">{escape(role)}: {count}</span>'
            for role, count in sorted(role_counts.items(), key=lambda item: item[0])
        )
        or '<span class="mini-chip mini-chip-neutral">no explicit roles</span>'
    )
    return op_badges, role_badges


def _descriptor_axes(node: SceneNode) -> str:
    normalized = _dedupe(node.descriptor_axes or [])
    return ", ".join(normalized) if normalized else "—"


def _label_bridge_subtype_lookup(labels: List[Dict[str, Any]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for label in labels:
        subtype = label.get("object_subtype_id")
        if not subtype:
            continue
        class_key = str(label.get("object_class_id") or label.get("canonical_name") or "").strip()
        if class_key and class_key not in lookup:
            lookup[class_key] = str(subtype)
    return lookup


def _render_identity_rows(labels: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for label in labels:
        preferred_operation = label.get("preferred_operation") or label.get("operation") or "unknown"
        rows.append(
            "<tr>"
            f"<td>{escape(str(label.get('observation_name', '')))}</td>"
            f"<td>{escape(str(label.get('canonical_name', '')))}</td>"
            f"<td>{escape(str(label.get('object_class_id') or '—'))}</td>"
            f"<td>{escape(str(label.get('object_subtype_id') or '—'))}</td>"
            f"<td>{escape(str(label.get('matched_alias') or '—'))}</td>"
            f"<td>{_format_chips(label.get('role_ids') or [], tone='role')}</td>"
            f"<td>{_format_inline_list(label.get('allowed_operations') or [])}</td>"
            f"<td><span class=\"badge {_operation_badge_class(preferred_operation)}\">{escape(str(preferred_operation))}</span></td>"
            f"<td>{escape(str(label.get('ontology_status', 'unknown')))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_runtime_bridge(scene_graph: Optional[SceneGraph], labels: List[Dict[str, Any]]) -> str:
    if scene_graph is None or not scene_graph.nodes:
        return (
            '<div class="ontology-subcard">'
            "<h4>Runtime Naming Bridge</h4>"
            '<p class="ontology-note">No scene graph nodes were provided, so Layer 2 / Layer 3 runtime names are unavailable in this report.</p>'
            "</div>"
        )

    subtype_lookup = _label_bridge_subtype_lookup(labels)
    rows: List[str] = []
    for node in scene_graph.nodes:
        class_key = str(node.object_class_id or node.class_name or "").strip()
        subtype = subtype_lookup.get(class_key) or "—"
        rows.append(
            "<tr>"
            f"<td><div><strong>{escape(node.id)}</strong></div><div class=\"meta-text\">roles: {escape(', '.join(node.role_ids) or '—')}</div></td>"
            f"<td>{escape(node.class_name or '—')}</td>"
            f"<td>{escape(subtype)}</td>"
            f"<td>{escape(node.hierarchical_name or '—')}</td>"
            f"<td>{_format_inline_list(node.referring_expressions or [])}</td>"
            f"<td>{escape(_descriptor_axes(node))}</td>"
            "</tr>"
        )

    return (
        '<div class="ontology-subcard">'
        "<h4>Runtime Naming Bridge</h4>"
        '<p class="ontology-note">'
        "This is the concrete layered naming output that feeds the tabletop scene graph: Layer 1 stays stable, while Layer 2 and Layer 3 separate descriptive names from referring aliases."
        "</p>"
        "<table>"
        "<thead><tr><th>Node</th><th>Layer 1 Class</th><th>Layer 1.5 Subtype</th><th>Layer 2 `hierarchical_name`</th><th>Layer 3 `referring_expressions`</th><th>Descriptor Axes</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _render_contract_cards() -> str:
    cards = [
        (
            "Layer 1",
            "Stable class identity",
            "plate / pen / bowl / box",
        ),
        (
            "Layer 1.5",
            "Optional subtype identity",
            "ceramic_plate / plastic_box",
        ),
        (
            "Layer 2",
            "Descriptor-driven instance name",
            "`hierarchical_name`",
        ),
        (
            "Layer 3",
            "Instance-level referring aliases",
            "`referring_expressions`",
        ),
    ]
    return "".join(
        (
            '<div class="layer-card">'
            f'<div class="layer-kicker">{escape(layer)}</div>'
            f'<div class="layer-title">{escape(title)}</div>'
            f'<div class="layer-body">{escape(body)}</div>'
            "</div>"
        )
        for layer, title, body in cards
    )


def render_ontology_section(labels: List[Any], scene_graph: Optional[SceneGraph] = None) -> str:
    """Render the M1 ontology section as a standalone HTML fragment."""

    normalized = [_normalize_label(label) for label in labels]
    has_runtime_nodes = bool(scene_graph and scene_graph.nodes)
    if not normalized and not has_runtime_nodes:
        return (
            '<div class="card">'
            "<h3>M1 - Ontology</h3>"
            "<p>No ontology data</p>"
            "</div>"
        )

    op_badges, role_badges = _summary_badges(normalized) if normalized else ("", "")
    identity_rows = _render_identity_rows(normalized) if normalized else ""
    runtime_bridge = _render_runtime_bridge(scene_graph, normalized)

    identity_block = (
        '<div class="ontology-subcard">'
        "<h4>Identity / Roles / Intrinsic Operations</h4>"
        '<p class="ontology-note">'
        "Identity stays separate from role and affordance. `container` is rendered as a role/capability, not as the primary class name."
        "</p>"
        "<table>"
        "<thead><tr><th>Observation</th><th>Canonical</th><th>Layer 1 Class</th><th>Layer 1.5 Subtype</th><th>Matched Alias</th><th>Roles</th><th>Allowed Operations</th><th>Preferred</th><th>Status</th></tr></thead>"
        f"<tbody>{identity_rows}</tbody>"
        "</table>"
        "</div>"
        if normalized
        else (
            '<div class="ontology-subcard">'
            "<h4>Identity / Roles / Intrinsic Operations</h4>"
            '<p class="ontology-note">No ontology label resolutions were passed into the report.</p>'
            "</div>"
        )
    )

    return (
        '<div class="card">'
        "<style>"
        ".card{background:#fff;border-radius:8px;padding:20px;margin-bottom:16px;"
        "box-shadow:0 1px 3px rgba(0,0,0,0.1);}"
        ".badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;"
        "font-weight:600;color:#fff;margin:0 2px 4px 0;}"
        ".badge-pick{background:#4CAF50;}"
        ".badge-push{background:#FF9800;}"
        ".badge-non_operable{background:#9E9E9E;}"
        ".badge-unknown{background:#2196F3;}"
        ".mini-chip{display:inline-flex;align-items:center;padding:2px 8px;border-radius:999px;"
        "font-size:11px;font-weight:600;margin:0 6px 6px 0;border:1px solid transparent;}"
        ".mini-chip-neutral{background:#e2e8f0;color:#0f172a;border-color:#cbd5e1;}"
        ".mini-chip-role{background:#dbeafe;color:#1d4ed8;border-color:#93c5fd;}"
        ".layer-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:14px 0 18px 0;}"
        ".layer-card{padding:14px;border:1px solid #dbe4ee;border-radius:12px;background:#f8fafc;}"
        ".layer-kicker{font-size:11px;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;color:#475569;margin-bottom:6px;}"
        ".layer-title{font-size:14px;font-weight:700;color:#0f172a;margin-bottom:6px;}"
        ".layer-body{font-size:13px;line-height:1.45;color:#475569;}"
        ".ontology-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:16px;}"
        ".ontology-subcard{border:1px solid #e2e8f0;border-radius:12px;padding:16px;background:#ffffff;}"
        ".ontology-note{margin:0 0 12px 0;color:#475569;font-size:13px;line-height:1.5;}"
        ".meta-text{color:#64748b;font-size:12px;}"
        "table{width:100%;border-collapse:collapse;font-size:13px;}"
        "th,td{padding:8px 12px;border-bottom:1px solid #eee;text-align:left;vertical-align:top;}"
        "th{background:#f8f9fa;border-bottom:2px solid #dee2e6;}"
        "code{background:#e2e8f0;padding:1px 5px;border-radius:4px;}"
        "</style>"
        "<h3>M1 - Ontology</h3>"
        '<p class="ontology-note">'
        "Ontology v2 is rendered with the approved layered contract: stable identity, optional subtype, descriptor-driven runtime naming, and separate referring aliases."
        "</p>"
        f'<p style="margin:0 0 10px 0;">Intrinsic operation distribution: {op_badges or "—"}</p>'
        f'<p style="margin:0 0 14px 0;">Role distribution: {role_badges}</p>'
        '<div class="ontology-subcard">'
        "<h4>Layered Naming Contract</h4>"
        '<p class="ontology-note">'
        "Role/capability and intrinsic affordance stay orthogonal to naming. The runtime scene graph should expose Layer 2 names without collapsing them back into naked class nouns."
        "</p>"
        f'<div class="layer-grid">{_render_contract_cards()}</div>'
        "</div>"
        '<div class="ontology-grid">'
        f"{identity_block}"
        f"{runtime_bridge}"
        "</div>"
        "</div>"
    )
