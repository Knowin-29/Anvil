"""M4 training data section renderer."""

from __future__ import annotations

import base64
import json
import os
import re
from html import escape
from io import BytesIO
from typing import Any, Dict, Iterable, List, Mapping

from PIL import Image


VALIDATOR_COLUMNS = [
    ("action_consistency", "ActionConsistency"),
    ("coordinate_range", "CoordinateRange"),
    ("image_exists", "ImageExists"),
    ("instruction_language", "InstructionLanguage"),
    ("ontology_consistency", "OntologyConsistency"),
    ("point_in_mask", "PointInMask"),
    ("referencing_uniqueness", "ReferencingUniqueness"),
    ("thinking_format", "ThinkingFormat"),
]


REX_TOKEN_PATTERNS = [
    (re.compile(r"(<\|user\|>)"), "token-user"),
    (re.compile(r"(<\|end\|>)"), "token-user"),
    (re.compile(r"(<\|thinking_start\|>)"), "token-thinking"),
    (re.compile(r"(<\|thinking_end\|>)"), "token-thinking"),
    (re.compile(r"(<\|action_start\|>)"), "token-action"),
    (re.compile(r"(<\|action_end\|>)"), "token-action"),
    (re.compile(r"(<\|object_ref_start\|>.*?<\|object_ref_end\|>)"), "token-ref"),
    (re.compile(r"(<\|box_start\|>.*?<\|box_end\|>)"), "token-box"),
]


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        data = value.to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _normalize_validator_name(name: Any) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    if "_" in raw:
        return raw.lower()
    return re.sub(r"(?<!^)(?=[A-Z])", "_", raw).lower()


def _display_validator_name(normalized_name: str) -> str:
    for key, display in VALIDATOR_COLUMNS:
        if key == normalized_name:
            return display
    parts = [part.capitalize() for part in normalized_name.split("_") if part]
    return "".join(parts) or normalized_name


def _thinking_steps(plan_result: Any) -> List[Any]:
    if isinstance(plan_result, Mapping):
        thinking = plan_result.get("thinking")
    else:
        thinking = getattr(plan_result, "thinking", None)
    if thinking is None:
        return []
    return list(getattr(thinking, "steps", thinking.get("steps", [])) if isinstance(thinking, Mapping) else getattr(thinking, "steps", []))


def _plan_status(plan_result: Any) -> str:
    if isinstance(plan_result, Mapping):
        return str(plan_result.get("status", "unknown"))
    return str(getattr(plan_result, "status", "unknown"))


def _plan_instruction(plan_result: Any) -> str:
    if isinstance(plan_result, Mapping):
        return str(plan_result.get("instruction", ""))
    return str(getattr(plan_result, "instruction", ""))


def _plan_actions(plan_result: Any) -> List[Any]:
    if isinstance(plan_result, Mapping):
        actions = plan_result.get("actions")
    else:
        actions = getattr(plan_result, "actions", None)
    return list(actions or [])


def _plan_notes(plan_result: Any) -> List[str]:
    if isinstance(plan_result, Mapping):
        notes = plan_result.get("notes")
    else:
        notes = getattr(plan_result, "notes", None)
    return [str(note) for note in (notes or []) if note]


def _action_node(action: Any) -> str:
    data = _mapping(action)
    action_type = escape(str(data.get("type", "action")))
    target = escape(str(data.get("target", "-")))
    destination = escape(str(data.get("destination", "")))
    reason = escape(str(data.get("reason", "")))
    summary = target
    if destination:
        summary = f"{summary} -> {destination}"
    description = f"<strong>Action: {action_type}</strong><br>{summary}"
    if reason:
        description += f"<br>{reason}"
    return f'<span class="flow-node flow-node-action">{description}</span>'


def _render_thinking_traces(plan_results: Iterable[Any]) -> str:
    cards = []
    for plan in plan_results:
        steps = _thinking_steps(plan)
        status = escape(_plan_status(plan))
        instruction = escape(_plan_instruction(plan))
        step_nodes = []
        for step in steps:
            perception_type = escape(str(getattr(step, "perception_type", step.get("perception_type", "")) if isinstance(step, Mapping) else getattr(step, "perception_type", "")))
            target = escape(str(getattr(step, "target", step.get("target", "")) if isinstance(step, Mapping) else getattr(step, "target", "")))
            content = escape(str(getattr(step, "content", step.get("content", "")) if isinstance(step, Mapping) else getattr(step, "content", "")))
            step_nodes.append(
                f'<span class="flow-node"><strong>{perception_type}</strong><br>{target}<br>{content}</span>'
            )
        action_nodes = [_action_node(action) for action in _plan_actions(plan)]
        nodes = step_nodes + action_nodes
        flow_html = "<span class=\"flow-arrow\">→</span>".join(nodes) if nodes else "<p>No thinking trace</p>"
        notes = _plan_notes(plan)
        notes_html = ""
        if notes:
            notes_html = "<p class=\"plan-notes\"><strong>Notes:</strong> " + escape(" | ".join(notes)) + "</p>"
        cards.append(
            '<div class="card">'
            f"<h4>{instruction or 'Untitled plan'}</h4>"
            f'<p><span class="badge badge-{status}">{status}</span></p>'
            f'<div class="thinking-flow">{flow_html}</div>'
            f"{notes_html}"
            "</div>"
        )
    if not cards:
        return '<div class="card"><h4>Thinking Trace</h4><p>No data</p></div>'
    return '<div class="card"><h3>Thinking Trace</h3>' + "".join(cards) + "</div>"


def _validator_names(rejected_entries: Iterable[Any]) -> List[str]:
    names = [name for name, _ in VALIDATOR_COLUMNS]
    for entry in rejected_entries:
        data = _mapping(entry)
        for key in ("critical_fails", "warning_fails"):
            for name in data.get(key, []) or []:
                normalized = _normalize_validator_name(name)
                if normalized and normalized not in names:
                    names.append(normalized)
    return names


def _entry_reason_map(entry: Mapping[str, Any]) -> Dict[str, str]:
    reason_map: Dict[str, str] = {}
    report = entry.get("report")
    report_results = getattr(report, "results", None)
    if report_results:
        for result in report_results:
            result_data = _mapping(result)
            name = _normalize_validator_name(result_data.get("validator_name"))
            reason = str(result_data.get("reason") or "").strip()
            if name and reason:
                reason_map[name] = reason

    joined_reasons = "; ".join(str(item) for item in (entry.get("reasons") or []) if item)
    if joined_reasons:
        for name in (entry.get("critical_fails") or []) + (entry.get("warning_fails") or []):
            normalized = _normalize_validator_name(name)
            if normalized and normalized not in reason_map:
                reason_map[normalized] = joined_reasons
    return reason_map


def _render_validator_heatmap(rejected_entries: Iterable[Any], accepted_count: int = 0) -> str:
    entries = [_mapping(entry) for entry in rejected_entries]
    validators = _validator_names(entries)
    total_samples = accepted_count + len(entries)
    if not entries:
        return '<div class="card"><h3>Validator Heatmap</h3><p>No data</p></div>'

    detail_id = f"validator-detail-{len(entries)}-{accepted_count}"
    header = "".join(f"<th>{escape(_display_validator_name(name))}</th>" for name in validators)
    fail_counts = {name: 0 for name in validators}
    warning_counts = {name: 0 for name in validators}
    rows = []
    for entry in entries:
        sample_id = escape(str(entry.get("sample_id", "")))
        instruction = escape(str(entry.get("instruction", "")))
        critical = {_normalize_validator_name(name) for name in (entry.get("critical_fails") or [])}
        warning = {_normalize_validator_name(name) for name in (entry.get("warning_fails") or [])}
        reason_map = _entry_reason_map(entry)
        cells = []
        for name in validators:
            if name in critical:
                cls = "fail"
                title = reason_map.get(name) or "critical fail"
                value = "fail"
                fail_counts[name] += 1
            elif name in warning:
                cls = "warning"
                title = reason_map.get(name) or "warning"
                value = "warn"
                warning_counts[name] += 1
            else:
                cls = "pass"
                title = "pass"
                value = "pass"
            cells.append(
                f'<td class="heatmap-cell {cls}" title="{escape(title)}" data-reason="{escape(title)}" '
                f'onclick="document.getElementById({json.dumps(detail_id)}).textContent=this.dataset.reason;">{value}</td>'
            )
        rows.append(f"<tr><td>{sample_id}</td><td>{instruction}</td>{''.join(cells)}</tr>")

    stats_cells = []
    denominator = max(total_samples, len(entries), 1)
    for name in validators:
        pass_count = max(denominator - fail_counts[name] - warning_counts[name], 0)
        rate = 100.0 * pass_count / denominator
        stats_cells.append(
            f'<td title="pass={pass_count}, warning={warning_counts[name]}, fail={fail_counts[name]}">{rate:.1f}%</td>'
        )

    return (
        '<div class="card">'
        '<h3>Validator Heatmap</h3>'
        '<p style="color:#6b7280;font-size:12px;margin-bottom:8px;">'
        'Pass rate estimates treat accepted samples as pass-only because only rejected validator reports are retained.'
        '</p>'
        '<div class="heatmap-wrap">'
        '<table>'
        f"<thead><tr><th>Sample</th><th>Instruction</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"<tfoot><tr><th colspan=\"2\">Pass rate</th>{''.join(stats_cells)}</tr></tfoot>"
        "</table>"
        "</div>"
        f'<div id="{detail_id}" class="validator-detail" style="margin-top:12px;padding:10px 12px;background:#f8fafc;border-radius:8px;border:1px solid #e5e7eb;">'
        "Click a heatmap cell to inspect its validator reason."
        "</div>"
        "</div>"
    )


def _highlight_rex_omni(text: str) -> str:
    if not text:
        return "<p>No Rex-Omni text</p>"

    raw = text
    placeholders = []
    for pattern, class_name in REX_TOKEN_PATTERNS:
        def _repl(match):
            token = match.group(1)
            placeholder = f"@@TOKEN_{len(placeholders)}@@"
            placeholders.append((placeholder, class_name, token))
            return placeholder

        raw = pattern.sub(_repl, raw)

    highlighted = escape(raw)
    for placeholder, class_name, token in placeholders:
        highlighted = highlighted.replace(
            placeholder,
            f'<span class="{class_name}">{escape(token)}</span>',
        )
    return f'<pre class="rex-omni">{highlighted}</pre>'


def _image_data_uri(image_path: str, max_width: int = 320) -> str:
    if not image_path or not os.path.exists(image_path):
        return ""
    image = Image.open(image_path).convert("RGB")
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, max(1, int(image.height * ratio))), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _sample_image_size(data: Mapping[str, Any]) -> tuple[int, int]:
    raw = data.get("image_size")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return max(int(raw[0]), 1), max(int(raw[1]), 1)
        except (TypeError, ValueError):
            pass
    return 1000, 1000


def _quantized_point_to_pixel(point: Any, image_size: tuple[int, int]) -> tuple[float, float] | None:
    if not point or len(point) < 2:
        return None
    try:
        width, height = image_size
        x = float(point[0]) / 999.0 * width
        y = float(point[1]) / 999.0 * height
    except (TypeError, ValueError):
        return None
    return x, y


def _render_sample_thumbnail(data: Mapping[str, Any]) -> str:
    image_path = str(data.get("image_path", ""))
    image_uri = _image_data_uri(image_path)
    if not image_uri:
        return '<div class="sample-thumb"><p style="color:#6b7280;margin:0;">Image unavailable</p></div>'

    image_size = _sample_image_size(data)
    points: List[str] = []
    for obj in data.get("scene_objects") or []:
        obj_data = _mapping(obj)
        point = _quantized_point_to_pixel(obj_data.get("grasp_point_quantized"), image_size)
        if point is None:
            continue
        points.append(
            f'<g class="quantized-overlay"><circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="10" fill="#38bdf8" fill-opacity="0.85" stroke="#ffffff" stroke-width="3" />'
            f'<text x="{point[0] + 12:.1f}" y="{point[1] - 12:.1f}" fill="#0f172a" font-size="22" font-weight="700" paint-order="stroke" stroke="#ffffff" stroke-width="4">{escape(str(obj_data.get("id", "object")))}</text></g>'
        )

    for action in data.get("actions") or []:
        action_data = _mapping(action)
        for field, color in (("grasp_point_quantized", "#f59e0b"), ("place_point_quantized", "#22c55e")):
            point = _quantized_point_to_pixel(action_data.get(field), image_size)
            if point is None:
                continue
            label = f"{action_data.get('type', 'action')}:{action_data.get('target', '-')}"
            points.append(
                f'<g class="quantized-overlay"><circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="8" fill="{color}" fill-opacity="0.9" stroke="#ffffff" stroke-width="3" />'
                f'<text x="{point[0] + 10:.1f}" y="{point[1] + 18:.1f}" fill="#ffffff" font-size="20" font-weight="700" paint-order="stroke" stroke="#0f172a" stroke-width="5">{escape(label)}</text></g>'
            )

    width, height = image_size
    return (
        '<div class="sample-thumb">'
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img" aria-label="Training sample thumbnail">'
        f'<image href="{escape(image_uri)}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none" />'
        f"{''.join(points)}"
        "</svg>"
        "</div>"
    )


def _render_sample_previews(training_samples: Iterable[Any]) -> str:
    cards = []
    for sample in training_samples:
        data = _mapping(sample)
        sample_id = escape(str(data.get("sample_id", "")))
        instruction = escape(str(data.get("instruction", "")))
        rex_text = str(data.get("rex_omni_text", ""))
        cards.append(
            '<div class="card">'
            f"<h4>{sample_id or 'Sample'}</h4>"
            '<div style="display:grid;grid-template-columns:minmax(220px,320px) 1fr;gap:16px;align-items:start;">'
            f'<div>{_render_sample_thumbnail(data)}</div>'
            '<div>'
            f"<p>{instruction}</p>"
            f"{_highlight_rex_omni(rex_text)}"
            "</div>"
            "</div>"
            "</div>"
        )
    if not cards:
        return '<div class="card"><h3>Rex-Omni Preview</h3><p>No data</p></div>'
    return '<div class="card"><h3>Rex-Omni Preview</h3>' + "".join(cards) + "</div>"


def render_training_section(plan_results: Iterable[Any], training_samples: Iterable[Any], rejected_entries: Iterable[Any]) -> str:
    """Render the M4 training section as a self-contained HTML fragment."""

    plan_results = list(plan_results or [])
    training_samples = list(training_samples or [])
    rejected_entries = list(rejected_entries or [])

    return (
        '<div class="card">'
        "<h3>M4 - Training Data</h3>"
        "<style>"
        ".thinking-flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px;}"
        ".flow-node{display:inline-block;padding:8px 12px;border-radius:6px;border:2px solid #1f2937;background:#f8fafc;font-size:12px;}"
        ".flow-node-action{border-color:#f59e0b;background:#fff7ed;}"
        ".flow-arrow{color:#9ca3af;margin:0 2px;}"
        ".heatmap-wrap{overflow:auto;}"
        ".heatmap-cell{min-width:54px;text-align:center;font-size:12px;color:#fff;border:1px solid #fff;}"
        ".heatmap-cell.pass{background:#4CAF50;}"
        ".heatmap-cell.fail{background:#ef4444;}"
        ".heatmap-cell.warning{background:#f59e0b;}"
        ".heatmap-cell.skip{background:#6b7280;}"
        ".rex-omni{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;}"
        ".token-user{color:#60a5fa;font-weight:600;}"
        ".token-thinking{color:#4ade80;}"
        ".token-action{color:#f59e0b;}"
        ".token-ref{color:#c084fc;}"
        ".token-box{color:#f87171;}"
        ".badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;color:#fff;}"
        ".badge-success{background:#16a34a;}"
        ".badge-scope_violation{background:#dc2626;}"
        ".badge-ambiguous{background:#d97706;}"
        ".badge-unresolved{background:#0ea5e9;}"
        ".badge-blocked_unresolvable{background:#7c3aed;}"
        ".badge-unknown{background:#6b7280;}"
        ".plan-notes{margin-top:8px;color:#4b5563;font-size:12px;}"
        ".sample-thumb{background:#e2e8f0;border-radius:10px;overflow:hidden;border:1px solid #cbd5e1;}"
        ".sample-thumb svg{display:block;width:100%;height:auto;}"
        "</style>"
        + _render_thinking_traces(plan_results)
        + _render_validator_heatmap(rejected_entries, accepted_count=len(training_samples))
        + _render_sample_previews(training_samples)
        + "</div>"
    )
