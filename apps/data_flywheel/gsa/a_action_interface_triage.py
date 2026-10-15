"""Offline triage helpers for A_action_interface review candidates."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .a_action_interface_review import (
    make_review_decision_template,
    normalize_action_interface_candidates,
    promote_action_interface_candidates,
    read_jsonl,
    write_jsonl,
    write_review_html,
)
from .taxonomy import SourceProvenance


PREFERRED_OBJECT_TERMS = (
    "can",
    "bowl",
    "cup",
    "mug",
    "spoon",
    "fork",
    "knife",
    "spatula",
    "carrot",
    "banana",
    "apple",
    "tomato",
    "sushi",
    "bottle",
    "brush",
    "cloth",
    "towel",
    "pot",
    "pan",
    "plate",
    "lid",
    "block",
    "toy",
    "fruit",
    "vegetable",
    "container",
    "box",
    "bag",
    "sponge",
)

FILTERED_OBJECT_TERMS = (
    "microwave",
    "faucet",
    "stove",
    "burner",
    "knob",
    "sink",
    "door",
    "stand",
    "shelf",
    "cabinet",
    "drawer",
    "counter",
    "wall",
    "table",
    "robot",
    "arm",
    "gripper",
    "hand",
    "handle",
    "surface",
    "floor",
    "oven",
)


def normalize_object_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"[_/-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def object_triage_class(name: Any) -> str:
    normalized = normalize_object_name(name)
    if not normalized:
        return "unknown"
    if _contains_term(normalized, FILTERED_OBJECT_TERMS):
        return "filtered_static_or_background"
    if _contains_term(normalized, PREFERRED_OBJECT_TERMS):
        return "preferred_manipulable"
    return "unknown"


def triage_candidates(
    candidates: Iterable[Dict[str, Any]],
    *,
    max_candidates: int = 200,
    max_per_object_name: int = 24,
    max_per_image: int = 6,
    include_unknown_objects: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    normalized = normalize_action_interface_candidates(candidates)
    annotated = [_annotate_candidate(row) for row in normalized]

    eligible: List[Dict[str, Any]] = []
    filtered_reasons: Counter[str] = Counter()
    for row in annotated:
        reason = _candidate_filter_reason(row, include_unknown_objects=include_unknown_objects)
        if reason:
            filtered_reasons[reason] += 1
            continue
        eligible.append(row)

    eligible.sort(key=_sort_key)
    selected = _select_balanced_review_set(
        eligible,
        max_candidates=max_candidates,
        max_per_object_name=max_per_object_name,
        max_per_image=max_per_image,
        filtered_reasons=filtered_reasons,
    )

    summary = build_triage_summary(annotated, selected, filtered_reasons)
    return selected, summary


def build_triage_summary(
    candidates: Iterable[Dict[str, Any]],
    selected: Iterable[Dict[str, Any]],
    filtered_reasons: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    rows = list(candidates)
    selected_rows = list(selected)
    source_counts = Counter(str(row.get("source_provenance", "")) for row in rows)
    verifier_counts = Counter(str(row.get("verifier_status", "")) for row in rows)
    action_counts = Counter(str(row.get("action_type", "")) for row in rows)
    object_counts = Counter(normalize_object_name(row.get("object_name")) for row in rows)
    object_class_counts = Counter(row.get("triage", {}).get("object_class", "unknown") for row in rows)

    selected_object_counts = Counter(normalize_object_name(row.get("object_name")) for row in selected_rows)
    selected_action_counts = Counter(str(row.get("action_type", "")) for row in selected_rows)
    selected_object_class_counts = Counter(
        row.get("triage", {}).get("object_class", "unknown") for row in selected_rows
    )

    by_object = []
    for object_name, count in object_counts.most_common():
        object_rows = [row for row in rows if normalize_object_name(row.get("object_name")) == object_name]
        vlm_count = sum(1 for row in object_rows if row.get("source_provenance") == SourceProvenance.VLM.value)
        geometry_valid_count = sum(1 for row in object_rows if _geometry_valid(row))
        by_object.append(
            {
                "object_name": object_name,
                "candidate_count": count,
                "vlm_count": vlm_count,
                "geometry_valid_count": geometry_valid_count,
                "selected_count": selected_object_counts.get(object_name, 0),
                "triage_class": object_triage_class(object_name),
            }
        )

    return {
        "candidate_count": len(rows),
        "selected_count": len(selected_rows),
        "source_counts": dict(source_counts),
        "verifier_counts": dict(verifier_counts),
        "action_counts": dict(action_counts),
        "object_class_counts": dict(object_class_counts),
        "selected_action_counts": dict(selected_action_counts),
        "selected_object_class_counts": dict(selected_object_class_counts),
        "filtered_reasons": dict(filtered_reasons or {}),
        "top_object_names": by_object[:40],
        "selected_top_object_names": selected_object_counts.most_common(40),
        "vlm_candidate_count": source_counts.get(SourceProvenance.VLM.value, 0),
        "adapter_candidate_count": source_counts.get("adapter", 0),
        "geometry_valid_count": sum(1 for row in rows if _geometry_valid(row)),
        "stable_training_label_count": sum(1 for row in rows if row.get("stable_training_label")),
    }


def write_triage_outputs(
    *,
    candidates_path: Path,
    output_dir: Path,
    max_candidates: int = 200,
    max_per_object_name: int = 24,
    max_per_image: int = 6,
    include_unknown_objects: bool = True,
) -> Dict[str, Any]:
    raw_candidates = read_jsonl(candidates_path)
    selected, summary = triage_candidates(
        raw_candidates,
        max_candidates=max_candidates,
        max_per_object_name=max_per_object_name,
        max_per_image=max_per_image,
        include_unknown_objects=include_unknown_objects,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    priority_path = output_dir / "review_priority_candidates.jsonl"
    template_path = output_dir / "review_decisions.template.jsonl"
    html_path = output_dir / "review_priority.html"
    reviewed_path = output_dir / "review_priority_reviewed_candidates.jsonl"
    pack_path = output_dir / "A_action_interface_priority_pack_manifest.jsonl"
    promotion_report_path = output_dir / "review_priority_promotion_report.json"
    summary_path = output_dir / "overnight_candidate_analysis.json"
    markdown_path = output_dir / "overnight_candidate_analysis.md"

    write_jsonl(priority_path, selected)
    write_jsonl(template_path, make_review_decision_template(selected))
    write_review_html(html_path, selected)

    promotion = promote_action_interface_candidates(selected)
    write_jsonl(reviewed_path, promotion.candidates)
    write_jsonl(pack_path, promotion.stable_labels)

    output_summary = {
        **summary,
        "input_candidates_path": str(candidates_path),
        "review_priority_candidates_path": str(priority_path),
        "review_decisions_template_path": str(template_path),
        "review_html_path": str(html_path),
        "reviewed_candidates_path": str(reviewed_path),
        "stable_pack_path": str(pack_path),
        "promotion_report_path": str(promotion_report_path),
        "promotion_summary": promotion.summary,
    }
    summary_path.write_text(json.dumps(output_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    promotion_report_path.write_text(json.dumps(promotion.summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_triage_markdown(output_summary), encoding="utf-8")
    output_summary["summary_path"] = str(summary_path)
    output_summary["markdown_report_path"] = str(markdown_path)
    return output_summary


def render_triage_markdown(summary: Mapping[str, Any]) -> str:
    source_counts = summary.get("source_counts", {})
    filtered_reasons = summary.get("filtered_reasons", {})
    top_objects = summary.get("top_object_names", [])[:20]
    selected_objects = summary.get("selected_top_object_names", [])[:20]

    lines = [
        "# A_action_interface Candidate Triage Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| candidate_count | {summary.get('candidate_count', 0)} |",
        f"| selected_count | {summary.get('selected_count', 0)} |",
        f"| vlm_candidate_count | {summary.get('vlm_candidate_count', 0)} |",
        f"| adapter_candidate_count | {summary.get('adapter_candidate_count', 0)} |",
        f"| geometry_valid_count | {summary.get('geometry_valid_count', 0)} |",
        f"| stable_training_label_count | {summary.get('stable_training_label_count', 0)} |",
        "",
        "## Source Counts",
        "",
        "| Source | Count |",
        "|---|---:|",
    ]
    for source, count in sorted(source_counts.items()):
        lines.append(f"| `{source}` | {count} |")

    lines.extend(["", "## Filtered Reasons", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in sorted(filtered_reasons.items()):
        lines.append(f"| `{reason}` | {count} |")

    lines.extend(["", "## Top Objects", "", "| Object | Count | VLM | Geometry Valid | Selected | Class |", "|---|---:|---:|---:|---:|---|"])
    for row in top_objects:
        lines.append(
            "| {object_name} | {candidate_count} | {vlm_count} | {geometry_valid_count} | "
            "{selected_count} | `{triage_class}` |".format(**row)
        )

    lines.extend(["", "## Selected Objects", "", "| Object | Selected Count |", "|---|---:|"])
    for item in selected_objects:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            lines.append(f"| {item[0]} | {item[1]} |")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- review priority candidates: `{summary.get('review_priority_candidates_path')}`",
            f"- review decisions template: `{summary.get('review_decisions_template_path')}`",
            f"- review HTML: `{summary.get('review_html_path')}`",
            f"- stable pack dry-run output: `{summary.get('stable_pack_path')}`",
            "",
            "The stable pack is expected to be empty until human review decisions are approved.",
        ]
    )
    return "\n".join(lines) + "\n"


def _annotate_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(candidate)
    object_class = object_triage_class(row.get("object_name"))
    score = _priority_score(row, object_class)
    row["triage"] = {
        "object_class": object_class,
        "priority_score": score,
        "object_name_normalized": normalize_object_name(row.get("object_name")),
    }
    return row


def _candidate_filter_reason(candidate: Mapping[str, Any], *, include_unknown_objects: bool) -> str:
    if candidate.get("source_provenance") != SourceProvenance.VLM.value:
        return "source_not_vlm"
    verification = candidate.get("verification") or {}
    if not verification.get("valid_schema", False):
        return "invalid_schema"
    if not verification.get("geometry_valid", False):
        return "invalid_geometry"
    object_class = candidate.get("triage", {}).get("object_class")
    if object_class == "filtered_static_or_background":
        return "filtered_static_or_background_object"
    if object_class == "unknown" and not include_unknown_objects:
        return "unknown_object"
    return ""


def _priority_score(candidate: Mapping[str, Any], object_class: str) -> float:
    score = 0.0
    if candidate.get("source_provenance") == SourceProvenance.VLM.value:
        score += 1000.0
    if _geometry_valid(candidate):
        score += 200.0
    if object_class == "preferred_manipulable":
        score += 120.0
    elif object_class == "unknown":
        score += 30.0
    elif object_class == "filtered_static_or_background":
        score -= 500.0
    if not candidate.get("risk_flags"):
        score += 10.0
    area_ratio = _bbox_area_ratio(candidate)
    if area_ratio is not None:
        score += max(0.0, 1.0 - min(area_ratio, 1.0)) * 20.0
    return round(score, 6)


def _sort_key(candidate: Mapping[str, Any]) -> Tuple[float, str, str]:
    triage = candidate.get("triage") or {}
    return (
        -float(triage.get("priority_score", 0.0)),
        str(candidate.get("object_name", "")),
        str(candidate.get("label_id", "")),
    )


def _select_balanced_review_set(
    eligible: List[Dict[str, Any]],
    *,
    max_candidates: int,
    max_per_object_name: int,
    max_per_image: int,
    filtered_reasons: Counter[str],
) -> List[Dict[str, Any]]:
    by_action: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_action[str(row.get("action_type", "unknown"))].append(row)

    action_order = [action for action in ("pick", "push") if action in by_action]
    action_order.extend(sorted(action for action in by_action if action not in set(action_order)))
    action_indices = {action: 0 for action in action_order}

    selected: List[Dict[str, Any]] = []
    by_object: Counter[str] = Counter()
    by_image: Counter[str] = Counter()
    while len(selected) < max_candidates and action_order:
        progressed = False
        for action in list(action_order):
            bucket = by_action[action]
            while action_indices[action] < len(bucket):
                row = bucket[action_indices[action]]
                action_indices[action] += 1
                object_key = normalize_object_name(row.get("object_name"))
                image_key = str(row.get("image_path") or row.get("sample_id") or "")
                if max_per_object_name > 0 and by_object[object_key] >= max_per_object_name:
                    filtered_reasons["per_object_cap"] += 1
                    continue
                if max_per_image > 0 and by_image[image_key] >= max_per_image:
                    filtered_reasons["per_image_cap"] += 1
                    continue
                selected.append(row)
                by_object[object_key] += 1
                by_image[image_key] += 1
                progressed = True
                break
            if len(selected) >= max_candidates:
                break
        if not progressed:
            break
    return selected


def _geometry_valid(candidate: Mapping[str, Any]) -> bool:
    verification = candidate.get("verification") or {}
    return bool(verification.get("geometry_valid"))


def _bbox_area_ratio(candidate: Mapping[str, Any]) -> Optional[float]:
    bbox = candidate.get("bbox")
    width = candidate.get("image_width")
    height = candidate.get("image_height")
    if not (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
        and isinstance(width, (int, float))
        and isinstance(height, (int, float))
        and width > 0
        and height > 0
    ):
        return None
    bbox_area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    return bbox_area / (float(width) * float(height))


def _contains_term(text: str, terms: Iterable[str]) -> bool:
    for term in terms:
        normalized_term = normalize_object_name(term)
        if not normalized_term:
            continue
        pattern = r"(^|\s)" + re.escape(normalized_term) + r"($|\s)"
        if re.search(pattern, text):
            return True
    return False
