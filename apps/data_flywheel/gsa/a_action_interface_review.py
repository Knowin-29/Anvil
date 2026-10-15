"""Review and promotion helpers for A_action_interface full-loop runs."""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from PIL import Image

from .a_action_interface_contract import (
    ACTION_INTERFACE_SCHEMA_VERSION,
    LABEL_STATUS_CANDIDATE,
    LABEL_STATUS_STABLE_TRAINING_LABEL,
    LABEL_STATUS_TEACHER_GENERATED,
    REVIEW_DECISION_APPROVE,
    REVIEW_DECISION_UNCERTAIN,
    REVIEW_DECISION_PENDING,
    REVIEW_DECISION_REJECT,
    REVIEW_DECISIONS,
    ensure_action_interface_label_id,
    verify_action_interface_candidate,
)
from .taxonomy import Bucket, SourceProvenance, VerifierStatus


STATUS_PROMOTION_BLOCKERS = {
    f"label_status_{LABEL_STATUS_CANDIDATE}",
    f"label_status_{LABEL_STATUS_TEACHER_GENERATED}",
    "not_auto_verifiable",
}


@dataclass
class PromotionResult:
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    stable_labels: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def load_action_interface_labels(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "labels" / Bucket.A_ACTION_INTERFACE.value / f"{Bucket.A_ACTION_INTERFACE.value}.jsonl"
    return read_jsonl(path)


def normalize_action_interface_candidate(record: Dict[str, Any]) -> Dict[str, Any]:
    out = ensure_action_interface_label_id(record)
    out.setdefault("schema_version", ACTION_INTERFACE_SCHEMA_VERSION)
    if not out.get("label_status"):
        out["label_status"] = (
            LABEL_STATUS_TEACHER_GENERATED
            if out.get("source_provenance") == SourceProvenance.VLM.value
            else LABEL_STATUS_CANDIDATE
        )
    out.setdefault("stable_training_label", False)
    out["stable_training_label"] = bool(out["stable_training_label"])
    verification = verify_action_interface_candidate(out)
    out["verification"] = verification.to_dict()
    out["promotion_blockers"] = verification.promotion_blockers
    return out


def normalize_action_interface_candidates(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_action_interface_candidate(record) for record in records]


def make_review_decision_template(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "label_id": candidate.get("label_id"),
                "decision": REVIEW_DECISION_PENDING,
                "reviewer": "",
                "reviewed_at": "",
                "notes": "",
                "reject_reason": "",
            }
        )
    return rows


def load_review_decisions(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not path or not path.exists():
        return {}
    decisions: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        label_id = row.get("label_id")
        decision = str(row.get("decision", REVIEW_DECISION_PENDING)).strip().lower()
        if not label_id:
            continue
        if decision not in REVIEW_DECISIONS:
            decision = REVIEW_DECISION_PENDING
        normalized = dict(row)
        normalized["decision"] = decision
        decisions[str(label_id)] = normalized
    return decisions


def promote_action_interface_candidates(
    candidates: Iterable[Dict[str, Any]],
    decisions: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> PromotionResult:
    decisions = decisions or {}
    reviewed: List[Dict[str, Any]] = []
    stable: List[Dict[str, Any]] = []
    decision_counts: Dict[str, int] = {
        REVIEW_DECISION_APPROVE: 0,
        REVIEW_DECISION_REJECT: 0,
        REVIEW_DECISION_UNCERTAIN: 0,
        REVIEW_DECISION_PENDING: 0,
    }
    source_counts: Dict[str, int] = {}

    for raw in candidates:
        candidate = normalize_action_interface_candidate(raw)
        label_id = str(candidate.get("label_id"))
        decision_row = decisions.get(label_id, {})
        decision = str(decision_row.get("decision", REVIEW_DECISION_PENDING)).strip().lower()
        if decision not in REVIEW_DECISIONS:
            decision = REVIEW_DECISION_PENDING

        decision_counts[decision] += 1
        source = str(candidate.get("source_provenance", ""))
        source_counts[source] = source_counts.get(source, 0) + 1

        blockers = _promotion_blockers_for_candidate(candidate, decision)
        candidate["review_decision"] = decision
        candidate["review"] = dict(decision_row) if decision_row else {}
        candidate["stable_training_label"] = False
        candidate["promotion_blockers"] = blockers

        if not blockers:
            promoted = dict(candidate)
            promoted["label_status"] = LABEL_STATUS_STABLE_TRAINING_LABEL
            promoted["stable_training_label"] = True
            promoted["verifier_status"] = VerifierStatus.AUTO_VERIFIABLE.value
            promoted["promotion_blockers"] = []
            promoted["review_decision"] = REVIEW_DECISION_APPROVE
            promoted_verification = verify_action_interface_candidate(promoted)
            promoted["verification"] = promoted_verification.to_dict()
            if promoted_verification.promotion_eligible:
                stable.append(promoted)
            else:
                candidate["promotion_blockers"] = promoted_verification.promotion_blockers or promoted_verification.errors

        reviewed.append(candidate)

    summary = {
        "candidate_count": len(reviewed),
        "stable_training_label_count": len(stable),
        "decision_counts": decision_counts,
        "source_counts": source_counts,
        "heuristic_candidate_count": sum(
            1 for row in reviewed if row.get("verifier_status") == VerifierStatus.HEURISTIC_ONLY.value
        ),
    }
    return PromotionResult(candidates=reviewed, stable_labels=stable, summary=summary)


def _promotion_blockers_for_candidate(candidate: Dict[str, Any], decision: str) -> List[str]:
    verification = verify_action_interface_candidate(candidate)
    blockers = [
        blocker
        for blocker in verification.promotion_blockers
        if blocker not in STATUS_PROMOTION_BLOCKERS
    ]
    blockers.extend(verification.errors)

    if decision != REVIEW_DECISION_APPROVE:
        blockers.append(f"review_decision_{decision}")
    if candidate.get("source_provenance") != SourceProvenance.VLM.value:
        blockers.append("source_not_vlm")
    if candidate.get("verifier_status") == VerifierStatus.HEURISTIC_ONLY.value:
        blockers.append("heuristic_only")
    if candidate.get("stable_training_label"):
        blockers.append("already_marked_stable_before_promotion")
    return sorted(set(blockers))


def write_review_html(path: Path, candidates: Iterable[Dict[str, Any]]) -> None:
    rows = list(candidates)
    cards = "\n".join(_candidate_card(row) for row in rows)
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>A_action_interface Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f4; color: #1f2328; }}
    h1 {{ font-size: 24px; margin-bottom: 6px; }}
    .summary {{ margin-bottom: 20px; color: #555; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ background: #fff; border: 1px solid #d8d8d2; border-radius: 6px; padding: 12px; }}
    .media {{ position: relative; width: 100%; background: #eee; overflow: hidden; border-radius: 4px; }}
    .media img, .media svg {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }}
    .meta {{ font-size: 12px; line-height: 1.45; margin-top: 10px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; word-break: break-all; }}
    .ok {{ color: #146c43; }}
    .warn {{ color: #9a5b00; }}
  </style>
</head>
<body>
  <h1>A_action_interface Review</h1>
  <div class="summary">Candidates: {len(rows)}. Approve labels in review_decisions.jsonl before promotion.</div>
  <div class="grid">
    {cards}
  </div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def _candidate_card(candidate: Dict[str, Any]) -> str:
    image_path = candidate.get("image_path")
    width = candidate.get("image_width") or 1
    height = candidate.get("image_height") or 1
    data_url = _image_data_url(image_path)
    overlay = _svg_overlay(candidate, float(width), float(height))
    aspect = f"{float(width)} / {float(height)}" if width and height else "4 / 3"
    blockers = candidate.get("promotion_blockers") or []
    blocker_class = "ok" if not blockers else "warn"
    return f"""
<div class="card">
  <div class="media" style="aspect-ratio: {aspect};">
    {_img_tag(data_url)}
    {overlay}
  </div>
  <div class="meta">
    <div><b>{_e(candidate.get("action_type"))}</b> / {_e(candidate.get("subtype"))}</div>
    <div class="mono">label_id={_e(candidate.get("label_id"))}</div>
    <div>object={_e(candidate.get("object_name"))} / {_e(candidate.get("object_id"))}</div>
    <div>source={_e(candidate.get("source_provenance"))}, verifier={_e(candidate.get("verifier_status"))}</div>
    <div>status={_e(candidate.get("label_status"))}, stable={_e(candidate.get("stable_training_label"))}</div>
    <div class="{blocker_class}">blockers={_e(blockers)}</div>
  </div>
</div>
"""


def _svg_overlay(candidate: Dict[str, Any], width: float, height: float) -> str:
    parts = [
        f'<svg viewBox="0 0 {width:g} {height:g}" preserveAspectRatio="none">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#0f62fe"/></marker></defs>',
    ]
    bbox = candidate.get("bbox")
    if _num_list(bbox, 4):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        parts.append(
            f'<rect x="{x1:g}" y="{y1:g}" width="{x2 - x1:g}" height="{y2 - y1:g}" '
            'fill="none" stroke="#d22" stroke-width="2"/>'
        )
    if _num_list(candidate.get("grasp_point_2d"), 2):
        _point(parts, candidate["grasp_point_2d"], "#198754")
        _vector(parts, candidate["grasp_point_2d"], candidate.get("jaw_axis_2d"), "#198754")
    if _num_list(candidate.get("push_contact_point_2d"), 2):
        _point(parts, candidate["push_contact_point_2d"], "#0f62fe")
        _vector(parts, candidate["push_contact_point_2d"], candidate.get("push_direction_2d"), "#0f62fe")
    region = candidate.get("place_region_2d")
    if isinstance(region, dict) and isinstance(region.get("points"), list):
        pts = " ".join(f"{float(p[0]):g},{float(p[1]):g}" for p in region["points"] if _num_list(p, 2))
        if pts:
            parts.append(f'<polygon points="{pts}" fill="rgba(255,193,7,.2)" stroke="#9a5b00" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _point(parts: List[str], point: Any, color: str) -> None:
    x, y = [float(v) for v in point]
    parts.append(f'<circle cx="{x:g}" cy="{y:g}" r="5" fill="{color}" stroke="#fff" stroke-width="1.5"/>')


def _vector(parts: List[str], point: Any, vector: Any, color: str) -> None:
    if not (_num_list(point, 2) and _num_list(vector, 2)):
        return
    x, y = [float(v) for v in point]
    dx, dy = [float(v) for v in vector]
    scale = 34.0
    parts.append(
        f'<line x1="{x:g}" y1="{y:g}" x2="{x + dx * scale:g}" y2="{y + dy * scale:g}" '
        f'stroke="{color}" stroke-width="3" marker-end="url(#arrow)"/>'
    )


def _image_data_url(image_path: Any) -> Optional[str]:
    if not image_path:
        return None
    try:
        img = Image.open(str(image_path)).convert("RGB")
        img.thumbnail((720, 720))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _img_tag(data_url: Optional[str]) -> str:
    if not data_url:
        return '<div class="mono" style="padding:12px;">image unavailable</div>'
    return f'<img src="{data_url}" alt="review image">'


def _num_list(value: Any, length: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == length and all(
        isinstance(item, (int, float)) for item in value
    )


def _e(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value))
