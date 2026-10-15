"""Acceptance review and verifier evaluation for A_action_interface."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .a_action_interface_contract import (
    REVIEW_DECISION_APPROVE,
    REVIEW_DECISION_PENDING,
    REVIEW_DECISION_REJECT,
    REVIEW_DECISION_UNCERTAIN,
    REVIEW_DECISIONS,
)
from .a_action_interface_review import (
    load_review_decisions,
    make_review_decision_template,
    normalize_action_interface_candidates,
    promote_action_interface_candidates,
    read_jsonl,
    write_jsonl,
    write_review_html,
)
from .a_action_interface_triage import triage_candidates


HUMAN_REVIEW_DECISIONS = {
    REVIEW_DECISION_APPROVE,
    REVIEW_DECISION_REJECT,
    REVIEW_DECISION_UNCERTAIN,
}


def prepare_acceptance_review_set(
    *,
    candidates_path: Path,
    output_dir: Path,
    sample_count: int = 100,
    max_per_object_name: int = 24,
    max_per_image: int = 6,
) -> Dict[str, Any]:
    raw_candidates = read_jsonl(candidates_path)
    selected, triage_summary = triage_candidates(
        raw_candidates,
        max_candidates=sample_count,
        max_per_object_name=max_per_object_name,
        max_per_image=max_per_image,
        include_unknown_objects=False,
    )
    selected = selected[:sample_count]

    output_dir.mkdir(parents=True, exist_ok=True)
    review_candidates_path = output_dir / "acceptance_review_candidates.jsonl"
    decisions_template_path = output_dir / "review_decisions.template.jsonl"
    active_decisions_path = output_dir / "review_decisions.jsonl"
    html_path = output_dir / "acceptance_review.html"
    summary_path = output_dir / "acceptance_review_setup_summary.json"
    markdown_path = output_dir / "acceptance_review_setup_summary.md"

    template = make_acceptance_review_decision_template(selected)
    write_jsonl(review_candidates_path, selected)
    write_jsonl(decisions_template_path, template)
    if not active_decisions_path.exists():
        write_jsonl(active_decisions_path, template)
    write_review_html(html_path, selected)

    summary = {
        "input_candidates_path": str(candidates_path),
        "review_candidates_path": str(review_candidates_path),
        "review_decisions_template_path": str(decisions_template_path),
        "active_review_decisions_path": str(active_decisions_path),
        "review_html_path": str(html_path),
        "sample_count_requested": sample_count,
        "selected_count": len(selected),
        "triage_summary": triage_summary,
        "human_review_summary": summarize_human_review(selected, {}),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_acceptance_setup_markdown(summary), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["markdown_report_path"] = str(markdown_path)
    return summary


def make_acceptance_review_decision_template(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = make_review_decision_template(candidates)
    for row in rows:
        row["allowed_decisions"] = [
            REVIEW_DECISION_APPROVE,
            REVIEW_DECISION_REJECT,
            REVIEW_DECISION_UNCERTAIN,
            REVIEW_DECISION_PENDING,
        ]
        row.setdefault("reject_reason", "")
    return rows


def summarize_human_review(
    candidates: Iterable[Dict[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = normalize_action_interface_candidates(candidates)
    overall = _empty_review_stats()
    by_action: Dict[str, Dict[str, Any]] = defaultdict(_empty_review_stats)
    by_object: Dict[str, Dict[str, Any]] = defaultdict(_empty_review_stats)
    by_source: Dict[str, Dict[str, Any]] = defaultdict(_empty_review_stats)
    pending_label_ids: List[str] = []

    for candidate in rows:
        label_id = str(candidate.get("label_id"))
        decision_row = decisions.get(label_id, {})
        decision = normalize_review_decision(decision_row.get("decision", REVIEW_DECISION_PENDING))
        candidate_key = {
            "action_type": str(candidate.get("action_type", "unknown")),
            "object_name": str(candidate.get("object_name", "unknown")),
            "source_provenance": str(candidate.get("source_provenance", "unknown")),
        }
        _update_review_stats(overall, decision)
        _update_review_stats(by_action[candidate_key["action_type"]], decision)
        _update_review_stats(by_object[candidate_key["object_name"]], decision)
        _update_review_stats(by_source[candidate_key["source_provenance"]], decision)
        if decision == REVIEW_DECISION_PENDING:
            pending_label_ids.append(label_id)

    _finalize_review_stats(overall)
    for group in (by_action, by_object, by_source):
        for stats in group.values():
            _finalize_review_stats(stats)

    return {
        "overall": overall,
        "by_action_type": dict(sorted(by_action.items())),
        "by_object_name": dict(sorted(by_object.items())),
        "by_source_provenance": dict(sorted(by_source.items())),
        "pending_label_ids": pending_label_ids,
    }


def build_verifier_eval_set(
    candidates: Iterable[Dict[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows = normalize_action_interface_candidates(candidates)
    eval_rows: List[Dict[str, Any]] = []
    for candidate in rows:
        label_id = str(candidate.get("label_id"))
        decision_row = decisions.get(label_id, {})
        decision = normalize_review_decision(decision_row.get("decision", REVIEW_DECISION_PENDING))
        if decision not in HUMAN_REVIEW_DECISIONS:
            continue
        row = dict(candidate)
        row["human_decision"] = decision
        row["human_review"] = dict(decision_row)
        eval_rows.append(row)
    return eval_rows


class AActionInterfaceVLMJudge:
    """VLM judge for one reviewed A_action_interface candidate."""

    def __init__(self, vlm_client: Any):
        self.vlm_client = vlm_client

    def judge(self, candidate: Mapping[str, Any]) -> Dict[str, Any]:
        prompt = build_vlm_judge_prompt(candidate)
        image_path = candidate.get("image_path")
        if hasattr(self.vlm_client, "chat"):
            raw = self.vlm_client.chat(
                prompt=prompt,
                image_path=str(image_path) if image_path else None,
                temperature=0.0,
                max_tokens=500,
                system_prompt="Return compact valid JSON only.",
            )
        else:
            raw = self.vlm_client.query(prompt)
        parsed = parse_vlm_judge_response(str(raw))
        return {
            "label_id": candidate.get("label_id"),
            "verifier_decision": parsed["decision"],
            "verifier_reason": parsed["reason"],
        }


def build_vlm_judge_prompt(candidate: Mapping[str, Any]) -> str:
    geometry = {
        "bbox": candidate.get("bbox"),
        "action_type": candidate.get("action_type"),
        "grasp_point_2d": candidate.get("grasp_point_2d"),
        "jaw_axis_2d": candidate.get("jaw_axis_2d"),
        "push_contact_point_2d": candidate.get("push_contact_point_2d"),
        "push_direction_2d": candidate.get("push_direction_2d"),
        "object_name": candidate.get("object_name"),
        "image_width": candidate.get("image_width"),
        "image_height": candidate.get("image_height"),
    }
    return (
        "Judge one robot action-interface label on the image.\n"
        "Use the overlaid/implicit image coordinate data below.\n"
        "Decision meanings:\n"
        "- approve: the action point and direction are visually plausible for the object.\n"
        "- reject: the point/direction is clearly wrong, on background, or unsuitable.\n"
        "- uncertain: the image/object is ambiguous or cannot be judged reliably.\n"
        "Return JSON only with keys decision and reason. No markdown.\n"
        f"LABEL_JSON: {json.dumps(geometry, ensure_ascii=False)}"
    )


def parse_vlm_judge_response(text: str) -> Dict[str, str]:
    parsed: Any = {}
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        parsed = _extract_first_json_object(text)
    if not isinstance(parsed, dict):
        return {"decision": REVIEW_DECISION_UNCERTAIN, "reason": "judge_parse_failed"}
    decision = normalize_review_decision(parsed.get("decision", REVIEW_DECISION_UNCERTAIN))
    if decision == REVIEW_DECISION_PENDING:
        decision = REVIEW_DECISION_UNCERTAIN
    reason = str(parsed.get("reason") or parsed.get("notes") or "").strip()
    return {"decision": decision, "reason": reason or "no_reason"}


def run_vlm_judge(
    eval_set: Iterable[Dict[str, Any]],
    judge: AActionInterfaceVLMJudge,
    *,
    max_cases: Optional[int] = None,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(eval_set):
        if max_cases is not None and idx >= max_cases:
            break
        predictions.append(judge.judge(candidate))
    return predictions


def evaluate_verifier_predictions(
    eval_set: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    eval_rows = list(eval_set)
    prediction_by_id = {str(row.get("label_id")): row for row in predictions}
    confusion = {
        human: {pred: 0 for pred in sorted(HUMAN_REVIEW_DECISIONS)}
        for human in sorted(HUMAN_REVIEW_DECISIONS)
    }
    error_breakdown: Counter[str] = Counter()
    false_positive: List[Dict[str, Any]] = []
    false_negative: List[Dict[str, Any]] = []
    uncertain_mismatch: List[Dict[str, Any]] = []
    missing_predictions: List[str] = []
    compared_count = 0
    correct_count = 0

    for row in eval_rows:
        label_id = str(row.get("label_id"))
        human = normalize_review_decision(row.get("human_decision"))
        prediction = prediction_by_id.get(label_id)
        if not prediction:
            missing_predictions.append(label_id)
            continue
        predicted = normalize_review_decision(prediction.get("verifier_decision"))
        if predicted == REVIEW_DECISION_PENDING:
            predicted = REVIEW_DECISION_UNCERTAIN
        compared_count += 1
        confusion[human][predicted] += 1
        if human == predicted:
            correct_count += 1
            continue

        mismatch = {
            "label_id": label_id,
            "human_decision": human,
            "verifier_decision": predicted,
            "object_name": row.get("object_name"),
            "action_type": row.get("action_type"),
            "verifier_reason": prediction.get("verifier_reason", ""),
        }
        error_breakdown[f"human_{human}_verifier_{predicted}"] += 1
        if predicted == REVIEW_DECISION_APPROVE and human != REVIEW_DECISION_APPROVE:
            false_positive.append(mismatch)
        if human == REVIEW_DECISION_APPROVE and predicted != REVIEW_DECISION_APPROVE:
            false_negative.append(mismatch)
        if human == REVIEW_DECISION_UNCERTAIN or predicted == REVIEW_DECISION_UNCERTAIN:
            uncertain_mismatch.append(mismatch)

    accuracy = correct_count / compared_count if compared_count else None
    return {
        "eval_count": len(eval_rows),
        "prediction_count": len(prediction_by_id),
        "compared_count": compared_count,
        "correct_count": correct_count,
        "verifier_accuracy": accuracy,
        "target_accuracy": 0.8,
        "target_met": accuracy >= 0.8 if accuracy is not None else None,
        "confusion_matrix": confusion,
        "missing_predictions": missing_predictions,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "uncertain_mismatch": uncertain_mismatch,
        "error_breakdown": dict(error_breakdown),
    }


def run_acceptance_evaluation(
    *,
    candidates_path: Path,
    decisions_path: Optional[Path],
    output_dir: Path,
    predictions_path: Optional[Path] = None,
) -> Dict[str, Any]:
    candidates = read_jsonl(candidates_path)
    decisions = load_review_decisions(decisions_path)
    human_summary = summarize_human_review(candidates, decisions)
    eval_set = build_verifier_eval_set(candidates, decisions)

    output_dir.mkdir(parents=True, exist_ok=True)
    human_summary_path = output_dir / "human_review_summary.json"
    human_markdown_path = output_dir / "human_review_summary.md"
    eval_set_path = output_dir / "verifier_eval_set.jsonl"
    reviewed_candidates_path = output_dir / "acceptance_reviewed_candidates.jsonl"
    stable_pack_path = output_dir / "A_action_interface_acceptance_pack_manifest.jsonl"
    promotion_report_path = output_dir / "acceptance_promotion_report.json"

    promotion = promote_action_interface_candidates(candidates, decisions)
    write_jsonl(reviewed_candidates_path, promotion.candidates)
    write_jsonl(stable_pack_path, promotion.stable_labels)
    write_jsonl(eval_set_path, eval_set)
    human_summary_path.write_text(json.dumps(human_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    human_markdown_path.write_text(render_human_review_markdown(human_summary), encoding="utf-8")
    promotion_report_path.write_text(json.dumps(promotion.summary, ensure_ascii=False, indent=2), encoding="utf-8")

    verifier_report = None
    verifier_report_path = None
    verifier_markdown_path = None
    if predictions_path and predictions_path.exists():
        predictions = read_jsonl(predictions_path)
        verifier_report = evaluate_verifier_predictions(eval_set, predictions)
        verifier_report_path = output_dir / "vlm_judge_verifier_eval_report.json"
        verifier_markdown_path = output_dir / "vlm_judge_verifier_eval_report.md"
        verifier_report_path.write_text(json.dumps(verifier_report, ensure_ascii=False, indent=2), encoding="utf-8")
        verifier_markdown_path.write_text(render_verifier_eval_markdown(verifier_report), encoding="utf-8")

    return {
        "candidates_path": str(candidates_path),
        "decisions_path": str(decisions_path) if decisions_path else None,
        "human_review_summary_path": str(human_summary_path),
        "human_review_markdown_path": str(human_markdown_path),
        "verifier_eval_set_path": str(eval_set_path),
        "reviewed_candidates_path": str(reviewed_candidates_path),
        "stable_pack_path": str(stable_pack_path),
        "promotion_report_path": str(promotion_report_path),
        "human_review_summary": human_summary,
        "promotion_summary": promotion.summary,
        "verifier_eval_report_path": str(verifier_report_path) if verifier_report_path else None,
        "verifier_eval_markdown_path": str(verifier_markdown_path) if verifier_markdown_path else None,
        "verifier_eval_report": verifier_report,
    }


def normalize_review_decision(value: Any) -> str:
    decision = str(value or REVIEW_DECISION_PENDING).strip().lower()
    return decision if decision in REVIEW_DECISIONS else REVIEW_DECISION_PENDING


def render_acceptance_setup_markdown(summary: Mapping[str, Any]) -> str:
    human = summary.get("human_review_summary", {}).get("overall", {})
    lines = [
        "# A_action_interface Acceptance Review Setup",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| sample_count_requested | {summary.get('sample_count_requested', 0)} |",
        f"| selected_count | {summary.get('selected_count', 0)} |",
        f"| reviewed_count | {human.get('reviewed_count', 0)} |",
        f"| pass_rate | {_fmt_rate(human.get('pass_rate'))} |",
        "",
        "## Outputs",
        "",
        f"- candidates: `{summary.get('review_candidates_path')}`",
        f"- decisions template: `{summary.get('review_decisions_template_path')}`",
        f"- active decisions: `{summary.get('active_review_decisions_path')}`",
        f"- review HTML: `{summary.get('review_html_path')}`",
        "",
        "Fill `review_decisions.jsonl` before running acceptance evaluation.",
    ]
    return "\n".join(lines) + "\n"


def render_human_review_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# A_action_interface Human Review Summary",
        "",
        "## Overall",
        "",
        _stats_table(summary.get("overall", {})),
        "",
        "## By Action Type",
        "",
        _group_stats_table(summary.get("by_action_type", {})),
        "",
        "## By Source Provenance",
        "",
        _group_stats_table(summary.get("by_source_provenance", {})),
        "",
        "Uncertain decisions are counted in `reviewed_count` and in the pass-rate denominator.",
    ]
    return "\n".join(lines) + "\n"


def render_verifier_eval_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# A_action_interface VLM Judge Verifier Eval",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| eval_count | {report.get('eval_count', 0)} |",
        f"| prediction_count | {report.get('prediction_count', 0)} |",
        f"| compared_count | {report.get('compared_count', 0)} |",
        f"| correct_count | {report.get('correct_count', 0)} |",
        f"| verifier_accuracy | {_fmt_rate(report.get('verifier_accuracy'))} |",
        f"| target_met | {report.get('target_met')} |",
        "",
        "## Confusion Matrix",
        "",
        "Rows are human decisions. Columns are VLM judge decisions.",
        "",
        _confusion_table(report.get("confusion_matrix", {})),
        "",
        "## Error Breakdown",
        "",
        "| Error | Count |",
        "|---|---:|",
    ]
    for key, count in sorted((report.get("error_breakdown") or {}).items()):
        lines.append(f"| `{key}` | {count} |")
    return "\n".join(lines) + "\n"


def _empty_review_stats() -> Dict[str, Any]:
    return {
        "candidate_count": 0,
        "reviewed_count": 0,
        "approve_count": 0,
        "reject_count": 0,
        "uncertain_count": 0,
        "pending_count": 0,
        "pass_rate": None,
        "target_pass_rate": 0.5,
        "target_met": None,
    }


def _update_review_stats(stats: Dict[str, Any], decision: str) -> None:
    stats["candidate_count"] += 1
    if decision == REVIEW_DECISION_APPROVE:
        stats["approve_count"] += 1
        stats["reviewed_count"] += 1
    elif decision == REVIEW_DECISION_REJECT:
        stats["reject_count"] += 1
        stats["reviewed_count"] += 1
    elif decision == REVIEW_DECISION_UNCERTAIN:
        stats["uncertain_count"] += 1
        stats["reviewed_count"] += 1
    else:
        stats["pending_count"] += 1


def _finalize_review_stats(stats: Dict[str, Any]) -> None:
    reviewed_count = int(stats["reviewed_count"])
    if reviewed_count:
        pass_rate = stats["approve_count"] / reviewed_count
        stats["pass_rate"] = pass_rate
        stats["target_met"] = pass_rate >= stats["target_pass_rate"]


def _stats_table(stats: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "| Metric | Value |",
            "|---|---:|",
            f"| candidate_count | {stats.get('candidate_count', 0)} |",
            f"| reviewed_count | {stats.get('reviewed_count', 0)} |",
            f"| approve_count | {stats.get('approve_count', 0)} |",
            f"| reject_count | {stats.get('reject_count', 0)} |",
            f"| uncertain_count | {stats.get('uncertain_count', 0)} |",
            f"| pending_count | {stats.get('pending_count', 0)} |",
            f"| pass_rate | {_fmt_rate(stats.get('pass_rate'))} |",
            f"| target_met | {stats.get('target_met')} |",
        ]
    )


def _group_stats_table(group: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "| Group | Candidates | Reviewed | Approve | Reject | Uncertain | Pending | Pass Rate | Target Met |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, stats in group.items():
        lines.append(
            f"| {name} | {stats.get('candidate_count', 0)} | {stats.get('reviewed_count', 0)} | "
            f"{stats.get('approve_count', 0)} | {stats.get('reject_count', 0)} | "
            f"{stats.get('uncertain_count', 0)} | {stats.get('pending_count', 0)} | "
            f"{_fmt_rate(stats.get('pass_rate'))} | {stats.get('target_met')} |"
        )
    return "\n".join(lines)


def _confusion_table(confusion: Mapping[str, Mapping[str, int]]) -> str:
    labels = sorted(HUMAN_REVIEW_DECISIONS)
    lines = ["| Human \\ Verifier | " + " | ".join(labels) + " |", "|" + "---|" * (len(labels) + 1)]
    for human in labels:
        row = confusion.get(human, {})
        lines.append("| " + human + " | " + " | ".join(str(row.get(pred, 0)) for pred in labels) + " |")
    return "\n".join(lines)


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def _extract_first_json_object(text: str) -> Any:
    depth = 0
    start = -1
    for idx, ch in enumerate(text):
        if ch == "{" and depth == 0:
            start = idx
            depth = 1
        elif ch == "{" and depth > 0:
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    return {}
    return {}
