#!/usr/bin/env python3
"""Hybrid rule scorer for the 10k database-log MS-Swift inference outputs."""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'`.,;:")
    return text


def extract_json(value: Any) -> Optional[Any]:
    text = clean_text(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def parse_bbox(value: Any) -> Optional[List[float]]:
    obj = extract_json(value)
    if isinstance(obj, dict):
        for key in ("bbox", "bbox_2d", "box"):
            box = obj.get(key)
            if isinstance(box, list) and len(box) >= 4:
                return normalize_box([float(x) for x in box[:4]])
    text = clean_text(value)
    patterns = [
        r'"(?:bbox|bbox_2d|box)"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        r"<box>\s*\[\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*,\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\]\s*</box>",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_box([float(match.group(i)) for i in range(1, 5)])
    return None


def parse_point(value: Any) -> Optional[Tuple[float, float]]:
    obj = extract_json(value)
    if isinstance(obj, dict):
        for key in ("point", "points"):
            point = obj.get(key)
            if isinstance(point, list) and point and isinstance(point[0], list):
                point = point[0]
            if isinstance(point, list) and len(point) >= 2:
                return normalize_point((float(point[0]), float(point[1])))
    text = clean_text(value)
    patterns = [
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
        r"\[\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*\]",
        r'"point"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_point((float(match.group(1)), float(match.group(2))))
    return None


def normalize_box(box: List[float]) -> List[float]:
    vals = [float(x) for x in box[:4]]
    if max(abs(x) for x in vals) <= 1.5:
        vals = [x * 1000.0 for x in vals]
    x1, y1, x2, y2 = vals
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def normalize_point(point: Tuple[float, float]) -> Tuple[float, float]:
    x, y = point
    if max(abs(x), abs(y)) <= 1.5:
        return x * 1000.0, y * 1000.0
    return x, y


def iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_box(point: Tuple[float, float], box: List[float], tol: float = 50.0) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - tol) <= x <= (x2 + tol) and (y1 - tol) <= y <= (y2 + tol)


def parse_count(value: Any) -> Optional[int]:
    match = re.search(r"-?\d+", clean_text(value))
    return int(match.group(0)) if match else None


def parse_choice(value: Any) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None
    match = re.match(r"^(?:answer|choice|option)?\s*[:：]?\s*([a-z])(?:[.)\s:：]|$)", text)
    if match and match.group(1) in list("abcdef"):
        return match.group(1)
    yes_no = {
        "yes": "yes",
        "no": "no",
        "true": "yes",
        "false": "no",
        "correct": "yes",
        "incorrect": "no",
    }
    for key, val in yes_no.items():
        if text == key or text.startswith(key + " "):
            return val
    return text if len(text) <= 64 else None


def comma_items(value: Any) -> List[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value).split(",")
    items = []
    for item in raw_items:
        norm = normalize_text(item)
        if norm:
            items.append(norm)
    return items


def f1_sets(pred_items: List[str], ref_items: List[str]) -> float:
    pred = set(pred_items)
    ref = set(ref_items)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = len(pred & ref)
    precision = tp / len(pred)
    recall = tp / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def infer_answer_type(row: Dict[str, Any], ref: str) -> str:
    prompt = str(row.get("prompt") or row.get("messages", [{}])[0].get("content", ""))
    prompt_l = prompt.lower()
    ref_l = clean_text(ref).lower()
    tags = row.get("tags") if isinstance(row.get("tags"), dict) else {}
    qc = tags.get("qc") if isinstance(tags.get("qc"), dict) else {}
    answer_format = str(qc.get("answer_format") or "").lower()
    task_family = str(qc.get("task_family") or "").lower()

    if "bbox" in answer_format or "bbox" in task_family or '"bbox"' in ref_l or "bounding box" in prompt_l:
        return "bbox"
    if "point" in answer_format or "point" in task_family or "list of tuples" in prompt_l:
        return "point"
    if "count" in answer_format or "how many" in prompt_l or "single non-negative integer" in prompt_l:
        return "count"
    if "choice" in answer_format or re.search(r"\boptions?\s*:\s*", prompt_l):
        return "choice"
    if "json" in answer_format or "output only a json" in prompt_l or isinstance(extract_json(ref), dict):
        return "json_schema"
    if len(normalize_text(ref)) <= 64:
        return "exact_text"
    return "open_text"


def reference_text(row: Dict[str, Any]) -> str:
    return clean_text(row.get("labels") or row.get("model_answer") or row.get("label") or "")


def ability_keys(row: Dict[str, Any]) -> List[str]:
    tags = row.get("tags") if isinstance(row.get("tags"), dict) else {}
    qc = tags.get("qc") if isinstance(tags.get("qc"), dict) else {}
    keys = qc.get("capabilities") if isinstance(qc.get("capabilities"), list) else None
    if not keys:
        keys = tags.get("ability_tags") if isinstance(tags.get("ability_tags"), list) else None
    filtered = [str(x) for x in (keys or []) if re.match(r"^(?:G|A|S|R)\.\d+$", str(x))]
    return filtered or ["unknown"]


def score_row(row: Dict[str, Any]) -> Dict[str, Any]:
    response = clean_text(row.get("response", ""))
    ref = reference_text(row)
    answer_type = infer_answer_type(row, ref)
    result: Dict[str, Any] = {
        "id": row.get("id") or row.get("req_id") or row.get("__index__"),
        "ok": bool(row.get("__ok__", True)),
        "answer_type": answer_type,
        "abilities": ability_keys(row),
        "scored": False,
        "correct": False,
        "score": None,
        "needs_judge": False,
        "reason": "",
    }
    if not result["ok"]:
        result["reason"] = row.get("error_type") or "inference_failed"
        return result
    if not response:
        result["reason"] = "empty_response"
        return result
    if not ref:
        result["reason"] = "missing_reference"
        return result

    if answer_type == "bbox":
        pred_box = parse_bbox(response)
        ref_box = parse_bbox(ref)
        if pred_box and ref_box:
            val = iou(pred_box, ref_box)
            result.update(scored=True, correct=val >= 0.5, score=val, metric="iou")
        else:
            result.update(reason="bbox_parse_failed", needs_judge=True)
        return result

    if answer_type == "point":
        pred_point = parse_point(response)
        ref_point = parse_point(ref)
        ref_box = parse_bbox(ref)
        if pred_point and ref_box:
            correct = point_in_box(pred_point, ref_box)
            result.update(scored=True, correct=correct, score=1.0 if correct else 0.0, metric="point_in_ref_box")
        elif pred_point and ref_point:
            dist = math.dist(pred_point, ref_point) / 1414.21356237
            result.update(scored=True, correct=dist <= 0.05, score=max(0.0, 1.0 - dist), metric="pck_0_05")
        else:
            result.update(reason="point_parse_failed", needs_judge=True)
        return result

    if answer_type == "count":
        pred_count = parse_count(response)
        ref_count = parse_count(ref)
        if pred_count is not None and ref_count is not None:
            result.update(scored=True, correct=pred_count == ref_count, score=1.0 if pred_count == ref_count else 0.0, metric="exact_count")
        else:
            result.update(reason="count_parse_failed", needs_judge=True)
        return result

    if answer_type == "choice":
        pred_choice = parse_choice(response)
        ref_choice = parse_choice(ref)
        if pred_choice and ref_choice:
            correct = pred_choice == ref_choice or pred_choice in ref_choice or ref_choice in pred_choice
            result.update(scored=True, correct=correct, score=1.0 if correct else 0.0, metric="choice_match")
        else:
            result.update(reason="choice_parse_failed", needs_judge=True)
        return result

    if answer_type == "json_schema":
        pred_json = extract_json(response)
        ref_json = extract_json(ref)
        if isinstance(pred_json, dict) and isinstance(ref_json, dict):
            key_scores = []
            for key, ref_val in ref_json.items():
                pred_val = pred_json.get(key)
                if isinstance(ref_val, str) and "," in ref_val:
                    key_scores.append(f1_sets(comma_items(pred_val or ""), comma_items(ref_val)))
                else:
                    key_scores.append(1.0 if normalize_text(pred_val) == normalize_text(ref_val) else 0.0)
            avg = sum(key_scores) / len(key_scores) if key_scores else 0.0
            result.update(scored=True, correct=avg >= 0.8, score=avg, metric="json_key_macro")
        elif isinstance(ref_json, dict):
            result.update(scored=True, correct=False, score=0.0, metric="json_validity", reason="pred_json_parse_failed")
        else:
            result.update(reason="ref_json_parse_failed", needs_judge=True)
        return result

    if answer_type == "exact_text":
        pred_norm = normalize_text(response)
        ref_norm = normalize_text(ref)
        correct = pred_norm == ref_norm or pred_norm in ref_norm or ref_norm in pred_norm
        result.update(scored=True, correct=correct, score=1.0 if correct else 0.0, metric="short_text_match")
        return result

    result.update(needs_judge=True, reason="open_text_not_rule_scored")
    return result


def pct(num: int, den: int) -> float:
    return num / den if den else 0.0


def summarize(items: List[Dict[str, Any]], tag: str, pred_path: Path) -> Dict[str, Any]:
    by_type: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"rows": 0, "scored": 0, "correct": 0, "scores": []})
    by_ability: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"rows": 0, "scored": 0, "correct": 0, "scores": []})
    reasons = Counter()
    for item in items:
        buckets = [by_type[item["answer_type"]]]
        for ability in item["abilities"]:
            buckets.append(by_ability[ability])
        for bucket in buckets:
            bucket["rows"] += 1
            if item["scored"]:
                bucket["scored"] += 1
                bucket["correct"] += int(bool(item["correct"]))
                if isinstance(item.get("score"), (int, float)):
                    bucket["scores"].append(float(item["score"]))
        if item.get("reason"):
            reasons[item["reason"]] += 1

    def finish(bucket: Dict[str, Any]) -> Dict[str, Any]:
        scores = bucket.pop("scores")
        bucket["accuracy"] = pct(bucket["correct"], bucket["scored"])
        bucket["coverage"] = pct(bucket["scored"], bucket["rows"])
        bucket["mean_score"] = statistics.mean(scores) if scores else None
        return bucket

    total_rows = len(items)
    total_scored = sum(1 for item in items if item["scored"])
    total_correct = sum(1 for item in items if item["scored"] and item["correct"])
    failed = sum(1 for item in items if not item["ok"])
    empty = reasons.get("empty_response", 0)
    return {
        "tag": tag,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_file": str(pred_path),
        "rows": total_rows,
        "inference_failed": failed,
        "empty_response": empty,
        "rule_scored": total_scored,
        "needs_judge": sum(1 for item in items if item["needs_judge"]),
        "accuracy_on_rule_scored": pct(total_correct, total_scored),
        "rule_coverage": pct(total_scored, total_rows),
        "by_type": {key: finish(dict(val)) for key, val in sorted(by_type.items())},
        "by_ability": {key: finish(dict(val)) for key, val in sorted(by_ability.items())},
        "reasons": dict(reasons.most_common()),
    }


def write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Logs Hybrid Score Summary",
        "",
        f"- Tag: `{summary['tag']}`",
        f"- Prediction file: `{summary['prediction_file']}`",
        f"- Rows: {summary['rows']}",
        f"- Rule coverage: {summary['rule_coverage']:.4f}",
        f"- Accuracy on rule-scored: {summary['accuracy_on_rule_scored']:.4f}",
        f"- Needs judge: {summary['needs_judge']}",
        "",
        "## By Type",
        "",
        "| Type | Rows | Scored | Coverage | Correct | Accuracy | Mean Score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary["by_type"].items():
        mean = "-" if row["mean_score"] is None else f"{row['mean_score']:.4f}"
        lines.append(
            f"| `{key}` | {row['rows']} | {row['scored']} | {row['coverage']:.4f} | "
            f"{row['correct']} | {row['accuracy']:.4f} | {mean} |"
        )
    lines.extend(["", "## By Ability", "", "| Ability | Rows | Scored | Coverage | Correct | Accuracy |", "|---|---:|---:|---:|---:|---:|"])
    for key, row in summary["by_ability"].items():
        lines.append(f"| `{key}` | {row['rows']} | {row['scored']} | {row['coverage']:.4f} | {row['correct']} | {row['accuracy']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(path: Path, summary: Dict[str, Any]) -> None:
    type_rows = []
    for key, row in summary["by_type"].items():
        mean = "-" if row["mean_score"] is None else f"{row['mean_score']:.4f}"
        type_rows.append(
            f"<tr><td><code>{html.escape(key)}</code></td><td>{row['rows']}</td><td>{row['scored']}</td>"
            f"<td>{row['coverage']:.4f}</td><td>{row['correct']}</td><td>{row['accuracy']:.4f}</td><td>{mean}</td></tr>"
        )
    ability_rows = []
    for key, row in summary["by_ability"].items():
        ability_rows.append(
            f"<tr><td><code>{html.escape(key)}</code></td><td>{row['rows']}</td><td>{row['scored']}</td>"
            f"<td>{row['coverage']:.4f}</td><td>{row['correct']}</td><td>{row['accuracy']:.4f}</td></tr>"
        )
    doc = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Logs Hybrid Score Summary</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;margin:24px;color:#111827;line-height:1.5}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}
th,td{{border:1px solid #d1d5db;padding:7px;text-align:left}}
th{{background:#f3f4f6}}
code{{background:#f3f4f6;padding:1px 4px;border-radius:3px}}
</style>
</head>
<body>
<h1>Logs Hybrid Score Summary</h1>
<p>Tag: <code>{html.escape(summary['tag'])}</code></p>
<p>Rows: {summary['rows']} | Rule coverage: {summary['rule_coverage']:.4f} | Accuracy on rule-scored: {summary['accuracy_on_rule_scored']:.4f} | Needs judge: {summary['needs_judge']}</p>
<h2>By Type</h2>
<table><thead><tr><th>Type</th><th>Rows</th><th>Scored</th><th>Coverage</th><th>Correct</th><th>Accuracy</th><th>Mean Score</th></tr></thead><tbody>{''.join(type_rows)}</tbody></table>
<h2>By Ability</h2>
<table><thead><tr><th>Ability</th><th>Rows</th><th>Scored</th><th>Coverage</th><th>Correct</th><th>Accuracy</th></tr></thead><tbody>{''.join(ability_rows)}</tbody></table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="logs_eval")
    parser.add_argument("--items-jsonl", default=None)
    args = parser.parse_args()

    pred_path = Path(args.pred)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [score_row(row) for row in iter_jsonl(pred_path)]
    summary = summarize(items, args.tag, pred_path)
    dump_json(out_dir / "summary.json", summary)
    write_markdown(out_dir / "summary.md", summary)
    write_html(out_dir / "summary.html", summary)
    item_path = Path(args.items_jsonl) if args.items_jsonl else out_dir / "items.jsonl"
    with item_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
