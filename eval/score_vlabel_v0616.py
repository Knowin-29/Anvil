#!/usr/bin/env python3
"""Rule-first v0616 VLabel scorer with frozen manifest support."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from vlabel_ir_v0616 import (
    build_typed_ir,
    canonical_relation,
    extract_choice,
    extract_count,
    greedy_match_f1,
    iou,
    manifest_entry,
    norm_text,
    parse_boxes_points,
    point_distance_ok,
    point_in_box,
    stable_hash,
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pred_text(row: Dict[str, Any]) -> str:
    return str(row.get("response") or row.get("model_answer") or "")


def prompt_from_output(row: Dict[str, Any]) -> str:
    for m in row.get("messages") or []:
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def build_manifest_from_gt(gts: List[Dict[str, Any]], outs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    manifest = []
    for i, gt in enumerate(gts):
        prompt = ""
        if outs and i < len(outs):
            prompt = prompt_from_output(outs[i])
        entry = manifest_entry(
            sample_id=gt.get("id") or gt.get("req_id") or f"row_{i}",
            source="derived_from_gt",
            task_family_raw=gt.get("task_family") or gt.get("task_family_raw") or "unknown",
            prompt=prompt,
            gt_answer=gt.get("model_answer", ""),
            capabilities=gt.get("capabilities") or [],
            original_index=gt.get("original_index", i),
        )
        if gt.get("gt_bbox") is not None:
            boxes, points, frame = parse_boxes_points(json.dumps({"bbox": gt.get("gt_bbox")}))
            for box in boxes:
                if box not in entry["gt_ir"]["boxes"]:
                    entry["gt_ir"]["boxes"].append(box)
            entry["gt_ir"]["coord_frame"] = entry["gt_ir"].get("coord_frame") or frame
            entry["gt_ir"]["components"] = sorted(set(entry["gt_ir"].get("components", []) + ["bbox"]))
            entry["coord_frame"] = entry["gt_ir"]["coord_frame"]
            entry["answer_type"] = "bbox_region"
            entry["scoring_mode"] = "max_iou_any"
            entry["evaluable"] = bool(entry["gt_ir"]["boxes"])
            entry["not_evaluable_reason"] = None if entry["gt_ir"]["boxes"] else "gt_no_bbox"
        manifest.append(entry)
    return manifest


def score_attribute_state(pred: str, gt_ir: Dict[str, Any]) -> tuple[Optional[bool], str, Dict[str, Any]]:
    gj = gt_ir.get("json")
    pj = build_typed_ir("", pred).get("json")
    if isinstance(gj, dict) and isinstance(pj, dict):
        common = set(gj) & set(pj)
        if not common:
            return False, "no_key_overlap", {"matched": 0, "total_keys": len(gj)}
        matched = 0
        for k in common:
            if norm_text(gj.get(k)) == norm_text(pj.get(k)):
                matched += 1
        key_acc = matched / max(1, len(common))
        return key_acc >= 0.5, f"key_match_{matched}/{len(common)}", {"matched": matched, "common": len(common)}
    gt_norm = norm_text(gt_ir.get("text", ""))
    pred_norm = norm_text(pred)
    if gt_norm and len(gt_norm) <= 80:
        return (gt_norm in pred_norm or pred_norm in gt_norm), "short_text_contains", {}
    return None, "needs_judge_semantic_attribute", {}


def score_one(pred: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    at = manifest["answer_type"]
    gt_ir = manifest["gt_ir"]
    if not manifest.get("evaluable", False):
        return {"scored": False, "correct": None, "reason": manifest.get("not_evaluable_reason"), "pending_judge": False}

    pred_ir = build_typed_ir("", pred, at)
    if at == "bbox_region":
        gt_boxes = gt_ir.get("boxes") or []
        pred_boxes = pred_ir.get("boxes") or []
        if not pred_boxes:
            return {"scored": True, "correct": False, "reason": "pred_no_bbox", "pending_judge": False}
        if len(gt_boxes) <= 1 or manifest.get("scoring_mode") == "max_iou_any":
            best = max(iou(pb, gb) for pb in pred_boxes for gb in gt_boxes) if gt_boxes else 0.0
            return {"scored": True, "correct": best >= 0.5, "reason": f"iou={best:.3f}", "pending_judge": False, "metrics": {"best_iou": best}}
        f1, tp, fp, fn, best = greedy_match_f1(pred_boxes, gt_boxes, 0.5)
        return {
            "scored": True,
            "correct": f1 >= 0.5,
            "reason": f"multi_f1={f1:.3f}",
            "pending_judge": False,
            "metrics": {"f1": f1, "tp": tp, "fp": fp, "fn": fn, "best_iou": best},
        }

    if at == "point_affordance":
        pred_points = pred_ir.get("points") or []
        pred_boxes = pred_ir.get("boxes") or []
        gt_points = gt_ir.get("points") or []
        gt_boxes = gt_ir.get("boxes") or []
        if pred_points and gt_boxes:
            ok = any(point_in_box(pp, gb) for pp in pred_points for gb in gt_boxes)
            return {"scored": True, "correct": ok, "reason": "point_in_gt_region", "pending_judge": False}
        if pred_points and gt_points:
            ok = any(point_distance_ok(pp, gp) for pp in pred_points for gp in gt_points)
            return {"scored": True, "correct": ok, "reason": "point_pck", "pending_judge": False}
        if pred_boxes and gt_boxes:
            best = max(iou(pb, gb) for pb in pred_boxes for gb in gt_boxes)
            return {"scored": True, "correct": best >= 0.3, "reason": f"region_iou={best:.3f}", "pending_judge": False}
        return {"scored": True, "correct": False, "reason": "pred_no_point_or_region", "pending_judge": False}

    if at == "choice":
        gt_choice = gt_ir.get("choice")
        pc = extract_choice(pred)
        if not pc:
            return {"scored": True, "correct": False, "reason": "pred_choice_unparseable", "pending_judge": False}
        return {"scored": True, "correct": pc == gt_choice, "reason": f"choice:{gt_choice}=={pc}", "pending_judge": False}

    if at == "count":
        gt_count = gt_ir.get("count")
        pc = extract_count(pred)
        if pc is None:
            return {"scored": True, "correct": False, "reason": "pred_count_unparseable", "pending_judge": False}
        return {"scored": True, "correct": pc == gt_count, "reason": f"count:{gt_count}=={pc}", "pending_judge": False}

    if at == "relation":
        gt_rel = gt_ir.get("relation")
        pr = canonical_relation(pred)
        if not pr:
            return {"scored": True, "correct": False, "reason": "pred_relation_unparseable", "pending_judge": False}
        return {"scored": True, "correct": pr == gt_rel, "reason": f"relation:{gt_rel}=={pr}", "pending_judge": False}

    if at == "attribute_state":
        correct, reason, metrics = score_attribute_state(pred, gt_ir)
        if correct is None:
            return {"scored": False, "correct": None, "reason": reason, "pending_judge": True, "metrics": metrics}
        return {"scored": True, "correct": correct, "reason": reason, "pending_judge": False, "metrics": metrics}

    if at == "action_free_text":
        return {"scored": False, "correct": None, "reason": "needs_judge_action_free_text", "pending_judge": True}

    return {"scored": False, "correct": None, "reason": "unknown_answer_type", "pending_judge": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_jsonl")
    ap.add_argument("gt_jsonl")
    ap.add_argument("tag")
    ap.add_argument("--manifest")
    ap.add_argument("--out_dir", default="${GSA_FORGE_ROOT}/eval")
    args = ap.parse_args()

    outs = read_jsonl(args.output_jsonl)
    gts = read_jsonl(args.gt_jsonl)
    if args.manifest:
        manifest = read_jsonl(args.manifest)
    else:
        manifest = build_manifest_from_gt(gts, outs)
    n = min(len(outs), len(gts), len(manifest))

    by_type = defaultdict(lambda: Counter())
    by_family = defaultdict(lambda: Counter())
    reasons = defaultdict(Counter)
    pending = []
    scored = correct = not_evaluable = pending_judge = 0
    row_results = []

    for i in range(n):
        m = manifest[i]
        at = m.get("answer_type", "unknown")
        fam = m.get("task_family_raw") or gts[i].get("task_family", "unknown")
        by_type[at]["total"] += 1
        by_family[fam]["total"] += 1
        if not m.get("evaluable", False):
            not_evaluable += 1
            by_type[at]["not_evaluable"] += 1
            by_family[fam]["not_evaluable"] += 1
            reasons[at][m.get("not_evaluable_reason") or "not_evaluable"] += 1
            continue
        by_type[at]["evaluable"] += 1
        by_family[fam]["evaluable"] += 1
        r = score_one(pred_text(outs[i]), m)
        reasons[at][r["reason"]] += 1
        row_results.append({"index": i, "sample_id": m.get("sample_id"), "answer_type": at, **r})
        if r.get("pending_judge"):
            pending_judge += 1
            by_type[at]["pending_judge"] += 1
            by_family[fam]["pending_judge"] += 1
            pending.append({
                "index": i,
                "sample_id": m.get("sample_id"),
                "answer_type": at,
                "task_family_raw": fam,
                "prompt": prompt_from_output(outs[i])[:2000],
                "gt": gts[i].get("model_answer", "")[:2000],
                "pred": pred_text(outs[i])[:2000],
                "judge_verdict": None,
                "judge_confidence": None,
                "audit_reason": r["reason"],
                "suggested_gt_fix": None,
            })
        if r.get("scored"):
            scored += 1
            by_type[at]["scored"] += 1
            by_family[fam]["scored"] += 1
            if r.get("correct") is True:
                correct += 1
                by_type[at]["correct"] += 1
                by_family[fam]["correct"] += 1

    def summarize(counter: Counter) -> Dict[str, Any]:
        d = dict(counter)
        d["acc"] = d.get("correct", 0) / d.get("scored", 1) if d.get("scored", 0) else None
        d["evaluable_rate"] = d.get("evaluable", 0) / d.get("total", 1) if d.get("total", 0) else 0.0
        return d

    report = {
        "checkpoint": args.tag,
        "output_jsonl": args.output_jsonl,
        "gt_jsonl": args.gt_jsonl,
        "manifest": args.manifest or "derived_from_gt",
        "total_rows": n,
        "output_rows": len(outs),
        "gt_rows": len(gts),
        "manifest_rows": len(manifest),
        "complete": len(outs) == len(gts) == len(manifest),
        "scored": scored,
        "correct": correct,
        "not_evaluable": not_evaluable,
        "pending_judge": pending_judge,
        "overall_rule_acc": correct / scored if scored else None,
        "by_answer_type": {k: summarize(v) for k, v in sorted(by_type.items())},
        "by_task_family": {k: summarize(v) for k, v in sorted(by_family.items())},
        "reasons": {k: dict(v.most_common(50)) for k, v in reasons.items()},
        "row_result_hash": stable_hash(row_results),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"vlabel_score_v0616_{args.tag}.json"
    pending_path = out_dir / f"vlabel_pending_judge_v0616_{args.tag}.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(pending_path, "w", encoding="utf-8") as f:
        for row in pending:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"=== {args.tag} v0616 rule-first scoring ===")
    print(f"Rows: {n} complete={report['complete']}")
    print(f"Scored: {scored} Correct: {correct} Acc: {(report['overall_rule_acc'] or 0)*100:.1f}%")
    print(f"Not evaluable: {not_evaluable} Pending judge: {pending_judge}")
    for k, v in report["by_answer_type"].items():
        acc = v["acc"]
        acc_s = "NA" if acc is None else f"{acc*100:.1f}%"
        print(f"  {k}: {acc_s} scored={v.get('scored',0)} eval={v.get('evaluable',0)}/{v.get('total',0)} pending={v.get('pending_judge',0)}")
    print(f"-> {report_path}")
    print(f"-> {pending_path}")


if __name__ == "__main__":
    main()
