#!/usr/bin/env python3
"""Score v3: Clean two-layer evaluation.

Layer 1 (Rule-based, reliable):
  - bbox IoU: GT and PRED both parseable as bbox -> IoU >= 0.5
  - Exact match: GT.strip() == PRED.strip() (after removing <think> tags)

Layer 2 (VLM-judge, pending):
  - Anything not scored by Layer 1 -> output to pending_judge.jsonl

Explicitly NOT scored (gt_invalid):
  - GT is <FAILED>, empty, or None

Usage:
    python3 eval/score_vlabel_v3.py <output.jsonl> <gt.jsonl> [tag]
"""
import json, re, sys
from collections import defaultdict, Counter
from pathlib import Path

def parse_bbox(text):
    """Extract [x1,y1,x2,y2] from various formats."""
    text = str(text)
    for key in ['bbox_2d', 'bbox']:
        m = re.search(key + r'"?\s*:?\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
        if m:
            return [float(m.group(i)) for i in range(1, 5)]
    m = re.search(r'<box>\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]</box>', text)
    if m:
        return [float(m.group(i)) for i in range(1, 5)]
    return None

def iou(a, b):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    if not a or not b:
        return 0.0
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    if xa1 > xa2: xa1, xa2 = xa2, xa1
    if ya1 > ya2: ya1, ya2 = ya2, ya1
    if xb1 > xb2: xb1, xb2 = xb2, xb1
    if yb1 > yb2: yb1, yb2 = yb2, yb1
    inter = max(0, min(xa2, xb2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def clean_response(text):
    """Remove <think>...</think> wrapper from model output."""
    text = str(text).strip()
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    return text.strip()

def main():
    out_file = sys.argv[1]
    gt_file = sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else 'ckpt'

    outs = [json.loads(l) for l in open(out_file)]
    gts = [json.loads(l) for l in open(gt_file)]
    n = min(len(outs), len(gts))

    # Results
    results = {
        'rule_bbox': {'correct': 0, 'total': 0, 'ious': []},
        'rule_exact': {'correct': 0, 'total': 0},
        'gt_invalid': 0,
        'needs_judge': 0,
    }
    by_family = defaultdict(lambda: {'bbox_correct': 0, 'bbox_total': 0,
                                      'exact_correct': 0, 'exact_total': 0,
                                      'needs_judge': 0, 'gt_invalid': 0})
    pending_judge = []

    for i in range(n):
        pred_raw = outs[i].get('response', '')
        g = gts[i]
        ga = g.get('model_answer', '')
        tf = g.get('task_family', 'unknown')

        # Clean response (remove <think> tags)
        pred = clean_response(pred_raw)
        gt_clean = clean_response(ga)

        # === GT invalid check ===
        if ga.strip() in ('<FAILED>', '', 'None') or ga.strip().startswith('<FAILED>'):
            results['gt_invalid'] += 1
            by_family[tf]['gt_invalid'] += 1
            continue

        # === Layer 1a: bbox IoU (ONLY for bbox_grounding family) ===
        gt_bbox = g.get('gt_bbox') or parse_bbox(ga)
        pred_bbox = parse_bbox(pred)
        if gt_bbox and pred_bbox and 'bbox_grounding' in tf:
            v = iou(pred_bbox, gt_bbox)
            results['rule_bbox']['total'] += 1
            results['rule_bbox']['ious'].append(v)
            if v >= 0.5:
                results['rule_bbox']['correct'] += 1
            by_family[tf]['bbox_total'] += 1
            by_family[tf]['bbox_correct'] += int(v >= 0.5)
            continue  # scored by bbox, done

        # === Layer 1b: Exact match (case-insensitive, whitespace-normalized) ===
        gt_norm = gt_clean.lower().strip()
        pred_norm = pred.lower().strip()
        if gt_norm and pred_norm and (gt_norm == pred_norm or
                                      (len(gt_norm) < 100 and gt_norm in pred_norm)):
            results['rule_exact']['correct'] += 1
            results['rule_exact']['total'] += 1
            by_family[tf]['exact_correct'] += 1
            by_family[tf]['exact_total'] += 1
            continue  # scored by exact match

        # Check if it's a short GT that doesn't match
        if len(gt_norm) < 100 and gt_norm and pred_norm:
            # Short GT but no match -> rule says incorrect
            results['rule_exact']['total'] += 1
            by_family[tf]['exact_total'] += 1
            continue

        # === Layer 2: Needs VLM judge ===
        results['needs_judge'] += 1
        by_family[tf]['needs_judge'] += 1
        pending_judge.append({
            'index': i,
            'task_family': tf,
            'gt': ga[:500],
            'pred': pred[:500],
            'capabilities': g.get('capabilities', []),
        })

    # Compute summary
    bbox_r = results['rule_bbox']
    exact_r = results['rule_exact']
    rule_total = bbox_r['total'] + exact_r['total']
    rule_correct = bbox_r['correct'] + exact_r['correct']

    # IoU stats
    ious = bbox_r['ious']
    iou_stats = {}
    if ious:
        ious_sorted = sorted(ious)
        iou_stats = {
            'mean': sum(ious) / len(ious),
            'median': ious_sorted[len(ious) // 2],
            'iou@0.5': sum(1 for x in ious if x >= 0.5) / len(ious),
            'iou@0.7': sum(1 for x in ious if x >= 0.7) / len(ious),
            'iou@0.9': sum(1 for x in ious if x >= 0.9) / len(ious),
            'parse_fail': sum(1 for x in ious if x == 0),
        }

    # Output report
    report = {
        'checkpoint': tag,
        'total_samples': n,
        'gt_invalid': results['gt_invalid'],
        'rule_scored': rule_total,
        'rule_correct': rule_correct,
        'rule_acc': rule_correct / rule_total if rule_total else 0,
        'needs_vlm_judge': results['needs_judge'],
        'bbox': {
            'total': bbox_r['total'],
            'correct': bbox_r['correct'],
            'acc': bbox_r['correct'] / bbox_r['total'] if bbox_r['total'] else 0,
            **iou_stats,
        },
        'exact_match': {
            'total': exact_r['total'],
            'correct': exact_r['correct'],
            'acc': exact_r['correct'] / exact_r['total'] if exact_r['total'] else 0,
        },
        'by_family': {k: dict(v) for k, v in by_family.items()},
    }

    # Save report
    eval_dir = Path(out_file).parent if '/' in out_file else Path('eval')
    report_path = f"${GSA_FORGE_ROOT}/eval/vlabel_v3_{tag}.json"
    json.dump(report, open(report_path, 'w'), indent=2, ensure_ascii=False)

    # Save pending judge list
    pending_path = f"${GSA_FORGE_ROOT}/eval/vlabel_v3_{tag}_pending.jsonl"
    with open(pending_path, 'w') as f:
        for item in pending_judge:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Print summary
    print(f"{'='*60}")
    print(f"  {tag} - Evaluation v3 (Rule + Pending Judge)")
    print(f"{'='*60}")
    print(f"  Total samples:     {n}")
    print(f"  GT invalid:        {results['gt_invalid']} (skipped)")
    print(f"  Rule-scored:       {rule_total} ({rule_correct} correct, acc={report['rule_acc']*100:.1f}%)")
    print(f"  Needs VLM-judge:   {results['needs_judge']} (output to pending.jsonl)")
    print(f"")
    print(f"  --- bbox IoU (HARD, most reliable) ---")
    print(f"  Samples:  {bbox_r['total']}")
    print(f"  Acc@0.5:  {iou_stats.get('iou@0.5',0)*100:.1f}%")
    print(f"  Mean IoU: {iou_stats.get('mean',0):.3f}")
    print(f"  Median:   {iou_stats.get('median',0):.3f}")
    print(f"  Acc@0.7:  {iou_stats.get('iou@0.7',0)*100:.1f}%")
    print(f"  Acc@0.9:  {iou_stats.get('iou@0.9',0)*100:.1f}%")
    print(f"")
    print(f"  --- Exact match ---")
    print(f"  Samples:  {exact_r['total']}")
    print(f"  Acc:      {report['exact_match']['acc']*100:.1f}%")
    print(f"")
    print(f"  --- By family (rule-scored) ---")
    for tf in sorted(by_family.keys()):
        d = by_family[tf]
        parts = []
        if d['bbox_total']: parts.append(f"bbox {d['bbox_correct']}/{d['bbox_total']}")
        if d['exact_total']: parts.append(f"exact {d['exact_correct']}/{d['exact_total']}")
        if d['needs_judge']: parts.append(f"judge={d['needs_judge']}")
        if d['gt_invalid']: parts.append(f"invalid={d['gt_invalid']}")
        print(f"    {tf}: {', '.join(parts)}")
    print(f"")
    print(f"  -> Report: {report_path}")
    print(f"  -> Pending: {pending_path} ({len(pending_judge)} items)")


if __name__ == '__main__':
    main()
