#!/usr/bin/env python3
"""Score v4: Improved multi-model evaluation for v0616 dataset.

Improvements over v3:
- Uses gt_ir structured fields (boxes, points) when available
- Point scoring: point in GT bbox or PCK distance
- answer_type based routing (from manifest)
- Better choice/attribute matching (case-insensitive, strip)
- Per answer_type breakdown

Usage:
    python3 eval/score_vlabel_v4.py <output.jsonl> <gt.jsonl> [tag]
"""
import json, re, sys, math
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


def parse_point(text):
    """Extract (x, y) point from text."""
    text = str(text)
    m = re.search(r'"point"?\s*:?\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.search(r'<point>\[\[(\d+),(\d+)\]\]</point>', text)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.search(r'\((\d+)\s*,\s*(\d+)\)', text)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


def iou(a, b):
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


def point_in_bbox(pt, bbox):
    """Check if point falls within bbox (with 5% tolerance on [0,1000] scale)."""
    if not pt or not bbox:
        return False
    x, y = pt
    x1, y1, x2, y2 = bbox
    tol = 50  # 5% of 1000
    return (x1 - tol) <= x <= (x2 + tol) and (y1 - tol) <= y <= (y2 + tol)


def point_distance(p1, p2, diag=1414.0):
    """Normalized distance (PCK style, scale [0,1000] diag=1414)."""
    if not p1 or not p2:
        return 1.0
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx*dx + dy*dy) / diag


def clean_response(text):
    text = str(text).strip()
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    return text.strip()


def normalize_choice(text):
    """Normalize choice answers for comparison."""
    text = text.strip().lower()
    text = re.sub(r'^(answer|choice|option)\s*[:?]\s*', '', text)
    text = re.sub(r'[.?,?;?]$', '', text)
    return text.strip()


def main():
    out_file = sys.argv[1]
    gt_file = sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else 'ckpt'

    outs = [json.loads(l) for l in open(out_file)]
    gts = [json.loads(l) for l in open(gt_file)]
    n = min(len(outs), len(gts))

    stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'ious': [], 'distances': []})
    overall = {'scored': 0, 'correct': 0, 'gt_invalid': 0, 'needs_judge': 0}

    for i in range(n):
        pred_raw = outs[i].get('response', '')
        g = gts[i]
        ga = g.get('model_answer', '')
        tf = g.get('task_family', 'unknown')
        answer_type = g.get('answer_type', 'unknown')
        gt_ir = g.get('gt_ir', {}) if isinstance(g.get('gt_ir'), dict) else {}

        pred = clean_response(pred_raw)
        gt_clean = clean_response(ga)

        # GT invalid
        if not ga or ga.strip() in ('<FAILED>', 'None', ''):
            overall['gt_invalid'] += 1
            continue

        scored = False

        # === BBOX evaluation ===
        if answer_type == 'bbox_region' or 'bbox_grounding' in tf:
            gt_boxes = gt_ir.get('boxes', [])
            gt_bbox = gt_boxes[0] if gt_boxes else parse_bbox(ga)
            pred_bbox = parse_bbox(pred)

            if gt_bbox and pred_bbox:
                v = iou(pred_bbox, gt_bbox)
                stats['bbox']['total'] += 1
                stats['bbox']['ious'].append(v)
                if v >= 0.5:
                    stats['bbox']['correct'] += 1
                scored = True

        # === POINT evaluation ===
        elif answer_type == 'point_affordance' or 'point' in tf:
            gt_points = gt_ir.get('points', [])
            gt_boxes = gt_ir.get('boxes', [])
            pred_pt = parse_point(pred)

            if pred_pt:
                if gt_boxes:
                    correct = point_in_bbox(pred_pt, gt_boxes[0])
                    stats['point']['total'] += 1
                    if correct:
                        stats['point']['correct'] += 1
                    scored = True
                elif gt_points:
                    dist = point_distance(pred_pt, tuple(gt_points[0][:2]))
                    stats['point']['total'] += 1
                    stats['point']['distances'].append(dist)
                    if dist <= 0.1:  # PCK@0.1 (10% of diagonal)
                        stats['point']['correct'] += 1
                    scored = True

        # === CHOICE evaluation ===
        elif answer_type == 'choice':
            gt_norm = normalize_choice(gt_clean)
            pred_norm = normalize_choice(pred)
            if gt_norm and pred_norm:
                stats['choice']['total'] += 1
                if gt_norm == pred_norm or gt_norm in pred_norm or pred_norm in gt_norm:
                    stats['choice']['correct'] += 1
                scored = True

        # === COUNT evaluation ===
        elif answer_type == 'count':
            gt_num = re.search(r'(\d+)', gt_clean)
            pred_num = re.search(r'(\d+)', pred)
            if gt_num and pred_num:
                stats['count']['total'] += 1
                if int(gt_num.group(1)) == int(pred_num.group(1)):
                    stats['count']['correct'] += 1
                scored = True

        # === RELATION evaluation ===
        elif answer_type == 'relation':
            gt_norm = normalize_choice(gt_clean)
            pred_norm = normalize_choice(pred)
            if gt_norm and pred_norm and len(gt_norm) < 200:
                stats['relation']['total'] += 1
                if gt_norm == pred_norm or gt_norm in pred_norm:
                    stats['relation']['correct'] += 1
                scored = True

        # === ATTRIBUTE/STATE evaluation ===
        elif answer_type == 'attribute_state':
            gt_norm = gt_clean.lower().strip()
            pred_norm = pred.lower().strip()
            if gt_norm and len(gt_norm) < 300:
                stats['attribute']['total'] += 1
                if gt_norm == pred_norm or gt_norm in pred_norm or pred_norm in gt_norm:
                    stats['attribute']['correct'] += 1
                scored = True

        # === FALLBACK: short text exact match ===
        if not scored:
            gt_norm = gt_clean.lower().strip()
            pred_norm = pred.lower().strip()
            if gt_norm and pred_norm and len(gt_norm) < 100:
                stats['exact_other']['total'] += 1
                if gt_norm == pred_norm or gt_norm in pred_norm:
                    stats['exact_other']['correct'] += 1
                scored = True

        if scored:
            overall['scored'] += 1
            # count correct across all
        else:
            overall['needs_judge'] += 1

    # Compute totals
    total_correct = sum(s['correct'] for s in stats.values())
    total_scored = sum(s['total'] for s in stats.values())

    # Print report
    print(f"\n{'='*65}")
    print(f"  {tag} ? Evaluation v4 (Multi-type Rule-based)")
    print(f"{'='*65}")
    print(f"  Total samples:     {n}")
    print(f"  GT invalid:        {overall['gt_invalid']}")
    print(f"  Rule-scored:       {total_scored} / {n - overall['gt_invalid']} evaluable")
    print(f"  Overall accuracy:  {total_correct}/{total_scored} = {total_correct/max(1,total_scored)*100:.1f}%")
    print(f"  Needs VLM judge:   {overall['needs_judge']}")
    print()
    print(f"  {'Type':<15} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    for typ in ['bbox', 'point', 'choice', 'count', 'relation', 'attribute', 'exact_other']:
        s = stats[typ]
        if s['total'] > 0:
            acc = s['correct'] / s['total'] * 100
            print(f"  {typ:<15} {s['correct']:>8} {s['total']:>8} {acc:>7.1f}%")

    # IoU details
    if stats['bbox']['ious']:
        ious = stats['bbox']['ious']
        print(f"\n  --- Bbox IoU details ---")
        print(f"  Mean IoU:   {sum(ious)/len(ious):.3f}")
        print(f"  Acc@0.5:    {sum(1 for x in ious if x>=0.5)/len(ious)*100:.1f}%")
        print(f"  Acc@0.7:    {sum(1 for x in ious if x>=0.7)/len(ious)*100:.1f}%")
        print(f"  Zero IoU:   {sum(1 for x in ious if x==0)}/{len(ious)}")

    print()

    # Save JSON report
    report = {
        'tag': tag, 'total': n, 'gt_invalid': overall['gt_invalid'],
        'rule_scored': total_scored, 'rule_correct': total_correct,
        'overall_acc': total_correct / max(1, total_scored),
        'needs_judge': overall['needs_judge'],
        'by_type': {k: {'correct': v['correct'], 'total': v['total'],
                        'acc': v['correct']/max(1,v['total'])}
                    for k, v in stats.items() if v['total'] > 0},
    }
    if stats['bbox']['ious']:
        report['bbox_iou_mean'] = sum(stats['bbox']['ious']) / len(stats['bbox']['ious'])

    report_path = f"${GSA_FORGE_ROOT}/eval/v4_{tag}.json"
    json.dump(report, open(report_path, 'w'), indent=2, ensure_ascii=False)
    print(f"  Report saved: {report_path}")


if __name__ == '__main__':
    main()
