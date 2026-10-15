#!/usr/bin/env python3
"""Score checkpoint replay outputs against verifier-correct ground truth.

Scoring per task_family:
  - bbox_grounding_*  : IoU vs GT bbox (correct if IoU >= 0.5)
  - point_or_affordance: point-in-GT-bbox or point distance (uses bbox if present)
  - spatial_relation_binary / closed_choice_token: key field exact match (pose_family / answer letter)
  - attribute_classification / state_classification: JSON parseable + key overlap
  - others (temporal/planning/describe): JSON-parseable / non-empty as format-correctness proxy

Outputs eval/vlabel_score_<tag>.json with per-family and overall metrics.
"""
import json, re, sys
from collections import defaultdict

def parse_bbox(text):
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
    if not a or not b:
        return None
    xa1, ya1, xa2, ya2 = a; xb1, yb1, xb2, yb2 = b
    if xa1 > xa2: xa1, xa2 = xa2, xa1
    if ya1 > ya2: ya1, ya2 = ya2, ya1
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / ua if ua > 0 else 0

def try_json(text):
    text = str(text)
    # strip code fences
    text = re.sub(r'```json|```', '', text)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            return None
    return None

def pose_family(text):
    j = try_json(text)
    if j and 'pose_family' in j:
        return str(j['pose_family']).lower()
    m = re.search(r'"pose_family"\s*:\s*"([^"]+)"', str(text))
    return m.group(1).lower() if m else None

def choice_letter(text):
    m = re.search(r'\b([A-D])\b', str(text).strip())
    return m.group(1) if m else None

def score_one(tf, pred, gt):
    """Return (is_correct, has_score). has_score=False means not scorable (skip)."""
    ga = gt.get('model_answer', '')
    if tf in ('bbox_grounding_single_or_few', 'bbox_grounding_many_objects'):
        pb = parse_bbox(pred)
        gb = gt.get('gt_bbox')
        if gb is None:
            return None, False
        v = iou(pb, gb)
        return (v is not None and v >= 0.5), True
    if tf == 'point_or_affordance':
        pb = parse_bbox(pred); gb = gt.get('gt_bbox')
        if gb:
            v = iou(pb, gb) if pb else None
            return (v is not None and v >= 0.5), True
        # fallback: both produce a point/bbox
        return (parse_bbox(pred) is not None), True
    if tf == 'spatial_relation_binary':
        pp = pose_family(pred); gp = pose_family(ga)
        if gp is None:
            return None, False
        return (pp == gp), True
    if tf == 'closed_choice_token':
        pc = choice_letter(pred); gc = choice_letter(ga)
        if gc is None:
            return None, False
        return (pc == gc), True
    if tf in ('attribute_classification', 'state_classification', 'robot_planning_json'):
        pj = try_json(pred); gj = try_json(ga)
        if gj is None:
            return (pj is not None), True  # format proxy
        if pj is None:
            return False, True
        # key overlap
        common = set(pj.keys()) & set(gj.keys())
        return (len(common) >= max(1, len(gj) // 2)), True
    # describe / temporal / other: format proxy (non-empty)
    return (len(str(pred).strip()) > 5), True

def main():
    out_file = sys.argv[1]
    gt_file = sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else 'ckpt'
    outs = [json.loads(l) for l in open(out_file)]
    gts = [json.loads(l) for l in open(gt_file)]
    n = min(len(outs), len(gts))

    by_family = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_cap = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_correct = 0; overall_total = 0

    for o, g in zip(outs[:n], gts[:n]):
        pred = o.get('response', '')
        tf = g.get('task_family', 'untagged')
        ok, scorable = score_one(tf, pred, g)
        if not scorable:
            continue
        by_family[tf]['total'] += 1
        by_family[tf]['correct'] += int(ok)
        overall_total += 1
        overall_correct += int(ok)
        for cap in g.get('capabilities', []):
            top = cap.split('.')[0]  # G/S/A
            by_cap[top]['total'] += 1
            by_cap[top]['correct'] += int(ok)

    report = {
        'checkpoint': tag,
        'overall': {'correct': overall_correct, 'total': overall_total,
                    'acc': overall_correct / overall_total if overall_total else 0},
        'by_family': {k: {**v, 'acc': v['correct']/v['total'] if v['total'] else 0}
                      for k, v in by_family.items()},
        'by_capability': {k: {**v, 'acc': v['correct']/v['total'] if v['total'] else 0}
                          for k, v in by_cap.items()},
    }
    out = f"${GSA_FORGE_ROOT}/eval/vlabel_score_{tag}.json"
    json.dump(report, open(out, 'w'), indent=2, ensure_ascii=False)
    print(f"=== {tag} ===")
    print(f"Overall acc: {report['overall']['acc']*100:.1f}% ({overall_correct}/{overall_total})")
    print("By task_family:")
    for k, v in sorted(report['by_family'].items()):
        print(f"  {v['acc']*100:5.1f}%  {k} (n={v['total']})")
    print("By capability (G/S/A):")
    for k, v in sorted(report['by_capability'].items()):
        print(f"  {v['acc']*100:5.1f}%  {k} (n={v['total']})")
    print(f"-> {out}")

if __name__ == '__main__':
    main()
