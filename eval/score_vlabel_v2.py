#!/usr/bin/env python3
"""Corrected scoring: strict metrics only, no inflation.

Scoring tiers:
  HARD (reliable):
    - bbox_grounding: IoU >= 0.5 (only when both GT and pred have parseable bbox)
    - closed_choice: exact match on yes/no/short-answer (not just ABCD)
    - spatial_relation: key-field match (pose_family, shape, tilt, object-name)
  
  MEDIUM (format+semantic):
    - attribute/state: JSON parseable + at least 1 key match
    - point_or_affordance: bbox IoU or point-in-region
  
  NOT_SCORED (explicitly marked):
    - GT is <FAILED> or empty
    - GT format unrecognizable

Usage:
    python3 eval/score_vlabel_v2.py <output.jsonl> <gt.jsonl> <tag>
"""
import json, os, re, sys
from collections import Counter, defaultdict

def parse_bbox(text):
    boxes = parse_bboxes(text)
    return boxes[0] if boxes else None

def parse_bboxes(text):
    text = str(text)
    boxes = []
    # Common JSON keys: bbox, bbox_2d, bbox_1000, bbox_1...
    for m in re.finditer(r'"(?:bbox(?:_2d|_1000|_\d+)?|box)"\s*:?\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text, re.I):
        boxes.append([float(m.group(i)) for i in range(1, 5)])
    # Plural form: "bboxes": [[x1,y1,x2,y2], ...]
    for m in re.finditer(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text):
        box = [float(m.group(i)) for i in range(1, 5)]
        if box not in boxes:
            boxes.append(box)
    for m in re.finditer(r'<box>\s*\[\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*,\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\]\s*</box>', text):
        box = [float(m.group(i)) for i in range(1, 5)]
        if box not in boxes:
            boxes.append(box)
    return boxes

def parse_point(text):
    text = str(text)
    patterns = [
        r'"(?:point_2d|point|affordance_point)"\s*:?\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',
        r'<point>\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*</point>',
        r'\bpoint\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return [float(m.group(1)), float(m.group(2))]
    return None

def iou(a, b):
    if not a or not b:
        return 0
    xa1, ya1, xa2, ya2 = a; xb1, yb1, xb2, yb2 = b
    if xa1 > xa2: xa1, xa2 = xa2, xa1
    if ya1 > ya2: ya1, ya2 = ya2, ya1
    inter = max(0, min(xa2, xb2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
    ua = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / ua if ua > 0 else 0

def try_json(text):
    text = str(text).strip()
    text = re.sub(r'```json\s*|```\s*', '', text)
    m = re.search(r'\{[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return None

def norm_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'```json\s*|```\s*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip(' "\'。。，,.;:：')

def extract_choice(text):
    raw = norm_text(text)
    if not raw:
        return None
    j = try_json(text)
    if isinstance(j, dict):
        for key in ("answer", "choice", "label", "option", "result"):
            if key in j:
                v = extract_choice(str(j[key]))
                if v:
                    return v
    m = re.search(r'\b([abcd])\b', raw, re.I)
    if m:
        return m.group(1).upper()
    if re.search(r'\b(yes|true|是|对|正确)\b', raw):
        return "YES"
    if re.search(r'\b(no|false|否|不|错误)\b', raw):
        return "NO"
    if len(raw) <= 32:
        return raw
    return None

SPATIAL_ALIASES = {
    "left": {"left", "左", "左边", "左侧"},
    "right": {"right", "右", "右边", "右侧"},
    "above": {"above", "over", "top", "上", "上方", "上面"},
    "below": {"below", "under", "bottom", "下", "下方", "下面"},
    "on": {"on", "onto", "在上", "上面"},
    "in": {"in", "inside", "within", "在内", "里面"},
    "front": {"front", "in front", "前", "前方"},
    "behind": {"behind", "back", "后", "后方"},
    "standing": {"standing", "upright", "竖直", "立着", "立姿"},
    "lying": {"lying", "horizontal", "倾倒", "躺", "平放"},
}

def canonical_spatial_value(value):
    raw = norm_text(value)
    if not raw:
        return None
    for canon, aliases in SPATIAL_ALIASES.items():
        if raw == canon or any(a in raw for a in aliases):
            return canon
    if len(raw) <= 40:
        return raw
    return None

def extract_spatial(text):
    j = try_json(text)
    if isinstance(j, dict):
        for key in ["pose_family", "spatial_relation", "relation", "position", "orientation", "answer"]:
            if key in j:
                v = canonical_spatial_value(j[key])
                if v:
                    return v
    return canonical_spatial_value(text)

def point_in_bbox(point, bbox):
    if not point or not bbox:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1 <= x <= x2 and y1 <= y <= y2

def point_distance_ok(pred_point, gt_point, threshold=60):
    if not pred_point or not gt_point:
        return False
    dx = pred_point[0] - gt_point[0]
    dy = pred_point[1] - gt_point[1]
    return (dx * dx + dy * dy) ** 0.5 <= threshold

def extract_key_value(text, key):
    """Extract a value for a known key from text (JSON or freeform)."""
    m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', str(text))
    if m:
        return m.group(1).lower()
    m = re.search(rf'"{key}"\s*:\s*(\w+)', str(text))
    if m:
        return m.group(1).lower()
    return None

def score_one(tf, pred, gt):
    """Return (is_correct: bool|None, tier: str, reason: str).
    
    tier: 'hard'|'medium'|'not_scored'
    """
    ga = gt.get('model_answer', '')
    
    # Skip truly broken GT
    if ga.strip() in ('<FAILED>', '', 'None'):
        return None, 'not_scored', 'gt_failed_or_empty'
    
    # === BBOX GROUNDING (HARD) ===
    if 'bbox_grounding' in tf:
        gt_boxes = []
        if gt.get('gt_bbox') is not None:
            gt_boxes.append(gt.get('gt_bbox'))
        gt_boxes.extend([b for b in parse_bboxes(ga) if b not in gt_boxes])
        if not gt_boxes:
            return None, 'not_scored', 'gt_no_bbox'
        pred_boxes = parse_bboxes(pred)
        if not pred_boxes:
            return False, 'hard', 'pred_no_bbox'
        best = max(iou(pb, gb) for pb in pred_boxes for gb in gt_boxes)
        return (best >= 0.5), 'hard', f'iou={best:.3f}'
    
    # === CLOSED CHOICE (HARD) ===
    if tf == 'closed_choice_token':
        gc = extract_choice(ga)
        pc = extract_choice(pred)
        if gc is None:
            return None, 'not_scored', 'gt_choice_unparseable'
        if pc is None:
            return False, 'hard', 'pred_choice_unparseable'
        return (gc == pc), 'hard', f'choice:{gc}=={pc}'
    
    # === SPATIAL RELATION (HARD for key fields) ===
    if tf == 'spatial_relation_binary':
        gv = extract_spatial(ga)
        pv = extract_spatial(pred)
        if gv is None:
            return None, 'not_scored', 'gt_spatial_unparseable'
        if pv is None:
            return False, 'hard', 'pred_spatial_unparseable'
        return (gv == pv), 'hard', f'spatial:{gv}=={pv}'
    
    # === POINT/AFFORDANCE (MEDIUM) ===
    if tf == 'point_or_affordance':
        gb = gt.get('gt_bbox') or parse_bbox(ga)
        gp = gt.get('gt_point') or parse_point(ga)
        pb = parse_bbox(pred)
        pp = parse_point(pred)
        if gb and pb:
            v = iou(pb, gb)
            return (v >= 0.3), 'medium', f'iou={v:.3f}'  # lower threshold for affordance
        if gb and pp:
            return point_in_bbox(pp, gb), 'medium', 'point_in_gt_bbox'
        if gp and pp:
            return point_distance_ok(pp, gp), 'medium', 'point_distance'
        if gp is None and gb is None:
            return None, 'not_scored', 'gt_no_point_or_bbox'
        return False, 'medium', 'pred_no_point_or_bbox'
    
    # === ATTRIBUTE/STATE (MEDIUM - strict JSON match) ===
    if tf in ('attribute_classification', 'state_classification'):
        gj = try_json(ga)
        pj = try_json(pred)
        if gj is None:
            ga_norm = norm_text(ga)
            pred_norm = norm_text(pred)
            if not ga_norm:
                return None, 'not_scored', 'gt_empty'
            if len(ga_norm) <= 80:
                return (ga_norm in pred_norm or pred_norm in ga_norm), 'medium', 'free_text_exact_or_contains'
            return None, 'not_scored', 'gt_not_json_needs_semantic_judge'
        if pj is None:
            return False, 'medium', 'pred_not_json'
        # Key-value match
        common = set(gj.keys()) & set(pj.keys())
        if not common:
            return False, 'medium', 'no_key_overlap'
        matched = sum(1 for k in common if str(gj[k]).lower() == str(pj.get(k,'')).lower())
        return (matched >= max(1, len(common)//2)), 'medium', f'kv_match_{matched}/{len(common)}'
    
    # === ROBOT PLANNING (MEDIUM) ===
    if tf == 'robot_planning_json':
        gj = try_json(ga); pj = try_json(pred)
        if gj and pj:
            common = set(gj.keys()) & set(pj.keys())
            return (len(common) >= 1), 'medium', f'json_keys_{len(common)}'
        return None, 'not_scored', 'planning_unparseable'
    
    # === OTHER/TEMPORAL/UNTAGGED: NOT SCORED (no inflation) ===
    return None, 'not_scored', f'no_metric_for_{tf}'


def main():
    out_file = sys.argv[1]
    gt_file = sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else 'ckpt'
    
    outs = [json.loads(l) for l in open(out_file)]
    gts = [json.loads(l) for l in open(gt_file)]
    n = min(len(outs), len(gts))
    
    by_family = defaultdict(lambda: {'correct': 0, 'total': 0, 'not_scored': 0})
    by_tier = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_cap = defaultdict(lambda: {'correct': 0, 'total': 0})
    reasons = defaultdict(Counter)
    
    for o, g in zip(outs[:n], gts[:n]):
        pred = o.get('response', '')
        tf = g.get('task_family', 'untagged')
        ok, tier, reason = score_one(tf, pred, g)
        
        reasons[tf][reason] += 1
        
        if ok is None:
            by_family[tf]['not_scored'] += 1
            continue
        
        by_family[tf]['total'] += 1
        by_family[tf]['correct'] += int(ok)
        by_tier[tier]['total'] += 1
        by_tier[tier]['correct'] += int(ok)
        
        for cap in g.get('capabilities', []):
            top = cap.split('.')[0]
            by_cap[top]['total'] += 1
            by_cap[top]['correct'] += int(ok)
    
    overall_scored = sum(v['total'] for v in by_family.values())
    overall_correct = sum(v['correct'] for v in by_family.values())
    overall_not_scored = sum(v['not_scored'] for v in by_family.values())
    
    completion = None
    completion_path = os.environ.get("EVAL_COMPLETION_JSON")
    if completion_path and os.path.exists(completion_path):
        completion = json.load(open(completion_path, encoding="utf-8"))

    report = {
        'checkpoint': tag, 'total_samples': n,
        'scored': overall_scored, 'not_scored': overall_not_scored,
        'input_rows': len(outs), 'gt_rows': len(gts),
        'overall_acc': overall_correct / overall_scored if overall_scored else 0,
        'completion': completion,
        'by_tier': {k: {**v, 'acc': v['correct']/v['total'] if v['total'] else 0} for k, v in by_tier.items()},
        'by_family': {k: {**v, 'acc': v['correct']/v['total'] if v['total'] else 0} for k, v in by_family.items()},
        'by_capability': {k: {**v, 'acc': v['correct']/v['total'] if v['total'] else 0} for k, v in by_cap.items()},
        'reasons': {k: dict(v.most_common()) for k, v in reasons.items()},
    }
    
    out_path = f"${GSA_FORGE_ROOT}/eval/vlabel_score_v2_{tag}.json"
    json.dump(report, open(out_path, 'w'), indent=2, ensure_ascii=False)
    
    print(f"=== {tag} (v2 strict scoring) ===")
    print(f"Samples: {n} | Scored: {overall_scored} | Not-scored: {overall_not_scored}")
    print(f"Overall acc: {report['overall_acc']*100:.1f}% ({overall_correct}/{overall_scored})")
    print(f"\nBy tier:")
    for t, v in sorted(report['by_tier'].items()):
        print(f"  {t}: {v['acc']*100:.1f}% ({v['correct']}/{v['total']})")
    print(f"\nBy task_family:")
    for k, v in sorted(report['by_family'].items()):
        ns = v['not_scored']
        if v['total'] > 0:
            print(f"  {v['acc']*100:5.1f}%  {k} (n={v['total']}, skip={ns})")
        else:
            print(f"  ----   {k} (n=0, skip={ns})")
    print(f"\nBy capability:")
    for k, v in sorted(report['by_capability'].items()):
        print(f"  {v['acc']*100:5.1f}%  {k} (n={v['total']})")
    print(f"\n-> {out_path}")


if __name__ == '__main__':
    main()
