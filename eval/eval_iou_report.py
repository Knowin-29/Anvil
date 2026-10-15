#!/usr/bin/env python3
"""计算 replay 输出与 log GT 的 IoU,生成对比报告。"""
import json, re, sys
from collections import defaultdict

def parse_bbox(text):
    text = str(text)
    # bbox_2d (grounding 模式) 或 bbox
    for key in ['bbox_2d', 'bbox']:
        m = re.search(key + r'"?\s*:?\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
        if m:
            return [float(m.group(i)) for i in range(1,5)]
    # <box>[[x1,y1],[x2,y2]]</box>
    m = re.search(r'<box>\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]</box>', text)
    if m:
        return [float(m.group(i)) for i in range(1,5)]
    return None

def iou(a, b):
    if not a or not b: return None
    xa1,ya1,xa2,ya2 = a; xb1,yb1,xb2,yb2 = b
    if xa1>xa2: xa1,xa2 = xa2,xa1
    if ya1>ya2: ya1,ya2 = ya2,ya1
    ix1,iy1 = max(xa1,xb1), max(ya1,yb1)
    ix2,iy2 = min(xa2,xb2), min(ya2,yb2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    ua = (xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter
    return inter/ua if ua>0 else 0

def main():
    out_file = sys.argv[1] if len(sys.argv)>1 else 'eval/replay_full_out.jsonl'
    gt_file = sys.argv[2] if len(sys.argv)>2 else 'eval/replay_gt.jsonl'
    outs = [json.loads(l) for l in open(out_file)]
    gts = [json.loads(l) for l in open(gt_file)]
    n = min(len(outs), len(gts))
    outs, gts = outs[:n], gts[:n]

    ious = []
    parse_ok = 0
    by_date = defaultdict(list)
    by_dim = defaultdict(list)
    for o, g in zip(outs, gts):
        resp = o.get('response','')
        pred = parse_bbox(resp)
        if pred is not None:
            parse_ok += 1
        v = iou(pred, g['gt_bbox'])
        if v is not None:
            ious.append(v)
            by_date[g['date']].append(v)
            by_dim[g['dimension']].append(v)

    def stats(arr):
        if not arr: return {}
        arr_s = sorted(arr)
        n = len(arr)
        return {
            'n': n, 'mean': sum(arr)/n, 'median': arr_s[n//2],
            'iou@0.5': sum(1 for x in arr if x>=0.5)/n,
            'iou@0.7': sum(1 for x in arr if x>=0.7)/n,
            'iou@0.9': sum(1 for x in arr if x>=0.9)/n,
        }

    report = {
        'total': n,
        'parse_ok': parse_ok,
        'parse_rate': parse_ok/n if n else 0,
        'overall': stats(ious),
        'by_date': {k: stats(v) for k,v in sorted(by_date.items())},
        'by_dimension': {k: stats(v) for k,v in by_dim.items()},
    }
    json.dump(report, open('eval/iou_report.json','w'), indent=2, ensure_ascii=False)

    print("="*60)
    print(f"  Replay IoU 评测 (checkpoint-1000)")
    print("="*60)
    print(f"  样本数: {n}, bbox解析成功: {parse_ok} ({parse_ok/n*100:.1f}%)")
    ov = report['overall']
    print(f"  Mean IoU:   {ov['mean']:.3f}")
    print(f"  Median IoU: {ov['median']:.3f}")
    print(f"  IoU@0.5: {ov['iou@0.5']*100:.1f}%")
    print(f"  IoU@0.7: {ov['iou@0.7']*100:.1f}%")
    print(f"  IoU@0.9: {ov['iou@0.9']*100:.1f}%")
    print(f"\n  按日期:")
    for d, s in report['by_date'].items():
        print(f"    {d}: n={s['n']} mean={s['mean']:.3f} @0.5={s['iou@0.5']*100:.0f}%")
    print(f"\n  按维度:")
    for d, s in report['by_dimension'].items():
        print(f"    {d}: n={s['n']} mean={s['mean']:.3f} @0.5={s['iou@0.5']*100:.0f}%")

if __name__ == '__main__':
    main()
