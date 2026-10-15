#!/usr/bin/env python3
"""Robust grounding IoU eval. Handles <think> blocks and thousandth-scale GT.

Parses the LAST bbox-looking array from a response (so a bbox stated after </think>
or inside the final JSON is captured), computes IoU vs thousandth GT. Reports parse
rate, mean IoU, IoU@{0.5,0.7}. GT and predictions must be in the SAME scale ([0,999]).

Usage:
    python eval/grounding_iou.py --pred <out.jsonl> --gt eval/clean_grounding_gt.jsonl
"""
import argparse
import json
import re
import statistics as st

BBOX_RE = re.compile(r'\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]')


def parse_last_bbox(text):
    """Return the last well-formed 4-int bbox in the text (prefers the final answer,
    not an intermediate one inside <think>). Prefer a bbox in a "bbox":[...] context."""
    text = str(text)
    # Prefer explicit bbox/bbox_2d field (take the last occurrence)
    keyed = re.findall(r'(?:bbox|bbox_2d)"?\s*:?\s*\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]', text)
    if keyed:
        return [float(v) for v in keyed[-1]]
    # Fallback: last 4-int array anywhere
    allm = BBOX_RE.findall(text)
    if allm:
        return [float(v) for v in allm[-1]]
    return None


def iou(a, b):
    if not a or not b:
        return None
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    if xa1 > xa2:
        xa1, xa2 = xa2, xa1
    if ya1 > ya2:
        ya1, ya2 = ya2, ya1
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--name", default="model")
    args = ap.parse_args()

    outs = [json.loads(l) for l in open(args.pred)]
    gts = [json.loads(l) for l in open(args.gt)]
    n = min(len(outs), len(gts))
    ious, parse_ok = [], 0
    for o, g in zip(outs[:n], gts[:n]):
        resp = o.get("response", o.get("messages", [{}])[-1].get("content", "") if o.get("messages") else "")
        p = parse_last_bbox(resp)
        if p is not None:
            parse_ok += 1
        v = iou(p, [float(x) for x in g["bbox"]])
        if v is not None:
            ious.append(v)
    res = {
        "name": args.name, "n": n,
        "parse_rate": round(100 * parse_ok / n, 1) if n else 0,
        "mean_iou": round(st.mean(ious), 3) if ious else 0,
        "iou@0.5": round(100 * sum(1 for i in ious if i >= 0.5) / len(ious), 1) if ious else 0,
        "iou@0.7": round(100 * sum(1 for i in ious if i >= 0.7) / len(ious), 1) if ious else 0,
    }
    print(json.dumps(res, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
