#!/usr/bin/env python3
"""Score flywheel output against HUMAN ground truth on the same images.

This is the objective quality evidence the paper needs: for every image that carries
open-dataset human GT, compare what the flywheel produced on that same image.

Metrics
  bbox  vs GT bbox : mean IoU, IoU@0.5, IoU@0.75, and "best-of-detection" IoU
                     (the detection form may name a different object, so we also
                     report the best-matching box among all detected boxes)
  point vs GT point: hit rate inside the GT-derived region, mean normalized distance

Important caveat we report honestly: the flywheel picks the object it considers most
manipulable, while the open GT refers to a phrase-specified object. So a low
single-box IoU is NOT purely a flywheel error -- it is partly a referent mismatch.
The best-of-detection number bounds the geometry quality separately from referent
selection, and both are reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(path: str) -> dict:
    gt = {}
    for ln in Path(path).read_text(errors="ignore").splitlines():
        if not ln.strip():
            continue
        try:
            d = json.loads(ln)
        except Exception:
            continue
        gt[d["image"]] = d
    return gt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="configs/data/gt_store.jsonl")
    ap.add_argument("--pred-glob", default="data/flywheel_gt/shard*/raw")
    ap.add_argument("--out", default="eval/gt_agreement.json")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    print(f"GT images: {len(gt):,}")

    # gather predictions per image per form
    preds = defaultdict(dict)
    for d in glob.glob(args.pred_glob):
        for jf in Path(d).glob("*.jsonl"):
            form = jf.stem
            for ln in jf.read_text(errors="ignore").splitlines():
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                img = r.get("image_path")
                if img in gt:
                    preds[img][form] = r.get("payload")

    res = {"n_gt_images": len(gt), "n_images_with_pred": len(preds)}

    # ---------- bbox agreement ----------
    ious_single, ious_best = [], []
    for img, forms in preds.items():
        g = gt[img]
        if g["gt_kind"] != "bbox":
            continue
        gb = g["gt"]
        p = forms.get("bbox")
        items = p if isinstance(p, list) else [p] if p else []
        if items and isinstance(items[0], dict):
            b = items[0].get("bbox_2d") or items[0].get("bbox")
            if isinstance(b, list) and len(b) == 4:
                ious_single.append(iou(b, gb))
        cand = []
        for f in ("bbox", "detection"):
            pp = forms.get(f)
            for it in (pp if isinstance(pp, list) else [pp] if pp else []):
                if isinstance(it, dict):
                    b = it.get("bbox_2d") or it.get("bbox")
                    if isinstance(b, list) and len(b) == 4:
                        cand.append(iou(b, gb))
        if cand:
            ious_best.append(max(cand))
    if ious_single:
        res["bbox_vs_gt"] = {
            "n": len(ious_single),
            "mean_iou": round(st.mean(ious_single), 4),
            "iou@0.5": round(100 * sum(i >= .5 for i in ious_single) / len(ious_single), 2),
            "iou@0.75": round(100 * sum(i >= .75 for i in ious_single) / len(ious_single), 2),
        }
    if ious_best:
        res["bbox_best_of_detection_vs_gt"] = {
            "n": len(ious_best),
            "mean_iou": round(st.mean(ious_best), 4),
            "iou@0.5": round(100 * sum(i >= .5 for i in ious_best) / len(ious_best), 2),
            "iou@0.75": round(100 * sum(i >= .75 for i in ious_best) / len(ious_best), 2),
        }

    # ---------- point agreement ----------
    hits, dists = [], []
    for img, forms in preds.items():
        g = gt[img]
        if g["gt_kind"] != "point":
            continue
        gpts = g["gt"]
        cand = []
        for f, key in (("point", "point_2d"), ("dof", "grasp_point_2d"),
                       ("affordance", "contact_point_2d")):
            pp = forms.get(f)
            for it in (pp if isinstance(pp, list) else [pp] if pp else []):
                if isinstance(it, dict) and isinstance(it.get(key), list):
                    cand.append(it[key])
        if not cand:
            continue
        best = min((math.dist(c, gp) for c in cand for gp in gpts), default=None)
        if best is None:
            continue
        dists.append(best / 1000.0)
        hits.append(best <= 100)     # within 10% of image diagonal-ish tolerance
    if dists:
        res["point_vs_gt"] = {
            "n": len(dists),
            "hit_rate@0.1": round(100 * sum(hits) / len(hits), 2),
            "mean_norm_dist": round(st.mean(dists), 4),
            "median_norm_dist": round(st.median(dists), 4),
        }

    res["caveat"] = ("The flywheel selects the most manipulable object while open GT "
                     "refers to a phrase-specified object, so single-box IoU mixes "
                     "geometry error with referent mismatch; best_of_detection bounds "
                     "geometry quality separately.")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
