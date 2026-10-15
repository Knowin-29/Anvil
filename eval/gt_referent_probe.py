#!/usr/bin/env python3
"""Referent-controlled agreement probe: ask the generator to ground the SAME phrase
the human annotator grounded, then compare boxes.

Why this exists: the unconditioned flywheel picks "the most manipulable object", while
open GT refers to a phrase-specified object. Comparing them mixes *referent selection*
with *localization quality*, which understates localization (measured: mean IoU 0.135
unconditioned, yet ~0.85 on the cases where the referent happened to coincide).

This probe isolates localization: same image, same referring expression, human box vs
generated box. That is the number the paper should quote for annotation geometry
quality, alongside the unconditioned number for the "free-choice" setting.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trains.flywheel.vlm_client import chat_json  # noqa: E402


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def probe_bbox(row: dict):
    phrase = (row.get("phrase") or "").strip()
    if not phrase:
        return None
    prompt = (f'Locate "{phrase}" in the image. Reply with ONLY a JSON array: '
              '[{"label":"<the phrase>","bbox_2d":[x1,y1,x2,y2]}] with integer '
              "coordinates in [0,1000), origin top-left.")
    r = chat_json(row["image"], prompt, temperature=0.0, max_tokens=250)
    it = r[0] if isinstance(r, list) and r else r
    if not isinstance(it, dict):
        return None
    b = it.get("bbox_2d") or it.get("bbox")
    if not (isinstance(b, list) and len(b) == 4):
        return None
    return iou([int(v) for v in b], row["gt"])


def probe_point(row: dict):
    q = (row.get("question") or "").replace("<image>", "").strip()
    if not q:
        return None
    prompt = (q + '\nReply with ONLY a JSON array: [{"point_2d":[x,y]}] with integer '
                  "coordinates in [0,1000), origin top-left.")
    r = chat_json(row["image"], prompt, temperature=0.0, max_tokens=250)
    it = r[0] if isinstance(r, list) and r else r
    if not isinstance(it, dict):
        return None
    p = it.get("point_2d")
    if not (isinstance(p, list) and len(p) == 2):
        return None
    import math
    return min(math.dist(p, g) for g in row["gt"]) / 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="configs/data/gt_store.jsonl")
    ap.add_argument("--n-bbox", type=int, default=300)
    ap.add_argument("--n-point", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="eval/gt_referent_agreement.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.gt).read_text().splitlines() if l.strip()]
    bx = [r for r in rows if r["gt_kind"] == "bbox"][: args.n_bbox]
    pt = [r for r in rows if r["gt_kind"] == "point"][: args.n_point]
    res: dict = {}

    ious = []
    with ThreadPoolExecutor(args.workers) as ex:
        for fu in as_completed([ex.submit(probe_bbox, r) for r in bx]):
            v = fu.result()
            if v is not None:
                ious.append(v)
    if ious:
        res["bbox_referent_controlled"] = {
            "n": len(ious), "mean_iou": round(st.mean(ious), 4),
            "median_iou": round(st.median(ious), 4),
            "iou@0.5": round(100 * sum(i >= .5 for i in ious) / len(ious), 2),
            "iou@0.75": round(100 * sum(i >= .75 for i in ious) / len(ious), 2),
        }

    dists = []
    with ThreadPoolExecutor(args.workers) as ex:
        for fu in as_completed([ex.submit(probe_point, r) for r in pt]):
            v = fu.result()
            if v is not None:
                dists.append(v)
    if dists:
        res["point_referent_controlled"] = {
            "n": len(dists),
            "hit_rate@0.1": round(100 * sum(d <= .1 for d in dists) / len(dists), 2),
            "hit_rate@0.05": round(100 * sum(d <= .05 for d in dists) / len(dists), 2),
            "mean_norm_dist": round(st.mean(dists), 4),
            "median_norm_dist": round(st.median(dists), 4),
        }

    res["interpretation"] = (
        "Referent-controlled: the generator grounds the SAME phrase the human "
        "annotator grounded, so this isolates localization quality from referent "
        "selection. Compare against eval/gt_agreement.json (free-choice setting).")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
