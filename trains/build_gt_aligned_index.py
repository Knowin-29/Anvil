#!/usr/bin/env python3
"""Build a GT-anchored image index + ground-truth store.

Selects images that ALREADY carry human ground truth in open datasets, and records
that GT in a normalized form. Running the flywheel on exactly these images gives us:

  1. an objective quality measure for the flywheel (its bbox/point vs the human GT
     on the SAME image) -- replacing unverifiable "human agreement" claims;
  2. a trustworthy benchmark test split whose answers are HUMAN GT, not model output;
  3. a clean train/val separation keyed on image identity.

Normalized GT record:
  {image, source, gt_kind: bbox|point, gt: [...], phrase: str}
All coordinates thousandth-scale [0,1000).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

_VG_IMG = "/path/to/datasets/datasets/transferred_datasets/datasets_train/robopoint-data/images/"
REMAP = [("/mnt/data/da/transferred_dataset/visual_genome/images/", _VG_IMG),
         ("/mnt/data/datasets/", "/path/to/datasets/datasets/"),
         ("/mnt/data/", "/path/to/datasets/"),
         ("/path/to/storage/", "/path/to/workspace/foundation/")]

D = "/path/to/datasets/datasets/dataset_jsons_qwen3_swift"
SOURCES = [
    # name, jsonl, gt_kind, scene
    ("visual_genome", f"{D}/visual_genome/vg_grounding_swift.jsonl", "bbox", "grounding"),
    ("robopoint_ref", f"{D}/robopoint_rft_point_ref_train_40000_test_2000_0417/"
                      "data_train_internvl_no_think_to_swift_onlypoint.jsonl", "point", "tabletop"),
    ("roborefit_rec", f"{D}/roborefit_rft_point_rec_train_35000_test_1000_0502/"
                      "data_train_internvl_no_think_to_swift_onlypoint.jsonl", "point", "tabletop"),
    ("prism_grasp", f"{D}/PRISM/train_internvl_no_ref_to_swift.jsonl", "point", "action_planning"),
]

_RE_BBOX = re.compile(r"\[\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\]\]")
_RE_PT_TAG = re.compile(r"<point>\[\[(\d+\.?\d*),\s*(\d+\.?\d*)\]\]</point>")
_RE_PT_JSON = re.compile(r'"point_2d"\s*:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)\]')
_RE_PHRASE = re.compile(r'"([^"]{3,80})"')


def remap(p: str) -> str:
    for a, b in REMAP:
        if p.startswith(a):
            return b + p[len(a):]
    return p


def parse_gt(ans: str, kind: str):
    if kind == "bbox":
        m = _RE_BBOX.search(ans)
        if m:
            x1, y1, x2, y2 = (int(v) for v in m.groups())
            if 0 <= x1 < x2 < 1001 and 0 <= y1 < y2 < 1001:
                return [x1, y1, x2, y2]
        return None
    pts = [[float(a), float(b)] for a, b in _RE_PT_TAG.findall(ans)]
    pts += [[float(a), float(b)] for a, b in _RE_PT_JSON.findall(ans)]
    pts = [p for p in pts if 0 <= p[0] < 1000 and 0 <= p[1] < 1000]
    return pts or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=6000)
    ap.add_argument("--index-out", default="configs/data/gt_images.index")
    ap.add_argument("--gt-out", default="configs/data/gt_store.jsonl")
    args = ap.parse_args()

    rows, stats, seen_img = [], Counter(), set()
    for name, path, kind, scene in SOURCES:
        if not os.path.isfile(path):
            print(f"  MISS {name}")
            continue
        taken = 0
        with open(path, errors="ignore") as f:
            for ln in f:
                if taken >= args.per_source:
                    break
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                imgs = d.get("images") or []
                if isinstance(imgs, str):
                    imgs = [imgs]
                if not imgs:
                    continue
                img = remap(str(imgs[0]))
                if img in seen_img or not os.path.exists(img):
                    continue
                msgs = d.get("messages") or []
                ans = str(msgs[-1].get("content", "")) if msgs else ""
                q = str(msgs[0].get("content", "")) if msgs else ""
                gt = parse_gt(ans, kind)
                if not gt:
                    continue
                ph = _RE_PHRASE.search(q)
                seen_img.add(img)
                taken += 1
                rows.append({"image": img, "source": name, "gt_kind": kind,
                             "gt": gt, "scene": scene,
                             "phrase": ph.group(1) if ph else "",
                             "question": q[:300]})
        stats[name] = taken
        print(f"  {taken:>6,}  {name} ({kind})", flush=True)

    random.seed(0)
    random.shuffle(rows)
    Path(args.gt_out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    Path(args.index_out).write_text(
        "\n".join(f"{r['image']}\t{r['scene']}" for r in rows) + "\n")
    print(f"\n{len(rows):,} GT-anchored images -> {args.index_out}")
    print(f"GT store -> {args.gt_out}")
    print(json.dumps(dict(stats), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
