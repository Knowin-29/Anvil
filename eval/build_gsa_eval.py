#!/usr/bin/env python3
"""Build reliable per-dimension eval sets from flywheel-verified data.

For every evaluable GSA dimension, produce a held-out eval set with:
  - locally-reachable images only,
  - coordinates normalized to thousandth [0,999] (flywheel stores raw pixels),
  - prompts stating the coordinate convention,
  - GT stored in the SAME thousandth scale (so metrics are meaningful),
  - split BY IMAGE from any training pool basename list (no leakage).

Dimensions & metrics (scored by score_gsa_eval.py):
  G_grounding          -> bbox IoU
  S_spatial            -> predicate exact match (+ subject/object match)
  A_identity_attr      -> attribute key/value overlap (F1)
  A_obj_state          -> state_value exact match
  A_action_precondition-> can_do_now accuracy
  A_action_interface   -> grasp_point inside GT bbox (point-in-box) + normalized dist

Outputs under eval/gsa_bench/<dim>_{eval,gt}.jsonl

Usage: python eval/build_gsa_eval.py --per-dim 150 [--exclude-train trains/clean_sft_train.jsonl]
"""
import argparse
import glob
import json
import os
import random

FLYWHEEL = "/path/to/Anvil/data/output/gsa_flywheel_exp/verified"
OUTDIR = "eval/gsa_bench"
SYS = ("You are an embodied perception assistant specialized in indoor scene understanding. "
       "All 2D coordinates use a normalized thousandth scale in [0, 999], origin at the top-left "
       "corner of the image. Respond precisely with structured JSON.")
NOTE = "All coordinates are normalized to [0, 999] (top-left origin)."


def clip999(v):
    return max(0, min(999, int(round(float(v)))))


def nb(bbox, sz):
    if not bbox or len(bbox) != 4 or not sz or len(sz) != 2 or not sz[0] or not sz[1]:
        return None
    w, h = sz
    o = [clip999(bbox[0] / w * 1000), clip999(bbox[1] / h * 1000),
         clip999(bbox[2] / w * 1000), clip999(bbox[3] / h * 1000)]
    return o if o[2] > o[0] and o[3] > o[1] else None


def npt(pt, sz):
    if not pt or len(pt) != 2 or not sz or len(sz) != 2 or not sz[0] or not sz[1]:
        return None
    return [clip999(pt[0] / sz[0] * 1000), clip999(pt[1] / sz[1] * 1000)]


def load_dim(dim):
    recs = []
    for fp in sorted(glob.glob(os.path.join(FLYWHEEL, dim, "*.jsonl"))):
        for l in open(fp):
            try:
                d = json.loads(l)
            except Exception:
                continue
            if os.path.exists(d.get("image_path", "")):
                recs.append(d)
    return recs


def build(dim, recs, n, exclude):
    random.shuffle(recs)
    out_e, out_g = [], []
    used = set()
    for d in recs:
        img = d["image_path"]
        base = os.path.basename(img)
        if base in exclude or base in used:
            continue
        sz = d.get("image_size")
        q = gt = None
        if dim == "G_grounding":
            obj = d.get("object_class") or d.get("object")
            b = nb(d.get("bbox"), sz)
            if not obj or not b:
                continue
            q = f"Detect and locate the '{obj}' in this image. Provide {{\"bbox\":[x1,y1,x2,y2]}}. {NOTE}"
            gt = {"object": obj, "bbox": b}
        elif dim == "S_spatial":
            s, p, o = d.get("subject"), d.get("predicate"), d.get("object")
            if not (s and p and o):
                continue
            q = (f"What is the spatial relationship between the '{s}' and the '{o}'? "
                 f"Answer as {{\"subject\":..,\"predicate\":..,\"object\":..}}.")
            gt = {"subject": s, "predicate": p, "object": o}
        elif dim == "A_identity_attr":
            obj = d.get("object_class") or d.get("object")
            attrs = d.get("attributes")
            if not obj or not attrs:
                continue
            q = f"Identify the visual attributes of the '{obj}'. Answer as {{\"object\":..,\"attributes\":{{..}}}}."
            gt = {"object": obj, "attributes": attrs}
        elif dim == "A_obj_state":
            obj = d.get("object_class") or d.get("object")
            sv = d.get("state_value")
            if not obj or not sv:
                continue
            q = f"What is the current state of the '{obj}'? Answer as {{\"object\":..,\"state_type\":..,\"state_value\":..}}."
            gt = {"object": obj, "state_type": d.get("state_type", ""), "state_value": sv}
        elif dim == "A_action_precondition":
            obj = d.get("object_class") or d.get("object")
            if not obj or d.get("can_do_now") is None:
                continue
            act = d.get("action_type", "manipulate")
            q = (f"Can a robot directly {act} the '{obj}' right now? "
                 f"Answer as {{\"can_do_now\":true/false,\"blocked_by\":[..]}}.")
            gt = {"object": obj, "action_type": act, "can_do_now": bool(d.get("can_do_now")),
                  "blocked_by": d.get("blocked_by", [])}
        elif dim == "A_action_interface":
            obj = d.get("object_class") or d.get("object")
            gp = npt(d.get("grasp_point_2d"), sz)
            gbox = nb(d.get("bbox"), sz)
            if not obj or not gp:
                continue
            act = d.get("action_type", "grasp")
            q = (f"How should a robot {act} the '{obj}'? Provide {{\"grasp_point_2d\":[x,y]}}. {NOTE}")
            gt = {"object": obj, "grasp_point_2d": gp, "bbox": gbox}
        else:
            continue
        used.add(base)
        out_e.append({"messages": [{"role": "system", "content": SYS},
                                   {"role": "user", "content": f"<image>\n{q}"}],
                      "images": [img]})
        out_g.append(gt)
        if len(out_e) >= n:
            break
    os.makedirs(OUTDIR, exist_ok=True)
    with open(f"{OUTDIR}/{dim}_eval.jsonl", "w") as fe, open(f"{OUTDIR}/{dim}_gt.jsonl", "w") as fg:
        for e, g in zip(out_e, out_g):
            fe.write(json.dumps(e, ensure_ascii=False) + "\n")
            fg.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"{dim}: {len(out_e)} eval samples")
    return len(out_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dim", type=int, default=150)
    ap.add_argument("--exclude-train", default="")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    random.seed(args.seed)

    exclude = set()
    if args.exclude_train and os.path.exists(args.exclude_train):
        for l in open(args.exclude_train):
            try:
                d = json.loads(l)
                for i in d.get("images", []):
                    exclude.add(os.path.basename(i))
            except Exception:
                pass
    print(f"excluding {len(exclude)} training image basenames")

    dims = ["G_grounding", "S_spatial", "A_identity_attr", "A_obj_state",
            "A_action_precondition", "A_action_interface"]
    total = 0
    for dim in dims:
        recs = load_dim(dim)
        total += build(dim, recs, args.per_dim, exclude)
    print(f"TOTAL eval samples: {total}  (A_robot_state skipped: 0% local images)")


if __name__ == "__main__":
    main()
