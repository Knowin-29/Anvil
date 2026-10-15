#!/usr/bin/env python3
"""Build a CLEAN local train + eval set from flywheel-verified data.

Fixes the coordinate-convention bug end to end:
  - Flywheel bboxes are RAW PIXELS + image_size -> normalized to thousandth [0,999].
  - Only records with a locally-reachable image AND a valid normalizable bbox are kept.
  - Prompts state the [0,999] convention explicitly.
  - Train and eval are split by IMAGE (no image in both) to prevent leakage.
  - GT for eval is stored in the SAME thousandth scale as training (so IoU is meaningful).

Outputs:
  trains/clean_sft_train.jsonl      (ms-swift SFT)
  eval/clean_grounding_eval.jsonl   (ms-swift infer prompts)
  eval/clean_grounding_gt.jsonl     (GT bbox in thousandth, aligned by line)

Usage: python trains/build_clean_local.py --eval-n 200 --train-cap 20000
"""
import argparse
import glob
import json
import os
import random

FLYWHEEL = "/path/to/Anvil/data/output/gsa_flywheel_exp/verified"
SYS = ("You are an embodied perception assistant specialized in indoor scene understanding. "
       "All 2D coordinates use a normalized thousandth scale in [0, 999], origin at the top-left "
       "corner of the image. Respond precisely with structured JSON.")
COORD_NOTE = "All coordinates are normalized to [0, 999] (top-left origin)."


def clip999(v):
    return max(0, min(999, int(round(float(v)))))


def norm_bbox(bbox, img_size):
    if not bbox or len(bbox) != 4 or not img_size or len(img_size) != 2:
        return None
    w, h = img_size
    if not w or not h:
        return None
    x1, y1, x2, y2 = bbox
    out = [clip999(x1 / w * 1000), clip999(y1 / h * 1000),
           clip999(x2 / w * 1000), clip999(y2 / h * 1000)]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def norm_point(pt, img_size):
    if not pt or len(pt) != 2 or not img_size or len(img_size) != 2:
        return None
    w, h = img_size
    if not w or not h:
        return None
    return [clip999(pt[0] / w * 1000), clip999(pt[1] / h * 1000)]


SYS_MD = ("You are an embodied perception assistant specialized in indoor scene understanding. "
          "All 2D coordinates use a normalized thousandth scale in [0, 999], origin at the top-left "
          "corner of the image. Respond precisely with structured JSON.")


def build_multidim(train_cap, exclude_basenames, seed=42):
    """Build a multi-dimension clean SFT set (all dims with local images), thousandth coords.
    Excludes any image whose basename is in exclude_basenames (the eval sets), preventing leakage."""
    import glob as _glob
    rnd = random.Random(seed)
    recs = []
    dims = ["G_grounding", "S_spatial", "A_identity_attr", "A_obj_state",
            "A_action_precondition", "A_action_interface"]
    for dim in dims:
        for fp in sorted(_glob.glob(os.path.join(FLYWHEEL, dim, "*.jsonl"))):
            pub = "from_public_datasets" in os.path.basename(fp)
            for l in open(fp):
                try:
                    d = json.loads(l)
                except Exception:
                    continue
                img = d.get("image_path", "")
                if not img or not os.path.exists(img):
                    continue
                if os.path.basename(img) in exclude_basenames:
                    continue
                sz = d.get("image_size")
                msg = _make_record(dim, d, sz, pub)
                if msg:
                    recs.append(msg)
    rnd.shuffle(recs)
    recs = recs[:train_cap]
    with open("trains/clean_sft_multidim.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(r["_dim"] for r in recs)
    for r in recs:
        r.pop("_dim", None)
    # rewrite without _dim
    with open("trains/clean_sft_multidim.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"multidim train: {len(recs)} -> trains/clean_sft_multidim.jsonl")
    print("  by dim:", dict(c))


def _make_record(dim, d, sz, pub):
    def nbb(b):
        if pub:
            if b and len(b) == 4 and all(0 <= v <= 999 for v in b) and b[2] > b[0] and b[3] > b[1]:
                return [int(v) for v in b]
            return None
        return norm_bbox(b, sz)

    def npt2(p):
        if pub:
            if p and len(p) == 2 and all(0 <= v <= 999 for v in p):
                return [int(v) for v in p]
            return None
        return norm_point(p, sz)

    obj = d.get("object_class") or d.get("object")
    q = resp = None
    if dim == "G_grounding":
        b = nbb(d.get("bbox"))
        if not obj or not b:
            return None
        q = f"Detect and locate the '{obj}' in this image. Provide {{\"bbox\":[x1,y1,x2,y2]}}. {COORD_NOTE}"
        resp = {"object": obj, "bbox": b}
    elif dim == "S_spatial":
        s, p, o = d.get("subject"), d.get("predicate"), d.get("object")
        if not (s and p and o):
            return None
        q = f"What is the spatial relationship between the '{s}' and the '{o}'? Answer as {{\"subject\":..,\"predicate\":..,\"object\":..}}."
        resp = {"subject": s, "predicate": p, "object": o}
    elif dim == "A_identity_attr":
        attrs = d.get("attributes")
        if not obj or not attrs:
            return None
        q = f"Identify the visual attributes of the '{obj}'. Answer as {{\"object\":..,\"attributes\":{{..}}}}."
        resp = {"object": obj, "attributes": attrs}
    elif dim == "A_obj_state":
        sv = d.get("state_value")
        if not obj or not sv:
            return None
        q = f"What is the current state of the '{obj}'? Answer as {{\"object\":..,\"state_type\":..,\"state_value\":..}}."
        resp = {"object": obj, "state_type": d.get("state_type", ""), "state_value": sv}
    elif dim == "A_action_precondition":
        if not obj or d.get("can_do_now") is None:
            return None
        act = d.get("action_type", "manipulate")
        q = f"Can a robot directly {act} the '{obj}' right now? Answer as {{\"can_do_now\":true/false,\"blocked_by\":[..]}}."
        resp = {"object": obj, "action_type": act, "can_do_now": bool(d.get("can_do_now")), "blocked_by": d.get("blocked_by", [])}
    elif dim == "A_action_interface":
        gp = npt2(d.get("grasp_point_2d"))
        if not obj or not gp:
            return None
        act = d.get("action_type", "grasp")
        q = f"How should a robot {act} the '{obj}'? Provide {{\"grasp_point_2d\":[x,y]}}. {COORD_NOTE}"
        resp = {"object": obj, "grasp_point_2d": gp}
    else:
        return None
    return {"messages": [{"role": "system", "content": SYS_MD},
                         {"role": "user", "content": f"<image>\n{q}"},
                         {"role": "assistant", "content": json.dumps(resp, ensure_ascii=False)}],
            "images": [d["image_path"]], "_dim": dim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--train-cap", type=int, default=20000)
    ap.add_argument("--multidim", action="store_true", help="build all-dimension training set")
    ap.add_argument("--exclude-eval-dir", default="eval/gsa_bench", help="dir with *_gt.jsonl eval sets to exclude")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    if args.multidim:
        import glob as _glob
        exclude = set()
        # eval GT files may lack an image field; the *_eval.jsonl files carry images[].
        for ef in _glob.glob(os.path.join(args.exclude_eval_dir, "*_eval.jsonl")):
            for l in open(ef):
                try:
                    d = json.loads(l)
                    for i in d.get("images", []):
                        exclude.add(os.path.basename(i))
                except Exception:
                    pass
        for gt in _glob.glob(os.path.join(args.exclude_eval_dir, "*_gt.jsonl")):
            for l in open(gt):
                try:
                    d = json.loads(l)
                    if d.get("image"):
                        exclude.add(os.path.basename(d["image"]))
                except Exception:
                    pass
        print(f"excluding {len(exclude)} eval image basenames")
        build_multidim(args.train_cap, exclude, args.seed)
        return

    # Collect valid G_grounding records (local image + normalizable bbox)
    recs = []
    for fp in sorted(glob.glob(os.path.join(FLYWHEEL, "G_grounding", "*.jsonl"))):
        pub = "from_public_datasets" in os.path.basename(fp)
        for l in open(fp):
            try:
                d = json.loads(l)
            except Exception:
                continue
            img = d.get("image_path", "")
            obj = d.get("object_class") or d.get("object")
            bbox = d.get("bbox")
            if not img or not obj or not bbox or not os.path.exists(img):
                continue
            if pub:  # already thousandth
                if len(bbox) == 4 and all(0 <= v <= 999 for v in bbox) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    nb = [int(round(v)) for v in bbox]
                else:
                    continue
            else:
                nb = norm_bbox(bbox, d.get("image_size"))
                if nb is None:
                    continue
            recs.append({"img": img, "obj": obj, "bbox": nb})

    print(f"valid grounding records (local img + normalizable bbox): {len(recs)}")
    if len(recs) < args.eval_n + 100:
        print("WARNING: few valid records; check flywheel image availability")

    # Split by image to prevent leakage
    random.shuffle(recs)
    eval_imgs = set()
    eval_recs, train_recs = [], []
    for r in recs:
        base = os.path.basename(r["img"])
        if len(eval_recs) < args.eval_n and base not in eval_imgs:
            eval_imgs.add(base)
            eval_recs.append(r)
        elif base not in eval_imgs:
            train_recs.append(r)
    train_recs = train_recs[:args.train_cap]

    def user_q(obj):
        return (f"<image>\nDetect and locate the '{obj}' in this image. Provide the bounding box "
                f"as {{\"bbox\": [x1,y1,x2,y2]}}. {COORD_NOTE}")

    with open("trains/clean_sft_train.jsonl", "w") as f:
        for r in train_recs:
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": user_q(r["obj"])},
                    {"role": "assistant", "content": json.dumps({"object": r["obj"], "bbox": r["bbox"]}, ensure_ascii=False)},
                ],
                "images": [r["img"]],
            }, ensure_ascii=False) + "\n")

    with open("eval/clean_grounding_eval.jsonl", "w") as fe, open("eval/clean_grounding_gt.jsonl", "w") as fg:
        for r in eval_recs:
            fe.write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": user_q(r["obj"])},
                ],
                "images": [r["img"]],
            }, ensure_ascii=False) + "\n")
            fg.write(json.dumps({"object": r["obj"], "bbox": r["bbox"], "image": r["img"]}, ensure_ascii=False) + "\n")

    print(f"train: {len(train_recs)} -> trains/clean_sft_train.jsonl")
    print(f"eval:  {len(eval_recs)} -> eval/clean_grounding_eval.jsonl (+_gt), split by image (no leak)")


if __name__ == "__main__":
    main()
