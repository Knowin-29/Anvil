#!/usr/bin/env python3
"""Build a clean, balanced, multi-source GSA training/eval set from the LOCAL registry.

Fixes the four data issues:
  (2) diversity: samples across MANY real GSA-relevant sources, not one washing-machine dump;
  (3) uses the remapped local registry (sources_registry_local.yaml, 41 available), keeping only
      GSA-relevant embodied/spatial/grounding sources (skips chart/doc/diagram QA);
  (4) desensitized RELATIVE image paths: rewrites `/mnt/*` and `/path/to/workspace/*` image refs to
      `images/<source>/<basename>` and records the real absolute root in an images_manifest, so
      the jsonl is HF-upload-ready; keeps the REAL source name (not "vlm_benchmark").
  Coordinates: normalizes pixel point/bbox to thousandth [0,999] via per-record width/height;
      sources already thousandth or QA-only pass through. Prompts carry the coord convention.

This is the reuse-ready path (user choice): public sources that are ALREADY annotated
(msswift_ready) are the backbone; the flywheel only fills missing dims later (needs VLM endpoint).

Usage:
    python trains/build_multisource_clean.py --registry \
        /path/to/Anvil/data/sources_registry_local.yaml \
        --out-train trains/gsa_multisource_train.jsonl --per-source-cap 8000
"""
import argparse
import json
import os
import random
import re
import yaml

# GSA-relevant sources only (embodied perception: grounding / spatial / pointing / affordance).
# Excludes chart/doc/diagram/table QA (chartqa, docvqa, plotqa, dvqa, figureqa, robut_wikisql,
# mapqa, raven, ai2d, scienceqa) which are general VQA, not embodied perception.
GSA_SOURCES = {
    "refspatial_2d_choice", "refspatial_2d_reasoning", "refspatial_3d_choice",
    "refspatial_3d_reasoning", "refspatial_3d_vacant", "refspatial_3d_visual_choice",
    "sat_spatial_qa", "whatsup_spatial", "vsr", "clevr", "visual7w",
    "robopoint_ref", "roborefit_rec", "fsd_free_point", "pixmo_points",
    "pixmo_pointing_explanations", "pixmo_count_counting", "tallyqa",
    "prism", "sharegpt4o",
}

IMG_REMAPS = [
    ("/mnt/data/datasets", "/path/to/datasets/datasets"),
    ("/mnt/data", "/path/to/datasets"),
    ("/path/to/storage", "/path/to/workspace/foundation"),
]
COORD_NOTE = "All coordinates are normalized to [0, 999] (top-left origin)."
POINT_RE = re.compile(r'"point_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]')
BBOX_RE = re.compile(r'"bbox(?:_2d)?"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]')
# Tolerance: coords may slightly exceed declared dims; beyond this the record is a source
# annotation error (e.g. FSD points at 708 on a 640-wide image) and is DROPPED as bad quality.
OOB_TOL = 1.08


def remap_img(p):
    for a, b in IMG_REMAPS:
        if p.startswith(a):
            return p.replace(a, b, 1)
    return p


def clip999(v):
    return max(0, min(999, int(round(v))))


def _all_coords(text):
    vals = [float(v) for tup in POINT_RE.findall(text) for v in tup]
    vals += [float(v) for tup in BBOX_RE.findall(text) for v in tup]
    return vals


def norm_coords_in_text(text, w, h):
    """Return (new_text, ok). Detect the coordinate scale robustly:

    Many 'point' sources (robopoint/roborefit/fsd/pixmo) store coords ALREADY in thousandth
    [0,999] even though they also carry the original width/height as metadata. So we do NOT
    blindly divide by w/h. Rule:
      - If all coords already lie within [0, ~1000] -> already thousandth, pass through.
      - Else if w/h known and coords look like pixels (max <= max(w,h)*tol) -> normalize by w/h.
      - Else (coords exceed both 1000 and the declared dims) -> genuine error, drop (ok=False).
    """
    vals = _all_coords(text)
    if not vals:
        return text, True
    mx = max(vals)
    # already normalized to thousandth
    if mx <= 1000 * OOB_TOL:
        return text, True
    # looks like pixels within the declared image -> normalize
    if w and h and mx <= max(w, h) * OOB_TOL:
        def _pt(m):
            x, y = float(m.group(1)), float(m.group(2))
            return f'"point_2d": [{clip999(x/w*1000)}, {clip999(y/h*1000)}]'

        def _bb(m):
            cs = [float(m.group(i)) for i in range(1, 5)]
            return (f'"bbox": [{clip999(cs[0]/w*1000)}, {clip999(cs[1]/h*1000)}, '
                    f'{clip999(cs[2]/w*1000)}, {clip999(cs[3]/h*1000)}]')
        return BBOX_RE.sub(_bb, POINT_RE.sub(_pt, text)), True
    # coords exceed both 1000 and declared dims -> unrecoverable, drop
    return text, False


def process_source(s, cap, seed):
    name = s["name"]
    path = s["path"]
    if not os.path.exists(path):
        return [], f"{name}: jsonl missing"
    out = []
    rnd = random.Random(seed)
    lines = []
    with open(path) as f:
        for i, l in enumerate(f):
            lines.append(l)
            if i > cap * 20:  # cap read for speed
                break
    rnd.shuffle(lines)
    kept = 0
    img_missing = 0
    oob_dropped = 0
    for l in lines:
        if kept >= cap:
            break
        try:
            d = json.loads(l)
        except Exception:
            continue
        imgs = d.get("images") or ([d["image"]] if d.get("image") else [])
        if not imgs:
            continue
        w = d.get("width"); h = d.get("height")
        # width/height may be a list (per-image) or scalar; take first numeric
        if isinstance(w, (list, tuple)):
            w = w[0] if w else None
        if isinstance(h, (list, tuple)):
            h = h[0] if h else None
        try:
            w = float(w) if w else None
            h = float(h) if h else None
        except (TypeError, ValueError):
            w = h = None
        rel_imgs = []
        ok = True
        for ip in imgs:
            real = remap_img(ip)
            if not os.path.exists(real):
                ok = False
                break
            rel = f"images/{name}/{os.path.basename(real)}"
            rel_imgs.append((rel, real))
        if not ok:
            img_missing += 1
            continue
        msgs = d.get("messages", [])
        new_msgs = []
        drop = False
        for m in msgs:
            c = m["content"]
            if m["role"] == "assistant":
                c2, ok = norm_coords_in_text(str(c), w, h)
                if not ok:
                    drop = True
                    break
                c = c2
            new_msgs.append({"role": m["role"], "content": c})
        if drop:
            oob_dropped += 1
            continue
        if not any(m["role"] == "assistant" for m in new_msgs):
            continue
        out.append({
            "messages": new_msgs,
            "images": [r for r, _ in rel_imgs],
            "_source": name,
            "_abs_images": [a for _, a in rel_imgs],
        })
        kept += 1
    return out, f"{name}: kept={kept} img_missing={img_missing} oob_dropped={oob_dropped}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out-train", default="trains/gsa_multisource_train.jsonl")
    ap.add_argument("--images-manifest", default="trains/gsa_multisource_images.json")
    ap.add_argument("--per-source-cap", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reg = yaml.safe_load(open(args.registry))
    srcs = [s for s in reg["sources"]
            if s.get("available_local") and s["name"] in GSA_SOURCES]
    print(f"GSA-relevant available sources: {len(srcs)}")

    all_recs = []
    img_map = {}
    reports = []
    for s in srcs:
        recs, rep = process_source(s, args.per_source_cap, args.seed)
        reports.append(rep)
        print("  " + rep)
        for r in recs:
            for rel, ab in zip(r["images"], r.pop("_abs_images")):
                img_map[rel] = ab
        all_recs += recs

    random.Random(args.seed).shuffle(all_recs)
    from collections import Counter
    dist = Counter(r["_source"] for r in all_recs)
    with open(args.out_train, "w") as f:
        for r in all_recs:
            r.pop("_source", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.images_manifest, "w") as f:
        json.dump({"note": "rel_path -> absolute source image (for materialize/HF upload)",
                   "count": len(img_map), "map": img_map}, f)
    print(f"\nTOTAL: {len(all_recs)} records, {len(img_map)} unique images")
    print("by source:", dict(dist.most_common()))
    print(f"-> {args.out_train}\n-> {args.images_manifest}")


if __name__ == "__main__":
    main()
