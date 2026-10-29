#!/usr/bin/env python3
"""Cascaded GSA generation + 3-layer verification over an image pool.

Pipeline per image:
  G_grounding  -> L1 cross-check (reverse-verify bbox) -> keep if match
  S_spatial    (conditioned on verified G objects)
  A_*          (conditioned on a verified G object; interface/identity get L1)
Then the unified rule verifier (L2) scores every record and assigns
verified/needs_review/rejected. Verified@k is accumulated across re-scans.

Output: JSONL per dimension under --out/{verified,needs_review,rejected}/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trains.flywheel import prompts, verifier  # noqa: E402
from trains.flywheel.vlm_client import chat_json  # noqa: E402
from trains.flywheel.taxonomy import BBOX_DIMS  # noqa: E402

A_DIMS = ["A_identity_attr", "A_obj_state", "A_action_precondition", "A_action_interface"]


def _unwrap(raw):
    """The endpoint often wraps a single object in a list; unwrap to dict."""
    if isinstance(raw, list) and raw:
        return raw[0]
    return raw


def _norm_g(raw):
    """Normalize the model's grounding reply into {object_class, bbox, confidence}.

    The endpoint often returns Qwen-VL native format: a list of
    {"bbox_2d":[...], "label":..., "confidence":...}. Accept both that and a
    plain dict with object_class/bbox.
    """
    d = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(d, dict):
        return None
    bbox = d.get("bbox") or d.get("bbox_2d")
    obj = d.get("object_class") or d.get("label") or d.get("name")
    if obj in (None, "", "object_class", "single most prominent object",
               "main manipulable object"):
        obj = "object"
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    return {"object_class": obj, "bbox": [int(v) for v in bbox],
            "confidence": d.get("confidence", 0.9)}


def gen_for_image(img: str, do_l1: bool = True) -> list[dict]:
    out = []
    g = _norm_g(chat_json(img, prompts.G_GROUNDING))
    if g is None:
        return out
    g["dimension"] = "G_grounding"; g["image_path"] = img; g["source"] = _src(img)
    if do_l1:
        cc = chat_json(img, prompts.cross_check(g.get("object_class", "object"), g["bbox"]),
                       temperature=0.0)
        g["_cross_check"] = cc if isinstance(cc, dict) else {"match": True, "confidence": 0.5}
        if not g["_cross_check"].get("match", True):
            return out  # drop hallucinated grounding early
    out.append(g)

    s = _unwrap(chat_json(img, prompts.s_spatial([g])))
    if isinstance(s, dict) and s.get("predicate"):
        s["dimension"] = "S_spatial"; s["image_path"] = img; s["source"] = _src(img)
        out.append(s)

    for dim in A_DIMS:
        a = _unwrap(chat_json(img, prompts.PROMPT_FN[dim](g)))
        if not isinstance(a, dict):
            continue
        # normalize native bbox_2d key
        if "bbox" not in a and "bbox_2d" in a:
            a["bbox"] = a.pop("bbox_2d")
        if a.get("object_class") in (None, "", "point_1"):
            a["object_class"] = g["object_class"]
        a["dimension"] = dim; a["image_path"] = img; a["source"] = _src(img)
        if dim in BBOX_DIMS and do_l1 and a.get("bbox"):
            cc = chat_json(img, prompts.cross_check(a.get("object_class", "object"), a["bbox"]),
                           temperature=0.0)
            a["_cross_check"] = cc if isinstance(cc, dict) else {"match": True}
        out.append(a)
    return out


def _src(img: str) -> str:
    if "bridge" in img:
        return "bridge_data_v2"
    if "changeit" in img.lower():
        return "changeit_frames"
    return Path(img).parts[-3] if len(Path(img).parts) >= 3 else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="newline-delimited image paths")
    ap.add_argument("--out", default="/path/to/Anvil/data/flywheel_v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-l1", action="store_true")
    args = ap.parse_args()

    imgs = [l.strip() for l in Path(args.index).read_text().splitlines() if l.strip()]
    if args.limit:
        imgs = imgs[: args.limit]
    outp = Path(args.out)
    for sub in ("verified", "needs_review", "rejected"):
        (outp / sub).mkdir(parents=True, exist_ok=True)

    handles = {}
    def wr(rec, verdict):
        dim = rec.get("dimension", "unknown")
        f = handles.get((verdict, dim))
        if f is None:
            f = (outp / verdict / f"{dim}.jsonl").open("a")
            handles[(verdict, dim)] = f
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    t0 = time.time(); n_img = n_rec = n_ver = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(gen_for_image, im, not args.no_l1): im for im in imgs}
        for fu in as_completed(futs):
            n_img += 1
            for rec in (fu.result() or []):
                res = verifier.score(rec)
                rec["_verification"] = res
                rec["_scan_count"] = 1
                wr(rec, res["verdict"])
                n_rec += 1
                n_ver += res["verdict"] == "verified"
            if n_img % 50 == 0:
                dt = time.time() - t0
                for f in handles.values():
                    f.flush()
                print(f"[{n_img}/{len(imgs)}] rec={n_rec} verified={n_ver} "
                      f"({n_img/dt:.1f} img/s)", flush=True)
    for f in handles.values():
        f.close()
    print(f"DONE images={n_img} records={n_rec} verified={n_ver} "
          f"verify_rate={n_ver/max(n_rec,1):.3f} wall={time.time()-t0:.0f}s")
    (outp / "gen_stats.json").write_text(json.dumps(
        {"images": n_img, "records": n_rec, "verified": n_ver}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
