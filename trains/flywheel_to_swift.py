#!/usr/bin/env python3
"""Convert clean-v1 verified flywheel records into ms-swift SFT jsonl.

Emits {"messages":[{role:user, content:<img>+question},{role:assistant,
content:<json answer>}], "images":[abs_path]} per record. Also writes
per-layer slices (G/S/A) for the per-dimension teacher runs (B1).

Dedup by (image_path, normalized-answer) so template variants collapse.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

LAYER = {"G_grounding": "G", "S_spatial": "S", "A_identity_attr": "A",
         "A_obj_state": "A", "A_action_precondition": "A", "A_robot_state": "A",
         "A_action_interface": "A"}

QTMPL = {
    "G_grounding": "Detect the main manipulable object and give its bounding box (thousandth scale).",
    "S_spatial": "Describe one spatial relation between two objects in the image.",
    "A_identity_attr": "Identify the main object and its attributes (color, material, shape, size).",
    "A_obj_state": "Report the physical state of the main object.",
    "A_action_precondition": "Can the main object be picked right now? Explain any blockers.",
    "A_robot_state": "Describe the robot end-effector state.",
    "A_action_interface": "Give a 2D grasp point and jaw axis for manipulating the main object.",
}


def answer_json(rec: dict) -> str:
    drop = {"_verification", "_scan_count", "_cross_check", "image_path",
            "dimension", "source"}
    return json.dumps({k: v for k, v in rec.items() if k not in drop}, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", default="data/flywheel_v1/verified/*.jsonl")
    ap.add_argument("--out", default="trains/gsa_sft_clean_v1.jsonl")
    ap.add_argument("--min-scan", type=int, default=1, help="min Verified@k to include")
    args = ap.parse_args()

    seen = set()
    by_layer = defaultdict(list)
    n_in = n_out = 0
    for jf in glob.glob(args.in_glob):
        dim = Path(jf).stem
        for ln in Path(jf).read_text(errors="ignore").splitlines():
            if not ln.strip():
                continue
            n_in += 1
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            if rec.get("_scan_count", 1) < args.min_scan:
                continue
            img = rec.get("image_path")
            if not img or not os.path.exists(img):
                continue
            ans = answer_json(rec)
            key = hashlib.md5(f"{img}|{ans}".encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            samp = {"messages": [
                {"role": "user", "content": "<image>" + QTMPL.get(dim, "Describe the scene.")},
                {"role": "assistant", "content": ans}], "images": [img]}
            by_layer[LAYER.get(dim, "A")].append(samp)
            n_out += 1

    out = Path(args.out)
    with out.open("w") as f:
        for L in by_layer.values():
            for s in L:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    for L, rows in by_layer.items():
        lp = out.with_name(out.stem + f"_{L}.jsonl")
        with lp.open("w") as f:
            for s in rows:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  layer {L}: {len(rows)} -> {lp.name}")
    print(f"in={n_in} out={n_out} dedup_removed={n_in - n_out} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
