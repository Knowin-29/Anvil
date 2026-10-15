#!/usr/bin/env python3
"""Build a LOCAL grounding eval set from flywheel-verified G_grounding data.

The date-sliced log eval (replay_strat) references aliyun-only images and cannot run here.
This builds a runnable local eval from the flywheel's verified G_grounding annotations whose
images exist on disk, with GT bbox, deduplicated against the training pool by image path.

Outputs:
  eval/local_grounding_eval.jsonl     (ms-swift infer format: messages + images)
  eval/local_grounding_gt.jsonl       (GT bbox aligned by line)

Usage: python eval/build_local_eval.py --n 200
"""
import argparse
import glob
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FLYWHEEL = os.environ.get("GSA_GROUNDING_SOURCE", "/path/to/Anvil/data/output/gsa_flywheel_exp/verified/G_grounding")
TRAIN = str(REPO_ROOT / "trains/gsa_sft_train.jsonl")
SYS = ("You are an embodied perception assistant specialized in indoor scene understanding. "
       "Detect objects and respond with a JSON object containing the bounding box.")


def load_train_images():
    imgs = set()
    if os.path.exists(TRAIN):
        with open(TRAIN) as f:
            for l in f:
                try:
                    d = json.loads(l)
                except Exception:
                    continue
                for i in d.get("images", []) or []:
                    imgs.add(os.path.basename(i))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="eval/local_grounding_eval.jsonl")
    ap.add_argument("--gt", default="eval/local_grounding_gt.jsonl")
    args = ap.parse_args()

    train_imgs = load_train_images()
    picked = 0
    seen = set()
    with open(args.out, "w") as fout, open(args.gt, "w") as fgt:
        for fp in sorted(glob.glob(os.path.join(FLYWHEEL, "*.jsonl"))):
            with open(fp) as f:
                for l in f:
                    try:
                        d = json.loads(l)
                    except Exception:
                        continue
                    img = d.get("image_path", "")
                    bbox = d.get("bbox")
                    obj = d.get("object_class") or d.get("object") or "object"
                    if not img or not os.path.exists(img) or not bbox:
                        continue
                    base = os.path.basename(img)
                    # dedup: skip images that appear in the training set or already picked
                    if base in train_imgs or base in seen:
                        continue
                    seen.add(base)
                    rec = {
                        "messages": [
                            {"role": "system", "content": SYS},
                            {"role": "user", "content": f"<image>\nLocate the '{obj}' in this image. "
                                                        f"Provide bounding box as {{\"bbox\": [x1,y1,x2,y2]}} in [0,999]."},
                        ],
                        "images": [img],
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fgt.write(json.dumps({"object": obj, "bbox": bbox, "image": img}, ensure_ascii=False) + "\n")
                    picked += 1
                    if picked >= args.n:
                        break
            if picked >= args.n:
                break
    print(f"built {picked} local eval samples (dedup'd vs training)")
    print(f"  -> {args.out}\n  -> {args.gt}")


if __name__ == "__main__":
    main()
