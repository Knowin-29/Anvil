#!/usr/bin/env python3
"""Filter ms-swift jsonl lines whose image(s) are unreachable on disk.

Some sources (notably bridgedatav2 under a now-deleted backup_420 dir, exposed via dead
symlinks) reference images that no longer exist locally. GRPO/GKD dataloaders fail hard on
missing images (SFT only warns). This tool drops lines with any unreachable image, writing a
cleaned file and a report. Non-destructive: writes <name>.clean.jsonl unless --inplace.

Usage:
    python trains/filter_missing_images.py --input trains/gsa_sft_train.jsonl
    python trains/filter_missing_images.py --input trains/gsa_grpo_train.jsonl --inplace
"""
import argparse
import json
import os
from pathlib import Path


def reachable(img: str, roots) -> bool:
    if not img:
        return False
    if img.startswith(("http://", "https://")):
        return True
    if os.path.exists(img):
        return True
    for r in roots:
        if r and os.path.exists(os.path.join(r, img)):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--image-roots", default=os.environ.get("IMAGE_ROOTS", "/path/to/workspace/foundation"))
    args = ap.parse_args()

    roots = [r for r in args.image_roots.split(":") if r]
    src = Path(args.input)
    out = src if args.inplace else src.with_suffix(".clean.jsonl")
    tmp = src.with_suffix(src.suffix + ".filt.tmp")

    kept = dropped = 0
    drop_dirs = {}
    with open(src, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            imgs = rec.get("images", [])
            if isinstance(imgs, str):
                imgs = [imgs]
            ok = all(reachable(i, roots) for i in imgs) if imgs else True
            if ok:
                fout.write(line + "\n")
                kept += 1
            else:
                dropped += 1
                for i in imgs:
                    if not reachable(i, roots):
                        d = "/".join(i.split("/")[:6])
                        drop_dirs[d] = drop_dirs.get(d, 0) + 1

    os.replace(tmp, out)
    report = {
        "input": str(src), "output": str(out),
        "kept": kept, "dropped": dropped,
        "total": kept + dropped,
        "drop_rate": round(dropped / max(kept + dropped, 1), 4),
        "dropped_by_dir": dict(sorted(drop_dirs.items(), key=lambda kv: -kv[1])[:20]),
    }
    rp = src.with_suffix(".filter_report.json")
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
