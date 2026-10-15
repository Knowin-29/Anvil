#!/usr/bin/env python3
"""Build a scene-labelled embodied image index for flywheel generation.

Pools every genuinely embodied image directory available locally (kitchen,
object-state, navigation, household, tabletop, BridgeData V2 frames, plus the
robopoint/refspatial image mirrors) into one index with a scene tag per line:

    <abs_path>\t<scene>

Scenes: embodied_home, kitchen, exploration, tabletop, object_state, ego.
Interleaves sources round-robin so a long generation run stays balanced even if
it is stopped early.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

# (root, scene, max_images) — caps keep any one pool from dominating.
POOLS = [
    ("/path/to/datasets/dataflywheel/input/kitchen", "kitchen", 44000),
    ("/path/to/datasets/dataflywheel/input/object_state", "object_state", 30000),
    ("/path/to/datasets/dataflywheel/input/navigation", "exploration", 26000),
    ("/path/to/datasets/dataflywheel/input/tabletop", "tabletop", 4000),
    ("/path/to/datasets/dataflywheel/input/household", "embodied_home", 4000),
    ("/path/to/datasets/bridge_raw_work/extracted/raw", "embodied_home", 31000),
    ("/path/to/datasets/datasets/transferred_datasets/datasets_train/robopoint-data/images",
     "tabletop", 60000),
    ("/path/to/datasets/datasets/transferred_datasets/datasets_train/refspatial",
     "exploration", 60000),
]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def scan(root: str, cap: int, stride: int = 1) -> list[str]:
    p = Path(root)
    if not p.is_dir():
        return []
    out = []
    for f in p.rglob("*"):
        if f.suffix.lower() in EXTS:
            out.append(str(f))
            if len(out) >= cap * max(stride, 1):
                break
    if stride > 1:
        out = out[::stride]
    return out[:cap]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/data/embodied_images.index")
    ap.add_argument("--stride", type=int, default=1,
                    help="subsample within a pool to cut trajectory redundancy")
    args = ap.parse_args()

    pools = []
    stats = Counter()
    for root, scene, cap in POOLS:
        # BridgeData frames are trajectory-dense -> stride them harder
        st = 3 if "bridge" in root else args.stride
        imgs = scan(root, cap, st)
        if imgs:
            pools.append([(i, scene) for i in imgs])
            stats[scene] += len(imgs)
            print(f"  {len(imgs):>7,}  {scene:<14} {root}", flush=True)

    # round-robin interleave so any prefix of the file is scene-balanced
    merged = []
    random.seed(0)
    for pool in pools:
        random.shuffle(pool)
    idx = 0
    while any(idx < len(p) for p in pools):
        for p in pools:
            if idx < len(p):
                merged.append(p[idx])
        idx += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"{p}\t{s}" for p, s in merged) + "\n")
    Path(str(out) + ".stats.json").write_text(json.dumps(
        {"total": len(merged), "by_scene": dict(stats.most_common())}, indent=2))
    print(f"\ntotal {len(merged):,} images -> {out}")
    print(json.dumps(dict(stats.most_common()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
