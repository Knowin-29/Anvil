#!/usr/bin/env python3
"""Index the recovered BridgeData V2 raw frames as a clean image source.

The old flywheel referenced bridge frames as flattened `episode_NNNNNN.png` under
`backup_420/bridgedatav2/` (now deleted). The recovered raw tree keeps the
original trajectory structure (`.../trajN/images0/im_K.jpg`), so the old
episode->frame mapping is unrecoverable. We therefore treat the raw frames as a
fresh image pool for clean-v1 flywheel generation (BridgeData V2 is public,
CC-BY-4.0).

Writes a newline-delimited index of absolute image paths + a small stats json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path(os.environ.get("BRIDGE_RAW_ROOT", "/path/to/datasets/bridge_raw_work/extracted/raw"))
OUT_INDEX = REPO_ROOT / "configs/data/bridge_images.index"
OUT_STATS = REPO_ROOT / "configs/data/bridge_images.stats.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1,
                    help="keep 1 of every N frames per trajectory to reduce redundancy")
    args = ap.parse_args()

    imgs = sorted(str(p) for p in RAW_ROOT.rglob("*.jpg"))
    imgs += sorted(str(p) for p in RAW_ROOT.rglob("*.png"))
    imgs = sorted(imgs)
    if args.stride > 1:
        imgs = imgs[:: args.stride]

    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_INDEX.write_text("\n".join(imgs) + "\n")
    scenes = {}
    for p in imgs:
        # .../raw/{rss|icra|...}/{scene}/...
        parts = p.split("/raw/", 1)
        key = parts[1].split("/")[0] if len(parts) > 1 else "other"
        scenes[key] = scenes.get(key, 0) + 1
    OUT_STATS.write_text(json.dumps({
        "source": "bridge_data_v2_raw",
        "license": "open (BridgeData V2, CC-BY-4.0)",
        "raw_root": str(RAW_ROOT),
        "stride": args.stride,
        "n_images": len(imgs),
        "top_scenes": dict(sorted(scenes.items(), key=lambda x: -x[1])[:15]),
    }, indent=2))
    print(f"indexed {len(imgs)} bridge images (stride={args.stride}) -> {OUT_INDEX}")
    print(json.dumps(dict(sorted(scenes.items(), key=lambda x: -x[1])[:10]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
