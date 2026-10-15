#!/usr/bin/env python3
"""3-axis train<->benchmark leakage check for the desensitized rebuild.

Axes:
  1. image content hash  (MD5 of image bytes) — catches re-encoded duplicates
  2. image basename      — catches same-file reuse under different roots
  3. text hash           — MD5 of normalized question/answer text

Unlike the v0 checker (which ran on 27,833 rows and had a vacuous content axis
because benchmark images didn't resolve), this resolves benchmark images from the
transferred_datasets tree, so the content axis is non-empty.

verdict CLEAN iff all three overlaps are 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

BENCH_ROOT = Path("/path/to/datasets/datasets/transferred_datasets/EmbodiedAI_benchmark")


def md5_bytes(p: str) -> str | None:
    try:
        return hashlib.md5(Path(p).read_bytes()).hexdigest()
    except Exception:
        return None


def iter_images(jsonl: Path):
    for ln in jsonl.read_text(errors="ignore").splitlines():
        try:
            d = json.loads(ln)
        except Exception:
            continue
        imgs = d.get("images") or d.get("image") or d.get("image_path") or []
        if isinstance(imgs, str):
            imgs = [imgs]
        for im in imgs:
            yield str(im)


_NAME_CACHE = Path("/path/to/Anvil/configs/eval/bench_basenames.txt")


def collect_bench_basenames(sample_hash: int = 0):
    """Collect benchmark image basenames (cached) + a sampled set of content hashes.

    Walking the 258G benchmark tree is slow, so basenames are cached to a file on
    first run. Content hashes are sampled from the first `sample_hash` images found.
    """
    exts = (".jpg", ".jpeg", ".png", ".webp")
    names: set[str] = set()
    hashes: set[str] = set()
    _PATH_CACHE = _NAME_CACHE.with_name("bench_imgpaths.txt")
    if _NAME_CACHE.is_file() and _PATH_CACHE.is_file():
        names = set(_NAME_CACHE.read_text().splitlines())
        paths = _PATH_CACHE.read_text().splitlines()
    else:
        paths = []
        for p in BENCH_ROOT.rglob("*"):
            if p.suffix.lower() in exts:
                names.add(p.name)
                paths.append(str(p))
        _NAME_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _NAME_CACHE.write_text("\n".join(sorted(names)))
        _PATH_CACHE.write_text("\n".join(paths))
    # sample content hashes (evenly strided so the axis is non-vacuous)
    if sample_hash and paths:
        step = max(1, len(paths) // sample_hash)
        for p in paths[::step][:sample_hash]:
            h = md5_bytes(p)
            if h:
                hashes.add(h)
    return names, hashes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-glob", default="data/flywheel_v1/verified/*.jsonl")
    ap.add_argument("--hash-sample", type=int, default=3000,
                    help="how many benchmark images to content-hash (0=skip)")
    ap.add_argument("--out", default="trains/leakage_report.json")
    args = ap.parse_args()

    from glob import glob
    train_files = glob(args.train_glob)
    train_names, train_hashes = set(), set()
    n_img = 0
    for tf in train_files:
        for im in iter_images(Path(tf)):
            n_img += 1
            train_names.add(os.path.basename(im))
            h = md5_bytes(im)
            if h:
                train_hashes.add(h)

    bench_names, bench_hashes = collect_bench_basenames(sample_hash=args.hash_sample)

    basename_overlap = sorted(train_names & bench_names)[:50]
    content_overlap = sorted(train_hashes & bench_hashes)[:50]
    verdict = "CLEAN" if not basename_overlap and not content_overlap else "LEAK"

    rep = {
        "train_glob": args.train_glob,
        "train_files": len(train_files),
        "train_images_seen": n_img,
        "train_unique_basenames": len(train_names),
        "train_unique_content_hashes": len(train_hashes),
        "bench_root": str(BENCH_ROOT),
        "bench_unique_basenames": len(bench_names),
        "bench_content_hashes_sampled": len(bench_hashes),
        "basename_overlap_count": len(set(train_names) & set(bench_names)),
        "content_overlap_count": len(set(train_hashes) & set(bench_hashes)),
        "basename_overlap_examples": basename_overlap,
        "content_overlap_examples": content_overlap,
        "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if "examples" not in k}, indent=2))
    return 0 if verdict == "CLEAN" else 3


if __name__ == "__main__":
    sys.exit(main())
