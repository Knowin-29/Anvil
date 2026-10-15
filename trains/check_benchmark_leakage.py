#!/usr/bin/env python3
"""Train/val/test <-> public-benchmark leakage checker for GSA-Forge.

Detects overlap between the training pool and evaluation benchmarks along three axes:
  1. Image content hash (perceptual-agnostic exact-content dedup via file bytes / pixel hash)
  2. Image basename (catches same-image-different-path)
  3. (Optional) prompt+response text hash (annotation-level dup)

This addresses the requirement that train/val/test must be deduplicated and that public
benchmarks must not overlap with the 90+ training sources (image/scene/annotation leakage).

Usage:
    python trains/check_benchmark_leakage.py \
        --train trains/gsa_sft_train.jsonl \
        --bench eval/replay_100.jsonl eval/replay_strat.jsonl \
        --out trains/leakage_report.json \
        [--hash content|basename|both] [--max-train N] [--image-roots /a:/b]

Output: JSON report with overlap counts, offending samples, and a clean/leaky verdict.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc


def resolve_image(img, roots):
    """Return the first existing absolute path for an image ref, or None."""
    if not img:
        return None
    if os.path.isabs(img) and os.path.exists(img):
        return img
    for r in roots:
        cand = os.path.join(r, img)
        if os.path.exists(cand):
            return cand
    return None


def file_content_hash(path, chunk=1 << 20):
    """SHA-256 of encoded file bytes: exact file identity, not perceptual identity."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def record_images(rec):
    imgs = rec.get("images", [])
    if isinstance(imgs, str):
        return [imgs]
    if isinstance(imgs, list):
        return [i for i in imgs if isinstance(i, str)]
    return []


def record_text_hash(rec):
    parts = []
    messages = rec.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
    if not parts:
        for key in ("prompt", "query", "question", "response", "answer", "solution"):
            value = rec.get(key)
            if isinstance(value, str):
                parts.append(value)
    normalized = " ".join(" ".join(parts).lower().split())
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_bench_index(bench_paths, roots, use_content):
    """Index benchmark records and count every unresolved content comparison."""
    by_basename = defaultdict(list)   # basename -> [(bench_file, ref)]
    by_content = defaultdict(list)    # content_hash -> [(bench_file, ref)]
    by_text = defaultdict(list)       # normalized text hash -> [bench_file]
    n_imgs = 0
    unresolved = 0
    for bp in bench_paths:
        for rec in iter_jsonl(bp):
            th = record_text_hash(rec)
            if th:
                by_text[th].append(bp)
            for ref in record_images(rec):
                n_imgs += 1
                base = os.path.basename(ref)
                by_basename[base].append((bp, ref))
                if use_content:
                    ap = resolve_image(ref, roots)
                    if ap:
                        ch = file_content_hash(ap)
                        if ch:
                            by_content[ch].append((bp, ref))
                        else:
                            unresolved += 1
                    else:
                        unresolved += 1
    return by_basename, by_content, by_text, n_imgs, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="training jsonl file(s)")
    ap.add_argument("--bench", nargs="+", required=True, help="benchmark jsonl file(s)")
    ap.add_argument("--out", default="trains/leakage_report.json")
    ap.add_argument("--hash", choices=["content", "basename", "both"], default="both")
    ap.add_argument("--max-train", type=int, default=0, help="cap train records scanned (0=all)")
    ap.add_argument("--image-roots", default=os.environ.get("IMAGE_ROOTS", "/path/to/workspace/foundation"))
    args = ap.parse_args()

    roots = [r for r in args.image_roots.split(":") if r]
    use_content = args.hash in ("content", "both")
    use_basename = args.hash in ("basename", "both")

    print(f"[leakage] indexing {len(args.bench)} benchmark file(s)...", flush=True)
    b_base, b_content, b_text, n_bench_imgs, bench_unresolved = build_bench_index(
        args.bench, roots, use_content)
    print(f"[leakage] benchmark images: {n_bench_imgs} "
          f"(unique basenames={len(b_base)}, content-hashed={len(b_content)})", flush=True)

    report = {
        "train_files": args.train,
        "bench_files": args.bench,
        "hash_mode": args.hash,
        "bench_images": n_bench_imgs,
        "bench_unresolved_images": bench_unresolved,
        "train_records": 0,
        "train_images": 0,
        "train_unresolved_images": 0,
        "basename_overlap": 0,
        "content_overlap": 0,
        "text_overlap": 0,
        "basename_examples": [],
        "content_examples": [],
        "text_examples": [],
    }
    seen_base_hits = set()
    seen_content_hits = set()
    seen_text_hits = set()

    stopped_early = False
    for tf in args.train:
        if stopped_early:
            break
        for rec in iter_jsonl(tf):
            if args.max_train and report["train_records"] >= args.max_train:
                stopped_early = True
                break
            report["train_records"] += 1
            th = record_text_hash(rec)
            if th in b_text and th not in seen_text_hits:
                seen_text_hits.add(th)
                report["text_overlap"] += 1
                if len(report["text_examples"]) < 25:
                    report["text_examples"].append(
                        {"train_file": tf, "bench": b_text[th][0]})
            for ref in record_images(rec):
                report["train_images"] += 1
                base = os.path.basename(ref)
                if use_basename and base in b_base and base not in seen_base_hits:
                    seen_base_hits.add(base)
                    report["basename_overlap"] += 1
                    if len(report["basename_examples"]) < 25:
                        report["basename_examples"].append(
                            {"train_ref": ref, "bench": b_base[base][0]})
                if use_content:
                    ap_ = resolve_image(ref, roots)
                    if ap_:
                        ch = file_content_hash(ap_)
                        if ch is None:
                            report["train_unresolved_images"] += 1
                        elif ch in b_content and ch not in seen_content_hits:
                            seen_content_hits.add(ch)
                            report["content_overlap"] += 1
                            if len(report["content_examples"]) < 25:
                                report["content_examples"].append(
                                    {"train_ref": ref, "bench": b_content[ch][0]})
                    else:
                        report["train_unresolved_images"] += 1

    hard_overlap = report["content_overlap"] > 0 or report["text_overlap"] > 0
    incomplete = stopped_early or (
        use_content and (bench_unresolved > 0 or report["train_unresolved_images"] > 0))
    if hard_overlap:
        report["verdict"] = "LEAKAGE_DETECTED"
    elif incomplete:
        report["verdict"] = "INCOMPLETE"
    else:
        report["verdict"] = "CLEAN"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"  train records={report['train_records']:,} images={report['train_images']:,}")
    print(f"  basename overlap: {report['basename_overlap']}")
    print(f"  content  overlap: {report['content_overlap']}")
    print(f"  text     overlap: {report['text_overlap']}")
    print(f"  unresolved images: train={report['train_unresolved_images']} "
          f"bench={report['bench_unresolved_images']}")
    print(f"  VERDICT: {report['verdict']}")
    print(f"  -> {args.out}")
    print("=" * 60)
    sys.exit({"CLEAN": 0, "LEAKAGE_DETECTED": 3, "INCOMPLETE": 4}[report["verdict"]])


if __name__ == "__main__":
    main()
