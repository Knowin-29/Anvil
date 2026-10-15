#!/usr/bin/env python3
"""Pre-scan vlabel input/GT files before expensive model inference."""
import argparse
import json
import os
from collections import Counter, defaultdict

try:
    from PIL import Image
except Exception:  # pragma: no cover - environment guard
    Image = None


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, json.loads(line)


def text_len(rec):
    total = 0
    for msg in rec.get("messages") or []:
        total += len(str(msg.get("content", "")))
    return total


def inspect_image(path):
    info = {"path": path, "exists": os.path.isfile(path)}
    if not info["exists"]:
        info["reason"] = "missing_image"
        return info
    if Image is None:
        return info
    try:
        with Image.open(path) as im:
            info.update({"width": im.width, "height": im.height, "mode": im.mode})
            pixels = im.width * im.height
            if pixels > 16_000_000:
                info["reason"] = "huge_image"
    except Exception as exc:
        info.update({"reason": "bad_image", "error": f"{type(exc).__name__}: {exc}"})
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--long-text-threshold", type=int, default=12000)
    ap.add_argument("--focus-start", type=int, default=180)
    ap.add_argument("--focus-end", type=int, default=260)
    args = ap.parse_args()

    inputs = list(iter_jsonl(args.input))
    gts = dict(iter_jsonl(args.gt))
    counters = Counter()
    by_family = Counter()
    text_lengths = []
    skip_candidates = []
    focus = []

    for i, rec in inputs:
        gt = gts.get(i, {})
        family = gt.get("task_family", "missing_gt")
        by_family[family] += 1
        imgs = rec.get("images") or []
        tl = text_len(rec)
        text_lengths.append(tl)
        reasons = []

        if len(imgs) == 0:
            counters["no_image"] += 1
            reasons.append("no_image")
        if len(imgs) > 1:
            counters["multi_image"] += 1
            reasons.append("multi_image")
        if tl > args.long_text_threshold:
            counters["long_text"] += 1
            reasons.append("long_text")

        image_infos = [inspect_image(p) for p in imgs]
        for info in image_infos:
            reason = info.get("reason")
            if reason:
                counters[reason] += 1
                reasons.append(reason)

        item = {
            "index": i,
            "id": gt.get("id"),
            "task_family": family,
            "text_len": tl,
            "num_images": len(imgs),
            "images": image_infos,
            "reasons": sorted(set(reasons)),
        }
        if reasons:
            skip_candidates.append(item)
        if args.focus_start <= i < args.focus_end:
            focus.append(item)

    sorted_lengths = sorted(text_lengths)
    n = len(sorted_lengths)
    pct = {}
    for name, q in [("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]:
        pct[name] = sorted_lengths[min(n - 1, int(n * q))] if n else 0

    report = {
        "input": args.input,
        "gt": args.gt,
        "total": len(inputs),
        "gt_total": len(gts),
        "matched_rows": min(len(inputs), len(gts)),
        "counters": dict(counters),
        "by_family": dict(by_family),
        "text_len": {
            "max": max(text_lengths) if text_lengths else 0,
            **pct,
        },
        "skip_candidate_count": len(skip_candidates),
        "skip_candidates": skip_candidates[:200],
        "focus_range": {"start": args.focus_start, "end": args.focus_end, "items": focus},
        "env_note": {
            "IMAGE_MAX_TOKEN_NUM": os.environ.get("IMAGE_MAX_TOKEN_NUM"),
            "VIDEO_MAX_TOKEN_NUM": os.environ.get("VIDEO_MAX_TOKEN_NUM"),
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote audit report: {args.out}")
    print(f"total={report['total']} skip_candidates={len(skip_candidates)} counters={dict(counters)}")


if __name__ == "__main__":
    main()
