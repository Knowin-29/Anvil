#!/usr/bin/env python3
"""Drop only non-runnable vlabel rows before swift infer.

Long text is kept because ms-swift can truncate with --truncation_strategy left.
Rows with missing/bad/huge images are skipped explicitly and written to a report.
"""
import argparse
import json
import os

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def image_reason(path, max_pixels):
    if not os.path.isfile(path):
        return "missing_image"
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            if im.width * im.height > max_pixels:
                return "huge_image"
    except Exception:
        return "bad_image"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out-input", required=True)
    ap.add_argument("--out-gt", required=True)
    ap.add_argument("--skip-report", required=True)
    ap.add_argument("--max-pixels", type=int, default=16_000_000)
    args = ap.parse_args()

    inputs = read_jsonl(args.input)
    gts = read_jsonl(args.gt)
    skipped = []
    kept = 0
    with open(args.out_input, "w", encoding="utf-8") as fi, \
         open(args.out_gt, "w", encoding="utf-8") as fg:
        for local_i, (inp, gt0) in enumerate(zip(inputs, gts)):
            gt = dict(gt0)
            gt.setdefault("original_index", local_i)
            reasons = []
            for img in inp.get("images") or []:
                reason = image_reason(img, args.max_pixels)
                if reason:
                    reasons.append(reason)
            if not inp.get("images"):
                reasons.append("no_image")
            if reasons:
                skipped.append({
                    "local_index": local_i,
                    "original_index": gt.get("original_index", local_i),
                    "id": gt.get("id"),
                    "task_family": gt.get("task_family"),
                    "images": inp.get("images", []),
                    "reasons": sorted(set(reasons)),
                })
                continue
            fi.write(json.dumps(inp, ensure_ascii=False) + "\n")
            fg.write(json.dumps(gt, ensure_ascii=False) + "\n")
            kept += 1

    report = {
        "input_n": len(inputs),
        "gt_n": len(gts),
        "kept": kept,
        "skipped": len(skipped),
        "skip_rows": skipped,
    }
    with open(args.skip_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Runnable rows: kept={kept}, skipped={len(skipped)}")
    print(f"skip_report -> {args.skip_report}")


if __name__ == "__main__":
    main()
