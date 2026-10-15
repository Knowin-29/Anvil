#!/usr/bin/env python3
"""Create aligned vlabel input/GT subsets by task family regex."""
import argparse
import json
import re


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--family-regex", required=True)
    ap.add_argument("--out-input", required=True)
    ap.add_argument("--out-gt", required=True)
    ap.add_argument("--trace-out", required=True)
    args = ap.parse_args()

    inputs = read_jsonl(args.input)
    gts = read_jsonl(args.gt)
    pat = re.compile(args.family_regex)
    kept = 0
    with open(args.out_input, "w", encoding="utf-8") as fi, \
         open(args.out_gt, "w", encoding="utf-8") as fg, \
         open(args.trace_out, "w", encoding="utf-8") as ft:
        for i, (inp, gt0) in enumerate(zip(inputs, gts)):
            if not pat.search(str(gt0.get("task_family", ""))):
                continue
            gt = dict(gt0)
            gt.setdefault("original_index", i)
            fi.write(json.dumps(inp, ensure_ascii=False) + "\n")
            fg.write(json.dumps(gt, ensure_ascii=False) + "\n")
            ft.write(json.dumps({
                "index": i,
                "id": gt.get("id"),
                "task_family": gt.get("task_family"),
                "gt_bbox_present": gt.get("gt_bbox") is not None,
                "images": inp.get("images", []),
            }, ensure_ascii=False) + "\n")
            kept += 1
    print(f"subset kept={kept} family_regex={args.family_regex}")
    print(f"input -> {args.out_input}")
    print(f"gt    -> {args.out_gt}")


if __name__ == "__main__":
    main()
