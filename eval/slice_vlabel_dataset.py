#!/usr/bin/env python3
"""Slice aligned vlabel input/GT jsonl files by [start, end)."""
import argparse
import json


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out-input", required=True)
    ap.add_argument("--out-gt", required=True)
    ap.add_argument("--trace-out", required=True)
    args = ap.parse_args()
    if args.start < 0 or args.end <= args.start:
        raise SystemExit("invalid range")

    inputs = read_jsonl(args.input)
    gts = read_jsonl(args.gt)
    end = min(args.end, len(inputs), len(gts))
    rows = list(range(args.start, end))

    with open(args.out_input, "w", encoding="utf-8") as fi, \
         open(args.out_gt, "w", encoding="utf-8") as fg, \
         open(args.trace_out, "w", encoding="utf-8") as ft:
        for i in rows:
            inp = inputs[i]
            gt = gts[i]
            fi.write(json.dumps(inp, ensure_ascii=False) + "\n")
            gt = dict(gt)
            gt.setdefault("original_index", i)
            fg.write(json.dumps(gt, ensure_ascii=False) + "\n")
            ft.write(json.dumps({
                "index": i,
                "id": gt.get("id"),
                "task_family": gt.get("task_family"),
                "images": inp.get("images", []),
                "text_len": sum(len(str(m.get("content", ""))) for m in inp.get("messages", [])),
            }, ensure_ascii=False) + "\n")
    print(f"Sliced {len(rows)} rows [{args.start}, {end})")
    print(f"input -> {args.out_input}")
    print(f"gt    -> {args.out_gt}")


if __name__ == "__main__":
    main()
