#!/usr/bin/env python3
"""Record remaining likely-failed cases after an interrupted swift infer run."""
import argparse
import json
import os


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    inputs = read_jsonl(args.input)
    gts = read_jsonl(args.gt)
    done = count_lines(args.result)
    log_tail = ""
    if os.path.exists(args.log):
        with open(args.log, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 16000))
            log_tail = f.read().decode("utf-8", errors="replace")
    with open(args.out, "w", encoding="utf-8") as f:
        for local_i in range(done, min(len(inputs), len(gts))):
            inp = inputs[local_i]
            gt = gts[local_i]
            f.write(json.dumps({
                "local_index": local_i,
                "id": gt.get("id"),
                "task_family": gt.get("task_family"),
                "images": inp.get("images", []),
                "text_len": sum(len(str(m.get("content", ""))) for m in inp.get("messages", [])),
                "log_tail": log_tail if local_i == done else "",
            }, ensure_ascii=False) + "\n")
    print(f"Wrote failed/remaining cases from local index {done}: {args.out}")


if __name__ == "__main__":
    main()
