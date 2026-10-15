#!/usr/bin/env python3
"""Validate eval output row counts before scoring/reporting."""
import argparse
import json
import os


def count_jsonl(path):
    if not os.path.exists(path):
        return 0, 0
    n = bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            try:
                json.loads(line)
            except Exception:
                bad += 1
    return n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    input_n, input_bad = count_jsonl(args.input)
    gt_n, gt_bad = count_jsonl(args.gt)
    result_n, result_bad = count_jsonl(args.result)
    expected = min(input_n, gt_n)
    ok = input_bad == gt_bad == result_bad == 0 and result_n == expected
    report = {
        "tag": args.tag,
        "input_n": input_n,
        "gt_n": gt_n,
        "expected_result_n": expected,
        "result_n": result_n,
        "bad_json": {"input": input_bad, "gt": gt_bad, "result": result_bad},
        "complete": ok,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit("incomplete eval output; refusing to mark run complete")


if __name__ == "__main__":
    main()
