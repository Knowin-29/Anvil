#!/usr/bin/env python3
"""构建 Rex v3 Benchmark 切片。

从 export 抽 10% 作 holdout，按 capability_tag 分片。
每片 50~100 条 → benchmarks/rex_v3_bench/<capability>.jsonl

用法:
    python scripts/build_benchmark.py --run-dir data/output/v3_run_xxx
    python scripts/build_benchmark.py --latest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List


def find_latest_run(output_root: str = "data/output") -> str:
    runs = sorted(glob.glob(os.path.join(output_root, "v3_run_*")))
    if not runs:
        print(f"No v3 runs found in {output_root}")
        sys.exit(1)
    return runs[-1]


def build_benchmark(
    run_dir: str,
    output_dir: str = "benchmarks/rex_v3_bench",
    holdout_ratio: float = 0.1,
    min_per_cap: int = 50,
    max_per_cap: int = 100,
    seed: int = 42,
) -> Dict[str, int]:
    """构建 benchmark 切片。"""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    annotations_dir = os.path.join(run_dir, "annotations")
    if not os.path.isdir(annotations_dir):
        print(f"No annotations dir: {annotations_dir}")
        return {}

    # 按 capability 加载
    pools: Dict[str, List[Dict]] = defaultdict(list)
    for cap_dir in os.listdir(annotations_dir):
        cap_path = os.path.join(annotations_dir, cap_dir)
        if not os.path.isdir(cap_path):
            continue
        for jsonl_file in os.listdir(cap_path):
            if not jsonl_file.endswith(".jsonl"):
                continue
            filepath = os.path.join(cap_path, jsonl_file)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # 仅取高质量样本
                    score = item.get("verifier_score")
                    if score is not None and score < 0.7:
                        continue
                    pools[cap_dir].append(item)

    # 按 capability 切片
    result: Dict[str, int] = {}
    for cap, items in pools.items():
        # 抽 holdout_ratio
        n_holdout = max(min_per_cap, int(len(items) * holdout_ratio))
        n_holdout = min(n_holdout, max_per_cap, len(items))

        selected = random.sample(items, n_holdout)

        output_path = os.path.join(output_dir, f"{cap}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in selected:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        result[cap] = n_holdout
        print(f"  {cap}: {n_holdout} samples → {output_path}")

    # 写 meta
    meta = {
        "source_run": run_dir,
        "holdout_ratio": holdout_ratio,
        "per_capability": result,
        "total": sum(result.values()),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBenchmark total: {sum(result.values())} samples across {len(result)} capabilities")
    return result


def main():
    parser = argparse.ArgumentParser(description="构建 Rex v3 Benchmark 切片")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output-dir", type=str, default="benchmarks/rex_v3_bench")
    parser.add_argument("--output-root", type=str, default="data/output")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run(args.output_root)

    print(f"Building benchmark from: {run_dir}")
    build_benchmark(run_dir, args.output_dir)


if __name__ == "__main__":
    main()
