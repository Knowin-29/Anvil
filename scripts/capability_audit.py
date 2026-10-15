#!/usr/bin/env python3
"""Capability Coverage 审计脚本。

输入: data/output/{run_id}/manifest.jsonl
输出: capability_coverage.json + 终端报告

用法:
    python scripts/capability_audit.py --run-dir data/output/v3_run_xxx
    python scripts/capability_audit.py --latest   # 自动找最新 run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List


def find_latest_run(output_root: str = "data/output") -> str:
    """找到最新的 run 目录。"""
    runs = sorted(glob.glob(os.path.join(output_root, "v3_run_*")))
    if not runs:
        print(f"No v3 runs found in {output_root}")
        sys.exit(1)
    return runs[-1]


def audit_run(run_dir: str) -> Dict[str, Any]:
    """审计单个 run 的能力覆盖。"""
    annotations_dir = os.path.join(run_dir, "annotations")
    if not os.path.isdir(annotations_dir):
        print(f"No annotations dir: {annotations_dir}")
        return {}

    per_capability: Dict[str, Dict[str, Any]] = {}

    for cap_dir in sorted(os.listdir(annotations_dir)):
        cap_path = os.path.join(annotations_dir, cap_dir)
        if not os.path.isdir(cap_path):
            continue

        count = 0
        scores: List[float] = []
        sub_tags: Dict[str, int] = defaultdict(int)

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

                    count += 1
                    if item.get("verifier_score") is not None:
                        scores.append(item["verifier_score"])
                    sub_tag = item.get("sub_tag", "unknown")
                    sub_tags[sub_tag] += 1

        pass_rate = sum(1 for s in scores if s >= 0.7) / len(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        per_capability[cap_dir] = {
            "count": count,
            "pass_rate": round(pass_rate, 3),
            "avg_score": round(avg_score, 3),
            "score_dist": {
                "high_0.8+": sum(1 for s in scores if s >= 0.8),
                "mid_0.5-0.8": sum(1 for s in scores if 0.5 <= s < 0.8),
                "low_<0.5": sum(1 for s in scores if s < 0.5),
            } if scores else {},
            "sub_tags": dict(sub_tags),
        }

    # 总统计
    total_count = sum(v["count"] for v in per_capability.values())
    result = {
        "run_dir": run_dir,
        "total_annotations": total_count,
        "num_capabilities": len(per_capability),
        "per_capability": per_capability,
    }

    return result


def print_report(result: Dict[str, Any]) -> None:
    """打印终端报告。"""
    print("=" * 60)
    print(f"  Capability Coverage Report")
    print(f"  Run: {result.get('run_dir', 'N/A')}")
    print(f"  Total annotations: {result.get('total_annotations', 0)}")
    print("=" * 60)
    print()
    print(f"{'Tag':<6} {'Count':<8} {'PassRate':<10} {'AvgScore':<10} {'SubTags'}")
    print("-" * 60)

    per_cap = result.get("per_capability", {})
    for tag in ["G", "S", "A", "R", "P", "F", "H", "E", "Aff"]:
        if tag in per_cap:
            info = per_cap[tag]
            sub_tags_str = ", ".join(f"{k}:{v}" for k, v in list(info["sub_tags"].items())[:3])
            print(f"{tag:<6} {info['count']:<8} {info['pass_rate']:<10.3f} {info['avg_score']:<10.3f} {sub_tags_str}")
        else:
            print(f"{tag:<6} {'—':<8} {'—':<10} {'—':<10}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Capability Coverage 审计")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--latest", action="store_true", help="使用最新 run")
    parser.add_argument("--output-root", type=str, default="data/output")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run(args.output_root)

    print(f"Auditing: {run_dir}")
    result = audit_run(run_dir)

    if not result:
        return

    # 打印报告
    print_report(result)

    # 保存 JSON
    output_path = os.path.join(run_dir, "capability_coverage.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
