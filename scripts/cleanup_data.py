#!/usr/bin/env python3
"""数据清理脚本 — 根据 Scanner 质量评分清除低质样本。

功能:
1. 读取 *.scores.jsonl sidecar 文件
2. 按阈值或 action=delete 标记筛选低质行
3. 生成清理后的 JSONL (原文件备份 → 覆写)
4. 输出清理报告

用法:
    python scripts/cleanup_data.py \
        --annotations-dir data/output/flywheel/v3_run_1778980547/annotations \
        --threshold 0.3 \
        --dry-run

    python scripts/cleanup_data.py \
        --annotations-dir data/output/flywheel/v3_run_1778980547/annotations \
        --threshold 0.4 \
        --execute
"""

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("cleanup")

SCORE_SUFFIX = ".scores.jsonl"


def load_delete_set(score_path: Path, threshold: float) -> Tuple[Set[int], Dict]:
    """加载需要删除的行索引集合。"""
    to_delete: Set[int] = set()
    stats = {"total": 0, "below_threshold": 0, "action_delete": 0}

    if not score_path.exists():
        return to_delete, stats

    with open(score_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["total"] += 1
            line_idx = obj.get("line_idx", -1)
            score = obj.get("ensemble_score", 1.0)
            action = obj.get("action", "keep")

            if score < threshold:
                to_delete.add(line_idx)
                stats["below_threshold"] += 1
            elif action == "delete":
                to_delete.add(line_idx)
                stats["action_delete"] += 1

    return to_delete, stats


def cleanup_file(
    jsonl_path: Path,
    to_delete: Set[int],
    execute: bool = False,
    backup: bool = True,
) -> Tuple[int, int]:
    """清理单个 JSONL 文件。返回 (保留数, 删除数)。"""
    kept = 0
    removed = 0
    clean_lines = []

    with open(jsonl_path, "r") as f:
        for idx, line in enumerate(f):
            if idx in to_delete:
                removed += 1
            else:
                clean_lines.append(line)
                kept += 1

    if execute and removed > 0:
        if backup:
            backup_path = jsonl_path.with_suffix(f".bak_{int(time.time())}.jsonl")
            shutil.copy2(jsonl_path, backup_path)
            logger.info(f"  Backup: {backup_path}")
        with open(jsonl_path, "w") as f:
            f.writelines(clean_lines)

    return kept, removed


def main():
    parser = argparse.ArgumentParser(description="根据 Scanner 评分清理低质数据")
    parser.add_argument("--annotations-dir", type=str, required=True, help="标注目录")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="低于此分数的标注将被删除 (default: 0.3)")
    parser.add_argument("--dry-run", action="store_true", help="只统计不执行删除")
    parser.add_argument("--execute", action="store_true", help="执行删除")
    parser.add_argument("--no-backup", action="store_true", help="不创建备份")
    parser.add_argument("--report", type=str, help="输出 JSON 报告路径")
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        logger.error("请指定 --dry-run 或 --execute")
        return

    ann_dir = Path(args.annotations_dir)
    if not ann_dir.exists():
        logger.error(f"目录不存在: {ann_dir}")
        return

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "threshold": args.threshold,
        "mode": "dry-run" if args.dry_run else "execute",
        "files": {},
        "summary": {"total_kept": 0, "total_removed": 0, "total_files": 0},
    }

    jsonl_files = sorted(ann_dir.rglob("*.jsonl"))
    jsonl_files = [f for f in jsonl_files if not f.name.endswith(SCORE_SUFFIX)
                   and ".bak_" not in f.name]

    for jsonl_path in jsonl_files:
        score_path = jsonl_path.with_suffix(SCORE_SUFFIX)
        to_delete, score_stats = load_delete_set(score_path, args.threshold)

        if not to_delete:
            continue

        kept, removed = cleanup_file(
            jsonl_path,
            to_delete,
            execute=args.execute,
            backup=not args.no_backup,
        )

        cap = jsonl_path.parent.name
        rel_path = str(jsonl_path.relative_to(ann_dir))
        logger.info(f"[{cap}] {rel_path}: kept={kept}, removed={removed} "
                    f"(scored={score_stats['total']})")

        report["files"][rel_path] = {
            "kept": kept,
            "removed": removed,
            "score_stats": score_stats,
        }
        report["summary"]["total_kept"] += kept
        report["summary"]["total_removed"] += removed
        report["summary"]["total_files"] += 1

    logger.info("=" * 60)
    logger.info(f"清理完成 — 模式: {'dry-run' if args.dry_run else 'execute'}")
    logger.info(f"  总保留: {report['summary']['total_kept']}")
    logger.info(f"  总删除: {report['summary']['total_removed']}")
    logger.info(f"  涉及文件: {report['summary']['total_files']}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"报告已保存: {args.report}")


if __name__ == "__main__":
    main()
