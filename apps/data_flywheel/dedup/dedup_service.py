"""去重检测服务 — 检测并标记/删除重复或高度相似的标注数据。

策略:
1. 精确去重: content JSON hash 完全相同
2. 语义去重: 同 sample + 同 capability 下文本相似度 > 阈值 (Jaccard on 3-gram)
3. 结构去重: 同 sample 下 bbox IoU > 0.9 的 grounding 标注

用法:
    python -m apps.data_flywheel.dedup.dedup_service \
        --annotations-dir data/output/flywheel/v3_run_xxx/annotations \
        --semantic-threshold 0.85 \
        --execute
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("dedup")


@dataclass
class DedupStats:
    """去重统计结果。"""
    total_annotations: int = 0
    exact_duplicates: int = 0
    semantic_duplicates: int = 0
    structural_duplicates: int = 0
    unique_kept: int = 0

    @property
    def total_removed(self) -> int:
        return self.exact_duplicates + self.semantic_duplicates + self.structural_duplicates

    def summary(self) -> str:
        return (
            f"Total: {self.total_annotations} | "
            f"Exact dup: {self.exact_duplicates} | "
            f"Semantic dup: {self.semantic_duplicates} | "
            f"Structural dup: {self.structural_duplicates} | "
            f"Kept: {self.unique_kept} ({self.unique_kept/max(self.total_annotations,1)*100:.1f}%)"
        )


def _content_hash(content: Dict) -> str:
    """计算 content 的确定性 hash (排除路径等非语义字段)。"""
    filtered = {k: v for k, v in content.items() if k not in ("image_path", "image_size")}
    canonical = json.dumps(filtered, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(canonical.encode()).hexdigest()


def _text_fingerprint(text: str) -> Set[str]:
    """提取文本的 3-gram 指纹集合。"""
    text = text.lower().strip()
    words = text.split()
    if len(words) < 3:
        return set(words)
    return set(" ".join(words[i:i+3]) for i in range(len(words) - 2))


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard 相似度。"""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _bbox_iou(bbox_a: List[int], bbox_b: List[int]) -> float:
    """计算两个 bbox 的 IoU。"""
    if len(bbox_a) < 4 or len(bbox_b) < 4:
        return 0.0
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _extract_text_fields(content: Dict) -> str:
    """从 content 中提取主要文本字段用于语义比较。"""
    parts = []
    for key in ("question", "answer", "instruction", "phrase", "referring_expression",
                "absent_object", "reasoning_chain", "false_claim", "true_description"):
        val = content.get(key, "")
        if isinstance(val, str) and val:
            parts.append(val)
    # 多轮对话
    turns = content.get("turns", [])
    if turns:
        for t in turns[:4]:
            if isinstance(t, dict):
                parts.append(t.get("content", ""))
    # 规划步骤
    steps = content.get("steps", [])
    if steps:
        for s in steps[:4]:
            if isinstance(s, dict):
                parts.append(f"{s.get('action','')} {s.get('target','')}")
    return " ".join(parts)


def _extract_bbox(content: Dict) -> Optional[List[int]]:
    """从 content 中提取 bbox。"""
    bbox = content.get("bbox", [])
    if not bbox:
        boxes = content.get("boxes", [[]])
        bbox = boxes[0] if boxes else []
    if bbox and len(bbox) >= 4:
        return [int(c) for c in bbox[:4]]
    return None


def dedup_annotations(
    annotations_dir: str,
    semantic_threshold: float = 0.85,
    iou_threshold: float = 0.9,
    dry_run: bool = True,
) -> Dict[str, DedupStats]:
    """对所有能力维度执行去重。

    Args:
        annotations_dir: 标注目录 (包含 G/, A/, S/ 等子目录)
        semantic_threshold: 语义相似度阈值 (Jaccard)
        iou_threshold: bbox IoU 阈值
        dry_run: 仅报告不删除

    Returns:
        各维度的去重统计
    """
    results = {}

    for cap_dir in sorted(os.listdir(annotations_dir)):
        cap_path = os.path.join(annotations_dir, cap_dir)
        if not os.path.isdir(cap_path):
            continue

        jsonl_file = os.path.join(cap_path, f"{cap_dir}.jsonl")
        if not os.path.exists(jsonl_file):
            continue

        stats = DedupStats()
        sample_groups: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)

        with open(jsonl_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                stats.total_annotations += 1
                try:
                    ann = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = ann.get("sample_id", "")
                sample_groups[sample_id].append((idx, ann))

        seen_hashes: Set[str] = set()
        kept_indices: Set[int] = set()
        dup_indices: Set[int] = set()

        for sample_id, items in sample_groups.items():
            # Phase 1: 精确去重
            unique_items: List[Tuple[int, Dict]] = []
            for idx, ann in items:
                content = ann.get("content", {})
                h = _content_hash(content)
                if h in seen_hashes:
                    stats.exact_duplicates += 1
                    dup_indices.add(idx)
                else:
                    seen_hashes.add(h)
                    unique_items.append((idx, ann))

            # Phase 2: 语义去重 (同 capability 内)
            cap_groups: Dict[str, List[Tuple[int, Dict, Set[str]]]] = defaultdict(list)
            for idx, ann in unique_items:
                cap = ann.get("capability_tag", "")
                content = ann.get("content", {})
                text = _extract_text_fields(content)
                fp = _text_fingerprint(text)
                cap_groups[cap].append((idx, ann, fp))

            semantic_kept: List[Tuple[int, Dict]] = []
            for cap, group in cap_groups.items():
                kept_in_cap: List[Tuple[int, Dict, Set[str]]] = []
                for idx, ann, fp in group:
                    is_dup = False
                    for _, _, existing_fp in kept_in_cap:
                        sim = _jaccard_similarity(fp, existing_fp)
                        if sim > semantic_threshold:
                            is_dup = True
                            break
                    if is_dup:
                        stats.semantic_duplicates += 1
                        dup_indices.add(idx)
                    else:
                        kept_in_cap.append((idx, ann, fp))
                        semantic_kept.append((idx, ann))

            # Phase 3: 结构去重 (bbox IoU)
            bbox_items: List[Tuple[int, List[int]]] = []
            for idx, ann in semantic_kept:
                content = ann.get("content", {})
                bbox = _extract_bbox(content)
                if bbox:
                    bbox_items.append((idx, bbox))

            for i in range(len(bbox_items)):
                if bbox_items[i][0] in dup_indices:
                    continue
                for j in range(i + 1, len(bbox_items)):
                    if bbox_items[j][0] in dup_indices:
                        continue
                    iou = _bbox_iou(bbox_items[i][1], bbox_items[j][1])
                    if iou > iou_threshold:
                        stats.structural_duplicates += 1
                        dup_indices.add(bbox_items[j][0])

            for idx, ann in semantic_kept:
                if idx not in dup_indices:
                    kept_indices.add(idx)

        stats.unique_kept = len(kept_indices)

        if not dry_run and stats.total_removed > 0:
            output_lines = []
            with open(jsonl_file, "r") as f:
                for idx, line in enumerate(f):
                    if idx in kept_indices:
                        output_lines.append(line)
            backup_path = jsonl_file + ".pre_dedup"
            os.rename(jsonl_file, backup_path)
            with open(jsonl_file, "w") as f:
                f.writelines(output_lines)
            logger.info(f"[{cap_dir}] Deduped: {stats.summary()}")
        else:
            logger.info(f"[{cap_dir}] (dry-run) {stats.summary()}")

        results[cap_dir] = stats

    return results


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Data deduplication service")
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--semantic-threshold", type=float, default=0.85)
    parser.add_argument("--iou-threshold", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--execute", action="store_true", default=False)
    args = parser.parse_args()

    dry_run = not args.execute
    print(f"Dedup Service - {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"  Dir: {args.annotations_dir}")
    print(f"  Semantic threshold: {args.semantic_threshold}")
    print(f"  IoU threshold: {args.iou_threshold}")
    print()

    results = dedup_annotations(
        annotations_dir=args.annotations_dir,
        semantic_threshold=args.semantic_threshold,
        iou_threshold=args.iou_threshold,
        dry_run=dry_run,
    )

    print("\n" + "=" * 60)
    print("DEDUP SUMMARY")
    print("=" * 60)
    total_in = 0
    total_out = 0
    for cap, stats in sorted(results.items()):
        total_in += stats.total_annotations
        total_out += stats.unique_kept
        dup_pct = stats.total_removed / max(stats.total_annotations, 1) * 100
        print(f"  [{cap:4s}] {stats.total_annotations:>8,} -> {stats.unique_kept:>8,} "
              f"(removed {stats.total_removed:,}, {dup_pct:.1f}% dup)")
    print(f"\n  TOTAL: {total_in:,} -> {total_out:,} "
          f"(removed {total_in - total_out:,}, {(total_in-total_out)/max(total_in,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
