"""Step8: 难例挖掘与回流 — 低 IoU 样本、场景图质量不足样本重新进入增强→标注循环。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def mine_hard_samples(
    eval_report_path: str,
    hard_iou_thresh: float = 0.3,
    output_jsonl: str = "data/output/flywheel/hard_samples.jsonl",
    scene_graph_jsonl: Optional[str] = None,
    depth_info_jsonl: Optional[str] = None,
    min_relation_count: int = 2,
) -> Dict[str, Any]:
    """从评估报告和场景图中提取难例。

    难例类型：
    - low_iou: IoU < 阈值的检测样本
    - low_relation_count: 场景图中关系数过少的图片
    - depth_estimation_failed: 深度估计失败的图片
    """
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    hard_samples: List[Dict] = []
    hard_images: set = set()

    # ---- 1. 低 IoU 难例 ----
    if os.path.exists(eval_report_path):
        with open(eval_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        for sample in report.get("samples", []):
            if sample.get("best_iou", 1.0) < hard_iou_thresh:
                hard_samples.append({
                    "image": sample["image"],
                    "phrase": sample["phrase"],
                    "gt_bbox": sample["gt_bbox"],
                    "best_iou": sample["best_iou"],
                    "reason": "low_iou",
                })
                hard_images.add(sample["image"])

    # ---- 2. 场景图关系数过少 ----
    if scene_graph_jsonl and os.path.exists(scene_graph_jsonl):
        with open(scene_graph_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sg = json.loads(line)
                n_objects = len(sg.get("objects", []))
                n_relations = len(sg.get("relations", []))
                # 有多个物体但关系很少 → 可能检测质量差或场景复杂
                if n_objects >= 2 and n_relations < min_relation_count:
                    img = sg["image"]
                    if img not in hard_images:
                        hard_samples.append({
                            "image": img,
                            "phrase": "",
                            "gt_bbox": [],
                            "best_iou": -1,
                            "reason": "low_relation_count",
                            "n_objects": n_objects,
                            "n_relations": n_relations,
                        })
                        hard_images.add(img)

    # ---- 3. 深度估计失败 ----
    if depth_info_jsonl and os.path.exists(depth_info_jsonl):
        with open(depth_info_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if not d.get("depth_valid", True):
                    img = d["image"]
                    if img not in hard_images:
                        hard_samples.append({
                            "image": img,
                            "phrase": "",
                            "gt_bbox": [],
                            "best_iou": -1,
                            "reason": "depth_estimation_failed",
                        })
                        hard_images.add(img)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in hard_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计各类难例数量
    reason_counts: Dict[str, int] = {}
    for s in hard_samples:
        r = s["reason"]
        reason_counts[r] = reason_counts.get(r, 0) + 1

    return {
        "hard_count": len(hard_samples),
        "total_evaluated": len(report.get("samples", [])) if os.path.exists(eval_report_path) else 0,
        "reason_counts": reason_counts,
        "output_jsonl": output_jsonl,
    }


def get_hard_image_paths(hard_samples_jsonl: str) -> List[str]:
    """从难例 JSONL 中提取需要重新处理的图片路径。"""
    paths: List[str] = []
    seen = set()
    if not os.path.exists(hard_samples_jsonl):
        return paths
    with open(hard_samples_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            p = item.get("image", "")
            if p and p not in seen:
                paths.append(p)
                seen.add(p)
    return paths
