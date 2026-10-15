"""Step6: 数据集导出 — 将 grounding/referring/pointing/spatial_qa 统一导出为训练格式 JSONL。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def export_dataset(
    grounding_jsonl: str,
    referring_jsonl: str,
    pointing_jsonl: str,
    output_jsonl: str = "data/output/flywheel/train_export.jsonl",
    spatial_qa_jsonl: Optional[str] = None,
    scene_graph_jsonl: Optional[str] = None,
    keep_only_verified: bool = False,
) -> Dict[str, Any]:
    """合并所有标注为统一训练格式。"""
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    records: List[Dict] = []

    # Grounding → detection 训练样本
    if os.path.exists(grounding_jsonl):
        with open(grounding_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                for phrase_data in item.get("phrases", []):
                    for box_data in phrase_data.get("boxes", []):
                        records.append({
                            "type": "grounding",
                            "image": item["image"],
                            "phrase": phrase_data["phrase"],
                            "bbox": box_data["bbox"],
                            "width": item.get("width"),
                            "height": item.get("height"),
                        })

    # Referring → referring 训练样本
    if os.path.exists(referring_jsonl):
        with open(referring_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                records.append({
                    "type": "referring",
                    "image": item["image"],
                    "expression": item.get("expression", ""),
                    "bbox": item.get("gt_bbox"),
                    "matched_boxes": item.get("matched_boxes", []),
                    "width": item.get("width"),
                    "height": item.get("height"),
                })

    # Pointing → pointing 训练样本
    if os.path.exists(pointing_jsonl):
        with open(pointing_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                records.append({
                    "type": "pointing",
                    "image": item["image"],
                    "phrase": item.get("phrase", ""),
                    "bbox": item.get("bbox"),
                    "point": item.get("point"),
                    "width": item.get("width"),
                    "height": item.get("height"),
                })

    # Spatial QA → spatial_qa 训练样本
    if spatial_qa_jsonl and os.path.exists(spatial_qa_jsonl):
        with open(spatial_qa_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if keep_only_verified and item.get("verified") is False:
                    continue
                rec = {
                    "type": f"spatial_qa_{item.get('type', 'unknown')}",
                    "image": item["image"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "qa_type": item.get("type"),
                    "predicate": item.get("predicate"),
                    "subject": item.get("subject"),
                    "object": item.get("object"),
                }
                # 保留思维链（如果有）
                if "thinking" in item:
                    rec["thinking"] = item["thinking"]
                if "is_negative" in item:
                    rec["is_negative"] = item["is_negative"]
                for k in (
                    "answer_format",
                    "answer_bbox",
                    "answer_point",
                    "supports_training",
                    "verified",
                    "verify_score",
                    "verify_reasons",
                ):
                    if k in item:
                        rec[k] = item[k]
                records.append(rec)

    # Scene Graph → scene_graph 训练样本（整图级别）
    if scene_graph_jsonl and os.path.exists(scene_graph_jsonl):
        with open(scene_graph_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                records.append({
                    "type": "scene_graph",
                    "image": item["image"],
                    "objects": item.get("objects", []),
                    "relations": item.get("relations", []),
                    "image_size": item.get("image_size"),
                })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {}
    for rec in records:
        t = rec["type"]
        stats[t] = stats.get(t, 0) + 1

    return {
        "total": len(records),
        "stats": stats,
        "output_jsonl": output_jsonl,
    }
