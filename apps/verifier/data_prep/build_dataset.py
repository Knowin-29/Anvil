"""Verifier 数据准备 — 从飞轮标注 + Scanner 分数构建 SFT/DPO/GRPO 训练数据。

流程:
1. SFT 数据: 高分标注 → (prompt, chosen_response) 对
2. DPO 数据: 同 sample 的高分 vs 低分 → (prompt, chosen, rejected) 三元组
3. GRPO 数据: 标注 + reward signal → (prompt, response, reward) 三元组

用法:
    python -m apps.verifier.data_prep.build_dataset \
        --annotations-dir data/output/flywheel/daily_20260519/annotations \
        --output-dir data/verifier/train_data \
        --format sft|dpo|grpo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [verifier.data_prep] %(message)s")
logger = logging.getLogger("verifier.data_prep")

ALL_CAPS = ["G", "A", "S", "R", "P", "F", "H", "E", "Aff"]

# ─── Prompt Templates ────────────────────────────────────────────────────────

VERIFY_PROMPT_TEMPLATE = """You are a quality verifier for embodied AI training data. Given an annotation, judge whether it is correct and high-quality.

Annotation:
- Capability: {capability}
- Sub-tag: {sub_tag}
- Content: {content_json}

Evaluate this annotation on:
1. Factual correctness (does it match what would be seen in the image?)
2. Completeness (are all required fields present and meaningful?)
3. Consistency (do the fields agree with each other?)

Respond with a JSON object:
{{"score": <0.0-1.0>, "rationale": "<brief explanation>", "issues": [<list of issues if any>]}}"""

VERIFY_RESPONSE_GOOD = """{{"score": {score}, "rationale": "{rationale}", "issues": []}}"""
VERIFY_RESPONSE_BAD = """{{"score": {score}, "rationale": "{rationale}", "issues": {issues}}}"""


def load_annotations_with_scores(annotations_dir: str) -> List[Dict]:
    """加载标注 + 对应的 scores（如果有）。"""
    all_anns = []

    for cap in ALL_CAPS:
        ann_file = os.path.join(annotations_dir, cap, f"{cap}.jsonl")
        score_file = os.path.join(annotations_dir, cap, f"{cap}.scores.jsonl")

        if not os.path.exists(ann_file):
            continue

        # Load scores index
        scores_map = {}
        if os.path.exists(score_file):
            with open(score_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    key = f"{obj.get('sample_id', '')}_{obj.get('sub_tag', '')}"
                    scores_map[key] = obj.get("ensemble_score", obj.get("score", 0.5))

        # Load annotations
        with open(ann_file) as f:
            for line in f:
                if not line.strip():
                    continue
                ann = json.loads(line)
                key = f"{ann.get('sample_id', '')}_{ann.get('sub_tag', '')}"
                ann["_score"] = scores_map.get(key, ann.get("confidence", 0.5))
                ann["_capability"] = cap
                all_anns.append(ann)

    logger.info(f"Loaded {len(all_anns)} annotations from {annotations_dir}")
    return all_anns


def build_sft_data(annotations: List[Dict], min_score: float = 0.6) -> List[Dict]:
    """构建 SFT 数据: 高质量标注 → (instruction, output) 对。"""
    sft_data = []

    for ann in annotations:
        score = ann["_score"]
        if score < min_score:
            continue

        content_json = json.dumps(ann.get("content", {}), ensure_ascii=False, indent=2)
        prompt = VERIFY_PROMPT_TEMPLATE.format(
            capability=ann.get("capability_tag", ann["_capability"]),
            sub_tag=ann.get("sub_tag", ""),
            content_json=content_json[:500],
        )

        response = json.dumps({
            "score": round(score, 2),
            "rationale": f"This {ann.get('sub_tag', '')} annotation is well-formed with correct structure and plausible content.",
            "issues": [],
        }, ensure_ascii=False)

        sft_data.append({
            "instruction": prompt,
            "output": response,
            "capability": ann["_capability"],
            "sub_tag": ann.get("sub_tag", ""),
        })

    logger.info(f"SFT data: {len(sft_data)} samples (min_score={min_score})")
    return sft_data


def build_dpo_data(annotations: List[Dict], score_gap: float = 0.3) -> List[Dict]:
    """构建 DPO 数据: 同 capability 的高分 vs 低分配对。"""
    # Group by capability
    by_cap = defaultdict(list)
    for ann in annotations:
        by_cap[ann["_capability"]].append(ann)

    dpo_data = []
    for cap, anns in by_cap.items():
        sorted_anns = sorted(anns, key=lambda x: x["_score"], reverse=True)
        high = [a for a in sorted_anns if a["_score"] >= 0.7]
        low = [a for a in sorted_anns if a["_score"] < 0.4]

        if not high or not low:
            continue

        for h in high[:50]:
            l = random.choice(low)
            content_h = json.dumps(h.get("content", {}), ensure_ascii=False)[:300]
            content_l = json.dumps(l.get("content", {}), ensure_ascii=False)[:300]

            prompt = f"Evaluate the quality of this {cap} annotation. Is it correct and complete?"

            chosen = json.dumps({
                "score": round(h["_score"], 2),
                "rationale": "High quality annotation with correct content.",
                "issues": [],
            })
            rejected = json.dumps({
                "score": round(l["_score"], 2),
                "rationale": "Low quality annotation with issues.",
                "issues": ["Incorrect or implausible content", "Missing required fields"],
            })

            dpo_data.append({
                "prompt": prompt + f"\n\nAnnotation: {content_h}",
                "chosen": chosen,
                "rejected": rejected,
                "capability": cap,
            })

    logger.info(f"DPO data: {len(dpo_data)} pairs (score_gap={score_gap})")
    return dpo_data


def build_grpo_data(annotations: List[Dict]) -> List[Dict]:
    """构建 GRPO 数据: 标注 + reward signal。"""
    grpo_data = []

    for ann in annotations:
        content_json = json.dumps(ann.get("content", {}), ensure_ascii=False)[:400]
        prompt = VERIFY_PROMPT_TEMPLATE.format(
            capability=ann.get("capability_tag", ann["_capability"]),
            sub_tag=ann.get("sub_tag", ""),
            content_json=content_json,
        )

        # Reward = normalized score
        reward = ann["_score"]

        response = json.dumps({
            "score": round(reward, 2),
            "rationale": f"Annotation quality assessment for {ann.get('sub_tag', '')}.",
            "issues": [] if reward >= 0.6 else ["Potential quality issues detected"],
        })

        grpo_data.append({
            "prompt": prompt,
            "response": response,
            "reward": round(reward, 3),
            "capability": ann["_capability"],
        })

    logger.info(f"GRPO data: {len(grpo_data)} samples")
    return grpo_data


def save_dataset(data: List[Dict], output_path: str, format_name: str):
    """保存数据集为 JSONL 格式。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} {format_name} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build Verifier Training Data")
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--output-dir", default="data/verifier/train_data")
    parser.add_argument("--format", choices=["sft", "dpo", "grpo", "all"], default="all")
    parser.add_argument("--min-score", type=float, default=0.6)
    args = parser.parse_args()

    annotations = load_annotations_with_scores(args.annotations_dir)

    if args.format in ("sft", "all"):
        sft = build_sft_data(annotations, min_score=args.min_score)
        save_dataset(sft, os.path.join(args.output_dir, "sft_train.jsonl"), "SFT")

    if args.format in ("dpo", "all"):
        dpo = build_dpo_data(annotations)
        save_dataset(dpo, os.path.join(args.output_dir, "dpo_train.jsonl"), "DPO")

    if args.format in ("grpo", "all"):
        grpo = build_grpo_data(annotations)
        save_dataset(grpo, os.path.join(args.output_dir, "grpo_train.jsonl"), "GRPO")


if __name__ == "__main__":
    main()
