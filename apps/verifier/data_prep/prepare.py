"""Verifier 数据准备 — 从飞轮标注数据构造 SFT/DPO/GRPO 训练集。

流程:
1. 从标注数据中采样 (positive: high confidence, negative: low confidence / hallucination)
2. 构造 judge prompt + label 对
3. 输出 SFT 格式 (instruction, output) 和 DPO 格式 (chosen, rejected)
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("verifier.data_prep")

JUDGE_SYSTEM_PROMPT = """You are a quality verifier for embodied AI training data. Given an annotation about a scene, determine if it is correct and high-quality.

Score the annotation from 0.0 to 1.0:
- 1.0: Perfectly correct, specific, and useful
- 0.7-0.9: Mostly correct with minor issues
- 0.4-0.6: Partially correct but has notable problems
- 0.1-0.3: Mostly incorrect or very low quality
- 0.0: Completely wrong or nonsensical

Respond with JSON: {"score": <float>, "reason": "<brief explanation>"}"""

CAP_VERIFY_TEMPLATES = {
    "G": "Verify this grounding annotation:\nObject: {name}\nBbox: {bbox}\nIs this a valid object detection with reasonable bounding box coordinates?",
    "A": "Verify this attribute annotation:\nObject: {object}\nAttribute type: {attr_type}\nQuestion: {question}\nAnswer: {answer}\nIs this attribute description accurate and specific?",
    "S": "Verify this spatial relation annotation:\nRelation: {subject} is {predicate} {object}\nIs this spatial relationship plausible for a real scene?",
    "R": "Verify this reasoning annotation:\nQuestion: {question}\nAnswer: {answer}\nType: {type}\nIs the reasoning chain logical and the answer correct?",
    "P": "Verify this planning annotation:\nInstruction: {instruction}\nSteps: {steps}\nAre the steps logical, complete, and achievable?",
    "F": "Verify this instruction annotation:\nInstruction: {instruction}\nExpected output: {expected_output}\nIs this a clear, unambiguous instruction with correct expected output?",
    "H": "Verify this hallucination annotation:\nQuestion: {question}\nAnswer: {answer}\nAbsent object: {absent_object}\nDoes this correctly identify something NOT in the scene?",
    "E": "Verify this multi-turn dialogue:\nTurns: {turns}\nIs the dialogue coherent, consistent, and grounded in the scene?",
    "Aff": "Verify this affordance annotation:\nObject: {object}\nAction: {action}\nContact point: {contact_point}\nIs this action physically plausible for this object?",
}


@dataclass
class VerifierSample:
    """单条 Verifier 训练样本。"""
    instruction: str
    output: str
    score: float
    capability: str
    source_annotation: Dict


def build_judge_prompt(annotation: Dict) -> str:
    """根据标注构造 judge prompt。"""
    cap = annotation.get("capability_tag", "")
    content = annotation.get("content", {})
    template = CAP_VERIFY_TEMPLATES.get(cap, "Verify this annotation:\n{content}")

    try:
        # 尝试用 content 字段填充模板
        prompt = template.format(**content)
    except (KeyError, TypeError):
        # fallback: 直接展示 content
        prompt = f"Verify this {cap} annotation:\n{json.dumps(content, ensure_ascii=False, indent=2)[:500]}"

    return prompt


def prepare_sft_data(
    annotations_dir: str,
    output_path: str,
    samples_per_cap: int = 500,
    score_threshold: float = 0.7,
) -> int:
    """从标注数据构造 SFT 训练集。

    策略: 用 confidence 作为伪标签
    - confidence >= score_threshold → positive (score=0.85-0.95)
    - confidence < 0.4 → negative (score=0.1-0.3)
    """
    caps = ["G", "A", "S", "R", "P", "F", "H", "E", "Aff"]
    all_samples = []

    for cap in caps:
        jsonl_file = os.path.join(annotations_dir, cap, f"{cap}.jsonl")
        if not os.path.exists(jsonl_file):
            continue

        positives = []
        negatives = []

        with open(jsonl_file) as f:
            for line in f:
                if not line.strip():
                    continue
                ann = json.loads(line)
                conf = ann.get("confidence", 0.5)
                if conf >= score_threshold:
                    positives.append(ann)
                elif conf < 0.4:
                    negatives.append(ann)

        # Sample
        pos_sample = random.sample(positives, min(samples_per_cap, len(positives)))
        neg_sample = random.sample(negatives, min(samples_per_cap // 3, len(negatives)))

        for ann in pos_sample:
            prompt = build_judge_prompt(ann)
            score = round(random.uniform(0.82, 0.95), 2)
            output = json.dumps({"score": score, "reason": "Annotation is specific, grounded, and consistent with scene context."})
            all_samples.append({"instruction": JUDGE_SYSTEM_PROMPT + "\n\n" + prompt, "output": output})

        for ann in neg_sample:
            prompt = build_judge_prompt(ann)
            score = round(random.uniform(0.1, 0.3), 2)
            output = json.dumps({"score": score, "reason": "Annotation appears incorrect or lacks specificity for the given scene."})
            all_samples.append({"instruction": JUDGE_SYSTEM_PROMPT + "\n\n" + prompt, "output": output})

    random.shuffle(all_samples)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"SFT data: {len(all_samples)} samples -> {output_path}")
    return len(all_samples)


def prepare_dpo_data(
    annotations_dir: str,
    output_path: str,
    pairs_per_cap: int = 200,
) -> int:
    """构造 DPO 训练集 (chosen vs rejected pairs)。

    策略: 同一 capability 下，high-conf 为 chosen，low-conf 为 rejected
    """
    caps = ["G", "A", "S", "R", "P", "F", "H", "E", "Aff"]
    all_pairs = []

    for cap in caps:
        jsonl_file = os.path.join(annotations_dir, cap, f"{cap}.jsonl")
        if not os.path.exists(jsonl_file):
            continue

        high_conf = []
        low_conf = []

        with open(jsonl_file) as f:
            for line in f:
                if not line.strip():
                    continue
                ann = json.loads(line)
                conf = ann.get("confidence", 0.5)
                if conf >= 0.75:
                    high_conf.append(ann)
                elif conf < 0.45:
                    low_conf.append(ann)

        n_pairs = min(pairs_per_cap, len(high_conf), len(low_conf))
        if n_pairs == 0:
            continue

        random.shuffle(high_conf)
        random.shuffle(low_conf)

        for i in range(n_pairs):
            good_ann = high_conf[i]
            bad_ann = low_conf[i]

            prompt = f"Rate the quality of this {cap} annotation on a scale of 0-1.\n\n"
            good_prompt = prompt + build_judge_prompt(good_ann)
            bad_prompt = prompt + build_judge_prompt(bad_ann)

            chosen = json.dumps({"score": round(random.uniform(0.8, 0.95), 2), "reason": "High quality, specific and grounded."})
            rejected = json.dumps({"score": round(random.uniform(0.1, 0.35), 2), "reason": "Low quality, generic or incorrect."})

            all_pairs.append({
                "prompt": JUDGE_SYSTEM_PROMPT + "\n\n" + good_prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

    random.shuffle(all_pairs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info(f"DPO data: {len(all_pairs)} pairs -> {output_path}")
    return len(all_pairs)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--output-dir", default="data/verifier_training")
    parser.add_argument("--samples-per-cap", type=int, default=500)
    args = parser.parse_args()

    sft_path = os.path.join(args.output_dir, "sft_train.jsonl")
    dpo_path = os.path.join(args.output_dir, "dpo_train.jsonl")

    n_sft = prepare_sft_data(args.annotations_dir, sft_path, args.samples_per_cap)
    n_dpo = prepare_dpo_data(args.annotations_dir, dpo_path, args.samples_per_cap // 2)

    print(f"\nVerifier training data prepared:")
    print(f"  SFT: {n_sft} samples -> {sft_path}")
    print(f"  DPO: {n_dpo} pairs -> {dpo_path}")
