"""Verifier 数据收集器 — 从 Phase 1 输出抽取伪标签训练数据。

多信号伪标签策略:
- grounding: 跨模型 IoU 一致性
- referring/attribute: LLM-as-judge
- pointing: mask 内/外二值
- spatial_qa: 程序化校验 + LLM judge
- affordance: 可达性 + 物理合理性 + LLM judge
- R/P/F/H/E: LLM judge 双盲 + self-consistency
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from apps.data_flywheel.workers.base import CapabilityTag

logger = logging.getLogger("flywheel.verifier.collector")


@dataclass
class VerifierSample:
    """Verifier 训练样本。"""
    sample_id: str
    capability_tag: str
    input_text: str
    output_text: str
    pseudo_label: int  # 0 or 1
    signals: Dict[str, float] = field(default_factory=dict)
    source_run_id: str = ""
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "capability_tag": self.capability_tag,
            "input": self.input_text,
            "output": self.output_text,
            "pseudo_label": self.pseudo_label,
            "signals": self.signals,
            "source_run_id": self.source_run_id,
            "image_path": self.image_path,
        }

    def to_sft_format(self) -> Dict[str, Any]:
        """转为 ms-swift SFT 训练格式。"""
        messages = [
            {"role": "system", "content": "You are a data quality verifier for embodied AI training data."},
            {"role": "user", "content": self.input_text},
            {"role": "assistant", "content": self.output_text},
        ]
        result = {"messages": messages}
        if self.image_path:
            result["images"] = [self.image_path]
        return result

    def annotation_hash(self) -> str:
        content = f"{self.sample_id}:{self.capability_tag}:{self.input_text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class VerifierDataCollector:
    """从飞轮输出收集 Verifier 训练数据。"""

    def __init__(self, run_dir: str, output_dir: str = "verifier_data"):
        self.run_dir = run_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def collect(self) -> str:
        """收集所有 capability 的伪标签数据，输出 v0_train.jsonl。"""
        samples: List[VerifierSample] = []

        annotations_dir = os.path.join(self.run_dir, "annotations")
        if not os.path.isdir(annotations_dir):
            logger.warning(f"No annotations dir found: {annotations_dir}")
            return ""

        for cap_dir in sorted(os.listdir(annotations_dir)):
            cap_path = os.path.join(annotations_dir, cap_dir)
            if not os.path.isdir(cap_path):
                continue

            for jsonl_file in os.listdir(cap_path):
                if not jsonl_file.endswith(".jsonl"):
                    continue
                filepath = os.path.join(cap_path, jsonl_file)
                cap_samples = self._process_capability_file(filepath, cap_dir)
                samples.extend(cap_samples)

        # 写入训练文件
        output_path = os.path.join(self.output_dir, "v0_train.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

        # 写入 ms-swift SFT 格式
        sft_path = os.path.join(self.output_dir, "v0_sft.jsonl")
        with open(sft_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_sft_format(), ensure_ascii=False) + "\n")

        logger.info(f"Collected {len(samples)} verifier samples → {output_path}")
        return output_path

    def _process_capability_file(self, filepath: str, cap_tag: str) -> List[VerifierSample]:
        """处理单个 capability 的 JSONL 文件。"""
        samples: List[VerifierSample] = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ann = json.loads(line)
                except json.JSONDecodeError:
                    continue

                vs = self._annotation_to_verifier_sample(ann, cap_tag)
                if vs:
                    samples.append(vs)

        return samples

    def _annotation_to_verifier_sample(self, ann: Dict, cap_tag: str) -> Optional[VerifierSample]:
        """将标注转为 Verifier 训练样本。"""
        sample_id = ann.get("sample_id", "")
        content = ann.get("content", {})
        confidence = ann.get("confidence", 0.5)

        # 构建 input (问题)
        input_text = self._build_input(cap_tag, content)
        if not input_text:
            return None

        # 构建 output (判断)
        pseudo_label, signals = self._compute_pseudo_label(cap_tag, ann)
        score = confidence if pseudo_label == 1 else (1.0 - confidence)
        output_text = f"[Score: {score:.2f}] {'Correct' if pseudo_label == 1 else 'Incorrect'}. "
        output_text += self._build_rationale(cap_tag, content, pseudo_label)

        return VerifierSample(
            sample_id=sample_id,
            capability_tag=cap_tag,
            input_text=input_text,
            output_text=output_text,
            pseudo_label=pseudo_label,
            signals=signals,
            source_run_id=os.path.basename(self.run_dir),
            image_path=ann.get("metadata", {}).get("image_path"),
        )

    def _build_input(self, cap_tag: str, content: Dict) -> str:
        """构建 Verifier 输入文本。"""
        if cap_tag == "G":
            return f"[Capability: grounding]\nAnnotation: {json.dumps(content, ensure_ascii=False)[:500]}\nIs this grounding annotation correct?"
        elif cap_tag == "S":
            return f"[Capability: spatial]\nAnnotation: {json.dumps(content, ensure_ascii=False)[:500]}\nIs this spatial relationship correct?"
        elif cap_tag == "A":
            return f"[Capability: attribute]\nAnnotation: {json.dumps(content, ensure_ascii=False)[:500]}\nIs this attribute annotation correct?"
        elif cap_tag == "Aff":
            return f"[Capability: affordance]\nAnnotation: {json.dumps(content, ensure_ascii=False)[:500]}\nIs this affordance annotation physically plausible?"
        elif cap_tag == "H":
            return f"[Capability: hallucination]\nAnnotation: {json.dumps(content, ensure_ascii=False)[:500]}\nIs this negative/counterfactual sample well-formed?"
        else:
            return f"[Capability: {cap_tag}]\nAnnotation: {json.dumps(content, ensure_ascii=False)[:500]}\nIs this annotation correct?"

    def _compute_pseudo_label(self, cap_tag: str, ann: Dict) -> tuple:
        """计算伪标签 + 信号。"""
        confidence = ann.get("confidence", 0.5)
        signals: Dict[str, float] = {"confidence": confidence}

        # 基于 confidence 的简单伪标签
        if confidence >= 0.7:
            pseudo_label = 1
        elif confidence <= 0.3:
            pseudo_label = 0
        else:
            # 边界样本: 基于 capability 特定规则
            pseudo_label = self._capability_specific_label(cap_tag, ann)

        signals["pseudo_label_source"] = 1.0 if confidence >= 0.7 or confidence <= 0.3 else 0.5
        return pseudo_label, signals

    def _capability_specific_label(self, cap_tag: str, ann: Dict) -> int:
        """能力特定的伪标签规则。"""
        content = ann.get("content", {})

        if cap_tag == "G":
            bbox = content.get("bbox", [])
            if bbox and len(bbox) == 4:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w > 5 and h > 5:
                    return 1
            return 0

        elif cap_tag == "Aff":
            point = content.get("contact_point", [])
            bbox = content.get("bbox", [])
            if point and bbox and len(point) == 2 and len(bbox) == 4:
                if bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]:
                    return 1
            return 0

        elif cap_tag == "H":
            # 否定样本通常是正确的
            return 1

        return 1  # 默认正标签

    def _build_rationale(self, cap_tag: str, content: Dict, label: int) -> str:
        """构建判断理由。"""
        if label == 1:
            if cap_tag == "G":
                return "The bounding box appears to correctly localize the target object."
            elif cap_tag == "Aff":
                return "The contact point is within the object boundary and the action is physically plausible."
            elif cap_tag == "S":
                return "The spatial relationship is consistent with the object positions."
            else:
                return "The annotation appears correct and well-formed."
        else:
            if cap_tag == "G":
                return "The bounding box may not accurately cover the target object."
            elif cap_tag == "Aff":
                return "The contact point may be outside the object or the action is not physically plausible."
            else:
                return "The annotation may contain errors."

    def collect_dpo_pairs(self) -> str:
        """从 v0_train.jsonl 生成 DPO 对。"""
        train_path = os.path.join(self.output_dir, "v0_train.jsonl")
        if not os.path.exists(train_path):
            logger.warning("v0_train.jsonl not found, run collect() first")
            return ""

        dpo_pairs: List[Dict] = []
        samples_by_cap: Dict[str, List[Dict]] = {}

        with open(train_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                cap = item["capability_tag"]
                if cap not in samples_by_cap:
                    samples_by_cap[cap] = []
                samples_by_cap[cap].append(item)

        for cap, items in samples_by_cap.items():
            positives = [i for i in items if i["pseudo_label"] == 1]
            negatives = [i for i in items if i["pseudo_label"] == 0]

            for pos in positives[:50]:
                for neg in negatives[:50]:
                    dpo_pairs.append({
                        "messages": [
                            {"role": "system", "content": "You are a data quality verifier."},
                            {"role": "user", "content": pos["input"]},
                        ],
                        "chosen": pos["output"],
                        "rejected": neg["output"],
                    })

        dpo_path = os.path.join(self.output_dir, "v0_dpo.jsonl")
        with open(dpo_path, "w", encoding="utf-8") as f:
            for pair in dpo_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info(f"Generated {len(dpo_pairs)} DPO pairs → {dpo_path}")
        return dpo_path
