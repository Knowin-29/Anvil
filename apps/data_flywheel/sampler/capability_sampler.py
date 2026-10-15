"""CapabilitySampler — 按 27B 能力缺口加权采样，生成 balanced 训练集。

核心逻辑:
  采样概率 ∝ gap_weight * available_count
  缺口越大的能力，采样权重越高
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("flywheel.sampler")

# 27B 能力缺口权重（来自 capability_gap_analysis.md）
DEFAULT_GAP_WEIGHTS: Dict[str, float] = {
    "G": 0.12,
    "S": 0.33,
    "A": 0.14,
    "R": 0.35,
    "P": 0.37,
    "F": 0.20,
    "H": 0.25,
    "E": 0.25,
    "Aff": 0.50,
}


@dataclass
class SamplerConfig:
    """采样器配置。"""
    gap_weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_GAP_WEIGHTS.copy())
    min_score_threshold: float = 0.7
    max_samples_per_capability: int = 5000
    total_budget: Optional[int] = None
    seed: int = 42


class CapabilitySampler:
    """按能力缺口加权采样，生成 capability-balanced 训练 JSONL。"""

    def __init__(self, config: Optional[SamplerConfig] = None):
        self.config = config or SamplerConfig()
        random.seed(self.config.seed)

    def sample_from_run(self, run_dir: str, output_path: Optional[str] = None) -> str:
        """从 run 输出中采样生成训练集。"""
        annotations_dir = os.path.join(run_dir, "annotations")
        if not os.path.isdir(annotations_dir):
            logger.warning(f"No annotations dir: {annotations_dir}")
            return ""

        # 按 capability 加载数据
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
                        # 过滤低分样本
                        score = item.get("verifier_score")
                        if score is not None and score < self.config.min_score_threshold:
                            continue
                        pools[cap_dir].append(item)

        if not pools:
            logger.warning("No valid annotations found")
            return ""

        # 计算每个 capability 的采样数量
        allocations = self._compute_allocations(pools)

        # 执行采样
        sampled: List[Dict] = []
        for cap, n_samples in allocations.items():
            pool = pools.get(cap, [])
            if not pool:
                continue
            n = min(n_samples, len(pool))
            selected = random.sample(pool, n)
            sampled.extend(selected)
            logger.info(f"  {cap}: sampled {n}/{len(pool)} (weight={self.config.gap_weights.get(cap, 0):.2f})")

        random.shuffle(sampled)

        # 写入输出
        if output_path is None:
            output_path = os.path.join(run_dir, "train_balanced.jsonl")

        with open(output_path, "w", encoding="utf-8") as f:
            for item in sampled:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Sampled {len(sampled)} total → {output_path}")
        return output_path

    def _compute_allocations(self, pools: Dict[str, List[Dict]]) -> Dict[str, int]:
        """计算每个 capability 的采样配额。"""
        weights = self.config.gap_weights
        total_budget = self.config.total_budget

        # 计算加权分数
        weighted_scores: Dict[str, float] = {}
        for cap, pool in pools.items():
            w = weights.get(cap, 0.1)
            weighted_scores[cap] = w * len(pool)

        total_weighted = sum(weighted_scores.values())
        if total_weighted == 0:
            return {}

        # 如果没有总预算，取所有可用样本
        if total_budget is None:
            total_budget = sum(len(p) for p in pools.values())

        allocations: Dict[str, int] = {}
        for cap, score in weighted_scores.items():
            proportion = score / total_weighted
            n = int(proportion * total_budget)
            n = min(n, self.config.max_samples_per_capability)
            n = min(n, len(pools[cap]))
            allocations[cap] = max(1, n)

        return allocations

    def get_sampling_report(self, pools: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """生成采样报告（用于 GUI 展示）。"""
        allocations = self._compute_allocations(pools)
        weights = self.config.gap_weights

        report = {
            "gap_weights": weights,
            "pool_sizes": {k: len(v) for k, v in pools.items()},
            "allocations": allocations,
            "total_sampled": sum(allocations.values()),
        }
        return report
