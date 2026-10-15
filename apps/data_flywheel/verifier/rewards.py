"""Verifier GRPO Reward Functions — ms-swift 兼容。

三个 reward function:
- format_reward: 检查输出格式 [Score: X.XX] + rationale
- consistency_reward: 与多信号伪标签的一致性
- downstream_reward: 下游任务表现（placeholder，需实际 benchmark 数据）
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_SCORE_PATTERN = re.compile(r"\[Score:\s*([\d.]+)\]")


def format_reward(completions: List[str], **kwargs) -> List[float]:
    """检查输出格式是否符合 [Score: X.XX] + rationale。"""
    rewards: List[float] = []
    for comp in completions:
        score = 0.0
        # 检查 [Score: X.XX] 格式
        match = _SCORE_PATTERN.search(comp)
        if match:
            score += 0.5
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 1.0:
                    score += 0.3
            except ValueError:
                pass
        # 检查有 rationale（至少 10 个字符的解释）
        if match:
            rationale = comp[match.end():].strip()
            if len(rationale) >= 10:
                score += 0.2
        rewards.append(score)
    return rewards


def consistency_reward(completions: List[str], **kwargs) -> List[float]:
    """与伪标签的一致性。"""
    pseudo_labels = kwargs.get("pseudo_label", [])
    if not pseudo_labels:
        return [0.5] * len(completions)

    rewards: List[float] = []
    for i, comp in enumerate(completions):
        label = pseudo_labels[i] if i < len(pseudo_labels) else 1
        match = _SCORE_PATTERN.search(comp)
        if not match:
            rewards.append(0.0)
            continue

        try:
            pred_score = float(match.group(1))
        except ValueError:
            rewards.append(0.0)
            continue

        # 正标签应该给高分，负标签应该给低分
        if label == 1:
            reward = pred_score  # 越高越好
        else:
            reward = 1.0 - pred_score  # 越低越好

        rewards.append(reward)

    return rewards


def downstream_reward(completions: List[str], **kwargs) -> List[float]:
    """下游任务表现 reward（placeholder）。

    完整实现需要:
    1. 用 verifier 过滤数据
    2. 训练小模型
    3. 在 capability benchmark 上评估
    4. 将提升回灌为 reward

    当前简化为: 与 format + consistency 的加权平均。
    """
    format_r = format_reward(completions, **kwargs)
    consistency_r = consistency_reward(completions, **kwargs)

    rewards: List[float] = []
    for f, c in zip(format_r, consistency_r):
        rewards.append(0.3 * f + 0.7 * c)

    return rewards
