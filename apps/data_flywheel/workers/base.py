"""Capability Worker 基类和核心类型定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from apps.data_flywheel.sources.base import Sample


class CapabilityTag(str, Enum):
    """9 项能力维度，与 binhui benchmark 雷达图对齐。"""
    G = "G"        # visual_grounding
    S = "S"        # spatial_understanding
    A = "A"        # attribute_state
    R = "R"        # scene_reasoning
    P = "P"        # task_planning
    F = "F"        # instruction_following
    H = "H"        # hallucination_control
    E = "E"        # multi_turn_consistency
    Aff = "Aff"    # affordance_finegrained

    @property
    def full_name(self) -> str:
        _names = {
            "G": "visual_grounding",
            "S": "spatial_understanding",
            "A": "attribute_state",
            "R": "scene_reasoning",
            "P": "task_planning",
            "F": "instruction_following",
            "H": "hallucination_control",
            "E": "multi_turn_consistency",
            "Aff": "affordance_finegrained",
        }
        return _names[self.value]


@dataclass
class CapabilityAnnotation:
    """Worker 输出的标注单元。"""
    sample_id: str
    capability_tag: CapabilityTag
    sub_tag: str
    content: Dict[str, Any]
    confidence: float = 1.0
    provenance: str = "heuristic"
    verifier_score: Optional[float] = None
    difficulty: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "capability_tag": self.capability_tag.value,
            "sub_tag": self.sub_tag,
            "content": self.content,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "verifier_score": self.verifier_score,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }


class CapabilityWorker(ABC):
    """能力 Worker 抽象基类。每个 Worker 负责一个能力维度的标注生成。"""

    tag: CapabilityTag

    @abstractmethod
    def process(self, sample: Sample, **kwargs) -> List[CapabilityAnnotation]:
        """处理单个样本，返回标注列表。"""
        ...

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}({self.tag.value})"

    def supports(self, sample: Sample) -> bool:
        """判断该 worker 是否支持处理此样本（可覆写）。"""
        return True
