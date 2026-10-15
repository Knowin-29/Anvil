"""数据源基类和核心数据结构。"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from PIL import Image


@dataclass
class Intrinsics:
    """相机内参。"""
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0

    def as_matrix(self) -> List[List[float]]:
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]


@dataclass
class Sample:
    """统一的多模态样本数据结构。"""
    sample_id: str
    source_type: str
    frames: List[Image.Image]
    metadata: Dict[str, Any] = field(default_factory=dict)
    intrinsics: Optional[Intrinsics] = None
    episode_id: Optional[str] = None
    frame_idx: Optional[int] = None

    @property
    def image(self) -> Image.Image:
        """便捷属性：返回第一帧（单帧场景）。"""
        return self.frames[0]

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def image_path(self) -> Optional[str]:
        """返回原始图片路径（如果有）。"""
        return self.metadata.get("image_path")

    @staticmethod
    def generate_id(prefix: str = "") -> str:
        short_uuid = uuid.uuid4().hex[:12]
        return f"{prefix}_{short_uuid}" if prefix else short_uuid


class DataSource(ABC):
    """数据源抽象基类。所有数据源必须实现此接口。"""

    source_type: str = "base"

    @abstractmethod
    def iter_samples(self) -> Iterator[Sample]:
        """惰性迭代所有样本。"""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """返回样本总数（可能是估计值）。"""
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """返回数据源元信息。"""
        return {"source_type": self.source_type}

    def __iter__(self) -> Iterator[Sample]:
        return self.iter_samples()
