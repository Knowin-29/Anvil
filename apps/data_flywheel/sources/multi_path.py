"""MultiPathSource — 多数据源加权混合采样器。"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import yaml

from apps.data_flywheel.sources.base import DataSource, Sample
from apps.data_flywheel.sources.registry import get_source

logger = logging.getLogger("flywheel.sources.multi_path")


@dataclass
class SourceConfig:
    """单个数据源配置。"""
    path: str
    source_type: str = "image_dir"
    weight: float = 1.0
    filters: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourcesConfig:
    """多数据源配置。"""
    sources: List[SourceConfig]

    @classmethod
    def from_yaml(cls, path: str) -> "SourcesConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        sources_raw = data.get("sources", [])
        sources: List[SourceConfig] = []
        for item in sources_raw:
            if isinstance(item, str):
                sources.append(SourceConfig(path=item))
            elif isinstance(item, dict):
                sources.append(SourceConfig(
                    path=item.get("path", ""),
                    source_type=item.get("source_type", "image_dir"),
                    weight=item.get("weight", 1.0),
                    filters=item.get("filters", {}),
                    extra={k: v for k, v in item.items()
                           if k not in ("path", "source_type", "weight", "filters")},
                ))
        return cls(sources=sources)

    @classmethod
    def from_paths(cls, paths: List[str], source_type: str = "image_dir") -> "SourcesConfig":
        """向后兼容: 纯路径列表 → 全部当指定类型。"""
        return cls(sources=[SourceConfig(path=p, source_type=source_type) for p in paths])


class MultiPathSource(DataSource):
    """多数据源加权混合采样。"""

    source_type = "multi"

    def __init__(self, config: SourcesConfig):
        self.config = config
        self._sources: Optional[List[tuple]] = None

    def _build_sources(self) -> List[tuple]:
        """构建 (DataSource, weight) 列表。"""
        if self._sources is not None:
            return self._sources

        result: List[tuple] = []
        for sc in self.config.sources:
            kwargs: Dict[str, Any] = {"paths": [sc.path]}
            kwargs.update(sc.filters)
            kwargs.update(sc.extra)
            try:
                source = get_source(sc.source_type, **kwargs)
                result.append((source, sc.weight))
                logger.info(f"Loaded source: {sc.source_type} @ {sc.path} (weight={sc.weight})")
            except Exception as e:
                logger.warning(f"Failed to create source {sc.source_type} @ {sc.path}: {e}")

        self._sources = result
        return result

    def iter_samples(self) -> Iterator[Sample]:
        """按权重交错迭代所有源。"""
        sources = self._build_sources()
        if not sources:
            return

        iterators = [(src.iter_samples(), w) for src, w in sources]
        buffers: List[List[Sample]] = [[] for _ in iterators]
        exhausted = [False] * len(iterators)

        while not all(exhausted):
            for i, (it, weight) in enumerate(iterators):
                if exhausted[i]:
                    continue
                n_pull = max(1, int(weight * 5))
                for _ in range(n_pull):
                    try:
                        buffers[i].append(next(it))
                    except StopIteration:
                        exhausted[i] = True
                        break

            combined = []
            for buf in buffers:
                combined.extend(buf)
                buf.clear()

            random.shuffle(combined)
            yield from combined

    def __len__(self) -> int:
        return sum(len(src) for src, _ in self._build_sources())

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "multi",
            "num_sources": len(self.config.sources),
            "sources": [{"path": s.path, "type": s.source_type, "weight": s.weight}
                        for s in self.config.sources],
        }
