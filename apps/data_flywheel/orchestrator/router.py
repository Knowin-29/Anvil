"""CapabilityRouter — 根据 sample 特征和配置决定走哪些 worker。"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.base import CapabilityTag, CapabilityWorker

logger = logging.getLogger("flywheel.orchestrator.router")


class CapabilityRouter:
    """根据 sample 特征和配置决定走哪些 worker。"""

    def __init__(
        self,
        workers: Dict[CapabilityTag, CapabilityWorker],
        enabled_capabilities: Optional[Set[CapabilityTag]] = None,
    ):
        self._workers = workers
        self._enabled = enabled_capabilities or set(CapabilityTag)

    def route(self, sample: Sample) -> List[CapabilityWorker]:
        """返回该 sample 应该经过的 worker 列表。"""
        result: List[CapabilityWorker] = []

        for tag, worker in self._workers.items():
            if tag not in self._enabled:
                continue
            if not worker.supports(sample):
                continue
            result.append(worker)

        logger.debug(f"Routed sample {sample.sample_id} to {len(result)} workers")
        return result

    @property
    def enabled_tags(self) -> Set[CapabilityTag]:
        return self._enabled

    def enable(self, tag: CapabilityTag) -> None:
        self._enabled.add(tag)

    def disable(self, tag: CapabilityTag) -> None:
        self._enabled.discard(tag)
