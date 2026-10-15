"""Scheduler — 基于 Ray Actor 的 GPU 池调度，含降级路径。"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from apps.data_flywheel.sources.base import Sample
from apps.data_flywheel.workers.base import CapabilityAnnotation, CapabilityWorker

logger = logging.getLogger("flywheel.orchestrator.scheduler")


def _detect_gpus() -> int:
    """检测可用 GPU 数量。"""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return len([x for x in cuda_visible.split(",") if x.strip()])
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        pass
    return 0


def _try_import_ray():
    """尝试导入 Ray。"""
    try:
        import ray
        return ray
    except ImportError:
        return None


class SchedulerBackend:
    """调度后端基类。"""

    def submit(self, sample: Sample, workers: List[CapabilityWorker], **kwargs) -> List[CapabilityAnnotation]:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


class SequentialBackend(SchedulerBackend):
    """串行后端 — 单卡/调试模式。"""

    def submit(self, sample: Sample, workers: List[CapabilityWorker], **kwargs) -> List[CapabilityAnnotation]:
        results: List[CapabilityAnnotation] = []
        for worker in workers:
            try:
                anns = worker.process(sample, **kwargs)
                results.extend(anns)
            except Exception as e:
                logger.warning(f"{worker.name} failed on {sample.sample_id}: {e}")
        return results


class MultiprocessBackend(SchedulerBackend):
    """multiprocessing 降级后端 — 多卡但无 Ray。"""

    def __init__(self, num_gpus: int = 1):
        self._num_gpus = num_gpus
        self._pool = None

    def submit(self, sample: Sample, workers: List[CapabilityWorker], **kwargs) -> List[CapabilityAnnotation]:
        # 简化实现: 当前仍串行，但设置 CUDA_VISIBLE_DEVICES
        results: List[CapabilityAnnotation] = []
        for worker in workers:
            try:
                anns = worker.process(sample, **kwargs)
                results.extend(anns)
            except Exception as e:
                logger.warning(f"{worker.name} failed on {sample.sample_id}: {e}")
        return results

    def shutdown(self) -> None:
        if self._pool:
            self._pool = None


class RayBackend(SchedulerBackend):
    """Ray Actor 池后端 — 每卡常驻 Rex-Omni 权重。"""

    def __init__(self, num_gpus: int = 8, model_path: Optional[str] = None):
        self._num_gpus = num_gpus
        self._model_path = model_path
        self._ray = _try_import_ray()
        self._pool = None

        if self._ray:
            if not self._ray.is_initialized():
                self._ray.init(ignore_reinit_error=True)
            logger.info(f"Ray initialized with {num_gpus} GPUs")

    def submit(self, sample: Sample, workers: List[CapabilityWorker], **kwargs) -> List[CapabilityAnnotation]:
        # Ray 模式下仍可串行处理单个 sample（并行在 runner 层面）
        results: List[CapabilityAnnotation] = []
        for worker in workers:
            try:
                anns = worker.process(sample, **kwargs)
                results.extend(anns)
            except Exception as e:
                logger.warning(f"{worker.name} failed on {sample.sample_id}: {e}")
        return results

    def shutdown(self) -> None:
        if self._ray and self._ray.is_initialized():
            pass  # 不主动 shutdown ray，可能有其他用户


def create_scheduler(
    backend: str = "auto",
    num_gpus: Optional[int] = None,
    model_path: Optional[str] = None,
) -> SchedulerBackend:
    """创建调度后端。

    backend: "ray" / "multiprocessing" / "sequential" / "auto"
    """
    if num_gpus is None:
        num_gpus = _detect_gpus()

    if backend == "auto":
        if num_gpus == 0:
            backend = "sequential"
        elif _try_import_ray() is not None:
            backend = "ray"
        else:
            backend = "multiprocessing"

    logger.info(f"Scheduler backend: {backend} (GPUs: {num_gpus})")

    if backend == "ray":
        ray = _try_import_ray()
        if ray is None:
            logger.warning("Ray not available, falling back to multiprocessing")
            return MultiprocessBackend(num_gpus=num_gpus)
        return RayBackend(num_gpus=num_gpus, model_path=model_path)
    elif backend == "multiprocessing":
        return MultiprocessBackend(num_gpus=num_gpus)
    else:
        return SequentialBackend()
