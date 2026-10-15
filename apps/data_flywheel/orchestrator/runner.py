"""Runner — 飞轮 v3.0 总入口，编排 source → router → scheduler → output。"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from apps.data_flywheel.orchestrator.router import CapabilityRouter
from apps.data_flywheel.orchestrator.scheduler import SchedulerBackend, create_scheduler
from apps.data_flywheel.sources.base import DataSource, Sample
from apps.data_flywheel.workers.base import CapabilityAnnotation, CapabilityTag, CapabilityWorker

logger = logging.getLogger("flywheel.orchestrator.runner")


def _infer_scene_tag(image_path: str) -> str:
    """从图片路径推断场景标签。"""
    path_lower = image_path.lower()
    if "vlm_dataset_root" in path_lower:
        if "raw_augment" in path_lower:
            return "anvil_augment"
        return "appliance"
    if "washing_machine" in path_lower:
        return "anvil_washing_machine"
    if "brands_images" in path_lower:
        return "brands"
    if "dataset_knob" in path_lower:
        return "knob"
    if "fsd_points" in path_lower or "prism" in path_lower:
        return "academic_spatial"
    if "clevr" in path_lower:
        return "academic_clevr"
    if "aokvqa" in path_lower or "vqav2" in path_lower:
        return "academic_vqa"
    if "refspatial" in path_lower:
        return "academic_refspatial"
    return "other"


@dataclass
class RunConfig:
    """Runner 配置。"""
    capabilities: Set[CapabilityTag] = field(default_factory=lambda: set(CapabilityTag))
    scheduler_backend: str = "auto"
    num_gpus: Optional[int] = None
    model_path: str = "IDEA-Research/Rex-Omni"
    output_root: str = "data/output"
    run_id: Optional[str] = None
    verifier_threshold: float = 0.7
    verifier_margin: float = 0.1
    vlm_url: str = "http://localhost:10301"
    vlm_model: str = "qwen2.5-vl-32b"


@dataclass
class RunResult:
    """Runner 输出结果。"""
    run_id: str
    total_samples: int
    total_annotations: int
    per_capability: Dict[str, int]
    elapsed_seconds: float
    output_dir: str


def _build_workers(capabilities: Set[CapabilityTag], **kwargs) -> Dict[CapabilityTag, CapabilityWorker]:
    """按需构建 worker 实例。"""
    from apps.data_flywheel.workers.grounding_worker import GroundingWorker
    from apps.data_flywheel.workers.spatial_worker import SpatialWorker
    from apps.data_flywheel.workers.attribute_worker import AttributeWorker
    from apps.data_flywheel.workers.reasoning_worker import ReasoningWorker
    from apps.data_flywheel.workers.planning_worker import PlanningWorker
    from apps.data_flywheel.workers.instruction_worker import InstructionWorker
    from apps.data_flywheel.workers.hallucination_worker import HallucinationWorker
    from apps.data_flywheel.workers.multiturn_worker import MultiturnWorker
    from apps.data_flywheel.workers.affordance_worker import AffordanceWorker

    _WORKER_MAP = {
        CapabilityTag.G: GroundingWorker,
        CapabilityTag.S: SpatialWorker,
        CapabilityTag.A: AttributeWorker,
        CapabilityTag.R: ReasoningWorker,
        CapabilityTag.P: PlanningWorker,
        CapabilityTag.F: InstructionWorker,
        CapabilityTag.H: HallucinationWorker,
        CapabilityTag.E: MultiturnWorker,
        CapabilityTag.Aff: AffordanceWorker,
    }

    workers: Dict[CapabilityTag, CapabilityWorker] = {}
    for tag in capabilities:
        if tag in _WORKER_MAP:
            workers[tag] = _WORKER_MAP[tag](**kwargs)

    return workers


def run(
    source: DataSource,
    config: Optional[RunConfig] = None,
) -> RunResult:
    """v3.0 飞轮总入口。"""
    if config is None:
        config = RunConfig()

    run_id = config.run_id or f"v3_run_{int(time.time())}"
    output_dir = os.path.join(config.output_root, run_id)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # 初始化结构化日志
    from apps.data_flywheel.orchestrator.logger import FlywheelLogger
    fly_logger = FlywheelLogger(log_dir=os.path.join(output_dir, "logs"))

    # 构建 VLM Client (所有 Worker 共享)
    from apps.data_flywheel.workers.vlm_client import VLMClient
    vlm_client = VLMClient(base_url=config.vlm_url, model=config.vlm_model)
    logger.info(f"VLM client initialized: {config.vlm_url} / {config.vlm_model}")

    # 构建 workers
    workers = _build_workers(config.capabilities, vlm_client=vlm_client)
    router = CapabilityRouter(workers, config.capabilities)
    scheduler = create_scheduler(
        backend=config.scheduler_backend,
        num_gpus=config.num_gpus,
        model_path=config.model_path,
    )

    # 按 capability 分组的输出文件
    annotation_files: Dict[str, Any] = {}
    per_capability: Dict[str, int] = {}

    # Verifier 集成（可选）
    verifier = None
    if config.verifier_threshold > 0:
        try:
            from apps.data_flywheel.verifier.server import VerifierInference
            verifier = VerifierInference(backend="transformers")
            logger.info("Verifier enabled (inline mode)")
        except Exception as e:
            logger.info(f"Verifier not available, skipping: {e}")

    start_time = time.time()
    total_samples = 0
    total_annotations = 0
    verified_pass = 0
    verified_fail = 0

    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    manifest_f = open(manifest_path, "w", encoding="utf-8")

    try:
        for sample in source:
            total_samples += 1
            routed_workers = router.route(sample)

            # 分离 GroundingWorker（优先执行）和下游 Worker
            grounding_workers = [w for w in routed_workers if w.tag == CapabilityTag.G]
            downstream_workers = [w for w in routed_workers if w.tag != CapabilityTag.G]

            # 先执行 Grounding
            grounding_annotations = []
            if grounding_workers:
                grounding_annotations = scheduler.submit(sample, grounding_workers)

            # 提取 grounding_data 传给下游（统一格式）
            grounding_data = []
            for ann in grounding_annotations:
                content = ann.content or {}
                # 统一字段: phrase→name, boxes→bbox
                name = content.get("name") or content.get("phrase", "object")
                bbox = content.get("bbox")
                if not bbox and content.get("boxes"):
                    bbox = content["boxes"][0] if content["boxes"] else None
                if bbox:
                    grounding_data.append({
                        "name": name,
                        "bbox": bbox,
                        "phrase": name,
                        "confidence": ann.confidence,
                    })

            # 执行下游 Worker（传入 grounding_data）
            downstream_annotations = scheduler.submit(
                sample, downstream_workers, grounding_data=grounding_data
            )

            annotations = grounding_annotations + downstream_annotations

            # 日志: 样本处理
            fly_logger.log_sample(sample.sample_id, sample.source_type, sample.image_path)

            # Verifier 评分（如果可用）
            if verifier:
                for ann in annotations:
                    try:
                        result = verifier.verify(
                            capability_tag=ann.capability_tag.value,
                            input_text=json.dumps(ann.content, ensure_ascii=False)[:300],
                            output_text="",
                        )
                        ann.verifier_score = result.get("score", 0.5)
                        if ann.verifier_score >= config.verifier_threshold:
                            verified_pass += 1
                        else:
                            verified_fail += 1
                    except Exception:
                        pass

            total_annotations += len(annotations)

            # 日志: 标注生成
            if annotations:
                caps = list({a.capability_tag.value for a in annotations})
                fly_logger.log_annotations(sample.sample_id, annotations, caps)

            # 进度日志（每 10 个样本）
            if total_samples % 10 == 0:
                fly_logger.log_progress(total_samples, len(source), per_capability.copy())

            # 写入按 capability 分组的 JSONL
            for ann in annotations:
                tag_str = ann.capability_tag.value
                per_capability[tag_str] = per_capability.get(tag_str, 0) + 1

                cap_dir = os.path.join(output_dir, "annotations", tag_str)
                os.makedirs(cap_dir, exist_ok=True)

                if tag_str not in annotation_files:
                    fpath = os.path.join(cap_dir, f"{tag_str}.jsonl")
                    annotation_files[tag_str] = open(fpath, "w", encoding="utf-8")

                annotation_files[tag_str].write(json.dumps(ann.to_dict(), ensure_ascii=False) + "\n")

            # 写 manifest
            scene_tag = _infer_scene_tag(sample.image_path or "")
            manifest_f.write(json.dumps({
                "sample_id": sample.sample_id,
                "source_type": sample.source_type,
                "image_path": sample.image_path,
                "scene": scene_tag,
                "num_annotations": len(annotations),
                "capabilities": list({a.capability_tag.value for a in annotations}),
            }, ensure_ascii=False) + "\n")

            if total_samples % 10 == 0:
                logger.info(f"Processed {total_samples} samples, {total_annotations} annotations")

    finally:
        manifest_f.close()
        for f in annotation_files.values():
            f.close()
        scheduler.shutdown()

    elapsed = time.time() - start_time

    # 写 capability_stats.json
    stats = {
        "run_id": run_id,
        "total_samples": total_samples,
        "total_annotations": total_annotations,
        "per_capability": per_capability,
        "elapsed_seconds": elapsed,
        "samples_per_second": total_samples / elapsed if elapsed > 0 else 0,
        "verifier": {
            "enabled": verifier is not None,
            "pass": verified_pass,
            "fail": verified_fail,
            "threshold": config.verifier_threshold,
        },
    }
    with open(os.path.join(output_dir, "capability_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # 日志: 运行完成
    fly_logger.log_complete(stats)

    logger.info(f"Run complete: {total_samples} samples, {total_annotations} annotations in {elapsed:.1f}s")

    return RunResult(
        run_id=run_id,
        total_samples=total_samples,
        total_annotations=total_annotations,
        per_capability=per_capability,
        elapsed_seconds=elapsed,
        output_dir=output_dir,
    )
