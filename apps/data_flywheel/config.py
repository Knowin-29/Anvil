"""飞轮全局配置，所有路径/阈值集中管理。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from apps.data_flywheel.ontology.store import resolve_store_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_ontology_store_path() -> str:
    return resolve_store_path(str(PROJECT_ROOT / "data" / "output" / "flywheel" / "ontology_store.json"))


@dataclass
class FlywheelConfig:
    # ---- 数据源 ----
    image_dirs: List[str] = field(default_factory=lambda: [
        str(PROJECT_ROOT / "data" / "input" / "images"),
    ])
    # ---- 模型 ----
    model_path: str = "IDEA-Research/Rex-Omni"
    backend: str = "transformers"
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_h"
    # ---- 模型推理参数 ----
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0       # 注: do_sample=False 时 top_p 必须为 1.0，否则 transformers 报 warning
    top_k: int = 1
    repetition_penalty: float = 1.05
    # ---- 输出 ----
    output_root: str = str(PROJECT_ROOT / "data" / "output" / "flywheel")
    # ---- 增强 ----
    augment_per_image: int = 2
    # ---- 标注 ----
    grounding_conf_thresh: float = 0.3
    filter_adjectives: bool = True  # 论文 Phrase Filtering
    caption_backend: str = "auto"  # auto / dashscope / openai / anthropic / local
    # ---- API 模式 ----
    use_api: bool = False  # True=全 API 模式，无需本地模型（精度较低）
    detector_models: List[str] = field(default_factory=list)  # 可选，多模型 A/B (空=仅 model_path)
    detector_primary: str = ""  # 主检测器 tag（空=第一个）
    detection_ab_max_images: int = 0  # A/B 评测图片上限（0=不限制）
    enable_detection_verifier: bool = True
    detection_verifier_policy: str = "drop"  # drop / rescore / flag
    grounding_min_area_ratio: float = 0.0005
    grounding_max_area_ratio: float = 0.9
    grounding_min_aspect_ratio: float = 0.1
    grounding_max_aspect_ratio: float = 10.0
    grounding_nms_iou_thresh: float = 0.7
    # ---- Ontology / Review ----
    enable_ontology_bootstrap: bool = True
    enable_ontology_runtime_matching: bool = True
    ontology_store_path: str = field(default_factory=_default_ontology_store_path)
    # ---- Perception Router ----
    detvlm_enabled: bool = False
    detvlm_url: str = "YOUR_DETVLM_URL"  # DetectionVLM vLLM 服务地址
    dashscope_enabled: bool = False
    dashscope_api_key: Optional[str] = None
    sam3_enabled: bool = False
    sam3_url: str = "YOUR_SAM3_URL"
    local_vlm_enabled: bool = False
    local_vlm_url: str = "http://localhost:10301"
    mask_output_dir: str = "data/output/masks"

    # ---- 深度估计 (新增) ----
    enable_depth_estimation: bool = True   # 是否启用 FFS 深度估计
    ffs_server_url: str = "YOUR_FFS_DEPTH_URL"
    ffs_timeout: int = 60                  # 单次请求超时 (秒)
    ffs_max_workers: int = 4               # 深度估计并发数
    default_fx: float = 500.0              # 默认相机内参 fx
    default_fy: float = 500.0              # 默认相机内参 fy
    default_cx: float = 640.0              # 默认相机内参 cx
    default_cy: float = 400.0              # 默认相机内参 cy

    # ---- Scene Graph ----
    enable_scene_graph: bool = True        # 构建场景图 + 生成空间关系 QA
    enable_3d_relations: bool = True       # 是否启用 3D 空间关系（需要深度估计）
    scene_graph_thresholds: Optional[Dict] = None  # 可覆盖默认阈值

    # ---- Spatial QA ----
    max_qa_per_image: int = 20             # 每张图最多生成的 QA 数量
    include_negative_qa: bool = True       # 生成负例 QA
    negative_qa_ratio: float = 0.3         # 负例占比
    enable_thinking_chain: bool = True     # QA 是否包含思维链
    qa_language: str = "en"                # QA 语言: en / zh
    enable_bbox_single_qa: bool = True
    enable_bbox_relational_qa: bool = True
    enable_point_single_qa: bool = True
    enable_point_relational_qa: bool = True
    enable_qa_verifier: bool = True
    qa_verify_iou_thresh: float = 0.7
    qa_verify_point_margin: float = 0.0
    keep_only_verified: bool = False

    # ---- 评估 ----
    iou_thresh: float = 0.5
    hard_sample_iou_thresh: float = 0.3

    # ---- 步骤开关（GUI 可独立勾选）----
    enable_augment: bool = True
    enable_caption: bool = True
    enable_grounding: bool = True
    enable_referring: bool = True
    enable_pointing: bool = True
    enable_eval: bool = True
    enable_hard_mining: bool = True

    # ---- 飞轮 ----
    rounds: int = 1
    max_images_per_round: int = 50
    # ---- 路径 ----
    relative_paths: bool = True  # JSONL 中使用相对路径

    # ---- 预估耗时（秒/张图，用于 GUI 显示）----
    @staticmethod
    def estimate_time(n_images: int, use_api: bool = False,
                      enable_augment: bool = True, enable_caption: bool = True,
                      enable_grounding: bool = True, enable_depth: bool = True,
                      enable_scene_graph: bool = True, enable_referring: bool = True,
                      enable_pointing: bool = True, enable_eval: bool = True) -> dict:
        """预估各步骤耗时，返回 {step: seconds}。"""
        # 基于 benchmark 数据（单图）
        MODEL_LOAD = 45.0 if not use_api else 0.0  # 模型加载（首次）
        PER_IMG = {
            "augment":      0.15,
            "caption":      5.5 if use_api else 8.0,   # API ~5.5s, 本地 ~8s
            "grounding":    0.5 if use_api else 3.0,    # API ~0.5s, 本地 ~3s
            "depth":        0.5,                         # FFS API ~0.5s
            "scene_graph":  0.02,                        # 纯计算
            "spatial_qa":   0.01,                        # 纯计算
            "referring":    0.0 if use_api else 2.0,     # 需要模型
            "pointing":     0.0 if use_api else 1.0,     # 需要模型
            "eval":         0.0 if use_api else 3.0,     # 需要模型
            "export":       0.01,
            "hard_mining":  0.01,
        }
        steps = {}
        if enable_augment:
            steps["① 图片增强"] = PER_IMG["augment"] * n_images
        if enable_caption:
            steps["② Caption"] = PER_IMG["caption"] * n_images
        if enable_grounding:
            steps["③ Grounding"] = PER_IMG["grounding"] * n_images
        if enable_depth and enable_scene_graph:
            steps["③.5 深度估计"] = PER_IMG["depth"] * n_images
        if enable_scene_graph:
            steps["③.6 场景图"] = PER_IMG["scene_graph"] * n_images
            steps["③.7 空间QA"] = PER_IMG["spatial_qa"] * n_images
        if enable_referring:
            steps["④ Referring"] = PER_IMG["referring"] * n_images
        if enable_pointing:
            steps["⑤ Pointing"] = PER_IMG["pointing"] * n_images
        steps["⑥ 导出"] = PER_IMG["export"] * n_images
        if enable_eval:
            steps["⑦ 评估"] = PER_IMG["eval"] * n_images
            steps["⑧ 难例挖掘"] = PER_IMG["hard_mining"] * n_images

        if not use_api and (enable_grounding or enable_referring or enable_pointing or enable_eval):
            steps["⓪ 模型加载"] = MODEL_LOAD

        total = sum(steps.values())
        steps["合计"] = total
        return steps

    @property
    def grounding_jsonl(self) -> str:
        return os.path.join(self.output_root, "grounding.jsonl")

    @property
    def depth_info_jsonl(self) -> str:
        return os.path.join(self.output_root, "depth_info.jsonl")

    @property
    def scene_graph_jsonl(self) -> str:
        return os.path.join(self.output_root, "scene_graph.jsonl")

    @property
    def spatial_qa_jsonl(self) -> str:
        return os.path.join(self.output_root, "spatial_qa.jsonl")

    @property
    def referring_jsonl(self) -> str:
        return os.path.join(self.output_root, "referring.jsonl")

    @property
    def pointing_jsonl(self) -> str:
        return os.path.join(self.output_root, "pointing.jsonl")

    @property
    def export_jsonl(self) -> str:
        return os.path.join(self.output_root, "train_export.jsonl")

    @property
    def eval_report(self) -> str:
        return os.path.join(self.output_root, "eval_report.json")

    @property
    def hard_samples_jsonl(self) -> str:
        return os.path.join(self.output_root, "hard_samples.jsonl")

    @property
    def vis_dir(self) -> str:
        return os.path.join(self.output_root, "visualizations")

    def ensure_dirs(self) -> None:
        for d in [self.output_root, self.vis_dir]:
            os.makedirs(d, exist_ok=True)
        for img_dir in self.image_dirs:
            os.makedirs(img_dir, exist_ok=True)
