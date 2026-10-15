"""Scanner 配置。"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class JudgeEndpoint:
    """单个 Judge 模型端点配置。"""
    name: str
    base_url: str = "http://localhost:10301"
    model: str = "qwen2.5-vl-32b"
    weight: float = 1.0
    max_tokens: int = 512
    temperature: float = 0.1


@dataclass
class ScannerConfig:
    """Scanner 全局配置。"""
    annotations_dir: str = "data/output/flywheel/v3_run_1778980547/annotations"
    score_suffix: str = ".scores.jsonl"
    judges: List[JudgeEndpoint] = field(default_factory=lambda: [
        JudgeEndpoint(name="qwen2.5-vl-32b", base_url="http://localhost:10301")
    ])
    batch_size: int = 8
    concurrency: int = 4
    pass_threshold: float = 0.6
    delete_threshold: float = 0.3
    scan_interval_seconds: int = 60
    max_image_size: int = 960
    capabilities: Optional[List[str]] = None  # None = scan all
    resume: bool = True
    max_samples_per_file: int = 0  # 0 = no limit
