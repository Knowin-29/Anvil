"""数据源抽象层 — 统一多模态数据接入。

支持 5 种数据形态:
- image_dir: 本地图片目录
- image_seq: 多帧 episode 序列
- video: 视频文件
- rgbd: RGBD 传感器数据
- hf: HuggingFace 数据集
"""

from apps.data_flywheel.sources.base import DataSource, Intrinsics, Sample
from apps.data_flywheel.sources.registry import get_source, list_sources, register_source

# 导入具体实现以触发 @register_source 注册
from apps.data_flywheel.sources import image_dir  # noqa: F401
from apps.data_flywheel.sources import image_seq  # noqa: F401
from apps.data_flywheel.sources import video  # noqa: F401
from apps.data_flywheel.sources import rgbd  # noqa: F401
from apps.data_flywheel.sources import huggingface  # noqa: F401

__all__ = [
    "DataSource",
    "Intrinsics",
    "Sample",
    "get_source",
    "list_sources",
    "register_source",
]
