"""HuggingFaceSource — 从 HuggingFace Hub 加载数据集。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional

from PIL import Image

from apps.data_flywheel.sources.base import DataSource, Sample
from apps.data_flywheel.sources.registry import register_source

logger = logging.getLogger("flywheel.sources.huggingface")


@register_source("hf")
class HuggingFaceSource(DataSource):
    """从 HuggingFace datasets 加载数据。"""

    def __init__(
        self,
        paths: List[str],
        split: str = "train",
        streaming: bool = True,
        image_column: str = "image",
        max_samples: Optional[int] = None,
    ):
        self.dataset_name = paths[0] if paths else ""
        self.split = split
        self.streaming = streaming
        self.image_column = image_column
        self.max_samples = max_samples
        self._dataset = None
        self._length: Optional[int] = None

    def _load_dataset(self):
        if self._dataset is not None:
            return self._dataset

        try:
            import datasets
        except ImportError:
            logger.error("'datasets' package not installed. Run: pip install datasets")
            return None

        token = os.environ.get("HF_TOKEN")
        try:
            self._dataset = datasets.load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=self.streaming,
                token=token,
            )
            if not self.streaming:
                self._length = len(self._dataset)
            logger.info(f"HuggingFaceSource: loaded {self.dataset_name} ({self.split})")
        except Exception as e:
            logger.error(f"Failed to load HF dataset '{self.dataset_name}': {e}")
            self._dataset = None

        return self._dataset

    def iter_samples(self) -> Iterator[Sample]:
        ds = self._load_dataset()
        if ds is None:
            return

        count = 0
        for item in ds:
            if self.max_samples and count >= self.max_samples:
                break

            img = item.get(self.image_column)
            if img is None:
                continue
            if not isinstance(img, Image.Image):
                try:
                    img = Image.open(img).convert("RGB")
                except Exception:
                    continue

            metadata = {k: v for k, v in item.items() if k != self.image_column}
            metadata["hf_dataset"] = self.dataset_name

            yield Sample(
                sample_id=Sample.generate_id("hf"),
                source_type="hf",
                frames=[img],
                metadata=metadata,
            )
            count += 1

    def __len__(self) -> int:
        if self._length is not None:
            return self._length
        if self.max_samples:
            return self.max_samples
        return 0  # streaming 模式无法预知长度

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "hf",
            "dataset_name": self.dataset_name,
            "split": self.split,
            "streaming": self.streaming,
        }
