"""RGBDSource — RGBD 传感器数据源，跳过 FFS 直用传感器深度。"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from PIL import Image

from apps.data_flywheel.sources.base import DataSource, Intrinsics, Sample
from apps.data_flywheel.sources.registry import register_source

logger = logging.getLogger("flywheel.sources.rgbd")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_DEPTH_EXTS = {".npy", ".exr", ".png"}


@register_source("rgbd")
class RGBDSource(DataSource):
    """从 RGBD 目录加载 RGB + 深度图对。"""

    def __init__(
        self,
        paths: List[str],
        depth_suffix: str = "_depth.npy",
        intrinsics: Optional[Dict[str, float]] = None,
        max_images: Optional[int] = None,
    ):
        self.paths = paths
        self.depth_suffix = depth_suffix
        self.intrinsics = Intrinsics(**intrinsics) if intrinsics else Intrinsics()
        self.max_images = max_images
        self._pairs: Optional[List[tuple]] = None

    def _find_depth_path(self, rgb_path: str) -> Optional[str]:
        stem = os.path.splitext(rgb_path)[0]
        depth_path = stem + self.depth_suffix
        if os.path.exists(depth_path):
            return depth_path
        for ext in _DEPTH_EXTS:
            candidate = stem + "_depth" + ext
            if os.path.exists(candidate):
                return candidate
        return None

    def _scan_pairs(self) -> List[tuple]:
        if self._pairs is not None:
            return self._pairs

        pairs: List[tuple] = []
        for path in self.paths:
            if not os.path.isdir(path):
                continue
            for f in sorted(glob.glob(os.path.join(path, "*"))):
                ext = os.path.splitext(f)[1].lower()
                if ext not in _IMAGE_EXTS:
                    continue
                if "_depth" in os.path.basename(f):
                    continue
                depth = self._find_depth_path(f)
                if depth:
                    pairs.append((f, depth))

        if self.max_images and len(pairs) > self.max_images:
            pairs = pairs[: self.max_images]

        self._pairs = pairs
        logger.info(f"RGBDSource: found {len(pairs)} RGB-D pairs")
        return pairs

    def iter_samples(self) -> Iterator[Sample]:
        for rgb_path, depth_path in self._scan_pairs():
            try:
                img = Image.open(rgb_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open {rgb_path}: {e}")
                continue

            try:
                if depth_path.endswith(".npy"):
                    depth = np.load(depth_path)
                else:
                    depth_img = Image.open(depth_path)
                    depth = np.array(depth_img).astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to load depth {depth_path}: {e}")
                depth = None

            yield Sample(
                sample_id=Sample.generate_id("rgbd"),
                source_type="rgbd",
                frames=[img],
                metadata={
                    "image_path": rgb_path,
                    "depth_path": depth_path,
                    "depth_map": depth,
                    "has_depth": depth is not None,
                },
                intrinsics=self.intrinsics,
            )

    def __len__(self) -> int:
        return len(self._scan_pairs())

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "rgbd",
            "paths": self.paths,
            "total_pairs": len(self),
        }
