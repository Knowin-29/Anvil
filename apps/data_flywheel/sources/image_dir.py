"""ImageDirSource — 本地图片目录数据源，兼容 v2.1 的 --image-dir。"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

from PIL import Image

from apps.data_flywheel.sources.base import DataSource, Sample
from apps.data_flywheel.sources.registry import register_source

logger = logging.getLogger("flywheel.sources.image_dir")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


@register_source("image_dir")
class ImageDirSource(DataSource):
    """从一个或多个本地目录加载图片。"""

    def __init__(
        self,
        paths: List[str],
        max_images: Optional[int] = None,
        min_resolution: Optional[List[int]] = None,
        recursive: bool = False,
    ):
        self.paths = paths
        self.max_images = max_images
        self.min_resolution = tuple(min_resolution) if min_resolution else None
        self.recursive = recursive
        self._file_list: Optional[List[str]] = None

    def _scan_files(self) -> List[str]:
        if self._file_list is not None:
            return self._file_list

        files: List[str] = []
        for path in self.paths:
            if not os.path.isdir(path):
                logger.warning(f"Directory not found, skipping: {path}")
                continue
            pattern = os.path.join(path, "**", "*") if self.recursive else os.path.join(path, "*")
            for f in sorted(glob.glob(pattern, recursive=self.recursive)):
                if os.path.splitext(f)[1].lower() in _IMAGE_EXTS:
                    files.append(f)

        if self.max_images and len(files) > self.max_images:
            files = files[: self.max_images]

        self._file_list = files
        logger.info(f"ImageDirSource: found {len(files)} images from {len(self.paths)} dirs")
        return files

    def iter_samples(self) -> Iterator[Sample]:
        for fpath in self._scan_files():
            try:
                img = Image.open(fpath).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open {fpath}: {e}")
                continue

            if self.min_resolution:
                w, h = img.size
                if w < self.min_resolution[0] or h < self.min_resolution[1]:
                    continue

            yield Sample(
                sample_id=Sample.generate_id("imgdir"),
                source_type="image_dir",
                frames=[img],
                metadata={"image_path": fpath, "filename": os.path.basename(fpath)},
            )

    def __len__(self) -> int:
        return len(self._scan_files())

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "image_dir",
            "paths": self.paths,
            "total_images": len(self),
        }
