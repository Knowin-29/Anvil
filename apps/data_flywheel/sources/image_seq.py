"""ImageSeqSource — 多帧 episode 序列数据源。

按文件名前缀分组为 episode，例如:
  ep001_frame_000.jpg, ep001_frame_001.jpg → episode "ep001"
"""

from __future__ import annotations

import glob
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional

from PIL import Image

from apps.data_flywheel.sources.base import DataSource, Sample
from apps.data_flywheel.sources.registry import register_source

logger = logging.getLogger("flywheel.sources.image_seq")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_DEFAULT_PATTERN = re.compile(r"^(.+?)_(?:frame_?)?(\d+)\.\w+$", re.IGNORECASE)


@register_source("image_seq")
class ImageSeqSource(DataSource):
    """按文件名前缀分组为多帧 episode。"""

    def __init__(
        self,
        paths: List[str],
        episode_pattern: Optional[str] = None,
        max_frames_per_episode: int = 50,
    ):
        self.paths = paths
        self.max_frames = max_frames_per_episode
        self._pattern = re.compile(episode_pattern) if episode_pattern else _DEFAULT_PATTERN
        self._episodes: Optional[Dict[str, List[str]]] = None

    def _scan_episodes(self) -> Dict[str, List[str]]:
        if self._episodes is not None:
            return self._episodes

        episodes: Dict[str, List[str]] = defaultdict(list)
        for path in self.paths:
            if not os.path.isdir(path):
                continue
            for f in sorted(glob.glob(os.path.join(path, "*"))):
                if os.path.splitext(f)[1].lower() not in _IMAGE_EXTS:
                    continue
                basename = os.path.basename(f)
                m = self._pattern.match(basename)
                if m:
                    ep_id = m.group(1)
                    episodes[ep_id].append(f)
                else:
                    episodes[basename].append(f)

        for ep_id in episodes:
            episodes[ep_id] = episodes[ep_id][: self.max_frames]

        self._episodes = dict(episodes)
        logger.info(f"ImageSeqSource: {len(episodes)} episodes from {len(self.paths)} dirs")
        return self._episodes

    def iter_samples(self) -> Iterator[Sample]:
        for ep_id, frame_paths in self._scan_episodes().items():
            frames = []
            for fp in frame_paths:
                try:
                    frames.append(Image.open(fp).convert("RGB"))
                except Exception as e:
                    logger.warning(f"Failed to open {fp}: {e}")

            if not frames:
                continue

            yield Sample(
                sample_id=Sample.generate_id("seq"),
                source_type="image_seq",
                frames=frames,
                metadata={
                    "image_path": frame_paths[0],
                    "frame_paths": frame_paths,
                    "num_frames": len(frames),
                },
                episode_id=ep_id,
                frame_idx=0,
            )

    def __len__(self) -> int:
        return len(self._scan_episodes())

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "image_seq",
            "paths": self.paths,
            "total_episodes": len(self),
        }
