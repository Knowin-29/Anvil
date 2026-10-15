"""VideoSource — 视频文件数据源，支持抽帧和关键帧检测。"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

from PIL import Image

from apps.data_flywheel.sources.base import DataSource, Sample
from apps.data_flywheel.sources.registry import register_source

logger = logging.getLogger("flywheel.sources.video")

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _load_video_backend():
    """尝试加载视频后端: decord > opencv。"""
    try:
        import decord  # noqa: F401
        return "decord"
    except ImportError:
        pass
    try:
        import cv2  # noqa: F401
        return "opencv"
    except ImportError:
        pass
    return None


@register_source("video")
class VideoSource(DataSource):
    """从视频文件抽帧生成样本。"""

    def __init__(
        self,
        paths: List[str],
        fps: float = 1.0,
        keyframe_only: bool = False,
        max_frames: int = 100,
    ):
        self.paths = paths
        self.fps = fps
        self.keyframe_only = keyframe_only
        self.max_frames = max_frames
        self._backend = _load_video_backend()
        self._video_files: Optional[List[str]] = None

        if self._backend is None:
            logger.warning("No video backend available (install decord or opencv-python)")

    def _scan_files(self) -> List[str]:
        if self._video_files is not None:
            return self._video_files

        files: List[str] = []
        for path in self.paths:
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in _VIDEO_EXTS:
                files.append(path)
            elif os.path.isdir(path):
                for f in sorted(glob.glob(os.path.join(path, "*"))):
                    if os.path.splitext(f)[1].lower() in _VIDEO_EXTS:
                        files.append(f)

        self._video_files = files
        return files

    def _extract_frames_decord(self, video_path: str) -> List[Image.Image]:
        import decord
        import numpy as np

        vr = decord.VideoReader(video_path)
        total = len(vr)
        video_fps = vr.get_avg_fps()
        step = max(1, int(video_fps / self.fps))
        indices = list(range(0, total, step))[: self.max_frames]
        frames_np = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(frames_np[i]) for i in range(len(indices))]

    def _extract_frames_opencv(self, video_path: str) -> List[Image.Image]:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(video_fps / self.fps))
        frames: List[Image.Image] = []
        idx = 0
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            idx += 1
        cap.release()
        return frames

    def iter_samples(self) -> Iterator[Sample]:
        if self._backend is None:
            logger.error("No video backend, skipping VideoSource")
            return

        for vpath in self._scan_files():
            try:
                if self._backend == "decord":
                    frames = self._extract_frames_decord(vpath)
                else:
                    frames = self._extract_frames_opencv(vpath)
            except Exception as e:
                logger.warning(f"Failed to extract frames from {vpath}: {e}")
                continue

            if not frames:
                continue

            yield Sample(
                sample_id=Sample.generate_id("video"),
                source_type="video",
                frames=frames,
                metadata={
                    "image_path": vpath,
                    "video_path": vpath,
                    "num_frames": len(frames),
                    "backend": self._backend,
                },
                episode_id=os.path.splitext(os.path.basename(vpath))[0],
            )

    def __len__(self) -> int:
        return len(self._scan_files())

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "video",
            "paths": self.paths,
            "backend": self._backend,
            "total_videos": len(self),
        }
