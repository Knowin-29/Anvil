"""数据源测试 — 验证每种 source 能正常迭代。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from apps.data_flywheel.sources import DataSource, Sample, get_source, list_sources
from apps.data_flywheel.sources.image_dir import ImageDirSource
from apps.data_flywheel.sources.image_seq import ImageSeqSource
from apps.data_flywheel.sources.multi_path import MultiPathSource, SourcesConfig


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> str:
    """创建临时图片目录。"""
    for i in range(3):
        img = Image.new("RGB", (640, 480), color=(i * 80, 100, 200))
        img.save(tmp_path / f"test_{i:03d}.jpg")
    return str(tmp_path)


@pytest.fixture
def tmp_seq_dir(tmp_path: Path) -> str:
    """创建临时 episode 序列目录。"""
    for ep in range(2):
        for frame in range(3):
            img = Image.new("RGB", (320, 240), color=(ep * 100, frame * 80, 50))
            img.save(tmp_path / f"ep{ep:03d}_frame_{frame:03d}.jpg")
    return str(tmp_path)


class TestImageDirSource:
    def test_scan_and_iterate(self, tmp_image_dir: str):
        source = ImageDirSource(paths=[tmp_image_dir])
        assert len(source) == 3

        samples = list(source.iter_samples())
        assert len(samples) == 3
        for s in samples:
            assert isinstance(s, Sample)
            assert s.source_type == "image_dir"
            assert s.num_frames == 1
            assert s.image.size == (640, 480)

    def test_max_images(self, tmp_image_dir: str):
        source = ImageDirSource(paths=[tmp_image_dir], max_images=2)
        assert len(source) == 2

    def test_missing_dir(self):
        source = ImageDirSource(paths=["/nonexistent/path"])
        assert len(source) == 0

    def test_registry(self, tmp_image_dir: str):
        source = get_source("image_dir", paths=[tmp_image_dir])
        assert isinstance(source, ImageDirSource)
        assert len(source) == 3


class TestImageSeqSource:
    def test_episode_grouping(self, tmp_seq_dir: str):
        source = ImageSeqSource(paths=[tmp_seq_dir])
        assert len(source) == 2  # 2 episodes

        samples = list(source.iter_samples())
        assert len(samples) == 2
        for s in samples:
            assert s.source_type == "image_seq"
            assert s.num_frames == 3
            assert s.episode_id is not None


class TestMultiPathSource:
    def test_from_paths(self, tmp_image_dir: str):
        config = SourcesConfig.from_paths([tmp_image_dir])
        source = MultiPathSource(config)
        samples = list(source.iter_samples())
        assert len(samples) == 3

    def test_multi_source(self, tmp_image_dir: str, tmp_seq_dir: str):
        config = SourcesConfig(sources=[
            __import__("apps.data_flywheel.sources.multi_path", fromlist=["SourceConfig"]).SourceConfig(
                path=tmp_image_dir, source_type="image_dir", weight=1.0
            ),
            __import__("apps.data_flywheel.sources.multi_path", fromlist=["SourceConfig"]).SourceConfig(
                path=tmp_seq_dir, source_type="image_seq", weight=1.0
            ),
        ])
        source = MultiPathSource(config)
        samples = list(source.iter_samples())
        assert len(samples) == 5  # 3 images + 2 episodes


class TestListSources:
    def test_registered_sources(self):
        sources = list_sources()
        assert "image_dir" in sources
        assert "image_seq" in sources
        assert "video" in sources
        assert "rgbd" in sources
        assert "hf" in sources
