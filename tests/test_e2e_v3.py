"""v3.0 端到端测试 — 验证全 capability 通过。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from apps.data_flywheel.sources import Sample
from apps.data_flywheel.sources.image_dir import ImageDirSource
from apps.data_flywheel.workers.base import CapabilityTag
from apps.data_flywheel.orchestrator.runner import RunConfig, run


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> str:
    """创建 5 张测试图片。"""
    for i in range(5):
        img = Image.new("RGB", (640, 480), color=(i * 50, 100, 200 - i * 30))
        img.save(tmp_path / f"test_{i:03d}.jpg")
    return str(tmp_path)


class TestE2EV3:
    """v3.0 端到端测试。"""

    def test_full_capability_run(self, tmp_image_dir: str, tmp_path: Path):
        """5 张图全 capability 通过。"""
        source = ImageDirSource(paths=[tmp_image_dir])
        output_root = str(tmp_path / "output")

        config = RunConfig(
            capabilities=set(CapabilityTag),
            scheduler_backend="sequential",
            output_root=output_root,
            run_id="test_e2e",
        )

        result = run(source, config)

        assert result.total_samples == 5
        assert result.total_annotations >= 0
        assert os.path.exists(os.path.join(output_root, "test_e2e", "manifest.jsonl"))
        assert os.path.exists(os.path.join(output_root, "test_e2e", "capability_stats.json"))

    def test_single_capability(self, tmp_image_dir: str, tmp_path: Path):
        """仅启用 Hallucination 能力。"""
        source = ImageDirSource(paths=[tmp_image_dir], max_images=2)
        output_root = str(tmp_path / "output_single")

        config = RunConfig(
            capabilities={CapabilityTag.H},
            scheduler_backend="sequential",
            output_root=output_root,
            run_id="test_single",
        )

        result = run(source, config)
        assert result.total_samples == 2

    def test_source_compatibility(self, tmp_image_dir: str, tmp_path: Path):
        """验证 ImageDirSource 向后兼容。"""
        source = ImageDirSource(paths=[tmp_image_dir])
        samples = list(source.iter_samples())
        assert len(samples) == 5
        for s in samples:
            assert s.source_type == "image_dir"
            assert s.image_path is not None

    def test_output_structure(self, tmp_image_dir: str, tmp_path: Path):
        """验证输出目录结构。"""
        source = ImageDirSource(paths=[tmp_image_dir], max_images=2)
        output_root = str(tmp_path / "output_struct")

        config = RunConfig(
            capabilities={CapabilityTag.H, CapabilityTag.A},
            scheduler_backend="sequential",
            output_root=output_root,
            run_id="test_struct",
        )

        result = run(source, config)

        run_dir = os.path.join(output_root, "test_struct")
        assert os.path.isdir(os.path.join(run_dir, "annotations"))
        assert os.path.isfile(os.path.join(run_dir, "manifest.jsonl"))
        assert os.path.isfile(os.path.join(run_dir, "capability_stats.json"))

        # 验证 manifest 格式
        with open(os.path.join(run_dir, "manifest.jsonl")) as f:
            for line in f:
                item = json.loads(line)
                assert "sample_id" in item
                assert "source_type" in item
