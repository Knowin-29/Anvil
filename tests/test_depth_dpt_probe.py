from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import probe_depth_dpt_v1 as probe


def _stat(label: str, value: float, uncertainty: float) -> dict:
    return {"label": label, "median_value": value, "uncertainty": uncertainty}


def test_inverse_depth_pair_uses_higher_as_nearer_and_fails_closed():
    clear = probe.pairwise_inverse_depth_probes([
        _stat("near", 2.0, 0.01),
        _stat("far", 1.0, 0.01),
    ])[0]
    assert clear["decision"] == "directional_probe"
    assert clear["nearer"] == "near"
    assert clear["value_semantics"] == "relative_inverse_depth"

    weak = probe.pairwise_inverse_depth_probes([
        _stat("first", 1.02, 0.1),
        _stat("second", 1.0, 0.1),
    ])[0]
    assert weak["decision"] == "indeterminate"
    assert weak["nearer"] is None
    assert "relative_margin" in weak["failed_thresholds"]
    assert "uncertainty_ratio" in weak["failed_thresholds"]


def test_inverse_depth_stats_preserve_nonmetric_semantics():
    values = np.full((100, 100), 10.0, dtype=np.float32)
    entity = {"label": "object", "bbox_2d": [100, 100, 900, 900]}
    stats = probe.inverse_depth_entity_stats(values, entity, [])
    assert stats["median_value"] == 10.0
    assert stats["value_semantics"] == "relative_inverse_depth"
    assert "median_depth_m" not in stats


def test_dpt_model_validation_is_hash_bound(tmp_path: Path, monkeypatch):
    contents = {
        "config.json": '{"model_type":"dpt","architectures":["DPTForDepthEstimation"]}',
        "preprocessor_config.json": (
            '{"image_processor_type":"DPTImageProcessor",'
            '"size":{"height":384,"width":384}}'
        ),
        "pytorch_model.bin": "fixture",
    }
    expected = {}
    for name, value in contents.items():
        path = tmp_path / name
        path.write_text(value, encoding="utf-8")
        expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(probe, "EXPECTED_MODEL_FILES", expected)
    manifest = probe.validate_model_artifact(tmp_path)
    assert manifest["nearer_rule"] == "higher_value_is_nearer"

    (tmp_path / "pytorch_model.bin").write_text("tampered", encoding="utf-8")
    with pytest.raises(SystemExit, match="hash mismatch"):
        probe.validate_model_artifact(tmp_path)


def test_dpt_interpreter_is_pinned(monkeypatch):
    monkeypatch.setattr(probe, "REQUIRED_INTERPRETER", Path(sys.executable))
    assert probe.validate_required_interpreter() == str(Path(sys.executable).resolve())
    with pytest.raises(SystemExit, match="pinned torch-2.6 interpreter"):
        probe.validate_required_interpreter("/usr/bin/python3")
