from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import probe_depth_da2_v1 as probe
from trains.flywheel import prompts, prompts_ext
from trains.flywheel.contracts import grounding


def _entity(label: str, box: list[int]) -> dict:
    return {"label": label, "bbox_2d": box}


def _stat(label: str, depth: float, uncertainty: float) -> dict:
    return {
        "label": label,
        "median_depth_m": depth,
        "uncertainty_m": uncertainty,
    }


def test_robust_entity_depth_excludes_smaller_overlap_from_large_box():
    depth = np.full((100, 100), 5.0, dtype=np.float32)
    depth[40:60, 40:60] = 1.0
    large = _entity("support", [100, 100, 900, 900])
    small = _entity("object", [400, 400, 600, 600])

    large_stats = probe.robust_entity_depth(depth, large, [small])
    small_stats = probe.robust_entity_depth(depth, small, [large])

    assert large_stats["median_depth_m"] == 5.0
    assert large_stats["excluded_smaller_or_equal_boxes"] == 1
    assert small_stats["median_depth_m"] == 1.0
    assert small_stats["excluded_smaller_or_equal_boxes"] == 0


def test_robust_entity_depth_fails_closed_on_insufficient_unique_pixels():
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    target = _entity("target", [100, 100, 900, 900])
    overlap = _entity("overlap", [100, 100, 900, 900])
    with pytest.raises(ValueError, match="insufficient uncontaminated"):
        probe.robust_entity_depth(depth, target, [overlap])


def test_pairwise_probe_requires_margin_and_uncertainty_thresholds():
    clear = probe.pairwise_depth_probes([
        _stat("near", 1.0, 0.02),
        _stat("far", 2.0, 0.03),
    ])[0]
    assert clear["decision"] == "directional_probe"
    assert clear["nearer"] == "near"

    small_margin = probe.pairwise_depth_probes([
        _stat("first", 1.0, 0.005),
        _stat("second", 1.02, 0.005),
    ])[0]
    assert small_margin["decision"] == "indeterminate"
    assert small_margin["nearer"] is None
    assert "absolute_margin" in small_margin["failed_thresholds"]

    uncertain = probe.pairwise_depth_probes([
        _stat("first", 1.0, 0.5),
        _stat("second", 1.3, 0.5),
    ])[0]
    assert uncertain["decision"] == "indeterminate"
    assert "uncertainty_ratio" in uncertain["failed_thresholds"]


def test_recover_trace_groundings_replays_exact_detection_and_cross_check():
    image = ROOT / "artifacts/upload_review_500_v2/images/kit-066-cf951e8ea8.jpg"
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    raw = [
        {"label": "black pot", "bbox_2d": [311, 593, 692, 1000]},
        {"label": "red chair", "bbox_2d": [809, 600, 1000, 875]},
    ]
    proposed = [grounding(item["label"], item["bbox_2d"]) for item in raw]
    common = {
        "image_sha256": image_sha,
        "image_path_local_only": str(image.resolve()),
        "error_type": "",
    }
    rows = [
        {**common, "prompt": prompts_ext.DETECTION, "response": raw},
        {
            **common,
            "prompt": prompts.cross_check_set(proposed),
            "response": [
                {"index": 0, "match": True, "confidence": 0.95},
                {"index": 1, "match": True, "confidence": 0.90},
            ],
        },
    ]
    candidate = {
        "image_path": str(image),
        "image_sha256": image_sha,
        "entities": None,
    }
    recovered = probe.recover_trace_groundings([candidate], rows)
    assert recovered[0]["entities"] == raw
    assert recovered[0]["grounding_origin"] == "frozen_v43_run_audit"

    tampered = copy.deepcopy(rows)
    tampered[1]["prompt"] += " changed"
    with pytest.raises(SystemExit, match="expected one frozen cross-check"):
        probe.recover_trace_groundings([candidate], tampered)


def test_model_artifact_validation_is_hash_bound(tmp_path: Path, monkeypatch):
    contents = {
        "config.json": (
            '{"model_type":"depth_anything","depth_estimation_type":"metric",'
            '"max_depth":20,"architectures":["DepthAnythingForDepthEstimation"]}'
        ),
        "preprocessor_config.json": (
            '{"image_processor_type":"DPTImageProcessor",'
            '"size":{"height":518,"width":518}}'
        ),
        "model.safetensors": "fixture",
    }
    expected = {}
    for name, value in contents.items():
        path = tmp_path / name
        path.write_text(value, encoding="utf-8")
        expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(probe, "EXPECTED_MODEL_FILES", expected)

    manifest = probe.validate_model_artifact(tmp_path)
    assert manifest["depth_estimation_type"] == "metric"

    (tmp_path / "model.safetensors").write_text("tampered", encoding="utf-8")
    with pytest.raises(SystemExit, match="hash mismatch"):
        probe.validate_model_artifact(tmp_path)


def test_probe_output_is_restricted_to_artifacts(tmp_path: Path):
    allowed = ROOT / "artifacts/depth_da2_v1/test-only"
    assert probe._require_artifact_output(allowed) == allowed.resolve()
    with pytest.raises(SystemExit, match="must stay under"):
        probe._require_artifact_output(tmp_path / "outside")
