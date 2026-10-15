import numpy as np
from PIL import Image
from pathlib import Path

from apps.data_flywheel.training_data.phase_c.validators.point_in_mask import PointInMaskValidator
from apps.data_flywheel.types import SceneObjectSample, TrainingSample, ThinkingTrace


def _sample(mask_path: Path, grasp_point_quantized: list[int]) -> TrainingSample:
    return TrainingSample(
        sample_id="sample-1",
        image_id="img-1",
        image_path=str(mask_path),
        image_size=(100, 100),
        data_source="real",
        instruction="pick pen",
        instruction_template_id="tpl",
        instruction_language="en",
        scene_objects=[
            SceneObjectSample(
                id="pen_0",
                class_name="pen",
                bbox_pixel=[10.0, 10.0, 30.0, 30.0],
                bbox_quantized=[100, 100, 300, 300],
                mask_file=str(mask_path),
                grasp_point_quantized=grasp_point_quantized,
            )
        ],
        blocking_dag=[],
        thinking=ThinkingTrace(steps=[]),
        actions=[],
        rex_omni_text="",
    )


def _write_mask(path: Path, foreground: list[tuple[int, int]]) -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    for x, y in foreground:
        mask[y, x] = 255
    Image.fromarray(mask, mode="L").save(path)


def test_point_in_mask_pass(tmp_path):
    mask_path = tmp_path / "mask.png"
    _write_mask(mask_path, [(50, 50)])
    sample = _sample(mask_path, [500, 500])

    result = PointInMaskValidator().check(sample, {})

    assert result.passed is True
    assert result.reason is None


def test_point_outside_mask_fail(tmp_path):
    mask_path = tmp_path / "mask.png"
    _write_mask(mask_path, [(10, 10)])
    sample = _sample(mask_path, [500, 500])

    result = PointInMaskValidator().check(sample, {})

    assert result.passed is False
    assert "outside mask" in result.reason


def test_point_at_boundary_passes(tmp_path):
    mask_path = tmp_path / "mask.png"
    _write_mask(mask_path, [(99, 99)])
    sample = _sample(mask_path, [999, 999])

    result = PointInMaskValidator().check(sample, {})

    assert result.passed is True


def test_mask_file_missing_fails(tmp_path):
    missing_path = tmp_path / "missing.png"
    sample = _sample(missing_path, [500, 500])

    result = PointInMaskValidator().check(sample, {})

    assert result.passed is False
    assert "mask file error" in result.reason
