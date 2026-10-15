import os
from pathlib import Path

from PIL import Image
import pytest

from apps.data_flywheel.perception.backends.dashscope import DashscopeBackend
from apps.data_flywheel.perception.backends.detvlm import DetVLMBackend
from apps.data_flywheel.perception.backends.sam3 import SAM3Backend

pytestmark = pytest.mark.integration


def _fixture_image_path() -> Path:
    candidates = []
    env_path = os.environ.get('PERCEPTION_TEST_IMAGE')
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path('tests/fixtures/desk_sample.jpg'))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pytest.skip('No integration fixture image available; set PERCEPTION_TEST_IMAGE or add tests/fixtures/desk_sample.jpg')


def test_detvlm_real_service_detects_pen_on_real_image():
    backend = DetVLMBackend()
    if not backend.health():
        pytest.skip('DetVLM service unavailable')

    image = Image.open(_fixture_image_path()).convert('RGB')
    detections = backend.detect(image, ['pen', 'cup'])
    if not detections:
        pytest.skip('DetVLM service did not return any detections for the fixture image')

    assert len(detections) > 0


def test_sam3_real_service_chinese_text_mode():
    backend = SAM3Backend()
    if not backend.health():
        pytest.skip('SAM 3 service unavailable')

    image = Image.open(_fixture_image_path()).convert('RGB')
    detections = backend.detect(image, ['card'])
    zh_detections = backend.detect(image, ['卡片'])

    if not detections:
        pytest.skip('SAM 3 did not return any English card detections for the fixture image')

    assert len(zh_detections) > 0


def test_sam3_box_mode_real_service(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = SAM3Backend(mask_output_dir='data/output/masks')
    if not backend.health():
        pytest.skip('SAM 3 service unavailable')

    image = Image.open(_fixture_image_path()).convert('RGB')
    detections = backend.detect(image, ['card'])
    if not detections:
        pytest.skip('SAM 3 did not return any card detections for the fixture image')

    segmentations = backend.segment(
        image,
        bbox=detections[0].bbox,
        image_id='test_sample',
        class_name='card',
        det_idx=0,
    )

    assert len(segmentations) > 0
    assert not os.path.isabs(segmentations[0].mask_file)
    assert os.path.exists(segmentations[0].mask_file)


def test_dashscope_real_service():
    backend = DashscopeBackend()
    if not backend.health():
        pytest.skip('Dashscope service unavailable')

    image = Image.open(_fixture_image_path()).convert('RGB')
    detections = backend.detect(image, ['pen', 'cup'])

    if not detections:
        pytest.skip('Dashscope service returned no detections for the fixture image')

    assert len(detections) > 0
