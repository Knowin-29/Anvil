from PIL import Image
import pytest

from apps.data_flywheel.perception.protocol import UnsupportedCapabilityError
from apps.data_flywheel.perception.backends.dashscope import DashscopeBackend


def _fake_image():
    return Image.new('RGB', (8, 8), color='white')


def test_dashscope_parses_openai_format(monkeypatch):
    def fake_detect_objects_api(image_path, categories=None):
        assert categories == ['cup']
        return [{'label': 'cup', 'bbox': [1, 2, 3, 4], 'confidence': 0.75}]

    monkeypatch.setattr(
        'apps.data_flywheel.perception.backends.dashscope.detect_objects_api',
        fake_detect_objects_api,
    )

    backend = DashscopeBackend(api_key='test-key')
    detections = backend.detect(_fake_image(), ['cup'])

    assert len(detections) == 1
    assert detections[0].class_name == 'cup'
    assert detections[0].bbox == [1.0, 2.0, 3.0, 4.0]
    assert detections[0].confidence == 0.75
    assert detections[0].source == 'dashscope'


def test_dashscope_segment_raises_unsupported():
    backend = DashscopeBackend(api_key='test-key')

    with pytest.raises(UnsupportedCapabilityError):
        backend.segment(_fake_image(), bbox=[0, 0, 10, 10])
