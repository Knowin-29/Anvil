from PIL import Image
import pytest

from apps.data_flywheel.perception.backends.local_vlm import LocalVLMBackend


def _fake_image():
    return Image.new('RGB', (8, 8), color='white')


def test_local_vlm_detect_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        LocalVLMBackend().detect(_fake_image(), ['pen'])


def test_local_vlm_health_returns_false():
    assert LocalVLMBackend().health() is False
