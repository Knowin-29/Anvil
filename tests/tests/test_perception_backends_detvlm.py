from PIL import Image
import pytest

from apps.data_flywheel.perception.protocol import (
    PerceptionBackendError,
    UnsupportedCapabilityError,
)


def _fake_image():
    return Image.new("RGB", (16, 16), color="white")


def test_detvlm_parses_response(monkeypatch):
    from apps.data_flywheel.perception.backends.detvlm import DetVLMBackend
    from apps.data_flywheel.detvlm_adapter import DetVLMAdapter

    def fake_inference(self, images, task="detection", categories=None, **kwargs):
        assert task == "detection"
        assert categories == ["pen", "cup"]
        assert images.size == (16, 16)
        return [{
            "extracted_predictions": {
                "pen": [
                    {"type": "box", "coords": [10, 20, 30, 40], "confidence": 0.9},
                    {"type": "point", "coords": [1, 2], "confidence": 0.1},
                ],
                "cup": [
                    {"type": "box", "coords": [5, 6, 7], "confidence": 0.7},
                    {"type": "box", "coords": [1, 2, 3, 4], "confidence": 0.8},
                ],
            }
        }]

    monkeypatch.setattr(DetVLMAdapter, "inference", fake_inference)

    backend = DetVLMBackend()
    detections = backend.detect(_fake_image(), ["pen", "cup"])

    assert len(detections) == 2
    assert detections[0].class_name == "pen"
    assert detections[0].bbox == [10.0, 20.0, 30.0, 40.0]
    assert detections[0].confidence == 0.9
    assert detections[0].source == "detvlm"
    assert detections[1].class_name == "cup"
    assert detections[1].bbox == [1.0, 2.0, 3.0, 4.0]
    assert detections[1].source == "detvlm"


def test_detvlm_segment_raises_unsupported():
    from apps.data_flywheel.perception.backends.detvlm import DetVLMBackend

    backend = DetVLMBackend()
    with pytest.raises(UnsupportedCapabilityError) as exc_info:
        backend.segment(_fake_image(), bbox=[0, 0, 10, 10])

    assert "detvlm" in str(exc_info.value)
    assert "segment" in str(exc_info.value)


def test_detvlm_capabilities_are_text_to_bbox_only():
    from apps.data_flywheel.perception.backends.detvlm import DetVLMBackend

    assert DetVLMBackend().capabilities() == {"text_to_bbox"}

