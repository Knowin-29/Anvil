import base64
import io
import os

from PIL import Image
import pytest

from apps.data_flywheel.perception.backends.sam3 import SAM3Backend


def _fake_image():
    return Image.new('RGB', (16, 16), color='white')


def _fake_png_base64():
    image = Image.new('L', (16, 16), color=0)
    for x in range(4, 12):
        for y in range(4, 12):
            image.putpixel((x, y), 255)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def test_sam3_capabilities():
    assert SAM3Backend().capabilities() == {'text_to_bbox', 'text_to_mask', 'box_to_mask'}


def test_sam3_detect_text_mode(monkeypatch):
    def fake_call_segment(self, image, bbox=None, text_prompt=None):
        assert text_prompt == 'pen'
        assert bbox is None
        return {'detections': [{'bbox': [1, 2, 3, 4], 'score': 0.6}]}

    monkeypatch.setattr(SAM3Backend, '_call_segment', fake_call_segment)

    backend = SAM3Backend()
    detections = backend.detect(_fake_image(), ['pen'])

    assert len(detections) == 1
    assert detections[0].class_name == 'pen'
    assert detections[0].bbox == [1.0, 2.0, 3.0, 4.0]
    assert detections[0].confidence == 0.6
    assert detections[0].source == 'sam3'


def test_sam3_segment_box_mode_produces_mask_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_call_segment(self, image, bbox=None, text_prompt=None):
        assert bbox == [2.0, 2.0, 14.0, 14.0]
        return {
            'detections': [
                {'bbox': [2, 2, 14, 14], 'score': 0.8, 'mask': _fake_png_base64()},
            ]
        }

    monkeypatch.setattr(SAM3Backend, '_call_segment', fake_call_segment)

    backend = SAM3Backend(mask_output_dir='data/output/masks')
    segmentations = backend.segment(
        _fake_image(),
        bbox=[2.0, 2.0, 14.0, 14.0],
        image_id='scene',
        class_name='pen',
        det_idx=0,
    )

    assert len(segmentations) == 1
    segmentation = segmentations[0]
    assert segmentation.mask_file == 'data/output/masks/scene_pen_0.png'
    assert os.path.exists(segmentation.mask_file)
    assert len(segmentation.grasp_point) == 2
    assert segmentation.detection.class_name == 'pen'


def test_sam3_persist_mask_naming_follows_convention(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = SAM3Backend(mask_output_dir='data/output/masks')

    path = backend._persist_mask(
        mask_b64=_fake_png_base64(),
        image_id='desk_001',
        class_name='pen',
        det_idx=2,
    )

    assert path == 'data/output/masks/desk_001_pen_2.png'
    assert not os.path.isabs(path)
    assert os.path.exists(path)


def test_sam3_mask_chinese_class_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = SAM3Backend(mask_output_dir='data/output/masks')

    path = backend._persist_mask(
        mask_b64=_fake_png_base64(),
        image_id='desk_001',
        class_name='笔',
        det_idx=0,
    )

    assert '笔' in path
    assert not os.path.isabs(path)


def test_sam3_reject_absolute_mask_output_dir():
    with pytest.raises(ValueError, match='relative'):
        SAM3Backend(mask_output_dir='/tmp/absolute/path/outside/repo')

def test_sam3_segment_skips_invalid_bbox(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_call_segment(self, image, bbox=None, text_prompt=None):
        return {
            'detections': [
                {'bbox': [1, 2, 3], 'score': 0.5, 'mask': _fake_png_base64()},
                {'bbox': [2, 2, 14, 14], 'score': 0.8, 'mask': _fake_png_base64()},
            ]
        }

    monkeypatch.setattr(SAM3Backend, '_call_segment', fake_call_segment)

    backend = SAM3Backend(mask_output_dir='data/output/masks')
    segmentations = backend.segment(
        _fake_image(),
        bbox=[2.0, 2.0, 14.0, 14.0],
        image_id='scene',
        class_name='pen',
        det_idx=0,
    )

    assert len(segmentations) == 1
    assert segmentations[0].detection.bbox == [2.0, 2.0, 14.0, 14.0]
