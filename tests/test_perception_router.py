from PIL import Image
import pytest

from apps.data_flywheel.config import FlywheelConfig
from apps.data_flywheel.perception.protocol import UnsupportedCapabilityError
from apps.data_flywheel.perception.router import PerceptionRouter
from apps.data_flywheel.types import Detection, Segmentation


def _fake_image():
    return Image.new('RGB', (8, 8), color='white')


class MockBackend:
    def __init__(self, capabilities, name='mock', detections=None):
        self.name = name
        self._capabilities = set(capabilities)
        self._detections = detections or []
        self.detect_called = False
        self.segment_calls = []

    def capabilities(self):
        return set(self._capabilities)

    def detect(self, image, categories, **kwargs):
        self.detect_called = True
        return list(self._detections)

    def segment(self, image, *, bbox=None, text_prompt=None, **kwargs):
        self.segment_calls.append({
            'bbox': bbox,
            'text_prompt': text_prompt,
            'image_id': kwargs.get('image_id'),
            'class_name': kwargs.get('class_name'),
            'det_idx': kwargs.get('det_idx'),
        })
        detection = Detection(
            class_name=kwargs['class_name'],
            bbox=list(bbox or [0.0, 0.0, 1.0, 1.0]),
            confidence=1.0,
            source=self.name,
        )
        return [
            Segmentation(
                detection=detection,
                mask_polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                mask_file=f"masks/{kwargs['class_name']}_{kwargs['det_idx']}.png",
                grasp_point=[0.5, 0.5],
            )
        ]

    def health(self):
        return True


def test_router_detect_picks_text_to_bbox_backend():
    backend_a = MockBackend({'caption'}, name='captioner')
    backend_b = MockBackend({'text_to_bbox'}, name='detector')
    router = PerceptionRouter([backend_a, backend_b])

    router.detect(_fake_image(), 'img_001', ['pen'])

    assert backend_b.detect_called
    assert not backend_a.detect_called


def test_router_prefer_named_backend():
    backend_a = MockBackend({'text_to_bbox'}, name='b1')
    backend_b = MockBackend({'text_to_bbox'}, name='b2')
    router = PerceptionRouter([backend_a, backend_b])

    router.detect(_fake_image(), 'img_001', ['pen'], prefer='b2')

    assert not backend_a.detect_called
    assert backend_b.detect_called


def test_router_raises_when_no_capable_backend():
    router = PerceptionRouter([MockBackend({'caption'})])

    with pytest.raises(UnsupportedCapabilityError):
        router.detect(_fake_image(), 'img_001', ['pen'])


def test_router_segment_bboxes_assigns_det_idx_per_class():
    segmenter = MockBackend({'box_to_mask'}, name='sam3')
    router = PerceptionRouter([segmenter])
    detections = [
        Detection('pen', [0.0, 0.0, 1.0, 1.0], 1.0, 'detvlm'),
        Detection('pen', [1.0, 1.0, 2.0, 2.0], 1.0, 'detvlm'),
        Detection('pen', [2.0, 2.0, 3.0, 3.0], 1.0, 'detvlm'),
        Detection('card', [3.0, 3.0, 4.0, 4.0], 1.0, 'detvlm'),
        Detection('card', [4.0, 4.0, 5.0, 5.0], 1.0, 'detvlm'),
    ]

    segmentations = router.segment_bboxes(_fake_image(), 'img_001', detections)

    assert len(segmentations) == 5
    assert [(call['class_name'], call['det_idx']) for call in segmenter.segment_calls] == [
        ('pen', 0),
        ('pen', 1),
        ('pen', 2),
        ('card', 0),
        ('card', 1),
    ]


def test_router_detect_and_segment_end_to_end():
    detector = MockBackend(
        {'text_to_bbox'},
        name='detvlm',
        detections=[Detection('pen', [0.0, 0.0, 2.0, 2.0], 0.9, 'detvlm')],
    )
    segmenter = MockBackend({'box_to_mask'}, name='sam3')
    router = PerceptionRouter([detector, segmenter])

    result = router.detect_and_segment(_fake_image(), 'img_001', ['pen'])

    assert len(result) == 1
    assert result[0].detection.class_name == 'pen'
    assert segmenter.segment_calls[0]['det_idx'] == 0


def test_router_from_config_registers_enabled_backends(monkeypatch):
    class FakeDetVLMBackend(MockBackend):
        def __init__(self, base_url=''):
            super().__init__({'text_to_bbox'}, name='detvlm')
            self.base_url = base_url

    class FakeDashscopeBackend(MockBackend):
        def __init__(self, api_key=None):
            super().__init__({'text_to_bbox', 'caption'}, name='dashscope')
            self.api_key = api_key

    class FakeSAM3Backend(MockBackend):
        def __init__(self, base_url='', mask_output_dir='data/output/masks'):
            super().__init__({'text_to_bbox', 'text_to_mask', 'box_to_mask'}, name='sam3')
            self.base_url = base_url
            self.mask_output_dir = mask_output_dir

    class FakeLocalVLMBackend(MockBackend):
        def __init__(self, base_url=''):
            super().__init__({'text_to_bbox'}, name='local_vlm')
            self.base_url = base_url

    monkeypatch.setattr('apps.data_flywheel.perception.backends.detvlm.DetVLMBackend', FakeDetVLMBackend)
    monkeypatch.setattr('apps.data_flywheel.perception.backends.dashscope.DashscopeBackend', FakeDashscopeBackend)
    monkeypatch.setattr('apps.data_flywheel.perception.backends.sam3.SAM3Backend', FakeSAM3Backend)
    monkeypatch.setattr('apps.data_flywheel.perception.backends.local_vlm.LocalVLMBackend', FakeLocalVLMBackend)

    config = FlywheelConfig(
        detvlm_enabled=True,
        dashscope_enabled=True,
        sam3_enabled=True,
        local_vlm_enabled=True,
        detvlm_url='http://detvlm',
        dashscope_api_key='dash-key',
        sam3_url='http://sam3',
        local_vlm_url='http://local',
        mask_output_dir='data/output/masks',
    )

    router = PerceptionRouter.from_config(config)

    assert [backend.name for backend in router.backends] == ['detvlm', 'dashscope', 'sam3', 'local_vlm']


def test_router_from_default_config_registers_no_optional_backends():
    config = FlywheelConfig()
    router = PerceptionRouter.from_config(config)

    assert router.backends == []


def test_router_from_config_skips_disabled_backends(monkeypatch):
    class FakeDetVLMBackend(MockBackend):
        def __init__(self, base_url=''):
            super().__init__({'text_to_bbox'}, name='detvlm')

    monkeypatch.setattr('apps.data_flywheel.perception.backends.detvlm.DetVLMBackend', FakeDetVLMBackend)

    config = FlywheelConfig(detvlm_enabled=True, dashscope_enabled=False, sam3_enabled=False, local_vlm_enabled=False)
    router = PerceptionRouter.from_config(config)

    assert [backend.name for backend in router.backends] == ['detvlm']
