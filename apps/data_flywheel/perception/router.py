from __future__ import annotations

from typing import List, Optional

from apps.data_flywheel.perception.protocol import PerceptionBackend, UnsupportedCapabilityError
from apps.data_flywheel.types import Detection, Segmentation


class PerceptionRouter:
    def __init__(self, backends: List[PerceptionBackend], mask_output_dir: str = 'data/output/masks'):
        self.backends = list(backends)
        self.mask_output_dir = mask_output_dir
        self._by_capability = {
            'text_to_bbox': [backend for backend in self.backends if 'text_to_bbox' in backend.capabilities()],
            'text_to_mask': [backend for backend in self.backends if 'text_to_mask' in backend.capabilities()],
            'box_to_mask': [backend for backend in self.backends if 'box_to_mask' in backend.capabilities()],
            'caption': [backend for backend in self.backends if 'caption' in backend.capabilities()],
        }

    @classmethod
    def from_config(cls, config) -> 'PerceptionRouter':
        backends: List[PerceptionBackend] = []
        if getattr(config, 'detvlm_enabled', False):
            from apps.data_flywheel.perception.backends.detvlm import DetVLMBackend
            backends.append(DetVLMBackend(base_url=config.detvlm_url))
        if getattr(config, 'dashscope_enabled', False):
            from apps.data_flywheel.perception.backends.dashscope import DashscopeBackend
            backends.append(DashscopeBackend(api_key=config.dashscope_api_key))
        if getattr(config, 'sam3_enabled', False):
            from apps.data_flywheel.perception.backends.sam3 import SAM3Backend
            backends.append(SAM3Backend(base_url=config.sam3_url, mask_output_dir=config.mask_output_dir))
        if getattr(config, 'local_vlm_enabled', False):
            from apps.data_flywheel.perception.backends.local_vlm import LocalVLMBackend
            backends.append(LocalVLMBackend(base_url=config.local_vlm_url))
        return cls(backends=backends, mask_output_dir=config.mask_output_dir)

    def detect(
        self,
        image,
        image_id: str,
        categories: List[str],
        prefer: Optional[str] = None,
    ) -> List[Detection]:
        backends = self._by_capability['text_to_bbox']
        if not backends:
            raise UnsupportedCapabilityError('router', 'text_to_bbox')
        backend = self._pick(backends, prefer)
        return backend.detect(image, categories, image_id=image_id)

    def segment_bboxes(self, image, image_id: str, detections: List[Detection]) -> List[Segmentation]:
        backends = self._by_capability['box_to_mask']
        if not backends:
            raise UnsupportedCapabilityError('router', 'box_to_mask')
        backend = backends[0]
        class_counts: dict[str, int] = {}
        results: List[Segmentation] = []
        for detection in detections:
            det_idx = class_counts.get(detection.class_name, 0)
            class_counts[detection.class_name] = det_idx + 1
            segmentations = backend.segment(
                image,
                bbox=detection.bbox,
                image_id=image_id,
                class_name=detection.class_name,
                det_idx=det_idx,
            )
            if segmentations:
                first = segmentations[0]
                results.append(
                    Segmentation(
                        detection=detection,
                        mask_polygon=first.mask_polygon,
                        mask_file=first.mask_file,
                        grasp_point=first.grasp_point,
                    )
                )
        return results

    def detect_and_segment(self, image, image_id: str, categories: List[str]) -> List[Segmentation]:
        detections = self.detect(image, image_id, categories)
        return self.segment_bboxes(image, image_id, detections)

    def _pick(self, backends: List[PerceptionBackend], prefer: Optional[str]):
        if prefer is None:
            return backends[0]
        for backend in backends:
            if backend.name == prefer:
                return backend
        raise ValueError(f"Preferred backend '{prefer}' not registered")
