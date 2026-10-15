from __future__ import annotations

import base64
import io
import os
import re
from typing import List, Optional, Set

import requests
from PIL import Image

from apps.data_flywheel.perception.grasp_point import grasp_point_from_mask
from apps.data_flywheel.perception.mask_io import save_mask_png
from apps.data_flywheel.perception.protocol import PerceptionBackend, PerceptionBackendError
from apps.data_flywheel.types import Detection, Segmentation


class SAM3Backend(PerceptionBackend):
    name = 'sam3'

    def __init__(
        self,
        base_url: str = 'YOUR_SAM3_URL',
        mask_output_dir: str = 'data/output/masks',
        timeout_seconds: int = 60,
    ):
        if os.path.isabs(mask_output_dir):
            raise ValueError(
                f'mask_output_dir must be repo-root relative, got absolute path: {mask_output_dir}'
            )
        self.base_url = base_url.rstrip('/')
        self.mask_output_dir = mask_output_dir
        self.timeout_seconds = timeout_seconds

    def capabilities(self) -> Set[str]:
        return {'text_to_bbox', 'text_to_mask', 'box_to_mask'}

    def detect(self, image: Image.Image, categories: List[str], **kwargs) -> List[Detection]:
        detections: List[Detection] = []
        for category in categories:
            raw = self._call_segment(image, text_prompt=category)
            for item in raw.get('detections', []):
                bbox = self._coerce_bbox(item.get('bbox'))
                if bbox is None:
                    continue
                detections.append(
                    Detection(
                        class_name=category,
                        bbox=bbox,
                        confidence=self._coerce_confidence(item.get('score', 1.0)),
                        source=self.name,
                    )
                )
        return detections

    def segment(
        self,
        image: Image.Image,
        *,
        bbox: Optional[List[float]] = None,
        text_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[Segmentation]:
        image_id = kwargs['image_id']
        class_name = kwargs['class_name']
        det_idx = kwargs['det_idx']
        raw = self._call_segment(image, bbox=bbox, text_prompt=text_prompt)

        segmentations: List[Segmentation] = []
        for item in raw.get('detections', []):
            mask_b64 = item.get('mask')
            det_bbox = self._coerce_bbox(item.get('bbox', bbox))
            if mask_b64 is None or det_bbox is None:
                continue
            mask_file = self._persist_mask(mask_b64, image_id, class_name, det_idx)
            mask_polygon = self._decode_mask_polygon(mask_b64)
            detection = Detection(
                class_name=class_name,
                bbox=det_bbox,
                confidence=self._coerce_confidence(item.get('score', 1.0)),
                source=self.name,
            )
            segmentations.append(
                Segmentation(
                    detection=detection,
                    mask_polygon=mask_polygon,
                    mask_file=mask_file,
                    grasp_point=grasp_point_from_mask(mask_polygon),
                )
            )
        return segmentations

    def _call_segment(
        self,
        image: Image.Image,
        bbox: Optional[List[float]] = None,
        text_prompt: Optional[str] = None,
    ) -> dict:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        payload = {'image': base64.b64encode(buffer.getvalue()).decode('utf-8')}
        if text_prompt is not None:
            payload['text_prompt'] = text_prompt
        if bbox is not None:
            payload['box_prompt'] = [float(coord) for coord in bbox]
        try:
            response = requests.post(
                f'{self.base_url}/segment',
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise PerceptionBackendError(self.name, exc) from exc

    def _persist_mask(self, mask_b64: str, image_id: str, class_name: str, det_idx: int) -> str:
        safe_class = re.sub(r'[^\w]+', '_', class_name)
        rel_path = f'{self.mask_output_dir}/{image_id}_{safe_class}_{det_idx}.png'
        save_mask_png(mask_b64, rel_path)
        return rel_path

    @staticmethod
    def _coerce_bbox(bbox: object) -> Optional[List[float]]:
        if not isinstance(bbox, list) or len(bbox) < 4:
            return None
        try:
            return [float(coord) for coord in bbox[:4]]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_confidence(score: object) -> float:
        try:
            return float(score)
        except (TypeError, ValueError):
            return 1.0

    def _decode_mask_polygon(self, mask_b64: str) -> List[List[float]]:
        import cv2
        import numpy as np

        image = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert('L')
        mask = (np.array(image) > 128).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.002 * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        return [[float(point[0][0]), float(point[0][1])] for point in simplified]

    def health(self) -> bool:
        try:
            response = requests.get(f'{self.base_url}/health', timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
