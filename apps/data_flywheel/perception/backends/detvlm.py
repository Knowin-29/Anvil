from __future__ import annotations

from typing import List, Set

from PIL import Image

from apps.data_flywheel.detvlm_adapter import DetVLMAdapter
from apps.data_flywheel.perception.protocol import (
    PerceptionBackend,
    PerceptionBackendError,
    UnsupportedCapabilityError,
)
from apps.data_flywheel.types import Detection


class DetVLMBackend(PerceptionBackend):
    name = "detvlm"

    def __init__(self, base_url: str = "YOUR_DETVLM_URL"):
        self._adapter = DetVLMAdapter(base_url=base_url)

    def capabilities(self) -> Set[str]:
        return {"text_to_bbox"}

    def detect(self, image: Image.Image, categories: List[str], **kwargs) -> List[Detection]:
        try:
            result = self._adapter.inference(images=image, task="detection", categories=categories)[0]
        except Exception as exc:
            raise PerceptionBackendError(self.name, exc) from exc

        detections: List[Detection] = []
        for class_name, items in result.get("extracted_predictions", {}).items():
            for item in items:
                if item.get("type") != "box":
                    continue
                coords = item.get("coords", [])
                if not isinstance(coords, list) or len(coords) < 4:
                    continue
                try:
                    bbox = [float(coord) for coord in coords[:4]]
                except (TypeError, ValueError):
                    continue
                confidence = item.get("confidence", item.get("score", 1.0))
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 1.0
                detections.append(
                    Detection(
                        class_name=class_name,
                        bbox=bbox,
                        confidence=confidence,
                        source=self.name,
                    )
                )
        return detections

    def segment(self, image: Image.Image, *, bbox=None, text_prompt=None, **kwargs):
        raise UnsupportedCapabilityError(self.name, "segment")

    def health(self) -> bool:
        try:
            payload = self._adapter.health()
            status = str(payload.get("status", "")).strip().lower()
            if status in {"ok", "healthy"}:
                return True
            return bool(payload.get("model_loaded"))
        except Exception:
            return False
