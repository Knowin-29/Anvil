from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import Iterator, List, Set

from PIL import Image

from apps.data_flywheel.api_detector import detect_objects_api
from apps.data_flywheel.captioner import _caption_dashscope
from apps.data_flywheel.perception.protocol import (
    PerceptionBackend,
    PerceptionBackendError,
    UnsupportedCapabilityError,
)
from apps.data_flywheel.types import Detection


@contextmanager
def _image_path(image: Image.Image | str) -> Iterator[str]:
    if isinstance(image, str):
        yield image
        return

    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        tmp_path = tmp.name
        tmp.close()
        image.save(tmp_path, format="PNG")
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@contextmanager
def _dashscope_api_key(api_key: str | None) -> Iterator[None]:
    if api_key is None:
        yield
        return

    previous = os.environ.get("DASHSCOPE_API_KEY")
    os.environ["DASHSCOPE_API_KEY"] = api_key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DASHSCOPE_API_KEY", None)
        else:
            os.environ["DASHSCOPE_API_KEY"] = previous


class DashscopeBackend(PerceptionBackend):
    name = "dashscope"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def capabilities(self) -> Set[str]:
        return {"text_to_bbox", "caption"}

    def detect(self, image: Image.Image | str, categories: List[str], **kwargs) -> List[Detection]:
        try:
            with _image_path(image) as image_path, _dashscope_api_key(self.api_key):
                raw_detections = detect_objects_api(image_path, categories=categories)
        except Exception as exc:
            raise PerceptionBackendError(self.name, exc) from exc

        detections: List[Detection] = []
        for item in raw_detections:
            bbox = item.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            try:
                normalized_bbox = [float(coord) for coord in bbox[:4]]
                confidence = float(item.get("confidence", item.get("score", 1.0)))
            except (TypeError, ValueError):
                continue
            detections.append(
                Detection(
                    class_name=str(item.get("label", "")),
                    bbox=normalized_bbox,
                    confidence=confidence,
                    source=self.name,
                )
            )
        return detections

    def caption(self, image: Image.Image | str) -> str | None:
        try:
            if isinstance(image, str):
                with _dashscope_api_key(self.api_key):
                    return _caption_dashscope(image)
            with _image_path(image) as image_path, _dashscope_api_key(self.api_key):
                return _caption_dashscope(image_path)
        except Exception as exc:
            raise PerceptionBackendError(self.name, exc) from exc

    def segment(self, image: Image.Image | str, *, bbox=None, text_prompt=None, **kwargs):
        raise UnsupportedCapabilityError(self.name, "segment")

    def health(self) -> bool:
        return bool(self.api_key or os.environ.get("DASHSCOPE_API_KEY"))
