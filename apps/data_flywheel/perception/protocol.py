from __future__ import annotations

from typing import Protocol, List, Optional, Set

from PIL import Image

from apps.data_flywheel.types import Detection, Segmentation

CAPABILITIES: Set[str] = {"text_to_bbox", "text_to_mask", "box_to_mask", "caption"}


class PerceptionError(Exception):
    """Base exception for perception API errors."""


class PerceptionBackendError(PerceptionError):
    def __init__(self, backend_name: str, cause: Exception):
        self.backend_name = backend_name
        self.cause = cause
        super().__init__(f"Backend '{backend_name}' failed: {cause}")


class UnsupportedCapabilityError(PerceptionError):
    def __init__(self, backend_name: str, capability: str):
        self.backend_name = backend_name
        self.capability = capability
        super().__init__(f"Backend '{backend_name}' does not support capability '{capability}'")


class MaskPersistenceError(PerceptionError):
    def __init__(self, path: str, cause: Exception):
        self.path = path
        self.cause = cause
        super().__init__(f"Failed to persist mask to {path}: {cause}")


class PerceptionBackend(Protocol):
    name: str

    def capabilities(self) -> Set[str]:
        ...

    def detect(self, image: Image.Image, categories: List[str], **kwargs) -> List[Detection]:
        ...

    def segment(
        self,
        image: Image.Image,
        *,
        bbox: Optional[List[float]] = None,
        text_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[Segmentation]:
        ...

    def health(self) -> bool:
        ...
