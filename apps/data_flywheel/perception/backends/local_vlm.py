from __future__ import annotations

from typing import List, Set

from PIL import Image

from apps.data_flywheel.perception.protocol import PerceptionBackend, UnsupportedCapabilityError
from apps.data_flywheel.types import Detection


class LocalVLMBackend(PerceptionBackend):
    name = 'local_vlm'

    def __init__(self, base_url: str = 'http://localhost:10301'):
        self.base_url = base_url.rstrip('/')

    def capabilities(self) -> Set[str]:
        return {'text_to_bbox'}

    def detect(self, image: Image.Image, categories: List[str], **kwargs) -> List[Detection]:
        raise NotImplementedError('LocalVLMBackend is a placeholder. Deploy the local VLM first.')

    def segment(self, image: Image.Image, *, bbox=None, text_prompt=None, **kwargs):
        raise UnsupportedCapabilityError(self.name, 'segment')

    def health(self) -> bool:
        return False
