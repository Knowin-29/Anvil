from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Protocol, runtime_checkable

from PIL import Image


class VLMBackendError(Exception):
    pass


@runtime_checkable
class VLMBackend(Protocol):
    name: str

    def ask(self, image: Image.Image, question: str, max_tokens: int = 512) -> str:
        ...


def _image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@dataclass
class _DashscopeClient:
    api_key: Optional[str] = None
    model: str = "qwen-vl-max"
    endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def ask(self, image: Image.Image, question: str, max_tokens: int = 512) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("dashscope response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return "\n".join(texts).strip()
        raise ValueError("dashscope response missing content")


class DashscopeVLMBackend:
    name = "dashscope"

    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-vl-max", client=None):
        self._client = client or _DashscopeClient(api_key=api_key, model=model)

    def ask(self, image: Image.Image, question: str, max_tokens: int = 512) -> str:
        try:
            return self._client.ask(image=image, question=question, max_tokens=max_tokens)
        except Exception as exc:
            raise VLMBackendError(f"dashscope: {exc}") from exc


class MockVLMBackend:
    name = "mock"

    def __init__(self, canned_response: str = '{"en": "mock", "zh": "假的", "en_contextual": "mock near mock"}'):
        self._canned_response = canned_response

    def ask(self, image: Image.Image, question: str, max_tokens: int = 512) -> str:
        return self._canned_response


class RemoteVLMBackend:
    name = "remote_vlm"

    def __init__(self, base_url: str):
        self.base_url = base_url

    def ask(self, image: Image.Image, question: str, max_tokens: int = 512) -> str:
        raise NotImplementedError("RemoteVLMBackend is a placeholder")
