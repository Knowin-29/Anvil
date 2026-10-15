import pytest
from PIL import Image

from apps.data_flywheel.scene_understanding.vlm.backend import (
    DashscopeVLMBackend,
    MockVLMBackend,
    RemoteVLMBackend,
    VLMBackendError,
)


def _image():
    return Image.new("RGB", (8, 8), color="white")


def test_mock_vlm_returns_canned_response():
    backend = MockVLMBackend(canned_response='{"en": "mock"}')

    assert backend.ask(_image(), "describe it") == '{"en": "mock"}'


def test_dashscope_vlm_ask_calls_underlying_client(monkeypatch):
    calls = []

    class FakeClient:
        def ask(self, image, question, max_tokens=512):
            calls.append((image.size, question, max_tokens))
            return "ok"

    backend = DashscopeVLMBackend(api_key="test-key", client=FakeClient())
    result = backend.ask(_image(), "what is this?", max_tokens=128)

    assert result == "ok"
    assert calls == [((8, 8), "what is this?", 128)]


def test_dashscope_vlm_error_wraps_as_vlm_backend_error(monkeypatch):
    class FakeClient:
        def ask(self, image, question, max_tokens=512):
            raise RuntimeError("boom")

    backend = DashscopeVLMBackend(api_key="test-key", client=FakeClient())

    with pytest.raises(VLMBackendError) as exc:
        backend.ask(_image(), "question")

    assert "dashscope" in str(exc.value)


def test_remote_vlm_placeholder_raises_not_implemented():
    backend = RemoteVLMBackend(base_url="http://example.com")

    with pytest.raises(NotImplementedError):
        backend.ask(_image(), "question")
