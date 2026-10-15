"""Tests for shared VLMClient local-vLLM compatibility helpers."""

from __future__ import annotations

import threading

from apps.data_flywheel.workers.vlm_client import VLMClient


def test_expand_env_with_default(monkeypatch):
    monkeypatch.delenv("VLLM_MODEL", raising=False)
    assert VLMClient.expand_env("${VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}") == (
        "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    monkeypatch.setenv("VLLM_MODEL", "local/model")
    assert VLMClient.expand_env("${VLLM_MODEL:-fallback}") == "local/model"


def test_chat_url_accepts_v1_and_root_base_url():
    local = VLMClient(base_url="http://localhost:10301/v1", model="m")
    assert local._chat_completions_url() == "http://localhost:10301/v1/chat/completions"

    root = VLMClient(base_url="http://localhost:10301", model="m")
    assert root._chat_completions_url() == "http://localhost:10301/v1/chat/completions"


def test_local_mode_has_no_auth_header():
    client = VLMClient(base_url="http://localhost:10301/v1", model="m", api_key="")
    session = client._get_session()
    assert "Authorization" not in session.headers


def test_api_key_sets_auth_header():
    client = VLMClient(base_url="http://localhost:10301/v1", model="m", api_key="secret")
    session = client._get_session()
    assert session.headers["Authorization"] == "Bearer secret"


def test_session_is_thread_local():
    client = VLMClient(base_url="http://localhost:10301/v1", model="m")
    sessions = [client._get_session()]

    def get_session():
        sessions.append(client._get_session())

    t = threading.Thread(target=get_session)
    t.start()
    t.join()

    assert sessions[0] is not sessions[1]


def test_chat_accepts_multiple_image_paths_and_pil_images(monkeypatch):
    from PIL import Image

    client = VLMClient(base_url="http://localhost:10301/v1", model="m", max_retries=0)
    client._encode_image = lambda path: f"data:image/jpeg;base64,path-{path}"
    client._encode_pil_image = lambda img: "data:image/jpeg;base64,pil"
    posted = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Session:
        def post(self, url, json, timeout):
            posted["url"] = url
            posted["json"] = json
            posted["timeout"] = timeout
            return _Resp()

    monkeypatch.setattr(client, "_get_session", lambda: _Session())

    result = client.chat(
        "describe",
        image_path="original.jpg",
        image_paths=["crop-path.jpg"],
        pil_images=[Image.new("RGB", (8, 8), "white")],
    )

    assert result == "ok"
    content = posted["json"]["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].endswith("path-original.jpg")
    assert content[1]["image_url"]["url"].endswith("path-crop-path.jpg")
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,pil"
    assert content[3]["text"] == "describe"
