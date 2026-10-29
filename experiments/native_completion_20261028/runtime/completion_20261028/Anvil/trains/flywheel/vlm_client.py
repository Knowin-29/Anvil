#!/usr/bin/env python3
"""Thin OpenAI-compatible multimodal client for the Anvil data flywheel.

The endpoint, model, and credential are supplied through environment variables.
Responses use native thousandth-scale coordinates. JSON extraction handles
markdown fences and thinking blocks; requests use backoff and an on-disk cache.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
import urllib.request
import urllib.parse
from pathlib import Path

_URL = os.environ.get("GSA_VLM_URL", "")
_KEY = os.environ.get("GSA_VLM_KEY", "")
_MODEL = os.environ.get("GSA_VLM_MODEL", "Qwen3.5-397B-A17B")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE = Path(os.environ.get("GSA_VLM_CACHE", str(_REPO_ROOT / "logs/vlm_cache")))
_CACHE.mkdir(parents=True, exist_ok=True)
_REQUEST_LOG = os.environ.get("GSA_VLM_REQUEST_LOG", "")
_RESPONSE_LOG = os.environ.get("GSA_VLM_RESPONSE_LOG", "")
_request_log_lock = threading.Lock()


def _image_data(path: str, max_bytes: int = 4_000_000) -> tuple[str, str, str]:
    """Return base64, detected MIME, and SHA-256 with a hard payload-size gate."""
    data = Path(path).read_bytes()
    if len(data) > max_bytes:
        raise ValueError(f"image exceeds {max_bytes} bytes: {path} ({len(data)} bytes)")
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        fmt = (image.format or "").upper()
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(fmt)
    if mime is None:
        raise ValueError(f"unsupported image format {fmt!r}: {path}")
    return base64.b64encode(data).decode(), mime, hashlib.sha256(data).hexdigest()


def _audit_request(record: dict) -> None:
    """Optionally log request metadata and prompt locally; never log key or base64."""
    _append_audit_record(_REQUEST_LOG, record)


def _append_audit_record(log_path: str, record: dict) -> None:
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _request_log_lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_json(text: str):
    """Best-effort: strip fences/think, then find the first balanced {...}."""
    if not text:
        return None
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    t = t.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    # try whole string, then first {...} / [...] span
    for cand in (t,):
        try:
            return json.loads(cand)
        except Exception:
            pass
    m = re.search(r"[\{\[].*[\}\]]", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _open_request(request, *, timeout=120):
    """Honor configured bypass hosts from both NO_PROXY spellings.

    A platform-provided lowercase list can otherwise hide the teacher host that
    the project's .env explicitly added to uppercase NO_PROXY. Other endpoints
    retain the standard proxy configuration.
    """
    proxies = urllib.request.getproxies()
    bypass = ",".join(filter(None, [proxies.get("no", ""), os.environ.get("NO_PROXY", "")]))
    host = urllib.parse.urlparse(request.full_url).hostname
    if host and urllib.request.proxy_bypass_environment(host, {"no": bypass}):
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


def chat_json(image_path: str, prompt: str, *, model: str | None = None,
              temperature: float = 0.2, max_tokens: int = 512,
              retries: int = 3, use_cache: bool = True):
    """Return complete parsed JSON (or None), with response accounting in the cache."""
    if not _URL:
        raise RuntimeError("GSA_VLM_URL is required")
    model = model or _MODEL
    b64, mime, image_sha = _image_data(image_path)
    key = hashlib.sha256(
        json.dumps(["anvil-v2", _URL, model, temperature, max_tokens, prompt, image_sha],
                   separators=(",", ":")).encode()).hexdigest()
    cf = _CACHE / f"{key}.json"
    if use_cache and cf.is_file():
        try:
            return json.loads(cf.read_text())["parsed"]
        except Exception:
            pass
    body = json.dumps({
        "model": model, "max_tokens": max_tokens, "temperature": temperature,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + b64}},
            {"type": "text", "text": prompt}]}],
    }).encode()
    last = None
    for attempt in range(retries):
        try:
            _audit_request({
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_key": key,
                "attempt": attempt + 1, "endpoint": _URL, "model": model,
                "image_path_local_only": image_path, "image_sha256": image_sha,
                "image_mime": mime, "temperature": temperature,
                "max_tokens": max_tokens, "prompt": prompt,
            })
            req = urllib.request.Request(_URL, data=body, headers={
                "Content-Type": "application/json", "Authorization": "Bearer " + _KEY})
            raw = json.load(_open_request(req, timeout=120))
            content = raw["choices"][0]["message"]["content"]
            finish_reason = raw["choices"][0].get("finish_reason")
            parsed = None if finish_reason == "length" else extract_json(content)
            # Verification calls deliberately bypass the cache, but still need
            # complete response accounting and the original model output.
            _append_audit_record(_RESPONSE_LOG, {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_key": key, "attempt": attempt + 1, "event": "response",
                "image_sha256": image_sha, "model": raw.get("model", model),
                "finish_reason": finish_reason, "usage": raw.get("usage"),
                "content": content, "parsed": parsed,
            })
            if use_cache:
                cf.write_text(json.dumps({"content": content, "parsed": parsed,
                                          "finish_reason": finish_reason,
                                          "usage": raw.get("usage"),
                                          "model": raw.get("model", model)}))
            return parsed
        except Exception as e:  # noqa
            last = e
            _append_audit_record(_RESPONSE_LOG, {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_key": key, "attempt": attempt + 1, "event": "request_error",
                "image_sha256": image_sha, "model": model, "error_type": type(e).__name__,
            })
            time.sleep(min(2 ** attempt, 8))
    return None
