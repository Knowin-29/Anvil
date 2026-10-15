#!/usr/bin/env python3
"""Smoke-check a local vLLM OpenAI-compatible chat endpoint."""
from __future__ import annotations

import argparse
import json
import sys

import requests


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local vLLM chat completions endpoint")
    parser.add_argument("--base-url", default="http://localhost:10301/v1")
    parser.add_argument("--model", default="/path/to/workspace/foundation/contributor/hf_models/Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    url = chat_url(args.base_url)
    try:
        resp = requests.post(url, json=payload, timeout=args.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
    except Exception as exc:
        print(f"ERROR: vLLM health check failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, "url": url, "model": args.model, "content": content}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
