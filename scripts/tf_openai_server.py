#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/chat/completions server for Qwen3.6-27B (qwen3_5
arch) backed by transformers 5.5.0 + accelerate device_map. Multimodal (image_url
data URIs). Continuous micro-batching worker for throughput. Slow but reliable when
vLLM/sglang lack qwen3_5 kernels.

Env:
  TF_MODEL            model dir
  TF_SERVED_NAME      served-model-name (default Qwen3.6-27B)
  TF_HOST/TF_PORT     bind (default 127.0.0.1:30000)
  TF_MAX_BATCH        micro-batch size (default 4)
  TF_BATCH_WAIT_MS    batch collect window ms (default 25)
  TF_MAX_NEW_TOKENS   hard cap on max_tokens (default 1024)
"""
from __future__ import annotations
import base64
import io
import json
import hashlib
import os
import re
import threading
import time
import uuid
from queue import Queue, Empty
from collections import deque
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
MODEL_PATH = os.environ["TF_MODEL"]
SERVED_NAME = os.environ.get("TF_SERVED_NAME", "Qwen3.6-27B")
HOST = os.environ.get("TF_HOST", "127.0.0.1")
PORT = int(os.environ.get("TF_PORT", "30000"))
MAX_BATCH = int(os.environ.get("TF_MAX_BATCH", "4"))
BATCH_WAIT_MS = int(os.environ.get("TF_BATCH_WAIT_MS", "25"))
MAX_NEW_TOKENS_CAP = int(os.environ.get("TF_MAX_NEW_TOKENS", "1024"))

_req_q: "Queue" = Queue()
_data_uri_re = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<b64>.+)$", re.S)
print(f"[tf-server] loading processor from {MODEL_PATH}", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor.tokenizer.padding_side = "left"
if os.environ.get("TF_MAX_PIXELS"):
    processor.image_processor.size["longest_edge"] = int(os.environ["TF_MAX_PIXELS"])
print("[tf-server] loading model (device_map=auto, bf16)...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map=os.environ.get("TF_DEVICE_MAP", "auto"),
    trust_remote_code=True,
)
adapters = {}
if os.environ.get("TF_ADAPTERS_FILE"):
    from peft import PeftModel
    adapters = json.loads(Path(os.environ["TF_ADAPTERS_FILE"]).read_text())
    if not adapters:
        raise ValueError("adapter manifest must not be empty")
    for index, (name, spec) in enumerate(adapters.items()):
        path = Path(spec["path"])
        actual = hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()
        if actual != spec["sha256"]:
            raise ValueError("adapter hash mismatch: " + name)
        if index == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name, is_trainable=False)
        else:
            model.load_adapter(path, adapter_name=name, is_trainable=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
model.eval()
print("[tf-server] model ready. hf_device_map:", getattr(model, "hf_device_map", None), flush=True)
def _load_image_from_url(url: str) -> Image.Image:
    m = _data_uri_re.match(url.strip())
    if not m:
        raise ValueError("only data: URIs supported")
    data = base64.b64decode(m.group("b64"))
    return Image.open(io.BytesIO(data)).convert("RGB")


def _to_hf_messages(messages):
    """Convert OpenAI messages -> Qwen chat messages (image objects inline)."""
    out = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        parts = []
        for item in content or []:
            it = item.get("type")
            if it == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            elif it == "image_url":
                url = item["image_url"]["url"] if isinstance(item["image_url"], dict) else item["image_url"]
                parts.append({"type": "image", "image": _load_image_from_url(url)})
        out.append({"role": role, "content": parts})
    return out


def _extract_images(hf_messages):
    imgs = []
    for msg in hf_messages:
        for part in msg["content"]:
            if part.get("type") == "image":
                imgs.append(part["image"])
    return imgs or None


def _build_inputs(job):
    hf_msgs = _to_hf_messages(job["messages"])
    try:
        text = processor.apply_chat_template(
            hf_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        text = processor.apply_chat_template(
            hf_msgs, tokenize=False, add_generation_prompt=True)
    images = _extract_images(hf_msgs)
    return text, images


def _run_single(job):
    return _run_batch([job])[0]


def _run_batch(jobs):
    """Batch only requests with the same model and generation configuration."""
    job = jobs[0]
    if adapters:
        model.set_adapter(job["model"])
        model.eval()
    built = [_build_inputs(item) for item in jobs]
    images = [image for _, group in built for image in (group or [])]
    inputs = processor(
        text=[text for text, _ in built], images=images or None, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    gen_kw = dict(
        max_new_tokens=min(int(job["max_tokens"]), MAX_NEW_TOKENS_CAP),
        do_sample=job["temperature"] > 0.0,
    )
    if job["temperature"] > 0.0:
        gen_kw["temperature"] = job["temperature"]
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kw)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [] if eos_ids is None else ([eos_ids] if isinstance(eos_ids, int) else eos_ids)
    results = []
    for index, tokens in enumerate(trimmed.tolist()):
        end = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), None)
        tokens = tokens[:end] if end is not None else tokens
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        finish = "stop" if end is not None or len(tokens) < gen_kw["max_new_tokens"] else "length"
        results.append({"text": text, "n_in": int(inputs["attention_mask"][index].sum()),
                        "n_out": len(tokens), "finish_reason": finish,
                        "batch_size": len(jobs), "batch_seconds": seconds})
    return results


def _worker():
    """One model execution at a time, with compatible requests micro-batched."""
    pending = deque()
    while True:
        job = pending.popleft() if pending else _req_q.get()
        jobs = [job]
        signature = lambda item: (item["model"], item["max_tokens"], item["temperature"])
        deadline = time.monotonic() + BATCH_WAIT_MS / 1000
        while len(jobs) < MAX_BATCH:
            try:
                candidate = _req_q.get(timeout=max(0, deadline-time.monotonic()))
            except Empty:
                break
            if signature(candidate) == signature(job):
                jobs.append(candidate)
            else:
                pending.append(candidate)
        try:
            results = _run_batch(jobs)
            for item, result in zip(jobs, results):
                item["result"] = result
        except Exception as e:  # noqa
            for item in jobs:
                item["error"] = repr(e)
        finally:
            for item in jobs:
                item["event"].set()


threading.Thread(target=_worker, daemon=True).start()
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": name, "object": "model"}
                                           for name in (adapters or {SERVED_NAME: None})]}


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    requested = body.get("model", SERVED_NAME)
    if requested not in (adapters or {SERVED_NAME: None}):
        return JSONResponse(status_code=400, content={"error": "unknown served model"})
    job = {
        "model": requested,
        "messages": body.get("messages", []),
        "max_tokens": body.get("max_tokens", 512),
        "temperature": float(body.get("temperature", 0.2)),
        "event": threading.Event(),
    }
    _req_q.put(job)
    job["event"].wait()
    if job.get("error"):
        return JSONResponse(status_code=500, content={"error": job["error"]})
    r = job["result"]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": r["text"]},
            "finish_reason": r["finish_reason"],
        }],
        "usage": {
            "prompt_tokens": r["n_in"],
            "completion_tokens": r["n_out"],
            "total_tokens": r["n_in"] + r["n_out"],
        },
        "inference": {"batch_size": r["batch_size"], "batch_seconds": r["batch_seconds"]},
    }
if __name__ == "__main__":
    print(f"[tf-server] serving {SERVED_NAME} on http://{HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
