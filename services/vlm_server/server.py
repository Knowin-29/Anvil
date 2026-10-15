#!/usr/bin/env python3
"""Qwen2.5-VL-32B OpenAI-compatible inference server.

Uses transformers + accelerate (device_map=auto) to shard across all GPUs.
Exposes /v1/chat/completions endpoint compatible with OpenAI API.

Usage:
    python services/vlm_server/server.py \
        --model /path/to/Qwen2.5-VL-32B-Instruct --port 10301
"""
import argparse
import logging
import time
import uuid
from typing import List

import torch
torch.backends.cudnn.enabled = False  # Avoid cuDNN version mismatch with driver 535
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vlm_server")

app = FastAPI(title="Qwen2.5-VL Server")

MODEL = None
PROCESSOR = None
MODEL_NAME = "qwen2.5-vl-32b"


class ChatMessage(BaseModel):
    role: str
    content: object


class ChatRequest(BaseModel):
    model: str = "qwen2.5-vl-32b"
    messages: List[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage


def load_model(model_path: str):
    """Load Qwen2.5-VL with device_map=auto."""
    global MODEL, PROCESSOR, MODEL_NAME
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    logger.info(f"Loading model from {model_path}...")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")

    PROCESSOR = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    MODEL.eval()
    MODEL_NAME = model_path.split("/")[-1].lower()
    logger.info(f"Model loaded: {MODEL_NAME}")


def build_inputs(messages):
    """Convert OpenAI messages to model inputs."""
    from qwen_vl_utils import process_vision_info

    qwen_msgs = []
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            qwen_msgs.append({
                "role": msg.role,
                "content": [{"type": "text", "text": content}],
            })
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    t = item.get("type", "")
                    if t == "text":
                        parts.append(item)
                    elif t == "image_url":
                        url = item["image_url"]["url"]
                        parts.append({"type": "image", "image": url})
                    elif t == "image":
                        parts.append({"type": "image", "image": item["image"]})
            qwen_msgs.append({"role": msg.role, "content": parts})
        else:
            qwen_msgs.append({
                "role": msg.role,
                "content": [{"type": "text", "text": str(content)}],
            })

    text = PROCESSOR.apply_chat_template(
        qwen_msgs, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(qwen_msgs)
    inputs = PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to("cuda")


import asyncio
import threading

_INFERENCE_LOCK = threading.Lock()


def _sync_inference(messages, max_tokens, temperature, top_p):
    """在锁保护下执行同步推理（避免并发写 GPU）。"""
    with _INFERENCE_LOCK:
        inputs = build_inputs(messages)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = MODEL.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                do_sample=temperature > 0.01,
            )
        gen_ids = out[0][prompt_len:]
        text = PROCESSOR.decode(gen_ids, skip_special_tokens=True)
        return text, prompt_len, len(gen_ids)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        loop = asyncio.get_event_loop()
        text, prompt_len, gen_len = await loop.run_in_executor(
            None, _sync_inference, request.messages, request.max_tokens,
            request.temperature, request.top_p,
        )
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[ChatChoice(
                message=ChatMessage(role="assistant", content=text)
            )],
            usage=Usage(
                prompt_tokens=prompt_len,
                completion_tokens=gen_len,
                total_tokens=prompt_len + gen_len,
            ),
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "gpus": torch.cuda.device_count()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=10301)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    load_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port)
