"""Qwen3-32B OpenAI-Compatible API Server (transformers backend).

启动:
    conda run -n roboevolve_vlm_train python services/qwen3_server/server.py

端口: 10302
用途: Verifier 推理 / 文本 Judge
"""

import argparse
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List, Optional

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [qwen3] %(message)s")
logger = logging.getLogger("qwen3_server")

app = Flask(__name__)

MODEL = None
TOKENIZER = None
DEVICE = None
MODEL_NAME = ""
GEN_LOCK = Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)


def load_model(model_path: str, device_map: str = "auto"):
    global MODEL, TOKENIZER, MODEL_NAME
    logger.info(f"Loading {model_path}...")
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    MODEL.eval()
    MODEL_NAME = model_path.split("/")[-1].lower()
    logger.info(f"Model loaded: {MODEL_NAME}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 2048)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)

    # Build prompt from messages
    text = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with GEN_LOCK:
        inputs = TOKENIZER(text, return_tensors="pt").to(MODEL.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = MODEL.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                do_sample=temperature > 0.01,
                pad_token_id=TOKENIZER.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        response_text = TOKENIZER.decode(new_tokens, skip_special_tokens=True)

    # OpenAI format response
    return jsonify({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": len(new_tokens),
            "total_tokens": input_len + len(new_tokens),
        },
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/path/to/environment/hf_models/Qwen/Qwen3-32B")
    parser.add_argument("--port", type=int, default=10302)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    load_model(args.model, device_map=args.device_map)

    print(f"\n{'='*50}")
    print(f"  Qwen3-32B Server")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Model: {args.model}")
    print(f"{'='*50}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
