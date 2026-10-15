"""本地 VLM 推理服务 — 使用 transformers 部署 Qwen3.6-27B。

提供 OpenAI 兼容的 /v1/chat/completions 接口。
使用 CUDA_VISIBLE_DEVICES 控制 GPU 分配。
"""

# Monkey-patch: 绕过 transformers 5.x 的 flash_attn 检测 bug
import importlib.metadata
_orig_packages_distributions = None
try:
    from transformers.utils import import_utils
    if hasattr(import_utils, 'PACKAGE_DISTRIBUTION_MAPPING'):
        if 'flash_attn' not in import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            import_utils.PACKAGE_DISTRIBUTION_MAPPING['flash_attn'] = ['flash-attn', 'flash_attn']
except Exception:
    pass

import base64
import io
import json
import logging
import os
import time
import re
import threading
from typing import Optional

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Inference lock - ensures only one request uses GPU at a time
_inference_lock = threading.Lock()

# #region agent log
import json as _json
_LOG_PATH = "/path/to/workspace/foundation/.cursor/debug-b68d1f.log"
def _dbg(msg, data=None, hyp="H1"):
    import time; entry = {"sessionId":"b68d1f","timestamp":int(time.time()*1000),"location":"local_vlm_server.py","message":msg,"data":data or {},"hypothesisId":hyp}
    with open(_LOG_PATH, "a") as f: f.write(_json.dumps(entry)+"\n")
_dbg("imports_ok", {"torch": torch.__version__, "model_path": os.environ.get("VLM_MODEL_PATH","")})
# #endregion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("local_vlm")

app = Flask(__name__)

MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "/path/to/environment/hf_models/Qwen/Qwen3.6-27B")
PORT = int(os.environ.get("VLM_PORT", "10301"))

# Global model/processor
model = None
processor = None


def load_model():
    global model, processor
    logger.info(f"Loading model from {MODEL_PATH}...")
    # #region agent log
    _dbg("load_model_start", {"model_path": MODEL_PATH, "exists": os.path.exists(MODEL_PATH)}, "H1")
    # #endregion
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    elapsed = time.time() - t0
    # #region agent log
    _dbg("load_model_done", {"elapsed_s": round(elapsed, 1), "device_map": str(getattr(model, 'hf_device_map', 'unknown'))[:200]}, "H1")
    # #endregion
    logger.info(f"Model loaded in {elapsed:.1f}s. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")


def decode_image(image_url: str) -> Optional[Image.Image]:
    """Decode base64 data URL or file path to PIL Image."""
    if image_url.startswith("data:"):
        match = re.match(r"data:image/\w+;base64,(.*)", image_url)
        if match:
            img_data = base64.b64decode(match.group(1))
            return Image.open(io.BytesIO(img_data)).convert("RGB")
    elif os.path.exists(image_url):
        return Image.open(image_url).convert("RGB")
    return None


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 2048)
    temperature = data.get("temperature", 0.3)

    # Extract text and images from messages
    text_parts = []
    images = []

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item["text"])
                elif item.get("type") == "image_url":
                    img_url = item["image_url"]["url"]
                    img = decode_image(img_url)
                    if img:
                        images.append(img)

    prompt_text = "\n".join(text_parts)

    # Build conversation for Qwen3.6 processor
    # Qwen3.6 accepts {"type": "image", "url": "..."} or {"type": "image", "image": PIL.Image}
    conversation = [{"role": "user", "content": []}]
    for img in images:
        conversation[0]["content"].append({"type": "image", "image": img})
    conversation[0]["content"].append({"type": "text", "text": prompt_text})

    try:
        with _inference_lock:
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": temperature > 0,
                }
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 0.9

                output_ids = model.generate(**inputs, **gen_kwargs)

            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[0][input_len:]
            response_text = processor.decode(generated, skip_special_tokens=True)

        return jsonify({
            "id": f"local-{int(time.time())}",
            "object": "chat.completion",
            "model": os.path.basename(MODEL_PATH),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(generated),
                "total_tokens": input_len + len(generated),
            },
        })
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "data": [{"id": os.path.basename(MODEL_PATH), "object": "model"}]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": os.path.basename(MODEL_PATH), "port": PORT})


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
