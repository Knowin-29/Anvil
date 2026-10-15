#!/usr/bin/env python3
"""Verifier Model Server — serve HF grounding/pointing models for cross-verification.

Loads specialist vision models (LocateAnything-3B for grounding, Molmo-7B-D for
pointing/affordance) via transformers + trust_remote_code, and exposes
task-specific endpoints:

  POST /v1/ground    {image_path, query}        -> {boxes: [[x1,y1,x2,y2],...]} (0-1000 norm)
  POST /v1/point     {image_path, query}        -> {points: [[x,y],...]} (0-1000 norm)
  GET  /health

Used by verifier_service.py multi-model cross-verification: the specialist model
independently predicts boxes/points, then the verifier compares geometrically
(IoU for grounding, point-in-box for pointing) instead of just asking yes/no.

Usage:
    VERIFIER_GROUNDING_MODEL=/path/to/workspace/foundation/models/nvidia/LocateAnything-3B \
    VERIFIER_POINTING_MODEL=/path/to/workspace/foundation/models/allenai/Molmo-7B-D-0924 \
    VERIFIER_MODEL_PORT=10310 CUDA_VISIBLE_DEVICES=2 \
    python scripts/verifier_model_server.py
"""

import base64
import io
import json
import logging
import os
import re
import time
from threading import Lock
from typing import List, Optional

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("verifier_model_server")

GROUNDING_MODEL = os.environ.get("VERIFIER_GROUNDING_MODEL", "/path/to/workspace/foundation/models/nvidia/LocateAnything-3B")
POINTING_MODEL = os.environ.get("VERIFIER_POINTING_MODEL", "/path/to/workspace/foundation/models/allenai/Molmo-7B-D-0924")
PORT = int(os.environ.get("VERIFIER_MODEL_PORT", "10310"))
ENABLE_GROUNDING = os.environ.get("VERIFIER_ENABLE_GROUNDING", "1") == "1"
ENABLE_POINTING = os.environ.get("VERIFIER_ENABLE_POINTING", "1") == "1"
MAX_IMAGE_SIZE = int(os.environ.get("VERIFIER_MAX_IMAGE_SIZE", "1024"))

app = Flask(__name__)

# Some custom modeling files (e.g. Molmo) declare spurious heavy imports
# (tensorflow/jax/keras) in check_imports that aren't needed at inference time.
# Patch check_imports to ignore those specific frameworks so trust_remote_code
# loading doesn't demand uninstalled deps. Genuine deps still raise normally
# at actual import time.
_SPURIOUS_IMPORTS = {"tensorflow", "tensorflow_text", "jax", "jaxlib", "flax", "keras", "flash_attn", "flash_attn_2_cuda"}
try:
    import importlib as _importlib
    import transformers.dynamic_module_utils as _dmu

    def _patched_check_imports(filename):
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        imports = re.findall(r"^\s*import\s+(\S+?)(?:\s|$|\.)", content, flags=re.MULTILINE)
        imports += re.findall(r"^\s*from\s+([a-zA-Z][\w.]*)\s+import", content, flags=re.MULTILINE)
        missing = []
        for imp in set(imports):
            top = imp.split(".")[0].strip()
            if not top or top in _SPURIOUS_IMPORTS:
                continue
            try:
                _importlib.import_module(top)
            except Exception:
                missing.append(top)
        missing = [m for m in missing if m]
        if missing:
            raise ImportError(
                f"This modeling file requires the following packages that were not found: {', '.join(missing)}"
            )
        return []
    def _safe_check_imports(filename):
        try:
            return _patched_check_imports(filename)
        except SyntaxError:
            return []
    _dmu.check_imports = _safe_check_imports
except Exception:
    pass

# Lazy-loaded models (loaded on first request to speed startup / allow health check)
_models = {"grounding": None, "pointing": None}
_load_lock = Lock()


def _load_image(image_path: str = None, image_b64: str = None):
    from PIL import Image
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
    elif image_b64:
        data = base64.b64decode(image_b64.split(",")[-1])
        img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        return None
    if max(img.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
    return img


def _load_grounding_model():
    """Load LocateAnything grounding model."""
    if _models["grounding"] is not None:
        return _models["grounding"]
    with _load_lock:
        if _models["grounding"] is not None:
            return _models["grounding"]
        import torch
        from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
        # Compat shim: LocateAnything's custom modeling overrides
        # _check_and_adjust_attn_implementation with an older signature that
        # doesn't accept transformers 5.x's `allow_all_kernels` kwarg. Patch the
        # base method to swallow unknown kwargs so the custom override works.
        try:
            from transformers.modeling_utils import PreTrainedModel
            _orig = PreTrainedModel._check_and_adjust_attn_implementation
            import inspect as _inspect
            if "allow_all_kernels" in _inspect.signature(_orig).parameters:
                def _patched(self, attn_implementation, is_init_check=False, **kwargs):
                    return _orig(self, attn_implementation, is_init_check=is_init_check)
                PreTrainedModel._check_and_adjust_attn_implementation = _patched
        except Exception as _e:
            logger.warning(f"attn impl shim skipped: {_e}")
        logger.info(f"Loading grounding model: {GROUNDING_MODEL}")
        processor = AutoProcessor.from_pretrained(GROUNDING_MODEL, trust_remote_code=True)
        # Pre-import the dynamic modeling module and patch the subclass override
        # whose signature lags transformers 5.x (missing allow_all_kernels kwarg).
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            cfg = AutoProcessor.from_pretrained  # noqa (ensure module cached)
            import transformers
            cls = get_class_from_dynamic_module(
                "modeling_locateanything.LocateAnythingForConditionalGeneration",
                GROUNDING_MODEL,
            )
            base = cls.__mro__[0]
            for klass in cls.__mro__:
                if "_check_and_adjust_attn_implementation" in klass.__dict__:
                    _override = klass.__dict__["_check_and_adjust_attn_implementation"]
                    def _patched(self, attn_implementation, is_init_check=False, **kwargs):
                        if attn_implementation == "magi":
                            return "magi"
                        return super(klass, self)._check_and_adjust_attn_implementation(
                            attn_implementation, is_init_check=is_init_check, **kwargs)
                    klass._check_and_adjust_attn_implementation = _patched
                    logger.info(f"Patched attn impl on {klass.__name__}")
                    break
        except Exception as _e:
            logger.warning(f"dynamic patch skipped: {_e}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                GROUNDING_MODEL, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="cuda",
            )
        except Exception:
            model = AutoModel.from_pretrained(
                GROUNDING_MODEL, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="cuda",
            )
        model.eval()
        _models["grounding"] = (model, processor)
        logger.info("Grounding model loaded")
        return _models["grounding"]


def _load_pointing_model():
    """Load Molmo pointing model."""
    if _models["pointing"] is not None:
        return _models["pointing"]
    with _load_lock:
        if _models["pointing"] is not None:
            return _models["pointing"]
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        logger.info(f"Loading pointing model: {POINTING_MODEL}")
        processor = AutoProcessor.from_pretrained(
            POINTING_MODEL, trust_remote_code=True, torch_dtype="auto", device_map="cuda",
        )
        # Compat shim: older Molmo modeling lacks `all_tied_weights_keys` that
        # transformers 5.9 expects during caching_allocator_warmup. Inject a
        # safe default on the dynamically-loaded class before instantiation.
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            mcls = get_class_from_dynamic_module(
                "modeling_molmo.MolmoForCausalLM", POINTING_MODEL,
            )
            if not hasattr(mcls, "all_tied_weights_keys"):
                mcls.all_tied_weights_keys = {}
                logger.info("Injected all_tied_weights_keys on MolmoForCausalLM")
        except Exception as _e:
            logger.warning(f"Molmo tied-weights shim skipped: {_e}")
        model = AutoModelForCausalLM.from_pretrained(
            POINTING_MODEL, trust_remote_code=True, torch_dtype="auto", device_map="cuda",
            attn_implementation="eager",
        )
        model.eval()
        _models["pointing"] = (model, processor)
        logger.info("Pointing model loaded")
        return _models["pointing"]


def _ground(image, query: str) -> List[List[int]]:
    """Run grounding model, return boxes normalized to [0,1000)."""
    import torch
    model, processor = _load_grounding_model()
    w, h = image.size
    prompt = f"Detect the {query}. Output bounding boxes."
    try:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
    except Exception as e:
        logger.warning(f"Grounding generate failed: {e}")
        return []
    return _parse_boxes(text, w, h)


def _point(image, query: str) -> List[List[int]]:
    """Run pointing model (Molmo), return points normalized to [0,1000)."""
    import torch
    model, processor = _load_pointing_model()
    w, h = image.size
    prompt = f"Point to the {query}."
    try:
        inputs = processor.process(images=[image], text=prompt)
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate_from_batch(
                inputs,
                generation_config=dict(max_new_tokens=256, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer,
            )
        gen_tokens = output[0, inputs["input_ids"].size(1):]
        text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)
    except Exception as e:
        logger.warning(f"Pointing generate failed: {e}")
        return []
    return _parse_molmo_points(text, w, h)


def _parse_boxes(text: str, w: int, h: int) -> List[List[int]]:
    """Parse bbox coordinates from model output, normalize to [0,1000)."""
    boxes = []
    for m in re.finditer(r"[\[\(]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[\]\)]", text):
        vals = [float(x) for x in m.groups()]
        mx = max(vals)
        if mx <= 1.0:
            box = [int(vals[0] * 1000), int(vals[1] * 1000), int(vals[2] * 1000), int(vals[3] * 1000)]
        elif mx <= 1000:
            box = [int(v) for v in vals]
        else:
            box = [int(vals[0] * 1000 / w), int(vals[1] * 1000 / h), int(vals[2] * 1000 / w), int(vals[3] * 1000 / h)]
        boxes.append(box)
    return boxes


def _parse_molmo_points(text: str, w: int, h: int) -> List[List[int]]:
    """Parse Molmo point output. Molmo outputs <point x="X" y="Y"> with X,Y in 0-100 percent."""
    points = []
    for m in re.finditer(r'x\d*="([\d.]+)"\s*y\d*="([\d.]+)"', text):
        x_pct, y_pct = float(m.group(1)), float(m.group(2))
        points.append([int(x_pct * 10), int(y_pct * 10)])  # 0-100 -> 0-1000
    if not points:
        for m in re.finditer(r"\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)", text):
            x, y = float(m.group(1)), float(m.group(2))
            mx = max(x, y)
            if mx <= 100:
                points.append([int(x * 10), int(y * 10)])
            elif mx <= 1000:
                points.append([int(x), int(y)])
            else:
                points.append([int(x * 1000 / w), int(y * 1000 / h)])
    return points


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "verifier_model_server",
        "grounding_enabled": ENABLE_GROUNDING,
        "pointing_enabled": ENABLE_POINTING,
        "grounding_loaded": _models["grounding"] is not None,
        "pointing_loaded": _models["pointing"] is not None,
    })


@app.route("/v1/ground", methods=["POST"])
def ground_endpoint():
    if not ENABLE_GROUNDING:
        return jsonify({"error": "grounding disabled", "boxes": []}), 200
    body = request.get_json(force=True)
    image = _load_image(body.get("image_path"), body.get("image_b64"))
    if image is None:
        return jsonify({"error": "no valid image", "boxes": []}), 200
    query = body.get("query", "object")
    t0 = time.time()
    boxes = _ground(image, query)
    return jsonify({"boxes": boxes, "latency_ms": int((time.time() - t0) * 1000)})


@app.route("/v1/point", methods=["POST"])
def point_endpoint():
    if not ENABLE_POINTING:
        return jsonify({"error": "pointing disabled", "points": []}), 200
    body = request.get_json(force=True)
    image = _load_image(body.get("image_path"), body.get("image_b64"))
    if image is None:
        return jsonify({"error": "no valid image", "points": []}), 200
    query = body.get("query", "object")
    t0 = time.time()
    points = _point(image, query)
    return jsonify({"points": points, "latency_ms": int((time.time() - t0) * 1000)})


if __name__ == "__main__":
    logger.info(f"Starting verifier model server on port {PORT}")
    logger.info(f"  Grounding: {GROUNDING_MODEL} (enabled={ENABLE_GROUNDING})")
    logger.info(f"  Pointing:  {POINTING_MODEL} (enabled={ENABLE_POINTING})")
    logger.info("Models load lazily on first request.")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
