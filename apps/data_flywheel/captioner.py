"""Step2: Caption 生成 — 支持本地 Rex-Omni 和多种 VLM API。"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger("flywheel.captioner")

# 加载 .env
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _image_to_base64(image_path: str, max_size: int = 1024) -> str:
    """将图片转为 base64，缩放到合理大小以节省 token。"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


CAPTION_PROMPT = (
    "Describe all objects you can see in this image. "
    "List each object as a simple noun (e.g. cup, bottle, plate, bowl, laptop, phone). "
    "Output format: 'There is {object1}, {object2}, ... and {objectN} in the image.' "
    "Be specific and exhaustive. Do not describe colors or positions."
)


# ---- API 后端 ----

def _caption_dashscope(image_path: str) -> Optional[str]:
    """通义千问 VL (qwen-vl-max) via DashScope API — 速度快、中文场景好。"""
    import httpx
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    b64 = _image_to_base64(image_path)
    resp = httpx.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "qwen-vl-max",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": CAPTION_PROMPT},
            ]}],
            "max_tokens": 300,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _caption_openai(image_path: str) -> Optional[str]:
    """OpenAI GPT-4o-mini via compatible API。"""
    import httpx
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    b64 = _image_to_base64(image_path)
    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": CAPTION_PROMPT},
            ]}],
            "max_tokens": 300,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _caption_anthropic(image_path: str) -> Optional[str]:
    """Anthropic Claude via Messages API。"""
    import httpx
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    b64 = _image_to_base64(image_path)
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": CAPTION_PROMPT},
            ]}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


def _caption_local_rex(image_path: str, model_path: str, backend: str) -> Optional[str]:
    """本地 Rex-Omni 检测 → 拼 caption（fallback）。"""
    from rex_omni import RexOmniWrapper
    common_categories = [
        "person", "car", "chair", "table", "cup", "bottle", "laptop",
        "phone", "book", "bag", "box", "plant", "dog", "cat", "food",
        "door", "window", "monitor", "keyboard", "mouse", "lamp",
        "shelf", "cabinet", "toy", "tool", "robot", "arm",
        "sofa", "bed", "sink", "toilet", "refrigerator", "oven",
        "microwave", "clock", "vase", "pillow", "towel", "basket",
        "plate", "bowl", "pan", "pot", "knife", "fork", "spoon",
        "glass", "mug", "tray", "cutting board", "napkin",
    ]
    model = RexOmniWrapper(
        model_path=model_path, backend=backend,
        max_tokens=2048, temperature=0.0, top_p=1.0, top_k=1,
        repetition_penalty=1.05,
    )
    image = Image.open(image_path).convert("RGB")
    det = model.inference(images=image, task="detection", categories=common_categories)[0]
    preds = det.get("extracted_predictions", {})
    found = [cat for cat, dets in preds.items() if dets]
    if found:
        if len(found) > 1:
            return f"There is {', '.join(found[:-1])} and {found[-1]} in the image."
        return f"There is a {found[0]} in the image."
    return "There is a table and a chair in the room."


# ---- API 探测 ----

_API_BACKENDS = [
    ("dashscope", _caption_dashscope, "通义千问 VL (qwen-vl-max)"),
    ("openai", _caption_openai, "OpenAI GPT-4o-mini"),
    ("anthropic", _caption_anthropic, "Anthropic Claude"),
]


def _default_probe_image() -> Optional[str]:
    probe_image = os.environ.get("REX_API_TEST_IMAGE", "").strip()
    if probe_image and os.path.exists(probe_image):
        return probe_image

    root = Path(__file__).resolve().parents[2]
    probe_dirs = [
        root / "data" / "data" / "input" / "debug",
        root / "data" / "input" / "images",
    ]
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for img_dir in probe_dirs:
        if not img_dir.is_dir():
            continue
        imgs = sorted(p for p in img_dir.glob("*.*") if p.suffix.lower() in image_exts)
        if imgs:
            return str(imgs[0])
    return None


def probe_apis(test_image: Optional[str] = None) -> Dict[str, Any]:
    """测试所有 API 后端的可用性，返回结果。"""
    if test_image is None:
        test_image = _default_probe_image()

    results: Dict[str, Any] = {}
    for name, fn, desc in _API_BACKENDS:
        if test_image is None:
            results[name] = {"available": False, "reason": "no test image", "desc": desc}
            continue
        try:
            caption = fn(test_image)
            if caption:
                results[name] = {"available": True, "desc": desc, "sample": caption[:100]}
            else:
                results[name] = {"available": False, "reason": "no API key", "desc": desc}
        except Exception as e:
            results[name] = {"available": False, "reason": str(e)[:200], "desc": desc}

    return results


def select_best_backend(probe_results: Optional[Dict] = None) -> Optional[str]:
    """选择第一个可用的 API 后端。优先级：dashscope > openai > anthropic。"""
    if probe_results is None:
        probe_results = probe_apis()
    for name, _, _ in _API_BACKENDS:
        if probe_results.get(name, {}).get("available"):
            return name
    return None


# ---- 主入口 ----

def generate_captions(
    image_paths: List[str],
    caption_backend: str = "auto",
    model_path: str = "IDEA-Research/Rex-Omni",
    backend: str = "transformers",
) -> List[Tuple[str, str]]:
    """为图片生成 caption。

    caption_backend:
      - "auto": 自动探测最快可用的 API，全部不可用则 fallback 到本地
      - "dashscope" / "openai" / "anthropic": 指定 API
      - "local": 本地 Rex-Omni
    """
    # 确定后端
    if caption_backend == "auto":
        probe = probe_apis(image_paths[0] if image_paths else None)
        chosen = select_best_backend(probe)
        if chosen:
            logger.info(f"Caption 后端: {chosen} ({probe[chosen].get('desc', '')})")
        else:
            logger.info("所有 API 不可用，fallback 到本地 Rex-Omni")
            chosen = "local"
    else:
        chosen = caption_backend

    fn_map = {name: fn for name, fn, _ in _API_BACKENDS}

    results: List[Tuple[str, str]] = []
    for i, img_path in enumerate(image_paths):
        try:
            if chosen == "local":
                caption = _caption_local_rex(img_path, model_path, backend)
            elif chosen in fn_map:
                caption = fn_map[chosen](img_path)
                if not caption:
                    caption = _caption_local_rex(img_path, model_path, backend)
            else:
                caption = _caption_local_rex(img_path, model_path, backend)

            results.append((img_path, caption or "There is a table in the room."))
            if (i + 1) % 5 == 0:
                logger.info(f"  Caption {i+1}/{len(image_paths)} done")
        except Exception as e:
            logger.warning(f"  Caption failed for {img_path}: {e}")
            results.append((img_path, "There is a table in the room."))

    return results


# 保留旧接口兼容
def generate_captions_with_rex(
    image_paths: List[str],
    model_path: str = "IDEA-Research/Rex-Omni",
    backend: str = "transformers",
) -> List[Tuple[str, str]]:
    return generate_captions(image_paths, caption_backend="local", model_path=model_path, backend=backend)


def generate_captions_from_file(manifest_path: str) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            results.append((item["image_path"], item["caption"]))
    return results
