"""VLM API 检测后端 — 用 DashScope qwen-vl-max 做目标检测，替代本地 Rex-Omni。"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger("flywheel.api_detector")

# 加载 .env
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _image_to_base64(image_path: str, max_size: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_image_size(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size  # (w, h)


def _parse_boxes_from_text(text: str, img_w: int, img_h: int) -> List[Dict]:
    """从 VLM 返回的文本中解析 bbox。支持多种格式。"""
    results = []

    # 尝试解析 JSON 数组
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "label" in item and "bbox" in item:
                    bbox = item["bbox"]
                    if isinstance(bbox, list) and len(bbox) == 4:
                        results.append({"label": item["label"], "bbox": [float(x) for x in bbox]})
            if results:
                return results
    except (json.JSONDecodeError, ValueError):
        pass

    # 尝试解析 JSON 代码块
    json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            for item in data:
                if isinstance(item, dict) and "label" in item and "bbox" in item:
                    bbox = item["bbox"]
                    if isinstance(bbox, list) and len(bbox) == 4:
                        results.append({"label": item["label"], "bbox": [float(x) for x in bbox]})
            if results:
                return results
        except (json.JSONDecodeError, ValueError):
            pass

    # 尝试逐行解析 "label: [x1, y1, x2, y2]" 格式
    for line in text.split("\n"):
        m = re.match(r'["\s-]*(\w[\w\s]*?)[\s:]+\[?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', line)
        if m:
            label = m.group(1).strip().lower()
            bbox = [float(m.group(i)) for i in range(2, 6)]
            results.append({"label": label, "bbox": bbox})

    return results


DETECT_PROMPT = (
    "Detect all objects in this image. For each object, output its label and bounding box "
    "as pixel coordinates [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right.\n"
    "Output ONLY a JSON array, no other text:\n"
    '[{{"label": "object_name", "bbox": [x1, y1, x2, y2]}}, ...]\n'
    "Be exhaustive. Include all visible objects."
)

DETECT_CATEGORIES_PROMPT = (
    "Detect the following objects in this image: {categories}.\n"
    "For each detected object, output its label and bounding box "
    "as pixel coordinates [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right.\n"
    "Output ONLY a JSON array, no other text:\n"
    '[{{"label": "object_name", "bbox": [x1, y1, x2, y2]}}, ...]\n'
    "If an object appears multiple times, include each instance. "
    "If an object is not found, skip it."
)


def _rescale_boxes(boxes: List[Dict], orig_w: int, orig_h: int, sent_w: int, sent_h: int) -> List[Dict]:
    """将 API 返回的 bbox 从发送尺寸空间缩放回原始图片尺寸。"""
    if sent_w <= 0 or sent_h <= 0 or (sent_w == orig_w and sent_h == orig_h):
        return boxes
    sx = orig_w / sent_w
    sy = orig_h / sent_h
    for b in boxes:
        bbox = b.get("bbox")
        if bbox and len(bbox) == 4:
            b["bbox"] = [round(bbox[0] * sx, 2), round(bbox[1] * sy, 2),
                         round(bbox[2] * sx, 2), round(bbox[3] * sy, 2)]
    return boxes


def _image_to_base64_with_size(image_path: str, max_size: int = 1024) -> Tuple[str, int, int]:
    """编码为 base64 并返回实际发送的 (b64, sent_w, sent_h)。"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    sent_w, sent_h = img.size
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), sent_w, sent_h


def detect_objects_api(
    image_path: str,
    categories: Optional[List[str]] = None,
) -> List[Dict]:
    """用 DashScope VLM API 检测图片中的物体，返回 [{label, bbox}, ...]。"""
    import httpx

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not set. Cannot use API detection.")

    b64, sent_w, sent_h = _image_to_base64_with_size(image_path)
    img_w, img_h = _get_image_size(image_path)

    if categories:
        prompt = DETECT_CATEGORIES_PROMPT.format(categories=", ".join(categories))
    else:
        prompt = DETECT_PROMPT

    resp = httpx.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "qwen-vl-max",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": 2000,
        },
        timeout=90,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    boxes = _parse_boxes_from_text(text, img_w, img_h)
    boxes = _rescale_boxes(boxes, img_w, img_h, sent_w, sent_h)
    return boxes


def detect_phrases_api(
    image_path: str,
    phrases: List[str],
) -> Dict[str, List[List[float]]]:
    """用 API 检测指定短语对应的 bbox。返回 {phrase: [[x1,y1,x2,y2], ...]}。"""
    import httpx

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not set.")

    b64, sent_w, sent_h = _image_to_base64_with_size(image_path)

    prompt = (
        f"Locate these objects in the image: {', '.join(phrases)}.\n"
        "For each object, output its bounding box as pixel coordinates [x1, y1, x2, y2].\n"
        "Output ONLY a JSON array:\n"
        '[{"label": "object_name", "bbox": [x1, y1, x2, y2]}, ...]\n'
        "If an object appears multiple times, include each instance. Skip objects not found."
    )

    resp = httpx.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "qwen-vl-max",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": 2000,
        },
        timeout=90,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    img_w, img_h = _get_image_size(image_path)
    boxes = _parse_boxes_from_text(text, img_w, img_h)
    boxes = _rescale_boxes(boxes, img_w, img_h, sent_w, sent_h)

    result: Dict[str, List[List[float]]] = {p: [] for p in phrases}
    for det in boxes:
        label = det["label"].lower()
        for p in phrases:
            if p.lower() in label or label in p.lower():
                result[p].append(det["bbox"])
                break
    return result
