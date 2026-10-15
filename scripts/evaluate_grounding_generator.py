"""Independent validation module for the G_grounding GENERATION stage.

This CLI reproduces the production grounding pipeline from the very beginning up
to (but NOT including) the rule verifier, for dimension "G_grounding":

    1. VLM object detection          (the generator itself)
    2. label building + bbox denorm   (build_label / denormalize_bbox)
    3. active bbox cross-check        (Layer-1 self-check, verify_bbox)

It is intentionally NOT wired into the live generator_service.py service. Every
previously hard-coded tunable (prompts, temperatures, token limits, image
encoding, bbox area filters, coordinate convention) is read from the shared
``grounding_generator`` section of configs/grounding_verifier_rules.yaml so the
generation stage can be tuned and validated on its own.

PRODUCTION PARITY — read before trusting the numbers:
  * Logic that is a *faithful copy* of generator_service.py (derive_source,
    parse_json, denormalize_bbox coordinate math, the build_label schema, and the
    detection prompt) is pinned by tests/test_grounding_generator_consistency.py.
    Those tests import both copies and fail if production drifts, so they must
    stay equivalent.
  * KNOWN, INTENTIONAL DEVIATIONS from production (do NOT "fix" them to match):
      - Cross-check (stage 3): the LIVE production path
        run_continuous() -> run_all_dimensions() does NOT run verify_bbox at all;
        cross-check only lived in the now-unused run_dimension(). So enabling
        cross_check here is STRICTER than the running generator. Keep
        cross_check.enabled=false in the config to mirror live behavior; flip it
        on only to study the (legacy) self-check stage in isolation.
      - cross_check.bbox_coord_space defaults to "normalized": production's
        verify_bbox sent PIXEL coords under "0-1000" wording (a quirk). Use
        "pixel" to reproduce production byte-for-byte; "normalized" is the
        corrected behavior.
      - Detection prompt area wording (95% vs production's stale 70%) — a wording
        fix tracked as an allowed delta in the consistency test.
      - chat_template_kwargs (enable_thinking) is applied unconditionally here;
        production only injects it for volceapi/dashscope endpoints.
      - bbox_filters.min_area_ratio is loosened from production's 0.001 (see the
        config comment) — tuning, not a faithful copy.

The emitted labels use the same schema the production generator writes to
``labels/G_grounding/*.jsonl``; with ``--labels-output`` they can be piped
straight into scripts/evaluate_grounding_verifier.py for end-to-end checks.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import requests
import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "grounding_verifier_rules.yaml"
VALID_BBOX_COORD_SPACES = {"pixel", "normalized"}
DIMENSION = "G_grounding"


# ============================================================
# Input loading
# ============================================================

def load_inputs(input_path: Path) -> list[dict[str, Any]]:
    """Load generation inputs.

    Supports three shapes:
      * a ``.txt`` image list, optionally tab-separated ``path\tsource\tscene``
        (same format as data/input/dimension_sources/G_grounding.txt)
      * a ``.jsonl`` file of objects, each with at least ``image_path``
      * a ``.json`` array / object-with-items of such objects
    """
    suffix = input_path.suffix.lower()
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if suffix == ".txt":
        items: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            image_path = parts[0].strip()
            if not image_path:
                continue
            scene_type = parts[2].strip() if len(parts) > 2 else "unknown"
            items.append({"image_path": image_path, "scene_type": scene_type})
        return items

    if suffix == ".jsonl":
        items = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            items.append(item)
        return items

    payload = json.loads(text)
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("JSON array items must be objects")
        return payload
    if isinstance(payload, dict):
        nested = payload.get("items")
        if nested is None:
            return [payload]
        if not isinstance(nested, list) or not all(isinstance(item, dict) for item in nested):
            raise ValueError("JSON object field 'items' must be a list of objects")
        return nested
    raise ValueError("Input must be .txt, .jsonl, a JSON array, or a JSON object")


# ============================================================
# Config loading
# ============================================================

def _expand_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return os.path.expandvars(value)


def load_grounding_generator_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and normalize the ``grounding_generator`` section."""
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    raw = payload.get("grounding_generator", {})
    if not isinstance(raw, dict):
        raise ValueError("Config field 'grounding_generator' must be a mapping")

    model_raw = raw.get("model", {}) or {}
    if not isinstance(model_raw, dict):
        raise ValueError("grounding_generator.model must be a mapping")
    model = {
        "endpoint": _expand_env(model_raw.get("endpoint", "")),
        "api_key_env": model_raw.get("api_key_env", ""),
        "model": _expand_env(model_raw.get("model", "")),
        "timeout_seconds": float(model_raw.get("timeout_seconds", 180)),
        "chat_template_kwargs": model_raw.get("chat_template_kwargs", {}) or {},
    }

    image_raw = raw.get("image", {}) or {}
    image = {
        "max_size": int(image_raw.get("max_size", 960)),
        "jpeg_quality": int(image_raw.get("jpeg_quality", 85)),
    }

    filters_raw = raw.get("bbox_filters", {}) or {}
    bbox_filters = {
        "max_area_ratio": float(filters_raw.get("max_area_ratio", 0.95)),
        "min_area_ratio": float(filters_raw.get("min_area_ratio", 0.001)),
    }

    detection_raw = raw.get("detection", {}) or {}
    if not detection_raw.get("prompt"):
        raise ValueError("grounding_generator.detection.prompt must be defined")
    detection = {
        "temperature": float(detection_raw.get("temperature", 0.7)),
        "max_tokens": int(detection_raw.get("max_tokens", 8192)),
        "retries": max(1, int(detection_raw.get("retries", 3))),
        "system_prompt": detection_raw.get("system_prompt", "") or "",
        "prompt": detection_raw["prompt"],
    }

    cross_raw = raw.get("cross_check", {}) or {}
    bbox_coord_space = str(cross_raw.get("bbox_coord_space", "normalized")).lower()
    if bbox_coord_space not in VALID_BBOX_COORD_SPACES:
        raise ValueError(f"cross_check.bbox_coord_space must be one of {sorted(VALID_BBOX_COORD_SPACES)}")
    cross_check = {
        "enabled": bool(cross_raw.get("enabled", True)),
        "temperature": float(cross_raw.get("temperature", 0.7)),
        "max_tokens": int(cross_raw.get("max_tokens", 8192)),
        "retries": max(1, int(cross_raw.get("retries", 1))),
        "keep_on_unparsed": bool(cross_raw.get("keep_on_unparsed", True)),
        "prompt": cross_raw.get("prompt", "") or "",
        "bbox_coord_space": bbox_coord_space,
    }
    if cross_check["enabled"] and not cross_check["prompt"]:
        raise ValueError("grounding_generator.cross_check.prompt must be defined when enabled")

    config = {
        "worker_count": max(1, int(raw.get("worker_count", 1))),
        "max_images": max(0, int(raw.get("max_images", 0))),
        "coordinate_normalized_range": int(raw.get("coordinate_normalized_range", 1000)),
        "default_confidence": float(raw.get("default_confidence", 0.5)),
        "model": model,
        "image": image,
        "bbox_filters": bbox_filters,
        "detection": detection,
        "cross_check": cross_check,
    }
    if not config["model"]["endpoint"]:
        raise ValueError("grounding_generator.model.endpoint must be configured")
    return config


# ============================================================
# Image + JSON helpers (mirror generator_service.VLMClient)
# ============================================================

def encode_image_data_url(image_path: str, max_size: int = 960, jpeg_quality: int = 85) -> str | None:
    try:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=jpeg_quality)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None


def get_image_size(image_path: str) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (1280, 960)


def parse_json(text: str) -> Any | None:
    """Reproduce generator_service.VLMClient._parse_json."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        if start < 0:
            start = text.find("{")
        end = max(text.rfind("]"), text.rfind("}")) + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _round_seconds(value: float) -> float:
    return round(float(value), 6)


def chat_completion_detail(
    model_cfg: dict[str, Any],
    image_path: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    retries: int,
    system_prompt: str = "",
    max_size: int = 960,
    jpeg_quality: int = 85,
) -> dict[str, Any]:
    """Single VLM /chat/completions call with retries; returns content + timing."""
    started = time.perf_counter()
    endpoint = str(model_cfg.get("endpoint", "")).rstrip("/")
    detail: dict[str, Any] = {
        "content": "",
        "raw_response": "",
        "request_seconds": 0.0,
        "encode_seconds": 0.0,
        "total_seconds": 0.0,
        "http_status": None,
        "error": None,
        "attempts": 0,
    }
    if not endpoint:
        detail["error"] = "missing_endpoint"
        detail["total_seconds"] = _round_seconds(time.perf_counter() - started)
        return detail

    encode_started = time.perf_counter()
    data_url = encode_image_data_url(image_path, max_size=max_size, jpeg_quality=jpeg_quality)
    detail["encode_seconds"] = _round_seconds(time.perf_counter() - encode_started)
    if not data_url:
        detail["error"] = "image_encode_failed"
        detail["total_seconds"] = _round_seconds(time.perf_counter() - started)
        return detail

    api_key = os.environ.get(model_cfg.get("api_key_env", ""), "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": prompt},
    ]})
    payload: dict[str, Any] = {
        "model": model_cfg.get("model", ""),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    chat_template_kwargs = model_cfg.get("chat_template_kwargs") or {}
    if chat_template_kwargs:
        # Mirror generator_service.py: thinking control rides in extra_body.
        payload["extra_body"] = {"chat_template_kwargs": chat_template_kwargs}

    timeout = float(model_cfg.get("timeout_seconds", 180))
    request_started = time.perf_counter()
    for attempt in range(retries):
        detail["attempts"] = attempt + 1
        try:
            response = requests.post(f"{endpoint}/chat/completions", json=payload, headers=headers, timeout=timeout)
            detail["http_status"] = response.status_code
            if response.status_code == 429:
                detail["error"] = "http_status_429"
                time.sleep(min(2 ** attempt, 10))
                continue
            if response.status_code != 200:
                detail["raw_response"] = getattr(response, "text", "") or ""
                detail["error"] = f"http_status_{response.status_code}"
                if attempt < retries - 1:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                break
            payload_json = response.json()
            msg = payload_json["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content or content == "null":
                content = msg.get("reasoning_content") or ""
            detail["content"] = content
            detail["raw_response"] = content
            detail["error"] = None
            break
        except Exception as exc:  # surface network/json errors in the report
            detail["error"] = f"{exc.__class__.__name__}: {exc}"
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 10))
                continue
            break
    detail["request_seconds"] = _round_seconds(time.perf_counter() - request_started)
    detail["total_seconds"] = _round_seconds(time.perf_counter() - started)
    return detail


# ============================================================
# Label building (mirror generator_service.py for G_grounding)
# ============================================================

def derive_source(image_path: str) -> str:
    p = image_path.lower()
    if "bridge" in p or "episode" in p or "bridgedatav2" in p:
        return "bridge_data_v2"
    if "erqa" in p or "embodied" in p:
        return "erqa_embodied"
    if "vlm_dataset_root" in p or "raw_benchmark" in p:
        return "vlm_benchmark"
    if "appliance_datasets" in p or "data_instruction" in p or "washing" in p:
        return "appliance"
    if "mit-states" in p or "mit_states" in p:
        return "mit_states"
    if "changeit" in p or "object-states" in p:
        return "changeit_frames"
    if "spatialsense" in p:
        return "spatialsense"
    if "where2place" in p:
        return "where2place"
    if "fixtures" in p:
        return "test_fixtures"
    return "other"


def denormalize_bbox(
    bbox: list[Any],
    img_w: int,
    img_h: int,
    norm_range: int = 1000,
    max_area_ratio: float = 0.95,
    min_area_ratio: float = 0.001,
) -> list[int] | None:
    """Convert a detected bbox to pixel coords with a configurable area band.

    INTENTIONAL DIVERGENCE FROM PRODUCTION (tracked by
    tests/test_grounding_generator_consistency.py; do NOT sync back to
    generator_service.denormalize_bbox without also fixing production):
    production only de-normalizes 0..norm_range coordinates when the image is
    LARGER than norm_range (``img_w > 1000 or img_h > 1000``). But the detection
    prompt always asks for 0-1000 normalized coords, so on sub-1000px images
    (e.g. 640x480 COCO/LVIS) production mis-reads normalized coords as raw pixels
    and rejects ~85% of boxes as out-of-bounds. Here we honor the declared
    0-1000 contract regardless of image size — the actual fix.
    """
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    if all(0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
        # Fractional 0-1 coordinates -> scale by image size.
        x1, y1, x2, y2 = int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h)
    elif all(0 <= v <= norm_range for v in [x1, y1, x2, y2]):
        # 0..norm_range normalized coordinates (the generator's declared output
        # convention) -> scale to pixels regardless of image size. This is the
        # fix: production also gated this on img > norm_range.
        x1 = int(x1 * img_w / norm_range)
        y1 = int(y1 * img_h / norm_range)
        x2 = int(x2 * img_w / norm_range)
        y2 = int(y2 * img_h / norm_range)
    else:
        # Some coordinate exceeds norm_range -> treat as raw pixel coordinates.
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return None
    area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    if area > img_area * max_area_ratio or area < img_area * min_area_ratio:
        return None
    return [x1, y1, x2, y2]


def build_label(
    item: dict[str, Any],
    image_path: str,
    img_w: int,
    img_h: int,
    sample_id: str,
    idx: int,
    config: dict[str, Any],
    scene_type: str = "unknown",
    source_override: str | None = None,
) -> dict[str, Any]:
    """Reproduce generator_service.build_label for the G_grounding branch.

    Parity note: production always derives ``source`` from the image path. This
    module additionally honors an explicit input-level ``source`` (passed as
    ``source_override``) when the caller carries dataset provenance, e.g. the
    LVIS/COCO-Rem ``from_dataset`` field. It falls back to ``derive_source`` when
    absent, so production-equivalent inputs behave identically.
    """
    source = source_override or item.get("source") or derive_source(image_path)
    obj_id = f"{sample_id}_obj{idx:02d}"
    label = {
        "pipeline": DIMENSION,
        "source": source,
        "dataset": source,
        "sample_id": sample_id,
        "image_path": image_path,
        "image_size": [img_w, img_h],
        "bucket": DIMENSION[0],
        "dimension": DIMENSION,
        "object_id": obj_id,
        "timestamp": datetime.now().isoformat(),
        "confidence": float(item.get("confidence", config["default_confidence"])),
        "scene_type": item.get("scene_type", scene_type),
        "object_class": item.get("object_name") or item.get("label") or item.get("name") or "unknown",
        "bbox": denormalize_bbox(
            item.get("bbox") or item.get("bbox_2d") or item.get("bounding_box") or [],
            img_w,
            img_h,
            norm_range=config["coordinate_normalized_range"],
            max_area_ratio=config["bbox_filters"]["max_area_ratio"],
            min_area_ratio=config["bbox_filters"]["min_area_ratio"],
        ),
    }
    return label


# ============================================================
# Generation stages
# ============================================================

def run_detection(image_path: str, config: dict[str, Any]) -> dict[str, Any]:
    """Call the generator VLM and parse its object list."""
    det = config["detection"]
    detail = chat_completion_detail(
        config["model"],
        image_path,
        det["prompt"],
        temperature=det["temperature"],
        max_tokens=det["max_tokens"],
        retries=det["retries"],
        system_prompt=det["system_prompt"],
        max_size=config["image"]["max_size"],
        jpeg_quality=config["image"]["jpeg_quality"],
    )
    parsed = parse_json(str(detail.get("content", "")))
    if isinstance(parsed, dict):
        parsed = [parsed]
    items = [it for it in parsed if isinstance(it, dict)] if isinstance(parsed, list) else []
    detail["objects"] = items
    detail["parsed_ok"] = parsed is not None
    return detail


def render_cross_check_prompt(template: str, obj: str, bbox: list[int]) -> str:
    # Only substitute the documented {object}/{bbox} placeholders. We must NOT use
    # str.format here: the production verify_bbox prompt embeds a literal JSON
    # schema ({"match": ...}) and str.format would treat those braces as fields
    # and raise KeyError.
    bbox_str = "[" + ",".join(str(int(v)) for v in bbox) + "]"
    return template.replace("{object}", obj).replace("{bbox}", bbox_str)


def cross_check_label(image_path: str, label: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Reverse-verify a label's bbox; reproduce generator_service.verify_bbox."""
    cc = config["cross_check"]
    obj = label.get("object_class", "") or label.get("object_name", "")
    bbox = label.get("bbox") or []
    if cc["bbox_coord_space"] == "normalized" and len(bbox) == 4:
        iw, ih = label.get("image_size", [1, 1])
        norm = config["coordinate_normalized_range"]
        iw = iw or 1
        ih = ih or 1
        prompt_bbox = [
            int(bbox[0] * norm / iw),
            int(bbox[1] * norm / ih),
            int(bbox[2] * norm / iw),
            int(bbox[3] * norm / ih),
        ]
    else:
        prompt_bbox = [int(v) for v in bbox]

    prompt = render_cross_check_prompt(cc["prompt"], obj, prompt_bbox)
    detail = chat_completion_detail(
        config["model"],
        image_path,
        prompt,
        temperature=cc["temperature"],
        max_tokens=cc["max_tokens"],
        retries=cc["retries"],
        max_size=config["image"]["max_size"],
        jpeg_quality=config["image"]["jpeg_quality"],
    )
    parsed = parse_json(str(detail.get("content", "")))
    if not parsed or not isinstance(parsed, dict):
        # Faithful to production: unparsed verification keeps the label.
        result = {
            "match": bool(cc["keep_on_unparsed"]),
            "verified_class": "",
            "confidence": 0.5,
            "parsed_ok": False,
        }
    else:
        match_val = parsed.get("match", True)
        if isinstance(match_val, str):
            match_val = match_val.lower() in ("true", "yes", "1")
        result = {
            "match": bool(match_val),
            "verified_class": parsed.get("actual_object", "") or parsed.get("object_class", ""),
            "confidence": float(parsed.get("confidence", 0.5)) if _is_number(parsed.get("confidence")) else 0.5,
            "parsed_ok": True,
        }
    result["raw_response"] = detail.get("raw_response", "")
    result["request_seconds"] = detail.get("request_seconds", 0.0)
    result["http_status"] = detail.get("http_status")
    result["error"] = detail.get("error")
    return result


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def generate_one_image(input_item: dict[str, Any], index: int, config: dict[str, Any]) -> dict[str, Any]:
    """Run detection -> label build -> cross-check for a single image."""
    started = time.perf_counter()
    image_path = input_item.get("image_path", "")
    scene_type = input_item.get("scene_type", "unknown")
    source_override = input_item.get("source")
    sample_id = input_item.get("sample_id") or f"gen_{DIMENSION}_{datetime.now().strftime('%Y%m%d')}_{index:04d}"

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "image_path": image_path,
        "scene_type": scene_type,
        "detection": {},
        "objects_detected": 0,
        "labels_built": 0,
        "labels_dropped_geometry": 0,
        "cross_check_failed": 0,
        "emitted_labels": [],
        "dropped_labels": [],
        "error": None,
    }

    if not image_path or not os.path.exists(image_path):
        result["error"] = "missing_image"
        result["item_seconds"] = _round_seconds(time.perf_counter() - started)
        return result

    detail = run_detection(image_path, config)
    result["detection"] = {
        "parsed_ok": detail.get("parsed_ok", False),
        "http_status": detail.get("http_status"),
        "error": detail.get("error"),
        "request_seconds": detail.get("request_seconds", 0.0),
        "attempts": detail.get("attempts", 0),
        "raw_response": detail.get("raw_response", ""),
    }
    objects = detail.get("objects", [])
    result["objects_detected"] = len(objects)
    if not objects:
        if detail.get("error"):
            result["error"] = detail.get("error")
        result["item_seconds"] = _round_seconds(time.perf_counter() - started)
        return result

    img_w, img_h = get_image_size(image_path)
    for idx, item in enumerate(objects):
        label = build_label(item, image_path, img_w, img_h, sample_id, idx, config, scene_type, source_override=source_override)
        if label.get("bbox") is None:
            result["labels_dropped_geometry"] += 1
            continue
        result["labels_built"] += 1

        if config["cross_check"]["enabled"]:
            cc = cross_check_label(image_path, label, config)
            label["_cross_check"] = {
                "match": cc["match"],
                "verified_class": cc["verified_class"],
                "confidence": cc["confidence"],
                "parsed_ok": cc["parsed_ok"],
                "http_status": cc["http_status"],
                "error": cc["error"],
            }
            if not cc["match"]:
                label["_cross_check_failed"] = True
                result["cross_check_failed"] += 1
                result["dropped_labels"].append(label)
                continue
        result["emitted_labels"].append(label)

    result["labels_emitted"] = len(result["emitted_labels"])
    result["item_seconds"] = _round_seconds(time.perf_counter() - started)
    return result


# ============================================================
# Orchestration + reporting
# ============================================================

class ProgressPrinter:
    def __init__(self, total: int, enabled: bool = False, stream: TextIO | None = None):
        self.total = max(0, int(total))
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.started = time.perf_counter()

    def update(self, done: int) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.started
        rate = done / elapsed if elapsed > 0 else 0.0
        total = self.total or done
        width = 24
        filled = int(width * done / total) if total else width
        bar = "#" * filled + "-" * (width - filled)
        message = f"Grounding generator [{bar}] {done}/{total} elapsed={elapsed:.1f}s rate={rate:.2f}/s"
        end = "\n" if done >= total else "\r"
        print(message, file=self.stream, end=end, flush=True)


def _generate_parallel(
    inputs: list[dict[str, Any]],
    config: dict[str, Any],
    show_progress: bool,
    progress_stream: TextIO | None,
) -> list[dict[str, Any]]:
    worker_count = max(1, int(config.get("worker_count", 1)))
    progress = ProgressPrinter(len(inputs), show_progress, progress_stream)
    if not inputs:
        progress.update(0)
        return []
    items: list[dict[str, Any] | None] = [None] * len(inputs)
    done = 0
    if worker_count <= 1 or len(inputs) <= 1:
        for idx, item in enumerate(inputs):
            items[idx] = generate_one_image(item, idx, config)
            done += 1
            progress.update(done)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(generate_one_image, item, idx, config): idx for idx, item in enumerate(inputs)}
            for future in as_completed(futures):
                idx = futures[future]
                items[idx] = future.result()
                done += 1
                progress.update(done)
    return [item for item in items if item is not None]


def _rate(passed: int, total: int) -> dict[str, float | int]:
    return {"passed": passed, "total": total, "rate": round(passed / total, 6) if total else 0.0}


def evaluate_grounding_generator(
    inputs: list[dict[str, Any]],
    config: dict[str, Any],
    show_progress: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    items = _generate_parallel(inputs, config, show_progress=show_progress, progress_stream=progress_stream)
    total_seconds = time.perf_counter() - started

    total_images = len(items)
    images_with_detection = sum(1 for it in items if it["objects_detected"] > 0)
    images_with_emitted = sum(1 for it in items if it.get("labels_emitted", 0) > 0)
    images_errored = sum(1 for it in items if it.get("error"))
    total_objects = sum(it["objects_detected"] for it in items)
    total_built = sum(it["labels_built"] for it in items)
    total_dropped_geometry = sum(it["labels_dropped_geometry"] for it in items)
    total_cross_failed = sum(it["cross_check_failed"] for it in items)
    total_emitted = sum(it.get("labels_emitted", 0) for it in items)

    return {
        "total_images": total_images,
        "worker_count": max(1, int(config.get("worker_count", 1))),
        "cross_check_enabled": config["cross_check"]["enabled"],
        "model": config["model"]["model"],
        "endpoint": config["model"]["endpoint"],
        "counts": {
            "objects_detected": total_objects,
            "labels_built": total_built,
            "labels_dropped_geometry": total_dropped_geometry,
            "cross_check_failed": total_cross_failed,
            "labels_emitted": total_emitted,
        },
        "stage_passrate": {
            "image_detection_success": _rate(images_with_detection, total_images),
            "image_emitted_any_label": _rate(images_with_emitted, total_images),
            "geometry_keep": _rate(total_built, total_objects),
            "cross_check_keep": _rate(total_built - total_cross_failed, total_built),
            "object_emit": _rate(total_emitted, total_objects),
        },
        "errors": {
            "images_errored": images_errored,
        },
        "timing": {
            "total_seconds": _round_seconds(total_seconds),
            "avg_seconds_per_image": _round_seconds(total_seconds / total_images) if total_images else 0.0,
        },
        "items": items,
    }


def collect_emitted_labels(report: dict[str, Any]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for item in report.get("items", []):
        labels.extend(item.get("emitted_labels", []))
    return labels


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the G_grounding generation stage (generator -> cross-check, before the rule verifier).")
    parser.add_argument("--input", required=True, type=Path, help="Input .txt image list, .jsonl, or .json with image_path entries.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Shared grounding YAML config.")
    parser.add_argument("--output", type=Path, help="Optional report JSON path. Defaults to stdout.")
    parser.add_argument("--labels-output", type=Path, help="Optional JSONL path for the emitted generator labels (feed to evaluate_grounding_verifier.py).")
    parser.add_argument("--worker-count", type=int, help="Override grounding_generator.worker_count.")
    parser.add_argument("--max-images", type=int, help="Override grounding_generator.max_images (0 = no limit).")
    parser.add_argument("--no-cross-check", action="store_true", help="Disable the Layer-1 bbox cross-check for this run.")
    parser.add_argument("--no-progress", action="store_true", help="Disable stderr progress output.")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON instead of pretty JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_grounding_generator_config(args.config)
    if args.worker_count is not None:
        config["worker_count"] = max(1, int(args.worker_count))
    if args.max_images is not None:
        config["max_images"] = max(0, int(args.max_images))
    if args.no_cross_check:
        config["cross_check"]["enabled"] = False

    inputs = load_inputs(args.input)
    if config["max_images"] > 0:
        inputs = inputs[: config["max_images"]]

    report = evaluate_grounding_generator(
        inputs,
        config,
        show_progress=not args.no_progress,
        progress_stream=sys.stderr,
    )

    if args.labels_output:
        labels = collect_emitted_labels(report)
        args.labels_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.labels_output, "w", encoding="utf-8") as f:
            for label in labels:
                f.write(json.dumps(label, ensure_ascii=False) + "\n")

    output = {"report": report}
    text = json.dumps(output, ensure_ascii=False, indent=None if args.compact else 2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
