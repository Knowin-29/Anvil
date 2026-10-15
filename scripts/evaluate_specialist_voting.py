"""Independent multi-model specialist verifier for G_grounding labels.

This CLI is intentionally not wired into verifier_service.py's production
pipeline. It evaluates user-provided image/bbox samples with 1-3 specialist
models and reports per-stage pass rates using a verifier-like JSON format.
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
from pathlib import Path
from typing import Any, TextIO

import requests
import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "grounding_verifier_rules.yaml"
VALID_VOTE_POLICIES = {"majority", "any", "all"}
ORIGINAL_PROMPT_INDEX = "orig"
ORIGINAL_BBOX_PROMPT = (
    'Detect the "{object}" in this image. Output ONLY its bounding box as '
    '[x1,y1,x2,y2] with coordinates normalized to 0-1000. If not present, output "none".'
)
ORIGINAL_POINTING_PROMPT = (
    'Point to {object} in this image. Output ONLY the point as (x,y) '
    'with coordinates normalized to 0-1000. If not present, output "none".'
)


def load_labels(input_path: Path) -> list[dict[str, Any]]:
    """Load labels from JSONL, a JSON array, or a JSON object with items."""
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if input_path.suffix.lower() == ".jsonl":
        labels = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            labels.append(item)
        return labels
    payload = json.loads(text)
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("JSON array items must be objects")
        return payload
    if isinstance(payload, dict):
        items = payload.get("items")
        if items is None:
            return [payload]
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise ValueError("JSON object field 'items' must be a list of objects")
        return items
    raise ValueError("Input must be JSONL, a JSON array, or a JSON object")


def _expand_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return os.path.expandvars(value)


def parse_prompt_index(value: Any) -> int | str:
    if isinstance(value, str) and value.strip().lower() == ORIGINAL_PROMPT_INDEX:
        return ORIGINAL_PROMPT_INDEX
    return int(value)


def load_specialist_voting_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the specialist_voting section from the shared grounding config."""
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    config = payload.get("specialist_voting", {})
    if not isinstance(config, dict):
        raise ValueError("Config field 'specialist_voting' must be a mapping")
    config = {
        "vote_policy": config.get("vote_policy", "any"),
        "max_models": int(config.get("max_models", 3)),
        "iou_threshold": float(config.get("iou_threshold", 0.4)),
        "enable_point_fallback": bool(config.get("enable_point_fallback", True)),
        "point_margin_1000": float(config.get("point_margin_1000", 50)),
        "timeout_seconds": float(config.get("timeout_seconds", 30)),
        "worker_count": int(config.get("worker_count", 1)),
        "bbox_prompt_index": parse_prompt_index(config.get("bbox_prompt_index", 0)),
        "pointing_prompt_index": parse_prompt_index(config.get("pointing_prompt_index", 0)),
        "bbox_prompts": config.get("bbox_prompts", []),
        "pointing_prompts": config.get("pointing_prompts", []),
        "models": config.get("models", []),
    }
    if config["vote_policy"] not in VALID_VOTE_POLICIES:
        raise ValueError(f"vote_policy must be one of {sorted(VALID_VOTE_POLICIES)}")
    enabled_models = []
    for model in config["models"]:
        if not model.get("enabled", True):
            continue
        expanded = {**model}
        expanded["endpoint"] = _expand_env(expanded.get("endpoint", ""))
        expanded["model"] = _expand_env(expanded.get("model", ""))
        enabled_models.append(expanded)
    max_models = max(1, min(3, config["max_models"]))
    config["worker_count"] = max(1, int(config.get("worker_count", 1)))
    config["models"] = enabled_models[:max_models]
    if not 1 <= len(config["models"]) <= 3:
        raise ValueError("specialist_voting must define 1-3 enabled models")
    if not config["bbox_prompts"] or not config["pointing_prompts"]:
        raise ValueError("specialist_voting must define bbox_prompts and pointing_prompts")
    return config


def _valid_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False
    return x1 < x2 and y1 < y2


def _to_1000_bbox(bbox: list[float], image_size: list[float]) -> list[float]:
    iw, ih = float(image_size[0]), float(image_size[1])
    return [bbox[0] * 1000 / iw, bbox[1] * 1000 / ih, bbox[2] * 1000 / iw, bbox[3] * 1000 / ih]


def format_bbox_1000(bbox: list[float]) -> str:
    return "[" + ",".join(str(int(v + 0.5)) for v in bbox) + "]"


def render_prompt(template: str, obj: str, bbox_1000: list[float]) -> str:
    return template.format(object=obj, bbox=format_bbox_1000(bbox_1000))


def _prompt_by_index(prompts: list[str], index: int | str, kind: str) -> str:
    if index == ORIGINAL_PROMPT_INDEX:
        if kind == "bbox":
            return ORIGINAL_BBOX_PROMPT
        if kind == "pointing":
            return ORIGINAL_POINTING_PROMPT
        raise ValueError(f"unknown original prompt kind: {kind}")
    if not isinstance(index, int):
        raise ValueError(f"{kind} prompt index must be an integer or '{ORIGINAL_PROMPT_INDEX}'")
    if index < 0 or index >= len(prompts):
        raise ValueError(f"{kind} prompt index {index} out of range 0-{len(prompts) - 1}")
    return prompts[index]


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_box(point: list[float], box: list[float], margin: float) -> bool:
    return (
        box[0] - margin <= point[0] <= box[2] + margin
        and box[1] - margin <= point[1] <= box[3] + margin
    )


def parse_boxes_1000(text: str) -> list[list[float]]:
    boxes = []
    pattern = r"[\[\(]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\]\)]"
    for match in re.finditer(pattern, text):
        vals = [float(v) for v in match.groups()]
        if max(abs(v) for v in vals) <= 1.0:
            vals = [v * 1000 for v in vals]
        if vals[0] < vals[2] and vals[1] < vals[3]:
            boxes.append(vals)
    return boxes


def parse_points_1000(text: str) -> list[list[float]]:
    points = []
    pattern = r"[\[\(]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\]\)]"
    for match in re.finditer(pattern, text):
        vals = [float(v) for v in match.groups()]
        if max(abs(v) for v in vals) <= 1.0:
            vals = [v * 1000 for v in vals]
        points.append(vals)
    return points


def encode_image_data_url(image_path: str, max_size: int = 960) -> str | None:
    try:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None


def _round_seconds(value: float) -> float:
    return round(float(value), 6)


def _chat_completion_detail(model_cfg: dict[str, Any], image_path: str, prompt: str, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    endpoint = model_cfg.get("endpoint", "").rstrip("/")
    base = {
        "content": "",
        "raw_response": "",
        "request_seconds": 0.0,
        "encode_seconds": 0.0,
        "total_seconds": 0.0,
        "http_status": None,
        "error": None,
    }
    if not endpoint:
        base["error"] = "missing_endpoint"
        base["total_seconds"] = _round_seconds(time.perf_counter() - started)
        return base

    encode_started = time.perf_counter()
    data_url = encode_image_data_url(image_path)
    base["encode_seconds"] = _round_seconds(time.perf_counter() - encode_started)
    if not data_url:
        base["error"] = "image_encode_failed"
        base["total_seconds"] = _round_seconds(time.perf_counter() - started)
        return base

    api_key = os.environ.get(model_cfg.get("api_key_env", ""), "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    messages = []
    if model_cfg.get("system_prompt"):
        messages.append({"role": "system", "content": model_cfg["system_prompt"]})
    messages.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": prompt},
    ]})
    payload = {
        "model": model_cfg.get("model", ""),
        "messages": messages,
        "temperature": model_cfg.get("temperature", 0.1),
        "max_tokens": model_cfg.get("max_tokens", 256),
    }
    if model_cfg.get("chat_template_kwargs"):
        payload["chat_template_kwargs"] = model_cfg["chat_template_kwargs"]

    request_started = time.perf_counter()
    try:
        response = requests.post(f"{endpoint}/chat/completions", json=payload, headers=headers, timeout=timeout)
        base["request_seconds"] = _round_seconds(time.perf_counter() - request_started)
        base["http_status"] = response.status_code
        if response.status_code != 200:
            base["raw_response"] = getattr(response, "text", "") or ""
            base["error"] = f"http_status_{response.status_code}"
            return base
        payload_json = response.json()
        content = payload_json["choices"][0]["message"].get("content", "") or ""
        base["content"] = content
        base["raw_response"] = content
        return base
    except Exception as exc:  # requests/json errors should be visible in the report.
        base["request_seconds"] = _round_seconds(time.perf_counter() - request_started)
        base["error"] = f"{exc.__class__.__name__}: {exc}"
        return base
    finally:
        base["total_seconds"] = _round_seconds(time.perf_counter() - started)


def _chat_completion(model_cfg: dict[str, Any], image_path: str, prompt: str, timeout: float) -> str:
    return str(_chat_completion_detail(model_cfg, image_path, prompt, timeout).get("content", ""))


def query_model_for_bbox_detail(
    model_cfg: dict[str, Any],
    image_path: str,
    obj: str,
    candidate_bbox_1000: list[float],
    timeout: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    prompt = render_prompt(
        _prompt_by_index(config["bbox_prompts"], config["bbox_prompt_index"], "bbox"),
        obj,
        candidate_bbox_1000,
    )
    detail = _chat_completion_detail(model_cfg, image_path, prompt, timeout)
    detail["values"] = parse_boxes_1000(str(detail.get("content", "")))
    return detail


def query_model_for_bbox(
    model_cfg: dict[str, Any],
    image_path: str,
    obj: str,
    candidate_bbox_1000: list[float],
    timeout: float,
    config: dict[str, Any],
) -> list[list[float]]:
    prompt = render_prompt(
        _prompt_by_index(config["bbox_prompts"], config["bbox_prompt_index"], "bbox"),
        obj,
        candidate_bbox_1000,
    )
    return parse_boxes_1000(_chat_completion(model_cfg, image_path, prompt, timeout))


def query_model_for_point_detail(
    model_cfg: dict[str, Any],
    image_path: str,
    obj: str,
    candidate_bbox_1000: list[float],
    timeout: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    prompt = render_prompt(
        _prompt_by_index(config["pointing_prompts"], config["pointing_prompt_index"], "pointing"),
        obj,
        candidate_bbox_1000,
    )
    detail = _chat_completion_detail(model_cfg, image_path, prompt, timeout)
    detail["values"] = parse_points_1000(str(detail.get("content", "")))
    return detail


def query_model_for_point(
    model_cfg: dict[str, Any],
    image_path: str,
    obj: str,
    candidate_bbox_1000: list[float],
    timeout: float,
    config: dict[str, Any],
) -> list[list[float]]:
    prompt = render_prompt(
        _prompt_by_index(config["pointing_prompts"], config["pointing_prompt_index"], "pointing"),
        obj,
        candidate_bbox_1000,
    )
    return parse_points_1000(_chat_completion(model_cfg, image_path, prompt, timeout))


def votes_required(model_count: int, vote_policy: str) -> int:
    if vote_policy == "any":
        return 1
    if vote_policy == "all":
        return model_count
    return model_count // 2 + 1


def _rate(passed: int, total: int) -> dict[str, float | int]:
    return {"passed": passed, "total": total, "rate": round(passed / total, 6) if total else 0.0}


def _prompt_number_for_report(index: int | str) -> int | str:
    if index == ORIGINAL_PROMPT_INDEX:
        return ORIGINAL_PROMPT_INDEX
    return int(index) + 1


def prompt_pair_info(config: dict[str, Any]) -> dict[str, Any]:
    bbox_index = config["bbox_prompt_index"]
    pointing_index = config["pointing_prompt_index"]
    bbox_number = _prompt_number_for_report(bbox_index)
    pointing_number = _prompt_number_for_report(pointing_index)
    return {
        "bbox_prompt_index": bbox_index,
        "pointing_prompt_index": pointing_index,
        "bbox_prompt_number": bbox_number,
        "pointing_prompt_number": pointing_number,
        "label": f"{bbox_number}-{pointing_number}",
        "python_index_label": f"{bbox_index}-{pointing_index}",
    }


def _detail_seconds(detail: dict[str, Any], key: str) -> float:
    value = detail.get(key, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _model_result_timing_fields(prefix: str, detail: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_raw_response": detail.get("raw_response", ""),
        f"{prefix}_request_seconds": _round_seconds(_detail_seconds(detail, "request_seconds")),
        f"{prefix}_encode_seconds": _round_seconds(_detail_seconds(detail, "encode_seconds")),
        f"{prefix}_total_seconds": _round_seconds(_detail_seconds(detail, "total_seconds")),
        f"{prefix}_http_status": detail.get("http_status"),
        f"{prefix}_error": detail.get("error"),
    }


def evaluate_one_label(label: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    item_started = time.perf_counter()
    obj = label.get("object_class") or label.get("object_name") or ""
    bbox = label.get("bbox")
    image_size = label.get("image_size") or [1000, 1000]
    input_valid = bool(label.get("image_path") and obj and _valid_bbox(bbox) and _valid_bbox([0, 0, image_size[0], image_size[1]]))
    model_results = []
    if not input_valid:
        return {
            "sample_id": label.get("sample_id"),
            "image_path": label.get("image_path"),
            "object_class": obj,
            "bbox": bbox,
            "input_valid": False,
            "model_results": model_results,
            "votes_required": votes_required(len(config["models"]), config["vote_policy"]),
            "agree_votes": 0,
            "tag": "rejected",
            "reason": "input_invalid",
            "item_seconds": _round_seconds(time.perf_counter() - item_started),
        }

    label_box = _to_1000_bbox([float(v) for v in bbox], image_size)
    timeout = config["timeout_seconds"]
    for model_cfg in config["models"]:
        bbox_detail = query_model_for_bbox_detail(model_cfg, label["image_path"], obj, label_box, timeout, config)
        boxes = list(bbox_detail.get("values", []) or [])
        best_iou = max((bbox_iou(label_box, box) for box in boxes), default=0.0)
        bbox_agree = best_iou >= config["iou_threshold"] if boxes else None
        bbox_request_seconds = _detail_seconds(bbox_detail, "request_seconds")
        result = {
            "model": model_cfg.get("name"),
            "call_success": bool(boxes),
            "bbox_used": bool(boxes),
            "bbox_result": boxes,
            "bbox_best_iou": round(best_iou, 4) if boxes else None,
            "bbox_agree": bbox_agree,
            "pointing_used": False,
            "pointing_result": [],
            "pointing_agree": None,
            "method": "grounding_iou",
            "agree": bool(bbox_agree),
            "score": round(best_iou, 4),
            "evidence": f"best_iou={best_iou:.3f}; boxes={len(boxes)}",
            "raw_response": bbox_detail.get("raw_response", ""),
            "request_seconds": _round_seconds(bbox_request_seconds),
            "call_total_seconds": _round_seconds(_detail_seconds(bbox_detail, "total_seconds")),
            "http_status": bbox_detail.get("http_status"),
            "error": bbox_detail.get("error"),
            **_model_result_timing_fields("bbox", bbox_detail),
            "pointing_raw_response": "",
            "pointing_request_seconds": 0.0,
            "pointing_encode_seconds": 0.0,
            "pointing_total_seconds": 0.0,
            "pointing_http_status": None,
            "pointing_error": None,
        }
        if not boxes and config["enable_point_fallback"]:
            point_detail = query_model_for_point_detail(model_cfg, label["image_path"], obj, label_box, timeout, config)
            points = list(point_detail.get("values", []) or [])
            hit = any(point_in_box(point, label_box, config["point_margin_1000"]) for point in points)
            point_request_seconds = _detail_seconds(point_detail, "request_seconds")
            result = {
                "model": model_cfg.get("name"),
                "call_success": bool(points),
                "bbox_used": False,
                "bbox_result": [],
                "bbox_best_iou": None,
                "bbox_agree": None,
                "pointing_used": bool(points),
                "pointing_result": points,
                "pointing_agree": hit if points else None,
                "method": "point_in_box",
                "agree": hit,
                "score": 1.0 if hit else 0.0,
                "evidence": f"points={len(points)}; in_box={hit}",
                "raw_response": point_detail.get("raw_response", ""),
                "request_seconds": _round_seconds(bbox_request_seconds + point_request_seconds),
                "call_total_seconds": _round_seconds(_detail_seconds(bbox_detail, "total_seconds") + _detail_seconds(point_detail, "total_seconds")),
                "http_status": point_detail.get("http_status"),
                "error": point_detail.get("error"),
                **_model_result_timing_fields("bbox", bbox_detail),
                **_model_result_timing_fields("pointing", point_detail),
            }
        model_results.append(result)

    agree_votes = sum(1 for result in model_results if result["agree"])
    required = votes_required(len(model_results), config["vote_policy"])
    passed = agree_votes >= required
    return {
        "sample_id": label.get("sample_id"),
        "image_path": label.get("image_path"),
        "object_class": obj,
        "bbox": bbox,
        "input_valid": True,
        "model_results": model_results,
        "votes_required": required,
        "agree_votes": agree_votes,
        "tag": "verified_auto" if passed else "needs_human_review",
        "reason": "" if passed else f"agree_votes={agree_votes}/{required}",
        "item_seconds": _round_seconds(time.perf_counter() - item_started),
    }


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
        message = f"Specialist voting [{bar}] {done}/{total} elapsed={elapsed:.1f}s rate={rate:.2f}/s"
        end = "\n" if done >= total else "\r"
        print(message, file=self.stream, end=end, flush=True)


def _evaluate_labels_parallel(
    labels: list[dict[str, Any]],
    config: dict[str, Any],
    show_progress: bool,
    progress_stream: TextIO | None,
) -> list[dict[str, Any]]:
    worker_count = max(1, int(config.get("worker_count", 1)))
    progress = ProgressPrinter(len(labels), show_progress, progress_stream)
    if not labels:
        progress.update(0)
        return []
    items: list[dict[str, Any] | None] = [None] * len(labels)
    done = 0
    if worker_count <= 1 or len(labels) <= 1:
        for idx, label in enumerate(labels):
            items[idx] = evaluate_one_label(label, config)
            done += 1
            progress.update(done)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(evaluate_one_label, label, config): idx for idx, label in enumerate(labels)}
            for future in as_completed(futures):
                idx = futures[future]
                items[idx] = future.result()
                done += 1
                progress.update(done)
    return [item for item in items if item is not None]


def _timing_summary(items: list[dict[str, Any]], total_seconds: float) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    request_seconds_values: list[float] = []
    item_seconds_values = [float(item.get("item_seconds", 0.0) or 0.0) for item in items]
    for item in items:
        for result in item.get("model_results", []):
            seconds = float(result.get("request_seconds", 0.0) or 0.0)
            if seconds < 0:
                seconds = 0.0
            request_seconds_values.append(seconds)
            model = str(result.get("model") or "unknown")
            entry = by_model.setdefault(model, {"count": 0, "success_count": 0, "total_seconds": 0.0, "max_seconds": 0.0})
            entry["count"] += 1
            if result.get("call_success"):
                entry["success_count"] += 1
            entry["total_seconds"] += seconds
            entry["max_seconds"] = max(float(entry["max_seconds"]), seconds)
    for entry in by_model.values():
        count = int(entry["count"])
        entry["total_seconds"] = _round_seconds(entry["total_seconds"])
        entry["max_seconds"] = _round_seconds(entry["max_seconds"])
        entry["avg_seconds"] = _round_seconds(entry["total_seconds"] / count) if count else 0.0
    request_total = sum(request_seconds_values)
    request_count = len(request_seconds_values)
    item_total = sum(item_seconds_values)
    return {
        "total_seconds": _round_seconds(total_seconds),
        "item_total_seconds": _round_seconds(item_total),
        "item_avg_seconds": _round_seconds(item_total / len(item_seconds_values)) if item_seconds_values else 0.0,
        "model_request_count": request_count,
        "model_request_total_seconds": _round_seconds(request_total),
        "model_request_avg_seconds": _round_seconds(request_total / request_count) if request_count else 0.0,
        "model_request_max_seconds": _round_seconds(max(request_seconds_values, default=0.0)),
        "by_model": by_model,
    }


def evaluate_specialist_voting(
    labels: list[dict[str, Any]],
    config: dict[str, Any],
    show_progress: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    items = _evaluate_labels_parallel(labels, config, show_progress=show_progress, progress_stream=progress_stream)
    total_seconds = time.perf_counter() - started
    total = len(items)
    input_valid_count = sum(1 for item in items if item["input_valid"])
    model_success_count = sum(1 for item in items if any(r["call_success"] for r in item["model_results"]))
    bbox_agree_count = sum(1 for item in items if any(r["method"] == "grounding_iou" and r["agree"] for r in item["model_results"]))
    point_agree_count = sum(1 for item in items if any(r["method"] == "point_in_box" and r["agree"] for r in item["model_results"]))
    vote_pass_count = sum(1 for item in items if item["tag"] == "verified_auto")
    review_count = sum(1 for item in items if item["tag"] == "needs_human_review")
    rejected_count = sum(1 for item in items if item["tag"] == "rejected")
    return {
        "total": total,
        "vote_policy": config["vote_policy"],
        "worker_count": max(1, int(config.get("worker_count", 1))),
        "model_count": len(config["models"]),
        "iou_threshold": config["iou_threshold"],
        "bbox_prompt_index": config["bbox_prompt_index"],
        "pointing_prompt_index": config["pointing_prompt_index"],
        "prompt_pair": prompt_pair_info(config),
        "timing": _timing_summary(items, total_seconds),
        "stage_passrate": {
            "input_valid": _rate(input_valid_count, total),
            "model_call_success": _rate(model_success_count, total),
            "bbox_iou_agree": _rate(bbox_agree_count, total),
            "point_fallback_agree": _rate(point_agree_count, total),
            "vote_pass": _rate(vote_pass_count, total),
        },
        "overall": {
            "verified_auto": {"count": vote_pass_count, "rate": round(vote_pass_count / total, 6) if total else 0.0},
            "needs_human_review": {"count": review_count, "rate": round(review_count / total, 6) if total else 0.0},
            "rejected": {"count": rejected_count, "rate": round(rejected_count / total, 6) if total else 0.0},
        },
        "items": items,
    }


def run_prompt_sweep(labels: list[dict[str, Any]], config: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"bbox", "pointing", "paired", "grid"}:
        raise ValueError("sweep mode must be one of bbox/pointing/paired/grid")
    pairs: list[tuple[int, int]] = []
    bbox_count = len(config["bbox_prompts"])
    point_count = len(config["pointing_prompts"])
    if mode == "bbox":
        pairs = [(idx, config["pointing_prompt_index"]) for idx in range(bbox_count)]
    elif mode == "pointing":
        pairs = [(config["bbox_prompt_index"], idx) for idx in range(point_count)]
    elif mode == "paired":
        pairs = [(idx, idx) for idx in range(min(bbox_count, point_count))]
    else:
        pairs = [(bbox_idx, point_idx) for bbox_idx in range(bbox_count) for point_idx in range(point_count)]

    results = []
    for bbox_idx, point_idx in pairs:
        candidate = {**config, "bbox_prompt_index": bbox_idx, "pointing_prompt_index": point_idx}
        report = evaluate_specialist_voting(labels, candidate)
        if mode == "bbox":
            prompt_id = f"bbox_{bbox_idx}"
            prompt_type = "bbox"
            tested_prompt_text = config["bbox_prompts"][bbox_idx]
        elif mode == "pointing":
            prompt_id = f"pointing_{point_idx}"
            prompt_type = "pointing"
            tested_prompt_text = config["pointing_prompts"][point_idx]
        else:
            prompt_id = f"bbox_{bbox_idx}__pointing_{point_idx}"
            prompt_type = mode
            tested_prompt_text = {
                "bbox": config["bbox_prompts"][bbox_idx],
                "pointing": config["pointing_prompts"][point_idx],
            }
        results.append({
            "prompt_id": prompt_id,
            "prompt_type": prompt_type,
            "bbox_prompt_index": bbox_idx,
            "pointing_prompt_index": point_idx,
            "tested_prompt_text": tested_prompt_text,
            "overall": report["overall"],
            "stage_passrate": report["stage_passrate"],
            "report": report,
        })
    return {"mode": mode, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate independent multi-model specialist verifier pass rates.")
    parser.add_argument("--input", required=True, type=Path, help="Input .jsonl or .json file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Shared grounding verifier YAML.")
    parser.add_argument("--output", type=Path, help="Optional output JSON path. Defaults to stdout.")
    parser.add_argument("--vote-policy", choices=sorted(VALID_VOTE_POLICIES), help="Override specialist_voting.vote_policy.")
    parser.add_argument("--bbox-prompt-index", help="Override specialist_voting.bbox_prompt_index; use 'orig' for the production pipeline prompt.")
    parser.add_argument("--pointing-prompt-index", help="Override specialist_voting.pointing_prompt_index; use 'orig' for the production pipeline prompt.")
    parser.add_argument("--worker-count", type=int, help="Override specialist_voting.worker_count for parallel sample evaluation.")
    parser.add_argument("--no-progress", action="store_true", help="Disable stderr progress output.")
    parser.add_argument(
        "--sweep-prompts",
        choices=["bbox", "pointing", "paired", "grid"],
        help="Evaluate prompt candidates in batch. grid runs all bbox x pointing combinations.",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON instead of pretty JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_specialist_voting_config(args.config)
    if args.vote_policy:
        config["vote_policy"] = args.vote_policy
    if args.bbox_prompt_index is not None:
        config["bbox_prompt_index"] = parse_prompt_index(args.bbox_prompt_index)
    if args.pointing_prompt_index is not None:
        config["pointing_prompt_index"] = parse_prompt_index(args.pointing_prompt_index)
    if args.worker_count is not None:
        config["worker_count"] = max(1, int(args.worker_count))
    labels = load_labels(args.input)
    report = evaluate_specialist_voting(labels, config, show_progress=not args.no_progress, progress_stream=sys.stderr)
    output: dict[str, Any] = {"report": report}
    if args.sweep_prompts:
        output["prompt_sweep"] = run_prompt_sweep(labels, config, args.sweep_prompts)
    text = json.dumps(output, ensure_ascii=False, indent=None if args.compact else 2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
