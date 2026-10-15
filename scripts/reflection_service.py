"""GSA Reflection Service — VLM-based cross-verification of generated annotations.

Features:
- Background thread samples verified/needs_review labels every 120s
- Sends image + annotation to VLM for correctness verification
- Downgrades incorrect labels (verified->needs_review, needs_review->rejected)
- Per-dimension VLM prompt templates

Flask app on port 10308.
"""

import json
import logging
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
VERIFIED_DIR = BASE_DIR / "data" / "output" / "gsa_flywheel_exp" / "verified"
REVIEW_DIR = BASE_DIR / "data" / "output" / "gsa_flywheel_exp" / "needs_review"
REJECTED_DIR = BASE_DIR / "data" / "output" / "gsa_flywheel_exp" / "rejected"

REFLECTION_INTERVAL = 120  # seconds between reflection rounds
SAMPLE_SIZE = int(os.environ.get("REFLECTION_SAMPLE_SIZE", "10"))

def _normalize_chat_url(raw: str) -> str:
    """Match generator_service behaviour: .env exposes VLM_URL as a `/v1` base,
    while the OpenAI-compatible chat endpoint is `/v1/chat/completions`.
    Accept either form so reflection stops 404-ing against the volceapi base."""
    raw = (raw or "").rstrip("/")
    if raw.endswith("/chat/completions"):
        return raw
    return f"{raw}/chat/completions"


VLM_URL = _normalize_chat_url(os.environ.get("VLM_URL", "http://localhost:8000/v1"))
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")
VLM_MODEL = os.environ.get("VLM_MODEL", "MLLM")

DOWNGRADE_CONFIDENCE_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("reflection_service")

# ---------------------------------------------------------------------------
# Flask App & State
# ---------------------------------------------------------------------------

app = Flask(__name__)
_lock = threading.Lock()

_state: dict = {
    "total_reflected": 0,
    "total_downgraded": 0,
    "last_reflection_time": None,
    "reflection_history": [],
}

# ---------------------------------------------------------------------------
# VLM Prompt Templates (per dimension)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATES = {
    "G_grounding": (
        "图中是否真的存在[{object_class}]？bbox [{bbox}] 是否准确框住了它？"
        "回答 JSON: {{\"correct\": bool, \"reason\": str, \"confidence\": float}}"
    ),
    "S_spatial": (
        "[{subject}] 和 [{object}] 之间的关系真的是 [{predicate}] 吗？"
        "回答 JSON: {{\"correct\": bool, \"reason\": str}}"
    ),
    "A_obj_state": (
        "[{object_class}] 的状态真的是 [{state_value}] 吗？"
        "回答 JSON: {{\"correct\": bool, \"reason\": str}}"
    ),
    "A_action_interface": (
        "抓取点 [{grasp_point_2d}] 对于抓取 [{object_class}] 是否合理？"
        "回答 JSON: {{\"correct\": bool, \"reason\": str}}"
    ),
    "A_action_precondition": (
        "对于动作 [{action_type}]，[{can_do_now}] 的判断是否正确？"
        "回答 JSON: {{\"correct\": bool, \"reason\": str}}"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_field(label: dict, key: str, default=None):
    if key in label:
        return label[key]
    answer = label.get("answer")
    if isinstance(answer, dict) and key in answer:
        return answer[key]
    return default


def _detect_pipeline(label: dict) -> str:
    pipeline = _get_field(label, "pipeline") or ""
    if pipeline:
        return pipeline
    bucket = _get_field(label, "bucket") or ""
    if bucket.startswith("G"):
        return "G_grounding"
    elif bucket.startswith("S"):
        return "S_spatial"
    elif "state" in bucket.lower():
        return "A_obj_state"
    elif "interface" in bucket.lower():
        return "A_action_interface"
    elif "precondition" in bucket.lower():
        return "A_action_precondition"
    return "G_grounding"


def _build_vlm_prompt(label: dict, pipeline: str) -> str:
    template = _PROMPT_TEMPLATES.get(pipeline, _PROMPT_TEMPLATES["G_grounding"])
    field_map = {
        "object_class": str(_get_field(label, "object_class") or _get_field(label, "object_name") or "unknown"),
        "bbox": str(_get_field(label, "bbox") or []),
        "subject": str(_get_field(label, "subject") or ""),
        "object": str(_get_field(label, "object") or ""),
        "predicate": str(_get_field(label, "predicate") or ""),
        "state_value": str(_get_field(label, "state_value") or ""),
        "grasp_point_2d": str(_get_field(label, "grasp_point_2d") or []),
        "action_type": str(_get_field(label, "action_type") or ""),
        "can_do_now": str(_get_field(label, "can_do_now") or ""),
    }
    try:
        return template.format(**field_map)
    except KeyError:
        return template


def _call_vlm(prompt: str, image_path: str = None) -> dict:
    """Call VLM API and return parsed JSON response."""
    messages = []
    content_parts = []

    if image_path and os.path.isfile(image_path):
        import base64
        with open(image_path, "rb") as img_f:
            img_b64 = base64.b64encode(img_f.read()).decode()
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    content_parts.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content_parts})

    headers = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"

    payload = {
        "model": VLM_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(VLM_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("VLM call failed: %s", e)
        return {"correct": True, "reason": "vlm_call_failed", "confidence": 0.0}


# ---------------------------------------------------------------------------
# Sampling & Reflection Logic
# ---------------------------------------------------------------------------

def _collect_samples(directory: Path, n: int) -> list:
    """Collect up to n random samples from jsonl files in directory."""
    all_entries = []
    if not directory.exists():
        return []

    for fp in directory.rglob("*.jsonl"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        entry["_source_file"] = str(fp)
                        entry["_source_dir"] = str(directory)
                        all_entries.append(entry)
        except (json.JSONDecodeError, OSError):
            continue

    if len(all_entries) <= n:
        return all_entries
    return random.sample(all_entries, n)


def _resolve_image_path(label: dict) -> str:
    """Try to resolve the image path for a label."""
    img = _get_field(label, "image_path") or _get_field(label, "image") or ""
    if img and os.path.isabs(img):
        return img
    if img:
        candidate = BASE_DIR / img
        if candidate.exists():
            return str(candidate)
    return ""


def _downgrade_sample(label: dict, source_dir: str, vlm_result: dict):
    """Move sample from current tier to lower tier."""
    pipeline = _detect_pipeline(label)
    clean_label = {k: v for k, v in label.items()
                   if not k.startswith("_")}

    enriched = {
        **clean_label,
        "_reflection_result": vlm_result,
        "_downgraded_at": datetime.now(timezone.utc).isoformat(),
    }

    if VERIFIED_DIR.as_posix() in source_dir:
        target_dir = REVIEW_DIR / pipeline
    else:
        target_dir = REJECTED_DIR / pipeline

    target_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    target_file = target_dir / f"{pipeline}_{date_str}_reflection.jsonl"

    with open(target_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def run_reflection():
    """Run one reflection round: sample, verify with VLM, downgrade if needed."""
    n_per_source = max(1, SAMPLE_SIZE // 2)
    samples_verified = _collect_samples(VERIFIED_DIR, n_per_source)
    samples_review = _collect_samples(REVIEW_DIR, n_per_source)
    all_samples = samples_verified + samples_review

    if not all_samples:
        logger.info("Reflection: no samples found")
        return

    logger.info("Reflection: processing %d samples", len(all_samples))
    downgraded = 0

    for sample in all_samples:
        pipeline = _detect_pipeline(sample)
        prompt = _build_vlm_prompt(sample, pipeline)
        image_path = _resolve_image_path(sample)
        vlm_result = _call_vlm(prompt, image_path)

        correct = vlm_result.get("correct", True)
        confidence = vlm_result.get("confidence", 0.0)

        if not correct and confidence >= DOWNGRADE_CONFIDENCE_THRESHOLD:
            source_dir = sample.get("_source_dir", "")
            _downgrade_sample(sample, source_dir, vlm_result)
            downgraded += 1

    with _lock:
        _state["total_reflected"] += len(all_samples)
        _state["total_downgraded"] += downgraded
        _state["last_reflection_time"] = datetime.now(timezone.utc).isoformat()
        history_entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "samples": len(all_samples),
            "downgraded": downgraded,
        }
        _state["reflection_history"].append(history_entry)
        if len(_state["reflection_history"]) > 20:
            _state["reflection_history"] = _state["reflection_history"][-20:]

    logger.info("Reflection complete: %d samples, %d downgraded", len(all_samples), downgraded)


# ---------------------------------------------------------------------------
# Background Thread
# ---------------------------------------------------------------------------

def _background_reflector():
    logger.info("Background reflector started (interval=%ds, sample_size=%d)", REFLECTION_INTERVAL, SAMPLE_SIZE)
    while True:
        try:
            run_reflection()
        except Exception:
            logger.exception("Error during reflection")
        time.sleep(REFLECTION_INTERVAL)


# ---------------------------------------------------------------------------
# Flask Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "reflection"})


@app.route("/stats", methods=["GET"])
def stats():
    with _lock:
        return jsonify({
            "total_reflected": _state["total_reflected"],
            "total_downgraded": _state["total_downgraded"],
            "last_reflection_time": _state["last_reflection_time"],
            "reflection_history": _state["reflection_history"],
        })


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reflector_thread = threading.Thread(target=_background_reflector, daemon=True)
    reflector_thread.start()
    logger.info("Starting Reflection Service on port 10308 (interval: %ds)", REFLECTION_INTERVAL)
    app.run(host="0.0.0.0", port=10308, debug=False)
