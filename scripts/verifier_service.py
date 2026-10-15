"""GSA Verifier Service — Pipeline-aware verification with data archival.

Features:
- 10s scan interval for near-real-time verification
- Automatic archival: verified -> verified/, rejected -> rejected/
- Incremental statistics persisted to disk (survives restarts)
- Per-pipeline pass rate tracking
- Human review queue management

Flask app on port 10306.
"""

import json
import logging
import os
import base64
import io
import re
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests as _requests
from flask import Flask, jsonify

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
LABEL_DIR = BASE_DIR / "data" / "output" / "gsa_flywheel_exp" / "labels"
VERIFICATION_DIR = BASE_DIR / "data" / "output" / "gsa_flywheel_exp" / "verification"
VERIFIED_DIR = BASE_DIR / "data" / "output" / "gsa_flywheel_exp" / "verified"
REJECTED_DIR = BASE_DIR / "data" / "output" / "gsa_flywheel_exp" / "rejected"
REVIEW_DIR = BASE_DIR / "data" / "output" / "gsa_flywheel_exp" / "needs_review"
STATE_FILE = BASE_DIR / "data" / "output" / "gsa_flywheel_exp" / "verifier_state.json"

# Multi-model verification (API-endpoint based, legacy)
MULTI_MODEL_ENDPOINTS = [ep.strip() for ep in os.environ.get("VERIFIER_MODEL_ENDPOINTS", "").split(",") if ep.strip()]
MULTI_MODEL_KEYS = [k.strip() for k in os.environ.get("VERIFIER_MODEL_KEYS", "").split(",") if k.strip()]
MULTI_MODEL_NAMES = [n.strip() for n in os.environ.get("VERIFIER_MODEL_NAMES", "").split(",") if n.strip()]
MULTI_MODEL_ENABLED = len(MULTI_MODEL_ENDPOINTS) > 0

# ============================================================
# Specialist cross-verification (SOTA detection/pointing models)
# ============================================================
# Strategy (per requirements):
#   - The generator's main model stays Qwen3.5-397B (NOT changed here).
#   - Specialist verification runs ONLY on already-generated samples.
#   - Preferred specialist: doubao-seed VLM (Ark Responses API, vision-capable,
#     independently predicts bbox/point for geometric IoU/point-in-box comparison).
#   - Optional: DetectionVLM API or local HF model server.
#   - NO silent degradation: if specialist verification is ENABLED but the
#     configured backend is unreachable, we surface the error (logged + counted)
#     rather than quietly skipping or falling back to a weaker check.
DETVLM_URL = os.environ.get("VERIFIER_DETVLM_URL", "")  # e.g. http://127.0.0.1:5054
VERIFIER_MODEL_SERVER = os.environ.get("VERIFIER_MODEL_SERVER", "")  # local HF server
# Doubao-Seed VLM (Ark Responses API, vision-capable, for geometric cross-check)
SEED_VLM_URL = os.environ.get("VERIFIER_SEED_URL", "https://ark.cn-beijing.volces.com/api/v3/responses")
SEED_VLM_KEY = os.environ.get("VERIFIER_SEED_KEY", "")
SEED_VLM_MODEL = os.environ.get("VERIFIER_SEED_MODEL", "doubao-seed-2-0-lite-260428")
SEED_VLM_ENABLED = os.environ.get("VERIFIER_SEED_ENABLED", "1") == "1"
# Specialist enabled if any vision backend is configured
SPECIALIST_API_ENABLED = os.environ.get("VERIFIER_SPECIALIST_API", "0") == "1"
SPECIALIST_VERIFY_ENABLED = bool(DETVLM_URL) or bool(VERIFIER_MODEL_SERVER) or SEED_VLM_ENABLED or SPECIALIST_API_ENABLED
# Hard-fail policy: when True, an unreachable specialist backend raises/logs loudly
SPECIALIST_STRICT = os.environ.get("VERIFIER_SPECIALIST_STRICT", "1") == "1"
SPECIALIST_IOU_THRESHOLD = float(os.environ.get("VERIFIER_IOU_THRESHOLD", "0.4"))
SPECIALIST_TIMEOUT = float(os.environ.get("VERIFIER_SPECIALIST_TIMEOUT", "30"))
# Fraction of verified-candidate labels to cross-verify (bounds cost)
SPECIALIST_SAMPLE_RATE = float(os.environ.get("VERIFIER_SPECIALIST_SAMPLE_RATE", "0.15"))
# Track specialist backend errors (surfaced in /stats; no silent degradation)
_specialist_errors = {"count": 0, "last_error": "", "backend": ""}

# Primary VLM for scanner verification (single-model fallback)
VLM_URL = os.environ.get("VLM_URL", "")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen3.5-397B-A17B")
SCANNER_VLM_ENABLED = bool(VLM_URL)

SCAN_INTERVAL = 10
RESCAN_INTERVAL = 60  # Re-verify existing data every 60s with stricter criteria
PASS_THRESHOLD = 0.85  # Score >= 0.85 to pass (was 0.8)

WEIGHTS = {
    "schema_valid": 0.20,
    "content_quality": 0.30,  # Increased: content matters most
    "geometry_valid": 0.25,   # Increased: bbox quality is critical
    "consistency": 0.15,
    "confidence_check": 0.10, # Decreased: model confidence is less reliable
}

GENERIC_NAMES: set = {
    "object", "thing", "item", "stuff", "something",
    "it", "this", "that", "one", "unknown",
}

NON_MANIPULABLE: set = {
    "wall", "floor", "ceiling", "background", "sky", "ground", "room",
    "door", "window", "staircase", "stairs", "carpet", "rug",
    "curtain", "blinds", "light_fixture", "chandelier",
    "countertop", "kitchen_counter", "bathroom_counter",
    "interior_door", "exterior_door", "sliding_door", "glass_door",
    "hallway", "corridor", "passage", "entrance",
    "baseboard", "molding", "trim", "railing", "banister",
    "fireplace", "mantle", "chimney",
    "garage_door", "fence", "gate",
}

NON_GRASPABLE: set = {
    "wall", "floor", "ceiling", "door", "window", "staircase",
    "chair_backrest", "backrest", "wall_piece", "damaged_wall",
    "carpet", "rug", "countertop", "fireplace",
    "fence", "gate", "garage_door", "railing", "banister",
    "stairs", "roof", "chimney", "mantle",
}

VALID_PREDICATES: set = {
    "left-of", "right-of", "above", "below", "on-top-of", "under",
    "inside", "next-to", "in-front-of", "behind", "overlapping",
    "near", "far-from", "larger-than", "smaller-than",
    "stacked-on", "supporting", "occluding", "occluded-by",
}

VALID_STATE_TYPES: set = {
    "pose", "open_close", "occupancy", "cleanliness", "damage",
    "liquid_level", "on_off", "rotation_angle", "temperature", "other",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("verifier_service")

# ---------------------------------------------------------------------------
# Flask App & State
# ---------------------------------------------------------------------------

app = Flask(__name__)
_lock = threading.Lock()

# Persistent state (loaded from disk on startup)
_state: dict = {
    "total_scanned": 0,
    "total_verified": 0,
    "total_rejected": 0,
    "total_needs_review": 0,
    "last_scan_time": None,
    "processed_files": [],
    "scan_count": 0,
    "pipeline_stats": {},
    "hourly_throughput": [],
    "start_time": None,
    "recent_results": [],  # sliding window for pass@100
    "recent_1000_results": [],  # sliding window for pass@1000
    "per_minute_throughput": [],  # [{ts, ver, rej, rev}]
    "rejection_reasons": {},  # {reason_key: count}
    "score_histogram": [0] * 10,  # 10 buckets [0-0.1,...,0.9-1.0]
}


def _load_state():
    """Load persisted state from disk."""
    global _state
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
            _state.update(saved)
            logger.info(f"Loaded persisted state: {_state['total_scanned']} previously scanned")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")


def _save_state():
    """Persist state to disk."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(_state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


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


def _is_valid_bbox(bbox) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        return x1 < x2 and y1 < y2
    except (TypeError, ValueError):
        return False


def _bbox_in_bounds(bbox, w: int = 1920, h: int = 1080) -> bool:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h


def _point_in_bbox(pt, bbox) -> bool:
    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
        return False
    try:
        px, py = float(pt[0]), float(pt[1])
        x1, y1, x2, y2 = (float(v) for v in bbox)
        return x1 <= px <= x2 and y1 <= py <= y2
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Per-pipeline verification checks
# ---------------------------------------------------------------------------

def _checks_G_grounding(label: dict) -> dict:
    obj_class = _get_field(label, "object_class") or _get_field(label, "object_name") or ""
    bbox = _get_field(label, "bbox")
    confidence = _get_field(label, "confidence")
    image_size = _get_field(label, "image_size") or [1920, 1080]

    # Schema: must have class + valid bbox
    schema = 1.0 if (obj_class and _is_valid_bbox(bbox)) else 0.0

    # Content quality — strict naming checks
    obj_lower = obj_class.strip().lower().replace(" ", "_")
    if not obj_class or obj_lower in GENERIC_NAMES:
        content = 0.0
    elif len(obj_class.strip()) < 3:
        content = 0.3  # Too short name
    elif obj_lower in NON_MANIPULABLE or any(nm in obj_lower for nm in NON_MANIPULABLE):
        content = 0.3  # Non-manipulable background elements
    elif "_" not in obj_lower and len(obj_lower.split()) == 1 and len(obj_lower) < 5:
        content = 0.6  # Single short word, likely too generic
    else:
        content = 1.0

    # Geometry — strict bbox quality
    if bbox and _is_valid_bbox(bbox):
        img_w, img_h = image_size[0], image_size[1]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_area = bbox_w * bbox_h
        img_area = img_w * img_h
        aspect_ratio = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)

        if x1 <= 1 and y1 <= 1 and x2 >= img_w - 2 and y2 >= img_h - 2:
            geometry = 0.1  # Full-image bbox = garbage
        elif bbox_area > 0.5 * img_area:
            geometry = 0.2  # Too large — stricter threshold
        elif bbox_area > 0.3 * img_area:
            geometry = 0.4  # Large but may be valid for furniture
        elif bbox_area < 0.001 * img_area:
            geometry = 0.3  # Too small (likely noise)
        elif aspect_ratio > 10:
            geometry = 0.4  # Extremely elongated (likely wrong)
        elif bbox_w < 5 or bbox_h < 5:
            geometry = 0.3  # Degenerate bbox
        else:
            geometry = 1.0

        # Image quality check
        if img_w < 200 or img_h < 200:
            geometry = min(geometry, 0.4)  # Too low resolution
        elif img_w < 400 or img_h < 400:
            geometry = min(geometry, 0.7)  # Low quality
    else:
        geometry = 0.0

    # Image quality penalty
    img_w, img_h = image_size[0], image_size[1]
    if img_w < 200 or img_h < 200:
        geometry = min(geometry, 0.4)
    elif img_w < 400 or img_h < 400:
        geometry = min(geometry, 0.7)

    # Consistency: check if bbox makes sense for the object name
    consistency = 1.0

    # Confidence: strict grading — detect template values
    if isinstance(confidence, (int, float)):
        if abs(confidence - 0.85) < 0.001 or abs(confidence - 0.9) < 0.001:
            conf_check = 0.6  # Template confidence value
        elif confidence >= 0.92:
            conf_check = 1.0
        elif confidence >= 0.7:
            conf_check = 0.7
        elif confidence >= 0.5:
            conf_check = 0.4
        else:
            conf_check = 0.0
    else:
        conf_check = 0.3

    return {"schema_valid": schema, "content_quality": content, "geometry_valid": geometry,
            "consistency": consistency, "confidence_check": conf_check}


def _checks_S_spatial(label: dict) -> dict:
    subject = _get_field(label, "subject") or ""
    predicate = _get_field(label, "predicate") or ""
    obj = _get_field(label, "object") or ""
    confidence = _get_field(label, "confidence")
    reasoning = _get_field(label, "reasoning") or ""

    schema = 1.0 if (subject.strip() and predicate.strip() and obj.strip()) else 0.0

    # Content: predicate must be valid, subject != object
    subj_lower = subject.strip().lower().replace(" ", "_")
    obj_lower = obj.strip().lower().replace(" ", "_")
    pred_lower = predicate.strip().lower()
    if pred_lower not in VALID_PREDICATES:
        content = 0.2
    elif subject.strip().lower() == obj.strip().lower():
        content = 0.0  # self-referential relationship
    elif subj_lower in GENERIC_NAMES or obj_lower in GENERIC_NAMES:
        content = 0.4  # generic object names
    else:
        content = 1.0

    # Semantic: both subject and object shouldn't be non-manipulable
    subj_is_nm = any(nm in subj_lower for nm in NON_MANIPULABLE)
    obj_is_nm = any(nm in obj_lower for nm in NON_MANIPULABLE)
    if subj_is_nm and obj_is_nm:
        content = min(content, 0.3)

    # Large objects can't be "on-top-of" (bed on-top-of X is wrong)
    large_objects = {"bed", "sofa", "couch", "table", "wardrobe", "cabinet", "refrigerator", "bathtub"}
    if pred_lower == "on-top-of" and any(lo in subj_lower for lo in large_objects):
        content = min(content, 0.3)

    geometry = 1.0  # spatial relations don't have geometry

    # Consistency: reasoning quality
    if reasoning and len(reasoning) > 15:
        consistency = 1.0
    elif reasoning:
        consistency = 0.8
    else:
        consistency = 0.5

    conf_check = 1.0 if (isinstance(confidence, (int, float)) and confidence >= 0.8) else (0.7 if isinstance(confidence, (int, float)) and confidence >= 0.5 else 0.3)

    return {"schema_valid": schema, "content_quality": content, "geometry_valid": geometry,
            "consistency": consistency, "confidence_check": conf_check}


def _checks_A_identity_attr(label: dict) -> dict:
    obj_class = _get_field(label, "object_class") or _get_field(label, "object_name") or ""
    bbox = _get_field(label, "bbox")
    attrs = _get_field(label, "attributes")
    confidence = _get_field(label, "confidence")
    image_size = _get_field(label, "image_size") or [1920, 1080]

    has_attrs = isinstance(attrs, dict) and sum(1 for v in attrs.values() if v not in (None, "", "null", "unknown")) >= 2
    schema = 1.0 if (obj_class and _is_valid_bbox(bbox) and isinstance(attrs, dict)) else 0.0

    # Content: need specific object name + at least 2 non-null attributes
    if not obj_class or obj_class.strip().lower() in GENERIC_NAMES:
        content = 0.2
    elif not has_attrs:
        content = 0.4
    else:
        content = 1.0

    # Geometry: bbox quality
    if bbox and _is_valid_bbox(bbox):
        img_w, img_h = image_size[0], image_size[1]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = img_w * img_h
        if x1 <= 1 and y1 <= 1 and x2 >= img_w - 2 and y2 >= img_h - 2:
            geometry = 0.2
        elif bbox_area > 0.7 * img_area:
            geometry = 0.4
        elif bbox_area < 0.001 * img_area:
            geometry = 0.3
        else:
            geometry = 1.0

        # Image quality check
        if img_w < 200 or img_h < 200:
            geometry = min(geometry, 0.4)  # Too low resolution
        elif img_w < 400 or img_h < 400:
            geometry = min(geometry, 0.7)  # Low quality
    else:
        geometry = 0.0

    consistency = 1.0
    conf_check = 1.0 if (isinstance(confidence, (int, float)) and confidence >= 0.8) else (0.7 if isinstance(confidence, (int, float)) and confidence >= 0.5 else 0.3)

    return {"schema_valid": schema, "content_quality": content, "geometry_valid": geometry,
            "consistency": consistency, "confidence_check": conf_check}


def _checks_A_obj_state(label: dict) -> dict:
    obj_class = _get_field(label, "object_class") or _get_field(label, "object_name") or ""
    state_type = _get_field(label, "state_type") or ""
    state_value = _get_field(label, "state_value")
    confidence = _get_field(label, "confidence")
    reasoning = _get_field(label, "reasoning") or ""

    schema = 1.0 if (obj_class and state_type and state_value not in (None, "")) else 0.0

    # Content: state_type must be valid AND state_value must be specific
    valid_types = {"pose", "open_close", "occupancy", "on_off", "orientation", "cleanliness", "damage", "liquid_level", "temperature"}
    generic_values = {"unknown", "normal", "default", "none", "n/a", ""}
    st_lower = state_type.strip().lower()
    sv_lower = str(state_value).strip().lower() if state_value else ""

    if st_lower not in valid_types:
        content = 0.3
    elif state_value and sv_lower in generic_values:
        content = 0.4
    elif state_value and len(str(state_value)) < 2:
        content = 0.5
    else:
        content = 1.0

    # state_value vs state_type mismatch detection
    _pose_values = {"upright", "tilted", "fallen", "inverted", "sideways", "leaning", "flat", "standing", "hanging"}
    _open_close_values = {"open", "closed", "half_open", "half-open", "ajar", "partially_open"}
    _on_off_values = {"on", "off"}
    _occupancy_values = {"empty", "full", "occupied", "vacant", "partially_full", "half_full"}
    if st_lower == "pose" and sv_lower in (_on_off_values | _open_close_values):
        content = 0.3
    elif st_lower == "open_close" and sv_lower in (_pose_values | _on_off_values) - {"open"}:
        content = 0.3
    elif st_lower == "on_off" and sv_lower not in _on_off_values and sv_lower not in generic_values:
        content = min(content, 0.4)

    # Geometry: check bbox quality
    bbox = _get_field(label, "bbox")
    image_size = _get_field(label, "image_size") or [1920, 1080]
    if bbox and _is_valid_bbox(bbox):
        img_w, img_h = image_size[0], image_size[1]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = img_w * img_h
        if x1 <= 1 and y1 <= 1 and x2 >= img_w - 2 and y2 >= img_h - 2:
            geometry = 0.2
        elif bbox_area > 0.8 * img_area:
            geometry = 0.4
        else:
            geometry = 1.0
    else:
        geometry = 0.5

    # Consistency: state should make sense for the object type
    if reasoning and len(reasoning) > 15:
        consistency = 1.0
    elif reasoning:
        consistency = 0.8
    elif obj_class and state_type:
        consistency = 0.6
    else:
        consistency = 0.5

    conf_check = 1.0 if (isinstance(confidence, (int, float)) and confidence >= 0.8) else (0.7 if isinstance(confidence, (int, float)) and confidence >= 0.5 else 0.3)

    return {"schema_valid": schema, "content_quality": content, "geometry_valid": geometry,
            "consistency": consistency, "confidence_check": conf_check}


def _checks_A_action_precondition(label: dict) -> dict:
    action_type = _get_field(label, "action_type") or ""
    can_do_now = _get_field(label, "can_do_now")
    blocked_by = _get_field(label, "blocked_by")
    failure_reason = _get_field(label, "failure_reason") or ""
    reasoning = _get_field(label, "reasoning") or ""
    confidence = _get_field(label, "confidence")

    schema = 1.0 if (action_type and isinstance(can_do_now, bool) and isinstance(blocked_by, list)) else 0.0

    # Content: action_type must be specific, failure_reason must be detailed
    valid_actions = {"pick", "push", "place", "reorient", "slide", "pull",
                     "open", "close", "toggle_on", "toggle_off", "pour",
                     "grasp", "stack", "unstack", "slice", "fill", "empty"}
    valid_reasons = {"none", "occluded", "stacked_under", "too_large", "too_small", "unstable", "fragile", "unreachable", "stacked", "no_clearance"}
    if action_type.strip().lower() not in valid_actions:
        content = 0.3
    elif not can_do_now and failure_reason.strip().lower() not in valid_reasons:
        content = 0.5
    else:
        content = 1.0

    # Geometry: check bbox
    bbox = _get_field(label, "bbox")
    image_size = _get_field(label, "image_size") or [1920, 1080]
    if bbox and _is_valid_bbox(bbox):
        img_w, img_h = image_size[0], image_size[1]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x1 <= 1 and y1 <= 1 and x2 >= img_w - 2 and y2 >= img_h - 2:
            geometry = 0.2
        else:
            geometry = 1.0
    else:
        geometry = 0.4

    # Consistency: logical checks
    if can_do_now is True and blocked_by:
        consistency = 0.0
    elif can_do_now is False and not blocked_by and failure_reason == "none":
        consistency = 0.0
    elif reasoning:
        consistency = 1.0
    else:
        consistency = 0.7

    conf_check = 1.0 if (isinstance(confidence, (int, float)) and confidence >= 0.8) else (0.7 if isinstance(confidence, (int, float)) and confidence >= 0.5 else 0.3)

    return {"schema_valid": schema, "content_quality": content, "geometry_valid": geometry,
            "consistency": consistency, "confidence_check": conf_check}


def _checks_A_action_interface(label: dict) -> dict:
    action_type = _get_field(label, "action_type") or ""
    grasp = _get_field(label, "grasp_point_2d")
    push = _get_field(label, "push_contact_point_2d")
    jaw_axis = _get_field(label, "jaw_axis_2d")
    bbox = _get_field(label, "bbox")
    confidence = _get_field(label, "confidence")
    reasoning = _get_field(label, "reasoning") or ""
    grasp_strategy = _get_field(label, "grasp_strategy") or ""
    obj_class = _get_field(label, "object_class") or _get_field(label, "object_name") or ""
    image_size = _get_field(label, "image_size") or [1920, 1080]

    has_point = (isinstance(grasp, (list, tuple)) and len(grasp) == 2) or \
                (isinstance(push, (list, tuple)) and len(push) == 2)
    schema = 1.0 if (action_type and has_point) else 0.0

    # Content: must have valid action type and strategy
    valid_actions = {"pick", "push", "place", "slide", "pull", "open", "close",
                     "toggle_on", "toggle_off", "pour", "reorient", "grasp",
                     "stack", "unstack", "slice", "fill", "empty"}
    valid_strategies = {"top_down", "side", "pinch", "edge", "wrap", ""}
    obj_lower = obj_class.strip().lower().replace(" ", "_")
    if action_type.strip().lower() not in valid_actions:
        content = 0.3
    elif grasp_strategy and grasp_strategy not in valid_strategies:
        content = 0.6
    elif any(ng in obj_lower for ng in NON_GRASPABLE):
        content = 0.2
    else:
        content = 1.0

    # Penalize pick on oversized objects
    if action_type.strip().lower() == "pick" and bbox and _is_valid_bbox(bbox):
        img_w, img_h = float(image_size[0]), float(image_size[1])
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = img_w * img_h
        if bbox_area > 0.4 * img_area:
            content = min(content, 0.3)

    # Geometry: point should be within bbox + grasp_point position check
    geometry = 0.4
    if bbox and _is_valid_bbox(bbox):
        pt = grasp if isinstance(grasp, (list, tuple)) and len(grasp) == 2 else push
        if pt and _point_in_bbox(pt, bbox):
            geometry = 1.0
            # Check if grasp_point is exactly bbox center (default value)
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            px, py = float(pt[0]), float(pt[1])
            if abs(px - cx) < 1.0 and abs(py - cy) < 1.0:
                content = min(content, 0.6)
            # Check if point is in central 30% of bbox
            bw, bh = x2 - x1, y2 - y1
            margin_x, margin_y = bw * 0.35, bh * 0.35
            inner_x1, inner_y1 = x1 + margin_x, y1 + margin_y
            inner_x2, inner_y2 = x2 - margin_x, y2 - margin_y
            if not (inner_x1 <= px <= inner_x2 and inner_y1 <= py <= inner_y2):
                geometry = max(geometry * 0.7, 0.5)
        elif pt:
            geometry = 0.3
        else:
            geometry = 0.5

    # jaw_axis templateness check (mild penalty - VLM tends to default [1,0])
    if isinstance(jaw_axis, (list, tuple)) and len(jaw_axis) == 2:
        try:
            jx, jy = float(jaw_axis[0]), float(jaw_axis[1])
            if (abs(jx - 1.0) < 0.01 and abs(jy) < 0.01) or \
               (abs(jx) < 0.01 and abs(jy - 1.0) < 0.01):
                geometry = min(geometry, 0.8)  # mild penalty, not 0.5
        except (TypeError, ValueError):
            pass

    # Consistency: reasoning quality
    if reasoning and len(reasoning) > 20:
        consistency = 1.0
    elif reasoning:
        consistency = 0.9
    else:
        consistency = 0.7

    conf_check = 1.0 if (isinstance(confidence, (int, float)) and confidence >= 0.8) else (0.7 if isinstance(confidence, (int, float)) and confidence >= 0.5 else 0.3)

    return {"schema_valid": schema, "content_quality": content, "geometry_valid": geometry,
            "consistency": consistency, "confidence_check": conf_check}


def _checks_A_robot_state(label: dict) -> dict:
    """Verify robot end-effector / gripper state labels (merged from a-robot-state branch)."""
    state_type = _get_field(label, "state_type") or ""
    state_value = _get_field(label, "state_value")
    camera_view = _get_field(label, "camera_view") or ""
    visual_evidence = _get_field(label, "visual_evidence") or ""
    confidence = _get_field(label, "confidence")
    target_obj = _get_field(label, "target_object_class")

    valid_states = {"gripper_empty", "holding_object", "object_between_fingers", "contact_state", "tool_ready"}
    valid_values = {"yes", "no", "unknown"}
    st_lower = state_type.strip().lower()
    sv_lower = str(state_value).strip().lower() if state_value else ""

    # Schema: state_type valid + state_value present + camera_view present
    schema = 1.0 if (st_lower in valid_states and sv_lower in valid_values and camera_view) else 0.0

    # Content: require concrete visual evidence; penalize generic/empty
    if sv_lower == "unknown":
        content = 0.5  # unknown is acceptable but low-information
    elif not visual_evidence or len(visual_evidence.strip()) < 10:
        content = 0.4  # weak/missing visual evidence
    elif st_lower == "holding_object" and sv_lower == "yes" and not target_obj:
        content = 0.4  # holding_object=yes must name target
    else:
        content = 1.0

    # Geometry: robot_state has no bbox requirement (whole-frame judgment)
    geometry = 1.0

    # Consistency: camera_view should be a robot camera (wrist preferred)
    if camera_view not in ("wrist", "head", "external"):
        consistency = 0.3
    elif camera_view == "wrist":
        consistency = 1.0
    else:
        consistency = 0.7

    conf_check = 1.0 if (isinstance(confidence, (int, float)) and confidence >= 0.8) else (0.7 if isinstance(confidence, (int, float)) and confidence >= 0.5 else 0.3)

    return {"schema_valid": schema, "content_quality": content, "geometry_valid": geometry,
            "consistency": consistency, "confidence_check": conf_check}


_PIPELINE_CHECKS = {
    "G_grounding": _checks_G_grounding,
    "S_spatial": _checks_S_spatial,
    "A_identity_attr": _checks_A_identity_attr,
    "A_obj_state": _checks_A_obj_state,
    "A_action_precondition": _checks_A_action_precondition,
    "A_action_interface": _checks_A_action_interface,
    "A_robot_state": _checks_A_robot_state,
}


# ---------------------------------------------------------------------------
# Scoring & Tagging
# ---------------------------------------------------------------------------

def compute_score(checks: dict) -> float:
    return round(sum(WEIGHTS[k] * checks[k] for k in WEIGHTS), 4)


def assign_tag(score: float) -> str:
    if score >= PASS_THRESHOLD:
        return "verified_auto"
    if score >= 0.6:
        return "needs_human_review"
    return "rejected"


# ---------------------------------------------------------------------------
# Verification & Archival
# ---------------------------------------------------------------------------

def verify_label(label: dict) -> dict:
    pipeline = _get_field(label, "pipeline") or ""
    if not pipeline:
        bucket = _get_field(label, "bucket") or ""
        if bucket.startswith("G"):
            pipeline = "G_grounding"
        elif bucket.startswith("S"):
            pipeline = "S_spatial"
        elif bucket.startswith("A"):
            pipeline = "A_identity_attr"

    check_fn = _PIPELINE_CHECKS.get(pipeline)
    if check_fn is None:
        checks = {k: 0.5 for k in WEIGHTS}
    else:
        checks = check_fn(label)

    score = compute_score(checks)
    tag = assign_tag(score)

    # Specialist cross-verification (independent bbox/point prediction + geometric
    # comparison). Sampled to bound API cost. Only on verified-candidate
    # grounding/grasp labels.
    specialist = None
    if (SPECIALIST_VERIFY_ENABLED and tag == "verified_auto"
            and pipeline in ("G_grounding", "A_identity_attr", "A_obj_state", "A_action_interface")
            and random.random() < SPECIALIST_SAMPLE_RATE):
        specialist = specialist_verify(label)
        if specialist is not None:
            if not specialist["agree"]:
                # Specialist disagrees -> downgrade to review (don't hard-reject; model may miss)
                tag = "needs_human_review"
                score = min(score, 0.75)
            else:
                # Specialist confirms -> small confidence boost
                score = min(1.0, score + 0.03)

    # Add reject/review reason (lowest scoring check dimension)
    reason = ""
    if tag != "verified_auto" and checks:
        worst_key = min(checks, key=lambda k: checks[k])
        worst_val = checks[worst_key]
        reason = f"{worst_key}={worst_val:.2f}"
        if specialist is not None and not specialist["agree"]:
            reason = f"specialist_disagree({specialist['method']}:{specialist['evidence']})"
    result = {"pipeline": pipeline, "checks": checks, "score": score, "tag": tag, "reason": reason}
    if specialist is not None:
        result["specialist"] = specialist
    return result


def archive_label(label: dict, verification: dict):
    """Archive label to appropriate directory based on verification tag."""
    tag = verification["tag"]
    pipeline = verification["pipeline"] or "unknown"

    if tag == "verified_auto":
        target_dir = VERIFIED_DIR / pipeline
    elif tag == "rejected":
        target_dir = REJECTED_DIR / pipeline
    else:
        target_dir = REVIEW_DIR / pipeline

    target_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    target_file = target_dir / f"{pipeline}_{date_str}.jsonl"

    enriched = {**label, "_verification": verification}
    with open(target_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def verify_file(filepath: Path) -> list:
    labels = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(json.loads(line))
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to read %s: %s", filepath, e)
        return []

    results = []
    for label in labels:
        result = verify_label(label)
        result["source_file"] = filepath.name
        results.append(result)
        archive_label(label, result)
    return results


def run_scan() -> dict:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(LABEL_DIR.rglob("*.jsonl"))

    # Track by file path + line count (detect appended content)
    file_offsets = _state.get("file_offsets", {})
    all_results = []

    for filepath in jsonl_files:
        fpath = str(filepath)
        prev_lines = file_offsets.get(fpath, 0)

        # Read only new lines since last scan
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
        except OSError:
            continue

        current_lines = len(all_lines)
        if current_lines <= prev_lines:
            continue

        new_lines = all_lines[prev_lines:]
        new_labels = []
        for line in new_lines:
            line = line.strip()
            if line:
                try:
                    new_labels.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not new_labels:
            file_offsets[fpath] = current_lines
            continue

        logger.info("Verifying %d new labels from %s (lines %d-%d)", len(new_labels), filepath.name, prev_lines+1, current_lines)

        for label in new_labels:
            result = verify_label(label)
            result["source_file"] = filepath.name
            all_results.append(result)
            archive_label(label, result)

        file_offsets[fpath] = current_lines

    if not all_results:
        return _state.get("latest_report", {})

    # Update state
    with _lock:
        _state["file_offsets"] = file_offsets

    report = _build_report(all_results)
    _save_verification_results(all_results, report)

    report = _build_report(all_results)
    _save_verification_results(all_results, report)

    with _lock:
        _state["total_scanned"] += len(all_results)
        _state["total_verified"] += report["counts"]["verified_auto"]
        _state["total_rejected"] += report["counts"]["rejected"]
        _state["total_needs_review"] += report["counts"]["needs_human_review"]
        _state["last_scan_time"] = datetime.now(timezone.utc).isoformat()
        _state["scan_count"] += 1
        _state["latest_report"] = report

        # Update pipeline stats
        for pl, stats in report.get("pipeline_stats", {}).items():
            if pl not in _state["pipeline_stats"]:
                _state["pipeline_stats"][pl] = {"total": 0, "verified": 0, "rejected": 0, "review": 0}
            _state["pipeline_stats"][pl]["total"] += stats["total"]
            _state["pipeline_stats"][pl]["verified"] += stats.get("verified_auto", 0)
            _state["pipeline_stats"][pl]["rejected"] += stats.get("rejected", 0)
            _state["pipeline_stats"][pl]["review"] += stats.get("needs_human_review", 0)

        # Track throughput per 5-minute interval (for smoother chart)
        interval_key = datetime.now().strftime("%Y-%m-%d %H:%M")[:-1] + "0"  # Round to 10min
        throughput = _state.get("hourly_throughput", [])
        if throughput and throughput[-1].get("hour") == interval_key:
            throughput[-1]["count"] += len(all_results)
        else:
            throughput.append({"hour": interval_key, "count": len(all_results)})
            if len(throughput) > 288:  # Keep 48 hours at 10min intervals
                throughput = throughput[-288:]
        _state["hourly_throughput"] = throughput

        # Update pass@100 sliding window — store actual scores for richer metrics
        for r in all_results:
            _state["recent_results"].append(r.get("score", 0.0))
        # Keep only last 200 for flexibility
        if len(_state["recent_results"]) > 200:
            _state["recent_results"] = _state["recent_results"][-200:]

        # Update pass@1000 sliding window
        for r in all_results:
            _state["recent_1000_results"].append(r.get("score", 0.0))
        if len(_state["recent_1000_results"]) > 1000:
            _state["recent_1000_results"] = _state["recent_1000_results"][-1000:]

        # Update per-minute throughput
        minute_key = datetime.now().strftime("%H:%M")
        pmt = _state.get("per_minute_throughput", [])
        ver_count = sum(1 for r in all_results if r.get("tag") == "verified_auto")
        rej_count = sum(1 for r in all_results if r.get("tag") == "rejected")
        rev_count = sum(1 for r in all_results if r.get("tag") == "needs_human_review")
        if pmt and pmt[-1].get("ts") == minute_key:
            pmt[-1]["ver"] += ver_count
            pmt[-1]["rej"] += rej_count
            pmt[-1]["rev"] += rev_count
        else:
            pmt.append({"ts": minute_key, "ver": ver_count, "rej": rej_count, "rev": rev_count})
        if len(pmt) > 120:
            pmt = pmt[-120:]
        _state["per_minute_throughput"] = pmt

        # Update rejection reasons (lowest-scoring check as reason)
        for r in all_results:
            if r.get("tag") == "rejected":
                checks = r.get("checks", {})
                if checks:
                    worst = min(checks, key=lambda k: checks[k])
                    _state["rejection_reasons"][worst] = _state["rejection_reasons"].get(worst, 0) + 1

        # Update score histogram
        for r in all_results:
            score = r.get("score", 0.0)
            bucket = min(int(score * 10), 9)
            _state["score_histogram"][bucket] += 1

    _save_state()
    return report


def _build_report(results: list) -> dict:
    counts = {"verified_auto": 0, "needs_human_review": 0, "rejected": 0}
    pipeline_stats = {}

    for r in results:
        tag = r.get("tag", "rejected")
        counts[tag] = counts.get(tag, 0) + 1
        pl = r.get("pipeline", "unknown")
        pipeline_stats.setdefault(pl, {"total": 0, "verified_auto": 0, "needs_human_review": 0, "rejected": 0})
        pipeline_stats[pl]["total"] += 1
        pipeline_stats[pl][tag] += 1

    total = len(results)
    pipeline_pass_rates = {}
    for pl, s in pipeline_stats.items():
        passed = s["verified_auto"] + s["needs_human_review"]
        pipeline_pass_rates[pl] = round(passed / s["total"], 4) if s["total"] > 0 else 0.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_labels": total,
        "counts": counts,
        "pipeline_stats": pipeline_stats,
        "pipeline_pass_rates": pipeline_pass_rates,
    }


def _save_verification_results(results: list, report: dict):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = VERIFICATION_DIR / f"results_{ts}.jsonl"
    report_path = VERIFICATION_DIR / f"report_{ts}.json"
    try:
        VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d results to %s", len(results), results_path)
    except OSError as e:
        logger.error("Failed to save results: %s", e)


# ---------------------------------------------------------------------------
# Background Scanner
# ---------------------------------------------------------------------------

def _background_scanner():
    logger.info("Background scanner started (scan=%ds, rescan=%ds)", SCAN_INTERVAL, RESCAN_INTERVAL)
    last_rescan = time.time()
    while True:
        try:
            run_scan()
        except Exception:
            logger.exception("Error during scan")

        # Periodic rescan: re-verify existing verified data with stricter criteria
        if time.time() - last_rescan > RESCAN_INTERVAL:
            try:
                _rescan_verified()
            except Exception:
                logger.exception("Error during rescan")
            last_rescan = time.time()

        time.sleep(SCAN_INTERVAL)


def _rescan_verified():
    """Re-scan verified data with stricter criteria. Demote low-quality samples.

    Performance optimizations:
    - Only rescan today's shard files (filename contains today's date)
    - Skip files where all labels have _scan_count >= 20 (fully verified)
    - Only rewrite files when actual changes occurred
    """
    if not VERIFIED_DIR.exists():
        return

    today_str = datetime.now().strftime("%Y%m%d")
    demoted = 0
    rescanned = 0

    for fp in VERIFIED_DIR.rglob("*.jsonl"):
        # Only rescan today's shard files
        if today_str not in fp.name:
            continue

        labels = []
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        labels.append(json.loads(line))
        except Exception:
            continue

        if not labels:
            continue

        # Skip files where all labels are fully verified
        if all(lb.get("_scan_count", 1) >= 20 for lb in labels):
            continue

        keep = []
        file_modified = False

        for label in labels:
            # Skip individual labels already fully verified
            if label.get("_scan_count", 1) >= 20:
                keep.append(label)
                continue

            rescanned += 1
            clean_label = {k: v for k, v in label.items() if k != "_verification"}
            result = verify_label(clean_label)

            if result["score"] < 0.9:
                demoted += 1
                file_modified = True
                pipeline = result["pipeline"] or "unknown"
                review_dir = REVIEW_DIR / pipeline
                review_dir.mkdir(parents=True, exist_ok=True)
                date_str = datetime.now().strftime("%Y%m%d")
                review_file = review_dir / f"{pipeline}_{date_str}_rescan.jsonl"
                enriched = {**clean_label, "_verification": result, "_demoted_from": "verified"}
                with open(review_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                with _lock:
                    _state["total_verified"] = max(0, _state["total_verified"] - 1)
                    _state["total_needs_review"] += 1
            else:
                # VLM secondary confirmation (sampled - max 10 per rescan cycle)
                if SCANNER_VLM_ENABLED and rescanned <= 10:
                    try:
                        vlm_result = _call_model_verify(VLM_URL, VLM_API_KEY, VLM_MODEL, _build_verify_prompt(clean_label))
                        if vlm_result.get("scene_type"):
                            label["scene_type"] = vlm_result["scene_type"]
                            file_modified = True
                        if not vlm_result.get("correct", True):
                            demoted += 1
                            file_modified = True
                            pipeline = result["pipeline"] or "unknown"
                            review_dir = REVIEW_DIR / pipeline
                            review_dir.mkdir(parents=True, exist_ok=True)
                            date_str = datetime.now().strftime("%Y%m%d")
                            review_file = review_dir / f"{pipeline}_{date_str}_vlm_demote.jsonl"
                            enriched = {**clean_label, "_verification": result, "_vlm_reject": vlm_result, "_demoted_from": "verified"}
                            with open(review_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                            with _lock:
                                _state["total_verified"] = max(0, _state["total_verified"] - 1)
                                _state["total_needs_review"] += 1
                            continue
                    except Exception as e:
                        logger.debug("VLM verify failed: %s", e)
                        continue
                label["_scan_count"] = label.get("_scan_count", 1) + 1
                file_modified = True
                keep.append(label)

        # Only rewrite when there are actual changes
        if file_modified:
            if keep:
                with open(fp, "w", encoding="utf-8") as f:
                    for item in keep:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif labels:
                fp.unlink()

    if rescanned > 0:
        logger.info("Rescan: checked %d verified labels, demoted %d (%.1f%%)", rescanned, demoted, demoted/rescanned*100 if rescanned else 0)
        _save_state()


# ---------------------------------------------------------------------------
# Multi-Model Cross-Verification
# ---------------------------------------------------------------------------

def _build_verify_prompt(label: dict) -> str:
    """Build a VLM prompt to verify a generated label. Also requests scene classification."""
    pipeline = label.get("pipeline", "")
    obj = label.get("object_class", "") or label.get("object_name", "")
    bbox = label.get("bbox", [])
    scene_req = ' Also classify the scene from: tabletop_manipulation/kitchen/living_room/bedroom/bathroom/office/warehouse/hallway/outdoor/robot_workspace/shelf_storage/other.'
    fmt = ' Reply JSON: {{"correct": true/false, "reason": "...", "scene_type": "..."}}'
    if pipeline == "G_grounding":
        return f'Verify: Is there a "{obj}" in this image at bbox {bbox}?{scene_req}{fmt}'
    elif pipeline == "S_spatial":
        subj = label.get("subject", "")
        pred = label.get("predicate", "")
        obj2 = label.get("object", "")
        return f'Verify: Is the spatial relation "{subj} {pred} {obj2}" correct?{scene_req}{fmt}'
    elif pipeline == "A_obj_state":
        st = label.get("state_value", "")
        return f'Verify: Is the state of "{obj}" really "{st}"?{scene_req}{fmt}'
    elif pipeline == "A_action_interface":
        gp = label.get("grasp_point_2d", [])
        return f'Verify: Is grasp point {gp} reasonable for "{obj}"?{scene_req}{fmt}'
    elif pipeline == "A_action_precondition":
        act = label.get("action_type", "")
        can = label.get("can_do_now", None)
        return f'Verify: For action "{act}" on "{obj}", is can_do_now={can} correct?{scene_req}{fmt}'
    return f'Verify this annotation: {json.dumps(label, ensure_ascii=False)[:300]}.{scene_req}{fmt}'



def _call_model_verify(endpoint: str, api_key: str, model_name: str, prompt: str) -> dict:
    """Call a single model for verification."""
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 256,
        }
        resp = _requests.post(f"{endpoint}/chat/completions", json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            return {"correct": True, "reason": f"API error {resp.status_code}"}
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "")
        import re
        m = re.search(r'\{[^}]*"correct"\s*:\s*(true|false)[^}]*\}', content, re.IGNORECASE)
        if m:
            parsed = json.loads(m.group(0).lower().replace("true", "true").replace("false", "false"))
            return {"correct": parsed.get("correct", True), "reason": parsed.get("reason", "")}
        return {"correct": True, "reason": "parse_failed"}
    except Exception as e:
        return {"correct": True, "reason": f"error: {str(e)[:100]}"}


def multi_model_verify(label: dict) -> dict:
    """Run multi-model verification on a label. Returns consensus result."""
    if not MULTI_MODEL_ENABLED:
        return {"consensus": "verified", "votes": [], "disagreement": False}

    prompt = _build_verify_prompt(label)
    votes = []

    with ThreadPoolExecutor(max_workers=len(MULTI_MODEL_ENDPOINTS)) as pool:
        futures = {}
        for i, ep in enumerate(MULTI_MODEL_ENDPOINTS):
            key = MULTI_MODEL_KEYS[i] if i < len(MULTI_MODEL_KEYS) else ""
            name = MULTI_MODEL_NAMES[i] if i < len(MULTI_MODEL_NAMES) else f"model_{i}"
            futures[pool.submit(_call_model_verify, ep, key, name, prompt)] = name

        for fut in as_completed(futures):
            model_name = futures[fut]
            result = fut.result()
            votes.append({"model": model_name, "correct": result["correct"], "reason": result["reason"]})

    correct_count = sum(1 for v in votes if v["correct"])
    total = len(votes)
    disagreement = 0 < correct_count < total

    if correct_count == total:
        consensus = "verified"
    elif correct_count == 0:
        consensus = "rejected"
    else:
        consensus = "needs_review"

    return {"consensus": consensus, "votes": votes, "disagreement": disagreement}


# ---------------------------------------------------------------------------
# Specialist HF Model Cross-Verification (geometric: IoU / point-in-box)
# ---------------------------------------------------------------------------

def _bbox_iou(box_a, box_b) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    try:
        xa = max(box_a[0], box_b[0]); ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2]); yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
    except (IndexError, TypeError):
        return 0.0


def _point_in_box(point, box, margin: int = 50) -> bool:
    """Check if a [x,y] point falls inside [x1,y1,x2,y2] box (with margin in 0-1000 units)."""
    try:
        return (box[0] - margin) <= point[0] <= (box[2] + margin) and \
               (box[1] - margin) <= point[1] <= (box[3] + margin)
    except (IndexError, TypeError):
        return False


def _encode_image_for_api(image_path: str, max_size: int = 960) -> Optional[str]:
    """Encode an image to a base64 data URL (resized) for VLM API calls."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None


def _api_vlm_predict(image_path: str, prompt: str) -> str:
    """Call the VLM API (Qwen3.5-397B) with an image + prompt, return raw text."""
    data_url = _encode_image_for_api(image_path)
    if not data_url or not VLM_URL:
        return ""
    try:
        headers = {"Content-Type": "application/json"}
        if VLM_API_KEY:
            headers["Authorization"] = f"Bearer {VLM_API_KEY}"
        payload = {
            "model": VLM_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0.1,
            "max_tokens": 256,
        }
        resp = _requests.post(f"{VLM_URL}/chat/completions", json=payload, headers=headers, timeout=SPECIALIST_TIMEOUT)
        if resp.status_code != 200:
            return ""
        return resp.json()["choices"][0]["message"].get("content", "") or ""
    except Exception:
        return ""


def _parse_boxes_1000(text: str) -> list:
    """Parse [x1,y1,x2,y2] boxes from text, assume already 0-1000 normalized."""
    boxes = []
    for m in re.finditer(r"[\[\(]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[\]\)]", text):
        vals = [float(x) for x in m.groups()]
        mx = max(vals)
        if mx <= 1.0:
            vals = [v * 1000 for v in vals]
        boxes.append([int(v) for v in vals])
    return boxes


def _parse_points_1000(text: str) -> list:
    """Parse (x,y) points from text, assume 0-1000 normalized."""
    points = []
    for m in re.finditer(r"[\[\(]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[\]\)]", text):
        x, y = float(m.group(1)), float(m.group(2))
        if max(x, y) <= 1.0:
            x, y = x * 1000, y * 1000
        points.append([int(x), int(y)])
    return points


def _api_ground(image_path: str, obj: str) -> list:
    """Ask VLM API to detect object bbox, return list of [x1,y1,x2,y2] in 0-1000."""
    prompt = (f'Detect the "{obj}" in this image. Output ONLY its bounding box as '
              f'[x1,y1,x2,y2] with coordinates normalized to 0-1000. If not present, output "none".')
    return _parse_boxes_1000(_api_vlm_predict(image_path, prompt))


def _api_point(image_path: str, query: str) -> list:
    """Ask VLM API to point at the target, return list of [x,y] in 0-1000."""
    prompt = (f'Point to {query} in this image. Output ONLY the point as (x,y) '
              f'with coordinates normalized to 0-1000. If not present, output "none".')
    return _parse_points_1000(_api_vlm_predict(image_path, query if False else prompt))


def _detvlm_call(image_path: str, task: str, categories: list) -> Optional[dict]:
    """Call the dedicated DetectionVLM API (/v1/detvlm). task: detection|pointing.

    Returns parsed response dict, or raises on failure when SPECIALIST_STRICT.
    Coordinates from DetectionVLM are in pixel space (bbox{x1,y1,x2,y2}, point{x,y}).
    """
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"detvlm: cannot read image {image_path}: {e}")
    payload = {
        "image": {"base64": img_b64},
        "task": task,
        "categories": categories,
        "return_visualization": False,
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    resp = _requests.post(f"{DETVLM_URL}/v1/detvlm", json=payload, timeout=SPECIALIST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _detvlm_ground(image_path: str, obj: str, iw: int, ih: int) -> list:
    """DetectionVLM detection -> list of [x1,y1,x2,y2] normalized to 0-1000."""
    data = _detvlm_call(image_path, "detection", [obj])
    boxes = []
    for pred in (data or {}).get("predictions", []):
        b = pred.get("bbox")
        if b and all(k in b for k in ("x1", "y1", "x2", "y2")):
            boxes.append([b["x1"] * 1000 / iw, b["y1"] * 1000 / ih,
                          b["x2"] * 1000 / iw, b["y2"] * 1000 / ih])
    return boxes


def _detvlm_point(image_path: str, query: str, iw: int, ih: int) -> list:
    """DetectionVLM pointing -> list of [x,y] normalized to 0-1000."""
    data = _detvlm_call(image_path, "pointing", [query])
    points = []
    for pred in (data or {}).get("predictions", []):
        p = pred.get("point")
        if p and "x" in p and "y" in p:
            points.append([p["x"] * 1000 / iw, p["y"] * 1000 / ih])
    return points


def _seed_vlm_predict(image_path: str, prompt: str) -> str:
    """Call doubao-seed VLM (Ark Responses API) with image, return raw text response."""
    if not SEED_VLM_KEY:
        raise RuntimeError("VERIFIER_SEED_KEY must be configured before model verification")
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
        # Resize if too large (keep under 200KB for API speed)
        from PIL import Image as _PILImage
        img = _PILImage.open(io.BytesIO(img_data)).convert("RGB")
        if max(img.size) > 800:
            img.thumbnail((800, 800))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        raise RuntimeError(f"seed_vlm: cannot encode image {image_path}: {e}")

    payload = {
        "model": SEED_VLM_MODEL,
        "stream": False,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"},
                {"type": "input_text", "text": prompt},
            ],
        }],
    }
    headers = {
        "Authorization": f"Bearer {SEED_VLM_KEY}",
        "Content-Type": "application/json",
    }
    resp = _requests.post(SEED_VLM_URL, json=payload, headers=headers, timeout=SPECIALIST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    # Extract text from Responses API output
    for output in data.get("output", []):
        if output.get("type") == "message":
            for content in output.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")
    return ""


def _seed_ground(image_path: str, obj: str) -> list:
    """Ask doubao-seed VLM to detect object bbox, return list of [x1,y1,x2,y2] in 0-1000."""
    prompt = (f'Detect the "{obj}" in this image. Output ONLY its bounding box as '
              f'[x1,y1,x2,y2] with coordinates normalized to 0-1000. If not present, output "none".')
    return _parse_boxes_1000(_seed_vlm_predict(image_path, prompt))


def _seed_point(image_path: str, query: str) -> list:
    """Ask doubao-seed VLM to point at target, return list of [x,y] in 0-1000."""
    prompt = (f'Point to {query} in this image. Output ONLY the point as [x,y] '
              f'with coordinates normalized to 0-1000. If not present, output "none".')
    return _parse_points_1000(_seed_vlm_predict(image_path, prompt))


def specialist_verify(label: dict) -> Optional[dict]:
    """Cross-verify grounding/pointing labels with a specialist HF model (geometric).

    For G_grounding / A_identity_attr / A_obj_state: ask grounding model to detect
      the object, compare its boxes to the label bbox via IoU.
    For A_action_interface (grasp point) / pointing: ask pointing model where to act,
      compare predicted point to label grasp_point / bbox via point-in-box.

    Returns {agree: bool, method: str, score: float, evidence: str} or None if N/A.
    """
    if not SPECIALIST_VERIFY_ENABLED:
        return None

    pipeline = label.get("pipeline", "") or label.get("dimension", "")
    image_path = label.get("image_path", "")
    if not image_path or not os.path.exists(image_path):
        return None

    obj = label.get("object_class") or label.get("object_name") or ""
    image_size = label.get("image_size") or [1000, 1000]
    iw, ih = image_size[0] or 1000, image_size[1] or 1000

    def _to_1000(bbox):
        """Convert label bbox (pixel) to 0-1000 normalized for comparison."""
        if not bbox or len(bbox) != 4:
            return None
        return [bbox[0] * 1000 / iw, bbox[1] * 1000 / ih, bbox[2] * 1000 / iw, bbox[3] * 1000 / ih]

    try:
        # Grounding-type: independent bbox prediction -> IoU comparison
        if pipeline in ("G_grounding", "A_identity_attr", "A_obj_state") and obj:
            label_box = _to_1000(label.get("bbox"))
            if label_box is None:
                return None
            pred_boxes, method = _specialist_predict_boxes(image_path, obj, iw, ih)
            if pred_boxes:
                best_iou = max(_bbox_iou(label_box, pb) for pb in pred_boxes)
                agree = best_iou >= SPECIALIST_IOU_THRESHOLD
                return {"agree": agree, "method": method, "score": round(best_iou, 3),
                        "evidence": f"best IoU={best_iou:.2f} vs {len(pred_boxes)} boxes"}
            # Point-in-box fallback (specialist pointing)
            pred_points, pmethod = _specialist_predict_points(image_path, obj, iw, ih)
            if not pred_points:
                return {"agree": False, "method": "point_in_box", "score": 0.0,
                        "evidence": f"could not locate '{obj}'"}
            hit = any(_point_in_box(p, label_box) for p in pred_points)
            return {"agree": hit, "method": "point_in_box", "score": 1.0 if hit else 0.0,
                    "evidence": f"{len(pred_points)} pts, in_box={hit}"}

        # Grasp/pointing-type: compare grasp point via point distance
        if pipeline == "A_action_interface" and obj:
            grasp = label.get("grasp_point_2d")
            if not grasp or len(grasp) != 2:
                return None
            grasp_1000 = [grasp[0] * 1000 / iw, grasp[1] * 1000 / ih]
            pred_points, method = _specialist_predict_points(image_path, f"where a robot should grasp the {obj}", iw, ih)
            if not pred_points:
                return {"agree": False, "method": method, "score": 0.0,
                        "evidence": f"no grasp point for '{obj}'"}
            min_dist = min(((p[0] - grasp_1000[0]) ** 2 + (p[1] - grasp_1000[1]) ** 2) ** 0.5 for p in pred_points)
            agree = min_dist <= 150
            score = max(0.0, 1.0 - min_dist / 500)
            return {"agree": agree, "method": method, "score": round(score, 3),
                    "evidence": f"min point dist={min_dist:.0f}/1000"}
    except Exception as e:
        # No silent degradation: surface specialist backend errors.
        _specialist_errors["count"] += 1
        _specialist_errors["last_error"] = str(e)[:200]
        if SPECIALIST_STRICT:
            logger.error("Specialist verification backend FAILED (no degradation): %s", e)
        else:
            logger.debug("specialist_verify error: %s", e)
        return None

    return None


def _specialist_predict_boxes(image_path: str, obj: str, iw: int, ih: int):
    """Predict bboxes (0-1000) using the configured specialist backend.

    Priority: DetectionVLM -> doubao-seed VLM -> local HF model server -> optional VLM API.
    Raises on backend failure (caught by caller; no silent fallback to weaker check).
    """
    # Preferred: dedicated DetectionVLM (if reachable)
    if DETVLM_URL:
        return _detvlm_ground(image_path, obj, iw, ih), "grounding_iou_detvlm"
    # doubao-seed VLM (vision-capable, SOTA, Ark API)
    if SEED_VLM_ENABLED:
        return _seed_ground(image_path, obj), "grounding_iou_seed"
    # Local HF model server (LocateAnything etc.)
    if VERIFIER_MODEL_SERVER:
        resp = _requests.post(
            f"{VERIFIER_MODEL_SERVER}/v1/ground",
            json={"image_path": image_path, "query": obj}, timeout=SPECIALIST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("boxes", []), "grounding_iou_local"
    # Optional grounding-capable VLM API (only if explicitly enabled)
    if SPECIALIST_API_ENABLED:
        return _api_ground(image_path, obj), "grounding_iou_api"
    return [], "none"


def _specialist_predict_points(image_path: str, query: str, iw: int, ih: int):
    """Predict points (0-1000) using the configured specialist backend."""
    if DETVLM_URL:
        return _detvlm_point(image_path, query, iw, ih), "point_detvlm"
    if SEED_VLM_ENABLED:
        return _seed_point(image_path, query), "point_seed"
    if VERIFIER_MODEL_SERVER:
        resp = _requests.post(
            f"{VERIFIER_MODEL_SERVER}/v1/point",
            json={"image_path": image_path, "query": query}, timeout=SPECIALIST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("points", []), "point_local"
    if SPECIALIST_API_ENABLED:
        return _api_point(image_path, query), "point_api"
    return [], "none"


# ---------------------------------------------------------------------------
# Flask Endpoints
# ---------------------------------------------------------------------------

@app.route("/status", methods=["GET"])
def status():
    with _lock:
        return jsonify({
            "service": "gsa_verifier",
            "port": 10306,
            "total_scanned": _state["total_scanned"],
            "total_verified": _state["total_verified"],
            "total_rejected": _state["total_rejected"],
            "total_needs_review": _state["total_needs_review"],
            "last_scan_time": _state["last_scan_time"],
            "scan_count": _state["scan_count"],
            "processed_files_count": len(_state["processed_files"]),
            "pipeline_stats": _state["pipeline_stats"],
        })


@app.route("/verify", methods=["GET"])
def verify():
    report = run_scan()
    return jsonify({"status": "scan_complete", "report": report})


@app.route("/report", methods=["GET"])
def report():
    with _lock:
        return jsonify(_state.get("latest_report", {}))


@app.route("/stats", methods=["GET"])
def stats():
    with _lock:
        total = _state["total_scanned"]
        recent = _state.get("recent_results", [])
        recent_100 = recent[-100:] if len(recent) >= 100 else recent
        pass_at_100 = round(sum(1 for s in recent_100 if s >= 0.85) / len(recent_100) * 100, 1) if recent_100 else 0.0
        avg_score_100 = round(sum(recent_100) / len(recent_100), 3) if recent_100 else 0.0

        recent_1k = _state.get("recent_1000_results", [])
        pass_at_1000 = round(sum(1 for s in recent_1k if s >= 0.85) / len(recent_1k) * 100, 1) if recent_1k else 0.0
        avg_score_1000 = round(sum(recent_1k) / len(recent_1k), 3) if recent_1k else 0.0

        # Compute verified@k from verified dir
        verified_at_k = {"k1": 0, "k5": 0, "k20": 0}
        try:
            if VERIFIED_DIR.exists():
                for fp in VERIFIED_DIR.rglob("*.jsonl"):
                    with open(fp, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    item = json.loads(line)
                                    sc = item.get("_scan_count", 1)
                                    if sc >= 1:
                                        verified_at_k["k1"] += 1
                                    if sc >= 5:
                                        verified_at_k["k5"] += 1
                                    if sc >= 20:
                                        verified_at_k["k20"] += 1
                                except json.JSONDecodeError:
                                    pass
        except Exception:
            pass

        return jsonify({
            "total_scanned": total,
            "total_verified": _state["total_verified"],
            "total_rejected": _state["total_rejected"],
            "total_needs_review": _state["total_needs_review"],
            "pass_rate": round(_state["total_verified"] / total * 100, 1) if total > 0 else 0.0,
            "reject_rate": round(_state["total_rejected"] / total * 100, 1) if total > 0 else 0.0,
            "pass_at_100": pass_at_100,
            "avg_score_100": avg_score_100,
            "pass_at_1000": pass_at_1000,
            "avg_score_1000": avg_score_1000,
            "recent_window_size": len(recent_100),
            "recent_1000_window_size": len(recent_1k),
            "pipeline_stats": _state["pipeline_stats"],
            "hourly_throughput": _state.get("hourly_throughput", [])[-24:],
            "per_minute_throughput": _state.get("per_minute_throughput", [])[-30:],
            "rejection_reasons": _state.get("rejection_reasons", {}),
            "score_histogram": _state.get("score_histogram", [0] * 10),
            "verified_at_k": verified_at_k,
            "specialist": {
                "enabled": SPECIALIST_VERIFY_ENABLED,
                "backend": ("detvlm" if DETVLM_URL else "seed" if SEED_VLM_ENABLED else "local_server" if VERIFIER_MODEL_SERVER else "api" if SPECIALIST_API_ENABLED else "none"),
                "strict": SPECIALIST_STRICT,
                "error_count": _specialist_errors["count"],
                "last_error": _specialist_errors["last_error"],
            },
        })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "verifier", "port": 10306})


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _load_state()
    if not _state.get("start_time"):
        _state["start_time"] = datetime.now(timezone.utc).isoformat()

    scanner_thread = threading.Thread(target=_background_scanner, daemon=True)
    scanner_thread.start()
    logger.info("Starting Verifier Service on port 10306 (scan interval: %ds)", SCAN_INTERVAL)
    app.run(host="0.0.0.0", port=10306, debug=False)
