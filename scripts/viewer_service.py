#!/usr/bin/env python3
"""GSA Data Flywheel Viewer Service — Rich Dashboard Edition.

Features:
- Multi-tab dashboard: Overview / Cases / Statistics / Schema / Export
- Detailed case viewer with image + annotations + VLM output
- Real-time statistics: throughput, pass rates, dimension breakdown
- Data format preview for each pipeline schema
- One-click export of verified data for training
- Auto-refresh every 30s

Runs on port 10307.
"""

import base64
import io
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request, send_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("GSA_DATA_ROOT", str(Path(__file__).resolve().parent.parent / "data" / "output" / "gsa_flywheel_exp")))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = DATA_ROOT / "labels"
VERIFIED_DIR = DATA_ROOT / "verified"
REJECTED_DIR = DATA_ROOT / "rejected"
REVIEW_DIR = DATA_ROOT / "needs_review"
VERIFICATION_DIR = DATA_ROOT / "verification"
VERIFIER_STATE_FILE = DATA_ROOT / "verifier_state.json"
VIEWER_PORT = int(os.environ.get("VIEWER_PORT", "10307"))

# Annotation platform paths
ANNOTATION_DIR = DATA_ROOT / "annotations"
ANNOTATION_FILE = ANNOTATION_DIR / "annotations.jsonl"
ANNOTATORS_FILE = ANNOTATION_DIR / "annotators.json"

DIMENSIONS = ["G_grounding", "S_spatial", "A_identity_attr", "A_obj_state", "A_action_precondition", "A_action_interface"]

# TTL cache for dashboard data (avoids re-scanning all files on every request)
_dashboard_cache: dict = {"data": None, "ts": 0}
_DASHBOARD_CACHE_TTL = 90  # seconds (raised: large datasets are expensive to rescan)

DIMENSION_COLORS = {
    "G_grounding": "#7ee787",
    "S_spatial": "#79c0ff",
    "A_identity_attr": "#f0c674",
    "A_obj_state": "#f08080",
    "A_action_precondition": "#ffa657",
    "A_action_interface": "#d2a8ff",
}

SCHEMA_EXAMPLES = {
    "G_grounding": {"pipeline": "G_grounding", "object_class": "cup", "bbox": [120, 80, 340, 290], "confidence": 0.92, "source": "bridge_data_v2"},
    "S_spatial": {"pipeline": "S_spatial", "subject": "cup", "predicate": "on-top-of", "object": "table", "confidence": 0.88},
    "A_identity_attr": {"pipeline": "A_identity_attr", "object_class": "mug", "bbox": [100, 50, 300, 250], "attributes": {"color": "blue", "material": "ceramic", "shape": "cylindrical", "size": "medium"}, "g_context": True, "confidence": 0.85},
    "A_obj_state": {"pipeline": "A_obj_state", "object_class": "drawer", "bbox": [50, 200, 400, 500], "state_type": "open_close", "state_value": "open", "s_context": False, "confidence": 0.91},
    "A_action_precondition": {"pipeline": "A_action_precondition", "scene_family": "tabletop_tidy", "action_type": "pick", "can_do_now": False, "blocked_by": ["plate_01"], "need_pre_action": "push_first", "failure_reason": "occluded", "s_context": True, "confidence": 0.87},
    "A_action_interface": {"pipeline": "A_action_interface", "scene_family": "tabletop_tidy", "action_type": "pick", "grasp_point_2d": [220, 150], "jaw_axis_2d": [0, 1], "g_context": True, "s_context": True, "confidence": 0.82},
}

# ---------------------------------------------------------------------------
# Prior Knowledge (Ontology Reference)
# ---------------------------------------------------------------------------

PRIOR_KNOWLEDGE = {
    "actions": [
        "pick", "put", "open", "close", "push", "pull", "slide",
        "toggle_on", "toggle_off", "slice", "pour", "fill", "empty",
        "clean", "heat", "cool", "break", "throw", "reorient",
        "grasp", "place_inside", "stack", "unstack", "navigate_to",
    ],
    "states": [
        "pose", "open_close", "occupancy", "on_off", "cleanliness",
        "temperature", "liquid_level", "damage", "orientation",
        "cooked", "sliced", "frozen", "wet", "dirty", "full", "empty",
    ],
    "attributes": [
        "color", "material", "shape", "size", "weight", "texture",
        "transparency", "rigidity", "fragility", "temperature",
    ],
    "predicates": [
        "left-of", "right-of", "above", "below", "on-top-of", "under",
        "inside", "next-to", "in-front-of", "behind", "overlapping",
        "near", "far-from", "stacked-on", "supporting", "occluding",
        "occluded-by", "containing", "attached-to",
    ],
    "scenes": [
        "tabletop_manipulation", "kitchen", "living_room", "bedroom",
        "bathroom", "office", "warehouse", "hallway", "staircase",
        "outdoor", "garage", "laundry_room", "dining_room",
        "toy_closeup", "robot_workspace", "shelf_storage",
    ],
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("viewer_service")

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list:
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return items


def _load_all_from_dir(directory: Path) -> list:
    items = []
    if not directory.exists():
        return items
    for fp in sorted(directory.rglob("*.jsonl")):
        items.extend(_load_jsonl(fp))
    return items


def _count_lines_in_dir(directory: Path):
    """Fast exact counts via line counting (no JSON parse).

    Returns (total_count, per_dimension_counts). Labels are organized in
    per-dimension subdirectories, so the parent dir name is the dimension.
    """
    total = 0
    per_dim: dict = {}
    if not directory.exists():
        return total, per_dim
    for fp in directory.rglob("*.jsonl"):
        try:
            with open(fp, "rb") as f:
                n = sum(1 for line in f if line.strip())
        except Exception:
            continue
        total += n
        dim = fp.parent.name
        per_dim[dim] = per_dim.get(dim, 0) + n
    return total, per_dim


def _load_recent_from_dir(directory: Path, max_labels: int = 8000) -> list:
    """Load a bounded sample of the most recent labels (newest files first).

    Used for secondary metrics (object diversity, scene/bbox/action stats, cases)
    that don't need the full dataset. Avoids loading millions of records.
    """
    items = []
    if not directory.exists():
        return items
    files = sorted(directory.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for fp in files:
        items.extend(_load_jsonl(fp))
        if len(items) >= max_labels:
            break
    return items[:max_labels]


def _encode_image_base64(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime = mime_map.get(suffix, "image/png")
    try:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""


def _derive_status(label: dict) -> str:
    """Derive display status from label verification info."""
    v = label.get("_verification", {})
    tag = v.get("tag", "")
    if tag == "verified_auto":
        return "verified"
    elif tag == "rejected":
        return "rejected"
    elif tag == "needs_human_review":
        return "review"
    return "pending"


def _compute_verified_at_k(vdir=None) -> dict:
    """Count verified labels by scan_count directly from files.

    Uses verifier_state.json aggregate if available (fast path); falls back to
    bounded file scan to avoid loading millions of records on large datasets.
    """
    _vdir = vdir or VERIFIED_DIR
    # Fast path: read pre-aggregated counts from verifier_state.json
    state_file = _vdir.parent / "verifier_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                st = json.load(f)
            vk = st.get("verified_at_k")
            if isinstance(vk, dict) and "k1" in vk:
                return {"k1": vk.get("k1", 0), "k5": vk.get("k5", 0), "k20": vk.get("k20", 0)}
        except Exception:
            pass

    k1 = k5 = k20 = 0
    if _vdir.exists():
        for fp in _vdir.rglob("*.jsonl"):
            try:
                for line in fp.read_text().splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    sc = item.get("_scan_count", 1)
                    if sc >= 1:
                        k1 += 1
                    if sc >= 5:
                        k5 += 1
                    if sc >= 20:
                        k20 += 1
            except Exception:
                continue
    return {"k1": k1, "k5": k5, "k20": k20}


def compute_dashboard_data(labels_dir=None, verified_dir=None, rejected_dir=None, review_dir=None) -> dict:
    """Compute comprehensive dashboard metrics (with TTL cache)."""
    global _dashboard_cache
    now = time.time()
    # Use cache if no custom dirs and cache is fresh
    if (labels_dir is None and verified_dir is None and rejected_dir is None and review_dir is None
            and _dashboard_cache["data"] is not None
            and (now - _dashboard_cache["ts"]) < _DASHBOARD_CACHE_TTL):
        return _dashboard_cache["data"]

    _labels_dir = labels_dir or LABELS_DIR
    _verified_dir = verified_dir or VERIFIED_DIR
    _rejected_dir = rejected_dir or REJECTED_DIR
    _review_dir = review_dir or REVIEW_DIR

    # Try verifier_state.json for quick stats before full scan
    state_file = (_labels_dir.parent / "verifier_state.json") if labels_dir else VERIFIER_STATE_FILE
    verifier_state = {}
    if state_file.exists():
        try:
            with open(state_file) as f:
                verifier_state = json.load(f)
        except Exception:
            pass

    # Compute pass@100 from verifier state — average score of last 100 verified samples
    recent_results = verifier_state.get("recent_results", [])
    recent_100 = recent_results[-100:] if len(recent_results) >= 100 else recent_results
    pass_at_100 = round(sum(1 for s in recent_100 if s >= 0.85) / len(recent_100) * 100, 1) if recent_100 else 0.0
    avg_score_100 = round(sum(recent_100) / len(recent_100), 3) if recent_100 else 0.0
    pass_at_100_window = len(recent_100)

    # Compute pass@1000
    recent_1k = verifier_state.get("recent_1000_results", [])
    pass_at_1000 = round(sum(1 for s in recent_1k if s >= 0.85) / len(recent_1k) * 100, 1) if recent_1k else 0.0

    # Fast exact counts (line counting, no JSON parse) — scales to millions
    total_labels, labels_per_dim = _count_lines_in_dir(_labels_dir)
    total_verified, verified_per_dim = _count_lines_in_dir(_verified_dir)
    total_rejected, rejected_per_dim = _count_lines_in_dir(_rejected_dir)
    total_review, review_per_dim = _count_lines_in_dir(_review_dir)

    # Bounded recent sample for secondary metrics (object/scene/bbox/action/cases)
    labels = _load_recent_from_dir(_labels_dir, max_labels=8000)

    # Dimension breakdown — exact counts from line-counting (parent dir = dimension)
    dim_stats = {}
    for dim in DIMENSIONS:
        count = labels_per_dim.get(dim, 0)
        v_count = verified_per_dim.get(dim, 0)
        r_count = rejected_per_dim.get(dim, 0)
        rev_count = review_per_dim.get(dim, 0)
        # pass_rate = verified / (verified + rejected + review) — all scanned samples
        scanned = v_count + r_count + rev_count
        pass_rate = round(v_count / scanned * 100, 1) if scanned > 0 else 0.0
        # progress = scanned / generated
        progress_pct = round(scanned / count * 100, 1) if count > 0 else 0.0
        # avg confidence from recent sample only (secondary metric)
        dim_recent = [l for l in labels if l.get("pipeline") == dim or l.get("dimension") == dim]
        avg_conf = round(sum(float(l.get("confidence", 0)) for l in dim_recent) / len(dim_recent), 3) if dim_recent else 0.0

        dim_stats[dim] = {
            "count": count,
            "verified": v_count,
            "rejected": r_count,
            "review": rev_count,
            "scanned": scanned,
            "pass_rate": pass_rate,
            "progress_pct": progress_pct,
            "avg_confidence": avg_conf,
            "color": DIMENSION_COLORS.get(dim, "#888"),
        }

    # Source breakdown
    source_counts = defaultdict(int)
    for label in labels:
        src = label.get("source") or label.get("dataset") or "other"
        source_counts[src] += 1

    # Sample cases — pick from ALL dimensions evenly (not just last N)
    sample_cases = []
    image_labels = {}
    for label in labels:
        img_path = label.get("image_path", "")
        if img_path and Path(img_path).exists():
            image_labels.setdefault(img_path, []).append(label)

    # Get latest labels per dimension (sorted by timestamp desc)
    per_dim_latest = {}
    for label in labels:
        dim = label.get("pipeline") or label.get("dimension", "")
        img_path = label.get("image_path", "")
        if img_path and Path(img_path).exists():
            per_dim_latest.setdefault(dim, []).append((label.get("timestamp", ""), img_path))

    # Take most recent 5 images per dimension (ensure all dims represented)
    selected_images = []
    for dim in DIMENSIONS:
        dim_entries = per_dim_latest.get(dim, [])
        dim_entries.sort(reverse=True)  # newest first
        seen = set()
        for _, img in dim_entries:
            if img not in seen:
                seen.add(img)
                selected_images.append((img, dim))
            if len(seen) >= 5:
                break

    # Build cases (newest first)
    seen_images = set()
    for img_path, primary_dim in selected_images:
        if len(sample_cases) >= 30:
            break
        if img_path in seen_images:
            continue
        seen_images.add(img_path)

        all_labels_for_image = image_labels.get(img_path, [])
        annotated_b64 = _render_multi_annotated_image(img_path, all_labels_for_image)
        primary = all_labels_for_image[0]
        sample_cases.append({
            "image_path": img_path,
            "filename": Path(img_path).name,
            "pipeline": primary.get("pipeline") or primary.get("dimension", ""),
            "confidence": primary.get("confidence", 0),
            "sample_id": primary.get("sample_id", ""),
            "source": primary.get("source", ""),
            "timestamp": primary.get("timestamp", ""),
            "label_count": len(all_labels_for_image),
            "label_data": {k: v for k, v in primary.items() if k not in ("image_path", "_verification")},
            "all_labels": [
                {k: v for k, v in l.items() if k not in ("image_path", "_verification")}
                for l in all_labels_for_image[:10]
            ],
            "verification_score": primary.get("_verification", {}).get("score"),
            "verification_tag": primary.get("_verification", {}).get("tag"),
            "status": _derive_status(primary),
            "scene_type": primary.get("scene_type", ""),
            "scan_count": primary.get("_scan_count", 0),
            "annotated_b64": annotated_b64,
        })

    # Hourly throughput from verifier state
    hourly_throughput = verifier_state.get("hourly_throughput", [])[-24:]

    # PassRate = total_verified / total_scanned (all samples that went through verifier)
    total_scanned_by_verifier = verifier_state.get("total_scanned", 0)
    total_verified_by_verifier = verifier_state.get("total_verified", 0)
    overall_pass_rate = round(total_verified_by_verifier / total_scanned_by_verifier * 100, 1) if total_scanned_by_verifier > 0 else 0.0
    total_processed = total_verified + total_rejected + total_review

    # Additional metrics
    all_objects = set()
    total_bbox_area_pct = []
    dead_letter_dir = DATA_ROOT / "dead_letter"
    dead_letter_count = 0
    if dead_letter_dir.exists():
        for fp in dead_letter_dir.rglob("*.jsonl"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    dead_letter_count += sum(1 for line in f if line.strip())
            except Exception:
                pass

    for label in labels:
        obj_cls = label.get("object_class") or label.get("object_name") or ""
        if obj_cls.strip():
            all_objects.add(obj_cls.strip().lower())
        bbox = label.get("bbox")
        img_size = label.get("image_size") or [1920, 1080]
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                img_w, img_h = float(img_size[0]), float(img_size[1])
                if x2 > x1 and y2 > y1 and img_w > 0 and img_h > 0:
                    pct = (x2 - x1) * (y2 - y1) / (img_w * img_h) * 100
                    total_bbox_area_pct.append(pct)
            except (TypeError, ValueError):
                pass

    unique_objects = len(all_objects)
    dims_with_data = sum(1 for d in DIMENSIONS if dim_stats.get(d, {}).get("count", 0) > 0)
    dim_coverage = round(dims_with_data / len(DIMENSIONS), 3) if DIMENSIONS else 0.0
    generator_self_check_rate = round(total_labels / (total_labels + dead_letter_count), 3) if (total_labels + dead_letter_count) > 0 else 0.0
    scanner_pass_rate = round(total_verified_by_verifier / total_scanned_by_verifier * 100, 1) if total_scanned_by_verifier > 0 else 0.0
    avg_bbox_area_pct = round(sum(total_bbox_area_pct) / len(total_bbox_area_pct), 2) if total_bbox_area_pct else 0.0

    # Action diversity
    all_actions = set()
    for label in labels:
        at = label.get("action_type", "")
        if at:
            all_actions.add(at.strip().lower())
    action_diversity = len(all_actions)

    # Scene type distribution
    scene_counts = {}
    for label in labels:
        st = label.get("scene_type", "unknown")
        if st:
            scene_counts[st] = scene_counts.get(st, 0) + 1

    result = {
        "summary": {
            "total_labels": total_labels,
            "total_scanned": total_scanned_by_verifier,
            "total_verified": total_verified_by_verifier,
            "total_rejected": total_rejected,
            "total_review": total_review,
            "overall_pass_rate": overall_pass_rate,
            "pass_at_100": pass_at_100,
            "avg_score_100": avg_score_100,
            "pass_at_100_window": pass_at_100_window,
            "pass_at_1000": pass_at_1000,
            "total_processed": total_processed,
            "unique_objects": unique_objects,
            "dim_coverage": dim_coverage,
            "generator_self_check_rate": generator_self_check_rate,
            "scanner_pass_rate": scanner_pass_rate,
            "avg_bbox_area_pct": avg_bbox_area_pct,
            "action_diversity": action_diversity,
            "verified_at_k": _compute_verified_at_k(_verified_dir),
        },
        "dim_stats": dim_stats,
        "source_counts": dict(source_counts),
        "scene_counts": scene_counts,
        "sample_cases": sample_cases[:30],
        "hourly_throughput": hourly_throughput,
        "schema_examples": SCHEMA_EXAMPLES,
        "dimension_colors": DIMENSION_COLORS,
        "prior_knowledge": PRIOR_KNOWLEDGE,
        "has_data": total_labels > 0,
    }

    # Store in cache for subsequent requests
    if labels_dir is None and verified_dir is None and rejected_dir is None and review_dir is None:
        _dashboard_cache["data"] = result
        _dashboard_cache["ts"] = time.time()

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    data = compute_dashboard_data()
    return render_template_string(HTML_TEMPLATE, **data)


@app.route("/api/data")
def api_data():
    data = compute_dashboard_data()
    return jsonify(data)


@app.route("/api/cases")
def api_cases():
    """Paginated case list with server-side filtering.

    Status filter determines which directory to read from:
      verified -> VERIFIED_DIR (has _verification.tag=verified_auto)
      rejected -> REJECTED_DIR
      review   -> REVIEW_DIR
      (none)   -> LABELS_DIR (raw generated, status=pending)
    Reading from the status-specific dir is required because raw LABELS_DIR
    entries have no _verification field (would all derive to "pending").
    """
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 20))
    dimension = request.args.get("dimension", "")
    scene = request.args.get("scene", "")
    status = request.args.get("status", "")

    status_dir = {
        "verified": VERIFIED_DIR,
        "rejected": REJECTED_DIR,
        "review": REVIEW_DIR,
    }
    src_dir = status_dir.get(status, LABELS_DIR)

    # If a specific dimension is requested, scope to its subdir for speed
    if dimension and (src_dir / dimension).exists():
        labels = _load_recent_from_dir(src_dir / dimension, max_labels=4000)
    else:
        labels = _load_recent_from_dir(src_dir, max_labels=4000)

    if dimension:
        labels = [l for l in labels if l.get("pipeline") == dimension or l.get("dimension") == dimension]
    if scene:
        labels = [l for l in labels if l.get("scene_type", "") == scene]
    # For status-specific dirs the entries already match; only re-derive for raw labels
    if status and src_dir is LABELS_DIR:
        labels = [l for l in labels if _derive_status(l) == status]

    total = len(labels)
    page_labels = labels[offset:offset + limit]

    cases = []
    for label in page_labels:
        img_path = label.get("image_path", "")
        cases.append({
            "label": {k: v for k, v in label.items() if k != "_verification"},
            "has_image": bool(img_path and Path(img_path).exists()),
            "verification": label.get("_verification", {}),
        })

    return jsonify({"cases": cases, "total": total, "offset": offset, "limit": limit})


@app.route("/api/case/<sample_id>")
def api_case_detail(sample_id):
    """Single case detail with annotated image."""
    labels = _load_all_from_dir(LABELS_DIR)
    for label in labels:
        if label.get("sample_id") == sample_id:
            img_path = label.get("image_path", "")
            img_b64 = ""
            annotated_b64 = ""
            if img_path and Path(img_path).exists():
                img_b64 = _encode_image_base64(Path(img_path))
                annotated_b64 = _render_annotated_image(img_path, label)
            return jsonify({
                "label": label,
                "image_b64": img_b64,
                "annotated_b64": annotated_b64,
                "verification": label.get("_verification", {}),
            })
    return jsonify({"error": "Not found"}), 404


@app.route("/api/image/<path:image_path>")
def api_image(image_path):
    """Serve an image file on demand instead of inlining base64."""
    p = Path("/") / image_path
    if not p.exists():
        p = DATA_ROOT / image_path
    if not p.is_file():
        return jsonify({"error": "Image not found"}), 404
    suffix = p.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mimetype = mime_map.get(suffix, "application/octet-stream")
    return send_file(str(p), mimetype=mimetype)


def _render_annotated_image(image_path: str, label: dict) -> str:
    """Render single label on image."""
    return _render_multi_annotated_image(image_path, [label])


def _render_multi_annotated_image(image_path: str, labels: list) -> str:
    """Render image with ALL labels' bbox/point/annotation overlays."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return ""

    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        colors = {
            "G_grounding": "#7ee787",
            "S_spatial": "#79c0ff",
            "A_identity_attr": "#f0c674",
            "A_obj_state": "#f08080",
            "A_action_precondition": "#ffa657",
            "A_action_interface": "#d2a8ff",
        }

        def denorm_bbox(bbox):
            """Convert stored bbox (pixel coords from denormalize_bbox) to draw coords.
            
            Labels store bbox in pixel coords (already denormalized by generator).
            Draw directly — they match the image's native resolution.
            """
            if not bbox or len(bbox) != 4:
                return None
            return [int(float(v)) for v in bbox]

        def denorm_point(pt):
            if not pt or len(pt) != 2:
                return None
            return [int(float(v)) for v in pt]

        for label in labels:
            pipeline = label.get("pipeline", "")
            color = colors.get(pipeline, "#ffffff")

            # Draw bbox
            bbox = label.get("bbox")
            if bbox:
                coords = denorm_bbox(bbox)
                if coords and not (coords[0] == 0 and coords[1] == 0 and coords[2] >= img_w - 2 and coords[3] >= img_h - 2):
                    draw.rectangle(coords, outline=color, width=3)
                    text = label.get("object_class") or label.get("object_name") or ""
                    conf = label.get("confidence", "")
                    label_text = f"{text} {conf:.2f}" if isinstance(conf, float) else text
                    if label_text:
                        ty = max(0, coords[1] - 16)
                        draw.rectangle([coords[0], ty, coords[0] + len(label_text) * 7 + 4, ty + 15], fill="#000000aa")
                        draw.text((coords[0] + 2, ty + 1), label_text, fill=color)

            # Draw grasp point
            grasp_pt = label.get("grasp_point_2d")
            if grasp_pt:
                pt = denorm_point(grasp_pt)
                if pt:
                    r = 8
                    draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill="#ff0000", outline="#ffffff", width=2)
                    draw.text((pt[0]+12, pt[1]-6), "grasp", fill="#ff4444")

            # Draw jaw axis
            jaw_axis = label.get("jaw_axis_2d")
            if jaw_axis and grasp_pt:
                pt = denorm_point(grasp_pt)
                if pt:
                    dx, dy = float(jaw_axis[0]), float(jaw_axis[1])
                    length = 25
                    draw.line([(pt[0]-int(dx*length), pt[1]-int(dy*length)),
                               (pt[0]+int(dx*length), pt[1]+int(dy*length))], fill="#00ff00", width=3)

            # Draw placement zones (semi-transparent overlay for empty regions)
            action_type = label.get("action_type", "")
            obj_name = label.get("object_class", "") or label.get("object_name", "")
            if action_type == "place" or "placement_zone" in str(obj_name):
                if bbox:
                    coords = denorm_bbox(bbox)
                    if coords:
                        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                        overlay_draw = ImageDraw.Draw(overlay)
                        overlay_draw.rectangle(coords, fill=(88, 166, 255, 40), outline="#58a6ff", width=2)
                        surface = label.get("place_surface", "surface")
                        overlay_draw.text((coords[0]+4, coords[1]+4), f"PLACE: {surface}", fill="#58a6ff")
                        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                        draw = ImageDraw.Draw(img)

            # Draw push point + direction
            push_pt = label.get("push_contact_point_2d")
            push_dir = label.get("push_direction_2d")
            if push_pt:
                pt = denorm_point(push_pt)
                if pt:
                    draw.ellipse([pt[0]-6, pt[1]-6, pt[0]+6, pt[1]+6], fill="#00aaff", outline="#ffffff", width=2)
                    if push_dir:
                        dx, dy = float(push_dir[0]), float(push_dir[1])
                        draw.line([(pt[0], pt[1]), (pt[0]+int(dx*40), pt[1]+int(dy*40))], fill="#00aaff", width=3)

            # State/action text annotations (bottom-left)
            if label.get("state_type"):
                ann = f"{label['state_type']}={label.get('state_value','')}"
                draw.text((5, img_h - 20), ann, fill="#ffffff")
            if label.get("can_do_now") is not None:
                status = "CAN DO" if label["can_do_now"] else f"BLOCKED: {label.get('failure_reason','')}"
                draw.text((5, img_h - 36), status, fill="#00ff00" if label["can_do_now"] else "#ff4444")

        # Encode
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    except Exception as e:
        logger.error(f"Render failed: {e}")
        return ""


@app.route("/api/export", methods=["POST"])
def api_export():
    """Trigger export of verified data."""
    verified = _load_all_from_dir(VERIFIED_DIR)
    if not verified:
        return jsonify({"error": "No verified data to export", "count": 0}), 404

    export_dir = DATA_ROOT / "training"
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_file = export_dir / f"verified_export_{ts}.jsonl"

    with open(export_file, "w", encoding="utf-8") as f:
        for item in verified:
            clean = {k: v for k, v in item.items() if k != "_verification"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    return jsonify({"status": "exported", "count": len(verified), "file": str(export_file)})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "viewer", "port": 10307})


@app.route("/data-sources")
def data_sources():
    """Serve the data sources analysis report (generated by build_data_sources_report.py)."""
    report_path = PROJECT_ROOT / "docs" / "2026-06-12-数据飞轮数据源分析.html"
    if report_path.exists():
        return send_file(str(report_path))
    # Fallback: generate on-the-fly if missing
    try:
        import subprocess
        subprocess.run(
            ["python3", str(PROJECT_ROOT / "scripts" / "build_data_sources_report.py")],
            check=True, timeout=60,
        )
        if report_path.exists():
            return send_file(str(report_path))
    except Exception as e:
        logger.warning(f"Failed to generate data sources report: {e}")
    return (
        "<h1>数据源分析报告未生成</h1>"
        "<p>请运行: <code>python scripts/build_data_sources_report.py</code></p>",
        404,
    )


@app.route("/flywheel")
def flywheel_page():
    """Serve the data flywheel system description page."""
    report_path = PROJECT_ROOT / "docs" / "data_flywheel_system.html"
    if report_path.exists():
        return send_file(str(report_path))
    try:
        import subprocess
        subprocess.run(
            ["python3", str(PROJECT_ROOT / "scripts" / "build_flywheel_report.py")],
            check=True, timeout=60,
        )
        if report_path.exists():
            return send_file(str(report_path))
    except Exception as e:
        logger.warning(f"Failed to generate flywheel report: {e}")
    return "<h1>数据飞轮说明未生成</h1><p>请运行: <code>python scripts/build_flywheel_report.py</code></p>", 404


@app.route("/training-eval")
def training_eval_page():
    """Serve the training and evaluation system description page."""
    report_path = PROJECT_ROOT / "docs" / "training_eval_system.html"
    if report_path.exists():
        return send_file(str(report_path))
    try:
        import subprocess
        subprocess.run(
            ["python3", str(PROJECT_ROOT / "scripts" / "build_training_report.py")],
            check=True, timeout=60,
        )
        if report_path.exists():
            return send_file(str(report_path))
    except Exception as e:
        logger.warning(f"Failed to generate training report: {e}")
    return "<h1>训练测评说明未生成</h1><p>请运行: <code>python scripts/build_training_report.py</code></p>", 404


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GSA Data Flywheel Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
@keyframes gradient-shift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes pulse-glow{0%,100%{box-shadow:0 0 20px rgba(99,102,241,0.1)}50%{box-shadow:0 0 40px rgba(99,102,241,0.2)}}
@keyframes float-in{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Space Grotesk','Noto Sans SC',sans-serif;background:linear-gradient(135deg,#e0e7ff 0%,#ede9fe 25%,#fce7f3 50%,#e0f2fe 75%,#ecfeff 100%);background-size:400% 400%;animation:gradient-shift 20s ease infinite;color:#1e293b;line-height:1.5;padding:1rem 1.2rem;min-height:100vh}
h1{font-size:1.8rem;font-weight:700;margin-bottom:.2rem;color:#fff;text-shadow:0 2px 10px rgba(0,0,0,0.3);letter-spacing:-0.5px}
h2{font-size:1.1rem;font-weight:600;margin:1.2rem 0 .6rem;color:#1e293b}
.subtitle{color:rgba(255,255,255,0.8);font-size:.82rem;margin-bottom:1rem}
.tabs{display:flex;gap:2px;background:rgba(255,255,255,0.2);border-radius:10px;padding:3px;margin-bottom:1rem;backdrop-filter:blur(8px)}
.tab{padding:.5rem 1.1rem;cursor:pointer;color:rgba(30,41,59,0.7);font-size:.82rem;font-weight:500;border-radius:8px;transition:all .2s}
.tab:hover{color:#1e293b;background:rgba(255,255,255,0.5)}
.tab.active{color:#fff;background:linear-gradient(135deg,#6366f1,#8b5cf6);box-shadow:0 2px 8px rgba(99,102,241,0.4)}
.tab-content{display:none;animation:float-in .3s ease}
.tab-content.active{display:block}
.grid{display:grid;gap:.8rem}
.grid-4{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
.grid-3{grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.grid-2{grid-template-columns:repeat(auto-fit,minmax(380px,1fr))}
.card{background:rgba(255,255,255,0.85);border:1px solid rgba(255,255,255,0.4);border-radius:14px;padding:1rem 1.1rem;transition:all .2s;backdrop-filter:blur(12px);box-shadow:0 2px 12px rgba(0,0,0,0.08)}
.card:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(99,102,241,0.2);border-color:rgba(99,102,241,0.3)}
.stat-value{font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;letter-spacing:-1px;color:#4f46e5}
.stat-label{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-top:.2rem}
.stat-sub{font-size:.65rem;color:#8b5cf6;margin-top:.15rem}
table{width:100%;border-collapse:collapse;margin-top:.4rem;font-size:.8rem;background:rgba(255,255,255,0.7);border-radius:10px;overflow:hidden}
th,td{padding:.5rem .7rem;text-align:left;border-bottom:1px solid rgba(99,102,241,0.1)}
th{color:#6366f1;font-weight:600;text-transform:uppercase;font-size:.68rem;letter-spacing:.04em;background:rgba(99,102,241,0.05)}
tr:hover td{background:rgba(99,102,241,0.04)}
.bar-bg{background:rgba(99,102,241,0.12);border-radius:6px;height:7px;width:100%;overflow:hidden}
.bar-fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#6366f1,#ec4899,#06b6d4);background-size:200% 100%;animation:shimmer 2.5s linear infinite}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:10px;font-size:.68rem;font-weight:600}
.badge-pass{background:rgba(34,197,94,0.15);color:#16a34a;border:1px solid rgba(34,197,94,0.3)}
.badge-fail{background:rgba(239,68,68,0.12);color:#dc2626;border:1px solid rgba(239,68,68,0.3)}
.badge-review{background:rgba(245,158,11,0.12);color:#d97706;border:1px solid rgba(245,158,11,0.3)}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:.3rem;box-shadow:0 0 6px currentColor}
.case-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:.8rem;margin-top:.6rem}
.case-card{background:rgba(255,255,255,0.8);border:1px solid rgba(99,102,241,0.15);border-radius:12px;overflow:hidden;position:relative;transition:all .2s;backdrop-filter:blur(8px);cursor:pointer}
.case-card:hover{transform:translateY(-2px);box-shadow:0 6px 24px rgba(99,102,241,0.15)}
.case-card img{width:100%;height:200px;object-fit:contain;background:#1a1a2e;border-bottom:1px solid rgba(99,102,241,0.1)}
.case-info{padding:.7rem;font-size:.78rem}
.case-info .meta{color:#64748b;margin-top:.2rem}
.dim-badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;color:#fff;margin-right:4px}
.status-badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;margin-left:4px}
.status-verified{background:#2ea043;color:#fff}
.status-review{background:#d29922;color:#fff}
.status-rejected{background:#f85149;color:#fff}
.case-badges{padding:6px 8px;display:flex;align-items:center;flex-wrap:wrap;gap:4px}
.json-block{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:.75rem;font-family:'JetBrains Mono',monospace;font-size:.75rem;overflow-x:auto;white-space:pre-wrap;color:#79c0ff;max-height:300px;overflow-y:auto}
.chart-bar{display:flex;align-items:center;gap:.5rem;margin:.3rem 0}
.chart-bar .label{width:140px;font-size:.78rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chart-bar .value{font-size:.72rem;color:#8b949e;min-width:40px;text-align:right}
.throughput-chart{display:flex;align-items:flex-end;gap:2px;height:80px;margin-top:.5rem}
.throughput-bar{background:#58a6ff;border-radius:2px 2px 0 0;min-width:8px;flex:1;transition:height .3s}
.export-btn{background:#238636;color:#fff;border:none;padding:.6rem 1.2rem;border-radius:6px;cursor:pointer;font-size:.85rem;font-weight:500}
.export-btn:hover{background:#2ea043}
.no-data{color:#8b949e;font-style:italic;padding:2rem;text-align:center}
</style>
</head>
<body>
<h1>GSA Data Flywheel Dashboard</h1>
<p class="subtitle">Real-time data production monitoring | vLLM dual-instance | Auto-refresh 30s</p>

{% if not has_data %}
<div class="card"><p class="no-data">No data yet. Start the flywheel pipeline to generate results.</p></div>
{% else %}

<!-- Tabs -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('overview')">Overview</div>
  <div class="tab" onclick="switchTab('cases')">Cases</div>
  <div class="tab" onclick="switchTab('statistics')">Statistics</div>
  <div class="tab" onclick="switchTab('schema')">Schema</div>
  <div class="tab" onclick="switchTab('export')">Export</div>
  <div class="tab" onclick="window.open('/data-sources','_blank')">数据源说明 ↗</div>
  <div class="tab" onclick="window.open('/flywheel','_blank')">数据飞轮说明 ↗</div>
  <div class="tab" onclick="window.open('/training-eval','_blank')">训练测评说明 ↗</div>
</div>

<!-- Overview Tab -->
<div class="tab-content active" id="tab-overview">
<div class="grid grid-4">
  <div class="card"><div class="stat-value">{{ summary.total_labels }}</div><div class="stat-label">Generated</div><div class="stat-sub">{{ summary.total_scanned }} scanned</div></div>
  <div class="card"><div class="stat-value">{{ summary.overall_pass_rate }}%</div><div class="stat-label">Pass Rate</div><div class="stat-sub">{{ summary.total_verified }}/{{ summary.total_scanned }} verified</div></div>
  <div class="card"><div class="stat-value">{{ summary.pass_at_100 }}%</div><div class="stat-label">Pass@100</div><div class="stat-sub">last {{ summary.pass_at_100_window }} samples</div></div>
  <div class="card"><div class="stat-value">{{ summary.get('pass_at_1000', 0) }}%</div><div class="stat-label">Pass@1000</div><div class="stat-sub">quality evolution</div></div>
  <div class="card"><div class="stat-value">{{ summary.avg_score_100 }}</div><div class="stat-label">Avg Score</div><div class="stat-sub">recent 100</div></div>
</div>
<div style="font-size:11px;color:#8b949e;padding:4px 0 12px;line-height:1.5">
  <b>Pass Rate</b> = Generator Verification 通过率（评判 Verifier 规则在当前阈值的严格程度）&nbsp;|&nbsp;
  <b>PASS@k</b> = 最近 k 个样本中通过校验的比例（滑动窗口，k 越大越稳定反映整体趋势）&nbsp;|&nbsp;
  <b>Verified@k</b> = 单个样本通过 k 次扫描校验仍保留（累积置信度，提供 PRM 奖励信号）
</div>

<h2>Quality Metrics</h2>
<div class="grid grid-4">
  <div class="card"><div class="stat-value">{{ summary.unique_objects }}</div><div class="stat-label">Unique Objects</div><div class="stat-sub">distinct object classes</div></div>
  <div class="card"><div class="stat-value">{{ "%.0f"|format(summary.dim_coverage * 100) }}%</div><div class="stat-label">Dim Coverage</div><div class="stat-sub">{{ "%.0f"|format(summary.dim_coverage * 6) }}/6 dims active</div></div>
  <div class="card"><div class="stat-value">{{ "%.1f"|format(summary.generator_self_check_rate * 100) }}%</div><div class="stat-label">Gen Self-Check</div><div class="stat-sub">labels / (labels+dead_letter)</div></div>
  <div class="card"><div class="stat-value">{{ summary.scanner_pass_rate }}%</div><div class="stat-label">Scanner Pass</div><div class="stat-sub">verified / scanned</div></div>
  <div class="card"><div class="stat-value">{{ summary.avg_bbox_area_pct }}%</div><div class="stat-label">Avg BBox Area</div><div class="stat-sub">bbox / image area</div></div>
</div>

<h2>Verified@k (样本级累积置信度)</h2>
<div class="grid grid-4">
  <div class="card"><div class="stat-value">{{ summary.get('verified_at_k', {}).get('k1', 0) }}</div><div class="stat-label">Verified@1</div><div class="stat-sub">通过1次扫描</div></div>
  <div class="card"><div class="stat-value">{{ summary.get('verified_at_k', {}).get('k5', 0) }}</div><div class="stat-label">Verified@5</div><div class="stat-sub">通过5次 · 可信数据</div></div>
  <div class="card"><div class="stat-value">{{ summary.get('verified_at_k', {}).get('k20', 0) }}</div><div class="stat-label">Verified@20</div><div class="stat-sub">通过20次 · 高置信训练数据</div></div>
  <div class="card"><div class="stat-value">{{ summary.get('action_diversity', 0) }}</div><div class="stat-label">Action Types</div><div class="stat-sub">distinct actions observed</div></div>
</div>

<h2>Dimension Breakdown</h2>
<div class="card">
<table>
<thead><tr><th>Pipeline</th><th>Generated</th><th>Verified</th><th>Review</th><th>Rejected</th><th>Pass Rate</th><th>Avg Conf</th><th>Verified@k</th></tr></thead>
<tbody>
{% for dim, stats in dim_stats.items() %}
<tr>
  <td><span class="status-dot" style="background:{{ stats.color }}"></span>{{ dim }}</td>
  <td>{{ stats.count }}</td>
  <td>{{ stats.verified }}</td>
  <td>{{ stats.review }}</td>
  <td>{{ stats.rejected }}</td>
  <td>{% if stats.scanned > 0 %}{{ stats.pass_rate }}%{% else %}-{% endif %}</td>
  <td>{{ stats.avg_confidence }}</td>
  <td><span style="font-size:.75rem;color:#79c0ff">{{ stats.verified }}</span></td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

<h2>Throughput (10-min intervals)</h2>
<div class="card">
{% if hourly_throughput %}
<div style="display:flex;gap:4px;align-items:flex-end">
  <div style="display:flex;flex-direction:column;justify-content:space-between;height:80px;font-size:9px;color:#8b949e;text-align:right;padding-right:4px">
    <span>max</span><span>50%</span><span>0</span>
  </div>
  <div class="throughput-chart" style="flex:1">
{% set max_count = hourly_throughput|map(attribute='count')|max if hourly_throughput else 1 %}
{% for h in hourly_throughput %}
<div class="throughput-bar" style="height:{{ (h.count / max_count * 100) if max_count > 0 else 5 }}%;min-height:3px" title="{{ h.hour }}: {{ h.count }} labels"></div>
{% endfor %}
  </div>
</div>
<div style="display:flex;justify-content:space-between;font-size:.7rem;color:#8b949e;margin-top:.3rem">
  <span>{{ hourly_throughput[0].hour if hourly_throughput else '' }}</span>
  <span>{{ hourly_throughput[-1].hour if hourly_throughput else '' }}</span>
</div>
{% else %}
<p class="no-data">No throughput data yet</p>
{% endif %}
</div>

<h2>Data Sources</h2>
<div class="card">
{% set palette = ["#7ee787","#79c0ff","#f0c674","#f08080","#d2a8ff","#ffa657","#a5d6ff","#ff7b72"] %}
{% set total_src = source_counts.values()|sum if source_counts else 1 %}
{% for src, cnt in source_counts.items() %}
<div class="chart-bar">
  <span class="label">{{ src }}</span>
  <div class="bar-bg" style="flex:1"><div class="bar-fill" style="width:{{ (cnt / total_src * 100) }}%;background:{{ palette[loop.index0 % palette|length] }}"></div></div>
  <span class="value">{{ cnt }} ({{ (cnt / total_src * 100)|round(1) }}%)</span>
</div>
{% endfor %}
</div>
</div>

<!-- Cases Tab -->
<div class="tab-content" id="tab-cases">
<h2>Recent Cases (annotated)</h2>
<div style="display:flex;gap:1rem;margin-bottom:1rem;align-items:center">
  <input type="text" id="caseSearch" placeholder="Search by filename, pipeline, source..." style="flex:1;padding:.5rem;background:#21262d;border:1px solid #30363d;border-radius:6px;color:#c9d1d9;font-size:.85rem" onkeyup="filterCases()">
  <select id="caseFilter" onchange="filterCases()" style="padding:.5rem;background:#21262d;border:1px solid #30363d;border-radius:6px;color:#c9d1d9">
    <option value="">All Pipelines</option>
    <option value="G_grounding">G_grounding</option>
    <option value="S_spatial">S_spatial</option>
    <option value="A_identity_attr">A_identity_attr</option>
    <option value="A_obj_state">A_obj_state</option>
    <option value="A_action_precondition">A_action_precondition</option>
    <option value="A_action_interface">A_action_interface</option>
  </select>
  <select id="sceneFilter" onchange="filterCases()" style="padding:.5rem;background:#21262d;border:1px solid #30363d;border-radius:6px;color:#c9d1d9">
    <option value="">All Scenes</option>
    <option value="tabletop_manipulation">tabletop_manipulation</option>
    <option value="indoor_robot">indoor_robot</option>
    <option value="kitchen_interaction">kitchen_interaction</option>
    <option value="indoor_general">indoor_general</option>
    <option value="object_state_change">object_state_change</option>
    <option value="object_closeup">object_closeup</option>
    <option value="household">household</option>
    <option value="living_room">living_room</option>
    <option value="mixed">mixed</option>
    <option value="unknown">unknown</option>
  </select>
  <select id="statusFilter" onchange="filterCases()" style="padding:.5rem;background:#21262d;border:1px solid #30363d;border-radius:6px;color:#c9d1d9">
    <option value="verified" selected>Verified</option>
    <option value="">All Status</option>
    <option value="review">Review</option>
    <option value="rejected">Rejected</option>
  </select>
  <button onclick="toggleView()" class="export-btn" style="padding:.4rem .8rem;font-size:.8rem" id="viewToggle">Grid View</button>
</div>
{% if sample_cases %}
<div class="case-grid" id="caseGrid">
{% for case in sample_cases %}
<div class="case-card" data-pipeline="{{ case.pipeline }}" data-filename="{{ case.filename }}" data-source="{{ case.source }}" data-status="{{ case.get('status', 'pending') }}" data-scene="{{ case.get('scene_type', '') }}" onclick="showDetail({{ loop.index0 }})">
  <div class="case-badges">
    <span class="dim-badge" style="background:{{ dimension_colors.get(case.pipeline, '#586069') }}">{{ case.pipeline }}</span>
    {% if case.get('scene_type') %}<span class="dim-badge" style="background:#21262d;border:1px solid #30363d;color:#8b949e;font-size:10px">{{ case.scene_type }}</span>{% endif %}
    {% if case.get('scan_count', 0) > 0 %}<span class="dim-badge" style="background:#0c2d48;color:#79c0ff;font-size:10px">@{{ case.scan_count }}</span>{% endif %}
    {% if case.get('status') == 'verified' %}<span class="status-badge status-verified">Verified</span>
    {% elif case.get('status') == 'review' %}<span class="status-badge status-review">Review</span>
    {% elif case.get('status') == 'rejected' %}<span class="status-badge status-rejected">Rejected</span>
    {% endif %}
    <span class="badge {% if case.confidence >= 0.8 %}badge-pass{% elif case.confidence >= 0.5 %}badge-review{% else %}badge-fail{% endif %}" style="margin-left:auto">{{ "%.2f"|format(case.confidence) }}</span>
  </div>
  {% if case.annotated_b64 %}
  <img src="{{ case.annotated_b64 }}" alt="{{ case.filename }}" loading="lazy">
  {% endif %}
  <div class="case-info">
    <div class="meta">{{ case.source }} · {{ case.get("label_count", 1) }} objects · {{ case.filename[:30] }}</div>
  </div>
</div>
{% endfor %}
</div>
<div style="text-align:center;margin:1rem 0">
  <button id="loadMoreBtn" onclick="loadMoreCases()" class="export-btn" style="padding:.5rem 1.5rem;font-size:.85rem">Load More</button>
  <span id="casesCountInfo" style="margin-left:1rem;color:#8b949e;font-size:.8rem"></span>
</div>
<!-- Detail Modal -->
<div id="detailModal" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.9);z-index:1000;overflow-y:auto;padding:2rem" onclick="if(event.target===this)closeDetail()">
  <div style="max-width:1200px;margin:0 auto;background:#161b22;border-radius:12px;overflow:hidden">
    <div style="display:flex;justify-content:space-between;align-items:center;padding:1rem 1.5rem;border-bottom:1px solid #30363d">
      <h3 id="detailTitle" style="color:#f0f6fc;margin:0">Case Detail</h3>
      <button onclick="closeDetail()" style="background:none;border:none;color:#8b949e;font-size:1.5rem;cursor:pointer">&times;</button>
    </div>
    <div style="padding:1.5rem">
      <img id="detailImg" src="" style="width:100%;max-height:600px;object-fit:contain;border-radius:8px;background:#0d1117">
      <div id="detailInfo" style="margin-top:1rem"></div>
      <div id="detailJson" class="json-block" style="margin-top:1rem;max-height:400px"></div>
    </div>
  </div>
</div>
{% else %}
<div class="card"><p class="no-data">No cases available yet. Data is being generated...</p></div>
{% endif %}
</div>

<!-- Statistics Tab -->
<div class="tab-content" id="tab-statistics">
<h2>Prior Knowledge (Ontology)</h2>
<div class="card" style="max-height:320px;overflow-y:auto;font-size:12px;line-height:1.6;margin-bottom:16px">
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
<div>
<h4 style="margin:0 0 6px;color:#58a6ff">Action Types</h4>
<div style="display:flex;flex-wrap:wrap;gap:4px">
{% for act in prior_knowledge.actions %}<span class="dim-badge" style="background:#21262d;border:1px solid #30363d;color:#c9d1d9;font-size:10px">{{ act }}</span>{% endfor %}
</div>
<h4 style="margin:10px 0 6px;color:#7ee787">State Types</h4>
<div style="display:flex;flex-wrap:wrap;gap:4px">
{% for st in prior_knowledge.states %}<span class="dim-badge" style="background:#1b4332;color:#7ee787;font-size:10px">{{ st }}</span>{% endfor %}
</div>
</div>
<div>
<h4 style="margin:0 0 6px;color:#f0c674">Attributes</h4>
<div style="display:flex;flex-wrap:wrap;gap:4px">
{% for attr in prior_knowledge.attributes %}<span class="dim-badge" style="background:#3d2e00;color:#f0c674;font-size:10px">{{ attr }}</span>{% endfor %}
</div>
<h4 style="margin:10px 0 6px;color:#d2a8ff">Spatial Predicates</h4>
<div style="display:flex;flex-wrap:wrap;gap:4px">
{% for pred in prior_knowledge.predicates %}<span class="dim-badge" style="background:#2d1b4e;color:#d2a8ff;font-size:10px">{{ pred }}</span>{% endfor %}
</div>
</div>
</div>
</div>

<h2>Scene Distribution</h2>
<div class="card" style="margin-bottom:16px">
{% if scene_counts %}
{% for scene, count in scene_counts.items() %}
<div style="display:flex;align-items:center;gap:8px;margin:4px 0;font-size:12px">
  <span style="width:160px;color:#8b949e">{{ scene }}</span>
  <div style="flex:1;height:8px;background:#21262d;border-radius:4px;overflow:hidden"><div style="height:100%;background:#58a6ff;width:{{ (count / (scene_counts.values()|list|max)) * 100 }}%;border-radius:4px"></div></div>
  <span style="min-width:40px;text-align:right;color:#c9d1d9">{{ count }}</span>
</div>
{% endfor %}
{% else %}
<p style="color:#8b949e;font-size:12px">No scene data yet</p>
{% endif %}
</div>

<h2>Verification Statistics</h2>
<div class="grid grid-3">
  <div class="card">
    <h2 style="margin-top:0">Pass/Fail Distribution</h2>
    {% set total_p = summary.total_verified + summary.total_rejected + summary.total_review %}
    {% if total_p > 0 %}
    <div class="chart-bar"><span class="label">Verified</span><div class="bar-bg" style="flex:1"><div class="bar-fill" style="width:{{ summary.total_verified / total_p * 100 }}%;background:#7ee787"></div></div><span class="value">{{ summary.total_verified }}</span></div>
    <div class="chart-bar"><span class="label">Needs Review</span><div class="bar-bg" style="flex:1"><div class="bar-fill" style="width:{{ summary.total_review / total_p * 100 }}%;background:#ffa657"></div></div><span class="value">{{ summary.total_review }}</span></div>
    <div class="chart-bar"><span class="label">Rejected</span><div class="bar-bg" style="flex:1"><div class="bar-fill" style="width:{{ summary.total_rejected / total_p * 100 }}%;background:#f08080"></div></div><span class="value">{{ summary.total_rejected }}</span></div>
    {% else %}
    <p class="no-data">No verification data</p>
    {% endif %}
  </div>
  <div class="card">
    <h2 style="margin-top:0">Per-Dimension Pass Rate</h2>
    {% for dim, stats in dim_stats.items() %}
    <div class="chart-bar"><span class="label">{{ dim }}</span><div class="bar-bg" style="flex:1"><div class="bar-fill" style="width:{{ stats.pass_rate }}%;background:{{ stats.color }}"></div></div><span class="value">{{ stats.pass_rate }}%</span></div>
    {% endfor %}
  </div>
  <div class="card">
    <h2 style="margin-top:0">Label Count by Dimension</h2>
    {% for dim, stats in dim_stats.items() %}
    <div class="chart-bar"><span class="label">{{ dim }}</span><div class="bar-bg" style="flex:1"><div class="bar-fill" style="width:{{ (stats.count / summary.total_labels * 100) if summary.total_labels > 0 else 0 }}%;background:{{ stats.color }}"></div></div><span class="value">{{ stats.count }}</span></div>
    {% endfor %}
  </div>
</div>
</div>

<!-- Schema Tab -->
<div class="tab-content" id="tab-schema">
<h2>Pipeline Schema Examples</h2>
<p style="color:#8b949e;font-size:.85rem;margin-bottom:1rem">Each pipeline produces labels in the following JSON format:</p>
<div class="grid grid-2">
{% for dim, example in schema_examples.items() %}
<div class="card">
  <h2 style="margin-top:0"><span class="status-dot" style="background:{{ dim_stats[dim].color if dim in dim_stats else '#888' }}"></span>{{ dim }}</h2>
  <div class="json-block">{{ example | tojson(indent=2) }}</div>
</div>
{% endfor %}
</div>
</div>

<!-- Export Tab -->
<div class="tab-content" id="tab-export">
<h2>Data Export for Training</h2>
<div class="card">
  <p style="margin-bottom:1rem">Export verified labels as JSONL for SFT training. Only labels with <code>verified_auto</code> tag are included.</p>
  <div class="grid grid-4" style="margin-bottom:1rem">
    <div><strong>{{ summary.total_verified }}</strong><br><span style="font-size:.8rem;color:#8b949e">Verified labels ready</span></div>
    <div><strong>{{ dim_stats|length }}</strong><br><span style="font-size:.8rem;color:#8b949e">Pipelines covered</span></div>
    <div><strong>{{ source_counts|length }}</strong><br><span style="font-size:.8rem;color:#8b949e">Data sources</span></div>
    <div><strong>{{ summary.overall_pass_rate }}%</strong><br><span style="font-size:.8rem;color:#8b949e">Overall pass rate</span></div>
  </div>
  <button class="export-btn" onclick="doExport()">Export Verified Data</button>
  <div id="export-result" style="margin-top:1rem;font-size:.85rem"></div>
</div>
</div>

{% endif %}

<script>
var caseData = {{ sample_cases | tojson }};
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}
var _casesOffset = 0;
var _casesLimit = 20;
var _casesTotal = 0;
function filterCases() {
  _casesOffset = 0;
  var grid = document.getElementById('caseGrid');
  if (grid) grid.innerHTML = '';
  _fetchCases(false);
}
function loadMoreCases() { _fetchCases(true); }
function _fetchCases(append) {
  const dimension = document.getElementById('caseFilter')?.value || '';
  const scene = document.getElementById('sceneFilter')?.value || '';
  const status = document.getElementById('statusFilter')?.value || '';
  const params = new URLSearchParams({dimension, scene, status, offset: _casesOffset, limit: _casesLimit});
  fetch('/api/cases?' + params.toString()).then(r => r.json()).then(data => {
    _casesTotal = data.total;
    var grid = document.getElementById('caseGrid');
    if (!append) grid.innerHTML = '';
    data.cases.forEach((c, i) => {
      var label = c.label || {};
      var idx = _casesOffset + i;
      var card = document.createElement('div');
      card.className = 'case-card';
      card.dataset.pipeline = label.pipeline || '';
      card.dataset.scene = label.scene_type || '';
      card.dataset.status = (c.verification && c.verification.tag) || 'pending';
      card.onclick = function(){ showCaseApi(label.sample_id); };
      var conf = label.confidence || 0;
      var confClass = conf >= 0.8 ? 'badge-pass' : conf >= 0.5 ? 'badge-review' : 'badge-fail';
      var imgHtml = '';
      if (c.has_image && label.image_path) {
        var imgPath = label.image_path.replace(/^\//, '');
        imgHtml = '<img src="/api/image/' + imgPath + '" loading="lazy" onerror="this.style.display=\'none\'">';
      }
      card.innerHTML = imgHtml
        + '<div class="case-badges"><span class="dim-badge" style="background:#586069">' + (label.pipeline||'') + '</span>'
        + (label.scene_type ? '<span class="dim-badge" style="background:#21262d;border:1px solid #30363d;color:#8b949e;font-size:10px">' + label.scene_type + '</span>' : '')
        + '<span class="badge ' + confClass + '" style="margin-left:auto">' + conf.toFixed(2) + '</span></div>'
        + '<div class="case-info"><div class="meta">' + (label.object_class||label.subject||'') + ' · ' + (label.source||'') + '</div></div>';
      grid.appendChild(card);
    });
    _casesOffset += data.cases.length;
    var btn = document.getElementById('loadMoreBtn');
    var info = document.getElementById('casesCountInfo');
    if (info) info.textContent = _casesOffset + ' / ' + _casesTotal;
    if (btn) btn.style.display = _casesOffset >= _casesTotal ? 'none' : '';
  });
}
function showCaseApi(sampleId) {
  if (!sampleId) return;
  fetch('/api/case/' + sampleId).then(r=>r.json()).then(data => {
    if (data.error) return;
    document.getElementById('detailTitle').textContent = (data.label.pipeline||'') + ' — ' + (data.label.filename||'');
    document.getElementById('detailImg').src = data.annotated_b64 || data.image_b64 || '';
    document.getElementById('detailInfo').innerHTML = (data.label.source||'') + ' | ' + (data.label.filename||'');
    document.getElementById('detailJson').textContent = JSON.stringify(data.label, null, 2);
    document.getElementById('detailModal').style.display = 'block';
  });
}
function toggleView() {
  var grid = document.getElementById('caseGrid');
  var btn = document.getElementById('viewToggle');
  if (grid.style.gridTemplateColumns === '1fr') {
    grid.style.gridTemplateColumns = '';
    btn.textContent = 'List View';
  } else {
    grid.style.gridTemplateColumns = '1fr';
    btn.textContent = 'Grid View';
  }
}
function showDetail(idx) {
  var c = caseData[idx];
  if (!c) return;
  document.getElementById('detailTitle').textContent = c.pipeline + ' — ' + c.filename;
  document.getElementById('detailImg').src = c.annotated_b64 || '';
  document.getElementById('detailInfo').innerHTML = '<span class="badge badge-pass">' + (c.confidence||0).toFixed(2) + '</span> ' + c.source + ' | ' + (c.label_count||1) + ' objects | ' + (c.timestamp||'').slice(0,16);
  document.getElementById('detailJson').textContent = JSON.stringify(c.all_labels || [c.label_data], null, 2);
  document.getElementById('detailModal').style.display = 'block';
}
function closeDetail() { document.getElementById('detailModal').style.display = 'none'; }
document.addEventListener('keydown', e => { if(e.key==='Escape') closeDetail(); });
function doExport() {
  fetch('/api/export', {method: 'POST'}).then(r => r.json()).then(data => {
    document.getElementById('export-result').innerHTML = data.error
      ? '<span style="color:#f08080">' + data.error + '</span>'
      : '<span style="color:#7ee787">Exported ' + data.count + ' labels to ' + data.file + '</span>';
  });
}
// Auto-refresh disabled for stability
// setInterval(() => { fetch('/api/data').then(r=>r.json()).then(d=>{if(d.has_data)location.reload()}).catch(()=>{}); }, 30000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Annotation Platform API
# ---------------------------------------------------------------------------

def _load_annotators() -> dict:
    if ANNOTATORS_FILE.exists():
        try:
            return json.loads(ANNOTATORS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_annotators(data: dict):
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATORS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _generate_api_key() -> str:
    return "sk-" + uuid.uuid4().hex[:32]


def _auth_annotator(req) -> tuple:
    """Authenticate via Bearer token, query param, cookie, or session username."""
    key = (req.headers.get("Authorization", "").replace("Bearer ", "").strip()
           or req.args.get("key", "").strip()
           or req.cookies.get("ann_key", "").strip())
    if key:
        annotators = _load_annotators()
        for username, info in annotators.items():
            if info.get("api_key") == key:
                return key, username
    # Fallback: allow unauthenticated access with guest identity
    return "guest", "guest"


@app.route("/history")
def history_index():
    """List available output archives."""
    # Scan both data/output/ and data/ for output_* directories
    project_data = DATA_ROOT.parent.parent  # data/
    output_dir = DATA_ROOT.parent  # data/output/
    archives = set()
    for d in project_data.iterdir():
        if d.is_dir() and d.name.startswith("output_"):
            archives.add((d.name, str(d)))
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("output_"):
            archives.add((d.name, str(d)))
    archives = sorted(archives, key=lambda x: x[0], reverse=True)
    links = "".join(f'<li><a href="/history/{a[0]}">{a[0]}</a> | <a href="/history/{a[0]}/annotate">annotate</a></li>' for a in archives)
    return f'<html><body style="background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:2rem"><h1>Output Archives</h1><ul style="font-size:1.1rem;line-height:2">{links}</ul><p><a href="/" style="color:#58a6ff">Back to Dashboard</a></p></body></html>'


@app.route("/history/<archive_name>")
def history_viewer(archive_name):
    """Viewer for historical output archive."""
    archive_path = DATA_ROOT.parent / archive_name
    if not archive_path.exists():
        archive_path = DATA_ROOT.parent.parent / archive_name
    if not archive_path.exists():
        return f"<h1>Archive '{archive_name}' not found</h1>", 404
    data = compute_dashboard_data(
        labels_dir=archive_path / "labels",
        verified_dir=archive_path / "verified",
        rejected_dir=archive_path / "rejected",
        review_dir=archive_path / "needs_review",
    )
    return render_template_string(HTML_TEMPLATE, **data)


@app.route("/history/<archive_name>/annotate")
def history_annotate(archive_name):
    """Annotate page for historical archive."""
    html_path = Path(__file__).parent / "annotation_ui.html"
    if not html_path.exists():
        return "<h1>annotation_ui.html not found</h1>", 404
    content = html_path.read_text(encoding="utf-8")
    # Inject archive_name so frontend sends it with API calls
    inject_script = f"""<script>window.GSA_ARCHIVE='{archive_name}';</script>"""
    content = content.replace("<script>", inject_script + "\n<script>", 1)
    return content


@app.route("/annotate")
def annotate_page():
    html_path = Path(__file__).parent / "annotation_ui.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>annotation_ui.html not found</h1>", 404


@app.route("/api/annotation/login", methods=["POST"])
def annotation_login():
    """Login with username+password. First login = auto register."""
    data = request.get_json(force=True) if request.is_json else {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "请输入用户名和密码"}), 400

    annotators = _load_annotators()

    if username in annotators:
        if annotators[username].get("password") != password:
            return jsonify({"error": "密码错误"}), 401
        api_key = annotators[username]["api_key"]
    else:
        api_key = _generate_api_key()
        annotators[username] = {
            "password": password,
            "api_key": api_key,
            "created": datetime.now().isoformat(),
            "label_count": 0,
        }
        _save_annotators(annotators)

    resp = jsonify({"annotator_name": username, "api_key": api_key, "message": "登录成功"})
    resp.set_cookie("ann_key", api_key, max_age=86400 * 30, httponly=False, samesite="Lax")
    return resp


@app.route("/api/annotation/cases", methods=["GET"])
def annotation_cases():
    key, name = _auth_annotator(request)
    if not key:
        return jsonify({"error": "unauthorized"}), 401

    pipeline_filter = request.args.get("pipeline", "")
    offset = int(request.args.get("offset", "0"))
    limit = int(request.args.get("limit", "20"))
    include_annotated = request.args.get("include_annotated", "false") == "true"
    source_filter = request.args.get("source", "")
    archive = request.args.get("archive", "")

    # Determine data directories (support historical archives)
    if archive:
        archive_path = DATA_ROOT.parent / archive
        if not archive_path.exists():
            archive_path = DATA_ROOT.parent.parent / archive
        # Handle nested structure (output_20260601/gsa_flywheel_exp/)
        if (archive_path / "gsa_flywheel_exp").exists():
            archive_path = archive_path / "gsa_flywheel_exp"
        _verified = archive_path / "verified"
        _review = archive_path / "needs_review"
        _labels = archive_path / "labels"
    else:
        _verified = VERIFIED_DIR
        _review = REVIEW_DIR
        _labels = LABELS_DIR

    annotated_ids = set()
    ann_file = (archive_path / "annotations" / "annotations.jsonl") if archive else ANNOTATION_FILE
    if ann_file.exists():
        for line in ann_file.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    annotated_ids.add(rec.get("case_id", ""))
                except Exception:
                    pass

    all_labels = []
    if source_filter:
        src_dirs = {
            "verified": [_verified],
            "needs_review": [_review],
            "labels": [_labels],
        }.get(source_filter, [_verified, _review, _labels])
    else:
        src_dirs = [_verified, _review, _labels]
    for src_dir in src_dirs:
        if not src_dir.exists():
            continue
        for fp in src_dir.rglob("*.jsonl"):
            try:
                for i, line in enumerate(fp.read_text().splitlines()):
                    if not line.strip():
                        continue
                    label = json.loads(line)
                    case_id = f"{fp.name}:{i}"
                    if pipeline_filter and label.get("pipeline", "") != pipeline_filter:
                        continue
                    if not include_annotated and case_id in annotated_ids:
                        continue
                    label["_case_id"] = case_id
                    label["_source_dir"] = src_dir.name
                    all_labels.append(label)
            except Exception:
                continue

    total = len(all_labels)
    page = all_labels[offset:offset+limit]

    cases = []
    for label in page:
        cases.append({
            "case_id": label["_case_id"],
            "pipeline": label.get("pipeline", "unknown"),
            "object_class": label.get("object_class", "") or label.get("object_name", ""),
            "bbox": label.get("bbox"),
            "confidence": label.get("confidence", 0),
            "source_dir": label["_source_dir"],
            "image_path": label.get("image_path", ""),
            "scan_count": label.get("_scan_count", 0),
            "label_data": {k: v for k, v in label.items() if not k.startswith("_")},
        })

    return jsonify({"items": cases, "total": total, "offset": offset})


@app.route("/api/annotation/case/<path:case_id>", methods=["GET"])
def annotation_case_detail(case_id):
    key, _ = _auth_annotator(request)
    if not key:
        return jsonify({"error": "unauthorized"}), 401

    archive = request.args.get("archive", "")
    filename, line_idx = case_id.rsplit(":", 1)
    line_idx = int(line_idx)

    # Determine search dirs based on archive
    if archive:
        archive_path = DATA_ROOT.parent / archive
        if not archive_path.exists():
            archive_path = DATA_ROOT.parent.parent / archive
        # Handle nested structure (e.g. output_20260601/gsa_flywheel_exp/)
        if (archive_path / "gsa_flywheel_exp").exists():
            archive_path = archive_path / "gsa_flywheel_exp"
        search_dirs = [archive_path / "verified", archive_path / "needs_review", archive_path / "labels"]
    else:
        search_dirs = [VERIFIED_DIR, REVIEW_DIR, LABELS_DIR]

    for src_dir in search_dirs:
        fp = None
        for candidate in src_dir.rglob(filename):
            fp = candidate
            break
        if not fp:
            continue
        lines = fp.read_text().splitlines()
        if line_idx < len(lines):
            label = json.loads(lines[line_idx])
            img_path = label.get("image_path", "")
            img_b64 = ""
            if img_path and Path(img_path).exists():
                try:
                    from PIL import Image as PILImage
                    img = PILImage.open(img_path).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
                except Exception:
                    pass
            return jsonify({
                "case_id": case_id,
                "label": {k: v for k, v in label.items() if k != "_verification"},
                "image_b64": img_b64,
                "verification": label.get("_verification", {}),
            })

    return jsonify({"error": "case not found"}), 404


@app.route("/api/annotation/submit", methods=["POST"])
def annotation_submit():
    key, name = _auth_annotator(request)
    if not key:
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(force=True)
    case_id = data.get("case_id", "")
    correct = data.get("correct")
    error_regions = data.get("error_regions", [])
    comment = data.get("comment", "")

    if not case_id or correct is None:
        return jsonify({"error": "case_id and correct are required"}), 400

    record = {
        "annotation_id": str(uuid.uuid4()),
        "case_id": case_id,
        "annotator_key": key,
        "annotator_name": name,
        "pipeline": data.get("pipeline", ""),
        "timestamp": datetime.now().isoformat(),
        "correct": bool(correct),
        "error_regions": error_regions,
        "comment": comment,
        "label_snapshot": data.get("label_data", {}),
    }

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATION_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Apply annotation: move label between verified/rejected
    _apply_annotation_to_data(case_id, correct, name, comment)

    return jsonify({"status": "ok", "annotation_id": record["annotation_id"]})


def _apply_annotation_to_data(case_id: str, correct: bool, annotator: str, comment: str):
    """Move label between dirs based on human annotation result."""
    try:
        filename, line_idx_str = case_id.rsplit(":", 1)
        line_idx = int(line_idx_str)
    except (ValueError, AttributeError):
        return

    source_file = None
    source_dir_name = None
    for src_dir in [VERIFIED_DIR, REVIEW_DIR, LABELS_DIR, REJECTED_DIR]:
        if not src_dir.exists():
            continue
        for candidate in src_dir.rglob(filename):
            source_file = candidate
            source_dir_name = src_dir.name
            break
        if source_file:
            break

    if not source_file or not source_file.exists():
        return

    try:
        lines = source_file.read_text(encoding="utf-8").splitlines()
        if line_idx >= len(lines) or not lines[line_idx].strip():
            return
        label = json.loads(lines[line_idx])
        label["_human_annotation"] = {
            "correct": correct,
            "annotator": annotator,
            "comment": comment,
            "annotated_at": datetime.now().isoformat(),
        }

        if not correct and source_dir_name in ("verified", "labels"):
            tdir = REJECTED_DIR / (label.get("pipeline") or "unknown")
            tdir.mkdir(parents=True, exist_ok=True)
            tf = tdir / f"human_rejected_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(tf, "a", encoding="utf-8") as f:
                f.write(json.dumps(label, ensure_ascii=False) + "\n")
            lines[line_idx] = ""
            remaining = [l for l in lines if l.strip()]
            source_file.write_text(("\n".join(remaining) + "\n") if remaining else "", encoding="utf-8")
            logger.info("Annotation: moved %s to rejected (by %s)", case_id, annotator)

        elif correct and source_dir_name in ("needs_review", "labels"):
            tdir = VERIFIED_DIR / (label.get("pipeline") or "unknown")
            tdir.mkdir(parents=True, exist_ok=True)
            tf = tdir / f"human_verified_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(tf, "a", encoding="utf-8") as f:
                f.write(json.dumps(label, ensure_ascii=False) + "\n")
            lines[line_idx] = ""
            remaining = [l for l in lines if l.strip()]
            source_file.write_text(("\n".join(remaining) + "\n") if remaining else "", encoding="utf-8")
            logger.info("Annotation: moved %s to verified (by %s)", case_id, annotator)

    except Exception as e:
        logger.warning("Failed to apply annotation for %s: %s", case_id, e)


@app.route("/api/annotation/stats", methods=["GET"])
def annotation_stats():
    key, _ = _auth_annotator(request)
    if not key:
        return jsonify({"error": "unauthorized"}), 401

    if not ANNOTATION_FILE.exists():
        return jsonify({"total": 0, "correct": 0, "wrong": 0, "by_pipeline": {}, "by_annotator": {}})

    records = []
    for line in ANNOTATION_FILE.read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    total = len(records)
    correct = sum(1 for r in records if r.get("correct"))
    wrong = total - correct
    by_pipeline = {}
    by_annotator = {}
    for r in records:
        pl = r.get("pipeline", "unknown")
        by_pipeline.setdefault(pl, {"total": 0, "correct": 0})
        by_pipeline[pl]["total"] += 1
        if r.get("correct"):
            by_pipeline[pl]["correct"] += 1
        ann = r.get("annotator_name", "unknown")
        by_annotator.setdefault(ann, {"total": 0, "correct": 0})
        by_annotator[ann]["total"] += 1
        if r.get("correct"):
            by_annotator[ann]["correct"] += 1

    return jsonify({"total": total, "correct": correct, "wrong": wrong, "by_pipeline": by_pipeline, "by_annotator": by_annotator})


@app.route("/api/annotation/export", methods=["GET"])
def annotation_export():
    key, _ = _auth_annotator(request)
    if not key:
        return jsonify({"error": "unauthorized"}), 401

    if not ANNOTATION_FILE.exists():
        return jsonify({"items": [], "count": 0})

    records = []
    for line in ANNOTATION_FILE.read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    fmt = request.args.get("format", "json")
    if fmt == "file":
        return send_file(str(ANNOTATION_FILE), as_attachment=True, download_name="annotations.jsonl")

    return jsonify({"items": records, "count": len(records)})


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting GSA Viewer Service on port %d", VIEWER_PORT)
    logger.info("Data root: %s", DATA_ROOT)
    app.run(host="0.0.0.0", port=VIEWER_PORT, debug=False)
