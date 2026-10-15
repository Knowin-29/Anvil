#!/usr/bin/env python3
"""Typed answer parsing and routing for the v0616 VLabel pipeline.

The functions in this file are prediction-blind when used for GT routing:
they only inspect the prompt and ground truth answer.  All coordinates are
normalized to a 0-1000 frame whenever the source format is recognizable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


ANSWER_TYPES = {
    "bbox_region",
    "point_affordance",
    "choice",
    "attribute_state",
    "count",
    "relation",
    "action_free_text",
    "unknown",
}

OLD_IMAGE_ROOT = "${DATA_ROOT}/qwen-proxy-log-mirror"
NEW_IMAGE_ROOT = "${DATA_ROOT}/datasets/qwen-proxy-log-mirror-fast"


def remap_image_path(path: str) -> str:
    return str(path).replace(OLD_IMAGE_ROOT, NEW_IMAGE_ROOT)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(obj: Any) -> str:
    return hashlib.sha1(stable_json_dumps(obj).encode("utf-8")).hexdigest()


def strip_think_and_fences(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.I | re.S)
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def norm_text(text: Any) -> str:
    text = strip_think_and_fences(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'。。，,.;:：")


def _iter_json_candidates(text: str) -> Iterable[Any]:
    """Yield JSON objects/lists decoded from any balanced JSON-looking span."""
    cleaned = strip_think_and_fences(text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        j = i - 1
        while j >= 0 and cleaned[j].isspace():
            j -= 1
        if j >= 0 and cleaned[j] in "[:,[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[i:])
            yield obj
        except Exception:
            continue


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _clamp(v: float, lo: float = 0.0, hi: float = 1000.0) -> float:
    return max(lo, min(hi, float(v)))


def _normalize_coord(v: float) -> float:
    v = float(v)
    if 0.0 <= v <= 1.5:
        return _clamp(v * 1000.0)
    return float(v)


def normalize_box_xyxy(box: Iterable[Any]) -> Optional[List[float]]:
    vals = list(box)
    if len(vals) != 4 or not all(_is_num(v) for v in vals):
        return None
    x1, y1, x2, y2 = [_normalize_coord(float(v)) for v in vals]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_box_cxcywh(box: Iterable[Any]) -> Optional[List[float]]:
    vals = list(box)
    if len(vals) != 4 or not all(_is_num(v) for v in vals):
        return None
    cx, cy, w, h = [_normalize_coord(float(v)) for v in vals]
    if w <= 0 or h <= 0:
        return None
    return [_clamp(cx - w / 2), _clamp(cy - h / 2), _clamp(cx + w / 2), _clamp(cy + h / 2)]


def normalize_point(point: Iterable[Any]) -> Optional[List[float]]:
    vals = list(point)
    if len(vals) != 2 or not all(_is_num(v) for v in vals):
        return None
    return [_normalize_coord(float(vals[0])), _normalize_coord(float(vals[1]))]


def _append_unique_box(boxes: List[List[float]], box: Optional[List[float]]) -> None:
    if not box:
        return
    rounded = [round(x, 4) for x in box]
    if all([round(x, 4) for x in b] != rounded for b in boxes):
        boxes.append(box)


def _append_unique_point(points: List[List[float]], point: Optional[List[float]]) -> None:
    if not point:
        return
    rounded = [round(x, 4) for x in point]
    if all([round(x, 4) for x in p] != rounded for p in points):
        points.append(point)


def _walk_for_boxes(obj: Any, boxes: List[List[float]], points: List[List[float]]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lk = str(key).lower()
            if lk in {"bbox", "bbox_2d", "bbox_1000", "box"} or re.fullmatch(r"bbox_\d+", lk):
                if isinstance(value, list):
                    if len(value) == 4 and all(_is_num(v) for v in value):
                        _append_unique_box(boxes, normalize_box_xyxy(value))
                    elif len(value) == 2 and all(isinstance(p, list) and len(p) == 2 for p in value):
                        flat = [value[0][0], value[0][1], value[1][0], value[1][1]]
                        _append_unique_box(boxes, normalize_box_xyxy(flat))
            elif lk in {"bboxes", "boxes"} and isinstance(value, list):
                for item in value:
                    if isinstance(item, list) and len(item) == 4:
                        _append_unique_box(boxes, normalize_box_xyxy(item))
                    elif isinstance(item, list) and len(item) == 2 and all(isinstance(p, list) for p in item):
                        flat = [item[0][0], item[0][1], item[1][0], item[1][1]]
                        _append_unique_box(boxes, normalize_box_xyxy(flat))
            elif lk in {"point", "point_2d", "affordance_point", "goal_pixel"}:
                if isinstance(value, list):
                    _append_unique_point(points, normalize_point(value))
            _walk_for_boxes(value, boxes, points)
    elif isinstance(obj, list):
        if len(obj) == 4 and all(_is_num(v) for v in obj):
            # JSON lists without a key are ambiguous; keep them as xyxy only if they
            # look like an ordered region rather than arbitrary answer numbers.
            _append_unique_box(boxes, normalize_box_xyxy(obj))
        elif len(obj) == 2 and all(isinstance(p, list) and len(p) == 2 for p in obj):
            flat = [obj[0][0], obj[0][1], obj[1][0], obj[1][1]]
            _append_unique_box(boxes, normalize_box_xyxy(flat))
        elif len(obj) == 2 and all(_is_num(v) for v in obj):
            _append_unique_point(points, normalize_point(obj))
        else:
            for item in obj:
                _walk_for_boxes(item, boxes, points)


def parse_boxes_points(text: Any) -> Tuple[List[List[float]], List[List[float]], str]:
    text = "" if text is None else str(text)
    boxes: List[List[float]] = []
    points: List[List[float]] = []
    raw = strip_think_and_fences(text)

    for obj in _iter_json_candidates(raw):
        _walk_for_boxes(obj, boxes, points)

    # Strong key fallback for truncated JSON.  This intentionally only accepts
    # known coordinate keys, not arbitrary numeric tuples in free text.
    for m in re.finditer(r'"(?:bbox(?:_2d|_1000|_\d+)?|box)"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', raw, re.I):
        _append_unique_box(boxes, normalize_box_xyxy([float(m.group(i)) for i in range(1, 5)]))
    for m in re.finditer(r'"(?:bbox(?:_2d|_1000|_\d+)?|box)"\s*:\s*\[\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', raw, re.I):
        _append_unique_box(boxes, normalize_box_xyxy([float(m.group(i)) for i in range(1, 5)]))

    for m in re.finditer(r'"(?:point_2d|point|affordance_point|goal_pixel)"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', raw, re.I):
        _append_unique_point(points, normalize_point([float(m.group(1)), float(m.group(2))]))

    for m in re.finditer(r"<box>\s*\[\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*,\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\]\s*</box>", raw, re.I):
        _append_unique_box(boxes, normalize_box_xyxy([float(m.group(i)) for i in range(1, 5)]))

    for m in re.finditer(r"<point>\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*</point>", raw, re.I):
        _append_unique_point(points, normalize_point([float(m.group(1)), float(m.group(2))]))

    for m in re.finditer(r"\b(?:click|move_mouse|double_click)\s*\(\s*x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)\s*\)", raw, re.I):
        _append_unique_point(points, normalize_point([float(m.group(1)), float(m.group(2))]))

    cxcywh_hint = re.search(r"(center_x|cx|x,\s*y,\s*w,\s*h|width,\s*height)", raw, re.I)
    if cxcywh_hint:
        cxcywh_pat = r"[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]"
        for m in re.finditer(cxcywh_pat, raw):
            _append_unique_box(boxes, normalize_box_cxcywh([float(m.group(i)) for i in range(1, 5)]))

    frame = "none"
    all_vals = [v for box in boxes for v in box] + [v for p in points for v in p]
    if all_vals:
        frame = "norm_1000" if max(all_vals) <= 1000.0 else "pixel_or_unknown"
    return boxes, points, frame


CHOICE_TRUE = {"yes", "true", "是", "对", "正确", "有", "可以", "能"}
CHOICE_FALSE = {"no", "false", "否", "不", "错误", "没有", "不能", "不可以"}


def extract_choice(text: Any) -> Optional[str]:
    raw = norm_text(text)
    if not raw:
        return None
    for obj in _iter_json_candidates(raw):
        if isinstance(obj, dict):
            for key in ("answer", "choice", "label", "option", "result"):
                if key in obj:
                    nested = extract_choice(obj[key])
                    if nested:
                        return nested
    m = re.search(r"\b([abcd])\b", raw, re.I)
    if m:
        return m.group(1).upper()
    tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", raw))
    if tokens & CHOICE_TRUE:
        return "YES"
    if tokens & CHOICE_FALSE:
        return "NO"
    if len(raw) <= 40:
        return raw
    return None


def extract_count(text: Any) -> Optional[int]:
    raw = norm_text(text)
    for obj in _iter_json_candidates(raw):
        if isinstance(obj, dict):
            for key in ("count", "number", "num", "answer"):
                if key in obj:
                    c = extract_count(obj[key])
                    if c is not None:
                        return c
    m = re.search(r"-?\d+", raw)
    if m:
        return int(m.group(0))
    cn = {
        "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    for k, v in cn.items():
        if k in raw:
            return v
    return None


REL_ALIASES = {
    "left": {"left", "左", "左边", "左侧"},
    "right": {"right", "右", "右边", "右侧"},
    "above": {"above", "over", "top", "上", "上方", "上面"},
    "below": {"below", "under", "bottom", "下", "下方", "下面"},
    "inside": {"in", "inside", "within", "在内", "里面"},
    "on": {"on", "onto", "在上"},
    "front": {"front", "in front", "前", "前方"},
    "behind": {"behind", "back", "后", "后方"},
    "standing": {"standing", "upright", "竖直", "立着", "直立", "立姿"},
    "lying": {"lying", "horizontal", "倾倒", "躺", "平放", "水平"},
    "open": {"open", "opened", "打开", "开启"},
    "closed": {"closed", "close", "关闭", "闭合", "合上"},
}


def canonical_relation(text: Any) -> Optional[str]:
    raw = norm_text(text)
    if not raw:
        return None
    for obj in _iter_json_candidates(raw):
        if isinstance(obj, dict):
            for key in ("pose_family", "spatial_relation", "relation", "position", "orientation", "state", "answer"):
                if key in obj:
                    v = canonical_relation(obj[key])
                    if v:
                        return v
    for canon, aliases in REL_ALIASES.items():
        if raw == canon:
            return canon
        for alias in aliases:
            if re.fullmatch(r"[a-z ]+", alias):
                if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", raw):
                    return canon
            elif alias in raw:
                return canon
    return raw if len(raw) <= 48 else None


def extract_json_dict(text: Any) -> Optional[Dict[str, Any]]:
    for obj in _iter_json_candidates(str(text)):
        if isinstance(obj, dict):
            return obj
    return None


def classify_answer_type(prompt: Any, gt_answer: Any, task_family: Any = None) -> Tuple[str, str]:
    prompt_n = norm_text(prompt)
    gt_n = norm_text(gt_answer)
    tf = norm_text(task_family)
    joined = f"{prompt_n} {gt_n} {tf}"

    boxes, points, _ = parse_boxes_points(gt_answer)
    prompt_mentions_bbox = bool(re.search(r"\b(bbox|bounding box|box|框|坐标框|定位)\b", joined))
    prompt_mentions_point = bool(re.search(r"\b(point|affordance|click|点击|抓取点|目标点|goal_pixel)\b", joined))

    if "bbox_grounding" in tf or (boxes and prompt_mentions_bbox and "point_or_affordance" not in tf):
        return "bbox_region", "task_family_or_schema_bbox"
    if "point_or_affordance" in tf or points or ("affordance" in joined):
        return "point_affordance", "task_family_or_schema_point"
    if "closed_choice" in tf or re.search(r"\b(a\.|b\.|c\.|d\.|yes/no|true/false|是否|判断)\b", joined):
        return "choice", "closed_choice_or_binary_prompt"
    if "count" in joined or "how many" in joined or "多少" in joined or "数量" in joined:
        return "count", "count_prompt"
    if "spatial_relation" in tf or any(k in joined for k in ("pose_family", "relation", "left", "right", "above", "below", "左右", "上下", "空间")):
        return "relation", "spatial_schema_or_prompt"
    if "attribute_classification" in tf or "state_classification" in tf:
        return "attribute_state", "attribute_or_state_family"
    if "robot_planning" in tf or any(k in joined for k in ("plan", "planning", "action", "步骤", "规划", "推动", "抓取")):
        return "action_free_text", "action_or_planning_family"
    if extract_json_dict(gt_answer):
        return "attribute_state", "json_gt_default"
    return "unknown", "no_supported_answer_type"


def build_typed_ir(prompt: Any, gt_answer: Any, task_family: Any = None) -> Dict[str, Any]:
    answer_type, route_reason = classify_answer_type(prompt, gt_answer, task_family)
    boxes, points, coord_frame = parse_boxes_points(gt_answer)
    ir: Dict[str, Any] = {
        "answer_type": answer_type,
        "route_reason": route_reason,
        "boxes": boxes,
        "points": points,
        "choice": extract_choice(gt_answer),
        "count": extract_count(gt_answer),
        "relation": canonical_relation(gt_answer),
        "json": extract_json_dict(gt_answer),
        "text": strip_think_and_fences(gt_answer),
    }
    components: List[str] = []
    if boxes:
        components.append("bbox")
    if points:
        components.append("point")
    if ir["choice"]:
        components.append("choice")
    if ir["count"] is not None:
        components.append("count")
    if ir["relation"]:
        components.append("relation")
    if ir["json"]:
        components.append("json")
    ir["components"] = components
    ir["coord_frame"] = coord_frame
    return ir


def manifest_entry(
    *,
    sample_id: str,
    source: str,
    task_family_raw: str,
    prompt: str,
    gt_answer: str,
    capabilities: Optional[List[str]] = None,
    original_index: Optional[int] = None,
) -> Dict[str, Any]:
    gt_ir = build_typed_ir(prompt, gt_answer, task_family_raw)
    answer_type = gt_ir["answer_type"]
    evaluable = True
    not_reason = None
    if not strip_think_and_fences(gt_answer) or strip_think_and_fences(gt_answer) in {"<FAILED>", "None"}:
        evaluable = False
        not_reason = "gt_empty_or_failed"
    elif answer_type == "bbox_region" and not gt_ir["boxes"]:
        evaluable = False
        not_reason = "gt_no_bbox"
    elif answer_type == "point_affordance" and not (gt_ir["points"] or gt_ir["boxes"]):
        evaluable = False
        not_reason = "gt_no_point_or_region"
    elif answer_type == "choice" and not gt_ir["choice"]:
        evaluable = False
        not_reason = "gt_choice_unparseable"
    elif answer_type == "count" and gt_ir["count"] is None:
        evaluable = False
        not_reason = "gt_count_unparseable"
    elif answer_type == "relation" and not gt_ir["relation"]:
        evaluable = False
        not_reason = "gt_relation_unparseable"
    elif answer_type == "unknown":
        evaluable = False
        not_reason = "unknown_answer_type"

    entry = {
        "sample_id": sample_id,
        "source": source,
        "task_family_raw": task_family_raw,
        "answer_type": answer_type,
        "components": gt_ir["components"],
        "gt_ir": gt_ir,
        "coord_frame": gt_ir["coord_frame"],
        "evaluable": evaluable,
        "route_reason": gt_ir["route_reason"],
        "not_evaluable_reason": not_reason,
        "capabilities": capabilities or [],
        "original_index": original_index,
    }
    entry["manifest_hash"] = stable_hash({
        "sample_id": entry["sample_id"],
        "answer_type": entry["answer_type"],
        "gt_ir": entry["gt_ir"],
        "evaluable": entry["evaluable"],
        "not_evaluable_reason": entry["not_evaluable_reason"],
    })
    return entry


def iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def point_in_box(point: List[float], box: List[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def point_distance_ok(pred: List[float], gt: List[float], alpha: float = 0.05) -> bool:
    threshold = alpha * math.sqrt(1000.0 * 1000.0 + 1000.0 * 1000.0)
    return math.dist(pred, gt) <= threshold


def greedy_match_f1(pred_boxes: List[List[float]], gt_boxes: List[List[float]], threshold: float = 0.5) -> Tuple[float, int, int, int, float]:
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            pairs.append((iou(pb, gb), pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    tp = 0
    best = pairs[0][0] if pairs else 0.0
    for score, pi, gi in pairs:
        if score < threshold:
            break
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tp += 1
    fp = max(0, len(pred_boxes) - tp)
    fn = max(0, len(gt_boxes) - tp)
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 0.0
    return f1, tp, fp, fn, best
