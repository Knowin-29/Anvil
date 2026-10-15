"""Step4: Referring 标注 — 生成丰富的 referring expression 并用 Rex-Omni 验证。

复用 geometry.py 的几何计算和类别知识库。
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from apps.data_flywheel.geometry import (
    bbox_area,
    bbox_center,
    bbox_contains,
    bbox_distance,
    bbox_iou,
    image_diagonal,
)

logger = logging.getLogger("flywheel.referring")


# ---- Expression 生成 ----

def _absolute_position_exprs(category: str, bbox: List[float], img_size: Tuple[int, int]) -> List[str]:
    w, h = img_size
    cx, cy = bbox_center(bbox)

    h_pos = "on the left" if cx < w / 3 else ("on the right" if cx > 2 * w / 3 else "in the center")
    v_pos = "at the top" if cy < h / 3 else ("at the bottom" if cy > 2 * h / 3 else "in the middle")

    area_ratio = bbox_area(bbox) / (w * h) if w * h > 0 else 0
    size_desc = "large" if area_ratio > 0.15 else ("small" if area_ratio < 0.02 else "")

    exprs = [
        f"the {category} {h_pos} of the image",
        f"the {category} {v_pos} {h_pos}",
    ]
    if size_desc:
        exprs.append(f"the {size_desc} {category}")
        exprs.append(f"the {size_desc} {category} {h_pos}")
    return exprs


def _relative_position_exprs(
    category: str, bbox: List[float],
    other_category: str, other_bbox: List[float],
    img_size: Tuple[int, int],
) -> List[str]:
    rcx, rcy = bbox_center(other_bbox)
    tcx, tcy = bbox_center(bbox)
    dx, dy = tcx - rcx, tcy - rcy

    ref_w = other_bbox[2] - other_bbox[0]
    ref_h = other_bbox[3] - other_bbox[1]
    min_dist = max(ref_w, ref_h) * 0.3

    if abs(dx) < min_dist and abs(dy) < min_dist:
        return []

    if abs(dx) > abs(dy):
        rel = "to the right of" if dx > 0 else "to the left of"
    else:
        rel = "below" if dy > 0 else "above"

    exprs = [f"the {category} {rel} the {other_category}"]

    img_diag = image_diagonal(*img_size)
    if img_diag > 0 and bbox_distance(bbox, other_bbox) < img_diag * 0.15:
        exprs.append(f"the {category} near the {other_category}")

    return exprs


def _containment_exprs(
    category: str, bbox: List[float],
    container_category: str, container_bbox: List[float],
) -> List[str]:
    if not bbox_contains(container_bbox, bbox):
        return []

    exprs = [
        f"the {category} on the {container_category}",
        f"the {category} in the {container_category}",
    ]

    cw = container_bbox[2] - container_bbox[0]
    ch = container_bbox[3] - container_bbox[1]
    if cw > 0 and ch > 0:
        tcx, tcy = bbox_center(bbox)
        rx = (tcx - container_bbox[0]) / cw
        ry = (tcy - container_bbox[1]) / ch

        h_in = "left side" if rx < 0.33 else ("right side" if rx > 0.67 else "center")
        v_in = "top" if ry < 0.33 else ("bottom" if ry > 0.67 else "middle")

        if h_in != "center" or v_in != "middle":
            exprs.append(f"the {category} on the {v_in} {h_in} of the {container_category}")

    return exprs


def generate_referring_expressions(
    category: str,
    bbox: List[float],
    img_size: Tuple[int, int],
    all_objects: List[Tuple[str, List[float]]],
) -> List[str]:
    """为一个物体生成多样化的 referring expression。"""
    exprs = []

    exprs.extend(_absolute_position_exprs(category, bbox, img_size))

    for other_cat, other_bbox in all_objects:
        if other_cat == category and bbox_iou(bbox, other_bbox) > 0.5:
            continue
        exprs.extend(_relative_position_exprs(category, bbox, other_cat, other_bbox, img_size))

    for other_cat, other_bbox in all_objects:
        if other_cat == category and bbox_iou(bbox, other_bbox) > 0.5:
            continue
        containment = _containment_exprs(category, bbox, other_cat, other_bbox)
        if containment:
            exprs.extend(containment)
        contained = _containment_exprs(other_cat, other_bbox, category, bbox)
        if contained:
            exprs.append(f"the {category} with the {other_cat}")

    seen = set()
    return [e for e in exprs if not (e in seen or seen.add(e))]


# ---- 验证 ----

def _verify_local(model, image_path: str, expr: str) -> List[List[float]]:
    image = Image.open(image_path).convert("RGB")
    result = model.inference(images=image, task="detection", categories=[expr])[0]
    preds = result.get("extracted_predictions", {})
    matched = []
    for _, dets in preds.items():
        for det in dets:
            if det.get("type") == "box":
                matched.append(det["coords"])
    return matched


def _verify_api(image_path: str, expr: str) -> List[List[float]]:
    from apps.data_flywheel.api_detector import detect_phrases_api
    result = detect_phrases_api(image_path, [expr])
    return result.get(expr, [])


# ---- 主入口 ----

def run_referring(
    grounding_jsonl: str,
    model_path: str = "IDEA-Research/Rex-Omni",
    backend: str = "transformers",
    output_jsonl: str = "data/output/flywheel/referring.jsonl",
    model=None,
    use_api: bool = False,
    max_exprs_per_object: int = 3,
) -> Dict[str, Any]:
    """从 grounding 标注生成 referring 数据。"""
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    grounding_data: List[Dict] = []
    with open(grounding_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                grounding_data.append(json.loads(line))

    if not grounding_data:
        with open(output_jsonl, "w") as f:
            pass
        return {"count": 0, "output_jsonl": output_jsonl}

    if not use_api and model is None:
        from rex_omni import RexOmniWrapper
        model = RexOmniWrapper(
            model_path=model_path, backend=backend,
            max_tokens=2048, temperature=0.0, top_p=1.0, top_k=1,
            repetition_penalty=1.05,
        )

    referring_annotations: List[Dict] = []

    for item in grounding_data:
        image_path = item["image"]
        img_w = item.get("width", 0)
        img_h = item.get("height", 0)
        if not img_w or not img_h:
            try:
                with Image.open(image_path) as img:
                    img_w, img_h = img.size
            except Exception:
                continue

        all_objects: List[Tuple[str, List[float]]] = []
        for phrase_data in item.get("phrases", []):
            for box_data in phrase_data.get("boxes", []):
                all_objects.append((phrase_data["phrase"], box_data["bbox"]))

        for phrase_data in item.get("phrases", []):
            phrase = phrase_data["phrase"]
            for box_data in phrase_data.get("boxes", []):
                bbox = box_data["bbox"]
                candidates = generate_referring_expressions(
                    category=phrase, bbox=bbox,
                    img_size=(img_w, img_h), all_objects=all_objects,
                )
                if not candidates:
                    continue

                selected = random.sample(candidates, min(max_exprs_per_object, len(candidates)))
                for expr in selected:
                    try:
                        if use_api:
                            matched_boxes = _verify_api(image_path, expr)
                        else:
                            matched_boxes = _verify_local(model, image_path, expr)

                        referring_annotations.append({
                            "image": image_path,
                            "expression": expr,
                            "gt_bbox": bbox,
                            "matched_boxes": matched_boxes,
                            "category": phrase,
                            "width": img_w,
                            "height": img_h,
                            "verified": len(matched_boxes) > 0,
                        })
                    except Exception as e:
                        logger.debug(f"Verify failed: {expr} — {e}")
                        continue

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in referring_annotations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    verified_count = sum(1 for a in referring_annotations if a.get("verified"))
    logger.info(f"Referring: {len(referring_annotations)} 条, 验证通过 {verified_count} 条")

    return {
        "count": len(referring_annotations),
        "verified": verified_count,
        "output_jsonl": output_jsonl,
    }
