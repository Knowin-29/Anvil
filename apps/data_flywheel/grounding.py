"""Step3: Grounding 标注 — spaCy 提取名词短语 → Phrase Filtering → 检测。"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Tuple

from PIL import Image

from apps.auto_label_engine.engine import (
    create_annotation,
    extract_noun_phrases,
    setup_spacy,
    visualize_grounding,
)
from apps.data_flywheel.phrase_filter import filter_phrases

logger = logging.getLogger("flywheel.grounding")


def _unique_phrases(phrases_with_spans: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    """去重，保留第一次出现的 span。"""
    seen = set()
    out = []
    for p, s, e in phrases_with_spans:
        if p not in seen:
            seen.add(p)
            out.append((p, s, e))
    return out


def _grounding_one_local(image_path, caption, nlp, model, filter_adjectives):
    """用本地 Rex-Omni 做单张图的 grounding。"""
    image = Image.open(image_path).convert("RGB")
    raw_phrases = extract_noun_phrases(caption, nlp)
    if filter_adjectives:
        raw_phrases = filter_phrases(raw_phrases, nlp)
    phrases_with_spans = _unique_phrases(raw_phrases)
    phrases = [p for p, _, _ in phrases_with_spans]
    if not phrases:
        return None

    # Rex-Omni 检测
    result = model.inference(images=image, task="detection", categories=phrases)[0]
    grounding_results = result.get("extracted_predictions", {})

    annotation = create_annotation(
        image_path=image_path,
        caption=caption,
        image_size=image.size,
        phrases_with_spans=phrases_with_spans,
        grounding_results=grounding_results,
    )
    return annotation if annotation.get("phrases") else None


def _grounding_one_api(image_path, caption, nlp, filter_adjectives):
    """用 VLM API 做单张图的 grounding。"""
    from apps.data_flywheel.api_detector import detect_phrases_api

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    raw_phrases = extract_noun_phrases(caption, nlp)
    if filter_adjectives:
        raw_phrases = filter_phrases(raw_phrases, nlp)
    phrases_with_spans = _unique_phrases(raw_phrases)
    phrases = [p for p, _, _ in phrases_with_spans]
    if not phrases:
        return None

    # API 检测
    phrase_boxes = detect_phrases_api(image_path, phrases)

    # 构建与 create_annotation 一致的格式
    annotation = {
        "image": image_path,
        "caption": caption,
        "width": img_w,
        "height": img_h,
        "phrases": [],
    }
    for p, s, e in phrases_with_spans:
        boxes = phrase_boxes.get(p, [])
        if boxes:
            annotation["phrases"].append({
                "phrase": p,
                "start_char": s,
                "end_char": e,
                "boxes": [{"bbox": b} for b in boxes],
            })
    return annotation if annotation["phrases"] else None


def _bbox_area(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = _bbox_area(a) + _bbox_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def _clip_bbox(bbox: List[float], width: float, height: float) -> List[float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _clean_annotation_boxes(
    annotation: Dict[str, Any],
    verifier_policy: str,
    min_area_ratio: float,
    max_area_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    nms_iou_thresh: float,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    width = float(annotation.get("width", 1) or 1)
    height = float(annotation.get("height", 1) or 1)
    img_area = max(1.0, width * height)
    stats = {
        "raw_boxes": 0,
        "kept_boxes": 0,
        "dropped_invalid": 0,
        "dropped_shape": 0,
        "dropped_nms": 0,
        "rescored_boxes": 0,
        "flagged_boxes": 0,
    }
    cleaned_phrases: List[Dict[str, Any]] = []

    for phrase in annotation.get("phrases", []):
        boxes = phrase.get("boxes", [])
        fixed: List[Dict[str, Any]] = []
        for box in boxes:
            stats["raw_boxes"] += 1
            raw_bbox = box.get("bbox")
            if not raw_bbox or len(raw_bbox) != 4:
                stats["dropped_invalid"] += 1
                continue

            bbox = _clip_bbox([float(v) for v in raw_bbox], width, height)
            area = _bbox_area(bbox)
            w = max(1e-6, bbox[2] - bbox[0])
            h = max(1e-6, bbox[3] - bbox[1])
            ar = w / h
            area_ratio = area / img_area
            abnormal = (
                area <= 1.0
                or area_ratio < min_area_ratio
                or area_ratio > max_area_ratio
                or ar < min_aspect_ratio
                or ar > max_aspect_ratio
            )

            cur = dict(box)
            cur["bbox"] = bbox
            if abnormal:
                if verifier_policy == "drop":
                    stats["dropped_shape"] += 1
                    continue
                if verifier_policy == "rescore":
                    old_conf = float(cur.get("confidence", 0.5) or 0.5)
                    cur["confidence"] = round(old_conf * 0.5, 4)
                    cur["abnormal"] = True
                    stats["rescored_boxes"] += 1
                elif verifier_policy == "flag":
                    cur["abnormal"] = True
                    stats["flagged_boxes"] += 1

            fixed.append(cur)

        # 同短语内 NMS
        fixed.sort(key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)
        kept: List[Dict[str, Any]] = []
        for cand in fixed:
            cand_bbox = cand.get("bbox", [])
            suppress = any(_bbox_iou(cand_bbox, k.get("bbox", [])) >= nms_iou_thresh for k in kept)
            if suppress:
                stats["dropped_nms"] += 1
                continue
            kept.append(cand)
            stats["kept_boxes"] += 1

        if kept:
            p = dict(phrase)
            p["boxes"] = kept
            cleaned_phrases.append(p)

    annotation["phrases"] = cleaned_phrases
    return annotation, stats


def run_grounding(
    image_caption_pairs: List[Tuple[str, str]],
    model_path: str = "IDEA-Research/Rex-Omni",
    backend: str = "transformers",
    output_jsonl: str = "data/output/flywheel/grounding.jsonl",
    vis_dir: str = "data/output/flywheel/visualizations",
    visualize: bool = True,
    filter_adjectives: bool = True,
    model=None,
    use_api: bool = False,
    enable_verifier: bool = True,
    verifier_policy: str = "drop",
    min_area_ratio: float = 0.0005,
    max_area_ratio: float = 0.9,
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 10.0,
    nms_iou_thresh: float = 0.7,
) -> Dict[str, Any]:
    """Grounding 标注。use_api=True 时用 VLM API 检测，无需本地模型。"""
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)

    nlp = setup_spacy()

    if not use_api and model is None:
        from rex_omni import RexOmniWrapper
        model = RexOmniWrapper(
            model_path=model_path, backend=backend,
            max_tokens=2048, temperature=0.0, top_p=1.0, top_k=1,
            repetition_penalty=1.05,
        )

    annotations: List[Dict] = []
    quality_stats = {
        "raw_boxes": 0,
        "kept_boxes": 0,
        "dropped_invalid": 0,
        "dropped_shape": 0,
        "dropped_nms": 0,
        "rescored_boxes": 0,
        "flagged_boxes": 0,
        "total_phrases": 0,
    }
    for idx, (image_path, caption) in enumerate(image_caption_pairs):
        try:
            if use_api:
                annotation = _grounding_one_api(image_path, caption, nlp, filter_adjectives)
            else:
                annotation = _grounding_one_local(image_path, caption, nlp, model, filter_adjectives)

            if annotation:
                if enable_verifier:
                    annotation, st = _clean_annotation_boxes(
                        annotation=annotation,
                        verifier_policy=verifier_policy,
                        min_area_ratio=min_area_ratio,
                        max_area_ratio=max_area_ratio,
                        min_aspect_ratio=min_aspect_ratio,
                        max_aspect_ratio=max_aspect_ratio,
                        nms_iou_thresh=nms_iou_thresh,
                    )
                    for k, v in st.items():
                        quality_stats[k] += int(v)
                    quality_stats["total_phrases"] += len(annotation.get("phrases", []))
                annotations.append(annotation)
                if visualize:
                    base = os.path.splitext(os.path.basename(image_path))[0]
                    vis_path = os.path.join(vis_dir, f"{base}_grounding.jpg")
                    image = Image.open(image_path).convert("RGB")
                    visualize_grounding(image, annotation, vis_path)
        except Exception as e:
            logger.warning(f"Grounding failed for {image_path}: {e}")
            continue

        if (idx + 1) % 5 == 0:
            logger.info(f"  Grounding {idx + 1}/{len(image_caption_pairs)} done")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in annotations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    avg_boxes_per_phrase = (
        round(quality_stats["kept_boxes"] / max(1, quality_stats["total_phrases"]), 4)
        if enable_verifier
        else 0.0
    )
    return {
        "count": len(annotations),
        "output_jsonl": output_jsonl,
        "quality_stats": {
            **quality_stats,
            "avg_boxes_per_phrase": avg_boxes_per_phrase,
            "verifier_policy": verifier_policy,
        },
    }
