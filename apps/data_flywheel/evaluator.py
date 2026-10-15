"""Step7: 评估 — 标注质量自洽检查（用 Rex-Omni 或 API 重新检测，计算 IoU）。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from PIL import Image

from apps.data_flywheel.geometry import bbox_iou


def _detect_local(model, image_path, phrase):
    """本地模型检测。"""
    image = Image.open(image_path).convert("RGB")
    result = model.inference(images=image, task="detection", categories=[phrase])[0]
    preds = result.get("extracted_predictions", {})
    boxes = []
    for _, dets in preds.items():
        for det in dets:
            if det.get("type") == "box":
                boxes.append(det["coords"])
    return boxes


def _detect_api(image_path, phrase):
    """API 检测。"""
    from apps.data_flywheel.api_detector import detect_phrases_api
    result = detect_phrases_api(image_path, [phrase])
    return result.get(phrase, [])


def evaluate_grounding(
    grounding_jsonl: str,
    model_path: str = "IDEA-Research/Rex-Omni",
    backend: str = "transformers",
    iou_thresh: float = 0.5,
    eval_report_path: str = "data/output/flywheel/eval_report.json",
    max_samples: int = 100,
    model=None,
    use_api: bool = False,
) -> Dict[str, Any]:
    """自洽评估：重新检测标注结果，计算 IoU。"""
    os.makedirs(os.path.dirname(eval_report_path) or ".", exist_ok=True)

    data: List[Dict] = []
    with open(grounding_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    data = data[:max_samples]
    if not data:
        report = {"total": 0, "avg_iou": 0, "pass_rate": 0, "samples": []}
        with open(eval_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    if not use_api and model is None:
        from rex_omni import RexOmniWrapper
        model = RexOmniWrapper(
            model_path=model_path, backend=backend,
            max_tokens=2048, temperature=0.0, top_p=1.0, top_k=1,
            repetition_penalty=1.05,
        )

    sample_results: List[Dict] = []
    total_iou = 0.0
    total_boxes = 0
    pass_count = 0

    for item in data:
        image_path = item["image"]
        for phrase_data in item.get("phrases", []):
            phrase = phrase_data["phrase"]
            gt_boxes = [b["bbox"] for b in phrase_data.get("boxes", [])]

            try:
                if use_api:
                    pred_boxes = _detect_api(image_path, phrase)
                else:
                    pred_boxes = _detect_local(model, image_path, phrase)
            except Exception:
                continue

            for gt_box in gt_boxes:
                best_iou = 0.0
                for pred_box in pred_boxes:
                    iou = bbox_iou(gt_box, pred_box)
                    best_iou = max(best_iou, iou)

                total_iou += best_iou
                total_boxes += 1
                if best_iou >= iou_thresh:
                    pass_count += 1

                sample_results.append({
                    "image": image_path,
                    "phrase": phrase,
                    "gt_bbox": gt_box,
                    "best_iou": round(best_iou, 4),
                    "pass": best_iou >= iou_thresh,
                })

    avg_iou = round(total_iou / total_boxes, 4) if total_boxes > 0 else 0.0
    pass_rate = round(pass_count / total_boxes, 4) if total_boxes > 0 else 0.0

    report = {
        "total_boxes": total_boxes,
        "avg_iou": avg_iou,
        "pass_rate": pass_rate,
        "iou_thresh": iou_thresh,
        "samples": sample_results,
    }

    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report
