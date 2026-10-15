"""Step5: Pointing 标注 — Box → SAM Mask → 点标注（fallback: bbox 中心点）。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


def _bbox_center(bbox: List[float]) -> List[float]:
    """Fallback: bbox 中心点。"""
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _point_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    """论文方法: SAM mask → 最小外接旋转矩形 → 对角线交点。"""
    try:
        import cv2
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        cx, cy = rect[0]
        # 验证点在 mask 内
        if mask[int(cy), int(cx)] > 0:
            return [float(cx), float(cy)]
        # fallback: mask 质心
        return [float(xs.mean()), float(ys.mean())]
    except Exception:
        return None


def run_pointing(
    grounding_jsonl: str,
    output_jsonl: str = "data/output/flywheel/pointing.jsonl",
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_h",
) -> Dict[str, Any]:
    """从 grounding 标注生成 pointing 数据。"""
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

    # 尝试加载 SAM
    sam_predictor = None
    if sam_checkpoint:
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            sam.to(device)
            sam_predictor = SamPredictor(sam)
        except Exception:
            sam_predictor = None

    pointing_annotations: List[Dict] = []
    for item in grounding_data:
        image_path = item["image"]
        image_np = None
        if sam_predictor:
            try:
                image_np = np.array(Image.open(image_path).convert("RGB"))
                sam_predictor.set_image(image_np)
            except Exception:
                image_np = None

        for phrase_data in item.get("phrases", []):
            phrase = phrase_data["phrase"]
            for box_data in phrase_data.get("boxes", []):
                bbox = box_data["bbox"]
                point = None

                # 优先用 SAM
                if sam_predictor and image_np is not None:
                    try:
                        x1, y1, x2, y2 = bbox
                        input_box = np.array([[x1, y1, x2, y2]])
                        masks, scores, _ = sam_predictor.predict(
                            box=input_box, multimask_output=False,
                        )
                        if masks is not None and len(masks) > 0:
                            point = _point_from_mask(masks[0])
                    except Exception:
                        pass

                # Fallback: bbox 中心
                if point is None:
                    point = _bbox_center(bbox)

                pointing_annotations.append({
                    "image": image_path,
                    "phrase": phrase,
                    "bbox": bbox,
                    "point": point,
                    "method": "sam" if (sam_predictor and point != _bbox_center(bbox)) else "center",
                    "width": item.get("width"),
                    "height": item.get("height"),
                })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for ann in pointing_annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    return {"count": len(pointing_annotations), "output_jsonl": output_jsonl}
