"""DetectionVLM API adapter — wraps DetectionVLMClient as a drop-in
replacement for RexOmniWrapper so the flywheel pipeline can use a remote
vLLM service without any local model loading."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from PIL import Image

logger = logging.getLogger("flywheel.detvlm_adapter")


class DetVLMAdapter:
    """Adapts a remote DetectionVLM API to the same interface as
    ``RexOmniWrapper`` so callers like ``grounding.py``, ``evaluator.py``
    and ``referring.py`` can use it transparently via::

        result = model.inference(images=pil_image, task="detection", categories=[...])[0]
        preds = result["extracted_predictions"]
    """

    def __init__(self, base_url: str = "YOUR_DETVLM_URL", **_kwargs):
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _image_to_base64(image: Union[str, Image.Image]) -> str:
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def inference(
        self,
        images: Union[str, Image.Image],
        task: str = "detection",
        categories: Optional[List[str]] = None,
        **_kwargs,
    ) -> List[Dict[str, Any]]:
        """Mimic ``RexOmniWrapper.inference`` return value.

        Returns a **list** with one dict per image (currently always one).
        Each dict has ``"extracted_predictions"`` keyed by category name,
        matching the Rex-Omni format.
        """
        if task not in ("detection", "pointing"):
            raise ValueError(f"Unsupported task: {task}")

        payload = {
            "image": {"base64": self._image_to_base64(images)},
            "task": task,
            "categories": categories or [],
            "return_visualization": False,
            "temperature": 0.0,
            "max_tokens": 4096,
        }

        resp = requests.post(
            f"{self.base_url}/v1/detvlm",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        api_result = resp.json()

        predictions = api_result.get("predictions", [])

        if task == "detection":
            extracted = self._to_rex_detection(predictions, categories or [])
        else:
            extracted = self._to_rex_pointing(predictions, categories or [])

        return [{"extracted_predictions": extracted}]

    @staticmethod
    def _to_rex_detection(
        predictions: List[Dict], categories: List[str]
    ) -> Dict[str, List[Dict]]:
        result: Dict[str, List[Dict]] = {cat: [] for cat in categories}
        for pred in predictions:
            label = pred.get("label", "")
            bbox_raw = pred.get("bbox", {})
            if isinstance(bbox_raw, dict):
                coords = [bbox_raw["x1"], bbox_raw["y1"], bbox_raw["x2"], bbox_raw["y2"]]
            elif isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                coords = bbox_raw
            else:
                continue

            matched_cat = None
            label_lower = label.lower()
            for cat in categories:
                if cat.lower() == label_lower or cat.lower() in label_lower or label_lower in cat.lower():
                    matched_cat = cat
                    break
            if matched_cat is None:
                matched_cat = label

            result.setdefault(matched_cat, []).append({
                "type": "box",
                "coords": [float(c) for c in coords],
                "confidence": pred.get("confidence", pred.get("score", 1.0)),
            })
        return result

    @staticmethod
    def _to_rex_pointing(
        predictions: List[Dict], categories: List[str]
    ) -> Dict[str, List[Dict]]:
        result: Dict[str, List[Dict]] = {cat: [] for cat in categories}
        for pred in predictions:
            label = pred.get("label", "")
            point_raw = pred.get("point", {})
            if isinstance(point_raw, dict):
                coords = [point_raw["x"], point_raw["y"]]
            elif isinstance(point_raw, list) and len(point_raw) == 2:
                coords = point_raw
            else:
                continue

            matched_cat = None
            label_lower = label.lower()
            for cat in categories:
                if cat.lower() == label_lower or cat.lower() in label_lower or label_lower in cat.lower():
                    matched_cat = cat
                    break
            if matched_cat is None:
                matched_cat = label

            result.setdefault(matched_cat, []).append({
                "type": "point",
                "coords": [float(c) for c in coords],
            })
        return result

    def health(self) -> Dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
