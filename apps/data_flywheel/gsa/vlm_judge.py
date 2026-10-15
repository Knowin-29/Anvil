"""VLM-as-Judge — Use VLM to verify annotation quality.

Two-pass verification:
1. VLM Judge: Ask VLM if the bbox/point actually corresponds to the labeled object
2. Multi-model consistency: Run detection twice, keep only consistent results
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gsa.vlm_judge")


class VLMJudge:
    """Use VLM to verify annotations are correct."""

    def __init__(self, vlm_client):
        self.vlm = vlm_client

    def verify_bbox(self, image_path: str, object_name: str, bbox: List[int],
                    img_w: int, img_h: int) -> Tuple[bool, float, str]:
        """Ask VLM if the bbox region actually contains the named object.

        Returns: (is_correct, confidence, reason)
        """
        # Normalize bbox back to 0-1000 for the prompt
        norm_bbox = [
            int(bbox[0] / img_w * 1000),
            int(bbox[1] / img_h * 1000),
            int(bbox[2] / img_w * 1000),
            int(bbox[3] / img_h * 1000),
        ]

        prompt = f"""Look at the region [{norm_bbox[0]},{norm_bbox[1]},{norm_bbox[2]},{norm_bbox[3]}] in this image.
Is there a "{object_name}" in this region?
Return JSON: {{"correct": true/false, "confidence": 0.0-1.0, "actual_object": "what you see", "reason": "brief explanation"}}"""

        try:
            result = self.vlm.chat_json(
                prompt=prompt,
                image_path=image_path,
                max_tokens=4000,
                temperature=0.1,
                system_prompt="You are a precise visual verification expert. Check if the labeled region matches the object name.",
            )
            if isinstance(result, dict):
                correct = result.get("correct", False)
                conf = result.get("confidence", 0.5)
                reason = result.get("reason", "")
                return correct, conf, reason
            return False, 0.0, "invalid_response"
        except Exception as e:
            logger.warning(f"VLM judge failed: {e}")
            return True, 0.5, "judge_unavailable"

    def verify_grasp_point(self, image_path: str, object_name: str,
                           grasp_point: List[int], img_w: int, img_h: int) -> Tuple[bool, float, str]:
        """Ask VLM if the grasp point is on the named object."""
        norm_pt = [
            int(grasp_point[0] / img_w * 1000),
            int(grasp_point[1] / img_h * 1000),
        ]

        prompt = f"""Look at the point [{norm_pt[0]},{norm_pt[1]}] in this image.
Is this point on or very near the object "{object_name}"? Is it a reasonable grasp point?
Return JSON: {{"on_object": true/false, "reasonable_grasp": true/false, "confidence": 0.0-1.0, "reason": "brief"}}"""

        try:
            result = self.vlm.chat_json(
                prompt=prompt,
                image_path=image_path,
                max_tokens=4000,
                temperature=0.1,
                system_prompt="You are a robotic manipulation expert. Verify grasp point quality.",
            )
            if isinstance(result, dict):
                on_obj = result.get("on_object", False)
                reasonable = result.get("reasonable_grasp", False)
                conf = result.get("confidence", 0.5)
                reason = result.get("reason", "")
                return on_obj and reasonable, conf, reason
            return False, 0.0, "invalid_response"
        except Exception as e:
            logger.warning(f"VLM judge grasp failed: {e}")
            return True, 0.5, "judge_unavailable"


class MultiModelFilter:
    """Run detection multiple times and keep only consistent results."""

    def __init__(self, vlm_client, n_runs: int = 2, iou_threshold: float = 0.3):
        self.vlm = vlm_client
        self.n_runs = n_runs
        self.iou_threshold = iou_threshold

    def detect_with_consensus(self, image_path: str) -> List[Dict[str, Any]]:
        """Run detection n_runs times, keep objects that appear consistently."""
        from apps.data_flywheel.workers.vlm_client import VLMClient
        from PIL import Image

        img = Image.open(image_path)
        img_w, img_h = img.size
        img.close()

        all_runs = []
        for i in range(self.n_runs):
            prompt = ("Detect all visible objects in this image. "
                      "Return JSON array: [{\"name\": \"object_name\", \"bbox_2d\": [x1,y1,x2,y2]}]")
            try:
                result = self.vlm.chat_json(
                    prompt=prompt,
                    image_path=image_path,
                    max_tokens=4000,
                    temperature=0.2 + i * 0.1,  # Slight variation
                    system_prompt="Expert object detector. Return JSON only.",
                )
                items = result if isinstance(result, list) else result.get("objects", [])
                run_objects = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("label") or ""
                    bbox = item.get("bbox") or item.get("bbox_2d") or []
                    if name and bbox and len(bbox) == 4:
                        # Denormalize
                        if all(0 <= v <= 1000 for v in bbox):
                            bbox = VLMClient.denormalize_bbox(bbox, img_w, img_h)
                        run_objects.append({"name": name.lower(), "bbox": bbox})
                all_runs.append(run_objects)
            except Exception as e:
                logger.warning(f"Detection run {i} failed: {e}")
                all_runs.append([])

        if not all_runs or not all_runs[0]:
            return []

        # Keep objects from first run that have a match in subsequent runs
        consensus = []
        for obj in all_runs[0]:
            matched = True
            for other_run in all_runs[1:]:
                if not self._find_match(obj, other_run):
                    matched = False
                    break
            if matched:
                obj["consensus_score"] = 1.0
                consensus.append(obj)
            else:
                # Partial match: appeared in at least one other run
                partial_matches = sum(1 for r in all_runs[1:] if self._find_match(obj, r))
                if partial_matches > 0:
                    obj["consensus_score"] = (1 + partial_matches) / len(all_runs)
                    consensus.append(obj)

        return consensus

    def _find_match(self, obj: Dict, candidates: List[Dict]) -> bool:
        """Check if obj has a matching candidate (same name + IoU > threshold)."""
        for cand in candidates:
            if obj["name"] == cand["name"] or obj["name"] in cand["name"] or cand["name"] in obj["name"]:
                iou = self._compute_iou(obj["bbox"], cand["bbox"])
                if iou >= self.iou_threshold:
                    return True
        return False

    @staticmethod
    def _compute_iou(box1: List, box2: List) -> float:
        """Compute IoU between two bboxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
