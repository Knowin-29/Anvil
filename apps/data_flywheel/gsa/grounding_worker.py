"""GSA Grounding Worker — Aligned with GSA taxonomy subtypes.

Produces G labels with proper subtypes:
- simple_grounding: single object by name
- descriptive_grounding: object by attributes
- relational_grounding: object by spatial relation to another
- counting: how many objects of a type
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from ..workers.vlm_client import VLMClient
from .taxonomy import Bucket, GSubtype, SourceProvenance, VerifierStatus
from .visual_evidence import GsaObjectRef, assign_object_ids

logger = logging.getLogger("gsa.grounding_worker")


class GSAGroundingWorker:
    """Generates G-bucket labels aligned with GSA taxonomy."""

    def __init__(self, vlm_client: VLMClient, use_consensus: bool = True):
        self.vlm = vlm_client
        self.use_consensus = use_consensus
        self.simple_only = False

    def generate(self, image_path: str, sample_id: str,
                 scene_family: str = "", scenario_id: str = "") -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        if self.use_consensus:
            from .vlm_judge import MultiModelFilter
            mf = MultiModelFilter(self.vlm, n_runs=2, iou_threshold=0.3)
            objects = mf.detect_with_consensus(image_path)
        else:
            objects = self._detect_objects(image_path)
        if not objects:
            return labels
        obj_refs = assign_object_ids(objects)
        labels.extend(self._simple_grounding(sample_id, obj_refs, scene_family, scenario_id))
        if self.simple_only:
            labels.extend(self._counting(sample_id, obj_refs, scene_family, scenario_id))
            return labels
        labels.extend(self._descriptive_grounding(image_path, sample_id, obj_refs, scene_family, scenario_id))
        labels.extend(self._relational_grounding(image_path, sample_id, obj_refs, scene_family, scenario_id))
        labels.extend(self._counting(sample_id, obj_refs, scene_family, scenario_id))
        return labels

    def _detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        prompt = ("Detect all visible objects in this indoor scene. "
                  "Return JSON array: [{\"name\": \"object_name\", \"bbox_2d\": [x1,y1,x2,y2]}]")
        try:
            # Get image dimensions for coordinate denormalization
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            img_w, img_h = img.size
            img.close()

            result = self.vlm.chat_json(prompt=prompt, image_path=image_path,
                max_tokens=4000, temperature=0.3,
                system_prompt="Expert object detector. Return JSON only.")
            items = result if isinstance(result, list) else result.get("objects", [])
            # Normalize different VLM output formats + denormalize coordinates
            from apps.data_flywheel.workers.vlm_client import VLMClient
            normalized = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("label") or item.get("object_name") or "unknown"
                bbox = item.get("bbox") or item.get("bbox_2d") or item.get("bounding_box")
                if bbox and len(bbox) == 4:
                    # Qwen VL returns coords in 0-1000 range, denormalize to pixels
                    if all(0 <= v <= 1000 for v in bbox):
                        bbox = VLMClient.denormalize_bbox(bbox, img_w, img_h)
                    normalized.append({"name": name, "bbox": bbox, "confidence": item.get("confidence", 0.8)})
            return normalized
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _simple_grounding(self, sid, objs, sf, sc):
        return [{
            "sample_id": sid, "bucket": Bucket.G.value,
            "subtype": GSubtype.SIMPLE_GROUNDING.value,
            "scene_family": sf, "scenario_id": sc,
            "object_id": o.object_id,
            "question": f"Find the {o.raw_object_name}.",
            "answer": {"bbox": o.bbox, "object_name": o.raw_object_name},
            "verifier_status": VerifierStatus.AUTO_VERIFIABLE.value,
            "source_provenance": SourceProvenance.VLM.value,
            "confidence": o.confidence,
        } for o in objs]

    def _descriptive_grounding(self, img, sid, objs, sf, sc):
        if len(objs) < 2:
            return []
        names = [o.raw_object_name for o in objs[:8]]
        prompt = (f"Objects: {json.dumps(names)}. Generate descriptive grounding queries "
                  "using attributes. JSON: [{\"query\":\"the red cup\",\"target_object\":\"cup\"}]")
        try:
            r = self.vlm.chat_json(prompt=prompt, image_path=img, max_tokens=512, temperature=0.5,
                system_prompt="Generate grounding data. JSON only.")
            queries = r if isinstance(r, list) else r.get("queries", [])
        except Exception:
            return []
        labels = []
        for q in queries:
            if not isinstance(q, dict):
                continue
            t = q.get("target_object", "")
            m = next((o for o in objs if t.lower() in o.raw_object_name.lower()), None)
            if m:
                labels.append({
                    "sample_id": sid, "bucket": Bucket.G.value,
                    "subtype": GSubtype.DESCRIPTIVE_GROUNDING.value,
                    "scene_family": sf, "scenario_id": sc,
                    "object_id": m.object_id,
                    "question": f"Find {q.get('query', t)}.",
                    "answer": {"bbox": m.bbox, "object_name": m.raw_object_name},
                    "verifier_status": VerifierStatus.AUTO_VERIFIABLE.value,
                    "source_provenance": SourceProvenance.VLM.value,
                })
        return labels

    def _relational_grounding(self, img, sid, objs, sf, sc):
        if len(objs) < 2:
            return []
        info = [{"name": o.raw_object_name, "bbox": o.bbox} for o in objs[:6]]
        prompt = (f"Objects: {json.dumps(info)}. Generate relational grounding queries. "
                  "JSON: [{\"query\":\"cup left of plate\",\"target\":\"cup\",\"reference\":\"plate\",\"relation\":\"left_of\"}]")
        try:
            r = self.vlm.chat_json(prompt=prompt, image_path=img, max_tokens=512, temperature=0.5,
                system_prompt="Generate spatial grounding data. JSON only.")
            queries = r if isinstance(r, list) else r.get("queries", [])
        except Exception:
            return []
        labels = []
        for q in queries:
            if not isinstance(q, dict):
                continue
            t = q.get("target", "")
            m = next((o for o in objs if t.lower() in o.raw_object_name.lower()), None)
            if m:
                labels.append({
                    "sample_id": sid, "bucket": Bucket.G.value,
                    "subtype": GSubtype.RELATIONAL_GROUNDING.value,
                    "scene_family": sf, "scenario_id": sc,
                    "object_id": m.object_id,
                    "question": f"Find {q.get('query', t)}.",
                    "answer": {"bbox": m.bbox, "object_name": m.raw_object_name},
                    "relation": q.get("relation", ""),
                    "reference_object": q.get("reference", ""),
                    "verifier_status": VerifierStatus.AUTO_VERIFIABLE.value,
                    "source_provenance": SourceProvenance.VLM.value,
                })
        return labels

    def _counting(self, sid, objs, sf, sc):
        counts: Dict[str, int] = {}
        for o in objs:
            counts[o.object_name] = counts.get(o.object_name, 0) + 1
        labels = [{
            "sample_id": sid, "bucket": Bucket.G.value,
            "subtype": GSubtype.COUNTING.value,
            "scene_family": sf, "scenario_id": sc,
            "question": f"How many {cls}(s) are there?",
            "answer": {"count": n, "object_class": cls},
            "verifier_status": VerifierStatus.AUTO_VERIFIABLE.value,
            "source_provenance": SourceProvenance.VLM.value,
        } for cls, n in counts.items()]
        labels.append({
            "sample_id": sid, "bucket": Bucket.G.value,
            "subtype": GSubtype.COUNTING.value,
            "scene_family": sf, "scenario_id": sc,
            "question": "How many objects total?",
            "answer": {"count": len(objs), "object_class": "all"},
            "verifier_status": VerifierStatus.AUTO_VERIFIABLE.value,
            "source_provenance": SourceProvenance.VLM.value,
        })
        return labels
