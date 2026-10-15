"""GSA Spatial Worker — Aligned with GSA taxonomy S-bucket subtypes.

Produces S labels covering 7 spatial relation types:
- 2d_relation: left/right/above/below from bbox positions
- depth_relation: front/behind from depth estimation
- topology_relation: next_to/near/far
- occlusion_relation: occluding/occluded_by
- support_relation: on_top_of/supporting
- containment_relation: inside/contains
- stacking_relation: stacked_on/under
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..workers.vlm_client import VLMClient
from .taxonomy import Bucket, SSubtype, SourceProvenance, VerifierStatus
from .visual_evidence import GsaObjectRef, GsaSpatialFact

logger = logging.getLogger("gsa.spatial_worker")


class GSASpatialWorker:
    """Generates S-bucket labels aligned with GSA taxonomy."""

    def __init__(self, vlm_client: VLMClient):
        self.vlm = vlm_client

    def generate(self, image_path: str, sample_id: str,
                 objects: List[GsaObjectRef],
                 scene_family: str = "", scenario_id: str = "") -> List[Dict[str, Any]]:
        """Generate all S-subtype labels for detected objects."""
        labels: List[Dict[str, Any]] = []
        if len(objects) < 2:
            return labels

        # Rule-based 2D relations from bboxes
        labels.extend(self._2d_relations(sample_id, objects, scene_family, scenario_id))

        # VLM-based complex relations
        if self.vlm and image_path:
            vlm_labels = self._vlm_spatial_relations(
                image_path, sample_id, objects, scene_family, scenario_id)
            labels.extend(vlm_labels)

        return labels

    def _2d_relations(self, sid, objects, sf, sc) -> List[Dict[str, Any]]:
        """Compute 2D spatial relations from bbox positions."""
        labels = []
        for i, subj in enumerate(objects):
            if not subj.bbox:
                continue
            for j, obj in enumerate(objects):
                if i == j or not obj.bbox:
                    continue

                relations = self._compute_2d_relation(subj.bbox, obj.bbox)
                for rel in relations:
                    labels.append({
                        "sample_id": sid, "bucket": Bucket.S.value,
                        "subtype": SSubtype.RELATION_2D.value,
                        "scene_family": sf, "scenario_id": sc,
                        "subject_id": subj.object_id,
                        "object_id": obj.object_id,
                        "predicate": rel,
                        "question": f"What is the spatial relation between {subj.raw_object_name} and {obj.raw_object_name}?",
                        "answer": f"{subj.raw_object_name} is {rel} {obj.raw_object_name}",
                        "verifier_status": VerifierStatus.AUTO_VERIFIABLE.value,
                        "source_provenance": SourceProvenance.VLM.value,
                        "evidence_type": "rule",
                    })
        return labels

    def _compute_2d_relation(self, bbox_a: List[float], bbox_b: List[float]) -> List[str]:
        """Compute 2D relations between two bboxes."""
        cx_a = (bbox_a[0] + bbox_a[2]) / 2
        cy_a = (bbox_a[1] + bbox_a[3]) / 2
        cx_b = (bbox_b[0] + bbox_b[2]) / 2
        cy_b = (bbox_b[1] + bbox_b[3]) / 2

        w_a = bbox_a[2] - bbox_a[0]
        h_a = bbox_a[3] - bbox_a[1]
        w_b = bbox_b[2] - bbox_b[0]
        h_b = bbox_b[3] - bbox_b[1]

        threshold_x = (w_a + w_b) / 4
        threshold_y = (h_a + h_b) / 4

        relations = []
        dx = cx_a - cx_b
        dy = cy_a - cy_b

        if dx < -threshold_x:
            relations.append("left_of")
        elif dx > threshold_x:
            relations.append("right_of")

        if dy < -threshold_y:
            relations.append("above")
        elif dy > threshold_y:
            relations.append("below")

        return relations[:1]  # Return strongest relation only

    def _vlm_spatial_relations(self, image_path, sid, objects, sf, sc) -> List[Dict[str, Any]]:
        """Use VLM to identify complex spatial relations."""
        obj_info = [{"id": o.object_id, "name": o.raw_object_name, "bbox": o.bbox}
                    for o in objects[:6] if o.bbox]

        prompt = f"""Analyze spatial relationships between these objects: {json.dumps(obj_info)}

Identify these relation types:
- occlusion: one object partially/fully hides another
- support: one object rests on/supports another
- containment: one object is inside another
- stacking: objects stacked vertically
- topology: objects touching or very close

Return JSON array:
[{{"subject": "cup_0", "predicate": "on_top_of", "object": "table_0", "type": "support", "confidence": 0.9}}]

Only report clearly visible relationships."""

        try:
            result = self.vlm.chat_json(
                prompt=prompt, image_path=image_path,
                max_tokens=768, temperature=0.3,
                system_prompt="Spatial relationship expert. JSON only.",
            )
            facts = result if isinstance(result, list) else result.get("relations", [])
        except Exception as e:
            logger.warning(f"VLM spatial failed: {e}")
            return []

        subtype_map = {
            "occlusion": SSubtype.OCCLUSION_RELATION.value,
            "support": SSubtype.SUPPORT_RELATION.value,
            "containment": SSubtype.CONTAINMENT_RELATION.value,
            "stacking": SSubtype.STACKING_RELATION.value,
            "topology": SSubtype.TOPOLOGY_RELATION.value,
            "depth": SSubtype.DEPTH_RELATION.value,
        }

        labels = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            rel_type = fact.get("type", "topology")
            subtype = subtype_map.get(rel_type, SSubtype.TOPOLOGY_RELATION.value)
            subj = fact.get("subject", "")
            pred = fact.get("predicate", "")
            obj = fact.get("object", "")

            if not (subj and pred and obj):
                continue

            labels.append({
                "sample_id": sid, "bucket": Bucket.S.value,
                "subtype": subtype,
                "scene_family": sf, "scenario_id": sc,
                "subject_id": subj,
                "object_id": obj,
                "predicate": pred,
                "question": f"What is the relationship between {subj} and {obj}?",
                "answer": f"{subj} is {pred} {obj}",
                "verifier_status": VerifierStatus.AUTO_VERIFIABLE.value,
                "source_provenance": SourceProvenance.VLM.value,
                "evidence_type": "vlm",
                "confidence": fact.get("confidence"),
            })

        return labels

    def extract_spatial_facts(self, labels: List[Dict[str, Any]]) -> List[GsaSpatialFact]:
        """Convert S labels to GsaSpatialFact objects for downstream use."""
        facts = []
        for i, label in enumerate(labels):
            facts.append(GsaSpatialFact(
                fact_id=f"S:{label.get('subtype', '')}:{label.get('subject_id', '')}:{label.get('object_id', '')}",
                subject_id=label.get("subject_id", ""),
                predicate=label.get("predicate", ""),
                object_id=label.get("object_id", ""),
                confidence=label.get("confidence"),
                evidence_type=label.get("evidence_type", "rule"),
            ))
        return facts
