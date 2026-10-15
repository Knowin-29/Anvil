"""A_identity_attr Pipeline — Object identity and attribute extraction.

Iterates over detected objects in a GsaEvidenceBundle, produces recognition
labels, and optionally queries a VLM for visual attributes (color, material,
shape, size).
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from .a_pipeline_schemas import AIdentityAttrLabel, validate_identity_attr
from .taxonomy import (
    AIdentityAttrSubtype,
    Bucket,
    KNOWN_COLORS,
    KNOWN_MATERIALS,
    KNOWN_SHAPES,
    SourceProvenance,
    VerifierStatus,
)
from .visual_evidence import GsaEvidenceBundle, GsaObjectRef

logger = logging.getLogger("gsa.a_identity_attr_pipeline")

_VLM_ATTR_PROMPT = (
    'Look at the object "{name}" at bbox [{bbox}].\n'
    "Identify its visual attributes:\n"
    "- color: the dominant color\n"
    "- material: what it's made of\n"
    "- shape: overall shape category\n"
    "- size: relative size (small/medium/large)\n\n"
    'Return JSON: {{"color": "...", "material": "...", "shape": "...", "size": "..."}}\n'
    "Only include attributes you can clearly observe."
)


class AIdentityAttrPipeline:
    """Pipeline that produces AIdentityAttrLabel entries from evidence bundles."""

    def __init__(self, vlm_client: Optional[Any] = None, enrich_attrs: bool = False) -> None:
        self.vlm_client = vlm_client
        self.enrich_attrs = enrich_attrs

    def run(self, bundle: GsaEvidenceBundle) -> List[AIdentityAttrLabel]:
        """Run identity + attribute extraction for all objects in *bundle*.
        
        By default only extracts object_class from G-stage detections (no VLM call).
        Set enrich_attrs=True to query VLM for color/material/shape/size.
        """
        labels: List[AIdentityAttrLabel] = []
        for obj in bundle.objects:
            label = self._build_label(obj, bundle)
            if self.enrich_attrs and self.vlm_client and bundle.image_path:
                self._enrich_with_vlm(label, obj, bundle.image_path)
            labels.append(label)
        return labels

    def _build_label(
        self, obj: GsaObjectRef, bundle: GsaEvidenceBundle
    ) -> AIdentityAttrLabel:
        """Create a base recognition label from an object reference."""
        return AIdentityAttrLabel(
            sample_id=bundle.sample_id,
            scene_family=bundle.scene_family,
            scenario_id=bundle.scenario_id,
            object_id=obj.object_id,
            subtype=AIdentityAttrSubtype.OBJECT_CLASS.value,
            object_class=obj.object_name,
            bbox=obj.bbox,
            g_context=obj.bbox is not None,
            source_provenance=SourceProvenance.VLM.value,
            verifier_status=VerifierStatus.AUTO_VERIFIABLE.value,
        )

    def _enrich_with_vlm(
        self,
        label: AIdentityAttrLabel,
        obj: GsaObjectRef,
        image_path: str,
    ) -> None:
        """Query VLM for visual attributes and merge into *label*."""
        bbox_str = ",".join(str(v) for v in obj.bbox) if obj.bbox else "0,0,1,1"
        prompt = _VLM_ATTR_PROMPT.format(name=obj.raw_object_name, bbox=bbox_str)

        try:
            response = self.vlm_client.chat(prompt, image_path=image_path, temperature=0.2)
            attrs = json.loads(response) if isinstance(response, str) else response
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            logger.warning("VLM attribute extraction failed for %s: %s", obj.object_id, exc)
            return
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected VLM error for %s: %s", obj.object_id, exc)
            return
        if not isinstance(attrs, dict):
            logger.warning("VLM attribute extraction returned non-object for %s", obj.object_id)
            return

        merged = label.attributes.copy()
        for key in ("color", "material", "shape", "size"):
            value = attrs.get(key)
            if value and isinstance(value, str):
                merged[key] = value.strip().lower()
        label.attributes = merged
        label.subtype = AIdentityAttrSubtype.MULTI_ATTR.value
