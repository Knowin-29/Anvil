"""A_state Pipeline — Object identity and state recognition.

Answers: "What is this object?" and "What state is it in?"
Consumes G detections and produces A_state labels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .taxonomy import (
    AStateSubtype,
    Bucket,
    SceneFamily,
    SourceProvenance,
    VerifierStatus,
    STATE_KEYS,
)
from .visual_evidence import GsaEvidenceBundle, GsaObjectRef

logger = logging.getLogger("gsa.a_state_pipeline")


@dataclass
class AStateLabel:
    """A single A_state label row."""

    sample_id: str
    bucket: str = Bucket.A_STATE.value
    subtype: str = ""
    scene_family: str = ""
    scenario_id: str = ""
    object_id: str = ""
    object_name: str = ""
    raw_object_name: str = ""
    bbox: Optional[List[float]] = None
    state_key: Optional[str] = None
    state_value: Optional[str] = None
    verifier_status: str = VerifierStatus.REVIEW_REQUIRED.value
    source_provenance: str = SourceProvenance.VLM.value
    evidence_ids: List[str] = field(default_factory=list)
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AStatePipeline:
    """Pipeline to produce A_state labels from an evidence bundle.

    Two stages:
    1. Object recognition: identity normalization
    2. State recognition: controlled state enum assignment
    """

    def __init__(self, vlm_client=None):
        self.vlm_client = vlm_client

    def run(self, bundle: GsaEvidenceBundle) -> List[AStateLabel]:
        """Process an evidence bundle and produce A_state labels."""
        labels: List[AStateLabel] = []

        for obj in bundle.objects:
            # Stage 1: Object recognition
            recog_label = self._build_recognition_label(bundle, obj)
            labels.append(recog_label)

            # Stage 2: State recognition (if VLM available)
            if self.vlm_client and bundle.image_path:
                state_labels = self._build_state_labels(bundle, obj)
                labels.extend(state_labels)

        return labels

    def _build_recognition_label(
        self, bundle: GsaEvidenceBundle, obj: GsaObjectRef
    ) -> AStateLabel:
        """Build an object_recognition label."""
        verifier = VerifierStatus.AUTO_VERIFIABLE.value
        if not obj.bbox:
            verifier = VerifierStatus.REVIEW_REQUIRED.value

        return AStateLabel(
            sample_id=bundle.sample_id,
            subtype=AStateSubtype.OBJECT_RECOGNITION.value,
            scene_family=bundle.scene_family,
            scenario_id=bundle.scenario_id,
            object_id=obj.object_id,
            object_name=obj.object_name,
            raw_object_name=obj.raw_object_name,
            bbox=obj.bbox,
            verifier_status=verifier,
            source_provenance=obj.source_provenance,
            evidence_ids=[f"G:{obj.object_id}"],
            confidence=obj.confidence,
        )

    def _build_state_labels(
        self, bundle: GsaEvidenceBundle, obj: GsaObjectRef
    ) -> List[AStateLabel]:
        """Use VLM to identify object states."""
        if not self.vlm_client or not bundle.image_path:
            return []

        prompt = self._state_prompt(obj)
        try:
            response = self.vlm_client.chat_json(
                prompt=prompt,
                image_path=bundle.image_path,
                max_tokens=512,
                temperature=0.3,
                system_prompt="You are an expert at identifying object states in indoor scenes. Respond in JSON only.",
            )
        except Exception as e:
            logger.warning(f"VLM state recognition failed for {obj.object_id}: {e}")
            return []

        labels: List[AStateLabel] = []
        states = response if isinstance(response, list) else response.get("states", [])

        for state_entry in states:
            if not isinstance(state_entry, dict):
                continue
            state_key = state_entry.get("state_key", "")
            state_value = state_entry.get("state_value", "")

            if state_key not in STATE_KEYS:
                continue

            verifier = VerifierStatus.AUTO_VERIFIABLE.value
            if state_key in {"clean_dirty", "whole_broken"}:
                verifier = VerifierStatus.REVIEW_REQUIRED.value

            labels.append(AStateLabel(
                sample_id=bundle.sample_id,
                subtype=AStateSubtype.STATE_RECOGNITION.value,
                scene_family=bundle.scene_family,
                scenario_id=bundle.scenario_id,
                object_id=obj.object_id,
                object_name=obj.object_name,
                raw_object_name=obj.raw_object_name,
                bbox=obj.bbox,
                state_key=state_key,
                state_value=state_value,
                verifier_status=verifier,
                source_provenance=SourceProvenance.VLM.value,
                evidence_ids=[f"G:{obj.object_id}", f"A_state:{obj.object_id}"],
                confidence=state_entry.get("confidence"),
            ))

        return labels

    def _state_prompt(self, obj: GsaObjectRef) -> str:
        bbox_info = ""
        if obj.bbox:
            bbox_info = f" (located at bbox [{obj.bbox[0]:.0f},{obj.bbox[1]:.0f},{obj.bbox[2]:.0f},{obj.bbox[3]:.0f}])"

        return f"""Look at the object "{obj.raw_object_name}"{bbox_info} in this image.
Identify its current physical states from these categories ONLY:
- open_closed: is it open or closed?
- empty_filled: is it empty or filled?
- upright_fallen: is it upright or fallen over?
- visible_occluded: is it fully visible or partially occluded?
- inside_outside: is it inside a container or outside?

Only report states that are clearly observable. Skip ambiguous ones.

Return JSON array:
[{{"state_key": "open_closed", "state_value": "open", "confidence": 0.9}}]"""
