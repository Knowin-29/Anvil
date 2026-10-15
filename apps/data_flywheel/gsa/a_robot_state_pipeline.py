"""Pipeline for robot end-effector / gripper state recognition."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .a_pipeline_schemas import ARobotStateLabel, validate_robot_state
from .taxonomy import (
    ARobotStateSubtype,
    CameraView,
    EndEffector,
    SourceProvenance,
    VerifierStatus,
)
from .visual_evidence import GsaEvidenceBundle

logger = logging.getLogger("gsa.a_robot_state_pipeline")

_ROBOT_CAMERA_KEYWORDS = re.compile(
    r"wrist|gripper|hand_camera|bridgedatav2|bridge_data_v2|episode_|"
    r"sharerobot|embodied_reasoner|robo2vlm",
    re.IGNORECASE,
)

_STATE_TYPES = [s.value for s in ARobotStateSubtype]
_STATE_VALUES = {"yes", "no", "unknown"}
_EVIDENCE_MIN_LEN = 12
_GENERIC_EVIDENCE = {
    "not enough evidence",
    "unclear",
    "unknown",
    "cannot determine",
    "not visible",
    "no evidence",
    "n/a",
    "none",
}
_END_EFFECTOR_EVIDENCE_TERMS = (
    "gripper", "jaw", "finger", "fingers", "tool", "end-effector", "end effector",
    "end-effector", "holding",
)

_VISIBILITY_PROMPT = """You are reviewing a robot wrist camera image plus two auxiliary crops.
Stage 1: judge whether the robot gripper/end-effector or tool is visible and clear enough for robot-state labeling.

Return JSON only:
{
  "visibility": {
    "gripper_visible": true|false|null,
    "gripper_clear": true|false|null,
    "tool_visible": true|false|null,
    "tool_clear": true|false|null,
    "suitable_for_robot_state": true|false|null,
    "visual_evidence": "specific visible evidence, not a generic statement"
  }
}

Rules:
- Bridge images are wrist-camera views; still judge visibility from pixels.
- Use null if the image is too ambiguous.
- Do not infer hidden gripper/tool state without visible evidence.
"""

_GRASP_STATE_PROMPT = """You are reviewing a robot wrist camera image plus two auxiliary crops.
Stage 2: judge grasp-related robot states only.

Return JSON only:
{
  "states": {
    "gripper_empty": {"value": "yes|no|unknown", "confidence": 0.0, "target_object_class": null, "visual_evidence": "specific visual evidence"},
    "holding_object": {"value": "yes|no|unknown", "confidence": 0.0, "target_object_class": "object name or null", "visual_evidence": "specific visual evidence"},
    "object_between_fingers": {"value": "yes|no|unknown", "confidence": 0.0, "target_object_class": "object name or null", "visual_evidence": "specific visual evidence"}
  }
}

Rules:
- Each yes/no must include concrete visual evidence from the gripper fingers, jaws, or object position.
- holding_object=yes requires a visible object held by the gripper and a target_object_class.
- object_between_fingers=yes means an object is visibly between the gripper fingers/jaws.
- Use unknown when the gripper/object relationship is not visible or unclear.
"""

_CONTACT_TOOL_PROMPT = """You are reviewing a robot wrist camera image plus two auxiliary crops.
Stage 3: judge contact and tool readiness only.

Return JSON only:
{
  "states": {
    "contact_state": {"value": "yes|no|unknown", "confidence": 0.0, "target_object_class": "object/surface name or null", "visual_evidence": "specific visual evidence"},
    "tool_ready": {"value": "yes|no|unknown", "confidence": 0.0, "target_object_class": null, "visual_evidence": "specific visual evidence"}
  }
}

Rules:
- contact_state=yes means the robot gripper/tool/end-effector is visibly touching an object or surface.
- contact_state does not mean an object is touching the table by itself.
- tool_ready=yes requires the gripper/tool to be clearly visible, unobstructed, and positioned for use.
- Use unknown if visibility is insufficient.
"""

_VERIFIER_PROMPT = """You are verifying candidate robot-state labels against the original image and auxiliary crops.
For each candidate, decide whether the state_value and visual_evidence are supported by visible pixels.

Return JSON only:
{
  "verifications": {
    "gripper_empty": {"result": "supported|unsupported|unsure", "reason": "short visual reason"},
    "holding_object": {"result": "supported|unsupported|unsure", "reason": "short visual reason"},
    "object_between_fingers": {"result": "supported|unsupported|unsure", "reason": "short visual reason"},
    "contact_state": {"result": "supported|unsupported|unsure", "reason": "short visual reason"},
    "tool_ready": {"result": "supported|unsupported|unsure", "reason": "short visual reason"}
  }
}

Rules:
- supported: visible evidence directly supports the candidate.
- unsupported: visible evidence contradicts the candidate.
- unsure: visibility or evidence is insufficient.
- Be strict: generic evidence is not supported.
"""


def _is_bridge_image_path(image_path: Optional[str]) -> bool:
    if not image_path:
        return False
    lp = image_path.lower()
    return any(k in lp for k in ("bridgedatav2", "bridge_data_v2", "episode_"))


def _infer_camera_view(image_path: Optional[str]) -> str:
    if not image_path:
        return CameraView.EXTERNAL.value
    lp = image_path.lower()
    if _is_bridge_image_path(image_path) or "wrist" in lp or "hand_camera" in lp:
        return CameraView.WRIST.value
    if "embodied_reasoner" in lp or ".image_cache" in lp or "head" in lp:
        return CameraView.HEAD.value
    return CameraView.EXTERNAL.value


def _infer_source_dataset(image_path: Optional[str]) -> str:
    if not image_path:
        return "vlm"
    lp = image_path.lower()
    if "fmb_extracted" in lp or "/fmb/" in lp:
        return "fmb"
    if "droid_100" in lp or "/droid/" in lp:
        return "droid_100"
    if "embodied_reasoner" in lp:
        return "embodied_reasoner"
    if "robo2vlm" in lp:
        return "robo2vlm"
    if "sharerobot" in lp:
        return "sharerobot"
    return "bridge_data_v2" if _is_bridge_image_path(image_path) else "vlm"


def _path_under_roots(image_path: Optional[str], roots: List[str]) -> bool:
    if not image_path:
        return False
    try:
        resolved = str(Path(image_path).resolve())
    except Exception:
        resolved = str(image_path)
    for root in roots:
        if not root:
            continue
        try:
            resolved_root = str(Path(root).resolve())
        except Exception:
            resolved_root = str(root)
        if resolved.startswith(resolved_root):
            return True
    return False


def _is_robot_camera(bundle: GsaEvidenceBundle) -> bool:
    path = bundle.image_path or ""
    ctx = bundle.business_context.get("source", "")
    return bool(_is_bridge_image_path(path) or _ROBOT_CAMERA_KEYWORDS.search(path) or _ROBOT_CAMERA_KEYWORDS.search(ctx))


def _parse_vlm_response(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        return {}


def _normalize_value(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text in _STATE_VALUES:
        return text
    if text in {"true", "1"}:
        return "yes"
    if text in {"false", "0"}:
        return "no"
    return "unknown"


def _normalize_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf < 0:
        return 0.0
    if conf > 1:
        return 1.0
    return conf


def _normalize_target(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "unknown"}:
        return None
    return text


def _normalize_evidence(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text or text.lower() in {"null", "none"}:
        return None
    return text


def _evidence_is_weak(evidence: Optional[str]) -> bool:
    if not evidence:
        return True
    text = evidence.strip().lower()
    if len(text) < _EVIDENCE_MIN_LEN:
        return True
    return text in _GENERIC_EVIDENCE


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def _visibility_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    payload = parsed.get("visibility", parsed)
    if not isinstance(payload, dict):
        return {}
    return {
        "gripper_visible": _normalize_bool(payload.get("gripper_visible")),
        "gripper_clear": _normalize_bool(payload.get("gripper_clear")),
        "tool_visible": _normalize_bool(payload.get("tool_visible")),
        "tool_clear": _normalize_bool(payload.get("tool_clear")),
        "suitable_for_robot_state": _normalize_bool(payload.get("suitable_for_robot_state")),
        "visual_evidence": _normalize_evidence(payload.get("visual_evidence")),
    }


def _extract_state_items(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize new multi-label and old single-label VLM response shapes."""
    states = parsed.get("states")
    items: List[Dict[str, Any]] = []
    if isinstance(states, dict):
        for state_type, payload in states.items():
            if state_type not in _STATE_TYPES:
                continue
            payload = payload if isinstance(payload, dict) else {"value": payload}
            items.append({
                "state_type": state_type,
                "state_value": payload.get("value", payload.get("state_value")),
                "confidence": payload.get("confidence"),
                "target_object_class": payload.get("target_object_class"),
                "visual_evidence": payload.get("visual_evidence", payload.get("evidence")),
            })
        return items

    state_type = parsed.get("state_type")
    if state_type in _STATE_TYPES:
        items.append({
            "state_type": state_type,
            "state_value": parsed.get("state_value"),
            "confidence": parsed.get("confidence"),
            "target_object_class": parsed.get("target_object_class"),
            "visual_evidence": parsed.get("visual_evidence", parsed.get("evidence")),
        })
    return items


def _merge_state_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_type = {item.get("state_type"): item for item in items if item.get("state_type") in _STATE_TYPES}
    merged: List[Dict[str, Any]] = []
    for state_type in _STATE_TYPES:
        item = dict(by_type.get(state_type, {}))
        item.setdefault("state_type", state_type)
        item.setdefault("state_value", "unknown")
        item.setdefault("confidence", None)
        item.setdefault("target_object_class", None)
        item.setdefault("visual_evidence", "state not returned by model")
        merged.append(item)
    return merged


class ARobotStatePipeline:
    """Recognizes robot gripper / end-effector state from camera images."""

    def __init__(
        self,
        vlm_client=None,
        auto_pass_confidence: float = 0.8,
        force_robot_state_vlm: bool = False,
        robot_state_vlm_roots: Optional[List[str]] = None,
    ):
        self.vlm_client = vlm_client
        self.auto_pass_confidence = auto_pass_confidence
        self.force_robot_state_vlm = force_robot_state_vlm
        self.robot_state_vlm_roots = robot_state_vlm_roots or []

    def accepts_image_path(self, image_path: Optional[str]) -> bool:
        if self.force_robot_state_vlm:
            return True
        if _path_under_roots(image_path, self.robot_state_vlm_roots):
            return True
        if not image_path:
            return False
        dummy = GsaEvidenceBundle(
            sample_id="accepts_image_path",
            image_path=image_path,
            objects=[],
            spatial_facts=[],
        )
        return _is_robot_camera(dummy)

    def run(self, bundle: GsaEvidenceBundle) -> List[ARobotStateLabel]:
        if not self.accepts_image_path(bundle.image_path):
            return []

        if self.vlm_client is None:
            logger.warning("No VLM client configured; skipping robot state inference.")
            return []

        crops = self._build_auxiliary_crops(bundle.image_path)
        try:
            visibility_raw = self.vlm_client.query(
                image_path=bundle.image_path,
                pil_images=crops,
                prompt=_VISIBILITY_PROMPT,
                temperature=0.2,
            )
            visibility_parsed = _parse_vlm_response(visibility_raw)
            if "states" in visibility_parsed or "state_type" in visibility_parsed:
                labels = self._build_labels(bundle, _extract_state_items(visibility_parsed), {})
                self._apply_cross_state_quality(labels)
                self._finalize_verifier_status(labels)
                return labels

            visibility = _visibility_payload(visibility_parsed)
            grasp_parsed = self._query_stage(bundle, crops, _GRASP_STATE_PROMPT)
            contact_parsed = self._query_stage(bundle, crops, _CONTACT_TOOL_PROMPT)
        except Exception:
            logger.exception("VLM call failed for sample %s", bundle.sample_id)
            return []

        items = _merge_state_items(_extract_state_items(grasp_parsed) + _extract_state_items(contact_parsed))
        labels = self._build_labels(bundle, items, visibility)
        self._apply_visibility_quality(labels, visibility)
        self._apply_cross_state_quality(labels)
        self._apply_verifier(bundle, crops, labels)
        self._finalize_verifier_status(labels)
        return labels

    def _query_stage(self, bundle: GsaEvidenceBundle, crops: List[Image.Image], prompt: str) -> Dict[str, Any]:
        raw = self.vlm_client.query(
            image_path=bundle.image_path,
            pil_images=crops,
            prompt=prompt,
            temperature=0.2,
        )
        parsed = _parse_vlm_response(raw)
        if not parsed:
            logger.warning("Could not parse staged VLM response for %s", bundle.sample_id)
        return parsed

    def _build_auxiliary_crops(self, image_path: Optional[str]) -> List[Image.Image]:
        if not image_path:
            return []
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            logger.debug("Could not build robot-state crops from %s", image_path, exc_info=True)
            return []

        w, h = img.size
        crop_size = int(min(w, h) * 0.72)
        left = max((w - crop_size) // 2, 0)
        top = max((h - crop_size) // 2, 0)
        center_crop = img.crop((left, top, min(left + crop_size, w), min(top + crop_size, h)))
        lower_half = img.crop((0, int(h * 0.4), w, h))
        return [center_crop, lower_half]

    def _build_labels(
        self,
        bundle: GsaEvidenceBundle,
        items: List[Dict[str, Any]],
        visibility: Dict[str, Any],
    ) -> List[ARobotStateLabel]:
        labels: List[ARobotStateLabel] = []
        camera_view = _infer_camera_view(bundle.image_path)

        for item in items:
            state_type = item.get("state_type")
            if state_type not in _STATE_TYPES:
                continue
            state_value = _normalize_value(item.get("state_value"))
            confidence = _normalize_confidence(item.get("confidence"))
            target_obj = _normalize_target(item.get("target_object_class"))
            visual_evidence = _normalize_evidence(item.get("visual_evidence"))
            quality_flags = self._quality_flags(
                state_type, state_value, confidence, target_obj, visual_evidence
            )

            label = ARobotStateLabel(
                sample_id=bundle.sample_id,
                scene_family=bundle.scene_family,
                scenario_id=bundle.scenario_id,
                camera_view=camera_view,
                end_effector=EndEffector.PARALLEL_JAW.value,
                state_type=state_type,
                subtype=state_type,
                state_value=state_value,
                target_object_class=target_obj,
                visual_evidence=visual_evidence,
                confidence=confidence,
                image_path=bundle.image_path,
                source_dataset=_infer_source_dataset(bundle.image_path),
                source_provenance=SourceProvenance.VLM.value,
                label_status="teacher_generated",
                stable_training_label=False,
                quality_flags=quality_flags,
            )

            self._apply_state_specific_quality(label, visibility)
            validation = validate_robot_state(label)
            if not validation.valid:
                label.quality_flags.extend(f"schema_invalid:{err}" for err in validation.errors)
                logger.warning("Label validation failed: %s", validation.errors)
            labels.append(label)
        return labels

    def _quality_flags(
        self,
        state_type: str,
        state_value: str,
        confidence: Optional[float],
        target_obj: Optional[str],
        visual_evidence: Optional[str],
    ) -> List[str]:
        flags: List[str] = []
        if confidence is None:
            flags.append("missing_confidence")
        elif confidence < self.auto_pass_confidence:
            flags.append("low_confidence")
        if state_value == "unknown":
            flags.append("unknown_state_value")
        if state_value in {"yes", "no"}:
            if not visual_evidence:
                flags.append("missing_visual_evidence")
            elif _evidence_is_weak(visual_evidence):
                flags.append("weak_visual_evidence")
        if state_type == ARobotStateSubtype.HOLDING_OBJECT.value:
            if state_value == "yes" and not target_obj:
                flags.append("holding_object_missing_target")
        return flags

    def _apply_state_specific_quality(self, label: ARobotStateLabel, visibility: Dict[str, Any]) -> None:
        evidence = (label.visual_evidence or "").lower()
        if label.state_type == ARobotStateSubtype.CONTACT_STATE.value and label.state_value == "yes":
            if not any(term in evidence for term in _END_EFFECTOR_EVIDENCE_TERMS):
                label.quality_flags.append("contact_evidence_not_end_effector")
        if label.state_type == ARobotStateSubtype.TOOL_READY.value and label.state_value == "yes":
            if visibility:
                clear = visibility.get("tool_clear") is True or visibility.get("gripper_clear") is True
                visible = visibility.get("tool_visible") is True or visibility.get("gripper_visible") is True
                if not (clear and visible):
                    label.state_value = "unknown"
                    label.quality_flags.append("tool_not_clearly_visible")
                    if "unknown_state_value" not in label.quality_flags:
                        label.quality_flags.append("unknown_state_value")

    def _apply_visibility_quality(self, labels: List[ARobotStateLabel], visibility: Dict[str, Any]) -> None:
        if not visibility:
            for label in labels:
                label.quality_flags.append("missing_visibility_stage")
            return
        not_clear = (
            visibility.get("suitable_for_robot_state") is False
            or visibility.get("gripper_visible") is False
            or visibility.get("gripper_clear") is False
        )
        if not not_clear:
            return
        for label in labels:
            if label.state_type != ARobotStateSubtype.TOOL_READY.value:
                label.quality_flags.append("gripper_not_visible_or_unclear")

    def _apply_cross_state_quality(self, labels: List[ARobotStateLabel]) -> None:
        by_type = {label.state_type: label for label in labels}
        empty = by_type.get(ARobotStateSubtype.GRIPPER_EMPTY.value)
        conflicting_types = [
            ARobotStateSubtype.HOLDING_OBJECT.value,
            ARobotStateSubtype.OBJECT_BETWEEN_FINGERS.value,
        ]
        if not empty or empty.state_value != "yes":
            return

        conflict = [
            by_type[t] for t in conflicting_types
            if t in by_type and by_type[t].state_value == "yes"
        ]
        if not conflict:
            return

        for label in [empty, *conflict]:
            if "mutually_exclusive_state_conflict" not in label.quality_flags:
                label.quality_flags.append("mutually_exclusive_state_conflict")

    def _apply_verifier(self, bundle: GsaEvidenceBundle, crops: List[Image.Image], labels: List[ARobotStateLabel]) -> None:
        candidates = [
            {
                "state_type": label.state_type,
                "state_value": label.state_value,
                "target_object_class": label.target_object_class,
                "visual_evidence": label.visual_evidence,
                "confidence": label.confidence,
            }
            for label in labels
        ]
        prompt = _VERIFIER_PROMPT + "\nCandidate labels:\n" + json.dumps(
            {"states": candidates}, ensure_ascii=False
        )
        try:
            parsed = self._query_stage(bundle, crops, prompt)
        except Exception:
            logger.exception("Verifier VLM call failed for sample %s", bundle.sample_id)
            for label in labels:
                label.quality_flags.append("verifier_failed")
            return

        verifications = parsed.get("verifications") if isinstance(parsed, dict) else None
        if not isinstance(verifications, dict):
            for label in labels:
                label.quality_flags.append("verifier_missing_result")
            return

        for label in labels:
            payload = verifications.get(label.state_type, {})
            if not isinstance(payload, dict):
                payload = {}
            result = str(payload.get("result", "unsure")).strip().lower()
            if result not in {"supported", "unsupported", "unsure"}:
                result = "unsure"
            label.verifier_result = result
            label.verifier_reason = _normalize_evidence(payload.get("reason"))
            if result == "unsupported":
                label.quality_flags.append("verifier_unsupported")
            elif result == "unsure":
                label.quality_flags.append("verifier_unsure")

    def _finalize_verifier_status(self, labels: List[ARobotStateLabel]) -> None:
        for label in labels:
            # Keep ordering stable while removing duplicate flags from repeated quality passes.
            label.quality_flags = list(dict.fromkeys(label.quality_flags))
            label.verifier_status = (
                VerifierStatus.REVIEW_REQUIRED.value
                if label.quality_flags else VerifierStatus.AUTO_VERIFIABLE.value
            )
