"""A_action_interface Pipeline — 2D geometric action interface for pick/push/place.

Refactored from a_action_pipeline.py to ONLY produce geometric outputs
(grasp points, push vectors, place regions). Precondition logic lives in
a_action_precondition_pipeline.py.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence

from .a_action_interface_contract import (
    LABEL_STATUS_CANDIDATE,
    LABEL_STATUS_TEACHER_GENERATED,
    ensure_action_interface_label_id,
    verify_action_interface_candidate,
)
from .a_pipeline_schemas import AActionInterfaceLabel, validate_action_interface
from .taxonomy import (
    AActionInterfaceSubtype, ActionType, LabelType,
    SourceProvenance, VerifierStatus,
)
from .visual_evidence import GsaEvidenceBundle, GsaObjectRef

logger = logging.getLogger("gsa.a_action_interface_pipeline")
_PUSH_ONLY_KW = {"card", "paper", "sheet", "thin", "flat", "coin", "ticket"}
_PIXEL_FRAMES = {"pixel", "pixels", "image_pixel", "image_pixels"}
_NORM_1000_FRAMES = {"normalized_1000", "qwen_1000", "0_1000", "0-1000"}
_NORM_01_FRAMES = {"normalized", "normalized_01", "relative", "0_1", "0-1"}
_KEY_ALIASES = {
    "grasp_point_2d": ("grasp_point_2d", "grasp_point", "grasp_center", "point"),
    "jaw_axis_2d": ("jaw_axis_2d", "jaw_axis", "jaw_direction", "gripper_axis"),
    "push_contact_point_2d": ("push_contact_point_2d", "push_contact_point", "contact_point"),
    "push_direction_2d": ("push_direction_2d", "push_direction", "direction"),
}


class AActionInterfacePipeline:
    """Produces AActionInterfaceLabel entries with pure geometric data."""

    def __init__(self, vlm_client=None, enable_repair: bool = True):
        self.vlm_client = vlm_client
        self.enable_repair = enable_repair

    def run(self, bundle: GsaEvidenceBundle) -> List[AActionInterfaceLabel]:
        labels: List[AActionInterfaceLabel] = []
        for obj in bundle.objects:
            for at in self._candidates(bundle, obj):
                lbl = self._build(bundle, obj, at)
                if lbl and validate_action_interface(lbl).valid:
                    labels.append(lbl)
                elif lbl:
                    r = validate_action_interface(lbl)
                    logger.warning("Validation failed obj=%s: %s", obj.object_id, r.errors or r.missing_fields)
        return labels

    def _candidates(self, bundle: GsaEvidenceBundle, obj: GsaObjectRef) -> List[str]:
        cands: List[str] = []
        push_only = any(kw in obj.object_name for kw in _PUSH_ONLY_KW)
        if not push_only:
            cands.append(ActionType.PICK.value)
        cands.append(ActionType.PUSH.value)
        if bundle.placement_regions:
            cands.append(ActionType.PLACE.value)
        return cands or [ActionType.PICK.value]

    def _build(self, bundle, obj, action_type) -> Optional[AActionInterfaceLabel]:
        if action_type == ActionType.PICK.value:
            gp, ja, generation_source = self._geom_pick(bundle, obj)
            return self._label(bundle, obj, action_type, LabelType.POINT.value,
                               AActionInterfaceSubtype.GRASP_POINT.value,
                               generation_source=generation_source,
                               grasp_point_2d=gp, jaw_axis_2d=ja)
        if action_type == ActionType.PUSH.value:
            cp, dr, generation_source = self._geom_push(bundle, obj)
            return self._label(bundle, obj, action_type, LabelType.VECTOR.value,
                               AActionInterfaceSubtype.PUSH_CONTACT.value,
                               generation_source=generation_source,
                               push_contact_point_2d=cp, push_direction_2d=dr)
        if action_type == ActionType.PLACE.value and bundle.placement_regions:
            region = bundle.placement_regions[0]
            return self._label(bundle, obj, action_type, LabelType.REGION.value,
                               AActionInterfaceSubtype.PLACE_REGION.value,
                               generation_source="heuristic",
                               place_region_2d={"type": "polygon", "points": region.polygon})
        return None

    # ---- Geometry: pick ----
    def _geom_pick(self, bundle, obj):
        gp, ja, generation_source = None, None, "heuristic"
        if self.vlm_client and bundle.image_path and obj.bbox:
            data = self._vlm_query(
                self._pick_prompt(bundle.image_path, obj),
                bundle.image_path,
                expected_keys=("grasp_point_2d", "jaw_axis_2d"),
            )
            if data:
                frame = data.get("coordinate_frame")
                gp = self._point_to_pixel(
                    self._value(data, "grasp_point_2d", "grasp_point", "grasp_center", "point"),
                    bundle.image_path,
                    frame,
                )
                ja = self._unit_vector(
                    self._value(data, "jaw_axis_2d", "jaw_axis", "jaw_direction", "gripper_axis")
                )
                if gp and ja:
                    generation_source = "vlm"
        if obj.bbox and not gp:
            cx = (obj.bbox[0] + obj.bbox[2]) / 2
            cy = (obj.bbox[1] + obj.bbox[3]) / 2
            gp = [cx, cy]
            w, h = obj.bbox[2] - obj.bbox[0], obj.bbox[3] - obj.bbox[1]
            ja = [1.0, 0.0] if w < h else [0.0, 1.0]
            generation_source = "heuristic"
        return gp, ja, generation_source

    # ---- Geometry: push ----

    def _geom_push(self, bundle, obj):
        cp, dr, generation_source = None, None, "heuristic"
        if self.vlm_client and bundle.image_path and obj.bbox:
            data = self._vlm_query(
                self._push_prompt(bundle.image_path, obj),
                bundle.image_path,
                expected_keys=("push_contact_point_2d", "push_direction_2d"),
            )
            if data:
                frame = data.get("coordinate_frame")
                cp = self._point_to_pixel(
                    self._value(data, "push_contact_point_2d", "push_contact_point", "contact_point"),
                    bundle.image_path,
                    frame,
                )
                dr = self._unit_vector(
                    self._value(data, "push_direction_2d", "push_direction", "direction")
                )
                if cp and dr:
                    generation_source = "vlm"
        if obj.bbox and not cp:
            cy = (obj.bbox[1] + obj.bbox[3]) / 2
            cp = [obj.bbox[0], cy]
            dr = [-1.0, 0.0]
            generation_source = "heuristic"
        return cp, dr, generation_source

    # ---- VLM helpers ----

    def _pick_prompt(self, image_path: str, obj: GsaObjectRef) -> str:
        size = self._image_size(image_path)
        size_text = f"Image size is {size[0]}x{size[1]} pixels.\n" if size else ""
        return (
            f"{size_text}"
            f'Object "{obj.raw_object_name}" is at bbox '
            f"[{obj.bbox[0]:.1f}, {obj.bbox[1]:.1f}, {obj.bbox[2]:.1f}, {obj.bbox[3]:.1f}] "
            "in the same pixel coordinate frame as the image.\n"
            "Choose one practical parallel-jaw grasp for this object.\n"
            "Rules:\n"
            "- Output no reasoning and no markdown.\n"
            "- The grasp point must be inside the bbox.\n"
            "- jaw_axis_2d is the gripper closing direction as a non-zero 2D vector.\n"
            '- coordinate_frame must be "pixel" if coordinates use image pixels, or "normalized_1000" only if using 0-1000 coordinates.\n'
            'Return exactly one line starting with FINAL_JSON: {"coordinate_frame":"pixel","grasp_point_2d":[x,y],"jaw_axis_2d":[dx,dy]}'
        )

    def _push_prompt(self, image_path: str, obj: GsaObjectRef) -> str:
        size = self._image_size(image_path)
        size_text = f"Image size is {size[0]}x{size[1]} pixels.\n" if size else ""
        return (
            f"{size_text}"
            f'Object "{obj.raw_object_name}" is at bbox '
            f"[{obj.bbox[0]:.1f}, {obj.bbox[1]:.1f}, {obj.bbox[2]:.1f}, {obj.bbox[3]:.1f}] "
            "in the same pixel coordinate frame as the image.\n"
            "Choose one practical push contact point on the object boundary and a direction to clear it.\n"
            "Rules:\n"
            "- Output no reasoning and no markdown.\n"
            "- The contact point should be on or inside the bbox boundary.\n"
            "- push_direction_2d is a non-zero 2D vector.\n"
            '- coordinate_frame must be "pixel" if coordinates use image pixels, or "normalized_1000" only if using 0-1000 coordinates.\n'
            'Return exactly one line starting with FINAL_JSON: {"coordinate_frame":"pixel","push_contact_point_2d":[x,y],"push_direction_2d":[dx,dy]}'
        )

    def _vlm_query(
        self,
        prompt: str,
        image_path: str,
        expected_keys: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(self.vlm_client, "chat"):
                raw = self.vlm_client.chat(
                    prompt=prompt,
                    image_path=image_path,
                    max_tokens=2048,
                    temperature=0.0,
                    system_prompt=(
                        "You are a robotic manipulation expert. "
                        "Return only the requested FINAL_JSON line."
                    ),
                )
                parsed = self._extract_expected_json(raw, expected_keys, log_failure=False)
                if parsed:
                    return parsed
                if self.enable_repair:
                    repaired = self._repair_json_response(raw, prompt, expected_keys)
                    if repaired:
                        return repaired
                logger.warning("No expected action-interface JSON found in VLM response: %s", raw[:240])
                return None

            res = self.vlm_client.chat_json(
                prompt=prompt,
                image_path=image_path,
                max_tokens=2048,
                temperature=0.0,
                system_prompt="You are a robotic manipulation expert. Respond in JSON only.",
            )
            return self._coerce_response_dict(res, expected_keys)
        except Exception as e:
            logger.warning("VLM query failed: %s", e)
        return None

    def _extract_expected_json(
        self,
        text: str,
        expected_keys: Sequence[str],
        log_failure: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        final_idx = text.rfind("FINAL_JSON")
        candidates: List[str] = []
        if final_idx >= 0:
            candidates.extend(self._json_object_strings(text[final_idx:]))
        candidates.extend(self._json_object_strings(text))

        for candidate in reversed(candidates):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            coerced = self._coerce_response_dict(parsed, expected_keys)
            if coerced:
                return coerced
        if log_failure:
            logger.warning("No expected action-interface JSON found in VLM response: %s", text[:240])
        return None

    def _repair_json_response(
        self,
        raw_response: str,
        original_prompt: str,
        expected_keys: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if not raw_response or not hasattr(self.vlm_client, "chat"):
            return None
        schema = self._schema_for_expected_keys(expected_keys)
        prompt = (
            "Convert the model response into the required action-interface JSON.\n"
            "Do not add reasoning, markdown, or any extra text.\n"
            f"Required schema: {schema}\n"
            "Use pixel coordinates if the response uses image/bbox pixel coordinates. "
            "If the response uses 0-1000 normalized coordinates, set coordinate_frame to normalized_1000.\n\n"
            "Original task:\n"
            f"{original_prompt}\n\n"
            "Model response to convert:\n"
            f"{raw_response}\n\n"
            f"Return exactly one line: FINAL_JSON: {schema}"
        )
        try:
            repaired = self.vlm_client.chat(
                prompt=prompt,
                image_path=None,
                max_tokens=1024,
                temperature=0.0,
                system_prompt="You convert text into valid JSON only.",
            )
        except Exception as exc:
            logger.warning("VLM JSON repair failed: %s", exc)
            return None
        return self._extract_expected_json(repaired, expected_keys)

    @staticmethod
    def _schema_for_expected_keys(expected_keys: Sequence[str]) -> str:
        if "grasp_point_2d" in expected_keys:
            return '{"coordinate_frame":"pixel","grasp_point_2d":[x,y],"jaw_axis_2d":[dx,dy]}'
        if "push_contact_point_2d" in expected_keys:
            return '{"coordinate_frame":"pixel","push_contact_point_2d":[x,y],"push_direction_2d":[dx,dy]}'
        return '{"coordinate_frame":"pixel"}'

    def _coerce_response_dict(
        self,
        response: Any,
        expected_keys: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if isinstance(response, dict):
            if any(
                self._value(response, *_KEY_ALIASES.get(key, (key,))) is not None
                for key in expected_keys
            ):
                return response
            return None
        if isinstance(response, list):
            for item in response:
                coerced = self._coerce_response_dict(item, expected_keys)
                if coerced:
                    return coerced
        return None

    @staticmethod
    def _json_object_strings(text: str) -> List[str]:
        candidates: List[str] = []
        depth = 0
        start = -1
        in_string = False
        escape_next = False
        for idx, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}":
                if depth:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        candidates.append(text[start:idx + 1])
                        start = -1
        return candidates

    @staticmethod
    def _value(data: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return None

    def _point_to_pixel(
        self,
        point: Any,
        image_path: str,
        coordinate_frame: Optional[str],
    ) -> Optional[List[float]]:
        pt = self._point(point)
        if pt is None:
            return None
        frame = (coordinate_frame or "").strip().lower()
        size = self._image_size(image_path)
        if not size:
            return pt
        w, h = size
        if frame in _PIXEL_FRAMES:
            return pt
        if frame in _NORM_01_FRAMES and all(0.0 <= v <= 1.0 for v in pt):
            return [pt[0] * w, pt[1] * h]
        if frame in _NORM_1000_FRAMES:
            return [pt[0] * w / 1000.0, pt[1] * h / 1000.0]
        if 0 <= pt[0] <= w and 0 <= pt[1] <= h:
            return pt
        if all(0 <= v <= 1000 for v in pt):
            return [pt[0] * w / 1000.0, pt[1] * h / 1000.0]
        return pt

    @staticmethod
    def _point(value: Any) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        if not all(isinstance(v, (int, float)) for v in value):
            return None
        return [float(value[0]), float(value[1])]

    @classmethod
    def _unit_vector(cls, value: Any) -> Optional[List[float]]:
        vec = cls._point(value)
        if vec is None:
            return None
        norm = math.hypot(vec[0], vec[1])
        if norm == 0:
            return None
        return [vec[0] / norm, vec[1] / norm]

    @staticmethod
    def _image_size(image_path: str) -> Optional[tuple[int, int]]:
        try:
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            w, h = img.size
            img.close()
            return w, h
        except Exception:
            return None

    # ---- Label factory ----

    def _label(self, bundle, obj, action_type, label_type, subtype, generation_source="heuristic", **kwargs):
        if generation_source == "vlm":
            source_provenance = SourceProvenance.VLM.value
            verifier_status = VerifierStatus.REVIEW_REQUIRED.value
            label_status = LABEL_STATUS_TEACHER_GENERATED
        else:
            source_provenance = SourceProvenance.ADAPTER.value
            verifier_status = VerifierStatus.HEURISTIC_ONLY.value
            label_status = LABEL_STATUS_CANDIDATE
        image_size = self._image_size(bundle.image_path) if bundle.image_path else None
        label = AActionInterfaceLabel(
            sample_id=bundle.sample_id,
            scene_family=bundle.scene_family,
            scenario_id=bundle.scenario_id,
            object_id=obj.object_id,
            source_provenance=source_provenance,
            verifier_status=verifier_status,
            label_status=label_status,
            stable_training_label=False,
            g_context=True,
            s_context=True,
            action_type=action_type,
            label_type=label_type,
            subtype=subtype,
            image_path=bundle.image_path,
            image_width=image_size[0] if image_size else None,
            image_height=image_size[1] if image_size else None,
            object_name=obj.raw_object_name,
            bbox=obj.bbox,
            **kwargs,
        )
        labeled = ensure_action_interface_label_id(label.to_dict())
        label.label_id = labeled["label_id"]
        verification = verify_action_interface_candidate(label.to_dict())
        label.verification = verification.to_dict()
        label.promotion_blockers = verification.promotion_blockers
        return label
