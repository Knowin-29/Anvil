"""Episode-level multi-frame A_robot_state inference for DROID/FMB manifests."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .a_pipeline_schemas import ARobotStateLabel, validate_robot_state
from .data_adapters.droid_robot_state_adapter import DEFAULT_DROID_EXTRACT_ROOT
from .data_adapters.fmb_robot_state_adapter import DEFAULT_FMB_EXTRACT_ROOT
from .data_adapters.robot_state_ingest import DEFAULT_DROID_100_ROOT, DEFAULT_FMB_ROOT
from .taxonomy import (
    ARobotStateSubtype,
    Bucket,
    CameraView,
    EndEffector,
    SourceProvenance,
    VerifierStatus,
)

logger = logging.getLogger("gsa.episode_robot_state_pipeline")

STATE_TYPES = [state.value for state in ARobotStateSubtype]
STATE_VALUES = {"yes", "no", "unknown"}
GENERIC_EVIDENCE = {
    "",
    "unknown",
    "unclear",
    "not enough evidence",
    "cannot determine",
    "n/a",
    "none",
}

EPISODE_PROMPT = """You are labeling robot gripper/end-effector state from an ordered episode.
The input images are sampled wrist-camera frames in chronological order.

Return JSON only:
{
  "frames": [
    {
      "frame_index": 0,
      "step_index": 0,
      "states": {
        "gripper_empty": {"value": "yes|no|unknown", "confidence": 0.0, "target_object_class": null, "visual_evidence": "specific cross-frame visual evidence"},
        "holding_object": {"value": "yes|no|unknown", "confidence": 0.0, "target_object_class": "object name or null", "visual_evidence": "specific cross-frame visual evidence"},
        "object_between_fingers": {"value": "yes|no|unknown", "confidence": 0.0, "target_object_class": "object name or null", "visual_evidence": "specific cross-frame visual evidence"},
        "contact_state": {"value": "yes|no|unknown", "confidence": 0.0, "target_object_class": "object/surface name or null", "visual_evidence": "specific cross-frame visual evidence"},
        "tool_ready": {"value": "yes|no|unknown", "confidence": 0.0, "target_object_class": null, "visual_evidence": "specific cross-frame visual evidence"}
      }
    }
  ],
  "episode_summary": {
    "episode_outcome": "success|failure|unknown",
    "state_timeline": ["short chronological state changes"],
    "summary_evidence": "specific evidence across frames",
    "confidence": 0.0
  },
  "dynamic_annotation": {
    "task_parse": {
      "instruction": "task instruction or primitive",
      "task_type": "pick|place|push|insert|open|close|move|unknown",
      "target_object_class": "object name or null",
      "destination": "destination or null",
      "instruction_role": "prior_only",
      "expected_events": ["short expected task events"]
    },
    "sampled_frames": [
      {"frame_index": 0, "step_index": 0, "phase": "pre|approach|interaction|transport|release|post|unknown", "phase_confidence": 0.0}
    ],
    "phase_segments": [
      {"phase": "pre|approach|interaction|transport|release|post|unknown", "start_step": 0, "end_step": 0, "state_summary": "short state summary", "evidence": "specific cross-frame evidence", "confidence": 0.0}
    ],
    "state_transitions": [
      {"transition_type": "grasp_acquired|contact_started|object_released|object_moved|tool_ready_changed|unknown", "from_step": 0, "to_step": 0, "before_state": {}, "after_state": {}, "target_object_class": "object name or null", "evidence": "specific transition evidence", "confidence": 0.0}
    ],
    "episode_outcome": {"value": "success|failure|partial|unknown", "evidence": "specific evidence across sampled frames", "confidence": 0.0}
  }
}

Rules:
- Return one frame entry for every sampled frame listed below.
- Each frame must include all five states.
- holding_object=yes requires a visible object held by the gripper and a target_object_class.
- object_between_fingers=yes means an object is visibly between gripper fingers/jaws.
- contact_state=yes only means the gripper/tool/end-effector touches an object or surface.
- Use unknown when visibility is insufficient.
- Instruction is a task prior only; do not mark a state or transition positive without visual or temporal evidence.
"""

TASK_CONTEXT_FIELDS = (
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
    "primitive",
)
TASK_TYPES = {"pick", "place", "push", "insert", "open", "close", "move", "unknown"}
PHASES = {"pre", "approach", "interaction", "transport", "release", "post", "unknown"}
TRANSITION_TYPES = {
    "grasp_acquired",
    "contact_started",
    "object_released",
    "object_moved",
    "tool_ready_changed",
    "unknown",
}
TASKS_EXPECTING_TRANSITIONS = {"pick", "place", "push", "insert"}


@dataclass(frozen=True)
class EpisodeFrame:
    source_dataset: str
    episode_id: str
    frame_index: int
    step_index: int
    image_path: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class EpisodeBatch:
    source_dataset: str
    episode_id: str
    frames: List[EpisodeFrame]


def _read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        logger.warning("Episode manifest not found: %s", path)
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skip malformed manifest line in %s: %s", path, exc)
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _sample_indices(length: int, frames_per_episode: int) -> List[int]:
    if length <= 0:
        return []
    if frames_per_episode <= 0 or length <= frames_per_episode:
        return list(range(length))
    if frames_per_episode == 1:
        return [length // 2]
    return sorted(
        {
            round(i * (length - 1) / (frames_per_episode - 1))
            for i in range(frames_per_episode)
        }
    )


def _source_episode_key(source: str, row: Dict[str, Any]) -> str:
    if source == "droid_100":
        return str(row.get("episode_id") or "ep_unknown")
    if source == "fmb":
        return str(row.get("trajectory_id") or "traj_unknown")
    return str(row.get("episode_id") or row.get("trajectory_id") or "episode_unknown")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "episode_unknown"


def episode_task_context(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return instruction/primitive context fields available on a manifest row."""
    return {
        key: row.get(key)
        for key in TASK_CONTEXT_FIELDS
        if row.get(key) not in (None, "")
    }


def is_episode_capable_row(row: Dict[str, Any]) -> bool:
    """Whether a manifest row has the minimum metadata for dynamic labels."""
    has_episode = row.get("episode_id") not in (None, "") or row.get("trajectory_id") not in (None, "")
    has_step = row.get("step_index") not in (None, "")
    return bool(has_episode and has_step and episode_task_context(row))


def group_manifest_rows(
    source_dataset: str,
    rows: Iterable[Dict[str, Any]],
    *,
    frames_per_episode: int = 6,
    max_episodes: Optional[int] = None,
) -> List[EpisodeBatch]:
    """Group extracted sampled-frame manifest rows into ordered episode batches."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        image_path = str(row.get("image_path") or "")
        if not image_path:
            continue
        grouped.setdefault(_source_episode_key(source_dataset, row), []).append(row)

    episodes: List[EpisodeBatch] = []
    for episode_id in sorted(grouped):
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
        ordered = sorted(grouped[episode_id], key=lambda r: _as_int(r.get("step_index")))
        selected = [ordered[i] for i in _sample_indices(len(ordered), frames_per_episode)]
        frames = [
            EpisodeFrame(
                source_dataset=source_dataset,
                episode_id=episode_id,
                frame_index=idx,
                step_index=_as_int(row.get("step_index")),
                image_path=str(row.get("image_path") or ""),
                metadata={k: v for k, v in row.items() if k != "image_path"},
            )
            for idx, row in enumerate(selected)
        ]
        if frames:
            episodes.append(EpisodeBatch(source_dataset=source_dataset, episode_id=episode_id, frames=frames))
    return episodes


def _dynamic_skip_reason(batch: EpisodeBatch, min_frames: int) -> Optional[str]:
    if len(batch.frames) < min_frames:
        return "too_few_frames"
    if not any(frame.metadata.get("step_index") not in (None, "") for frame in batch.frames):
        return "missing_step_index"
    if not any(episode_task_context(frame.metadata) for frame in batch.frames):
        return "missing_task_context"
    return None


def filter_dynamic_capable_batches(
    batches: Iterable[EpisodeBatch],
    min_frames: int = 3,
) -> Tuple[List[EpisodeBatch], List[Dict[str, Any]]]:
    """Filter episode batches for optional dynamic labels and record skip reasons."""
    capable: List[EpisodeBatch] = []
    skipped: List[Dict[str, Any]] = []
    for batch in batches:
        reason = _dynamic_skip_reason(batch, min_frames)
        if reason:
            skipped.append({
                "source_dataset": batch.source_dataset,
                "episode_id": batch.episode_id,
                "skip_reason": reason,
                "frame_count": len(batch.frames),
            })
        else:
            capable.append(batch)
    return capable, skipped


def _manifest_path(source: str, override: Dict[str, Any]) -> Optional[Path]:
    manifest = override.get("manifest") or override.get("manifest_path")
    if not manifest:
        if source == "droid_100":
            manifest = "droid_100_robot_state_manifest.jsonl"
        elif source == "fmb":
            manifest = "fmb_robot_state_manifest.jsonl"
        else:
            return None
    manifest_path = Path(manifest)
    if manifest_path.is_absolute():
        return manifest_path

    root = override.get("root") or override.get("root_override") or override.get("extract_root")
    if not root:
        if source == "droid_100":
            root = DEFAULT_DROID_EXTRACT_ROOT
        elif source == "fmb":
            root = DEFAULT_FMB_EXTRACT_ROOT
        else:
            return None
    return Path(root) / manifest_path


def discover_episode_batches(
    sources: Iterable[str],
    overrides: Dict[str, Dict[str, Any]],
    *,
    frames_per_episode: int = 6,
    max_episodes_per_source: Optional[int] = None,
) -> List[EpisodeBatch]:
    """Load extracted episode manifests and return sampled episode batches."""
    batches: List[EpisodeBatch] = []
    for source in sources:
        override = dict(overrides.get(source, {}))
        if source == "droid_100" and not any(k in override for k in ("root", "root_override", "extract_root")):
            override["root"] = DEFAULT_DROID_100_ROOT
        if source == "fmb" and not any(k in override for k in ("root", "root_override", "extract_root")):
            override["root"] = DEFAULT_FMB_ROOT
        manifest_path = _manifest_path(source, override)
        if manifest_path is None:
            logger.warning("Episode robot-state source has no manifest/root configured: %s", source)
            continue
        rows = _read_jsonl(manifest_path)
        batches.extend(
            group_manifest_rows(
                source,
                rows,
                frames_per_episode=frames_per_episode,
                max_episodes=max_episodes_per_source,
            )
        )
    return batches


def _parse_json(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}


def _normalize_value(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else "unknown"
    if text in STATE_VALUES:
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
    return max(0.0, min(1.0, conf))


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text or text.lower() in {"null", "none"}:
        return None
    return text


def _frame_payloads(parsed: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    frames = parsed.get("frames")
    if not isinstance(frames, list):
        return {}
    payloads: Dict[int, Dict[str, Any]] = {}
    for item in frames:
        if not isinstance(item, dict):
            continue
        payloads[_as_int(item.get("frame_index"), -1)] = item
    return payloads


def _state_payload(frame_payload: Dict[str, Any], state_type: str) -> Dict[str, Any]:
    states = frame_payload.get("states")
    if isinstance(states, dict):
        payload = states.get(state_type, {})
        if isinstance(payload, dict):
            return payload
        return {"value": payload}
    return {}


def _evidence_weak(evidence: Optional[str]) -> bool:
    if evidence is None:
        return True
    text = evidence.strip().lower()
    return len(text) < 12 or text in GENERIC_EVIDENCE


class ARobotStateEpisodePipeline:
    """Runs multi-frame VLM robot-state inference over grouped episodes."""

    def __init__(self, vlm_client=None, auto_pass_confidence: float = 0.8):
        self.vlm_client = vlm_client
        self.auto_pass_confidence = auto_pass_confidence

    def run_from_config(
        self,
        episode_cfg: Dict[str, Any],
        dataset_overrides: Dict[str, Dict[str, Any]],
        *,
        include_dynamic: bool = False,
    ):
        if not episode_cfg.get("enabled", False):
            return ([], [], []) if include_dynamic else ([], [])
        sources = episode_cfg.get("sources") or ["droid_100", "fmb"]
        frames_per_episode = _as_int(episode_cfg.get("frames_per_episode"), 6)
        raw_max = episode_cfg.get("max_episodes_per_source")
        max_episodes = None if raw_max in (None, "", 0, "0") else _as_int(raw_max)
        batches = discover_episode_batches(
            sources,
            dataset_overrides,
            frames_per_episode=frames_per_episode,
            max_episodes_per_source=max_episodes,
        )
        labels: List[ARobotStateLabel] = []
        summaries: List[Dict[str, Any]] = []
        dynamic_labels: List[Dict[str, Any]] = []
        dynamic_sources = set(episode_cfg.get("dynamic_sources") or sources)
        min_dynamic_frames = _as_int(episode_cfg.get("min_frames_per_dynamic_episode"), 3)

        for batch in batches:
            dynamic_requested = include_dynamic and batch.source_dataset in dynamic_sources
            dynamic_capable = False
            if dynamic_requested:
                capable, skipped = filter_dynamic_capable_batches([batch], min_dynamic_frames)
                dynamic_capable = bool(capable)
                for item in skipped:
                    logger.info(
                        "Skip dynamic robot-state label: source=%s episode=%s reason=%s frames=%s",
                        item["source_dataset"], item["episode_id"], item["skip_reason"], item["frame_count"],
                    )
            result = self.run_episode(batch, include_dynamic=dynamic_capable, dynamic_cfg=episode_cfg)
            if dynamic_capable:
                batch_labels, summary, dynamic = result
                if dynamic:
                    dynamic_labels.append(dynamic)
            else:
                batch_labels, summary = result
            labels.extend(batch_labels)
            summaries.append(summary)
        return (labels, summaries, dynamic_labels) if include_dynamic else (labels, summaries)

    def run_episode(
        self,
        batch: EpisodeBatch,
        *,
        include_dynamic: bool = False,
        dynamic_cfg: Optional[Dict[str, Any]] = None,
    ):
        if self.vlm_client is None:
            logger.warning("No VLM client configured; skipping episode robot-state inference.")
            labels: List[ARobotStateLabel] = []
            summary = self._fallback_summary(batch, ["vlm_client_missing"])
            if include_dynamic:
                return labels, summary, self._dynamic_from_payload(batch, {}, dynamic_cfg or {}, ["vlm_client_missing"])
            return labels, summary

        prompt = self._build_prompt(batch)
        try:
            raw = self.vlm_client.query(
                image_paths=[frame.image_path for frame in batch.frames],
                prompt=prompt,
                temperature=0.2,
            )
            parsed = _parse_json(raw)
        except Exception:
            logger.exception("Episode VLM call failed for %s:%s", batch.source_dataset, batch.episode_id)
            labels = self._labels_from_payload(batch, {}, ["episode_vlm_failed"])
            summary = self._fallback_summary(batch, ["episode_vlm_failed"])
            if include_dynamic:
                return labels, summary, self._dynamic_from_payload(batch, {}, dynamic_cfg or {}, ["episode_vlm_failed"])
            return labels, summary

        labels = self._labels_from_payload(batch, parsed, [])
        summary = self._summary_from_payload(batch, parsed)
        if include_dynamic:
            return labels, summary, self._dynamic_from_payload(batch, parsed, dynamic_cfg or {}, [])
        return labels, summary

    def _build_prompt(self, batch: EpisodeBatch) -> str:
        frame_meta = [
            {
                "frame_index": frame.frame_index,
                "step_index": frame.step_index,
                "image_path": frame.image_path,
                "metadata": {
                    key: frame.metadata.get(key)
                    for key in (
                        "language_instruction",
                        "language_instruction_2",
                        "language_instruction_3",
                        "primitive",
                        "gripper_position",
                        "action_gripper_position",
                        "action_gripper",
                        "tcp_force_norm",
                        "tcp_torque_norm",
                    )
                    if frame.metadata.get(key) not in (None, "")
                },
            }
            for frame in batch.frames
        ]
        context = {
            "source_dataset": batch.source_dataset,
            "episode_id": batch.episode_id,
            "sampled_frames": frame_meta,
        }
        return EPISODE_PROMPT + "\nEpisode metadata:\n" + json.dumps(context, ensure_ascii=False)

    def _labels_from_payload(
        self,
        batch: EpisodeBatch,
        parsed: Dict[str, Any],
        base_flags: List[str],
    ) -> List[ARobotStateLabel]:
        payloads = _frame_payloads(parsed)
        labels: List[ARobotStateLabel] = []
        for frame in batch.frames:
            frame_payload = payloads.get(frame.frame_index)
            frame_flags = list(base_flags)
            if frame_payload is None:
                frame_payload = {}
                frame_flags.append("missing_episode_frame")
            for state_type in STATE_TYPES:
                label = self._build_label(batch, frame, frame_payload, state_type, frame_flags)
                labels.append(label)
        self._apply_cross_state_quality(labels)
        self._finalize(labels)
        return labels

    def _build_label(
        self,
        batch: EpisodeBatch,
        frame: EpisodeFrame,
        frame_payload: Dict[str, Any],
        state_type: str,
        inherited_flags: List[str],
    ) -> ARobotStateLabel:
        payload = _state_payload(frame_payload, state_type)
        flags = list(inherited_flags)
        if not payload:
            flags.append("missing_episode_state")
        state_value = _normalize_value(payload.get("value", payload.get("state_value")))
        confidence = _normalize_confidence(payload.get("confidence"))
        target = _normalize_text(payload.get("target_object_class"))
        evidence = _normalize_text(payload.get("visual_evidence", payload.get("evidence")))

        if confidence is None:
            flags.append("missing_confidence")
        elif confidence < self.auto_pass_confidence:
            flags.append("low_confidence")
        if state_value == "unknown":
            flags.append("unknown_state_value")
        if state_value in {"yes", "no"} and _evidence_weak(evidence):
            flags.append("weak_visual_evidence")
        if state_type == ARobotStateSubtype.HOLDING_OBJECT.value and state_value == "yes" and not target:
            flags.append("holding_object_missing_target")

        safe_episode = _safe_id(batch.episode_id)
        task_instruction = self._task_instruction(batch)
        task_type = self._normalize_task_type(None, task_instruction)
        label = ARobotStateLabel(
            sample_id=f"{batch.source_dataset}_episode_{safe_episode}_step{frame.step_index:05d}_{state_type}",
            object_id=f"gripper_{batch.source_dataset}_{safe_episode}_step{frame.step_index:05d}",
            camera_view=CameraView.WRIST.value,
            end_effector=EndEffector.PARALLEL_JAW.value,
            state_type=state_type,
            subtype=state_type,
            state_value=state_value,
            target_object_class=target,
            visual_evidence=evidence,
            confidence=confidence,
            image_path=frame.image_path,
            source_dataset=batch.source_dataset,
            episode_id=batch.episode_id,
            step_index=frame.step_index,
            episode_phase=_normalize_text(frame_payload.get("phase")),
            task_instruction=task_instruction,
            task_type=task_type,
            task_target_object_class=target,
            instruction_role="prior_only" if task_instruction else None,
            source_provenance=SourceProvenance.VLM.value,
            label_status="teacher_generated_episode",
            stable_training_label=False,
            quality_flags=flags,
        )
        validation = validate_robot_state(label)
        if not validation.valid:
            label.quality_flags.extend(f"schema_invalid:{err}" for err in validation.errors)
        return label

    def _apply_cross_state_quality(self, labels: List[ARobotStateLabel]) -> None:
        by_frame: Dict[str, Dict[str, ARobotStateLabel]] = {}
        for label in labels:
            by_frame.setdefault(label.image_path or label.sample_id, {})[label.state_type] = label
        for states in by_frame.values():
            empty = states.get(ARobotStateSubtype.GRIPPER_EMPTY.value)
            if not empty or empty.state_value != "yes":
                continue
            conflicts = [
                states.get(ARobotStateSubtype.HOLDING_OBJECT.value),
                states.get(ARobotStateSubtype.OBJECT_BETWEEN_FINGERS.value),
            ]
            active = [label for label in conflicts if label is not None and label.state_value == "yes"]
            if not active:
                continue
            for label in [empty, *active]:
                label.quality_flags.append("mutually_exclusive_state_conflict")

    def _finalize(self, labels: List[ARobotStateLabel]) -> None:
        for label in labels:
            label.quality_flags = list(dict.fromkeys(label.quality_flags))
            label.verifier_status = (
                VerifierStatus.REVIEW_REQUIRED.value
                if label.quality_flags else VerifierStatus.AUTO_VERIFIABLE.value
            )

    def _dynamic_from_payload(
        self,
        batch: EpisodeBatch,
        parsed: Dict[str, Any],
        cfg: Dict[str, Any],
        base_flags: List[str],
    ) -> Dict[str, Any]:
        dynamic = parsed.get("dynamic_annotation") if isinstance(parsed, dict) else None
        flags = list(base_flags)
        if not isinstance(dynamic, dict):
            dynamic = {}
            flags.append("missing_dynamic_annotation")

        task_parse = self._normalize_task_parse(batch, dynamic.get("task_parse"))
        sampled_frames = self._normalize_dynamic_frames(batch, dynamic.get("sampled_frames"))
        phase_segments = self._normalize_phase_segments(dynamic.get("phase_segments"))
        transitions = self._normalize_transitions(dynamic.get("state_transitions"))
        outcome = self._normalize_dynamic_outcome(dynamic.get("episode_outcome"), parsed)

        min_conf = _normalize_confidence(cfg.get("dynamic_min_confidence"))
        if min_conf is None:
            min_conf = self.auto_pass_confidence
        self._finalize_dynamic_quality(
            flags, task_parse, phase_segments, transitions, outcome, parsed, min_conf
        )
        flags = list(dict.fromkeys(flags))
        return {
            "bucket": "A_robot_state_dynamic",
            "label_version": str(cfg.get("dynamic_label_version") or "v1"),
            "source_dataset": batch.source_dataset,
            "episode_id": batch.episode_id,
            "task_parse": task_parse,
            "sampled_frames": sampled_frames,
            "phase_segments": phase_segments,
            "state_transitions": transitions,
            "episode_outcome": outcome,
            "quality": {
                "verifier_status": (
                    VerifierStatus.REVIEW_REQUIRED.value
                    if flags else VerifierStatus.AUTO_VERIFIABLE.value
                ),
                "quality_flags": flags,
                "skip_reason": None,
            },
        }

    def _normalize_task_parse(self, batch: EpisodeBatch, payload: Any) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        instruction = _normalize_text(payload.get("instruction")) or self._task_instruction(batch)
        task_type = self._normalize_task_type(payload.get("task_type"), instruction)
        expected = payload.get("expected_events")
        if not isinstance(expected, list):
            expected = self._expected_events(task_type)
        else:
            expected = [str(item) for item in expected if item not in (None, "")]
        return {
            "instruction": instruction,
            "task_type": task_type,
            "target_object_class": _normalize_text(payload.get("target_object_class")),
            "destination": _normalize_text(payload.get("destination")),
            "instruction_role": "prior_only",
            "expected_events": expected,
        }

    def _normalize_task_type(self, value: Any, instruction: Optional[str]) -> str:
        text = str(value or "").strip().lower()
        if text in TASK_TYPES:
            return text
        inst = (instruction or "").lower()
        for task_type in ("pick", "place", "push", "insert", "open", "close", "move"):
            if task_type in inst:
                return task_type
        if any(token in inst for token in ("grasp", "grab", "lift")):
            return "pick"
        if any(token in inst for token in ("put", "set")):
            return "place"
        return "unknown"

    def _expected_events(self, task_type: str) -> List[str]:
        if task_type == "pick":
            return ["approach_target", "contact_target", "close_gripper", "lift_or_move_target"]
        if task_type == "place":
            return ["transport_object", "approach_destination", "release_object"]
        if task_type == "push":
            return ["approach_target", "contact_target", "move_target"]
        if task_type == "insert":
            return ["approach_target", "contact_target", "insert_target"]
        return []

    def _normalize_phase(self, value: Any) -> str:
        phase = str(value or "unknown").strip().lower()
        return phase if phase in PHASES else "unknown"

    def _normalize_transition_type(self, value: Any) -> str:
        transition = str(value or "unknown").strip().lower()
        return transition if transition in TRANSITION_TYPES else "unknown"

    def _normalize_dynamic_frames(self, batch: EpisodeBatch, payload: Any) -> List[Dict[str, Any]]:
        by_index: Dict[int, Dict[str, Any]] = {}
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    by_index[_as_int(item.get("frame_index"), -1)] = item
        frames: List[Dict[str, Any]] = []
        for frame in batch.frames:
            item = by_index.get(frame.frame_index, {})
            frames.append({
                "frame_index": frame.frame_index,
                "step_index": frame.step_index,
                "image_path": frame.image_path,
                "phase": self._normalize_phase(item.get("phase")),
                "phase_confidence": _normalize_confidence(item.get("phase_confidence")),
            })
        return frames

    def _normalize_phase_segments(self, payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            out.append({
                "phase": self._normalize_phase(item.get("phase")),
                "start_step": _as_int(item.get("start_step")),
                "end_step": _as_int(item.get("end_step")),
                "state_summary": _normalize_text(item.get("state_summary")),
                "evidence": _normalize_text(item.get("evidence")),
                "confidence": _normalize_confidence(item.get("confidence")),
            })
        return out

    def _normalize_transitions(self, payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            before = item.get("before_state") if isinstance(item.get("before_state"), dict) else {}
            after = item.get("after_state") if isinstance(item.get("after_state"), dict) else {}
            out.append({
                "transition_type": self._normalize_transition_type(item.get("transition_type")),
                "from_step": _as_int(item.get("from_step")),
                "to_step": _as_int(item.get("to_step")),
                "before_state": before,
                "after_state": after,
                "target_object_class": _normalize_text(item.get("target_object_class")),
                "evidence": _normalize_text(item.get("evidence")),
                "confidence": _normalize_confidence(item.get("confidence")),
            })
        return out

    def _normalize_dynamic_outcome(self, payload: Any, parsed: Dict[str, Any]) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        summary = parsed.get("episode_summary") if isinstance(parsed.get("episode_summary"), dict) else {}
        value = str(payload.get("value") or summary.get("episode_outcome") or "unknown").strip().lower()
        if value not in {"success", "failure", "partial", "unknown"}:
            value = "unknown"
        return {
            "value": value,
            "evidence": _normalize_text(payload.get("evidence") or summary.get("summary_evidence")),
            "confidence": _normalize_confidence(payload.get("confidence", summary.get("confidence"))),
        }

    def _frame_has_state(self, parsed: Dict[str, Any], state_type: str, value: str = "yes") -> bool:
        for frame_payload in _frame_payloads(parsed).values():
            payload = _state_payload(frame_payload, state_type)
            if _normalize_value(payload.get("value", payload.get("state_value"))) == value:
                return True
        return False

    def _finalize_dynamic_quality(
        self,
        flags: List[str],
        task_parse: Dict[str, Any],
        phase_segments: List[Dict[str, Any]],
        transitions: List[Dict[str, Any]],
        outcome: Dict[str, Any],
        parsed: Dict[str, Any],
        min_confidence: float,
    ) -> None:
        task_type = task_parse.get("task_type", "unknown")
        transition_types = {item.get("transition_type") for item in transitions}
        if not task_parse.get("instruction"):
            flags.append("missing_task_instruction")
        if task_type == "unknown":
            flags.append("unknown_task_type")
        if not phase_segments:
            flags.append("missing_phase_segments")
        if task_type in TASKS_EXPECTING_TRANSITIONS and not transitions:
            flags.append("missing_expected_transition")
        for segment in phase_segments:
            conf = segment.get("confidence")
            if conf is None or conf < min_confidence:
                flags.append("low_phase_confidence")
            if _evidence_weak(segment.get("evidence")):
                flags.append("weak_phase_evidence")
        for transition in transitions:
            conf = transition.get("confidence")
            if conf is None or conf < min_confidence:
                flags.append("low_transition_confidence")
            if _evidence_weak(transition.get("evidence")):
                flags.append("weak_transition_evidence")
        outcome_conf = outcome.get("confidence")
        if outcome_conf is None or outcome_conf < min_confidence:
            flags.append("low_outcome_confidence")
        if _evidence_weak(outcome.get("evidence")):
            flags.append("weak_outcome_evidence")

        if "grasp_acquired" in transition_types and not self._frame_has_state(parsed, ARobotStateSubtype.HOLDING_OBJECT.value):
            flags.append("transition_frame_conflict")
        if "contact_started" in transition_types and not self._frame_has_state(parsed, ARobotStateSubtype.CONTACT_STATE.value):
            flags.append("transition_frame_conflict")
        if task_type == "push" and "grasp_acquired" in transition_types:
            flags.append("instruction_transition_conflict")
        if task_type == "pick" and outcome.get("value") == "success" and not self._frame_has_state(parsed, ARobotStateSubtype.HOLDING_OBJECT.value):
            flags.append("instruction_transition_conflict")
        phases = {segment.get("phase") for segment in phase_segments}
        if task_type == "place" and outcome.get("value") == "success":
            if "object_released" not in transition_types and "release" not in phases:
                flags.append("instruction_transition_conflict")

    def _summary_from_payload(self, batch: EpisodeBatch, parsed: Dict[str, Any]) -> Dict[str, Any]:
        payload = parsed.get("episode_summary") if isinstance(parsed, dict) else None
        if not isinstance(payload, dict):
            return self._fallback_summary(batch, ["missing_episode_summary"])
        confidence = _normalize_confidence(payload.get("confidence"))
        flags: List[str] = []
        if confidence is None:
            flags.append("missing_confidence")
        elif confidence < self.auto_pass_confidence:
            flags.append("low_confidence")
        evidence = _normalize_text(payload.get("summary_evidence"))
        if _evidence_weak(evidence):
            flags.append("weak_summary_evidence")
        outcome = str(payload.get("episode_outcome") or "unknown").strip().lower()
        if outcome not in {"success", "failure", "unknown"}:
            outcome = "unknown"
            flags.append("invalid_episode_outcome")
        timeline = payload.get("state_timeline")
        if not isinstance(timeline, list):
            timeline = []
            flags.append("missing_state_timeline")
        return {
            "bucket": "A_robot_state_episode",
            "episode_id": batch.episode_id,
            "source_dataset": batch.source_dataset,
            "frame_count": len(batch.frames),
            "sampled_steps": [frame.step_index for frame in batch.frames],
            "task_instruction": self._task_instruction(batch),
            "state_timeline": timeline,
            "episode_outcome": outcome,
            "summary_evidence": evidence,
            "confidence": confidence,
            "quality_flags": list(dict.fromkeys(flags)),
        }

    def _fallback_summary(self, batch: EpisodeBatch, flags: List[str]) -> Dict[str, Any]:
        return {
            "bucket": "A_robot_state_episode",
            "episode_id": batch.episode_id,
            "source_dataset": batch.source_dataset,
            "frame_count": len(batch.frames),
            "sampled_steps": [frame.step_index for frame in batch.frames],
            "task_instruction": self._task_instruction(batch),
            "state_timeline": [],
            "episode_outcome": "unknown",
            "summary_evidence": None,
            "confidence": None,
            "quality_flags": list(dict.fromkeys(flags)),
        }

    def _task_instruction(self, batch: EpisodeBatch) -> Optional[str]:
        for frame in batch.frames:
            for key in ("language_instruction", "language_instruction_2", "language_instruction_3", "primitive"):
                value = _normalize_text(frame.metadata.get(key))
                if value:
                    return value
        return None
