"""FMB adapter — extracted wrist frames -> A_robot_state candidate labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..a_pipeline_schemas import ARobotStateLabel
from ..taxonomy import ARobotStateSubtype, CameraView, EndEffector, VerifierStatus
from .robot_state_ingest import RobotStateDatasetAdapter, finalize_robot_state_label

logger = logging.getLogger("gsa.data_adapters.fmb_robot_state")

DEFAULT_FMB_EXTRACT_ROOT = (
    "/path/to/datasets/downloads/fmb/fmb_extracted"
)
DEFAULT_MANIFEST = "fmb_robot_state_manifest.jsonl"


def _read_manifest(path: Path, max_samples: int) -> List[dict]:
    samples: List[dict] = []
    if not path.exists():
        logger.warning("FMB robot-state manifest not found: %s", path)
        return samples
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skip malformed FMB manifest line: %s", exc)
                continue
            if isinstance(item, dict):
                samples.append(item)
    return samples


def _manifest_path(dataset_path: Path, config: Dict[str, Any]) -> Path:
    manifest = config.get("manifest") or config.get("manifest_path") or DEFAULT_MANIFEST
    manifest_path = Path(manifest)
    if manifest_path.is_absolute():
        return manifest_path
    extract_root = config.get("extract_root")
    if extract_root:
        return Path(extract_root) / manifest_path
    return dataset_path / manifest_path


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _state_candidates(sample: dict) -> List[tuple[str, float, str]]:
    gripper_pose = _as_float(sample.get("gripper_pose"))
    action_gripper = _as_float(sample.get("action_gripper"))
    force_norm = _as_float(sample.get("tcp_force_norm"))
    torque_norm = _as_float(sample.get("tcp_torque_norm"))
    primitive = str(sample.get("primitive") or "").lower()

    candidates = [
        (ARobotStateSubtype.GRIPPER_EMPTY.value, 0.35, "fmb_wrist_candidate"),
        (ARobotStateSubtype.HOLDING_OBJECT.value, 0.35, "fmb_wrist_candidate"),
        (ARobotStateSubtype.OBJECT_BETWEEN_FINGERS.value, 0.35, "fmb_wrist_candidate"),
        (ARobotStateSubtype.CONTACT_STATE.value, 0.35, "fmb_force_torque_candidate"),
        (ARobotStateSubtype.TOOL_READY.value, 0.25, "fmb_task_candidate"),
    ]

    if gripper_pose is not None:
        if gripper_pose >= 0.5:
            candidates[0] = (ARobotStateSubtype.GRIPPER_EMPTY.value, 0.42, "closed_gripper_hint")
            candidates[1] = (ARobotStateSubtype.HOLDING_OBJECT.value, 0.45, "closed_gripper_hint")
            candidates[2] = (ARobotStateSubtype.OBJECT_BETWEEN_FINGERS.value, 0.45, "closed_gripper_hint")
        else:
            candidates[0] = (ARobotStateSubtype.GRIPPER_EMPTY.value, 0.45, "open_gripper_hint")

    if action_gripper is not None and abs(action_gripper) > 0.1:
        candidates[2] = (ARobotStateSubtype.OBJECT_BETWEEN_FINGERS.value, 0.40, "gripper_action_hint")

    if (force_norm is not None and force_norm >= 5.0) or (torque_norm is not None and torque_norm >= 0.5):
        candidates[3] = (ARobotStateSubtype.CONTACT_STATE.value, 0.50, "force_torque_contact_hint")

    if any(token in primitive for token in ("insert", "grasp", "reorient", "move", "place")):
        candidates[4] = (ARobotStateSubtype.TOOL_READY.value, 0.35, "primitive_task_hint")

    return candidates


def _visual_evidence(sample: dict) -> str:
    parts = []
    for key in (
        "trajectory_id",
        "step_index",
        "primitive",
        "gripper_pose",
        "action_gripper",
        "tcp_force_norm",
        "tcp_torque_norm",
        "has_wrist_2",
    ):
        value = sample.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        task_family = metadata.get("task_family")
        object_id = metadata.get("object_id")
        if task_family:
            parts.append(f"task_family={task_family}")
        if object_id is not None:
            parts.append(f"object_id={object_id}")
    return "; ".join(parts)


class FmbRobotStateAdapter(RobotStateDatasetAdapter):
    """RobotStateDatasetAdapter for extracted FMB wrist frames."""

    name = "fmb"

    def resolve_dataset_path(self, config: Dict[str, Any]) -> Path:
        if config.get("root"):
            return Path(config["root"])
        if config.get("root_override"):
            return Path(config["root_override"])
        if config.get("extract_root"):
            return Path(config["extract_root"])
        return Path(DEFAULT_FMB_EXTRACT_ROOT)

    def discover(self, dataset_path: Path, max_samples: int, config: Dict[str, Any]) -> List[dict]:
        return _read_manifest(_manifest_path(dataset_path, config), max_samples)

    def convert(self, samples: List[dict], dataset_path: Path, config: Dict[str, Any]) -> List[ARobotStateLabel]:
        labels: List[ARobotStateLabel] = []
        for sample in samples:
            image_path = str(sample.get("image_path") or "")
            image_exists = bool(image_path and Path(image_path).exists())
            trajectory_id = str(sample.get("trajectory_id") or "traj_unknown").replace("/", "_")
            step_index = int(sample.get("step_index") or 0)
            evidence = _visual_evidence(sample)

            for state_type, confidence, hint in _state_candidates(sample):
                label = ARobotStateLabel(
                    sample_id=f"fmb_{trajectory_id}_step{step_index:05d}_{state_type}",
                    object_id=f"gripper_fmb_{trajectory_id}_step{step_index:05d}",
                    camera_view=CameraView.WRIST.value,
                    end_effector=EndEffector.PARALLEL_JAW.value,
                    state_type=state_type,
                    state_value="unknown",
                    confidence=confidence,
                    verifier_status=VerifierStatus.REVIEW_REQUIRED.value,
                    visual_evidence=evidence,
                    quality_flags=[hint, "needs_vlm_or_human_review"],
                )
                finalized = finalize_robot_state_label(
                    label,
                    image_path=image_path,
                    source_dataset=self.name,
                    image_exists=image_exists,
                )
                finalized.verifier_status = VerifierStatus.REVIEW_REQUIRED.value
                labels.append(finalized)

        logger.info("Converted %d FMB manifest rows into %d labels", len(samples), len(labels))
        return labels
