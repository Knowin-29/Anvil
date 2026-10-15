"""DROID adapter — extracted wrist frames -> A_robot_state candidate labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..a_pipeline_schemas import ARobotStateLabel
from ..taxonomy import ARobotStateSubtype, CameraView, EndEffector, VerifierStatus
from .robot_state_ingest import RobotStateDatasetAdapter, finalize_robot_state_label

logger = logging.getLogger("gsa.data_adapters.droid_robot_state")


DEFAULT_DROID_EXTRACT_ROOT = (
    "/path/to/datasets/downloads/droid/droid_100_extracted"
)
DEFAULT_MANIFEST = "droid_100_robot_state_manifest.jsonl"


def _read_manifest(path: Path, max_samples: int) -> List[dict]:
    samples: List[dict] = []
    if not path.exists():
        logger.warning("DROID robot-state manifest not found: %s", path)
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
                logger.warning("Skip malformed DROID manifest line: %s", exc)
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


def _state_candidates(sample: dict) -> List[tuple[str, float]]:
    """Return weak robot-state candidates from DROID step metadata.

    Gripper numeric state is only a hint. It does not prove whether an object is
    present, so all values remain unknown and review_required.
    """
    candidates = [
        (ARobotStateSubtype.GRIPPER_EMPTY.value, 0.35),
        (ARobotStateSubtype.HOLDING_OBJECT.value, 0.30),
        (ARobotStateSubtype.OBJECT_BETWEEN_FINGERS.value, 0.30),
    ]

    gp = sample.get("gripper_position")
    agp = sample.get("action_gripper_position")
    try:
        numeric = [float(v) for v in (gp, agp) if v is not None]
    except (TypeError, ValueError):
        numeric = []
    if numeric:
        spread = max(numeric) - min(numeric)
        if spread > 0.05:
            candidates[2] = (ARobotStateSubtype.OBJECT_BETWEEN_FINGERS.value, 0.38)
    return candidates


def _visual_evidence(sample: dict) -> str:
    parts = []
    for key in (
        "language_instruction",
        "language_instruction_2",
        "language_instruction_3",
        "gripper_position",
        "action_gripper_position",
        "action_gripper_velocity",
        "reward",
        "is_last",
    ):
        value = sample.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return "; ".join(parts)


class DroidRobotStateAdapter(RobotStateDatasetAdapter):
    """RobotStateDatasetAdapter for DROID droid_100 extracted wrist frames."""

    name = "droid_100"

    def resolve_dataset_path(self, config: Dict[str, Any]) -> Path:
        if config.get("root"):
            return Path(config["root"])
        if config.get("root_override"):
            return Path(config["root_override"])
        if config.get("extract_root"):
            return Path(config["extract_root"])
        return Path(DEFAULT_DROID_EXTRACT_ROOT)

    def discover(self, dataset_path: Path, max_samples: int, config: Dict[str, Any]) -> List[dict]:
        return _read_manifest(_manifest_path(dataset_path, config), max_samples)

    def convert(self, samples: List[dict], dataset_path: Path, config: Dict[str, Any]) -> List[ARobotStateLabel]:
        labels: List[ARobotStateLabel] = []
        for sample in samples:
            image_path = str(sample.get("image_path") or "")
            image_exists = bool(image_path and Path(image_path).exists())
            episode_id = str(sample.get("episode_id") or "ep_unknown")
            step_index = int(sample.get("step_index") or 0)
            evidence = _visual_evidence(sample)

            for state_type, confidence in _state_candidates(sample):
                label = ARobotStateLabel(
                    sample_id=f"droid100_{episode_id}_step{step_index:05d}_{state_type}",
                    object_id=f"gripper_droid100_{episode_id}_step{step_index:05d}",
                    camera_view=CameraView.WRIST.value,
                    end_effector=EndEffector.PARALLEL_JAW.value,
                    state_type=state_type,
                    state_value="unknown",
                    confidence=confidence,
                    verifier_status=VerifierStatus.REVIEW_REQUIRED.value,
                    visual_evidence=evidence,
                    quality_flags=["weak_gripper_hint", "needs_vlm_or_human_review"],
                )
                finalized = finalize_robot_state_label(
                    label,
                    image_path=image_path,
                    source_dataset=self.name,
                    image_exists=image_exists,
                )
                finalized.verifier_status = VerifierStatus.REVIEW_REQUIRED.value
                labels.append(finalized)

        logger.info("Converted %d DROID manifest rows into %d labels", len(samples), len(labels))
        return labels
