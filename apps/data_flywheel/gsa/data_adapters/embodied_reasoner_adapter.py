"""Embodied-Reasoner Adapter — trajectory JSON -> A_robot_state labels."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..a_pipeline_schemas import (
    AActionPreconditionLabel,
    ARobotStateLabel,
    validate_action_precondition,
    validate_robot_state,
)
from ..taxonomy import (
    ARobotStateSubtype,
    CameraView,
    EndEffector,
    FailureReason,
    PreAction,
    SourceProvenance,
    VerifierStatus,
)
from .robot_state_image_utils import resolve_er_image
from .robot_state_ingest import RobotStateDatasetAdapter, finalize_robot_state_label
from .robot_state_text_rules import infer_state_from_er_action

logger = logging.getLogger("gsa.data_adapters.embodied_reasoner")


def _unique_suffix(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _load_trajectories(dataset_path: Path, trajectory_json: str) -> List[Dict]:
    traj_path = dataset_path / trajectory_json
    if not traj_path.exists():
        logger.warning("Embodied-Reasoner trajectory JSON not found: %s", traj_path)
        return []
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load trajectory JSON %s: %s", traj_path, exc)
        return []
    if not isinstance(data, list):
        logger.warning("Unexpected trajectory JSON type: %s", type(data))
        return []
    return data


def discover_er_step_samples(
    dataset_path: Path,
    max_samples: int,
    config: Dict[str, Any],
) -> List[dict]:
    """Expand trajectories into per-action step samples."""
    trajectory_json = config.get(
        "trajectory_json", "data_engine_origin_trajectory_9390.json"
    )
    trajectories = _load_trajectories(dataset_path, trajectory_json)
    if not trajectories:
        return []

    samples: List[dict] = []
    for traj in trajectories:
        if len(samples) >= max_samples:
            break
        traj_id = traj.get("id")
        key_actions = traj.get("key_actions") or []
        images = traj.get("images") or []
        for action_idx, action in enumerate(key_actions):
            if len(samples) >= max_samples:
                break
            samples.append(
                {
                    "_traj_id": traj_id,
                    "_action_idx": action_idx,
                    "_action": action,
                    "_image_rel": images[action_idx] if action_idx < len(images) else "",
                    "_scene": traj.get("scene"),
                    "_tasktype": traj.get("tasktype"),
                }
            )
    return samples


def convert_to_precondition_labels(trajs: List[Dict]) -> List[AActionPreconditionLabel]:
    """Extract action precondition labels from trajectory reasoning."""
    labels: List[AActionPreconditionLabel] = []
    amap = {"pick": "pick", "push": "push", "place": "place",
            "grasp": "pick", "put": "place"}
    fmap = {"blocked": "occluded", "occluded": "occluded",
            "unreachable": "unreachable"}
    for i, traj in enumerate(trajs):
        action = traj.get("action", {})
        mapped = amap.get(action.get("type", "pick"), "pick")
        success = traj.get("success", traj.get("feasible", True))
        failure = traj.get("failure_reason", "")
        fr = fmap.get(failure, FailureReason.UNKNOWN.value) if not success else None
        label = AActionPreconditionLabel(
            sample_id=f"embodied_reasoner_precond_{i}",
            object_id=action.get("target", f"obj_{i}"),
            action_type=mapped,
            can_do_now=bool(success),
            failure_reason=fr,
            need_pre_action=PreAction.NONE.value,
            source_provenance=SourceProvenance.PUBLIC_DATASET.value,
            verifier_status=VerifierStatus.AUTO_VERIFIABLE.value,
            s_context=True,
        )
        if validate_action_precondition(label).valid:
            labels.append(label)
    logger.info("Converted %d -> A_action_precondition", len(labels))
    return labels


class EmbodiedReasonerRobotStateAdapter(RobotStateDatasetAdapter):
    """RobotStateDatasetAdapter for Embodied-Reasoner key_actions trajectories."""

    name = "embodied_reasoner"

    def discover(self, dataset_path: Path, max_samples: int, config: Dict[str, Any]) -> List[dict]:
        return discover_er_step_samples(dataset_path, max_samples, config)

    def convert(self, samples: List[dict], dataset_path: Path, config: Dict[str, Any]) -> List[ARobotStateLabel]:
        labels: List[ARobotStateLabel] = []
        cache_dir = config.get("image_cache_dir")
        cache_path = Path(cache_dir) if cache_dir else dataset_path / ".image_cache"

        for sample in samples:
            action = sample.get("_action") or {}
            action_name = str(action.get("action", ""))
            mapped = infer_state_from_er_action(action_name)
            if mapped is None:
                continue

            state_type, state_value = mapped
            rel_image = str(sample.get("_image_rel") or "")
            image_path, image_exists = resolve_er_image(rel_image, dataset_path, cache_path)

            traj_id = sample.get("_traj_id", 0)
            action_idx = sample.get("_action_idx", 0)
            object_type = action.get("objectType") or None
            sample_id = (
                f"er_{traj_id}_{action_idx}_{_unique_suffix(action_name, rel_image)}"
            )

            label = ARobotStateLabel(
                sample_id=sample_id,
                object_id=f"gripper_er_{traj_id}_{action_idx}",
                camera_view=CameraView.EXTERNAL.value,
                end_effector=EndEffector.PARALLEL_JAW.value,
                state_type=state_type,
                state_value=state_value,
                target_object_class=object_type if object_type else None,
                confidence=0.9 if image_exists else 0.65,
            )
            finalized = finalize_robot_state_label(
                label,
                image_path=image_path,
                source_dataset=self.name,
                image_exists=image_exists,
            )
            if validate_robot_state(finalized).valid:
                labels.append(finalized)

        logger.info("Converted %d Embodied-Reasoner step labels", len(labels))
        return labels
