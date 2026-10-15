"""ShareRobot adapter — planning + trajectory -> A_robot_state labels."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..a_pipeline_schemas import ARobotStateLabel
from ..taxonomy import ARobotStateSubtype, EndEffector
from .robot_state_ingest import RobotStateDatasetAdapter, finalize_robot_state_label
from .robot_state_text_rules import (
    infer_camera_view,
    infer_state_from_instruction,
    infer_state_from_text,
)

logger = logging.getLogger("gsa.data_adapters.sharerobot")

DEFAULT_PLANNING_TASKS = [
    "future_prediction_task",
    "past_description_task",
    "planning_task",
    "planning_with_context_task",
]


def _conversation_text(item: dict) -> str:
    parts: List[str] = []
    for turn in item.get("conversations", []):
        value = turn.get("value", "")
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _unique_suffix(raw_id: Any) -> str:
    return hashlib.sha1(str(raw_id).encode("utf-8")).hexdigest()[:12]


class ShareRobotRobotStateAdapter(RobotStateDatasetAdapter):
    name = "sharerobot"

    def discover(self, dataset_path: Path, max_samples: int, config: Dict[str, Any]) -> List[dict]:
        samples: List[dict] = []
        planning_dirs = config.get("planning_dirs", ["planning/train"])
        task_filter = config.get("planning_tasks_for_robot_state", DEFAULT_PLANNING_TASKS)

        task_files: List[Path] = []
        for rel_dir in planning_dirs:
            base = dataset_path / rel_dir
            if not base.exists():
                continue
            for json_path in sorted(base.glob("*.json")):
                if task_filter and json_path.stem not in task_filter:
                    continue
                task_files.append(json_path)

        per_file = max(1, max_samples // max(len(task_files), 1)) if task_files else 0
        for json_path in task_files:
            if len(samples) >= max_samples:
                break
            task_name = json_path.stem
            try:
                items = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skip planning file %s: %s", json_path, exc)
                continue
            if not isinstance(items, list):
                continue
            limit = min(per_file, max_samples - len(samples))
            for item in items[:limit]:
                enriched = dict(item)
                enriched["_source_kind"] = "planning"
                enriched["_task_file"] = task_name
                samples.append(enriched)

        traj_rel = config.get("trajectory_json", "trajectory/trajectory.json")
        traj_path = dataset_path / traj_rel
        if traj_path.exists() and len(samples) < max_samples:
            try:
                traj_items = json.loads(traj_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skip trajectory file %s: %s", traj_path, exc)
                traj_items = []
            if isinstance(traj_items, list):
                remaining = max_samples - len(samples)
                for item in traj_items[:remaining]:
                    enriched = dict(item)
                    enriched["_source_kind"] = "trajectory"
                    samples.append(enriched)

        return samples

    def convert(self, samples: List[dict], dataset_path: Path, config: Dict[str, Any]) -> List[ARobotStateLabel]:
        labels: List[ARobotStateLabel] = []
        images_root = Path(config.get("frame_images_root") or config.get("images_root", "images"))
        if not images_root.is_absolute():
            images_root = dataset_path / images_root

        for idx, sample in enumerate(samples):
            kind = sample.get("_source_kind", "planning")
            if kind == "trajectory":
                label = self._convert_trajectory(sample, idx, images_root)
            else:
                label = self._convert_planning(sample, idx, images_root)
            if label is not None:
                labels.append(label)
        return labels

    def _resolve_image(self, rel_path: str, images_root: Path) -> Tuple[Optional[str], bool]:
        if not rel_path:
            return None, False
        candidate = images_root / rel_path
        if candidate.exists():
            return str(candidate), True
        return str(candidate), False

    def _convert_planning(self, sample: dict, idx: int, images_root: Path) -> Optional[ARobotStateLabel]:
        text = _conversation_text(sample)
        mapped = infer_state_from_text(text)
        if mapped is None:
            return None

        state_type, state_value = mapped
        images = sample.get("image") or []
        step = sample.get("selected_step", 0)
        if not isinstance(step, int) or step < 0 or step >= len(images):
            step = 0 if images else -1
        rel_image = images[step] if step >= 0 else ""
        image_path, image_exists = self._resolve_image(str(rel_image), images_root)

        raw_id = sample.get("id", idx)
        task_file = sample.get("_task_file", "planning")
        sample_id = (
            f"sharerobot_{task_file}_{_unique_suffix(raw_id)}_f{step}"
        )

        label = ARobotStateLabel(
            sample_id=sample_id,
            object_id=f"gripper_{idx}",
            camera_view=infer_camera_view(str(rel_image), sample.get("meta_data")),
            end_effector=EndEffector.PARALLEL_JAW.value,
            state_type=state_type,
            state_value=state_value,
            confidence=0.85 if image_exists else 0.6,
        )
        return finalize_robot_state_label(
            label,
            image_path=image_path,
            source_dataset=self.name,
            image_exists=image_exists,
        )

    def _convert_trajectory(self, sample: dict, idx: int, images_root: Path) -> Optional[ARobotStateLabel]:
        instruction = sample.get("instruction", "")
        mapped = infer_state_from_instruction(instruction)
        if mapped is None:
            return None

        state_type, state_value = mapped
        rel_image = sample.get("image_path", "")
        image_path, image_exists = self._resolve_image(str(rel_image), images_root)
        raw_id = sample.get("id", idx)
        sample_id = f"sharerobot_traj_{raw_id}_{_unique_suffix(rel_image)}"

        traj = sample.get("trajectory") or sample.get("points") or []
        if state_type == ARobotStateSubtype.CONTACT_STATE.value and len(traj) >= 3:
            state_value = "yes"

        label = ARobotStateLabel(
            sample_id=sample_id,
            object_id=f"gripper_traj_{raw_id}",
            camera_view=infer_camera_view(str(rel_image), sample.get("meta_data")),
            end_effector=EndEffector.PARALLEL_JAW.value,
            state_type=state_type,
            state_value=state_value,
            confidence=0.8 if image_exists else 0.55,
        )
        return finalize_robot_state_label(
            label,
            image_path=image_path,
            source_dataset=self.name,
            image_exists=image_exists,
        )
