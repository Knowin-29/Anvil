"""BridgeDataV2 adapter — wrist images -> A_robot_state candidate labels."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from ..a_pipeline_schemas import ARobotStateLabel
from ..taxonomy import ARobotStateSubtype, CameraView, EndEffector, VerifierStatus
from .robot_state_image_utils import DEFAULT_BRIDGE_IMAGES_ROOT
from .robot_state_ingest import RobotStateDatasetAdapter, finalize_robot_state_label

logger = logging.getLogger("gsa.data_adapters.bridge_robot_state")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def resolve_bridge_images_root(config: Dict[str, Any], dataset_path: Path) -> Path:
    if config.get("images_root"):
        root = Path(config["images_root"])
        if not root.is_absolute():
            root = dataset_path / root
        return root
    env_root = os.environ.get("BRIDGE_IMAGES_ROOT")
    if env_root:
        return Path(env_root)
    return Path(DEFAULT_BRIDGE_IMAGES_ROOT)


class BridgeRobotStateAdapter(RobotStateDatasetAdapter):
    name = "bridge_data_v2"

    def discover(self, dataset_path: Path, max_samples: int, config: Dict[str, Any]) -> List[dict]:
        images_root = resolve_bridge_images_root(config, dataset_path)
        if not images_root.exists():
            logger.warning("Bridge images root not found: %s", images_root)
            return []

        samples: List[dict] = []
        for image_path in sorted(images_root.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            samples.append({"image_path": str(image_path), "_stem": image_path.stem})
            if len(samples) >= max_samples:
                break
        return samples

    def convert(self, samples: List[dict], dataset_path: Path, config: Dict[str, Any]) -> List[ARobotStateLabel]:
        labels: List[ARobotStateLabel] = []
        for idx, sample in enumerate(samples):
            image_path = str(sample.get("image_path") or "")
            stem = str(sample.get("_stem") or Path(image_path).stem or idx)
            image_exists = bool(image_path and Path(image_path).exists())

            label = ARobotStateLabel(
                sample_id=f"bridge_{stem}",
                object_id=f"gripper_bridge_{idx}",
                camera_view=CameraView.WRIST.value,
                end_effector=EndEffector.PARALLEL_JAW.value,
                state_type=ARobotStateSubtype.GRIPPER_EMPTY.value,
                state_value="unknown",
                confidence=0.3,
                verifier_status=VerifierStatus.REVIEW_REQUIRED.value,
            )
            finalized = finalize_robot_state_label(
                label,
                image_path=image_path,
                source_dataset=self.name,
                image_exists=image_exists,
            )
            finalized.verifier_status = VerifierStatus.REVIEW_REQUIRED.value
            labels.append(finalized)
        return labels
