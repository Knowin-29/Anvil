"""Pluggable ingest registry for A_robot_state structured datasets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..a_pipeline_schemas import ARobotStateLabel, validate_robot_state
from ..taxonomy import SourceProvenance, VerifierStatus

logger = logging.getLogger("gsa.data_adapters.robot_state_ingest")

ROBOT_STATE_VALUES = frozenset({"yes", "no", "unknown"})
DEFAULT_SHARE_ROBOT_ROOT = (
    "/path/to/datasets/downloads/sharerobot"
)
DEFAULT_EMBODIED_REASONER_ROOT = (
    "/path/to/datasets/downloads/embodied_reasoner"
)
DEFAULT_ROBO2VLM_ROOT = (
    "/path/to/datasets/downloads/robo2vlm"
)
DEFAULT_BRIDGE_ROOT = (
    "/path/to/datasets/downloads/bridge_data_v2"
)
DEFAULT_DROID_100_ROOT = (
    "/path/to/datasets/downloads/droid/droid_100_extracted"
)
DEFAULT_FMB_ROOT = (
    "/path/to/datasets/downloads/fmb/fmb_extracted"
)


class RobotStateDatasetAdapter(ABC):
    """Unified interface for A_robot_state dataset sources."""

    name: str = ""
    pipeline: str = "A_robot_state"

    @abstractmethod
    def discover(self, dataset_path: Path, max_samples: int, config: Dict[str, Any]) -> List[dict]:
        ...

    @abstractmethod
    def convert(self, samples: List[dict], dataset_path: Path, config: Dict[str, Any]) -> List[ARobotStateLabel]:
        ...

    def resolve_dataset_path(self, config: Dict[str, Any]) -> Path:
        if config.get("root"):
            return Path(config["root"])
        if config.get("root_override"):
            return Path(config["root_override"])
        return Path(config.get("dir", "."))


def finalize_robot_state_label(
    label: ARobotStateLabel,
    *,
    image_path: Optional[str],
    source_dataset: str,
    image_exists: bool,
) -> ARobotStateLabel:
    """Normalize fields shared by all robot-state adapters."""
    label.subtype = label.state_type
    label.source_provenance = SourceProvenance.PUBLIC_DATASET.value
    if image_path:
        label.image_path = image_path
    label.source_dataset = source_dataset
    if not image_exists:
        label.verifier_status = VerifierStatus.REVIEW_REQUIRED.value
    return label


class RobotStateIngestRegistry:
    """Register adapters and ingest enabled datasets."""

    def __init__(self) -> None:
        self._adapters: Dict[str, RobotStateDatasetAdapter] = {}

    def register(self, adapter: RobotStateDatasetAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> Optional[RobotStateDatasetAdapter]:
        return self._adapters.get(name)

    def ingest_one(
        self,
        name: str,
        config: Dict[str, Any],
        max_samples: int,
    ) -> List[ARobotStateLabel]:
        adapter = self._adapters.get(name)
        if adapter is None:
            logger.warning("No robot state adapter registered for %s", name)
            return []

        dataset_path = adapter.resolve_dataset_path(config)
        if not dataset_path.exists():
            logger.warning("Dataset path not found for %s: %s", name, dataset_path)
            return []

        samples = adapter.discover(dataset_path, max_samples, config)
        labels = adapter.convert(samples, dataset_path, config)
        valid: List[ARobotStateLabel] = []
        for label in labels:
            result = validate_robot_state(label)
            if result.valid:
                valid.append(label)
            else:
                logger.debug("Dropped invalid label %s: %s", label.sample_id, result.errors)
        logger.info(
            "Ingested %s: discovered=%d converted=%d valid=%d path=%s",
            name,
            len(samples),
            len(labels),
            len(valid),
            dataset_path,
        )
        return valid

    def ingest_all(
        self,
        enabled: List[str],
        overrides: Dict[str, Dict[str, Any]],
        max_per_source: int,
    ) -> List[ARobotStateLabel]:
        all_labels: List[ARobotStateLabel] = []
        seen_ids: set[str] = set()
        stats: Dict[str, Dict[str, int]] = {}

        for name in enabled:
            cfg = dict(overrides.get(name, {}))
            if name == "sharerobot" and "root" not in cfg and "root_override" not in cfg:
                cfg.setdefault("root", DEFAULT_SHARE_ROBOT_ROOT)
            elif name == "embodied_reasoner" and "root" not in cfg and "root_override" not in cfg:
                cfg.setdefault("root", DEFAULT_EMBODIED_REASONER_ROOT)
            elif name == "robo2vlm" and "root" not in cfg and "root_override" not in cfg:
                cfg.setdefault("root", DEFAULT_ROBO2VLM_ROOT)
            elif name == "bridge_data_v2" and "root" not in cfg and "root_override" not in cfg:
                cfg.setdefault("root", DEFAULT_BRIDGE_ROOT)
            elif name == "droid_100" and "root" not in cfg and "root_override" not in cfg:
                cfg.setdefault("root", DEFAULT_DROID_100_ROOT)
            elif name == "fmb" and "root" not in cfg and "root_override" not in cfg:
                cfg.setdefault("root", DEFAULT_FMB_ROOT)
            labels = self.ingest_one(name, cfg, max_per_source)
            subtype_counts: Dict[str, int] = {}
            added = 0
            for label in labels:
                if label.sample_id in seen_ids:
                    continue
                seen_ids.add(label.sample_id)
                all_labels.append(label)
                subtype_counts[label.state_type] = subtype_counts.get(label.state_type, 0) + 1
                added += 1
            stats[name] = {"added": added, **subtype_counts}

        for name, counts in stats.items():
            logger.info("Robot state ingest stats [%s]: %s", name, counts)
        return all_labels


def build_robot_state_registry() -> RobotStateIngestRegistry:
    from .bridge_robot_state_adapter import BridgeRobotStateAdapter
    from .droid_robot_state_adapter import DroidRobotStateAdapter
    from .embodied_reasoner_adapter import EmbodiedReasonerRobotStateAdapter
    from .fmb_robot_state_adapter import FmbRobotStateAdapter
    from .robo2vlm_adapter import Robo2VLMAdapter
    from .sharerobot_adapter import ShareRobotRobotStateAdapter

    registry = RobotStateIngestRegistry()
    registry.register(ShareRobotRobotStateAdapter())
    registry.register(EmbodiedReasonerRobotStateAdapter())
    registry.register(Robo2VLMAdapter())
    registry.register(BridgeRobotStateAdapter())
    registry.register(DroidRobotStateAdapter())
    registry.register(FmbRobotStateAdapter())
    return registry
