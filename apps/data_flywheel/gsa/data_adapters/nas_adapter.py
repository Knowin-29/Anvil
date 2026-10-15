"""NAS Adapter — Unified interface for reading NAS-hosted datasets.

Discovers and loads datasets from anvil-oss/nas-backup/datasets/test_contributor/
and routes them to the appropriate A pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("gsa.data_adapters.nas")

NAS_ROOT = Path("/path/to/datasets")
DIRECT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DATASET_REGISTRY: Dict[str, Dict] = {
    "paco-lvis": {
        "dir": "001_done_P1_A0_paco-lvis",
        "pipeline": "A_identity_attr",
        "description": "641K part masks, object attributes",
        "adapter": "paco_lvis_adapter",
    },
    "mit-states": {
        "dir": "013_done_P1_A1_mit-states",
        "pipeline": "A_obj_state",
        "description": "63K images, object state annotations",
        "adapter": "mit_states_adapter",
    },
    "embodied-reasoner": {
        "dir": "011_done_P1_A1_embodied-reasoner",
        "pipeline": "A_robot_state",
        "description": "9K trajectories, robot reasoning",
        "adapter": "embodied_reasoner_adapter",
        "ingest": "robot_state",
    },
    "sharerobot": {
        "dir": "public_sources/downloads/sharerobot",
        "root_override": "/path/to/datasets/downloads/sharerobot",
        "pipeline": "A_robot_state",
        "description": "ShareRobot planning/trajectory for robot state",
        "adapter": "sharerobot_adapter",
        "ingest": "robot_state",
    },
    "visual-genome": {
        "dir": "019_done_P1_A1_visual-genome",
        "pipeline": "A_identity_attr",
        "description": "Attributes and relationships",
        "adapter": "visual_genome_adapter",
    },
    "fsd-dataset": {
        "dir": "004_done_P1_A0_fsd-dataset",
        "pipeline": "A_action_interface",
        "description": "Robot task frames and instructions; no native action point labels",
        "adapter": "a_action_interface_offline_probe",
    },
    "molmo2-vp": {
        "dir": "038_done_P0_molmo2-vp",
        "pipeline": "A_action_interface",
        "description": "Video point labels; point-coordinate auxiliary probe data",
        "adapter": "a_action_interface_offline_probe",
    },
}


def _has_any_file(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _has_direct_images(path: Path) -> bool:
    """Return true only for directly readable image files, not zip/tar payloads."""
    if not path.exists():
        return False
    return any(
        f.is_file() and f.suffix.lower() in DIRECT_IMAGE_EXTENSIONS
        for f in path.rglob("*")
    )


def discover_available_datasets() -> List[Dict]:
    """Scan NAS root and return list of available datasets with metadata."""
    available = []
    if not NAS_ROOT.exists():
        logger.warning(f"NAS root not found: {NAS_ROOT}")
        return available

    for name, info in DATASET_REGISTRY.items():
        ds_path = NAS_ROOT / info["dir"]
        if ds_path.exists():
            raw_path = ds_path / "raw"
            extracted_path = ds_path / "extracted"
            has_raw = _has_any_file(raw_path)
            has_extracted = _has_any_file(extracted_path)
            has_direct_images = (
                _has_direct_images(extracted_path) or _has_direct_images(raw_path)
            )
            available.append({
                "name": name,
                "path": str(ds_path),
                "pipeline": info["pipeline"],
                "has_raw": has_raw,
                "has_extracted": has_extracted,
                "has_direct_images": has_direct_images,
                "adapter_ready": has_raw or has_extracted,
                # ready means ready for image-directory runners. Raw zip/tar
                # payloads can be adapter-ready but must not be treated as
                # directly runnable image folders.
                "ready": has_direct_images,
                "description": info["description"],
            })
    return available


def get_dataset_path(name: str) -> Optional[Path]:
    """Get the path to a named dataset, or None if not available."""
    info = DATASET_REGISTRY.get(name)
    if not info:
        return None
    ds_path = NAS_ROOT / info["dir"]
    if ds_path.exists():
        return ds_path
    return None


def get_images_from_dataset(name: str, max_images: int = 100) -> List[str]:
    """Get image paths from a named dataset."""
    ds_path = get_dataset_path(name)
    if not ds_path:
        return []

    image_dirs = [ds_path / "extracted", ds_path / "raw"]
    images = []
    for img_dir in image_dirs:
        if not img_dir.exists():
            continue
        for f in sorted(img_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in DIRECT_IMAGE_EXTENSIONS:
                images.append(str(f))
                if len(images) >= max_images:
                    return images
    return images
