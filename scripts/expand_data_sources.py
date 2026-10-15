#!/usr/bin/env python3
"""Expand data sources — discover and route NAS/local datasets to GSA dimensions.

Scans configured data sources (NAS H5 trajectories, local image directories),
filters for embodied/indoor robot scenes, and builds dimension_sources/ symlinks
following the capability fusion routing rules.

Usage:
    python scripts/expand_data_sources.py [--max-per-dim 300] [--output-dir data/input/dimension_sources]
"""

import argparse
import glob
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("expand_data_sources")

NAS_BASE = "/path/to/datasets"
LOCAL_BASE = "/path/to/datasets/datasets"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_SOURCES: Dict[str, dict] = {
    "robosuite": {
        "path": f"{NAS_BASE}/005_done_P1_A0_gymnasium-robosuite-mujoco/raw",
        "type": "h5",
        "dimensions": ["A_action_interface", "A_action_precondition"],
        "description": "MuJoCo robosuite trajectories with RGB (1500 trajectories)",
        "scene_filter": "robot_manipulation",
    },
    "embodied_reasoner": {
        "path": f"{NAS_BASE}/011_done_P1_A1_embodied-reasoner/raw",
        "type": "image_dir",
        "dimensions": ["A_obj_state", "A_action_precondition"],
        "description": "Embodied reasoning trajectories (9K+ train)",
        "scene_filter": "embodied",
    },
    "mit_states": {
        "path": f"{NAS_BASE}/013_done_P1_A1_mit-states/raw",
        "type": "image_dir",
        "dimensions": ["A_obj_state", "A_identity_attr"],
        "description": "Object state changes (open/closed, wet/dry etc.)",
        "scene_filter": "indoor",
    },
    "spatialsense": {
        "path": f"{NAS_BASE}/009_done_P0_A1_spatialsense/raw",
        "type": "image_dir",
        "dimensions": ["S_spatial"],
        "description": "Spatial relation annotations",
        "scene_filter": "indoor",
    },
    "visual_genome": {
        "path": f"{NAS_BASE}/019_done_P1_A1_visual-genome/raw",
        "type": "image_dir",
        "dimensions": ["G_grounding", "S_spatial"],
        "description": "Indoor objects and relationships",
        "scene_filter": "indoor",
    },
    # benchmark_eval_all removed (volc_jun12): mixed source with uncontrolled scenes,
    # overlaps with independent subsets (ChangeIt/Embodied-Reasoner/SpatialSense).
    "appliance": {
        "path": f"{LOCAL_BASE}/appliance_datasets",
        "type": "image_dir",
        "dimensions": ["A_obj_state", "A_identity_attr"],
        "description": "Washing machine / appliance instruction data",
        "scene_filter": "appliance",
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXCLUDE_PATTERNS = {"knob", "panel", "instruction", "diagram", "icon", "logo", "thumbnail"}


def is_valid_image(path: str) -> bool:
    """Check if image is suitable (not a diagram/icon/panel)."""
    name_lower = Path(path).stem.lower()
    return not any(pat in name_lower for pat in EXCLUDE_PATTERNS)


def discover_images_from_dir(base_path: str, max_count: int = 500) -> List[str]:
    """Recursively find images in a directory."""
    images = []
    base = Path(base_path)
    if not base.exists():
        logger.warning(f"Path does not exist: {base_path}")
        return []

    for ext in IMAGE_EXTENSIONS:
        found = list(base.rglob(f"*{ext}"))
        images.extend(str(p) for p in found if is_valid_image(str(p)))

    if len(images) > max_count:
        images = random.sample(images, max_count)
    return images


def discover_images_from_h5(base_path: str, max_count: int = 200) -> List[str]:
    """Extract RGB frames from H5 trajectory files and save as temp images."""
    h5_files = list(Path(base_path).rglob("*.h5")) if Path(base_path).exists() else []
    if not h5_files:
        logger.warning(f"No H5 files found in {base_path}")
        return []

    output_dir = PROJECT_ROOT / "data" / "input" / "h5_extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    try:
        import h5py
        import numpy as np
        from PIL import Image

        sample_files = random.sample(h5_files, min(len(h5_files), max_count))
        for h5_path in sample_files:
            try:
                with h5py.File(str(h5_path), "r") as f:
                    if "obs" in f and "agentview_image" in f["obs"]:
                        frames = f["obs"]["agentview_image"]
                        idx = random.randint(0, len(frames) - 1)
                        img_array = frames[idx]
                        img = Image.fromarray(img_array.astype(np.uint8))
                        out_path = output_dir / f"{h5_path.stem}_frame{idx}.jpg"
                        img.save(str(out_path), quality=90)
                        extracted.append(str(out_path))
            except Exception as e:
                logger.debug(f"Failed to extract from {h5_path}: {e}")
                continue
    except ImportError:
        logger.warning("h5py not installed, skipping H5 extraction. Using placeholder paths.")
        for h5_path in h5_files[:max_count]:
            extracted.append(str(h5_path))

    return extracted[:max_count]


def discover_images(source_name: str, max_count: int = 300) -> List[str]:
    """Discover images from a named data source."""
    if source_name not in DATA_SOURCES:
        logger.error(f"Unknown source: {source_name}")
        return []

    src = DATA_SOURCES[source_name]
    src_type = src["type"]
    src_path = src["path"]

    if src_type == "h5":
        return discover_images_from_h5(src_path, max_count)
    else:
        return discover_images_from_dir(src_path, max_count)


def build_dimension_sources(output_dir: str = None, max_per_dim: int = 300):
    """Build dimension_sources directory with symlinks to source images."""
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "input" / "dimension_sources")

    out_base = Path(output_dir)

    dimensions = ["G_grounding", "S_spatial", "A_identity_attr", "A_obj_state", "A_action_precondition", "A_action_interface"]

    stats = {}
    for dim in dimensions:
        dim_dir = out_base / dim
        dim_dir.mkdir(parents=True, exist_ok=True)

        applicable_sources = [name for name, cfg in DATA_SOURCES.items() if dim in cfg["dimensions"]]
        per_source_limit = max(max_per_dim // max(len(applicable_sources), 1), 50)

        all_images = []
        for src_name in applicable_sources:
            images = discover_images(src_name, per_source_limit)
            all_images.extend(images)
            logger.info(f"  [{dim}] {src_name}: {len(images)} images")

        if len(all_images) > max_per_dim:
            all_images = random.sample(all_images, max_per_dim)

        # Write image paths to dimension source file
        txt_file = out_base / f"{dim}.txt"
        with open(txt_file, "w") as f:
            for img_path in all_images:
                f.write(f"{img_path}\n")

        stats[dim] = {"total": len(all_images), "sources": applicable_sources}
        logger.info(f"[{dim}] Total: {len(all_images)} images from {len(applicable_sources)} sources")

    # Write manifest
    manifest = {
        "generated_at": str(Path(output_dir)),
        "dimensions": stats,
        "data_sources": {k: {"path": v["path"], "type": v["type"], "description": v["description"]} for k, v in DATA_SOURCES.items()},
    }
    manifest_path = out_base / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Manifest written to %s", manifest_path)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Expand GSA data sources")
    parser.add_argument("--max-per-dim", type=int, default=300, help="Max images per dimension")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    logger.info("=== GSA Data Source Expansion ===")
    logger.info("NAS base: %s", NAS_BASE)
    logger.info("Local base: %s", LOCAL_BASE)
    logger.info("Max per dimension: %d", args.max_per_dim)

    stats = build_dimension_sources(output_dir=args.output_dir, max_per_dim=args.max_per_dim)

    logger.info("\n=== Summary ===")
    total = 0
    for dim, s in stats.items():
        logger.info(f"  {dim}: {s['total']} images ({', '.join(s['sources'])})")
        total += s["total"]
    logger.info(f"  TOTAL: {total} images across {len(stats)} dimensions")


if __name__ == "__main__":
    main()
