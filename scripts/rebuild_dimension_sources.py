#!/usr/bin/env python3
"""[DEPRECATED 2026-07-19] Rebuild dimension_sources with scene-type classified data sources.

DEPRECATED: This script hardcodes 8 sources and is NO LONGER the source of truth.
The live flywheel reads `data/input/dimension_sources/manifest.json` (generated from the
GSA_Forge sources registry), NOT this script. Kept for reference/reproducibility only.
To regenerate source lists, use the manifest-driven path documented in
docs/GSA数据生成_校验_交互.md. Do not run this unless you intend to override the manifest.
"""
import os
import sys

if __name__ == "__main__" and os.environ.get("ALLOW_DEPRECATED_REBUILD") != "1":
    sys.stderr.write(
        "[DEPRECATED] rebuild_dimension_sources.py is no longer the source of truth.\n"
        "The live flywheel uses data/input/dimension_sources/manifest.json.\n"
        "Set ALLOW_DEPRECATED_REBUILD=1 to run anyway.\n"
    )
    sys.exit(2)
import random
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "input" / "dimension_sources"

NAS_BASE = "/path/to/datasets"
LOCAL_BASE = "/path/to/datasets/datasets"
BRIDGE_BASE = "/path/to/environment/backup_420/bridgedatav2/data/images"

# Scene-classified data sources
SOURCES = {
    "bridge_v2_tabletop": {
        "path": BRIDGE_BASE,
        "scene_type": "tabletop_manipulation",
        "dimensions": ["G_grounding", "S_spatial", "A_identity_attr", "A_obj_state", "A_action_interface", "A_action_precondition"],
        "extensions": [".png", ".jpg"],
        "max_images": 400,
    },
    "benchmark_robot_indoor": {
        "path": f"{LOCAL_BASE}/benchmark_eval_all",
        "scene_type": "kitchen_interaction",
        "dimensions": ["G_grounding", "S_spatial", "A_identity_attr", "A_obj_state", "A_action_interface", "A_action_precondition"],
        "extensions": [".jpg", ".png"],
        "max_images": 600,
    },
    "egoobjects_daily": {
        "path": f"{NAS_BASE}/015_done_P1_A1_egoobjects/raw",
        "scene_type": "living_room",
        "dimensions": ["G_grounding", "A_identity_attr", "A_obj_state"],
        "extensions": [".jpg", ".png", ".jpeg"],
        "max_images": 200,
    },
    "visual_genome_indoor": {
        "path": f"{NAS_BASE}/019_done_P1_A1_visual-genome/raw",
        "scene_type": "living_room",
        "dimensions": ["G_grounding", "S_spatial"],
        "extensions": [".jpg", ".png"],
        "max_images": 200,
    },
    "mit_states_objects": {
        "path": f"{NAS_BASE}/013_done_P1_A1_mit-states/raw",
        "scene_type": "tabletop_manipulation",
        "dimensions": ["A_obj_state", "A_identity_attr"],
        "extensions": [".jpg", ".png", ".jpeg"],
        "max_images": 200,
    },
    "spatialsense_relations": {
        "path": f"{NAS_BASE}/009_done_P0_A1_spatialsense/raw",
        "scene_type": "living_room",
        "dimensions": ["S_spatial"],
        "extensions": [".jpg", ".png"],
        "max_images": 150,
    },
    "embodied_reasoner": {
        "path": f"{NAS_BASE}/011_done_P1_A1_embodied-reasoner/raw",
        "scene_type": "kitchen_interaction",
        "dimensions": ["A_action_precondition", "A_obj_state"],
        "extensions": [".jpg", ".png", ".jpeg"],
        "max_images": 200,
    },
    "robosuite_sim": {
        "path": f"{NAS_BASE}/005_done_P1_A0_gymnasium-robosuite-mujoco/raw",
        "scene_type": "tabletop_manipulation",
        "dimensions": ["A_action_interface", "A_action_precondition"],
        "extensions": [".h5"],
        "max_images": 100,
    },
}

DIMENSIONS = ["G_grounding", "S_spatial", "A_identity_attr", "A_obj_state", "A_action_precondition", "A_action_interface"]
EXCLUDE_PATTERNS = {"knob", "panel", "instruction", "diagram", "icon", "logo", "thumbnail", "template"}


def find_images(path, extensions, max_count):
    """Find images in a directory."""
    images = []
    base = Path(path)
    if not base.exists():
        print(f"  [SKIP] Path not found: {path}")
        return []
    for ext in extensions:
        if ext == ".h5":
            images.extend(str(p) for p in base.rglob(f"*{ext}"))
        else:
            images.extend(str(p) for p in base.rglob(f"*{ext}") 
                         if not any(pat in p.stem.lower() for pat in EXCLUDE_PATTERNS))
    if len(images) > max_count:
        images = random.sample(images, max_count)
    return images


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    dim_images = {d: [] for d in DIMENSIONS}
    manifest = {"sources": {}, "dimensions": {}}
    
    for src_name, cfg in SOURCES.items():
        images = find_images(cfg["path"], cfg["extensions"], cfg["max_images"])
        print(f"[{src_name}] Found {len(images)} images (scene: {cfg['scene_type']})")
        manifest["sources"][src_name] = {"count": len(images), "scene_type": cfg["scene_type"]}
        
        for dim in cfg["dimensions"]:
            # Tag each image with source and scene_type
            for img in images:
                dim_images[dim].append(f"{img}\t{src_name}\t{cfg['scene_type']}")
    
    # Write per-dimension files
    for dim in DIMENSIONS:
        entries = dim_images[dim]
        # Limit per dimension
        if len(entries) > 500:
            entries = random.sample(entries, 500)
        random.shuffle(entries)
        
        txt_file = OUTPUT_DIR / f"{dim}.txt"
        with open(txt_file, "w") as f:
            for entry in entries:
                f.write(entry + "\n")
        
        manifest["dimensions"][dim] = len(entries)
        print(f"  [{dim}] {len(entries)} images written")
    
    # Write manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nDone. Manifest: {manifest_path}")
    print(f"Total: {sum(manifest['dimensions'].values())} entries across {len(DIMENSIONS)} dimensions")


if __name__ == "__main__":
    main()
