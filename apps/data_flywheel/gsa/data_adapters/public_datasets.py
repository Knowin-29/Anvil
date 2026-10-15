"""Public Dataset Adapter — Integrate contributor's GSA data/benchmark collections.

Maps public datasets to A pipeline capabilities:
- UMD_RGBD_Part_Affordance -> A_action_interface (grasp/scoop/contain affordances)
- AGD20K -> A_action_interface (affordance grounding)
- VAW -> A_identity_attr (visual attributes)
- Where2Place -> A_action_interface (placement affordance)
- RoboVQA -> A_action_precondition + A_robot_state
- Ego4D_Hands_Objects -> A_robot_state (hand-object interaction)
- ChangeIt -> A_obj_state (state changes)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("gsa.data_adapters.public_datasets")

GSA_DATA_ROOT = Path("/path/to/workspace/foundation/contributor/GSA_data_collection")
GSA_BENCH_ROOT = Path("/path/to/workspace/foundation/contributor/GSA_benchmark_collection")

DATASET_PIPELINE_MAP = {
    # GSA_data_collection
    "08_VAW": {
        "pipeline": "A_identity_attr",
        "description": "Visual Attributes in the Wild - 620K attribute annotations",
        "subtypes": ["color", "material", "shape", "size", "multi_attr"],
        "root": GSA_DATA_ROOT,
    },
    "10_UMD_RGBD_Part_Affordance": {
        "pipeline": "A_action_interface",
        "description": "Part-level affordance (grasp/cut/scoop/contain/pound/support)",
        "subtypes": ["grasp_point", "push_contact", "place_region"],
        "root": GSA_DATA_ROOT,
    },
    "11_AGD20K": {
        "pipeline": "A_action_interface",
        "description": "20K affordance grounding images",
        "subtypes": ["grasp_point", "push_contact"],
        "root": GSA_DATA_ROOT,
    },
    "12_Ego4D_Hands_Objects": {
        "pipeline": "A_robot_state",
        "description": "Ego4D hand-object interaction clips",
        "subtypes": ["holding_object", "contact_state"],
        "root": GSA_DATA_ROOT,
    },
    "13_ChangeIt": {
        "pipeline": "A_obj_state",
        "description": "Object state change detection",
        "subtypes": ["pose", "open_close", "occupancy"],
        "root": GSA_DATA_ROOT,
    },
    # GSA_benchmark_collection
    "04_RoboVQA": {
        "pipeline": "A_action_precondition",
        "description": "65GB robot VQA - action feasibility reasoning",
        "subtypes": ["can_pick", "can_push", "blocked"],
        "root": GSA_BENCH_ROOT,
    },
    "05_Where2Place": {
        "pipeline": "A_action_interface",
        "description": "Spatial affordance grounding for placement",
        "subtypes": ["place_region"],
        "root": GSA_BENCH_ROOT,
    },
    "08_PointArena": {
        "pipeline": "A_action_interface",
        "description": "Point-based action grounding benchmark",
        "subtypes": ["grasp_point"],
        "root": GSA_BENCH_ROOT,
    },
}


def discover_public_datasets() -> List[Dict]:
    """Discover available public datasets from contributor's collections."""
    available = []
    for dir_name, info in DATASET_PIPELINE_MAP.items():
        ds_path = info["root"] / dir_name
        if ds_path.exists():
            has_download = (ds_path / "download.sh").exists()
            has_data = any(ds_path.rglob("*.json")) or any(ds_path.rglob("*.jsonl"))
            available.append({
                "name": dir_name,
                "path": str(ds_path),
                "pipeline": info["pipeline"],
                "subtypes": info["subtypes"],
                "description": info["description"],
                "has_download_script": has_download,
                "has_data": has_data,
                "ready": has_data,
            })
    return available


def get_download_commands() -> Dict[str, str]:
    """Get download commands for datasets that need downloading."""
    commands = {}
    for dir_name, info in DATASET_PIPELINE_MAP.items():
        ds_path = info["root"] / dir_name
        script = ds_path / "download.sh"
        if script.exists():
            commands[dir_name] = f"cd {ds_path} && bash download.sh"
    return commands


def get_affordance_categories() -> Dict[str, List[str]]:
    """Return affordance categories from UMD dataset for validation."""
    return {
        "grasp": ["handle", "body", "rim", "edge"],
        "cut": ["blade", "edge"],
        "scoop": ["bowl", "concave_surface"],
        "contain": ["interior", "cavity"],
        "pound": ["head", "flat_surface"],
        "support": ["flat_top", "base"],
        "wrap-grasp": ["cylindrical_body"],
    }
