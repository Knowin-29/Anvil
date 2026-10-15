#!/usr/bin/env python3
"""GSA Data Export for Training — Convert verified labels to SFT format.

Features:
- Reads from verified/ directory (only auto-verified labels)
- Converts to multi-turn conversation format for SFT training
- Supports per-dimension ratio sampling
- Deduplication by sample_id
- Shuffle with reproducible seed
- Outputs to training/ directory for train_teacher.sh consumption

Usage:
    python scripts/data_export_for_training.py
    python scripts/data_export_for_training.py --max-samples 5000
    python scripts/data_export_for_training.py --dimension A_identity_attr --max-samples 1000
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("data_export")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERIFIED_DIR = PROJECT_ROOT / "data" / "output" / "gsa_flywheel_exp" / "verified"
# volc_jun12: Export directly to GSA_volc for HuggingFace upload
GSA_VOLC_DIR = Path("/path/to/datasets/GSA_volc")
TRAINING_DIR = GSA_VOLC_DIR / "annotations"
METADATA_DIR = GSA_VOLC_DIR / "metadata"

DIMENSIONS = ["G_grounding", "S_spatial", "A_identity_attr", "A_obj_state", "A_action_precondition", "A_action_interface"]

# Training-side required fields (F1-F5 from flywheel4training_upgrade_prompts.md)
NORM_RANGE = 1000


def compute_annotation_id(label: dict) -> str:
    """Stable annotation_id for template dedup (training-side F1)."""
    key = "|".join([
        label.get("image_path", ""),
        str(label.get("bbox", "")),
        label.get("object_class", ""),
        label.get("pipeline", label.get("dimension", "")),
    ])
    return hashlib.md5(key.encode()).hexdigest()[:16]


def validate_coordinates(label: dict) -> bool:
    """Coordinate legality check (training-side F4)."""
    bbox = label.get("bbox")
    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        if any(v < 0 for v in [x1, y1, x2, y2]):
            return False

    grasp_pt = label.get("grasp_point_2d")
    if grasp_pt and isinstance(grasp_pt, (list, tuple)) and len(grasp_pt) == 2:
        x, y = grasp_pt
        if x < 0 or y < 0:
            return False

    return True


def detect_output_format(label: dict) -> str:
    """Classify format: bbox/point/text (training-side F3 monitor)."""
    if label.get("bbox"):
        return "bbox"
    if label.get("grasp_point_2d") or label.get("points"):
        return "point"
    return "text"


def derive_stage(label: dict) -> str:
    """Determine training stage: sft/grpo/eval (training-side F5)."""
    source = label.get("source", "")
    if source in ("benchmark_eval", "test_fixtures"):
        return "eval"
    scene = label.get("scene_type", "")
    if scene in ("desktop_selfcollect", "api_log"):
        return "grpo"
    return "sft"

# Target ratios for balanced training (sum to 1.0)
DEFAULT_RATIOS = {
    "G_grounding": 0.20,
    "S_spatial": 0.20,
    "A_identity_attr": 0.15,
    "A_obj_state": 0.15,
    "A_action_precondition": 0.15,
    "A_action_interface": 0.15,
}

# SFT conversation templates per dimension
CONVERSATION_TEMPLATES = {
    "G_grounding": {
        "system": "You are a visual grounding assistant. Identify and locate objects in images with bounding boxes.",
        "user_template": "Identify the object '{object_class}' in this image and provide its bounding box coordinates.",
        "assistant_template": 'The {object_class} is located at bounding box [{bbox}] with confidence {confidence:.2f}.',
    },
    "S_spatial": {
        "system": "You are a spatial reasoning assistant. Analyze spatial relationships between objects in images.",
        "user_template": "What is the spatial relationship between '{subject}' and '{object}'?",
        "assistant_template": "The {subject} is {predicate} the {object}.",
    },
    "A_identity_attr": {
        "system": "You are a visual attribute recognition assistant. Identify objects and their visual properties.",
        "user_template": "Describe the visual attributes of the object at [{bbox}].",
        "assistant_template": 'This is a {object_class}. Attributes: color={color}, material={material}, shape={shape}, size={size}.',
    },
    "A_obj_state": {
        "system": "You are an object state recognition assistant. Determine the current state of objects.",
        "user_template": "What is the current state of the {object_class} at [{bbox}]?",
        "assistant_template": "The {object_class} has state type '{state_type}' with value '{state_value}'.",
    },
    "A_action_precondition": {
        "system": "You are a robot action planning assistant. Determine if actions are feasible given the current scene.",
        "user_template": "Can the robot perform a '{action_type}' action on the target object now?",
        "assistant_template": '{precondition_response}',
    },
    "A_action_interface": {
        "system": "You are a robot grasp planning assistant. Determine optimal grasp points and approach vectors.",
        "user_template": "Determine the optimal {action_type} interface for the target object.",
        "assistant_template": 'For {action_type} action: grasp point at [{grasp_point}], jaw axis [{jaw_axis}].',
    },
}


def load_verified_labels() -> Dict[str, List[dict]]:
    """Load all verified labels grouped by dimension."""
    by_dimension = defaultdict(list)

    if not VERIFIED_DIR.exists():
        logger.warning(f"Verified directory not found: {VERIFIED_DIR}")
        return by_dimension

    for fp in sorted(VERIFIED_DIR.rglob("*.jsonl")):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    label = json.loads(line)
                    dim = label.get("pipeline") or label.get("dimension") or "unknown"
                    by_dimension[dim].append(label)
                except json.JSONDecodeError:
                    continue

    for dim, labels in by_dimension.items():
        logger.info(f"  {dim}: {len(labels)} verified labels")

    return by_dimension


def deduplicate(labels: List[dict]) -> List[dict]:
    """Remove duplicates by sample_id."""
    seen = set()
    unique = []
    for label in labels:
        key = label.get("sample_id") or label.get("object_id") or hashlib.md5(json.dumps(label, sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(label)
    return unique


def label_to_conversation(label: dict, dimension: str) -> Optional[dict]:
    """Convert a label to SFT conversation format."""
    template = CONVERSATION_TEMPLATES.get(dimension)
    if not template:
        return None

    try:
        if dimension == "G_grounding":
            obj_class = label.get("object_class", "object")
            bbox = label.get("bbox", [0, 0, 0, 0])
            confidence = float(label.get("confidence", 0.5))
            user_msg = template["user_template"].format(object_class=obj_class)
            asst_msg = template["assistant_template"].format(
                object_class=obj_class, bbox=",".join(map(str, bbox)), confidence=confidence
            )

        elif dimension == "S_spatial":
            subject = label.get("subject", "object A")
            predicate = label.get("predicate", "near")
            obj = label.get("object", "object B")
            user_msg = template["user_template"].format(subject=subject, object=obj)
            asst_msg = template["assistant_template"].format(subject=subject, predicate=predicate, object=obj)

        elif dimension == "A_identity_attr":
            obj_class = label.get("object_class", "object")
            bbox = label.get("bbox", [0, 0, 0, 0])
            attrs = label.get("attributes", {})
            user_msg = template["user_template"].format(bbox=",".join(map(str, bbox)))
            asst_msg = template["assistant_template"].format(
                object_class=obj_class,
                color=attrs.get("color", "unknown"),
                material=attrs.get("material", "unknown"),
                shape=attrs.get("shape", "unknown"),
                size=attrs.get("size", "unknown"),
            )

        elif dimension == "A_obj_state":
            obj_class = label.get("object_class", "object")
            bbox = label.get("bbox", [0, 0, 0, 0])
            state_type = label.get("state_type", "unknown")
            state_value = label.get("state_value", "unknown")
            user_msg = template["user_template"].format(object_class=obj_class, bbox=",".join(map(str, bbox)))
            asst_msg = template["assistant_template"].format(
                object_class=obj_class, state_type=state_type, state_value=state_value
            )

        elif dimension == "A_action_precondition":
            action_type = label.get("action_type", "pick")
            can_do = label.get("can_do_now", True)
            blocked_by = label.get("blocked_by", [])
            failure_reason = label.get("failure_reason", "none")
            user_msg = template["user_template"].format(action_type=action_type)
            if can_do:
                response = f"Yes, the robot can perform '{action_type}' now. No obstacles detected."
            else:
                response = f"No, '{action_type}' is blocked. Reason: {failure_reason}. Blocked by: {', '.join(blocked_by) if blocked_by else 'unknown'}."
            asst_msg = template["assistant_template"].format(precondition_response=response)

        elif dimension == "A_action_interface":
            action_type = label.get("action_type", "pick")
            grasp_point = label.get("grasp_point_2d", [0, 0])
            jaw_axis = label.get("jaw_axis_2d", [1, 0])
            user_msg = template["user_template"].format(action_type=action_type)
            asst_msg = template["assistant_template"].format(
                action_type=action_type,
                grasp_point=",".join(map(str, grasp_point)),
                jaw_axis=",".join(map(str, jaw_axis)),
            )
        else:
            return None

        conversation = {
            "conversations": [
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ],
            "metadata": {
                "pipeline": dimension,
                "source": label.get("source", ""),
                "sample_id": label.get("sample_id", ""),
                "image_path": label.get("image_path", ""),
                "confidence": float(label.get("confidence", 0)),
            },
            # Training-side required fields (volc_jun12)
            "annotation_id": compute_annotation_id(label),
            "dimension": dimension,
            "stage": derive_stage(label),
            "output_format": detect_output_format(label),
        }

        # Add image if available
        img_path = label.get("image_path", "")
        if img_path:
            conversation["image"] = img_path

        return conversation

    except Exception as e:
        logger.debug(f"Failed to convert label: {e}")
        return None


def sample_by_ratio(by_dimension: Dict[str, List[dict]], max_samples: int, ratios: Dict[str, float], seed: int = 42) -> List[dict]:
    """Sample labels according to dimension ratios."""
    rng = random.Random(seed)
    sampled = []

    for dim, ratio in ratios.items():
        target_count = int(max_samples * ratio)
        available = by_dimension.get(dim, [])
        if not available:
            continue
        count = min(target_count, len(available))
        sampled.extend(rng.sample(available, count))

    # Fill remaining slots from any dimension
    remaining = max_samples - len(sampled)
    if remaining > 0:
        all_labels = []
        for labels in by_dimension.values():
            all_labels.extend(labels)
        extra = [l for l in all_labels if l not in sampled]
        if extra:
            sampled.extend(rng.sample(extra, min(remaining, len(extra))))

    rng.shuffle(sampled)
    return sampled


def export_training_data(
    max_samples: int = 10000,
    dimension: Optional[str] = None,
    ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Path:
    """Main export function."""
    logger.info("Loading verified labels...")
    by_dimension = load_verified_labels()

    total_available = sum(len(v) for v in by_dimension.values())
    if total_available == 0:
        logger.error("No verified labels found. Run the flywheel pipeline first.")
        sys.exit(1)

    logger.info(f"Total verified labels: {total_available}")

    # Deduplicate per dimension
    for dim in list(by_dimension.keys()):
        before = len(by_dimension[dim])
        by_dimension[dim] = deduplicate(by_dimension[dim])
        after = len(by_dimension[dim])
        if before != after:
            logger.info(f"  {dim}: deduped {before} -> {after}")

    # Filter to single dimension if specified
    if dimension:
        if dimension not in by_dimension:
            logger.error(f"Dimension '{dimension}' not found in verified data")
            sys.exit(1)
        by_dimension = {dimension: by_dimension[dimension]}

    # Sample
    if ratios is None:
        ratios = DEFAULT_RATIOS
    sampled = sample_by_ratio(by_dimension, max_samples, ratios, seed)
    logger.info(f"Sampled {len(sampled)} labels for training")

    # Coordinate validation (F4) before conversion
    valid_sampled = [l for l in sampled if validate_coordinates(l)]
    filtered_count = len(sampled) - len(valid_sampled)
    if filtered_count > 0:
        logger.info(f"Filtered {filtered_count} samples with invalid coordinates")
    sampled = valid_sampled

    # Convert to conversations
    conversations = []
    for label in sampled:
        dim = label.get("pipeline") or label.get("dimension") or "unknown"
        conv = label_to_conversation(label, dim)
        if conv:
            conversations.append(conv)

    logger.info(f"Converted {len(conversations)} labels to SFT format")

    # Write output to GSA_volc
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = TRAINING_DIR / f"gsa_sft_data_{ts}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # Also write a "latest" file
    latest_file = TRAINING_DIR / "gsa_sft_data.jsonl"
    with open(latest_file, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # Compute distribution statistics (F2/F3 monitoring)
    dim_counts = defaultdict(int)
    fmt_counts = defaultdict(int)
    stage_counts = defaultdict(int)
    for conv in conversations:
        dim_counts[conv.get("dimension", "unknown")] += 1
        fmt_counts[conv.get("output_format", "text")] += 1
        stage_counts[conv.get("stage", "sft")] += 1

    # Write export metadata
    meta = {
        "export_time": datetime.now().isoformat(),
        "total_samples": len(conversations),
        "source_labels": total_available,
        "max_samples": max_samples,
        "seed": seed,
        "dimension_filter": dimension,
        "coordinate_filtered": filtered_count,
        "dimension_counts": dict(dim_counts),
        "format_distribution": dict(fmt_counts),
        "stage_distribution": dict(stage_counts),
        "output_file": str(output_file),
        "output_dir": str(TRAINING_DIR),
    }
    meta_file = METADATA_DIR / f"export_meta_{ts}.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Export complete: {output_file}")
    logger.info(f"Dimension distribution:")
    for dim, count in meta["dimension_counts"].items():
        if count > 0:
            logger.info(f"  {dim}: {count} ({count/len(conversations)*100:.1f}%)")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Export verified GSA labels to SFT training format")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum samples to export")
    parser.add_argument("--dimension", type=str, default=None, help="Filter to single dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    export_training_data(
        max_samples=args.max_samples,
        dimension=args.dimension,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
