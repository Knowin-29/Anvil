"""Teacher Training — Prepare GSA data for SFT training.

Converts GSA labels (G/S/A_state/A_action) into conversation-format
training data suitable for VLM SFT (Supervised Fine-Tuning).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("teacher.data_prep")


def gsa_label_to_conversation(label: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single GSA label to a conversation training sample."""
    bucket = label.get("bucket", "")
    subtype = label.get("subtype", "")
    image_path = label.get("image_path", "")

    if bucket == "G":
        return _g_to_conversation(label)
    elif bucket == "S":
        return _s_to_conversation(label)
    elif bucket == "A_state":
        return _a_state_to_conversation(label)
    elif bucket == "A_action":
        return _a_action_to_conversation(label)
    elif bucket == "A_robot_state":
        return _a_robot_state_to_conversation(label)
    return {}


def _g_to_conversation(label: Dict) -> Dict:
    question = label.get("question", "")
    answer = label.get("answer", {})
    image_path = label.get("image_path", "")

    if isinstance(answer, dict):
        bbox = answer.get("bbox")
        obj_name = answer.get("object_name", "")
        count = answer.get("count")

        if bbox:
            response = f"The {obj_name} is located at [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]."
        elif count is not None:
            response = f"There are {count} {answer.get('object_class', 'objects')} in the image."
        else:
            response = json.dumps(answer, ensure_ascii=False)
    else:
        response = str(answer)

    return {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ],
        "image": image_path,
        "metadata": {"bucket": "G", "subtype": label.get("subtype", "")},
    }


def _s_to_conversation(label: Dict) -> Dict:
    question = label.get("question", "")
    answer_text = label.get("answer", "")
    image_path = label.get("image_path", "")

    return {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": str(answer_text)},
        ],
        "image": image_path,
        "metadata": {"bucket": "S", "subtype": label.get("subtype", "")},
    }


def _a_state_to_conversation(label: Dict) -> Dict:
    obj_name = label.get("raw_object_name", label.get("object_name", "object"))
    state_key = label.get("state_key", "")
    state_value = label.get("state_value", "")
    image_path = label.get("image_path", "")

    if state_key and state_value:
        question = f"What is the state of the {obj_name}? Is it {state_key.replace('_', ' or ')}?"
        response = f"The {obj_name} is {state_value}."
    else:
        question = f"What object is this?"
        response = f"This is a {obj_name}."

    return {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ],
        "image": image_path,
        "metadata": {"bucket": "A_state", "subtype": label.get("subtype", "")},
    }


def _a_robot_state_to_conversation(label: Dict) -> Dict:
    state_type = label.get("state_type", label.get("subtype", "gripper_empty"))
    state_value = label.get("state_value", "unknown")
    image_path = label.get("image_path", "")
    target = label.get("target_object_class")

    prompts = {
        "gripper_empty": "Is the robot gripper empty?",
        "holding_object": "Is the robot gripper holding an object?",
        "object_between_fingers": "Is there an object between the gripper fingers?",
        "contact_state": "Is the gripper in contact with a surface or object?",
        "tool_ready": "Is the end-effector tool ready for the next action?",
    }
    question = prompts.get(state_type, "What is the current gripper state?")

    if state_value == "yes":
        answer = f"Yes, the gripper state is {state_type.replace('_', ' ')}."
    elif state_value == "no":
        answer = f"No, the gripper state is not {state_type.replace('_', ' ')}."
    else:
        answer = f"The gripper state appears to be {state_type.replace('_', ' ')} (uncertain)."
    if target and state_type == "holding_object":
        answer += f" Held object: {target}."

    return {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "image": image_path,
        "metadata": {
            "bucket": "A_robot_state",
            "subtype": label.get("subtype", state_type),
            "source_dataset": label.get("source_dataset", ""),
            "label_status": label.get("label_status", ""),
            "stable_training_label": bool(label.get("stable_training_label")),
            "quality_flags": label.get("quality_flags", []),
        },
    }


def _a_action_to_conversation(label: Dict) -> Dict:
    obj_id = label.get("object_id", "object")
    action_type = label.get("action_type", "pick")
    image_path = label.get("image_path", "")

    if action_type == "pick":
        grasp = label.get("grasp_point_2d", [0, 0])
        jaw = label.get("jaw_axis_2d", [1, 0])
        can_pick = label.get("can_pick_now", True)
        blocked = label.get("blocked_by", [])

        question = f"How should I grasp the {obj_id}? Provide the grasp point and jaw axis."
        response = (f"Grasp point: [{grasp[0]:.1f}, {grasp[1]:.1f}], "
                    f"Jaw axis: [{jaw[0]:.2f}, {jaw[1]:.2f}], "
                    f"Can pick now: {can_pick}")
        if blocked:
            response += f", Blocked by: {blocked}"

    elif action_type == "push":
        contact = label.get("push_contact_point_2d", [0, 0])
        direction = label.get("push_direction_2d", [1, 0])
        question = f"How should I push the {obj_id}?"
        response = (f"Push contact: [{contact[0]:.1f}, {contact[1]:.1f}], "
                    f"Direction: [{direction[0]:.2f}, {direction[1]:.2f}]")

    elif action_type == "place":
        region = label.get("place_region_2d", {})
        question = f"Where should I place the {obj_id}?"
        response = f"Place region: {json.dumps(region)}"
    else:
        question = f"What action for {obj_id}?"
        response = json.dumps(label, ensure_ascii=False)

    return {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ],
        "image": image_path,
        "metadata": {"bucket": "A_action", "subtype": label.get("subtype", "")},
    }


def _is_training_eligible(label: Dict[str, Any]) -> bool:
    """Gate labels before converting them into SFT samples."""
    if label.get("bucket") == "A_robot_state":
        return bool(label.get("stable_training_label"))
    return label.get("verifier_status") == "auto_verifiable"


def prepare_training_data(
    labels_dir: str,
    output_path: str,
    min_confidence: float = 0.0,
) -> int:
    """Convert all GSA labels to training format."""
    all_samples = []

    for bucket_dir in Path(labels_dir).iterdir():
        if not bucket_dir.is_dir():
            continue
        for jsonl_file in bucket_dir.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    label = json.loads(line)
                    if not _is_training_eligible(label):
                        continue
                    conv = gsa_label_to_conversation(label)
                    if conv and conv.get("conversations"):
                        all_samples.append(conv)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Prepared {len(all_samples)} training samples -> {output_path}")
    return len(all_samples)
