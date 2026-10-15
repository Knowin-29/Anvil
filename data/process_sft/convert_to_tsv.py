"""
Convert verified flywheel annotations (JSONL) to Rex-Omni TSV training format.

Usage:
    python convert_to_tsv.py --input_dir /path/to/verified/ --output_dir /path/to/tsv/
"""
import argparse
import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def encode_image_to_base64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return None


def build_grounding_annotation(sample: dict) -> dict:
    bbox = sample.get("bbox")
    label = sample.get("object_class", "object")
    if bbox is None:
        return None
    return {"boxes": [{"bbox": bbox, "phrase": label}]}


def build_spatial_annotation(sample: dict) -> dict:
    subject = sample.get("subject", "")
    predicate = sample.get("predicate", "")
    obj = sample.get("object", "")
    if not all([subject, predicate, obj]):
        return None
    question = f"What is the spatial relationship between {subject} and {obj}?"
    answer = f"The {subject} is {predicate} the {obj}."
    return {"conversations": [
        {"from": "human", "value": f"<image>\n{question}"},
        {"from": "gpt", "value": answer},
    ]}


def build_obj_state_annotation(sample: dict) -> dict:
    obj_class = sample.get("object_class", "object")
    state_value = sample.get("state_value", "")
    state_type = sample.get("state_type", "")
    if not state_value:
        return None
    question = f"What is the current state of the {obj_class}?"
    answer = f"The {obj_class} is in a {state_type} state: {state_value}."
    return {"conversations": [
        {"from": "human", "value": f"<image>\n{question}"},
        {"from": "gpt", "value": answer},
    ]}


def build_identity_attr_annotation(sample: dict) -> dict:
    obj_class = sample.get("object_class", "object")
    attributes = sample.get("attributes", {})
    if not attributes:
        return None
    attr_str = ", ".join(f"{k}: {v}" for k, v in attributes.items())
    question = f"Describe the visual attributes of the {obj_class}."
    answer = f"The {obj_class} has: {attr_str}."
    return {"conversations": [
        {"from": "human", "value": f"<image>\n{question}"},
        {"from": "gpt", "value": answer},
    ]}


def build_action_precondition_annotation(sample: dict) -> dict:
    obj_class = sample.get("object_class", "object")
    can_do_now = sample.get("can_do_now", True)
    blocked_by = sample.get("blocked_by", [])
    need_pre_action = sample.get("need_pre_action", "none")
    action_type = sample.get("action_type", "interact with")
    if can_do_now and not blocked_by:
        answer = f"The robot can directly {action_type} the {obj_class} without preconditions."
    elif blocked_by:
        blockers = ", ".join(blocked_by) if isinstance(blocked_by, list) else str(blocked_by)
        answer = f"The {obj_class} is blocked by: {blockers}. Pre-action needed: {need_pre_action}."
    else:
        answer = f"The {obj_class} requires pre-action: {need_pre_action}."
    question = f"Can a robot directly {action_type} the {obj_class}? What preconditions exist?"
    return {"conversations": [
        {"from": "human", "value": f"<image>\n{question}"},
        {"from": "gpt", "value": answer},
    ]}


def build_action_interface_annotation(sample: dict) -> dict:
    obj_class = sample.get("object_class", "object")
    action_type = sample.get("action_type", "")
    grasp_point = sample.get("grasp_point_2d", "")
    jaw_axis = sample.get("jaw_axis_2d", "")
    parts = []
    if action_type:
        parts.append(f"action type: {action_type}")
    if grasp_point:
        parts.append(f"grasp point: {grasp_point}")
    if jaw_axis:
        parts.append(f"jaw axis: {jaw_axis}")
    if not parts:
        return None
    answer = f"Action interface for {obj_class}: {'; '.join(parts)}."
    question = f"How should a robot grasp the {obj_class}?"
    return {"conversations": [
        {"from": "human", "value": f"<image>\n{question}"},
        {"from": "gpt", "value": answer},
    ]}


CONVERTERS = {
    "G_grounding": build_grounding_annotation,
    "S_spatial": build_spatial_annotation,
    "A_obj_state": build_obj_state_annotation,
    "A_identity_attr": build_identity_attr_annotation,
    "A_action_precondition": build_action_precondition_annotation,
    "A_action_interface": build_action_interface_annotation,
}


def process_dimension(input_dir, dimension, max_samples=None):
    dim_dir = os.path.join(input_dir, dimension)
    if not os.path.isdir(dim_dir):
        return []
    converter = CONVERTERS.get(dimension)
    if not converter:
        return []
    results = []
    for jf in sorted(Path(dim_dir).glob("*.jsonl")):
        with open(jf) as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                except:
                    continue
                image_path = sample.get("image_path", "")
                if not image_path or not os.path.isfile(image_path):
                    continue
                ann = converter(sample)
                if ann:
                    results.append((image_path, ann))
                if max_samples and len(results) >= max_samples:
                    return results
    return results


def write_tsv_dataset(samples, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f"{prefix}.images.tsv")
    ann_path = os.path.join(output_dir, f"{prefix}.annotations.tsv")
    idx_path = os.path.join(output_dir, f"{prefix}.annotations.tsv.lineidx")
    img_offsets = {}
    ann_offsets = []
    with open(img_path, "w") as img_f, open(ann_path, "w") as ann_f, open(idx_path, "w") as idx_f:
        for image_path, annotation in samples:
            if image_path not in img_offsets:
                img_offset = img_f.tell()
                img_b64 = encode_image_to_base64(image_path)
                if not img_b64:
                    continue
                img_f.write(f"{img_b64}\n")
                img_offsets[image_path] = img_offset
            ann_offset = ann_f.tell()
            ann_offsets.append(ann_offset)
            ann_f.write(f"{img_offsets[image_path]}\t{json.dumps(annotation, ensure_ascii=False)}\n")
            idx_f.write(f"{ann_offset}\n")
    print(f"Written {len(ann_offsets)} samples ({len(img_offsets)} images) to {output_dir}/{prefix}.*")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/path/to/Anvil/data/output/gsa_flywheel_exp/verified")
    parser.add_argument("--output_dir", default="/path/to/datasets/dataflywheel/output/rex_omni_tsv")
    parser.add_argument("--max_samples_per_dim", type=int, default=None)
    args = parser.parse_args()

    all_samples = []
    for dim in CONVERTERS:
        print(f"Processing {dim}...")
        samples = process_dimension(args.input_dir, dim, args.max_samples_per_dim)
        print(f"  → {len(samples)} samples")
        all_samples.extend(samples)

    grounding = [(p, a) for p, a in all_samples if "boxes" in a]
    vqa = [(p, a) for p, a in all_samples if "conversations" in a]

    if grounding:
        write_tsv_dataset(grounding, args.output_dir, "embodiedforge_grounding")
    if vqa:
        write_tsv_dataset(vqa, args.output_dir, "embodiedforge_vqa")


if __name__ == "__main__":
    main()
