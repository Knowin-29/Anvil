#!/usr/bin/env python3
"""
Convert GSA Flywheel verified data to ms-swift training format.

Input:  verified/ directory with per-dimension JSONL files
Output: gsa_sft_train.jsonl (SFT format) + gsa_grpo_train.jsonl (GRPO prompt-only)
        + data_report.json (缺失统计, 不再静默跳过)

ms-swift SFT format:
  {"messages": [...], "images": ["path"]}

ms-swift GRPO format (prompt only, no assistant response):
  {"messages": [{"role":"user","content":"<image>\\n..."}], "images": ["path"]}

多云路径可移植:
  --image_roots 指定逗号分隔的多个根目录, 按顺序 fallback 查找图片
  输出 jsonl 中 images 存相对路径 (相对于匹配到的 root)
"""
import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List


VERIFIED_DIR = "/path/to/Anvil/data/output/gsa_flywheel_exp/verified"

SYSTEM_PROMPT = (
    "You are an embodied perception assistant specialized in indoor scene understanding. "
    "You can detect objects, understand spatial relationships, identify object attributes and states, "
    "and reason about robot manipulation affordances. Respond precisely with structured JSON."
)


def convert_grounding(sample: dict) -> Optional[Tuple[str, str]]:
    bbox = sample.get("bbox")
    obj_class = sample.get("object_class", "")
    confidence = sample.get("confidence", 0.9)
    if not bbox or not obj_class:
        return None
    query = f"Detect the {obj_class} in the image and provide its bounding box coordinates [x1,y1,x2,y2]."
    response = json.dumps({
        "object": obj_class, "bbox": bbox, "confidence": round(float(confidence), 2)
    }, ensure_ascii=False)
    return query, response


def convert_spatial(sample: dict) -> Optional[Tuple[str, str]]:
    subject = sample.get("subject", "")
    predicate = sample.get("predicate", "")
    obj = sample.get("object", "")
    if not all([subject, predicate, obj]):
        return None
    query = f"What is the spatial relationship between the {subject} and the {obj}?"
    response = json.dumps({
        "subject": subject, "predicate": predicate, "object": obj,
        "description": f"The {subject} is {predicate} the {obj}."
    }, ensure_ascii=False)
    return query, response


def convert_identity_attr(sample: dict) -> Optional[Tuple[str, str]]:
    obj_class = sample.get("object_class", "")
    attributes = sample.get("attributes", {})
    bbox = sample.get("bbox")
    if not obj_class or not attributes:
        return None
    query = f"Identify the visual attributes of the {obj_class} in the image."
    resp = {"object": obj_class, "attributes": attributes}
    if bbox:
        resp["bbox"] = bbox
    return query, json.dumps(resp, ensure_ascii=False)


def convert_obj_state(sample: dict) -> Optional[Tuple[str, str]]:
    obj_class = sample.get("object_class", "")
    state_type = sample.get("state_type", "")
    state_value = sample.get("state_value", "")
    if not obj_class or not state_value:
        return None
    query = f"What is the current state of the {obj_class}?"
    response = json.dumps({
        "object": obj_class, "state_type": state_type, "state_value": state_value
    }, ensure_ascii=False)
    return query, response


def convert_action_precondition(sample: dict) -> Optional[Tuple[str, str]]:
    obj_class = sample.get("object_class", "")
    action_type = sample.get("action_type", "manipulate")
    can_do_now = sample.get("can_do_now", True)
    blocked_by = sample.get("blocked_by", [])
    need_pre_action = sample.get("need_pre_action", "none")
    if not obj_class:
        return None
    query = f"Can a robot directly {action_type} the {obj_class}? Analyze preconditions."
    response = json.dumps({
        "object": obj_class, "action_type": action_type,
        "can_do_now": can_do_now, "blocked_by": blocked_by,
        "need_pre_action": need_pre_action
    }, ensure_ascii=False)
    return query, response


def convert_action_interface(sample: dict) -> Optional[Tuple[str, str]]:
    obj_class = sample.get("object_class", "")
    action_type = sample.get("action_type", "")
    grasp_point = sample.get("grasp_point_2d", "")
    jaw_axis = sample.get("jaw_axis_2d", "")
    if not obj_class or not action_type:
        return None
    query = f"How should a robot grasp the {obj_class}? Provide the action interface."
    resp = {"object": obj_class, "action_type": action_type}
    if grasp_point:
        resp["grasp_point_2d"] = grasp_point
    if jaw_axis:
        resp["jaw_axis_2d"] = jaw_axis
    return query, json.dumps(resp, ensure_ascii=False)


CONVERTERS = {
    "G_grounding": convert_grounding,
    "S_spatial": convert_spatial,
    "A_identity_attr": convert_identity_attr,
    "A_obj_state": convert_obj_state,
    "A_action_precondition": convert_action_precondition,
    "A_action_interface": convert_action_interface,
}


def resolve_image_path(raw_path: str, image_roots: List[str]) -> Optional[Tuple[str, str]]:
    """Try to resolve an image path against multiple roots.
    
    Returns (resolved_absolute_path, relative_path) or None if not found anywhere.
    Supports:
      1. Direct absolute path exists -> use as-is, compute relative to matching root
      2. Try stripping known prefixes and re-rooting to each image_root
    """
    if not raw_path:
        return None

    # Case 1: file exists at the given path directly
    if os.path.isfile(raw_path):
        for root in image_roots:
            if raw_path.startswith(root):
                rel = os.path.relpath(raw_path, root)
                return raw_path, rel
        return raw_path, raw_path

    # Case 2: try stripping known prefixes and checking under each root
    known_prefixes = [
        "/path/to/workspace/foundation/",
        "/path/to/environment/",
        "/path/to/datasets/",
        "/path/to/datasets/",
        "/path/to/storage/contributor/",
        "/mnt/data/datasets/",
    ]
    
    for prefix in known_prefixes:
        if raw_path.startswith(prefix):
            suffix = raw_path[len(prefix):]
            for root in image_roots:
                candidate = os.path.join(root, suffix)
                if os.path.isfile(candidate):
                    rel = os.path.relpath(candidate, root)
                    return candidate, rel
    
    # Case 3: try the basename under each root recursively (last resort, expensive)
    # Skip this for performance - just report as missing
    return None


def load_dimension_data(verified_dir: str, dimension: str) -> list:
    dim_dir = os.path.join(verified_dir, dimension)
    if not os.path.isdir(dim_dir):
        return []
    samples = []
    for jf in sorted(Path(dim_dir).glob("*.jsonl")):
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def build_sft_record(query: str, response: str, image_path: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{query}"},
            {"role": "assistant", "content": response},
        ],
        "images": [image_path],
    }


def build_grpo_record(query: str, image_path: str, dimension: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{query}"},
        ],
        "images": [image_path],
        "dimension": dimension,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert flywheel data to ms-swift format")
    parser.add_argument("--input_dir", default=VERIFIED_DIR)
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / "trains"))
    parser.add_argument("--grpo_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_roots", type=str, default=None,
                        help="Comma-separated image root directories for path resolution. "
                             "Falls back to IMAGE_ROOTS env var.")
    parser.add_argument("--use_relative_paths", action="store_true", default=False,
                        help="Store relative paths in output (portable across clouds)")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve image roots
    roots_str = args.image_roots or os.environ.get("IMAGE_ROOTS", "")
    image_roots = [r.strip() for r in roots_str.replace(",", ":").split(":") if r.strip()]
    if not image_roots:
        image_roots = ["/path/to/workspace/foundation", "/path/to/datasets"]
    
    print(f"Image roots ({len(image_roots)}):")
    for r in image_roots:
        exists = os.path.isdir(r)
        print(f"  {'OK' if exists else 'MISSING'} {r}")

    sft_records = []
    grpo_records = []
    
    # Detailed tracking
    stats = {}
    missing_report = defaultdict(lambda: {"total": 0, "missing": 0, "missing_paths": []})

    for dimension, converter in CONVERTERS.items():
        samples = load_dimension_data(args.input_dir, dimension)
        dim_count = 0
        dim_missing = 0
        dim_convert_fail = 0

        for sample in samples:
            raw_path = sample.get("image_path", "")
            if not raw_path:
                dim_convert_fail += 1
                continue
            
            resolved = resolve_image_path(raw_path, image_roots)
            if resolved is None:
                dim_missing += 1
                # Track missing by source directory
                src_dir = os.path.dirname(raw_path)[:80]
                key = f"{dimension}|{src_dir}"
                missing_report[key]["total"] += 1
                missing_report[key]["missing"] += 1
                if len(missing_report[key]["missing_paths"]) < 3:
                    missing_report[key]["missing_paths"].append(raw_path)
                continue
            
            abs_path, rel_path = resolved
            use_path = rel_path if args.use_relative_paths else abs_path
            
            result = converter(sample)
            if result is None:
                dim_convert_fail += 1
                continue
            query, response = result
            sft_records.append(build_sft_record(query, response, use_path))
            if random.random() < args.grpo_ratio:
                grpo_records.append(build_grpo_record(query, use_path, dimension))
            dim_count += 1

        stats[dimension] = {
            "valid": dim_count,
            "missing_image": dim_missing,
            "convert_fail": dim_convert_fail,
            "total_annotations": len(samples),
        }
        
        miss_pct = (dim_missing / max(len(samples), 1)) * 100
        print(f"  [{dimension}] {dim_count} valid | {dim_missing} missing ({miss_pct:.1f}%) | {dim_convert_fail} convert_fail | {len(samples)} total")

    random.shuffle(sft_records)
    random.shuffle(grpo_records)

    sft_path = os.path.join(args.output_dir, "gsa_sft_train.jsonl")
    with open(sft_path, "w") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    grpo_path = os.path.join(args.output_dir, "gsa_grpo_train.jsonl")
    with open(grpo_path, "w") as f:
        for rec in grpo_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Generate data report
    total_valid = sum(s["valid"] for s in stats.values())
    total_missing = sum(s["missing_image"] for s in stats.values())
    total_annotations = sum(s["total_annotations"] for s in stats.values())
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "image_roots": image_roots,
        "use_relative_paths": args.use_relative_paths,
        "summary": {
            "total_annotations": total_annotations,
            "total_valid": total_valid,
            "total_missing_image": total_missing,
            "missing_rate": f"{(total_missing / max(total_annotations, 1)) * 100:.1f}%",
            "sft_samples": len(sft_records),
            "grpo_samples": len(grpo_records),
        },
        "per_dimension": stats,
        "missing_by_source": {k: v for k, v in sorted(missing_report.items(), key=lambda x: -x[1]["missing"])[:20]},
    }
    
    report_path = os.path.join(args.output_dir, "data_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SFT samples:     {len(sft_records):>6} -> {sft_path}")
    print(f"GRPO prompts:    {len(grpo_records):>6} -> {grpo_path}")
    print(f"Data report:            -> {report_path}")
    print(f"{'='*60}")
    print(f"Total annotations: {total_annotations}")
    print(f"Total valid:       {total_valid} ({total_valid/max(total_annotations,1)*100:.1f}%)")
    print(f"Total missing:     {total_missing} ({total_missing/max(total_annotations,1)*100:.1f}%)")
    if total_missing > 0:
        print(f"\n  WARNING: {total_missing} samples skipped due to missing images!")
        print(f"  Check data_report.json for details.")
        print(f"  Top missing sources:")
        for key, info in sorted(missing_report.items(), key=lambda x: -x[1]["missing"])[:5]:
            print(f"    {key}: {info['missing']} missing")


if __name__ == "__main__":
    main()
