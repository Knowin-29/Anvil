#!/usr/bin/env python3
"""
Convert GSA_volc (HuggingFace flywheel dataset) into ms-swift training jsonl
with absolute image paths, image-existence filtering, dedup and per-dimension
statistics.

The GSA_volc annotations store relative image paths under the `images` field
(e.g. "images/bridgev2/episode_001234.png"). At training time we must join them
with IMAGE_ROOT so swift can read the files.

Usage:
    python trains/convert_gsa_volc.py \
        --volc_root /mnt/data/huggingface_downloads/datasets/GSA_volc \
        --out_sft trains/gsa_volc_sft.jsonl \
        --out_grpo trains/gsa_volc_grpo.jsonl \
        --report trains/gsa_volc_report.json
"""
import argparse
import json
import os
from collections import Counter


def parse_dimension(record: dict) -> str:
    """Extract dimension tag from a record (top-level field or assistant JSON)."""
    if record.get("dimension"):
        return record["dimension"]
    msgs = record.get("messages", [])
    asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    asst = asst.strip()
    if asst.startswith("{"):
        try:
            return json.loads(asst).get("dimension", "unknown")
        except json.JSONDecodeError:
            return "unknown"
    return "unknown"


def is_grounding_full_image_bbox(record: dict, dim: str) -> bool:
    """True if a G_grounding sample uses a degenerate full-image bbox.

    From the audit (docs/GSA_volc_data_analysis.md): full-image bboxes are
    semantically fine for attribute/state dims (background surfaces) but are
    noise for the grounding/localization task. Only drop them for G_grounding.
    """
    if dim != "G_grounding":
        return False
    msgs = record.get("messages", [])
    asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    obj = None
    s = asst.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return False
    if not isinstance(obj, dict):
        return False
    bbox = obj.get("bbox")
    size = obj.get("image_size")
    if not (isinstance(bbox, list) and len(bbox) == 4 and isinstance(size, list) and len(size) == 2):
        return False
    W, H = size
    if not (W and H):
        return False
    try:
        area = (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1]))
    except (ValueError, TypeError):
        return False
    return area / (W * H) >= 0.98


def valid_record(record: dict, require_assistant: bool) -> bool:
    msgs = record.get("messages")
    if not msgs or not isinstance(msgs, list) or len(msgs) < 2:
        return False
    if not record.get("images"):
        return False
    has_user = any(m.get("role") == "user" for m in msgs)
    if not has_user:
        return False
    if require_assistant:
        asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
        if not asst.strip():
            return False
    return True


def abs_images(record: dict, image_root: str):
    """Return absolute image paths; None if any image is missing on disk.
    
    Three-level fallback:
      1. Direct join: image_root/rel (works for bridgev2 etc.)
      2. Basename fallback: image_root/<source_prefix>/basename (changeit/benchmark/kitchen etc.)
      3. Source-name remap: image_root/<mapped_prefix>/basename (household_where2place?other)
    """
    SOURCE_REMAP = {"household_where2place": "other"}
    
    out = []
    for rel in record.get("images", []):
        if os.path.isabs(rel):
            if os.path.isfile(rel):
                out.append(rel)
                continue
            return None
        
        # Level 1: direct join
        full = os.path.join(image_root, rel)
        if os.path.isfile(full):
            out.append(full)
            continue
        
        # Level 2: basename in source prefix dir
        parts = rel.split("/")
        if len(parts) >= 2:
            src_prefix = "/".join(parts[:2])  # e.g. "images/changeit"
            base = os.path.basename(rel)
            cand = os.path.join(image_root, src_prefix, base)
            if os.path.isfile(cand):
                out.append(cand)
                continue
        
        # Level 3: source-name remap
        if len(parts) >= 2:
            src_name = parts[1]
            mapped = SOURCE_REMAP.get(src_name)
            if mapped:
                cand = os.path.join(image_root, "images", mapped, base)
                if os.path.isfile(cand):
                    out.append(cand)
                    continue
        
        # All levels failed
        return None
    return out


def convert_file(in_path, image_root, require_assistant):
    """Yield (converted_record, dimension) for valid, image-present records.
    Returns generator plus collects stats via closure list."""
    stats = {"total": 0, "invalid": 0, "missing_image": 0, "dup": 0, "kept": 0,
             "by_dim": Counter(), "missing_by_dim": Counter()}
    seen = set()
    out = []
    if not os.path.isfile(in_path):
        return out, stats
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid"] += 1
                continue
            if not valid_record(rec, require_assistant):
                stats["invalid"] += 1
                continue
            dim = parse_dimension(rec)
            if is_grounding_full_image_bbox(rec, dim):
                stats["grounding_fullimg"] = stats.get("grounding_fullimg", 0) + 1
                continue
            imgs = abs_images(rec, image_root)
            if imgs is None:
                stats["missing_image"] += 1
                stats["missing_by_dim"][dim] += 1
                continue
            # dedup by (first image, assistant response or user content)
            msgs = rec.get("messages", [])
            asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
            user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            key = (imgs[0], asst or user)
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            new_rec = {"messages": rec["messages"], "images": imgs}
            if rec.get("dimension"):
                new_rec["dimension"] = rec["dimension"]
            out.append((new_rec, dim))
            stats["kept"] += 1
            stats["by_dim"][dim] += 1
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volc_root", default="/mnt/data/huggingface_downloads/datasets/GSA_volc")
    ap.add_argument("--out_sft", default="trains/gsa_volc_sft.jsonl")
    ap.add_argument("--out_grpo", default="trains/gsa_volc_grpo.jsonl")
    ap.add_argument("--report", default="trains/gsa_volc_report.json")
    args = ap.parse_args()

    sft_in = os.path.join(args.volc_root, "annotations", "sft_train.jsonl")
    grpo_in = os.path.join(args.volc_root, "annotations", "grpo_train.jsonl")

    print(f"VOLC_ROOT: {args.volc_root}")
    print(f"SFT in:    {sft_in}")
    print(f"GRPO in:   {grpo_in}")
    print()

    print("Converting SFT ...")
    sft_recs, sft_stats = convert_file(sft_in, args.volc_root, require_assistant=True)
    with open(args.out_sft, "w") as f:
        for rec, _ in sft_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  SFT kept {sft_stats['kept']} / {sft_stats['total']} "
          f"(missing_img={sft_stats['missing_image']}, dup={sft_stats['dup']}, "
          f"invalid={sft_stats['invalid']})")

    print("Converting GRPO ...")
    grpo_recs, grpo_stats = convert_file(grpo_in, args.volc_root, require_assistant=False)
    with open(args.out_grpo, "w") as f:
        for rec, _ in grpo_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  GRPO kept {grpo_stats['kept']} / {grpo_stats['total']} "
          f"(missing_img={grpo_stats['missing_image']}, dup={grpo_stats['dup']}, "
          f"invalid={grpo_stats['invalid']})")

    report = {
        "volc_root": args.volc_root,
        "sft": {
            "total": sft_stats["total"], "kept": sft_stats["kept"],
            "missing_image": sft_stats["missing_image"], "dup": sft_stats["dup"],
            "grounding_fullimg_dropped": sft_stats.get("grounding_fullimg", 0),
            "invalid": sft_stats["invalid"],
            "by_dim": dict(sft_stats["by_dim"]),
            "missing_by_dim": dict(sft_stats["missing_by_dim"]),
        },
        "grpo": {
            "total": grpo_stats["total"], "kept": grpo_stats["kept"],
            "missing_image": grpo_stats["missing_image"], "dup": grpo_stats["dup"],
            "invalid": grpo_stats["invalid"],
            "by_dim": dict(grpo_stats["by_dim"]),
        },
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {args.report}")
    print("\nSFT by dimension:")
    for k, v in sft_stats["by_dim"].most_common():
        print(f"  {v:>7}  {k}")


if __name__ == "__main__":
    main()
