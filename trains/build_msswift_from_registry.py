#!/usr/bin/env python3
"""
Build ms-swift training data from sources_registry.yaml.

Reads the registry, filters available_aliyun sources, applies ratio weighting,
and produces three output files:
  - gsa_sft_train.jsonl  (full messages: system+user+assistant)
  - gsa_grpo_train.jsonl (prompt-only: system+user, with dimension tag)
  - gsa_opd_train.jsonl  (on-policy prompt: same as GRPO, for OPD distillation)
  + data_report.json     (per-source/dimension statistics)

Usage:
    python trains/build_msswift_from_registry.py [--registry data/sources_registry.yaml]
    python trains/build_msswift_from_registry.py --grpo_ratio 0.3 --opd_ratio 0.2
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml


SYSTEM_PROMPT = (
    "You are an embodied perception assistant specialized in indoor scene understanding. "
    "You can detect objects, understand spatial relationships, identify object attributes and states, "
    "and reason about robot manipulation affordances. Respond precisely with structured output."
)


def load_registry(registry_path: str) -> dict:
    with open(registry_path) as f:
        return yaml.safe_load(f)


def validate_msswift_record(record: dict) -> bool:
    """Check that a record has valid ms-swift structure."""
    msgs = record.get("messages")
    if not msgs or not isinstance(msgs, list):
        return False
    if len(msgs) < 2:
        return False
    # Must have images field with at least one non-empty path
    images = record.get("images", [])
    if not images or not images[0]:
        return False
    has_user = any(m.get("role") == "user" for m in msgs)
    has_assistant = any(m.get("role") == "assistant" for m in msgs)
    return has_user and has_assistant


def quality_filter(record: dict) -> bool:
    """Filter out low-quality samples that would harm training."""
    msgs = record.get("messages", [])
    asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
    # Drop truly empty responses only
    if not asst.strip():
        return False
    return True


def dedup_by_image_response(records: List[dict]) -> List[dict]:
    """Deduplicate records with same (image, assistant_response).
    
    For datasets where the same annotation is repeated with multiple prompt
    templates, keep only one random variant per unique (image, response) pair.
    """
    seen = {}
    for rec in records:
        imgs = rec.get("images", [""])
        img = imgs[0] if imgs else ""
        msgs = rec.get("messages", [])
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        key = (img, asst)
        if key not in seen:
            seen[key] = rec
    return list(seen.values())


def check_image_exists(record: dict, sample_rate: float = 0.01) -> Optional[bool]:
    """Spot-check image existence at sample_rate probability."""
    if random.random() > sample_rate:
        return None
    images = record.get("images", [])
    if not images:
        return None
    img_path = images[0]
    return os.path.isfile(img_path)


def make_grpo_record(record: dict, dimension: str) -> dict:
    """Strip assistant response, keep only prompt (system+user)."""
    msgs = record.get("messages", [])
    prompt_msgs = []
    for m in msgs:
        if m.get("role") == "assistant":
            break
        prompt_msgs.append(m)
    if not prompt_msgs:
        prompt_msgs = [{"role": "user", "content": "<image>\nDescribe this scene."}]
    return {
        "messages": prompt_msgs,
        "images": record.get("images", []),
        "dimension": dimension,
    }


def load_source_records(source: dict) -> List[dict]:
    """Load records from a msswift_ready source."""
    path = source["path"]
    if not os.path.isfile(path):
        print(f"  [WARN] File not found: {path}")
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def apply_ratio(records: List[dict], ratio: float, seed: int) -> List[dict]:
    """Apply ratio weighting: ratio >= 1 means repeat, < 1 means subsample."""
    if ratio <= 0:
        return []
    if ratio == 1.0:
        return records
    rng = random.Random(seed)
    if ratio < 1.0:
        k = int(len(records) * ratio)
        return rng.sample(records, min(k, len(records)))
    else:
        full_copies = int(ratio)
        frac = ratio - full_copies
        result = records * full_copies
        if frac > 0:
            k = int(len(records) * frac)
            result += rng.sample(records, min(k, len(records)))
        return result


def main():
    parser = argparse.ArgumentParser(description="Build ms-swift data from registry")
    parser.add_argument("--registry", default=None,
                        help="Path to sources_registry.yaml")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: trains/)")
    parser.add_argument("--grpo_ratio", type=float, default=0.3,
                        help="Fraction of SFT data to also emit as GRPO prompts")
    parser.add_argument("--opd_ratio", type=float, default=0.2,
                        help="Fraction of SFT data to emit as OPD on-policy prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_raw", action="store_true",
                        help="Skip format:raw sources (they need extract scripts)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve paths relative to GSA_Forge root
    script_dir = Path(__file__).resolve().parent
    forge_root = script_dir.parent
    registry_path = args.registry or str(forge_root / "configs/data/sources_clean.yaml")
    output_dir = args.output_dir or str(script_dir)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Registry: {registry_path}")
    print(f"Output:   {output_dir}")
    print(f"GRPO ratio: {args.grpo_ratio}, OPD ratio: {args.opd_ratio}")
    print()

    registry = load_registry(registry_path)
    sources = registry.get("sources", [])

    all_sft = []
    all_grpo = []
    all_opd = []
    stats = {}
    total_missing_images = 0
    total_checked = 0

    for src in sources:
        if not src.get("available_aliyun", False):
            continue
        if src.get("ratio", 0) <= 0:
            continue
        if src.get("format") == "raw":
            if args.skip_raw:
                print(f"  [SKIP-RAW] {src['name']}")
                continue
            else:
                print(f"  [SKIP-RAW] {src['name']} (format=raw, use dedicated extract script)")
                stats[src["name"]] = {
                    "status": "skipped_raw",
                    "est_samples": src.get("est_samples", 0),
                    "notes": "Needs extract_log_to_swift.py",
                }
                continue

        name = src["name"]
        ratio = src.get("ratio", 1.0)
        dims = src.get("dimensions", [])
        dim_str = dims[0] if dims else "unknown"

        print(f"  Loading [{name}] ratio={ratio} ...", end=" ", flush=True)
        t0 = time.time()
        records = load_source_records(src)
        if not records:
            print("EMPTY")
            stats[name] = {"loaded": 0, "after_ratio": 0, "status": "empty"}
            continue

        # --- Quality filtering ---
        before_filter = len(records)
        records = [r for r in records if validate_msswift_record(r) and quality_filter(r)]
        filtered_out = before_filter - len(records)

        # --- Deduplication (if specified in registry) ---
        dedup_key = src.get("dedup_key", "")
        deduped_out = 0
        if dedup_key == "image_response":
            before_dedup = len(records)
            records = dedup_by_image_response(records)
            deduped_out = before_dedup - len(records)

        # Spot-check images
        src_missing = 0
        src_checked = 0
        for r in records[:200]:
            result = check_image_exists(r, sample_rate=1.0)
            if result is not None:
                src_checked += 1
                if not result:
                    src_missing += 1
        total_missing_images += src_missing
        total_checked += src_checked

        # Apply ratio
        weighted = apply_ratio(records, ratio, args.seed + hash(name) % 10000)

        elapsed = time.time() - t0
        extra = ""
        if filtered_out > 0:
            extra += f" filtered={filtered_out}"
        if deduped_out > 0:
            extra += f" deduped={deduped_out}"
        print(f"{before_filter} loaded -> {len(records)} clean -> {len(weighted)} after ratio ({elapsed:.1f}s){extra}")

        # Build SFT records (already in format)
        all_sft.extend(weighted)

        # Build GRPO (prompt-only subset)
        grpo_k = int(len(weighted) * args.grpo_ratio)
        grpo_subset = random.sample(weighted, min(grpo_k, len(weighted)))
        for r in grpo_subset:
            all_grpo.append(make_grpo_record(r, dim_str))

        # Build OPD (on-policy prompt subset)
        opd_k = int(len(weighted) * args.opd_ratio)
        opd_subset = random.sample(weighted, min(opd_k, len(weighted)))
        for r in opd_subset:
            all_opd.append(make_grpo_record(r, dim_str))

        stats[name] = {
            "loaded": before_filter,
            "filtered_out": filtered_out,
            "deduped_out": deduped_out,
            "clean": len(records),
            "after_ratio": len(weighted),
            "grpo_samples": len(grpo_subset),
            "opd_samples": len(opd_subset),
            "dimensions": dims,
            "ratio": ratio,
            "image_check": f"{src_checked - src_missing}/{src_checked} OK",
            "general": src.get("general", False),
        }

    # Shuffle all
    print(f"\nShuffling {len(all_sft)} SFT, {len(all_grpo)} GRPO, {len(all_opd)} OPD ...")
    random.shuffle(all_sft)
    random.shuffle(all_grpo)
    random.shuffle(all_opd)

    # Write outputs
    sft_path = os.path.join(output_dir, "gsa_sft_train.jsonl")
    grpo_path = os.path.join(output_dir, "gsa_grpo_train.jsonl")
    opd_path = os.path.join(output_dir, "gsa_opd_train.jsonl")

    for path, data, label in [
        (sft_path, all_sft, "SFT"),
        (grpo_path, all_grpo, "GRPO"),
        (opd_path, all_opd, "OPD"),
    ]:
        with open(path, "w") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {label}: {len(data):>8} samples -> {path}")

    # Report
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "registry": registry_path,
        "seed": args.seed,
        "grpo_ratio": args.grpo_ratio,
        "opd_ratio": args.opd_ratio,
        "summary": {
            "sft_total": len(all_sft),
            "grpo_total": len(all_grpo),
            "opd_total": len(all_opd),
            "sources_loaded": sum(1 for v in stats.values() if v.get("status") != "skipped_raw"),
            "sources_skipped_raw": sum(1 for v in stats.values() if v.get("status") == "skipped_raw"),
            "image_spot_check": f"{total_checked - total_missing_images}/{total_checked} OK",
        },
        "per_source": stats,
    }
    report_path = os.path.join(output_dir, "data_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {report_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SFT total:   {len(all_sft):>10}")
    print(f"  GRPO total:  {len(all_grpo):>10}")
    print(f"  OPD total:   {len(all_opd):>10}")
    print(f"  Image check: {total_checked - total_missing_images}/{total_checked} OK")
    if total_missing_images > 0:
        miss_pct = total_missing_images / max(total_checked, 1) * 100
        print(f"  WARNING: {total_missing_images} images missing ({miss_pct:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
