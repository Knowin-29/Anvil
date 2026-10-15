#!/usr/bin/env python3
"""Clean data sources — template dedup, coordinate validation, think-with-point/bbox checks.

Implements the cleaning pipeline addressing training-side feedback (F1-F4):
- F1: Template redundancy dedup (same annotation → multiple prompt templates)
- F3: bbox/point/text format balance monitoring
- F4: Coordinate legality validation (bbox geometry, point bounds, duplicates)
- Think with point/bbox direction: validate output format consistency

Usage:
    python scripts/clean_data_sources.py --input data/output/gsa_flywheel_exp/verified --dry-run
    python scripts/clean_data_sources.py --input data/output/gsa_flywheel_exp/verified --execute
    python scripts/clean_data_sources.py --input /path/to/datasets/GSA_volc/annotations --execute
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean_sources")

NORM_RANGE = 1000


def validate_bbox(bbox, norm_range: int = NORM_RANGE) -> Tuple[bool, str]:
    """Validate bbox coordinates for think-with-point/bbox compliance."""
    if bbox is None:
        return True, "no_bbox"
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "bbox_not_4_elements"
    x1, y1, x2, y2 = bbox
    if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
        return False, "bbox_non_numeric"
    if x2 <= x1:
        return False, "bbox_x2_le_x1"
    if y2 <= y1:
        return False, "bbox_y2_le_y1"
    if any(v < 0 for v in [x1, y1, x2, y2]):
        return False, "bbox_negative"
    if any(v > norm_range for v in [x1, y1, x2, y2]):
        return False, "bbox_exceeds_range"
    area = (x2 - x1) * (y2 - y1)
    max_area = norm_range * norm_range * 0.95
    if area > max_area:
        return False, "bbox_area_gt_95pct"
    min_area = norm_range * norm_range * 0.0005
    if area < min_area:
        return False, "bbox_area_too_small"
    return True, "ok"


def validate_point(point, norm_range: int = NORM_RANGE) -> Tuple[bool, str]:
    """Validate a single point coordinate."""
    if point is None:
        return True, "no_point"
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return False, "point_not_2d"
    x, y = point
    if not all(isinstance(v, (int, float)) for v in [x, y]):
        return False, "point_non_numeric"
    if x < 0 or x > norm_range or y < 0 or y > norm_range:
        return False, "point_out_of_range"
    return True, "ok"


def validate_points_set(points: List, norm_range: int = NORM_RANGE) -> Tuple[bool, str]:
    """Validate a set of points (affordance/grasp points)."""
    if not points:
        return True, "no_points"
    if len(points) > 15:
        return False, "too_many_points_gt15"
    seen = set()
    for pt in points:
        valid, reason = validate_point(pt, norm_range)
        if not valid:
            return False, f"point_invalid:{reason}"
        pt_key = (round(pt[0], 1), round(pt[1], 1))
        if pt_key in seen:
            return False, "duplicate_point"
        seen.add(pt_key)
    return True, "ok"


def compute_annotation_id(sample: dict) -> str:
    """Compute a stable annotation_id for template dedup (F1)."""
    img = sample.get("image_path", "")
    bbox = sample.get("bbox")
    obj_class = sample.get("object_class", "")
    dimension = sample.get("dimension") or sample.get("pipeline", "")
    key_parts = [img, str(bbox), obj_class, dimension]
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()[:16]


def detect_format_type(sample: dict) -> str:
    """Classify output format: bbox / point / text."""
    if sample.get("bbox"):
        return "bbox"
    if sample.get("grasp_point_2d") or sample.get("points"):
        return "point"
    return "text"


def clean_jsonl_file(
    input_path: Path,
    output_path: Optional[Path],
    execute: bool = False,
) -> Dict:
    """Clean a single JSONL file. Returns stats."""
    stats = {
        "total": 0,
        "kept": 0,
        "removed": 0,
        "reasons": Counter(),
        "format_dist": Counter(),
        "dimension_dist": Counter(),
        "dedup_removed": 0,
    }

    samples = []
    seen_annotations = set()

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                stats["removed"] += 1
                stats["reasons"]["json_parse_error"] += 1
                continue

            # Template dedup (F1): same annotation different templates
            ann_id = compute_annotation_id(sample)
            if ann_id in seen_annotations:
                stats["removed"] += 1
                stats["reasons"]["template_dedup"] += 1
                stats["dedup_removed"] += 1
                continue
            seen_annotations.add(ann_id)

            # Coordinate validation (F4)
            bbox = sample.get("bbox")
            if bbox:
                valid, reason = validate_bbox(bbox)
                if not valid:
                    stats["removed"] += 1
                    stats["reasons"][f"bbox:{reason}"] += 1
                    continue

            grasp_pt = sample.get("grasp_point_2d")
            if grasp_pt:
                valid, reason = validate_point(grasp_pt)
                if not valid:
                    stats["removed"] += 1
                    stats["reasons"][f"grasp_point:{reason}"] += 1
                    continue

            # Validate placement zones if present
            place_zone = sample.get("placement_zone")
            if place_zone and isinstance(place_zone, (list, tuple)) and len(place_zone) == 4:
                valid, reason = validate_bbox(place_zone)
                if not valid:
                    stats["removed"] += 1
                    stats["reasons"][f"place_zone:{reason}"] += 1
                    continue

            # Track format distribution (F3)
            fmt = detect_format_type(sample)
            stats["format_dist"][fmt] += 1

            # Track dimension distribution (F2)
            dim = sample.get("dimension") or sample.get("pipeline", "unknown")
            stats["dimension_dist"][dim] += 1

            # Attach annotation_id for downstream dedup
            sample["annotation_id"] = ann_id
            samples.append(sample)
            stats["kept"] += 1

    if execute and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return stats


def clean_directory(input_dir: Path, output_dir: Optional[Path], execute: bool) -> Dict:
    """Clean all JSONL files in a directory tree."""
    total_stats = {
        "files_processed": 0,
        "total_samples": 0,
        "kept": 0,
        "removed": 0,
        "reasons": Counter(),
        "format_dist": Counter(),
        "dimension_dist": Counter(),
    }

    jsonl_files = sorted(input_dir.rglob("*.jsonl"))
    jsonl_files = [f for f in jsonl_files if ".bak_" not in f.name]

    for jf in jsonl_files:
        rel = jf.relative_to(input_dir)
        out_path = (output_dir / rel) if output_dir else None

        file_stats = clean_jsonl_file(jf, out_path, execute)
        total_stats["files_processed"] += 1
        total_stats["total_samples"] += file_stats["total"]
        total_stats["kept"] += file_stats["kept"]
        total_stats["removed"] += file_stats["removed"]
        total_stats["reasons"] += file_stats["reasons"]
        total_stats["format_dist"] += file_stats["format_dist"]
        total_stats["dimension_dist"] += file_stats["dimension_dist"]

        if file_stats["removed"] > 0:
            logger.info(f"  {rel}: {file_stats['kept']}/{file_stats['total']} kept "
                       f"(-{file_stats['removed']})")

    return total_stats


def main():
    parser = argparse.ArgumentParser(description="Clean GSA data sources")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, help="Output directory (default: overwrite in-place)")
    parser.add_argument("--dry-run", action="store_true", help="Only report, no changes")
    parser.add_argument("--execute", action="store_true", help="Execute cleaning")
    parser.add_argument("--report", type=str, help="Save JSON report to path")
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        logger.error("Specify --dry-run or --execute")
        return

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    execute = args.execute

    if not input_dir.exists():
        logger.error(f"Input not found: {input_dir}")
        return

    logger.info(f"Cleaning: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Mode: {'EXECUTE' if execute else 'DRY-RUN'}")

    stats = clean_directory(input_dir, output_dir if execute else None, execute)

    print(f"\n{'='*60}")
    print(f"  Cleaning Report")
    print(f"{'='*60}")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Total samples:   {stats['total_samples']}")
    print(f"  Kept:            {stats['kept']} ({100*stats['kept']/max(1,stats['total_samples']):.1f}%)")
    print(f"  Removed:         {stats['removed']} ({100*stats['removed']/max(1,stats['total_samples']):.1f}%)")
    print(f"\n  Removal reasons:")
    for reason, count in stats["reasons"].most_common():
        print(f"    {reason}: {count}")
    print(f"\n  Format distribution (F3 monitor):")
    total_fmt = sum(stats["format_dist"].values())
    for fmt, count in stats["format_dist"].most_common():
        print(f"    {fmt}: {count} ({100*count/max(1,total_fmt):.1f}%)")
    print(f"\n  Dimension distribution (F2 monitor):")
    total_dim = sum(stats["dimension_dist"].values())
    for dim, count in stats["dimension_dist"].most_common():
        print(f"    {dim}: {count} ({100*count/max(1,total_dim):.1f}%)")

    if args.report:
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input": str(input_dir),
            "mode": "execute" if execute else "dry-run",
            **stats,
            "reasons": dict(stats["reasons"]),
            "format_dist": dict(stats["format_dist"]),
            "dimension_dist": dict(stats["dimension_dist"]),
        }
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
