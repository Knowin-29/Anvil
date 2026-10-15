#!/usr/bin/env python3
"""Manage scene tags — scan dimension_sources and verified data to maintain tag registry.

Scans dimension_sources/*.txt (col 3) and verified/*.jsonl (scene_type field),
aggregates tag statistics, updates manifest.json scene_types list.
Supports dynamic tag discovery: unknown tags auto-registered.

Usage:
    python scripts/manage_scene_tags.py [--update-manifest]
    python scripts/manage_scene_tags.py --stats
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scene_tags")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIMENSION_SOURCES = PROJECT_ROOT / "data" / "input" / "dimension_sources"
VERIFIED_DIR = PROJECT_ROOT / "data" / "output" / "gsa_flywheel_exp" / "verified"
MANIFEST = DIMENSION_SOURCES / "manifest.json"


def scan_dimension_sources() -> Counter:
    """Scan all dimension_sources/*.txt for scene_type (col 3)."""
    tags = Counter()
    for txt_file in sorted(DIMENSION_SOURCES.glob("*.txt")):
        with open(txt_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                tag = parts[2].strip() if len(parts) > 2 else "unknown"
                if tag:
                    tags[tag] += 1
    return tags


def scan_verified_data() -> Counter:
    """Scan verified/*.jsonl for scene_type field."""
    tags = Counter()
    if not VERIFIED_DIR.exists():
        return tags
    for jsonl_file in VERIFIED_DIR.rglob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    tag = sample.get("scene_type", "unknown")
                    if tag:
                        tags[tag] += 1
                except json.JSONDecodeError:
                    continue
    return tags


def update_manifest(all_tags: Counter) -> None:
    """Update manifest.json with discovered scene_types."""
    if not MANIFEST.exists():
        logger.warning(f"Manifest not found: {MANIFEST}")
        return

    with open(MANIFEST) as f:
        manifest = json.load(f)

    sorted_tags = sorted(all_tags.keys())
    manifest["scene_types"] = sorted_tags
    manifest["scene_tag_counts"] = dict(all_tags.most_common())

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated manifest with {len(sorted_tags)} scene tags")


def main():
    parser = argparse.ArgumentParser(description="Manage scene tags for GSA flywheel")
    parser.add_argument("--update-manifest", action="store_true", help="Update manifest.json")
    parser.add_argument("--stats", action="store_true", help="Print tag statistics")
    args = parser.parse_args()

    logger.info("Scanning dimension_sources...")
    src_tags = scan_dimension_sources()
    logger.info(f"  Found {len(src_tags)} unique tags from dimension_sources ({sum(src_tags.values())} entries)")

    logger.info("Scanning verified data...")
    ver_tags = scan_verified_data()
    logger.info(f"  Found {len(ver_tags)} unique tags from verified ({sum(ver_tags.values())} samples)")

    all_tags = src_tags + ver_tags

    if args.stats or not args.update_manifest:
        print(f"\n{'Tag':<30} {'Sources':>10} {'Verified':>10} {'Total':>10}")
        print("-" * 65)
        for tag, total in all_tags.most_common():
            print(f"{tag:<30} {src_tags.get(tag, 0):>10} {ver_tags.get(tag, 0):>10} {total:>10}")
        print(f"\n{'TOTAL':<30} {sum(src_tags.values()):>10} {sum(ver_tags.values()):>10} {sum(all_tags.values()):>10}")

    if args.update_manifest:
        update_manifest(all_tags)


if __name__ == "__main__":
    main()
