#!/usr/bin/env python3
"""Build a frozen, source-clean v3 generation index from the approved v2 images."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trains.flywheel.prompts_ext import LABEL_VERSION, PROMPT_VERSION


DEFAULT_EXCLUDED_BUCKETS = ("household", "refspatial", "robopoint")
DEFAULT_EXCLUDED_PATH_PATTERN = (
    r"/reasoning/|/stimuli/|QA_Imgs|candidate_blocks_images|mix_res|depth_images"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected object at {path}:{line_no}")
        rows.append(row)
    return rows


def prior_images(path: Path) -> tuple[set[str], set[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    paths = {
        str(item["image_path"])
        for item in manifest.get("selected", [])
        if item.get("image_path")
    }
    hashes = {
        file_sha256(Path(image_path))
        for image_path in paths
        if Path(image_path).is_file()
    }
    return paths, hashes


def round_robin_scenes(rows: list[dict]) -> list[dict]:
    """Keep the full set deterministic while making every short prefix diverse."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["scene"]].append(row)
    for scene_rows in grouped.values():
        scene_rows.sort(key=lambda row: row["image_sha256"])
    ordered: list[dict] = []
    offsets = Counter()
    while len(ordered) < len(rows):
        progressed = False
        for scene in sorted(grouped):
            index = offsets[scene]
            if index < len(grouped[scene]):
                ordered.append(grouped[scene][index])
                offsets[scene] += 1
                progressed = True
        if not progressed:
            break
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest", type=Path,
        default=Path("artifacts/upload_review_500_v2/manifest.jsonl"))
    parser.add_argument(
        "--image-root", type=Path,
        default=Path("artifacts/upload_review_500_v2"))
    parser.add_argument(
        "--exclude-audit-manifest", type=Path,
        default=Path("data/semantic_audit_v2/manifest.json"))
    parser.add_argument(
        "--out-index", type=Path,
        default=Path("artifacts/manifests/v3_generation.index"))
    parser.add_argument(
        "--out-manifest", type=Path,
        default=Path("artifacts/manifests/v3_generation_manifest.json"))
    parser.add_argument(
        "--exclude-source-bucket", action="append",
        default=list(DEFAULT_EXCLUDED_BUCKETS))
    parser.add_argument(
        "--exclude-source-path-regex", default=DEFAULT_EXCLUDED_PATH_PATTERN)
    parser.add_argument("--round-id", default="semantic-audit-v3")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for target in (args.out_index, args.out_manifest):
        if target.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite existing file: {target}")

    source_rows = load_jsonl(args.source_manifest)
    old_paths, old_hashes = prior_images(args.exclude_audit_manifest)
    excluded_buckets = set(args.exclude_source_bucket)
    excluded_pattern = re.compile(args.exclude_source_path_regex)
    counts = Counter()
    selected = []
    seen_hashes = set()

    for row in source_rows:
        upload_path = row.get("upload_path")
        upload_sha = row.get("upload_sha256")
        if not upload_path or not upload_sha:
            counts["missing_manifest_identity"] += 1
            continue
        image_path = (args.image_root / upload_path).resolve()
        if not image_path.is_file():
            counts["missing_image"] += 1
            continue
        if str(image_path) in old_paths or upload_sha in old_hashes:
            counts["prior_audit_image"] += 1
            continue
        if row.get("source_bucket") in excluded_buckets:
            counts[f"source_bucket/{row.get('source_bucket')}"] += 1
            continue
        if excluded_pattern.search(str(row.get("source_path", ""))):
            counts["derived_or_multipanel_source"] += 1
            continue
        actual_sha = file_sha256(image_path)
        if actual_sha != upload_sha:
            raise ValueError(f"upload hash mismatch: {image_path}")
        if actual_sha in seen_hashes:
            counts["duplicate_upload_hash"] += 1
            continue
        seen_hashes.add(actual_sha)
        selected.append({
            "id": row.get("id"),
            "image_path": str(image_path),
            "image_sha256": actual_sha,
            "scene": row.get("scene", "unknown"),
            "source_bucket": row.get("source_bucket", "unknown"),
        })

    if len(selected) < 40:
        raise SystemExit(f"only {len(selected)} eligible unseen images; need at least 40")

    selected = round_robin_scenes(selected)
    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_index.write_text(
        "".join(f'{row["image_path"]}\t{row["scene"]}\n' for row in selected),
        encoding="utf-8")
    payload = {
        "schema_version": 1,
        "round_id": args.round_id,
        "prompt_version": PROMPT_VERSION,
        "label_version": LABEL_VERSION,
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "excluded_audit_manifest": str(args.exclude_audit_manifest),
        "excluded_audit_manifest_sha256": file_sha256(args.exclude_audit_manifest),
        "excluded_source_buckets": sorted(excluded_buckets),
        "excluded_source_path_regex": args.exclude_source_path_regex,
        "eligible_images": len(selected),
        "eligible_by_scene": dict(sorted(Counter(
            row["scene"] for row in selected).items())),
        "eligible_by_source_bucket": dict(sorted(Counter(
            row["source_bucket"] for row in selected).items())),
        "excluded_counts": dict(sorted(counts.items())),
        "selected": selected,
    }
    args.out_manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "eligible_images": payload["eligible_images"],
        "eligible_by_scene": payload["eligible_by_scene"],
        "eligible_by_source_bucket": payload["eligible_by_source_bucket"],
        "excluded_counts": payload["excluded_counts"],
        "out_index": str(args.out_index),
        "out_manifest": str(args.out_manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
