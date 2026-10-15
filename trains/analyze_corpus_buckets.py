#!/usr/bin/env python3
"""Bucket-level contamination analysis of the corpus_v2 source pool.

Groups every corpus image path into a path-prefix bucket, intersects each bucket
with the exact image paths used by every evaluation release that has on-disk
images, and reports retained rows/images under candidate exclusion policies.

Path-prefix bucketing is deliberate: gsa_meta.source is unreliable because five
sources carry a directory tail (``train`` -> PRISM, ``images`` -> roborefit,
``3D`` -> refspatial/3D, ``stimuli`` -> mineanybuild, ``*_QA_Imgs`` -> phyblock).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

CORPUS_POOL = Path("/path/to/Anvil/trains/corpus_v2")

# Bucket key = longest stable dataset-identifying path prefix.
BUCKET_RULES: list[tuple[str, str]] = [
    ("/path/to/datasets/datasets/originDatasets_transfer_from_hpc3/PRISM/", "prism"),
    ("/path/to/datasets/dataflywheel/input/household/where2place/", "where2place"),
    ("/path/to/datasets/datasets/transferred_datasets/datasets_train/robopoint-data/images/VG_100K", "robopoint_data_vg_mirror"),
    ("/path/to/datasets/datasets/transferred_datasets/datasets_train/robopoint-data/", "robopoint_data"),
    ("/path/to/datasets/datasets/originDatasets_transfer_from_hpc3/robopoint_rft_", "robopoint_rft"),
    ("/path/to/datasets/da/transferred_dataset/visual_genome/", "visual_genome_typo_root"),
    ("/path/to/datasets/datasets/originDatasets_transfer_from_hpc3/roborefit_rft_", "roborefit_rft"),
    ("/path/to/datasets/datasets/originDatasets_transfer_from_hpc3/refspatial/", "refspatial_origin"),
    ("/path/to/datasets/datasets/transferred_datasets/datasets_train/refspatial/", "refspatial_transferred"),
    ("/path/to/datasets/datasets/originDatasets_transfer_from_hpc3/FSD_points_rft_", "fsd_rft"),
    ("/path/to/datasets/datasets/originDatasets_transfer_from_hpc3/SAT_rft_", "sat_rft"),
    ("/path/to/datasets/datasets/transferred_datasets/EmbodiedAI_benchmark/VSI-590K/", "vsi_590k"),
    ("/path/to/datasets/GSA_volc/", "gsa_volc_dead"),
    ("/path/to/datasets/dataflywheel/input/", "dataflywheel_other"),
    ("/path/to/datasets/bridge_raw_work/", "bridge_raw"),
    ("/path/to/datasets/datasets/originDatasets_transfer_from_hpc3/", "origin_other"),
]


def bucket_of(path: str) -> str:
    for prefix, name in BUCKET_RULES:
        if path.startswith(prefix):
            return name
    return "unbucketed:" + "/".join(path.split("/")[:7])


def eval_image_paths() -> dict[str, set[str]]:
    """Exact on-disk image paths referenced by evaluation releases."""
    out: dict[str, set[str]] = {}
    gsa = REPOSITORY_ROOT / "trains/gsa_bench/test.jsonl"
    paths: set[str] = set()
    for line in gsa.read_text(encoding="utf-8").splitlines():
        if line.strip():
            paths.update(str(p) for p in json.loads(line)["images"])
    out["gsa_bench_test"] = paths

    w2p = Path(
        "/path/to/datasets/datasets/transferred_datasets/"
        "EmbodiedAI_benchmark/where2place"
    )
    names = set()
    for line in (w2p / "point_questions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            names.add(str(json.loads(line)["image"]))
    out["where2place_basenames"] = names
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=CORPUS_POOL)
    parser.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "artifacts/corpus_source_clean_v1/bucket_analysis.json")
    args = parser.parse_args()

    evals = eval_image_paths()
    gsa_test = evals["gsa_bench_test"]
    w2p_names = evals["where2place_basenames"]

    rows_by_bucket: collections.Counter[str] = collections.Counter()
    images_by_bucket: dict[str, set[str]] = collections.defaultdict(set)
    src_by_bucket: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    gsa_hits: dict[str, set[str]] = collections.defaultdict(set)
    w2p_hits: dict[str, set[str]] = collections.defaultdict(set)
    total_rows = 0

    for shard in sorted(args.pool.glob("shard_*.jsonl")):
        print(f"scanning {shard.name}", file=sys.stderr, flush=True)
        with shard.open(errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                total_rows += 1
                source = str(record.get("gsa_meta", {}).get("source", "?"))
                for raw in record.get("images") or []:
                    bucket = bucket_of(raw)
                    rows_by_bucket[bucket] += 1
                    images_by_bucket[bucket].add(raw)
                    src_by_bucket[bucket][source] += 1
                    if raw in gsa_test:
                        gsa_hits[bucket].add(raw)
                    if os.path.basename(raw) in w2p_names and "where2place" in raw:
                        w2p_hits[bucket].add(raw)

    report = {
        "pool": str(args.pool),
        "total_rows": total_rows,
        "buckets": {
            bucket: {
                "rows": rows_by_bucket[bucket],
                "unique_images": len(images_by_bucket[bucket]),
                "gsa_test_exact_path_hits": len(gsa_hits.get(bucket, ())),
                "where2place_basename_hits": len(w2p_hits.get(bucket, ())),
                "gsa_meta_sources": dict(src_by_bucket[bucket].most_common()),
            }
            for bucket in sorted(rows_by_bucket, key=lambda b: -rows_by_bucket[b])
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {b: {k: v for k, v in d.items() if k != "gsa_meta_sources"} for b, d in report["buckets"].items()},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
