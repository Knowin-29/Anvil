#!/usr/bin/env python3
"""Build a source-clean GSA training corpus from the corpus_v2 pool.

Two independent exclusion layers are applied, in order:

Layer A (bucket): whole contaminated source buckets are dropped by *image path
prefix*, never by ``gsa_meta.source``. Five pool sources carry a directory tail
instead of a dataset name (``train`` -> PRISM, ``images`` -> roborefit/robopoint,
``3D`` -> refspatial/3D, ``stimuli`` -> mineanybuild, ``*_QA_Imgs`` -> phyblock),
so source-field filtering silently retains contaminated rows.

Layer B (content): every surviving image is fingerprinted and dropped if it
collides with any evaluation image on encoded bytes, decoded RGB, deterministic
JPEG normalization, exact evaluation path, release-qualified source identity, or
the frozen perceptual near-duplicate gate. This is the "content-hash split"
required by docs/multibench_leakage_audit_v1.md before any retrain.

Outputs a corpus, a frozen manifest, and an audit-ready training fingerprint
cache plus binding so tools/audit_multibench_leakage.py can consume it directly.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.audit_general_retention_v1_leakage import (  # noqa: E402
    build_bk,
    fingerprint_training_images,
    image_distance,
)
from tools.freeze_general_retention_v1_sources import canonical_hash  # noqa: E402
from trains.analyze_corpus_buckets import BUCKET_RULES, bucket_of  # noqa: E402

CORPUS_POOL = Path("/path/to/Anvil/trains/corpus_v2")
EVAL_ROOT = Path(
    "/path/to/datasets/datasets/transferred_datasets/EmbodiedAI_benchmark"
)

# Buckets whose images are known-dead on disk; excluded under every policy.
DEAD_BUCKETS = {"visual_genome_typo_root", "gsa_volc_dead"}

# P3: drop every bucket that produced a hard leak or a perceptual candidate
# against any evaluation release, plus the whole Visual Genome family because the
# CV-Bench near-duplicates live in the robopoint-data VG_100K_2 mirror.
P3_EXCLUDED = {
    "prism",
    "where2place",
    "robopoint_data",
    "robopoint_data_vg_mirror",
    "robopoint_rft",
    "visual_genome_typo_root",
}

# P4: P3 plus RoboRefit. Under P3 the audit still reports 143 normalized-prompt
# overlaps against the internal GSA test split, every one of them a RoboRefit
# referring-expression template reused on a *different* image with a different
# answer. Those are template overlaps rather than annotation leaks, but the
# auditor is fail-closed by design and must not be relaxed, so the source is
# dropped instead of the detector.
P4_EXCLUDED = P3_EXCLUDED | {"roborefit_rft"}

# P5: P4 plus the RefSpatial family, i.e. every bucket sharing a root with a
# scored evaluation release even without an observed collision.
P5_EXCLUDED = P4_EXCLUDED | {
    "refspatial_origin",
    "refspatial_transferred",
}

POLICIES = {"P3": P3_EXCLUDED, "P4": P4_EXCLUDED, "P5": P5_EXCLUDED}

EXCLUSION_RATIONALE = {
    "prism": "212 encoded+decoded+JPEG exact pairs against the GSA internal test split; PRISM also hides behind gsa_meta.source='train' for 11,828 rows",
    "where2place": "90/100 Where2Place evaluation images plus 2 RefSpatial location near pairs at phash=0",
    "robopoint_data": "9 exact pairs against the GSA internal test split via images/object_ref",
    "robopoint_data_vg_mirror": "13 CV-Bench perceptual near-duplicates in the VG_100K_2 mirror; same root as the Visual Genome family",
    "robopoint_rft": "11 perceptual near-duplicates against the GSA internal test split and 373 exact evaluation image paths",
    "visual_genome_typo_root": "same Visual Genome content as the robopoint-data mirror that carries the CV-Bench near pairs; also 100% dead links under the jsonl typo root",
    "refspatial_origin": "P5 only: shares the RefSpatial release root with the scored RefSpatial-location cell",
    "refspatial_transferred": "P5 only: shares the RefSpatial release root with the scored RefSpatial-location cell",
    "roborefit_rft": "P4/P5: 332 exact GSA internal test image paths, and 143 normalized referring-expression prompt templates shared with the internal test split (different images, different answers, but the auditor treats an exact normalized prompt as a hard candidate)",
    "gsa_volc_dead": "backup_420 deleted upstream; every path is a dead symlink",
}

def load_eval_fingerprints(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty evaluation fingerprint manifest: {path}")
    return rows


def eval_identity_index(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build every exclusion index the content layer needs."""
    gsa_test = REPOSITORY_ROOT / "trains/gsa_bench/test.jsonl"
    eval_paths: set[str] = set()
    for line in gsa_test.read_text(encoding="utf-8").splitlines():
        if line.strip():
            eval_paths.update(str(p) for p in json.loads(line)["images"])

    w2p_names: set[str] = set()
    questions = EVAL_ROOT / "where2place" / "point_questions.jsonl"
    for line in questions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            w2p_names.add(os.path.basename(str(json.loads(line)["image"])))

    cv_basenames: set[str] = set()
    for row in eval_rows:
        if row["dataset"] != "cv_bench":
            continue
        name = str((row.get("source_metadata") or {}).get("source_filename") or "")
        if name:
            cv_basenames.add(os.path.basename(name).casefold())

    return {
        "sha256": {row["sha256"] for row in eval_rows},
        "decoded_rgb": {row["decoded_rgb_sha256"] for row in eval_rows},
        "jpeg": {row["pillow_default_jpeg_sha256"] for row in eval_rows},
        "eval_paths": eval_paths,
        "where2place_basenames": w2p_names,
        "cv_bench_basenames": cv_basenames,
    }

def select_rows(
    pool: Path,
    excluded: set[str],
    *,
    target: int,
    per_bucket_cap_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream the pool, drop excluded buckets, and reservoir-cap each bucket.

    Reservoir sampling keeps the retained set independent of shard order so the
    corpus is not dominated by whichever large bucket happens to be scanned
    first, while remaining deterministic under a fixed seed.
    """
    rng = random.Random(seed)
    cap = max(1, int(target * per_bucket_cap_frac))
    reservoirs: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen_by_bucket: collections.Counter[str] = collections.Counter()
    stats = {
        "pool_rows": 0,
        "dropped_bucket_excluded": 0,
        "dropped_multi_bucket_row": 0,
        "dropped_no_image": 0,
        "dropped_duplicate_key": 0,
    }
    seen_keys: set[str] = set()

    for shard in sorted(pool.glob("shard_*.jsonl")):
        print(f"[select] {shard.name}", file=sys.stderr, flush=True)
        with shard.open(errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                stats["pool_rows"] += 1
                images = record.get("images") or []
                if not images:
                    stats["dropped_no_image"] += 1
                    continue
                buckets = {bucket_of(str(p)) for p in images}
                if buckets & (excluded | DEAD_BUCKETS):
                    stats["dropped_bucket_excluded"] += 1
                    continue
                if len(buckets) != 1:
                    stats["dropped_multi_bucket_row"] += 1
                    continue
                bucket = next(iter(buckets))
                answer = "".join(
                    str(m.get("content", ""))
                    for m in record.get("messages") or []
                    if m.get("role") == "assistant"
                )
                key = hashlib.sha256(
                    (str(images[0]) + "|" + answer[:400]).encode("utf-8")
                ).hexdigest()
                if key in seen_keys:
                    stats["dropped_duplicate_key"] += 1
                    continue
                seen_keys.add(key)
                seen_by_bucket[bucket] += 1
                store = reservoirs[bucket]
                if len(store) < cap:
                    store.append(record)
                else:
                    index = rng.randrange(seen_by_bucket[bucket])
                    if index < cap:
                        store[index] = record

    rows = [record for bucket in sorted(reservoirs) for record in reservoirs[bucket]]
    rng.shuffle(rows)
    stats["per_bucket_cap"] = cap
    stats["retained_before_content_filter"] = len(rows)
    stats["bucket_pool_rows_after_exclusion"] = dict(seen_by_bucket.most_common())
    return rows, stats

def content_excluded_images(
    fingerprints: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    index: dict[str, Any],
    *,
    phash_radius: int,
    dhash_radius: int,
    max_mae: float,
    min_correlation: float,
) -> tuple[dict[str, str], dict[str, int]]:
    """Return image paths that must be excluded, keyed by exclusion reason."""
    excluded: dict[str, str] = {}
    reasons: collections.Counter[str] = collections.Counter()

    def mark(raw: str, reason: str) -> None:
        if raw not in excluded:
            excluded[raw] = reason
            reasons[reason] += 1

    for row in fingerprints:
        raw = row["raw_path"]
        if row.get("state") != "OK":
            mark(raw, "unreadable_image")
            continue
        if row["sha256"] in index["sha256"]:
            mark(raw, "encoded_byte_collision")
        elif row["decoded_rgb_sha256"] in index["decoded_rgb"]:
            mark(raw, "decoded_rgb_collision")
        elif row["pillow_default_jpeg_sha256"] in index["jpeg"]:
            mark(raw, "jpeg_normalized_collision")
        elif raw in index["eval_paths"] or row["path"] in index["eval_paths"]:
            mark(raw, "exact_evaluation_path")
        else:
            name = os.path.basename(raw)
            if name in index["where2place_basenames"] and "where2place" in raw.casefold():
                mark(raw, "where2place_source_identity")
            elif (
                name.casefold() in index["cv_bench_basenames"]
                and "vg_100k" in raw.casefold()
            ):
                mark(raw, "cv_bench_source_identity")

    survivors = [
        row for row in fingerprints
        if row.get("state") == "OK" and row["raw_path"] not in excluded
    ]
    tree = build_bk(survivors)
    if tree is not None:
        for evaluation in eval_rows:
            eval_phash = int(evaluation["phash64"], 16)
            eval_dhash = int(evaluation["dhash64"], 16)
            eval_aspect = evaluation["width"] / evaluation["height"]
            for _, position in tree.query(eval_phash, phash_radius):
                train = survivors[position]
                if train["raw_path"] in excluded:
                    continue
                if (eval_dhash ^ int(train["dhash64"], 16)).bit_count() > dhash_radius:
                    continue
                train_aspect = train["width"] / train["height"]
                if abs(math.log(eval_aspect / train_aspect)) > 0.02:
                    continue
                pixel = image_distance(Path(evaluation["path"]), Path(train["path"]))
                if (
                    pixel["normalized_rgb_mae"] <= max_mae
                    and pixel["grayscale_correlation"] >= min_correlation
                ):
                    mark(train["raw_path"], "perceptual_near_duplicate")
    return excluded, dict(reasons)

def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload)
    os.replace(partial, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    )
    atomic_write(path, payload)


def write_fingerprint_cache(
    cache: Path, rows: list[dict[str, Any]], corpus_sha: str, paths: list[str]
) -> dict[str, Any]:
    payload = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )
    atomic_write(cache, payload)
    binding = {
        "schema_version": 1,
        "corpus_sha256": corpus_sha,
        "path_set_sha256": canonical_hash(sorted(paths)),
        "fingerprint_schema_version": 2,
        "unique_image_paths": len(paths),
        "fingerprint_manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }
    binding["identity_sha256"] = canonical_hash(binding)
    atomic_write(
        cache.with_suffix(".binding.json"),
        (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return binding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--pool", type=Path, default=CORPUS_POOL)
    parser.add_argument("--target", type=int, default=150_000)
    parser.add_argument("--per-bucket-cap-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--phash-radius", type=int, default=4)
    parser.add_argument("--dhash-radius", type=int, default=4)
    parser.add_argument("--max-mae", type=float, default=0.06)
    parser.add_argument("--min-correlation", type=float, default=0.97)
    parser.add_argument(
        "--eval-fingerprints",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/leakage_audit_full_v1/eval_image_fingerprints.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    excluded_buckets = POLICIES[args.policy]
    out_dir = args.out_dir or (
        REPOSITORY_ROOT / f"trains/corpus_source_clean_v1/{args.policy.lower()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = load_eval_fingerprints(args.eval_fingerprints)
    index = eval_identity_index(eval_rows)
    print(
        f"[eval] {len(eval_rows)} unique evaluation images, "
        f"{len(index['eval_paths'])} exact evaluation paths",
        file=sys.stderr,
        flush=True,
    )

    rows, select_stats = select_rows(
        args.pool,
        excluded_buckets,
        target=args.target,
        per_bucket_cap_frac=args.per_bucket_cap_frac,
        seed=args.seed,
    )
    print(f"[select] retained {len(rows)} rows", file=sys.stderr, flush=True)

    paths = {}
    for record in rows:
        for raw in record["images"]:
            paths[str(raw)] = Path(str(raw))
    print(f"[fingerprint] {len(paths)} unique images", file=sys.stderr, flush=True)
    fingerprints = fingerprint_training_images(paths, workers=args.workers)

    excluded_images, exclusion_reasons = content_excluded_images(
        fingerprints,
        eval_rows,
        index,
        phash_radius=args.phash_radius,
        dhash_radius=args.dhash_radius,
        max_mae=args.max_mae,
        min_correlation=args.min_correlation,
    )
    print(f"[content] excluding {len(excluded_images)} images: {exclusion_reasons}",
          file=sys.stderr, flush=True)

    kept_rows = [
        record for record in rows
        if not any(str(raw) in excluded_images for raw in record["images"])
    ]
    kept_paths = sorted({str(raw) for record in kept_rows for raw in record["images"]})
    kept_path_set = set(kept_paths)
    kept_fingerprints = sorted(
        (row for row in fingerprints if row["raw_path"] in kept_path_set),
        key=lambda row: row["raw_path"],
    )
    if len(kept_fingerprints) != len(kept_paths):
        raise RuntimeError("fingerprint coverage mismatch after content filtering")

    corpus_path = out_dir / "train.jsonl"
    write_corpus(corpus_path, kept_rows)
    corpus_sha = file_sha256(corpus_path)
    cache_path = out_dir / "training_image_fingerprints.jsonl"
    binding = write_fingerprint_cache(cache_path, kept_fingerprints, corpus_sha, kept_paths)

    by_form = collections.Counter()
    by_scene = collections.Counter()
    by_source = collections.Counter()
    by_bucket = collections.Counter()
    bucket_images = collections.defaultdict(set)
    for record in kept_rows:
        meta = record.get("gsa_meta") or {}
        by_form[str(meta.get("form", "unknown"))] += 1
        by_scene[str(meta.get("scene", "unknown"))] += 1
        by_source[str(meta.get("source", "unknown"))] += 1
        bucket = bucket_of(str(record["images"][0]))
        by_bucket[bucket] += 1
        bucket_images[bucket].update(str(p) for p in record["images"])

    manifest = {
        "schema_version": 1,
        "corpus_id": f"gsa-source-clean-v1-{args.policy.lower()}",
        "policy": args.policy,
        "built_by": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__)),
        },
        "source_pool": {
            "path": str(args.pool.resolve()),
            "shards": sorted(p.name for p in args.pool.glob("shard_*.jsonl")),
            "rows": select_stats["pool_rows"],
        },
        "corpus": {
            "path": str(corpus_path.resolve()),
            "rows": len(kept_rows),
            "sha256": corpus_sha,
            "unique_images": len(kept_paths),
        },
        "training_fingerprint_cache": {
            "path": str(cache_path.resolve()),
            "sha256": file_sha256(cache_path),
            "binding_identity_sha256": binding["identity_sha256"],
            "rows": len(kept_fingerprints),
        },
        "exclusion_decisions": {
            "layer_a_bucket_exclusion": {
                "method": "image path prefix, never gsa_meta.source",
                "excluded_buckets": sorted(excluded_buckets | DEAD_BUCKETS),
                "rationale": {
                    bucket: EXCLUSION_RATIONALE.get(bucket, "dead image root")
                    for bucket in sorted(excluded_buckets | DEAD_BUCKETS)
                },
                "bucket_rules": [list(rule) for rule in BUCKET_RULES],
                "source_field_unreliable_examples": {
                    "train": "PRISM",
                    "images": "roborefit / robopoint-data",
                    "3D": "refspatial/3D",
                    "stimuli": "mineanybuild",
                    "Counterfactual_QA_Imgs": "phyblock",
                },
            },
            "layer_b_content_exclusion": {
                "axes": [
                    "encoded_bytes", "decoded_rgb", "pillow_default_jpeg",
                    "exact_evaluation_path", "release_qualified_source_identity",
                    "perceptual_near_duplicate",
                ],
                "excluded_images": len(excluded_images),
                "by_reason": exclusion_reasons,
                "near_duplicate_gate": {
                    "phash64_hamming_max": args.phash_radius,
                    "dhash64_hamming_max": args.dhash_radius,
                    "absolute_log_aspect_ratio_max": 0.02,
                    "normalized_rgb_mae_max": args.max_mae,
                    "grayscale_correlation_min": args.min_correlation,
                },
            },
        },
        "selection": select_stats,
        "distribution": {
            "by_form": dict(by_form.most_common()),
            "by_scene": dict(by_scene.most_common()),
            "by_gsa_meta_source": dict(by_source.most_common()),
            "by_path_bucket": {
                bucket: {
                    "rows": by_bucket[bucket],
                    "unique_images": len(bucket_images[bucket]),
                }
                for bucket in sorted(by_bucket, key=lambda b: -by_bucket[b])
            },
        },
        "evaluation_fingerprint_source": {
            "path": str(args.eval_fingerprints.resolve()),
            "sha256": file_sha256(args.eval_fingerprints),
            "unique_images": len(eval_rows),
            "reuse_note": "evaluation-side fingerprints reused from leakage_audit_full_v1",
        },
        "held_out_policy": {
            "scannet_val_frame_bundles_v3": "may_enter_training=false, excluded",
        },
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    atomic_write(
        out_dir / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    excl_rows = [
        {"raw_path": raw, "reason": reason}
        for raw, reason in sorted(excluded_images.items())
    ]
    atomic_write(
        out_dir / "content_excluded_images.jsonl",
        b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in excl_rows
        ),
    )
    print(json.dumps({
        "policy": args.policy,
        "rows": len(kept_rows),
        "unique_images": len(kept_paths),
        "corpus": str(corpus_path),
        "corpus_sha256": corpus_sha,
        "content_excluded_images": len(excluded_images),
        "by_reason": exclusion_reasons,
        "identity_sha256": manifest["identity_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
