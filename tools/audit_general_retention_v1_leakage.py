#!/usr/bin/env python3
"""Audit General Avg evaluation images against the frozen E1 post-training corpus."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.freeze_general_retention_v1_sources import _image_features, canonical_hash


class LeakageError(RuntimeError):
    """Raised when the leakage audit cannot establish complete coverage."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(payload)
    os.replace(partial, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LeakageError(f"bad JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise LeakageError(f"non-object JSON at {path}:{line_number}")
                rows.append(row)
    return rows


def resolve_training_path(raw: str) -> Path:
    if raw == "/mnt/data":
        return Path("/path/to/datasets")
    if raw.startswith("/mnt/data/"):
        return Path("/path/to/datasets") / raw.removeprefix("/mnt/data/")
    return Path(raw)


def collect_corpus_paths(
    corpus: Path,
) -> tuple[int, dict[str, Path], dict[str, set[str]]]:
    paths: dict[str, Path] = {}
    source_labels: dict[str, set[str]] = {}
    rows = 0
    with corpus.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LeakageError(f"bad corpus JSON at line {line_number}: {exc}") from exc
            images = item.get("images")
            if not isinstance(images, list) or not images:
                raise LeakageError(f"missing images list at corpus line {line_number}")
            for raw in images:
                if not isinstance(raw, str) or not raw:
                    raise LeakageError(f"bad image path at corpus line {line_number}")
                paths[raw] = resolve_training_path(raw)
                label = str(item.get("gsa_meta", {}).get("source", ""))
                if label:
                    source_labels.setdefault(raw, set()).add(label)
    return rows, paths, source_labels


def fingerprint_one(raw_path: str, resolved: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "raw_path": raw_path,
        "path": str(resolved),
        "state": "ERROR",
    }
    try:
        stat = resolved.stat()
        if not resolved.is_file():
            raise OSError("not a regular file")
        raw = resolved.read_bytes()
        features = _image_features(raw)
        record.update(
            {
                **features,
                "mtime_ns": stat.st_mtime_ns,
                "state": "OK",
            }
        )
    except (OSError, ValueError, RuntimeError) as exc:
        record["error"] = str(exc)
    return record


def fingerprint_training_images(
    paths: dict[str, Path],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(paths)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fingerprint_one, raw, resolved): raw
            for raw, resolved in paths.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 1000 == 0 or completed == total:
                print(f"fingerprinted {completed}/{total} training images", file=sys.stderr, flush=True)
    return sorted(results, key=lambda row: row["raw_path"])


class BKNode:
    def __init__(self, value: int, indices: list[int] | None = None) -> None:
        self.value = value
        self.indices = indices or []
        self.children: dict[int, BKNode] = {}

    def add(self, value: int, index: int) -> None:
        node = self
        while True:
            distance = (node.value ^ value).bit_count()
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value, [index])
                return
            node = child

    def query(self, value: int, radius: int) -> Iterable[tuple[int, int]]:
        distance = (self.value ^ value).bit_count()
        if distance <= radius:
            for index in self.indices:
                yield distance, index
        lower, upper = distance - radius, distance + radius
        for edge, child in self.children.items():
            if lower <= edge <= upper:
                yield from child.query(value, radius)


def build_bk(records: list[dict[str, Any]]) -> BKNode | None:
    root: BKNode | None = None
    for index, record in enumerate(records):
        value = int(record["phash64"], 16)
        if root is None:
            root = BKNode(value, [index])
        else:
            root.add(value, index)
    return root


def normalized_question(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def image_distance(eval_path: Path, train_path: Path) -> dict[str, float]:
    eval_image = cv2.imread(str(eval_path), cv2.IMREAD_COLOR)
    train_image = cv2.imread(str(train_path), cv2.IMREAD_COLOR)
    if eval_image is None or train_image is None:
        return {"normalized_rgb_mae": math.inf, "grayscale_correlation": -1.0}
    size = (128, 128)
    eval_resized = cv2.resize(eval_image, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    train_resized = cv2.resize(train_image, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    mae = float(np.mean(np.abs(eval_resized - train_resized)) / 255.0)
    eval_gray = cv2.cvtColor(eval_resized, cv2.COLOR_BGR2GRAY).reshape(-1)
    train_gray = cv2.cvtColor(train_resized, cv2.COLOR_BGR2GRAY).reshape(-1)
    if float(np.std(eval_gray)) == 0.0 or float(np.std(train_gray)) == 0.0:
        correlation = 1.0 if np.array_equal(eval_gray, train_gray) else 0.0
    else:
        correlation = float(np.corrcoef(eval_gray, train_gray)[0, 1])
    return {"normalized_rgb_mae": mae, "grayscale_correlation": correlation}


def near_duplicate_candidates(
    eval_images: list[dict[str, Any]],
    training: list[dict[str, Any]],
    *,
    phash_radius: int,
    dhash_radius: int,
    max_mae: float,
    min_correlation: float,
) -> list[dict[str, Any]]:
    tree = build_bk(training)
    if tree is None:
        return []
    candidates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for evaluation in eval_images:
        eval_phash = int(evaluation["phash64"], 16)
        eval_dhash = int(evaluation["dhash64"], 16)
        eval_aspect = evaluation["width"] / evaluation["height"]
        for phash_distance, index in tree.query(eval_phash, phash_radius):
            train = training[index]
            if (
                train["sha256"] == evaluation["sha256"]
                or train["decoded_rgb_sha256"] == evaluation["decoded_rgb_sha256"]
            ):
                continue
            dhash_distance = (eval_dhash ^ int(train["dhash64"], 16)).bit_count()
            if dhash_distance > dhash_radius:
                continue
            train_aspect = train["width"] / train["height"]
            if abs(math.log(eval_aspect / train_aspect)) > 0.02:
                continue
            pair = (evaluation["sha256"], train["sha256"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            pixel = image_distance(Path(evaluation["path"]), Path(train["path"]))
            if (
                pixel["normalized_rgb_mae"] <= max_mae
                and pixel["grayscale_correlation"] >= min_correlation
            ):
                candidates.append(
                    {
                        "eval_dataset": evaluation["dataset"],
                        "eval_image_id": evaluation["image_id"],
                        "eval_role": evaluation["evaluation_role"],
                        "eval_path": evaluation["path"],
                        "eval_sha256": evaluation["sha256"],
                        "training_path": train["raw_path"],
                        "training_resolved_path": train["path"],
                        "training_sha256": train["sha256"],
                        "phash_hamming": phash_distance,
                        "dhash_hamming": dhash_distance,
                        **pixel,
                        "adjudication": "UNRESOLVED",
                    }
                )
    return sorted(candidates, key=lambda row: (
        row["eval_dataset"], row["eval_image_id"], row["eval_role"], row["training_path"]
    ))


def _dataset_marker_present(dataset: str, values: Iterable[str]) -> bool:
    aliases = {
        "ai2d_test": ("ai2d",),
        "scienceqa_test": ("scienceqa",),
        "textvqa_val": ("textvqa",),
    }[dataset]
    normalized = re.sub(r"[^a-z0-9]+", "", " ".join(values).casefold())
    return any(alias in normalized for alias in aliases)


def source_id_collisions(
    eval_images: list[dict[str, Any]],
    training: list[dict[str, Any]],
    source_labels: dict[str, set[str]],
) -> list[dict[str, Any]]:
    collisions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    eval_ids: dict[str, set[str]] = {}
    for evaluation in eval_images:
        eval_ids.setdefault(evaluation["dataset"], set()).add(str(evaluation["image_id"]))
    for train in training:
        raw_path = train["raw_path"]
        path_tokens = set(Path(raw_path).parts) | set(Path(train["path"]).parts)
        path_tokens.add(Path(raw_path).stem)
        path_tokens.add(Path(train["path"]).stem)
        provenance = [raw_path, train["path"], *sorted(source_labels.get(raw_path, set()))]
        for dataset, dataset_eval_ids in eval_ids.items():
            if not _dataset_marker_present(dataset, provenance):
                continue
            for image_id in sorted(path_tokens & dataset_eval_ids):
                key = (dataset, image_id, raw_path)
                if key in seen:
                    continue
                seen.add(key)
                collisions.append(
                    {
                        "eval_dataset": dataset,
                        "eval_image_id": image_id,
                        "training_path": raw_path,
                        "training_source_labels": sorted(source_labels.get(raw_path, set())),
                        "policy": "dataset-qualified source ID is a hard leak",
                    }
                )
    return sorted(
        collisions,
        key=lambda row: (row["eval_dataset"], row["eval_image_id"], row["training_path"]),
    )


def image_question_collisions(
    corpus: Path,
    exact_eval_shas: set[str],
    sha_by_raw_path: dict[str, str],
    eval_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not exact_eval_shas:
        return []
    questions_by_sha: dict[str, list[dict[str, str]]] = {}
    for sample in eval_samples:
        sha = sample["source_image_sha256"]
        if sha in exact_eval_shas:
            questions_by_sha.setdefault(sha, []).append(
                {"uid": sample["uid"], "question": normalized_question(sample["question"])}
            )
    collisions: list[dict[str, Any]] = []
    with corpus.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            row_shas = {
                sha_by_raw_path[path]
                for path in item.get("images", [])
                if sha_by_raw_path.get(path) in exact_eval_shas
            }
            if not row_shas:
                continue
            user_text = " ".join(
                str(message.get("content", ""))
                for message in item.get("messages", [])
                if message.get("role") == "user"
            )
            normalized_user = normalized_question(user_text)
            for sha in row_shas:
                for question in questions_by_sha.get(sha, []):
                    if question["question"] and question["question"] in normalized_user:
                        collisions.append(
                            {
                                "corpus_line": line_number,
                                "eval_uid": question["uid"],
                                "image_sha256": sha,
                            }
                        )
    return collisions


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("/path/to/Anvil/trains/gsa_bench/train_shuf.jsonl"),
    )
    parser.add_argument("--expected-corpus-rows", type=int, default=333396)
    parser.add_argument(
        "--expected-corpus-sha256",
        default="4627f5500862a316434f1c2f22f541f763c0a5556bfadd5c607451811c8ce2d3",
    )
    parser.add_argument(
        "--eval-images",
        type=Path,
        default=root / "artifacts/general_retention_v1/eval_images.jsonl",
    )
    parser.add_argument(
        "--eval-samples",
        type=Path,
        default=root / "artifacts/general_retention_v1/eval_samples.jsonl",
    )
    parser.add_argument(
        "--eval-adapter-images",
        type=Path,
        default=root / "artifacts/general_retention_v1/eval_adapter_images.jsonl",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "artifacts/general_retention_v1/training_image_fingerprints.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts/general_retention_v1_leakage.json",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--phash-radius", type=int, default=4)
    parser.add_argument("--dhash-radius", type=int, default=4)
    parser.add_argument("--max-mae", type=float, default=0.06)
    parser.add_argument("--min-correlation", type=float, default=0.97)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    corpus_sha = file_sha256(args.corpus)
    rows, paths, source_labels = collect_corpus_paths(args.corpus)
    path_set_sha = canonical_hash(sorted(paths))
    preflight_errors: list[str] = []
    if corpus_sha != args.expected_corpus_sha256:
        preflight_errors.append("post-training corpus sha256 drift")
    if rows != args.expected_corpus_rows:
        preflight_errors.append(f"post-training corpus rows {rows} != {args.expected_corpus_rows}")

    training: list[dict[str, Any]]
    cache_binding = args.cache.with_suffix(".binding.json")
    if args.reuse_cache and args.cache.is_file() and cache_binding.is_file():
        binding = json.loads(cache_binding.read_text())
        if (
            binding.get("corpus_sha256") != corpus_sha
            or binding.get("path_set_sha256") != path_set_sha
            or binding.get("fingerprint_schema_version") != 2
        ):
            raise LeakageError("fingerprint cache binding drift")
        training = load_jsonl(args.cache)
        retry_paths = {
            row["raw_path"]: paths[row["raw_path"]]
            for row in training
            if row.get("state") != "OK"
            or "decoded_rgb_sha256" not in row
        }
        if retry_paths:
            print(
                f"repairing {len(retry_paths)} incomplete cached fingerprints",
                file=sys.stderr,
                flush=True,
            )
            repaired = {
                row["raw_path"]: row
                for row in fingerprint_training_images(retry_paths, workers=args.workers)
            }
            training = [repaired.get(row["raw_path"], row) for row in training]
            cache_payload = b"".join(
                (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
                for row in training
            )
            _atomic_write(args.cache, cache_payload)
            binding["fingerprint_manifest_sha256"] = hashlib.sha256(cache_payload).hexdigest()
            binding.pop("identity_sha256", None)
            binding["identity_sha256"] = canonical_hash(binding)
            _atomic_write(
                cache_binding,
                (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
    else:
        training = fingerprint_training_images(paths, workers=args.workers)
        cache_payload = b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in training
        )
        _atomic_write(args.cache, cache_payload)
        binding = {
            "schema_version": 1,
            "corpus_sha256": corpus_sha,
            "path_set_sha256": path_set_sha,
            "fingerprint_schema_version": 2,
            "unique_image_paths": len(paths),
            "fingerprint_manifest_sha256": hashlib.sha256(cache_payload).hexdigest(),
        }
        binding["identity_sha256"] = canonical_hash(binding)
        _atomic_write(
            cache_binding,
            (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    fingerprint_errors = [row for row in training if row.get("state") != "OK"]
    source_manifest_path = root / "artifacts/general_retention_v1/source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_eval_images = [
        {**row, "evaluation_role": "official_source"}
        for row in load_jsonl(args.eval_images)
    ]
    adapter_eval_images = load_jsonl(args.eval_adapter_images)
    eval_images = source_eval_images + adapter_eval_images
    eval_samples = load_jsonl(args.eval_samples)
    complete = not preflight_errors and not fingerprint_errors and len(training) == len(paths)
    ok_training = [row for row in training if row.get("state") == "OK"]
    eval_by_sha: dict[str, list[dict[str, Any]]] = {}
    eval_by_rgb_sha: dict[str, list[dict[str, Any]]] = {}
    for image in eval_images:
        eval_by_sha.setdefault(image["sha256"], []).append(image)
        eval_by_rgb_sha.setdefault(image["decoded_rgb_sha256"], []).append(image)
    exact_pairs = []
    normalized_exact_pairs = []
    exact_shas: set[str] = set()
    for train in ok_training:
        for evaluation in eval_by_sha.get(train["sha256"], []):
            exact_shas.add(train["sha256"])
            exact_pairs.append(
                {
                    "eval_dataset": evaluation["dataset"],
                    "eval_image_id": evaluation["image_id"],
                    "eval_role": evaluation["evaluation_role"],
                    "eval_sha256": evaluation["sha256"],
                    "training_path": train["raw_path"],
                }
            )
        for evaluation in eval_by_rgb_sha.get(train["decoded_rgb_sha256"], []):
            if train["sha256"] == evaluation["sha256"]:
                continue
            normalized_exact_pairs.append(
                {
                    "eval_dataset": evaluation["dataset"],
                    "eval_image_id": evaluation["image_id"],
                    "eval_role": evaluation["evaluation_role"],
                    "eval_sha256": evaluation["sha256"],
                    "decoded_rgb_sha256": evaluation["decoded_rgb_sha256"],
                    "training_path": train["raw_path"],
                    "training_sha256": train["sha256"],
                    "policy": "decoded RGB exact identity is a hard leak",
                }
            )

    source_id_pairs = source_id_collisions(
        source_eval_images,
        ok_training,
        source_labels,
    ) if complete else []

    near = near_duplicate_candidates(
        eval_images,
        ok_training,
        phash_radius=args.phash_radius,
        dhash_radius=args.dhash_radius,
        max_mae=args.max_mae,
        min_correlation=args.min_correlation,
    ) if complete else []
    sha_by_raw = {row["raw_path"]: row["sha256"] for row in ok_training}
    question_pairs = image_question_collisions(
        args.corpus, exact_shas, sha_by_raw, eval_samples
    ) if complete else []

    report: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": "general-retention-v1",
        "state": (
            "PASS"
            if complete
            and not exact_pairs
            and not normalized_exact_pairs
            and not source_id_pairs
            and not near
            and not question_pairs
            else "BLOCKED"
        ),
        "posttraining_corpus": str(args.corpus.resolve()),
        "posttraining_corpus_sha256": corpus_sha,
        "posttraining_corpus_rows": rows,
        "auditor_sha256": file_sha256(Path(__file__)),
        "source_manifest": {
            "path": str(source_manifest_path.resolve()),
            "file_sha256": file_sha256(source_manifest_path),
            "identity_sha256": source_manifest["identity_sha256"],
        },
        "evaluation_manifest_sha256": {
            "official_source_images": file_sha256(args.eval_images),
            "adapter_transport_images": file_sha256(args.eval_adapter_images),
            "samples": file_sha256(args.eval_samples),
        },
        "training_unique_image_paths": len(paths),
        "training_fingerprints_ok": len(ok_training),
        "training_fingerprint_errors": len(fingerprint_errors),
        "eval_samples": len(eval_samples),
        "eval_official_source_images": len(source_eval_images),
        "eval_adapter_transport_images": len(adapter_eval_images),
        "eval_image_representations": len(eval_images),
        "exact_image_collisions": len(exact_pairs) if complete else None,
        "transformation_normalized_exact_image_collisions": (
            len(normalized_exact_pairs) if complete else None
        ),
        "dataset_qualified_source_id_collisions": (
            len(source_id_pairs) if complete else None
        ),
        "near_duplicate_image_collisions_unresolved": len(near) if complete else None,
        "image_question_collisions": len(question_pairs) if complete else None,
        "near_duplicate_protocol": {
            "phash64_hamming_max": args.phash_radius,
            "dhash64_hamming_max": args.dhash_radius,
            "absolute_log_aspect_ratio_max": 0.02,
            "normalized_rgb_mae_max": args.max_mae,
            "grayscale_correlation_min": args.min_correlation,
            "candidate_policy": "all retained candidates require adjudication",
        },
        "preflight_errors": preflight_errors,
        "fingerprint_error_examples": fingerprint_errors[:100],
        "exact_collision_examples": exact_pairs[:100],
        "transformation_normalized_exact_collision_examples": normalized_exact_pairs[:100],
        "dataset_qualified_source_id_collision_examples": source_id_pairs[:100],
        "near_duplicate_candidates": near,
        "image_question_collision_examples": question_pairs[:100],
        "cache_binding": str(cache_binding.resolve()),
        "cache_binding_identity_sha256": binding["identity_sha256"],
    }
    report["identity_sha256"] = canonical_hash(report)
    _atomic_write(
        args.output,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "state": report["state"],
        "training_images": len(training),
        "fingerprint_errors": len(fingerprint_errors),
        "exact": report["exact_image_collisions"],
        "normalized_exact": report["transformation_normalized_exact_image_collisions"],
        "source_id": report["dataset_qualified_source_id_collisions"],
        "near_unresolved": report["near_duplicate_image_collisions_unresolved"],
        "image_question": report["image_question_collisions"],
        "identity_sha256": report["identity_sha256"],
    }, indent=2))
    return 0 if report["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
