#!/usr/bin/env python3
"""Fail-closed E1 train-versus-main-benchmark leakage audit.

The audit covers the five datasets in ``experiments/result_registry.yaml``. It
reuses the training-image fingerprint cache only after verifying its corpus and
file bindings, then checks encoded bytes, decoded RGB pixels, a deterministic
JPEG normalization, dataset-qualified source identifiers, available scene
identifiers, normalized text, and perceptual near-duplicate candidates.

Exit codes: 0 PASS, 3 LEAKAGE_DETECTED, 4 BLOCKED/INCOMPLETE.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.audit_general_retention_v1_leakage import (  # noqa: E402
    collect_corpus_paths,
    file_sha256,
    near_duplicate_candidates,
)
from tools.freeze_general_retention_v1_sources import (  # noqa: E402
    _image_features,
    canonical_hash,
)


DATA_ROOT = Path(
    "/path/to/datasets/datasets/transferred_datasets/EmbodiedAI_benchmark"
)
DEFAULT_CORPUS = Path(
    "/path/to/Anvil/trains/gsa_bench/train_shuf.jsonl"
)
DEFAULT_CORPUS_ROWS = 333396
DEFAULT_CORPUS_SHA256 = (
    "4627f5500862a316434f1c2f22f541f763c0a5556bfadd5c607451811c8ce2d3"
)


class FullLeakageError(RuntimeError):
    """Raised when a required audit input cannot be verified."""


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload)
    os.replace(partial, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    payload = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    )
    atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def normalize_text(value: Any) -> str:
    """Normalize publication-facing text without erasing semantic numbers."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    value = value.casefold().replace("<image>", " ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def text_sha(value: Any) -> str | None:
    normalized = normalize_text(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else None


def extract_referents(value: Any) -> list[str]:
    """Extract explicit referent-like fields from structured answers."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    found: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key.casefold() in {"label", "object", "target", "target_class"}:
                    normalized = normalize_text(child)
                    if normalized:
                        found.add(normalized)
                else:
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return sorted(found)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
    return cleaned or "image"


def materialize_eval_image(raw: bytes, destination_stem: Path) -> tuple[Path, dict[str, Any]]:
    features = _image_features(raw)
    suffix = ".jpg" if features["format"] in {"jpeg", "jpg"} else ".png"
    path = destination_stem.with_suffix(suffix)
    if path.is_file():
        if file_sha256(path) != features["sha256"]:
            raise FullLeakageError(f"materialized evaluation image drift: {path}")
    else:
        atomic_write(path, raw)
    return path.resolve(), features


def make_eval_record(
    *,
    dataset: str,
    uid: str,
    raw: bytes,
    cache_dir: Path,
    prompt: Any,
    answer: Any,
    referents: Iterable[Any],
    source_ids: Iterable[str],
    scene_ids: Iterable[str],
    source_identity_status: str,
    scene_identity_status: str,
    source_metadata: dict[str, Any] | None = None,
    existing_path: Path | None = None,
) -> dict[str, Any]:
    if existing_path is not None:
        if not existing_path.is_file():
            raise FullLeakageError(
                f"missing materialized evaluation image: {existing_path}"
            )
        features = _image_features(raw)
        if file_sha256(existing_path) != features["sha256"]:
            raise FullLeakageError(
                f"materialized evaluation image/source bytes drift: {existing_path}"
            )
        path = existing_path.resolve()
    else:
        path, features = materialize_eval_image(
            raw, cache_dir / dataset / _safe_name(uid)
        )
    normalized_referents = sorted(
        {normalized for item in referents if (normalized := normalize_text(item))}
    )
    prompt_hash = text_sha(prompt)
    answer_hash = text_sha(answer)
    return {
        **features,
        "dataset": dataset,
        "uid": uid,
        "image_id": uid,
        "evaluation_role": "main_table_test",
        "path": str(path),
        "prompt_sha256": prompt_hash,
        "answer_sha256": answer_hash,
        "prompt_answer_sha256": text_sha(
            f"{normalize_text(prompt)}\n{normalize_text(answer)}"
        ) if prompt_hash and answer_hash else None,
        "referent_sha256": sorted(filter(None, (text_sha(x) for x in normalized_referents))),
        "source_ids": sorted({normalize_text(x) for x in source_ids if normalize_text(x)}),
        "scene_ids": sorted({normalize_text(x) for x in scene_ids if normalize_text(x)}),
        "source_identity_status": source_identity_status,
        "scene_identity_status": scene_identity_status,
        "source_metadata": source_metadata or {},
    }


def load_cv_bench(cache_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise FullLeakageError("pyarrow is required for CV-Bench") from exc
    records: list[dict[str, Any]] = []
    assets: list[Path] = []
    for kind, filename in (("2D", "test_2d.parquet"), ("3D", "test_3d.parquet")):
        path = DATA_ROOT / "CV-Bench" / filename
        if not path.is_file():
            raise FullLeakageError(f"missing CV-Bench asset: {path}")
        assets.append(path)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                raw = (row.get("image") or {}).get("bytes")
                if not raw:
                    raise FullLeakageError(f"CV-Bench {kind}/{row.get('idx')} has no image")
                uid = f"{kind}_{row['idx']}"
                source_filename = str(row.get("source_filename") or "")
                source_dataset = str(row.get("source_dataset") or row.get("source") or "")
                # CV-Bench does not expose a uniform scene column. Hypersim is the
                # one current source with an explicit scene token in its source path.
                scene_tokens = [
                    part for part in Path(source_filename).parts
                    if re.fullmatch(r"ai_\d+_\d+", part.casefold())
                ]
                scene_ids = [f"{source_dataset}:{token}" for token in scene_tokens]
                records.append(make_eval_record(
                    dataset="cv_bench",
                    uid=uid,
                    raw=raw,
                    cache_dir=cache_dir,
                    prompt=row.get("prompt") or row.get("question"),
                    answer=row.get("answer"),
                    referents=[row.get("target_class")],
                    source_ids=[f"{source_dataset}:{source_filename}"],
                    scene_ids=scene_ids,
                    source_identity_status="PROVIDED_BY_RELEASE",
                    scene_identity_status=(
                        "DERIVED_FROM_SOURCE_PATH"
                        if scene_ids else "NOT_PROVIDED_BY_RELEASE"
                    ),
                    source_metadata={
                        "source": row.get("source"),
                        "source_dataset": source_dataset,
                        "source_filename": source_filename,
                    },
                    existing_path=REPOSITORY_ROOT / ".cache/eval_assets/cv_bench" / f"{uid}.png",
                ))
    return records, assets


def load_embspatial(cache_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    path = DATA_ROOT / "EmbSpatial-Bench" / "embspatial_bench.json"
    if not path.is_file():
        raise FullLeakageError(f"missing EmbSpatial asset: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    records: list[dict[str, Any]] = []
    for row in rows:
        uid = str(row["question_id"])
        try:
            raw = base64.b64decode(row["image"], validate=True)
        except Exception as exc:
            raise FullLeakageError(f"invalid EmbSpatial image at {uid}") from exc
        answer_options = list(row.get("answer_options") or [])
        answer_index = row.get("answer")
        answer = (
            answer_options[answer_index]
            if isinstance(answer_index, int) and 0 <= answer_index < len(answer_options)
            else answer_index
        )
        objects = list(row.get("objects") or [])
        records.append(make_eval_record(
            dataset="embspatial",
            uid=uid,
            raw=raw,
            cache_dir=cache_dir,
            prompt=row.get("question"),
            answer=answer,
            referents=[obj.get("name") for obj in objects if isinstance(obj, dict)],
            source_ids=[f"{row.get('data_source')}:{uid}"],
            scene_ids=[f"{row.get('data_source')}:{row.get('scene_id')}"],
            source_identity_status="QUESTION_ID_PROVIDED_BY_RELEASE",
            scene_identity_status="PROVIDED_BY_RELEASE",
            source_metadata={
                "data_source": row.get("data_source"),
                "scene_id": row.get("scene_id"),
            },
            existing_path=REPOSITORY_ROOT / ".cache/eval_assets/embspatial" / f"{uid}.jpg",
        ))
    return records, [path]


def load_where2place(cache_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    base = DATA_ROOT / "where2place"
    question_path = base / "point_questions.jsonl"
    if not question_path.is_file():
        raise FullLeakageError(f"missing Where2Place questions: {question_path}")
    records: list[dict[str, Any]] = []
    assets = [question_path]
    for line_number, line in enumerate(question_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        image_ref = str(row["image"])
        image_path = base / "images" / image_ref
        if not image_path.is_file():
            raise FullLeakageError(f"missing Where2Place image at line {line_number}: {image_path}")
        assets.append(image_path)
        records.append(make_eval_record(
            dataset="where2place",
            uid=str(row["question_id"]),
            raw=image_path.read_bytes(),
            cache_dir=cache_dir,
            prompt=row.get("text"),
            answer=None,
            referents=[],
            source_ids=[f"where2place:{image_ref}"],
            scene_ids=[],
            source_identity_status="IMAGE_ID_PROVIDED_BY_RELEASE",
            scene_identity_status="NOT_PROVIDED_BY_RELEASE",
            source_metadata={"image": image_ref, "category": row.get("category")},
            existing_path=image_path,
        ))
    return records, assets


def load_refspatial(cache_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise FullLeakageError("pyarrow is required for RefSpatial-Bench") from exc
    records: list[dict[str, Any]] = []
    assets: list[Path] = []
    for split in ("location", "placement", "unseen"):
        path = DATA_ROOT / "RefSpatial-Bench" / "data" / f"{split}-00000-of-00001.parquet"
        if not path.is_file():
            raise FullLeakageError(f"missing RefSpatial asset: {path}")
        assets.append(path)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                raw = (row.get("image") or {}).get("bytes")
                if not raw:
                    raise FullLeakageError(f"RefSpatial {split}/{row.get('id')} has no image")
                uid = f"{split}_{row['id']}"
                records.append(make_eval_record(
                    dataset="refspatial_bench",
                    uid=uid,
                    raw=raw,
                    cache_dir=cache_dir,
                    prompt=row.get("prompt"),
                    answer=None,
                    referents=[row.get("object")],
                    source_ids=[f"refspatial:{split}:{row.get('id')}"],
                    scene_ids=[],
                    source_identity_status="ROW_ID_PROVIDED_BY_RELEASE",
                    scene_identity_status="NOT_PROVIDED_BY_RELEASE",
                    source_metadata={"split": split, "step": row.get("step")},
                    existing_path=REPOSITORY_ROOT / ".cache/eval_assets/refspatial_bench" / f"{uid}.jpg",
                ))
    return records, assets


def load_gsa_test(cache_dir: Path, path: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    if not path.is_file():
        raise FullLeakageError(f"missing GSA test: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        images = row.get("images") or []
        messages = row.get("messages") or []
        if not images or len(messages) < 2:
            raise FullLeakageError(f"malformed GSA test row {line_number}")
        image_path = Path(str(images[0]))
        if not image_path.is_file():
            raise FullLeakageError(f"unresolved GSA test image {line_number}: {image_path}")
        meta = row.get("gsa_meta") or {}
        answer = messages[1].get("content")
        records.append(make_eval_record(
            dataset="gsa_bench_test",
            uid=f"{meta.get('form', 'unknown')}_{line_number - 1}",
            raw=image_path.read_bytes(),
            cache_dir=cache_dir,
            prompt=messages[0].get("content"),
            answer=answer,
            referents=extract_referents(answer),
            source_ids=[f"gsa:{meta.get('source')}:{image_path.resolve()}"],
            scene_ids=[],
            source_identity_status="SOURCE_LABEL_AND_IMAGE_PATH_PROVIDED",
            scene_identity_status="COARSE_SCENE_LABEL_ONLY",
            source_metadata={
                "source": meta.get("source"),
                "scene_label": meta.get("scene"),
                "original_image_path": str(image_path.resolve()),
            },
            existing_path=image_path,
        ))
    return records, [path]


def build_eval_records(cache_dir: Path, gsa_test: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    records: list[dict[str, Any]] = []
    assets: list[Path] = []
    for loader in (load_cv_bench, load_embspatial, load_where2place, load_refspatial):
        batch, bound = loader(cache_dir)
        records.extend(batch)
        assets.extend(bound)
    batch, bound = load_gsa_test(cache_dir, gsa_test)
    records.extend(batch)
    assets.extend(bound)
    return records, sorted(set(assets), key=str)


def unique_eval_images(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sha: dict[str, dict[str, Any]] = {}
    for record in records:
        sha = record["sha256"]
        if sha not in by_sha:
            by_sha[sha] = {
                **record,
                "datasets": [record["dataset"]],
                "uids": [record["uid"]],
            }
        else:
            by_sha[sha]["datasets"] = sorted(set(by_sha[sha]["datasets"] + [record["dataset"]]))
            by_sha[sha]["uids"].append(record["uid"])
    return sorted(by_sha.values(), key=lambda row: (row["dataset"], row["uid"]))


def load_extra_eval_records(names: list[str], cache_dir: Path):
    """Audit the exact single-image local evaluation inputs, never infer scene IDs."""
    from eval.gsa_eval.registry import get_spec
    records, assets, counts = [], [], {}
    for name in names:
        if name in counts:
            raise FullLeakageError(f"duplicate extra benchmark: {name}")
        spec = get_spec(name)
        if not spec.available or spec.loader is None or not spec.n_expected:
            raise FullLeakageError(f"extra benchmark lacks a frozen loader/count: {name}")
        samples = spec.loader(cache_dir / "expanded_inputs")
        if len(samples) != spec.n_expected or len({s.uid for s in samples}) != len(samples):
            raise FullLeakageError(f"extra benchmark coverage/UID mismatch: {name}")
        counts[name] = len(samples)
        for sample in samples:
            if len(sample.image_paths) != 1:
                raise FullLeakageError(f"extra benchmark is not single-image: {name}/{sample.uid}")
            path = Path(sample.image_paths[0])
            answer = sample.gt
            if isinstance(answer, str) and Path(answer).is_absolute() and Path(answer).is_file():
                assets.append(Path(answer))
                answer = {"reference_file_sha256": file_sha256(Path(answer))}
            records.append(make_eval_record(
                dataset=name, uid=sample.uid, raw=path.read_bytes(), cache_dir=cache_dir,
                prompt=sample.prompt, answer=answer, referents=[], source_ids=[], scene_ids=[],
                source_identity_status="NOT_PROVIDED_BY_RELEASE",
                scene_identity_status="NOT_PROVIDED_BY_RELEASE",
                source_metadata={"evaluation_uid": sample.uid, "group": sample.group,
                                 "limitation": "Local UID is not a recovered source/scene ID"},
                existing_path=path))
            assets.append(path)
    return records, sorted(set(assets), key=str), counts


def load_verified_training_cache(
    *,
    corpus: Path,
    expected_rows: int,
    expected_sha256: str,
    cache: Path,
) -> tuple[list[dict[str, Any]], dict[str, set[str]], int, dict[str, Any]]:
    corpus_sha = file_sha256(corpus)
    rows, paths, source_labels = collect_corpus_paths(corpus)
    if corpus_sha != expected_sha256:
        raise FullLeakageError("post-training corpus sha256 drift")
    if rows != expected_rows:
        raise FullLeakageError(f"post-training corpus rows {rows} != {expected_rows}")
    binding_path = cache.with_suffix(".binding.json")
    if not cache.is_file() or not binding_path.is_file():
        raise FullLeakageError("verified training fingerprint cache is missing")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    expected_binding = {
        "corpus_sha256": corpus_sha,
        "path_set_sha256": canonical_hash(sorted(paths)),
        "unique_image_paths": len(paths),
        "fingerprint_schema_version": 2,
    }
    for key, expected in expected_binding.items():
        if binding.get(key) != expected:
            raise FullLeakageError(f"training fingerprint binding drift: {key}")
    if file_sha256(cache) != binding.get("fingerprint_manifest_sha256"):
        raise FullLeakageError("training fingerprint cache file hash drift")
    training: list[dict[str, Any]] = []
    with cache.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("state") != "OK":
                raise FullLeakageError(f"incomplete training fingerprint at line {line_number}")
            for key in ("sha256", "decoded_rgb_sha256", "pillow_default_jpeg_sha256", "phash64", "dhash64"):
                if not row.get(key):
                    raise FullLeakageError(f"training fingerprint lacks {key} at line {line_number}")
            training.append(row)
    if len(training) != len(paths):
        raise FullLeakageError("training fingerprint cache coverage mismatch")
    return training, source_labels, rows, binding


def _pair_id(kind: str, evaluation: dict[str, Any], training: dict[str, Any]) -> str:
    return canonical_hash({
        "kind": kind,
        "eval_dataset": evaluation["dataset"],
        "eval_sha256": evaluation["sha256"],
        "training_sha256": training["sha256"],
    })


def image_collisions(
    eval_images: list[dict[str, Any]], training: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    axes = {
        "encoded_bytes": "sha256",
        "decoded_rgb": "decoded_rgb_sha256",
        "pillow_default_jpeg": "pillow_default_jpeg_sha256",
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for axis, field in axes.items():
        train_by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in training:
            train_by_value[row[field]].append(row)
        pairs: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for evaluation in eval_images:
            for train in train_by_value.get(evaluation[field], []):
                key = (evaluation["sha256"], train["sha256"])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({
                    "pair_id": _pair_id(axis, evaluation, train),
                    "axis": axis,
                    "eval_dataset": evaluation["dataset"],
                    "eval_datasets": evaluation.get("datasets", [evaluation["dataset"]]),
                    "eval_uids": evaluation.get("uids", [evaluation["uid"]]),
                    "eval_path": evaluation["path"],
                    "eval_sha256": evaluation["sha256"],
                    "training_path": train["raw_path"],
                    "training_resolved_path": train["path"],
                    "training_sha256": train["sha256"],
                })
        output[axis] = sorted(pairs, key=lambda row: row["pair_id"])
    return output


def _path_normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "/", value.casefold()).strip("/")


def source_and_scene_collisions(
    eval_records: list[dict[str, Any]],
    training: list[dict[str, Any]],
    source_labels: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_basename: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_resolved_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in training:
        by_basename[Path(row["raw_path"]).name.casefold()].append(row)
        by_resolved_path[str(Path(row["path"]).resolve())].append(row)
        for component in set(Path(row["raw_path"]).parts) | set(Path(row["path"]).parts):
            normalized_component = normalize_text(component)
            if normalized_component:
                by_component[normalized_component].append(row)

    source_hits: list[dict[str, Any]] = []
    scene_hits: list[dict[str, Any]] = []
    seen_source: set[tuple[str, str]] = set()
    seen_scene: set[tuple[str, str]] = set()
    for evaluation in eval_records:
        metadata = evaluation.get("source_metadata") or {}
        candidates: list[dict[str, Any]] = []
        if evaluation["dataset"] == "where2place":
            image_ref = str(metadata.get("image") or "")
            candidates = by_basename.get(Path(image_ref).name.casefold(), [])
            candidates = [
                row for row in candidates
                if "where2place" in _path_normalized(
                    " ".join([row["raw_path"], *source_labels.get(row["raw_path"], set())])
                )
            ]
        elif evaluation["dataset"] == "gsa_bench_test":
            expected = str(metadata.get("original_image_path") or "")
            candidates = by_resolved_path.get(expected, [])
        elif evaluation["dataset"] == "cv_bench":
            source_filename = str(metadata.get("source_filename") or "")
            if source_filename:
                suffix = _path_normalized(source_filename)
                candidates = [
                    row for row in by_basename.get(Path(source_filename).name.casefold(), [])
                    if _path_normalized(
                        " ".join([row["raw_path"], row["path"]])
                    ).endswith(suffix)
                ]
        for train in candidates:
            key = (evaluation["uid"], train["raw_path"])
            if key in seen_source:
                continue
            seen_source.add(key)
            source_hits.append({
                "eval_dataset": evaluation["dataset"],
                "eval_uid": evaluation["uid"],
                "eval_source_ids": evaluation["source_ids"],
                "training_path": train["raw_path"],
                "training_source_labels": sorted(source_labels.get(train["raw_path"], set())),
                "policy": "dataset-qualified source identity is a hard leak",
            })

        for scene_id in evaluation["scene_ids"]:
            scene_token = normalize_text(scene_id.split(":", 1)[-1])
            if not scene_token:
                continue
            for train in by_component.get(scene_token, []):
                key = (scene_id, train["raw_path"])
                if key in seen_scene:
                    continue
                seen_scene.add(key)
                scene_hits.append({
                    "eval_dataset": evaluation["dataset"],
                    "eval_uid": evaluation["uid"],
                    "eval_scene_id": scene_id,
                    "training_path": train["raw_path"],
                    "training_source_labels": sorted(source_labels.get(train["raw_path"], set())),
                    "policy": "dataset-qualified scene identity is a hard leak",
                })
    return (
        sorted(source_hits, key=lambda row: (row["eval_dataset"], row["eval_uid"], row["training_path"])),
        sorted(scene_hits, key=lambda row: (row["eval_dataset"], row["eval_uid"], row["training_path"])),
    )


def text_collision_summary(
    corpus: Path, eval_records: list[dict[str, Any]]
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    fields = ("prompt_sha256", "answer_sha256", "prompt_answer_sha256")
    eval_index: dict[str, dict[str, list[dict[str, str]]]] = {
        field: defaultdict(list) for field in fields
    }
    eval_referents: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eval_records:
        item = {"dataset": row["dataset"], "uid": row["uid"]}
        for field in fields:
            if row.get(field):
                eval_index[field][row[field]].append(item)
        for digest in row.get("referent_sha256", []):
            eval_referents[digest].append(item)

    matched: dict[str, set[str]] = {field: set() for field in fields}
    matched["referent_sha256"] = set()
    hard_examples: list[dict[str, Any]] = []
    seen_hard: set[tuple[str, str]] = set()
    with corpus.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            prompt = " ".join(
                str(message.get("content", "")) for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            )
            answer = " ".join(
                str(message.get("content", "")) for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            )
            values = {
                "prompt_sha256": text_sha(prompt),
                "answer_sha256": text_sha(answer),
                "prompt_answer_sha256": text_sha(
                    f"{normalize_text(prompt)}\n{normalize_text(answer)}"
                ) if prompt and answer else None,
            }
            for field, digest in values.items():
                if digest and digest in eval_index[field]:
                    matched[field].add(digest)
                    if field in {"prompt_sha256", "prompt_answer_sha256"}:
                        for item in eval_index[field][digest]:
                            key = (field, digest)
                            if key in seen_hard:
                                continue
                            seen_hard.add(key)
                            hard_examples.append({
                                "axis": field.removesuffix("_sha256"),
                                "sha256": digest,
                                "eval_dataset": item["dataset"],
                                "eval_uid": item["uid"],
                                "training_corpus_line": line_number,
                                "policy": "exact normalized evaluation prompt is a hard annotation leak candidate",
                            })
            for referent in extract_referents(answer):
                digest = text_sha(referent)
                if digest in eval_referents:
                    matched["referent_sha256"].add(digest)
    summary = {f"unique_{key.removesuffix('_sha256')}_overlaps": len(value) for key, value in matched.items()}
    return summary, sorted(hard_examples, key=lambda row: (row["axis"], row["sha256"]))


def load_adjudications(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    decisions: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = str(row.get("pair_id") or "")
            decision = str(row.get("decision") or "").upper()
            reviewer = str(row.get("reviewer") or "").strip()
            comment = str(row.get("comment") or "").strip()
            if not pair_id or pair_id in decisions:
                raise FullLeakageError(f"invalid/duplicate adjudication pair_id at line {line_number}")
            if decision not in {"LEAK", "NOT_LEAK"} or not reviewer or not comment:
                raise FullLeakageError(f"incomplete adjudication at line {line_number}")
            decisions[pair_id] = row
    return decisions


def adjudicate_near_candidates(
    candidates: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    known = {row["pair_id"] for row in candidates}
    unknown = sorted(set(decisions) - known)
    if unknown:
        raise FullLeakageError(f"adjudication contains {len(unknown)} unknown pair_id values")
    unresolved, leaks, cleared = [], [], []
    for row in candidates:
        decision = decisions.get(row["pair_id"])
        if decision is None:
            unresolved.append({**row, "decision": "UNRESOLVED", "reviewer": "", "comment": ""})
        elif str(decision["decision"]).upper() == "LEAK":
            leaks.append({**row, **decision})
        else:
            cleared.append({**row, **decision})
    return unresolved, leaks, cleared


def determine_state(
    *,
    hard_leak: bool,
    unresolved_near: int,
    metadata_blockers: int,
) -> str:
    """Legacy single-axis verdict, retained verbatim for backward compatibility.

    This conflates two independent things: whether the training corpus is
    contaminated (which we control) and whether the evaluation release exposes a
    unique scene identity (which the benchmark publisher controls). Prefer
    ``determine_contamination_verdict`` and ``determine_metadata_verdict``.
    """
    if hard_leak:
        return "LEAKAGE_DETECTED"
    if unresolved_near or metadata_blockers:
        return "BLOCKED"
    return "PASS"


def determine_contamination_verdict(*, hard_leak: bool, unresolved_near: int) -> str:
    """Training-corpus contamination only. Thresholds are unchanged from v1.

    Covers encoded bytes, decoded RGB, deterministic JPEG normalization,
    release-qualified source identity, scene identity, normalized-text hard
    candidates, adjudicated perceptual leaks, and unresolved perceptual
    candidates. Evaluation-release metadata completeness is deliberately excluded
    because no corpus rebuild can change it.
    """
    if hard_leak:
        return "LEAKAGE_DETECTED"
    if unresolved_near:
        return "BLOCKED_PENDING_ADJUDICATION"
    return "PASS"


def determine_metadata_verdict(*, metadata_blockers: int) -> str:
    """Evaluation-release metadata completeness. A publisher property, not a leak."""
    if metadata_blockers:
        return "INCOMPLETE_EVALUATION_RELEASE_METADATA"
    return "COMPLETE"


def benchmark_breakdown(
    *,
    dataset: str,
    eval_records: list[dict[str, Any]],
    collisions: dict[str, list[dict[str, Any]]],
    source_hits: list[dict[str, Any]],
    scene_hits: list[dict[str, Any]],
    text_hard: list[dict[str, Any]],
    near: list[dict[str, Any]],
    unresolved_near: list[dict[str, Any]],
    near_leaks: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_records = [row for row in eval_records if row["dataset"] == dataset]
    dataset_shas = {row["sha256"] for row in dataset_records}

    def image_axis(axis: str) -> dict[str, int]:
        rows = [
            row for row in collisions[axis]
            if dataset in row.get("eval_datasets", [row["eval_dataset"]])
        ]
        affected_shas = {row["eval_sha256"] for row in rows}
        return {
            "pairs": len({(row["eval_sha256"], row["training_sha256"]) for row in rows}),
            "eval_images": len(affected_shas),
            "eval_records": sum(row["sha256"] in affected_shas for row in dataset_records),
        }

    source_rows = [row for row in source_hits if row["eval_dataset"] == dataset]
    scene_rows = [row for row in scene_hits if row["eval_dataset"] == dataset]
    text_rows = [row for row in text_hard if row["eval_dataset"] == dataset]
    near_rows = [
        row for row in near
        if dataset in row.get("eval_datasets", [row["eval_dataset"]])
    ]
    unresolved_rows = [
        row for row in unresolved_near
        if dataset in row.get("eval_datasets", [row["eval_dataset"]])
    ]
    near_leak_rows = [
        row for row in near_leaks
        if dataset in row.get("eval_datasets", [row["eval_dataset"]])
    ]
    missing_source = sum(
        row["source_identity_status"] == "NOT_PROVIDED_BY_RELEASE"
        for row in dataset_records
    )
    missing_scene = sum(
        row["scene_identity_status"] in {
            "NOT_PROVIDED_BY_RELEASE", "COARSE_SCENE_LABEL_ONLY"
        }
        for row in dataset_records
    )
    hard_leak = bool(
        image_axis("encoded_bytes")["pairs"]
        or image_axis("decoded_rgb")["pairs"]
        or image_axis("pillow_default_jpeg")["pairs"]
        or source_rows or scene_rows or text_rows or near_leak_rows
    )
    metadata_blockers = int(bool(missing_source or missing_scene))
    state = determine_state(
        hard_leak=hard_leak,
        unresolved_near=len(unresolved_rows),
        metadata_blockers=metadata_blockers,
    )
    contamination_verdict = determine_contamination_verdict(
        hard_leak=hard_leak, unresolved_near=len(unresolved_rows)
    )
    metadata_verdict = determine_metadata_verdict(metadata_blockers=metadata_blockers)
    return {
        "state": state,
        "contamination_verdict": contamination_verdict,
        "metadata_completeness_verdict": metadata_verdict,
        "verdict_axis_note": (
            "contamination_verdict reflects the training corpus and is the axis this "
            "project controls; metadata_completeness_verdict reflects what the "
            "evaluation release publishes and is unaffected by any corpus rebuild"
        ),
        "records": len(dataset_records),
        "unique_encoded_images": len(dataset_shas),
        "encoded_bytes": image_axis("encoded_bytes"),
        "decoded_rgb": image_axis("decoded_rgb"),
        "pillow_default_jpeg": image_axis("pillow_default_jpeg"),
        "source_id_pairs": len(source_rows),
        "scene_id_pairs": len(scene_rows),
        "normalized_text_hard_candidates": len(text_rows),
        "perceptual_candidates": len(near_rows),
        "perceptual_unresolved": len(unresolved_rows),
        "perceptual_adjudicated_leak": len(near_leak_rows),
        "missing_source_identity_records": missing_source,
        "missing_unique_scene_identity_records": missing_scene,
    }


def refspatial_location_breakdown(
    *,
    eval_records: list[dict[str, Any]],
    collisions: dict[str, list[dict[str, Any]]],
    near: list[dict[str, Any]],
    unresolved_near: list[dict[str, Any]],
    near_leaks: list[dict[str, Any]],
) -> dict[str, Any]:
    location_records = [
        row for row in eval_records
        if row["dataset"] == "refspatial_bench" and row["uid"].startswith("location_")
    ]

    def exact(axis: str) -> dict[str, int]:
        rows = [
            row for row in collisions[axis]
            if "refspatial_bench" in row.get("eval_datasets", [row["eval_dataset"]])
            and any(str(uid).startswith("location_") for uid in row.get("eval_uids", []))
        ]
        affected = {row["eval_sha256"] for row in rows}
        return {
            "pairs": len({(row["eval_sha256"], row["training_sha256"]) for row in rows}),
            "eval_images": len(affected),
            "eval_records": sum(row["sha256"] in affected for row in location_records),
        }

    def location_near(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row for row in rows
            if "refspatial_bench" in row.get("eval_datasets", [row["eval_dataset"]])
            and any(str(uid).startswith("location_") for uid in row.get("eval_uids", []))
        ]

    unresolved = location_near(unresolved_near)
    leaks = location_near(near_leaks)
    exact_leak = any(exact(axis)["pairs"] for axis in collisions)
    return {
        "name": "location",
        "records": len(location_records),
        "state": determine_state(
            hard_leak=bool(exact_leak or leaks),
            unresolved_near=len(unresolved),
            metadata_blockers=0,
        ),
        "contamination_verdict": determine_contamination_verdict(
            hard_leak=bool(exact_leak or leaks), unresolved_near=len(unresolved)
        ),
        "encoded_bytes": exact("encoded_bytes"),
        "decoded_rgb": exact("decoded_rgb"),
        "pillow_default_jpeg": exact("pillow_default_jpeg"),
        "perceptual_candidates": len(location_near(near)),
        "perceptual_unresolved": len(unresolved),
        "perceptual_adjudicated_leak": len(leaks),
        "unresolved_uids": sorted({
            uid for row in unresolved for uid in row.get("eval_uids", [])
            if str(uid).startswith("location_")
        }),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--expected-corpus-rows", type=int, default=DEFAULT_CORPUS_ROWS)
    parser.add_argument("--expected-corpus-sha256", default=DEFAULT_CORPUS_SHA256)
    parser.add_argument(
        "--training-cache", type=Path,
        default=REPOSITORY_ROOT / "artifacts/general_retention_v1/training_image_fingerprints.jsonl",
    )
    parser.add_argument(
        "--gsa-test", type=Path,
        default=REPOSITORY_ROOT / "trains/gsa_bench/test.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPOSITORY_ROOT / "artifacts/leakage_audit_full_v1",
    )
    parser.add_argument("--phash-radius", type=int, default=4)
    parser.add_argument("--dhash-radius", type=int, default=4)
    parser.add_argument("--max-mae", type=float, default=0.06)
    parser.add_argument("--min-correlation", type=float, default=0.97)
    parser.add_argument("--adjudications", type=Path, default=None)
    parser.add_argument("--extra-benchmarks", nargs="*", default=[],
                        help="Additional registered single-image full subsets; original five remain included")
    parser.add_argument(
        "--exit-on",
        choices=("legacy_state", "contamination"),
        default="legacy_state",
        help=(
            "which verdict axis drives the process exit code; legacy_state preserves v1 "
            "behaviour (default), contamination ignores evaluation-release metadata "
            "completeness, which no corpus rebuild can influence"
        ),
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    cache_dir = REPOSITORY_ROOT / ".cache/leakage_audit_full_v1/eval_images"
    report_path = output_dir / "report.json"
    try:
        training, source_labels, corpus_rows, binding = load_verified_training_cache(
            corpus=args.corpus,
            expected_rows=args.expected_corpus_rows,
            expected_sha256=args.expected_corpus_sha256,
            cache=args.training_cache,
        )
        eval_records, source_assets = build_eval_records(cache_dir, args.gsa_test)
        expected_counts = {
            "cv_bench": 2638,
            "embspatial": 3640,
            "where2place": 100,
            "refspatial_bench": 277,
            "gsa_bench_test": 1463,
        }
        if set(args.extra_benchmarks) & set(expected_counts):
            raise FullLeakageError("extra benchmarks repeat an existing audit dataset")
        extra_records, extra_assets, extra_counts = load_extra_eval_records(
            args.extra_benchmarks, cache_dir)
        eval_records.extend(extra_records)
        source_assets.extend(extra_assets)
        source_assets = sorted(set(source_assets), key=str)
        expected_counts.update(extra_counts)
        observed_counts = Counter(row["dataset"] for row in eval_records)
        if dict(observed_counts) != expected_counts:
            raise FullLeakageError(
                f"evaluation coverage mismatch: {dict(observed_counts)} != {expected_counts}"
            )
        eval_images = unique_eval_images(eval_records)
        eval_manifest_path = output_dir / "eval_image_fingerprints.jsonl"
        eval_manifest_sha = write_jsonl(eval_manifest_path, eval_images)

        collisions = image_collisions(eval_images, training)
        source_hits, scene_hits = source_and_scene_collisions(
            eval_records, training, source_labels
        )
        text_summary, text_hard = text_collision_summary(args.corpus, eval_records)

        near_raw = near_duplicate_candidates(
            eval_images,
            training,
            phash_radius=args.phash_radius,
            dhash_radius=args.dhash_radius,
            max_mae=args.max_mae,
            min_correlation=args.min_correlation,
        )
        training_by_path = {row["raw_path"]: row for row in training}
        eval_by_sha = {row["sha256"]: row for row in eval_images}
        near: list[dict[str, Any]] = []
        for row in near_raw:
            evaluation = eval_by_sha[row["eval_sha256"]]
            train = training_by_path[row["training_path"]]
            near.append({
                **row,
                "eval_datasets": evaluation.get("datasets", [evaluation["dataset"]]),
                "eval_uids": evaluation.get("uids", [evaluation["uid"]]),
                "pair_id": _pair_id("perceptual_near_duplicate", evaluation, train),
            })
        near = sorted(near, key=lambda row: row["pair_id"])
        adjudication_path = (
            args.adjudications.resolve()
            if args.adjudications else output_dir / "near_duplicate_adjudications.jsonl"
        )
        unresolved_near, near_leaks, cleared_near = adjudicate_near_candidates(
            near, load_adjudications(adjudication_path)
        )

        hard_image_pairs = {
            (row["eval_sha256"], row["training_sha256"])
            for axis_rows in collisions.values()
            for row in axis_rows
        }
        source_pair_count = len(source_hits)
        scene_pair_count = len(scene_hits)
        missing_source = Counter(
            row["dataset"] for row in eval_records
            if row["source_identity_status"] == "NOT_PROVIDED_BY_RELEASE"
        )
        missing_scene = Counter(
            row["dataset"] for row in eval_records
            if row["scene_identity_status"] in {
                "NOT_PROVIDED_BY_RELEASE", "COARSE_SCENE_LABEL_ONLY"
            }
        )
        metadata_blockers = []
        if missing_source:
            metadata_blockers.append("source identity unavailable for some evaluation rows")
        if missing_scene:
            metadata_blockers.append(
                "unique scene identity unavailable for some evaluation rows; exact image checks do not prove scene separation"
            )

        hard_leak = bool(
            hard_image_pairs or source_hits or scene_hits or text_hard or near_leaks
        )
        state = determine_state(
            hard_leak=hard_leak,
            unresolved_near=len(unresolved_near),
            metadata_blockers=len(metadata_blockers),
        )
        contamination_verdict = determine_contamination_verdict(
            hard_leak=hard_leak, unresolved_near=len(unresolved_near)
        )
        metadata_completeness_verdict = determine_metadata_verdict(
            metadata_blockers=len(metadata_blockers)
        )
        per_benchmark = {
            dataset: benchmark_breakdown(
                dataset=dataset,
                eval_records=eval_records,
                collisions=collisions,
                source_hits=source_hits,
                scene_hits=scene_hits,
                text_hard=text_hard,
                near=near,
                unresolved_near=unresolved_near,
                near_leaks=near_leaks,
            )
            for dataset in expected_counts
        }
        per_benchmark["refspatial_bench"]["scored_subgroup"] = (
            refspatial_location_breakdown(
                eval_records=eval_records,
                collisions=collisions,
                near=near,
                unresolved_near=unresolved_near,
                near_leaks=near_leaks,
            )
        )
        claim_implications = {
            "scope": "only checkpoints trained on the audited E1 corpus are implicated",
            "unaffected_by_this_audit_alone": [
                "base and public baseline checkpoints that did not consume the E1 corpus",
                "the separately audited AI2D/ScienceQA/TextVQA General retention suite",
            ],
            "gsa_joint_sft_e1": {
                "cv_bench": (
                    "WITHHOLD_PENDING_"
                    f"{per_benchmark['cv_bench']['perceptual_unresolved']}_NEAR_DUPLICATE_DECISIONS"
                ),
                "embspatial": "NO_IMAGE_OR_TEXT_COLLISION_DETECTED_UNDER_V1_PROTOCOL",
                "refspatial_location": (
                    "WITHHOLD_PENDING_NEAR_DUPLICATE_DECISIONS:"
                    + ",".join(
                        per_benchmark["refspatial_bench"]["scored_subgroup"]["unresolved_uids"]
                    )
                ),
                "refspatial_other_groups": (
                    "CONTAMINATED_EXACT_IMAGES_NOT_USED_BY_MAIN_LOCATION_CELL:"
                    f"{per_benchmark['refspatial_bench']['encoded_bytes']['eval_images']}"
                ),
                "where2place": (
                    "QUARANTINE_EXACT_EVAL_IMAGES_IN_TRAINING:"
                    f"{per_benchmark['where2place']['encoded_bytes']['eval_images']}"
                    f"/{per_benchmark['where2place']['unique_encoded_images']}"
                ),
                "gsa_bench_test": (
                    "QUARANTINE_EXACT_TRAIN_TEST_IMAGE_COLLISIONS:"
                    f"{per_benchmark['gsa_bench_test']['encoded_bytes']['eval_images']}"
                ),
            },
        }

        artifact_hashes = {
            "encoded_byte_collisions.jsonl": write_jsonl(
                output_dir / "encoded_byte_collisions.jsonl", collisions["encoded_bytes"]
            ),
            "decoded_rgb_collisions.jsonl": write_jsonl(
                output_dir / "decoded_rgb_collisions.jsonl", collisions["decoded_rgb"]
            ),
            "jpeg_normalized_collisions.jsonl": write_jsonl(
                output_dir / "jpeg_normalized_collisions.jsonl", collisions["pillow_default_jpeg"]
            ),
            "source_id_collisions.jsonl": write_jsonl(
                output_dir / "source_id_collisions.jsonl", source_hits
            ),
            "scene_id_collisions.jsonl": write_jsonl(
                output_dir / "scene_id_collisions.jsonl", scene_hits
            ),
            "normalized_text_collisions.jsonl": write_jsonl(
                output_dir / "normalized_text_collisions.jsonl", text_hard
            ),
            "near_duplicate_candidates.jsonl": write_jsonl(
                output_dir / "near_duplicate_candidates.jsonl", near
            ),
            "near_duplicate_pending_review.jsonl": write_jsonl(
                output_dir / "near_duplicate_pending_review.jsonl", unresolved_near
            ),
        }
        report: dict[str, Any] = {
            "schema_version": 1,
            "audit_id": "multi-benchmark-leakage-v1",
            "state": state,
            "contamination_verdict": contamination_verdict,
            "metadata_completeness_verdict": metadata_completeness_verdict,
            "verdict_axes": {
                "protocol_version": "v1.1-separated-verdicts",
                "rationale": (
                    "v1 folded evaluation-release metadata completeness into the same "
                    "verdict as training-corpus contamination. Those are independent: a "
                    "public release that ships no unique scene identifier cannot be fixed "
                    "by rebuilding the training corpus, so a fully decontaminated corpus "
                    "could never be reported as clean. The axes are now reported "
                    "separately; no detection threshold was changed."
                ),
                "contamination_verdict": {
                    "value": contamination_verdict,
                    "owner": "this project (training corpus)",
                    "covers": [
                        "encoded_byte_collisions",
                        "decoded_rgb_collisions",
                        "jpeg_normalized_collisions",
                        "release_qualified_source_id_collisions",
                        "scene_id_collisions",
                        "normalized_text_hard_candidates",
                        "perceptual_adjudicated_leak",
                        "perceptual_unresolved",
                    ],
                    "actionable": True,
                },
                "metadata_completeness_verdict": {
                    "value": metadata_completeness_verdict,
                    "owner": "benchmark publishers (evaluation releases)",
                    "covers": [
                        "missing_source_identity_records",
                        "missing_unique_scene_identity_records",
                    ],
                    "actionable_by_corpus_rebuild": False,
                    "interpretation": (
                        "a zero in the scene column means no collision was detected among "
                        "available identifiers; it does not prove scene separation where "
                        "the public release lacks a unique scene identity. This is "
                        "disclosed, not suppressed, and is unrelated to training-corpus "
                        "contamination."
                    ),
                },
                "legacy_state_note": (
                    "the top-level `state` and `claim_eligible` fields retain their v1 "
                    "semantics for backward compatibility with existing consumers"
                ),
            },
            "auditor": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__)),
            },
            "posttraining_corpus": {
                "path": str(args.corpus.resolve()),
                "sha256": file_sha256(args.corpus),
                "rows": corpus_rows,
                "unique_images": len(training),
                "fingerprint_cache_sha256": file_sha256(args.training_cache),
                "fingerprint_binding_identity_sha256": binding["identity_sha256"],
            },
            "evaluation": {
                "records_by_dataset": dict(sorted(observed_counts.items())),
                "records": len(eval_records),
                "unique_encoded_images": len(eval_images),
                "eval_image_manifest": str(eval_manifest_path),
                "eval_image_manifest_sha256": eval_manifest_sha,
                "source_assets": [
                    {"path": str(path.resolve()), "sha256": file_sha256(path)}
                    for path in source_assets
                ],
            },
            "collision_counts": {
                "encoded_byte_pairs": len(collisions["encoded_bytes"]),
                "decoded_rgb_pairs": len(collisions["decoded_rgb"]),
                "jpeg_normalized_pairs": len(collisions["pillow_default_jpeg"]),
                "unique_hard_image_pairs": len(hard_image_pairs),
                "source_id_pairs": source_pair_count,
                "scene_id_pairs": scene_pair_count,
                **text_summary,
                "normalized_text_hard_candidates": len(text_hard),
                "perceptual_candidates": len(near),
                "perceptual_unresolved": len(unresolved_near),
                "perceptual_adjudicated_leak": len(near_leaks),
                "perceptual_adjudicated_not_leak": len(cleared_near),
            },
            "per_benchmark": per_benchmark,
            "claim_implications": claim_implications,
            "identity_coverage": {
                "missing_source_identity_by_dataset": dict(sorted(missing_source.items())),
                "missing_unique_scene_identity_by_dataset": dict(sorted(missing_scene.items())),
                "metadata_blockers": metadata_blockers,
            },
            "near_duplicate_protocol": {
                "phash64_hamming_max": args.phash_radius,
                "dhash64_hamming_max": args.dhash_radius,
                "absolute_log_aspect_ratio_max": 0.02,
                "normalized_rgb_mae_max": args.max_mae,
                "grayscale_correlation_min": args.min_correlation,
                "adjudication_policy": "every retained candidate requires LEAK or NOT_LEAK with reviewer and comment",
                "adjudication_path": str(adjudication_path),
            },
            "artifact_sha256": artifact_hashes,
            # Unchanged v1 semantics. Not redefined in terms of contamination_verdict
            # so that any existing consumer keeps observing identical values.
            "claim_eligible": state == "PASS",
            "contamination_clean": contamination_verdict == "PASS",
        }
        report["identity_sha256"] = canonical_hash(report)
        write_json(report_path, report)
    except (OSError, ValueError, json.JSONDecodeError, FullLeakageError) as exc:
        report = {
            "schema_version": 1,
            "audit_id": "multi-benchmark-leakage-v1",
            "state": "INCOMPLETE",
            "contamination_verdict": "INCOMPLETE",
            "metadata_completeness_verdict": "INCOMPLETE",
            "error": str(exc),
            "claim_eligible": False,
            "contamination_clean": False,
        }
        report["identity_sha256"] = canonical_hash(report)
        write_json(report_path, report)
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 4

    print(json.dumps({
        "state": report["state"],
        "contamination_verdict": report["contamination_verdict"],
        "metadata_completeness_verdict": report["metadata_completeness_verdict"],
        "contamination_clean": report["contamination_clean"],
        "records_by_dataset": report["evaluation"]["records_by_dataset"],
        "collision_counts": report["collision_counts"],
        "metadata_blockers": report["identity_coverage"]["metadata_blockers"],
        "report": str(report_path),
        "identity_sha256": report["identity_sha256"],
    }, indent=2))
    # Exit codes keep v1 meaning so scripts/run_leakage_audit_full.sh and any CI
    # gate behave identically. --exit-on selects which axis drives the code.
    driving_state = (
        report["state"] if args.exit_on == "legacy_state"
        else {
            "PASS": "PASS",
            "LEAKAGE_DETECTED": "LEAKAGE_DETECTED",
            "BLOCKED_PENDING_ADJUDICATION": "BLOCKED",
        }[report["contamination_verdict"]]
    )
    return 0 if driving_state == "PASS" else (3 if driving_state == "LEAKAGE_DETECTED" else 4)


if __name__ == "__main__":
    raise SystemExit(main())
