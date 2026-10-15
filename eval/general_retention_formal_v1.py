#!/usr/bin/env python3
"""Frozen, provenance-bound general-retention evaluation on four L20 GPUs.

This module is intentionally separate from gsa-eval-v7.  It converts the three
official VLMEvalKit TSV releases into a compact runtime index during CPU preflight,
then runs four exact stride shards through the repository's proven
``TransformersEngine`` primitive.  Scoring is deterministic and never uses a judge.
"""
from __future__ import annotations

import argparse
import ast
import base64
import binascii
import csv
import functools
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import string
import sys
import tempfile
import subprocess
from typing import Any, Callable, Iterable

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL = "general-retention-formal-v1"
SUITE_ID = "general-retention-v1"
NUM_SHARDS = 4
BATCH_SIZE = 16
DATASET_ORDER = ("ai2d_test", "scienceqa_test", "textvqa_val")
MODEL_ROLES = ("base", "e1")
CANONICAL_CONFIG = ROOT / "configs/eval/general_retention_formal_v1.yaml"
FORMAL_SOURCE_IDENTITY = "4687e366d03aafa35e673fb88129d66d55088943ff44f35110996a1669fa8897"
FORMAL_CROSSWALK_IDENTITY = "2c9c978cea2f792619ea470b5b5e214d9a66e3af936b93105ee5941f5f9d9c06"
FORMAL_LEAKAGE_IDENTITY = "bc55a602d262b2e1ffef5313dcd9c8ee8cbd4500b48140029cbae20acf344629"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_THINK_BLOCK = re.compile(r"<think>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCTUATION = (
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
    "_", "-", ">", "<", "@", "`", ",", "?", "!",
)
_MODEL_METADATA_NAMES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
    "chat_template.json",
    "chat_template.jinja",
)


class FormalProtocolError(ValueError):
    """Raised when a frozen input or run artifact drifts."""


PublicInfer = Callable[[list[dict[str, Any]], int], list[tuple[str, bool]]]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FormalProtocolError(f"missing file: {path}")
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path, *, include_mtime: bool = False) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FormalProtocolError(f"missing file: {path}")
    stat = path.stat()
    value: dict[str, Any] = {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": file_hash(path),
    }
    if include_mtime:
        value["mtime_ns"] = stat.st_mtime_ns
    return value


def verify_file_binding(binding: Any, *, deep: bool) -> Path:
    if not isinstance(binding, dict):
        raise FormalProtocolError("file binding must be an object")
    expected = {"path", "bytes", "sha256"}
    if set(binding) not in (expected, expected | {"mtime_ns"}):
        raise FormalProtocolError("file binding schema drift")
    path = Path(str(binding.get("path"))).expanduser().resolve()
    if str(path) != binding.get("path") or not path.is_file():
        raise FormalProtocolError(f"bound file is missing or non-canonical: {path}")
    stat = path.stat()
    if stat.st_size != binding.get("bytes"):
        raise FormalProtocolError(f"bound file size drift: {path}")
    if "mtime_ns" in binding and stat.st_mtime_ns != binding.get("mtime_ns"):
        raise FormalProtocolError(f"bound file mtime drift: {path}")
    sha256 = binding.get("sha256")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise FormalProtocolError(f"invalid bound sha256: {path}")
    if deep and file_hash(path) != sha256:
        raise FormalProtocolError(f"bound file hash drift: {path}")
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FormalProtocolError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FormalProtocolError(f"invalid {label}: {path} ({exc.msg})") from exc
    if not isinstance(value, dict):
        raise FormalProtocolError(f"{label} must be an object: {path}")
    return value


def load_jsonl(path: Path, label: str, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FormalProtocolError(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormalProtocolError(
                    f"{path}:{line_number}: invalid JSON ({exc.msg})"
                ) from exc
            if not isinstance(row, dict):
                raise FormalProtocolError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    if not rows and not allow_empty:
        raise FormalProtocolError(f"{label} has no rows: {path}")
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FormalProtocolError(f"missing protocol config: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FormalProtocolError("protocol config must be a mapping")
    return value


def _atomic_text(path: Path, text: str, *, idempotent: bool = True) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if idempotent and path.read_text(encoding="utf-8") == text:
            return
        raise FormalProtocolError(f"refusing to overwrite bound artifact: {path}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: Any, *, idempotent: bool = True) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        idempotent=idempotent,
    )


def atomic_jsonl(
    path: Path, rows: Iterable[dict[str, Any]], *, idempotent: bool = True
) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_text(path, text, idempotent=idempotent)


def _verify_identity(value: dict[str, Any], expected: str, label: str) -> None:
    identity = value.get("identity_sha256")
    unsigned = {key: item for key, item in value.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise FormalProtocolError(f"{label} internal identity drift")
    if identity != expected:
        raise FormalProtocolError(f"{label} identity differs from frozen config")


def _dataset_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or [row.get("id") for row in datasets] != list(DATASET_ORDER):
        raise FormalProtocolError(f"datasets must be ordered exactly as {DATASET_ORDER}")
    return datasets


def validate_protocol_config(
    config: dict[str, Any], *, allow_test_fixture: bool = False
) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("protocol_id") != PROTOCOL
        or config.get("suite_id") != SUITE_ID
    ):
        raise FormalProtocolError("protocol config identity drift")
    generation = config.get("generation") or {}
    required_generation = {
        "engine": "swift_transformers_engine",
        "logical_shards": NUM_SHARDS,
        "batch_size_per_gpu": BATCH_SIZE,
        "temperature": 0.0,
        "thinking": False,
        "device_map_per_worker": "cuda:0",
    }
    for key, expected in required_generation.items():
        if generation.get(key) != expected:
            raise FormalProtocolError(f"generation.{key} must be {expected!r}")
    health = config.get("health_admission") or {}
    if health.get("max_malformed_percent") != 5.0 or health.get("max_truncation_percent") != 1.0:
        raise FormalProtocolError("health admission thresholds drift")
    is_fixture = bool(config.get("test_fixture"))
    if is_fixture and not allow_test_fixture:
        raise FormalProtocolError("test_fixture configs are forbidden by the formal API")
    if not is_fixture:
        evidence = config.get("source_evidence") or {}
        pinned = {
            "source_manifest_identity_sha256": FORMAL_SOURCE_IDENTITY,
            "crosswalk_identity_sha256": FORMAL_CROSSWALK_IDENTITY,
            "leakage_identity_sha256": FORMAL_LEAKAGE_IDENTITY,
        }
        for key, expected in pinned.items():
            if evidence.get(key) != expected:
                raise FormalProtocolError(f"formal source pin drift: {key}")
    expected_metrics = {
        "ai2d_test": (3088, "multiple_choice_accuracy", 32),
        "scienceqa_test": (2017, "multiple_choice_accuracy", 32),
        "textvqa_val": (5000, "official_vqa_soft_accuracy", 64),
    }
    for item in _dataset_configs(config):
        dataset = item["id"]
        n, metric, max_tokens = expected_metrics[dataset]
        if not is_fixture and int(item.get("expected_n", -1)) != n:
            raise FormalProtocolError(f"{dataset} expected_n drift")
        if item.get("metric") != metric or int(item.get("max_tokens", -1)) != max_tokens:
            raise FormalProtocolError(f"{dataset} metric/token contract drift")


def _decode_adapter_image(raw: str, *, dataset: str, index: str) -> tuple[bytes, str]:
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise FormalProtocolError(
            f"{dataset}:{index}: image is neither an earlier index reference nor valid base64"
        ) from exc
    if not decoded:
        raise FormalProtocolError(f"{dataset}:{index}: empty decoded image")
    return decoded, hashlib.sha256(decoded).hexdigest()


def _parse_textvqa_answers(value: str, *, index: str) -> list[str]:
    try:
        answers = ast.literal_eval(value)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise FormalProtocolError(f"textvqa_val:{index}: invalid answer list") from exc
    if not isinstance(answers, list) or not all(isinstance(item, str) for item in answers):
        raise FormalProtocolError(f"textvqa_val:{index}: answer must be a string list")
    return answers


def _mc_prompt(row: dict[str, str], choices: list[str]) -> str:
    parts: list[str] = []
    hint = str(row.get("hint", "")).strip()
    if hint:
        parts.append(f"Hint: {hint}")
    parts.append(f"Question: {row['question']}")
    parts.append("Options:")
    parts.extend(f"{string.ascii_uppercase[index]}. {choice}" for index, choice in enumerate(choices))
    parts.append("Select the correct answer. Respond with exactly one option letter (for example, A).")
    return "\n".join(parts)


def _textvqa_prompt(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Answer the question using the text in the image. Respond with only the short answer."
    )


def _read_frozen_dataset(
    entry: dict[str, Any],
    primary_by_index: dict[str, dict[str, Any]],
    adapter_by_id: dict[str, dict[str, Any]],
    crosswalk: dict[str, Any],
    *,
    verify_image_files: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = str(entry["id"])
    tsv = Path(str(entry["tsv_path"])).expanduser().resolve()
    if not tsv.is_file():
        raise FormalProtocolError(f"missing official adapter TSV: {tsv}")
    if file_hash(tsv, "md5") != str(entry["md5"]).lower():
        raise FormalProtocolError(f"{dataset} release MD5 drift")
    if file_hash(tsv) != str(entry["sha256"]).lower():
        raise FormalProtocolError(f"{dataset} release SHA-256 drift")

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    seen: set[str] = set()
    row_images: dict[str, dict[str, Any]] = {}
    verified_paths: set[Path] = set()
    runtime_rows: list[dict[str, Any]] = []
    row_mapping_digest = hashlib.sha256()
    reference_rows = 0
    with tsv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"index", "image", "question", "answer"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise FormalProtocolError(f"{dataset} TSV columns drift")
        for row_number, row in enumerate(reader, start=2):
            index = str(row.get("index", ""))
            if not index or index in seen or index not in primary_by_index:
                raise FormalProtocolError(f"{dataset}: bad index at TSV row {row_number}: {index!r}")
            seen.add(index)
            primary = primary_by_index[index]
            if row.get("question") != primary.get("question"):
                raise FormalProtocolError(f"{dataset}:{index}: question drift")
            choices = primary.get("choices")
            if dataset == "textvqa_val":
                answers = _parse_textvqa_answers(str(row.get("answer", "")), index=index)
                if answers != primary.get("answer"):
                    raise FormalProtocolError(f"{dataset}:{index}: answers drift")
            else:
                if not isinstance(choices, list) or not choices:
                    raise FormalProtocolError(f"{dataset}:{index}: choices are missing")
                if row.get("answer") != primary.get("answer"):
                    raise FormalProtocolError(f"{dataset}:{index}: answer drift")
                for offset, expected in enumerate(choices):
                    if row.get(string.ascii_uppercase[offset], "") != expected:
                        raise FormalProtocolError(f"{dataset}:{index}: choice drift")
                answers = str(primary["answer"])
            if dataset == "ai2d_test" and row.get("image_path") != primary.get(
                "expected_adapter_image_path"
            ):
                raise FormalProtocolError(f"{dataset}:{index}: image_path drift")

            image_reference_index: str | None = None
            raw_field = str(row.get("image", ""))
            adapter_image_id = str(row.get("image_path") or primary["image_id"])
            if raw_field in row_images:
                if dataset != "textvqa_val":
                    raise FormalProtocolError(f"{dataset}:{index}: unexpected image reference")
                image_reference_index = raw_field
                reference_rows += 1
                adapter_image = row_images[raw_field]
            else:
                decoded, image_sha256 = _decode_adapter_image(
                    raw_field, dataset=dataset, index=index
                )
                adapter_image = adapter_by_id.get(adapter_image_id)
                if adapter_image is None:
                    raise FormalProtocolError(
                        f"{dataset}:{index}: adapter image ID is absent from manifest"
                    )
                if adapter_image.get("sha256") != image_sha256:
                    raise FormalProtocolError(
                        f"{dataset}:{index}: decoded image hash differs from adapter manifest"
                    )
                path = Path(adapter_image["path"]).expanduser().resolve()
                if verify_image_files and path not in verified_paths:
                    if file_hash(path) != image_sha256:
                        raise FormalProtocolError(f"{dataset}:{index}: extracted adapter image drift")
                    try:
                        with Image.open(io.BytesIO(decoded)) as image:
                            image.verify()
                    except (OSError, ValueError) as exc:
                        raise FormalProtocolError(f"{dataset}:{index}: undecodable image") from exc
                    verified_paths.add(path)
            row_images[index] = adapter_image

            if adapter_image.get("adapter_image_id") != adapter_image_id:
                raise FormalProtocolError(f"{dataset}:{index}: adapter image identity drift")
            if adapter_image.get("primary_image_id") != primary.get("image_id"):
                raise FormalProtocolError(f"{dataset}:{index}: primary image identity drift")
            if adapter_image.get("primary_image_sha256") != primary.get("source_image_sha256"):
                raise FormalProtocolError(f"{dataset}:{index}: primary image hash drift")
            row_binding = {
                "adapter_index": index,
                "primary_uid": primary["uid"],
                "primary_image_id": primary["image_id"],
                "adapter_image_id": adapter_image_id,
                "adapter_image_sha256": adapter_image["sha256"],
                "primary_image_sha256": primary["source_image_sha256"],
                "mapping_method": adapter_image["mapping_method"],
                "primary_abc_label": primary.get("abc_label"),
                "upstream_is_abc": primary.get("upstream_is_abc"),
                "image_reference_index": image_reference_index,
            }
            row_mapping_digest.update(
                canonical_json(row_binding).encode("utf-8") + b"\n"
            )
            prompt = (
                _textvqa_prompt(str(row["question"]))
                if dataset == "textvqa_val"
                else _mc_prompt(row, choices)
            )
            runtime_rows.append(
                {
                    "dataset": dataset,
                    "uid": str(primary["uid"]),
                    "adapter_index": index,
                    "adapter_image_id": adapter_image_id,
                    "image_path": str(Path(adapter_image["path"]).resolve()),
                    "image_sha256": str(adapter_image["sha256"]),
                    "question_sha256": str(primary["question_sha256"]),
                    "prompt": prompt,
                    "choices": choices,
                    "answer": answers,
                    "metric": entry["metric"],
                    "max_tokens": int(entry["max_tokens"]),
                }
            )

    if len(runtime_rows) != int(entry["expected_n"]):
        raise FormalProtocolError(
            f"{dataset}: {len(runtime_rows)} rows != expected {entry['expected_n']}"
        )
    if seen != set(primary_by_index):
        raise FormalProtocolError(f"{dataset}: adapter/primary row coverage differs")
    if reference_rows != int(crosswalk.get("adapter_image_reference_rows", -1)):
        raise FormalProtocolError(f"{dataset}: image-reference count drift")
    observed_mapping = row_mapping_digest.hexdigest()
    if observed_mapping != crosswalk.get("row_mapping_sha256"):
        raise FormalProtocolError(f"{dataset}: row-mapping digest drift")
    unique_adapter_ids = len({row["adapter_image_id"] for row in runtime_rows})
    if unique_adapter_ids != int(crosswalk.get("unique_adapter_images", -1)):
        raise FormalProtocolError(f"{dataset}: unique adapter-image ID count drift")
    return runtime_rows, {
        "id": dataset,
        "rows": len(runtime_rows),
        "unique_adapter_image_ids": unique_adapter_ids,
        "unique_adapter_image_byte_hashes": len(
            {row["image_sha256"] for row in runtime_rows}
        ),
        "image_reference_rows": reference_rows,
        "row_mapping_sha256": observed_mapping,
        "tsv": file_binding(tsv),
    }


def _python_tree_binding(root: Path, paths: Iterable[Path] | None = None) -> dict[str, Any]:
    root = root.expanduser().resolve()
    selected = sorted(paths if paths is not None else root.rglob("*.py"))
    selected = [path.resolve() for path in selected if path.is_file()]
    digest = hashlib.sha256()
    total_bytes = 0
    for path in selected:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise FormalProtocolError(f"runtime code path escapes package root: {path}") from exc
        raw = path.read_bytes()
        total_bytes += len(raw)
        digest.update(str(relative).encode("utf-8") + b"\0" + raw + b"\0")
    if not selected:
        raise FormalProtocolError(f"runtime code tree is empty: {root}")
    return {
        "root": str(root),
        "python_files": len(selected),
        "python_bytes": total_bytes,
        "python_tree_sha256": digest.hexdigest(),
    }


def _package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise FormalProtocolError(f"runtime package cannot be resolved: {name}")
    return Path(spec.origin).resolve().parent


@functools.lru_cache(maxsize=1)
def runtime_binding() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in ("ms-swift", "torch", "transformers", "safetensors", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    swift_root = _package_root("swift")
    swift_repo = swift_root.parent
    try:
        swift_head = subprocess.run(
            ["git", "-C", str(swift_repo), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        swift_status = subprocess.run(
            [
                "git", "-C", str(swift_repo), "status", "--short",
                "--untracked-files=all", "--", str(swift_root.relative_to(swift_repo)),
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise FormalProtocolError("cannot bind editable ms-swift repository state") from exc
    transformers_root = _package_root("transformers")
    transformer_paths: list[Path] = []
    for relative in (
        "__init__.py",
        "modeling_utils.py",
        "processing_utils.py",
        "image_processing_utils.py",
        "models/auto/auto_factory.py",
        "models/auto/configuration_auto.py",
    ):
        transformer_paths.append(transformers_root / relative)
    transformer_paths.extend((transformers_root / "generation").glob("*.py"))
    transformer_paths.extend((transformers_root / "models/qwen3_5").glob("*.py"))
    payload = {
        "python": sys.version.split()[0],
        "packages": versions,
        "swift_code": {
            **_python_tree_binding(swift_root),
            "git_head": swift_head,
            "git_status_scoped": swift_status,
            "git_status_scoped_sha256": hashlib.sha256(
                swift_status.encode("utf-8")
            ).hexdigest(),
        },
        "transformers_code": _python_tree_binding(
            transformers_root, transformer_paths
        ),
    }
    return {**payload, "identity_sha256": content_hash(payload)}


def query_l20_hardware(
    selectors: list[str], nvidia_smi_bin: str, *, require_idle: bool
) -> dict[str, Any]:
    if len(selectors) != NUM_SHARDS or len(set(selectors)) != NUM_SHARDS:
        raise FormalProtocolError("hardware admission requires four unique GPU selectors")
    resolved_bin = shutil.which(nvidia_smi_bin)
    if resolved_bin is None:
        candidate = Path(nvidia_smi_bin).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved_bin = str(candidate.resolve())
    if resolved_bin is None:
        raise FormalProtocolError(f"nvidia-smi executable is unavailable: {nvidia_smi_bin}")
    smi = str(Path(resolved_bin).resolve())
    try:
        gpu_query = subprocess.run(
            [
                smi,
                "--query-gpu=index,name,memory.total,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        process_query = subprocess.run(
            [
                smi,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise FormalProtocolError(f"nvidia-smi hardware query failed: {exc.stderr.strip()}") from exc
    physical: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(gpu_query.stdout)):
        if not row:
            continue
        if len(row) != 4:
            raise FormalProtocolError(f"unexpected nvidia-smi GPU row: {row}")
        index, name, memory, uuid = (item.strip() for item in row)
        try:
            memory_mib = int(memory)
        except ValueError as exc:
            raise FormalProtocolError(f"invalid nvidia-smi memory value: {memory}") from exc
        physical.append(
            {"index": index, "name": name, "memory_mib": memory_mib, "uuid": uuid}
        )
    selected: list[dict[str, Any]] = []
    for selector in selectors:
        matches = [gpu for gpu in physical if selector in (gpu["index"], gpu["uuid"])]
        if len(matches) != 1:
            raise FormalProtocolError(
                f"GPU selector does not resolve to one physical GPU: {selector}"
            )
        selected.append({"selector": selector, **matches[0]})
    if len({gpu["uuid"] for gpu in selected}) != NUM_SHARDS:
        raise FormalProtocolError("GPU selectors do not resolve to four unique physical devices")
    for gpu in selected:
        if "L20" not in gpu["name"].upper() or int(gpu["memory_mib"]) < 44 * 1024:
            raise FormalProtocolError(
                f"GPU {gpu['index']} is not a >=44 GiB L20: "
                f"{gpu['name']} ({gpu['memory_mib']} MiB)"
            )
        if "MIG" in gpu["name"].upper() or gpu["uuid"].upper().startswith("MIG"):
            raise FormalProtocolError(f"GPU selector resolves to a MIG slice: {gpu['uuid']}")
    selected_uuids = {gpu["uuid"] for gpu in selected}
    processes: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(process_query.stdout)):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 4:
            raise FormalProtocolError(
                f"unexpected nvidia-smi compute-process row: {row}"
            )
        uuid, pid = row[0].strip(), row[1].strip()
        if uuid in selected_uuids:
            processes.append(
                {
                    "gpu_uuid": uuid,
                    "pid": pid,
                    "process_name": row[2].strip() if len(row) > 2 else "",
                    "used_gpu_memory_mib": row[3].strip() if len(row) > 3 else "",
                }
            )
    if require_idle and processes:
        raise FormalProtocolError(f"selected GPUs have compute processes: {processes}")
    payload: dict[str, Any] = {
        "contract": "exactly_4_unique_physical_L20_at_least_44_GiB",
        "nvidia_smi": file_binding(Path(smi)),
        "selectors": selectors,
        "devices": selected,
        "compute_processes_at_query": processes,
        "idle_at_query": not processes,
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def verify_hardware_inventory(
    inventory: Any, *, live: bool, require_idle: bool
) -> None:
    if not isinstance(inventory, dict):
        raise FormalProtocolError("hardware inventory must be an object")
    identity = inventory.get("identity_sha256")
    unsigned = {key: value for key, value in inventory.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise FormalProtocolError("hardware inventory identity drift")
    if (
        inventory.get("contract")
        != "exactly_4_unique_physical_L20_at_least_44_GiB"
        or inventory.get("idle_at_query") is not True
        or inventory.get("compute_processes_at_query") != []
    ):
        raise FormalProtocolError("hardware inventory was not admitted idle exact-4xL20")
    verify_file_binding(inventory.get("nvidia_smi"), deep=True)
    devices = inventory.get("devices")
    selectors = inventory.get("selectors")
    if (
        not isinstance(devices, list)
        or len(devices) != NUM_SHARDS
        or not isinstance(selectors, list)
        or len(selectors) != NUM_SHARDS
        or len({row.get("uuid") for row in devices if isinstance(row, dict)}) != NUM_SHARDS
    ):
        raise FormalProtocolError("hardware inventory device coverage drift")
    if live:
        current = query_l20_hardware(
            [str(value) for value in selectors],
            str(inventory["nvidia_smi"]["path"]),
            require_idle=require_idle,
        )
        if current != inventory:
            raise FormalProtocolError("live hardware inventory differs from bound admission")


def evaluation_code_binding(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    evidence = config["source_evidence"]
    paths = (
        Path(__file__).resolve(),
        ROOT / "eval/gsa_eval/runner.py",
        ROOT / "scripts/run_general_retention_dp.sh",
        config_path.expanduser().resolve(),
        Path(evidence["vqa_reference_path"]).expanduser().resolve(),
    )
    files = {path.name: file_binding(path) for path in paths}
    expected_reference_sha = str(evidence["vqa_reference_sha256"])
    if files[Path(evidence["vqa_reference_path"]).name]["sha256"] != expected_reference_sha:
        raise FormalProtocolError("pinned VQA scorer reference drift")
    return {"files": files, "identity_sha256": content_hash(files)}


def _source_evidence(
    config: dict[str, Any], *, verify_image_files: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence = config["source_evidence"]
    source_path = Path(evidence["source_manifest_path"])
    sample_path = Path(evidence["sample_manifest_path"])
    adapter_path = Path(evidence["adapter_image_manifest_path"])
    crosswalk_path = Path(evidence["crosswalk_path"])
    leakage_path = Path(evidence["leakage_report_path"])
    source = load_json(source_path, "source manifest")
    crosswalk = load_json(crosswalk_path, "adapter crosswalk")
    leakage = load_json(leakage_path, "leakage report")
    _verify_identity(source, str(evidence["source_manifest_identity_sha256"]), "source manifest")
    _verify_identity(crosswalk, str(evidence["crosswalk_identity_sha256"]), "crosswalk")
    _verify_identity(leakage, str(evidence["leakage_identity_sha256"]), "leakage report")
    if source.get("suite_id") != SUITE_ID or crosswalk.get("suite_id") != SUITE_ID:
        raise FormalProtocolError("source/crosswalk suite drift")
    if source.get("crosswalk_identity_sha256") != crosswalk.get("identity_sha256"):
        raise FormalProtocolError("source manifest does not bind the selected crosswalk")
    fingerprinted = leakage.get("training_fingerprints_ok")
    unique_training_images = leakage.get("training_unique_image_paths")
    if (
        leakage.get("state") != "PASS"
        or not isinstance(fingerprinted, int)
        or fingerprinted <= 0
        or fingerprinted != unique_training_images
        or leakage.get("training_fingerprint_errors") != 0
    ):
        raise FormalProtocolError("leakage audit is not PASS with complete fingerprints")
    zero_fields = (
        "exact_image_collisions",
        "transformation_normalized_exact_image_collisions",
        "dataset_qualified_source_id_collisions",
        "near_duplicate_image_collisions_unresolved",
        "image_question_collisions",
    )
    for field in zero_fields:
        if leakage.get(field) != 0:
            raise FormalProtocolError(f"leakage gate failed: {field}={leakage.get(field)!r}")
    if leakage.get("preflight_errors") not in ([], None):
        raise FormalProtocolError("leakage report contains preflight errors")
    leakage_source = leakage.get("source_manifest") or {}
    if (
        Path(str(leakage_source.get("path"))).expanduser().resolve()
        != source_path.expanduser().resolve()
        or leakage_source.get("file_sha256") != file_hash(source_path)
        or leakage_source.get("identity_sha256") != source.get("identity_sha256")
    ):
        raise FormalProtocolError("leakage report/source manifest binding drift")

    if file_hash(sample_path) != source.get("sample_manifest_sha256"):
        raise FormalProtocolError("sample manifest hash drift")
    if file_hash(adapter_path) != source.get("adapter_image_manifest_sha256"):
        raise FormalProtocolError("adapter image manifest hash drift")
    samples = load_jsonl(sample_path, "frozen evaluation samples")
    adapter_images = load_jsonl(adapter_path, "frozen adapter images")
    if len(samples) != source.get("total_samples"):
        raise FormalProtocolError("frozen sample count drift")
    if len(adapter_images) != source.get("total_unique_adapter_images"):
        raise FormalProtocolError("frozen adapter-image count drift")

    samples_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in DATASET_ORDER:
        selected = [row for row in samples if row.get("dataset") == dataset]
        by_index = {str(row.get("adapter_index")): row for row in selected}
        if len(by_index) != len(selected):
            raise FormalProtocolError(f"{dataset}: duplicate primary adapter index")
        samples_by_dataset[dataset] = by_index
    adapter_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in DATASET_ORDER:
        selected = [row for row in adapter_images if row.get("dataset") == dataset]
        by_id = {str(row.get("adapter_image_id")): row for row in selected}
        if len(by_id) != len(selected):
            raise FormalProtocolError(f"{dataset}: duplicate adapter image ID")
        adapter_by_dataset[dataset] = by_id
    crosswalks = {str(row.get("dataset")): row for row in crosswalk.get("datasets", [])}
    if set(crosswalks) != set(DATASET_ORDER):
        raise FormalProtocolError("crosswalk dataset coverage drift")

    runtime_rows: list[dict[str, Any]] = []
    dataset_receipts: list[dict[str, Any]] = []
    global_uids: set[str] = set()
    for entry in _dataset_configs(config):
        dataset = str(entry["id"])
        rows, receipt = _read_frozen_dataset(
            entry,
            samples_by_dataset[dataset],
            adapter_by_dataset[dataset],
            crosswalks[dataset],
            verify_image_files=verify_image_files,
        )
        for row in rows:
            global_uid = f"{dataset}:{row['uid']}"
            if global_uid in global_uids:
                raise FormalProtocolError(f"duplicate global UID: {global_uid}")
            global_uids.add(global_uid)
            row["global_uid"] = global_uid
            row["request_index"] = len(runtime_rows)
            runtime_rows.append(row)
        dataset_receipts.append(receipt)
    if len(runtime_rows) != source.get("total_samples"):
        raise FormalProtocolError("runtime/source total sample count drift")

    bindings = {
        "source_manifest": file_binding(source_path),
        "source_manifest_identity_sha256": source["identity_sha256"],
        "sample_manifest": file_binding(sample_path),
        "adapter_image_manifest": file_binding(adapter_path),
        "crosswalk": file_binding(crosswalk_path),
        "crosswalk_identity_sha256": crosswalk["identity_sha256"],
        "leakage_report": file_binding(leakage_path),
        "leakage_identity_sha256": leakage["identity_sha256"],
        "leakage_state": leakage["state"],
        "leakage_zero_fields": {field: leakage[field] for field in zero_fields},
        "datasets": dataset_receipts,
    }
    return runtime_rows, bindings


def build_input_receipt(
    config_path: Path,
    runtime_index_path: Path,
    *,
    hardware_inventory: dict[str, Any],
    verify_image_files: bool = True,
    allow_test_fixture: bool = False,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    runtime_index_path = runtime_index_path.expanduser().resolve()
    config = load_yaml(config_path)
    validate_protocol_config(config, allow_test_fixture=allow_test_fixture)
    verify_hardware_inventory(hardware_inventory, live=False, require_idle=True)
    rows, sources = _source_evidence(config, verify_image_files=verify_image_files)
    atomic_jsonl(runtime_index_path, rows)
    runtime_index = file_binding(runtime_index_path)
    dataset_counts = {
        dataset: sum(row["dataset"] == dataset for row in rows) for dataset in DATASET_ORDER
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "suite_id": SUITE_ID,
        "test_fixture": bool(config.get("test_fixture")),
        "config": file_binding(config_path),
        "sources": sources,
        "runtime_index": {
            **runtime_index,
            "records": len(rows),
            "dataset_counts": dataset_counts,
            "record_identity_sha256": content_hash(rows),
        },
        "generation": config["generation"],
        "hardware_admission": hardware_inventory,
        "health_admission": config["health_admission"],
        "aggregate": config["aggregate"],
        "evaluation_code": evaluation_code_binding(config_path, config),
        "runtime": runtime_binding(),
        "training_materialization_eligible": False,
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def verify_input_receipt(
    path: Path,
    *,
    deep_sources: bool = False,
    allow_test_fixture: bool = False,
    live_hardware: bool = False,
    require_idle_hardware: bool = False,
) -> dict[str, Any]:
    if require_idle_hardware and not live_hardware:
        raise FormalProtocolError("idle hardware verification requires a live query")
    path = path.expanduser().resolve()
    receipt = load_json(path, "input receipt")
    identity = receipt.get("identity_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise FormalProtocolError("input receipt identity drift")
    if receipt.get("protocol") != PROTOCOL or receipt.get("suite_id") != SUITE_ID:
        raise FormalProtocolError("input receipt protocol drift")
    if receipt.get("test_fixture") is True and not allow_test_fixture:
        raise FormalProtocolError("test fixture receipt is forbidden by the formal API")
    config_path = verify_file_binding(receipt.get("config"), deep=True)
    config = load_yaml(config_path)
    validate_protocol_config(config, allow_test_fixture=allow_test_fixture)
    if receipt.get("evaluation_code") != evaluation_code_binding(config_path, config):
        raise FormalProtocolError("evaluation code binding drift")
    if receipt.get("runtime") != runtime_binding():
        raise FormalProtocolError("evaluation runtime binding drift")
    verify_hardware_inventory(
        receipt.get("hardware_admission"),
        live=live_hardware,
        require_idle=require_idle_hardware,
    )
    sources = receipt.get("sources") or {}
    for key in (
        "source_manifest", "sample_manifest", "adapter_image_manifest", "crosswalk",
        "leakage_report",
    ):
        verify_file_binding(sources.get(key), deep=deep_sources)
    for dataset in sources.get("datasets", []):
        verify_file_binding(dataset.get("tsv"), deep=deep_sources)
    runtime = receipt.get("runtime_index") or {}
    runtime_path = verify_file_binding(
        {key: runtime[key] for key in ("path", "bytes", "sha256")}, deep=True
    )
    rows = load_jsonl(runtime_path, "runtime index")
    if len(rows) != runtime.get("records") or content_hash(rows) != runtime.get(
        "record_identity_sha256"
    ):
        raise FormalProtocolError("runtime index record binding drift")
    if [row.get("request_index") for row in rows] != list(range(len(rows))):
        raise FormalProtocolError("runtime index request order drift")
    observed_counts = {
        dataset: sum(row.get("dataset") == dataset for row in rows) for dataset in DATASET_ORDER
    }
    if observed_counts != runtime.get("dataset_counts"):
        raise FormalProtocolError("runtime index dataset counts drift")
    return receipt


def _weight_paths(model: Path, config: dict[str, Any]) -> list[Path]:
    if config.get("peft_type") or config.get("base_model_name_or_path"):
        raise FormalProtocolError("unmerged adapter config is not accepted")
    adapter_artifacts = sorted(model.rglob("adapter_config.json"))
    adapter_artifacts += sorted(model.rglob("adapter_model.safetensors"))
    if adapter_artifacts:
        raise FormalProtocolError(f"unmerged adapter artifact is not accepted: {adapter_artifacts[0]}")
    index = model / "model.safetensors.index.json"
    single = model / "model.safetensors"
    if index.is_file():
        value = load_json(index, "model weight index")
        weight_map = value.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise FormalProtocolError("empty model weight map")
        paths = [model / name for name in sorted({str(name) for name in weight_map.values()})]
    elif single.is_file():
        paths = [single]
    else:
        raise FormalProtocolError(f"no merged safetensors weights found: {model}")
    resolved = model.resolve()
    if any(resolved not in path.resolve().parents for path in paths):
        raise FormalProtocolError("model weight index escapes model directory")
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise FormalProtocolError("model weight index references missing or empty files")
    return paths


def model_inventory(model: Path) -> dict[str, Any]:
    model = model.expanduser().resolve()
    config = load_json(model / "config.json", "model config")
    architectures = config.get("architectures")
    model_type = config.get("model_type")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or not isinstance(architectures[0], str)
        or not isinstance(model_type, str)
        or not isinstance(config.get("vision_config"), dict)
    ):
        raise FormalProtocolError("model is not a single merged vision-language architecture")
    weights = _weight_paths(model, config)
    metadata = {
        name: file_binding(model / name)
        for name in _MODEL_METADATA_NAMES
        if (model / name).is_file()
    }
    weight_bindings = [file_binding(path, include_mtime=True) for path in weights]
    content_payload = {
        "architecture": architectures[0],
        "model_type": model_type,
        "metadata_files": {
            name: {"bytes": binding["bytes"], "sha256": binding["sha256"]}
            for name, binding in sorted(metadata.items())
        },
        "weight_files": [
            {
                "relative_path": str(Path(binding["path"]).relative_to(model)),
                "bytes": binding["bytes"],
                "sha256": binding["sha256"],
            }
            for binding in weight_bindings
        ],
    }
    payload: dict[str, Any] = {
        "path": str(model),
        "architecture": architectures[0],
        "model_type": model_type,
        "source_kind": "standalone_merged_weights",
        "metadata_files": metadata,
        "weight_files": weight_bindings,
        "weight_bytes_fully_sha256_hashed": True,
        "checkpoint_content_identity_sha256": content_hash(content_payload),
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def verify_model_inventory(inventory: Any, *, deep: bool) -> None:
    if not isinstance(inventory, dict):
        raise FormalProtocolError("model inventory must be an object")
    identity = inventory.get("identity_sha256")
    unsigned = {key: value for key, value in inventory.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise FormalProtocolError("model inventory identity drift")
    model = Path(str(inventory.get("path"))).expanduser().resolve()
    if str(model) != inventory.get("path"):
        raise FormalProtocolError("model path is not canonical")
    config = load_json(model / "config.json", "model config")
    if (
        config.get("architectures") != [inventory.get("architecture")]
        or config.get("model_type") != inventory.get("model_type")
        or not isinstance(config.get("vision_config"), dict)
    ):
        raise FormalProtocolError("model architecture identity drift")
    weights = _weight_paths(model, config)
    expected_metadata = {
        name for name in _MODEL_METADATA_NAMES if (model / name).is_file()
    }
    metadata = inventory.get("metadata_files")
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata:
        raise FormalProtocolError("model metadata inventory coverage drift")
    for name, binding in metadata.items():
        verify_file_binding(binding, deep=deep)
    weight_bindings = inventory.get("weight_files")
    if not isinstance(weight_bindings, list) or len(weight_bindings) != len(weights):
        raise FormalProtocolError("model weight inventory coverage drift")
    for expected, binding in zip(weights, weight_bindings):
        path = verify_file_binding(binding, deep=deep)
        if path != expected.resolve():
            raise FormalProtocolError("model weight path/order drift")
    content_payload = {
        "architecture": inventory.get("architecture"),
        "model_type": inventory.get("model_type"),
        "metadata_files": {
            name: {"bytes": binding["bytes"], "sha256": binding["sha256"]}
            for name, binding in sorted(metadata.items())
        },
        "weight_files": [
            {
                "relative_path": str(
                    Path(binding["path"]).resolve().relative_to(model)
                ),
                "bytes": binding["bytes"],
                "sha256": binding["sha256"],
            }
            for binding in weight_bindings
        ],
    }
    if inventory.get("checkpoint_content_identity_sha256") != content_hash(
        content_payload
    ):
        raise FormalProtocolError("model checkpoint content identity drift")


def bind_model(
    role: str,
    model: Path,
    input_receipt: Path,
    *,
    allow_test_fixture: bool = False,
) -> dict[str, Any]:
    if role not in MODEL_ROLES:
        raise FormalProtocolError(f"unsupported model role: {role}")
    input_receipt = input_receipt.expanduser().resolve()
    verify_input_receipt(input_receipt, allow_test_fixture=allow_test_fixture)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "role": role,
        "input_receipt": file_binding(input_receipt),
        "model": model_inventory(model),
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def verify_model_binding(
    path: Path, *, deep: bool, allow_test_fixture: bool = False
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    binding = load_json(path, "model binding")
    identity = binding.get("identity_sha256")
    unsigned = {key: value for key, value in binding.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise FormalProtocolError("model binding identity drift")
    if binding.get("protocol") != PROTOCOL or binding.get("role") not in MODEL_ROLES:
        raise FormalProtocolError("model binding protocol/role drift")
    receipt_path = verify_file_binding(binding.get("input_receipt"), deep=True)
    verify_input_receipt(receipt_path, allow_test_fixture=allow_test_fixture)
    verify_model_inventory(binding.get("model"), deep=deep)
    return binding


def _runtime_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    value = receipt["runtime_index"]
    path = verify_file_binding(
        {key: value[key] for key in ("path", "bytes", "sha256")}, deep=True
    )
    return load_jsonl(path, "runtime index")


def _verify_runtime_images(rows: list[dict[str, Any]]) -> None:
    """Re-hash each selected transport once immediately before model loading."""
    expected_by_path: dict[Path, str] = {}
    for row in rows:
        path = Path(str(row.get("image_path"))).expanduser().resolve()
        expected = str(row.get("image_sha256"))
        if not _SHA256.fullmatch(expected):
            raise FormalProtocolError(f"{row.get('global_uid')}: invalid runtime image hash")
        previous = expected_by_path.setdefault(path, expected)
        if previous != expected:
            raise FormalProtocolError(f"runtime image path maps to multiple hashes: {path}")
    for path, expected in expected_by_path.items():
        if file_hash(path) != expected:
            raise FormalProtocolError(f"runtime image drift before inference: {path}")


def _real_infer(
    public_inputs: list[dict[str, Any]], max_tokens: int, *, model: str, batch_size: int
) -> list[tuple[str, bool]]:
    from swift import InferRequest, RequestConfig
    from eval.gsa_eval.runner import get_engine

    engine = get_engine(model, batch_size, device_map="cuda:0", thinking=False)
    requests = [
        InferRequest(messages=item["messages"], images=item["images"])
        for item in public_inputs
    ]
    responses = engine.infer(
        requests, RequestConfig(max_tokens=max_tokens, temperature=0.0)
    )
    if len(responses) != len(requests):
        raise FormalProtocolError("inference engine returned an unexpected response count")
    output: list[tuple[str, bool]] = []
    for response in responses:
        try:
            choice = response.choices[0]
            text = choice.message.content
            if text is None:
                text = ""
            if not isinstance(text, str):
                raise TypeError("response content is not a string")
            output.append((text, choice.finish_reason == "length"))
        except Exception as exc:
            raise FormalProtocolError("malformed inference-engine response object") from exc
    return output


def infer_shard(
    *,
    receipt_path: Path,
    binding_path: Path,
    shard: int,
    num_shards: int,
    batch_size: int,
    out: Path,
    infer_public: PublicInfer | None = None,
    allow_test_fixture: bool = False,
) -> dict[str, Any]:
    if num_shards != NUM_SHARDS or not 0 <= shard < NUM_SHARDS:
        raise FormalProtocolError("formal inference requires shard 0..3/4")
    if batch_size != BATCH_SIZE:
        raise FormalProtocolError(f"formal inference requires batch_size={BATCH_SIZE}")
    receipt_path = receipt_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    receipt = verify_input_receipt(
        receipt_path, allow_test_fixture=allow_test_fixture
    )
    binding = verify_model_binding(
        binding_path, deep=False, allow_test_fixture=allow_test_fixture
    )
    if binding["input_receipt"] != file_binding(receipt_path):
        raise FormalProtocolError("model binding/input receipt mismatch")
    rows = _runtime_rows(receipt)
    selected = rows[shard::NUM_SHARDS]
    if not selected:
        raise FormalProtocolError(f"empty formal shard: {shard}")
    _verify_runtime_images(selected)
    print(
        f"[pre-infer] shard={shard}/{NUM_SHARDS} records={len(selected)} "
        f"unique_images={len({row['image_path'] for row in selected})} images_verified=true",
        flush=True,
    )
    infer = infer_public
    if infer is None:
        model = str(binding["model"]["path"])

        def infer(public: list[dict[str, Any]], max_tokens: int) -> list[tuple[str, bool]]:
            return _real_infer(public, max_tokens, model=model, batch_size=batch_size)

    generated: dict[int, tuple[str, bool]] = {}
    for dataset in DATASET_ORDER:
        batch_rows = [row for row in selected if row["dataset"] == dataset]
        if not batch_rows:
            continue
        budgets = {int(row["max_tokens"]) for row in batch_rows}
        if len(budgets) != 1:
            raise FormalProtocolError(f"{dataset}: runtime max-token drift")
        public = [
            {
                "messages": [{"role": "user", "content": row["prompt"]}],
                "images": [row["image_path"]],
            }
            for row in batch_rows
        ]
        max_tokens = budgets.pop()
        print(
            f"[infer] shard={shard}/{NUM_SHARDS} dataset={dataset} "
            f"records={len(batch_rows)} max_tokens={max_tokens} status=starting",
            flush=True,
        )
        responses = infer(public, max_tokens)
        if len(responses) != len(batch_rows):
            raise FormalProtocolError("inference callback returned wrong response count")
        for row, response in zip(batch_rows, responses):
            if (
                not isinstance(response, tuple)
                or len(response) != 2
                or not isinstance(response[0], str)
                or not isinstance(response[1], bool)
            ):
                raise FormalProtocolError("inference callback response schema drift")
            generated[int(row["request_index"])] = response
        print(
            f"[infer] shard={shard}/{NUM_SHARDS} dataset={dataset} "
            f"records={len(batch_rows)} status=complete",
            flush=True,
        )
    if set(generated) != {int(row["request_index"]) for row in selected}:
        raise FormalProtocolError("inference callback coverage drift")

    receipt_sha = file_hash(receipt_path)
    records = []
    for row in selected:
        request_index = int(row["request_index"])
        response, hit_token_cap = generated[request_index]
        records.append(
            {
                "protocol": PROTOCOL,
                "suite_id": SUITE_ID,
                "shard": shard,
                "num_shards": NUM_SHARDS,
                "request_index": request_index,
                "dataset": row["dataset"],
                "global_uid": row["global_uid"],
                "response": response,
                "hit_token_cap": hit_token_cap,
                "max_tokens": int(row["max_tokens"]),
                "temperature": 0.0,
                "thinking": False,
                "batch_size_per_gpu": BATCH_SIZE,
                "model_role": binding["role"],
                "model_binding_identity_sha256": binding["identity_sha256"],
                "model_inventory_identity_sha256": binding["model"]["identity_sha256"],
                "model_checkpoint_content_identity_sha256": binding["model"][
                    "checkpoint_content_identity_sha256"
                ],
                "input_receipt_sha256": receipt_sha,
                "runtime_index_sha256": receipt["runtime_index"]["sha256"],
                "hardware_admission_identity_sha256": receipt["hardware_admission"][
                    "identity_sha256"
                ],
            }
        )
    atomic_jsonl(out, records, idempotent=False)
    return {
        "status": "generated",
        "role": binding["role"],
        "shard": shard,
        "records": len(records),
        "out": str(out.resolve()),
    }


def _expected_prediction(
    row: dict[str, Any], receipt: dict[str, Any], binding: dict[str, Any], shard: int
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "suite_id": SUITE_ID,
        "shard": shard,
        "num_shards": NUM_SHARDS,
        "request_index": int(row["request_index"]),
        "dataset": row["dataset"],
        "global_uid": row["global_uid"],
        "max_tokens": int(row["max_tokens"]),
        "temperature": 0.0,
        "thinking": False,
        "batch_size_per_gpu": BATCH_SIZE,
        "model_role": binding["role"],
        "model_binding_identity_sha256": binding["identity_sha256"],
        "model_inventory_identity_sha256": binding["model"]["identity_sha256"],
        "model_checkpoint_content_identity_sha256": binding["model"][
            "checkpoint_content_identity_sha256"
        ],
        "input_receipt_sha256": file_hash(Path(binding["input_receipt"]["path"])),
        "runtime_index_sha256": receipt["runtime_index"]["sha256"],
        "hardware_admission_identity_sha256": receipt["hardware_admission"][
            "identity_sha256"
        ],
    }


def verify_shard(
    *,
    receipt_path: Path,
    binding_path: Path,
    shard_path: Path,
    expected_shard: int,
    allow_test_fixture: bool = False,
) -> list[dict[str, Any]]:
    if not 0 <= expected_shard < NUM_SHARDS:
        raise FormalProtocolError("invalid expected shard")
    receipt = verify_input_receipt(
        receipt_path, allow_test_fixture=allow_test_fixture
    )
    binding = verify_model_binding(
        binding_path, deep=False, allow_test_fixture=allow_test_fixture
    )
    if binding["input_receipt"] != file_binding(receipt_path):
        raise FormalProtocolError("model binding/input receipt mismatch")
    runtime_rows = _runtime_rows(receipt)
    expected_rows = runtime_rows[expected_shard::NUM_SHARDS]
    records = load_jsonl(shard_path, "prediction shard")
    if len(records) != len(expected_rows):
        raise FormalProtocolError("prediction shard row count drift")
    allowed_keys = set(_expected_prediction(expected_rows[0], receipt, binding, expected_shard)) | {
        "response", "hit_token_cap"
    }
    for row, prediction in zip(expected_rows, records):
        if set(prediction) != allowed_keys:
            raise FormalProtocolError("prediction shard schema drift")
        expected = _expected_prediction(row, receipt, binding, expected_shard)
        for key, value in expected.items():
            if prediction.get(key) != value:
                raise FormalProtocolError(
                    f"prediction shard provenance/order drift at {row['global_uid']}: {key}"
                )
        if not isinstance(prediction.get("response"), str) or not isinstance(
            prediction.get("hit_token_cap"), bool
        ):
            raise FormalProtocolError("prediction response metadata drift")
    return records


def _answer_span(text: str) -> tuple[str | None, bool]:
    if not text:
        return None, False
    stripped = _THINK_BLOCK.sub(" ", text)
    if _THINK_OPEN.search(stripped):
        return None, True
    stripped = stripped.strip()
    if _THINK_CLOSE.search(stripped):
        matches = list(_THINK_CLOSE.finditer(stripped))
        stripped = stripped[matches[-1].end():].strip()
    return (stripped or None), False


def parse_multiple_choice(text: str, choices: list[str]) -> tuple[str | None, bool]:
    span, unclosed = _answer_span(text)
    if span is None:
        return None, unclosed
    letters = string.ascii_uppercase[: len(choices)]
    patterns = (
        rf"\s*\(?([{letters}])\)?[\s.]*",
        rf"\s*(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?([{letters}])\)?[\s.]*",
        rf"\s*the\s+(?:correct\s+)?answer\s+is\s+\(?([{letters}])\)?[\s.]*",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, span, re.IGNORECASE)
        if match:
            return match.group(1).upper(), unclosed
    normalized = " ".join(span.casefold().split()).rstrip(".")
    option_matches = [
        string.ascii_uppercase[index]
        for index, choice in enumerate(choices)
        if normalized == " ".join(str(choice).casefold().split()).rstrip(".")
    ]
    return (option_matches[0] if len(option_matches) == 1 else None), unclosed


def _process_punctuation(value: str) -> str:
    output = value
    for mark in _PUNCTUATION:
        if mark + " " in value or " " + mark in value or _COMMA_STRIP.search(value):
            output = output.replace(mark, "")
        else:
            output = output.replace(mark, " ")
    return _PERIOD_STRIP.sub("", output)


def _reference_digit_article(reference_path: Path) -> Callable[[str], str]:
    """Compile only the pinned helper definition, without importing VLMEvalKit."""
    source = reference_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(reference_path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_process_digit_article"
    ]
    if len(selected) != 1 or not isinstance(selected[0], ast.FunctionDef):
        raise FormalProtocolError("pinned VQA scorer helper definition drift")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, str(reference_path), "exec"), {"__builtins__": __builtins__}, namespace)
    helper = namespace.get("_process_digit_article")
    if not callable(helper):
        raise FormalProtocolError("failed to load pinned VQA normalization helper")
    return helper


def _vqa_normalize(answer: str, digit_article: Callable[[str], str]) -> str:
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    return str(digit_article(_process_punctuation(answer)))


def _vqa_soft_credit(
    prediction: str, answers: list[str], digit_article: Callable[[str], str]
) -> float:
    pred = _vqa_normalize(prediction, digit_article)
    ground_truths = [_vqa_normalize(answer, digit_article) for answer in answers]
    if not ground_truths:
        raise FormalProtocolError("TextVQA sample has no annotator answers")
    credits = []
    for current in range(len(ground_truths)):
        matching_other = sum(
            answer == pred
            for index, answer in enumerate(ground_truths)
            if index != current
        )
        credits.append(min(1.0, matching_other / 3.0))
    return sum(credits) / len(credits)


def score_predictions(
    *,
    receipt_path: Path,
    binding_path: Path,
    shard_paths: list[Path],
    out: Path | None,
    allow_test_fixture: bool = False,
) -> dict[str, Any]:
    if len(shard_paths) != NUM_SHARDS:
        raise FormalProtocolError("scoring requires exactly four prediction shards")
    receipt = verify_input_receipt(
        receipt_path, allow_test_fixture=allow_test_fixture
    )
    binding = verify_model_binding(
        binding_path, deep=False, allow_test_fixture=allow_test_fixture
    )
    if binding["input_receipt"] != file_binding(receipt_path):
        raise FormalProtocolError("model binding/input receipt mismatch")
    records: list[dict[str, Any]] = []
    shard_bindings = []
    for shard, path in enumerate(shard_paths):
        records.extend(
            verify_shard(
                receipt_path=receipt_path,
                binding_path=binding_path,
                shard_path=path,
                expected_shard=shard,
                allow_test_fixture=allow_test_fixture,
            )
        )
        shard_bindings.append(file_binding(path))
    records.sort(key=lambda row: row["request_index"])
    runtime_rows = _runtime_rows(receipt)
    if [row["request_index"] for row in records] != list(range(len(runtime_rows))):
        raise FormalProtocolError("merged predictions do not cover every request exactly once")
    if [row["global_uid"] for row in records] != [row["global_uid"] for row in runtime_rows]:
        raise FormalProtocolError("merged prediction UID order drift")

    config = load_yaml(Path(receipt["config"]["path"]))
    reference_path = Path(config["source_evidence"]["vqa_reference_path"])
    digit_article = _reference_digit_article(reference_path)
    per_sample: list[dict[str, Any]] = []
    for sample, prediction in zip(runtime_rows, records):
        response = prediction["response"]
        hit_cap = bool(prediction["hit_token_cap"])
        if sample["metric"] == "multiple_choice_accuracy":
            parsed, unclosed = parse_multiple_choice(response, sample["choices"])
            credit = 1.0 if parsed == sample["answer"] else 0.0
        elif sample["metric"] == "official_vqa_soft_accuracy":
            parsed, unclosed = _answer_span(response)
            credit = (
                _vqa_soft_credit(parsed, sample["answer"], digit_article)
                if parsed is not None
                else 0.0
            )
        else:
            raise FormalProtocolError(f"unknown metric: {sample['metric']}")
        malformed = parsed is None
        truncated = hit_cap or unclosed
        per_sample.append(
            {
                "dataset": sample["dataset"],
                "global_uid": sample["global_uid"],
                "credit": credit,
                "parsed": not malformed,
                "malformed": malformed,
                "truncated": truncated,
                "hit_token_cap": hit_cap,
                "unclosed_think": unclosed,
            }
        )

    health = receipt["health_admission"]
    benchmarks: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        rows = [row for row in per_sample if row["dataset"] == dataset]
        malformed = sum(row["malformed"] for row in rows)
        truncated = sum(row["truncated"] for row in rows)
        malformed_rate = 100.0 * malformed / len(rows)
        truncation_rate = 100.0 * truncated / len(rows)
        violations = []
        if malformed_rate > float(health["max_malformed_percent"]):
            violations.append("malformed_rate_exceeds_5_percent")
        if truncation_rate > float(health["max_truncation_percent"]):
            violations.append("truncation_rate_exceeds_1_percent")
        benchmarks[dataset] = {
            "metric": rows and next(
                item["metric"] for item in runtime_rows if item["dataset"] == dataset
            ),
            "n": len(rows),
            "raw_score_percent": 100.0 * sum(row["credit"] for row in rows) / len(rows),
            "malformed_count": malformed,
            "malformed_percent": malformed_rate,
            "output_validity_percent": 100.0 - malformed_rate,
            "truncation_count": truncated,
            "truncation_percent": truncation_rate,
            "health_admitted": not violations,
            "health_violations": violations,
            "malformed_response_credit": 0.0,
        }
    raw_macro = sum(benchmarks[key]["raw_score_percent"] for key in DATASET_ORDER) / len(
        DATASET_ORDER
    )
    all_admitted = all(benchmarks[key]["health_admitted"] for key in DATASET_ORDER)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "suite_id": SUITE_ID,
        "status": "FORMAL_SCORE_ADMITTED" if all_admitted else "SCORED_HEALTH_NOT_ADMITTED",
        "model_role": binding["role"],
        "input_receipt": file_binding(receipt_path),
        "model_binding": file_binding(binding_path),
        "model_binding_identity_sha256": binding["identity_sha256"],
        "model_checkpoint_content_identity_sha256": binding["model"][
            "checkpoint_content_identity_sha256"
        ],
        "hardware_admission_identity_sha256": receipt["hardware_admission"][
            "identity_sha256"
        ],
        "prediction_shards": shard_bindings,
        "coverage_complete": len(records) == len(runtime_rows),
        "malformed_scored_as_zero": True,
        "health_admission_separate_from_raw_scoring": True,
        "benchmarks": benchmarks,
        "raw_general_avg_percent": raw_macro,
        "general_avg_percent": raw_macro if all_admitted else None,
        "all_cells_health_admitted": all_admitted,
        "per_sample": per_sample,
        "training_materialization_eligible": False,
    }
    payload["identity_sha256"] = content_hash(payload)
    if out is not None:
        atomic_json(out, payload, idempotent=False)
    return payload


def _verify_result(path: Path, *, allow_test_fixture: bool = False) -> dict[str, Any]:
    result = load_json(path, "formal score result")
    identity = result.get("identity_sha256")
    unsigned = {key: value for key, value in result.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise FormalProtocolError("formal score result identity drift")
    if result.get("protocol") != PROTOCOL or result.get("coverage_complete") is not True:
        raise FormalProtocolError("formal score result protocol/coverage drift")
    receipt_path = verify_file_binding(result.get("input_receipt"), deep=True)
    binding_path = verify_file_binding(result.get("model_binding"), deep=True)
    binding = verify_model_binding(
        binding_path, deep=False, allow_test_fixture=allow_test_fixture
    )
    if binding.get("input_receipt") != file_binding(receipt_path):
        raise FormalProtocolError("result model/input receipt binding drift")
    if (
        result.get("model_role") != binding.get("role")
        or result.get("model_binding_identity_sha256") != binding.get("identity_sha256")
    ):
        raise FormalProtocolError("result model identity drift")
    shards = result.get("prediction_shards")
    if not isinstance(shards, list) or len(shards) != NUM_SHARDS:
        raise FormalProtocolError("result must bind exactly four prediction shards")
    shard_paths: list[Path] = []
    for shard, shard_binding in enumerate(shards):
        shard_path = verify_file_binding(shard_binding, deep=True)
        shard_paths.append(shard_path)
        verify_shard(
            receipt_path=receipt_path,
            binding_path=binding_path,
            shard_path=shard_path,
            expected_shard=shard,
            allow_test_fixture=allow_test_fixture,
        )
    rebuilt = score_predictions(
        receipt_path=receipt_path,
        binding_path=binding_path,
        shard_paths=shard_paths,
        out=None,
        allow_test_fixture=allow_test_fixture,
    )
    if rebuilt != result:
        raise FormalProtocolError("result scores/health do not recompute from bound shards")
    return result


def compare_results(
    base_path: Path,
    e1_path: Path,
    out: Path,
    *,
    allow_test_fixture: bool = False,
) -> dict[str, Any]:
    base = _verify_result(base_path, allow_test_fixture=allow_test_fixture)
    e1 = _verify_result(e1_path, allow_test_fixture=allow_test_fixture)
    if base.get("model_role") != "base" or e1.get("model_role") != "e1":
        raise FormalProtocolError("paired comparison requires base and e1 roles")
    if base.get("input_receipt") != e1.get("input_receipt"):
        raise FormalProtocolError("paired results use different input receipts")
    if (
        base.get("hardware_admission_identity_sha256")
        != e1.get("hardware_admission_identity_sha256")
    ):
        raise FormalProtocolError("paired results use different hardware admissions")
    if base.get("model_binding_identity_sha256") == e1.get("model_binding_identity_sha256"):
        raise FormalProtocolError("paired results bind the same model")
    if (
        base.get("model_checkpoint_content_identity_sha256")
        == e1.get("model_checkpoint_content_identity_sha256")
    ):
        raise FormalProtocolError("paired results bind byte-identical checkpoints")
    base_items = base.get("per_sample") or []
    e1_items = e1.get("per_sample") or []
    if [row.get("global_uid") for row in base_items] != [row.get("global_uid") for row in e1_items]:
        raise FormalProtocolError("paired per-sample coverage/order drift")
    benchmarks: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        base_cell = base["benchmarks"][dataset]
        e1_cell = e1["benchmarks"][dataset]
        paired = [
            (left, right)
            for left, right in zip(base_items, e1_items)
            if left["dataset"] == dataset
        ]
        improved = sum(right["credit"] > left["credit"] for left, right in paired)
        regressed = sum(right["credit"] < left["credit"] for left, right in paired)
        unchanged = len(paired) - improved - regressed
        base_score = float(base_cell["raw_score_percent"])
        e1_score = float(e1_cell["raw_score_percent"])
        benchmarks[dataset] = {
            "n": len(paired),
            "base_raw_score_percent": base_score,
            "e1_raw_score_percent": e1_score,
            "e1_minus_base_points": e1_score - base_score,
            "retention_ratio_percent": (100.0 * e1_score / base_score) if base_score else None,
            "paired_improved": improved,
            "paired_regressed": regressed,
            "paired_unchanged": unchanged,
            "base_health_admitted": base_cell["health_admitted"],
            "e1_health_admitted": e1_cell["health_admitted"],
        }
    admitted = bool(base["all_cells_health_admitted"] and e1["all_cells_health_admitted"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "suite_id": SUITE_ID,
        "status": "PAIRED_GENERAL_RETENTION_ADMITTED" if admitted else "PAIRED_SCORED_HEALTH_NOT_ADMITTED",
        "input_receipt": base["input_receipt"],
        "hardware_admission_identity_sha256": base[
            "hardware_admission_identity_sha256"
        ],
        "base_result": file_binding(base_path),
        "e1_result": file_binding(e1_path),
        "benchmarks": benchmarks,
        "base_raw_general_avg_percent": base["raw_general_avg_percent"],
        "e1_raw_general_avg_percent": e1["raw_general_avg_percent"],
        "raw_general_avg_delta_points": (
            e1["raw_general_avg_percent"] - base["raw_general_avg_percent"]
        ),
        "base_general_avg_percent": base["general_avg_percent"],
        "e1_general_avg_percent": e1["general_avg_percent"],
        "paired_health_admitted": admitted,
        "anti_forgetting_is_reported_not_threshold_tuned": True,
        "training_materialization_eligible": False,
    }
    payload["identity_sha256"] = content_hash(payload)
    atomic_json(out, payload, idempotent=False)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--runtime-index", type=Path, required=True)
    preflight.add_argument("--gpu-selector", action="append", required=True)
    preflight.add_argument("--nvidia-smi-bin", required=True)
    preflight.add_argument("--out", type=Path, required=True)

    hardware_status = sub.add_parser("hardware-status")
    hardware_status.add_argument("--gpu-selector", action="append", required=True)
    hardware_status.add_argument("--nvidia-smi-bin", required=True)

    verify_receipt = sub.add_parser("verify-receipt")
    verify_receipt.add_argument("--input-receipt", type=Path, required=True)
    verify_receipt.add_argument("--deep-sources", action="store_true")
    verify_receipt.add_argument("--live-hardware", action="store_true")
    verify_receipt.add_argument("--require-idle-hardware", action="store_true")

    bind = sub.add_parser("bind-model")
    bind.add_argument("--role", choices=MODEL_ROLES, required=True)
    bind.add_argument("--model", type=Path, required=True)
    bind.add_argument("--input-receipt", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)

    verify_binding = sub.add_parser("verify-model-binding")
    verify_binding.add_argument("--model-binding", type=Path, required=True)
    verify_binding.add_argument("--deep", action="store_true")

    infer = sub.add_parser("infer-shard")
    infer.add_argument("--input-receipt", type=Path, required=True)
    infer.add_argument("--model-binding", type=Path, required=True)
    infer.add_argument("--shard", type=int, required=True)
    infer.add_argument("--num-shards", type=int, required=True)
    infer.add_argument("--batch-size", type=int, required=True)
    infer.add_argument("--out", type=Path, required=True)

    verify = sub.add_parser("verify-shard")
    verify.add_argument("--input-receipt", type=Path, required=True)
    verify.add_argument("--model-binding", type=Path, required=True)
    verify.add_argument("--shard", type=Path, required=True)
    verify.add_argument("--expected-shard", type=int, required=True)

    score = sub.add_parser("score")
    score.add_argument("--input-receipt", type=Path, required=True)
    score.add_argument("--model-binding", type=Path, required=True)
    score.add_argument("--predictions", type=Path, nargs=NUM_SHARDS, required=True)
    score.add_argument("--out", type=Path, required=True)

    verify_result = sub.add_parser("verify-result")
    verify_result.add_argument("--result", type=Path, required=True)
    verify_result.add_argument("--require-admitted", action="store_true")

    compare = sub.add_parser("compare")
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--e1", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "hardware-status":
        hardware = query_l20_hardware(
            [str(value) for value in args.gpu_selector],
            args.nvidia_smi_bin,
            require_idle=False,
        )
        output = {
            "status": "HARDWARE_IDLE" if hardware["idle_at_query"] else "HARDWARE_BUSY",
            "hardware": hardware,
        }
    elif args.command == "preflight":
        if args.config.expanduser().resolve() != CANONICAL_CONFIG.resolve():
            raise FormalProtocolError(
                f"formal CLI requires the canonical config: {CANONICAL_CONFIG}"
            )
        hardware = query_l20_hardware(
            [str(value) for value in args.gpu_selector],
            args.nvidia_smi_bin,
            require_idle=True,
        )
        receipt = build_input_receipt(
            args.config,
            args.runtime_index,
            hardware_inventory=hardware,
        )
        atomic_json(args.out, receipt)
        output = {
            "status": "FORMAL_INPUT_PREFLIGHT_READY",
            "receipt": str(args.out.resolve()),
            "identity_sha256": receipt["identity_sha256"],
            "records": receipt["runtime_index"]["records"],
            "dataset_counts": receipt["runtime_index"]["dataset_counts"],
        }
    elif args.command == "verify-receipt":
        receipt = verify_input_receipt(
            args.input_receipt,
            deep_sources=args.deep_sources,
            live_hardware=args.live_hardware,
            require_idle_hardware=args.require_idle_hardware,
        )
        output = {"status": "INPUT_RECEIPT_VERIFIED", "identity_sha256": receipt["identity_sha256"]}
    elif args.command == "bind-model":
        binding = bind_model(args.role, args.model, args.input_receipt)
        atomic_json(args.out, binding)
        output = {
            "status": "MODEL_BOUND",
            "role": args.role,
            "identity_sha256": binding["identity_sha256"],
            "model_identity_sha256": binding["model"]["identity_sha256"],
        }
    elif args.command == "verify-model-binding":
        binding = verify_model_binding(args.model_binding, deep=args.deep)
        output = {"status": "MODEL_BINDING_VERIFIED", "role": binding["role"]}
    elif args.command == "infer-shard":
        output = infer_shard(
            receipt_path=args.input_receipt,
            binding_path=args.model_binding,
            shard=args.shard,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            out=args.out,
        )
    elif args.command == "verify-shard":
        records = verify_shard(
            receipt_path=args.input_receipt,
            binding_path=args.model_binding,
            shard_path=args.shard,
            expected_shard=args.expected_shard,
        )
        output = {"status": "SHARD_VERIFIED", "records": len(records)}
    elif args.command == "score":
        result = score_predictions(
            receipt_path=args.input_receipt,
            binding_path=args.model_binding,
            shard_paths=args.predictions,
            out=args.out,
        )
        output = {
            "status": result["status"],
            "raw_general_avg_percent": result["raw_general_avg_percent"],
            "general_avg_percent": result["general_avg_percent"],
        }
    elif args.command == "verify-result":
        result = _verify_result(args.result)
        if args.require_admitted and result.get("status") != "FORMAL_SCORE_ADMITTED":
            raise FormalProtocolError(
                f"result health admission failed: {result.get('status')}"
            )
        output = {
            "status": "FORMAL_RESULT_VERIFIED",
            "model_role": result["model_role"],
            "score_status": result["status"],
            "general_avg_percent": result["general_avg_percent"],
        }
    elif args.command == "compare":
        result = compare_results(args.base, args.e1, args.out)
        output = {
            "status": result["status"],
            "raw_general_avg_delta_points": result["raw_general_avg_delta_points"],
        }
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FormalProtocolError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
