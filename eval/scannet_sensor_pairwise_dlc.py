#!/usr/bin/env python3
"""Bound 8-way inference utilities for the formal ScanNet pairwise-depth release.

This module never computes depth labels. It validates the blinded full_v3 request
artifact, passes only ``messages`` and ``images`` to the already verified Swift
TransformersEngine path, and emits keyed response shards for the separate scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import scannet_sensor_pairwise as pair_eval  # noqa: E402


RUN_PROTOCOL = "gsa-depth-v5.0-oracle-pairwise-dlc-v1"
EXPECTED_REQUEST_MANIFEST_SHA256 = (
    "68e41d3e586a765c1dc3b4559fe7fb2b81054a5fdb8ec22996f12976b1abd846"
)
EXPECTED_REQUESTS_SHA256 = (
    "f4873707b32f1fa4e0ad79ad5c20d61c06c19bb84de06ce405fac8a7dee38d28"
)
EXPECTED_REQUEST_RECORDS_SHA256 = (
    "09c6ef120fdea7a03fca1cbe17473e6e664e15b802ed5eaeea0ac1dce3559710"
)
EXPECTED_REQUESTS = 120
NUM_SHARDS = 8
BATCH_SIZE = 4
MAX_TOKENS = 16
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVAL_ID = re.compile(r"^gsa-sdepth-v50-[0-9a-f]{20}$")
_REQUEST_KEYS = {
    "eval_id",
    "evaluation_protocol",
    "track",
    "messages",
    "images",
    "image_sha256",
    "pair",
}
_PAIR_KEYS = {"A", "B", "bbox_coordinate_space"}
_ENTITY_KEYS = {"name", "bbox_2d"}
_PREDICTION_KEYS = {
    "protocol",
    "evaluation_protocol",
    "shard",
    "num_shards",
    "request_index",
    "eval_id",
    "response",
    "hit_token_cap",
    "model_role",
    "model_binding_identity_sha256",
    "model_inventory_identity_sha256",
    "input_receipt_sha256",
    "requests_file_sha256",
    "request_records_sha256",
}
_MODEL_BINDING_KEYS = {
    "schema_version",
    "protocol",
    "role",
    "input_receipt",
    "model",
    "identity_sha256",
}
_MODEL_INVENTORY_KEYS = {
    "path",
    "architecture",
    "source_kind",
    "metadata_files",
    "weight_files",
    "weight_bytes_fully_sha256_hashed",
    "identity_sha256",
}
_PREDICTION_MANIFEST_KEYS = {
    "schema_version",
    "protocol",
    "evaluation_protocol",
    "track",
    "input_receipt",
    "model_binding",
    "model_role",
    "model_inventory_identity_sha256",
    "prediction_shards",
    "merged_predictions",
    "n_predictions",
    "one_row_per_request",
    "all_token_cap_metadata_present",
    "identity_sha256",
}
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
    "chat_template.jinja",
)


class RunnerError(ValueError):
    """Raised when inference inputs or resumable outputs are not exactly bound."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RunnerError(f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path, *, include_mtime: bool = False) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RunnerError(f"missing file: {path}")
    stat = path.stat()
    binding: dict[str, Any] = {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": file_hash(path),
    }
    if include_mtime:
        binding["mtime_ns"] = stat.st_mtime_ns
    return binding


def load_json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RunnerError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunnerError(f"invalid {label}: {path} ({exc.msg})") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"{label} must be a JSON object: {path}")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RunnerError(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunnerError(
                f"{path}:{line_number}: invalid JSON ({exc.msg})"
            ) from exc
        if not isinstance(value, dict):
            raise RunnerError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RunnerError(f"refusing to overwrite a different artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RunnerError(f"refusing to overwrite a different artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _bbox(value: Any, context: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise RunnerError(f"{context}: bbox must contain four integers")
    if any(item < 0 or item > 1000 for item in value):
        raise RunnerError(f"{context}: bbox is outside 0-1000")
    if value[0] >= value[2] or value[1] >= value[3]:
        raise RunnerError(f"{context}: bbox has non-positive area")
    return value


def validate_public_request(row: dict[str, Any], *, verify_image: bool) -> None:
    eval_id = row.get("eval_id")
    context = str(eval_id or "<missing-eval-id>")
    if set(row) != _REQUEST_KEYS:
        raise RunnerError(f"{context}: public request schema drift")
    if not isinstance(eval_id, str) or not _EVAL_ID.fullmatch(eval_id):
        raise RunnerError(f"{context}: invalid eval_id")
    if row.get("evaluation_protocol") != pair_eval.EVALUATION_PROTOCOL:
        raise RunnerError(f"{context}: evaluation protocol drift")
    if row.get("track") != pair_eval.TRACK_S_ORACLE:
        raise RunnerError(f"{context}: only the oracle-pair track is accepted")
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or set(messages[0]) != {"role", "content"}
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise RunnerError(f"{context}: message schema drift")
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise RunnerError(f"{context}: exactly one image path is required")
    image_sha256 = row.get("image_sha256")
    if not isinstance(image_sha256, str) or not _SHA256.fullmatch(image_sha256):
        raise RunnerError(f"{context}: invalid image hash")
    if verify_image:
        image_path = Path(images[0])
        if not image_path.is_file() or file_hash(image_path) != image_sha256:
            raise RunnerError(f"{context}: image missing or hash drifted")
    pair = row.get("pair")
    if not isinstance(pair, dict) or set(pair) != _PAIR_KEYS:
        raise RunnerError(f"{context}: public pair schema drift")
    if pair.get("bbox_coordinate_space") != "image_normalized_0_1000_xyxy":
        raise RunnerError(f"{context}: coordinate contract drift")
    for choice in ("A", "B"):
        entity = pair.get(choice)
        if not isinstance(entity, dict) or set(entity) != _ENTITY_KEYS:
            raise RunnerError(f"{context}/{choice}: public entity schema drift")
        if not isinstance(entity.get("name"), str) or not entity["name"].strip():
            raise RunnerError(f"{context}/{choice}: public name is absent")
        _bbox(entity.get("bbox_2d"), f"{context}/{choice}")
    serialised = json.dumps(row, ensure_ascii=False).casefold()
    private_keys = {f'"{key.casefold()}"' for key in pair_eval._FORBIDDEN_PUBLIC_KEYS}
    if any(key in serialised for key in private_keys):
        raise RunnerError(f"{context}: private field leaked into request")
    if "[instance" in serialised or '"bbox_source"' in serialised:
        raise RunnerError(f"{context}: internal identity or sensor provenance leaked")


def validate_request_release(
    manifest_path: Path, requests_path: Path, *, verify_images: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = manifest_path.expanduser().resolve()
    requests_path = requests_path.expanduser().resolve()
    if file_hash(manifest_path) != EXPECTED_REQUEST_MANIFEST_SHA256:
        raise RunnerError("formal full_v3 request manifest hash drift")
    if file_hash(requests_path) != EXPECTED_REQUESTS_SHA256:
        raise RunnerError("formal full_v3 request JSONL hash drift")
    manifest = load_json(manifest_path, "request manifest")
    if manifest.get("evaluation_protocol") != pair_eval.EVALUATION_PROTOCOL:
        raise RunnerError("request manifest evaluation protocol drift")
    if manifest.get("track") != pair_eval.TRACK_S_ORACLE:
        raise RunnerError("request manifest track drift")
    if manifest.get("n_source") != EXPECTED_REQUESTS or manifest.get("n_requests") != EXPECTED_REQUESTS:
        raise RunnerError("request manifest count drift")
    if manifest.get("label_balance") != pair_eval.FORMAL_LABEL_BALANCE:
        raise RunnerError("request manifest A/B balance drift")
    assignment = manifest.get("ab_assignment") or {}
    if assignment.get("salt") != pair_eval.AB_PERMUTATION_SALT:
        raise RunnerError("request manifest A/B salt drift")
    source = manifest.get("source") or {}
    if (
        source.get("formal_source_release") != pair_eval.FORMAL_SOURCE_RELEASE
        or source.get("formal_source_identity_match") is not True
        or source.get("formal_source_ready") is not True
        or source.get("labels_sha256") != pair_eval.FORMAL_SOURCE_LABELS_SHA256
        or source.get("records_sha256") != pair_eval.FORMAL_SOURCE_RECORDS_SHA256
        or source.get("source_report_sha256") != pair_eval.FORMAL_SOURCE_REPORT_SHA256
    ):
        raise RunnerError("request manifest formal source binding drift")
    if Path(str(manifest.get("requests_path"))).resolve() != requests_path:
        raise RunnerError("request manifest path does not bind the supplied JSONL")
    if manifest.get("requests_file_sha256") != EXPECTED_REQUESTS_SHA256:
        raise RunnerError("request manifest file hash drift")
    if manifest.get("request_records_sha256") != EXPECTED_REQUEST_RECORDS_SHA256:
        raise RunnerError("request manifest canonical hash drift")
    blinding = manifest.get("blinding") or {}
    if (
        blinding.get("internal_instance_ids_exposed") is not False
        or blinding.get("sensor_measurements_exposed") is not False
        or blinding.get("answers_exposed") is not False
    ):
        raise RunnerError("request manifest blinding contract drift")

    rows = load_jsonl(requests_path, "formal request JSONL")
    if len(rows) != EXPECTED_REQUESTS:
        raise RunnerError(f"expected {EXPECTED_REQUESTS} requests, found {len(rows)}")
    for row in rows:
        validate_public_request(row, verify_image=verify_images)
    if pair_eval.canonical_hash(rows) != EXPECTED_REQUEST_RECORDS_SHA256:
        raise RunnerError("request JSONL canonical records drift")
    eval_ids = [row["eval_id"] for row in rows]
    image_hashes = [row["image_sha256"] for row in rows]
    if len(set(eval_ids)) != len(rows) or len(set(image_hashes)) != len(rows):
        raise RunnerError("request eval IDs or images are not unique")
    return manifest, rows


def evaluation_code_binding() -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        Path(pair_eval.__file__).resolve(),
        ROOT / "eval/gsa_eval/runner.py",
        ROOT / "scripts/run_scannet_sensor_pairwise_eval_l20.sh",
    ]
    files = {path.name: file_binding(path) for path in paths}
    return {"files": files, "identity_sha256": content_hash(files)}


def runtime_binding() -> dict[str, Any]:
    distributions = ("ms-swift", "torch", "transformers", "safetensors")
    versions: dict[str, str] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    payload = {"python": sys.version.split()[0], "packages": versions}
    return {**payload, "identity_sha256": content_hash(payload)}


def _build_input_receipt(
    manifest_path: Path, requests_path: Path, *, verify_images: bool
) -> dict[str, Any]:
    manifest, rows = validate_request_release(
        manifest_path, requests_path, verify_images=verify_images
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": RUN_PROTOCOL,
        "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
        "track": pair_eval.TRACK_S_ORACLE,
        "request_release": {
            "manifest": file_binding(manifest_path),
            "requests": file_binding(requests_path),
            "request_records_sha256": EXPECTED_REQUEST_RECORDS_SHA256,
            "n_requests": len(rows),
            "n_unique_images": len({row["image_sha256"] for row in rows}),
            "label_balance": manifest["label_balance"],
            "ab_assignment": manifest["ab_assignment"],
            "blinded_model_fields": ["messages", "images"],
            "private_fields_passed_to_model": [],
        },
        "formal_source": {
            "release": pair_eval.FORMAL_SOURCE_RELEASE,
            "labels_sha256": pair_eval.FORMAL_SOURCE_LABELS_SHA256,
            "records_sha256": pair_eval.FORMAL_SOURCE_RECORDS_SHA256,
            "report_sha256": pair_eval.FORMAL_SOURCE_REPORT_SHA256,
            "training_materialization_eligible": False,
        },
        "generation": {
            "hardware": "exactly_8xL20",
            "num_shards": NUM_SHARDS,
            "shard_rule": "stable_request_order_stride_8",
            "temperature": 0.0,
            "thinking": False,
            "max_tokens": MAX_TOKENS,
            "batch_size_per_gpu": BATCH_SIZE,
            "device_map": "cuda:0_with_one_visible_GPU_per_worker",
        },
        "evaluation_code": evaluation_code_binding(),
        "runtime": runtime_binding(),
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def build_input_receipt(manifest_path: Path, requests_path: Path) -> dict[str, Any]:
    return _build_input_receipt(manifest_path, requests_path, verify_images=True)


def verify_input_receipt(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    receipt = load_json(path, "input receipt")
    identity = receipt.pop("identity_sha256", None)
    if identity != content_hash(receipt):
        raise RunnerError("input receipt identity drift")
    receipt["identity_sha256"] = identity
    release = receipt.get("request_release") or {}
    manifest_path = Path(str((release.get("manifest") or {}).get("path")))
    requests_path = Path(str((release.get("requests") or {}).get("path")))
    rebuilt = build_input_receipt_without_image_rehash(manifest_path, requests_path)
    if rebuilt != receipt:
        raise RunnerError("input receipt no longer matches the request release or code")
    return receipt


def build_input_receipt_without_image_rehash(
    manifest_path: Path, requests_path: Path
) -> dict[str, Any]:
    return _build_input_receipt(manifest_path, requests_path, verify_images=False)


def _weight_paths(model: Path, config: dict[str, Any]) -> list[Path]:
    if config.get("peft_type") or config.get("base_model_name_or_path"):
        raise RunnerError("unmerged adapter config is not accepted")
    adapter_files = sorted(model.rglob("adapter_config.json"))
    adapter_files.extend(sorted(model.rglob("adapter_model.safetensors")))
    adapter_files.extend(sorted(model.rglob("adapter_model.bin")))
    if adapter_files:
        raise RunnerError(f"unmerged adapter artifact is not accepted: {adapter_files[0]}")
    index = model / "model.safetensors.index.json"
    single = model / "model.safetensors"
    if index.is_file():
        value = load_json(index, "model safetensors index")
        weight_map = value.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RunnerError(f"empty model weight map: {index}")
        names = sorted({str(name) for name in weight_map.values()})
        paths = [model / name for name in names]
    elif single.is_file():
        paths = [single]
    else:
        raise RunnerError(f"no merged safetensors weights found under {model}")
    model = model.resolve()
    if any(model not in path.resolve().parents for path in paths):
        raise RunnerError("model weight index escapes the model directory")
    if not all(path.is_file() for path in paths):
        raise RunnerError("model weight index references missing files")
    return paths


def _verify_qwen_model_config(config: dict[str, Any]) -> None:
    architectures = config.get("architectures")
    if (
        config.get("model_type") != "qwen3_5"
        or not isinstance(architectures, list)
        or "Qwen3_5ForConditionalGeneration" not in architectures
        or not isinstance(config.get("vision_config"), dict)
    ):
        raise RunnerError("model is not the required merged Qwen3.5-VL architecture")


def model_inventory(model: Path) -> dict[str, Any]:
    model = model.expanduser().resolve()
    config_path = model / "config.json"
    config = load_json(config_path, "model config")
    _verify_qwen_model_config(config)
    weights = _weight_paths(model, config)
    metadata = {
        name: file_binding(model / name)
        for name in _MODEL_METADATA_NAMES
        if (model / name).is_file()
    }
    weight_bindings = [file_binding(path, include_mtime=True) for path in weights]
    payload: dict[str, Any] = {
        "path": str(model),
        "architecture": "Qwen3_5ForConditionalGeneration",
        "source_kind": "standalone_merged_weights",
        "metadata_files": metadata,
        "weight_files": weight_bindings,
        "weight_bytes_fully_sha256_hashed": True,
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def _verify_model_inventory(inventory: dict[str, Any], *, deep: bool) -> None:
    if not isinstance(inventory, dict) or set(inventory) != _MODEL_INVENTORY_KEYS:
        raise RunnerError("model inventory schema drift")
    identity = inventory.get("identity_sha256")
    without_identity = {key: value for key, value in inventory.items() if key != "identity_sha256"}
    if identity != content_hash(without_identity):
        raise RunnerError("model inventory identity drift")
    if (
        inventory.get("architecture") != "Qwen3_5ForConditionalGeneration"
        or inventory.get("source_kind") != "standalone_merged_weights"
        or inventory.get("weight_bytes_fully_sha256_hashed") is not True
    ):
        raise RunnerError("model inventory contract drift")
    model_value = inventory.get("path")
    if not isinstance(model_value, str) or not model_value:
        raise RunnerError("model inventory path is invalid")
    model = Path(model_value).expanduser().resolve()
    if str(model) != model_value:
        raise RunnerError("model inventory path is not canonical")
    config = load_json(model / "config.json", "model config")
    _verify_qwen_model_config(config)
    expected_weights = _weight_paths(model, config)
    metadata = inventory.get("metadata_files")
    if not isinstance(metadata, dict):
        raise RunnerError("model metadata inventory is invalid")
    expected_metadata = {
        name for name in _MODEL_METADATA_NAMES if (model / name).is_file()
    }
    if set(metadata) != expected_metadata:
        raise RunnerError("model metadata inventory coverage drift")
    for name in sorted(expected_metadata):
        binding = metadata[name]
        if not isinstance(binding, dict) or file_binding(model / name) != binding:
            raise RunnerError(f"model metadata drift: {model / name}")
    weight_bindings = inventory.get("weight_files")
    if not isinstance(weight_bindings, list) or len(weight_bindings) != len(expected_weights):
        raise RunnerError("model weight inventory coverage drift")
    for binding, expected_path in zip(weight_bindings, expected_weights):
        if not isinstance(binding, dict) or set(binding) != {
            "path",
            "bytes",
            "sha256",
            "mtime_ns",
        }:
            raise RunnerError("model weight binding schema drift")
        path = Path(str(binding.get("path"))).expanduser().resolve()
        if path != expected_path.resolve():
            raise RunnerError("model weight inventory path drift")
        if not path.is_file():
            raise RunnerError(f"model weight missing: {path}")
        stat = path.stat()
        if stat.st_size != binding.get("bytes") or stat.st_mtime_ns != binding.get("mtime_ns"):
            raise RunnerError(f"model weight stat drift: {path}")
        if deep and file_hash(path) != binding.get("sha256"):
            raise RunnerError(f"model weight hash drift: {path}")


def bind_model(role: str, model: Path, input_receipt: Path) -> dict[str, Any]:
    if role not in {"base", "merged"}:
        raise RunnerError(f"unsupported model role: {role}")
    receipt_path = input_receipt.expanduser().resolve()
    verify_input_receipt(receipt_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": RUN_PROTOCOL,
        "role": role,
        "input_receipt": file_binding(receipt_path),
        "model": model_inventory(model),
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def verify_model_binding(path: Path, *, deep: bool) -> dict[str, Any]:
    path = path.expanduser().resolve()
    binding = load_json(path, "model binding")
    if set(binding) != _MODEL_BINDING_KEYS:
        raise RunnerError("model binding schema drift")
    identity = binding.pop("identity_sha256", None)
    if identity != content_hash(binding):
        raise RunnerError("model binding identity drift")
    binding["identity_sha256"] = identity
    if (
        binding.get("schema_version") != 1
        or binding.get("protocol") != RUN_PROTOCOL
        or binding.get("role") not in {"base", "merged"}
    ):
        raise RunnerError("model binding protocol or role drift")
    receipt_binding = binding.get("input_receipt")
    if not isinstance(receipt_binding, dict):
        raise RunnerError("model binding input receipt schema drift")
    receipt_path = Path(str(receipt_binding.get("path")))
    if file_binding(receipt_path) != receipt_binding:
        raise RunnerError("model binding input receipt drift")
    verify_input_receipt(receipt_path)
    _verify_model_inventory(binding.get("model") or {}, deep=deep)
    return binding


PublicInfer = Callable[[list[dict[str, Any]], int], list[tuple[str, bool]]]


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
        raise RunnerError("inference engine returned an unexpected response count")
    output: list[tuple[str, bool]] = []
    for response in responses:
        try:
            choice = response.choices[0]
            output.append(
                (choice.message.content or "", choice.finish_reason == "length")
            )
        except Exception:
            output.append(("", False))
    return output


def _prediction_context(receipt_path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    receipt = verify_input_receipt(receipt_path)
    release = receipt["request_release"]
    return {
        "model_role": binding["role"],
        "model_binding_identity_sha256": binding["identity_sha256"],
        "model_inventory_identity_sha256": binding["model"]["identity_sha256"],
        "input_receipt_sha256": file_hash(receipt_path),
        "requests_file_sha256": release["requests"]["sha256"],
        "request_records_sha256": release["request_records_sha256"],
    }


def infer_shard(
    *,
    receipt_path: Path,
    binding_path: Path,
    shard: int,
    num_shards: int,
    batch_size: int,
    max_tokens: int,
    out: Path,
    infer_public: PublicInfer | None = None,
) -> dict[str, Any]:
    if num_shards != NUM_SHARDS or not 0 <= shard < NUM_SHARDS:
        raise RunnerError(f"formal inference requires shard 0..{NUM_SHARDS - 1}/{NUM_SHARDS}")
    if batch_size != BATCH_SIZE or max_tokens != MAX_TOKENS:
        raise RunnerError(
            f"formal inference requires batch_size={BATCH_SIZE}, max_tokens={MAX_TOKENS}"
        )
    receipt_path = receipt_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    out = out.expanduser().resolve()
    binding = verify_model_binding(binding_path, deep=False)
    if binding["input_receipt"]["sha256"] != file_hash(receipt_path):
        raise RunnerError("model binding and input receipt differ")
    receipt = verify_input_receipt(receipt_path)
    requests_path = Path(receipt["request_release"]["requests"]["path"])
    _, rows = validate_request_release(
        Path(receipt["request_release"]["manifest"]["path"]),
        requests_path,
        verify_images=False,
    )
    if out.exists():
        summary = verify_shard(
            receipt_path=receipt_path,
            binding_path=binding_path,
            shard_path=out,
            expected_shard=shard,
        )
        summary["status"] = "reused"
        return summary
    selected = list(enumerate(rows))[shard::NUM_SHARDS]
    if len(selected) != EXPECTED_REQUESTS // NUM_SHARDS:
        raise RunnerError(f"unexpected shard size: {len(selected)}")
    public_inputs = [
        {"messages": row["messages"], "images": row["images"]}
        for _, row in selected
    ]
    if infer_public is None:
        outputs = _real_infer(
            public_inputs,
            max_tokens,
            model=binding["model"]["path"],
            batch_size=batch_size,
        )
    else:
        outputs = infer_public(public_inputs, max_tokens)
    if len(outputs) != len(selected):
        raise RunnerError("inference callback returned an unexpected response count")
    context = _prediction_context(receipt_path, binding)
    records: list[dict[str, Any]] = []
    for (index, request), output in zip(selected, outputs):
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not isinstance(output[0], str)
            or not isinstance(output[1], bool)
        ):
            raise RunnerError("inference output must be (response string, capped bool)")
        records.append(
            {
                "protocol": RUN_PROTOCOL,
                "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
                "shard": shard,
                "num_shards": NUM_SHARDS,
                "request_index": index,
                "eval_id": request["eval_id"],
                "response": output[0],
                "hit_token_cap": output[1],
                **context,
            }
        )
    atomic_jsonl(out, records)
    summary = verify_shard(
        receipt_path=receipt_path,
        binding_path=binding_path,
        shard_path=out,
        expected_shard=shard,
    )
    summary["status"] = "generated"
    return summary


def verify_shard(
    *, receipt_path: Path, binding_path: Path, shard_path: Path,
    expected_shard: int | None = None
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    binding = verify_model_binding(binding_path, deep=False)
    if binding["input_receipt"]["sha256"] != file_hash(receipt_path):
        raise RunnerError("model binding and shard input receipt differ")
    receipt = verify_input_receipt(receipt_path)
    _, requests = validate_request_release(
        Path(receipt["request_release"]["manifest"]["path"]),
        Path(receipt["request_release"]["requests"]["path"]),
        verify_images=False,
    )
    records = load_jsonl(shard_path, "prediction shard")
    if not records:
        raise RunnerError(f"empty prediction shard: {shard_path}")
    shard_values = {row.get("shard") for row in records}
    if len(shard_values) != 1 or not isinstance(next(iter(shard_values)), int):
        raise RunnerError(f"invalid shard identity: {shard_path}")
    shard = next(iter(shard_values))
    if expected_shard is not None and shard != expected_shard:
        raise RunnerError(f"expected shard {expected_shard}, found {shard}")
    if not 0 <= shard < NUM_SHARDS:
        raise RunnerError(f"invalid shard index: {shard}")
    expected = list(enumerate(requests))[shard::NUM_SHARDS]
    if len(records) != len(expected):
        raise RunnerError(f"shard {shard} row count drift")
    context = _prediction_context(receipt_path, binding)
    for record, (index, request) in zip(records, expected):
        if set(record) != _PREDICTION_KEYS:
            raise RunnerError(f"shard {shard}: prediction schema drift")
        expected_fields = {
            "protocol": RUN_PROTOCOL,
            "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
            "shard": shard,
            "num_shards": NUM_SHARDS,
            "request_index": index,
            "eval_id": request["eval_id"],
            **context,
        }
        if any(record.get(key) != value for key, value in expected_fields.items()):
            raise RunnerError(f"shard {shard}/{index}: prediction binding drift")
        if not isinstance(record.get("response"), str):
            raise RunnerError(f"shard {shard}/{index}: response is not a string")
        if not isinstance(record.get("hit_token_cap"), bool):
            raise RunnerError(f"shard {shard}/{index}: token-cap metadata is absent")
    return {
        "shard": shard,
        "records": len(records),
        "sha256": file_hash(shard_path),
        "model_role": binding["role"],
    }


def merge_shards(
    *, receipt_path: Path, binding_path: Path, shard_paths: list[Path],
    out: Path, manifest_out: Path
) -> dict[str, Any]:
    if len(shard_paths) != NUM_SHARDS:
        raise RunnerError(f"expected exactly {NUM_SHARDS} prediction shard paths")
    receipt_path = receipt_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    binding = verify_model_binding(binding_path, deep=True)
    if binding["input_receipt"]["sha256"] != file_hash(receipt_path):
        raise RunnerError("model binding and merge input receipt differ")
    receipt = verify_input_receipt(receipt_path)
    _, requests = validate_request_release(
        Path(receipt["request_release"]["manifest"]["path"]),
        Path(receipt["request_release"]["requests"]["path"]),
        verify_images=False,
    )
    summaries = [
        verify_shard(
            receipt_path=receipt_path,
            binding_path=binding_path,
            shard_path=path,
        )
        for path in shard_paths
    ]
    if {summary["shard"] for summary in summaries} != set(range(NUM_SHARDS)):
        raise RunnerError("prediction shards do not cover shard_0..shard_7 exactly")
    records: list[dict[str, Any]] = []
    for path in shard_paths:
        records.extend(load_jsonl(path, "prediction shard"))
    records.sort(key=lambda row: row["request_index"])
    if [row["request_index"] for row in records] != list(range(EXPECTED_REQUESTS)):
        raise RunnerError("merged predictions do not cover every request exactly once")
    if [row["eval_id"] for row in records] != [row["eval_id"] for row in requests]:
        raise RunnerError("merged prediction order or eval IDs drifted")
    atomic_jsonl(out, records)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": RUN_PROTOCOL,
        "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
        "track": pair_eval.TRACK_S_ORACLE,
        "input_receipt": file_binding(receipt_path),
        "model_binding": file_binding(binding_path),
        "model_role": binding["role"],
        "model_inventory_identity_sha256": binding["model"]["identity_sha256"],
        "prediction_shards": [
            file_binding(path)
            for _, path in sorted(
                zip((summary["shard"] for summary in summaries), shard_paths)
            )
        ],
        "merged_predictions": file_binding(out),
        "n_predictions": len(records),
        "one_row_per_request": True,
        "all_token_cap_metadata_present": all(
            isinstance(row["hit_token_cap"], bool) for row in records
        ),
    }
    payload["identity_sha256"] = content_hash(payload)
    atomic_json(manifest_out, payload)
    return payload


def verify_scoring_bundle(
    *, manifest_path: Path, binding_path: Path, predictions_path: Path
) -> dict[str, Any]:
    """Verify the complete formal inference provenance before labels are scored."""
    manifest_path = manifest_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    predictions_path = predictions_path.expanduser().resolve()
    manifest = load_json(manifest_path, "prediction manifest")
    if set(manifest) != _PREDICTION_MANIFEST_KEYS:
        raise RunnerError("prediction manifest schema drift")
    identity = manifest.get("identity_sha256")
    unsigned = {
        key: value for key, value in manifest.items() if key != "identity_sha256"
    }
    if identity != content_hash(unsigned):
        raise RunnerError("prediction manifest identity drift")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol") != RUN_PROTOCOL
        or manifest.get("evaluation_protocol") != pair_eval.EVALUATION_PROTOCOL
        or manifest.get("track") != pair_eval.TRACK_S_ORACLE
        or manifest.get("n_predictions") != EXPECTED_REQUESTS
        or manifest.get("one_row_per_request") is not True
        or manifest.get("all_token_cap_metadata_present") is not True
    ):
        raise RunnerError("prediction manifest contract drift")

    if manifest.get("model_binding") != file_binding(binding_path):
        raise RunnerError("prediction manifest model binding drift")
    binding = verify_model_binding(binding_path, deep=True)
    if (
        manifest.get("model_role") != binding["role"]
        or manifest.get("model_inventory_identity_sha256")
        != binding["model"]["identity_sha256"]
    ):
        raise RunnerError("prediction manifest model identity drift")

    receipt_binding = binding["input_receipt"]
    if manifest.get("input_receipt") != receipt_binding:
        raise RunnerError("prediction manifest input receipt drift")
    receipt_path = Path(receipt_binding["path"]).expanduser().resolve()
    if receipt_binding != file_binding(receipt_path):
        raise RunnerError("prediction bundle input receipt file drift")
    receipt = verify_input_receipt(receipt_path)
    formal_source = receipt.get("formal_source")
    if formal_source != {
        "release": pair_eval.FORMAL_SOURCE_RELEASE,
        "labels_sha256": pair_eval.FORMAL_SOURCE_LABELS_SHA256,
        "records_sha256": pair_eval.FORMAL_SOURCE_RECORDS_SHA256,
        "report_sha256": pair_eval.FORMAL_SOURCE_REPORT_SHA256,
        "training_materialization_eligible": False,
    }:
        raise RunnerError("prediction bundle formal source drift")
    release = receipt.get("request_release") or {}
    if (
        release.get("requests", {}).get("sha256") != EXPECTED_REQUESTS_SHA256
        or release.get("manifest", {}).get("sha256")
        != EXPECTED_REQUEST_MANIFEST_SHA256
        or release.get("request_records_sha256")
        != EXPECTED_REQUEST_RECORDS_SHA256
        or release.get("n_requests") != EXPECTED_REQUESTS
        or release.get("n_unique_images") != EXPECTED_REQUESTS
        or release.get("label_balance") != pair_eval.FORMAL_LABEL_BALANCE
    ):
        raise RunnerError("prediction bundle request release drift")
    _, requests = validate_request_release(
        Path(release["manifest"]["path"]),
        Path(release["requests"]["path"]),
        verify_images=False,
    )

    if manifest.get("merged_predictions") != file_binding(predictions_path):
        raise RunnerError("prediction manifest merged file drift")
    records = load_jsonl(predictions_path, "merged predictions")
    if len(records) != EXPECTED_REQUESTS:
        raise RunnerError("merged prediction row count drift")
    context = _prediction_context(receipt_path, binding)
    for index, (record, request) in enumerate(zip(records, requests)):
        if set(record) != _PREDICTION_KEYS:
            raise RunnerError(f"merged prediction {index}: schema drift")
        expected_fields = {
            "protocol": RUN_PROTOCOL,
            "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
            "shard": index % NUM_SHARDS,
            "num_shards": NUM_SHARDS,
            "request_index": index,
            "eval_id": request["eval_id"],
            **context,
        }
        if any(record.get(key) != value for key, value in expected_fields.items()):
            raise RunnerError(f"merged prediction {index}: provenance drift")
        if not isinstance(record.get("response"), str):
            raise RunnerError(f"merged prediction {index}: response is not a string")
        if not isinstance(record.get("hit_token_cap"), bool):
            raise RunnerError(
                f"merged prediction {index}: token-cap metadata is absent"
            )

    shard_bindings = manifest.get("prediction_shards")
    if not isinstance(shard_bindings, list) or len(shard_bindings) != NUM_SHARDS:
        raise RunnerError("prediction manifest shard coverage drift")
    shard_records: list[dict[str, Any]] = []
    for shard, shard_binding in enumerate(shard_bindings):
        if not isinstance(shard_binding, dict):
            raise RunnerError("prediction manifest shard binding is invalid")
        shard_path = Path(str(shard_binding.get("path"))).expanduser().resolve()
        if shard_binding != file_binding(shard_path):
            raise RunnerError(f"prediction shard {shard} file drift")
        verify_shard(
            receipt_path=receipt_path,
            binding_path=binding_path,
            shard_path=shard_path,
            expected_shard=shard,
        )
        shard_records.extend(load_jsonl(shard_path, f"prediction shard {shard}"))
    shard_records.sort(key=lambda row: row["request_index"])
    if shard_records != records:
        raise RunnerError("merged predictions differ from the verified shards")

    provenance: dict[str, Any] = {
        "verified": True,
        "protocol": RUN_PROTOCOL,
        "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
        "track": pair_eval.TRACK_S_ORACLE,
        "prediction_manifest": file_binding(manifest_path),
        "prediction_manifest_identity_sha256": identity,
        "merged_predictions": manifest["merged_predictions"],
        "prediction_shards": shard_bindings,
        "model_binding": manifest["model_binding"],
        "model_binding_identity_sha256": binding["identity_sha256"],
        "model_role": binding["role"],
        "model_inventory": binding["model"],
        "input_receipt": manifest["input_receipt"],
        "input_receipt_identity_sha256": receipt["identity_sha256"],
        "request_release": release,
        "formal_source": formal_source,
        "evaluation_code": receipt["evaluation_code"],
        "runtime": receipt["runtime"],
        "n_predictions": len(records),
    }
    provenance["identity_sha256"] = content_hash(provenance)
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--request-manifest", type=Path, required=True)
    preflight.add_argument("--requests", type=Path, required=True)
    preflight.add_argument("--out", type=Path, required=True)

    bind = sub.add_parser("bind-model")
    bind.add_argument("--role", choices=("base", "merged"), required=True)
    bind.add_argument("--model", type=Path, required=True)
    bind.add_argument("--input-receipt", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)

    verify_model = sub.add_parser("verify-model-binding")
    verify_model.add_argument("--binding", type=Path, required=True)
    verify_model.add_argument("--deep", action="store_true")

    infer = sub.add_parser("infer-shard")
    infer.add_argument("--input-receipt", type=Path, required=True)
    infer.add_argument("--model-binding", type=Path, required=True)
    infer.add_argument("--shard", type=int, required=True)
    infer.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    infer.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    infer.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    infer.add_argument("--out", type=Path, required=True)

    verify = sub.add_parser("verify-shard")
    verify.add_argument("--input-receipt", type=Path, required=True)
    verify.add_argument("--model-binding", type=Path, required=True)
    verify.add_argument("--shard", type=Path, required=True)
    verify.add_argument("--expected-shard", type=int)

    merge = sub.add_parser("merge-shards")
    merge.add_argument("--input-receipt", type=Path, required=True)
    merge.add_argument("--model-binding", type=Path, required=True)
    merge.add_argument("--predictions", nargs="+", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("--manifest-out", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            payload = build_input_receipt(args.request_manifest, args.requests)
            atomic_json(args.out, payload)
        elif args.command == "bind-model":
            payload = bind_model(args.role, args.model, args.input_receipt)
            atomic_json(args.out, payload)
        elif args.command == "verify-model-binding":
            payload = verify_model_binding(args.binding, deep=args.deep)
        elif args.command == "infer-shard":
            payload = infer_shard(
                receipt_path=args.input_receipt,
                binding_path=args.model_binding,
                shard=args.shard,
                num_shards=args.num_shards,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                out=args.out,
            )
        elif args.command == "verify-shard":
            payload = verify_shard(
                receipt_path=args.input_receipt,
                binding_path=args.model_binding,
                shard_path=args.shard,
                expected_shard=args.expected_shard,
            )
        else:
            payload = merge_shards(
                receipt_path=args.input_receipt,
                binding_path=args.model_binding,
                shard_paths=args.predictions,
                out=args.out,
                manifest_out=args.manifest_out,
            )
    except (RunnerError, pair_eval.ContractError) as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
