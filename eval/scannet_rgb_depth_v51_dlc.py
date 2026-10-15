#!/usr/bin/env python3
"""Bound four-L20 inference for the ScanNet RGB-depth v5.1 formal40 release."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import scannet_rgb_depth_v51 as pair_eval  # noqa: E402


RUN_PROTOCOL = "gsa-depth-v5.1-formal40-paired-gs-dlc-v1"
PREDICTED_REQUEST_PROTOCOL = "gsa-depth-v5.1-formal40-s-pred-request-v1"
NUM_SHARDS = 4
BATCH_SIZE = 4
STAGE_MAX_TOKENS = {
    pair_eval.TRACK_G: 96,
    pair_eval.TRACK_S_ORACLE: 16,
    pair_eval.TRACK_S_PRED: 16,
}
TRACK_ALIASES = {
    "g": pair_eval.TRACK_G,
    pair_eval.TRACK_G: pair_eval.TRACK_G,
    "s-oracle": pair_eval.TRACK_S_ORACLE,
    "s_oracle": pair_eval.TRACK_S_ORACLE,
    pair_eval.TRACK_S_ORACLE: pair_eval.TRACK_S_ORACLE,
    "s-pred": pair_eval.TRACK_S_PRED,
    "s_pred": pair_eval.TRACK_S_PRED,
    pair_eval.TRACK_S_PRED: pair_eval.TRACK_S_PRED,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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
_MODEL_INVENTORY_KEYS = {
    "path",
    "architecture",
    "model_type",
    "source_kind",
    "metadata_files",
    "weight_files",
    "weight_bytes_fully_sha256_hashed",
    "identity_sha256",
}
_MODEL_BINDING_KEYS = {
    "schema_version",
    "protocol",
    "role",
    "input_receipt",
    "model",
    "identity_sha256",
}
_PREDICTION_KEYS = {
    "protocol",
    "evaluation_protocol",
    "track",
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
    "request_manifest_sha256",
    "requests_file_sha256",
    "request_records_sha256",
    "decoding",
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
    "request_manifest",
    "requests",
    "request_records_sha256",
    "decoding",
    "prediction_shards",
    "merged_predictions",
    "n_predictions",
    "one_row_per_request",
    "all_token_cap_metadata_present",
    "identity_sha256",
}
_PREDICTED_REQUEST_MANIFEST_KEYS = {
    "schema_version",
    "protocol",
    "evaluation_protocol",
    "track",
    "input_receipt",
    "model_binding",
    "model_binding_identity_sha256",
    "g_prediction_manifest",
    "g_predictions",
    "requests",
    "request_records_sha256",
    "source_records",
    "valid_grounding_records",
    "invalid_groundings",
    "complete_g_coverage",
    "training_materialization_eligible",
    "identity_sha256",
}


class RunnerError(ValueError):
    """Raised when inference provenance or execution contracts drift."""


@dataclass(frozen=True)
class RequestContext:
    track: str
    request_manifest: dict[str, Any]
    requests: dict[str, Any]
    request_records_sha256: str
    rows: tuple[dict[str, Any], ...]


PublicInfer = Callable[[list[dict[str, Any]], int], list[tuple[str, bool]]]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


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


def _verify_file_binding(binding: Any, *, deep: bool) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
        raise RunnerError("file binding schema drift")
    path = Path(str(binding.get("path"))).expanduser().resolve()
    if str(path) != binding.get("path") or not path.is_file():
        raise RunnerError(f"bound file is missing or non-canonical: {path}")
    if path.stat().st_size != binding.get("bytes"):
        raise RunnerError(f"bound file size drift: {path}")
    value = binding.get("sha256")
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RunnerError(f"bound file hash is invalid: {path}")
    if deep and file_hash(path) != value:
        raise RunnerError(f"bound file hash drift: {path}")
    return path


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


def load_jsonl(path: Path, label: str, *, allow_empty: bool = False) -> list[dict[str, Any]]:
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
            raise RunnerError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(value, dict):
            raise RunnerError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    if not rows and not allow_empty:
        raise RunnerError(f"{label} has no rows: {path}")
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
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


def atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


def _normalise_track(value: str) -> str:
    try:
        return TRACK_ALIASES[value]
    except KeyError as exc:
        raise RunnerError(f"unknown track: {value}") from exc


def decoding_contract(track: str) -> dict[str, Any]:
    track = _normalise_track(track)
    return {
        "thinking": False,
        "temperature": 0.0,
        "max_tokens": STAGE_MAX_TOKENS[track],
        "batch_size_per_gpu": BATCH_SIZE,
    }


def evaluation_code_binding() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        Path(pair_eval.__file__).resolve(),
        ROOT / "eval/gsa_eval/runner.py",
        ROOT / "scripts/run_scannet_rgb_depth_v51_eval_l20_4gpu.sh",
    )
    files = {path.name: file_binding(path) for path in paths}
    return {"files": files, "identity_sha256": content_hash(files)}


def runtime_binding() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ("ms-swift", "torch", "transformers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    payload = {"python": sys.version.split()[0], "packages": versions}
    return {**payload, "identity_sha256": content_hash(payload)}


def _build_input_receipt(release_manifest: Path, *, verify_images: bool) -> dict[str, Any]:
    release_manifest = release_manifest.expanduser().resolve()
    try:
        release = pair_eval.validate_request_release(
            release_manifest, verify_images=verify_images
        )
    except pair_eval.ContractError as exc:
        raise RunnerError(f"formal request release failed: {exc}") from exc
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": RUN_PROTOCOL,
        "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
        "request_release": {
            "manifest": file_binding(release_manifest),
            "identity_sha256": release.manifest["identity_sha256"],
            "static_requests": release.manifest["requests"],
            "records": pair_eval.EXPECTED_RECORDS,
            "unique_images": pair_eval.EXPECTED_RECORDS,
            "label_balance": pair_eval.EXPECTED_LABEL_BALANCE,
            "blinded_model_fields": ["messages", "images"],
            "private_fields_passed_to_model": [],
        },
        "formal_source": release.source.binding,
        "generation": {
            "hardware": "exactly_4_physical_L20",
            "logical_shards": NUM_SHARDS,
            "shard_rule": "stable_request_order_stride_4",
            "stage_decoding": {
                track: decoding_contract(track)
                for track in (pair_eval.TRACK_G, pair_eval.TRACK_S_ORACLE, pair_eval.TRACK_S_PRED)
            },
            "same_bound_model_all_stages": True,
        },
        "evaluation_code": evaluation_code_binding(),
        "runtime": runtime_binding(),
        "training_materialization_eligible": False,
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def build_input_receipt(release_manifest: Path) -> dict[str, Any]:
    return _build_input_receipt(release_manifest, verify_images=True)


def verify_input_receipt(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    receipt = load_json(path, "input receipt")
    identity = receipt.get("identity_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise RunnerError("input receipt identity drift")
    if receipt.get("schema_version") != 1 or receipt.get("protocol") != RUN_PROTOCOL:
        raise RunnerError("input receipt protocol drift")
    release = receipt.get("request_release") or {}
    release_manifest = _verify_file_binding(release.get("manifest"), deep=True)
    rebuilt = _build_input_receipt(release_manifest, verify_images=False)
    if rebuilt != receipt:
        raise RunnerError("input receipt no longer matches release/code/runtime")
    return receipt


def _weight_paths(model: Path, config: dict[str, Any]) -> list[Path]:
    if config.get("peft_type") or config.get("base_model_name_or_path"):
        raise RunnerError("unmerged adapter config is not accepted")
    adapter_files = sorted(model.rglob("adapter_config.json"))
    adapter_files += sorted(model.rglob("adapter_model.safetensors"))
    adapter_files += sorted(model.rglob("adapter_model.bin"))
    if adapter_files:
        raise RunnerError(f"unmerged adapter artifact is not accepted: {adapter_files[0]}")
    index = model / "model.safetensors.index.json"
    single = model / "model.safetensors"
    if index.is_file():
        value = load_json(index, "model safetensors index")
        weight_map = value.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RunnerError(f"empty model weight map: {index}")
        paths = [model / name for name in sorted({str(name) for name in weight_map.values()})]
    elif single.is_file():
        paths = [single]
    else:
        raise RunnerError(f"no merged safetensors weights found under {model}")
    resolved_model = model.resolve()
    if any(resolved_model not in path.resolve().parents for path in paths):
        raise RunnerError("model weight index escapes the model directory")
    if not all(path.is_file() for path in paths):
        raise RunnerError("model weight index references missing files")
    return paths


def _model_identity_fields(config: dict[str, Any]) -> tuple[str, str]:
    architectures = config.get("architectures")
    model_type = config.get("model_type")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or not isinstance(architectures[0], str)
        or not architectures[0]
        or not isinstance(model_type, str)
        or not model_type
        or not isinstance(config.get("vision_config"), dict)
    ):
        raise RunnerError("model config is not a single merged vision-language architecture")
    return architectures[0], model_type


def model_inventory(model: Path) -> dict[str, Any]:
    model = model.expanduser().resolve()
    config = load_json(model / "config.json", "model config")
    architecture, model_type = _model_identity_fields(config)
    weights = _weight_paths(model, config)
    metadata = {
        name: file_binding(model / name)
        for name in _MODEL_METADATA_NAMES
        if (model / name).is_file()
    }
    payload: dict[str, Any] = {
        "path": str(model),
        "architecture": architecture,
        "model_type": model_type,
        "source_kind": "standalone_merged_weights",
        "metadata_files": metadata,
        "weight_files": [file_binding(path, include_mtime=True) for path in weights],
        "weight_bytes_fully_sha256_hashed": True,
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def _verify_model_inventory(inventory: Any, *, deep: bool) -> None:
    if not isinstance(inventory, dict) or set(inventory) != _MODEL_INVENTORY_KEYS:
        raise RunnerError("model inventory schema drift")
    identity = inventory.get("identity_sha256")
    unsigned = {key: value for key, value in inventory.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise RunnerError("model inventory identity drift")
    if (
        inventory.get("source_kind") != "standalone_merged_weights"
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
    architecture, model_type = _model_identity_fields(config)
    if inventory.get("architecture") != architecture or inventory.get("model_type") != model_type:
        raise RunnerError("model architecture identity drift")
    weights = _weight_paths(model, config)
    metadata = inventory.get("metadata_files")
    expected_metadata = {name for name in _MODEL_METADATA_NAMES if (model / name).is_file()}
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata:
        raise RunnerError("model metadata inventory coverage drift")
    for name in sorted(expected_metadata):
        if metadata[name] != file_binding(model / name):
            raise RunnerError(f"model metadata drift: {model / name}")
    weight_bindings = inventory.get("weight_files")
    if not isinstance(weight_bindings, list) or len(weight_bindings) != len(weights):
        raise RunnerError("model weight inventory coverage drift")
    for binding, expected_path in zip(weight_bindings, weights):
        if not isinstance(binding, dict) or set(binding) != {
            "path",
            "bytes",
            "sha256",
            "mtime_ns",
        }:
            raise RunnerError("model weight binding schema drift")
        path = Path(str(binding.get("path"))).expanduser().resolve()
        if path != expected_path.resolve() or not path.is_file():
            raise RunnerError("model weight path drift")
        stat = path.stat()
        if stat.st_size != binding.get("bytes") or stat.st_mtime_ns != binding.get("mtime_ns"):
            raise RunnerError(f"model weight stat drift: {path}")
        if deep and file_hash(path) != binding.get("sha256"):
            raise RunnerError(f"model weight hash drift: {path}")


def bind_model(role: str, model: Path, input_receipt: Path) -> dict[str, Any]:
    if role not in {"base", "evaluated"}:
        raise RunnerError(f"unsupported model role: {role}")
    input_receipt = input_receipt.expanduser().resolve()
    verify_input_receipt(input_receipt)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": RUN_PROTOCOL,
        "role": role,
        "input_receipt": file_binding(input_receipt),
        "model": model_inventory(model),
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def verify_model_binding(path: Path, *, deep: bool) -> dict[str, Any]:
    path = path.expanduser().resolve()
    binding = load_json(path, "model binding")
    if set(binding) != _MODEL_BINDING_KEYS:
        raise RunnerError("model binding schema drift")
    identity = binding.get("identity_sha256")
    unsigned = {key: value for key, value in binding.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise RunnerError("model binding identity drift")
    if (
        binding.get("schema_version") != 1
        or binding.get("protocol") != RUN_PROTOCOL
        or binding.get("role") not in {"base", "evaluated"}
    ):
        raise RunnerError("model binding protocol or role drift")
    receipt_path = _verify_file_binding(binding.get("input_receipt"), deep=True)
    verify_input_receipt(receipt_path)
    _verify_model_inventory(binding.get("model"), deep=deep)
    return binding


def _release_from_receipt(receipt: dict[str, Any], *, verify_images: bool) -> pair_eval.RequestRelease:
    release_binding = (receipt.get("request_release") or {}).get("manifest")
    release_path = _verify_file_binding(release_binding, deep=True)
    try:
        release = pair_eval.validate_request_release(release_path, verify_images=verify_images)
    except pair_eval.ContractError as exc:
        raise RunnerError(f"formal request release failed: {exc}") from exc
    if (
        release.manifest["identity_sha256"]
        != (receipt.get("request_release") or {}).get("identity_sha256")
        or release.source.binding != receipt.get("formal_source")
    ):
        raise RunnerError("input receipt formal release drift")
    return release


def _static_request_context(
    receipt_path: Path, binding_path: Path, track: str
) -> RequestContext:
    if track not in {pair_eval.TRACK_G, pair_eval.TRACK_S_ORACLE}:
        raise RunnerError(f"{track}: not a static request track")
    receipt = verify_input_receipt(receipt_path)
    binding = verify_model_binding(binding_path, deep=False)
    if binding["input_receipt"] != file_binding(receipt_path):
        raise RunnerError("model binding and input receipt differ")
    release = _release_from_receipt(receipt, verify_images=False)
    item = release.manifest["requests"][track]
    return RequestContext(
        track=track,
        request_manifest=file_binding(Path(receipt["request_release"]["manifest"]["path"])),
        requests=item["file"],
        request_records_sha256=item["records_sha256"],
        rows=tuple(release.requests[track]),
    )


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
            output.append((choice.message.content or "", choice.finish_reason == "length"))
        except Exception:
            output.append(("", False))
    return output


def _prediction_context(
    receipt_path: Path, binding: dict[str, Any], context: RequestContext
) -> dict[str, Any]:
    return {
        "model_role": binding["role"],
        "model_binding_identity_sha256": binding["identity_sha256"],
        "model_inventory_identity_sha256": binding["model"]["identity_sha256"],
        "input_receipt_sha256": file_hash(receipt_path),
        "request_manifest_sha256": context.request_manifest["sha256"],
        "requests_file_sha256": context.requests["sha256"],
        "request_records_sha256": context.request_records_sha256,
        "decoding": decoding_contract(context.track),
    }


def _get_request_context(
    *,
    receipt_path: Path,
    binding_path: Path,
    track: str,
    dynamic_request_manifest: Path | None,
    verify_images: bool,
) -> RequestContext:
    track = _normalise_track(track)
    if track != pair_eval.TRACK_S_PRED:
        if dynamic_request_manifest is not None:
            raise RunnerError("dynamic request manifest is reserved for S-pred")
        return _static_request_context(receipt_path, binding_path, track)
    if dynamic_request_manifest is None:
        raise RunnerError("S-pred requires --dynamic-request-manifest")
    verified = verify_predicted_request_release(
        manifest_path=dynamic_request_manifest,
        binding_path=binding_path,
        verify_images=verify_images,
    )
    manifest = verified["manifest"]
    if manifest["input_receipt"] != file_binding(receipt_path):
        raise RunnerError("S-pred request release and input receipt differ")
    return RequestContext(
        track=track,
        request_manifest=file_binding(dynamic_request_manifest),
        requests=manifest["requests"],
        request_records_sha256=manifest["request_records_sha256"],
        rows=tuple(verified["rows"]),
    )


def infer_shard(
    *,
    receipt_path: Path,
    binding_path: Path,
    track: str,
    shard: int,
    num_shards: int,
    batch_size: int,
    max_tokens: int,
    out: Path,
    dynamic_request_manifest: Path | None = None,
    infer_public: PublicInfer | None = None,
) -> dict[str, Any]:
    track = _normalise_track(track)
    if num_shards != NUM_SHARDS or not 0 <= shard < NUM_SHARDS:
        raise RunnerError(f"formal inference requires shard 0..{NUM_SHARDS - 1}/{NUM_SHARDS}")
    expected_decoding = decoding_contract(track)
    if batch_size != BATCH_SIZE or max_tokens != expected_decoding["max_tokens"]:
        raise RunnerError(
            f"{track}: requires batch_size={BATCH_SIZE}, max_tokens={expected_decoding['max_tokens']}"
        )
    receipt_path = receipt_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    out = out.expanduser().resolve()
    binding = verify_model_binding(binding_path, deep=False)
    if binding["input_receipt"] != file_binding(receipt_path):
        raise RunnerError("model binding and input receipt differ")
    context = _get_request_context(
        receipt_path=receipt_path,
        binding_path=binding_path,
        track=track,
        dynamic_request_manifest=dynamic_request_manifest,
        verify_images=False,
    )
    if out.exists():
        summary = verify_shard(
            receipt_path=receipt_path,
            binding_path=binding_path,
            track=track,
            shard_path=out,
            expected_shard=shard,
            dynamic_request_manifest=dynamic_request_manifest,
        )
        summary["status"] = "reused"
        return summary
    selected = list(enumerate(context.rows))[shard::NUM_SHARDS]
    public_inputs = [
        {"messages": row["messages"], "images": row["images"]} for _, row in selected
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
    provenance = _prediction_context(receipt_path, binding, context)
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
                "track": track,
                "shard": shard,
                "num_shards": NUM_SHARDS,
                "request_index": index,
                "eval_id": request["eval_id"],
                "response": output[0],
                "hit_token_cap": output[1],
                **provenance,
            }
        )
    atomic_jsonl(out, records)
    summary = verify_shard(
        receipt_path=receipt_path,
        binding_path=binding_path,
        track=track,
        shard_path=out,
        expected_shard=shard,
        dynamic_request_manifest=dynamic_request_manifest,
    )
    summary["status"] = "generated"
    return summary


def verify_shard(
    *,
    receipt_path: Path,
    binding_path: Path,
    track: str,
    shard_path: Path,
    expected_shard: int,
    dynamic_request_manifest: Path | None = None,
) -> dict[str, Any]:
    track = _normalise_track(track)
    if not 0 <= expected_shard < NUM_SHARDS:
        raise RunnerError("expected shard index is invalid")
    receipt_path = receipt_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    binding = verify_model_binding(binding_path, deep=False)
    if binding["input_receipt"] != file_binding(receipt_path):
        raise RunnerError("model binding and shard input receipt differ")
    context = _get_request_context(
        receipt_path=receipt_path,
        binding_path=binding_path,
        track=track,
        dynamic_request_manifest=dynamic_request_manifest,
        verify_images=False,
    )
    records = load_jsonl(shard_path, "prediction shard", allow_empty=True)
    expected = list(enumerate(context.rows))[expected_shard::NUM_SHARDS]
    if len(records) != len(expected):
        raise RunnerError(f"{track}/shard {expected_shard}: row count drift")
    provenance = _prediction_context(receipt_path, binding, context)
    for record, (index, request) in zip(records, expected):
        if set(record) != _PREDICTION_KEYS:
            raise RunnerError(f"{track}/shard {expected_shard}: prediction schema drift")
        expected_fields = {
            "protocol": RUN_PROTOCOL,
            "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
            "track": track,
            "shard": expected_shard,
            "num_shards": NUM_SHARDS,
            "request_index": index,
            "eval_id": request["eval_id"],
            **provenance,
        }
        if any(record.get(key) != value for key, value in expected_fields.items()):
            raise RunnerError(f"{track}/shard {expected_shard}/{index}: provenance drift")
        if not isinstance(record.get("response"), str):
            raise RunnerError(f"{track}/shard {expected_shard}/{index}: response is not a string")
        if not isinstance(record.get("hit_token_cap"), bool):
            raise RunnerError(f"{track}/shard {expected_shard}/{index}: cap metadata absent")
    return {
        "track": track,
        "shard": expected_shard,
        "records": len(records),
        "sha256": file_hash(shard_path),
        "model_role": binding["role"],
    }


def merge_shards(
    *,
    receipt_path: Path,
    binding_path: Path,
    track: str,
    shard_paths: list[Path],
    out: Path,
    manifest_out: Path,
    dynamic_request_manifest: Path | None = None,
) -> dict[str, Any]:
    track = _normalise_track(track)
    if len(shard_paths) != NUM_SHARDS:
        raise RunnerError(f"expected exactly {NUM_SHARDS} prediction shard paths")
    receipt_path = receipt_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    binding = verify_model_binding(binding_path, deep=True)
    if binding["input_receipt"] != file_binding(receipt_path):
        raise RunnerError("model binding and merge input receipt differ")
    context = _get_request_context(
        receipt_path=receipt_path,
        binding_path=binding_path,
        track=track,
        dynamic_request_manifest=dynamic_request_manifest,
        verify_images=False,
    )
    summaries = [
        verify_shard(
            receipt_path=receipt_path,
            binding_path=binding_path,
            track=track,
            shard_path=path,
            expected_shard=shard,
            dynamic_request_manifest=dynamic_request_manifest,
        )
        for shard, path in enumerate(shard_paths)
    ]
    records: list[dict[str, Any]] = []
    for path in shard_paths:
        records.extend(load_jsonl(path, "prediction shard", allow_empty=True))
    records.sort(key=lambda row: row["request_index"])
    if [row["request_index"] for row in records] != list(range(len(context.rows))):
        raise RunnerError(f"{track}: merged predictions do not cover every request")
    if [row["eval_id"] for row in records] != [row["eval_id"] for row in context.rows]:
        raise RunnerError(f"{track}: merged prediction order drift")
    atomic_jsonl(out, records)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": RUN_PROTOCOL,
        "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
        "track": track,
        "input_receipt": file_binding(receipt_path),
        "model_binding": file_binding(binding_path),
        "model_role": binding["role"],
        "model_inventory_identity_sha256": binding["model"]["identity_sha256"],
        "request_manifest": context.request_manifest,
        "requests": context.requests,
        "request_records_sha256": context.request_records_sha256,
        "decoding": decoding_contract(track),
        "prediction_shards": [file_binding(path) for path in shard_paths],
        "merged_predictions": file_binding(out),
        "n_predictions": len(records),
        "one_row_per_request": len(records) == len(context.rows),
        "all_token_cap_metadata_present": all(
            isinstance(row.get("hit_token_cap"), bool) for row in records
        ),
    }
    payload["identity_sha256"] = content_hash(payload)
    atomic_json(manifest_out, payload)
    return payload


def verify_prediction_bundle(
    *,
    manifest_path: Path,
    binding_path: Path,
    predictions_path: Path,
    expected_track: str,
    dynamic_request_manifest: Path | None = None,
    deep_model: bool = True,
) -> dict[str, Any]:
    expected_track = _normalise_track(expected_track)
    manifest_path = manifest_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    predictions_path = predictions_path.expanduser().resolve()
    manifest = load_json(manifest_path, "prediction manifest")
    if set(manifest) != _PREDICTION_MANIFEST_KEYS:
        raise RunnerError("prediction manifest schema drift")
    identity = manifest.get("identity_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise RunnerError("prediction manifest identity drift")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol") != RUN_PROTOCOL
        or manifest.get("evaluation_protocol") != pair_eval.EVALUATION_PROTOCOL
        or manifest.get("track") != expected_track
        or manifest.get("one_row_per_request") is not True
        or manifest.get("all_token_cap_metadata_present") is not True
        or manifest.get("decoding") != decoding_contract(expected_track)
    ):
        raise RunnerError("prediction manifest contract drift")
    if manifest.get("model_binding") != file_binding(binding_path):
        raise RunnerError("prediction manifest model binding drift")
    binding = verify_model_binding(binding_path, deep=deep_model)
    if (
        manifest.get("model_role") != binding["role"]
        or manifest.get("model_inventory_identity_sha256")
        != binding["model"]["identity_sha256"]
    ):
        raise RunnerError("prediction manifest model identity drift")
    receipt_path = _verify_file_binding(manifest.get("input_receipt"), deep=True)
    if binding["input_receipt"] != manifest["input_receipt"]:
        raise RunnerError("prediction manifest/model receipt drift")
    if expected_track == pair_eval.TRACK_S_PRED and dynamic_request_manifest is None:
        dynamic_request_manifest = Path(str((manifest.get("request_manifest") or {}).get("path")))
    context = _get_request_context(
        receipt_path=receipt_path,
        binding_path=binding_path,
        track=expected_track,
        dynamic_request_manifest=dynamic_request_manifest,
        verify_images=False,
    )
    if (
        manifest.get("request_manifest") != context.request_manifest
        or manifest.get("requests") != context.requests
        or manifest.get("request_records_sha256") != context.request_records_sha256
        or manifest.get("n_predictions") != len(context.rows)
    ):
        raise RunnerError("prediction manifest request binding drift")
    if manifest.get("merged_predictions") != file_binding(predictions_path):
        raise RunnerError("prediction manifest merged file drift")
    rows = load_jsonl(predictions_path, "merged predictions", allow_empty=True)
    if len(rows) != len(context.rows):
        raise RunnerError("merged prediction count drift")
    provenance = _prediction_context(receipt_path, binding, context)
    for index, (row, request) in enumerate(zip(rows, context.rows)):
        if set(row) != _PREDICTION_KEYS:
            raise RunnerError(f"merged prediction {index}: schema drift")
        expected = {
            "protocol": RUN_PROTOCOL,
            "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
            "track": expected_track,
            "shard": index % NUM_SHARDS,
            "num_shards": NUM_SHARDS,
            "request_index": index,
            "eval_id": request["eval_id"],
            **provenance,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise RunnerError(f"merged prediction {index}: provenance drift")
        if not isinstance(row.get("response"), str) or not isinstance(
            row.get("hit_token_cap"), bool
        ):
            raise RunnerError(f"merged prediction {index}: response metadata drift")
    shard_bindings = manifest.get("prediction_shards")
    if not isinstance(shard_bindings, list) or len(shard_bindings) != NUM_SHARDS:
        raise RunnerError("prediction manifest shard coverage drift")
    shard_rows: list[dict[str, Any]] = []
    for shard, shard_binding in enumerate(shard_bindings):
        path = _verify_file_binding(shard_binding, deep=True)
        verify_shard(
            receipt_path=receipt_path,
            binding_path=binding_path,
            track=expected_track,
            shard_path=path,
            expected_shard=shard,
            dynamic_request_manifest=dynamic_request_manifest,
        )
        shard_rows.extend(load_jsonl(path, "prediction shard", allow_empty=True))
    shard_rows.sort(key=lambda row: row["request_index"])
    if shard_rows != rows:
        raise RunnerError("merged predictions differ from verified shards")
    verified_provenance: dict[str, Any] = {
        "verified": True,
        "protocol": RUN_PROTOCOL,
        "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
        "track": expected_track,
        "prediction_manifest": file_binding(manifest_path),
        "prediction_manifest_identity_sha256": identity,
        "merged_predictions": manifest["merged_predictions"],
        "prediction_shards": shard_bindings,
        "model_binding": manifest["model_binding"],
        "model_binding_identity_sha256": binding["identity_sha256"],
        "model_inventory_identity_sha256": binding["model"]["identity_sha256"],
        "input_receipt": manifest["input_receipt"],
        "request_manifest": context.request_manifest,
        "requests": context.requests,
        "request_records_sha256": context.request_records_sha256,
        "decoding": manifest["decoding"],
        "n_predictions": len(rows),
    }
    verified_provenance["identity_sha256"] = content_hash(verified_provenance)
    return {
        "manifest": manifest,
        "rows": rows,
        "requests": list(context.rows),
        "provenance": verified_provenance,
        "receipt": verify_input_receipt(receipt_path),
    }


def build_predicted_request_release(
    *,
    binding_path: Path,
    g_manifest_path: Path,
    g_predictions_path: Path,
    out: Path,
    manifest_out: Path,
) -> dict[str, Any]:
    binding_path = binding_path.expanduser().resolve()
    g_manifest_path = g_manifest_path.expanduser().resolve()
    g_predictions_path = g_predictions_path.expanduser().resolve()
    out = out.expanduser().resolve()
    manifest_out = manifest_out.expanduser().resolve()
    if out.exists() or manifest_out.exists():
        if not out.is_file() or not manifest_out.is_file():
            raise RunnerError("partial S-pred request release already exists")
        verified = verify_predicted_request_release(
            manifest_path=manifest_out,
            binding_path=binding_path,
            g_manifest_path=g_manifest_path,
            g_predictions_path=g_predictions_path,
            verify_images=True,
        )
        return {**verified["manifest"], "build_status": "reused_verified"}
    g_bundle = verify_prediction_bundle(
        manifest_path=g_manifest_path,
        binding_path=binding_path,
        predictions_path=g_predictions_path,
        expected_track=pair_eval.TRACK_G,
        deep_model=True,
    )
    if any(row["hit_token_cap"] for row in g_bundle["rows"]):
        raise RunnerError("G output hit token cap; refusing an incomplete S-pred release")
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for request, prediction in zip(g_bundle["requests"], g_bundle["rows"]):
        parsed = pair_eval.parse_grounding_response(prediction["response"])
        if not parsed.valid:
            invalid.append({"eval_id": request["eval_id"], "status": parsed.status})
            continue
        rows.append(pair_eval.format_predicted_s_request(request, parsed))
    atomic_jsonl(out, rows)
    binding = verify_model_binding(binding_path, deep=False)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PREDICTED_REQUEST_PROTOCOL,
        "evaluation_protocol": pair_eval.EVALUATION_PROTOCOL,
        "track": pair_eval.TRACK_S_PRED,
        "input_receipt": binding["input_receipt"],
        "model_binding": file_binding(binding_path),
        "model_binding_identity_sha256": binding["identity_sha256"],
        "g_prediction_manifest": file_binding(g_manifest_path),
        "g_predictions": file_binding(g_predictions_path),
        "requests": file_binding(out),
        "request_records_sha256": pair_eval.canonical_hash(rows),
        "source_records": pair_eval.EXPECTED_RECORDS,
        "valid_grounding_records": len(rows),
        "invalid_groundings": invalid,
        "complete_g_coverage": len(g_bundle["rows"]) == pair_eval.EXPECTED_RECORDS,
        "training_materialization_eligible": False,
    }
    payload["identity_sha256"] = content_hash(payload)
    atomic_json(manifest_out, payload)
    return {**payload, "build_status": "generated"}


def verify_predicted_request_release(
    *,
    manifest_path: Path,
    binding_path: Path,
    g_manifest_path: Path | None = None,
    g_predictions_path: Path | None = None,
    verify_images: bool,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    manifest = load_json(manifest_path, "S-pred request manifest")
    if set(manifest) != _PREDICTED_REQUEST_MANIFEST_KEYS:
        raise RunnerError("S-pred request manifest schema drift")
    identity = manifest.get("identity_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    if identity != content_hash(unsigned):
        raise RunnerError("S-pred request manifest identity drift")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol") != PREDICTED_REQUEST_PROTOCOL
        or manifest.get("evaluation_protocol") != pair_eval.EVALUATION_PROTOCOL
        or manifest.get("track") != pair_eval.TRACK_S_PRED
        or manifest.get("source_records") != pair_eval.EXPECTED_RECORDS
        or manifest.get("complete_g_coverage") is not True
        or manifest.get("training_materialization_eligible") is not False
    ):
        raise RunnerError("S-pred request release contract drift")
    if manifest.get("model_binding") != file_binding(binding_path):
        raise RunnerError("S-pred request model binding drift")
    binding = verify_model_binding(binding_path, deep=False)
    if (
        manifest.get("model_binding_identity_sha256") != binding["identity_sha256"]
        or manifest.get("input_receipt") != binding["input_receipt"]
    ):
        raise RunnerError("S-pred request model/receipt identity drift")
    bound_g_manifest = _verify_file_binding(manifest.get("g_prediction_manifest"), deep=True)
    bound_g_predictions = _verify_file_binding(manifest.get("g_predictions"), deep=True)
    if g_manifest_path is not None and bound_g_manifest != g_manifest_path.expanduser().resolve():
        raise RunnerError("S-pred request G manifest argument drift")
    if g_predictions_path is not None and bound_g_predictions != g_predictions_path.expanduser().resolve():
        raise RunnerError("S-pred request G predictions argument drift")
    g_bundle = verify_prediction_bundle(
        manifest_path=bound_g_manifest,
        binding_path=binding_path,
        predictions_path=bound_g_predictions,
        expected_track=pair_eval.TRACK_G,
        deep_model=False,
    )
    if any(row["hit_token_cap"] for row in g_bundle["rows"]):
        raise RunnerError("S-pred request source G output hit token cap")
    expected_rows: list[dict[str, Any]] = []
    expected_invalid: list[dict[str, str]] = []
    for request, prediction in zip(g_bundle["requests"], g_bundle["rows"]):
        parsed = pair_eval.parse_grounding_response(prediction["response"])
        if parsed.valid:
            expected_rows.append(pair_eval.format_predicted_s_request(request, parsed))
        else:
            expected_invalid.append({"eval_id": request["eval_id"], "status": parsed.status})
    request_path = _verify_file_binding(manifest.get("requests"), deep=True)
    rows = load_jsonl(request_path, "S-pred requests", allow_empty=True)
    if rows != expected_rows:
        raise RunnerError("S-pred requests do not exactly derive from verified G outputs")
    if (
        manifest.get("valid_grounding_records") != len(rows)
        or manifest.get("invalid_groundings") != expected_invalid
        or manifest.get("request_records_sha256") != pair_eval.canonical_hash(rows)
    ):
        raise RunnerError("S-pred request coverage or canonical hash drift")
    for row in rows:
        try:
            pair_eval.validate_public_request(row, expected_track=pair_eval.TRACK_S_PRED)
        except pair_eval.ContractError as exc:
            raise RunnerError(f"invalid S-pred public request: {exc}") from exc
        if verify_images:
            image = Path(row["images"][0])
            if file_hash(image) != row["image_sha256"]:
                raise RunnerError(f"S-pred request image hash drift: {row['eval_id']}")
    return {
        "manifest": manifest,
        "rows": rows,
        "invalid_groundings": expected_invalid,
        "g_bundle": g_bundle,
    }


def dry_run() -> dict[str, Any]:
    release = pair_eval.validate_request_release(
        pair_eval.DEFAULT_RELEASE / "release.manifest.json", verify_images=True
    )
    shard_counts = {
        track: [len(list(enumerate(rows))[shard::NUM_SHARDS]) for shard in range(NUM_SHARDS)]
        for track, rows in release.requests.items()
    }
    passed = (
        len(release.source.pairs) == pair_eval.EXPECTED_RECORDS
        and all(counts == [10, 10, 10, 10] for counts in shard_counts.values())
        and pair_eval.parse_grounding_response(
            '{"A":{"bbox_2d":[1,2,100,200]},"B":{"bbox_2d":[300,4,900,800]}}'
        ).valid
    )
    return {
        "status": "DLC_DRY_RUN_PASS" if passed else "DLC_DRY_RUN_FAIL",
        "synthetic_or_contract_fixture_only": True,
        "records": len(release.source.pairs),
        "static_shard_counts": shard_counts,
        "logical_shards": NUM_SHARDS,
        "training_materialization_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--release-manifest", type=Path, required=True)
    preflight.add_argument("--out", type=Path, required=True)

    bind = sub.add_parser("bind-model")
    bind.add_argument("--role", choices=("base", "evaluated"), required=True)
    bind.add_argument("--model", type=Path, required=True)
    bind.add_argument("--input-receipt", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)

    verify_model = sub.add_parser("verify-model-binding")
    verify_model.add_argument("--binding", type=Path, required=True)
    verify_model.add_argument("--deep", action="store_true")

    infer = sub.add_parser("infer-shard")
    infer.add_argument("--input-receipt", type=Path, required=True)
    infer.add_argument("--model-binding", type=Path, required=True)
    infer.add_argument("--track", choices=sorted(TRACK_ALIASES), required=True)
    infer.add_argument("--dynamic-request-manifest", type=Path)
    infer.add_argument("--shard", type=int, required=True)
    infer.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    infer.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    infer.add_argument("--max-tokens", type=int, required=True)
    infer.add_argument("--out", type=Path, required=True)

    verify = sub.add_parser("verify-shard")
    verify.add_argument("--input-receipt", type=Path, required=True)
    verify.add_argument("--model-binding", type=Path, required=True)
    verify.add_argument("--track", choices=sorted(TRACK_ALIASES), required=True)
    verify.add_argument("--dynamic-request-manifest", type=Path)
    verify.add_argument("--shard", type=Path, required=True)
    verify.add_argument("--expected-shard", type=int, required=True)

    merge = sub.add_parser("merge-shards")
    merge.add_argument("--input-receipt", type=Path, required=True)
    merge.add_argument("--model-binding", type=Path, required=True)
    merge.add_argument("--track", choices=sorted(TRACK_ALIASES), required=True)
    merge.add_argument("--dynamic-request-manifest", type=Path)
    merge.add_argument("--predictions", nargs="+", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("--manifest-out", type=Path, required=True)

    build_pred = sub.add_parser("build-s-pred-requests")
    build_pred.add_argument("--model-binding", type=Path, required=True)
    build_pred.add_argument("--g-manifest", type=Path, required=True)
    build_pred.add_argument("--g-predictions", type=Path, required=True)
    build_pred.add_argument("--out", type=Path, required=True)
    build_pred.add_argument("--manifest-out", type=Path, required=True)

    verify_pred = sub.add_parser("verify-s-pred-requests")
    verify_pred.add_argument("--model-binding", type=Path, required=True)
    verify_pred.add_argument("--manifest", type=Path, required=True)
    verify_pred.add_argument("--g-manifest", type=Path)
    verify_pred.add_argument("--g-predictions", type=Path)

    sub.add_parser("dry-run")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            payload = build_input_receipt(args.release_manifest)
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
                track=args.track,
                shard=args.shard,
                num_shards=args.num_shards,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                out=args.out,
                dynamic_request_manifest=args.dynamic_request_manifest,
            )
        elif args.command == "verify-shard":
            payload = verify_shard(
                receipt_path=args.input_receipt,
                binding_path=args.model_binding,
                track=args.track,
                shard_path=args.shard,
                expected_shard=args.expected_shard,
                dynamic_request_manifest=args.dynamic_request_manifest,
            )
        elif args.command == "merge-shards":
            payload = merge_shards(
                receipt_path=args.input_receipt,
                binding_path=args.model_binding,
                track=args.track,
                shard_paths=args.predictions,
                out=args.out,
                manifest_out=args.manifest_out,
                dynamic_request_manifest=args.dynamic_request_manifest,
            )
        elif args.command == "build-s-pred-requests":
            payload = build_predicted_request_release(
                binding_path=args.model_binding,
                g_manifest_path=args.g_manifest,
                g_predictions_path=args.g_predictions,
                out=args.out,
                manifest_out=args.manifest_out,
            )
        elif args.command == "verify-s-pred-requests":
            payload = verify_predicted_request_release(
                manifest_path=args.manifest,
                binding_path=args.model_binding,
                g_manifest_path=args.g_manifest,
                g_predictions_path=args.g_predictions,
                verify_images=True,
            )["manifest"]
        else:
            payload = dry_run()
    except (RunnerError, pair_eval.ContractError) as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("status") != "DLC_DRY_RUN_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
