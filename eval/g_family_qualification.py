#!/usr/bin/env python3
"""Independent qualification pilot for the scoped schema-v3 G specialist.

The 32 records cover bbox, detection, mask, and point on eight image-disjoint
qualification images.  The v2 trainer sees only the separate train/dev splits.
Eight images are still too small for a publication claim, so this module produces
only a directional engineering decision; it never emits a global teacher-eligibility
PASS or authorizes joint/MOPD training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.gsa_eval.parsing import parse_boxes, parse_json_items, parse_points  # noqa: E402


PROTOCOL = "g-family-specialist-qualification-pilot-v2"
RELEASE_PROTOCOL = "semantic-audit-v3-g-family-scoped-v2-independent-qualification"
SOURCE_AUDIT_PROTOCOL = "semantic-audit-v3-g-family-scoped-v1"
CORPUS_VERSION = "gsa-bench-v3-g-family-scoped-v2"
G_FORMS = ("bbox", "detection", "mask", "point")
EXPECTED_COUNTS = {"train": 84, "dev": 32, "qualification": 32}
EXPECTED_UNIQUE_IMAGES = {"train": 24, "dev": 8, "qualification": 8}
EXPECTED_EVAL_PER_FORM = 8
MIN_PUBLICATION_IMAGES = 30
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260820
ALLOWED_WORLD_SIZES = (4, 8)
ALLOWED_TRAINING_SEEDS = (42, 43, 44)
FROZEN_EFFECTIVE_BATCH = 8
FROZEN_EPOCHS = 5
FROZEN_MAX_STEPS = math.ceil(EXPECTED_COUNTS["train"] * FROZEN_EPOCHS / FROZEN_EFFECTIVE_BATCH)
FROZEN_NUM_SHARDS = 8
_CHECKPOINT_RE = re.compile(r"^checkpoint-([1-9][0-9]*)$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path, *, hash_bytes: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    binding: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
    if hash_bytes:
        binding["sha256"] = file_hash(path)
    return binding


def load_json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label}: {path} ({exc.msg})") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != payload:
            raise ValueError(f"refusing to overwrite a different bound artifact: {path}")
        return
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def same_path(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).expanduser().resolve(strict=False) == expected.resolve(strict=False)


def resolve_checkpoint(source: Path) -> Path:
    """Resolve the unique highest numeric checkpoint beneath ``source``."""
    source = source.expanduser().resolve()
    direct = _CHECKPOINT_RE.match(source.name)
    if direct and (source / "adapter_config.json").is_file():
        candidates = [source]
    else:
        if not source.is_dir():
            raise ValueError(f"adapter source is not a directory: {source}")
        candidates = []
        for config in source.rglob("adapter_config.json"):
            match = _CHECKPOINT_RE.match(config.parent.name)
            if match:
                candidates.append(config.parent.resolve())
    usable: list[tuple[int, Path]] = []
    for candidate in sorted(set(candidates)):
        match = _CHECKPOINT_RE.match(candidate.name)
        if match and (candidate / "adapter_model.safetensors").is_file():
            usable.append((int(match.group(1)), candidate))
    if not usable:
        raise ValueError(f"no usable numeric LoRA checkpoint under {source}")
    highest = max(step for step, _ in usable)
    winners = [path for step, path in usable if step == highest]
    if len(winners) != 1:
        joined = ", ".join(str(path) for path in winners)
        raise ValueError(f"ambiguous latest checkpoint-{highest}: {joined}")
    return winners[0]


def checkpoint_binding(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    match = _CHECKPOINT_RE.match(checkpoint.name)
    if not checkpoint.is_dir() or not match:
        raise ValueError(f"checkpoint must be a numeric checkpoint-N directory: {checkpoint}")
    config_path = checkpoint / "adapter_config.json"
    weights_path = checkpoint / "adapter_model.safetensors"
    config = load_json(config_path, "adapter config")
    binding: dict[str, Any] = {
        "path": str(checkpoint),
        "step": int(match.group(1)),
        "adapter_config": file_binding(config_path),
        "adapter_weights": file_binding(weights_path),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
    }
    trainer_state = checkpoint / "trainer_state.json"
    if trainer_state.is_file():
        binding["trainer_state"] = file_binding(trainer_state)
    binding["identity_sha256"] = content_hash(binding)
    return binding


def _find_training_receipt(checkpoint: Path) -> Path:
    current = checkpoint.resolve()
    for _ in range(6):
        candidate = current / "g_family_qualification_inputs_v2.json"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    raise ValueError(
        "selected G checkpoint is not descended from g_family_qualification_inputs_v2.json"
    )


def _training_args_path(checkpoint: Path) -> Path:
    args_path = checkpoint.resolve().parent / "args.json"
    if not args_path.is_file():
        raise ValueError(f"selected G checkpoint has no sibling args.json: {args_path}")
    return args_path


def _corpus_file(manifest_path: Path, manifest: dict[str, Any], key: str) -> Path:
    spec = (manifest.get("corpus_files") or {}).get(key)
    if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
        raise ValueError(f"manifest is missing corpus_files.{key}")
    path = (manifest_path.parent / spec["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"manifest-bound corpus file is missing: {path}")
    if file_hash(path) != spec.get("sha256"):
        raise ValueError(f"corpus file hash drift: {key}")
    rows = sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
    if rows != spec.get("records"):
        raise ValueError(f"corpus row-count drift: {key}")
    return path


def _validate_rows(
    rows: list[dict[str, Any]], split: str, expected: int
) -> tuple[set[str], Counter[str]]:
    if len(rows) != expected:
        raise ValueError(f"{split} rows must be {expected}, found {len(rows)}")
    sample_ids: set[str] = set()
    image_hashes: set[str] = set()
    forms: Counter[str] = Counter()
    image_forms: dict[str, list[str]] = defaultdict(list)
    image_hash_cache: dict[Path, str] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        form = row.get("gsa_form")
        meta = row.get("gsa_meta") or {}
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{split} contains a row without sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate {split} sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if form not in G_FORMS or row.get("gsa_family") != "G":
            raise ValueError(f"{sample_id}: row is outside the G family")
        if row.get("gsa_contract_version") != "gsa-contract-v2":
            raise ValueError(f"{sample_id}: unexpected GSA contract")
        if meta.get("split") != split or meta.get("image_sha256") is None:
            raise ValueError(f"{sample_id}: split/image binding is missing")
        if meta.get("release_protocol") != RELEASE_PROTOCOL:
            raise ValueError(f"{sample_id}: release protocol drift")
        if meta.get("authorization") != "G_specialist_qualification_only":
            raise ValueError(f"{sample_id}: authorization drift")
        if meta.get("semantic_audit_global_verdict") != "BLOCKED":
            raise ValueError(f"{sample_id}: global BLOCKED state was not preserved")
        images = row.get("images") or []
        if len(images) != 1:
            raise ValueError(f"{sample_id}: expected exactly one image")
        image_path = Path(str(images[0])).expanduser().resolve()
        if not image_path.is_file():
            raise ValueError(f"{sample_id}: image is missing: {image_path}")
        observed_hash = image_hash_cache.setdefault(image_path, file_hash(image_path))
        if observed_hash != meta.get("image_sha256"):
            raise ValueError(f"{sample_id}: image hash drift")
        messages = row.get("messages") or []
        if len(messages) < 3 or messages[-1].get("role") != "assistant":
            raise ValueError(f"{sample_id}: messages do not contain an assistant target")
        try:
            target = json.loads(messages[-1].get("content", ""))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{sample_id}: assistant target is not JSON") from exc
        if canonical_json(target) != canonical_json(row.get("gsa_payload")):
            raise ValueError(f"{sample_id}: assistant target and gsa_payload differ")
        image_hash = str(meta["image_sha256"])
        image_hashes.add(image_hash)
        image_forms[image_hash].append(form)
        forms[form] += 1
    if split in {"dev", "qualification"}:
        for image_hash, present in image_forms.items():
            if sorted(present) != sorted(G_FORMS):
                raise ValueError(
                    f"{split} image {image_hash} must have exactly one row per G form"
                )
    return image_hashes, forms


def _check_manifest_audit_binding(
    manifest: dict[str, Any], name: str, path: Path, observed_hash: str
) -> None:
    spec = (manifest.get("audit_bindings") or {}).get(name)
    if not isinstance(spec, dict):
        raise ValueError(f"manifest is missing audit_bindings.{name}")
    if not same_path(spec.get("path"), path) or spec.get("sha256") != observed_hash:
        raise ValueError(f"manifest audit binding drift: {name}")


def validate_release_inputs(
    *,
    manifest_path: Path,
    scoped_report_path: Path,
    full_report_path: Path,
    base_model: Path,
    checkpoint: Path,
    source: Path,
    joint_checkpoint: Path | None = None,
    eval_world_size: int = 8,
    batch_size: int = 4,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    scoped_report_path = scoped_report_path.expanduser().resolve()
    full_report_path = full_report_path.expanduser().resolve()
    base_model = base_model.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    source = source.expanduser().resolve()

    if resolve_checkpoint(source) != checkpoint:
        raise ValueError("selected G checkpoint is not the unique latest numeric checkpoint")
    manifest = load_json(manifest_path, "G-only corpus manifest")
    if manifest.get("protocol") != RELEASE_PROTOCOL:
        raise ValueError("manifest release protocol drift")
    if manifest.get("corpus_version") != CORPUS_VERSION:
        raise ValueError("manifest corpus version drift")
    if manifest.get("g_family_scoped_release_verdict") != "G_ONLY_PASS":
        raise ValueError("G-only scoped audit is not released")
    if manifest.get("global_audit_verdict") != "BLOCKED":
        raise ValueError("global audit BLOCKED state was not preserved")
    if manifest.get("authorization") != "G_specialist_qualification_only":
        raise ValueError("manifest authorization drift")
    if manifest.get("counts") != EXPECTED_COUNTS:
        raise ValueError("manifest must bind the frozen 84/32/32 split")
    if manifest.get("unique_images") != EXPECTED_UNIQUE_IMAGES:
        raise ValueError("manifest must bind the frozen 24/8/8 image split")
    if set(manifest.get("forms") or []) != set(G_FORMS):
        raise ValueError("manifest G forms drift")
    if any((manifest.get("overlap") or {}).get(key) != 0 for key in (
        "train_dev", "train_qualification", "dev_qualification"
    )):
        raise ValueError("manifest reports cross-split image overlap")
    exposure = manifest.get("exposure_policy") or {}
    if exposure.get("qualification_seen_by_training_launcher") is not False:
        raise ValueError("manifest does not freeze qualification away from training")

    train_path = _corpus_file(manifest_path, manifest, "train_G_only")
    dev_path = _corpus_file(manifest_path, manifest, "dev_G_only")
    qualification_path = _corpus_file(manifest_path, manifest, "qualification_G_only")
    train_rows = load_jsonl(train_path, "G train split")
    dev_rows = load_jsonl(dev_path, "G dev split")
    qualification_rows = load_jsonl(qualification_path, "G qualification split")
    train_images, train_forms = _validate_rows(train_rows, "train", EXPECTED_COUNTS["train"])
    dev_images, dev_forms = _validate_rows(dev_rows, "dev", EXPECTED_COUNTS["dev"])
    qualification_images, qualification_forms = _validate_rows(
        qualification_rows, "qualification", EXPECTED_COUNTS["qualification"]
    )
    if train_images & dev_images or train_images & qualification_images or dev_images & qualification_images:
        raise ValueError("observed cross-split image hash overlap")
    if len(train_images) != EXPECTED_UNIQUE_IMAGES["train"]:
        raise ValueError("observed train image count drift")
    if len(dev_images) != EXPECTED_UNIQUE_IMAGES["dev"]:
        raise ValueError("observed dev image count drift")
    if len(qualification_images) != EXPECTED_UNIQUE_IMAGES["qualification"]:
        raise ValueError("observed qualification image count drift")
    if any(dev_forms[form] != EXPECTED_EVAL_PER_FORM for form in G_FORMS):
        raise ValueError("dev split must contain eight rows per G form")
    if any(qualification_forms[form] != EXPECTED_EVAL_PER_FORM for form in G_FORMS):
        raise ValueError("qualification split must contain eight rows per G form")
    manifest_dev_images = set((manifest.get("image_split") or {}).get("dev_image_sha256") or [])
    manifest_qualification_images = set(
        (manifest.get("image_split") or {}).get("qualification_image_sha256") or []
    )
    manifest_train_images = set((manifest.get("image_split") or {}).get("train_image_sha256") or [])
    if (manifest_dev_images != dev_images or manifest_qualification_images != qualification_images
            or manifest_train_images != train_images):
        raise ValueError("manifest image lists do not replay from corpus rows")
    if manifest.get("by_form", {}).get("dev") != dict(dev_forms):
        raise ValueError("manifest dev form counts drift")
    if manifest.get("by_form", {}).get("qualification") != dict(qualification_forms):
        raise ValueError("manifest qualification form counts drift")
    if manifest.get("by_form", {}).get("train") != dict(train_forms):
        raise ValueError("manifest training form counts drift")

    scoped = load_json(scoped_report_path, "G-only scoped report")
    full = load_json(full_report_path, "full schema-v3 audit report")
    scoped_hash = file_hash(scoped_report_path)
    full_hash = file_hash(full_report_path)
    if scoped.get("protocol") != SOURCE_AUDIT_PROTOCOL or scoped.get("verdict") != "G_ONLY_PASS":
        raise ValueError("scoped report is not the released G-only audit")
    if scoped.get("global_audit_verdict") != "BLOCKED":
        raise ValueError("scoped report did not preserve global BLOCKED")
    if full.get("verdict") != "BLOCKED":
        raise ValueError("full schema-v3 audit no longer matches the scoped release premise")
    _check_manifest_audit_binding(manifest, "scoped_report", scoped_report_path, scoped_hash)
    _check_manifest_audit_binding(manifest, "full_compiler_report", full_report_path, full_hash)

    config_path = base_model / "config.json"
    if not config_path.is_file():
        raise ValueError(f"base model config is missing: {config_path}")
    selected_binding = checkpoint_binding(checkpoint)
    training_receipt_path = _find_training_receipt(checkpoint)
    training_receipt = load_json(training_receipt_path, "G qualification training receipt")
    training_args_path = _training_args_path(checkpoint)
    training_args = load_json(training_args_path, "G trainer args")
    if training_receipt.get("protocol") != RELEASE_PROTOCOL:
        raise ValueError("training receipt protocol drift")
    if training_receipt.get("authorization") != "G_specialist_qualification_only":
        raise ValueError("training receipt authorization drift")
    if training_receipt.get("global_audit_verdict") != "BLOCKED":
        raise ValueError("training receipt did not preserve global BLOCKED")
    if not same_path((training_receipt.get("manifest") or {}).get("path"), manifest_path):
        raise ValueError("training receipt is bound to a different manifest path")
    if (training_receipt.get("manifest") or {}).get("sha256") != file_hash(manifest_path):
        raise ValueError("training receipt manifest hash drift")
    if not same_path((training_receipt.get("scoped_report") or {}).get("path"), scoped_report_path):
        raise ValueError("training receipt is bound to a different scoped report")
    if (training_receipt.get("scoped_report") or {}).get("sha256") != scoped_hash:
        raise ValueError("training receipt scoped report hash drift")
    if not same_path((training_receipt.get("full_report") or {}).get("path"), full_report_path):
        raise ValueError("training receipt is bound to a different full audit report")
    if (training_receipt.get("full_report") or {}).get("sha256") != full_hash:
        raise ValueError("training receipt full audit report hash drift")
    if not same_path((training_receipt.get("model_config") or {}).get("path"), config_path):
        raise ValueError("training receipt is bound to a different base model")
    if (training_receipt.get("model_config") or {}).get("sha256") != file_hash(config_path):
        raise ValueError("training receipt base model config hash drift")
    if (training_receipt.get("train_rows") != 84
            or training_receipt.get("dev_rows") != 32
            or training_receipt.get("qualification_rows") != 32):
        raise ValueError("training receipt row counts drift")
    hardware = training_receipt.get("hardware")
    hardware_world_sizes = {"4xL20": 4, "8xL20": 8}
    if hardware not in hardware_world_sizes:
        raise ValueError("training receipt hardware drift")
    training_world_size = hardware_world_sizes[hardware]
    receipt_world_size = training_receipt.get("world_size", training_world_size)
    if receipt_world_size != training_world_size:
        raise ValueError("training receipt world size drift")
    per_device_batch = training_receipt.get("per_device_batch_size")
    grad_accum = training_receipt.get("gradient_accumulation_steps")
    effective_batch = training_receipt.get("effective_batch_size")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
               for value in (per_device_batch, grad_accum, effective_batch)):
        raise ValueError("training receipt batch configuration is invalid")
    if (training_world_size * per_device_batch * grad_accum != effective_batch
            or effective_batch != FROZEN_EFFECTIVE_BATCH):
        raise ValueError("training receipt effective batch drift")
    if training_receipt.get("epochs_for_step_calculation") != FROZEN_EPOCHS:
        raise ValueError("training receipt epoch calculation drift")
    if training_receipt.get("max_steps") != FROZEN_MAX_STEPS:
        raise ValueError("training receipt frozen max steps drift")
    args_world_size = training_args.get("local_world_size")
    if args_world_size != training_world_size:
        raise ValueError("trainer args world size differs from the training receipt")
    args_batch = training_args.get("per_device_train_batch_size")
    args_grad_accum = training_args.get("gradient_accumulation_steps")
    if args_batch != per_device_batch or args_grad_accum != grad_accum:
        raise ValueError("trainer args batch configuration differs from the training receipt")
    if training_args.get("max_steps") != FROZEN_MAX_STEPS:
        raise ValueError("trainer args max steps drift")
    args_seed = training_args.get("seed")
    args_data_seed = training_args.get("data_seed")
    if args_seed not in ALLOWED_TRAINING_SEEDS or args_data_seed != args_seed:
        raise ValueError("trainer args seed/data_seed drift")
    receipt_seed = training_receipt.get("seed")
    if receipt_seed is not None and receipt_seed != args_seed:
        raise ValueError("trainer args seed differs from the training receipt")
    if training_world_size == 4 and training_receipt.get("data_seed") != args_seed:
        raise ValueError("4xL20 training receipt must bind matching seed and data_seed")
    seed = args_seed
    if seed not in ALLOWED_TRAINING_SEEDS:
        raise ValueError("training receipt seed must be one of 42, 43, or 44")
    args_zero_stage = ((training_args.get("deepspeed") or {}).get("zero_optimization") or {}).get("stage")
    if args_zero_stage != 2:
        raise ValueError("trainer args must bind DeepSpeed ZeRO-2")
    if training_world_size == 4:
        if training_receipt.get("deepspeed") != "zero2":
            raise ValueError("4xL20 training receipt must bind DeepSpeed ZeRO-2")
        if per_device_batch != 1 or grad_accum != 2:
            raise ValueError("4xL20 matched training requires per-device batch 1 and GA 2")
        if training_receipt.get("wandb_mode") != "offline":
            raise ValueError("4xL20 training receipt must bind offline experiment tracking")
        if training_receipt.get("dataloader_num_workers") != training_args.get("dataloader_num_workers"):
            raise ValueError("4xL20 dataloader worker receipt/args drift")
        if training_receipt.get("dataset_num_proc") != training_args.get("dataset_num_proc"):
            raise ValueError("4xL20 dataset worker receipt/args drift")
    if training_args.get("max_length") != 3072:
        raise ValueError("trainer args max_length drift")
    learning_rate = training_args.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or not math.isclose(float(learning_rate), 1e-5):
        raise ValueError("trainer args learning rate drift")
    args_train = {Path(value).expanduser().resolve(strict=False)
                  for value in training_args.get("dataset") or []}
    args_dev = {Path(value).expanduser().resolve(strict=False)
                for value in training_args.get("val_dataset") or []}
    if args_train != {train_path} or args_dev != {dev_path}:
        raise ValueError("trainer args dataset paths are not exactly train/dev")
    if batch_size != 4:
        raise ValueError("frozen qualification evaluation requires batch_size=4")
    if eval_world_size not in ALLOWED_WORLD_SIZES:
        raise ValueError("evaluation world size must be exactly 4 or 8")
    if selected_binding["step"] != training_receipt.get("max_steps"):
        raise ValueError("selected G checkpoint is not the completed frozen max step")
    trainer_state_binding = selected_binding.get("trainer_state")
    if not isinstance(trainer_state_binding, dict):
        raise ValueError("selected G checkpoint has no trainer_state.json")
    trainer_state = load_json(Path(trainer_state_binding["path"]), "G trainer state")
    if int(trainer_state.get("global_step", -1)) != selected_binding["step"]:
        raise ValueError("G trainer state global_step does not match checkpoint-N")
    if training_receipt.get("qualification_exposed_to_trainer") is not False:
        raise ValueError("training receipt exposed qualification to the trainer")
    trainer_paths = {
        Path(value).expanduser().resolve(strict=False)
        for value in training_receipt.get("trainer_dataset_paths") or []
    }
    if trainer_paths != {train_path, dev_path} or qualification_path in trainer_paths:
        raise ValueError("training receipt dataset paths are not exactly train/dev")
    for name, path in (("train", train_path), ("dev", dev_path)):
        spec = training_receipt.get(name) or {}
        if not same_path(spec.get("path"), path) or spec.get("sha256") != file_hash(path):
            raise ValueError(f"training receipt {name} binding drift")
    if (training_receipt.get("qualification") or {}).get("sha256") != file_hash(qualification_path):
        raise ValueError("training receipt qualification binding drift")
    if not same_path((training_receipt.get("qualification") or {}).get("path"), qualification_path):
        raise ValueError("training receipt qualification path drift")

    adapter_base = selected_binding.get("base_model_name_or_path")
    if isinstance(adapter_base, str) and Path(adapter_base).is_absolute():
        if Path(adapter_base).resolve(strict=False) != base_model:
            raise ValueError("G adapter declares a different base model")

    joint_binding = checkpoint_binding(joint_checkpoint) if joint_checkpoint else None
    if joint_binding:
        joint_base = joint_binding.get("base_model_name_or_path")
        if isinstance(joint_base, str) and Path(joint_base).is_absolute():
            if Path(joint_base).resolve(strict=False) != base_model:
                raise ValueError("joint reference adapter declares a different base model")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "result_label": "G-specialist qualification pilot",
        "scope": "G_family_bbox_detection_mask_point_only",
        "authorization": "evaluation_only",
        "publication_claim_eligible": False,
        "teacher_authorization": "NONE",
        "joint_training_authorized": False,
        "mopd_training_authorized": False,
        "global_semantic_audit_verdict": "BLOCKED",
        "data": {
            "manifest": file_binding(manifest_path),
            "train": file_binding(train_path),
            "dev": file_binding(dev_path),
            "qualification": file_binding(qualification_path),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "qualification_rows": len(qualification_rows),
            "train_images": len(train_images),
            "dev_images": len(dev_images),
            "qualification_images": len(qualification_images),
            "qualification_per_form": dict(qualification_forms),
            "cross_split_image_sha256_overlap": 0,
            "split_role": "post_training_qualification_holdout",
            "held_out_from_gradient_updates": True,
            "used_during_training_for_validation_and_model_selection": False,
        },
        "audit": {
            "scoped_report": file_binding(scoped_report_path),
            "full_report": file_binding(full_report_path),
            "scoped_verdict": "G_ONLY_PASS",
            "global_verdict": "BLOCKED",
        },
        "training_receipt": file_binding(training_receipt_path),
        "training_args": file_binding(training_args_path),
        "training": {
            "hardware": hardware,
            "world_size": training_world_size,
            "per_device_batch_size": per_device_batch,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": effective_batch,
            "epochs": FROZEN_EPOCHS,
            "max_steps": FROZEN_MAX_STEPS,
            "seed": seed,
            "seed_binding": "receipt_and_args" if receipt_seed is not None else "legacy_args_json",
            "deepspeed": "zero2",
        },
        "models": {
            "base_path": str(base_model),
            "base_config": file_binding(config_path),
            "specialist_adapter": selected_binding,
            "joint_reference_adapter": joint_binding,
        },
        "generation": {
            "hardware": f"exactly_{eval_world_size}xL20",
            "eval_world_size": eval_world_size,
            "num_shards": FROZEN_NUM_SHARDS,
            "max_concurrent_workers": eval_world_size,
            "scheduling": "fixed_8_logical_shards_round_robin_waves",
            "batch_size": batch_size,
            "temperature": 0.0,
            "thinking": False,
            "max_tokens": 512,
            "coordinate_contract": "0_to_1000_thousandth",
        },
        "limitations": [
            "only eight unique qualification images (eight rows per form)",
            "no independent G test split exists in this scoped 148-row release",
            "joint reference is optional and never changes authorization",
        ],
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def model_inventory(model: Path) -> dict[str, Any]:
    """Bind model metadata and the exact weight-file inventory without rehashing 20 GB."""
    model = model.expanduser().resolve()
    config = model / "config.json"
    if not config.is_file():
        raise ValueError(f"model config is missing: {config}")
    files: dict[str, Any] = {"config.json": file_binding(config)}
    index = model / "model.safetensors.index.json"
    single = model / "model.safetensors"
    weights: list[dict[str, Any]] = []
    if index.is_file():
        files[index.name] = file_binding(index)
        value = load_json(index, "model safetensors index")
        names = sorted(set((value.get("weight_map") or {}).values()))
        if not names:
            raise ValueError(f"empty model weight map: {index}")
        for name in names:
            path = model / str(name)
            weights.append(file_binding(path, hash_bytes=False))
    elif single.is_file():
        weights.append(file_binding(single, hash_bytes=False))
    else:
        raise ValueError(f"no safetensors weights found under {model}")
    for optional in ("tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja"):
        path = model / optional
        if path.is_file():
            files[optional] = file_binding(path)
    inventory = {"path": str(model), "metadata_files": files, "weight_files": weights}
    inventory["identity_sha256"] = content_hash(inventory)
    return inventory


def bind_model(
    *, role: str, model: Path, input_receipt: Path, adapter: Path | None = None
) -> dict[str, Any]:
    receipt_path = input_receipt.expanduser().resolve()
    receipt = load_json(receipt_path, "qualification input receipt")
    if receipt.get("protocol") != PROTOCOL:
        raise ValueError("input receipt protocol drift")
    if role not in {"base", "specialist", "joint_reference"}:
        raise ValueError(f"unsupported model role: {role}")
    expected_adapter = None
    if role == "specialist":
        expected_adapter = (receipt.get("models") or {}).get("specialist_adapter")
    elif role == "joint_reference":
        expected_adapter = (receipt.get("models") or {}).get("joint_reference_adapter")
        if not expected_adapter:
            raise ValueError("input receipt has no joint reference")
    observed_adapter = checkpoint_binding(adapter) if adapter else None
    if expected_adapter != observed_adapter:
        raise ValueError(f"{role} adapter does not match the input receipt")
    if role == "base" and model.expanduser().resolve() != Path(receipt["models"]["base_path"]):
        raise ValueError("base model path does not match input receipt")
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "role": role,
        "source_kind": "standalone_base" if role == "base" else "merged_lora",
        "input_receipt": file_binding(receipt_path),
        "model": model_inventory(model),
        "adapter": observed_adapter,
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def verify_model_binding(path: Path, *, deep: bool = False) -> dict[str, Any]:
    binding = load_json(path, "model binding")
    identity = binding.pop("identity_sha256", None)
    if identity != content_hash(binding):
        raise ValueError(f"model binding identity drift: {path}")
    binding["identity_sha256"] = identity
    observed_inventory = model_inventory(Path(binding["model"]["path"]))
    if observed_inventory != binding.get("model"):
        raise ValueError(f"model inventory drift: {path}")
    if deep and binding.get("adapter") is not None:
        observed_adapter = checkpoint_binding(Path(binding["adapter"]["path"]))
        if observed_adapter != binding.get("adapter"):
            raise ValueError(f"adapter checkpoint drift: {path}")
    return binding


_EVAL_SCHEMA = {
    "bbox": '[{"label":"<name>","bbox_2d":[x1,y1,x2,y2]}]',
    "detection": '[{"label":"<name>","bbox_2d":[x1,y1,x2,y2]}, ...]',
    "mask": '[{"label":"<name>","polygon":[[x1,y1],[x2,y2],...]}]',
    "point": '[{"label":"<name>","point_2d":[x,y]}]',
}


def evaluation_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages") or []
    prompt_messages = [
        {"role": str(message["role"]), "content": str(message.get("content", ""))}
        for message in messages
        if message.get("role") in {"system", "user"}
    ]
    if not prompt_messages or prompt_messages[-1]["role"] != "user":
        raise ValueError(f"{row.get('sample_id')}: missing user prompt")
    form = row["gsa_form"]
    suffix = (
        "\nEvaluation contract: return only JSON in this exact form: "
        f"{_EVAL_SCHEMA[form]}. Express every coordinate on a 0-1000 "
        "thousandth scale relative to the image; do not use raw pixels."
    )
    prompt_messages[-1]["content"] = prompt_messages[-1]["content"].rstrip() + suffix
    return prompt_messages


def infer_shard(
    *, receipt_path: Path, binding_path: Path, shard: int, num_shards: int,
    batch_size: int, max_tokens: int, out: Path
) -> dict[str, Any]:
    if not 0 <= shard < num_shards:
        raise ValueError("invalid shard index")
    receipt_path = receipt_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    receipt = load_json(receipt_path, "qualification input receipt")
    expected_shards = (receipt.get("generation") or {}).get("num_shards")
    if expected_shards != FROZEN_NUM_SHARDS or num_shards != expected_shards:
        raise ValueError("inference shard count does not match the bound evaluation receipt")
    if max_tokens != (receipt.get("generation") or {}).get("max_tokens"):
        raise ValueError("inference max_tokens does not match the bound evaluation receipt")
    if batch_size != (receipt.get("generation") or {}).get("batch_size"):
        raise ValueError("inference batch_size does not match the bound evaluation receipt")
    binding = verify_model_binding(binding_path, deep=False)
    if binding["input_receipt"]["sha256"] != file_hash(receipt_path):
        raise ValueError("model binding and input receipt differ")
    qualification_binding = receipt["data"]["qualification"]
    qualification_path = Path(qualification_binding["path"])
    if file_hash(qualification_path) != qualification_binding["sha256"]:
        raise ValueError("qualification corpus drift before inference")
    rows = load_jsonl(qualification_path, "G qualification holdout")
    selected = list(enumerate(rows))[shard::num_shards]
    if not selected:
        raise ValueError(f"empty inference shard {shard}/{num_shards}")

    from swift import InferRequest, RequestConfig
    from eval.gsa_eval.runner import get_engine

    engine = get_engine(
        binding["model"]["path"], batch_size, device_map="cuda:0", thinking=False
    )
    requests = [
        InferRequest(messages=evaluation_messages(row), images=list(row["images"]))
        for _, row in selected
    ]
    responses = engine.infer(
        requests, RequestConfig(max_tokens=max_tokens, temperature=0.0)
    )
    if len(responses) != len(selected):
        raise RuntimeError("inference engine returned an unexpected response count")
    records: list[dict[str, Any]] = []
    for (index, row), response in zip(selected, responses):
        choice = response.choices[0]
        records.append(
            {
                "protocol": PROTOCOL,
                "sample_index": index,
                "shard": shard,
                "num_shards": num_shards,
                "sample_id": row["sample_id"],
                "form": row["gsa_form"],
                "image_sha256": row["gsa_meta"]["image_sha256"],
                "response": choice.message.content or "",
                "hit_token_cap": choice.finish_reason == "length",
                "model_role": binding["role"],
                "model_identity_sha256": binding["identity_sha256"],
                "input_receipt_sha256": file_hash(receipt_path),
            }
        )
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError(f"refusing to overwrite inference shard: {out}")
    temp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp, out)
    return {"role": binding["role"], "shard": shard, "records": len(records), "out": str(out)}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _valid_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    values = [_number(item) for item in value]
    if any(item is None for item in values):
        return None
    x, y = values  # type: ignore[misc]
    return (x, y) if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0 else None


def _valid_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    values = [_number(item) for item in value]
    if any(item is None for item in values):
        return None
    x1, y1, x2, y2 = values  # type: ignore[misc]
    if not all(0.0 <= item <= 1000.0 for item in values):
        return None
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _valid_polygon(value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    points = [_valid_point(point) for point in value]
    if any(point is None for point in points):
        return None
    return [point for point in points if point is not None]


def _label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return normalized or None


def _box_iou(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _polygon_iou(a: list[tuple[float, float]] | None,
                 b: list[tuple[float, float]] | None) -> float:
    if not a or not b:
        return 0.0
    import numpy as np
    from PIL import Image, ImageDraw

    size = 1001
    masks = []
    for polygon in (a, b):
        image = Image.new("1", (size, size), 0)
        ImageDraw.Draw(image).polygon(polygon, outline=1, fill=1)
        masks.append(np.asarray(image, dtype=bool))
    union = np.logical_or(masks[0], masks[1]).sum()
    if not union:
        return 0.0
    return float(np.logical_and(masks[0], masks[1]).sum() / union)


def _max_matching(edges: list[list[int]], n_gt: int) -> int:
    matched = [-1] * n_gt

    def visit(pred: int, seen: set[int]) -> bool:
        for gt in edges[pred]:
            if gt in seen:
                continue
            seen.add(gt)
            if matched[gt] == -1 or visit(matched[gt], seen):
                matched[gt] = pred
                return True
        return False

    return sum(visit(pred, set()) for pred in range(len(edges)))


def _detection_counts(pred: list[dict[str, Any]], gt: list[dict[str, Any]],
                      *, label_aware: bool) -> tuple[int, int, int]:
    edges: list[list[int]] = []
    for p in pred:
        targets = []
        for index, g in enumerate(gt):
            labels_ok = not label_aware or p["label"] == g["label"]
            if labels_ok and _box_iou(p["box"], g["box"]) >= 0.5:
                targets.append(index)
        edges.append(targets)
    tp = _max_matching(edges, len(gt))
    return tp, len(pred) - tp, len(gt) - tp


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 1.0


def _structured_items(text: str) -> list[dict[str, Any]]:
    return [item for item in parse_json_items(text) if isinstance(item, dict)]


def _score_row(row: dict[str, Any], text: str) -> dict[str, Any]:
    form = row["gsa_form"]
    gt_items = row["gsa_payload"]
    items = _structured_items(text)
    primary = (row.get("gsa_context") or {}).get("primary") or {}
    result: dict[str, Any] = {"parse_failure": False, "format_compliant": False}

    if form == "bbox":
        gt = gt_items[0]
        gt_box = _valid_box(gt.get("bbox_2d") or gt.get("bbox"))
        if gt_box is None or _label(gt.get("label")) is None:
            raise ValueError(f"invalid bbox GT: {row['sample_id']}")
        candidates = []
        for item in items:
            box = _valid_box(item.get("bbox_2d") or item.get("bbox"))
            if box:
                candidates.append({"box": box, "label": _label(item.get("label"))})
        if not candidates:
            fallback = [_valid_box(box) for box in parse_boxes(text)]
            candidates = [{"box": box, "label": None} for box in fallback if box]
        if not candidates:
            result["parse_failure"] = True
            iou = 0.0
            pred_label = None
        else:
            iou = _box_iou(candidates[0]["box"], gt_box)
            pred_label = candidates[0]["label"]
        result.update(
            {
                "bbox_iou": iou,
                "iou_at_0_5": int(iou >= 0.5),
                "iou_at_0_75": int(iou >= 0.75),
                "label_match": int(pred_label is not None and pred_label == _label(gt.get("label"))),
                "predicted_objects": len(candidates),
                "format_compliant": len(candidates) == 1 and len(items) == 1,
                "primary": float(iou >= 0.5),
            }
        )
        return result

    if form == "detection":
        gt = [
            {"box": _valid_box(item.get("bbox_2d") or item.get("bbox")),
             "label": _label(item.get("label"))}
            for item in gt_items
        ]
        if any(item["box"] is None or item["label"] is None for item in gt):
            raise ValueError(f"invalid detection GT: {row['sample_id']}")
        pred = []
        for item in items:
            box = _valid_box(item.get("bbox_2d") or item.get("bbox"))
            label = _label(item.get("label"))
            if box and label:
                pred.append({"box": box, "label": label})
        label_counts = _detection_counts(pred, gt, label_aware=True)
        geometry_counts = _detection_counts(pred, gt, label_aware=False)
        result.update(
            {
                "parse_failure": not pred,
                "format_compliant": bool(pred) and len(pred) == len(items),
                "label_tp": label_counts[0], "label_fp": label_counts[1], "label_fn": label_counts[2],
                "geometry_tp": geometry_counts[0], "geometry_fp": geometry_counts[1], "geometry_fn": geometry_counts[2],
                "predicted_objects": len(pred), "gt_objects": len(gt),
                "primary": _f1(*label_counts),
            }
        )
        return result

    if form == "mask":
        gt = gt_items[0]
        gt_polygon = _valid_polygon(gt.get("polygon"))
        if gt_polygon is None or _label(gt.get("label")) is None:
            raise ValueError(f"invalid mask GT: {row['sample_id']}")
        candidates = []
        for item in items:
            polygon = _valid_polygon(item.get("polygon"))
            if polygon:
                candidates.append({"polygon": polygon, "label": _label(item.get("label"))})
        if not candidates:
            result["parse_failure"] = True
            iou = 0.0
            pred_label = None
        else:
            iou = _polygon_iou(candidates[0]["polygon"], gt_polygon)
            pred_label = candidates[0]["label"]
        result.update(
            {
                "mask_iou": iou,
                "iou_at_0_5": int(iou >= 0.5),
                "iou_at_0_75": int(iou >= 0.75),
                "label_match": int(pred_label is not None and pred_label == _label(gt.get("label"))),
                "predicted_objects": len(candidates),
                "format_compliant": len(candidates) == 1 and len(items) == 1,
                "primary": iou,
            }
        )
        return result

    if form == "point":
        gt = gt_items[0]
        gt_point = _valid_point(gt.get("point_2d") or gt.get("point"))
        primary_box = _valid_box(primary.get("bbox"))
        if gt_point is None or primary_box is None or _label(gt.get("label")) is None:
            raise ValueError(f"invalid point GT: {row['sample_id']}")
        candidates = []
        for item in items:
            point = _valid_point(item.get("point_2d") or item.get("point"))
            if point:
                candidates.append({"point": point, "label": _label(item.get("label"))})
        if not candidates:
            fallback = [_valid_point(point) for point in parse_points(text)]
            candidates = [{"point": point, "label": None} for point in fallback if point]
        if not candidates or gt_point is None or primary_box is None:
            result["parse_failure"] = not candidates
            distance = 1.0
            inside = False
            pred_label = None
        else:
            point = candidates[0]["point"]
            distance = math.dist(point, gt_point) / 1000.0
            inside = primary_box[0] <= point[0] <= primary_box[2] and primary_box[1] <= point[1] <= primary_box[3]
            pred_label = candidates[0]["label"]
        result.update(
            {
                "normalized_distance": distance,
                "hit_at_0_1": int(distance <= 0.1),
                "inside_primary_bbox": int(inside),
                "label_match": int(pred_label is not None and pred_label == _label(gt.get("label"))),
                "predicted_objects": len(candidates),
                "format_compliant": len(candidates) == 1 and len(items) == 1,
                "primary": float(inside),
            }
        )
        return result
    raise ValueError(f"unsupported form: {form}")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def score_predictions(
    rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    expected = {row["sample_id"] for row in rows}
    if set(predictions) != expected:
        missing = sorted(expected - set(predictions))
        extra = sorted(set(predictions) - expected)
        raise ValueError(f"prediction coverage mismatch: missing={missing} extra={extra}")
    samples: list[dict[str, Any]] = []
    by_form: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pred = predictions[row["sample_id"]]
        scored = _score_row(row, str(pred.get("response", "")))
        record = {
            "sample_id": row["sample_id"],
            "image_sha256": row["gsa_meta"]["image_sha256"],
            "form": row["gsa_form"],
            "primary": _round(scored["primary"]),
            "parse_failure": bool(scored["parse_failure"]),
            "format_compliant": bool(scored["format_compliant"]),
            "hit_token_cap": bool(pred.get("hit_token_cap", False)),
            "metrics": {key: _round(value) if isinstance(value, float) else value
                        for key, value in scored.items() if key not in {"primary", "parse_failure", "format_compliant"}},
        }
        samples.append(record)
        by_form[row["gsa_form"]].append(record)

    forms: dict[str, Any] = {}
    bbox = by_form["bbox"]
    forms["bbox"] = {
        "n": len(bbox),
        "mean_iou": _round(_mean(item["metrics"]["bbox_iou"] for item in bbox)),
        "iou_at_0_5": _round(_mean(item["metrics"]["iou_at_0_5"] for item in bbox)),
        "iou_at_0_75": _round(_mean(item["metrics"]["iou_at_0_75"] for item in bbox)),
        "label_accuracy": _round(_mean(item["metrics"]["label_match"] for item in bbox)),
        "primary_metric": "iou_at_0_5",
    }
    forms["bbox"]["primary_value"] = forms["bbox"]["iou_at_0_5"]

    detection = by_form["detection"]
    sums = {name: sum(item["metrics"][name] for item in detection)
            for name in ("label_tp", "label_fp", "label_fn", "geometry_tp", "geometry_fp", "geometry_fn")}
    ltp, lfp, lfn = sums["label_tp"], sums["label_fp"], sums["label_fn"]
    gtp, gfp, gfn = sums["geometry_tp"], sums["geometry_fp"], sums["geometry_fn"]
    forms["detection"] = {
        "n_images": len(detection),
        "gt_objects": sum(item["metrics"]["gt_objects"] for item in detection),
        "predicted_objects": sum(item["metrics"]["predicted_objects"] for item in detection),
        "label_aware_tp": ltp, "label_aware_fp": lfp, "label_aware_fn": lfn,
        "label_aware_precision_at_0_5": _round(ltp / (ltp + lfp) if ltp + lfp else 0.0),
        "label_aware_recall_at_0_5": _round(ltp / (ltp + lfn) if ltp + lfn else 0.0),
        "label_aware_f1_at_0_5": _round(_f1(ltp, lfp, lfn)),
        "geometry_only_f1_at_0_5": _round(_f1(gtp, gfp, gfn)),
        "primary_metric": "label_aware_f1_at_0_5",
    }
    forms["detection"]["primary_value"] = forms["detection"]["label_aware_f1_at_0_5"]

    mask = by_form["mask"]
    forms["mask"] = {
        "n": len(mask),
        "mean_polygon_iou": _round(_mean(item["metrics"]["mask_iou"] for item in mask)),
        "iou_at_0_5": _round(_mean(item["metrics"]["iou_at_0_5"] for item in mask)),
        "iou_at_0_75": _round(_mean(item["metrics"]["iou_at_0_75"] for item in mask)),
        "label_accuracy": _round(_mean(item["metrics"]["label_match"] for item in mask)),
        "rasterization": "1001x1001_thousandth_grid",
        "primary_metric": "mean_polygon_iou",
    }
    forms["mask"]["primary_value"] = forms["mask"]["mean_polygon_iou"]

    point = by_form["point"]
    forms["point"] = {
        "n": len(point),
        "inside_primary_bbox": _round(_mean(item["metrics"]["inside_primary_bbox"] for item in point)),
        "hit_at_0_1": _round(_mean(item["metrics"]["hit_at_0_1"] for item in point)),
        "mean_normalized_distance": _round(_mean(item["metrics"]["normalized_distance"] for item in point)),
        "label_accuracy": _round(_mean(item["metrics"]["label_match"] for item in point)),
        "primary_metric": "inside_primary_bbox",
    }
    forms["point"]["primary_value"] = forms["point"]["inside_primary_bbox"]

    macro = _round(_mean(forms[form]["primary_value"] for form in G_FORMS))
    parse_failures = sum(item["parse_failure"] for item in samples)
    token_caps = sum(item["hit_token_cap"] for item in samples)
    format_compliant = sum(item["format_compliant"] for item in samples)
    trusted = parse_failures / len(samples) <= 0.25 and token_caps / len(samples) <= 0.15
    return {
        "n_records": len(samples),
        "n_unique_images": len({item["image_sha256"] for item in samples}),
        "forms": forms,
        "macro_primary": macro,
        "trust": {
            "status": "TRUSTED" if trusted else "WITHHELD",
            "parse_failures": parse_failures,
            "parse_failure_rate": _round(parse_failures / len(samples)),
            "hit_token_cap": token_caps,
            "hit_token_cap_rate": _round(token_caps / len(samples)),
            "format_compliance_rate": _round(format_compliant / len(samples)),
            "thresholds": {"max_parse_failure_rate": 0.25, "max_hit_token_cap_rate": 0.15},
        },
        "samples": samples,
    }


def _load_prediction_files(paths: list[Path], receipt_path: Path,
                           binding: dict[str, Any]) -> dict[str, dict[str, Any]]:
    receipt = load_json(receipt_path, "qualification input receipt")
    expected_shards = (receipt.get("generation") or {}).get("num_shards")
    if expected_shards != FROZEN_NUM_SHARDS:
        raise ValueError("qualification input receipt has an invalid shard count")
    qualification_binding = (receipt.get("data") or {}).get("qualification") or {}
    qualification_path = Path(str(qualification_binding.get("path", ""))).expanduser().resolve()
    if (not qualification_path.is_file()
            or file_hash(qualification_path) != qualification_binding.get("sha256")):
        raise ValueError("qualification corpus drift before scoring")
    expected_rows = load_jsonl(qualification_path, "G qualification holdout")
    expected_receipt = file_hash(receipt_path)
    predictions: dict[str, dict[str, Any]] = {}
    shard_ids: set[int] = set()
    for path in paths:
        match = re.search(r"shard_([0-9]+)\.jsonl$", path.name)
        if not match:
            raise ValueError(f"prediction filename does not bind a shard id: {path}")
        file_shard = int(match.group(1))
        shard_ids.add(file_shard)
        for row in load_jsonl(path, "inference shard"):
            if row.get("protocol") != PROTOCOL:
                raise ValueError(f"prediction protocol drift: {path}")
            if row.get("model_identity_sha256") != binding["identity_sha256"]:
                raise ValueError(f"prediction model binding drift: {path}")
            if row.get("input_receipt_sha256") != expected_receipt:
                raise ValueError(f"prediction input receipt drift: {path}")
            if row.get("shard") != file_shard or row.get("num_shards") != expected_shards:
                raise ValueError(f"prediction row shard binding drift: {path}")
            sample_index = row.get("sample_index")
            if (not isinstance(sample_index, int) or isinstance(sample_index, bool)
                    or not 0 <= sample_index < len(expected_rows)
                    or sample_index % expected_shards != file_shard):
                raise ValueError(f"prediction sample index is in the wrong shard: {path}")
            expected_row = expected_rows[sample_index]
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in predictions:
                raise ValueError(f"duplicate/invalid prediction sample: {sample_id}")
            if (sample_id != expected_row.get("sample_id")
                    or row.get("form") != expected_row.get("gsa_form")
                    or row.get("image_sha256") != (expected_row.get("gsa_meta") or {}).get("image_sha256")
                    or row.get("model_role") != binding.get("role")):
                raise ValueError(f"prediction row provenance drift: {path}")
            predictions[sample_id] = row
    expected_ids = set(range(expected_shards))
    if shard_ids != expected_ids:
        raise ValueError(
            f"expected shard_0..shard_{expected_shards - 1}, found {sorted(shard_ids)}"
        )
    if len(predictions) != len(expected_rows):
        raise ValueError(
            f"expected {len(expected_rows)} bound predictions, found {len(predictions)}"
        )
    return predictions


def build_model_result(*, receipt_path: Path, binding_path: Path,
                       prediction_paths: list[Path]) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    binding_path = binding_path.expanduser().resolve()
    receipt = load_json(receipt_path, "qualification input receipt")
    binding = verify_model_binding(binding_path, deep=True)
    qualification_path = Path(receipt["data"]["qualification"]["path"])
    rows = load_jsonl(qualification_path, "G qualification holdout")
    predictions = _load_prediction_files(prediction_paths, receipt_path, binding)
    scores = score_predictions(rows, predictions)
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "result_label": "G-specialist qualification pilot",
        "model_role": binding["role"],
        "publication_claim_eligible": False,
        "teacher_authorization": "NONE",
        "input_receipt": file_binding(receipt_path),
        "model_binding": file_binding(binding_path),
        "prediction_shards": [file_binding(path) for path in sorted(prediction_paths)],
        **scores,
    }
    payload["identity_sha256"] = content_hash(payload)
    return payload


def _paired_bootstrap(base: dict[str, Any], specialist: dict[str, Any]) -> dict[str, Any]:
    base_by_image: dict[str, list[float]] = defaultdict(list)
    specialist_by_image: dict[str, list[float]] = defaultdict(list)
    for item in base["samples"]:
        base_by_image[item["image_sha256"]].append(float(item["primary"]))
    for item in specialist["samples"]:
        specialist_by_image[item["image_sha256"]].append(float(item["primary"]))
    if set(base_by_image) != set(specialist_by_image):
        raise ValueError("base/specialist image sets differ")
    images = sorted(base_by_image)
    if len(images) != 8 or any(
        len(base_by_image[image]) != 4 or len(specialist_by_image[image]) != 4
        for image in images
    ):
        raise ValueError("paired pilot requires eight images with four G forms each")
    deltas = [
        _mean(specialist_by_image[image]) - _mean(base_by_image[image])
        for image in images
    ]
    rng = random.Random(BOOTSTRAP_SEED)
    boot = []
    for _ in range(BOOTSTRAP_SAMPLES):
        boot.append(_mean(deltas[rng.randrange(len(deltas))] for _ in images))
    boot.sort()

    def percentile(q: float) -> float:
        index = min(len(boot) - 1, max(0, round(q * (len(boot) - 1))))
        return boot[index]

    return {
        "unit": "unique_image_cluster",
        "n_images": len(images),
        "observed_mean_delta": _round(_mean(deltas)),
        "exploratory_bootstrap_95_ci": [_round(percentile(0.025)), _round(percentile(0.975))],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "publication_inference_allowed": False,
    }


def _comparison(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_samples = {
        (item["sample_id"], item["image_sha256"], item["form"])
        for item in base["samples"]
    }
    candidate_samples = {
        (item["sample_id"], item["image_sha256"], item["form"])
        for item in candidate["samples"]
    }
    if base_samples != candidate_samples or len(base_samples) != 32:
        raise ValueError("paired results do not cover the same 32 qualification records")
    form_delta = {
        form: _round(candidate["forms"][form]["primary_value"] - base["forms"][form]["primary_value"])
        for form in G_FORMS
    }
    return {
        "macro_primary_base": base["macro_primary"],
        "macro_primary_candidate": candidate["macro_primary"],
        "macro_primary_delta": _round(candidate["macro_primary"] - base["macro_primary"]),
        "per_form_primary_delta": form_delta,
        "paired_image_analysis": _paired_bootstrap(base, candidate),
    }


def compile_comparison(base_path: Path, specialist_path: Path,
                       joint_path: Path | None = None) -> dict[str, Any]:
    base = load_json(base_path, "base qualification result")
    specialist = load_json(specialist_path, "specialist qualification result")
    for label, payload in (("base", base), ("specialist", specialist)):
        identity = payload.pop("identity_sha256", None)
        if identity != content_hash(payload):
            raise ValueError(f"{label} qualification result identity drift")
        payload["identity_sha256"] = identity
    if base.get("protocol") != PROTOCOL or specialist.get("protocol") != PROTOCOL:
        raise ValueError("qualification result protocol drift")
    if base.get("model_role") != "base" or specialist.get("model_role") != "specialist":
        raise ValueError("base/specialist roles are invalid")
    if base.get("input_receipt") != specialist.get("input_receipt"):
        raise ValueError("base and specialist results use different inputs")
    if base.get("n_records") != 32 or specialist.get("n_records") != 32:
        raise ValueError("base/specialist results are not the fixed 32-record pilot")
    if base.get("n_unique_images") != 8 or specialist.get("n_unique_images") != 8:
        raise ValueError("base/specialist results are not the fixed eight-image pilot")
    paired = _comparison(base, specialist)
    deltas = paired["per_form_primary_delta"]
    trust_ok = base["trust"]["status"] == specialist["trust"]["status"] == "TRUSTED"
    if not trust_ok:
        decision = "UNTRUSTED_NO_DIRECTIONAL_DECISION"
    elif (paired["macro_primary_delta"] > 0
          and sum(value >= 0 for value in deltas.values()) >= 3
          and min(deltas.values()) >= -0.125):
        decision = "G_PILOT_DIRECTIONAL_CONTINUE"
    elif (paired["macro_primary_delta"] < 0
          and sum(value <= 0 for value in deltas.values()) >= 3):
        decision = "G_PILOT_NOT_SUPPORTED_REVISE_OR_STOP"
    else:
        decision = "G_PILOT_INCONCLUSIVE"

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "result_label": "G-specialist qualification pilot",
        "evidence_status": "PILOT_ONLY_PUBLICATION_CLAIM_WITHHELD",
        "engineering_decision": decision,
        "publication_claim_eligible": False,
        "teacher_authorization": "NONE",
        "global_teacher_eligibility_status": "NOT_EVALUATED",
        "joint_training_authorized": False,
        "mopd_training_authorized": False,
        "global_semantic_audit_verdict": "BLOCKED",
        "base_result": file_binding(base_path),
        "specialist_result": file_binding(specialist_path),
        "specialist_vs_base": paired,
        "decision_rule": {
            "continue_if": "trusted; macro delta > 0; >=3/4 form deltas nonnegative; no form delta < -0.125",
            "role": "engineering triage only",
        },
        "publication_blockers": [
            f"{base['n_unique_images']} unique images < preregistered minimum {MIN_PUBLICATION_IMAGES}",
            "no independent G-family test split is available in this scoped release",
        ],
        "required_next_evidence": (
            "Evaluate the frozen model identities on a larger, image-disjoint, "
            "human-audited G test set before any paper claim or teacher-eligibility decision."
        ),
    }
    if joint_path:
        joint = load_json(joint_path, "joint reference qualification result")
        identity = joint.pop("identity_sha256", None)
        if identity != content_hash(joint):
            raise ValueError("joint qualification result identity drift")
        joint["identity_sha256"] = identity
        if joint.get("model_role") != "joint_reference":
            raise ValueError("optional joint result has the wrong role")
        if joint.get("input_receipt") != base.get("input_receipt"):
            raise ValueError("joint reference result uses different inputs")
        report["joint_reference_result"] = file_binding(joint_path)
        report["specialist_vs_joint_reference"] = _comparison(joint, specialist)
        report["joint_reference_note"] = (
            "Optional diagnostic only; its training corpus is not asserted to be matched "
            "and it cannot authorize joint/MOPD training."
        )
    report["identity_sha256"] = content_hash(report)
    return report


def _paths(values: list[str]) -> list[Path]:
    return [Path(value).expanduser().resolve() for value in values]


def native_identity(row: dict[str, Any]) -> tuple:
    images = tuple(str(Path(x["path"] if isinstance(x, dict) else x).resolve()) for x in row["images"])
    prompt = tuple((m["role"], m["content"]) for m in row["messages"] if m["role"] != "assistant")
    return images, prompt


def native_join_predictions(source_rows: list[dict], prediction_rows: list[dict]) -> dict[str, dict]:
    expected = {native_identity(row): row for row in source_rows if not row.get("gsa_eval_blocked_upstream")}
    inferred = {native_identity(row): row for row in prediction_rows}
    if len(inferred) != len(prediction_rows) or len(expected) != sum(not row.get("gsa_eval_blocked_upstream") for row in source_rows):
        raise ValueError("duplicate native prompt/image identity")
    if set(inferred) != set(expected):
        raise ValueError("native prediction prompt/image coverage differs from source")
    result = {}
    for key, original in expected.items():
        references = [m["content"] for m in original["messages"] if m["role"] == "assistant"]
        if len(references) != 1 or inferred[key].get("labels") != references[0]:
            raise ValueError("native reference binding drift")
        sample_id = original.get("gsa_eval_original_sample_id", original["sample_id"])
        if sample_id in result:
            raise ValueError("join each context variant separately; original sample IDs must be unique")
        result[sample_id] = inferred[key]
    return result


def native_context_rows(source_rows: list[dict], variant: str, generated: dict[str, dict] | None = None,
                        forms: tuple[str, ...] = ("depth", "affordance", "dof", "action_cot")) -> list[dict]:
    """Freeze downstream contexts without substituting reference answers for failed parents."""
    import copy
    from trains.flywheel.contracts import training_instruction
    from trains.flywheel.validators import validate_structural

    if variant not in ("oracle", "student_cascade", "corrupted"):
        raise ValueError("unknown native context intervention")
    if not forms or set(forms) - {"depth", "affordance", "dof", "action_cot"}:
        raise ValueError("context interventions require declared downstream forms")
    if variant == "student_cascade" and generated is None:
        raise ValueError("student cascade requires actual upstream generations")
    def reject_constant(value):
        raise ValueError(f"nonfinite upstream JSON value: {value}")
    by_image_form = {}
    for row in source_rows:
        key = row["gsa_meta"]["image_sha256"], row["gsa_form"]
        if key in by_image_form:
            raise ValueError("context pilot needs one source target per image/form")
        by_image_form[key] = row
    needed = {"depth": ("detection",), "affordance": ("detection", "bbox", "depth"),
              "dof": ("detection", "bbox", "depth", "affordance"),
              "action_cot": ("detection", "bbox", "depth", "affordance", "dof")}
    rows = []
    for original in source_rows:
        form = original["gsa_form"]
        if form not in forms:
            continue
        row = copy.deepcopy(original)
        row["gsa_eval_original_sample_id"] = original["sample_id"]
        row["sample_id"] += "::" + variant
        context = copy.deepcopy(original["gsa_context"])
        spatial_keys = {"depth": (), "affordance": ("depth_order",),
                        "dof": ("depth_order", "affordance"),
                        "action_cot": ("depth_order", "affordance", "dof")}[form]
        context["spatial"] = {key: context["spatial"][key] for key in spatial_keys}
        intervention = {"variant": variant, "upstream": {}, "reference_targets_unchanged": True}
        blocked = []
        if variant == "student_cascade":
            parents = {}
            for parent_form in needed[form]:
                parent = by_image_form.get((original["gsa_meta"]["image_sha256"], parent_form))
                if parent is None:
                    raise ValueError("historical reference closure is incomplete")
                parent_id = parent["sample_id"]
                prediction = generated.get(parent_id)
                if prediction is None:
                    blocked.append({"sample_id": parent_id, "reason": "parent_not_generated"})
                    continue
                response = prediction.get("response", "")
                if not isinstance(response, str):
                    raise ValueError("upstream response must be text")
                response = re.sub(r"^\s*<think>\s*</think>\s*", "", response).strip()
                prefix = {"G": "<GROUND>", "S": "<RELATE>", "A": "<ACT>"}[parent["gsa_family"]]
                value, valid = None, False
                try:
                    value = json.loads(response[len(prefix):], parse_constant=reject_constant) if response.startswith(prefix) else None
                    valid, _ = validate_structural(parent_form, value)
                except (ValueError, TypeError):
                    pass
                intervention["upstream"][parent_form] = {"sample_id": parent_id,
                    "response_sha256": hashlib.sha256(prediction["response"].encode()).hexdigest(),
                    "structurally_parseable": bool(valid)}
                if prediction.get("finish_reason") == "length":
                    blocked.append({"sample_id": parent_id, "reason": "parent_token_cap"})
                elif not valid:
                    blocked.append({"sample_id": parent_id, "reason": "parent_structural_failure"})
                else:
                    parents[parent_form] = value
            if not blocked:
                groundings = [{"object_class": item["label"], "bbox": item.get("bbox_2d") or item.get("bbox")}
                              for item in parents["detection"]]
                if form == "depth":
                    primary = copy.deepcopy(groundings[0])
                else:
                    anchor = parents["bbox"][0]
                    name = original["gsa_context"]["primary"]["object_class"]
                    if _label(anchor["label"]) != _label(name):
                        blocked.append({"sample_id": intervention["upstream"]["bbox"]["sample_id"],
                                        "reason": "named_anchor_identity_changed"})
                    primary = {"object_class": name, "bbox": anchor.get("bbox_2d") or anchor.get("bbox")}
                context = {"contract_version": original["gsa_context"]["contract_version"],
                           "primary": primary, "groundings": groundings,
                           "spatial": {key: parents[{"depth_order": "depth"}.get(key, key)] for key in spatial_keys}}
        elif variant == "corrupted":
            if form == "depth":
                boxes = [item["bbox"] for item in context["groundings"]]
                for index, item in enumerate(context["groundings"]):
                    item["bbox"] = boxes[(index + 1) % len(boxes)]
                intervention["perturbation"] = "cyclically_assign_grounding_boxes_to_wrong_named_entities"
            elif form == "affordance":
                values = context["spatial"]["depth_order"]
                ranks = sorted((item["rank"], item["relative_depth"]) for item in values)
                for item, (rank, depth) in zip(sorted(values, key=lambda item: item["rank"]), reversed(ranks)):
                    item["rank"], item["relative_depth"] = rank, depth
                intervention["perturbation"] = "reverse_depth_order_preserving_labels_and_rank_value_consistency"
            elif form == "dof":
                x1, _, x2, _ = context["primary"]["bbox"]
                for part in context["spatial"]["affordance"]:
                    bx1, by1, bx2, by2 = part["bbox_2d"]
                    part["bbox_2d"] = [x1 + x2 - bx2, by1, x1 + x2 - bx1, by2]
                    part["contact_point_2d"][0] = x1 + x2 - part["contact_point_2d"][0]
                intervention["perturbation"] = "reflect_contact_region_within_named_object_horizontal_extent"
            else:
                for grasp in context["spatial"]["dof"]:
                    x, y = grasp["jaw_axis_2d"]
                    grasp["jaw_axis_2d"] = [-y, x]
                    grasp["wrist_deg"] = (grasp["wrist_deg"] + 180) % 180 - 90
                intervention["perturbation"] = "rotate_grasp_axis_and_symmetric_wrist_by_90_degrees"
        row["gsa_context_intervention"] = intervention
        row["gsa_eval_blocked_upstream"] = blocked
        row["gsa_provided_context"] = None if blocked else context
        if not blocked:
            prefix = "<RELATE>" if form == "depth" else "<ACT>"
            prompt = "<image>" + training_instruction(form, context).replace(
                "Return only a JSON array;", f"Return {prefix} followed by a JSON array;")
            users = [message for message in row["messages"] if message["role"] == "user"]
            if len(users) != 1:
                raise ValueError("context pilot requires one user instruction per task")
            if variant == "oracle" and users[0]["content"] != prompt:
                raise ValueError("oracle context re-render changed the original instruction")
            users[0]["content"] = prompt
        intervention["visible_input_changed"] = native_identity(row) != native_identity(original)
        rows.append(row)
    return rows


def native_action_reference_scores(row: dict[str, Any], payload: list, *, usable: bool) -> dict[str, Any]:
    """Single-reference A geometry; this does not score plan meaning or alternate grasps."""
    form = row["gsa_form"]
    references = row["gsa_payload"]
    if len(references) != 1:
        raise ValueError("native A diagnostic requires exactly one historical reference")
    reference = references[0]
    if form == "affordance":
        scores = {"action_match": 0.0, "contact_in_reference_region": 0.0,
                  "part_bbox_iou": 0.0, "single_reference_affordance_score": 0.0}
        box = _valid_box(reference.get("bbox_2d"))
        if box is None or _label(reference.get("affordance")) is None:
            raise ValueError("invalid historical affordance reference")
        if usable and len(payload) == 1:
            prediction = payload[0]
            point = _valid_point(prediction.get("contact_point_2d"))
            predicted_box = _valid_box(prediction.get("bbox_2d"))
            if point is not None and predicted_box is not None:
                scores["action_match"] = float(_label(prediction.get("affordance")) == _label(reference["affordance"]))
                scores["contact_in_reference_region"] = float(box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])
                scores["part_bbox_iou"] = _box_iou(predicted_box, box)
                scores["single_reference_affordance_score"] = scores["action_match"] * (
                    scores["contact_in_reference_region"] + scores["part_bbox_iou"]) / 2
        return {"scores": scores}
    if form == "dof":
        scores = {"grasp_point_score": 0.0, "axis_score": 0.0, "approach_match": 0.0,
                  "wrist_score": 0.0, "single_reference_dof_score": 0.0}
        errors = {"normalized_grasp_distance": None, "axis_error_deg_mod180": None,
                  "wrist_error_deg_mod180": None}
        point = _valid_point(reference.get("grasp_point_2d"))
        axis = reference.get("jaw_axis_2d")
        if (point is None or not isinstance(axis, list) or len(axis) != 2
                or any(_number(value) is None for value in axis) or math.hypot(*axis) == 0
                or _number(reference.get("wrist_deg")) is None
                or reference.get("approach") not in ("top", "side", "front")):
            raise ValueError("invalid historical DoF reference")
        if usable and len(payload) == 1:
            prediction = payload[0]
            predicted_point = _valid_point(prediction.get("grasp_point_2d"))
            predicted_axis = prediction.get("jaw_axis_2d")
            if (predicted_point is not None and isinstance(predicted_axis, list) and len(predicted_axis) == 2
                    and all(_number(value) is not None for value in predicted_axis)
                    and math.hypot(*predicted_axis) > 0 and _number(prediction.get("wrist_deg")) is not None):
                distance = math.dist(point, predicted_point) / 1000.0
                cosine = abs(sum(a * b for a, b in zip(axis, predicted_axis))) / (math.hypot(*axis) * math.hypot(*predicted_axis))
                angle = math.degrees(math.acos(min(1.0, max(0.0, cosine))))
                wrist = abs((prediction["wrist_deg"] - reference["wrist_deg"] + 90) % 180 - 90)
                errors.update(normalized_grasp_distance=distance, axis_error_deg_mod180=angle,
                              wrist_error_deg_mod180=wrist)
                scores.update(grasp_point_score=max(0.0, 1 - distance / .10),
                              axis_score=max(0.0, 1 - angle / 30),
                              approach_match=float(prediction.get("approach") == reference["approach"]),
                              wrist_score=max(0.0, 1 - wrist / 30))
                scores["single_reference_dof_score"] = math.prod(scores[key] for key in (
                    "grasp_point_score", "axis_score", "approach_match", "wrist_score")) ** .25
        return {"scores": scores, "errors_on_usable_outputs": errors}
    if form == "action_cot":
        point = _valid_point(reference.get("target_point_2d"))
        if point is None:
            raise ValueError("invalid historical action target reference")
        distance = None
        if usable and len(payload) == 1:
            predicted = _valid_point(payload[0].get("target_point_2d"))
            if predicted is not None:
                distance = math.dist(point, predicted) / 1000.0
        return {"scores": {"target_point_score": max(0.0, 1 - distance / .10) if distance is not None else 0.0},
                "errors_on_usable_outputs": {"normalized_target_distance": distance},
                "plan_semantic_score": None,
                "plan_semantic_score_status": "MISSING_HUMAN_CHECKLIST_AND_ALTERNATIVES"}
    raise ValueError(f"not an A reference form: {form}")


def native_diagnostic(source: Path, predictions: list[tuple[str, Path]]) -> dict[str, Any]:
    """Audit exact native inference joins; these diagnostics never qualify teachers."""
    from trains.flywheel.validators import validate

    source_rows = load_jsonl(source, "native source")
    indexed = {native_identity(row): row for row in source_rows}
    if not indexed or len(indexed) != len(source_rows):
        raise ValueError("native source has empty or duplicated prompt/image identities")
    image_hashes = {}
    for row in source_rows:
        if len(row["images"]) != 1:
            raise ValueError("native diagnostic requires one image per item")
        image = str(Path(row["images"][0]).resolve())
        if image not in image_hashes:
            image_hashes[image] = file_hash(Path(image))
        if image_hashes[image] != row["gsa_meta"]["image_sha256"]:
            raise ValueError("native image binding drift")
    models = {}
    for role, path in predictions:
        if role in models:
            raise ValueError("duplicate native model role")
        rows = load_jsonl(path, "native predictions")
        joined = native_join_predictions(source_rows, rows)
        items, counts, measures = [], defaultdict(Counter), defaultdict(lambda: defaultdict(list))
        for key, original in indexed.items():
            blocked = original.get("gsa_eval_blocked_upstream") or []
            prediction = joined.get(original.get("gsa_eval_original_sample_id", original["sample_id"]))
            form, family = original["gsa_form"], original["gsa_family"]
            prefix = {"G": "<GROUND>", "S": "<RELATE>", "A": "<ACT>"}[family]
            response = re.sub(r"^\s*<think>\s*</think>\s*", "", prediction["response"]).strip() if prediction else ""
            delimited = response.startswith(prefix)
            text = response[len(prefix):] if delimited else response
            payload, reason, valid = [], "invalid_json_array", False
            try:
                payload = json.loads(text)
                if isinstance(payload, list) and payload and all(isinstance(x, dict) for x in payload):
                    valid, reason = validate(form, payload, original["gsa_context"])
            except (json.JSONDecodeError, TypeError):
                pass
            counts[form].update(n=1, stage_delimiter_correct=int(delimited), strict_contract_pass=int(valid))
            item = {"sample_id": original["sample_id"], "form": form, "stage_delimiter_correct": delimited,
                    "strict_contract_pass": valid, "contract_error": None if valid else reason}
            if "gsa_context_intervention" in original:
                item["context_intervention"] = original["gsa_context_intervention"]
                item["generation_status"] = "BLOCKED_UPSTREAM" if blocked else "GENERATED"
                item["blocked_upstream"] = blocked
                counts[form].update(generation_attempted=int(not blocked), blocked_upstream=int(bool(blocked)))
                provided = original.get("gsa_provided_context")
                input_valid, input_reason = validate(form, payload, provided) if provided and payload else (False, "no_usable_output")
                item["provided_context_contract_pass"] = bool(input_valid and delimited)
                item["provided_context_contract_error"] = None if input_valid and delimited else input_reason
            if family == "G":
                scores = _score_row(original, text)
                item["g_reference_scores"] = scores
                metric = {"bbox": "bbox_iou", "detection": "primary", "mask": "mask_iou", "point": "hit_at_0_1"}[form]
                measures[form][metric].append(scores[metric])
            elif form == "depth":
                gt = {_label(x["label"]): x["rank"] for x in original["gsa_payload"]}
                pred = {_label(x.get("label")): x.get("rank") for x in payload} if valid else {}
                keys = sorted(gt)
                pairs = [(keys[a], keys[b]) for a in range(len(keys)) for b in range(a + 1, len(keys))]
                correct = sum((gt[a] < gt[b]) == (pred[a] < pred[b]) for a, b in pairs) if set(gt) == set(pred) else 0
                item["depth_pairs"] = {"correct": correct, "total": len(pairs)}
                measures[form]["mean_per_item_pair_accuracy"].append(correct / len(pairs) if pairs else 0)
            elif family == "A":
                diagnostics = native_action_reference_scores(original, payload, usable=valid and delimited)
                item["a_reference_diagnostics"] = diagnostics
                for metric, value in diagnostics["scores"].items():
                    measures[form][metric].append(value)
            items.append(item)
        known_finish = [row for row in rows if row.get("finish_reason") in ("stop", "length")]
        health = {"generated_records": len(rows), "known_finish_reasons": len(known_finish),
                  "token_cap_hits": sum(row.get("finish_reason") == "length" for row in rows),
                  "all_finish_reasons_recorded": len(known_finish) == len(rows),
                  "records_with_token_counts": sum(isinstance(row.get("completion_tokens"), int) for row in rows)}
        models[role] = {"prediction": file_binding(path), "coverage": len(rows), "items": items, "output_health": health,
                        "per_form": {form: {**dict(counts[form]), "reference_diagnostics": {
                            metric: _mean(values) for metric, values in measures[form].items()}} for form in counts}}
    variants = sorted({row.get("gsa_context_intervention", {}).get("variant", "oracle") for row in source_rows})
    return {"schema": "anvil-native-oracle-diagnostic-v1" if variants == ["oracle"] else "anvil-native-context-diagnostic-v1",
            "source": file_binding(source), "context_variants": variants,
            "evaluator_sha256": file_hash(Path(__file__)), "models": models,
            "action_reference_protocol": {
                "version": "single-historical-reference-geometry-v1",
                "denominator": "all source cases; malformed, wrong-stage and contract-invalid predictions score zero",
                "point_error_scale": "Euclidean distance in 0..1000 coordinates divided by 1000",
                "point_zero_at": .10, "axis_and_wrist_zero_at_deg": 30, "angular_symmetry_deg": 180,
                "alternative_reference_coverage": "not established",
                "plan_semantic_score": None,
            },
            "teacher_eligible": False, "publication_claim_eligible": False,
            "limitations": ["Context variants describe this frozen pilot only; no matched-budget global teacher qualification is granted.",
                            "Legacy Swift CLI outputs omit finish reasons; native API output health is recorded separately when available.",
                            "Reference measurements are small-sample diagnostics; A geometry uses one historical target and cannot judge all valid alternative grasps.",
                            "Action target geometry does not measure plan semantics; missing human checklists are not replaced by lexical similarity.",
                            "Machine preaudit and formal human confirmation remain separate."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    native = sub.add_parser("native-diagnostic", help="Compare native oracle predictions without qualification admission")
    native.add_argument("--source", type=Path, required=True)
    native.add_argument("--prediction", nargs=2, action="append", metavar=("ROLE", "JSONL"), required=True)
    native.add_argument("--out", type=Path, required=True)

    resolve = sub.add_parser("resolve-checkpoint")
    resolve.add_argument("--source", type=Path, required=True)

    checkpoint_id = sub.add_parser("checkpoint-id")
    checkpoint_id.add_argument("--checkpoint", type=Path, required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--scoped-report", type=Path, required=True)
    preflight.add_argument("--full-report", type=Path, required=True)
    preflight.add_argument("--base-model", type=Path, required=True)
    preflight.add_argument("--source", type=Path, required=True)
    preflight.add_argument("--checkpoint", type=Path, required=True)
    preflight.add_argument("--joint-checkpoint", type=Path)
    preflight.add_argument("--eval-world-size", type=int, choices=ALLOWED_WORLD_SIZES, default=8)
    preflight.add_argument("--batch-size", type=int, choices=(4,), default=4)
    preflight.add_argument("--out", type=Path, required=True)

    bind = sub.add_parser("bind-model")
    bind.add_argument("--role", choices=("base", "specialist", "joint_reference"), required=True)
    bind.add_argument("--model", type=Path, required=True)
    bind.add_argument("--adapter", type=Path)
    bind.add_argument("--input-receipt", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)

    verify = sub.add_parser("verify-model-binding")
    verify.add_argument("--binding", type=Path, required=True)
    verify.add_argument("--deep", action="store_true")

    infer = sub.add_parser("infer-shard")
    infer.add_argument("--input-receipt", type=Path, required=True)
    infer.add_argument("--model-binding", type=Path, required=True)
    infer.add_argument("--shard", type=int, required=True)
    infer.add_argument("--num-shards", type=int, default=8)
    infer.add_argument("--batch-size", type=int, default=4)
    infer.add_argument("--max-tokens", type=int, default=512)
    infer.add_argument("--out", type=Path, required=True)

    score = sub.add_parser("score")
    score.add_argument("--input-receipt", type=Path, required=True)
    score.add_argument("--model-binding", type=Path, required=True)
    score.add_argument("--predictions", nargs="+", required=True)
    score.add_argument("--out", type=Path, required=True)

    compare = sub.add_parser("compare")
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--specialist", type=Path, required=True)
    compare.add_argument("--joint", type=Path)
    compare.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "native-diagnostic":
            if args.out.exists():
                raise ValueError("native diagnostic requires a fresh output")
            payload = native_diagnostic(args.source, [(role, Path(path)) for role, path in args.prediction])
            atomic_json(args.out, payload)
            print(json.dumps({role: model["per_form"] for role, model in payload["models"].items()}, indent=2))
            return 0
        if args.command == "resolve-checkpoint":
            print(resolve_checkpoint(args.source))
            return 0
        if args.command == "checkpoint-id":
            print(checkpoint_binding(args.checkpoint)["identity_sha256"])
            return 0
        if args.command == "preflight":
            payload = validate_release_inputs(
                manifest_path=args.manifest,
                scoped_report_path=args.scoped_report,
                full_report_path=args.full_report,
                base_model=args.base_model,
                source=args.source,
                checkpoint=args.checkpoint,
                joint_checkpoint=args.joint_checkpoint,
                eval_world_size=args.eval_world_size,
                batch_size=args.batch_size,
            )
            atomic_json(args.out, payload)
            print(json.dumps({"status": "READY", "receipt": str(args.out.resolve()),
                              "identity_sha256": payload["identity_sha256"]}, indent=2))
            return 0
        if args.command == "bind-model":
            payload = bind_model(role=args.role, model=args.model,
                                 input_receipt=args.input_receipt, adapter=args.adapter)
            atomic_json(args.out, payload)
            print(json.dumps({"status": "BOUND", "role": args.role,
                              "identity_sha256": payload["identity_sha256"]}, indent=2))
            return 0
        if args.command == "verify-model-binding":
            payload = verify_model_binding(args.binding, deep=args.deep)
            print(json.dumps({"status": "BOUND", "role": payload["role"],
                              "identity_sha256": payload["identity_sha256"]}, indent=2))
            return 0
        if args.command == "infer-shard":
            payload = infer_shard(receipt_path=args.input_receipt,
                                  binding_path=args.model_binding, shard=args.shard,
                                  num_shards=args.num_shards, batch_size=args.batch_size,
                                  max_tokens=args.max_tokens, out=args.out)
            print(json.dumps(payload, indent=2))
            return 0
        if args.command == "score":
            payload = build_model_result(
                receipt_path=args.input_receipt,
                binding_path=args.model_binding,
                prediction_paths=_paths(args.predictions),
            )
            atomic_json(args.out, payload)
            print(json.dumps({"status": payload["trust"]["status"],
                              "role": payload["model_role"],
                              "macro_primary": payload["macro_primary"],
                              "out": str(args.out.resolve())}, indent=2))
            return 0
        if args.command == "compare":
            payload = compile_comparison(args.base, args.specialist, args.joint)
            atomic_json(args.out, payload)
            print(json.dumps({"evidence_status": payload["evidence_status"],
                              "engineering_decision": payload["engineering_decision"],
                              "publication_claim_eligible": False,
                              "teacher_authorization": "NONE",
                              "out": str(args.out.resolve())}, indent=2))
            return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
