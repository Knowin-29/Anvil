#!/usr/bin/env python3
"""CPU-only admission checks for general-retention and specialist evidence.

This script never downloads data, imports a model, or touches CUDA. It distinguishes
missing assets (TBD), executable but non-formal configurations (diagnostic), and
evidence that has passed every locally checkable formal gate (formal).
"""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

import yaml
from PIL import Image


VALID_STATES = {"formal", "diagnostic", "TBD"}
INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - release identity, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"registry must be a mapping: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version in {path}")
    return payload


def _state(value: Any, context: str) -> str:
    value = str(value)
    if value not in VALID_STATES and not value.startswith(("formal_", "diagnostic_")):
        raise ValueError(f"invalid current_state/target_tier {value!r} for {context}")
    return value


def inspect_checkpoint(path: Path) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not path.is_dir():
        return False, [f"missing model directory: {path}"]
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        candidate = path / name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            reasons.append(f"missing or empty {name}")
    if not any(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in ("chat_template.json", "chat_template.jinja")
    ):
        reasons.append("missing or empty chat template")
    if any(path.rglob("*.incomplete")):
        reasons.append("checkpoint contains .incomplete files")
    index_path = next((path / name for name in INDEX_FILES if (path / name).is_file()), None)
    if index_path is None:
        reasons.append("missing weight index")
    else:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            referenced = sorted(set((index.get("weight_map") or {}).values()))
            if not referenced:
                reasons.append("empty weight_map")
            for relative in referenced:
                shard = path / relative
                if not shard.is_file() or shard.stat().st_size == 0:
                    reasons.append(f"missing or empty shard: {relative}")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            reasons.append(f"unreadable weight index: {exc}")
    return not reasons, reasons


def inspect_tsv(entry: dict[str, Any], *, decode_images: bool) -> dict[str, Any]:
    path = Path(entry["local_path"])
    report: dict[str, Any] = {
        "id": entry["id"],
        "declared_state": _state(entry["current_state"], entry["id"]),
        "path": str(path),
        "state": "TBD",
        "reasons": [],
    }
    if not path.is_file():
        report["reasons"].append("official split file is missing")
        return report

    actual_md5 = file_md5(path)
    report["md5"] = actual_md5
    if actual_md5 != str(entry["md5"]).lower():
        report["state"] = "diagnostic"
        report["reasons"].append(
            f"release md5 mismatch: {actual_md5} != {entry['md5']}"
        )
        return report

    required = set(entry["required_columns"])
    image_columns = tuple(entry["image_columns"])
    ids: set[str] = set()
    image_hashes: set[str] = set()
    n = 0
    errors: list[str] = []
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = sorted(required - columns)
        if missing:
            errors.append(f"missing columns: {missing}")
        if not any(name in columns for name in image_columns):
            errors.append(f"none of image columns present: {image_columns}")
        if errors:
            report["state"] = "diagnostic"
            report["reasons"].extend(errors)
            return report
        for row in reader:
            n += 1
            uid = str(row.get(entry["id_column"], ""))
            if not uid or uid in ids:
                errors.append(f"empty or duplicate sample id at row {n + 1}: {uid!r}")
                break
            ids.add(uid)
            if not str(row.get("question", "")).strip() or not str(row.get("answer", "")).strip():
                errors.append(f"empty question/answer at row {n + 1}")
                break
            if decode_images:
                raw = row.get("image", "")
                image_path = row.get("image_path", "")
                try:
                    if raw:
                        image_bytes = base64.b64decode(raw, validate=True)
                    elif image_path:
                        source = Path(image_path)
                        if not source.is_absolute():
                            source = path.parent / source
                        image_bytes = source.read_bytes()
                    else:
                        raise ValueError("empty image and image_path")
                    if not image_bytes:
                        raise ValueError("empty image payload")
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.verify()
                    image_hashes.add(hashlib.sha256(image_bytes).hexdigest())
                except (OSError, ValueError, binascii.Error) as exc:
                    errors.append(f"invalid image at row {n + 1}: {exc}")
                    break

    report["n"] = n
    report["unique_images"] = len(image_hashes) if decode_images else None
    if n != int(entry["expected_n"]):
        errors.append(f"row count {n} != expected {entry['expected_n']}")
    if errors:
        report["state"] = "diagnostic"
        report["reasons"].extend(errors)
    elif not decode_images:
        report["state"] = "diagnostic"
        report["reasons"].append("image payloads were not decoded (--metadata-only)")
    else:
        report["state"] = "formal"
    return report


def inspect_corpus(entry: dict[str, Any]) -> dict[str, Any]:
    path = Path(entry["path"])
    report: dict[str, Any] = {
        "id": entry["id"],
        "path": str(path),
        "state": "TBD",
        "reasons": [],
    }
    if not path.is_file():
        report["reasons"].append("post-training corpus is missing")
        return report

    actual_sha256 = file_sha256(path)
    with path.open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    report.update({"rows": rows, "sha256": actual_sha256})
    if rows != int(entry["expected_rows"]):
        report["reasons"].append(
            f"row count {rows} != expected {entry['expected_rows']}"
        )
    if actual_sha256 != str(entry["sha256"]).lower():
        report["reasons"].append(
            f"sha256 mismatch: {actual_sha256} != {entry['sha256']}"
        )
    report["state"] = "formal" if not report["reasons"] else "diagnostic"
    return report


def inspect_leakage_audit(entry: dict[str, Any]) -> dict[str, Any]:
    path = Path(entry["required_path"])
    report = {"path": str(path), "state": "TBD", "reasons": []}
    if not path.is_file():
        report["reasons"].append("frozen leakage audit is missing")
        return report
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report["state"] = "diagnostic"
        report["reasons"].append(f"unreadable leakage audit: {exc}")
        return report
    for key, expected in entry["required_fields"].items():
        if payload.get(key) != expected:
            report["reasons"].append(
                f"{key}={payload.get(key)!r}, required={expected!r}"
            )
    report["state"] = "formal" if not report["reasons"] else "diagnostic"
    return report


def inspect_general(registry: dict[str, Any], *, decode_images: bool) -> dict[str, Any]:
    suite = registry["suite"]
    _state(suite["current_state"], suite["id"])
    datasets = [inspect_tsv(item, decode_images=decode_images) for item in registry["benchmarks"]]
    models = []
    for item in registry["models"]:
        declared = _state(item["current_state"], item["id"])
        ok, reasons = inspect_checkpoint(Path(item["path"]))
        models.append({
            "id": item["id"],
            "declared_state": declared,
            "state": declared if ok else "TBD",
            "path": item["path"],
            "required_for_initial_release": bool(
                item.get("required_for_initial_release", True)
            ),
            "reasons": reasons,
        })
    corpora = [inspect_corpus(item) for item in registry["posttraining_corpora"]]
    runner_path = Path(registry["formal_runner"]["required_path"])
    runner = {
        "path": str(runner_path),
        "state": "formal" if runner_path.is_file() else "TBD",
        "reasons": [] if runner_path.is_file() else [registry["formal_runner"]["blocker"]],
    }
    leakage = inspect_leakage_audit(registry["leakage_audit"])
    ready = (
        all(item["state"] == "formal" for item in datasets)
        and all(
            item["state"] in {"formal", "diagnostic"}
            for item in models
            if item["required_for_initial_release"]
        )
        and all(item["state"] == "formal" for item in corpora)
        and runner["state"] == "formal"
        and leakage["state"] == "formal"
    )
    return {
        "suite_id": suite["id"],
        "state": "formal" if ready else "TBD",
        "ready_for_formal_gpu": ready,
        "datasets": datasets,
        "models": models,
        "posttraining_corpora": corpora,
        "runner": runner,
        "leakage_audit": leakage,
    }


def inspect_specialists(registry: dict[str, Any]) -> dict[str, Any]:
    models = []
    for item in registry["models"]:
        declared = _state(item["current_state"], item["id"])
        path_value = item.get("path")
        reasons: list[str] = []
        if not path_value:
            state = "TBD"
            reasons.append("no exact released checkpoint is registered")
        else:
            ok, checkpoint_reasons = inspect_checkpoint(Path(path_value))
            reasons.extend(checkpoint_reasons)
            wrapper_value = item.get("wrapper")
            wrapper_ok = bool(wrapper_value and Path(wrapper_value).is_file())
            if not wrapper_ok:
                reasons.append("required same-protocol wrapper is missing")
            if ok and wrapper_ok and item.get("table_eligible") is True:
                state = "formal"
            elif ok and wrapper_ok:
                state = "diagnostic"
            else:
                state = "TBD"
        models.append({
            "id": item["id"],
            "declared_state": declared,
            "target_tier": item["target_tier"],
            "state": state,
            "path": path_value,
            "harness": item["harness"],
            "reasons": reasons,
        })
    return {"models": models, "ready_ids": [m["id"] for m in models if m["state"] == "formal"]}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--general-registry",
        type=Path,
        default=root / "configs/eval/general_retention_v1.yaml",
    )
    parser.add_argument(
        "--specialist-registry",
        type=Path,
        default=root / "configs/models/spatial_specialists.yaml",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="skip base64 decoding; useful only for a quick diagnostic, never formal admission",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    general = inspect_general(load_yaml(args.general_registry), decode_images=not args.metadata_only)
    if args.metadata_only and general["ready_for_formal_gpu"]:
        general["ready_for_formal_gpu"] = False
        general["state"] = "TBD"
        general["metadata_only"] = True
    specialists = inspect_specialists(load_yaml(args.specialist_registry))
    report = {"general": general, "specialists": specialists}
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(
            f"general {general['suite_id']}: {general['state']} "
            f"ready_for_formal_gpu={general['ready_for_formal_gpu']}"
        )
        for item in general["datasets"]:
            print(f"  dataset {item['id']}: {item['state']} - {'; '.join(item['reasons']) or 'ok'}")
        for item in general["posttraining_corpora"]:
            print(f"  corpus {item['id']}: {item['state']} - {'; '.join(item['reasons']) or 'ok'}")
        print(f"  runner: {general['runner']['state']} - {'; '.join(general['runner']['reasons']) or 'ok'}")
        print(
            f"  leakage: {general['leakage_audit']['state']} - "
            f"{'; '.join(general['leakage_audit']['reasons']) or 'ok'}"
        )
        print("specialists:")
        for item in specialists["models"]:
            print(f"  {item['id']}: {item['state']} - {'; '.join(item['reasons']) or 'ok'}")
    return 0 if general["ready_for_formal_gpu"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
