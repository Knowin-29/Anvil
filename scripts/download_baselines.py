#!/usr/bin/env python3
"""Download reviewed Hugging Face baselines with integrity and provenance checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download


IGNORE_PATTERNS = [
    "*.gguf",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "original/*",
]
COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


class IntegrityError(ValueError):
    """A remote revision or local checkpoint failed a fixed integrity check."""


def _ignored(relative_path: str) -> bool:
    path = Path(relative_path)
    return any(path.match(pattern) for pattern in IGNORE_PATTERNS)


def _family(entry: dict[str, Any]) -> str:
    repo_id = str(entry.get("repo_id", "")).lower()
    model_id = str(entry.get("id", "")).lower()
    if "internvl" in repo_id or "internvl" in model_id:
        return "internvl_hf"
    if repo_id.startswith("qwen/") and "vl" in repo_id:
        return "qwen_vl"
    return "image_text_to_text"


def _required_files(entry: dict[str, Any]) -> tuple[set[str], tuple[tuple[str, ...], ...]]:
    exact = {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }
    family = _family(entry)
    if family == "internvl_hf":
        exact.update({"processor_config.json", "chat_template.jinja"})
        alternatives = (INDEX_FILES,)
    elif family == "qwen_vl":
        exact.add("chat_template.json")
        alternatives = (INDEX_FILES,)
    else:
        alternatives = (INDEX_FILES, ("chat_template.json", "chat_template.jinja"))
    return exact, alternatives


def _validate_required_names(
    names: set[str], entry: dict[str, Any], context: str
) -> None:
    exact, alternatives = _required_files(entry)
    missing = sorted(exact - names)
    for choices in alternatives:
        if not any(choice in names for choice in choices):
            missing.append("one of: " + ", ".join(choices))
    if missing:
        raise IntegrityError(f"{context} lacks required files: {missing}")


def _present_required_names(names: set[str], entry: dict[str, Any]) -> set[str]:
    exact, alternatives = _required_files(entry)
    selected = set(exact)
    for choices in alternatives:
        selected.add(next(choice for choice in choices if choice in names))
    return selected


def remote_inventory(info: Any) -> list[dict[str, Any]]:
    inventory = []
    for sibling in getattr(info, "siblings", ()) or ():
        name = str(sibling.rfilename)
        if _ignored(name):
            continue
        size = getattr(sibling, "size", None)
        if size is None:
            lfs = getattr(sibling, "lfs", None)
            size = getattr(lfs, "size", 0) if lfs is not None else 0
        inventory.append({"path": name, "size_bytes": int(size or 0)})
    return sorted(inventory, key=lambda item: item["path"])


def inspect_remote(entry: dict[str, Any], api: HfApi) -> dict[str, Any]:
    requested_revision = str(entry.get("revision", "main"))
    info = api.model_info(
        entry["repo_id"], revision=requested_revision, files_metadata=True
    )
    resolved_sha = str(getattr(info, "sha", "") or "")
    if not COMMIT_SHA.fullmatch(resolved_sha):
        raise IntegrityError(
            f"remote {entry['repo_id']} returned invalid resolved sha {resolved_sha!r}"
        )
    revision_is_pinned = bool(COMMIT_SHA.fullmatch(requested_revision))
    if revision_is_pinned and resolved_sha.lower() != requested_revision.lower():
        raise IntegrityError(
            f"remote revision drift for {entry['repo_id']}: requested "
            f"{requested_revision}, resolved {resolved_sha}"
        )

    inventory = remote_inventory(info)
    names = {item["path"] for item in inventory}
    _validate_required_names(names, entry, f"remote {entry['repo_id']}")
    sizes = {item["path"]: item["size_bytes"] for item in inventory}
    empty_required = sorted(
        name for name in _present_required_names(names, entry) if sizes[name] <= 0
    )
    if empty_required:
        raise IntegrityError(
            f"remote {entry['repo_id']} has empty required files: {empty_required}"
        )
    return {
        "requested_revision": requested_revision,
        "resolved_sha": resolved_sha,
        "revision_is_pinned_commit": revision_is_pinned,
        "files": inventory,
        "total_bytes": sum(item["size_bytes"] for item in inventory),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and ".cache" not in path.relative_to(root).parts
    )


def inspect_local(
    path: Path,
    entry: dict[str, Any],
    expected_remote_files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not path.is_dir():
        raise IntegrityError(f"checkpoint directory is missing: {path}")

    incomplete = sorted(
        str(candidate.relative_to(path))
        for candidate in path.rglob("*.incomplete")
        if candidate.is_file()
    )
    if incomplete:
        raise IntegrityError(f"checkpoint contains incomplete downloads: {incomplete}")

    payload = _payload_files(path)
    names = {str(candidate.relative_to(path)) for candidate in payload}
    _validate_required_names(names, entry, f"local checkpoint {path}")
    empty_required = sorted(
        name
        for name in _present_required_names(names, entry)
        if (path / name).stat().st_size == 0
    )
    if empty_required:
        raise IntegrityError(
            f"local checkpoint {path} has empty required files: {empty_required}"
        )

    index_name = next((name for name in INDEX_FILES if name in names), None)
    assert index_name is not None  # enforced by _validate_required_names
    try:
        index = json.loads((path / index_name).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"unreadable weight index {path / index_name}: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise IntegrityError(f"weight index has no non-empty weight_map: {path / index_name}")

    raw_references = list(weight_map.values())
    invalid_references = [
        value
        for value in raw_references
        if not isinstance(value, str)
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ]
    if invalid_references:
        raise IntegrityError(f"weight index has unsafe shard paths: {invalid_references}")
    referenced = sorted(set(raw_references))
    missing_shards = [name for name in referenced if not (path / name).is_file()]
    empty_shards = [
        name for name in referenced if (path / name).is_file() and (path / name).stat().st_size == 0
    ]
    if missing_shards or empty_shards:
        raise IntegrityError(
            f"checkpoint weight shards invalid: missing={missing_shards}, empty={empty_shards}"
        )

    remote_sizes = {
        item["path"]: int(item.get("size_bytes") or 0)
        for item in (expected_remote_files or [])
    }
    missing_remote = sorted(set(remote_sizes) - names)
    size_mismatches = []
    if missing_remote:
        raise IntegrityError(f"local checkpoint lacks remote payload files: {missing_remote}")
    for name, expected_size in remote_sizes.items():
        if expected_size and (actual_size := (path / name).stat().st_size) != expected_size:
            size_mismatches.append(
                {"path": name, "expected": expected_size, "actual": actual_size}
            )
    if size_mismatches:
        raise IntegrityError(f"local payload size mismatches: {size_mismatches}")

    inventory = [
        {
            "path": str(candidate.relative_to(path)),
            "size_bytes": candidate.stat().st_size,
            "sha256": _sha256(candidate),
        }
        for candidate in payload
    ]
    return {
        "status": "complete",
        "files": inventory,
        "total_bytes": sum(item["size_bytes"] for item in inventory),
        "weight_index": index_name,
        "weight_shards": referenced,
    }


def complete(
    path: Path,
    entry: dict[str, Any] | None = None,
    expected_remote_files: list[dict[str, Any]] | None = None,
) -> bool:
    """Compatibility predicate backed by the strict checkpoint validator."""
    if entry is None:
        entry = {"id": "generic", "repo_id": "generic/image-text-to-text"}
    try:
        inspect_local(path, entry, expected_remote_files)
    except IntegrityError:
        return False
    return True


def build_audit_record(
    *,
    model_id: str,
    entry: dict[str, Any],
    destination: Path,
    resolved_path: str,
    remote: dict[str, Any],
    local: dict[str, Any],
    download_performed: bool,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 2,
        "model_id": model_id,
        "repo_id": entry["repo_id"],
        "requested_revision": remote["requested_revision"],
        "resolved_revision": remote["resolved_sha"],
        "revision_is_pinned_commit": remote["revision_is_pinned_commit"],
        "resolved_path": resolved_path,
        "destination": str(destination),
        "validated_at_utc": now,
        "download_performed": download_performed,
        "downloaded_at_utc": now if download_performed else None,
        "estimated_repo_bytes": remote["total_bytes"],
        "local_bytes": local["total_bytes"],
        "remote_inventory": remote["files"],
        "local_inventory": local["files"],
        "weight_index": local["weight_index"],
        "weight_shards": local["weight_shards"],
    }


def _validate_entry(model_id: str, entry: dict[str, Any]) -> None:
    if entry.get("source") != "huggingface" or not entry.get("repo_id"):
        raise SystemExit(f"{model_id} is not a downloadable Hugging Face entry")
    if entry.get("harness") != "unified_gsa_eval":
        raise SystemExit(
            f"{model_id} is excluded from the unified harness: "
            f"{entry.get('exclusion_reason', 'protocol mismatch')}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--registry", default="configs/models/baselines.yaml")
    ap.add_argument("--min-free-gib", type=float, default=30.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="validate remote metadata and any existing local checkpoint; do not download")
    ap.add_argument("--audit-dir", default="artifacts/downloads")
    args = ap.parse_args()

    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    entries = {item["id"]: item for item in registry["models"]}
    unknown = sorted(set(args.ids) - set(entries))
    if unknown:
        raise SystemExit(f"unknown model ids: {', '.join(unknown)}")
    if args.check_only and args.force:
        raise SystemExit("--check-only and --force are mutually exclusive")

    api = HfApi()
    audit_dir = Path(args.audit_dir)

    for model_id in args.ids:
        entry = entries[model_id]
        _validate_entry(model_id, entry)
        try:
            remote = inspect_remote(entry, api)
        except IntegrityError as exc:
            raise SystemExit(str(exc)) from exc
        destination = Path(entry["path"])

        local = None
        if destination.exists():
            try:
                local = inspect_local(destination, entry, remote["files"])
            except IntegrityError as exc:
                if args.check_only:
                    raise SystemExit(str(exc)) from exc
                print(f"[repair] {model_id}: {exc}")

        if args.check_only:
            local_status = "complete" if local is not None else "missing"
            print(
                f"[check] {model_id}: remote={remote['resolved_sha']} "
                f"files={len(remote['files'])} local={local_status}"
            )
            continue

        download_performed = False
        resolved = str(destination.resolve())
        if local is not None and not args.force:
            print(f"[skip] {model_id}: complete checkpoint already exists at {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(destination.parent).free
            reserve = int(args.min_free_gib * 1024 ** 3)
            if remote["total_bytes"] and free < remote["total_bytes"] + reserve:
                raise SystemExit(
                    f"insufficient disk for {model_id}: need about "
                    f"{remote['total_bytes'] / 1024 ** 3:.1f} GiB plus "
                    f"{args.min_free_gib:.1f} GiB reserve; have {free / 1024 ** 3:.1f} GiB"
                )

            print(
                f"[download] {entry['repo_id']}@{remote['resolved_sha']} -> {destination} "
                f"(estimated {remote['total_bytes'] / 1024 ** 3:.1f} GiB)"
            )
            resolved = snapshot_download(
                repo_id=entry["repo_id"],
                revision=remote["resolved_sha"],
                local_dir=str(destination),
                ignore_patterns=IGNORE_PATTERNS,
            )
            download_performed = True
            try:
                local = inspect_local(destination, entry, remote["files"])
            except IntegrityError as exc:
                raise SystemExit(f"incomplete checkpoint after download: {exc}") from exc

        assert local is not None
        record = build_audit_record(
            model_id=model_id,
            entry=entry,
            destination=destination,
            resolved_path=resolved,
            remote=remote,
            local=local,
            download_performed=download_performed,
        )
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / f"{model_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[ready] {model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
