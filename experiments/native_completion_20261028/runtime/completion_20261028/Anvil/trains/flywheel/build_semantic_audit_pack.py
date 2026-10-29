#!/usr/bin/env python3
"""Build two immutable, aligned roots for a v3 semantic-audit round."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_FORMS = (
    "action_cot", "affordance", "bbox", "depth",
    "detection", "dof", "mask", "point",
)
REQUIRED_IMAGES = 40


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def normalized_path(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def ensure_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit(f"refusing to overwrite non-empty audit output: {path}")


def require_version(record: dict[str, Any], path: Path, line_number: int) -> None:
    sample_id = record.get("sample_id", f"{path}:{line_number}")
    for field in ("label_version", "prompt_version"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"{sample_id}: missing non-empty {field}")


def load_forms(source: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_form: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    label_root = source / "labels"
    paths = sorted(label_root.rglob("*.jsonl"))
    if not paths:
        raise SystemExit(f"no label JSONL files found below {label_root}")
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            require_version(record, path, line_number)
            form = record.get("task_form")
            image = record.get("image_path")
            sample_id = record.get("sample_id")
            if not all(isinstance(value, str) and value for value in (form, image, sample_id)):
                raise SystemExit(f"{path}:{line_number}: task_form, image_path, and sample_id are required")
            previous = by_form[form].get(image)
            if previous is not None and content_hash(previous) != content_hash(record):
                raise SystemExit(f"conflicting duplicate label for form={form} image={image}")
            by_form[form][image] = record
    forms = tuple(sorted(by_form))
    if forms != tuple(sorted(REQUIRED_FORMS)):
        raise SystemExit(f"expected exactly eight forms {list(REQUIRED_FORMS)}, found {list(forms)}")
    return dict(by_form)


def source_fingerprint(by_form: dict[str, dict[str, dict[str, Any]]]) -> str:
    records = [
        {"form": form, "image_path": image, "record": record}
        for form in sorted(by_form)
        for image, record in sorted(by_form[form].items())
    ]
    return content_hash(records)


def load_exclusions(manifests: list[Path]) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    hashes: set[str] = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = manifest.get("selected")
        if not isinstance(selected, list):
            raise SystemExit(f"exclude manifest has no selected list: {manifest_path}")
        for item in selected:
            image = item.get("image_path")
            if not isinstance(image, str) or not image:
                raise SystemExit(f"exclude manifest has an item without image_path: {manifest_path}")
            paths.add(normalized_path(image))
            digest = item.get("image_sha256")
            if isinstance(digest, str) and digest:
                hashes.add(digest)
                continue
            image_path = Path(image)
            if not image_path.is_file():
                raise SystemExit(
                    "cannot verify exclusion image hash; file is absent and manifest "
                    f"has no image_sha256: {image}"
                )
            hashes.add(file_hash(image_path))
    return paths, hashes


def eligible_images(
    by_form: dict[str, dict[str, dict[str, Any]]],
    excluded_paths: set[str],
    excluded_hashes: set[str],
) -> tuple[set[str], dict[str, str]]:
    common = set.intersection(*(set(by_form[form]) for form in REQUIRED_FORMS))
    image_hashes: dict[str, str] = {}
    eligible: set[str] = set()
    for image in sorted(common):
        path = Path(image)
        if not path.is_file():
            raise SystemExit(f"audit source image does not exist: {image}")
        digest = file_hash(path)
        image_hashes[image] = digest
        if normalized_path(image) in excluded_paths or digest in excluded_hashes:
            continue
        eligible.add(image)
    return eligible, image_hashes


def select_images(
    by_form: dict[str, dict[str, dict[str, Any]]],
    n: int,
    seed: int,
    excluded_paths: set[str] | None = None,
    excluded_hashes: set[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    eligible, image_hashes = eligible_images(
        by_form, excluded_paths or set(), excluded_hashes or set()
    )
    if len(eligible) < n:
        raise SystemExit(f"only {len(eligible)} non-overlapping images have all eight forms; requested {n}")
    by_scene: dict[str, list[str]] = defaultdict(list)
    for image in eligible:
        scene = by_form["bbox"][image].get("scene_type", "unknown")
        by_scene[scene].append(image)
    for images in by_scene.values():
        images.sort(key=lambda value: stable_key(seed, value))
    selected: list[str] = []
    scenes = sorted(by_scene)
    while len(selected) < n:
        progressed = False
        for scene in scenes:
            if by_scene[scene] and len(selected) < n:
                selected.append(by_scene[scene].pop(0))
                progressed = True
        if not progressed:
            break
    return selected, {image: image_hashes[image] for image in selected}


def write_root(
    root: Path,
    by_form: dict[str, dict[str, dict[str, Any]]],
    images: list[str],
    audit_id: str,
) -> None:
    for form in REQUIRED_FORMS:
        records = [by_form[form][image] for image in images]
        pipeline = records[0]["pipeline"]
        if any(record.get("pipeline") != pipeline for record in records):
            raise SystemExit(f"form {form} spans multiple pipelines")
        out = root / "labels" / pipeline / f"{form}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "adapter_stats.json").write_text(
        json.dumps({
            "semantic_audit": True,
            "schema_version": 3,
            "audit_id": audit_id,
            "images": len(images),
            "forms": len(REQUIRED_FORMS),
        }, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/viewer_v3_pilot_clean"))
    parser.add_argument("--out", type=Path, default=Path("data/semantic_audit_v3"))
    parser.add_argument("--images", type=int, default=REQUIRED_IMAGES)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--round", default="v3")
    parser.add_argument("--audit-id")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()

    if args.images != REQUIRED_IMAGES:
        raise SystemExit(f"v3 protocol requires exactly {REQUIRED_IMAGES} images")
    if not isinstance(args.round, str) or not args.round.strip():
        raise SystemExit("--round must be non-empty")
    ensure_empty_output(args.out)
    by_form = load_forms(args.source)
    fingerprint = source_fingerprint(by_form)
    excluded_paths, excluded_hashes = load_exclusions(args.exclude_manifest)
    images, image_hashes = select_images(
        by_form, args.images, args.seed, excluded_paths, excluded_hashes
    )
    audit_id = args.audit_id or f"semantic-audit-{args.round}-{fingerprint[:12]}"

    for form in REQUIRED_FORMS:
        pipelines = {by_form[form][image].get("pipeline") for image in images}
        if len(pipelines) != 1 or not all(isinstance(value, str) and value for value in pipelines):
            raise SystemExit(f"form {form} spans invalid or multiple pipelines: {sorted(map(str, pipelines))}")

    sample_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    form_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for image in images:
        item: dict[str, Any] = {
            "image_path": image,
            "image_sha256": image_hashes[image],
            "sample_ids": {},
            "label_hashes": {},
            "label_versions": {},
            "prompt_versions": {},
        }
        for form in REQUIRED_FORMS:
            record = by_form[form][image]
            sample_id = record["sample_id"]
            if sample_id in sample_ids:
                raise SystemExit(f"duplicate sample_id across selected labels: {sample_id}")
            sample_ids.add(sample_id)
            digest = content_hash(record)
            item["sample_ids"][form] = sample_id
            item["label_hashes"][form] = digest
            item["label_versions"][form] = record["label_version"]
            item["prompt_versions"][form] = record["prompt_version"]
            form_records[form].append({"sample_id": sample_id, "label_sha256": digest})
        selected.append(item)

    per_form_label_hashes = {
        form: content_hash(form_records[form]) for form in REQUIRED_FORMS
    }
    scene_counts = Counter(by_form["bbox"][image].get("scene_type", "unknown") for image in images)
    source_counts = Counter(by_form["bbox"][image].get("source_bucket", "unknown") for image in images)
    roots = [args.out / "annotator_a", args.out / "annotator_b"]
    manifest = {
        "schema_version": 3,
        "round": args.round,
        "audit_id": audit_id,
        "seed": args.seed,
        "source_fingerprint": fingerprint,
        "source_records": sum(len(records) for records in by_form.values()),
        "exclude_manifests": [str(path) for path in args.exclude_manifest],
        "images": len(images),
        "forms": list(REQUIRED_FORMS),
        "records_per_annotator": len(images) * len(REQUIRED_FORMS),
        "roots": [str(root) for root in roots],
        "per_form_label_hashes": per_form_label_hashes,
        "scene_counts": dict(sorted(scene_counts.items())),
        "source_bucket_counts": dict(sorted(source_counts.items())),
        "selected": selected,
    }

    # All validation above is complete. Only now is it safe to create the round.
    args.out.mkdir(parents=True, exist_ok=True)
    for root in roots:
        write_root(root, by_form, images, audit_id)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_fields = (
        "audit_id", "round", "source_fingerprint", "images", "forms",
        "records_per_annotator", "scene_counts",
    )
    print(json.dumps({key: manifest[key] for key in summary_fields}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
