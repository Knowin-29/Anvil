#!/usr/bin/env python3
"""Create a train/dev/qualification split for the scoped G-family pilot.

The v1 release has 32 train images and eight complete validation images.  No G
checkpoint has been trained from that release, so v2 freezes those eight complete
images as qualification-only and selects eight complete images from the old train
pool as dev.  The remaining 24 images are the only gradient-update split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_PROTOCOL = "semantic-audit-v3-g-family-scoped-v1"
PROTOCOL = "semantic-audit-v3-g-family-scoped-v2-independent-qualification"
CORPUS_VERSION = "gsa-bench-v3-g-family-scoped-v2"
FORMS = ("bbox", "detection", "mask", "point")
SEED = 20260820


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


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _rank(image_sha256: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{image_sha256}".encode("ascii")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"refusing to overwrite different split: {path}")
    if path.exists():
        return
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def materialize(source: Path, out: Path, seed: int = SEED) -> dict[str, Any]:
    source = source.resolve()
    out = out.resolve()
    source_manifest_path = source / "manifest.json"
    source_manifest = load_json(source_manifest_path)
    if source_manifest.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("source corpus protocol drift")
    if source_manifest.get("counts") != {"train": 116, "val": 32}:
        raise ValueError("source corpus is not the frozen 116/32 release")
    if source_manifest.get("unique_images") != {"train": 32, "val": 8}:
        raise ValueError("source corpus is not the frozen 32/8 image release")
    if source_manifest.get("global_audit_verdict") != "BLOCKED":
        raise ValueError("source corpus did not preserve global BLOCKED")
    if source_manifest.get("g_family_scoped_release_verdict") != "G_ONLY_PASS":
        raise ValueError("source G-only release is not eligible")

    source_files: dict[str, Path] = {}
    for key in ("train_G_only", "val_G_only"):
        spec = source_manifest["corpus_files"][key]
        path = (source / spec["path"]).resolve()
        if file_hash(path) != spec["sha256"]:
            raise ValueError(f"source corpus hash drift: {key}")
        source_files[key] = path
    old_train = load_jsonl(source_files["train_G_only"])
    old_val = load_jsonl(source_files["val_G_only"])
    if len(old_train) != 116 or len(old_val) != 32:
        raise ValueError("source corpus row-count drift")

    train_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qual_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in old_train:
        train_by_image[row["gsa_meta"]["image_sha256"]].append(row)
    for row in old_val:
        qual_by_image[row["gsa_meta"]["image_sha256"]].append(row)
    if len(train_by_image) != 32 or len(qual_by_image) != 8:
        raise ValueError("source image counts drift")
    if set(train_by_image) & set(qual_by_image):
        raise ValueError("source train/validation image overlap")
    for image_sha, rows in qual_by_image.items():
        if sorted(row["gsa_form"] for row in rows) != sorted(FORMS):
            raise ValueError(f"qualification candidate is not form-complete: {image_sha}")

    complete_train_images = [
        image_sha for image_sha, rows in train_by_image.items()
        if sorted(row["gsa_form"] for row in rows) == sorted(FORMS)
    ]
    if len(complete_train_images) < 8:
        raise ValueError("fewer than eight form-complete dev candidates")
    dev_images = set(sorted(complete_train_images, key=lambda value: (_rank(value, seed), value))[:8])
    qualification_images = set(qual_by_image)
    gradient_train_images = set(train_by_image) - dev_images
    if (len(gradient_train_images), len(dev_images), len(qualification_images)) != (24, 8, 8):
        raise ValueError("v2 image split did not produce 24/8/8")

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "qualification": []}
    for source_row in old_train + old_val:
        image_sha = source_row["gsa_meta"]["image_sha256"]
        if image_sha in qualification_images:
            split = "qualification"
        elif image_sha in dev_images:
            split = "dev"
        else:
            split = "train"
        row = json.loads(json.dumps(source_row))
        row["gsa_meta"]["split"] = split
        row["gsa_meta"]["source_release_protocol"] = SOURCE_PROTOCOL
        row["gsa_meta"]["release_protocol"] = PROTOCOL
        row["gsa_meta"]["corpus_version"] = CORPUS_VERSION
        row["gsa_meta"]["qualification_exposure"] = (
            "never_train_or_dev" if split == "qualification" else "not_qualification"
        )
        splits[split].append(row)
    for split in splits:
        splits[split].sort(
            key=lambda row: (row["gsa_meta"]["image_sha256"], FORMS.index(row["gsa_form"]), row["sample_id"])
        )

    if sum(len(rows) for rows in splits.values()) != 148:
        raise ValueError("v2 did not preserve all 148 accepted records")
    if len(splits["dev"]) != 32 or len(splits["qualification"]) != 32:
        raise ValueError("dev and qualification must each have eight rows per form")
    all_ids = [row["sample_id"] for rows in splits.values() for row in rows]
    if len(set(all_ids)) != 148:
        raise ValueError("v2 sample IDs are not an exact partition")

    out.mkdir(parents=True, exist_ok=True)
    names = {
        "train": "train_G_only.jsonl",
        "dev": "dev_G_only.jsonl",
        "qualification": "qualification_G_only.jsonl",
    }
    for split, name in names.items():
        _write_jsonl(out / name, splits[split])

    image_lists = {
        "train": sorted(gradient_train_images),
        "dev": sorted(dev_images),
        "qualification": sorted(qualification_images),
    }
    by_form = {
        split: dict(sorted(Counter(row["gsa_form"] for row in rows).items()))
        for split, rows in splits.items()
    }
    corpus_files = {
        f"{split}_G_only": {
            "path": names[split],
            "sha256": file_hash(out / names[split]),
            "records": len(splits[split]),
        }
        for split in names
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "source_release_protocol": SOURCE_PROTOCOL,
        "corpus_version": CORPUS_VERSION,
        "release_scope": "G_family_only_independent_qualification_pilot",
        "family": "G",
        "forms": list(FORMS),
        "authorization": "G_specialist_qualification_only",
        "forbidden_uses": [
            "global_semantic_audit_pass_claim",
            "joint_training",
            "mopd_training",
            "S_or_A_specialist_training",
            "qualification_rows_as_training_or_dev",
            "publication_claim_from_eight_qualification_images",
        ],
        "g_family_scoped_release_verdict": "G_ONLY_PASS",
        "global_audit_verdict": "BLOCKED",
        "label_policy": source_manifest.get("label_policy"),
        "contract_version": source_manifest.get("contract_version"),
        "accepted_records": 148,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "by_form": by_form,
        "unique_images": {split: len(values) for split, values in image_lists.items()},
        "overlap": {
            "train_dev": len(set(image_lists["train"]) & set(image_lists["dev"])),
            "train_qualification": len(set(image_lists["train"]) & set(image_lists["qualification"])),
            "dev_qualification": len(set(image_lists["dev"]) & set(image_lists["qualification"])),
        },
        "image_split": {
            "policy": "v1-val-as-qualification_complete-dev-hashrank-v1",
            "seed": seed,
            "dev_selection": "eight lowest sha256(seed:image_sha256) among form-complete v1 train images",
            "qualification_selection": "all eight form-complete images from frozen v1 val",
            "train_image_sha256": image_lists["train"],
            "dev_image_sha256": image_lists["dev"],
            "qualification_image_sha256": image_lists["qualification"],
            "sha256": content_hash(image_lists),
        },
        "exposure_policy": {
            "train": "gradient_updates_only",
            "dev": "trainer_eval_and_checkpoint_selection_only",
            "qualification": "post_training_model_comparison_only",
            "qualification_seen_by_training_launcher": False,
        },
        "publication_claim_eligible": False,
        "publication_blocker": "qualification contains only eight unique images",
        "corpus_files": corpus_files,
        "source_bindings": {
            "manifest": binding(source_manifest_path),
            "train_G_only": binding(source_files["train_G_only"]),
            "val_G_only": binding(source_files["val_G_only"]),
        },
        "audit_bindings": source_manifest.get("audit_bindings"),
        "notes": (
            "All 148 accepted v1 records and all 40 audited images are preserved. "
            "Only train/dev are visible to the v2 trainer; qualification remains an "
            "eight-image engineering pilot and is not publication evidence."
        ),
    }
    manifest["partition_sample_ids_sha256"] = content_hash(sorted(all_ids))
    manifest_path = out / "manifest.json"
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != text:
        raise ValueError(f"refusing to overwrite different manifest: {manifest_path}")
    if not manifest_path.exists():
        temp = manifest_path.with_suffix(f".tmp.{os.getpid()}")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("trains/gsa_bench_v3_g_family_scoped_v1"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("trains/gsa_bench_v3_g_family_scoped_v2"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    try:
        manifest = materialize(args.source, args.out, args.seed)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "READY",
        "protocol": manifest["protocol"],
        "counts": manifest["counts"],
        "unique_images": manifest["unique_images"],
        "by_form": manifest["by_form"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
