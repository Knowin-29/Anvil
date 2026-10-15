#!/usr/bin/env python3
"""Minimal, dependency-free command surface for Anvil experiment preparation."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def _images(row: dict) -> list[str]:
    values = row.get("images") or row.get("image") or []
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values if value]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_corpus(args: argparse.Namespace) -> int:
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"input corpus is missing: {source}")
    if not 1 <= args.dev_permille <= 500:
        raise SystemExit("--dev-permille must be in [1, 500]")

    targets = [output / "train.jsonl", output / "dev.jsonl", output / "manifest.json"]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite an existing split; pass --force only for a deliberate rebuild: "
            + ", ".join(map(str, existing))
        )
    output.mkdir(parents=True, exist_ok=True)

    union_find = UnionFind()
    source_digest = hashlib.sha256()
    source_rows = 0
    missing_images = 0
    with source.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            source_digest.update(line)
            try:
                row = json.loads(line)
            except Exception as exc:
                raise SystemExit(f"invalid JSON at {source}:{line_number}: {exc}") from exc
            images = _images(row)
            if not images:
                missing_images += 1
                continue
            source_rows += 1
            first = images[0]
            union_find.find(first)
            for image in images[1:]:
                union_find.union(first, image)
    if missing_images:
        raise SystemExit(
            f"refusing to split {missing_images} row(s) without image identity; "
            "repair or exclude them first"
        )

    components: dict[str, list[str]] = collections.defaultdict(list)
    for image in union_find.parent:
        components[union_find.find(image)].append(image)
    component_key = {
        root: min(images) for root, images in components.items()
    }

    def is_dev(image: str) -> bool:
        root = union_find.find(image)
        key = f"{args.salt}\0{component_key[root]}".encode()
        bucket = int(hashlib.sha256(key).hexdigest()[:16], 16) % 1000
        return bucket < args.dev_permille

    temp_train = output / ".train.jsonl.tmp"
    temp_dev = output / ".dev.jsonl.tmp"
    counts = {
        "train": collections.Counter(),
        "dev": collections.Counter(),
    }
    image_sets = {"train": set(), "dev": set()}
    row_counts = collections.Counter()
    try:
        with source.open("rb") as source_handle, temp_train.open("wb") as train_handle, temp_dev.open("wb") as dev_handle:
            for line_number, line in enumerate(source_handle, 1):
                row = json.loads(line)
                images = _images(row)
                assignments = {is_dev(image) for image in images}
                if len(assignments) != 1:
                    raise SystemExit(
                        f"connected-component split invariant failed at row {line_number}"
                    )
                split = "dev" if assignments.pop() else "train"
                (dev_handle if split == "dev" else train_handle).write(line)
                row_counts[split] += 1
                image_sets[split].update(images)
                form = str((row.get("gsa_meta") or {}).get("form") or "unknown")
                counts[split][form] += 1
        if image_sets["train"] & image_sets["dev"]:
            raise SystemExit("train/dev image overlap detected after split")
        missing_forms = set(counts["train"]) - set(counts["dev"])
        if missing_forms:
            raise SystemExit(f"dev split is missing forms: {sorted(missing_forms)}")
        os.replace(temp_train, output / "train.jsonl")
        os.replace(temp_dev, output / "dev.jsonl")
    finally:
        temp_train.unlink(missing_ok=True)
        temp_dev.unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "split_id": args.split_id,
        "builder": "engineering/anvil.py split-corpus",
        "source": {
            "path": str(args.input),
            "rows": source_rows,
            "sha256": source_digest.hexdigest(),
        },
        "rule": {
            "unit": "connected component of shared image paths",
            "assignment": "sha256(salt + NUL + lexicographically-smallest-image) mod 1000",
            "salt": args.salt,
            "dev_permille": args.dev_permille,
        },
        "train": {
            "path": str(args.output / "train.jsonl"),
            "rows": row_counts["train"],
            "unique_images": len(image_sets["train"]),
            "by_form": dict(sorted(counts["train"].items())),
            "sha256": _sha256(output / "train.jsonl"),
        },
        "dev": {
            "path": str(args.output / "dev.jsonl"),
            "rows": row_counts["dev"],
            "unique_images": len(image_sets["dev"]),
            "by_form": dict(sorted(counts["dev"].items())),
            "sha256": _sha256(output / "dev.jsonl"),
        },
        "checks": {
            "row_conservation": row_counts["train"] + row_counts["dev"] == source_rows,
            "image_disjoint": not bool(image_sets["train"] & image_sets["dev"]),
            "all_forms_in_dev": not bool(set(counts["train"]) - set(counts["dev"])),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


def prepare_scannet_flywheel(args: argparse.Namespace) -> int:
    """Select bounded RGB inputs from verified official-train extraction manifests."""
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing an existing preparation output: {output}")
    if not 1 <= args.dev_permille <= 500 or args.max_image_bytes < 1:
        raise SystemExit("invalid development fraction or image size limit")
    source_root = args.source_root.resolve()
    split_root = args.split_root.resolve()
    splits = {}
    split_bindings = {}
    for name in ("train", "val", "test"):
        path = split_root / f"scannetv2_{name}.txt"
        splits[name] = set(path.read_text().split())
        split_bindings[name] = {"sha256": _sha256(path), "scenes": len(splits[name])}
    groups = {name: {scene.split("_")[0] for scene in scenes} for name, scenes in splits.items()}
    if groups["train"] & (groups["val"] | groups["test"]):
        raise SystemExit("official train and held-out splits share a physical scene group")

    records = []
    source_bindings = []
    seen_scenes = set()
    rejected = collections.Counter()
    manifests = sorted(source_root.glob("shard_*/scaleout_manifest.json"))
    if not manifests:
        raise SystemExit("no ScanNet scaleout manifests found")
    for manifest_path in manifests:
        digest = _sha256(manifest_path)
        pinned = manifest_path.with_suffix(".sha256").read_text().split()[0]
        if pinned != digest:
            raise SystemExit(f"source manifest hash drift: {manifest_path}")
        document = json.loads(manifest_path.read_text())
        policy = document.get("use_policy") or {}
        if (document.get("protocol") != "gsa-scannet-train-sens-cpu-scaleout-v1"
                or document.get("verdict") != "SCALEOUT_VALIDATED_TRAIN_SPLIT_ONLY"
                or policy.get("training_materialization_eligible") is not True
                or policy.get("official_train_only") is not True
                or policy.get("official_split") != "train"):
            raise SystemExit(f"source is not eligible official-train material: {manifest_path}")
        for name, binding in split_bindings.items():
            if document["official_splits"][name]["sha256"] != binding["sha256"]:
                raise SystemExit(f"source split binding drift: {name}")
        source_bindings.append({"path": str(manifest_path), "sha256": digest})
        for scene in document["scenes"]:
            scene_id = scene["scene_id"]
            if (not re.fullmatch(r"scene\d{4}_\d{2}", scene_id)
                    or scene_id not in splits["train"] or scene.get("official_split") != "train"
                    or scene_id in seen_scenes):
                raise SystemExit(f"invalid, held-out or duplicate scene: {scene_id}")
            seen_scenes.add(scene_id)
            scene_root = manifest_path.parent / scene_id
            frames = sorted(scene["frames"], key=lambda frame: hashlib.sha256(
                f"{args.salt}\0{scene_id}\0{frame['frame_index']}".encode()).hexdigest())
            selected = None
            for frame in frames:
                binding = frame["outputs"]["color.png"]
                relative = Path(binding["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise SystemExit("image path escapes its scene")
                image = (scene_root / relative).resolve()
                image.relative_to(scene_root.resolve())
                size = image.stat().st_size
                if size != binding["size_bytes"]:
                    raise SystemExit(f"source image size drift: {image}")
                if size > args.max_image_bytes:
                    rejected["over_upload_size_limit"] += 1
                    continue
                if _sha256(image) != binding["sha256"]:
                    raise SystemExit(f"source image hash drift: {image}")
                selected = (frame, image, binding["sha256"])
                break
            if selected is None:
                rejected["scene_without_usable_image"] += 1
                continue
            frame, image, image_sha = selected
            group = scene_id.split("_")[0]
            bucket = int(hashlib.sha256(f"{args.salt}\0{group}".encode()).hexdigest()[:16], 16) % 1000
            split = "dev" if bucket < args.dev_permille else "train"
            records.append({
                "images": [str(image)], "image_sha256": image_sha,
                "scene_id": scene_id, "scene_group": group, "frame_index": frame["frame_index"],
                "source": "scannet_official_train", "split": split,
                "label_provenance": "unlabelled_source",
            })
    records.sort(key=lambda row: hashlib.sha256(
        f"{args.salt}\0{row['scene_id']}".encode()).hexdigest())
    partitions = {name: [row for row in records if row["split"] == name] for name in ("train", "dev")}
    if not all(partitions.values()):
        raise SystemExit("both train and development partitions must be nonempty")
    # Duplicate bytes can cross scan roots; never silently split them across the two arms.
    identities = {name: {row["image_sha256"] for row in rows} for name, rows in partitions.items()}
    if identities["train"] & identities["dev"]:
        raise SystemExit("train/development images share byte identity")
    output.mkdir(parents=True)
    for name, rows in partitions.items():
        (output / f"{name}.index").write_text(
            "".join(row["images"][0] + "\tscannet_indoor\n" for row in rows))
    (output / "images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    manifest = {
        "schema": "anvil-scannet-flywheel-input-v1", "builder_sha256": _sha256(Path(__file__)),
        "source_manifests": source_bindings, "official_splits": split_bindings,
        "selection": {"images_per_scan": 1, "order": "sha256(salt, scene, frame)", "salt": args.salt,
                      "max_image_bytes": args.max_image_bytes},
        "split_rule": {"unit": "sceneXXXX physical scene group", "dev_permille": args.dev_permille},
        "images": len(records), "physical_scene_groups": len({row["scene_group"] for row in records}),
        "partitions": {name: {"images": len(rows), "scene_groups": len({r["scene_group"] for r in rows}),
                              "index_sha256": _sha256(output / f"{name}.index")}
                       for name, rows in partitions.items()},
        "unlabelled_corpus_sha256": _sha256(output / "images.jsonl"), "rejections": dict(rejected),
        "use_policy": {"generation_input_only": True, "benchmark_ground_truth": False,
                       "training_labels_eligible": False, "redistribution_eligible": False,
                       "remaining_checks": ["cross-benchmark overlap", "generated-label semantic review",
                                            "signed ScanNet terms archival before public release"]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("schema", "images", "physical_scene_groups", "partitions", "rejections")}, indent=2))
    return 0


def filter_corpus(args: argparse.Namespace) -> int:
    """Quarantine complete named source roots in a new byte-preserving derivative."""
    source, output = args.input.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing an existing filtered corpus: {output}")
    roots = [Path(os.path.abspath(value)) for value in args.exclude_image_root]
    if any(len(root.parts) < 5 for root in roots):
        raise SystemExit("exclusions must name a dataset root, not a broad filesystem root")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    excluded_by_root = collections.Counter()
    kept_images, removed_images = set(), set()
    forms = collections.Counter()
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix=".anvil-filter-", dir=output.parent) as temp_name:
        temporary = Path(temp_name)
        target = temporary / "train.jsonl"
        with source.open("rb") as input_handle, target.open("wb") as output_handle:
            for line in input_handle:
                digest.update(line)
                row = json.loads(line)
                images = _images(row)
                if not images:
                    raise SystemExit("input record lacks image identity")
                counts["input_rows"] += 1
                matching = {str(root) for root in roots for image in images
                            if Path(os.path.abspath(image)).is_relative_to(root)}
                if matching:
                    counts["removed_rows"] += 1
                    removed_images.update(images)
                    excluded_by_root.update(matching)
                    continue
                output_handle.write(line)
                counts["kept_rows"] += 1
                kept_images.update(images)
                forms[str(row.get("gsa_form") or (row.get("gsa_meta") or {}).get("form") or "unknown")] += 1
        if digest.hexdigest() != args.expected_sha256 or counts["input_rows"] != args.expected_rows:
            raise SystemExit("input corpus hash or row count differs from the bound audit")
        if any(not excluded_by_root[str(root)] for root in roots):
            raise SystemExit("an exclusion root matched no records; check the exact source path")
        manifest = {
            "schema": "anvil-source-filter-v1", "builder_sha256": _sha256(Path(__file__)),
            "source": {"path": str(source), "sha256": digest.hexdigest(), "rows": counts["input_rows"]},
            "exclusions": {"rule": "any image path under a named root removes the entire row",
                           "roots": [str(root) for root in roots], "rows_by_root": dict(excluded_by_root),
                           "rows": counts["removed_rows"], "images_in_removed_rows": len(removed_images)},
            "corpus": {"rows": counts["kept_rows"], "unique_images": len(kept_images),
                       "sha256": _sha256(target), "by_form": dict(sorted(forms.items()))},
            "status": "PENDING_LEAKAGE_REAUDIT", "source_unchanged": _sha256(source) == digest.hexdigest(),
        }
        if not manifest["source_unchanged"]:
            raise SystemExit("source changed during filtering")
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        output.mkdir()
        os.replace(target, output / "train.jsonl")
        os.replace(temporary / "manifest.json", output / "manifest.json")
    print(json.dumps(manifest, indent=2))
    return 0


def _sensor_mask_polygon(mask, *, min_iou=.95, max_vertices=48):
    """Encode one visible instance as a simple polygon and verify its pixel support.

    A single polygon cannot represent substantial holes or disconnected material.
    Reject those cases instead of filling them with background supervision. Scores
    describe fidelity to the official projection, not human visual correctness.
    """
    import cv2
    import numpy as np

    if not .95 <= min_iou <= 1 or not 6 <= max_vertices <= 128:
        raise ValueError("polygon fidelity must be in [.95,1] and vertex budget in [6,128]")
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2 or not mask.any() or not np.isin(mask, [0, 1]).all():
        raise ValueError("sensor mask must be a nonempty binary image")
    _, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    dominant = (components == component).astype(np.uint8)
    component_fraction = float(dominant.sum()) / int(mask.sum())
    evidence = {"dominant_component_fraction": component_fraction}
    if component_fraction < .98:
        return None, {**evidence, "reason": "disconnected_instance"}
    contours, _ = cv2.findContours(dominant, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
    hole_fraction = float(np.logical_and(filled, ~dominant.astype(bool)).sum()) / max(1, int(filled.sum()))
    evidence["filled_hole_fraction"] = hole_fraction
    if hole_fraction > .02:
        return None, {**evidence, "reason": "unrepresentable_holes"}

    def simple(poly):
        def cross(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def on(a, b, c):
            return cross(a, b, c) == 0 and all(min(a[k], b[k]) <= c[k] <= max(a[k], b[k]) for k in (0, 1))

        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                c, d = poly[j], poly[(j + 1) % n]
                if (cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0
                        or on(a, b, c) or on(a, b, d) or on(c, d, a) or on(c, d, b)):
                    return False
        return True

    height, width = mask.shape
    perimeter = cv2.arcLength(contour, True)
    candidates = []
    for fraction in (.0005, .001, .002, .003, .005, .0075, .01, .015, .02):
        vertices = cv2.approxPolyDP(contour, fraction * perimeter, True).reshape(-1, 2)
        poly = np.clip(np.rint(vertices * np.array([1000 / width, 1000 / height])), 0, 999).astype(int).tolist()
        poly = [point for index, point in enumerate(poly) if point != poly[index - 1]]
        if not 6 <= len(poly) <= max_vertices or len({tuple(p) for p in poly}) != len(poly) or not simple(poly):
            continue
        pixels = np.rint(np.array(poly) * np.array([width / 1000, height / 1000])).astype(np.int32)
        raster = np.zeros_like(mask)
        cv2.fillPoly(raster, [pixels], 1)
        intersection = int(np.logical_and(raster, mask).sum())
        union = int(np.logical_or(raster, mask).sum())
        iou = intersection / union
        if iou >= min_iou:
            candidates.append((len(poly), -iou, poly, {"polygon_raster_iou": iou,
                "polygon_precision": intersection / int(raster.sum()),
                "polygon_recall": intersection / int(mask.sum()), "vertices": len(poly),
                "approximation_perimeter_fraction": fraction}))
    if not candidates:
        return None, {**evidence, "reason": "no_faithful_compact_polygon"}
    _, _, polygon, measurement = min(candidates, key=lambda item: item[:3])
    return polygon, {**evidence, **measurement, "reason": None}


def prepare_jacquard_references(args: argparse.Namespace) -> int:
    """Bind source-native grasp references without relabelling centers as contacts."""
    import io
    import zipfile
    import numpy as np
    from PIL import Image

    source, output = args.archive.resolve(), args.output.resolve()
    if output.exists() or _sha256(source) != args.expected_sha256:
        raise ValueError("Jacquard preparation requires a fresh output and the exact bound archive")
    suffixes = ("_RGB.png", "_mask.png", "_perfect_depth.tiff", "_stereo_depth.tiff", "_grasps.txt")
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if len(set(names)) != len(names):
            raise ValueError("duplicate Jacquard archive members")
        # Validate the entire archive before any extraction, including paths not
        # selected below. A source hash is a binding, not a path-safety guarantee.
        for item in members:
            path = Path(item.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in item.filename
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("unsafe Jacquard archive member")
        if archive.testzip() is not None:
            raise ValueError("Jacquard archive CRC failure")
        prefixes = sorted(name[:-len("_grasps.txt")] for name in names if name.endswith("_grasps.txt"))
        if not prefixes:
            raise ValueError("Jacquard archive has no grasp annotations")
        objects = sorted({Path(prefix).parent.name for prefix in prefixes},
                         key=lambda value: hashlib.sha256(f"{args.seed}:{value}".encode()).hexdigest())
        if not 1 <= args.dev_objects < len(objects):
            raise ValueError("reserve at least one development object and one training object")
        dev_objects = set(objects[:args.dev_objects])
        counts, records, rgb_owners = collections.Counter(), [], {}
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".jacquard-", dir=output.parent) as temporary:
            staged = Path(temporary)
            for prefix in prefixes:
                object_id = Path(prefix).parent.name
                scene_id = Path(prefix).name
                if not re.fullmatch(r"[0-4]_" + re.escape(object_id), scene_id):
                    raise ValueError("Jacquard scene prefix does not match its object identity")
                blobs = {suffix: archive.read(prefix + suffix) for suffix in suffixes}
                with Image.open(io.BytesIO(blobs["_RGB.png"])) as image:
                    rgb = np.array(image.convert("RGB"))
                mask = np.array(Image.open(io.BytesIO(blobs["_mask.png"])))
                height, width = rgb.shape[:2]
                if mask.shape != (height, width) or not np.isin(mask, [0, 255]).all() or not mask.any():
                    raise ValueError("Jacquard binary mask is not aligned with its RGB")
                depth_diagnostics = {}
                for suffix in ("_perfect_depth.tiff", "_stereo_depth.tiff"):
                    depth = np.array(Image.open(io.BytesIO(blobs[suffix])))
                    if depth.shape != mask.shape or depth.dtype != np.float32 or not np.isfinite(depth).all():
                        raise ValueError("Jacquard depth is not finite aligned float32")
                    depth_diagnostics[suffix] = {"positive_pixels": int((depth > 0).sum()),
                                                "nonpositive_pixels": int((depth <= 0).sum())}
                grasps = np.loadtxt(io.BytesIO(blobs["_grasps.txt"]), delimiter=";", ndmin=2)
                if (grasps.shape[1] != 5 or not len(grasps) or not np.isfinite(grasps).all()
                        or (grasps[:, 3:] <= 0).any()):
                    raise ValueError("Jacquard grasp rows must have five finite values and positive jaw dimensions")
                xy = np.rint(grasps[:, :2]).astype(np.int64)
                inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
                foreground = np.zeros(len(xy), dtype=bool)
                foreground[inside] = mask[xy[inside, 1], xy[inside, 0]] > 0
                scene_counts = {"scenes": 1, "grasp_rows": len(grasps),
                                "distinct_xy_theta_opening": len(np.unique(grasps[:, :4], axis=0)),
                                "centers_outside_image": int((~inside).sum()),
                                "centers_on_background": int((inside & ~foreground).sum()),
                                "scenes_with_background_centers": int(bool((inside & ~foreground).any()))}
                # Background centers are retained. They are not necessarily bad
                # grasps: a gripper center can lie between two material contacts.
                decoded_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                if decoded_sha in rgb_owners and rgb_owners[decoded_sha] != object_id:
                    raise ValueError("identical decoded RGB assigned to distinct Jacquard objects")
                rgb_owners[decoded_sha] = object_id
                bindings = {}
                for suffix, raw in blobs.items():
                    relative = Path("source") / (prefix + suffix)
                    path = staged / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as handle:
                        handle.write(raw)
                    bindings[suffix] = {"path": str(relative), "archive_member": prefix + suffix,
                                        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                records.append({"sample_id": "jacquard_" + scene_id, "object_id": object_id,
                    "scene_id": scene_id, "split": "dev" if object_id in dev_objects else "train",
                    "width": width, "height": height, "decoded_rgb_sha256": decoded_sha,
                    "files": bindings, "source_grasps_xy_theta_opening_jaw_size": grasps.tolist(),
                    "coordinate_diagnostic": scene_counts, "depth_diagnostic": depth_diagnostics,
                    "training_eligible": False, "teacher_eligible": False,
                    "native_A_reference_coverage": "not_established"})
                counts.update(scene_counts)
            if _sha256(source) != args.expected_sha256:
                raise ValueError("Jacquard source archive changed during preparation")
            references = staged / "source_references.jsonl"
            references.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))
            manifest = {"schema": "anvil-jacquard-source-references-v1", "status": "SOURCE_BOUND_NATIVE_MAPPING_PENDING",
                "builder_sha256": _sha256(Path(__file__)), "archive": {"path": str(source),
                    "bytes": source.stat().st_size, "sha256": args.expected_sha256, "crc": "PASS"},
                "source_documentation": "https://jacquard.liris.cnrs.fr/database.php",
                "access_and_redistribution": "Public samples acquired; full-data access and redistribution terms remain separate.",
                "objects": len(objects), "counts": dict(counts),
                "split_rule": {"unit": "source_object_id", "seed": args.seed, "dev_objects": sorted(dev_objects),
                               "train_objects": sorted(set(objects) - dev_objects)},
                "partitions": dict(collections.Counter(row["split"] for row in records)),
                "references": {"path": references.name, "sha256": _sha256(references), "rows": len(records)},
                "coordinates": {"columns": ["center_x_px", "center_y_px", "theta_source_deg", "opening_px", "jaw_size_px"],
                    "image_y": "down", "angles": "Unmodified source degrees; account for image-axis convention before conversion.",
                    "background_test": "np.rint(center_xy), original binary mask; repeated jaw sizes retained",
                    "native_mapping": "No conversion of gripper center to Anvil material-contact grasp_point_2d."},
                "admission": {"training": False, "teacher_qualification": False, "paper_metrics": False,
                    "complete_A_alternatives": False, "human_confirmation": False,
                    "limitations": ["Source-native simulation references do not establish complete action-plan semantics.",
                                    "Object partition is a preparation split, not an official benchmark split.",
                                    "Contamination screening and independent review are pending."]}}
            (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            if output.exists():
                raise ValueError("Jacquard destination appeared during preparation")
            staged.rename(output)
    print(json.dumps(manifest, indent=2))
    return 0


def prepare_coco_masks(args: argparse.Namespace) -> int:
    """Prepare native masks from official train2017 annotations, with RGB review pending."""
    import concurrent.futures
    import io
    import sys
    import urllib.request
    import zipfile
    import numpy as np
    from PIL import Image
    from pycocotools import mask as coco_mask

    sys.path.insert(0, str(args.worktree.resolve()))
    from trains.flywheel.contracts import context_for, grounding, task_columns, training_instruction
    from trains.flywheel.validators import validate

    output, cache = args.output.resolve(), args.image_cache.resolve()
    min_iou = getattr(args, "mask_min_iou", .95)
    max_vertices = getattr(args, "polygon_max_vertices", 48)
    min_side = getattr(args, "min_box_side_px", 20)
    if output.exists() or args.limit <= 0 or not 1 <= args.workers <= 16 or not 0 < args.dev_permille < 1000:
        raise ValueError("COCO preparation needs a fresh output, positive limit, 1–16 workers and a train/dev split")
    if not .95 <= min_iou <= 1 or not 6 <= max_vertices <= 128 or min_side < 20:
        raise ValueError("COCO fidelity/vertex/size constraints must meet the original minimums")
    if _sha256(args.annotations_zip) != args.expected_annotations_sha256:
        raise ValueError("official annotation archive binding drift")
    with zipfile.ZipFile(args.annotations_zip) as archive:
        # Never infer the training split from an arbitrary JSON filename or use
        # the validation annotations packaged in the same official archive.
        with archive.open("annotations/instances_train2017.json") as handle:
            data = json.load(handle)
    classes = {item["id"]: item["name"] for item in data["categories"]}
    images = {item["id"]: item for item in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError("duplicate COCO source image IDs")
    allowed = {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
               "banana", "apple", "orange", "broccoli", "carrot", "sandwich", "pizza",
               "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
               "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
               "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}
    requested_categories = getattr(args, "categories", None)
    if requested_categories is not None:
        if not requested_categories or len(set(requested_categories)) != len(requested_categories) or set(requested_categories) - allowed:
            raise ValueError("COCO categories must be a unique nonempty subset of the supported classes")
        allowed = set(requested_categories)
    excluded_images, excluded_rgb, exclusions = set(), set(), []
    for path in getattr(args, "exclude_corpus", []):
        path = path.resolve()
        bound_sha = _sha256(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError("COCO exclusion corpus is empty")
        for row in rows:
            meta = row.get("gsa_meta", {})
            if (meta.get("source") != "coco_official_train2017" or type(meta.get("source_image_id")) is not int
                    or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("decoded_rgb_sha256", "")))):
                raise ValueError("COCO exclusions require explicit official image and decoded RGB identities")
            excluded_images.add(meta["source_image_id"])
            excluded_rgb.add(meta["decoded_rgb_sha256"])
        if _sha256(path) != bound_sha:
            raise ValueError("COCO exclusion corpus changed while reading")
        exclusions.append({"path": str(path), "sha256": bound_sha, "rows": len(rows)})
    counts = collections.Counter((a["image_id"], a["category_id"]) for a in data["annotations"])
    pools, rejected = collections.defaultdict(list), collections.Counter()
    for annotation in data["annotations"]:
        label = classes[annotation["category_id"]]
        if label not in allowed:
            continue
        if annotation["image_id"] in excluded_images:
            rejected["previously_reserved_source_image"] += 1
            continue
        if (annotation.get("iscrowd") or counts[(annotation["image_id"], annotation["category_id"])] != 1
                or not isinstance(annotation.get("segmentation"), list)):
            rejected["crowd_or_ambiguous_category_or_nonpolygon"] += 1
            continue
        image = images[annotation["image_id"]]
        x, y, w, h = annotation["bbox"]
        if (not re.fullmatch(r"[0-9]{12}\.jpg", image["file_name"])
                or "/train2017/" not in image.get("coco_url", "")):
            raise ValueError("COCO image is not explicitly an official train2017 object")
        if (x < 2 or y < 2 or x + w >= image["width"] - 2 or y + h >= image["height"] - 2
                or min(w, h) < min_side or not .005 <= annotation["area"] / (image["width"] * image["height"]) <= .65):
            rejected["border_small_or_dominant_instance"] += 1
            continue
        pools[label].append(annotation)
    for pool in pools.values():
        pool.sort(key=lambda a: hashlib.sha256(f"{args.seed}:{a['id']}".encode()).hexdigest(), reverse=True)
    selected, seen = [], set()
    # Round-robin categories and fixed hashes prevent the canary from being
    # dominated by frequent easy classes. No model scores enter selection.
    while len(selected) < args.limit and any(pools.values()):
        for label in sorted(pools):
            if not pools[label] or len(selected) >= args.limit:
                continue
            annotation = pools[label].pop()
            image_id = annotation["image_id"]
            if image_id in seen:
                rejected["one_target_per_image"] += 1
                continue
            info = images[image_id]
            encoded = coco_mask.frPyObjects(annotation["segmentation"], info["height"], info["width"])
            # COCO decodes in Fortran order; OpenCV drawing requires C layout.
            mask = np.ascontiguousarray(coco_mask.decode(coco_mask.merge(encoded)))
            polygon, measurement = _sensor_mask_polygon(mask, min_iou=min_iou, max_vertices=max_vertices)
            if polygon is None:
                rejected["mask:" + measurement["reason"]] += 1
                continue
            x, y, w, h = annotation["bbox"]
            box = [round(1000 * x / info["width"]), round(1000 * y / info["height"]),
                   round(1000 * (x + w) / info["width"]), round(1000 * (y + h) / info["height"])]
            context = context_for("bbox", [grounding(label, box)])
            payload = [{"label": label, "polygon": polygon}]
            valid, reason = validate("mask", payload, context)
            if not valid:
                rejected["native_contract:" + reason] += 1
                continue
            selected.append({"annotation": annotation, "image": info, "payload": payload,
                             "context": context, "measurement": measurement})
            seen.add(image_id)
    if not selected:
        raise ValueError("no representable native COCO masks")
    print(f"COCO_SELECTED images={len(selected)} categories={len(pools)}", flush=True)
    cache.mkdir(parents=True, exist_ok=True)
    cache_index = cache / "image_bindings.json"
    cached = json.loads(cache_index.read_text()) if cache_index.is_file() else {}

    def image_bytes(item):
        info = item["image"]
        path = cache / info["file_name"]
        url = "https://s3.amazonaws.com/images.cocodataset.org/train2017/" + info["file_name"]
        if path.is_file() and info["file_name"] in cached:
            raw = path.read_bytes()
            binding = cached[info["file_name"]]
            if binding != {"url": url, "sha256": hashlib.sha256(raw).hexdigest()}:
                raise ValueError("cached COCO image binding drift")
        else:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.direct_download else urllib.request.build_opener()
            for attempt in range(5):
                try:
                    with opener.open(url, timeout=45) as response:
                        raw = response.read(4000001)
                    break
                except (OSError, TimeoutError):
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            if len(raw) > 4000000:
                raise ValueError("COCO image exceeds the frozen 4MB limit")
            if path.exists() and path.read_bytes() != raw:
                raise ValueError("unbound cached image differs from the official download")
        with Image.open(io.BytesIO(raw)) as rgb:
            if rgb.size != (info["width"], info["height"]) or rgb.format != "JPEG":
                raise ValueError("COCO image dimensions/encoding do not match its annotation")
            rgb.load()
            decoded = hashlib.sha256(rgb.convert("RGB").tobytes()).hexdigest()
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(raw)
        return path, hashlib.sha256(raw).hexdigest(), decoded, url

    records, provenance = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(image_bytes, item) for item in selected]
        for number, (item, future) in enumerate(zip(selected, futures), 1):
            # Preserve deterministic selection/order. A download failure aborts
            # this candidate build; it never silently swaps in a new image.
            path, image_sha, decoded_sha, url = future.result()
            if decoded_sha in excluded_rgb:
                raise ValueError("COCO candidate duplicates a reserved RGB under another source image ID")
            info, annotation = item["image"], item["annotation"]
            split = "dev" if int(hashlib.sha256(f"{args.seed}:{decoded_sha}".encode()).hexdigest()[:16], 16) % 1000 < args.dev_permille else "train"
            payload, context = item["payload"], item["context"]
            prompt = training_instruction("mask", context).replace("Return only a JSON array;", "Return <GROUND> followed by a JSON array;")
            uid = "coco_train2017_" + str(annotation["id"]) + "_mask"
            record = {"sample_id": uid, "images": [str(path)], "gsa_payload": payload,
                      "messages": [{"role": "system", "content": "Return the requested native structured output with its stage delimiter."},
                                   {"role": "user", "content": "<image>" + prompt},
                                   {"role": "assistant", "content": "<GROUND>" + json.dumps(payload)}],
                      **task_columns("mask", context, source="coco_official_train2017", scene="coco_train2017", image_id=image_sha)}
            record["gsa_meta"].update({"split": split, "source_image_id": info["id"], "source_annotation_id": annotation["id"],
                "decoded_rgb_sha256": decoded_sha, "label_provenance": "official_coco_instance_polygon_conversion",
                "teacher_eligible": False, "linked_causal_trace": False,
                "training_instruction_version": "anvil-native-coco-mask-v1", "polygon_measurement": item["measurement"]})
            records.append(record)
            provenance.append({"sample_id": uid, "image": info, "annotation": annotation,
                               "download_url": url, "image_sha256": image_sha, "decoded_rgb_sha256": decoded_sha,
                               "split": split, "polygon_measurement": item["measurement"]})
            cached[info["file_name"]] = {"url": url, "sha256": image_sha}
            if number % 8 == 0:
                # Persist completed download bindings so a later network failure
                # does not force those source images to be downloaded again.
                pending_index = cache_index.with_suffix('.pending.json')
                pending_index.write_text(json.dumps(cached, indent=2) + '\n')
                pending_index.replace(cache_index)
                print(f"COCO_PREPARED images={number}/{len(selected)}", flush=True)
    if len({row["gsa_meta"]["decoded_rgb_sha256"] for row in records}) != len(records):
        raise ValueError("duplicate decoded COCO image identities; retain cache and resolve before materialization")
    for binding in exclusions:
        if _sha256(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("COCO exclusion corpus changed during preparation")
    # This mutable download cache is distinct from immutable output provenance.
    cache_temporary = cache / "image_bindings.json.tmp"
    with cache_temporary.open("x") as handle:
        json.dump(cached, handle, indent=2)
        handle.write("\n")
    os.replace(cache_temporary, cache_index)
    output.mkdir(parents=True)
    for name, rows in (("audit_corpus", records), ("source_provenance", provenance),
                       ("train", [r for r in records if r["gsa_meta"]["split"] == "train"]),
                       ("dev", [r for r in records if r["gsa_meta"]["split"] == "dev"])):
        (output / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"schema": "anvil-native-coco-mask-candidate-v1", "status": "PENDING_CONTENT_AND_SEMANTIC_AUDIT",
                "builder_sha256": _sha256(Path(__file__)), "source_archive_sha256": args.expected_annotations_sha256,
                "source_member": "annotations/instances_train2017.json", "records": len(records),
                "selection": {"seed": args.seed, "limit": args.limit, "category_round_robin": sorted(pools),
                              "unique_category_per_image": True, "one_target_per_image": True,
                              "minimum_polygon_iou": min_iou, "maximum_polygon_vertices": max_vertices,
                              "minimum_bbox_side_pixels": min_side, "excluded_corpora": exclusions,
                              "excluded_source_images": len(excluded_images)},
                "split_rule": {"unit": "decoded_rgb_sha256", "dev_permille": args.dev_permille, "seed": args.seed},
                "partitions": dict(collections.Counter(r["gsa_meta"]["split"] for r in records)),
                "categories": dict(collections.Counter(r["gsa_payload"][0]["label"] for r in records)),
                "rejections": dict(rejected), "download": {"workers": args.workers, "direct": args.direct_download},
                "files": {p.name: {"path": str(p), "sha256": _sha256(p)} for p in output.glob("*.jsonl")},
                "training_labels_eligible": False, "human_validation_complete": False, "teacher_eligible": False,
                "benchmark_ground_truth": False, "redistribution_eligible": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"records": len(records), "partitions": manifest["partitions"], "status": manifest["status"]}))
    return 0


def prepare_sensor_supervision(args: argparse.Namespace) -> int:
    """Derive native G/S training candidates from bound official-train RGB-D frames."""
    import sys
    worktree = args.worktree.resolve()
    sys.path.insert(0, str(worktree / "tools"))
    sys.path.insert(0, str(worktree))
    import cv2
    import numpy as np
    import build_depth_v50_scannet_sensor as sensor
    from trains.flywheel.contracts import context_for, grounding, task_columns, training_instruction
    from trains.flywheel.validators import validate

    output = args.output.resolve()
    if output.exists() or args.limit < 0:
        raise SystemExit("sensor preparation needs a fresh output and nonnegative limit")
    input_root = args.input_root.resolve()
    manifest_path, index_path = input_root / "manifest.json", input_root / "images.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if (_sha256(index_path) != manifest["unlabelled_corpus_sha256"]
            or manifest["schema"] != "anvil-scannet-flywheel-input-v1"):
        raise SystemExit("sensor source index binding drift")
    _, official = sensor.load_split_manifests(worktree / "configs/scannet_splits")
    label_maps = sensor.load_label_map(args.label_map)
    scenes = {}
    for binding in manifest["source_manifests"]:
        path = Path(binding["path"])
        if _sha256(path) != binding["sha256"]:
            raise SystemExit("sensor extraction manifest drift")
        extraction = json.loads(path.read_text())
        policy = extraction["use_policy"]
        if (extraction["verdict"] != "SCALEOUT_VALIDATED_TRAIN_SPLIT_ONLY"
                or policy.get("training_materialization_eligible") is not True
                or policy.get("official_train_only") is not True):
            raise SystemExit("sensor extraction is not authorized official training material")
        for scene in extraction["scenes"]:
            name = scene["scene_id"]
            if name in scenes or official.get(name) != "train" or scene["official_split"] != "train":
                raise SystemExit("duplicate or non-training scene in sensor extraction")
            scenes[name] = (path.parent / name, scene)
    images = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    partitions = collections.defaultdict(set)
    for row in images:
        group = row["scene_id"].split("_")[0]
        bucket = int(hashlib.sha256(f"{manifest['selection']['salt']}\0{group}".encode()).hexdigest()[:16], 16) % 1000
        expected = "dev" if bucket < manifest["split_rule"]["dev_permille"] else "train"
        if row["scene_group"] != group or row["split"] != expected:
            raise SystemExit("physical scene split binding drift")
        partitions[group].add(row["split"])
    if any(len(value) != 1 for value in partitions.values()):
        raise SystemExit("physical scene group crosses train/dev")
    if args.limit:
        images = images[:args.limit]
    records, provenance = [], []
    rejections = collections.Counter()
    for number, row in enumerate(images):
        scene_root, scene = scenes[row["scene_id"]]
        if scene.get("depth_shift_raw_units_per_meter") != 1000.0:
            raise SystemExit("sensor depth encoding must be millimeters")
        frames = [frame for frame in scene["frames"] if frame["frame_index"] == row["frame_index"]]
        if len(frames) != 1:
            raise SystemExit("source frame must resolve exactly once")
        files = {}
        for name in ("color.png", "depth_native.png", "instance.png", "semantic.png",
                     "intrinsic_color.txt", "intrinsic_depth.txt", "extrinsic_color.txt", "extrinsic_depth.txt"):
            binding = frames[0]["outputs"][name]
            path = (scene_root / binding["path"]).resolve()
            if (not path.is_relative_to(scene_root.resolve()) or path.stat().st_size != binding["size_bytes"]
                    or _sha256(path) != binding["sha256"]):
                raise SystemExit(f"sensor frame binding drift: {path}")
            files[name] = {"path": str(path), "sha256": binding["sha256"]}
        if (str(Path(row["images"][0]).resolve()) != files["color.png"]["path"]
                or row["image_sha256"] != files["color.png"]["sha256"]):
            raise SystemExit("RGB source does not match the frozen index")
        arrays = {name: sensor.load_png(Path(files[name]["path"]))[0]
                  for name in ("color.png", "depth_native.png", "instance.png", "semantic.png")}
        rgb, depth = arrays["color.png"], arrays["depth_native.png"]
        if arrays["instance.png"].shape != rgb.shape[:2] or arrays["semantic.png"].shape != rgb.shape[:2]:
            raise SystemExit("RGB/instance/semantic shapes differ")
        mats = {name: sensor.read_matrix(Path(files[name + ".txt"]["path"]))
                for name in ("intrinsic_color", "intrinsic_depth", "extrinsic_color", "extrinsic_depth")}
        registered = sensor.register_depth_to_color(depth, mats["intrinsic_depth"], mats["extrinsic_depth"],
                                                    mats["intrinsic_color"], mats["extrinsic_color"], rgb.shape[:2])
        aggregation_path = args.scannet_root / row["scene_id"] / (row["scene_id"] + ".aggregation.json")
        aggregation, aggregation_sha = sensor.load_aggregation(aggregation_path, row["scene_id"], allow_bare_scene_id=True)
        accepted, rejected = sensor.instance_statistics(arrays["instance.png"], arrays["semantic.png"],
                                                        registered, depth.shape, label_maps, aggregation)
        rejections.update(rejected)
        visible_names = collections.Counter(aggregation.get(int(k), "").casefold()
                                            for k in np.unique(arrays["instance.png"]) if k > 0)
        candidates, mask_measurements = [], []
        # Native G questions contain no target geometry: require a unique visible name.
        unique = [item for item in accepted if visible_names[item.aggregation_label.casefold()] == 1]
        for item in sorted(unique, key=lambda x: (-x.mask_pixels, x.instance_id))[:2]:
            ctx = context_for("bbox", [grounding(item.aggregation_label, list(item.bbox_2d))])
            candidates.append(("bbox", [{"label": item.aggregation_label, "bbox_2d": list(item.bbox_2d)}], ctx))
            mask = (arrays["instance.png"] == item.instance_id).astype(np.uint8)
            if args.include_masks:
                polygon, measurement = _sensor_mask_polygon(mask)
                mask_measurements.append({"instance_id": item.instance_id,
                                          "label": item.aggregation_label, **measurement})
                if polygon is not None:
                    candidates.append(("mask", [{"label": item.aggregation_label, "polygon": polygon}], ctx))
                else:
                    rejections["mask:" + measurement["reason"]] += 1
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            point = [round(1000 * x / mask.shape[1]), round(1000 * y / mask.shape[0])]
            # Quantization must not move the supervised point out of the instance.
            qx, qy = round(point[0] * mask.shape[1] / 1000), round(point[1] * mask.shape[0] / 1000)
            if distance[y, x] >= 3 and 0 <= qy < mask.shape[0] and 0 <= qx < mask.shape[1] and mask[qy, qx]:
                candidates.append(("point", [{"label": item.aggregation_label, "point_2d": point}], ctx))
        pair = sensor.choose_pair(accepted)
        if pair:
            payload = [sensor._target_document(pair.nearer, 1), sensor._target_document(pair.farther, 2)]
            # Input order is independent of depth; never teach a copy-the-input-order shortcut.
            ordered = sorted((pair.nearer, pair.farther), key=lambda item: hashlib.sha256(
                f"{row['image_sha256']}\0{item.instance_id}".encode()).hexdigest())
            ctx = context_for("depth", [grounding(item.display_label, list(item.bbox_2d)) for item in ordered],
                              {"relation_mode": "pairwise", "depth_order": payload})
            candidates.append(("depth", payload, ctx))
        else:
            rejections["no_separated_depth_pair"] += 1
        emitted = []
        for form, payload, ctx in candidates:
            valid, reason = validate(form, payload, ctx)
            if not valid:
                rejections[f"native_contract:{form}:{reason}"] += 1
                continue
            family = "S" if form == "depth" else "G"
            delimiter = "<RELATE>" if family == "S" else "<GROUND>"
            prompt = training_instruction(form, ctx).replace("Return only a JSON array;", f"Return {delimiter} followed by a JSON array;")
            sample_id = hashlib.sha256(json.dumps([row["image_sha256"], form, payload], sort_keys=True).encode()).hexdigest()
            record = {"sample_id": sample_id, "images": row["images"], "gsa_payload": payload,
                      "messages": [{"role": "system", "content": "Return the requested native structured output with its stage delimiter."},
                                   {"role": "user", "content": "<image>" + prompt},
                                   {"role": "assistant", "content": delimiter + json.dumps(payload)}],
                      **task_columns(form, ctx, source="scannet_official_train_sensor", scene=row["scene_id"], image_id=row["image_sha256"])}
            record["gsa_meta"].update({"split": row["split"], "scene_group": row["scene_group"],
                "label_provenance": "official_instance_projection_and_registered_depth",
                "teacher_eligible": False, "linked_causal_trace": False,
                "training_instruction_version": "anvil-native-sensor-v1"})
            if form == "mask":
                record["gsa_meta"]["polygon_measurement"] = next(
                    value for value in mask_measurements if value["label"] == payload[0]["label"])
            records.append(record)
            emitted.append(sample_id)
        provenance.append({"scene_id": row["scene_id"], "frame_index": row["frame_index"],
                           "split": row["split"], "files": files,
                           "aggregation": {"path": str(aggregation_path), "sha256": aggregation_sha},
                           "accepted_instances": [sensor._measurement_document(item) for item in accepted],
                           "mask_measurements": mask_measurements,
                           "sample_ids": emitted})
        if (number + 1) % 50 == 0:
            print(f"SENSOR_PREPARED frames={number + 1}/{len(images)} candidates={len(records)}", flush=True)
    if len({r["sample_id"] for r in records}) != len(records):
        raise SystemExit("duplicate supervised target identity; resolve source duplication before materializing")
    split_images = {name: {r["gsa_meta"]["image_sha256"] for r in records if r["gsa_meta"]["split"] == name}
                    for name in ("train", "dev")}
    if split_images["train"] & split_images["dev"]:
        raise SystemExit("sensor supervision images cross train/dev")
    output.mkdir(parents=True)
    for name, selected in (("train", [r for r in records if r["gsa_meta"]["split"] == "train"]),
                           ("dev", [r for r in records if r["gsa_meta"]["split"] == "dev"]),
                           ("audit_corpus", records), ("source_provenance", provenance)):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
    report = {"schema": "anvil-native-sensor-supervision-candidate-v1", "status": "PENDING_CONTENT_AND_SEMANTIC_AUDIT",
              "builder_sha256": _sha256(Path(__file__)), "sensor_builder_sha256": _sha256(Path(sensor.__file__)),
              "input_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
              "label_map_sha256": _sha256(args.label_map), "attempted_images": len(images),
              "include_masks": args.include_masks,
              "mask_encoding": ({"version": "single-visible-polygon-v1", "min_component_fraction": .98,
                  "max_hole_fraction": .02, "min_roundtrip_iou": .95, "vertices": [6, 48],
                  "human_quality_established": False} if args.include_masks else None),
              "records": len(records), "images": len({r["gsa_meta"]["image_sha256"] for r in records}),
              "partitions": {split: dict(collections.Counter(r["gsa_form"] for r in records if r["gsa_meta"]["split"] == split))
                             for split in ("train", "dev")}, "rejections": dict(rejections),
              "files": {p.name: {"sha256": _sha256(p), "path": str(p)} for p in output.glob("*.jsonl")},
              "use_policy": {"training_labels_eligible": False, "benchmark_ground_truth": False,
                             "teacher_eligible": False, "redistribution_eligible": False},
              "limitations": ["Sensor-derived training candidates; no new human ground truth.",
                              "No A targets or linked G-S-A causal traces.",
                              "Retain original physical-scene splits; audit content and visual semantics before training."]}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "attempted_images", "records", "images", "partitions", "rejections")}, indent=2))
    return 0


def _process_start(pid: int) -> str | None:
    try:
        # comm can contain spaces or parentheses; field 22 follows the final ')'.
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]
    except FileNotFoundError:
        return None


def _native_legacy_group(group: dict, validators, contracts) -> tuple[list, dict]:
    """Join frozen same-image labels conservatively; never repair their answers."""
    accepted, rejected, closed = [], collections.Counter(), {}
    payloads = {}
    for form, entries in group.items():
        if len(entries) != 1:
            rejected[form + ":ambiguous_source_rows"] += len(entries)
            continue
        row, line_number = entries[0]
        try:
            answer = row["messages"][-1]
            if answer["role"] != "assistant":
                raise ValueError("missing assistant")
            text = answer["content"].strip()
            if text.startswith("```json\n") and text.endswith("\n```"):
                text = text[8:-4]
            payload = json.loads(text)
            if not isinstance(payload, list) or not payload or not all(isinstance(x, dict) for x in payload):
                raise ValueError("not a nonempty array of objects")
            if form not in {"detection", "depth"} and len(payload) != 1:
                raise ValueError("ambiguous single-object target")
            payloads[form] = (payload, row, line_number)
        except (KeyError, IndexError, TypeError, ValueError):
            rejected[form + ":invalid_or_ambiguous_answer"] += 1

    def label(item):
        return str(item.get("label", "")).strip().casefold()

    def box(item):
        return item.get("bbox_2d") or item.get("bbox")

    def valid_box(item):
        return validators.validate("bbox", [json.loads(json.dumps(item))])[0]

    detection = payloads.get("detection", ([],))[0]
    detection_ok = validators.validate("detection", detection)[0]
    anchors = detection if detection_ok else []
    main = payloads.get("bbox", ([],))[0]
    main = main[0] if main and valid_box(main[0]) else None
    dependencies = {"depth": ["detection"], "affordance": ["detection", "depth"],
                    "dof": ["detection", "depth", "affordance"],
                    "action_cot": ["detection", "depth", "affordance", "dof"]}
    for form in contracts.FORM_SPECS:
        if form not in payloads:
            continue
        payload, row, line_number = payloads[form]
        deps = dependencies.get(form, [])
        if any(dep not in closed for dep in deps):
            rejected[form + ":unaccepted_upstream"] += 1
            continue
        primary = None
        if form == "bbox":
            primary = main
        elif form in {"point", "mask"}:
            candidates = [item for item in ([main] if main else []) + anchors
                          if label(item) == label(payload[0])]
            if candidates:
                # Two different boxes for the same name are an ambiguous anchor.
                distinct = {tuple(box(item)) for item in candidates}
                if len(distinct) == 1:
                    primary = candidates[0]
        elif form in {"affordance", "dof", "action_cot"} and main:
            # The legacy A prompt targets the main object. Require the same exact
            # box and name in detection, rather than inventing a cross-form match.
            primary = next((item for item in anchors
                            if label(item) == label(main) and box(item) == box(main)), None)
        if form not in {"detection", "depth"} and primary is None:
            rejected[form + ":missing_unambiguous_anchor"] += 1
            continue
        if form == "depth" and (len(payload) != len(anchors)
                                or len({label(x) for x in payload}) != len(payload)):
            rejected[form + ":nonbijective_depth_entities"] += 1
            continue
        groundings = [contracts.grounding(x["label"], box(x)) for x in anchors]
        if form in {"bbox", "point", "mask"}:
            groundings = [contracts.grounding(primary["label"], box(primary))]
        # Hash order depends only on entity identity, never on the current depth
        # target or its rank. Keep the named A primary separately from that order.
        groundings.sort(key=lambda item: hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest())
        spatial = {("depth_order" if dep == "depth" else dep): closed[dep]
                   for dep in deps if dep != "detection"}
        context = contracts.context_for(form, groundings, spatial)
        if primary:
            context["primary"] = contracts.grounding(primary["label"], box(primary))
        frozen_payload = json.dumps(payload, sort_keys=True)
        valid, reason = validators.validate(form, payload, context)
        if not valid or json.dumps(payload, sort_keys=True) != frozen_payload:
            rejected[form + ":" + (reason if not valid else "validator_would_rewrite_answer")] += 1
            continue
        instruction = contracts.training_instruction(form, context)
        family = contracts.FORM_SPECS[form]["family"]
        delimiter = {"G": "<GROUND>", "S": "<RELATE>", "A": "<ACT>"}[family]
        instruction = instruction.replace(" Return only a JSON array;", f" Return {delimiter} followed by a JSON array;")
        accepted.append({"form": form, "payload": payload, "context": context,
                         "instruction": instruction, "delimiter": delimiter,
                         "source_row": row, "source_line": line_number,
                         "upstream_source_lines": [payloads[dep][2] for dep in deps]})
        closed[form] = payload
    return accepted, dict(rejected)


def prepare_native_replay(args: argparse.Namespace) -> int:
    """Create a source-bound native SFT derivative of an already audited corpus."""
    import sys
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(args.worktree.resolve()))
    from trains.flywheel import contracts, validators

    output = args.output.resolve()
    if output.exists():
        raise SystemExit("native replay preparation needs a fresh output")
    binding = _clean_sft_inputs(args)
    excluded = set()
    exclude_bindings = []
    for path in args.exclude_corpus:
        path = path.resolve()
        exclude_bindings.append({"path": str(path), "sha256": _sha256(path)})
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for image in _images(row):
                excluded.add(contracts.image_sha256(image))
    selected, rejections = {}, {}
    for split in ("train", "dev"):
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        rejected = collections.Counter()
        path = Path(binding["files"][split]["path"])
        with path.open() as handle:
            for number, line in enumerate(handle, 1):
                row = json.loads(line)
                form = (row.get("gsa_meta") or {}).get("form")
                images = _images(row)
                if form not in contracts.FORM_SPECS:
                    rejected[str(form) + ":not_native_form"] += 1
                elif len(images) != 1:
                    rejected[str(form) + ":not_single_image"] += 1
                else:
                    groups[images[0]][form].append((row, number))
        candidates = []
        for image, group in groups.items():
            rows, reasons = _native_legacy_group(group, validators, contracts)
            rejected.update(reasons)
            candidates.extend((image, row) for row in rows)
        unique_images = sorted({image for image, _ in candidates})
        with ThreadPoolExecutor(max_workers=8) as pool:
            hashes = dict(zip(unique_images, pool.map(contracts.image_sha256, unique_images)))
        records = []
        identities = collections.Counter((hashes[image], item["form"]) for image, item in candidates)
        for image, item in candidates:
            digest = hashes[image]
            form = item["form"]
            if digest in excluded:
                rejected[form + ":reserved_historical_image"] += 1
                continue
            # Byte-identical image aliases cannot multiply a task's weight.
            key = (digest, form)
            if identities[key] > 1:
                rejected[form + ":ambiguous_duplicate_image_form"] += 1
                continue
            meta = item["source_row"]["gsa_meta"]
            identity = hashlib.sha256(f"{binding['files'][split]['sha256']}:{item['source_line']}".encode()).hexdigest()
            record = {"sample_id": "anvil-native-replay-" + identity[:24], "images": [image],
                      "gsa_payload": item["payload"], "messages": [
                          {"role": "system", "content": "Return the requested native structured output with its stage delimiter."},
                          {"role": "user", "content": "<image>" + item["instruction"]},
                          {"role": "assistant", "content": item["delimiter"] + json.dumps(item["payload"], ensure_ascii=False)}]}
            record.update(contracts.task_columns(form, item["context"], source=meta.get("source", "unknown"),
                                                 scene=meta.get("scene", "unknown"), image_id=digest))
            record["gsa_meta"].update({"split": split, "label_provenance": "legacy_synthetic_contract_replay",
                "human_reviewed": False, "teacher_eligible": False, "linked_causal_trace": False,
                "source_corpus_sha256": binding["files"][split]["sha256"], "source_line": item["source_line"],
                "upstream_source_lines": item["upstream_source_lines"],
                "training_instruction_version": "anvil-native-replay-v1"})
            records.append(record)
        selected[split] = sorted(records, key=lambda row: row["sample_id"])
        rejections[split] = dict(sorted(rejected.items()))
    image_sets = {split: {row["gsa_meta"]["image_sha256"] for row in rows} for split, rows in selected.items()}
    overlap = image_sets["train"] & image_sets["dev"]
    for split, rows in selected.items():
        # Preserve the frozen source split. Quarantine every occurrence of shared
        # image bytes from this derivative, including the development side.
        removed = [row for row in rows if row["gsa_meta"]["image_sha256"] in overlap]
        reasons = collections.Counter(rejections[split])
        reasons.update(row["gsa_form"] + ":cross_split_image_bytes" for row in removed)
        rejections[split] = dict(sorted(reasons.items()))
        selected[split] = [row for row in rows if row["gsa_meta"]["image_sha256"] not in overlap]
        image_sets[split] -= overlap
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-replay-", dir=output.parent) as temporary:
        staging = Path(temporary)
        files = {}
        for split, rows in selected.items():
            for family in (None, "G", "S", "A"):
                subset = rows if family is None else [row for row in rows if row["gsa_family"] == family]
                name = split + ("_" + family if family else "") + ".jsonl"
                path = staging / name
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset))
                files[name] = {"rows": len(subset), "sha256": _sha256(path)}
        report = {"schema": "anvil-native-synthetic-replay-v1", "inputs": binding,
                  "excluded_corpora": exclude_bindings, "files": files,
                  "by_form": {split: dict(collections.Counter(row["gsa_form"] for row in rows)) for split, rows in selected.items()},
                  "unique_images": {split: len(images) for split, images in image_sets.items()},
                  "rejections": rejections, "image_byte_disjoint": True,
                  "cross_split_image_bytes_quarantined": sorted(overlap),
                  "source_train_dev_assignments_preserved": True, "teacher_eligible": False,
                  "benchmark_ground_truth": False, "paper_metrics_admitted": False,
                  "scope": "Native-format synthetic SFT derivative; not human-verified labels or observed causal traces",
                  "builder_sha256": _sha256(Path(__file__)),
                  "validator_sha256": _sha256(Path(validators.__file__)),
                  "contracts_sha256": _sha256(Path(contracts.__file__))}
        (staging / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    print(json.dumps({key: report[key] for key in ("by_form", "unique_images", "files", "rejections")}, indent=2))
    return 0


def _clean_sft_inputs(args: argparse.Namespace) -> dict:
    corpus, worktree, model = args.corpus.resolve(), args.worktree.resolve(), args.model.resolve()
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source"]["path"]).resolve()
    source_sha = _sha256(source)
    if source_sha != manifest["source"]["sha256"]:
        raise SystemExit("source corpus changed since splitting")
    if not all(manifest["checks"].get(key) is True for key in
               ("row_conservation", "image_disjoint", "all_forms_in_dev")):
        raise SystemExit("split has not passed conservation/image-disjoint/form checks")
    paths = {"manifest": manifest_path, "source": source,
             "model_config": model / "config.json", "tokenizer": model / "tokenizer.json",
             "sft_launcher": worktree / "scripts/run_sft_9b_lora_l20_dlc.sh",
             "spatial_launcher": worktree / "scripts/run_eval_adapter_dp.sh",
             "retention_launcher": worktree / "scripts/run_general_retention_dp.sh",
             "retention_audit": args.retention_audit.resolve(),
             "spatial_audit": args.spatial_audit.resolve()}
    for split in ("train", "dev"):
        path = corpus / f"{split}.jsonl"
        if _sha256(path) != manifest[split]["sha256"] or manifest[split]["rows"] < 1:
            raise SystemExit(f"{split} corpus differs from the frozen split")
        paths[split] = path
    retention = json.loads(paths["retention_audit"].read_text())
    spatial = json.loads(paths["spatial_audit"].read_text())
    if (retention.get("state") != "PASS"
            or retention.get("posttraining_corpus_sha256") != source_sha
            or retention.get("posttraining_corpus_rows") != manifest["source"]["rows"]):
        raise SystemExit("retention audit does not clear this complete source corpus")
    if (spatial.get("contamination_verdict") != "PASS"
            or spatial.get("contamination_clean") is not True
            or spatial.get("posttraining_corpus", {}).get("sha256") != source_sha):
        raise SystemExit("spatial content audit does not clear this complete source corpus")
    return {
        "files": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in paths.items()},
        "train_rows": manifest["train"]["rows"], "dev_rows": manifest["dev"]["rows"],
        "spatial_metadata_status": spatial.get("metadata_completeness_verdict"),
        "scope": "fresh-base SFT candidate; content audits do not certify semantic label accuracy",
    }


def _source_group(image: str) -> str:
    """Conservative sequence grouping; unknown layouts stay in one directory."""
    path = Path(image)
    match = re.match(r"(.+?)_cam\d+", path.stem)
    if match:
        if "robopoint" in image.casefold():
            return "robopoint:" + match[1]
        return str(path.parent / match[1])
    floor = re.search(r"(FloorPlan\d+)", image)
    if floor:
        return image[:floor.start()] + floor[1]
    for parent in path.parents:
        if re.fullmatch(r"traj\d+", parent.name):
            return str(parent)
    return str(path.parent)


def prepare_native_feedback(args: argparse.Namespace) -> int:
    """Freeze current native flywheel targets with exact upstream provenance."""
    import sys
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(args.worktree.resolve()))
    from trains.flywheel import contracts, validators

    output = args.output.resolve()
    if output.exists() or args.max_images < 1 or not 1 <= args.dev_permille <= 500:
        raise SystemExit("native feedback requires a fresh output, positive image limit and valid development fraction")
    excluded, excluded_bindings = set(), []
    for path in args.exclude_corpus:
        excluded_bindings.append({"path": str(path.resolve()), "sha256": _sha256(path)})
        for line in path.read_text().splitlines():
            for image in _images(json.loads(line)):
                excluded.add(contracts.image_sha256(image))
    sources = []
    for root in args.raw_root:
        for path in sorted(root.resolve().glob("shard*/raw/*.jsonl")):
            sources.append({"path": str(path), "snapshot_bytes": path.stat().st_size})
    if not sources or len({x["path"] for x in sources}) != len(sources):
        raise SystemExit("raw roots must resolve to a nonempty, unique set of shard form files")
    snapshot_manifest = getattr(args, "snapshot_manifest", None)
    if snapshot_manifest:
        snapshot = json.loads(snapshot_manifest.read_text())
        if ({item["path"] for item in sources} != {item["path"] for item in snapshot["source_prefixes"]}
                or snapshot["label_version"] != args.label_version
                or snapshot["selection"]["salt"] != args.salt
                or snapshot["selection"]["max_images"] != args.max_images
                or snapshot["split_rule"]["dev_permille"] != args.dev_permille
                or snapshot["excluded_corpora"] != excluded_bindings):
            raise SystemExit("snapshot replay arguments differ from their frozen source recipe")
        sources = snapshot["source_prefixes"]
    versions, images = collections.Counter(), collections.defaultdict(set)

    def scan(binding):
        digest, consumed, number = hashlib.sha256(), 0, 0
        with Path(binding["path"]).open("rb") as handle:
            while consumed < binding["snapshot_bytes"]:
                line = handle.readline(binding["snapshot_bytes"] - consumed)
                if not line.endswith(b"\n"):
                    break
                consumed += len(line)
                digest.update(line)
                number += 1
                try:
                    row = json.loads(line)
                except ValueError:
                    yield number, None, hashlib.sha256(line).hexdigest()
                    continue
                yield number, row, hashlib.sha256(line).hexdigest()
        measured = {"prefix_bytes": consumed, "prefix_sha256": digest.hexdigest(), "lines": number}
        if "prefix_sha256" in binding and any(binding[key] != value for key, value in measured.items()):
            raise SystemExit("raw source prefix changed while freezing: " + binding["path"])
        binding.update(measured)

    for binding in sources:
        for _, row, _ in scan(binding):
            if row is None:
                versions["invalid_json"] += 1
                continue
            versions[str(row.get("label_version"))] += 1
            if row.get("label_version") != args.label_version:
                continue
            digest, image = row.get("image_sha256"), row.get("image_path")
            if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) and isinstance(image, str):
                if digest not in excluded:
                    images[digest].add(image)
    selected = sorted(images, key=lambda digest: hashlib.sha256(f"{args.salt}\0{digest}".encode()).hexdigest())[:args.max_images]
    selected_set = set(selected)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for binding in sources:
        for number, row, digest in scan(binding):
            if (row and row.get("label_version") == args.label_version
                    and row.get("image_sha256") in selected_set):
                groups[row["image_sha256"]][row.get("form")].append((row, {
                    "path": binding["path"], "line": number, "record_sha256": digest}))
    image_paths = sorted({path for digest in selected for path in images[digest]})
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual_hashes = dict(zip(image_paths, pool.map(contracts.image_sha256, image_paths)))
    if any(actual_hashes[path] != digest for digest in selected for path in images[digest]):
        raise SystemExit("raw label image bytes differ from their declared hashes")
    dependencies = {"depth": ["detection"], "affordance": ["detection", "depth"],
                    "dof": ["detection", "depth", "affordance"],
                    "action_cot": ["detection", "depth", "affordance", "dof"]}
    partitions, rejections = {"train": [], "dev": []}, collections.Counter()
    group_sets = {"train": set(), "dev": set()}
    for digest in selected:
        identities = {_source_group(path) for path in images[digest]}
        if len(identities) != 1:
            rejections["ambiguous_image_source_groups"] += 1
            continue
        group_id = identities.pop()
        bucket = int(hashlib.sha256(f"{args.salt}\0{group_id}".encode()).hexdigest()[:16], 16) % 1000
        split = "dev" if bucket < args.dev_permille else "train"
        closed = {}
        for form, spec in contracts.FORM_SPECS.items():
            entries = groups[digest].get(form, [])
            if len(entries) != 1:
                rejections[form + (":missing" if not entries else ":duplicate_source")] += 1
                continue
            row, raw_binding = entries[0]
            payload, context = row.get("payload"), row.get("gsa_context") or {}
            semantic, verification = row.get("_semantic_gate") or {}, row.get("_verification") or {}
            if (verification.get("verdict") != "verified" or verification.get("form") != form
                    or semantic.get("scene", {}).get("eligible") is not True
                    or (spec["family"] in {"S", "A"} and semantic.get("depth", {}).get("valid") is not True)
                    or (spec["family"] == "A" and semantic.get("manipulation", {}).get("valid") is not True)):
                rejections[form + ":source_verification_missing"] += 1
                continue
            deps = dependencies.get(form, [])
            if any(dep not in closed for dep in deps):
                rejections[form + ":unaccepted_upstream"] += 1
                continue
            if deps:
                upstream_g = [contracts.grounding(item["label"], item["bbox_2d"])
                              for item in closed["detection"]["payload"]]
                linked = context.get("groundings") == upstream_g
                for dep in deps[1:]:
                    linked = linked and context.get("spatial", {}).get("depth_order" if dep == "depth" else dep) == closed[dep]["payload"]
                if not linked:
                    rejections[form + ":upstream_payload_mismatch"] += 1
                    continue
            payload_copy = json.loads(json.dumps(payload))
            valid, reason = validators.validate(form, payload_copy, context)
            if not valid or payload_copy != payload:
                rejections[form + ":" + (reason if not valid else "validator_would_rewrite_answer")] += 1
                continue
            if form == "depth":
                names = [str(item.get("label", "")).casefold() for item in payload]
                if len(set(names)) != len(names) or len(names) != len(context.get("groundings", [])):
                    rejections[form + ":nonbijective_depth_entities"] += 1
                    continue
            # Keep only previous-stage targets in task metadata as well as prompts.
            context = json.loads(json.dumps(context))
            context["spatial"] = {("depth_order" if dep == "depth" else dep): closed[dep]["payload"]
                                  for dep in deps if dep != "detection"}
            try:
                instruction = contracts.training_instruction(form, context)
            except ValueError:
                rejections[form + ":invalid_instruction_context"] += 1
                continue
            delimiter = {"G": "<GROUND>", "S": "<RELATE>", "A": "<ACT>"}[spec["family"]]
            instruction = instruction.replace(" Return only a JSON array;", f" Return {delimiter} followed by a JSON array;")
            record = {"sample_id": "anvil-feedback-" + raw_binding["record_sha256"][:24],
                      "images": [row["image_path"]], "gsa_payload": payload, "messages": [
                          {"role": "system", "content": "Return the requested native structured output with its stage delimiter."},
                          {"role": "user", "content": "<image>" + instruction},
                          {"role": "assistant", "content": delimiter + json.dumps(payload, ensure_ascii=False)}]}
            record.update(contracts.task_columns(form, context, source=row.get("source", "unknown"),
                                                 scene=row.get("scene", "unknown"), image_id=digest))
            record["gsa_meta"].update({"split": split, "source_group": group_id,
                "label_provenance": "external_teacher_native_verified", "label_version": args.label_version,
                "human_reviewed": False, "teacher_eligible": False, "linked_causal_trace": False,
                "raw_binding": raw_binding, "upstream_raw_bindings": [closed[dep]["raw_binding"] for dep in deps],
                "training_instruction_version": "anvil-native-feedback-v1"})
            partitions[split].append(record)
            group_sets[split].add(group_id)
            closed[form] = {"payload": payload, "raw_binding": raw_binding}
    if group_sets["train"] & group_sets["dev"]:
        raise SystemExit("source groups cross train/dev")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-feedback-", dir=output.parent) as temporary:
        staging, files = Path(temporary), {}
        for split, records in partitions.items():
            records.sort(key=lambda row: row["sample_id"])
            for family in (None, "G", "S", "A"):
                rows = records if family is None else [row for row in records if row["gsa_family"] == family]
                name = split + ("_" + family if family else "") + ".jsonl"
                path = staging / name
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                files[name] = {"rows": len(rows), "sha256": _sha256(path)}
        audit = staging / "audit_corpus.jsonl"
        audit.write_bytes((staging / "train.jsonl").read_bytes() + (staging / "dev.jsonl").read_bytes())
        files[audit.name] = {"rows": sum(len(rows) for rows in partitions.values()), "sha256": _sha256(audit)}
        report = {"schema": "anvil-native-feedback-candidates-v1", "status": "PENDING_CONTENT_AND_INDEPENDENT_AUDITS",
            "training_eligible": False, "teacher_eligible": False, "benchmark_ground_truth": False,
            "source_prefixes": sources, "source_versions": dict(versions), "excluded_corpora": excluded_bindings,
            "label_version": args.label_version, "selection": {"salt": args.salt, "max_images": args.max_images,
                "selected_images": len(selected), "eligible_source_images": len(images), "rule": "lowest salted image SHA256"},
            "split_rule": {"unit": "source sequence group; unknown layouts use whole parent directory", "dev_permille": args.dev_permille},
            "files": files, "rejections": dict(rejections), "source_groups_disjoint": True,
            "by_form": {split: dict(collections.Counter(row["gsa_form"] for row in rows)) for split, rows in partitions.items()},
            "source_groups": {split: len(values) for split, values in group_sets.items()},
            "builder_sha256": _sha256(Path(__file__)), "validator_sha256": _sha256(Path(validators.__file__)),
            "contracts_sha256": _sha256(Path(contracts.__file__)),
            "notes": ["Current generation verification is machine-only, not independent or human agreement.",
                      "Native tasks preserve exact stored upstream targets; current-stage targets never enter prompts.",
                      "Source sequence grouping does not establish complete physical-scene disjointness."]}
        (staging / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    print(json.dumps({key: report[key] for key in ("selection", "by_form", "files", "source_groups", "rejections")}, indent=2))
    return 0


def clean_native_feedback(args: argparse.Namespace) -> int:
    """Exclude known evaluation scenes and every near-duplicate candidate group."""
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit("feedback cleaning requires a fresh derivative")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = source / "audit_corpus.jsonl"
    if _sha256(audit) != manifest["files"][audit.name]["sha256"]:
        raise SystemExit("candidate source changed before cleaning")
    spatial_root = source / "audit/spatial"
    spatial = json.loads((spatial_root / "report.json").read_text())
    if spatial.get("posttraining_corpus", {}).get("sha256") != _sha256(audit):
        raise SystemExit("near-duplicate report is not bound to the source candidates")
    eval_path, near_path = spatial_root / "eval_image_fingerprints.jsonl", spatial_root / "near_duplicate_candidates.jsonl"
    reserved = {_source_group(row["path"]) for row in map(json.loads, eval_path.read_text().splitlines())
                if "robopoint" in row["path"].casefold()}
    reserved.update(_source_group(row["training_path"]) for row in map(json.loads, near_path.read_text().splitlines()))
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    rejected_images, removed_rows = set(), 0
    partitions, source_groups = {"train": [], "dev": []}, {"train": set(), "dev": set()}
    salt, fraction = manifest["selection"]["salt"], manifest["split_rule"]["dev_permille"]
    for row in rows:
        group = _source_group(row["images"][0])
        if group in reserved:
            rejected_images.add(row["gsa_meta"]["image_sha256"])
            removed_rows += 1
            continue
        bucket = int(hashlib.sha256(f"{salt}\0{group}".encode()).hexdigest()[:16], 16) % 1000
        split = "dev" if bucket < fraction else "train"
        row["gsa_meta"].update(split=split, source_group=group)
        partitions[split].append(row)
        source_groups[split].add(group)
    if source_groups["train"] & source_groups["dev"]:
        raise SystemExit("cleaned feedback groups cross train/dev")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-feedback-clean-", dir=output.parent) as temporary:
        staging, files = Path(temporary), {}
        for split, items in partitions.items():
            items.sort(key=lambda row: row["sample_id"])
            for family in (None, "G", "S", "A"):
                selected = items if family is None else [row for row in items if row["gsa_family"] == family]
                if not selected:
                    raise SystemExit(f"cleaned feedback has no {split}/{family} samples")
                path = staging / (split + ("_" + family if family else "") + ".jsonl")
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
                files[path.name] = {"rows": len(selected), "sha256": _sha256(path)}
        combined = staging / "audit_corpus.jsonl"
        combined.write_bytes((staging / "train.jsonl").read_bytes() + (staging / "dev.jsonl").read_bytes())
        files[combined.name] = {"rows": len(rows) - removed_rows, "sha256": _sha256(combined)}
        report = {**manifest, "status": "PENDING_CLEAN_DERIVATIVE_REAUDIT", "files": files,
                  "parent": {"path": str(source), "manifest_sha256": _sha256(manifest_path)},
                  "quarantine": {"rule": "whole known RoboPoint evaluation scene and whole perceptual-candidate source group",
                    "reserved_groups": sorted(reserved), "removed_rows": removed_rows,
                    "removed_images": len(rejected_images), "removed_image_sha256": sorted(rejected_images),
                    "evidence": [{"path": str(path), "sha256": _sha256(path)}
                                 for path in (spatial_root / "report.json", eval_path, near_path)]},
                  "by_form": {split: dict(collections.Counter(row["gsa_form"] for row in items)) for split, items in partitions.items()},
                  "source_groups": {split: len(values) for split, values in source_groups.items()},
                  "review_pool": {"path": str(source / "preaudit_v1"), "scope": "independent frozen parent-generation sample; quarantined cases retain their votes"},
                  "split_rule": {"unit": "canonical RoboPoint scene ID across mounts/subfolders; otherwise source sequence group", "dev_permille": fraction},
                  "builder_sha256": _sha256(Path(__file__))}
        (staging / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        (staging / "preaudit_v1").symlink_to(source / "preaudit_v1", target_is_directory=True)
        staging.rename(output)
    print(json.dumps({"rows": files["audit_corpus.jsonl"]["rows"], "sha256": files["audit_corpus.jsonl"]["sha256"],
                      "removed_rows": removed_rows, "removed_images": len(rejected_images),
                      "by_form": report["by_form"], "source_groups": report["source_groups"]}, indent=2))
    return 0


def _completed_sft_lineage(version: Path, binding: dict, model: Path, steps: int, seed: int, lr: float) -> dict:
    config_path = version / "args.json"
    config = json.loads(config_path.read_text())
    expected = {"model": str(model.resolve()), "dataset": [binding["files"]["train"]["path"]],
                "val_dataset": [binding["files"]["dev"]["path"]], "adapters": [],
                "resume_from_checkpoint": None, "seed": seed, "data_seed": seed,
                "learning_rate": lr, "max_steps": steps, "tuner_type": "lora",
                "lora_rank": 64, "lora_alpha": 128, "lora_dropout": 0.05,
                "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4}
    mismatch = [key for key, value in expected.items() if config.get(key) != value]
    if mismatch:
        raise SystemExit(f"actual trainer configuration differs from the queued experiment: {mismatch}")
    state_path = version / f"checkpoint-{steps}/trainer_state.json"
    state = json.loads(state_path.read_text())
    if state.get("global_step") != steps or state.get("max_steps") != steps:
        raise SystemExit("fresh SFT has not reached its full step budget")
    selected = Path(state.get("best_model_checkpoint") or state_path.parent).resolve()
    if not selected.is_relative_to(version.resolve()) or not re.fullmatch(r"checkpoint-\d+", selected.name):
        raise SystemExit("validation-selected adapter escapes the completed fresh run")
    paths = {"trainer_args": config_path, "final_state": state_path,
             "selected_adapter_config": selected / "adapter_config.json",
             "selected_adapter_weights": selected / "adapter_model.safetensors"}
    return {"inputs": binding, "files": {key: {"path": str(path), "sha256": _sha256(path)}
                                        for key, path in paths.items()},
            "selected_adapter": str(selected), "paper_admission": "pending complete evaluation and score health"}


_INFERENCE_DISTRIBUTED_ENV = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "NODE_RANK",
    "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "_PATCH_WORLD_SIZE", "LOCAL_SIZE", "NPROC_PER_NODE",
)


def _native_engine(model: Path, adapter: Path, output: Path, max_new_tokens: int = 512):
    """Use the same Swift model/template path while retaining response health fields."""
    # Training launchers accept RANK as a LoRA hyperparameter. A single-process
    # inference must not inherit it as torch/Swift's distributed global rank.
    for key in _INFERENCE_DISTRIBUTED_ENV:
        os.environ.pop(key, None)
    from swift.pipelines.infer.infer import SwiftInfer
    command = ["--model", str(model), "--adapters", str(adapter), "--infer_backend", "transformers",
               "--max_batch_size", "4", "--max_new_tokens", str(max_new_tokens), "--max_length", "3072",
               "--temperature", "0", "--seed", "17", "--enable_thinking", "false",
               "--stream", "false", "--load_args", "false", "--torch_dtype", "bfloat16"]
    engine = SwiftInfer(command)
    actual = {key: getattr(engine.args, key) for key in (
        "model", "adapters", "infer_backend", "max_batch_size", "max_new_tokens", "max_length",
        "temperature", "seed", "enable_thinking", "stream", "load_args")}
    if (actual["model"] != str(model) or actual["adapters"] != [str(adapter)]
            or actual["max_new_tokens"] != max_new_tokens or actual["enable_thinking"] is not False):
        raise ValueError("native inference resolved unexpected model/template settings")
    with (output / "inference_args.json").open("x") as handle:
        json.dump({"command": command, "actual": actual, "image_max_token_num": os.environ.get("IMAGE_MAX_TOKEN_NUM"),
                   "adapter_files": {name: _sha256(adapter / name) for name in ("adapter_config.json", "adapter_model.safetensors")}}, handle, indent=2)
    return engine


def _native_infer_batches(engine, rows: list[dict], output: Path) -> list[dict]:
    from swift import InferRequest
    import copy
    active = [row for row in rows if not row.get("gsa_eval_blocked_upstream")]
    predictions = []
    with output.open("x") as handle:
        for start in range(0, len(active), 4):
            batch = active[start:start + 4]
            messages = [[{"role": item["role"], "content": item["content"]}
                         for item in row["messages"] if item["role"] in ("system", "user")] for row in batch]
            if any(not value or value[-1]["role"] != "user" for value in messages):
                raise ValueError("native inference requires a terminal user prompt")
            requests = [InferRequest(messages=copy.deepcopy(value), images=list(row["images"]))
                        for row, value in zip(batch, messages)]
            started = time.monotonic()
            responses = engine.infer(requests, engine.args.get_request_config(), use_tqdm=False)
            elapsed = time.monotonic() - started
            if len(responses) != len(batch):
                raise ValueError("native engine response coverage drift")
            for row, prompt, response in zip(batch, messages, responses):
                choice = response.choices[0]
                references = [item["content"] for item in row["messages"] if item["role"] == "assistant"]
                if len(references) != 1:
                    raise ValueError("native inference requires one bound reference stored outside the request")
                prediction = {"sample_id": row["sample_id"], "images": list(row["images"]), "messages": prompt,
                              "labels": references[0], "response": choice.message.content or "",
                              "finish_reason": choice.finish_reason,
                              "prompt_tokens": response.usage.prompt_tokens,
                              "completion_tokens": response.usage.completion_tokens,
                              "batch_index": start // 4, "batch_size": len(batch), "batch_seconds": elapsed}
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                predictions.append(prediction)
            handle.flush()
            print(f"NATIVE_INFER {output.name} {min(start + 4, len(active))}/{len(active)}", flush=True)
    return predictions


def _native_context_pack(args, source_rows: list[dict], pack: Path) -> None:
    from eval.g_family_qualification import native_context_rows, native_join_predictions
    import copy
    if pack.exists():
        raise ValueError("context preparation requires a fresh pack")
    pack.mkdir(parents=True)
    adapter = args.teachers.resolve() / "joint/selected_adapter"
    initial = json.loads((args.teachers / "joint/completed_lineage.json").read_text())["persistent_selected_adapter"]
    for binding in initial["files"].values():
        if _sha256(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("frozen context-generator adapter binding drift")
    engine = _native_engine(args.model.resolve(), adapter, pack)
    generated, files = {}, {}

    def freeze(name, rows):
        path = pack / (name + ".jsonl")
        with path.open("x") as handle:
            handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        files[path.name] = {"path": str(path), "sha256": _sha256(path), "rows": len(rows)}
        return path

    # Grounding is generated without reference answers. Subsequent stages consume
    # these actual outputs recursively, including structural failures in coverage.
    for name, forms in (("G", None), ("depth", ("depth",)), ("affordance", ("affordance",)), ("dof", ("dof",))):
        rows = ([row for row in source_rows if row["gsa_family"] == "G"] if forms is None
                else native_context_rows(source_rows, "student_cascade", generated, forms))
        freeze("cascade_" + name + "_inputs", rows)
        predicted_path = pack / ("cascade_" + name + "_predictions.jsonl")
        predicted = _native_infer_batches(engine, rows, predicted_path)
        generated.update(native_join_predictions(rows, predicted))
        files[predicted_path.name] = {"path": str(predicted_path), "sha256": _sha256(predicted_path), "rows": len(predicted)}
    variants = {}
    for variant in ("oracle", "student_cascade", "corrupted"):
        rows = native_context_rows(source_rows, variant, generated if variant == "student_cascade" else None)
        for original in source_rows:
            if original["gsa_family"] == "G":
                row = copy.deepcopy(original)
                row["gsa_eval_original_sample_id"] = row["sample_id"]
                row["sample_id"] += "::" + variant
                row["gsa_context_intervention"] = {"variant": variant, "visible_input_changed": False,
                                                   "reference_targets_unchanged": True, "upstream": {}}
                row["gsa_provided_context"] = copy.deepcopy(row["gsa_context"])
                row["gsa_eval_blocked_upstream"] = []
                rows.append(row)
        rows.sort(key=lambda row: row["gsa_eval_original_sample_id"])
        freeze(variant, rows)
        variants[variant] = {"records": len(rows), "blocked_upstream": sum(bool(row["gsa_eval_blocked_upstream"]) for row in rows),
                             "visible_inputs_changed": sum(row["gsa_context_intervention"]["visible_input_changed"] for row in rows)}
    manifest = {"schema": "anvil-native-context-pilot-v1", "status": "COMPLETE", "variants": variants,
                "files": files, "source": {"path": str(args.history.resolve() / "qualification.jsonl"),
                    "sha256": _sha256(args.history.resolve() / "qualification.jsonl")},
                "generator_adapter": initial, "inference_args_sha256": _sha256(pack / "inference_args.json"),
                "teacher_eligible": False, "paper_metrics_admitted": False,
                "notes": ["Grounding, depth, affordance and DoF are generated sequentially by the same frozen initial joint student.",
                          "Unavailable or structurally failed parents block generation; no ground-truth fallback and no denominator removal.",
                          "Corruptions change supplied context only. Original references and all source cases remain fixed.",
                          "Alternate target identities/grasps and action-plan checklists remain incomplete; this is a small diagnostic pilot."]}
    pending_manifest = pack / ".manifest.pending.json"
    with pending_manifest.open("x") as handle:
        json.dump(manifest, handle, indent=2)
    pending_manifest.rename(pack / "manifest.json")
    print("NATIVE_CONTEXT_PACK_COMPLETE", json.dumps(variants), flush=True)


def _validate_native_context_pack(pack: Path, source: dict) -> dict:
    manifest = json.loads((pack / "manifest.json").read_text())
    if manifest.get("status") != "COMPLETE" or manifest.get("source") != source:
        raise ValueError("shared contexts do not match the frozen qualification source")
    bindings = manifest.get("files", {})
    for variant in ("oracle", "student_cascade", "corrupted"):
        name = variant + ".jsonl"
        if name not in bindings or Path(bindings[name]["path"]).resolve() != (pack / name).resolve():
            raise ValueError("shared context variant binding is missing or points elsewhere")
    for binding in bindings.values():
        if _sha256(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("shared context file binding drift")
    return manifest


def run_native_context_eval(args: argparse.Namespace) -> int:
    """Wait for one bound training job, then evaluate shared recursive contexts."""
    import sys
    worktree, output = args.worktree.resolve(), args.output.resolve()
    sys.path.insert(0, str(worktree))
    from eval.g_family_qualification import load_jsonl, native_context_rows, native_diagnostic, native_identity
    if output.exists() or args.gpu not in range(8):
        raise ValueError("context evaluation requires a fresh output and allocated GPU")
    after_run, pack = args.after_run.resolve(), args.context_pack.resolve()
    previous = json.loads((after_run / "launch.json").read_text())
    if previous["role"] != args.role or previous["arm"] != args.arm or previous["environment"]["CUDA_VISIBLE_DEVICES"] != str(args.gpu):
        raise ValueError("context evaluation role/arm/GPU differs from its preceding training job")
    birth = _process_start(args.after_pid)
    if birth is None:
        raise ValueError("preceding training handle is not live at queue creation")
    source = args.history.resolve() / "qualification.jsonl"
    history = json.loads((args.history / "manifest.json").read_text())
    if _sha256(source) != history["files"]["qualification.jsonl"]["sha256"]:
        raise ValueError("historical native qualification source drift")
    rows = load_jsonl(source, "native qualification source")
    native_context_rows(rows, "oracle")  # Preflight exact rendering before queueing.
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate historical native sample identity")
    for row in rows:
        if row["gsa_meta"]["split"] != "qualification" or _sha256(Path(row["images"][0])) != row["gsa_meta"]["image_sha256"]:
            raise ValueError("qualification image or split drift")
        expected_forms = {"depth": {"detection"}, "affordance": {"detection", "depth"},
                          "dof": {"detection", "depth", "affordance"},
                          "action_cot": {"detection", "depth", "affordance", "dof"}}.get(row["gsa_form"], set())
        parents = [by_id[sample_id] for sample_id in row["gsa_meta"]["upstream_reviewed_sample_ids"]]
        if {parent["gsa_form"] for parent in parents} != expected_forms or len(parents) != len(expected_forms):
            raise ValueError("declared historical dependency closure differs from the context pilot")
        for parent in parents:
            if parent["gsa_meta"]["image_sha256"] != row["gsa_meta"]["image_sha256"]:
                raise ValueError("historical dependency crosses source images")
            if parent["gsa_form"] == "detection":
                observed = [{"object_class": item["label"], "bbox": item["bbox_2d"]} for item in parent["gsa_payload"]]
                reference = row["gsa_context"]["groundings"]
            else:
                observed = parent["gsa_payload"]
                reference = row["gsa_context"]["spatial"][{"depth": "depth_order"}.get(parent["gsa_form"], parent["gsa_form"])]
            if observed != reference:
                raise ValueError("historical upstream payload binding drift")
    initial_lineage = args.teachers.resolve() / args.role / "completed_lineage.json"
    bindings = {"source": {"path": str(source), "sha256": _sha256(source)},
                "initial_lineage": {"path": str(initial_lineage), "sha256": _sha256(initial_lineage)},
                "driver": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))}}
    for name, path in (("evaluator", worktree / "eval/g_family_qualification.py"),
                       ("contracts", worktree / "trains/flywheel/contracts.py"),
                       ("validators", worktree / "trains/flywheel/validators.py"),
                       ("model_config", args.model.resolve() / "config.json")):
        bindings[name] = {"path": str(path), "sha256": _sha256(path)}
    frozen_contexts = getattr(args, "frozen_contexts", False)
    if frozen_contexts:
        if args.prepare_contexts or args.context_builder_pid:
            raise ValueError("frozen contexts cannot also request a live context builder")
        _validate_native_context_pack(pack, bindings["source"])
        bindings["frozen_context_manifest"] = {
            "path": str(pack / "manifest.json"), "sha256": _sha256(pack / "manifest.json")}
        context_birth = None
    elif args.prepare_contexts:
        if args.role != "G" or args.arm != "control" or pack.exists():
            raise ValueError("only the first G control worker prepares the fresh shared context pack")
        context_birth = None
    else:
        context_birth = _process_start(args.context_builder_pid) if args.context_builder_pid else None
        if context_birth is None:
            raise ValueError("shared context-builder handle is not live")
    output.mkdir(parents=True)
    receipt = {"schema": "anvil-native-context-eval-queue-v1", "role": args.role, "arm": args.arm,
        "gpu": args.gpu, "after_run": str(after_run), "after_pid": args.after_pid, "after_process_start": birth,
        "context_pack": str(pack), "context_builder_pid": args.context_builder_pid, "context_builder_start": context_birth,
        "frozen_contexts": frozen_contexts,
        "files": bindings, "paper_metrics_admitted": False, "teacher_eligible": False}

    def save(status):
        receipt["status"] = status
        receipt["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save("WAITING_FOR_BOUND_TRAINING")
    print(f"WAITING_FOR_BOUND_TRAINING pid={args.after_pid} birth={birth}", flush=True)
    while _process_start(args.after_pid) == birth:
        time.sleep(30)
    completed = json.loads((after_run / "launch.json").read_text())
    if completed["status"] != "COMPLETED_DIAGNOSTIC" or completed.get("files") != previous.get("files"):
        save("PRECEDING_RUN_NOT_COMPLETED_OR_INPUTS_CHANGED")
        return 3
    for binding in bindings.values():
        if _sha256(Path(binding["path"])) != binding["sha256"]:
            save("QUEUED_INPUT_DRIFT")
            return 3
    save("WAITING_FOR_GPU")
    while True:
        status = subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True)
        if status.returncode == 0 and not status.stdout.strip():
            break
        time.sleep(30)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), IMAGE_MAX_TOKEN_NUM="1536", HF_HUB_OFFLINE="1")
    if args.prepare_contexts:
        save("GENERATING_SHARED_STUDENT_CASCADE")
        # The worker subprocess releases the generator model before scoring teachers.
        command = [sys.executable, str(Path(__file__).resolve()), "generate-native-context-pack",
                   "--worktree", str(worktree), "--history", str(args.history.resolve()), "--teachers", str(args.teachers.resolve()),
                   "--model", str(args.model.resolve()), "--output", str(pack)]
        code = subprocess.run(command, cwd=worktree).returncode
        if code:
            receipt["generator_exit_code"] = code
            save("CONTEXT_GENERATION_FAILED")
            return code
    elif not frozen_contexts:
        save("WAITING_FOR_SHARED_CONTEXTS")
        while not (pack / "manifest.json").is_file():
            if _process_start(args.context_builder_pid) != context_birth:
                save("CONTEXT_BUILDER_EXITED_WITHOUT_PACK")
                return 3
            time.sleep(30)
    manifest = _validate_native_context_pack(pack, bindings["source"])
    variants = {variant: load_jsonl(pack / (variant + ".jsonl"), "frozen native contexts")
                for variant in ("oracle", "student_cascade", "corrupted")}
    unique = {}
    for cases in variants.values():
        for row in cases:
            if row.get("gsa_eval_blocked_upstream"):
                continue
            key = native_identity(row)
            if key in unique and unique[key]["gsa_payload"] != row["gsa_payload"]:
                raise ValueError("identical context prompts have inconsistent reference labels")
            unique.setdefault(key, row)
    query_path = output / "unique_queries.jsonl"
    with query_path.open("x") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in unique.values())
    candidates = {args.role + "_" + args.arm: completed["completed"]["persistent_selected_adapter"]}
    if args.arm == "control":
        candidates[args.role + "_initial"] = json.loads(initial_lineage.read_text())["persistent_selected_adapter"]
    receipt["context_manifest_sha256"] = _sha256(pack / "manifest.json")
    receipt["unique_queries"] = len(unique)
    receipt["candidates"] = candidates
    for name, adapter in candidates.items():
        for binding in adapter["files"].values():
            if _sha256(Path(binding["path"])) != binding["sha256"]:
                raise ValueError("context evaluation adapter binding drift")
        model_output = output / name
        model_output.mkdir()
        save("EVALUATING_" + name)
        command = [sys.executable, str(Path(__file__).resolve()), "native-infer", "--worktree", str(worktree),
                   "--model", str(args.model.resolve()), "--adapter", adapter["path"], "--source", str(query_path),
                   "--output", str(model_output)]
        code = subprocess.run(command, cwd=worktree).returncode
        if code:
            receipt["inference_exit_code"] = code
            save("CONTEXT_EVALUATION_FAILED_" + name)
            return code
        predicted = load_jsonl(model_output / "predictions.jsonl", "context predictions")
        by_identity = {native_identity(row): row for row in predicted}
        if len(by_identity) != len(predicted) or set(by_identity) != set(unique):
            raise ValueError("context inference coverage differs from the frozen query union")
        for variant, cases in variants.items():
            prediction_path = model_output / (variant + "_predictions.jsonl")
            with prediction_path.open("x") as handle:
                handle.writelines(json.dumps(by_identity[native_identity(row)], ensure_ascii=False) + "\n"
                                  for row in cases if not row.get("gsa_eval_blocked_upstream"))
            diagnostic = native_diagnostic(pack / (variant + ".jsonl"), [(name, prediction_path)])
            with (model_output / (variant + "_diagnostic.json")).open("x") as handle:
                json.dump(diagnostic, handle, indent=2)
    save("COMPLETED_CONTEXT_DIAGNOSTICS")
    return 0


def generate_native_context_pack(args: argparse.Namespace) -> int:
    import sys
    sys.path.insert(0, str(args.worktree.resolve()))
    rows = [json.loads(line) for line in (args.history / "qualification.jsonl").read_text().splitlines()]
    _native_context_pack(args, rows, args.output.resolve())
    return 0


def native_infer(args: argparse.Namespace) -> int:
    import sys
    sys.path.insert(0, str(args.worktree.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("native inference requires an empty output directory")
    rows = [json.loads(line) for line in args.source.read_text().splitlines()]
    engine = _native_engine(args.model.resolve(), args.adapter.resolve(), args.output.resolve())
    _native_infer_batches(engine, rows, args.output.resolve() / "predictions.jsonl")
    return 0


def _sft_message_projection(rows: list[dict]) -> list[dict]:
    """Keep SFT model inputs verbatim while retaining heterogeneous audits elsewhere."""
    projected = []
    for row in rows:
        if (not isinstance(row.get("messages"), list) or not row["messages"]
                or row["messages"][-1].get("role") != "assistant"
                or not isinstance(row.get("images"), list) or not row["images"]):
            raise ValueError("SFT projection requires complete messages and explicit images")
        if any(row.get(key) for key in ("videos", "audios", "objects", "tools")):
            raise ValueError("SFT projection does not support additional modality/tool inputs")
        projected.append({"messages": row["messages"], "images": row["images"]})
    return projected


def run_teacher_refresh(args: argparse.Namespace) -> int:
    """Continue one saved specialist and evaluate its fixed-budget terminal adapter."""
    import random
    worktree, output = args.worktree.resolve(), args.output.resolve()
    history, teachers, model = args.history.resolve(), args.teachers.resolve(), args.model.resolve()
    if output.exists() or args.steps < 1 or args.seed < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise SystemExit("refresh requires a fresh output, positive steps/rate and nonnegative seed")
    if args.gpu not in range(8):
        raise SystemExit("refresh GPU must be one of 0..7")
    suffix = "" if args.role == "joint" else "_" + args.role
    train, dev, qualification = history / f"train{suffix}.jsonl", history / f"dev{suffix}.jsonl", history / "qualification.jsonl"
    old_launch = json.loads((teachers / "launch.json").read_text())
    previous = next(job for job in old_launch["jobs"] if job["name"] == args.role)
    if _sha256(train) != previous["train_sha256"] or _sha256(dev) != previous["dev_sha256"]:
        raise SystemExit("historical native training/development bindings changed")
    previous_lineage = json.loads((teachers / args.role / "completed_lineage.json").read_text())
    initial = previous_lineage["persistent_selected_adapter"]
    for item in initial["files"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise SystemExit("initial teacher adapter or lineage changed")
    paths = {"historical_train": train, "historical_dev": dev, "qualification": qualification,
             "initial_lineage": teachers / args.role / "completed_lineage.json",
             "launcher": worktree / "scripts/run_sft_9b_lora_l20_dlc.sh",
             "model_config": model / "config.json"}
    historical_rows = [json.loads(line) for line in train.read_text().splitlines()]
    if not historical_rows or not dev.stat().st_size:
        raise SystemExit("empty historical teacher data")
    rows, admission = historical_rows, "HISTORICAL_HUMAN_LABEL_REPLAY_CONTROL"
    feedback_binding = None
    if args.arm == "feedback":
        if args.feedback is None:
            raise SystemExit("feedback arm requires --feedback")
        root = args.feedback.resolve()
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        corpus = root / "audit_corpus.jsonl"
        source_sha = _sha256(corpus)
        if source_sha != manifest["files"][corpus.name]["sha256"]:
            raise SystemExit("feedback corpus binding changed")
        retention = json.loads((root / "audit/retention.json").read_text())
        spatial = json.loads((root / "audit/spatial/report.json").read_text())
        if (retention.get("state") != "PASS" or retention.get("posttraining_corpus_sha256") != source_sha
                or spatial.get("contamination_verdict") != "PASS"
                or spatial.get("posttraining_corpus", {}).get("sha256") != source_sha):
            raise SystemExit("feedback does not pass both bound contamination screens")
        review_path = args.review_report.resolve() if args.review_report else root / "preaudit_v1/report.json"
        review = json.loads(review_path.read_text())
        for binding in review["provenance"].values():
            if _sha256(Path(binding["path"])) != binding["sha256"]:
                raise SystemExit("independent feedback review binding drift")
        if review.get("gate_passed") is not True or review.get("raw_decision_agreement", 0) <= .9:
            raise SystemExit("feedback independent machine agreement has not exceeded 90%")
        public_path = Path(review["provenance"]["public_inputs"]["path"])
        public = {row["sample_id"]: row for row in map(json.loads, public_path.read_text().splitlines())}
        # A rejection by either reviewer excludes the image and every dependent
        # task on it. Do not train the conflicting targets while humans adjudicate.
        rejected_images = set()
        for side in ("a", "b"):
            stream_path = Path(review["provenance"][f"reviewer_{side}"]["path"])
            for row in map(json.loads, stream_path.read_text().splitlines()):
                if not row["supported"]:
                    rejected_images.add(public[row["sample_id"]]["image_sha256"])
        feedback_train = root / f"train{suffix}.jsonl"
        if _sha256(feedback_train) != manifest["files"][feedback_train.name]["sha256"]:
            raise SystemExit("feedback training slice changed")
        new_rows = [row for row in map(json.loads, feedback_train.read_text().splitlines())
                    if row["gsa_meta"]["image_sha256"] not in rejected_images]
        for row in new_rows:
            observed = public.get(row["sample_id"])
            if observed and (observed["payload"] != row["gsa_payload"]
                             or observed["image_sha256"] != row["gsa_meta"]["image_sha256"]
                             or observed["instruction"] != "\n".join(m["content"] for m in row["messages"] if m["role"] == "user")):
                raise SystemExit("reviewed feedback task no longer matches its visible review input")
        if not new_rows:
            raise SystemExit("no feedback training rows survive independent review exclusions")
        replay_count = math.ceil(len(new_rows) / 3)
        rows = new_rows + [historical_rows[i % len(historical_rows)] for i in range(replay_count)]
        feedback_binding = {"manifest_sha256": _sha256(manifest_path), "source_sha256": source_sha,
                            "review_path": str(review_path), "review_sha256": _sha256(review_path), "new_rows": len(new_rows),
                            "historical_replay_rows": replay_count, "review_rejected_images": sorted(rejected_images)}
        admission = "PROVISIONAL_MACHINE_VERIFIED_FEEDBACK_WITH_HUMAN_REVIEW_PENDING"
        paths.update(feedback_train=feedback_train, feedback_manifest=manifest_path,
                     feedback_review=review_path, retention=root / "audit/retention.json",
                     spatial=root / "audit/spatial/report.json")
        calibration = review_path.parent / "calibration.json"
        if calibration.is_file():
            paths["review_calibration"] = calibration
    # Never consume a historical development or qualification image during refresh.
    reserved = {row["gsa_meta"]["image_sha256"] for split in ("dev", "qualification")
                for row in map(json.loads, (history / f"{split}.jsonl").read_text().splitlines())}
    if any(row["gsa_meta"]["image_sha256"] in reserved for row in rows):
        raise SystemExit("refresh training contains a reserved historical image")
    output.mkdir(parents=True)
    random.Random(args.seed).shuffle(rows)
    train_path = output / "train.jsonl"
    train_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    paths["actual_train"] = train_path
    if args.project_sft_messages:
        train_path = output / "sft_messages.jsonl"
        train_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                      for row in _sft_message_projection(rows)))
        paths["model_training_inputs"] = train_path
    runtime = Path(tempfile.mkdtemp(prefix=f"anvil-refresh-{args.role}-{args.arm}-", dir="/tmp"))
    (output / "runtime").symlink_to(runtime, target_is_directory=True)
    environment = {**os.environ, "GSA_ROOT": str(worktree), "MODEL": str(model),
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "MASTER_PORT": str(29610 + args.gpu),
        "INITIAL_ADAPTER": initial["path"], "TRAIN": str(train_path), "VAL": str(dev),
        "OUT": str(runtime / "sft"), "BS": "1", "GA": "8", "RANK": "64", "ALPHA": "128",
        "DROPOUT": "0.05", "SEED": str(args.seed), "LR": str(args.lr), "STEPS": str(args.steps),
        "EVAL_STEPS": str(args.steps), "SAVE_STEPS": str(args.steps), "HOURS": str(args.hours),
        "DATALOADER_NUM_WORKERS": "2", "DATASET_NUM_PROC": "2", "REPORT_TO": "none",
        "COUNT_INPUT_TOKENS": "1", "PYTHONUNBUFFERED": "1", "IMAGE_MAX_TOKEN_NUM": "1536"}
    receipt = {"schema": "anvil-independent-teacher-refresh-v1", "role": args.role, "arm": args.arm,
        "status": "PREPARED", "training_admission": admission, "initial_adapter": initial,
        "feedback": feedback_binding, "files": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in paths.items()},
        "environment": {key: environment[key] for key in ("CUDA_VISIBLE_DEVICES", "MASTER_PORT", "INITIAL_ADAPTER", "TRAIN", "VAL", "OUT", "BS", "GA", "RANK", "ALPHA", "DROPOUT", "SEED", "LR", "STEPS", "EVAL_STEPS", "SAVE_STEPS", "HOURS", "COUNT_INPUT_TOKENS", "IMAGE_MAX_TOKEN_NUM")},
        "train_rows": len(rows), "steps": args.steps, "effective_batch": 8,
        "sft_message_projection": args.project_sft_messages,
        "teacher_eligible": False, "paper_metrics_admitted": False, "builder_sha256": _sha256(Path(__file__)),
        "notes": ["Optimizer state resets in both arms; initial teacher weights are identical within each pair.",
                  "Steps and example exposures match; actual input tokens are recorded, not assumed equal.",
                  "Development is historical and fixed; the terminal checkpoint is predetermined.",
                  "Only oracle-context held-out diagnostics are produced here; no automatic teacher qualification."]}
    receipt_path = output / "launch.json"

    def save(status):
        receipt["status"] = status
        receipt["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    launcher = ["bash", str(paths["launcher"])]
    save("PREFLIGHT")
    subprocess.run(launcher, cwd=worktree, env={**environment, "PREFLIGHT_ONLY": "1"}, check=True)
    if args.preflight_only:
        save("PREFLIGHT_ONLY")
        return 0
    save("TRAINING")
    code = subprocess.run(launcher, cwd=worktree, env={**environment, "PREFLIGHT_ONLY": "0"}).returncode
    if code:
        receipt["training_exit_code"] = code
        save("TRAINING_FAILED")
        return code
    versions = list((runtime / "sft").glob("v*/args.json"))
    if len(versions) != 1:
        raise SystemExit("refresh must produce exactly one fresh runtime version")
    actual_path, version = versions[0], versions[0].parent
    actual = json.loads(actual_path.read_text())
    expected = {"dataset": [str(train_path)], "val_dataset": [str(dev)], "adapters": [initial["path"]],
                "resume_from_checkpoint": None, "model": str(model), "max_steps": args.steps,
                "seed": args.seed, "data_seed": args.seed, "learning_rate": args.lr,
                "gradient_accumulation_steps": 8, "per_device_train_batch_size": 1,
                "include_num_input_tokens_seen": True}
    if any(actual.get(key) != value for key, value in expected.items()):
        raise SystemExit("actual refresh configuration differs from its receipt")
    checkpoint = version / f"checkpoint-{args.steps}"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state.get("global_step") != args.steps:
        raise SystemExit("refresh step budget incomplete")
    finite = ("import sys,torch;from safetensors import safe_open;torch.set_num_threads(2)\n"
              "with safe_open(sys.argv[1],framework='pt',device='cpu') as h:\n"
              " assert h.keys()\n"
              " for k in h.keys(): assert torch.isfinite(h.get_tensor(k)).all(), k\n")
    shell = ["bash", "-c", 'source "$1/.env"; shift; exec "$@"', "anvil-refresh", str(worktree)]
    subprocess.run(shell + ["python", "-c", finite, str(checkpoint / "adapter_model.safetensors")], env=environment, check=True)
    completed_paths = {"trainer_args": actual_path, "final_state": checkpoint / "trainer_state.json",
                       "selected_adapter_config": checkpoint / "adapter_config.json",
                       "selected_adapter_weights": checkpoint / "adapter_model.safetensors"}
    lineage = {"files": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in completed_paths.items()}}
    receipt["completed"] = {"steps": args.steps, "num_input_tokens_seen": state.get("num_input_tokens_seen"),
                            "selected_adapter_finite": True,
                            "persistent_selected_adapter": _persist_selected_adapter(lineage, output)}
    save("TRAINED_EVALUATING")
    predictions = output / "qualification_predictions.jsonl"
    inference = ["swift", "infer", "--model", str(model), "--adapters", str(output / "selected_adapter"),
                 "--val_dataset", str(qualification), "--result_path", str(predictions),
                 "--infer_backend", "transformers", "--max_batch_size", "4", "--max_new_tokens", "512",
                 "--max_length", "3072", "--temperature", "0", "--seed", str(args.seed),
                 "--enable_thinking", "false", "--stream", "false", "--load_args", "false"]
    receipt["inference_command"] = inference
    save("TRAINED_EVALUATING")
    inference_environment = {key: value for key, value in environment.items() if key not in _INFERENCE_DISTRIBUTED_ENV}
    inference_shell = ["bash", "-c", 'source "$1/.env"; shift; unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE NODE_RANK GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE _PATCH_WORLD_SIZE LOCAL_SIZE NPROC_PER_NODE; exec "$@"', "anvil-refresh", str(worktree)]
    subprocess.run(inference_shell + inference, cwd=worktree, env=inference_environment, check=True)
    diagnostic = output / "qualification_diagnostic.json"
    subprocess.run(shell + ["python", "eval/g_family_qualification.py", "native-diagnostic",
        "--source", str(qualification), "--prediction", args.role + "_" + args.arm, str(predictions),
        "--prediction", "initial", str(teachers / args.role / "qualification_predictions.jsonl"),
        "--out", str(diagnostic)], cwd=worktree, env=environment, check=True)
    receipt["evaluation"] = {"predictions_sha256": _sha256(predictions), "diagnostic_sha256": _sha256(diagnostic)}
    save("COMPLETED_DIAGNOSTIC")
    return 0


def recover_teacher_refresh(args: argparse.Namespace) -> int:
    """Recover evaluation from a completed saved adapter without restarting training."""
    import copy
    import sys
    worktree, original, output = args.worktree.resolve(), args.after_run.resolve(), args.output.resolve()
    if output.exists() or args.gpu not in range(8):
        raise ValueError("evaluation recovery requires a fresh output and allocated GPU")
    initial_bytes = (original / "launch.json").read_bytes()
    previous = json.loads(initial_bytes)
    if previous["environment"]["CUDA_VISIBLE_DEVICES"] != str(args.gpu):
        raise ValueError("recovery GPU does not match original training allocation")
    birth = _process_start(args.after_pid)
    if birth is None and previous["status"] not in ("TRAINED_EVALUATING", "COMPLETED_DIAGNOSTIC"):
        raise ValueError("training is neither live nor completed with a saved adapter")
    for record in previous["files"].values():
        if _sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("original training input binding drift")
    output.mkdir(parents=True)
    with (output / "original_launch_at_queue.json").open("xb") as handle:
        handle.write(initial_bytes)
    receipt = copy.deepcopy(previous)
    receipt["builder_sha256"] = _sha256(Path(__file__))
    receipt["recovery"] = {"original_run": str(original), "after_pid": args.after_pid, "process_start": birth,
                           "reason": "single-process inference inherited LoRA RANK as distributed rank",
                           "training_restarted": False, "cleared_inference_environment": list(_INFERENCE_DISTRIBUTED_ENV)}

    def save(status):
        receipt["status"] = status
        receipt["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save("WAITING_FOR_BOUND_TRAINING")
    if birth is not None:
        while _process_start(args.after_pid) == birth:
            time.sleep(15)
    terminal_bytes = (original / "launch.json").read_bytes()
    terminal = json.loads(terminal_bytes)
    with (output / "original_launch_terminal.json").open("xb") as handle:
        handle.write(terminal_bytes)
    if (terminal["status"] not in ("TRAINED_EVALUATING", "COMPLETED_DIAGNOSTIC")
            or terminal["files"] != previous["files"] or not terminal.get("completed", {}).get("selected_adapter_finite")
            or terminal["completed"]["steps"] != terminal["steps"]):
        save("PRECEDING_TRAINING_NOT_RECOVERABLE")
        return 3
    adapter = terminal["completed"]["persistent_selected_adapter"]
    for record in [*terminal["files"].values(), *adapter["files"].values(), *terminal["initial_adapter"]["files"].values()]:
        if _sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("completed training/adapter binding drift")
    state = json.loads(Path(adapter["files"]["final_state"]["path"]).read_text())
    if state["global_step"] != terminal["steps"] or state.get("num_input_tokens_seen") != terminal["completed"]["num_input_tokens_seen"]:
        raise ValueError("saved training budget differs from completion receipt")
    receipt["completed"] = terminal["completed"]
    receipt["recovery"]["terminal_launch_sha256"] = _sha256(output / "original_launch_terminal.json")
    save("WAITING_FOR_GPU")
    while True:
        status = subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True)
        if status.returncode == 0 and not status.stdout.strip():
            break
        time.sleep(15)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), IMAGE_MAX_TOKEN_NUM=terminal["environment"]["IMAGE_MAX_TOKEN_NUM"],
                      HF_HUB_OFFLINE="1")
    sys.path.insert(0, str(worktree))
    from eval.g_family_qualification import load_jsonl, native_diagnostic
    source = Path(terminal["files"]["qualification"]["path"])
    rows = load_jsonl(source, "recovery qualification source")
    inference = output / "inference"
    inference.mkdir()
    save("TRAINED_EVALUATING")
    engine = _native_engine(Path(terminal["files"]["model_config"]["path"]).parent, Path(adapter["path"]), inference)
    predictions = output / "qualification_predictions.jsonl"
    _native_infer_batches(engine, rows, predictions)
    initial_predictions = args.teachers.resolve() / terminal["role"] / "qualification_predictions.jsonl"
    diagnostic = native_diagnostic(source, [(terminal["role"] + "_" + terminal["arm"], predictions),
                                             ("initial", initial_predictions)])
    diagnostic_path = output / "qualification_diagnostic.json"
    with diagnostic_path.open("x") as handle:
        json.dump(diagnostic, handle, indent=2)
    receipt["evaluation"] = {"predictions_sha256": _sha256(predictions), "diagnostic_sha256": _sha256(diagnostic_path)}
    receipt["inference_command"] = "native API single-process recovery; see inference/inference_args.json"
    save("COMPLETED_DIAGNOSTIC")
    print("RECOVERY_COMPLETED_DIAGNOSTIC", terminal["role"], terminal["arm"], flush=True)
    return 0


def _persist_selected_adapter(lineage: dict, output: Path) -> dict:
    """Keep the selected LoRA and provenance durable when bulky runtime is elsewhere."""
    target = output / "selected_adapter"
    if target.exists():
        raise SystemExit("refusing to overwrite a preserved selected adapter")
    filenames = {"selected_adapter_config": "adapter_config.json",
                 "selected_adapter_weights": "adapter_model.safetensors",
                 "trainer_args": "trainer_args.json", "final_state": "trainer_state.json"}
    files = {}
    with tempfile.TemporaryDirectory(prefix=".selected-adapter-", dir=output) as temporary:
        staging = Path(temporary) / "adapter"
        staging.mkdir()
        for key, name in filenames.items():
            binding = lineage["files"][key]
            shutil.copyfile(binding["path"], staging / name)
            if _sha256(staging / name) != binding["sha256"]:
                raise SystemExit(f"selected adapter changed while preserving it: {key}")
            files[key] = {"path": str(target / name), "sha256": binding["sha256"]}
        staging.rename(target)
    return {"path": str(target), "files": files}


def queue_clean_sft(args: argparse.Namespace) -> int:
    """Wait for the bound evaluation queue, then run fresh SFT and full evaluation."""
    import fcntl

    worktree, output = args.worktree.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit("queued SFT requires a fresh output directory; no implicit resume")
    if (args.seed < 0 or not math.isfinite(args.lr) or args.lr <= 0
            or args.hours < 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", args.tag)):
        raise SystemExit("invalid seed, learning rate, hour cap or result tag")
    binding = _clean_sft_inputs(args)
    started = _process_start(args.after_pid)
    command_path = Path(f"/proc/{args.after_pid}/cmdline")
    if started is None or not command_path.is_file():
        raise SystemExit("the preceding evaluation queue must be confirmed live when binding")
    command = command_path.read_bytes().split(b"\0")
    if not any(Path(os.fsdecode(part)).name == "queue_anvil_completed_eval.sh" for part in command if part):
        raise SystemExit("after-pid is not the Anvil completion evaluation queue")
    if not args.after_log.is_file():
        raise SystemExit("preceding evaluation queue log is missing")
    steps = math.ceil(binding["train_rows"] / 32)
    runtime = args.runtime_root.resolve() if args.runtime_root else output
    if runtime != output and runtime.exists():
        raise SystemExit("external runtime requires a fresh directory")
    run_root, merged, retention = runtime / "sft", runtime / "merged", output / "retention"
    if (worktree / "eval_results" / args.tag).exists():
        raise SystemExit("evaluation tag is already in use")
    env = {**os.environ, "GSA_ROOT": str(worktree), "MODEL": str(args.model.resolve()),
           "TRAIN": binding["files"]["train"]["path"], "VAL": binding["files"]["dev"]["path"],
           "OUT": str(run_root), "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "BS": "1", "GA": "4",
           "RANK": "64", "ALPHA": "128", "DROPOUT": "0.05", "SEED": str(args.seed),
           "LR": str(args.lr), "STEPS": str(steps), "EVAL_STEPS": "1000", "SAVE_STEPS": "1000",
           "HOURS": str(args.hours), "DATALOADER_NUM_WORKERS": "4", "DATASET_NUM_PROC": "8",
           "REPORT_TO": "none", "PYTHONUNBUFFERED": "1", "DS_IGNORE_CUDA_DETECTION": "1",
           "NCCL_IB_DISABLE": "1", "NCCL_P2P_LEVEL": "PIX"}
    launcher = ["bash", binding["files"]["sft_launcher"]["path"]]
    subprocess.run(launcher, cwd=worktree, env={**env, "PREFLIGHT_ONLY": "1"}, check=True)
    receipt = {"schema": "anvil-queued-clean-sft-v1", "inputs": binding,
               "after_pid": args.after_pid, "after_process_start": started,
               "after_log": str(args.after_log.resolve()), "output": str(output), "tag": args.tag,
               "runtime_root": str(runtime), "external_runtime": runtime != output,
               "steps": steps, "effective_batch": 32, "seed": args.seed, "lr": args.lr,
               "initialization": "fresh base, new LoRA; no P4 continuation",
               "builder_sha256": _sha256(Path(__file__))}
    print(json.dumps(receipt, indent=2), flush=True)
    if args.preflight_only:
        print("CLEAN_SFT_QUEUE_PREFLIGHT_PASS: no waiting or training started", flush=True)
        return 0
    output.mkdir(parents=True)
    if runtime != output:
        runtime.mkdir(parents=True)
        (output / "runtime").symlink_to(runtime, target_is_directory=True)
    (output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"WAITING_FOR_EVALUATION_QUEUE pid={args.after_pid} birth={started}", flush=True)
    while _process_start(args.after_pid) == started:
        time.sleep(30)
    if "EVALUATION_QUEUE_FINISHED" not in args.after_log.read_text(errors="replace"):
        raise SystemExit("preceding evaluation queue stopped without its terminal completion marker")
    # Share the existing queue lock, and check actual compute processes as well.
    # A released lock alone does not prove that GPU jobs have stopped.
    with (worktree / "output/.anvil_completed_eval.lock").open("a") as lock:
        print("WAITING_FOR_GPU_ALLOCATION", flush=True)
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(30)
        while True:
            status = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                    text=True, capture_output=True)
            if status.returncode == 0 and not status.stdout.strip():
                break
            time.sleep(30)
        if _clean_sft_inputs(args) != binding:
            raise SystemExit("queued inputs or launchers changed while waiting; refusing stale execution")
        if run_root.exists() or (worktree / "eval_results" / args.tag).exists():
            raise SystemExit("queued output or evaluation tag was occupied while waiting")
        env["PREFLIGHT_ONLY"] = "0"
        print("CLEAN_SFT_STARTING", flush=True)
        subprocess.run(launcher, cwd=worktree, env=env, check=True)
        versions = [path for path in run_root.glob("v*") if path.is_dir() and (path / "args.json").is_file()]
        if len(versions) != 1:
            raise SystemExit("expected one versioned fresh SFT run")
        version = versions[0]
        lineage = _completed_sft_lineage(version, binding, args.model, steps, args.seed, args.lr)
        finite_check = (
            "import sys, torch; from safetensors import safe_open; torch.set_num_threads(2)\n"
            "with safe_open(sys.argv[1], framework='pt', device='cpu') as handle:\n"
            " assert handle.keys(), 'empty adapter'\n"
            " for key in handle.keys():\n"
            "  assert torch.isfinite(handle.get_tensor(key)).all(), f'nonfinite adapter: {key}'\n"
            "print('SELECTED_ADAPTER_FINITE')\n"
        )
        subprocess.run(["bash", "-c", 'source "$1/.env"; exec python -c "$2" "$3"', "anvil-finite",
                        str(worktree), finite_check, lineage["files"]["selected_adapter_weights"]["path"]],
                       cwd=worktree, env=env, check=True)
        lineage["selected_adapter_finite"] = True
        if runtime != output:
            lineage["persistent_selected_adapter"] = _persist_selected_adapter(lineage, output)
        (output / "completed_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")
        print("CLEAN_SFT_COMPLETE_STARTING_SPATIAL_EVALUATION", flush=True)
        eval_env = {**env, "NUM_GPUS": "8", "THINKING": "off", "REQUIRE_COMPLETED_STEPS": str(steps),
                    "CELL_TIMEOUT": "4h", "MERGED_DIR": str(merged)}
        spatial_result = subprocess.run(["bash", binding["files"]["spatial_launcher"]["path"], args.tag, str(version),
                        "cv_bench embspatial refspatial_bench where2place gsa_bench_test", "0", "16"],
                       cwd=worktree, env=eval_env)
        # A spatial score-health rejection must not suppress the independent
        # retention experiment. Keep its failing exit status and all raw evidence.
        exit_codes = {"spatial": spatial_result.returncode, "retention": None}
        (output / "evaluation_exit_codes.json").write_text(json.dumps(exit_codes, indent=2) + "\n")
        print(f"SPATIAL_EVALUATION_EXIT={spatial_result.returncode}; starting independent retention evaluation", flush=True)
        retention_env = {**eval_env, "CUDA_VISIBLE_DEVICES": "0,1,2,3", "BASE_MODEL": str(args.model.resolve()),
                         "E1_MODEL": str(merged), "OUT": str(retention), "RUN_STAGE": "paired",
                         "RESUME": "0", "CELL_TIMEOUT": "8h"}
        result = subprocess.run(["bash", binding["files"]["retention_launcher"]["path"]],
                                cwd=worktree, env=retention_env)
        exit_codes["retention"] = result.returncode
        (output / "evaluation_exit_codes.json").write_text(json.dumps(exit_codes, indent=2) + "\n")
        base_result, candidate_result = retention / "results/base.json", retention / "results/e1.json"
        if base_result.is_file() and candidate_result.is_file():
            subprocess.run(["bash", "-c", 'source "$1/.env"; shift; exec python "$@"', "anvil-parser-audit",
                            str(worktree), str(worktree / "eval/general_retention_mc_compat.py"),
                            "--base-result", str(base_result), "--candidate-result", str(candidate_result),
                            "--out", str(retention / "results/mc_compat_v1.json")], cwd=worktree, env=env, check=True)
        print(f"CLEAN_SFT_EVALUATION_FINISHED spatial_exit={spatial_result.returncode} retention_exit={result.returncode}; inspect score eligibility", flush=True)
        return spatial_result.returncode or result.returncode


def prepare_vacancy_repair(args: argparse.Namespace) -> int:
    """Restore audited vacancy supervision with deterministic replay, never test labels."""
    root, output = args.worktree.resolve(), args.output.resolve()
    package = Path(__file__).resolve().parents[1]
    p4 = root / "trains/corpus_source_clean_v1/p4/train.jsonl"
    p5 = package / "engineering/data/p5_retention_clean_split_v1/train.jsonl"
    if output.exists():
        raise SystemExit("vacancy repair requires a fresh output directory")
    expected_p4 = "1afebc5a30ebdb98d3d9173c53d1b5e350f2b9d06d2693015ee35726516b6f82"
    if _sha256(p4) != expected_p4:
        raise SystemExit("P4 source differs from the audited corpus")
    p5_manifest = json.loads((p5.parent / "manifest.json").read_text())
    if _sha256(p5) != p5_manifest["train"]["sha256"]:
        raise SystemExit("P5 training split differs from its bound manifest")
    vacancy = []
    with p4.open() as handle:
        for line in handle:
            row = json.loads(line)
            if (row.get("gsa_meta") or {}).get("source") == "refspatial_3d_vacant":
                vacancy.append(row)
    if len(vacancy) != 3812:
        raise SystemExit("vacancy source count drift")
    vacancy_images = {p for row in vacancy for p in _images(row)}
    replay = []
    with p5.open() as handle:
        for line in handle:
            row = json.loads(line)
            if not vacancy_images.intersection(_images(row)):
                key = hashlib.sha256(b"anvil-vacancy-replay-v1\0" + line.encode()).hexdigest()
                replay.append((key, row))
    replay.sort(key=lambda value: value[0])
    replay = [row for _, row in replay[:3812]]
    if len(replay) != 3812:
        raise SystemExit("insufficient replay rows")
    output.mkdir(parents=True)
    corpus = output / "candidates.jsonl"
    with corpus.open("x") as handle:
        for row in vacancy + replay:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    split_corpus(argparse.Namespace(input=corpus, output=output / "split", force=False,
                 split_id="anvil-vacancy-repair-v1", salt="anvil-vacancy-repair-v1", dev_permille=50))
    receipt = {"schema": "anvil-vacancy-repair-v1", "vacancy_rows": 3812,
               "replay_rows": 3812, "p4_sha256": _sha256(p4), "p5_train_sha256": _sha256(p5),
               "source_policy": "P4 image-audited RefSpatial training source plus P5 train-only replay; no public-test labels.",
               "scope": "Source-coverage adaptation, not a causal flywheel ablation or full MTD.",
               "selection": "Frozen recipe; no test-driven checkpoint selection. Image-disjoint internal dev, not scene-disjoint."}
    (output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    vacancy = subparsers.add_parser("prepare-vacancy-repair", help="build bounded audited vacancy/replay adaptation data")
    vacancy.add_argument("--worktree", type=Path, required=True)
    vacancy.add_argument("--output", type=Path, required=True)
    vacancy.set_defaults(func=prepare_vacancy_repair)
    split = subparsers.add_parser(
        "split-corpus", help="build an image-disjoint deterministic train/dev split"
    )
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--split-id", default="anvil-p4-image-disjoint-v1")
    split.add_argument("--salt", default="anvil-p4-image-disjoint-v1")
    split.add_argument("--dev-permille", type=int, default=10)
    split.add_argument("--force", action="store_true")
    split.set_defaults(func=split_corpus)
    source = subparsers.add_parser("prepare-scannet-flywheel", help="prepare source-bound, scene-disjoint generation inputs")
    source.add_argument("--source-root", type=Path, required=True)
    source.add_argument("--split-root", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)
    source.add_argument("--salt", default="anvil-scannet-scene-v1")
    source.add_argument("--dev-permille", type=int, default=100)
    source.add_argument("--max-image-bytes", type=int, default=4000000)
    source.set_defaults(func=prepare_scannet_flywheel)
    sensor = subparsers.add_parser("prepare-sensor-supervision", help="derive native G/S candidates from frozen training RGB-D")
    for flag in ("worktree", "input-root", "output", "scannet-root", "label-map"):
        sensor.add_argument("--" + flag, type=Path, required=True)
    sensor.add_argument("--limit", type=int, default=0, help="First N frozen source images; 0 selects all")
    sensor.add_argument("--include-masks", action="store_true", help="Also derive fidelity-checked single polygons from official instance masks")
    sensor.set_defaults(func=prepare_sensor_supervision)
    jacquard = subparsers.add_parser("prepare-jacquard-references", help="bind source-native grasp geometry without inventing Anvil contact labels")
    for flag in ("archive", "output"):
        jacquard.add_argument("--" + flag, type=Path, required=True)
    jacquard.add_argument("--expected-sha256", required=True)
    jacquard.add_argument("--dev-objects", type=int, default=2)
    jacquard.add_argument("--seed", default="anvil-jacquard-reference-v1")
    jacquard.set_defaults(func=prepare_jacquard_references)
    coco = subparsers.add_parser("prepare-coco-masks", help="derive native mask candidates from official COCO train2017 polygons")
    for flag in ("worktree", "annotations-zip", "image-cache", "output"):
        coco.add_argument("--" + flag, type=Path, required=True)
    coco.add_argument("--expected-annotations-sha256", required=True)
    coco.add_argument("--limit", type=int, default=64)
    coco.add_argument("--workers", type=int, default=4)
    coco.add_argument("--seed", default="anvil-coco-mask-v1")
    coco.add_argument("--dev-permille", type=int, default=100)
    coco.add_argument("--mask-min-iou", type=float, default=.95)
    coco.add_argument("--polygon-max-vertices", type=int, default=48)
    coco.add_argument("--min-box-side-px", type=float, default=20)
    coco.add_argument("--categories", nargs="+", help="Explicit supported source categories; omitted uses all40")
    coco.add_argument("--exclude-corpus", type=Path, action="append", default=[],
                      help="Reserve all official COCO source/RGB identities in this prior corpus; repeatable")
    coco.add_argument("--direct-download", action="store_true", help="Fetch public official S3 objects without environment proxies")
    coco.set_defaults(func=prepare_coco_masks)
    replay = subparsers.add_parser("prepare-native-replay", help="recover strict native tasks from audited same-image synthetic labels")
    for flag in ("worktree", "corpus", "output", "model", "retention-audit", "spatial-audit"):
        replay.add_argument("--" + flag, type=Path, required=True)
    replay.add_argument("--exclude-corpus", type=Path, action="append", required=True,
                        help="Reserve all image bytes in this corpus; may be repeated")
    replay.set_defaults(func=prepare_native_replay)
    feedback = subparsers.add_parser("prepare-native-feedback", help="freeze native flywheel candidates with exact upstream bindings")
    for flag in ("worktree", "output"):
        feedback.add_argument("--" + flag, type=Path, required=True)
    feedback.add_argument("--raw-root", type=Path, action="append", required=True)
    feedback.add_argument("--snapshot-manifest", type=Path,
                          help="Replay the exact frozen raw byte prefixes despite subsequent generator appends")
    feedback.add_argument("--exclude-corpus", type=Path, action="append", required=True)
    feedback.add_argument("--label-version", default="anvil-label-v3.5")
    feedback.add_argument("--salt", default="anvil-native-feedback-v1")
    feedback.add_argument("--max-images", type=int, default=2048)
    feedback.add_argument("--dev-permille", type=int, default=100)
    feedback.set_defaults(func=prepare_native_feedback)
    clean_feedback = subparsers.add_parser("clean-native-feedback", help="quarantine complete evaluation scene groups from frozen native candidates")
    clean_feedback.add_argument("--source", type=Path, required=True)
    clean_feedback.add_argument("--output", type=Path, required=True)
    clean_feedback.set_defaults(func=clean_native_feedback)
    recovery = subparsers.add_parser("recover-teacher-refresh", help="recover evaluation from a bound saved Teacher adapter without restarting training")
    for flag in ("worktree", "after-run", "output", "teachers"):
        recovery.add_argument("--" + flag, type=Path, required=True)
    recovery.add_argument("--after-pid", type=int, required=True)
    recovery.add_argument("--gpu", type=int, required=True)
    recovery.set_defaults(func=recover_teacher_refresh)

    context_pack = subparsers.add_parser("generate-native-context-pack", help="generate a frozen recursive student-context diagnostic pack on an allocated GPU")
    for flag in ("worktree", "history", "teachers", "model", "output"):
        context_pack.add_argument("--" + flag, type=Path, required=True)
    context_pack.set_defaults(func=generate_native_context_pack)
    native_inference = subparsers.add_parser("native-infer", help="infer native tasks with per-item finish reasons and token counts")
    for flag in ("worktree", "model", "adapter", "source", "output"):
        native_inference.add_argument("--" + flag, type=Path, required=True)
    native_inference.set_defaults(func=native_infer)
    context_eval = subparsers.add_parser("run-native-context-eval", help="queue shared oracle/student/corrupted-context evaluation after a bound live training job")
    for flag in ("worktree", "history", "teachers", "model", "output", "after-run", "context-pack"):
        context_eval.add_argument("--" + flag, type=Path, required=True)
    context_eval.add_argument("--after-pid", type=int, required=True)
    context_eval.add_argument("--context-builder-pid", type=int)
    context_mode = context_eval.add_mutually_exclusive_group()
    context_mode.add_argument("--prepare-contexts", action="store_true")
    context_mode.add_argument("--frozen-contexts", action="store_true",
                              help="reuse a complete hash-bound pack without a live generator")
    context_eval.add_argument("--role", choices=("G", "S", "A", "joint"), required=True)
    context_eval.add_argument("--arm", choices=("control", "feedback"), required=True)
    context_eval.add_argument("--gpu", type=int, required=True)
    context_eval.set_defaults(func=run_native_context_eval)
    refresh = subparsers.add_parser("run-teacher-refresh", help="continue one teacher with replay or independently reviewed feedback")
    for flag in ("worktree", "history", "teachers", "model", "output"):
        refresh.add_argument("--" + flag, type=Path, required=True)
    refresh.add_argument("--feedback", type=Path)
    refresh.add_argument("--review-report", type=Path,
                         help="Explicit independent review report; original and calibrated rounds remain separate")
    refresh.add_argument("--role", choices=("G", "S", "A", "joint"), required=True)
    refresh.add_argument("--arm", choices=("control", "feedback"), required=True)
    refresh.add_argument("--gpu", type=int, required=True)
    refresh.add_argument("--steps", type=int, default=512)
    refresh.add_argument("--seed", type=int, default=17)
    refresh.add_argument("--lr", type=float, default=2e-5)
    refresh.add_argument("--hours", type=int, default=5)
    refresh.add_argument("--preflight-only", action="store_true")
    refresh.add_argument("--project-sft-messages", action="store_true", help="Keep only exact messages/images in the SFT reader; preserve full audited rows separately")
    refresh.set_defaults(func=run_teacher_refresh)
    quarantine = subparsers.add_parser("filter-corpus", help="exclude named image-source roots into a fresh derivative")
    quarantine.add_argument("--input", type=Path, required=True)
    quarantine.add_argument("--output", type=Path, required=True)
    quarantine.add_argument("--expected-rows", type=int, required=True)
    quarantine.add_argument("--expected-sha256", required=True)
    quarantine.add_argument("--exclude-image-root", action="append", required=True)
    quarantine.set_defaults(func=filter_corpus)
    queue = subparsers.add_parser("queue-clean-sft", help="queue fresh SFT and full evaluations after a live evaluation job")
    for flag in ("worktree", "corpus", "output", "model", "retention-audit", "spatial-audit", "after-log"):
        queue.add_argument("--" + flag, type=Path, required=True)
    queue.add_argument("--after-pid", type=int, required=True)
    queue.add_argument("--runtime-root", type=Path,
                       help="Fresh directory for bulky checkpoints/merged weights; selected LoRA is preserved in --output")
    queue.add_argument("--tag", required=True)
    queue.add_argument("--seed", type=int, default=17)
    queue.add_argument("--lr", type=float, default=1e-4)
    queue.add_argument("--hours", type=int, default=30)
    queue.add_argument("--preflight-only", action="store_true")
    queue.set_defaults(func=queue_clean_sft)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
