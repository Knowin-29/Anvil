#!/usr/bin/env python3
"""Bind the official ShareRobot affordance test split to a native region contract.

No train/test relabeling, model annotation, invented contact points, or score-based
selection. This exports a declared, image-screened Anvil evaluation subset; it is
not a new score on the complete official 200-example benchmark. Source scene IDs
are absent, so image screening must not be called proven scene-level separation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import sys
import time
import urllib.request

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.arxiv_source_scale import Fingerprints, fingerprint, sha
from tools.build_anvil_pool import write_json
from trains.flywheel.native_training import canonical, strict_json, validate_target

REPOSITORY = "BAAI/ShareRobot-Bench"
REVISION = "68e9c15600f85f601b9cf492e4d994b50ed4cd4d"
CONTRACT = "anvil-affordance-region-v1"
VERSION = "anvil-sharerobot-official-test-v1"
QUESTION = re.compile(r'<image>\nYou are a robot using the joint control\. The task is "(.+)"\. '
                      r'Please predict a possible affordance area of the end effector\?')


def require(value, message):
    if not value:
        raise ValueError(message)


def safe_path(root, relative):
    p = PurePosixPath(relative)
    require(not p.is_absolute() and ".." not in p.parts and "\\" not in relative, "unsafe source path")
    path = Path(root).joinpath(*p.parts)
    require(path.resolve().is_relative_to(Path(root).resolve()), "source symlink escapes root")
    return path


def verify_file(root, entry):
    require(entry.get("type") == "file", "not a source file")
    path = safe_path(root, entry["path"])
    raw = path.read_bytes()
    require(len(raw) == entry["size"], "upstream file size mismatch")
    checksum = hashlib.sha256(raw).hexdigest()
    if entry.get("lfs"):
        require(checksum == entry["lfs"]["oid"], "upstream LFS SHA256 mismatch")
    else:
        blob = b"blob " + str(len(raw)).encode() + b"\0" + raw
        require(hashlib.sha1(blob).hexdigest() == entry["oid"], "upstream Git blob mismatch")
    return {"path": entry["path"], "bytes": len(raw), "sha256": checksum}


def fetch_tree():
    # A fixed public dataset revision, no token or arbitrary remote endpoint.
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/affordance?recursive=true&limit=1000"
    with urllib.request.urlopen(url, timeout=40) as response:
        require(not response.headers.get("Link"), "unexpected paginated upstream tree")
        tree = strict_json(response.read())
    files = [entry for entry in tree if entry["type"] == "file"]
    require(len(files) == 201 and any(e["path"] == "affordance/test.json" for e in files),
            "pinned official split inventory changed")
    return files


def native_row(row, source_root, annotations_sha256, fp):
    require(type(row.get("id")) is int and row["id"] >= 0, "invalid official ID")
    relative = row.get("image")
    require(isinstance(relative, str) and relative.startswith("images/"), "invalid image reference")
    path = safe_path(Path(source_root) / "affordance", relative)
    messages = row.get("conversations")
    require(isinstance(messages, list) and len(messages) == 2
            and [m.get("from") for m in messages] == ["human", "gpt"], "invalid official conversation")
    match = QUESTION.fullmatch(messages[0]["value"])
    require(match is not None, "unknown official question wrapper")
    instruction = match.group(1)
    require(bool(instruction.strip()), "empty instruction")
    pixels = strict_json(messages[1]["value"])
    require(isinstance(pixels, list) and len(pixels) == 4 and all(type(v) is int for v in pixels),
            "official region must be four integer pixel coordinates")
    width, height = fp["width"], fp["height"]
    x1, y1, x2, y2 = pixels
    require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, "official region outside decoded image")
    box = [round(v * 1000 / extent, 6) for v, extent in zip(pixels, (width, height, width, height))]
    target = {"affordance": instruction, "region_bbox_2d": box}
    validate_target("affordance", CONTRACT, target, "")
    uid = hashlib.sha256(canonical([VERSION, REVISION, row["id"], fp["decoded_rgb_sha256"], target]).encode()).hexdigest()
    prompt = ('<image>\nFor instruction ' + json.dumps(instruction) +
              ', localize the annotated instruction-conditioned affordance region. Return one JSON object '
              'with affordance (the instruction) and region_bbox_2d ([x1,y1,x2,y2], image-relative 0-1000). '
              'Predict a region, not an object box or a contact point.')
    return {"id": uid, "images": [str(path)], "messages": [
        {"role": "user", "content": prompt}, {"role": "assistant", "content": canonical(target)}],
        "anvil_meta": {"schema": VERSION, "source": "ShareRobot-Bench", "source_repository": REPOSITORY,
            "source_revision": REVISION, "source_annotation_id": str(row["id"]),
            "source_image_path": "affordance/" + relative, "official_annotations_sha256": annotations_sha256,
            "source_bbox_pixels": pixels, "source_image_size_wh": [width, height],
            "native_form": "affordance", "native_variant": CONTRACT, "split": "test",
            "training_approved": False, "benchmark_eligible": True,
            "group_id": "ShareRobot-Bench:image:" + fp["decoded_rgb_sha256"], "group_unit": "source_image",
            "image_sha256": fp["sha256"], "decoded_rgb_sha256": fp["decoded_rgb_sha256"],
            "coordinate_scale": 1000, "verification": "pinned_official_test_region_and_training_image_exclusion",
            "new_human_reviewed": False, "strong_model_verified": False, "legacy_full_contact_schema": False,
            "upstream_declared_license": "apache-2.0", "public_release_eligible": False}}


def load_training_protection(index, manifest_path):
    manifest_sha = sha(manifest_path); manifest = strict_json(Path(manifest_path).read_bytes())
    require(sha(index) == manifest["selection_sqlite_sha256"] and manifest.get("selection_complete") is True,
            "pool index is not bound to a completed selection")
    protected = Fingerprints(); count = 0
    with sqlite3.connect(f"{Path(index).resolve().as_uri()}?mode=ro", uri=True) as db:
        # Scan the compact images table, not a random-I/O millions-row join.
        for raw, decoded, ph, dh in db.execute("SELECT file_sha256,decoded_sha256,phash64,dhash64 FROM images WHERE split='train'"):
            protected.add({"sha256": raw, "decoded_rgb_sha256": decoded, "phash64": ph, "dhash64": dh}); count += 1
    eligibility = manifest["eligibility"]
    require(sha(eligibility["inventory"]) == eligibility["inventory_sha256"], "historical eligibility inventory drift")
    historical = strict_json(Path(eligibility["inventory"]).read_bytes())["inputs"]["ANVIL_SOURCE_FINGERPRINTS"]
    require(sha(historical["path"]) == historical["sha256"], "historical training fingerprints changed")
    historical_count = 0
    with Path(historical["path"]).open() as stream:
        for line in stream:
            if line.strip(): protected.add(strict_json(line)); historical_count += 1
    return protected, {"pool_manifest": str(manifest_path), "pool_manifest_sha256": manifest_sha,
        "pool_index": str(index), "pool_index_sha256": manifest["selection_sqlite_sha256"],
        "pool_training_images": count, "historical_training": historical,
        "historical_training_images": historical_count,
        "scope": "Bound pool plus bound historical training fingerprints; no claim about unrecorded exposures or unavailable scene IDs."}


def convert(args):
    start = time.monotonic(); code_sha = sha(Path(__file__))
    require(not args.output.exists(), "Use a new immutable output directory")
    upstream = fetch_tree()
    with ThreadPoolExecutor(args.workers) as workers:
        verified = list(workers.map(lambda e: verify_file(args.source_root, e), upstream))
    file_hashes = {r["path"]: r["sha256"] for r in verified}
    source = args.source_root / "affordance/test.json"; source_sha = sha(source)
    entries = strict_json(source.read_bytes())
    require(len(entries) == 200 and len({r["id"] for r in entries}) == 200, "official test row identity changed")
    require({"affordance/" + r["image"] for r in entries} == set(file_hashes) - {"affordance/test.json"},
            "official image references do not match pinned files")
    protected, binding = load_training_protection(args.pool_index, args.pool_manifest)
    def inspect(row):
        p = safe_path(args.source_root / "affordance", row["image"])
        with Image.open(p) as image:
            require(image.getexif().get(274, 1) == 1, "orientation requires an explicit coordinate transform")
        fp = fingerprint(p)
        require(fp["sha256"] == file_hashes["affordance/" + row["image"]], "image changed during conversion")
        return row, fp
    skipped = Counter(); audit = []; output = []; seen = set()
    with ThreadPoolExecutor(args.workers) as workers:
        for row, fp in workers.map(inspect, entries):
            reason = None
            try:
                mapped = native_row(row, args.source_root, source_sha, fp)
                key = (fp["decoded_rgb_sha256"], mapped["messages"][-1]["content"])
                if protected.matches(fp): reason = "training_image_match_exact_or_near"
                elif key in seen: reason = "duplicate_image_instruction_region"
            except ValueError as error:
                reason = str(error)
            audit.append({"source_id": row["id"], "image": row["image"], **fp,
                          "accepted": reason is None, "reason": reason})
            if reason: skipped[reason] += 1
            else: seen.add(key); output.append(mapped)
    # Input hashes are checked again before publishing any completion manifest.
    require(sha(source) == source_sha and sha(Path(__file__)) == code_sha
            and sha(args.pool_manifest) == binding["pool_manifest_sha256"]
            and sha(args.pool_index) == binding["pool_index_sha256"]
            and sha(binding["historical_training"]["path"]) == binding["historical_training"]["sha256"],
            "input/code drift during conversion")
    require(output, "No independent official test records remain")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in (("evaluation.jsonl", sorted(output, key=lambda r: r["id"])), ("source_audit.jsonl", audit)):
        with (args.output / name).open("x") as stream:
            for row in records: stream.write(canonical(row) + "\n")
    result = {"schema": VERSION, "state": "official_test_subset_exported", "source_repository": REPOSITORY,
        "source_revision": REVISION, "upstream_declared_license": "apache-2.0", "source_root": str(args.source_root),
        "source_files": verified, "source_records": 200, "records": len(output), "skipped": dict(skipped),
        "unique_images": len({r["anvil_meta"]["decoded_rgb_sha256"] for r in output}),
        "native_form": "affordance", "native_contract": CONTRACT, "source_split": "test",
        "protected_training": binding, "image_screening": "byte/RGB equality or both pHash64 and dHash64 Hamming<=4",
        "scene_level_isolation_proven": False, "training_approved": False, "model_calls": 0, "new_human_reviews": 0,
        "benchmark_scored": False, "public_release_eligible": False, "release_complete": False,
        "scope": "Derived Anvil evaluation subset with normalized boxes, not a claim of full official-benchmark evaluation or contact-point supervision.",
        "files": {name: sha(args.output / name) for name in ("evaluation.jsonl", "source_audit.jsonl")},
        "converter_sha256": code_sha, "seconds": time.monotonic() - start}
    write_json(args.output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "pool-index", "pool-manifest", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); require(1 <= args.workers <= 32, "workers must be 1..32")
    result = convert(args)
    print(json.dumps({k: result[k] for k in ("state", "records", "unique_images", "skipped", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
