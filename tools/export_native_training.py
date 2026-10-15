#!/usr/bin/env python3
"""Export ALL selected training rows as source-bound, per-form SFT streams.

This is a training adapter, not a new split or a semantic annotation operation.
No sampling, target repair, image copying, or silent malformed-row exclusion.
QA uses a separate reference-only protocol. Exports retain local image paths and
therefore are NOT the anonymized/portable public data release.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
import hashlib
import json
import multiprocessing
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.arxiv_source_scale import fingerprint, sha
from tools.build_anvil_pool import write_json
from trains.flywheel import native_training as native

VERSION = "anvil-native-training-export-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


@lru_cache(maxsize=50000)
def checked_image(path, canonical_path, expected_raw, expected_decoded):
    if path == canonical_path:
        actual = sha(path)
        require(actual == expected_raw, "Selected training image bytes changed")
        return actual
    fp = fingerprint(path)
    require(fp["decoded_rgb_sha256"] == expected_decoded, "Alternate source image differs from selected image")
    return fp["sha256"]


def convert_selected(raw, entry, selection_sha):
    """Verify an unchanged source row before admitting its reference to training."""
    row = native.strict_json(raw); meta = row.get("anvil_meta", {})
    form = "QA" if meta.get("record_type") == "qa_replay" else meta.get("native_form")
    variant = meta.get("native_contract", meta.get("native_variant", meta.get("schema")))
    source, group = meta.get("source"), meta.get("group_id")
    require(entry["split"] == "train" and entry["uid"] == row.get("id")
            and entry["form"] == form and entry["variant"] == variant and entry["source"] == source,
            "Source identity/form/variant disagrees with selected training row")
    # Match the index builder's three explicit namespaces, retaining the
    # original source group separately in the training record.
    if source in native.DENSE:
        expected_group = "decoded_image:" + str(group)
    elif form == "QA" or str(group).startswith(str(source) + ":"):
        expected_group = group
    else:
        expected_group = str(source) + ":" + str(group)
    require(entry["scene_group"] == expected_group, "Source group disagrees with selected group")
    images = row.get("images")
    require(isinstance(images, list) and len(images) == 1 and isinstance(images[0], str), "One source image required")
    actual_raw = checked_image(images[0], entry["image_path"], entry["file_sha256"], entry["decoded_sha256"])
    binding = {"selection_sha256": selection_sha, "source_sha256": entry["source_sha256"],
        "source_line_sha256": hashlib.sha256(raw).hexdigest(), "source_uid": entry["uid"],
        "source_offset": entry["byte_offset"], "source_size": entry["byte_length"],
        "image_path": images[0], "image_sha256": actual_raw, "decoded_rgb_sha256": entry["decoded_sha256"],
        "split": "train", "source": source, "group_id": group, "selected_group_id": entry["scene_group"],
        "native_form": form, "native_variant": variant}
    bound = native.bind_training_record(raw, binding)
    return form, native.canonical(bound) + "\n"


def convert_batch(job):
    selection_sha, entries = job
    result = []
    for raw, entry in entries:
        try:
            result.append(convert_selected(raw, entry, selection_sha))
        except Exception as error:
            raise ValueError(f"Selected training row {entry['uid']} ({entry['source']}/{entry['form']}, "
                             f"offset {entry['byte_offset']}): {error}") from error
    return result


def source_batches(db, selection_sha, batch_size=512, *, split="train"):
    require(split in {"train", "benchmark"}, "Unknown selected split")
    query = """SELECT s.uid,s.split,s.source,s.form,s.variant,s.scene_group,s.byte_offset,s.byte_length,
        s.source_path,s.source_sha256,s.image_path,s.decoded_sha256,i.file_sha256
        FROM selection s JOIN images i USING(image_id) WHERE s.split=?
        ORDER BY s.file_id,s.byte_offset"""
    cursor = db.execute(query, (split,)); names = [d[0] for d in cursor.description]
    path = None; stream = None; previous_end = 0; batch = []
    try:
        for values in cursor:
            entry = dict(zip(names, values))
            if entry["source_path"] != path:
                if stream is not None: stream.close()
                path = entry["source_path"]; stream = Path(path).open("rb", buffering=8 << 20); previous_end = 0
            offset, length = entry["byte_offset"], entry["byte_length"]
            require(offset >= previous_end and length > 0, "Overlapping or unordered selected source offsets")
            stream.seek(offset); raw = stream.read(length); previous_end = offset + length
            require(len(raw) == length and raw.endswith(b"\n"), "Truncated selected source row")
            batch.append((raw, entry))
            if len(batch) == batch_size:
                yield selection_sha, batch; batch = []
        if batch: yield selection_sha, batch
    finally:
        if stream is not None: stream.close()


def bounded_results(jobs, workers):
    if workers == 1:
        yield from map(convert_batch, jobs)
        return
    # Bound both source prefetch and result memory, preserving deterministic order.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = deque()
        for _ in range(workers * 2):
            job = next(jobs, None)
            if job is None: break
            pending.append(pool.submit(convert_batch, job))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None: pending.append(pool.submit(convert_batch, job))


def export(args):
    start = time.monotonic(); code_sha = sha(Path(__file__)); admission_sha = sha(Path(native.__file__))
    require(not args.output.exists(), "Use a new immutable training export directory")
    manifest_sha = sha(args.manifest); manifest = native.strict_json(args.manifest.read_bytes())
    require(manifest.get("selection_complete") is True, "Incomplete pool index")
    selection_sha = sha(args.index)
    require(selection_sha == manifest["selection_sqlite_sha256"], "Pool index SHA256 mismatch")
    for item in manifest["inputs"]:
        require(sha(item["path"]) == item["expected_sha256"] and sha(item["manifest"]) == item["manifest_sha256"],
                "Bound source/manifest changed before training export")
    db = sqlite3.connect(":memory:"); db.execute("PRAGMA temp_store=MEMORY")
    src = sqlite3.connect(f"{args.index.resolve().as_uri()}?mode=ro", uri=True)
    src.backup(db, pages=4096); src.close()
    counts = dict(db.execute("SELECT form,count(*) FROM selection WHERE split='train' GROUP BY form"))
    require(set(counts) <= set(native.FAMILIES) | {"QA"}, "Unsupported training form")
    require(sum(counts.values()) == manifest["observed"]["rows_by_split"]["train"], "Training count differs from manifest")
    images = db.execute("SELECT count(DISTINCT image_id) FROM selection WHERE split='train'").fetchone()[0]
    args.output.mkdir(parents=True, exist_ok=False)
    def progress(stage, **values):
        message = {"stage": stage, "seconds": time.monotonic() - start, **values}
        write_json(args.output / "progress.json", message); print(json.dumps(message), flush=True)
    progress("source_bound", records=sum(counts.values()), selected_training_images=images)
    written = Counter()
    try:
        with ExitStack() as stack:
            handles = {form: stack.enter_context((args.output / f"train_{form}.jsonl.pending").open("x")) for form in counts}
            for batch in bounded_results(iter(source_batches(db, selection_sha)), args.workers):
                previous = sum(written.values())
                for form, line in batch:
                    handles[form].write(line); written[form] += 1
                if sum(written.values()) // 50000 != previous // 50000:
                    progress("exporting", records=sum(written.values()), by_form=dict(written))
        require(dict(written) == counts, "Export omitted or duplicated selected training records")
        for item in manifest["inputs"]:
            require(sha(item["path"]) == item["expected_sha256"] and sha(item["manifest"]) == item["manifest_sha256"],
                    "Source changed during training export")
        require(sha(args.index) == selection_sha and sha(args.manifest) == manifest_sha
                and sha(Path(__file__)) == code_sha and sha(Path(native.__file__)) == admission_sha,
                "Selection or exporter code changed during export")
        files = {}
        for form, count in sorted(written.items()):
            name = f"train_{form}.jsonl"; (args.output / (name + ".pending")).rename(args.output / name)
            files[name] = {"records": count, "sha256": sha(args.output / name), "form": form,
                           "family": native.FAMILIES.get(form, "QA")}
        result = {"schema": VERSION, "state": "all_selected_training_rows_exported", "records": sum(written.values()),
            "unique_images": images, "by_form": dict(written), "files": files, "dropped_records": 0,
            "targets_repaired": 0, "qa_reference_only": True, "selection_sha256": selection_sha,
            "pool_manifest_sha256": manifest_sha, "pool_index": str(args.index), "pool_manifest": str(args.manifest),
            "source_inputs": manifest["inputs"], "exporter_sha256": code_sha, "admission_sha256": admission_sha,
            "image_validation": "Encoded SHA checked on each worker's first use; alternate paths require identical decoded RGB.",
            "tokenization_complete": False, "model_training_complete": False, "public_release_eligible": False,
            "scope": "Complete local training streams with source targets and local image paths; not an anonymous portable dataset release.",
            "workers": args.workers, "seconds": time.monotonic() - start}
        write_json(args.output / "manifest.json", result); progress("complete", records=result["records"])
        return result
    except BaseException as error:
        write_json(args.output / "failure.json", {"state": "failed_no_complete_training_export", "type": type(error).__name__,
                                                   "reason": str(error), "written_records": sum(written.values())})
        raise
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); require(1 <= args.workers <= 32, "workers must be 1..32")
    export(args)


if __name__ == "__main__":
    main()
