#!/usr/bin/env python3
"""Build a score-blind, replayable local Anvil pool index without copying images.

Native files retain their frozen scene split and are mandatory. Dense-source
benchmark rows must belong to a supplied, hash-bound eligibility inventory.
Exact count targets never override these constraints. Unknown data rights stay
local-only, independently of the converter's source-code license.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.arxiv_source_scale import Fingerprints, FORMS, fingerprint, sha

VERSION = "anvil-local-source-pool-index-v1"
DENSE_FORMS = {"bbox", "detection", "mask", "point", "direction", "metric_comparison", "relation_graph"}


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def rank(seed, value):
    return int.from_bytes(hashlib.sha256((seed + "|" + value).encode()).digest()[:8], "big") & ((1 << 63) - 1)


def manifest_digest(manifest, name):
    value = manifest.get("files", {}).get(name)
    if isinstance(value, dict):
        value = value.get("sha256")
    if value is None:
        value = manifest.get({"records.jsonl": "records_sha256", "train.jsonl": "train_sha256",
                              "evaluation.jsonl": "evaluation_sha256"}.get(name, ""))
    require(isinstance(value, str) and len(value) == 64, "No explicit file hash in completion manifest: " + name)
    return value


def file_rows(path):
    with Path(path).open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            require(bool(raw.strip()), "Blank input record lines are unsupported")
            yield offset, len(raw), raw, json.loads(raw)


def apportion(total, weights, caps=None, floors=None, seed="quota"):
    """Deterministic weighted Hamilton allocation with hard lower/upper bounds."""
    keys = sorted(weights)
    caps = {k: int((caps or {}).get(k, total)) for k in keys}
    result = {k: int((floors or {}).get(k, 0)) for k in keys}
    require(total >= 0 and all(0 <= result[k] <= caps[k] for k in keys), "Invalid quota bounds")
    remaining = total - sum(result.values())
    require(0 <= remaining <= sum(caps[k] - result[k] for k in keys), "Insufficient stratum capacity")
    while remaining:
        active = [k for k in keys if result[k] < caps[k]]
        denominator = sum(weights[k] for k in active)
        require(denominator > 0, "No positively weighted capacity")
        shares = {k: remaining * weights[k] / denominator for k in active}
        additions = {k: min(caps[k] - result[k], math.floor(shares[k])) for k in active}
        for k, n in additions.items():
            result[k] += n
        remaining -= sum(additions.values())
        if not remaining:
            break
        order = sorted((k for k in active if result[k] < caps[k]),
                       key=lambda k: (-(shares[k] - math.floor(shares[k])), rank(seed, encode(k)), k))
        require(order, "Quota allocation stalled")
        for k in order[:remaining]:
            result[k] += 1
            remaining -= 1
    return result


def load_native_row(row, requested_split):
    meta = row.get("anvil_meta", {})
    source = meta.get("source", row.get("source"))
    form = meta.get("native_form", row.get("native_form_candidate", row.get("native_form")))
    variant = meta.get("native_contract", meta.get("native_variant", row.get("schema_version", row.get("contract"))))
    group = meta.get("group_id", row.get("scene_group"))
    original = meta.get("split", row.get("source_split", row.get("partition")))
    if original == "train":
        split = "train"
    elif original in {"evaluation", "validation", "valid_seen", "valid_unseen", "test", "heldout",
                      "evaluation_source_only"}:
        split = "benchmark"
    else:
        raise ValueError("Native input has no supported frozen split: " + str(original))
    require(split == requested_split, "Native input split conflicts with requested split")
    require(form in FORMS and source and variant and group, "Incomplete native contract/source/group")
    if not group.startswith(str(source) + ":"):
        group = str(source) + ":" + group
    images = row.get("images", [row.get("input", {}).get("image")])
    require(isinstance(images, list) and len(images) == 1 and isinstance(images[0], str),
            "Current index requires exactly one source image per native atomic record")
    image_hash = (meta.get("image_sha256") or meta.get("input_image_sha256")
                  or meta.get("source_provenance", {}).get("image_sha256")
                  or row.get("input", {}).get("image_sha256") or row.get("image_sha256"))
    require(isinstance(image_hash, str) and len(image_hash) == 64, "Native image lacks pinned byte hash")
    if split == "train":
        require(meta.get("training_approved", row.get("training_approved")) is True,
                "Native training input lacks source training approval")
    return {"source": str(source), "form": form, "variant": str(variant), "scene": group,
            "split": split, "image": images[0], "image_hash": image_hash, "original_split": original}


def choose_images(images, desired_train, desired_eval, target_images, seed):
    """Mandatory-native first; stratified benchmark then dense train capacity."""
    native_counts = Counter()
    frozen = {"train": set(), "benchmark": set()}
    for key, item in images.items():
        native_counts.update(item["mandatory"])
        if item.get("fixed_split"):
            frozen[item["fixed_split"]].add(key)
    need = {"train": desired_train - native_counts["train"],
            "benchmark": desired_eval - native_counts["benchmark"]}
    require(min(need.values()) >= 0, "Mandatory native rows exceed record targets")
    require(len(frozen["train"] | frozen["benchmark"]) <= target_images,
            "Mandatory native images exceed image budget")
    native_train_fp = Fingerprints()
    benchmark_fp = Fingerprints()
    for key in frozen["train"]:
        native_train_fp.add(images[key]["fp"])
    for key in frozen["benchmark"]:
        require(not native_train_fp.matches(images[key]["fp"]), "Frozen native train/benchmark image overlap")
        benchmark_fp.add(images[key]["fp"])

    eval_available = {}
    for key, item in images.items():
        if not item["dense"]:
            continue
        if item.get("fixed_split") == "benchmark":
            require(item["eligible"], "Frozen native benchmark overlaps a dense noneligible image")
        if item["eligible"] and item.get("fixed_split") != "train" and not native_train_fp.matches(item["fp"]):
            eval_available[key] = item
    weights = Counter()
    for item in eval_available.values():
        weights.update(item["dense"])
    require(sum(weights.values()) >= need["benchmark"], "Insufficient eligible benchmark rows")
    requested = apportion(need["benchmark"], weights, dict(weights), seed=seed + "|benchmark-quota")
    source_weights = Counter()
    for (source, form), count in weights.items():
        source_weights[source] += count
    source_requested = apportion(need["benchmark"], source_weights, dict(source_weights),
                                 seed=seed + "|benchmark-source-quota")
    bench = set(frozen["benchmark"])
    have = Counter()
    for key in bench:
        if key in eval_available:
            have.update(eval_available[key]["dense"])
    # Reserve whole-image capacity by source. One-per-image forms (e.g. detection)
    # cannot dictate a large, mostly discarded benchmark image cohort: actual
    # record-form quotas are capped by the selected cohort below, after this step.
    # The image budget and source proportions are chosen before seeing any score.
    def source_rows(item, source):
        return sum(n for (src, form), n in item["dense"].items() if src == source)
    for source in sorted(source_requested, key=lambda k: (rank(seed, k), k)):
        candidates = sorted((key for key, item in eval_available.items()
                             if key not in bench and source_rows(item, source)),
                            key=lambda key: (-source_rows(eval_available[key], source),
                                             -sum(eval_available[key]["dense"].values()),
                                             rank(seed + "|eval-image", key), key))
        present = sum(n for (src, form), n in have.items() if src == source)
        for key in candidates:
            if present >= source_requested[source]:
                break
            bench.add(key)
            have.update(eval_available[key]["dense"])
            present += source_rows(eval_available[key], source)
        require(present >= source_requested[source], "Benchmark source cannot meet requested count")
    # Retain at least one group for every available source/form; do not invent
    # records to reproduce the unconstrained full-inventory form proportions.
    for stratum in sorted(weights):
        if have[stratum]:
            continue
        candidates = [key for key, item in eval_available.items() if item["dense"].get(stratum)]
        key = min(candidates, key=lambda key: (-eval_available[key]["dense"][stratum],
                                               rank(seed + "|eval-image", key), key))
        bench.add(key)
        have.update(eval_available[key]["dense"])
    for key in bench:
        benchmark_fp.add(images[key]["fp"])
    train = set(frozen["train"])
    require(not train & bench, "Mandatory image split conflict")
    for key in train:
        require(not benchmark_fp.matches(images[key]["fp"]), "Frozen native training overlaps selected benchmark")
    slots = target_images - len(train | bench)
    require(slots >= 0, "Stratified benchmark plus native images exceed image budget")
    train_available = []
    excluded_near = 0
    for key, item in images.items():
        if not item["dense"] or key in train or key in bench or item.get("fixed_split") == "benchmark":
            continue
        if benchmark_fp.matches(item["fp"]):
            excluded_near += 1
            continue
        train_available.append(key)
    train_available.sort(key=lambda key: (-sum(images[key]["dense"].values()),
                                         rank(seed + "|train-image", key), key))
    require(len(train_available) >= slots, "Insufficient eligible image groups to reach exact image budget")
    train.update(train_available[:slots])
    capacity = {split: sum(sum(images[key]["dense"].values()) for key in group)
                for split, group in (("train", train), ("benchmark", bench))}
    deficits = {s: max(0, need[s] - capacity[s]) for s in need}
    summary = {"mandatory_rows": dict(native_counts), "dense_rows_needed": need,
               "dense_rows_available_in_selected_images": capacity, "record_shortfall": deficits,
               "selected_images": {"train": len(train), "benchmark": len(bench)},
               "dense_train_images_excluded_for_near_benchmark_overlap": excluded_near,
               "benchmark_source_capacity_targets": dict(source_requested),
               "unconstrained_benchmark_form_reference_not_hard_targets": {encode(k): v for k, v in requested.items()}}
    return train, bench, summary


def build(args):
    start = time.monotonic()
    output = args.output.resolve()
    require(not output.exists(), "Output must be a new immutable directory")
    output.mkdir(parents=True)
    progress_path = output / "progress.json"
    def progress(stage, **values):
        result = {"stage": stage, "elapsed_seconds": round(time.monotonic() - start, 3), **values}
        write_json(progress_path, result)
        print(encode(result), flush=True)
    code_sha = sha(Path(__file__))
    eligibility_digest = sha(args.eligible_index)
    inventory_digest = sha(args.eligible_inventory)
    inventory = json.loads(args.eligible_inventory.read_text())
    require(inventory.get("state") == "inventory_complete_not_frozen_benchmark", "Unsupported eligibility inventory")
    eligibility = sqlite3.connect(args.eligible_index.resolve().as_uri() + "?mode=ro", uri=True)
    eligible_groups = {r[0] for r in eligibility.execute("SELECT group_id FROM images")}
    inputs = []; images = {}; seen = set(); scene_splits = {}; native_rows = []
    native_counts = Counter(); dense_counts = Counter()
    def image_entry(fp, path, declared=None):
        key = fp["decoded_rgb_sha256"]
        if declared is not None:
            require(key == declared, "Decoded image identity disagreement")
        if key not in images:
            images[key] = {"fp": fp, "path": str(path), "dense": Counter(), "mandatory": Counter(),
                           "fixed_split": None, "eligible": key in eligible_groups}
        else:
            old = images[key]["fp"]
            require(all(old[k] == fp[k] for k in ("width", "height", "phash64", "dhash64")),
                    "Same decoded image with inconsistent geometry/perceptual identity")
        return key
    for source in ("coco_train2017", "pixmo_points"):
        folder = args.dense_root / source
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        require(sha(manifest_path) == inventory["inputs"][source]["manifest_sha256"], "Dense source manifest mismatch")
        image_file = folder / "images.jsonl"
        require(sha(image_file) == manifest_digest(manifest, "images.jsonl"), "Dense image inventory changed")
        for _, _, _, row in file_rows(image_file):
            image_entry(row, row["path"], row["group_id"])
        for prior in ("train", "evaluation_candidate"):
            path = folder / (prior + ".jsonl")
            expected = manifest_digest(manifest, path.name)
            require(expected == inventory["inputs"][source][prior]["sha256"], "Eligibility/source snapshot mismatch")
            inputs.append({"path": str(path.resolve()), "expected_sha256": expected, "source": source,
                           "kind": "dense", "prior_split": prior, "manifest": str(manifest_path),
                           "manifest_sha256": sha(manifest_path)})
    for file_id, item in enumerate(inputs, 1):
        h = hashlib.sha256(); n = 0
        for offset, length, raw, row in file_rows(item["path"]):
            h.update(raw); meta = row["anvil_meta"]; uid = row["id"]
            require(uid not in seen, "Duplicate record ID across source inputs")
            seen.add(uid)
            require(meta["source"] == item["source"] and meta["split"] == item["prior_split"],
                    "Dense source/split identity mismatch")
            require(meta["verification"] == "source_annotation_and_deterministic_geometry", "Unsupported dense provenance")
            require(meta["native_form"] in DENSE_FORMS, "Unexpected dense native form")
            key = meta["group_id"]
            require(key in images, "Dense record missing image binding")
            stratum = (item["source"], meta["native_form"])
            images[key]["dense"][stratum] += 1; dense_counts[stratum] += 1; n += 1
        require(h.hexdigest() == item["expected_sha256"], "Dense source records hash mismatch")
        item.update(file_id=file_id, records=n, bytes=Path(item["path"]).stat().st_size)
        progress("dense_counted", file_id=file_id, records=n, total_records=len(seen))
    del seen

    @lru_cache(maxsize=None)
    def native_fingerprint(path):
        return fingerprint(path)
    for source_name, requested_split, path_text, manifest_text in args.native_source:
        path, manifest_path = Path(path_text), Path(manifest_text)
        require(path.is_file() and manifest_path.is_file(), "Required native source/manifest is not complete")
        manifest = json.loads(manifest_path.read_text())
        expected = manifest_digest(manifest, path.name)
        file_id = len(inputs) + 1; h = hashlib.sha256(); n = 0
        for offset, length, raw, row in file_rows(path):
            h.update(raw); value = load_native_row(row, requested_split)
            fp = native_fingerprint(value["image"])
            require(fp["sha256"] == value["image_hash"], "Native image bytes changed")
            key = image_entry(fp, value["image"])
            split, scene = value["split"], value["scene"]
            require(scene not in scene_splits or scene_splits[scene] == split, "Native physical scene spans both splits")
            scene_splits[scene] = split
            require(images[key]["fixed_split"] in (None, split), "Native decoded image spans both splits")
            images[key]["fixed_split"] = split
            images[key]["mandatory"][split] += 1
            native_counts[(split, value["source"], value["form"], value["variant"])] += 1
            native_rows.append((row["id"], file_id, offset, length, value["source"], value["form"],
                                value["variant"], key, scene, value["original_split"], split))
            n += 1
        require(h.hexdigest() == expected, "Native source records hash mismatch")
        inputs.append({"path": str(path.resolve()), "expected_sha256": expected, "source": source_name,
                       "kind": "native", "prior_split": requested_split, "manifest": str(manifest_path.resolve()),
                       "manifest_sha256": sha(manifest_path), "file_id": file_id, "records": n,
                       "bytes": path.stat().st_size})
        progress("native_bound", source=source_name, split=requested_split, records=n)
    require(len({r[0] for r in native_rows}) == len(native_rows), "Duplicate mandatory native IDs")
    train, bench, selection = choose_images(images, args.train_records, args.benchmark_records,
                                            args.images, args.seed)
    progress("images_selected", **selection)
    require(not any(selection["record_shortfall"].values()), "Selected source/stratum/image constraints leave a record shortfall")
    image_ids = {key: i for i, key in enumerate(sorted(train | bench), 1)}
    db_path = output / "selection.sqlite"
    db = sqlite3.connect(db_path.as_uri() + "?mode=rwc", uri=True)
    db.executescript("""PRAGMA journal_mode=DELETE; PRAGMA synchronous=NORMAL; PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-524288;
        CREATE TABLE inputs(file_id INTEGER PRIMARY KEY,path TEXT NOT NULL,sha256 TEXT NOT NULL,
                            source TEXT NOT NULL,kind TEXT NOT NULL,manifest TEXT,manifest_sha256 TEXT);
        CREATE TABLE images(image_id INTEGER PRIMARY KEY,decoded_sha256 TEXT UNIQUE NOT NULL,
                            file_sha256 TEXT NOT NULL,path TEXT NOT NULL,split TEXT NOT NULL,
                            dense_benchmark_eligible INTEGER NOT NULL,phash64 TEXT NOT NULL,dhash64 TEXT NOT NULL);
        CREATE TABLE records(rid INTEGER PRIMARY KEY,uid TEXT UNIQUE NOT NULL,file_id INTEGER NOT NULL,
                             byte_offset INTEGER NOT NULL,byte_length INTEGER NOT NULL,
                             source TEXT NOT NULL,form TEXT NOT NULL,variant TEXT NOT NULL,
                             image_id INTEGER NOT NULL,scene_group TEXT NOT NULL,
                             original_split TEXT NOT NULL,mandatory INTEGER NOT NULL,selection_rank INTEGER NOT NULL);
        CREATE TABLE selected(rid INTEGER PRIMARY KEY,reason TEXT NOT NULL);
        CREATE TABLE quotas(split TEXT NOT NULL,source TEXT NOT NULL,form TEXT NOT NULL,records INTEGER NOT NULL,
                            PRIMARY KEY(split,source,form));
        CREATE VIEW selection AS SELECT r.*,i.split,i.decoded_sha256,f.path AS source_path,
          f.sha256 AS source_sha256,i.path AS image_path,s.reason
          FROM selected s JOIN records r USING(rid) JOIN images i USING(image_id) JOIN inputs f USING(file_id);""")
    db.executemany("INSERT INTO inputs VALUES(?,?,?,?,?,?,?)",
        [(v["file_id"], v["path"], v["expected_sha256"], v["source"], v["kind"], v["manifest"], v["manifest_sha256"]) for v in inputs])
    db.executemany("INSERT INTO images VALUES(?,?,?,?,?,?,?,?)",
        [(image_ids[k], k, images[k]["fp"]["sha256"], images[k]["path"], "train" if k in train else "benchmark",
          int(images[k]["eligible"]), images[k]["fp"]["phash64"], images[k]["fp"]["dhash64"]) for k in sorted(image_ids)])
    insertion = "INSERT INTO records(uid,file_id,byte_offset,byte_length,source,form,variant,image_id,scene_group,original_split,mandatory,selection_rank) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
    db.executemany(insertion, [(uid, fi, off, size, source, form, variant, image_ids[key], scene, prior,
                              1, rank(args.seed + "|record", uid))
                             for uid, fi, off, size, source, form, variant, key, scene, prior, split in native_rows])
    indexed = len(native_rows)
    for item in inputs:
        if item["kind"] != "dense": continue
        batch = []; h = hashlib.sha256()
        for offset, length, raw, row in file_rows(item["path"]):
            h.update(raw); meta = row["anvil_meta"]; key = meta["group_id"]
            if key not in image_ids: continue
            require(key not in bench or images[key]["eligible"], "Dense benchmark image is not eligible")
            uid = row["id"]
            batch.append((uid, item["file_id"], offset, length, meta["source"], meta["native_form"],
                          meta["schema"], image_ids[key], "decoded_image:" + key, meta["split"],
                          0, rank(args.seed + "|record", uid)))
            if len(batch) >= 10000:
                db.executemany(insertion, batch); indexed += len(batch); batch.clear()
                if indexed % 100000 < 10000:
                    db.commit(); progress("indexing_rows", indexed=indexed)
        if batch: db.executemany(insertion, batch); indexed += len(batch)
        require(h.hexdigest() == item["expected_sha256"], "Input changed between counting and indexing")
        db.commit()
    db.execute("CREATE INDEX records_image_rank ON records(image_id,selection_rank,uid)")
    db.execute("CREATE INDEX records_stratum_rank ON records(source,form,selection_rank,uid)")
    db.execute("CREATE INDEX records_mandatory_images ON records(image_id) WHERE mandatory=1")
    db.execute("INSERT INTO selected SELECT rid,'mandatory_native' FROM records WHERE mandatory=1")
    db.execute("""INSERT INTO selected
        SELECT rid,'image_coverage_anchor' FROM
        (SELECT r.rid,row_number() OVER(PARTITION BY image_id ORDER BY selection_rank,uid) rn
         FROM records r WHERE NOT EXISTS
           (SELECT 1 FROM records n WHERE n.image_id=r.image_id AND n.mandatory=1))
        WHERE rn=1""")
    for split, target in (("train", args.train_records), ("benchmark", args.benchmark_records)):
        mandatory = db.execute("SELECT count(*) FROM records r JOIN images i USING(image_id) WHERE i.split=? AND mandatory=1", (split,)).fetchone()[0]
        capacities = dict(((s, f), n) for s, f, n in db.execute(
            "SELECT source,form,count(*) FROM records r JOIN images i USING(image_id) WHERE i.split=? AND mandatory=0 GROUP BY source,form", (split,)))
        floors = dict(((s, f), n) for s, f, n in db.execute(
            "SELECT source,form,count(*) FROM records r JOIN images i USING(image_id) JOIN selected v USING(rid) WHERE i.split=? AND mandatory=0 GROUP BY source,form", (split,)))
        weights = {k: dense_counts[k] for k in capacities}
        quotas = apportion(target - mandatory, weights, capacities, floors, args.seed + "|" + split)
        db.executemany("INSERT INTO quotas VALUES(?,?,?,?)", [(split, s, f, n) for (s, f), n in quotas.items()])
        for (source, form), quota in sorted(quotas.items()):
            remaining = quota - floors.get((source, form), 0)
            db.execute("""INSERT INTO selected
                SELECT r.rid,'source_form_rank' FROM records r JOIN images i USING(image_id)
                WHERE i.split=? AND r.source=? AND r.form=? AND mandatory=0
                  AND NOT EXISTS(SELECT 1 FROM selected s WHERE s.rid=r.rid)
                ORDER BY selection_rank,uid LIMIT ?""", (split, source, form, remaining))
        db.commit(); progress("records_selected", split=split, target=target)
    # Eligibility is independently rechecked by actual selected record IDs, not just images.
    db.execute("ATTACH DATABASE ? AS eligible", (args.eligible_index.resolve().as_uri() + "?mode=ro",))
    bad = db.execute("""SELECT count(*) FROM selection s WHERE s.split='benchmark' AND s.mandatory=0
        AND NOT EXISTS(SELECT 1 FROM eligible.records e WHERE e.id=s.uid AND e.group_id=s.decoded_sha256)""").fetchone()[0]
    require(bad == 0, "Selected benchmark row is absent from bound eligibility index")
    counts = dict(db.execute("SELECT split,count(*) FROM selection GROUP BY split"))
    require(counts == {"train": args.train_records, "benchmark": args.benchmark_records}, "Exact split row counts not achieved")
    require(db.execute("SELECT count(DISTINCT image_id) FROM selection").fetchone()[0] == args.images,
            "Selected records do not cover exact image budget")
    require(db.execute("SELECT count(*) FROM records WHERE mandatory=1").fetchone()[0] ==
            db.execute("SELECT count(*) FROM selection WHERE mandatory=1").fetchone()[0], "Mandatory native row lost")
    require(db.execute("""SELECT count(*) FROM(SELECT scene_group FROM selection WHERE mandatory=1
        GROUP BY scene_group HAVING count(DISTINCT split)>1)""").fetchone()[0] == 0, "Native scene split conflict")
    by_stratum = [{"split": s, "source": src, "form": form, "variant": variant, "records": n}
                  for s, src, form, variant, n in db.execute(
                    "SELECT split,source,form,variant,count(*) FROM selection GROUP BY split,source,form,variant ORDER BY split,source,form,variant")]
    logical = hashlib.sha256(); selected_count = 0; replayed = 0
    # Validate one deterministic offset per (source,form,split); bind every selected index row.
    probes = set()
    for row in db.execute("SELECT uid,file_id,byte_offset,byte_length,split,source,form,variant,decoded_sha256,scene_group FROM selection ORDER BY rid"):
        logical.update((encode(list(row)) + "\n").encode()); selected_count += 1
        uid, fi, off, size, split, src, form, variant, key, scene = row
        probe = (split, src, form, variant)
        if probe not in probes:
            with Path(inputs[fi-1]["path"]).open("rb") as stream:
                stream.seek(off); raw = stream.read(size); restored = json.loads(raw)
            require(restored["id"] == uid, "Replay source offset does not reconstruct selected ID")
            probes.add(probe); replayed += 1
    forms = {r["form"] for r in by_stratum}
    missing_forms = sorted(set(FORMS) - forms)
    missing_by_split = {split: sorted(set(FORMS) - {r["form"] for r in by_stratum if r["split"] == split})
                        for split in ("train", "benchmark")}
    require(sha(Path(__file__)) == code_sha, "Allocator changed while running")
    require(sha(args.eligible_index) == eligibility_digest, "Eligibility index changed while running")
    require(sha(args.eligible_inventory) == inventory_digest, "Eligibility inventory changed while running")
    db.commit(); db.execute("DETACH DATABASE eligible"); db.close(); eligibility.close()
    manifest = {"schema": VERSION, "state": "count_complete_partial_source_pool",
        "selection_complete": True, "release_complete": False, "public_release_eligible": False,
        "qa_replay_count": 0, "required_qa_replay": 464000, "qa_replay_missing": 464000,
        "paper_pool_complete": False,
        "benchmark_scored": False, "all_records_semantically_reviewed": False,
        "targets": {"train_records": args.train_records, "benchmark_records": args.benchmark_records,
                    "total_records": args.train_records + args.benchmark_records, "unique_images": args.images},
        "observed": {"rows_by_split": counts, "selected_records": selected_count, "unique_images": args.images},
        "by_source_form_variant": by_stratum, "missing_native_forms": missing_forms,
        "missing_native_forms_by_split": missing_by_split,
        "pending_sources": args.pending_source, "selection": selection, "seed": args.seed,
        "algorithm": "Frozen mandatory native scenes; source-weighted benchmark whole-image capacity and form coverage; density then stable hash image ordering; image anchors plus capacity-capped weighted source/form Hamilton record quotas; stable record hash; no scores.",
        "partition_scope": "Native physical scenes/episodes retained. COCO/PixMo decoded-image groups are NOT a proof of physical-scene isolation.",
        "eligibility": {"index": str(args.eligible_index.resolve()), "index_sha256": eligibility_digest,
                        "inventory": str(args.eligible_inventory.resolve()), "inventory_sha256": inventory_digest,
                        "scope": "Eligibility is limited to the histories bound by supplied inventory; later training/teacher-feedback histories must be separately bound before claiming independent evaluation."},
        "rights": "All selected data local_only until dataset/component redistribution rights are verified. No image copied.",
        "inputs": inputs, "allocator_sha256": code_sha, "selection_sqlite_sha256": sha(db_path),
        "logical_selected_rows_sha256": logical.hexdigest(), "replayed_offset_strata": replayed,
        "verification": "All input record bytes hash checked, dense source bytes checked on both passes, native image bytes/decoded identities checked, all selected IDs counted and hashed; stratified offset replay.",
        "seconds": time.monotonic() - start}
    write_json(output / "manifest.json", manifest)
    progress("complete", selected_records=selected_count, unique_images=args.images, state=manifest["state"])
    return manifest


def qa_image_quotas(images, target, seed):
    """Choose genuine replay groups by density then fixed hash, never by scores."""
    require(type(target) is int and target > 0, "Positive QA target required")
    require(all(type(v["qa_count"]) is int and v["qa_count"] > 0 for v in images.values()),
            "Invalid QA image capacity")
    quotas = {}; remaining = target
    for key in sorted(images, key=lambda k: (-images[k]["qa_count"], rank(seed + "|qa-image", k), k)):
        if not remaining:
            break
        quotas[key] = min(remaining, images[key]["qa_count"])
        remaining -= quotas[key]
    require(remaining == 0, "Insufficient real QA capacity")
    return quotas


def selection_digest(db, split=None):
    h = hashlib.sha256(); count = 0
    query = ("SELECT uid,file_id,byte_offset,byte_length,split,source,form,variant,decoded_sha256,scene_group "
             "FROM selection" + (" WHERE split=?" if split else "") + " ORDER BY rid")
    for row in db.execute(query, (split,) if split else ()):
        h.update((encode(list(row)) + "\n").encode()); count += 1
    return h.hexdigest(), count


def replay_selected_offsets(db, inputs):
    """Read every selected source row in ascending offset order, not random IO."""
    counts = Counter()
    for item in inputs:
        previous_end = 0
        with Path(item["path"]).open("rb", buffering=8 << 20) as stream:
            query = ("SELECT uid,byte_offset,byte_length,source,form,variant FROM selection "
                     "WHERE file_id=? ORDER BY byte_offset")
            for uid, offset, size, source, form, variant in db.execute(query, (item["file_id"],)):
                require(offset >= previous_end and size > 0, "Overlapping or invalid selected source offsets")
                stream.seek(offset); raw = stream.read(size); row = json.loads(raw)
                require(len(raw) == size and row["id"] == uid, "Selected source identity changed")
                meta = row.get("anvil_meta", {})
                actual_form = "QA" if meta.get("record_type") == "qa_replay" else meta.get(
                    "native_form", row.get("native_form_candidate", row.get("native_form")))
                actual_variant = meta.get("native_contract", meta.get("native_variant", meta.get(
                    "schema", row.get("schema_version", row.get("contract")))))
                require(meta.get("source", row.get("source")) == source and actual_form == form
                        and actual_variant == variant, "Selected source form/variant changed")
                previous_end = offset + size; counts[item["file_id"]] += 1
    return sum(counts.values())


def merge_replay(args):
    """New version: immutable old benchmark + native train + QA, same total budget.

    Work happens in a private in-memory copy. The old database and all source
    records remain untouched. The final file is written sequentially to avoid
    pathological network-filesystem random-page IO.
    """
    start = time.monotonic(); output = args.output.resolve()
    require(not output.exists(), "Output must be a new immutable directory")
    code_sha = sha(Path(__file__))
    prior_sha = sha(args.base_manifest)
    prior = json.loads(args.base_manifest.read_text())
    require(prior.get("selection_complete") is True, "Base selection is incomplete")
    require(sha(args.base_index) == prior["selection_sqlite_sha256"], "Base index SHA256 mismatch")
    qa_manifest_path = args.qa_root / "manifest.json"
    qa_manifest_sha = sha(qa_manifest_path)
    qa_manifest = json.loads(qa_manifest_path.read_text())
    require(qa_manifest.get("schema") == "anvil-clevr-official-qa-replay-v1"
            and qa_manifest.get("state") == "official_QA_matched_local_image_candidates",
            "Unsupported or incomplete QA source")
    bound_benchmark = [v for v in qa_manifest["protections"] if v.get("kind") == "fixed_pool_benchmark"]
    require(len(bound_benchmark) == 1 and bound_benchmark[0]["sha256"] == prior_sha
            and bound_benchmark[0]["index_sha256"] == prior["selection_sqlite_sha256"],
            "QA protection is not bound to this exact benchmark")
    for name in ("train_candidates.jsonl", "images.jsonl"):
        require(sha(args.qa_root / name) == manifest_digest(qa_manifest, name), "QA source SHA256 mismatch")
    output.mkdir(parents=True)
    def progress(stage, **values):
        value = {"stage": stage, "elapsed_seconds": time.monotonic() - start, **values}
        write_json(output / "progress.json", value); print(encode(value), flush=True)
    src = sqlite3.connect(args.base_index.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA temp_store=MEMORY")
    src.backup(db, pages=4096); src.close()
    require(db.execute("PRAGMA quick_check").fetchone()[0] == "ok", "Base database failed integrity check")
    original_benchmark, benchmark_rows = selection_digest(db, "benchmark")
    targets = prior["targets"]
    require(benchmark_rows == targets["benchmark_records"], "Base benchmark count mismatch")
    require(selection_digest(db)[0] == prior["logical_selected_rows_sha256"], "Base logical selection changed")
    inputs = []
    for fi, path, checksum, source, kind, manifest, manifest_sha in db.execute("SELECT * FROM inputs ORDER BY file_id"):
        require(sha(path) == checksum and sha(manifest) == manifest_sha, "Base source/manifest changed")
        inputs.append({"file_id": fi, "path": path, "expected_sha256": checksum, "source": source,
                       "kind": kind, "manifest": manifest, "manifest_sha256": manifest_sha})
    old_mandatory = db.execute("SELECT count(*) FROM records WHERE mandatory=1").fetchone()[0]
    progress("base_bound_in_memory", benchmark_rows=benchmark_rows, mandatory_rows=old_mandatory)
    image_ids = {}; benchmark_fp = Fingerprints()
    for iid, pixel, raw, path, split, _, phash, dhash in db.execute("SELECT * FROM images"):
        image_ids[pixel] = (iid, split)
        if split == "benchmark":
            benchmark_fp.add({"sha256": raw, "decoded_rgb_sha256": pixel, "phash64": phash, "dhash64": dhash})
    def add_image(fp, path):
        key = fp["decoded_rgb_sha256"]
        require(not benchmark_fp.matches(fp), "New training image overlaps frozen benchmark")
        if key in image_ids:
            iid, split = image_ids[key]
            require(split == "train", "New source image conflicts with benchmark split")
            return iid
        iid = db.execute("INSERT INTO images(decoded_sha256,file_sha256,path,split,dense_benchmark_eligible,phash64,dhash64) "
                         "VALUES(?,?,?,'train',0,?,?)", (key, fp["sha256"], path, fp["phash64"], fp["dhash64"])).lastrowid
        image_ids[key] = (iid, "train")
        return iid
    insertion = ("INSERT INTO records(uid,file_id,byte_offset,byte_length,source,form,variant,image_id,"
                 "scene_group,original_split,mandatory,selection_rank) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)")
    def add_input(path, manifest_path, manifest, source, kind):
        checksum = manifest_digest(manifest, path.name); fi = max(v["file_id"] for v in inputs) + 1
        require(sha(path) == checksum, "New input source SHA256 mismatch")
        item = {"file_id": fi, "path": str(path.resolve()), "expected_sha256": checksum,
                "source": source, "kind": kind, "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha(manifest_path)}
        inputs.append(item)
        db.execute("INSERT INTO inputs VALUES(?,?,?,?,?,?,?)", (fi, item["path"], checksum, source, kind,
                                                               item["manifest"], item["manifest_sha256"]))
        return fi
    qa_images = {}
    for _, _, _, item in file_rows(args.qa_root / "images.jsonl"):
        key = item["decoded_rgb_sha256"]
        require(key not in qa_images and item["split"] == "train", "Duplicate/nontraining QA image")
        qa_images[key] = item
    qa_quotas = qa_image_quotas(qa_images, args.qa_records, args.seed)
    qa_image_ids = {key: add_image(qa_images[key], qa_images[key]["path"]) for key in qa_quotas}
    fi = add_input(args.qa_root / "train_candidates.jsonl", qa_manifest_path, qa_manifest, "CLEVR", "qa_replay")
    seen_qa = set(); qa_counts = Counter(); qa_total = 0
    for offset, size, raw, row in file_rows(args.qa_root / "train_candidates.jsonl"):
        meta = row["anvil_meta"]; key = meta["decoded_rgb_sha256"]
        require(row["id"] not in seen_qa, "Repeated QA source identity"); seen_qa.add(row["id"])
        require(key in qa_images and meta.get("record_type") == "qa_replay" and meta.get("split") == "train"
                and meta.get("training_approved") is True and meta.get("source") == "CLEVR"
                and meta.get("schema") == qa_manifest["schema"], "QA source has invalid training scope")
        info = qa_images[key]
        require(meta["image_sha256"] == info["sha256"] and row["images"] == [info["path"]]
                and meta["group_id"] == info["group_id"], "QA source and image inventory disagree")
        qa_counts[key] += 1
        if key not in qa_quotas or qa_counts[key] > qa_quotas[key]:
            continue
        db.execute(insertion, (row["id"], fi, offset, size, meta["source"], "QA", meta["schema"],
                              qa_image_ids[key], meta["group_id"], "train", 0,
                              rank(args.seed + "|record", row["id"])))
        qa_total += 1
    require(qa_total == args.qa_records and all(qa_counts[k] == v["qa_count"] for k, v in qa_images.items()),
            "QA counts do not match source inventory")
    require(len(seen_qa) == qa_manifest["counts"]["train_candidate_qa"], "QA manifest count mismatch")
    del seen_qa
    progress("qa_added", rows=qa_total, image_groups=len(qa_quotas))
    @lru_cache(maxsize=None)
    def native_fp(path):
        return fingerprint(path)
    new_native = 0
    for source, split, path_text, manifest_text in args.native_source:
        require(split == "train", "This refinement cannot add or alter benchmark records")
        path, manifest_path = Path(path_text), Path(manifest_text)
        manifest = json.loads(manifest_path.read_text())
        fi = add_input(path, manifest_path, manifest, source, "native")
        for offset, size, raw, row in file_rows(path):
            value = load_native_row(row, "train"); fp = native_fp(value["image"])
            require(fp["sha256"] == value["image_hash"], "New native image bytes changed")
            iid = add_image(fp, value["image"])
            db.execute(insertion, (row["id"], fi, offset, size, value["source"], value["form"], value["variant"],
                                  iid, value["scene"], value["original_split"], 1,
                                  rank(args.seed + "|record", row["id"])))
            new_native += 1
    benchmark_images = {r[0] for r in db.execute("SELECT image_id FROM images WHERE split='benchmark'")}
    fixed_train = {r[0] for r in db.execute("SELECT DISTINCT r.image_id FROM records r JOIN images i USING(image_id) "
                                          "WHERE i.split='train' AND (r.mandatory=1 OR r.form='QA')")}
    available = list(db.execute("SELECT r.image_id,i.decoded_sha256,count(*) FROM records r JOIN images i USING(image_id) "
                               "WHERE i.split='train' AND r.mandatory=0 AND r.form!='QA' GROUP BY r.image_id"))
    slots = targets["unique_images"] - len(benchmark_images) - len(fixed_train)
    require(slots >= 0, "QA plus frozen native groups exceed image budget")
    candidates = sorted((r for r in available if r[0] not in fixed_train),
                        key=lambda r: (-r[2], rank(args.seed + "|train-image", r[1]), r[1]))
    require(len(candidates) >= slots, "Insufficient train image groups")
    keep = fixed_train | {r[0] for r in candidates[:slots]} | benchmark_images
    drop = [(iid,) for iid, split in image_ids.values() if iid not in keep]
    db.execute("CREATE TEMP TABLE dropped_images(image_id INTEGER PRIMARY KEY)")
    db.executemany("INSERT INTO dropped_images VALUES(?)", drop)
    db.execute("DELETE FROM selected WHERE rid IN (SELECT rid FROM records JOIN images USING(image_id) WHERE split='train')")
    db.execute("DELETE FROM records WHERE image_id IN (SELECT image_id FROM dropped_images)")
    db.execute("DELETE FROM images WHERE image_id IN (SELECT image_id FROM dropped_images)")
    db.execute("DELETE FROM quotas WHERE split='train'")
    db.execute("INSERT INTO selected SELECT rid,'mandatory_native_or_qa' FROM records JOIN images USING(image_id) "
               "WHERE split='train' AND (mandatory=1 OR form='QA')")
    db.execute("""INSERT INTO selected SELECT rid,'image_coverage_anchor' FROM
        (SELECT r.rid,row_number() OVER(PARTITION BY r.image_id ORDER BY selection_rank,uid) rn
         FROM records r JOIN images i USING(image_id) WHERE i.split='train'
         AND NOT EXISTS(SELECT 1 FROM records n WHERE n.image_id=r.image_id AND (n.mandatory=1 OR n.form='QA')))
        WHERE rn=1""")
    # Derive mandatory mass directly; do not rely on a source summary's split labels.
    mandatory_train = db.execute("SELECT count(*) FROM records JOIN images USING(image_id) WHERE split='train' AND mandatory=1").fetchone()[0]
    required = targets["train_records"] - mandatory_train - qa_total
    capacities = {(s, f): n for s, f, n in db.execute("SELECT source,form,count(*) FROM records JOIN images USING(image_id) "
                  "WHERE split='train' AND mandatory=0 AND form!='QA' GROUP BY source,form")}
    floors = {(s, f): n for s, f, n in db.execute("SELECT source,form,count(*) FROM selection "
              "WHERE split='train' AND mandatory=0 AND form!='QA' GROUP BY source,form")}
    quotas = apportion(required, capacities, capacities, floors, args.seed + "|train")
    for (source, form), quota in sorted(quotas.items()):
        db.execute("INSERT INTO quotas VALUES('train',?,?,?)", (source, form, quota))
        db.execute("""INSERT INTO selected SELECT r.rid,'source_form_rank' FROM records r JOIN images i USING(image_id)
            WHERE i.split='train' AND r.source=? AND r.form=? AND r.mandatory=0
            AND NOT EXISTS(SELECT 1 FROM selected s WHERE s.rid=r.rid)
            ORDER BY selection_rank,uid LIMIT ?""", (source, form, quota - floors.get((source, form), 0)))
    counts = dict(db.execute("SELECT split,count(*) FROM selection GROUP BY split"))
    require(counts == {"benchmark": targets["benchmark_records"], "train": targets["train_records"]}, "Split count mismatch")
    require(selection_digest(db, "benchmark") == (original_benchmark, benchmark_rows), "Frozen benchmark changed")
    require(db.execute("SELECT count(DISTINCT image_id) FROM selection").fetchone()[0] == targets["unique_images"], "Image budget mismatch")
    require(db.execute("SELECT count(*) FROM selection WHERE mandatory=1").fetchone()[0] == old_mandatory + new_native, "Mandatory native rows lost")
    require(db.execute("SELECT count(*) FROM selection WHERE form='QA' AND split='train'").fetchone()[0] == args.qa_records, "QA quota mismatch")
    require(db.execute("SELECT count(*) FROM (SELECT scene_group FROM selection WHERE mandatory=1 "
                       "GROUP BY scene_group HAVING count(DISTINCT split)>1)").fetchone()[0] == 0, "Native scene split conflict")
    progress("recipe_selected", **counts, qa_records=qa_total, native_added=new_native, replaced_train_images=len(drop))
    replayed = replay_selected_offsets(db, inputs)
    require(replayed == sum(counts.values()), "Incomplete source-offset replay")
    logical, count = selection_digest(db)
    by_form = [{"split": s, "source": src, "form": form, "variant": variant, "records": n}
               for s, src, form, variant, n in db.execute("SELECT split,source,form,variant,count(*) FROM selection "
                                                         "GROUP BY split,source,form,variant ORDER BY split,source,form,variant")]
    missing = {split: sorted(set(FORMS) - {r["form"] for r in by_form if r["split"] == split}) for split in counts}
    for item in inputs:
        require(sha(item["path"]) == item["expected_sha256"] and sha(item["manifest"]) == item["manifest_sha256"],
                "Source/manifest changed during selection")
    require(sha(args.base_index) == prior["selection_sqlite_sha256"] and sha(args.base_manifest) == prior_sha
            and sha(qa_manifest_path) == qa_manifest_sha
            and sha(args.qa_root / "images.jsonl") == manifest_digest(qa_manifest, "images.jsonl")
            and sha(Path(__file__)) == code_sha, "Frozen input/code changed")
    db.commit(); path = output / "selection.sqlite"
    disk = sqlite3.connect(path); db.backup(disk, pages=4096); disk.close(); db.close()
    manifest = {"schema": VERSION, "state": "recipe_counts_complete_local_index", "selection_complete": True,
                "release_complete": False, "public_release_eligible": False, "benchmark_scored": False,
                "all_records_semantically_reviewed": False, "targets": targets,
                "observed": {"rows_by_split": counts, "selected_records": count, "unique_images": targets["unique_images"]},
                "qa_replay_count": qa_total, "required_qa_replay": args.qa_records, "qa_replay_missing": 0,
                "recipe_counts_complete": True,
                "paper_pool_complete": (targets["train_records"] == 5220000
                                        and targets["benchmark_records"] == 580000
                                        and targets["unique_images"] == 160000
                                        and qa_total == 464000 and not any(missing.values())),
                "missing_native_forms_by_split": missing, "by_source_form_variant": by_form,
                "seed": args.seed, "inputs": inputs, "allocator_sha256": code_sha,
                "selection_sqlite_sha256": sha(path), "logical_selected_rows_sha256": logical,
                "frozen_benchmark_logical_sha256": original_benchmark, "all_selected_offsets_replayed": replayed,
                "parent_manifest_sha256": prior_sha, "parent_index_sha256": prior["selection_sqlite_sha256"],
                "qa_source_manifest_sha256": qa_manifest_sha, "eligibility": prior["eligibility"],
                "partition_scope": prior["partition_scope"], "rights": prior["rights"],
                "selection": {"new_native_rows": new_native, "qa_image_groups": len(qa_quotas),
                              "dropped_nonmandatory_train_images": len(drop), "mandatory_train_rows": mandatory_train,
                              "dense_train_capacity": sum(capacities.values()), "dense_train_selected": required},
                "seconds": time.monotonic() - start}
    write_json(output / "manifest.json", manifest)
    progress("complete", selected_records=count, qa_records=qa_total, missing_native_forms_by_split=missing)
    return manifest


def reselect_dense_split(db, split, total, seed):
    """Keep native/QA anchors and fill remaining capacity without changing image splits."""
    db.execute("DELETE FROM selected WHERE rid IN (SELECT rid FROM records JOIN images USING(image_id) WHERE split=?)", (split,))
    db.execute("DELETE FROM quotas WHERE split=?", (split,))
    db.execute("INSERT INTO selected SELECT rid,'mandatory_native_or_qa' FROM records JOIN images USING(image_id) "
               "WHERE split=? AND (mandatory=1 OR form='QA')", (split,))
    frozen = db.execute("SELECT count(*) FROM selection WHERE split=?", (split,)).fetchone()[0]
    db.execute("""INSERT INTO selected SELECT rid,'image_coverage_anchor' FROM
        (SELECT r.rid,row_number() OVER(PARTITION BY r.image_id ORDER BY selection_rank,uid) rn
         FROM records r JOIN images i USING(image_id) WHERE i.split=?
         AND NOT EXISTS(SELECT 1 FROM records n WHERE n.image_id=r.image_id AND (n.mandatory=1 OR n.form='QA'))) WHERE rn=1""", (split,))
    required = total - frozen
    caps = {(s, f): n for s, f, n in db.execute("SELECT source,form,count(*) FROM records JOIN images USING(image_id) "
            "WHERE split=? AND mandatory=0 AND form!='QA' GROUP BY source,form", (split,))}
    floors = {(s, f): n for s, f, n in db.execute("SELECT source,form,count(*) FROM selection "
              "WHERE split=? AND mandatory=0 AND form!='QA' GROUP BY source,form", (split,))}
    quotas = apportion(required, caps, caps, floors, seed)
    for (source, form), amount in sorted(quotas.items()):
        db.execute("INSERT INTO quotas VALUES(?,?,?,?)", (split, source, form, amount))
        db.execute("""INSERT INTO selected SELECT r.rid,'source_form_rank' FROM records r JOIN images i USING(image_id)
            WHERE i.split=? AND r.source=? AND r.form=? AND r.mandatory=0 AND r.form!='QA'
            AND NOT EXISTS(SELECT 1 FROM selected s WHERE s.rid=r.rid) ORDER BY selection_rank,uid LIMIT ?""",
            (split, source, form, amount - floors.get((source, form), 0)))
    return {"dense_selected": required, "dense_capacity": sum(caps.values()), "native_and_qa_retained": frozen}


def extend_benchmark(args):
    """Version an unscored pool with independent native test rows.

    The old benchmark remains immutable, but this explicitly creates a DIFFERENT
    benchmark version. Dense benchmark rows are reallocated score-blind to keep
    the agreed count/image budgets, retaining every original mandatory target.
    Train stays exact by default. Explicit --rebalance-dense-training preserves
    all native/QA targets while reselecting dense train rows, when a strict image
    budget otherwise leaves insufficient benchmark capacity. No old scores transfer.
    """
    start = time.monotonic(); code_sha = sha(Path(__file__))
    require(not args.output.exists(), "Use a new immutable output directory")
    parent_sha = sha(args.base_manifest); parent = json.loads(args.base_manifest.read_text())
    require(parent.get("benchmark_scored") is False and parent.get("recipe_counts_complete") is True,
            "Only a complete-count, explicitly unscored pool may be extended")
    require(sha(args.base_index) == parent["selection_sqlite_sha256"], "Base index SHA256 mismatch")
    native_sha = sha(args.native_manifest); native = json.loads(args.native_manifest.read_text())
    require(native.get("schema") == "anvil-sharerobot-official-test-v1"
            and native.get("state") == "official_test_subset_exported"
            and native.get("training_approved") is False and native.get("source_split") == "test",
            "Expected independent official affordance test export")
    binding = native["protected_training"]
    require(binding["pool_index_sha256"] == parent["selection_sqlite_sha256"]
            and binding["pool_manifest_sha256"] == parent_sha, "New test exclusion does not bind this exact pool")
    source_sha = manifest_digest(native, args.native_records.name)
    require(sha(args.native_records) == source_sha, "Native test file SHA256 mismatch")
    db = sqlite3.connect(":memory:"); db.execute("PRAGMA temp_store=MEMORY")
    with sqlite3.connect(f"{args.base_index.resolve().as_uri()}?mode=ro", uri=True) as source:
        source.backup(db, pages=4096)
    require(db.execute("PRAGMA quick_check").fetchone()[0] == "ok", "Base index integrity failure")
    original_train = selection_digest(db, "train")
    original_benchmark = selection_digest(db, "benchmark")
    old_mandatory = db.execute("SELECT count(*) FROM selection WHERE mandatory=1").fetchone()[0]
    frozen_train = {r[0] for r in db.execute("SELECT rid FROM selection WHERE split='train' AND (mandatory=1 OR form='QA')")}
    rebalance_train = bool(getattr(args, "rebalance_dense_training", False))
    inputs = list(parent["inputs"])
    for item in inputs:
        require(sha(item["path"]) == item["expected_sha256"] and sha(item["manifest"]) == item["manifest_sha256"],
                "Parent input source/manifest drift")
    train_fp = Fingerprints(); image_ids = {}; next_image = 1
    for iid, decoded, raw, split, ph, dh in db.execute(
            "SELECT image_id,decoded_sha256,file_sha256,split,phash64,dhash64 FROM images"):
        image_ids[decoded] = (iid, split); next_image = max(next_image, iid + 1)
        if split == "train": train_fp.add({"sha256": raw, "decoded_rgb_sha256": decoded, "phash64": ph, "dhash64": dh})
    fid = max(item["file_id"] for item in inputs) + 1
    item = {"file_id": fid, "path": str(args.native_records), "expected_sha256": source_sha,
            "source": "sharerobot_official_test", "kind": "native",
            "manifest": str(args.native_manifest), "manifest_sha256": native_sha}
    inputs.append(item)
    db.execute("INSERT INTO inputs VALUES(?,?,?,?,?,?,?)", (fid, item["path"], source_sha,
               item["source"], item["kind"], item["manifest"], native_sha))
    added = 0; new_images = 0
    for offset, length, raw, row in file_rows(args.native_records):
        value = load_native_row(row, "benchmark"); meta = row["anvil_meta"]
        require(value["source"] == "ShareRobot-Bench" and value["form"] == "affordance"
                and meta.get("benchmark_eligible") is True and meta.get("training_approved") is False,
                "Native test scope changed")
        fp = fingerprint(value["image"])
        require(fp["sha256"] == value["image_hash"] and not train_fp.matches(fp), "Native test image changed/overlaps training")
        key = fp["decoded_rgb_sha256"]
        if key in image_ids:
            iid, split = image_ids[key]; require(split == "benchmark", "Native test collides with training image")
        else:
            iid = next_image; next_image += 1; new_images += 1
            db.execute("INSERT INTO images VALUES(?,?,?,?,?,?,?,?)", (iid, key, fp["sha256"], value["image"],
                "benchmark", 0, fp["phash64"], fp["dhash64"]))
            image_ids[key] = (iid, "benchmark")
        db.execute("""INSERT INTO records(uid,file_id,byte_offset,byte_length,source,form,variant,image_id,
            scene_group,original_split,mandatory,selection_rank) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (row["id"], fid, offset, length, value["source"], value["form"], value["variant"], iid,
             value["scene"], value["original_split"], 1, rank(args.seed + "|record", row["id"])))
        added += 1
    require(added == native["records"] and added > 0, "Native test count mismatch")
    train_images = {r[0] for r in db.execute("SELECT image_id FROM images WHERE split='train'")}
    fixed = {r[0] for r in db.execute("SELECT DISTINCT image_id FROM records JOIN images USING(image_id) WHERE mandatory=1 AND split='benchmark'")}
    dense = list(db.execute("SELECT image_id,decoded_sha256,count(*) FROM records JOIN images USING(image_id) "
                           "WHERE split='benchmark' AND mandatory=0 GROUP BY image_id"))
    image_seed = args.seed
    if rebalance_train:
        # Retain ALL previous benchmark images and add the independent test images.
        # The displaced images come only from nonmandatory, non-QA training groups.
        fixed_benchmark = {r[0] for r in db.execute("SELECT image_id FROM images WHERE split='benchmark'")}
        fixed = {r[0] for r in db.execute("SELECT DISTINCT image_id FROM records JOIN images USING(image_id) "
                                        "WHERE split='train' AND (mandatory=1 OR form='QA')")}
        dense = list(db.execute("SELECT image_id,decoded_sha256,count(*) FROM records JOIN images USING(image_id) "
                               "WHERE split='train' AND mandatory=0 AND form!='QA' GROUP BY image_id"))
        fixed_other = fixed_benchmark
        # Existing train record rank remains unchanged; only image budget differs.
        image_seed = parent["seed"] + "|train"
    else:
        fixed_other = train_images
    slots = parent["observed"]["unique_images"] - len(fixed_other) - len(fixed)
    optional = sorted((r for r in dense if r[0] not in fixed),
                      key=lambda r: (-r[2], rank(image_seed + "|image", r[1]), r[1]))
    require(0 <= slots <= len(optional), "Insufficient benchmark image budget/capacity")
    keep = fixed_other | fixed | {r[0] for r in optional[:slots]}
    drop = [(iid,) for iid, split in image_ids.values() if iid not in keep]
    db.execute("CREATE TEMP TABLE dropped_images(image_id INTEGER PRIMARY KEY)")
    db.executemany("INSERT INTO dropped_images VALUES(?)", drop)
    db.execute("DELETE FROM selected WHERE rid IN (SELECT rid FROM records WHERE image_id IN (SELECT image_id FROM dropped_images))")
    db.execute("DELETE FROM records WHERE image_id IN (SELECT image_id FROM dropped_images)")
    db.execute("DELETE FROM images WHERE image_id IN (SELECT image_id FROM dropped_images)")
    benchmark_selection = reselect_dense_split(db, "benchmark", parent["observed"]["rows_by_split"]["benchmark"], args.seed)
    train_selection = None
    if rebalance_train:
        train_selection = reselect_dense_split(db, "train", parent["observed"]["rows_by_split"]["train"], parent["seed"] + "|train")
    counts = dict(db.execute("SELECT split,count(*) FROM selection GROUP BY split"))
    current_train = selection_digest(db, "train")
    require(counts == parent["observed"]["rows_by_split"], "Count budget changed")
    require(rebalance_train or current_train == original_train, "Frozen training records changed without explicit opt-in")
    require(frozen_train == {r[0] for r in db.execute("SELECT rid FROM selection WHERE split='train' AND (mandatory=1 OR form='QA')")},
            "Frozen native training or QA changed")
    unique_images = db.execute("SELECT count(DISTINCT image_id) FROM selection").fetchone()[0]
    require(unique_images == parent["observed"]["unique_images"], "Image count budget changed")
    require(db.execute("SELECT count(*) FROM selection WHERE mandatory=1").fetchone()[0] == old_mandatory + added,
            "Existing mandatory native records lost")
    by_form = [{"split": s, "source": src, "form": form, "variant": v, "records": n}
               for s, src, form, v, n in db.execute("SELECT split,source,form,variant,count(*) FROM selection "
                                                   "GROUP BY split,source,form,variant ORDER BY split,source,form,variant")]
    missing = {split: sorted(set(FORMS) - {r["form"] for r in by_form if r["split"] == split}) for split in counts}
    replayed = replay_selected_offsets(db, inputs); logical, count = selection_digest(db)
    require(replayed == count == sum(counts.values()), "Incomplete source offset replay")
    benchmark, benchmark_count = selection_digest(db, "benchmark")
    for item in inputs:
        require(sha(item["path"]) == item["expected_sha256"] and sha(item["manifest"]) == item["manifest_sha256"], "Source drift during refinement")
    require(sha(args.base_index) == parent["selection_sqlite_sha256"] and sha(args.base_manifest) == parent_sha
            and sha(Path(__file__)) == code_sha, "Parent/code drift during refinement")
    args.output.mkdir(parents=True, exist_ok=False); db.commit()
    path = args.output / "selection.sqlite"
    with sqlite3.connect(path) as disk: db.backup(disk, pages=4096)
    db.close()
    result = {**parent, "state": "count_and_native_form_complete_local_index" if not any(missing.values()) else "recipe_counts_complete_local_index",
        "inputs": inputs, "seed": args.seed, "allocator_sha256": code_sha, "seconds": time.monotonic() - start,
        "selection_sqlite_sha256": sha(path), "logical_selected_rows_sha256": logical,
        "parent_manifest_sha256": parent_sha, "parent_index_sha256": parent["selection_sqlite_sha256"],
        "benchmark_scored": False, "release_complete": False, "public_release_eligible": False,
        "by_source_form_variant": by_form, "missing_native_forms_by_split": missing,
        "paper_pool_complete": False, "native_form_coverage_complete": not any(missing.values()),
        "paper_alignment_scope": "Counts and eleven native form labels complete; source variants/composition and method claims still require separate paper alignment.",
        "all_selected_offsets_replayed": replayed, "frozen_benchmark_logical_sha256": benchmark,
        "frozen_training_logical_sha256": current_train[0],
        "training_version_change": {"old_sha256": original_train[0], "new_sha256": current_train[0],
            "dense_rebalance_enabled": rebalance_train, "all_rows_unchanged": current_train == original_train,
            "native_and_qa_unchanged": True},
        "benchmark_version_change": {"old_sha256": original_benchmark[0], "new_sha256": benchmark,
            "records": benchmark_count, "new_native_records": added, "new_image_groups": new_images,
            "removed_nonmandatory_image_groups": len(drop), "removed_groups_split": "train" if rebalance_train else "benchmark",
            "old_scores_transferable": False},
        "selection": {"parent_selection": parent["selection"], "added_official_test_rows": added,
                      "benchmark": benchmark_selection, "train": train_selection}}
    write_json(args.output / "manifest.json", result)
    print(encode({"state": result["state"], "observed": result["observed"], "missing_native_forms_by_split": missing,
                  "new_test_records": added, "seconds": result["seconds"]}), flush=True)
    return result


def benchmark_main():
    p = argparse.ArgumentParser(description=extend_benchmark.__doc__)
    for name in ("base-index", "base-manifest", "native-records", "native-manifest", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--seed", default="Anvil-native-benchmark-v4")
    p.add_argument("--rebalance-dense-training", action="store_true",
                   help="Explicitly reselect dense train rows to free image slots; native train and QA stay exact")
    extend_benchmark(p.parse_args(sys.argv[2:]))


def merge_main():
    p = argparse.ArgumentParser(description=merge_replay.__doc__)
    for name in ("base-index", "base-manifest", "qa-root", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--qa-records", type=int, default=464000)
    p.add_argument("--seed", default="Anvil-complete-recipe-v3")
    p.add_argument("--native-source", nargs=4, action="append", default=[],
                   metavar=("NAME", "SPLIT", "JSONL", "MANIFEST"))
    args = p.parse_args(sys.argv[2:])
    try:
        merge_replay(args)
    except Exception as error:
        if args.output.exists() and not (args.output / "manifest.json").exists():
            write_json(args.output / "failure.json", {"state": "failed_no_complete_pool", "type": type(error).__name__,
                                                       "reason": str(error), "release_complete": False})
        raise


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "extend-benchmark":
        return benchmark_main()
    if len(sys.argv) > 1 and sys.argv[1] == "complete-with-qa":
        return merge_main()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dense-root", type=Path, required=True)
    p.add_argument("--eligible-index", type=Path, required=True)
    p.add_argument("--eligible-inventory", type=Path, required=True)
    p.add_argument("--native-source", nargs=4, action="append", default=[],
                   metavar=("NAME", "SPLIT", "JSONL", "MANIFEST"))
    p.add_argument("--pending-source", action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-records", type=int, default=5220000)
    p.add_argument("--benchmark-records", type=int, default=580000)
    p.add_argument("--images", type=int, default=160000)
    p.add_argument("--seed", default="Anvil-source-pool-v1")
    args = p.parse_args()
    if min(args.train_records, args.benchmark_records, args.images) <= 0:
        p.error("Positive record/image targets are required")
    try:
        result = build(args)
        print(encode({k: result[k] for k in ("state", "observed", "missing_native_forms", "pending_sources")}))
    except Exception as error:
        if args.output.exists() and not (args.output / "manifest.json").exists():
            write_json(args.output / "failure.json", {"state": "failed_no_complete_pool", "type": type(error).__name__,
                                                     "reason": str(error), "release_complete": False})
        raise


if __name__ == "__main__":
    main()
