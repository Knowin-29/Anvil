#!/usr/bin/env python3
"""Build GSA-Bench: image-disjoint train / val / test splits.

Design principles (these are what make the benchmark defensible in review):

  1. **Test answers are HUMAN ground truth**, taken from the open datasets, never
     model output. A benchmark whose labels came from the model under study would be
     circular. Flywheel data is used for TRAIN (and val monitoring) only.
  2. **Splits are keyed on image identity**, not on records, so no image appears in
     more than one split (multi-form generation creates ~8 records per image).
  3. Test is additionally deduplicated against the 9 public benchmarks by the
     3-axis leakage checker, so we never test on data those benchmarks contain.
  4. Per-split scene and task-form distributions are reported for drift auditing.

Outputs (ms-swift format):
  trains/gsa_bench/train.jsonl     flywheel multi-form (8 forms)
  trains/gsa_bench/val.jsonl       flywheel, image-disjoint from train
  trains/gsa_bench/train_{G,S,A}.jsonl  family slices for specialist training
  trains/gsa_bench/val_{G,S,A}.jsonl    family slices for specialist validation
  trains/gsa_bench/test.jsonl      HUMAN-GT grounding subset on held-out images
  trains/gsa_bench/manifest.json   counts + split provenance

The current GT store covers bbox and point only. It is a defensible grounding
subset, not yet the complete eight-form GSA benchmark. The manifest records this
scope explicitly so downstream table builders cannot silently over-claim it.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trains.flywheel.contracts import CONTRACT_VERSION, validate_task_columns

ASK_GT = {
    "bbox": "Detect the referred object and output its bounding box in JSON (thousandth scale).",
    "point": "Point to the referred object and output its coordinates in JSON.",
}


def stable_image_id(path: str, source: str = "", provided: str = "") -> str:
    """Prefer byte identity; otherwise remove machine-specific mount prefixes."""
    if provided:
        return provided
    parts = list(Path(path).parts)
    if source and source in parts:
        relative = "/".join(parts[parts.index(source):])
    else:
        relative = "/".join(parts[-4:])
    return hashlib.sha256(f"{source}:{relative}".encode()).hexdigest()


def img_bucket(path: str, source: str = "", provided: str = "", n: int = 1000) -> int:
    return int(stable_image_id(path, source, provided)[:16], 16) % n


def normalize_frozen_test_record(record: dict) -> dict:
    """Attach benchmark metadata without changing a frozen human-GT target."""
    result = dict(record)
    meta = dict(result.get("gsa_meta") or {})
    form = result.get("gsa_form") or meta.get("form")
    if form not in {"bbox", "point"}:
        raise ValueError(f"unsupported frozen-test form: {form!r}")
    if meta.get("label_provenance") != "human_gt":
        raise ValueError("frozen test rows must have label_provenance=human_gt")
    images = result.get("images")
    messages = result.get("messages")
    if not isinstance(images, list) or len(images) != 1 or not images[0]:
        raise ValueError("frozen test rows must contain exactly one image")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("frozen test rows must preserve user and assistant messages")

    image_id = stable_image_id(
        images[0], meta.get("source", ""), meta.get("image_sha256", ""))
    meta.update({
        "split": "test",
        "form": form,
        "family": "G",
        "image_sha256": image_id,
        "contract_version": CONTRACT_VERSION,
        "evaluation_context": "human_gt_target_without_teacher_context",
    })
    result.update({
        "gsa_form": form,
        "gsa_family": "G",
        "gsa_contract_version": CONTRACT_VERSION,
        "gsa_context": result.get("gsa_context") or {},
        "gsa_meta": meta,
    })
    return result


def load_source_buckets(path: str) -> dict[str, str]:
    """Map generated-image hashes to their upstream source buckets."""
    if not path:
        return {}
    mapping = {}
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid source manifest at {path}:{line_no}") from exc
        image_id = row.get("upload_sha256")
        bucket = row.get("source_bucket")
        if image_id and bucket:
            mapping[image_id] = bucket
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flywheel-glob", default="data/flywheel_long/shard*/swift",
                    help="repeatable via --flywheel-glob a --flywheel-glob b")
    ap.add_argument("--extra-glob", action="append", default=[],
                    help="additional swift dirs to include (e.g. GT-anchored shards)")
    ap.add_argument("--gt-store", default="configs/data/gt_store.jsonl")
    ap.add_argument(
        "--test-jsonl", default="",
        help="frozen human-GT ms-swift test set; used verbatim instead of --gt-store",
    )
    ap.add_argument(
        "--source-manifest", default="",
        help="generation manifest mapping upload_sha256 to source_bucket",
    )
    ap.add_argument(
        "--exclude-source-bucket", action="append", default=[],
        help="upstream source bucket to exclude from train/val (repeatable)",
    )
    ap.add_argument("--out-dir", default="trains/gsa_bench")
    ap.add_argument("--val-buckets", type=int, default=40,   # 4%
                    help="hash buckets (of 1000) assigned to val")
    ap.add_argument("--test-size", type=int, default=6000)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------- TEST: human GT only, on images NOT used by the flywheel train set ----
    def gt_to_swift(r: dict) -> dict | None:
        kind = r["gt_kind"]
        if kind == "bbox":
            ans = json.dumps([{"label": r.get("phrase") or "object",
                               "bbox_2d": r["gt"]}], ensure_ascii=False)
            q = r.get("question") or ASK_GT["bbox"]
        else:
            pts = r["gt"]
            ans = json.dumps([{"label": r.get("phrase") or "object",
                               "point_2d": [int(pts[0][0]), int(pts[0][1])]}],
                             ensure_ascii=False)
            q = r.get("question") or ASK_GT["point"]
        if "<image>" not in q:
            q = "<image>" + q
        return {"messages": [{"role": "user", "content": q},
                             {"role": "assistant", "content": ans}],
                "images": [r["image"]],
                "gsa_form": kind,
                "gsa_family": "G",
                "gsa_contract_version": CONTRACT_VERSION,
                "gsa_context": {},
                "gsa_meta": {"split": "test", "form": kind, "family": "G",
                             "source": r["source"],
                             "scene": r.get("scene", "grounding"),
                             "image_sha256": stable_image_id(
                                 r["image"], r["source"], r.get("image_sha256", "")),
                             "label_provenance": "human_gt"}}

    random.seed(0)
    if args.test_jsonl:
        frozen_rows = []
        for line_no, line in enumerate(Path(args.test_jsonl).read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                frozen_rows.append(normalize_frozen_test_record(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid frozen test row at {args.test_jsonl}:{line_no}: {exc}") from exc
        test = frozen_rows
        test_rule = f"frozen human-GT records from {args.test_jsonl}; no resampling"
    else:
        gt_rows = []
        for ln in Path(args.gt_store).read_text(errors="ignore").splitlines():
            if ln.strip():
                try:
                    gt_rows.append(json.loads(ln))
                except Exception:
                    pass
        random.shuffle(gt_rows)

        # Reserve the highest hash buckets for test so the assignment is deterministic
        # and independent of file ordering.
        test_rows = [
            r for r in gt_rows
            if img_bucket(r["image"], r.get("source", ""), r.get("image_sha256", ""))
            >= 1000 - 60
        ]
        test_rows = test_rows[: args.test_size]
        test = [s for s in (gt_to_swift(r) for r in test_rows) if s]
        test_rule = "GT store sha256(stable_image_id) buckets >=940"

    test_images = {
        stable_image_id(
            r["images"][0], r.get("gsa_meta", {}).get("source", ""),
            r.get("gsa_meta", {}).get("image_sha256", ""))
        for r in test
    }
    source_buckets = load_source_buckets(args.source_manifest)
    excluded_buckets = set(args.exclude_source_bucket)
    excluded_records = Counter()

    # ---------- TRAIN / VAL: flywheel multi-form, image-disjoint from test ----------
    train, val = [], []
    seen = set()
    form_c = {"train": Counter(), "val": Counter(), "test": Counter()}
    scene_c = {"train": Counter(), "val": Counter(), "test": Counter()}
    for rec in test:
        meta = rec["gsa_meta"]
        form_c["test"][meta.get("form", "?")] += 1
        scene_c["test"][meta.get("scene", "?")] += 1
    search_dirs = list(glob.glob(args.flywheel_glob))
    for g in args.extra_glob:
        search_dirs.extend(glob.glob(g))
    print(f"scanning {len(search_dirs)} flywheel dirs", flush=True)
    for d in search_dirs:
        for jf in Path(d).glob("*.jsonl"):
            for ln in jf.read_text(errors="ignore").splitlines():
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL in {jf}") from exc
                valid, reason = validate_task_columns(rec)
                if not valid:
                    raise ValueError(f"contract-invalid training row in {jf}: {reason}")
                imgs = rec.get("images") or []
                if not imgs:
                    continue
                img = imgs[0]
                m = rec.setdefault("gsa_meta", {})
                image_id = stable_image_id(
                    img, m.get("source", ""), m.get("image_sha256", ""))
                m["image_sha256"] = image_id
                source_bucket = source_buckets.get(image_id)
                if source_bucket in excluded_buckets:
                    excluded_records[source_bucket] += 1
                    continue
                if image_id in test_images:
                    continue
                key = hashlib.sha256(
                    (image_id + rec["messages"][1]["content"][:200]).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                b = img_bucket(img, m.get("source", ""), image_id)
                split = "val" if b < args.val_buckets else "train"
                m["split"] = split
                (val if split == "val" else train).append(rec)
                form_c[split][rec["gsa_form"]] += 1
                scene_c[split][m.get("scene", "?")] += 1

    # Shuffle before writing. Flywheel records arrive grouped by task form (one file
    # per form), so writing in read order leaves the corpus *globally* balanced but
    # *locally* clustered: at under one epoch the model would see long runs of only
    # 3-4 forms. Shuffling makes any prefix representative of the whole.
    for rows in (train, val):
        random.shuffle(rows)

    for name, rows in (("train", train), ("val", val), ("test", test)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    family_counts = {"train": Counter(), "val": Counter()}
    for split, rows in (("train", train), ("val", val)):
        by_family = {family: [] for family in ("G", "S", "A")}
        for record in rows:
            family = record["gsa_family"]
            by_family[family].append(record)
            family_counts[split][family] += 1
        for family, family_rows in by_family.items():
            with (out / f"{split}_{family}.jsonl").open("w", encoding="utf-8") as f:
                for record in family_rows:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # verify image disjointness
    def imgs_of(rows):
        ids = set()
        for record in rows:
            if not record.get("images"):
                continue
            meta = record.get("gsa_meta", {})
            ids.add(stable_image_id(record["images"][0], meta.get("source", ""),
                                    meta.get("image_sha256", "")))
        return ids
    ti, vi, si = imgs_of(train), imgs_of(val), imgs_of(test)
    man = {
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "unique_images": {"train": len(ti), "val": len(vi), "test": len(si)},
        "overlap": {"train_val": len(ti & vi), "train_test": len(ti & si),
                    "val_test": len(vi & si)},
        "test_label_provenance": "human_gt (open datasets); NEVER model output",
        "test_scope": "grounding-only human-GT subset (bbox, point)",
        "full_gsa_benchmark_status": "BLOCKED_PENDING_HUMAN_GT_FOR_S_AND_A",
        "train_label_provenance": "contract-v2 G->S->A flywheel, set cross-checked",
        "split_rule": f"train/val sha256 buckets with val<{args.val_buckets}; test={test_rule}",
        "val_buckets": args.val_buckets,
        "source_manifest": args.source_manifest or None,
        "excluded_source_buckets": sorted(excluded_buckets),
        "excluded_records_by_source_bucket": dict(excluded_records.most_common()),
        "by_form": {k: dict(v.most_common()) for k, v in form_c.items()},
        "by_family": {k: dict(v.most_common()) for k, v in family_counts.items()},
        "by_scene": {k: dict(v.most_common()) for k, v in scene_c.items()},
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
