#!/usr/bin/env python3
"""Materialize a v3 semantic-audit release into an ms-swift corpus.

The v3 audit is a calibration/release gate, not a benchmark test-set builder.
This command deliberately has a narrow trust policy:

* the supplied compiler report must be a fresh schema-v3 PASS;
* reviewed samples use the A/B (or explicit adjudication) decision;
* an unreviewed record must pass the strict deterministic validator;
* a reviewed-true audit candidate may pass the structural validator, but it is
  marked ``human_semantic_audit`` so downstream contract checks can require an
  explicit opt-in;
* reviewed-false samples are never emitted, even when their machine validator
  happens to pass.

The result is an image-disjoint train/val pilot corpus with family slices and a
manifest bound to the audit pack, source fingerprint, and compiler-report hash.
Existing output is never overwritten; run this after the final audit report is
compiled and inspect the manifest before submitting any GPU task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trains.flywheel.contracts import CONTRACT_VERSION, FORM_SPECS, image_sha256, task_columns, training_instruction
from trains.flywheel.validators import validate, validate_structural

FORMS = [
    "action_cot", "affordance", "bbox", "depth",
    "detection", "dof", "mask", "point",
]

ASK = {
    "bbox": "Detect the main manipulable object and output its bounding box in JSON (thousandth scale).",
    "detection": "Detect all distinct manipulable objects and output labelled bounding boxes in JSON.",
    "mask": "Segment the most prominent manipulable object as a polygon in JSON.",
    "point": "Point to the most prominent manipulable object in JSON.",
    "depth": "Order the visible objects from nearest to farthest with relative depths, in JSON.",
    "affordance": "Identify the sub-part to contact to use the main object, with a contact point, in JSON.",
    "dof": "Specify a parallel-jaw grasp (contact point, jaw axis, approach, wrist angle) in JSON.",
    "action_cot": "Plan a physically appropriate manipulation of the main object, step by step, in JSON.",
}

SYSTEM_PROMPT = (
    "You are an embodied perception assistant. Use the visible image and the "
    "provided task contract; return only the requested structured JSON."
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSON at {path}:{line_number}")
        rows.append(row)
    return rows


def _latest_annotations(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        snapshot = row.get("label_snapshot")
        sample_id = snapshot.get("sample_id") if isinstance(snapshot, dict) else None
        if isinstance(sample_id, str) and sample_id:
            latest[sample_id] = row
    return latest


def _load_adjudication(path: Path | None) -> dict[str, bool]:
    if path is None:
        return {}
    out: dict[str, bool] = {}
    for row in _read_jsonl(path):
        sample_id = row.get("sample_id")
        value = row.get("correct")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(value, bool):
            raise ValueError(f"adjudication rows require sample_id and Boolean correct: {path}")
        if sample_id in out:
            raise ValueError(f"duplicate adjudication for {sample_id}")
        if value is False and not isinstance(row.get("error_code"), str):
            raise ValueError(f"negative adjudication lacks error_code: {sample_id}")
        out[sample_id] = value
    return out


def _compiler_report(
    pack: Path,
    report_path: Path,
    adjudication: Path | None,
    min_accuracy: float,
    min_agreement: float,
) -> tuple[dict[str, Any], str]:
    """Recompute the report and reject a stale/foreign report file."""
    from eval.compile_semantic_audit import compile_report

    try:
        supplied = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read compiler report {report_path}: {exc}") from exc
    if not isinstance(supplied, dict) or supplied.get("verdict") != "PASS":
        raise ValueError("semantic audit report must have verdict=PASS")
    if supplied.get("schema_version") != 3:
        raise ValueError("materialization requires a schema-v3 audit report")
    declared_manifest = supplied.get("manifest")
    if not isinstance(declared_manifest, str) or Path(declared_manifest).resolve() != (pack / "manifest.json").resolve():
        raise ValueError("compiler report is bound to a different audit pack")

    args = argparse.Namespace(
        pack=pack,
        adjudication=adjudication,
        min_accuracy=min_accuracy,
        min_agreement=min_agreement,
    )
    recomputed, rc = compile_report(args)
    if rc != 0 or recomputed.get("verdict") != "PASS":
        raise ValueError("recomputed semantic audit is not PASS; report may be stale")
    # Compare decision-bearing fields rather than formatting/path fields. This catches
    # annotations appended after the report was compiled without requiring byte identity.
    for field in ("schema_version", "verdict", "by_form", "unresolved_sample_ids", "errors", "annotator_keys"):
        if supplied.get(field) != recomputed.get(field):
            raise ValueError(f"compiler report is stale or inconsistent in field {field!r}")
    return supplied, sha256_file(report_path)


def _pack_truth(
    pack: Path,
    report: dict[str, Any],
    adjudication_path: Path | None,
) -> tuple[dict[str, bool], dict[str, str], dict[str, Any]]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3 or manifest.get("forms") != FORMS:
        raise ValueError("pack is not the fixed schema-v3 eight-form audit pack")
    expected: dict[str, str] = {}
    for item in manifest.get("selected", []):
        for form, sample_id in (item.get("sample_ids") or {}).items():
            if form not in FORMS or not isinstance(sample_id, str):
                raise ValueError("invalid sample mapping in audit manifest")
            if sample_id in expected:
                raise ValueError(f"duplicate audit sample_id {sample_id}")
            expected[sample_id] = form
    if len(expected) != 40 * len(FORMS):
        raise ValueError(f"audit pack must contain 320 samples, found {len(expected)}")

    a = _latest_annotations(pack / "annotator_a/annotations/annotations.jsonl")
    b = _latest_annotations(pack / "annotator_b/annotations/annotations.jsonl")
    adjudication = _load_adjudication(adjudication_path)
    truth: dict[str, bool] = {}
    decisions: dict[str, str] = {}
    for sample_id, form in expected.items():
        va = a.get(sample_id, {}).get("correct")
        vb = b.get(sample_id, {}).get("correct")
        if not isinstance(va, bool) or not isinstance(vb, bool):
            raise ValueError(f"missing Boolean A/B decision for {sample_id}")
        if va == vb:
            truth[sample_id] = va
            decisions[sample_id] = "agreed"
        elif sample_id in adjudication:
            truth[sample_id] = adjudication[sample_id]
            decisions[sample_id] = "adjudicated"
        else:
            raise ValueError(f"unresolved disagreement for {sample_id}")
    if report.get("unresolved_sample_ids"):
        raise ValueError("compiler report still contains unresolved disagreements")
    return truth, expected, manifest


def _load_source_records(source: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    labels = source / "labels"
    if not labels.is_dir():
        raise ValueError(f"source labels directory is missing: {labels}")
    for path in sorted(labels.rglob("*.jsonl")):
        for row in _read_jsonl(path):
            sample_id = row.get("sample_id")
            form = row.get("task_form")
            if not isinstance(sample_id, str) or not sample_id or form not in FORMS:
                raise ValueError(f"source row lacks sample_id/task_form: {path}")
            if sample_id in records:
                raise ValueError(f"duplicate source sample_id: {sample_id}")
            records[sample_id] = row
    if not records:
        raise ValueError(f"source contains no labels: {source}")
    return records


def _make_swift(row: dict[str, Any], split: str, audit: dict[str, Any], report_sha: str,
                reviewed: bool) -> dict[str, Any]:
    form = row["task_form"]
    image = Path(str(row.get("image_path", ""))).resolve()
    if not image.is_file():
        raise ValueError(f"source image is missing: {image}")
    ctx = row.get("gsa_context")
    if not isinstance(ctx, dict):
        raise ValueError(f"source context is not an object: {row.get('sample_id')}")
    payload = row.get("payload")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"source payload is not a non-empty list: {row.get('sample_id')}")
    digest = image_sha256(str(image))
    source = str(row.get("source") or row.get("dataset") or "unknown")
    scene = str(row.get("scene_type") or row.get("scene") or "embodied_home")
    out = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<image>" + ASK[form]},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "images": [str(image)],
        "sample_id": row["sample_id"],
        "gsa_payload": payload,
    }
    out.update(task_columns(form, ctx, source=source, scene=scene, image_id=digest))
    out["gsa_meta"].update({
        "split": split,
        "label_provenance": "human_semantic_audit" if reviewed else "machine_verified_v3",
        "audit_reviewed": reviewed,
        "semantic_audit_verdict": "PASS",
        "semantic_audit_id": audit.get("audit_id"),
        "semantic_audit_source_fingerprint": audit.get("source_fingerprint"),
        "semantic_audit_report_sha256": report_sha,
        "source_sample_id": row["sample_id"],
        "label_version": row.get("label_version"),
        "prompt_version": row.get("prompt_version"),
    })
    return out


def _split_rows(rows: list[dict[str, Any]], seed: int, val_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[row["gsa_meta"]["image_sha256"]].append(row)
    if len(by_image) < 2:
        raise ValueError("at least two unique images are required for an image-disjoint split")
    image_keys = sorted(by_image)
    target = max(1, min(len(image_keys) - 1, round(len(image_keys) * val_fraction)))
    ranked = sorted(
        image_keys,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(),
    )
    val_images = set(ranked[:target])

    def forms(images: set[str]) -> set[str]:
        return {row["gsa_form"] for key in images for row in by_image[key]}

    # Keep every form represented in both splits. Moving a whole image preserves
    # the no-image-overlap guarantee and is deterministic.
    for _ in range(len(FORMS) * 2):
        changed = False
        for form in FORMS:
            for destination, origin in ((val_images, set(image_keys) - val_images),
                                        (set(image_keys) - val_images, val_images)):
                if form in forms(destination):
                    continue
                candidates = [key for key in origin if any(r["gsa_form"] == form for r in by_image[key])]
                if not candidates or len(destination) >= len(image_keys) - 1:
                    continue
                key = min(candidates)
                destination.add(key)
                origin.discard(key)
                changed = True
        if not changed:
            break
    train_images = set(image_keys) - val_images
    if forms(train_images) != set(FORMS) or forms(val_images) != set(FORMS):
        raise ValueError(
            f"image split does not cover all forms: train={sorted(forms(train_images))}, "
            f"val={sorted(forms(val_images))}")
    train = [row for key in train_images for row in by_image[key]]
    val = [row for key in val_images for row in by_image[key]]
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(val)
    return train, val


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    pack = args.pack.resolve()
    source = args.source.resolve()
    report_path = args.audit.resolve()
    adjudication = args.adjudication.resolve() if args.adjudication else None
    if adjudication is None:
        candidate = pack / "adjudication.jsonl"
        if candidate.is_file():
            adjudication = candidate
    report, report_sha = _compiler_report(
        pack, report_path, adjudication, args.min_accuracy, args.min_agreement)
    truth, expected, pack_manifest = _pack_truth(pack, report, adjudication)
    source_records = _load_source_records(source)

    selected_by_id = {sid for sid in expected}
    rows: list[dict[str, Any]] = []
    stats = Counter()
    excluded = Counter()
    seen_images: set[str] = set()
    for sample_id, row in sorted(source_records.items()):
        form = row["task_form"]
        payload = row.get("payload")
        strict, strict_reason = validate(form, payload, row.get("gsa_context"))
        reviewed = sample_id in selected_by_id
        # The default release scope is the fixed, independently reviewed 40-image
        # sample. Full-pool materialization is intentionally explicit: those rows
        # are strict machine-filtered calibration candidates, not human consensus.
        if args.scope == "audited" and not reviewed:
            excluded["outside_audit_scope"] += 1
            continue
        if reviewed:
            if truth[sample_id] is False:
                excluded["human_false"] += 1
                continue
            structural, structural_reason = validate_structural(form, payload, row.get("gsa_context"))
            if not structural:
                raise ValueError(f"human-true row is structurally invalid {sample_id}: {structural_reason}")
            if not strict:
                stats["human_structural_override"] += 1
        elif not strict:
            excluded[f"strict:{strict_reason}"] += 1
            continue
        try:
            image_path = Path(str(row.get("image_path", ""))).resolve()
            digest = image_sha256(str(image_path))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot hash source image for {sample_id}: {exc}") from exc
        source_bucket = row.get("source_bucket")
        if not isinstance(source_bucket, str) or not source_bucket.strip():
            raise ValueError(f"source bucket provenance is missing for {sample_id}")
        if source_bucket in {"household", "refspatial"}:
            excluded[f"source_bucket:{source_bucket}"] += 1
            continue
        materialized = _make_swift(row, "pending", pack_manifest, report_sha, reviewed)
        materialized["gsa_meta"]["image_sha256"] = digest
        rows.append(materialized)
        stats["reviewed" if reviewed else "unreviewed_strict"] += 1
        seen_images.add(digest)

    if not rows:
        raise ValueError("no rows survived the post-audit materialization policy")
    train, val = _split_rows(rows, args.seed, args.val_fraction)
    for row in train:
        row["gsa_meta"]["split"] = "train"
    for row in val:
        row["gsa_meta"]["split"] = "val"

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=str(out.parent)))
    try:
        _write_jsonl(tmp / "train.jsonl", train)
        _write_jsonl(tmp / "val.jsonl", val)
        for split, split_rows in (("train", train), ("val", val)):
            for family in ("G", "S", "A"):
                _write_jsonl(
                    tmp / f"{split}_{family}.jsonl",
                    [row for row in split_rows if row["gsa_family"] == family],
                )
        all_images = {
            split: {row["gsa_meta"]["image_sha256"] for row in split_rows}
            for split, split_rows in (("train", train), ("val", val))
        }
        forms = {split: Counter(row["gsa_form"] for row in split_rows)
                 for split, split_rows in (("train", train), ("val", val))}
        families = {split: Counter(row["gsa_family"] for row in split_rows)
                    for split, split_rows in (("train", train), ("val", val))}
        manifest = {
            "schema_version": 1,
            "corpus_version": "gsa-bench-v3-pilot-clean",
            "label_policy": (
                "human_consensus_only" if args.scope == "audited"
                else "human_consensus_plus_strict_machine_verified"
            ),
            "scope": args.scope,
            "source_root": str(source),
            "source_records": len(source_records),
            "accepted_records": len(rows),
            "excluded_records": sum(excluded.values()),
            "excluded_by_reason": dict(excluded),
            "review_stats": dict(stats),
            "counts": {"train": len(train), "val": len(val)},
            "unique_images": {"train": len(all_images["train"]), "val": len(all_images["val"])},
            "overlap": {
                "train_val": len(all_images["train"] & all_images["val"]),
                # This pilot has no benchmark test split. Keep the full split
                # contract so existing qualification gates cannot mistake an
                # omitted field for an unverified overlap check.
                "train_test": 0,
                "val_test": 0,
            },
            "excluded_source_buckets": ["household", "refspatial"],
            "by_form": {split: dict(counter) for split, counter in forms.items()},
            "by_family": {split: dict(counter) for split, counter in families.items()},
            "semantic_audit": {
                "pack": str(pack),
                "pack_manifest_sha256": sha256_file(pack / "manifest.json"),
                "report": str(report_path),
                "report_sha256": report_sha,
                "audit_id": pack_manifest.get("audit_id"),
                "source_fingerprint": pack_manifest.get("source_fingerprint"),
                "verdict": report.get("verdict"),
            },
            "split_rule": {
                "image_disjoint": True,
                "seed": args.seed,
                "val_fraction": args.val_fraction,
            },
            "contract_version": CONTRACT_VERSION,
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.rename(out)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def materialize_reviewed_native(args: argparse.Namespace) -> dict:
    """Reuse resolved historical labels as native teacher candidates, preserving holdouts.

    A candidate pool is not a global generator PASS or a teacher-eligibility report.
    Upstream labels must themselves be resolved true and match the stored context.
    """
    from eval.compile_semantic_audit import compile_report, content_hash

    pack, output = args.pack.resolve(), args.out.resolve()
    if output.exists():
        raise ValueError(f"refusing existing native candidate output: {output}")
    if args.adjudication is None or args.preserve_split is None:
        raise ValueError("reviewed-native requires --adjudication and --preserve-split")
    report, _ = compile_report(argparse.Namespace(
        pack=pack, adjudication=args.adjudication.resolve(),
        min_accuracy=args.min_accuracy, min_agreement=args.min_agreement))
    if report.get("schema_version") != 3 or report.get("errors"):
        raise ValueError("historical review has integrity errors or is not schema v3")
    manifest = json.loads((pack / "manifest.json").read_text())
    split_manifest = json.loads(args.preserve_split.read_text())
    split_map = {}
    for split in ("train", "dev", "qualification"):
        for digest in split_manifest["image_split"][split + "_image_sha256"]:
            if digest in split_map:
                raise ValueError("preserved image splits overlap")
            split_map[digest] = split
    if set(split_map) != {item["image_sha256"] for item in manifest["selected"]}:
        raise ValueError("preserved splits must cover exactly the audited image pool")
    streams = [_latest_annotations(pack / f"annotator_{side}/annotations/annotations.jsonl")
               for side in ("a", "b")]
    adjudication = {row["sample_id"]: row for row in _read_jsonl(args.adjudication)}
    if len(adjudication) != len(_read_jsonl(args.adjudication)):
        raise ValueError("duplicate adjudications")
    packed = {}
    for path in sorted((pack / "annotator_a/labels").rglob("*.jsonl")):
        for row in _read_jsonl(path):
            if row["sample_id"] in packed:
                raise ValueError("duplicate packed label")
            packed[row["sample_id"]] = row
    dependencies = {"depth": ["detection"], "affordance": ["detection", "depth"],
                    "dof": ["detection", "depth", "affordance"],
                    "action_cot": ["detection", "depth", "affordance", "dof"]}
    partitions = {split: [] for split in ("train", "dev", "qualification")}
    exclusions = Counter()
    for item in manifest["selected"]:
        digest = item["image_sha256"]
        if image_sha256(item["image_path"]) != digest:
            raise ValueError("audited image binding drift")
        rows, accepted, resolutions = {}, {}, {}
        for form, sample_id in item["sample_ids"].items():
            row = packed[sample_id]
            expected_hash = item["label_hashes"][form]
            if (row.get("task_form") != form or row.get("image_path") != item["image_path"]
                    or content_hash(row) != expected_hash
                    or any(content_hash(s[sample_id]["label_snapshot"]) != expected_hash for s in streams)):
                raise ValueError(f"audited label binding drift: {sample_id}")
            left, right = (s[sample_id]["correct"] for s in streams)
            if not isinstance(left, bool) or not isinstance(right, bool):
                raise ValueError("historical decisions must be Boolean")
            decision = left if left == right else None
            resolution = "historical_human_agreement" if left == right else "unresolved"
            if left != right and sample_id in adjudication:
                adjudicated = adjudication[sample_id]
                if adjudicated.get("label_sha256") != expected_hash:
                    raise ValueError("adjudication label binding drift")
                decision = adjudicated["correct"]
                resolution = "historical_human_adjudication"
            valid, reason = validate(form, row["payload"], row.get("gsa_context"))
            accepted[form] = decision is True and valid
            if not accepted[form]:
                exclusions[f"{form}:" + ("unresolved" if decision is None else
                            "human_false" if not decision else "contract_" + reason)] += 1
            rows[form], resolutions[form] = row, resolution
        closed = {}
        for form in ("bbox", "detection", "mask", "point", "depth", "affordance", "dof", "action_cot"):
            row = rows[form]
            if not accepted[form]:
                continue
            deps = dependencies.get(form, [])
            if not all(closed.get(dep, False) for dep in deps):
                exclusions[f"{form}:unaccepted_upstream"] += 1
                continue
            context = row.get("gsa_context") or {}
            if deps:
                groundings = [{"object_class": entry["label"], "bbox": entry["bbox_2d"]}
                              for entry in rows["detection"]["payload"]]
                linked = context.get("groundings") == groundings
                for dep in deps[1:]:
                    key = "depth_order" if dep == "depth" else dep
                    linked = linked and context.get("spatial", {}).get(key) == rows[dep]["payload"]
                if not linked:
                    exclusions[f"{form}:upstream_payload_mismatch"] += 1
                    continue
            family = FORM_SPECS[form]["family"]
            delimiter = {"G": "<GROUND>", "S": "<RELATE>", "A": "<ACT>"}[family]
            instruction = training_instruction(form, context)
            # Native generation requests JSON only. Replace that output clause for
            # the staged teacher format instead of appending a contradictory rule.
            output_clause = " Return only a JSON array;"
            if instruction.count(output_clause) != 1:
                raise ValueError("native instruction output contract changed")
            instruction = instruction.replace(
                output_clause, f" Return {delimiter} followed by a JSON array;")
            sample = {
                "messages": [
                    {"role": "system", "content": "Return the requested native structured output with its stage delimiter."},
                    {"role": "user", "content": "<image>" + instruction},
                    {"role": "assistant", "content": delimiter + json.dumps(row["payload"], ensure_ascii=False)},
                ],
                "images": [item["image_path"]], "sample_id": row["sample_id"],
                "gsa_payload": row["payload"],
            }
            sample.update(task_columns(form, context, source=str(row.get("source") or row.get("dataset") or "unknown"),
                                       scene=str(row.get("scene_type") or row.get("scene") or "unknown"), image_id=digest))
            sample["gsa_meta"].update({
                "split": split_map[digest], "label_provenance": "human_semantic_audit",
                "review_decision": resolutions[form], "source_label_sha256": item["label_hashes"][form],
                "source_audit_only": bool(row.get("_audit_only")),
                "upstream_reviewed_sample_ids": [item["sample_ids"][dep] for dep in deps],
                "native_teacher_candidate": True, "teacher_eligible": False,
                "global_generator_audit_verdict": report["verdict"],
                "training_instruction_version": "anvil-native-reviewed-v2",
                "linked_causal_trace": False,
            })
            partitions[split_map[digest]].append(sample)
            closed[form] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".anvil-reviewed-native-", dir=output.parent) as temporary:
        staging = Path(temporary) / "corpus"
        staging.mkdir()
        bindings = {}
        for split, rows in partitions.items():
            rows.sort(key=lambda row: row["sample_id"])
            for family in (None, "G", "S", "A"):
                selected = rows if family is None else [row for row in rows if row["gsa_family"] == family]
                if not selected:
                    raise ValueError(f"no accepted native candidates for {split}/{family}")
                name = split + ("_" + family if family else "") + ".jsonl"
                (staging / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
                bindings[name] = {"rows": len(selected), "sha256": sha256_file(staging / name)}
        (staging / "audit_corpus.jsonl").write_bytes((staging / "train.jsonl").read_bytes() + (staging / "dev.jsonl").read_bytes())
        bindings["audit_corpus.jsonl"] = {"rows": len(partitions["train"]) + len(partitions["dev"]),
                                         "sha256": sha256_file(staging / "audit_corpus.jsonl")}
        sources = [pack / "manifest.json", args.preserve_split.resolve(), args.adjudication.resolve(),
                   *(pack / f"annotator_{side}/annotations/annotations.jsonl" for side in ("a", "b"))]
        result = {
            "schema": "anvil-reviewed-native-candidates-v1", "status": "PENDING_CONTENT_AUDITS",
            "training_eligible": False, "teacher_eligible": False, "benchmark_ground_truth": False,
            "global_generator_audit_verdict": report["verdict"],
            "unresolved_historical_items_excluded": report.get("unresolved_sample_ids", []),
            "files": bindings, "excluded_by_reason": dict(exclusions),
            "by_form": {split: dict(Counter(row["gsa_form"] for row in rows)) for split, rows in partitions.items()},
            "source_bindings": [{"path": str(p), "sha256": sha256_file(p)} for p in sources],
            "builder_sha256": sha256_file(Path(__file__)),
            "validator_sha256": sha256_file(Path(__file__).parent / "flywheel/validators.py"),
            "notes": ["Historical train/dev/qualification image assignments are unchanged.",
                      "Only resolved-true labels passing current strict contracts and exact upstream bindings are candidates.",
                      "Historical audit-only flags are preserved; current strict revalidation is required.",
                      "Oracle-context native teacher tasks do not establish causal G-to-S-to-A traces or specialist superiority."],
        }
        (staging / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        staging.rename(output)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=Path("data/semantic_audit_v3"))
    parser.add_argument("--source", type=Path, default=Path("data/viewer_v3_audit_ready_v34"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/audits/pilot_semantic_audit_v3.json"))
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--out", type=Path, default=Path("trains/gsa_bench_v3_pilot_clean"))
    parser.add_argument(
        "--scope", choices=("audited", "full"), default="audited",
        help=("audited emits only the fixed 40-image human-reviewed pool; "
              "full additionally emits unreviewed rows that pass strict validators"),
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    parser.add_argument("--min-agreement", type=float, default=0.80)
    parser.add_argument("--reviewed-native", action="store_true",
                        help="Prepare only resolved historical native teacher candidates; no global audit promotion")
    parser.add_argument("--preserve-split", type=Path,
                        help="Existing G-family manifest whose train/dev/qualification image assignments are retained")
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1")
    try:
        if args.reviewed_native:
            materialize_reviewed_native(args)
        else:
            materialize(args)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
