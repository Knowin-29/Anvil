#!/usr/bin/env python3
"""Materialize the fail-closed schema-v3 G-only qualification corpus.

Only independently reviewed bbox, detection, mask, and point labels with a
final human-true decision, a strict validator PASS, and no ``_audit_only`` flag
are released.  The output deliberately has no generic joint-training slice.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.compile_g_family_scoped_release import (  # noqa: E402
    AUTHORIZATION,
    CORPUS_VERSION,
    FORBIDDEN_USES,
    G_FORMS,
    PROTOCOL,
    REVIEWER_C_KEY,
    SCOPED_PASS,
    canonical_json,
    compile_scoped,
    content_hash,
    file_hash,
    load_json,
    same_path,
)
from eval.compile_semantic_audit import content_hash as label_content_hash  # noqa: E402
from trains.flywheel.contracts import (  # noqa: E402
    CONTRACT_VERSION,
    image_sha256,
    task_columns,
)
from trains.flywheel.validators import validate  # noqa: E402
from trains.materialize_semantic_audit_v3 import ASK, SYSTEM_PROMPT  # noqa: E402

SPLIT_POLICY = "image-sha256-disjoint-g-only-v1"
LABEL_POLICY = "final_human_true_and_strict_validator"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing JSONL input: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number} ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSON at {path}:{line_number}")
        rows.append(row)
    return rows


def _latest_annotations(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        snapshot = row.get("label_snapshot")
        sample_id = snapshot.get("sample_id") if isinstance(snapshot, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"annotation lacks label_snapshot.sample_id: {path}")
        latest[sample_id] = row
    return latest


def _reviewer_c_decisions(path: Path) -> dict[str, bool]:
    decisions: dict[str, bool] = {}
    for row in _read_jsonl(path):
        sample_id = row.get("sample_id")
        correct = row.get("correct")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or not isinstance(correct, bool)
            or row.get("reviewer_key") != REVIEWER_C_KEY
        ):
            raise ValueError(f"invalid reviewer-C adjudication row: {sample_id!r}")
        if sample_id in decisions:
            raise ValueError(f"duplicate reviewer-C adjudication: {sample_id}")
        decisions[sample_id] = correct
    if not decisions:
        raise ValueError("reviewer-C adjudication is empty")
    return decisions


def _packed_labels(pack: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    root = pack / "annotator_a/labels"
    if not root.is_dir():
        raise ValueError(f"missing packed labels directory: {root}")
    for path in sorted(root.rglob("*.jsonl")):
        for row in _read_jsonl(path):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in labels:
                raise ValueError(f"invalid or duplicate packed sample_id: {sample_id!r}")
            labels[sample_id] = row
    return labels


def _fresh_scoped_report(
    scoped_report_path: Path,
    pack: Path,
    adjudication: Path,
    full_report: Path,
    min_accuracy: float,
    min_agreement: float,
) -> tuple[dict[str, Any], str]:
    supplied = load_json(scoped_report_path, "G-only scoped compiler report")
    recomputed, rc = compile_scoped(
        argparse.Namespace(
            pack=pack,
            adjudication=adjudication,
            full_report=full_report,
            min_accuracy=min_accuracy,
            min_agreement=min_agreement,
        )
    )
    if rc != 0 or recomputed.get("verdict") != SCOPED_PASS:
        raise ValueError("recomputed G-only release gate is not G_ONLY_PASS")
    if supplied != recomputed:
        raise ValueError("G-only scoped compiler report is stale or inconsistent")
    if supplied.get("protocol") != PROTOCOL:
        raise ValueError("G-only scoped compiler protocol mismatch")
    if supplied.get("authorization") != AUTHORIZATION:
        raise ValueError("G-only scoped compiler authorization mismatch")
    return supplied, file_hash(scoped_report_path)


def _pack_index(
    pack: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    labels = _packed_labels(pack)
    entries: dict[str, dict[str, Any]] = {}
    image_by_sample: dict[str, str] = {}
    selected = manifest.get("selected")
    if not isinstance(selected, list) or len(selected) != 40:
        raise ValueError("G-only materialization requires 40 selected v3 images")
    for item in selected:
        if not isinstance(item, dict):
            raise ValueError("invalid selected item in v3 manifest")
        image_path = item.get("image_path")
        image_digest = item.get("image_sha256")
        if not isinstance(image_path, str) or not isinstance(image_digest, str):
            raise ValueError("v3 selected item lacks image path/hash")
        if image_sha256(image_path) != image_digest:
            raise ValueError(f"v3 selected image hash drift: {image_path}")
        sample_ids = item.get("sample_ids") or {}
        label_hashes = item.get("label_hashes") or {}
        for form in G_FORMS:
            sample_id = sample_ids.get(form)
            if not isinstance(sample_id, str) or sample_id in entries:
                raise ValueError(f"invalid or duplicate {form} sample mapping: {sample_id!r}")
            row = labels.get(sample_id)
            if (
                row is None
                or row.get("task_form") != form
                or row.get("image_path") != image_path
                or label_content_hash(row) != label_hashes.get(form)
            ):
                raise ValueError(f"packed label/manifest drift for {sample_id}")
            entries[sample_id] = {
                "row": row,
                "form": form,
                "label_sha256": label_hashes[form],
                "image_sha256": image_digest,
            }
            image_by_sample[sample_id] = image_digest
    if len(entries) != 40 * len(G_FORMS):
        raise ValueError(f"expected 160 G audit labels, found {len(entries)}")
    return entries, image_by_sample


def _final_decisions(
    pack: Path,
    adjudication_path: Path,
    entries: dict[str, dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, str]]:
    a = _latest_annotations(pack / "annotator_a/annotations/annotations.jsonl")
    b = _latest_annotations(pack / "annotator_b/annotations/annotations.jsonl")
    reviewer_c = _reviewer_c_decisions(adjudication_path)
    decisions: dict[str, bool] = {}
    provenance: dict[str, str] = {}
    for sample_id in entries:
        va = a.get(sample_id, {}).get("correct")
        vb = b.get(sample_id, {}).get("correct")
        if not isinstance(va, bool) or not isinstance(vb, bool):
            raise ValueError(f"missing paired Boolean A/B votes for {sample_id}")
        if va == vb:
            decisions[sample_id] = va
            provenance[sample_id] = "a_b_agreement"
        elif sample_id in reviewer_c:
            decisions[sample_id] = reviewer_c[sample_id]
            provenance[sample_id] = "reviewer_c_adjudication"
        else:
            raise ValueError(f"unresolved G-family disagreement: {sample_id}")
    return decisions, provenance


def _make_swift_row(
    row: dict[str, Any],
    split: str,
    decision_provenance: str,
    source_label_sha256: str,
    image_digest: str,
    scoped_report: dict[str, Any],
    scoped_report_sha256: str,
) -> dict[str, Any]:
    form = row["task_form"]
    payload = row["payload"]
    context = row["gsa_context"]
    source = str(row.get("source") or row.get("dataset") or "unknown")
    scene = str(row.get("scene_type") or row.get("scene") or "embodied_home")
    out: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<image>" + ASK[form]},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "images": [str(Path(row["image_path"]).resolve())],
        "sample_id": row["sample_id"],
        "gsa_payload": payload,
    }
    out.update(task_columns(form, context, source=source, scene=scene, image_id=image_digest))
    out["gsa_meta"].update(
        {
            "split": split,
            "label_provenance": "human_semantic_audit_g_family_scoped",
            "audit_reviewed": True,
            "source_audit_only": False,
            "review_decision": decision_provenance,
            "release_protocol": PROTOCOL,
            "release_scope": "G_family_only",
            "authorization": AUTHORIZATION,
            "g_family_scoped_release_verdict": SCOPED_PASS,
            "semantic_audit_global_verdict": scoped_report["global_audit_verdict"],
            "g_family_scoped_report_sha256": scoped_report_sha256,
            "g_family_compiler_inputs_sha256": scoped_report["compiler_inputs_sha256"],
            "source_label_sha256": source_label_sha256,
            "source_sample_id": row["sample_id"],
            "label_version": row.get("label_version"),
            "prompt_version": row.get("prompt_version"),
        }
    )
    return out


def _split_rows(
    rows: list[dict[str, Any]],
    seed: int,
    val_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[row["gsa_meta"]["image_sha256"]].append(row)
    if len(by_image) < 2:
        raise ValueError("G-only corpus needs at least two unique images")
    image_keys = sorted(by_image)
    target = max(1, min(len(image_keys) - 1, round(len(image_keys) * val_fraction)))
    ranked = sorted(
        image_keys,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    providers = {
        form: {key for key, values in by_image.items() if any(row["gsa_form"] == form for row in values)}
        for form in G_FORMS
    }
    if any(len(values) < 2 for values in providers.values()):
        raise ValueError("every G form needs at least two accepted images for disjoint splits")

    val_images: set[str] = set()
    for form in G_FORMS:
        if val_images & providers[form]:
            continue
        candidate = next(
            (
                key
                for key in ranked
                if key in providers[form]
                and all(len(source - (val_images | {key})) >= 1 for source in providers.values())
            ),
            None,
        )
        if candidate is None:
            raise ValueError(f"cannot create an image-disjoint val split covering {form}")
        val_images.add(candidate)
    for key in ranked:
        if len(val_images) >= target:
            break
        if key in val_images:
            continue
        if all(len(source - (val_images | {key})) >= 1 for source in providers.values()):
            val_images.add(key)
    train_images = set(image_keys) - val_images

    def covered(keys: set[str]) -> set[str]:
        return {
            row["gsa_form"]
            for key in keys
            for row in by_image[key]
        }

    if covered(train_images) != set(G_FORMS) or covered(val_images) != set(G_FORMS):
        raise ValueError("train/val split does not cover every G form")
    train = [row for key in sorted(train_images) for row in by_image[key]]
    val = [row for key in sorted(val_images) for row in by_image[key]]
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(val)
    split_binding = {
        "policy": SPLIT_POLICY,
        "seed": seed,
        "val_fraction": val_fraction,
        "train_image_sha256": sorted(train_images),
        "val_image_sha256": sorted(val_images),
    }
    split_binding["sha256"] = content_hash(split_binding)
    return train, val, split_binding


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    pack = args.pack.resolve()
    adjudication = args.adjudication.resolve()
    full_report_path = args.full_report.resolve()
    scoped_report_path = args.scoped_report.resolve()
    scoped_report, scoped_report_sha = _fresh_scoped_report(
        scoped_report_path,
        pack,
        adjudication,
        full_report_path,
        args.min_accuracy,
        args.min_agreement,
    )
    manifest_path = pack / "manifest.json"
    pack_manifest = load_json(manifest_path, "schema-v3 manifest")
    if pack_manifest.get("schema_version") != 3:
        raise ValueError("G-only materialization requires a schema-v3 pack")
    entries, _ = _pack_index(pack, pack_manifest)
    decisions, decision_provenance = _final_decisions(pack, adjudication, entries)

    accepted: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for sample_id in sorted(entries):
        entry = entries[sample_id]
        row = entry["row"]
        if decisions[sample_id] is not True:
            excluded["human_false"] += 1
            continue
        if row.get("_audit_only") is True:
            excluded["audit_only"] += 1
            continue
        valid, reason = validate(row["task_form"], row.get("payload"), row.get("gsa_context"))
        if not valid:
            excluded[f"strict_invalid:{row['task_form']}:{reason}"] += 1
            continue
        accepted.append(
            _make_swift_row(
                row,
                "pending",
                decision_provenance[sample_id],
                entry["label_sha256"],
                entry["image_sha256"],
                scoped_report,
                scoped_report_sha,
            )
        )
    if not accepted:
        raise ValueError("no G rows survived final-human-true + strict validation")
    if {row["gsa_form"] for row in accepted} != set(G_FORMS):
        raise ValueError("accepted G-only corpus does not cover every G form")

    train, val, image_split = _split_rows(accepted, args.seed, args.val_fraction)
    for row in train:
        row["gsa_meta"]["split"] = "train"
        row["gsa_meta"]["image_split_sha256"] = image_split["sha256"]
    for row in val:
        row["gsa_meta"]["split"] = "val"
        row["gsa_meta"]["image_split_sha256"] = image_split["sha256"]

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=str(out.parent)))
    try:
        train_path = tmp / "train_G_only.jsonl"
        val_path = tmp / "val_G_only.jsonl"
        _write_jsonl(train_path, train)
        _write_jsonl(val_path, val)
        corpus_files = {
            "train_G_only": {
                "path": "train_G_only.jsonl",
                "sha256": file_hash(train_path),
                "records": len(train),
            },
            "val_G_only": {
                "path": "val_G_only.jsonl",
                "sha256": file_hash(val_path),
                "records": len(val),
            },
        }
        by_form = {
            split: dict(sorted(Counter(row["gsa_form"] for row in rows).items()))
            for split, rows in (("train", train), ("val", val))
        }
        full_report_binding = scoped_report["input_bindings"]["full_compiler_report"]
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "corpus_version": CORPUS_VERSION,
            "release_scope": "G_family_only",
            "family": "G",
            "forms": list(G_FORMS),
            "authorization": AUTHORIZATION,
            "forbidden_uses": FORBIDDEN_USES,
            "g_family_scoped_release_verdict": SCOPED_PASS,
            "global_audit_verdict": scoped_report["global_audit_verdict"],
            "label_policy": LABEL_POLICY,
            "contract_version": CONTRACT_VERSION,
            "audit_universe_records": 40 * len(G_FORMS),
            "accepted_records": len(accepted),
            "excluded_records": sum(excluded.values()),
            "excluded_by_reason": dict(sorted(excluded.items())),
            "counts": {"train": len(train), "val": len(val)},
            "by_form": by_form,
            "unique_images": {
                "train": len(image_split["train_image_sha256"]),
                "val": len(image_split["val_image_sha256"]),
            },
            "overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
            "image_split": image_split,
            "accepted_sample_ids_sha256": content_hash(
                sorted(row["sample_id"] for row in accepted)
            ),
            "corpus_files": corpus_files,
            "audit_bindings": {
                "pack": str(pack),
                "scoped_report": {
                    "path": str(scoped_report_path),
                    "sha256": scoped_report_sha,
                },
                "full_compiler_report": full_report_binding,
                "compiler_inputs_sha256": scoped_report["compiler_inputs_sha256"],
                "audit_image_set_sha256": scoped_report["audit_image_set_sha256"],
                "inputs": scoped_report["input_bindings"],
            },
            "notes": (
                "This corpus is a G-family-only qualification release. The global "
                "semantic audit remains BLOCKED; this corpus must not be used as a "
                "joint or multi-teacher distillation release."
            ),
        }
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.rename(out)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("data/semantic_audit_v3"))
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=Path("data/semantic_audit_v3/adjudication.jsonl"),
    )
    parser.add_argument(
        "--full-report",
        type=Path,
        default=Path("artifacts/audits/pilot_semantic_audit_v3_adjudicated.json"),
    )
    parser.add_argument(
        "--scoped-report",
        type=Path,
        default=Path("artifacts/audits/pilot_semantic_audit_v3_g_family_scoped_v1.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("trains/gsa_bench_v3_g_family_scoped_v1"),
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    parser.add_argument("--min-agreement", type=float, default=0.80)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1")
    try:
        materialize(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
