#!/usr/bin/env python3
"""Fail-closed preflight for the schema-v3 G-only qualification corpus."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    SCOPED_PASS,
    content_hash,
    file_hash,
    load_json,
    same_path,
)
from trains.flywheel.contracts import (  # noqa: E402
    CONTRACT_VERSION,
    image_sha256,
    validate_task_columns,
)
from trains.flywheel.validators import validate  # noqa: E402
from trains.materialize_g_family_scoped_v3 import (  # noqa: E402
    LABEL_POLICY,
    SPLIT_POLICY,
    _final_decisions,
    _fresh_scoped_report,
    _pack_index,
    _read_jsonl,
)
from trains.materialize_semantic_audit_v3 import ASK, SYSTEM_PROMPT  # noqa: E402


def _critical_meta(
    row: dict[str, Any],
    entry: dict[str, Any],
    split: str,
    report: dict[str, Any],
    report_sha: str,
    split_sha: str,
) -> None:
    sample_id = row.get("sample_id")
    meta = row.get("gsa_meta")
    if not isinstance(meta, dict):
        raise ValueError(f"row lacks gsa_meta: {sample_id}")
    expected = {
        "form": entry["form"],
        "family": "G",
        "image_sha256": entry["image_sha256"],
        "contract_version": CONTRACT_VERSION,
        "split": split,
        "label_provenance": "human_semantic_audit_g_family_scoped",
        "audit_reviewed": True,
        "source_audit_only": False,
        "release_protocol": PROTOCOL,
        "release_scope": "G_family_only",
        "authorization": AUTHORIZATION,
        "g_family_scoped_release_verdict": SCOPED_PASS,
        "semantic_audit_global_verdict": "BLOCKED",
        "g_family_scoped_report_sha256": report_sha,
        "g_family_compiler_inputs_sha256": report["compiler_inputs_sha256"],
        "source_label_sha256": entry["label_sha256"],
        "source_sample_id": sample_id,
        "image_split_sha256": split_sha,
        "label_version": entry["row"].get("label_version"),
        "prompt_version": entry["row"].get("prompt_version"),
    }
    for field, value in expected.items():
        if meta.get(field) != value:
            raise ValueError(f"row {sample_id} has invalid scoped metadata {field!r}")


def _expected_accepted(
    pack: Path,
    adjudication: Path,
    entries: dict[str, dict[str, Any]],
) -> tuple[set[str], dict[str, str], Counter[str]]:
    decisions, provenance = _final_decisions(pack, adjudication, entries)
    accepted: set[str] = set()
    excluded: Counter[str] = Counter()
    for sample_id, entry in entries.items():
        row = entry["row"]
        if decisions[sample_id] is not True:
            excluded["human_false"] += 1
            continue
        if row.get("_audit_only") is True:
            excluded["audit_only"] += 1
            continue
        valid, reason = validate(row["task_form"], row.get("payload"), row.get("gsa_context"))
        if valid:
            accepted.add(sample_id)
        else:
            excluded[f"strict_invalid:{row['task_form']}:{reason}"] += 1
    return accepted, provenance, excluded


def _verify_rows(
    corpus_dir: Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
    report_sha: str,
    entries: dict[str, dict[str, Any]],
    expected_ids: set[str],
    provenance: dict[str, str],
) -> None:
    image_split = manifest.get("image_split")
    if not isinstance(image_split, dict):
        raise ValueError("corpus manifest lacks image_split binding")
    split_payload = dict(image_split)
    declared_split_sha = split_payload.pop("sha256", None)
    if image_split.get("policy") != SPLIT_POLICY or content_hash(split_payload) != declared_split_sha:
        raise ValueError("corpus image_split hash/policy drift")
    split_images = {
        "train": set(image_split.get("train_image_sha256") or []),
        "val": set(image_split.get("val_image_sha256") or []),
    }
    if not split_images["train"] or not split_images["val"]:
        raise ValueError("G-only train/val image split is empty")
    if split_images["train"] & split_images["val"]:
        raise ValueError("G-only train/val image overlap is non-zero")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    actual_images: dict[str, set[str]] = {"train": set(), "val": set()}
    actual_by_form: dict[str, Counter[str]] = {}
    for split in ("train", "val"):
        key = f"{split}_G_only"
        binding = (manifest.get("corpus_files") or {}).get(key)
        if not isinstance(binding, dict) or binding.get("path") != f"{key}.jsonl":
            raise ValueError(f"invalid corpus file binding for {key}")
        path = corpus_dir / binding["path"]
        if not path.is_file() or file_hash(path) != binding.get("sha256"):
            raise ValueError(f"corpus file hash drift: {path}")
        rows = _read_jsonl(path)
        if binding.get("records") != len(rows) or not rows:
            raise ValueError(f"corpus file record count mismatch: {path}")
        split_rows[split] = rows
        actual_by_form[split] = Counter()
        for row in rows:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in all_ids:
                raise ValueError(f"duplicate or invalid released sample_id: {sample_id!r}")
            entry = entries.get(sample_id)
            if entry is None or sample_id not in expected_ids:
                raise ValueError(f"unapproved sample entered G-only corpus: {sample_id}")
            source = entry["row"]
            if row.get("gsa_form") not in G_FORMS or row.get("gsa_family") != "G":
                raise ValueError(f"non-G row entered scoped corpus: {sample_id}")
            ok, reason = validate_task_columns(row)
            if not ok:
                raise ValueError(f"contract-invalid released row {sample_id}: {reason}")
            if row.get("gsa_payload") != source.get("payload") or row.get("gsa_context") != source.get("gsa_context"):
                raise ValueError(f"released payload/context drift: {sample_id}")
            expected_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "<image>" + ASK[row["gsa_form"]]},
                {
                    "role": "assistant",
                    "content": json.dumps(source["payload"], ensure_ascii=False),
                },
            ]
            if row.get("messages") != expected_messages:
                raise ValueError(f"released training messages drift: {sample_id}")
            valid, reason = validate(row["gsa_form"], row.get("gsa_payload"), row.get("gsa_context"))
            if not valid:
                raise ValueError(f"strict-invalid released row {sample_id}: {reason}")
            images = row.get("images")
            expected_image = str(Path(source["image_path"]).resolve())
            if images != [expected_image]:
                raise ValueError(f"released image path drift: {sample_id}")
            actual_image_sha = image_sha256(expected_image)
            if actual_image_sha != entry["image_sha256"]:
                raise ValueError(f"released image content drift: {sample_id}")
            _critical_meta(
                row,
                entry,
                split,
                report,
                report_sha,
                declared_split_sha,
            )
            if row["gsa_meta"].get("review_decision") != provenance[sample_id]:
                raise ValueError(f"released review-decision provenance drift: {sample_id}")
            if actual_image_sha not in split_images[split]:
                raise ValueError(f"released row is assigned to wrong image split: {sample_id}")
            all_ids.add(sample_id)
            actual_images[split].add(actual_image_sha)
            actual_by_form[split][row["gsa_form"]] += 1

    if all_ids != expected_ids:
        missing = sorted(expected_ids - all_ids)
        extra = sorted(all_ids - expected_ids)
        raise ValueError(f"released sample set drift: missing={missing[:3]} extra={extra[:3]}")
    if content_hash(sorted(all_ids)) != manifest.get("accepted_sample_ids_sha256"):
        raise ValueError("accepted sample-id set hash drift")
    if actual_images != split_images:
        raise ValueError("released row images differ from the bound image split")
    if manifest.get("counts") != {
        split: len(rows) for split, rows in split_rows.items()
    }:
        raise ValueError("manifest split counts drift")
    if manifest.get("by_form") != {
        split: dict(sorted(counts.items())) for split, counts in actual_by_form.items()
    }:
        raise ValueError("manifest form counts drift")
    if any(set(counts) != set(G_FORMS) for counts in actual_by_form.values()):
        raise ValueError("each G-only split must cover all four G forms")
    if manifest.get("unique_images") != {
        split: len(images) for split, images in actual_images.items()
    }:
        raise ValueError("manifest unique-image counts drift")
    if manifest.get("overlap") != {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }:
        raise ValueError("manifest overlap contract drift")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    pack = args.pack.resolve()
    adjudication = args.adjudication.resolve()
    full_report = args.full_report.resolve()
    scoped_report_path = args.scoped_report.resolve()
    manifest_path = args.manifest.resolve()
    corpus_dir = manifest_path.parent
    report, report_sha = _fresh_scoped_report(
        scoped_report_path,
        pack,
        adjudication,
        full_report,
        args.min_accuracy,
        args.min_agreement,
    )
    if report.get("global_audit_verdict") != "BLOCKED" or report.get("global_blocking_forms") != ["depth"]:
        raise ValueError("G-only release requires depth to be the sole global blocking form")
    manifest = load_json(manifest_path, "G-only corpus manifest")
    expected_header = {
        "protocol": PROTOCOL,
        "corpus_version": CORPUS_VERSION,
        "release_scope": "G_family_only",
        "family": "G",
        "forms": list(G_FORMS),
        "authorization": AUTHORIZATION,
        "forbidden_uses": FORBIDDEN_USES,
        "g_family_scoped_release_verdict": SCOPED_PASS,
        "global_audit_verdict": "BLOCKED",
        "label_policy": LABEL_POLICY,
    }
    for field, value in expected_header.items():
        if manifest.get(field) != value:
            raise ValueError(f"G-only corpus manifest mismatch in {field!r}")
    if manifest.get("audit_universe_records") != 160:
        raise ValueError("G-only corpus audit universe must be exactly 160 records")

    bindings = manifest.get("audit_bindings")
    if not isinstance(bindings, dict) or not same_path(bindings.get("pack"), pack):
        raise ValueError("G-only corpus is bound to a different audit pack")
    scoped_binding = bindings.get("scoped_report") or {}
    if (
        not same_path(scoped_binding.get("path"), scoped_report_path)
        or scoped_binding.get("sha256") != report_sha
    ):
        raise ValueError("G-only corpus/scoped-report binding drift")
    if bindings.get("full_compiler_report") != report["input_bindings"]["full_compiler_report"]:
        raise ValueError("G-only corpus/full-compiler binding drift")
    if bindings.get("inputs") != report.get("input_bindings"):
        raise ValueError("G-only corpus audit-input bindings drift")
    if bindings.get("compiler_inputs_sha256") != report.get("compiler_inputs_sha256"):
        raise ValueError("G-only corpus compiler-input hash drift")
    if bindings.get("audit_image_set_sha256") != report.get("audit_image_set_sha256"):
        raise ValueError("G-only corpus audit image-set hash drift")

    for name, binding in report["input_bindings"].items():
        path = Path(binding["path"])
        if not path.is_file() or file_hash(path) != binding.get("sha256"):
            raise ValueError(f"bound audit input drift: {name}")
    pack_manifest = load_json(pack / "manifest.json", "schema-v3 manifest")
    entries, _ = _pack_index(pack, pack_manifest)
    expected_ids, provenance, excluded = _expected_accepted(pack, adjudication, entries)
    if manifest.get("accepted_records") != len(expected_ids):
        raise ValueError("manifest accepted-record count differs from release policy")
    if manifest.get("excluded_records") != 160 - len(expected_ids):
        raise ValueError("manifest excluded-record count differs from release policy")
    if manifest.get("excluded_by_reason") != dict(sorted(excluded.items())):
        raise ValueError("manifest exclusion reasons differ from release policy")
    _verify_rows(
        corpus_dir,
        manifest,
        report,
        report_sha,
        entries,
        expected_ids,
        provenance,
    )
    forbidden_files = (
        "train.jsonl",
        "val.jsonl",
        "train_S.jsonl",
        "val_S.jsonl",
        "train_A.jsonl",
        "val_A.jsonl",
    )
    present = [name for name in forbidden_files if (corpus_dir / name).exists()]
    if present:
        raise ValueError(f"generic/joint or non-G slices present in G-only corpus: {present}")
    return {
        "verdict": SCOPED_PASS,
        "protocol": PROTOCOL,
        "authorization": AUTHORIZATION,
        "global_audit_verdict": "BLOCKED",
        "global_blocking_forms": ["depth"],
        "corpus": str(corpus_dir),
        "records": len(expected_ids),
        "train_records": manifest["counts"]["train"],
        "val_records": manifest["counts"]["val"],
        "image_split_sha256": manifest["image_split"]["sha256"],
        "compiler_inputs_sha256": report["compiler_inputs_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scoped-report", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    parser.add_argument("--min-agreement", type=float, default=0.80)
    args = parser.parse_args()
    try:
        result = verify(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
