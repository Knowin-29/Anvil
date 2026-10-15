#!/usr/bin/env python3
"""Fail-closed verifier for the composite semantic-audit pilot corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.compile_semantic_composite_gate import (  # noqa: E402
    COMPOSITE_CORPUS_VERSIONS,
    COMPOSITE_PROTOCOLS,
    compile_composite,
)
from trains.flywheel.contracts import FORM_SPECS, image_sha256, validate_task_columns  # noqa: E402
from trains.flywheel.validators import validate_structural  # noqa: E402

FORMS = {
    "action_cot", "affordance", "bbox", "depth",
    "detection", "dof", "mask", "point",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {path} ({exc.msg})") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing corpus split: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number} ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(row)
    return rows


def same_path(left: Any, right: Path) -> bool:
    return isinstance(left, str) and Path(left).resolve() == right.resolve()


def verify_component_bindings(
    report: dict[str, Any],
    semantic: dict[str, Any],
    packs: dict[str, Path],
    adjudications: dict[str, Path],
) -> None:
    declared_packs = report.get("component_packs") or {}
    corpus_packs = semantic.get("packs") or {}
    report_manifests = report.get("component_manifests") or {}
    corpus_manifests = semantic.get("component_manifests") or {}
    corpus_pack_hashes = semantic.get("pack_manifest_sha256") or {}
    corpus_adjudications = semantic.get("component_adjudications") or {}
    corpus_audit_ids = semantic.get("audit_ids") or {}
    for component in ("v3", "v4"):
        pack = packs[component]
        manifest_path = pack / "manifest.json"
        manifest = load_json(manifest_path, f"{component} pack manifest")
        manifest_sha = file_hash(manifest_path)
        if not same_path(declared_packs.get(component), pack):
            raise ValueError(f"composite report is bound to a different {component} pack")
        if not same_path(corpus_packs.get(component), pack):
            raise ValueError(f"corpus is bound to a different {component} pack")
        report_binding = report_manifests.get(component) or {}
        if not same_path(report_binding.get("path"), manifest_path):
            raise ValueError(f"composite report {component} manifest path mismatch")
        if report_binding.get("sha256") != manifest_sha:
            raise ValueError(f"composite report {component} manifest hash drift")
        if corpus_manifests.get(component) != report_binding:
            raise ValueError(f"corpus {component} manifest binding differs from report")
        if corpus_pack_hashes.get(component) != manifest_sha:
            raise ValueError(f"corpus {component} pack manifest hash drift")
        if corpus_audit_ids.get(component) != manifest.get("audit_id"):
            raise ValueError(f"corpus {component} audit_id mismatch")

        adjudication = adjudications[component]
        if not adjudication.is_file():
            raise ValueError(f"missing {component} adjudication: {adjudication}")
        adjudication_binding = corpus_adjudications.get(component) or {}
        if not same_path(adjudication_binding.get("path"), adjudication):
            raise ValueError(f"corpus {component} adjudication path mismatch")
        if adjudication_binding.get("sha256") != file_hash(adjudication):
            raise ValueError(f"corpus {component} adjudication hash drift")


def verify_rows(
    corpus_dir: Path,
    manifest: dict[str, Any],
    audit_sha: str,
    report: dict[str, Any],
) -> None:
    split_rows = {
        split: read_jsonl(corpus_dir / f"{split}.jsonl")
        for split in ("train", "val")
    }
    expected_counts = manifest.get("counts") or {}
    for split, rows in split_rows.items():
        if expected_counts.get(split) != len(rows) or not rows:
            raise ValueError(f"{split} count mismatch or empty split")

    sample_ids: set[str] = set()
    image_forms: set[tuple[str, str]] = set()
    images: dict[str, set[str]] = {"train": set(), "val": set()}
    by_form: dict[str, Counter[str]] = {}
    by_family: dict[str, Counter[str]] = {}
    component_hashes = report["component_report_hashes"]
    composite_hash = report["composite_inputs_sha256"]
    for split, rows in split_rows.items():
        by_form[split] = Counter()
        by_family[split] = Counter()
        for row in rows:
            ok, reason = validate_task_columns(row)
            if not ok:
                raise ValueError(f"contract-invalid released row: {reason}")
            sample_id = row.get("sample_id")
            form = row.get("gsa_form")
            family = row.get("gsa_family")
            meta = row.get("gsa_meta") or {}
            if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
                raise ValueError(f"duplicate or empty released sample_id: {sample_id!r}")
            if form not in FORMS or FORM_SPECS[form]["family"] != family:
                raise ValueError(f"released row has invalid form/family: {form}/{family}")
            if meta.get("split") != split:
                raise ValueError(f"released row split mismatch: {sample_id}")
            if (
                meta.get("label_provenance") != "human_semantic_audit"
                or meta.get("audit_reviewed") is not True
                or meta.get("source_audit_only") is not False
            ):
                raise ValueError(f"released row lacks human-only provenance: {sample_id}")
            if row.get("_audit_only") is True:
                raise ValueError(f"audit-only row entered released corpus: {sample_id}")
            component = "v4" if form == "depth" else "v3"
            if meta.get("semantic_audit_component") != component:
                raise ValueError(f"wrong audit component for {sample_id}")
            if component == "v4" and meta.get("depth_scope") != report.get("depth_scope"):
                raise ValueError(f"depth scope mismatch in released row {sample_id}")
            if meta.get("semantic_audit_report_sha256") != audit_sha:
                raise ValueError(f"composite report hash mismatch in row {sample_id}")
            if meta.get("semantic_audit_composite_inputs_sha256") != composite_hash:
                raise ValueError(f"composite input hash mismatch in row {sample_id}")
            if meta.get("semantic_audit_component_report_sha256") != component_hashes[
                f"{component}_report_sha256"
            ]:
                raise ValueError(f"component report hash mismatch in row {sample_id}")
            image_paths = row.get("images")
            if not isinstance(image_paths, list) or len(image_paths) != 1:
                raise ValueError(f"released row must contain exactly one image: {sample_id}")
            image_path = Path(str(image_paths[0]))
            if not image_path.is_file():
                raise ValueError(f"released row image is missing: {image_path}")
            image_sha = image_sha256(str(image_path))
            if meta.get("image_sha256") != image_sha:
                raise ValueError(f"released row image hash drift: {sample_id}")
            key = (image_sha, form)
            if key in image_forms:
                raise ValueError(f"duplicate released image/form pair: {key}")
            payload = row.get("gsa_payload")
            valid, structural_reason = validate_structural(form, payload, row.get("gsa_context"))
            if not valid:
                raise ValueError(f"released row is structurally invalid: {sample_id} ({structural_reason})")
            sample_ids.add(sample_id)
            image_forms.add(key)
            images[split].add(image_sha)
            by_form[split][form] += 1
            by_family[split][family] += 1

    if images["train"] & images["val"]:
        raise ValueError("train/val image overlap is non-zero")
    if any(set(by_form[split]) != FORMS for split in ("train", "val")):
        raise ValueError("each split must cover all eight forms")
    if manifest.get("accepted_records") != len(sample_ids):
        raise ValueError("accepted_records does not match released rows")
    expected_overlap = {"train_val": 0, "train_test": 0, "val_test": 0}
    if manifest.get("overlap") != expected_overlap:
        raise ValueError(f"corpus overlap contract mismatch: {manifest.get('overlap')}")
    if manifest.get("by_form") != {
        split: dict(counter) for split, counter in by_form.items()
    }:
        raise ValueError("manifest by_form counts do not match released rows")
    if manifest.get("by_family") != {
        split: dict(counter) for split, counter in by_family.items()
    }:
        raise ValueError("manifest by_family counts do not match released rows")
    if manifest.get("unique_images") != {
        split: len(values) for split, values in images.items()
    }:
        raise ValueError("manifest unique_images counts do not match released rows")

    for split, rows in split_rows.items():
        for family in ("G", "S", "A"):
            family_rows = read_jsonl(corpus_dir / f"{split}_{family}.jsonl")
            expected_ids = {
                row["sample_id"] for row in rows if row["gsa_family"] == family
            }
            actual_ids = {row.get("sample_id") for row in family_rows}
            if actual_ids != expected_ids or len(actual_ids) != len(family_rows):
                raise ValueError(f"{split}_{family} slice differs from canonical split")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = args.audit.resolve()
    manifest_path = args.manifest.resolve()
    packs = {"v3": args.v3_pack.resolve(), "v4": args.v4_pack.resolve()}
    adjudications = {
        "v3": args.v3_adjudication.resolve(),
        "v4": args.v4_adjudication.resolve(),
    }
    report = load_json(audit_path, "composite semantic audit report")
    depth_scope = report.get("depth_scope") or "total_order"
    expected_protocol = COMPOSITE_PROTOCOLS.get(depth_scope)
    expected_corpus_version = COMPOSITE_CORPUS_VERSIONS.get(depth_scope)
    if report.get("protocol") != expected_protocol or report.get("verdict") != "PASS":
        raise ValueError(f"required composite PASS is absent: protocol={report.get('protocol')!r}, verdict={report.get('verdict')!r}")
    declared_packs = report.get("component_packs") or {}
    if not same_path(declared_packs.get("v3"), packs["v3"]):
        raise ValueError("composite report is bound to a different v3 pack")
    if not same_path(declared_packs.get("v4"), packs["v4"]):
        raise ValueError("composite report is bound to a different v4 pack")
    recomputed, rc = compile_composite(Namespace(
        # Preserve the report's path spelling so the recomputed report is byte-
        # semantically comparable whether the compiler was invoked with relative
        # or absolute pack paths.
        v3_pack=Path(declared_packs["v3"]),
        v3_adjudication=adjudications["v3"],
        v4_pack=Path(declared_packs["v4"]),
        v4_adjudication=adjudications["v4"],
    ))
    if rc != 0 or recomputed.get("verdict") != "PASS":
        raise ValueError("recomputed composite audit is not PASS")
    for field in (
        "protocol", "verdict", "thresholds", "component_packs",
        "component_manifests", "component_report_hashes", "by_form",
        "blockers", "unresolved_sample_ids", "composite_inputs_sha256",
    ):
        if report.get(field) != recomputed.get(field):
            raise ValueError(f"composite report is stale or inconsistent in {field!r}")

    manifest = load_json(manifest_path, "composite corpus manifest")
    if (
        manifest.get("corpus_version") != expected_corpus_version
        or manifest.get("depth_scope") != depth_scope
        or manifest.get("label_policy") != "human_consensus_only"
        or manifest.get("scope") != "audited"
    ):
        raise ValueError("corpus is not the audited human-consensus composite release")
    if not {"household", "refspatial"}.issubset(
        set(manifest.get("excluded_source_buckets") or [])
    ):
        raise ValueError("corpus does not declare household/refspatial exclusions")
    semantic = manifest.get("semantic_audit") or {}
    audit_sha = file_hash(audit_path)
    if (
        semantic.get("protocol") != expected_protocol
        or semantic.get("depth_scope") != depth_scope
        or semantic.get("verdict") != "PASS"
        or not same_path(semantic.get("report"), audit_path)
        or semantic.get("report_sha256") != audit_sha
        or semantic.get("composite_inputs_sha256") != report.get("composite_inputs_sha256")
        or semantic.get("component_report_hashes") != report.get("component_report_hashes")
    ):
        raise ValueError("corpus/composite report binding mismatch")
    verify_component_bindings(report, semantic, packs, adjudications)
    verify_rows(manifest_path.parent, manifest, audit_sha, report)
    return {
        "verdict": "PASS",
        "protocol": expected_protocol,
        "depth_scope": depth_scope,
        "audit": str(audit_path),
        "corpus": str(manifest_path.parent),
        "records": manifest.get("accepted_records"),
        "composite_inputs_sha256": report.get("composite_inputs_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v3-pack", type=Path, required=True)
    parser.add_argument("--v3-adjudication", type=Path, required=True)
    parser.add_argument("--v4-pack", type=Path, required=True)
    parser.add_argument("--v4-adjudication", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
