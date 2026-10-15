#!/usr/bin/env python3
"""Compile the schema-v3 G-family audit into a narrowly scoped release gate.

This gate does not reinterpret the schema-v3 audit as a global PASS.  It only
authorizes a G specialist qualification when bbox, detection, mask, and point
are complete and eligible after the independent A/B review and reviewer-C
adjudication.  Depth and every non-G form remain outside this release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.compile_semantic_audit import (  # noqa: E402
    DEFAULT_MIN_ACCURACY,
    DEFAULT_MIN_AGREEMENT,
    V3_FORMS,
    compile_report,
)

PROTOCOL = "semantic-audit-v3-g-family-scoped-v1"
CORPUS_VERSION = "gsa-bench-v3-g-family-scoped-v1"
G_FORMS = ("bbox", "detection", "mask", "point")
REVIEWER_C_KEY = "reviewer-c-v3"
SCOPED_PASS = "G_ONLY_PASS"
SCOPED_BLOCKED = "G_ONLY_BLOCKED"
AUTHORIZATION = "G_specialist_qualification_only"
FORBIDDEN_USES = [
    "global_semantic_audit_pass_claim",
    "joint_training",
    "mopd_training",
    "S_or_A_specialist_training",
]


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


def same_path(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).expanduser().resolve(strict=False) == expected.resolve(strict=False)


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


def file_binding(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return {"path": str(path.resolve()), "sha256": file_hash(path)}


def _reviewer_c_errors(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing reviewer-C adjudication: {path}"]
    rows = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path}:{line_number}: adjudication is not an object")
            continue
        if row.get("reviewer_key") != REVIEWER_C_KEY:
            errors.append(
                f"{path}:{line_number}: reviewer_key must be {REVIEWER_C_KEY!r}"
            )
    if rows == 0:
        errors.append("reviewer-C adjudication is empty")
    return errors


def _report_drift_errors(
    supplied: dict[str, Any],
    recomputed: dict[str, Any],
    manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    if not same_path(supplied.get("manifest"), manifest_path):
        errors.append("full compiler report is bound to a different pack manifest")
    for field in (
        "schema_version",
        "protocol",
        "depth_scope",
        "verdict",
        "thresholds",
        "by_form",
        "unresolved_sample_ids",
        "errors",
        "annotator_keys",
    ):
        if supplied.get(field) != recomputed.get(field):
            errors.append(f"full compiler report is stale in field {field!r}")
    return errors


def compile_scoped(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    pack = args.pack.resolve()
    adjudication = args.adjudication.resolve()
    full_report_path = args.full_report.resolve()
    min_accuracy = float(args.min_accuracy)
    min_agreement = float(args.min_agreement)
    blockers: list[str] = []

    if min_accuracy < DEFAULT_MIN_ACCURACY or min_agreement < DEFAULT_MIN_AGREEMENT:
        blockers.append(
            "G-only thresholds cannot be lower than accuracy=0.80 and agreement=0.80"
        )
    blockers.extend(_reviewer_c_errors(adjudication))

    manifest_path = pack / "manifest.json"
    try:
        manifest = load_json(manifest_path, "schema-v3 manifest")
    except ValueError as exc:
        manifest = {}
        blockers.append(str(exc))
    if manifest.get("schema_version") != 3:
        blockers.append("G-only release requires schema_version=3")
    if manifest.get("forms") != V3_FORMS or manifest.get("images") != 40:
        blockers.append("G-only release requires the fixed 40-image, eight-form v3 pack")

    recomputed, _ = compile_report(
        argparse.Namespace(
            pack=pack,
            adjudication=adjudication,
            min_accuracy=min_accuracy,
            min_agreement=min_agreement,
        )
    )
    try:
        supplied = load_json(full_report_path, "full v3 compiler report")
    except ValueError as exc:
        supplied = {}
        blockers.append(str(exc))
    blockers.extend(_report_drift_errors(supplied, recomputed, manifest_path))

    if recomputed.get("schema_version") != 3:
        blockers.append("recomputed compiler report is not schema v3")
    if recomputed.get("errors"):
        blockers.append("recomputed v3 compiler report contains integrity errors")
    if recomputed.get("verdict") != "BLOCKED":
        blockers.append(
            "this scoped protocol is reserved for a globally BLOCKED v3 audit"
        )
    target_metrics: dict[str, Any] = {}
    by_form = recomputed.get("by_form") or {}
    for form in G_FORMS:
        metrics = by_form.get(form)
        if not isinstance(metrics, dict):
            blockers.append(f"recomputed compiler report is missing {form}")
            continue
        target_metrics[form] = metrics
        if metrics.get("expected") != 40 or metrics.get("paired") != 40:
            blockers.append(f"{form} lacks all 40 paired A/B decisions")
        if metrics.get("unresolved") != 0:
            blockers.append(f"{form} still has unresolved A/B disagreements")
        if metrics.get("eligible") is not True:
            blockers.append(f"{form} does not meet the scoped release thresholds")

    unresolved_g = [
        sample_id
        for sample_id in recomputed.get("unresolved_sample_ids") or []
        if any(sample_id.startswith(f"v3_{form}_") for form in G_FORMS)
    ]
    if unresolved_g:
        blockers.append("G-family unresolved sample IDs remain")
    depth_metrics = by_form.get("depth") or {}
    if depth_metrics.get("eligible") is not False:
        blockers.append("this scoped protocol is allowed only while v3 depth remains blocked")
    blocking_forms = sorted(
        form
        for form, metrics in by_form.items()
        if isinstance(metrics, dict) and metrics.get("eligible") is not True
    )
    if blocking_forms != ["depth"]:
        blockers.append(
            "G-only release requires depth to be the sole globally blocked form"
        )

    binding_specs = {
        "pack_manifest": (manifest_path, "schema-v3 manifest"),
        "annotator_a_annotations": (
            pack / "annotator_a/annotations/annotations.jsonl",
            "annotator-A annotations",
        ),
        "annotator_b_annotations": (
            pack / "annotator_b/annotations/annotations.jsonl",
            "annotator-B annotations",
        ),
        "reviewer_c_adjudication": (adjudication, "reviewer-C adjudication"),
        "full_compiler_report": (full_report_path, "full v3 compiler report"),
    }
    input_bindings: dict[str, dict[str, str]] = {}
    for name, (path, label) in binding_specs.items():
        try:
            input_bindings[name] = file_binding(path, label)
        except ValueError as exc:
            blockers.append(str(exc))

    image_set = sorted(
        {
            item.get("image_sha256")
            for item in manifest.get("selected", [])
            if isinstance(item, dict) and isinstance(item.get("image_sha256"), str)
        }
    )
    if len(image_set) != 40:
        blockers.append(f"v3 audit image set must contain 40 unique hashes, found {len(image_set)}")

    scoped_status = SCOPED_PASS if not blockers else SCOPED_BLOCKED
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "verdict": scoped_status,
        "release_scope": "G_family_only",
        "family": "G",
        "forms": list(G_FORMS),
        "authorization": AUTHORIZATION,
        "forbidden_uses": FORBIDDEN_USES,
        "global_audit_verdict": recomputed.get("verdict"),
        "global_audit_blockers_preserved": recomputed.get("verdict") != "PASS",
        "global_blocking_forms": blocking_forms,
        "thresholds": {
            "min_accuracy": min_accuracy,
            "min_agreement": min_agreement,
        },
        "by_form": target_metrics,
        "unresolved_g_sample_ids": unresolved_g,
        "input_bindings": input_bindings,
        "compiler_inputs_sha256": content_hash(
            {
                "protocol": PROTOCOL,
                "thresholds": [min_accuracy, min_agreement],
                "input_bindings": input_bindings,
            }
        ),
        "audit_image_set_sha256": content_hash(image_set),
        "blockers": sorted(set(blockers)),
        "notes": (
            "G_ONLY_PASS authorizes only a bbox/detection/mask/point specialist "
            "qualification. It is not a global semantic-audit PASS and does not "
            "authorize joint or multi-teacher on-policy distillation training."
        ),
    }
    return report, 0 if scoped_status == SCOPED_PASS else 2


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
        "--out",
        type=Path,
        default=Path("artifacts/audits/pilot_semantic_audit_v3_g_family_scoped_v1.json"),
    )
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY)
    parser.add_argument("--min-agreement", type=float, default=DEFAULT_MIN_AGREEMENT)
    args = parser.parse_args()
    report, return_code = compile_scoped(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
