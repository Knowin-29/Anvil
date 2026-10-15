#!/usr/bin/env python3
"""Combine resolved v3 non-depth evidence with a registered depth audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from compile_semantic_audit import (  # noqa: E402
    DEFAULT_MIN_ACCURACY,
    DEFAULT_MIN_AGREEMENT,
    V3_FORMS,
    canonical_json,
    compile_report,
    file_hash,
)

NON_DEPTH_FORMS = [form for form in V3_FORMS if form != "depth"]
COMPOSITE_PROTOCOLS = {
    "total_order": "semantic-audit-composite-v3-nondepth-v4-depth",
    "pairwise": "semantic-audit-composite-v3-nondepth-v42-pairwise-depth",
}
COMPOSITE_CORPUS_VERSIONS = {
    "total_order": "gsa-bench-v4-composite-pilot-clean",
    "pairwise": "gsa-bench-v42-pairwise-composite-pilot-clean",
}

# ``depth_scope`` cannot identify the evidence protocol: v4.2, v4.3, and
# sensor-derived schema5 records are all pairwise but have different trust and
# release semantics.
DEPTH_COMPONENT_CONTRACTS = {
    (4, "depth-only-calibration"): {
        "key": "v4",
        "name": "v4_total_order_depth",
        "composite_protocol": COMPOSITE_PROTOCOLS["total_order"],
        "corpus_version": COMPOSITE_CORPUS_VERSIONS["total_order"],
        "training_materialization": True,
    },
    (4, "depth-pairwise-calibration-v1"): {
        "key": "v4",
        "name": "v42_pairwise_depth",
        "composite_protocol": COMPOSITE_PROTOCOLS["pairwise"],
        "corpus_version": COMPOSITE_CORPUS_VERSIONS["pairwise"],
        "training_materialization": True,
    },
    (4, "depth-v4.3-pairwise-semantic-audit-v1"): {
        "key": "v43_depth",
        "name": "v43_pairwise_depth",
        "composite_protocol": "semantic-audit-composite-v3-nondepth-v43-pairwise-depth",
        "corpus_version": "gsa-bench-v43-pairwise-composite-pilot-clean",
        "training_materialization": True,
    },
    (5, "depth-v5.0-scannet-sensor-calibration-v1"): {
        "key": "v5_scannet_sensor_depth",
        "name": "v50_scannet_sensor_pairwise_depth",
        "composite_protocol": (
            "semantic-audit-composite-v3-nondepth-v50-scannet-sensor-depth"
        ),
        "corpus_version": "gsa-benchmark-v50-scannet-sensor-calibration-heldout-v1",
        "training_materialization": False,
        "qualification_scope": "held_out_benchmark_component_calibration",
    },
}


def _depth_args(args: argparse.Namespace) -> tuple[Path, Path | None]:
    explicit = getattr(args, "depth_pack", None)
    pack = explicit or getattr(args, "v4_pack", None)
    adjudication = (
        getattr(args, "depth_adjudication", None)
        if explicit is not None
        else getattr(args, "v4_adjudication", None)
    )
    if pack is None:
        raise ValueError("a depth audit pack is required")
    return Path(pack), Path(adjudication) if adjudication is not None else None


def depth_component_contract(report: dict[str, Any]) -> dict[str, Any] | None:
    schema = report.get("schema_version")
    protocol = report.get("protocol")
    # The historical schema4 total-order report and old synthetic tests omitted
    # the protocol field. Keep that one compatibility path.
    if schema == 4 and protocol is None:
        protocol = "depth-only-calibration"
    contract = DEPTH_COMPONENT_CONTRACTS.get((schema, protocol))
    return dict(contract) if contract is not None else None


def report_hash(report: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(report).encode("utf-8")).hexdigest()


def compile_component(pack: Path, adjudication: Path | None) -> dict[str, Any]:
    report, _ = compile_report(Namespace(
        pack=pack,
        adjudication=adjudication,
        min_accuracy=DEFAULT_MIN_ACCURACY,
        min_agreement=DEFAULT_MIN_AGREEMENT,
    ))
    return report


def form_blockers(source: str, form: str, row: dict[str, Any] | None) -> list[str]:
    if not isinstance(row, dict):
        return [f"{source}:{form}:missing_report_row"]
    blockers: list[str] = []
    if row.get("paired") != 40:
        blockers.append(f"{source}:{form}:paired={row.get('paired')}/40")
    if row.get("unresolved") != 0:
        blockers.append(f"{source}:{form}:unresolved={row.get('unresolved')}")
    agreement = row.get("raw_agreement")
    if not isinstance(agreement, (int, float)) or agreement < DEFAULT_MIN_AGREEMENT:
        blockers.append(f"{source}:{form}:raw_agreement={agreement}<0.80")
    accuracy = row.get("resolved_accuracy")
    if not isinstance(accuracy, (int, float)) or accuracy < DEFAULT_MIN_ACCURACY:
        blockers.append(f"{source}:{form}:resolved_accuracy={accuracy}<0.80")
    if row.get("eligible") is not True:
        blockers.append(f"{source}:{form}:not_eligible")
    return blockers


def compile_composite(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    depth_pack, depth_adjudication = _depth_args(args)
    v3 = compile_component(args.v3_pack, args.v3_adjudication)
    depth = compile_component(depth_pack, depth_adjudication)
    blockers: list[str] = []
    if v3.get("schema_version") != 3:
        blockers.append(f"v3:schema_version={v3.get('schema_version')}")
    contract = depth_component_contract(depth)
    if contract is None:
        blockers.append(
            "depth:unregistered_component="
            f"schema{depth.get('schema_version')}:{depth.get('protocol')!r}"
        )
        contract = dict(DEPTH_COMPONENT_CONTRACTS[(4, "depth-only-calibration")])
    depth_key = contract["key"]
    depth_scope = depth.get("depth_scope") or "total_order"
    if depth_scope not in {"total_order", "pairwise"}:
        blockers.append(f"{depth_key}:unsupported_depth_scope={depth_scope!r}")
    if depth.get("schema_version") == 5 and depth_scope != "pairwise":
        blockers.append(f"{depth_key}:schema5_requires_pairwise_depth")
    protocol = contract["composite_protocol"]
    for error in v3.get("errors") or []:
        blockers.append(f"v3:integrity:{error}")
    for error in depth.get("errors") or []:
        blockers.append(f"{depth_key}:integrity:{error}")

    combined_forms: dict[str, Any] = {}
    for form in NON_DEPTH_FORMS:
        row = (v3.get("by_form") or {}).get(form)
        blockers.extend(form_blockers("v3", form, row))
        combined_forms[form] = {"source": "v3", **(row or {})}
    depth_row = (depth.get("by_form") or {}).get("depth")
    blockers.extend(form_blockers(depth_key, "depth", depth_row))
    combined_forms["depth"] = {"source": depth_key, **(depth_row or {})}

    blockers = list(dict.fromkeys(blockers))
    component_hashes = {
        "v3_report_sha256": report_hash(v3),
        f"{depth_key}_report_sha256": report_hash(depth),
    }
    training_eligible = bool(contract["training_materialization"])
    if depth.get("training_materialization") is not None:
        training_eligible = (
            training_eligible
            and (depth.get("training_materialization") or {}).get("eligible") is True
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": protocol,
        "depth_scope": depth_scope,
        "depth_component": {
            "key": depth_key,
            "name": contract["name"],
            "schema_version": depth.get("schema_version"),
            "audit_protocol": depth.get("protocol"),
            "corpus_version": contract["corpus_version"],
        },
        "training_materialization": {
            "eligible": training_eligible,
            "reason": (
                "audited_depth_component_allows_training"
                if training_eligible
                else "held_out_sensor_records_not_trainable"
            ),
        },
        "qualification_scope": contract.get(
            "qualification_scope", "training_component_qualification"
        ),
        "training_release_verdict": "PASS" if training_eligible else "BLOCKED",
        "training_release_blocker": (
            None
            if training_eligible
            else "requires_official_train_source_manifest_and_no_scene_overlap_gate"
        ),
        "verdict": "PASS" if not blockers else "BLOCKED",
        "thresholds": {
            "min_accuracy": DEFAULT_MIN_ACCURACY,
            "min_agreement": DEFAULT_MIN_AGREEMENT,
        },
        "component_packs": {
            "v3": str(args.v3_pack),
            depth_key: str(depth_pack),
        },
        "component_manifests": {
            "v3": {
                "path": v3.get("manifest"),
                "sha256": file_hash(Path(v3["manifest"]))
                if isinstance(v3.get("manifest"), str) and Path(v3["manifest"]).is_file()
                else None,
            },
            depth_key: {
                "path": depth.get("manifest"),
                "sha256": file_hash(Path(depth["manifest"]))
                if isinstance(depth.get("manifest"), str) and Path(depth["manifest"]).is_file()
                else None,
            },
        },
        "component_report_hashes": component_hashes,
        "by_form": combined_forms,
        "blockers": blockers,
        "unresolved_sample_ids": {
            "v3_non_depth": [
                sample_id
                for sample_id in v3.get("unresolved_sample_ids") or []
                if any(
                    sample_id in item.get("sample_ids", {}).values()
                    and item.get("sample_ids", {}).get("depth") != sample_id
                    for item in json.loads(
                        Path(v3["manifest"]).read_text(encoding="utf-8")
                    ).get("selected", [])
                )
            ] if isinstance(v3.get("manifest"), str) and Path(v3["manifest"]).is_file() else [],
            f"{depth_key}_depth": depth.get("unresolved_sample_ids") or [],
        },
        "component_reports": {"v3": v3, depth_key: depth},
        "notes": (
            (
                "Schema5 PASS is limited to held-out benchmark-component calibration. "
                "Official ScanNet val/test sensor records are not a training release and "
                "cannot authorize joint, S-family, or MOPD training. A future training "
                "release requires an independent official-train source manifest and a "
                "no-scene-overlap gate."
            )
            if depth.get("schema_version") == 5
            else
            "The standalone depth component comes from an independently audited "
            f"{contract['name']} source with depth_scope={depth_scope}. It does not "
            "substitute for human-final v3 scene-level depth records used only to "
            "close each A-family cascade. V3 non-depth disagreements still require "
            "adjudication, and raw agreement is never altered by adjudication."
        ),
    }
    report["composite_inputs_sha256"] = hashlib.sha256(canonical_json({
        "component_report_hashes": component_hashes,
        "component_manifests": report["component_manifests"],
        "protocol": report["protocol"],
        "depth_scope": report["depth_scope"],
        "depth_component": report["depth_component"],
        "training_materialization": report["training_materialization"],
        "qualification_scope": report["qualification_scope"],
        "training_release_verdict": report["training_release_verdict"],
        "training_release_blocker": report["training_release_blocker"],
        "thresholds": report["thresholds"],
    }).encode("utf-8")).hexdigest()
    return report, 0 if report["verdict"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-pack", type=Path, required=True)
    parser.add_argument("--v3-adjudication", type=Path)
    parser.add_argument("--depth-pack", type=Path)
    parser.add_argument("--depth-adjudication", type=Path)
    parser.add_argument("--v4-pack", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--v4-adjudication", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/audits/pilot_semantic_audit_composite.json"),
    )
    args = parser.parse_args()
    if args.depth_pack is None and args.v4_pack is None:
        parser.error("--depth-pack is required")
    if args.depth_pack is not None and args.v4_pack is not None:
        parser.error("use --depth-pack, not both depth/v4 aliases")
    report, return_code = compile_composite(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
