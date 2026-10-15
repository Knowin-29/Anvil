#!/usr/bin/env python3
"""Materialize only a training-eligible v3-nondepth + depth composite corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.compile_semantic_composite_gate import (  # noqa: E402
    DEPTH_COMPONENT_CONTRACTS,
    NON_DEPTH_FORMS,
    compile_composite,
)
from trains.flywheel.contracts import CONTRACT_VERSION  # noqa: E402
from trains.flywheel.validators import (  # noqa: E402
    validate,
    validate_depth_v4_record,
)
from trains.materialize_semantic_audit_v3 import (  # noqa: E402
    FORMS,
    _latest_annotations,
    _load_adjudication,
    _make_swift,
    _read_jsonl,
    _split_rows,
    _write_jsonl,
    sha256_file,
)


CASCADE_FORMS = ("affordance", "dof", "action_cot")
BASE_DEPENDENCIES = ("bbox", "detection", "depth")
DEPENDENCY_POLICY = "v3-audited-cascade-closure-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _same_value(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _load_labels(pack: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for path in sorted((pack / "annotator_a/labels").rglob("*.jsonl")):
        for row in _read_jsonl(path):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in labels:
                raise ValueError(f"invalid or duplicate packed label: {sample_id}")
            labels[sample_id] = row
    return labels


def _load_pack_state(
    pack: Path,
    adjudication_path: Path | None,
) -> dict[str, Any]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    a = _latest_annotations(pack / "annotator_a/annotations/annotations.jsonl")
    b = _latest_annotations(pack / "annotator_b/annotations/annotations.jsonl")
    adjudication = _load_adjudication(adjudication_path)
    labels = _load_labels(pack)
    entries: dict[str, dict[str, Any]] = {}
    for item in manifest.get("selected", []):
        for form, sample_id in (item.get("sample_ids") or {}).items():
            if not isinstance(sample_id, str) or not sample_id or sample_id in entries:
                raise ValueError(f"invalid or duplicate manifest sample_id: {sample_id!r}")
            row = labels.get(sample_id)
            if (
                row is None
                or row.get("task_form") != form
                or row.get("image_path") != item.get("image_path")
            ):
                raise ValueError(f"packed label identity mismatch for {sample_id}")
            label_hash = (item.get("label_hashes") or {}).get(form)
            if not isinstance(label_hash, str) or _content_hash(row) != label_hash:
                raise ValueError(f"packed label hash drift for {sample_id}")
            va = a.get(sample_id, {}).get("correct")
            vb = b.get(sample_id, {}).get("correct")
            if not isinstance(va, bool) or not isinstance(vb, bool):
                raise ValueError(f"missing paired Boolean votes for {sample_id}")
            decision: bool | None
            if va == vb:
                decision = va
            else:
                decision = adjudication.get(sample_id)
            entries[sample_id] = {
                "decision": decision,
                "form": form,
                "item": item,
                "label_sha256": label_hash,
                "row": row,
            }
    expected = int(manifest.get("images", 0)) * len(manifest.get("forms") or [])
    if expected <= 0 or len(entries) != expected:
        raise ValueError(
            f"{pack}: expected {expected} selected rows, found {len(entries)}"
        )
    return {"entries": entries, "manifest": manifest}


def _accepted_rows(
    state: dict[str, Any],
    forms: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    manifest = state["manifest"]
    entries = state["entries"]
    accepted: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    expected = sum(
        1
        for item in manifest.get("selected", [])
        for form in (item.get("sample_ids") or {})
        if form in forms
    )
    if expected != int(manifest.get("images", 0)) * len(forms):
        raise ValueError(f"audit pack does not contain every requested form: {sorted(forms)}")
    for item in manifest.get("selected", []):
        for form, sample_id in (item.get("sample_ids") or {}).items():
            if form not in forms:
                continue
            entry = entries[sample_id]
            decision = entry["decision"]
            if decision is None:
                raise ValueError(f"unresolved disagreement for {sample_id}")
            stats["adjudicated_or_agreed"] += 1
            if not decision:
                stats["human_false"] += 1
                continue
            stats["human_true_raw"] += 1
            row = entry["row"]
            if row.get("_audit_only") is True:
                stats["audit_only"] += 1
                continue
            if form == "depth" and manifest.get("schema_version") == 4:
                valid, reason = validate_depth_v4_record(row)
            else:
                valid, reason = validate(form, row.get("payload"), row.get("gsa_context"))
            if not valid:
                stats[f"strict_invalid:{form}:{reason}"] += 1
                continue
            if row.get("source_bucket") in {"household", "refspatial"}:
                raise ValueError(f"blocked source bucket re-entered audit corpus: {sample_id}")
            accepted.append(row)
            stats["human_true"] += 1
    return accepted, stats


def _dependency_binding(entry: dict[str, Any]) -> dict[str, str]:
    return {
        "sample_id": entry["row"]["sample_id"],
        "label_sha256": entry["label_sha256"],
        "payload_sha256": _content_hash(entry["row"].get("payload")),
    }


def _grounding_payload(payload: Any) -> list[dict[str, Any]] | None:
    if not isinstance(payload, list):
        return None
    out: list[dict[str, Any]] = []
    for value in payload:
        if not isinstance(value, dict):
            return None
        box = value.get("bbox_2d") if "bbox_2d" in value else value.get("bbox")
        out.append({"object_class": value.get("label"), "bbox": box})
    return out


def _context_matches_dependency(row: dict[str, Any], form: str, payload: Any) -> bool:
    context = row.get("gsa_context")
    if not isinstance(context, dict):
        return False
    spatial = context.get("spatial")
    if not isinstance(spatial, dict):
        spatial = {}
    if form == "bbox":
        groundings = _grounding_payload(payload)
        return bool(groundings) and _same_value(context.get("primary"), groundings[0])
    if form == "detection":
        return _same_value(context.get("groundings"), _grounding_payload(payload))
    if form == "depth":
        return _same_value(spatial.get("depth_order"), payload)
    if form in {"affordance", "dof"}:
        return _same_value(spatial.get(form), payload)
    raise KeyError(f"unsupported cascade dependency: {form}")


def _dependency_closed_v3_rows(
    rows: list[dict[str, Any]],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, str]]], dict[str, Any]]:
    """Release A rows only when their audited v3 cascade is dependency-closed."""
    candidates = {row["sample_id"]: row for row in rows}
    entries = state["entries"]
    released = [row for row in rows if row.get("task_form") not in CASCADE_FORMS]
    released_a: set[str] = set()
    bindings: dict[str, dict[str, dict[str, str]]] = {}
    released_by_form: Counter[str] = Counter()
    excluded_by_reason: Counter[str] = Counter()
    dependency_references: Counter[str] = Counter()

    for item in state["manifest"].get("selected", []):
        sample_ids = item.get("sample_ids") or {}
        for form in CASCADE_FORMS:
            sample_id = sample_ids.get(form)
            if sample_id not in candidates:
                continue
            requirements = list(BASE_DEPENDENCIES)
            if form in {"dof", "action_cot"}:
                requirements.append("affordance")
            if form == "action_cot":
                requirements.append("dof")

            row = candidates[sample_id]
            row_bindings: dict[str, dict[str, str]] = {}
            rejection: str | None = None
            for dependency_form in requirements:
                dependency_id = sample_ids.get(dependency_form)
                entry = entries.get(dependency_id)
                if entry is None or entry.get("decision") is not True:
                    rejection = f"{form}:dependency_not_human_true:{dependency_form}"
                    break
                dependency_row = entry["row"]
                if dependency_row.get("_audit_only") is True:
                    rejection = f"{form}:dependency_audit_only:{dependency_form}"
                    break
                dependency_valid, dependency_reason = validate(
                    dependency_form,
                    dependency_row.get("payload"),
                    dependency_row.get("gsa_context"),
                )
                if not dependency_valid:
                    rejection = (
                        f"{form}:dependency_not_strict:{dependency_form}:"
                        f"{dependency_reason}"
                    )
                    break
                if dependency_form != "depth":
                    if dependency_id not in candidates:
                        rejection = f"{form}:dependency_not_strict:{dependency_form}"
                        break
                    if dependency_form in CASCADE_FORMS and dependency_id not in released_a:
                        rejection = f"{form}:dependency_not_closed:{dependency_form}"
                        break
                if not _context_matches_dependency(
                    row, dependency_form, dependency_row.get("payload")
                ):
                    rejection = f"{form}:context_mismatch:{dependency_form}"
                    break
                row_bindings[dependency_form] = _dependency_binding(entry)

            if rejection is not None:
                excluded_by_reason[rejection] += 1
                continue
            released.append(row)
            released_a.add(sample_id)
            bindings[sample_id] = row_bindings
            released_by_form[form] += 1
            dependency_references.update(row_bindings.keys())

    binding_rows = [
        {"sample_id": sample_id, "dependencies": bindings[sample_id]}
        for sample_id in sorted(bindings)
    ]
    stats = {
        "policy": DEPENDENCY_POLICY,
        "released_by_form": dict(released_by_form),
        "excluded_by_reason": dict(excluded_by_reason),
        "dependency_reference_counts": dict(dependency_references),
        "bindings_sha256": _content_hash(binding_rows),
    }
    return released, bindings, stats


def _fresh_composite_report(
    report_path: Path,
    v3_pack: Path,
    v3_adjudication: Path | None,
    depth_pack: Path,
    depth_adjudication: Path | None,
) -> tuple[dict[str, Any], str]:
    try:
        supplied = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read composite report: {exc}") from exc
    if (
        not isinstance(supplied, dict)
        or supplied.get("protocol") not in {
            contract["composite_protocol"]
            for contract in DEPTH_COMPONENT_CONTRACTS.values()
        }
        or supplied.get("verdict") != "PASS"
    ):
        raise ValueError("composite semantic audit report must be a protocol-matching PASS")
    declared = supplied.get("component_packs") or {}
    depth_component = supplied.get("depth_component") or {}
    depth_key = depth_component.get("key")
    if not isinstance(depth_key, str) or not depth_key:
        raise ValueError("composite report lacks a registered depth component")
    if Path(str(declared.get("v3"))).resolve() != v3_pack.resolve():
        raise ValueError("composite report is bound to a different v3 pack")
    if Path(str(declared.get(depth_key))).resolve() != depth_pack.resolve():
        raise ValueError("composite report is bound to a different depth pack")
    recomputed, rc = compile_composite(argparse.Namespace(
        v3_pack=Path(declared["v3"]),
        v3_adjudication=v3_adjudication,
        depth_pack=Path(declared[depth_key]),
        depth_adjudication=depth_adjudication,
    ))
    if rc != 0 or recomputed.get("verdict") != "PASS":
        raise ValueError("recomputed composite audit is not PASS")
    fields = (
        "protocol", "depth_scope", "depth_component", "training_materialization",
        "qualification_scope", "training_release_verdict", "training_release_blocker",
        "verdict", "thresholds", "component_manifests", "component_report_hashes",
        "by_form", "blockers", "unresolved_sample_ids", "composite_inputs_sha256",
    )
    for field in fields:
        if supplied.get(field) != recomputed.get(field):
            raise ValueError(f"composite report is stale or inconsistent in {field!r}")
    return supplied, sha256_file(report_path)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    v3_pack = args.v3_pack.resolve()
    explicit_depth = getattr(args, "depth_pack", None)
    depth_pack_value = explicit_depth or getattr(args, "v4_pack", None)
    if depth_pack_value is None:
        raise ValueError("a depth audit pack is required")
    depth_pack = depth_pack_value.resolve()
    v3_adjudication = args.v3_adjudication.resolve()
    depth_adjudication_value = (
        getattr(args, "depth_adjudication", None)
        if explicit_depth is not None
        else getattr(args, "v4_adjudication", None)
    )
    if depth_adjudication_value is None:
        raise ValueError("depth adjudication file is required for hash binding")
    depth_adjudication = depth_adjudication_value.resolve()
    for label, path in (
        ("v3 adjudication", v3_adjudication),
        ("depth adjudication", depth_adjudication),
    ):
        if not path.is_file():
            raise ValueError(f"{label} file is required for hash binding: {path}")
    report_path = args.audit.resolve()
    report, report_sha = _fresh_composite_report(
        report_path, v3_pack, v3_adjudication, depth_pack, depth_adjudication
    )
    if (report.get("training_materialization") or {}).get("eligible") is not True:
        raise ValueError("held_out_sensor_records_not_trainable")
    depth_key = report["depth_component"]["key"]
    v3_state = _load_pack_state(v3_pack, v3_adjudication)
    depth_state = _load_pack_state(depth_pack, depth_adjudication)
    v3_rows, v3_stats = _accepted_rows(v3_state, set(NON_DEPTH_FORMS))
    depth_rows, depth_stats = _accepted_rows(depth_state, {"depth"})
    v3_rows, dependency_bindings, dependency_stats = _dependency_closed_v3_rows(
        v3_rows, v3_state
    )
    v3_manifest = v3_state["manifest"]
    depth_manifest = depth_state["manifest"]
    audit_identity = {
        "audit_id": f"composite:{report['composite_inputs_sha256'][:16]}",
        "source_fingerprint": report["composite_inputs_sha256"],
    }
    rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    seen_image_forms: set[tuple[str, str]] = set()
    for component, component_rows in (("v3", v3_rows), (depth_key, depth_rows)):
        for row in component_rows:
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in seen_sample_ids:
                raise ValueError(f"duplicate or empty accepted sample_id: {sample_id!r}")
            swift = _make_swift(row, "pending", audit_identity, report_sha, reviewed=True)
            image_form = (
                str(swift["gsa_meta"]["image_sha256"]),
                str(swift["gsa_form"]),
            )
            if image_form in seen_image_forms:
                raise ValueError(f"duplicate accepted image/form pair: {image_form}")
            seen_sample_ids.add(sample_id)
            seen_image_forms.add(image_form)
            swift["gsa_meta"].update({
                "semantic_audit_component": component,
                "source_audit_only": False,
                "semantic_audit_component_report_sha256": report[
                    "component_report_hashes"
                ][f"{component}_report_sha256"],
                "semantic_audit_composite_inputs_sha256": report[
                    "composite_inputs_sha256"
                ],
            })
            if sample_id in dependency_bindings:
                dependencies = dependency_bindings[sample_id]
                swift["gsa_meta"].update({
                    "semantic_audit_dependency_policy": DEPENDENCY_POLICY,
                    "semantic_audit_dependencies": dependencies,
                    "semantic_audit_dependencies_sha256": _content_hash(dependencies),
                })
            if component == depth_key:
                swift["gsa_meta"]["depth_scope"] = row.get("depth_scope") or "total_order"
            rows.append(swift)
    if not rows or {row["gsa_form"] for row in rows} != set(FORMS):
        raise ValueError("accepted composite rows do not cover all eight forms")
    train, val = _split_rows(rows, args.seed, args.val_fraction)
    for row in train:
        row["gsa_meta"]["split"] = "train"
    for row in val:
        row["gsa_meta"]["split"] = "val"

    out = args.out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
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
        images = {
            split: {row["gsa_meta"]["image_sha256"] for row in split_rows}
            for split, split_rows in (("train", train), ("val", val))
        }
        manifest = {
            "schema_version": 1,
            "corpus_version": report["depth_component"]["corpus_version"],
            "depth_scope": report["depth_scope"],
            "label_policy": "human_consensus_only",
            "scope": "audited",
            "accepted_records": len(rows),
            "review_stats": {"v3": dict(v3_stats), depth_key: dict(depth_stats)},
            "dependency_closure": dependency_stats,
            "counts": {"train": len(train), "val": len(val)},
            "unique_images": {key: len(value) for key, value in images.items()},
            "overlap": {
                "train_val": len(images["train"] & images["val"]),
                "train_test": 0,
                "val_test": 0,
            },
            "excluded_source_buckets": ["household", "refspatial"],
            "by_form": {
                split: dict(Counter(row["gsa_form"] for row in split_rows))
                for split, split_rows in (("train", train), ("val", val))
            },
            "by_family": {
                split: dict(Counter(row["gsa_family"] for row in split_rows))
                for split, split_rows in (("train", train), ("val", val))
            },
            "semantic_audit": {
                "protocol": report["protocol"],
                "depth_scope": report["depth_scope"],
                "depth_component": report["depth_component"],
                "training_materialization": report["training_materialization"],
                "report": str(report_path),
                "report_sha256": report_sha,
                "verdict": report["verdict"],
                "composite_inputs_sha256": report["composite_inputs_sha256"],
                "component_report_hashes": report["component_report_hashes"],
                "component_manifests": report["component_manifests"],
                "component_adjudications": {
                    "v3": {
                        "path": str(v3_adjudication),
                        "sha256": sha256_file(v3_adjudication),
                    },
                    depth_key: {
                        "path": str(depth_adjudication),
                        "sha256": sha256_file(depth_adjudication),
                    },
                },
                "packs": {"v3": str(v3_pack), depth_key: str(depth_pack)},
                "pack_manifest_sha256": {
                    "v3": sha256_file(v3_pack / "manifest.json"),
                    depth_key: sha256_file(depth_pack / "manifest.json"),
                },
                "audit_ids": {
                    "v3": v3_manifest.get("audit_id"),
                    depth_key: depth_manifest.get("audit_id"),
                },
            },
            "split_rule": {
                "image_disjoint": True,
                "seed": args.seed,
                "val_fraction": args.val_fraction,
            },
            "contract_version": CONTRACT_VERSION,
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
    parser.add_argument("--v3-pack", type=Path, required=True)
    parser.add_argument("--v3-adjudication", type=Path, required=True)
    parser.add_argument("--depth-pack", type=Path)
    parser.add_argument("--depth-adjudication", type=Path)
    parser.add_argument("--v4-pack", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--v4-adjudication", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("trains/gsa_bench_v4_composite_pilot_clean")
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if args.depth_pack is None and args.v4_pack is None:
        parser.error("--depth-pack is required")
    if args.depth_pack is not None and args.v4_pack is not None:
        parser.error("use --depth-pack, not both depth/v4 aliases")
    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1")
    try:
        materialize(args)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
