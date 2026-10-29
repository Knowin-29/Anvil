#!/usr/bin/env python3
"""Adapt multi-form flywheel output into the EXISTING GSA viewer's layout.

The established viewer (`Rex-Embodied-anvil_dev/scripts/viewer_service.py`, 1711
lines: dashboard + case browser + human annotation platform + stats + export)
expects:

    <root>/labels/<dimension>/*.jsonl      raw generated
    <root>/verified/<dimension>/*.jsonl    accepted   (needs `_verification`)
    <root>/needs_review/<dimension>/...
    <root>/rejected/<dimension>/...

with per-record fields: `pipeline` / `dimension`, `scene_type`, `image_path`,
`sample_id`, plus the payload fields inline (bbox, point_2d, ...).

Our multi-form output is keyed by *task form* (bbox/detection/mask/dof/...), which is
a finer axis than the viewer's 6 GSA dimensions. This maps form -> dimension while
keeping `task_form` on the record, and inlines the payload so the viewer's bbox/point
overlays work unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

# task form -> viewer dimension
FORM2DIM = {
    "bbox": "G_grounding",
    "detection": "G_grounding",
    "point": "G_grounding",
    "mask": "G_grounding",
    "depth": "S_spatial",
    "affordance": "A_action_interface",
    "dof": "A_action_interface",
    "action_cot": "A_action_precondition",
}
HELD_OUT_PROTOCOLS = {"gsa-depth-v5.0-scannet-sensor-pairwise"}
HELD_OUT_SOURCE_SCHEMAS = {"gsa-scannet-sensor-depth-source-v1"}


def adapt(rec: dict, form: str, idx: int, id_prefix: str = "mf") -> dict:
    if (
        rec.get("protocol_version") in HELD_OUT_PROTOCOLS
        or rec.get("source_artifact_schema") in HELD_OUT_SOURCE_SCHEMAS
    ):
        raise ValueError(
            "held-out ScanNet sensor records require the schema-5 audit builder; "
            "generic viewer conversion is forbidden"
        )
    payload = rec.get("payload")
    items = payload if isinstance(payload, list) else [payload]
    first = items[0] if items and isinstance(items[0], dict) else {}
    context = rec.get("gsa_context") or rec.get("g_context") or {}
    g = context.get("primary") if isinstance(context.get("primary"), dict) else context

    if id_prefix == "mf":
        sample_id = f"mf_{form}_{idx:08d}"
    else:
        identity = "|".join((
            str(rec.get("image_sha256", "")),
            form,
            str(rec.get("label_version", "")),
            json.dumps(items, ensure_ascii=False, sort_keys=True),
        ))
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        sample_id = f"{id_prefix}_{form}_{digest}"

    out = {
        "sample_id": sample_id,
        "pipeline": FORM2DIM.get(form, "G_grounding"),
        "dimension": FORM2DIM.get(form, "G_grounding"),
        "task_form": form,
        "image_path": rec.get("image_path", ""),
        "source": rec.get("source", "unknown"),
        "dataset": rec.get("source", "unknown"),
        "scene_type": rec.get("scene", "embodied_home"),
        "object_class": first.get("label") or first.get("object_class") or g.get("object_class"),
        "confidence": first.get("confidence", 0.9),
        "label_version": rec.get("label_version"),
        "prompt_version": rec.get("prompt_version"),
        "payload": items,          # full multi-item payload (detection/depth lists)
        "gsa_context": context,
    }
    # Inline the primary geometry so the viewer's existing overlays work as-is.
    for k_src, k_dst in (("bbox_2d", "bbox"), ("bbox", "bbox"),
                         ("point_2d", "point_2d"),
                         ("grasp_point_2d", "grasp_point_2d"),
                         ("contact_point_2d", "contact_point_2d"),
                         ("jaw_axis_2d", "jaw_axis_2d"),
                         ("polygon", "polygon")):
        if k_src in first and k_dst not in out:
            out[k_dst] = first[k_src]
    if "bbox" not in out and g.get("bbox"):
        out["bbox"] = g["bbox"]
    if rec.get("_cross_check"):
        out["_cross_check"] = rec["_cross_check"]
    if rec.get("_verification"):
        out["_verification"] = rec["_verification"]
    else:
        out["_verification"] = {"tag": "verified_auto", "verdict": "verified"}
    if rec.get("_scan_count"):
        out["_scan_count"] = rec["_scan_count"]
    if rec.get("_semantic_gate"):
        out["_semantic_gate"] = rec["_semantic_gate"]
    if rec.get("_audit_only"):
        out["_audit_only"] = True
        out["_audit_reason"] = rec.get("_audit_reason", "semantic_review")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dirs", nargs="+", required=True,
                    help="multiform output dirs (each containing raw/<form>.jsonl)")
    ap.add_argument("--out", required=True, help="viewer data root to create")
    ap.add_argument("--status", default="verified",
                    choices=["verified", "labels", "needs_review", "rejected"])
    ap.add_argument("--limit-per-form", type=int, default=0)
    ap.add_argument("--id-prefix", default="mf",
                    help="non-default prefix enables content-addressed sample IDs")
    ap.add_argument("--require-empty", action="store_true",
                    help="refuse to merge into a non-empty viewer root")
    ap.add_argument("--source-manifest", default="",
                    help="upload manifest with upload_sha256/source_bucket")
    ap.add_argument("--exclude-source-bucket", action="append", default=[],
                    help="source bucket to omit (repeatable)")
    args = ap.parse_args()

    outroot = Path(args.out)
    if args.require_empty and outroot.exists() and any(outroot.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty viewer root: {outroot}")
    stats = Counter()
    idx = 0
    sample_ids = set()
    source_buckets = {}
    if args.source_manifest:
        for line_no, line in enumerate(Path(args.source_manifest).read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid source manifest at {args.source_manifest}:{line_no}") from exc
            if row.get("upload_sha256") and row.get("source_bucket"):
                source_buckets[row["upload_sha256"]] = row["source_bucket"]
    excluded_buckets = set(args.exclude_source_bucket)
    # write into BOTH labels/ and the status dir: the viewer's default (no status
    # filter) reads labels/, while the Verified tab reads verified/.
    for form, dim in FORM2DIM.items():
        rows = []
        for d in args.in_dirs:
            p = Path(d) / "raw" / f"{form}.jsonl"
            if not p.is_file():
                continue
            for ln in p.read_text(errors="ignore").splitlines():
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                source_bucket = source_buckets.get(rec.get("image_sha256", ""))
                if source_bucket in excluded_buckets:
                    stats[f"excluded/{source_bucket}"] += 1
                    continue
                idx += 1
                adapted = adapt(rec, form, idx, args.id_prefix)
                if adapted["sample_id"] in sample_ids:
                    raise ValueError(
                        f"duplicate sample_id generated: {adapted['sample_id']}")
                sample_ids.add(adapted["sample_id"])
                adapted["source_bucket"] = source_bucket or "unknown"
                rows.append(adapted)
                if args.limit_per_form and len(rows) >= args.limit_per_form:
                    break
            if args.limit_per_form and len(rows) >= args.limit_per_form:
                break
        if not rows:
            continue
        for sub in ("labels", args.status):
            dd = outroot / sub / dim
            dd.mkdir(parents=True, exist_ok=True)
            with (dd / f"{form}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"{dim}/{form}"] = len(rows)

    # the viewer also reads these dirs; create them so its scans don't fail
    for sub in ("needs_review", "rejected"):
        (outroot / sub).mkdir(parents=True, exist_ok=True)

    included = {k: v for k, v in stats.items() if not k.startswith("excluded/")}
    excluded = {k.removeprefix("excluded/"): v for k, v in stats.items()
                if k.startswith("excluded/")}
    stats_payload = {
        "included_records": sum(included.values()),
        "excluded_records": sum(excluded.values()),
        "included_by_dimension_form": included,
        "excluded_by_source_bucket": excluded,
    }
    (outroot / "adapter_stats.json").write_text(json.dumps(stats_payload, indent=2))
    print(f"wrote viewer data root: {outroot}")
    for k, v in sorted(included.items()):
        print(f"  {k:<40} {v:>8,}")
    print(f"  {'INCLUDED':<40} {sum(included.values()):>8,}")
    for k, v in sorted(excluded.items()):
        print(f"  {'excluded/' + k:<40} {v:>8,}")
    print(f"  {'EXCLUDED':<40} {sum(excluded.values()):>8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
