#!/usr/bin/env python3
"""Audit empty semantics and structural-null supervision in v069 data.

This audit is intentionally focused on training-signal clarity:
  - semantic negatives such as "There are none.";
  - structural empty JSON such as {"object": null, "robot_state": {}, "distance": ""};
  - nullable optional fields that would be pruned by build_v069_pack.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ROOT = Path("${GSA_FORGE_ROOT}")
TRAINS = ROOT / "trains"
sys.path.insert(0, str(TRAINS))

from build_v069_pack import (  # noqa: E402
    assistant_text,
    has_meaningful_supervision,
    image_ok,
    iter_jsonl,
    parse_jsonish,
    prune_nullish_fields,
    semantic_negative_kind,
    user_text,
)


def default_sources() -> Dict[str, Path]:
    return {
        "base_gsa_sft_train": TRAINS / "gsa_sft_train.jsonl",
        "gsa_volc_sft_v069": TRAINS / "gsa_volc_sft_v069.jsonl",
        "pixmo_points_tag_norm1000_v3": TRAINS / "pixmo_points_tag_norm1000_v3.jsonl",
        "refcoco_grounding_tag_norm1000_v3": TRAINS / "refcoco_grounding_tag_norm1000_v3.jsonl",
        "desktop_bbox_fused_train": TRAINS / "desktop_bbox_fused_train.jsonl",
        "desktop_bbox_shortname_phrasingdedup_train": TRAINS / "desktop_bbox_shortname_phrasingdedup_train.jsonl",
        "gsa_allcorrect_train_native_v1": TRAINS / "gsa_allcorrect_train_native_v1.jsonl",
        "aobjstate_verifier_v1": TRAINS / "aobjstate_verifier_v1.jsonl",
        "gsa_train_v069": TRAINS / "gsa_train_v069.jsonl",
        "gsa_val_v069": TRAINS / "gsa_val_v069.jsonl",
    }


def scan_file(
    name: str,
    path: Path,
    limit: Optional[int],
    sample_limit: int,
    check_images: bool,
    sample_rows: list[dict[str, Any]],
) -> Dict[str, Any]:
    counters = Counter()
    neg_kinds = Counter()
    nullish_keys = Counter()
    object_null_examples = 0
    start = time.time()
    if not path.exists():
        return {"path": str(path), "missing": True}

    for line_no, rec in enumerate(iter_jsonl(path), 1):
        if limit and line_no > limit:
            break
        counters["read"] += 1
        if check_images and not image_ok(rec):
            counters["image_missing"] += 1
        text = assistant_text(rec)
        neg_kind = semantic_negative_kind(text)
        if neg_kind:
            counters["semantic_negative"] += 1
            neg_kinds[neg_kind] += 1
            if len([r for r in sample_rows if r["source"] == name and r["category"] == "semantic_negative"]) < sample_limit:
                sample_rows.append({
                    "source": name,
                    "line": line_no,
                    "category": "semantic_negative",
                    "image": (rec.get("images") or [""])[0],
                    "user": user_text(rec)[:500],
                    "assistant": text[:500],
                })

        obj = parse_jsonish(text)
        if obj is None:
            counters["assistant_non_json"] += 1
            continue
        counters["assistant_json"] += 1
        if isinstance(obj, dict):
            _, prune_stats = prune_nullish_fields(obj)
            for key, value in prune_stats.items():
                key = str(key)
                if key.startswith("nullish_removed:"):
                    nullish_keys[key.split(":", 1)[1]] += int(value)
            if "object" in obj and obj.get("object") is None:
                counters["object_null"] += 1
                object_null_examples += 1
            if not has_meaningful_supervision(obj):
                counters["structural_empty_json"] += 1
                if len([r for r in sample_rows if r["source"] == name and r["category"] == "structural_empty_json"]) < sample_limit:
                    sample_rows.append({
                        "source": name,
                        "line": line_no,
                        "category": "structural_empty_json",
                        "image": (rec.get("images") or [""])[0],
                        "user": user_text(rec)[:500],
                        "assistant": text[:500],
                    })

    return {
        "path": str(path),
        "limit": limit,
        "elapsed_sec": round(time.time() - start, 2),
        "counters": dict(counters),
        "semantic_negative_kinds": dict(neg_kinds.most_common()),
        "top_nullish_keys": dict(nullish_keys.most_common(30)),
        "object_null_examples_seen": object_null_examples,
    }


def parse_source_arg(value: str) -> tuple[str, Path, Optional[int]]:
    parts = value.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("source must be name:path[:limit]")
    limit = int(parts[2]) if len(parts) == 3 and parts[2] else None
    return parts[0], Path(parts[1]), limit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=TRAINS / "gsa_v069_empty_semantics_audit.json")
    ap.add_argument("--samples-jsonl", type=Path, default=ROOT / "trains/diag/out/gsa_v069_empty_semantics_samples.jsonl")
    ap.add_argument("--merged-sample", type=int, default=500_000)
    ap.add_argument("--sample-limit", type=int, default=8)
    ap.add_argument("--check-images", action="store_true")
    ap.add_argument(
        "--source",
        action="append",
        type=parse_source_arg,
        help="Override sources. Format: name:path[:limit]. May be repeated.",
    )
    args = ap.parse_args()

    if args.source:
        sources = {name: (path, limit) for name, path, limit in args.source}
    else:
        sources = {}
        for name, path in default_sources().items():
            limit = args.merged_sample if name == "gsa_train_v069" else None
            sources[name] = (path, limit)

    sample_rows: list[dict[str, Any]] = []
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy_note": (
            "Semantic negatives are useful for absence calibration, but should be capped by source. "
            "Structural-empty JSON is treated as dirty data unless another field carries real supervision."
        ),
        "sources": {},
    }
    for name, (path, limit) in sources.items():
        report["sources"][name] = scan_file(name, path, limit, args.sample_limit, args.check_images, sample_rows)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.samples_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.samples_jsonl.open("w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "out_json": str(args.out_json),
        "samples_jsonl": str(args.samples_jsonl),
        "sources": {k: v.get("counters", {}) for k, v in report["sources"].items()},
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
