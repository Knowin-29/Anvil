#!/usr/bin/env python3
"""Build the v0616 VLabel benchmark from correct-balanced logs.

Outputs are aligned by row:
  - vlabel_input_v0616.jsonl: ms-swift inference input
  - vlabel_gt_v0616.jsonl: ground truth with typed IR summary
  - evaluable_manifest_v0616.jsonl: frozen prediction-blind denominator
  - build_report_v0616.json: counts and manifest hash
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from vlabel_ir_v0616 import manifest_entry, remap_image_path, stable_hash


DEFAULT_SOURCE = "${DATA_ROOT}/datasets/correct-balanced-train-test-v1/test.jsonl"
DEFAULT_OUT = "${GSA_EVAL_ROOT}/v0616"


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prompt_from_messages(row: Dict) -> str:
    msgs = row.get("messages") or []
    for m in msgs:
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return str(row.get("prompt", ""))


def swift_messages(row: Dict) -> List[Dict[str, str]]:
    msgs = []
    for m in row.get("messages") or []:
        role = m.get("role")
        if role == "assistant":
            continue
        if role in {"system", "user"}:
            content = str(m.get("content", ""))
            if role == "user" and "<image>" not in content:
                content = "<image>\n" + content
            msgs.append({"role": role, "content": content})
    if not msgs:
        content = str(row.get("prompt", ""))
        if "<image>" not in content:
            content = "<image>\n" + content
        msgs = [{"role": "user", "content": content}]
    return msgs


def row_task_family(row: Dict) -> str:
    qc = (row.get("tags") or {}).get("qc") or {}
    return qc.get("task_family") or "untagged"


def row_capabilities(row: Dict) -> List[str]:
    qc = (row.get("tags") or {}).get("qc") or {}
    caps = qc.get("capabilities") or []
    return caps if isinstance(caps, list) else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="v0616")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check_images", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / f"vlabel_input_{args.tag}.jsonl"
    gt_path = out_dir / f"vlabel_gt_{args.tag}.jsonl"
    manifest_path = out_dir / f"evaluable_manifest_{args.tag}.jsonl"
    report_path = out_dir / f"build_report_{args.tag}.json"

    counts = Counter()
    answer_counts = Counter()
    family_counts = Counter()
    not_eval = Counter()
    missing_images = []
    manifest_hashes = []

    with open(input_path, "w", encoding="utf-8") as fi, \
            open(gt_path, "w", encoding="utf-8") as fg, \
            open(manifest_path, "w", encoding="utf-8") as fm:
        for idx, row in enumerate(iter_jsonl(args.source)):
            if args.limit and counts["rows_seen"] >= args.limit:
                break
            counts["rows_seen"] += 1
            sample_id = row.get("id") or row.get("req_id") or f"{args.tag}_{idx}"
            images = [remap_image_path(p) for p in (row.get("images") or [])]
            if args.check_images:
                bad = [p for p in images if not os.path.isfile(p)]
                if bad:
                    missing_images.append({"index": idx, "id": sample_id, "images": bad})
                    counts["missing_image_rows"] += 1
                    continue
            prompt = prompt_from_messages(row)
            gt_answer = row.get("model_answer", "")
            task_family = row_task_family(row)
            caps = row_capabilities(row)

            manifest = manifest_entry(
                sample_id=sample_id,
                source="correct-balanced-train-test-v1/test",
                task_family_raw=task_family,
                prompt=prompt,
                gt_answer=gt_answer,
                capabilities=caps,
                original_index=idx,
            )
            manifest_hashes.append(manifest["manifest_hash"])
            answer_counts[manifest["answer_type"]] += 1
            family_counts[task_family] += 1
            if not manifest["evaluable"]:
                not_eval[manifest["not_evaluable_reason"]] += 1

            fi.write(json.dumps({"messages": swift_messages(row), "images": images}, ensure_ascii=False) + "\n")
            fg.write(json.dumps({
                "id": sample_id,
                "req_id": row.get("req_id"),
                "task_family": task_family,
                "capabilities": caps,
                "model_answer": gt_answer,
                "model": row.get("model"),
                "original_index": idx,
                "answer_type": manifest["answer_type"],
                "gt_ir": manifest["gt_ir"],
                "manifest_hash": manifest["manifest_hash"],
            }, ensure_ascii=False) + "\n")
            fm.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            counts["rows_written"] += 1

    report = {
        "tag": args.tag,
        "source": args.source,
        "input_path": str(input_path),
        "gt_path": str(gt_path),
        "manifest_path": str(manifest_path),
        "rows_seen": counts["rows_seen"],
        "rows_written": counts["rows_written"],
        "missing_image_rows": counts["missing_image_rows"],
        "answer_type_counts": dict(answer_counts),
        "task_family_counts": dict(family_counts),
        "not_evaluable_reasons": dict(not_eval),
        "manifest_set_hash": stable_hash(manifest_hashes),
        "missing_images_preview": missing_images[:20],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    print(f"-> {report_path}")


if __name__ == "__main__":
    main()
