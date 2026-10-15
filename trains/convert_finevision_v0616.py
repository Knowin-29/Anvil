#!/usr/bin/env python3
"""Convert selected FineVision subsets into v0616 ms-swift shards.

This is a local-parquet adapter only.  It does not download FineVision.  The
current machine does not expose the documented FineVision root, so this script
is intentionally safe to run as a path/status check until parquet files are
mounted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "data" / "sources_registry_v0616.yaml"


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_subset_files(root: Path, subset: str) -> List[Path]:
    candidates = []
    for pattern in [f"**/{subset}/**/*.parquet", f"**/{subset}*.parquet"]:
        candidates.extend(root.glob(pattern))
    return sorted(set(candidates))


def image_placeholder(images: Any) -> List[str]:
    """FineVision stores image bytes/PIL objects in parquet.

    A full image materialization policy is intentionally not guessed here.  Once
    the local root is mounted, this function should be extended to write images
    to ${GSA_TRAIN_ROOT}/v0616/finevision_images and return those paths.
    """
    return []


def text_turns(row: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    texts = row.get("texts")
    if isinstance(texts, list):
        for t in texts:
            if isinstance(t, dict):
                user = t.get("user") or t.get("question") or t.get("prompt")
                assistant = t.get("assistant") or t.get("answer") or t.get("response")
                if user and assistant:
                    yield {"user": str(user), "assistant": str(assistant)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--finevision_root", default=None)
    ap.add_argument("--out_dir", default="${GSA_TRAIN_ROOT}/v0616/finevision_selected")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--max_rows_per_subset", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(args.registry)
    fv = cfg.get("finevision", {})
    root = Path(args.finevision_root or fv.get("local_root", ""))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "registry": args.registry,
        "finevision_root": str(root),
        "root_exists": root.exists(),
        "dry_run": args.dry_run,
        "subsets": {},
        "note": "No network download is attempted. Mount local parquet root before conversion.",
    }
    if not root.exists():
        report_path = out_dir / "finevision_v0616_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"-> {report_path}")
        return

    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        report["error"] = f"pyarrow unavailable: {exc}"
        report_path = out_dir / "finevision_v0616_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    for subset in fv.get("selected_subsets", []):
        name = subset["name"]
        files = find_subset_files(root, name)
        stats = {"files": [str(p) for p in files], "rows_seen": 0, "records_written": 0, "status": "ok" if files else "missing"}
        out_path = out_dir / f"{name}_v0616.jsonl"
        if files and not args.dry_run:
            with open(out_path, "w", encoding="utf-8") as out:
                for path in files:
                    table = pq.read_table(path)
                    for row in table.to_pylist():
                        stats["rows_seen"] += 1
                        if args.max_rows_per_subset and stats["rows_seen"] > args.max_rows_per_subset:
                            break
                        images = image_placeholder(row.get("images"))
                        if not images:
                            continue
                        for turn in text_turns(row):
                            rec = {
                                "messages": [
                                    {"role": "user", "content": "<image>\n" + turn["user"]},
                                    {"role": "assistant", "content": turn["assistant"]},
                                ],
                                "images": images,
                                "v0616_source": f"finevision:{name}",
                                "v0616_bucket": subset.get("bucket", "general_vqa"),
                            }
                            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            stats["records_written"] += 1
                    if args.max_rows_per_subset and stats["rows_seen"] > args.max_rows_per_subset:
                        break
        report["subsets"][name] = stats

    report_path = out_dir / "finevision_v0616_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    print(f"-> {report_path}")


if __name__ == "__main__":
    main()
