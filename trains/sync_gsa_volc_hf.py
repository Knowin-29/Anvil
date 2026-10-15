#!/usr/bin/env python3
"""Manifest-driven sync for YOUR_ORG/Anvil-Data.

The HF repo has too many files for recursive tree listing to be reliable in this
environment. This script avoids directory downloads and downloads explicit files
listed in metadata/image_entries.jsonl instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import hf_hub_download


REPO_ID = os.environ.get("ANVIL_HF_DATASET", "YOUR_ORG/Anvil-Data")
REPO_TYPE = "dataset"
DEFAULT_ROOT = Path(os.environ.get("HF_DOWNLOAD_ROOT", "data/downloads")) / "GSA_volc"

CORE_FILES = [
    "README.md",
    ".gitattributes",
    "GSA_volc_数据质量评估报告.html",
    "metadata/manifest.json",
    "metadata/image_manifest.json",
    "metadata/image_entries.jsonl",
    "metadata/training_manifest.json",
    "metadata/source_roots.json",
    "image_mapping.json",
    "annotations/sft_train.jsonl",
    "annotations/grpo_train.jsonl",
    "train/sft_train.jsonl",
    "train/grpo_train.jsonl",
    "train/bbox_grounding_1m/stats.json",
    "train/bbox_grounding_1m/sft_train.jsonl",
    "train/bbox_grounding_1m/grpo_train.jsonl",
]


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def download_one(filename: str, root: Path, token: str, force: bool = False) -> tuple[str, bool, str]:
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE,
            local_dir=str(root),
            token=token,
            force_download=force,
        )
        return filename, True, ""
    except Exception as e:  # keep sync going and summarize failures.
        return filename, False, repr(e)


def iter_image_entries(root: Path) -> Iterable[str]:
    path = root / "metadata" / "image_entries.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            p = obj.get("image_path")
            if isinstance(p, str) and p:
                yield p


def read_failures(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            p = obj.get("path")
            if isinstance(p, str):
                out.append(p)
    return out


def sync_files(files: List[str], root: Path, token: str, workers: int, force: bool, fail_path: Path) -> dict:
    ok = fail = done = 0
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w"
    t0 = time.time()
    with fail_path.open(mode, encoding="utf-8") as fail_f:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(download_one, p, root, token, force) for p in files]
            for fut in as_completed(futures):
                p, success, err = fut.result()
                done += 1
                if success:
                    ok += 1
                else:
                    fail += 1
                    fail_f.write(json.dumps({"path": p, "error": err}, ensure_ascii=False) + "\n")
                    fail_f.flush()
                if done == 1 or done % 1000 == 0 or done == len(files):
                    rate = done / max(1.0, time.time() - t0)
                    log(f"progress {done}/{len(files)} ok={ok} fail={fail} rate={rate:.1f}/s")
    return {"requested": len(files), "ok": ok, "failed": fail, "failure_log": str(fail_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force-core", action="store_true")
    ap.add_argument("--retry-failures", type=Path)
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("HF_TOKEN is required", file=sys.stderr)
        raise SystemExit(2)

    args.root.mkdir(parents=True, exist_ok=True)
    log(f"sync root={args.root}")

    log("sync core files")
    core_report = sync_files(
        CORE_FILES,
        args.root,
        token,
        workers=min(args.workers, 8),
        force=args.force_core,
        fail_path=args.root / "sync_logs" / "core_failures.jsonl",
    )

    image_report = {"requested": 0, "ok": 0, "failed": 0, "skipped_existing": 0}
    if not args.skip_images:
        if args.retry_failures:
            images = read_failures(args.retry_failures)
            log(f"retry failures from {args.retry_failures}: {len(images)}")
        else:
            all_images = list(dict.fromkeys(iter_image_entries(args.root)))
            images = [p for p in all_images if not (args.root / p).is_file()]
            image_report["skipped_existing"] = len(all_images) - len(images)
            log(f"image manifest total={len(all_images)} missing={len(images)} existing={image_report['skipped_existing']}")
        if args.limit:
            images = images[: args.limit]
        if images:
            image_report.update(
                sync_files(
                    images,
                    args.root,
                    token,
                    workers=args.workers,
                    force=False,
                    fail_path=args.root / "sync_logs" / "image_failures.jsonl",
                )
            )

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_id": REPO_ID,
        "root": str(args.root),
        "core": core_report,
        "images": image_report,
    }
    out = args.root / "sync_logs" / "sync_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"report -> {out}")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
