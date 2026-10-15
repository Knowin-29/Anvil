#!/usr/bin/env python3
"""Run resumable overnight A_action_interface full-loop chunks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.data_flywheel.gsa.a_action_interface_review import read_jsonl, write_jsonl


DEFAULT_OUTPUT_ROOT = Path("data/output/gsa_v0/a_action_interface_overnight")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-images", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--max-runtime-hours", type=float, default=12.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default="/path/to/environment/env/rex/bin/python")
    parser.add_argument("--max-objects-per-image", type=int, default=3)
    parser.add_argument("--max-bbox-area-ratio", type=float, default=0.45)
    parser.add_argument("--enable-json-repair", action="store_true")
    parser.add_argument("--force-chunks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    _load_env(ROOT / ".env")

    plan = _build_chunk_plan(args)
    if args.dry_run:
        print(json.dumps({"output_root": str(args.output_root), "chunks": plan}, ensure_ascii=False, indent=2))
        return 0

    _write_status(args.output_root, {"state": "running", "started_at": _now(), "plan": plan})
    start_time = time.time()
    deadline = start_time + args.max_runtime_hours * 3600

    print(
        f"[overnight] start target_images={args.target_images} chunk_size={args.chunk_size} "
        f"image_offset={args.image_offset} output_root={args.output_root}",
        flush=True,
    )

    for chunk in plan:
        if time.time() >= deadline:
            print("[overnight] max runtime reached before starting next chunk", flush=True)
            break
        chunk_dir = args.output_root / chunk["chunk_name"]
        done_path = chunk_dir / "DONE"
        if done_path.exists() and not args.force_chunks:
            print(f"[overnight] skip {chunk['chunk_name']} already DONE", flush=True)
            continue
        exit_code = _run_chunk(args, chunk)
        _merge_outputs(args.output_root)
        if exit_code != 0:
            print(f"[overnight] stop after failed chunk {chunk['chunk_name']} exit={exit_code}", flush=True)
            _write_status(args.output_root, {"state": "failed", "failed_chunk": chunk, "updated_at": _now()})
            return exit_code

    summary = _merge_outputs(args.output_root)
    state = "complete" if summary["done_chunks"] >= len(plan) else "paused"
    _write_status(args.output_root, {"state": state, "updated_at": _now(), "summary": summary, "plan": plan})
    print(f"[overnight] {state} {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return 0


def _build_chunk_plan(args: argparse.Namespace) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    remaining = args.target_images
    chunk_idx = 0
    while remaining > 0:
        count = min(args.chunk_size, remaining)
        offset = args.image_offset + chunk_idx * args.chunk_size
        chunks.append(
            {
                "chunk_index": chunk_idx,
                "chunk_name": f"chunk_{chunk_idx:03d}",
                "image_offset": offset,
                "max_images": count,
            }
        )
        remaining -= count
        chunk_idx += 1
    return chunks


def _run_chunk(args: argparse.Namespace, chunk: Dict[str, Any]) -> int:
    chunk_dir = args.output_root / chunk["chunk_name"]
    chunk_dir.mkdir(parents=True, exist_ok=True)
    running_path = chunk_dir / "RUNNING"
    failed_path = chunk_dir / "FAILED"
    done_path = chunk_dir / "DONE"
    log_path = chunk_dir / "chunk.log"
    summary_path = chunk_dir / "chunk_summary.json"

    failed_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)
    running_path.write_text(_now(), encoding="utf-8")
    started = time.time()

    cmd = [
        args.python,
        "scripts/run_a_action_interface_full_loop.py",
        "--max-images",
        str(chunk["max_images"]),
        "--image-offset",
        str(chunk["image_offset"]),
        "--run-dir",
        str(chunk_dir),
        "--max-objects-per-image",
        str(args.max_objects_per_image),
        "--max-bbox-area-ratio",
        str(args.max_bbox_area_ratio),
    ]
    if args.enable_json_repair:
        cmd.append("--enable-json-repair")
    if args.force_chunks:
        cmd.append("--force-run")

    print(
        f"[overnight] run {chunk['chunk_name']} offset={chunk['image_offset']} "
        f"images={chunk['max_images']}",
        flush=True,
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== START {chunk['chunk_name']} { _now() } ===\n")
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)

    duration = round(time.time() - started, 2)
    chunk_summary = {
        **chunk,
        "run_dir": str(chunk_dir),
        "exit_code": proc.returncode,
        "duration_seconds": duration,
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "ended_at": _now(),
        "metrics": _chunk_metrics(chunk_dir),
    }
    summary_path.write_text(json.dumps(chunk_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    running_path.unlink(missing_ok=True)

    if proc.returncode == 0:
        done_path.write_text(_now(), encoding="utf-8")
        print(
            f"[overnight] done {chunk['chunk_name']} duration={duration}s "
            f"candidates={chunk_summary['metrics']['candidate_count']}",
            flush=True,
        )
    else:
        failed_path.write_text(_now(), encoding="utf-8")
        print(f"[overnight] failed {chunk['chunk_name']} exit={proc.returncode}", flush=True)
    return proc.returncode


def _chunk_metrics(chunk_dir: Path) -> Dict[str, Any]:
    report_path = chunk_dir / "reports" / "a_action_interface_promotion_report.json"
    report = _read_json(report_path) or {}
    selected = read_jsonl(chunk_dir / "selected_images.jsonl")
    labels = read_jsonl(chunk_dir / "labels" / "A_action_interface" / "A_action_interface.jsonl")
    return {
        "selected_image_count": len(selected),
        "candidate_count": int(report.get("candidate_count", len(labels))),
        "stable_training_label_count": int(report.get("stable_training_label_count", 0)),
        "source_counts": report.get("source_counts", {}),
        "heuristic_candidate_count": int(report.get("heuristic_candidate_count", 0)),
    }


def _merge_outputs(output_root: Path) -> Dict[str, Any]:
    chunk_dirs = sorted(path for path in output_root.glob("chunk_*") if path.is_dir())
    merged_candidates: List[Dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    selected_images = 0
    stable_count = 0
    done_chunks = 0
    failed_chunks = 0

    for chunk_dir in chunk_dirs:
        if (chunk_dir / "DONE").exists():
            done_chunks += 1
        if (chunk_dir / "FAILED").exists():
            failed_chunks += 1
        selected_images += len(read_jsonl(chunk_dir / "selected_images.jsonl"))
        candidates = read_jsonl(
            chunk_dir / "review_queue" / "A_action_interface" / "A_action_interface_candidates.jsonl"
        )
        merged_candidates.extend(candidates)
        for row in candidates:
            source_counts[str(row.get("source_provenance", ""))] += 1
            object_counts[str(row.get("object_name", ""))] += 1
            action_counts[str(row.get("action_type", ""))] += 1
        stable_count += len(read_jsonl(chunk_dir / "packs" / "A_action_interface_pack_manifest.jsonl"))

    merged_path = output_root / "merged_A_action_interface_candidates.jsonl"
    write_jsonl(merged_path, merged_candidates)
    summary = {
        "updated_at": _now(),
        "chunk_count": len(chunk_dirs),
        "done_chunks": done_chunks,
        "failed_chunks": failed_chunks,
        "selected_image_count": selected_images,
        "candidate_count": len(merged_candidates),
        "stable_training_label_count": stable_count,
        "source_counts": dict(source_counts),
        "heuristic_candidate_count": source_counts.get("adapter", 0),
        "top_object_names": object_counts.most_common(30),
        "action_counts": dict(action_counts),
        "merged_candidates_path": str(merged_path),
    }
    (output_root / "overnight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown_report(output_root / "overnight_quality_report.md", summary)
    return summary


def _write_markdown_report(path: Path, summary: Dict[str, Any]) -> None:
    source_counts = summary.get("source_counts", {})
    candidate_count = int(summary.get("candidate_count", 0))
    vlm_count = int(source_counts.get("vlm", 0))
    adapter_count = int(source_counts.get("adapter", 0))
    vlm_rate = (vlm_count / candidate_count) if candidate_count else 0.0
    adapter_rate = (adapter_count / candidate_count) if candidate_count else 0.0
    lines = [
        "# A_action_interface Overnight Quality Report",
        "",
        f"更新时间：{summary.get('updated_at')}",
        "",
        "## Summary",
        "",
        f"- Images selected: `{summary.get('selected_image_count', 0)}`",
        f"- Candidates: `{candidate_count}`",
        f"- Stable training labels: `{summary.get('stable_training_label_count', 0)}`",
        f"- VLM candidates: `{vlm_count}` ({vlm_rate:.1%})",
        f"- Heuristic fallback candidates: `{adapter_count}` ({adapter_rate:.1%})",
        f"- Done chunks: `{summary.get('done_chunks', 0)}`",
        f"- Failed chunks: `{summary.get('failed_chunks', 0)}`",
        "",
        "## Top Object Names",
        "",
        "| object_name | count |",
        "|---|---:|",
    ]
    for name, count in summary.get("top_object_names", [])[:30]:
        lines.append(f"| {name or '-'} | {count} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `stable_training_label` remains 0 unless human review decisions approve labels.",
            "- `adapter` means heuristic fallback and must not be promoted automatically.",
            "- Use the merged candidate file for batch review triage.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_status(output_root: Path, payload: Dict[str, Any]) -> None:
    existing = _read_json(output_root / "overnight_status.json") or {}
    existing.update(payload)
    (output_root / "overnight_status.json").write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
