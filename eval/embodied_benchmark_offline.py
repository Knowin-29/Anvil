#!/usr/bin/env python3
"""Offline bridge from embodied-arena tasks to MS-Swift inference results.

This script intentionally lives in GSA_Forge and imports embodied-arena as a
read-only dependency. It does not modify benchmark files.

Workflow:
  1. export: task YAML -> MS-Swift JSONL with task/doc metadata.
  2. score: Swift/vLLM result JSONL -> benchmark-native samples/results files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Resolve roots from environment so the desensitized snapshot is runnable.
# Falls back to the literal placeholder only when the env var is unset.
REPO_ROOT = Path(os.environ.get("GSA_FORGE_ROOT", "${GSA_FORGE_ROOT}"))
_BENCH_ROOT_ENV = os.environ.get("BENCHMARK_ROOT", "${BENCHMARK_ROOT}")
DEFAULT_BENCH_ROOT = Path(_BENCH_ROOT_ENV) / "embodied-arena"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "eval" / "benchmark_runs"

PROMOTED9_TASKS = [
    "internal_A3_binary_v3_holdout_candidate",
    "internal_G1_point_v3_phase15_candidate",
    "internal_G1_bbox_v3_phase15_candidate",
    "internal_G2_point_v3_phase15_candidate",
    "internal_G2_bbox_v3_phase15_candidate",
    "internal_G3_point_v31_holdout_candidate",
    "internal_G5_count_v31_holdout_promoted",
    "internal_S1_choice_v3_phase15_candidate",
    "internal_S3_same_substrate_v3_promoted",
]

PROFILE_TASKS = [
    "internal_G1_point_table_freeze_candidate",
    "internal_G2_point_table_freeze_candidate",
    "internal_G3_point_table_freeze_candidate",
    "internal_G4_point",
    "internal_G5_count_table_freeze_candidate",
    "internal_S1_choice_table_freeze_candidate",
    "internal_S2_choice",
    "internal_S3_choice_table_freeze_candidate",
    "internal_S4_multihop",
    "internal_A1_recognition",
    "internal_A2_attribute",
    "internal_A3_choice_table_freeze_candidate",
    "internal_R3_open_table_freeze_candidate",
]


def task_ability(task_name: str) -> str:
    parts = task_name.split("_")
    for part in parts:
        if part in {"G1", "G2", "G3", "G4", "G5", "S1", "S2", "S3", "S4", "A1", "A2", "A3", "R3"}:
            if "bbox" in parts:
                return f"{part}_bbox"
            if "point" in parts:
                return f"{part}_point"
            if "count" in parts:
                return f"{part}_count"
            if "choice" in parts:
                return f"{part}_choice"
            if "binary" in parts:
                return f"{part}_binary"
            return part
    return "unknown"


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def dump_json(path: Path, data: Any, *, indent: Optional[int] = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=json_default)
        f.write("\n")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
            n += 1
    return n


def setup_benchmark_import(bench_root: Path) -> None:
    root = bench_root.resolve()
    if not (root / "embodied_eval").is_dir():
        raise FileNotFoundError(f"embodied-arena root not found: {root}")
    sys.path.insert(0, str(root))


def parse_tasks(task_set: str, tasks: Optional[str]) -> List[str]:
    if tasks:
        return [x.strip() for x in tasks.split(",") if x.strip()]
    if task_set == "promoted9":
        return list(PROMOTED9_TASKS)
    if task_set == "profile":
        return list(PROFILE_TASKS)
    raise ValueError(f"unknown task set: {task_set}")


def get_task_objects(bench_root: Path, task_names: Sequence[str]):
    setup_benchmark_import(bench_root)
    from embodied_eval.tasks import TaskManager, get_task_dict  # type: ignore

    manager = TaskManager()
    missing = [t for t in task_names if t not in manager.all_tasks]
    if missing:
        raise ValueError(f"tasks not found in embodied-arena: {missing}")
    return get_task_dict(list(task_names), manager)


def call_doc_to_target(task: Any, doc: Dict[str, Any]) -> str:
    fn = getattr(task.config, "doc_to_target", None)
    kwargs = getattr(task.config, "dataset_kwargs", None)
    if not callable(fn):
        return ""
    try:
        value = fn(doc, kwargs) if kwargs is not None else fn(doc)
    except TypeError:
        value = fn(doc)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=json_default)


def extract_visual_paths(doc: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    images: List[str] = []
    videos: List[str] = []

    def add_path(value: Any) -> None:
        if not value:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                add_path(item)
            return
        if isinstance(value, dict):
            add_path(value.get("path") or value.get("image") or value.get("video"))
            return
        p = str(value)
        lower = p.lower()
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv")):
            videos.append(p)
        elif lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")):
            images.append(p)

    add_path(doc.get("image"))
    add_path(doc.get("images"))
    add_path(doc.get("video"))
    add_path(doc.get("videos"))

    # Deduplicate while preserving order.
    images = list(OrderedDict((x, None) for x in images).keys())
    videos = list(OrderedDict((x, None) for x in videos).keys())
    return images, videos


def build_user_content(prompt: str, num_images: int, num_videos: int) -> str:
    markers: List[str] = []
    if "<image>" not in prompt:
        markers.extend(["<image>"] * num_images)
    if "<video>" not in prompt:
        markers.extend(["<video>"] * num_videos)
    if markers:
        return "\n".join(markers + [prompt])
    return prompt


def export_tasks(args: argparse.Namespace) -> None:
    bench_root = Path(args.bench_root)
    task_names = parse_tasks(args.task_set, args.tasks)
    task_dict = get_task_objects(bench_root, task_names)
    output = Path(args.output)
    limit = int(args.limit) if args.limit is not None else None

    def rows() -> Iterable[Dict[str, Any]]:
        for task_name, task in task_dict.items():
            max_docs = len(task.eval_docs) if limit is None else min(limit, len(task.eval_docs))
            for doc_id in range(max_docs):
                doc = dict(task.eval_docs[doc_id])
                prompt = task.doc_to_text(doc)
                images, videos = extract_visual_paths(doc)
                label = call_doc_to_target(task, doc)
                content = build_user_content(prompt, len(images), len(videos))
                row: Dict[str, Any] = {
                    "id": f"{task_name}::{doc_id}",
                    "task_name": task_name,
                    "doc_id": doc_id,
                    "prompt": prompt,
                    "messages": [
                        {"role": "user", "content": content},
                        {"role": "assistant", "content": label},
                    ],
                    "benchmark": "embodied-arena",
                    "swift_infer_format": "ms-swift/messages+images/offline-benchmark",
                }
                if images:
                    row["images"] = images
                if videos:
                    row["videos"] = videos
                # Keep light metadata for debugging; the scorer reloads canonical docs.
                for key in ("image", "images", "video", "metadata", "id", "query", "question"):
                    if key in doc and key not in row:
                        row[f"doc_{key}"] = doc[key]
                yield row

    count = write_jsonl(output, rows())
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bench_root": str(bench_root),
        "task_set": args.task_set,
        "tasks": task_names,
        "limit": limit,
        "rows": count,
        "output": str(output),
    }
    dump_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_predictions(path: Path) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], Dict[str, int]]:
    preds: Dict[Tuple[str, int], Dict[str, Any]] = {}
    stats = {"rows": 0, "ok": 0, "failed": 0, "missing_keys": 0}
    for row in iter_jsonl(path):
        stats["rows"] += 1
        task_name = row.get("task_name")
        doc_id = row.get("doc_id")
        if task_name is None or doc_id is None:
            stats["missing_keys"] += 1
            continue
        key = (str(task_name), int(doc_id))
        if row.get("__ok__", True) and row.get("response") is not None:
            stats["ok"] += 1
            preds[key] = row
        else:
            stats["failed"] += 1
            preds.setdefault(key, row)
    return preds, stats


def make_sample(task: Any, task_name: str, doc_id: int, doc: Dict[str, Any], prompt: str, response: str) -> Dict[str, Any]:
    sample_result = task.process_results(doc, [[response]])
    sample: Dict[str, Any] = {
        "doc_id": doc_id,
        "doc": prompt,
        "resps": [[response]],
    }
    for key in ["image", "images", "video"]:
        if doc.get(key) is not None:
            sample[key] = doc[key]
    sample.update(sample_result or {})
    sample.setdefault("task_name", task_name)
    return sample


def score_predictions(args: argparse.Namespace) -> None:
    bench_root = Path(args.bench_root)
    pred_path = Path(args.pred)
    out_dir = Path(args.output_dir)
    task_names = parse_tasks(args.task_set, args.tasks)
    task_dict = get_task_objects(bench_root, task_names)
    preds, pred_stats = load_predictions(pred_path)
    limit = int(args.limit) if args.limit is not None else None

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bench_root": str(bench_root),
        "prediction_file": str(pred_path),
        "task_set": args.task_set,
        "prediction_stats": pred_stats,
        "tasks": {},
    }

    for task_name, task in task_dict.items():
        samples: List[Dict[str, Any]] = []
        missing = 0
        failed = 0
        max_docs = len(task.eval_docs) if limit is None else min(limit, len(task.eval_docs))
        for doc_id in range(max_docs):
            doc = dict(task.eval_docs[doc_id])
            prompt = task.doc_to_text(doc)
            pred_row = preds.get((task_name, doc_id))
            if not pred_row:
                missing += 1
                response = ""
            elif not pred_row.get("__ok__", True):
                failed += 1
                response = ""
            else:
                response = str(pred_row.get("response", ""))
            samples.append(make_sample(task, task_name, doc_id, doc, prompt, response))

        results = task.aggregate_results(samples)
        config = task.dump_config()
        samples_path = out_dir / f"samples_{task_name}.json"
        results_path = out_dir / f"results_{task_name}.json"
        configs_path = out_dir / f"configs_{task_name}.json"
        write_jsonl(samples_path, samples)
        dump_json(results_path, results)
        dump_json(configs_path, config)
        summary["tasks"][task_name] = {
            "ability": task_ability(task_name),
            "n": len(samples),
            "missing_predictions": missing,
            "failed_predictions": failed,
            "results": results,
            "samples": str(samples_path),
            "results_file": str(results_path),
        }

    dump_json(out_dir / "summary.json", summary)
    write_ability_summary(out_dir / "ability_summary.json", summary)
    write_markdown_summary(out_dir / "summary.md", summary)
    write_html_summary(out_dir / "summary.html", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


def primary_metric(results: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    if "overall" in results and isinstance(results["overall"], (int, float)):
        return "overall", float(results["overall"])
    for key, value in results.items():
        if isinstance(value, (int, float)):
            return key, float(value)
    return "-", None


def write_ability_summary(path: Path, summary: Dict[str, Any]) -> None:
    buckets: Dict[str, Dict[str, Any]] = {}
    for task_name, row in summary["tasks"].items():
        ability = row.get("ability") or task_ability(task_name)
        bucket = buckets.setdefault(ability, {"tasks": 0, "n": 0, "missing": 0, "failed": 0, "overall_scores": []})
        bucket["tasks"] += 1
        bucket["n"] += int(row.get("n", 0))
        bucket["missing"] += int(row.get("missing_predictions", 0))
        bucket["failed"] += int(row.get("failed_predictions", 0))
        metric, score = primary_metric(row.get("results", {}))
        if metric == "overall" and score is not None:
            bucket["overall_scores"].append(score)
    for bucket in buckets.values():
        scores = bucket.pop("overall_scores")
        bucket["mean_overall"] = (sum(scores) / len(scores)) if scores else None
    dump_json(path, buckets)


def write_markdown_summary(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Embodied Benchmark Offline Summary",
        "",
        f"- Created: {summary.get('created_at')}",
        f"- Prediction file: `{summary.get('prediction_file')}`",
        f"- Task set: `{summary.get('task_set')}`",
        "",
        "| Ability | Task | N | Missing | Failed | Metric | Score |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for task_name, row in summary["tasks"].items():
        metric, score = primary_metric(row.get("results", {}))
        score_text = "-" if score is None else f"{score:.4f}"
        lines.append(
            f"| `{row.get('ability', task_ability(task_name))}` | `{task_name}` | {row['n']} | {row['missing_predictions']} | "
            f"{row['failed_predictions']} | `{metric}` | {score_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_summary(path: Path, summary: Dict[str, Any]) -> None:
    rows = []
    for task_name, row in summary["tasks"].items():
        metric, score = primary_metric(row.get("results", {}))
        score_text = "-" if score is None else f"{score:.4f}"
        rows.append(
            "<tr>"
            f"<td><code>{row.get('ability', task_ability(task_name))}</code></td>"
            f"<td><code>{task_name}</code></td>"
            f"<td>{row['n']}</td>"
            f"<td>{row['missing_predictions']}</td>"
            f"<td>{row['failed_predictions']}</td>"
            f"<td><code>{metric}</code></td>"
            f"<td>{score_text}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang=\"zh\">
<head>
<meta charset=\"utf-8\">
<title>Embodied Benchmark Offline Summary</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;margin:24px;line-height:1.5;color:#111827}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #d1d5db;padding:8px;text-align:left}}
th{{background:#f3f4f6}}
code{{background:#f3f4f6;padding:1px 4px;border-radius:3px}}
</style>
</head>
<body>
<h1>Embodied Benchmark Offline Summary</h1>
<p>Created: {summary.get('created_at')}</p>
<p>Prediction file: <code>{summary.get('prediction_file')}</code></p>
<p>Task set: <code>{summary.get('task_set')}</code></p>
<table>
<thead><tr><th>Ability</th><th>Task</th><th>N</th><th>Missing</th><th>Failed</th><th>Metric</th><th>Score</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def list_tasks(args: argparse.Namespace) -> None:
    tasks = parse_tasks(args.task_set, args.tasks)
    print("\n".join(tasks))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bench-root", default=str(DEFAULT_BENCH_ROOT))
    common.add_argument("--task-set", choices=["promoted9", "profile"], default="promoted9")
    common.add_argument("--tasks", default=None, help="Comma-separated task names; overrides --task-set.")
    common.add_argument("--limit", default=None, help="Max docs per task for smoke/debug.")

    p = sub.add_parser("list-tasks", parents=[common])
    p.set_defaults(func=list_tasks)

    p = sub.add_parser("export", parents=[common])
    p.add_argument("--output", required=True, help="Output MS-Swift JSONL path.")
    p.set_defaults(func=export_tasks)

    p = sub.add_parser("score", parents=[common])
    p.add_argument("--pred", required=True, help="Swift/vLLM result JSONL.")
    p.add_argument("--output-dir", required=True, help="Directory for samples/results summary.")
    p.set_defaults(func=score_predictions)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
