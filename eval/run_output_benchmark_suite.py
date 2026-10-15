#!/usr/bin/env python3
"""Run embodied-arena benchmark suite for checkpoints under GSA_Forge/output."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(os.environ.get("GSA_FORGE_ROOT", "${GSA_FORGE_ROOT}"))
OUTPUT_ROOT = ROOT / "output"
RUNNER = ROOT / "eval" / "run_embodied_benchmark_offline.sh"
DEFAULT_RUN_ROOT = ROOT / "eval" / "benchmark_runs"
BASE_REGISTRY = ROOT / "eval" / "checkpoints_v0616.json"

KEY_OUTPUT_PATTERNS = [
    r"sft_27b_teacher/.*/keep_milestones/checkpoint-1000$",
    r"sft_27b_teacher/.*/keep_milestones/checkpoint-5000$",
    r"sft_27b_v7_fresh/.*/checkpoint-3000$",
    r"sft_9b_v1/.*/checkpoint-1000$",
    r"sft_9b_v1/.*/checkpoint-2000$",
    r"sft_9b_v1/.*/checkpoint-3000$",
    r"sft_9b_lora_v1/.*/v9-20260620-030752/checkpoint-12000$",
]


def safe_tag(path: Path, root: Path = OUTPUT_ROOT) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "__", str(rel).strip("/"))
    return tag.strip("_") or "checkpoint"


def is_full_checkpoint(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin")) or (path / "model.safetensors.index.json").is_file()
    return has_weights or (path / "trainer_state.json").is_file()


def is_adapter_checkpoint(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file()


def discover_output_checkpoints(output_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for config in sorted(output_root.rglob("config.json")):
        ckpt = config.parent
        if not is_full_checkpoint(ckpt):
            continue
        rows.append({"tag": safe_tag(ckpt, output_root), "model_path": str(ckpt), "source": "output"})
    for adapter_config in sorted(output_root.rglob("adapter_config.json")):
        ckpt = adapter_config.parent
        if not is_adapter_checkpoint(ckpt):
            continue
        rows.append({"tag": safe_tag(ckpt, output_root), "model_path": str(ckpt), "source": "output_adapter"})
    return rows


def key_output_filter(rows: Iterable[Dict[str, str]], output_root: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        rel = str(Path(row["model_path"]).relative_to(output_root))
        if any(re.search(pattern, rel) for pattern in KEY_OUTPUT_PATTERNS):
            out.append(row)
    return out


def load_base_rows() -> List[Dict[str, str]]:
    if not BASE_REGISTRY.is_file():
        return []
    data = json.loads(BASE_REGISTRY.read_text(encoding="utf-8"))
    rows = []
    for tag in ("base_27b", "base_9b"):
        item = data.get(tag)
        if item and Path(item["model_path"]).is_dir():
            rows.append({"tag": tag, "model_path": item["model_path"], "source": "registry"})
    return rows


def select_checkpoints(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.paths:
        rows = []
        for raw in args.paths:
            for part in raw.split(","):
                path = Path(part.strip())
                if path:
                    rows.append({"tag": safe_tag(path), "model_path": str(path), "source": "explicit"})
    else:
        output_root = Path(args.output_root)
        all_rows = discover_output_checkpoints(output_root)
        rows = key_output_filter(all_rows, output_root) if args.preset == "key-output" else all_rows
    if args.include_base:
        rows = load_base_rows() + rows
    if args.include_regex:
        rx = re.compile(args.include_regex)
        rows = [row for row in rows if rx.search(row["tag"]) or rx.search(row["model_path"])]
    if args.exclude_regex:
        rx = re.compile(args.exclude_regex)
        rows = [row for row in rows if not (rx.search(row["tag"]) or rx.search(row["model_path"]))]
    if args.max_checkpoints:
        rows = rows[: args.max_checkpoints]
    return rows


def primary_metric(results: Dict[str, Any]) -> tuple[str, Optional[float]]:
    if isinstance(results.get("overall"), (int, float)):
        return "overall", float(results["overall"])
    for key, value in results.items():
        if isinstance(value, (int, float)):
            return key, float(value)
    return "-", None


def read_task_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Optional[Path]) -> int:
    if path is None or not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def first_file(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def infer_artifacts(run_root: Path) -> Dict[str, Any]:
    infer_dir = run_root / "infer"
    pred = first_file(p for p in sorted(infer_dir.glob("*.jsonl")) if not p.name.endswith(".failed.jsonl"))
    failed = first_file(sorted(infer_dir.glob("*.failed.jsonl")))
    log_path = first_file(sorted(infer_dir.glob("*.log")))
    export_manifest = read_task_summary(run_root / "swift_infer.jsonl.manifest.json")
    return {
        "export_rows": export_manifest.get("rows"),
        "prediction_file": str(pred) if pred else None,
        "prediction_rows": count_jsonl(pred),
        "failed_file": str(failed) if failed else None,
        "failed_file_rows": count_jsonl(failed),
        "log": str(log_path) if log_path else None,
        "diagnostic": log_diagnostic(log_path),
    }


def compact_cell(value: Any, limit: int = 180) -> str:
    text = "-" if value in (None, "") else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", "\\|")
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def log_diagnostic(log_path: Optional[Path]) -> str:
    if log_path is None or not log_path.is_file():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    markers = [
        "backend_fallback",
        "init_failed",
        "not supported",
        "Qwen3_5ForConditionalGeneration",
        "out of memory",
        "cuda",
        "RuntimeError",
        "ValueError",
        "ValidationError",
        "Traceback",
        "ERROR",
    ]
    interesting = []
    for line in lines[-300:]:
        low = line.lower()
        if any(marker.lower() in low for marker in markers):
            interesting.append(line.strip())
    if interesting:
        return compact_cell(interesting[-1], 500)
    for line in reversed(lines):
        if line.strip():
            return compact_cell(line.strip(), 500)
    return ""


def summarize_run(row: Dict[str, str], run_root: Path, returncode: int) -> Dict[str, Any]:
    summary_path = run_root / "scored" / "summary.json"
    summary = read_task_summary(summary_path)
    artifacts = infer_artifacts(run_root)
    out: Dict[str, Any] = {
        "tag": row["tag"],
        "model_path": row["model_path"],
        "source": row.get("source"),
        "returncode": returncode,
        "run_root": str(run_root),
        "summary": str(summary_path) if summary_path.exists() else None,
        "prediction_stats": summary.get("prediction_stats", {}),
        **artifacts,
        "tasks": {},
    }
    for task_name, task_row in summary.get("tasks", {}).items():
        metric, score = primary_metric(task_row.get("results", {}))
        out["tasks"][task_name] = {
            "ability": task_row.get("ability"),
            "n": task_row.get("n"),
            "missing": task_row.get("missing_predictions"),
            "failed": task_row.get("failed_predictions"),
            "metric": metric,
            "score": score,
            "results": task_row.get("results", {}),
        }
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, suite: Dict[str, Any]) -> None:
    lines = [
        "# Output Benchmark Suite",
        "",
        f"- Created: {suite['created_at']}",
        f"- Task set: `{suite['task_set']}`",
        f"- Limit: `{suite['limit']}`",
        f"- Run root: `{suite['suite_root']}`",
        "",
        "| Checkpoint | Return | Task | Ability | N | Missing | Failed | Metric | Score | Pred Rows | Failed Rows | Diagnostic |",
        "|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for run in suite["runs"]:
        if not run["tasks"]:
            lines.append(
                f"| `{run['tag']}` | {run['returncode']} | `<no summary>` | - | {run.get('export_rows') or 0} | 0 | 0 | "
                f"`returncode` | {run['returncode']} | {run.get('prediction_rows') or 0} | "
                f"{run.get('failed_file_rows') or 0} | {compact_cell(run.get('diagnostic'))} |"
            )
            continue
        for task_name, task in run["tasks"].items():
            score = task["score"]
            score_text = "-" if score is None else f"{score:.4f}"
            lines.append(
                f"| `{run['tag']}` | {run['returncode']} | `{task_name}` | `{task.get('ability') or '-'}` | {task.get('n') or 0} | "
                f"{task.get('missing') or 0} | {task.get('failed') or 0} | `{task.get('metric')}` | {score_text} |"
                f" {run.get('prediction_rows') or 0} | {run.get('failed_file_rows') or 0} | {compact_cell(run.get('diagnostic'))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(path: Path, suite: Dict[str, Any]) -> None:
    rows = []
    for run in suite["runs"]:
        if not run["tasks"]:
            rows.append(
                f"<tr><td><code>{html.escape(run['tag'])}</code></td>"
                f"<td>{run['returncode']}</td><td><code>&lt;no summary&gt;</code></td><td>-</td>"
                f"<td>{run.get('export_rows') or 0}</td><td>0</td><td>0</td><td><code>returncode</code></td>"
                f"<td>{run['returncode']}</td><td>{run.get('prediction_rows') or 0}</td>"
                f"<td>{run.get('failed_file_rows') or 0}</td><td>{html.escape(compact_cell(run.get('diagnostic'), 260))}</td></tr>"
            )
            continue
        for task_name, task in run["tasks"].items():
            score = task["score"]
            score_text = "-" if score is None else f"{score:.4f}"
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(run['tag'])}</code></td>"
                f"<td>{run['returncode']}</td>"
                f"<td><code>{html.escape(task_name)}</code></td>"
                f"<td><code>{html.escape(str(task.get('ability') or '-'))}</code></td>"
                f"<td>{task.get('n') or 0}</td>"
                f"<td>{task.get('missing') or 0}</td>"
                f"<td>{task.get('failed') or 0}</td>"
                f"<td><code>{html.escape(str(task.get('metric')))}</code></td>"
                f"<td>{score_text}</td>"
                f"<td>{run.get('prediction_rows') or 0}</td>"
                f"<td>{run.get('failed_file_rows') or 0}</td>"
                f"<td>{html.escape(compact_cell(run.get('diagnostic'), 260))}</td>"
                "</tr>"
            )
    doc = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Output Benchmark Suite</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;margin:24px;line-height:1.5;color:#111827}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #d1d5db;padding:7px;text-align:left;vertical-align:top}}
th{{background:#f3f4f6;position:sticky;top:0}}
code{{background:#f3f4f6;padding:1px 4px;border-radius:3px}}
</style>
</head>
<body>
<h1>Output Benchmark Suite</h1>
<p>Created: {html.escape(suite['created_at'])}</p>
<p>Task set: <code>{html.escape(suite['task_set'])}</code>; limit: <code>{html.escape(str(suite['limit']))}</code></p>
<p>Run root: <code>{html.escape(suite['suite_root'])}</code></p>
<table>
<thead><tr><th>Checkpoint</th><th>Return</th><th>Task</th><th>Ability</th><th>N</th><th>Missing</th><th>Failed</th><th>Metric</th><th>Score</th><th>Pred Rows</th><th>Failed Rows</th><th>Diagnostic</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def run_checkpoint(row: Dict[str, str], args: argparse.Namespace, suite_root: Path) -> Dict[str, Any]:
    run_root = suite_root / row["tag"]
    env = os.environ.copy()
    env.update(
        {
            "CKPT_TAG": row["tag"],
            "MODEL_PATH": row["model_path"],
            "RUN_ROOT": str(run_root),
            "TASK_SET": args.task_set,
            "LIMIT": str(args.limit),
            "PYTHON_BIN": args.python_bin,
            "REPO_ROOT": args.ms_swift_root,
            "RESET_OUTPUT": "true" if args.reset_output else env.get("RESET_OUTPUT", "false"),
            "ALLOW_FAILED": "true" if args.allow_failed else "false",
            "ENABLE_QWEN35_VLLM_TIPS": "true" if args.qwen35_vllm_tips else "false",
        }
    )
    if args.tasks:
        env["TASKS"] = args.tasks
    if args.skip_inference:
        env["SKIP_INFERENCE"] = "true"
    print(f"[suite] running {row['tag']} -> {run_root}", flush=True)
    proc = subprocess.run(["bash", str(RUNNER)], cwd=str(ROOT), env=env, check=False)
    return summarize_run(row, run_root, proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["key-output", "all-output"], default="key-output")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--paths", action="append", help="Explicit checkpoint path(s), comma-separated or repeated.")
    parser.add_argument("--include-base", action="store_true", help="Also evaluate base_27b/base_9b from checkpoint registry.")
    parser.add_argument("--include-regex", default=None)
    parser.add_argument("--exclude-regex", default=None)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--task-set", choices=["promoted9", "profile"], default="promoted9")
    parser.add_argument("--tasks", default=None, help="Comma-separated task override.")
    parser.add_argument("--limit", default="0", help="Docs per task; 0 means full benchmark.")
    parser.add_argument("--suite-root", default=None)
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    parser.add_argument("--ms-swift-root", default=os.environ.get("REPO_ROOT", os.environ.get("MS_SWIFT_ROOT", str(ROOT / ".." / "ms-swift"))))
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument("--allow-failed", action="store_true", default=True)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--qwen35-vllm-tips", action="store_true", default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = select_checkpoints(args)
    if args.discover_only:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        raise SystemExit("no checkpoints selected")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    suite_root = Path(args.suite_root) if args.suite_root else DEFAULT_RUN_ROOT / f"output_suite_{args.task_set}_{stamp}"
    suite_root.mkdir(parents=True, exist_ok=True)
    write_json(suite_root / "manifest.json", {"checkpoints": rows, "args": vars(args)})

    runs = []
    for row in rows:
        run = run_checkpoint(row, args, suite_root)
        runs.append(run)
        if args.stop_on_error and run["returncode"] != 0:
            break

    suite = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_set": args.task_set,
        "tasks": args.tasks,
        "limit": args.limit,
        "suite_root": str(suite_root),
        "runs": runs,
    }
    write_json(suite_root / "suite_summary.json", suite)
    write_markdown(suite_root / "suite_summary.md", suite)
    write_html(suite_root / "suite_summary.html", suite)
    print(f"[suite] summary: {suite_root / 'suite_summary.md'}", flush=True)
    return 0 if all(run["returncode"] == 0 for run in runs) else 2


if __name__ == "__main__":
    sys.exit(main())
