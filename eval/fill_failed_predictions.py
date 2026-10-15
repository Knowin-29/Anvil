#!/usr/bin/env python3
"""Append failed prediction rows for missing JSONL inference outputs.

This is a small safety net for eval runners. If model initialization fails
before MS-Swift writes predictions, downstream scorers still receive one row per
input item with ``__ok__=false`` instead of producing an empty report.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if line.strip():
                yield index, json.loads(line)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def row_key(index: int, row: Dict[str, Any]) -> str:
    task_name = row.get("task_name")
    doc_id = row.get("doc_id")
    if task_name is not None and doc_id is not None:
        return f"task::{task_name}::{doc_id}"
    if row.get("__index__") is not None:
        return f"index::{row['__index__']}"
    return f"index::{index}"


def existing_keys(path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not path.exists():
        return keys
    for index, row in iter_jsonl(path):
        keys.add(row_key(index, row))
    return keys


def split_label(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    data = copy.deepcopy(raw)
    labels = data.get("labels") or data.get("model_answer") or data.get("label")
    messages = data.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "assistant":
            labels = last.get("content", labels)
            data["messages"] = messages[:-1]
    return data, None if labels is None else str(labels)


def build_failed_row(
    index: int,
    raw: Dict[str, Any],
    *,
    error_type: str,
    error_message: str,
    log_path: str = "",
) -> Dict[str, Any]:
    data, labels = split_label(raw)
    row = {
        **data,
        "__index__": index,
        "__ok__": False,
        "response": "",
        "labels": labels,
        "error_type": error_type,
        "error_message": error_message,
        "inference_log": log_path,
    }
    return row


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def mark_missing_failed(
    *,
    source: Path,
    pred: Path,
    failed: Optional[Path] = None,
    error_type: str = "InferenceRunnerFailed",
    error_message: str = "inference runner failed before writing this row",
    log_path: str = "",
) -> Dict[str, Any]:
    keys = existing_keys(pred)
    rows: List[Dict[str, Any]] = []
    source_rows = 0
    skipped_existing = 0
    for index, raw in iter_jsonl(source):
        source_rows += 1
        key = row_key(index, raw)
        if key in keys:
            skipped_existing += 1
            continue
        rows.append(
            build_failed_row(
                index,
                raw,
                error_type=error_type,
                error_message=error_message,
                log_path=log_path,
            )
        )
    append_jsonl(pred, rows)
    if failed is not None:
        append_jsonl(failed, rows)
    return {
        "event": "mark_missing_failed",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(source),
        "pred": str(pred),
        "failed": str(failed) if failed else None,
        "source_rows": source_rows,
        "existing_prediction_rows": len(keys),
        "skipped_existing": skipped_existing,
        "appended_failed": len(rows),
        "result_rows": count_jsonl(pred),
        "error_type": error_type,
        "error_message": error_message,
        "log_path": log_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Input MS-Swift JSONL.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL to append missing failed rows to.")
    parser.add_argument("--failed", default=None, help="Optional failed JSONL mirror.")
    parser.add_argument("--error-type", default="InferenceRunnerFailed")
    parser.add_argument("--error-message", default="inference runner failed before writing this row")
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    summary = mark_missing_failed(
        source=Path(args.source),
        pred=Path(args.pred),
        failed=Path(args.failed) if args.failed else None,
        error_type=args.error_type,
        error_message=args.error_message,
        log_path=args.log,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
