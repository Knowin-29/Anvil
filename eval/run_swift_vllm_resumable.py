#!/usr/bin/env python3
"""Resumable batched MS-Swift inference for converted eval JSONL.

The default backend is vLLM. For Qwen3.5 checkpoints whose architecture is not
accepted by the installed vLLM build, the runner can automatically fall back to
the MS-Swift transformers backend so evaluation produces usable rows instead of
an empty summary.
"""
from __future__ import annotations

import copy
import importlib.metadata as md
import json
import os
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Keep backward compatibility with earlier shell snippets that used VLLM_* for
# runner settings, but hide those non-vLLM env vars before importing vLLM.
VLLM_ENV_ALIASES = {
    "VLLM_TENSOR_PARALLEL_SIZE": "SWIFT_VLLM_TENSOR_PARALLEL_SIZE",
    "VLLM_GPU_MEMORY_UTILIZATION": "SWIFT_VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_MAX_MODEL_LEN": "SWIFT_VLLM_MAX_MODEL_LEN",
    "VLLM_MAX_NUM_SEQS": "SWIFT_VLLM_MAX_NUM_SEQS",
    "VLLM_ENFORCE_EAGER": "SWIFT_VLLM_ENFORCE_EAGER",
    "VLLM_LIMIT_MM_PER_PROMPT": "SWIFT_VLLM_LIMIT_MM_PER_PROMPT",
    "VLLM_ENGINE_KWARGS": "SWIFT_VLLM_ENGINE_KWARGS",
    "VLLM_ENABLE_PREFIX_CACHING": "SWIFT_VLLM_ENABLE_PREFIX_CACHING",
    "VLLM_ATTENTION_BACKEND": "SWIFT_VLLM_ATTENTION_BACKEND",
}
for old_name, new_name in VLLM_ENV_ALIASES.items():
    if old_name in os.environ:
        os.environ.setdefault(new_name, os.environ[old_name])
        os.environ.pop(old_name, None)


from swift.infer_engine import InferRequest
from swift.metrics import InferStats
from swift.pipelines.infer.infer import SwiftInfer
from fill_failed_predictions import mark_missing_failed
try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001 - progress bar is optional.
    tqdm = None


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    # Serialize the whole batch before opening the file so a bad object cannot
    # leave a partially written batch and then duplicate rows on retry/resume.
    lines = [
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=json_default) + "\n"
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.writelines(lines)
        f.flush()


def safe_usage(resp: Any) -> Dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for pkg in ["ms-swift", "vllm", "torch", "transformers", "flash-linear-attention", "flash-attn", "flashinfer-python"]:
        try:
            versions[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001 - diagnostic only.
            versions[pkg] = "missing"
    return versions


def resolve_length_policy() -> Tuple[str, str]:
    max_model_len = env_int("SWIFT_VLLM_MAX_MODEL_LEN", 16384)
    max_new_tokens = env_int("MAX_NEW_TOKENS", 1024)
    margin = env_int("MAX_INPUT_MARGIN", 32)
    max_length = os.environ.get("MAX_LENGTH")
    if max_length in (None, ""):
        # SWIFT max_length applies before generation. Leave headroom inside vLLM max_model_len.
        max_length = str(max(1, max_model_len - max_new_tokens - margin))

    policy = os.environ.get("OVERLONG_POLICY", "skip").lower().replace("-", "_")
    if policy == "skip":
        # SWIFT maps CLI `delete` to template `raise`; with infer_engine.strict=False,
        # overlong rows become per-row failures instead of aborting the full job.
        truncation_strategy = "delete"
    elif policy in {"truncate_left", "left"}:
        truncation_strategy = "left"
    elif policy in {"truncate_right", "right"}:
        truncation_strategy = "right"
    elif policy in {"raise", "error"}:
        truncation_strategy = "delete"
    else:
        raise ValueError("OVERLONG_POLICY must be one of: skip, truncate_left, truncate_right, raise")
    truncation_strategy = os.environ.get("TRUNCATION_STRATEGY", truncation_strategy)
    return max_length, truncation_strategy


def optional_arg(args: List[str], name: str, value: Optional[str]) -> None:
    if value not in (None, ""):
        args.extend([name, value])


def normalize_backend(value: str) -> str:
    value = (value or "vllm").lower()
    if value == "pt":
        return "transformers"
    return value


def should_fallback_to_transformers(error: BaseException) -> bool:
    text = "".join(traceback.format_exception_only(type(error), error))
    markers = [
        "architectures",
        "are not supported",
        "not supported for now",
        "ModelConfig",
    ]
    return all(marker in text for marker in markers[:2]) or ("ModelConfig" in text and "not supported" in text)


def mark_init_failure(result_path: Path, failed_path: Path, error: BaseException) -> None:
    if not env_bool("MARK_INIT_FAILED_ROWS", True):
        return
    val_dataset = Path(os.environ["VAL_DATASET"])
    if not val_dataset.is_file():
        print(json.dumps({
            "event": "mark_init_failed_rows_skipped",
            "reason": "val_dataset_not_found",
            "val_dataset": str(val_dataset),
        }, ensure_ascii=False), flush=True)
        return
    summary = mark_missing_failed(
        source=val_dataset,
        pred=result_path,
        failed=failed_path,
        error_type=type(error).__name__,
        error_message=str(error).splitlines()[0] if str(error) else type(error).__name__,
        log_path=os.environ.get("LOG_PATH", ""),
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def build_args() -> List[str]:
    max_length, truncation_strategy = resolve_length_policy()
    infer_backend = normalize_backend(os.environ.get("INFER_BACKEND", "vllm"))
    args = [
        "--model", os.environ["MODEL_PATH"],
        "--val_dataset", os.environ["VAL_DATASET"],
        "--val_dataset_shuffle", "false",
        "--remove_unused_columns", "false",
        "--infer_backend", infer_backend,
        "--enable_thinking", os.environ.get("ENABLE_THINKING", "false"),
        "--max_new_tokens", os.environ.get("MAX_NEW_TOKENS", "1024"),
        "--max_length", max_length,
        "--truncation_strategy", truncation_strategy,
        "--load_from_cache_file", os.environ.get("LOAD_FROM_CACHE_FILE", "false"),
        "--result_path", os.environ.get("RESULT_PATH", "/tmp/swift_vllm_unused.jsonl"),
    ]
    if infer_backend == "vllm":
        args.extend([
            "--vllm_tensor_parallel_size", os.environ.get("SWIFT_VLLM_TENSOR_PARALLEL_SIZE", "2"),
            "--vllm_gpu_memory_utilization", os.environ.get("SWIFT_VLLM_GPU_MEMORY_UTILIZATION", "0.9"),
            "--vllm_max_model_len", os.environ.get("SWIFT_VLLM_MAX_MODEL_LEN", "16384"),
            "--vllm_max_num_seqs", os.environ.get("SWIFT_VLLM_MAX_NUM_SEQS", "16"),
            "--vllm_enforce_eager", os.environ.get("SWIFT_VLLM_ENFORCE_EAGER", "false"),
            "--vllm_limit_mm_per_prompt", os.environ.get("SWIFT_VLLM_LIMIT_MM_PER_PROMPT", '{"image": 10, "video": 0}'),
        ])
        optional_arg(args, "--vllm_engine_kwargs", os.environ.get("SWIFT_VLLM_ENGINE_KWARGS"))
        optional_arg(args, "--vllm_enable_prefix_caching", os.environ.get("SWIFT_VLLM_ENABLE_PREFIX_CACHING"))
        optional_arg(args, "--vllm_max_lora_rank", os.environ.get("SWIFT_VLLM_MAX_LORA_RANK"))
    else:
        args.extend([
            "--max_batch_size",
            os.environ.get("SWIFT_TRANSFORMERS_MAX_BATCH_SIZE", os.environ.get("TRANSFORMERS_BATCH_SIZE", "1")),
        ])
    optional_arg(args, "--attn_impl", os.environ.get("ATTN_IMPL"))
    optional_arg(args, "--val_dataset_sample", os.environ.get("VAL_DATASET_SAMPLE"))
    optional_arg(args, "--adapters", os.environ.get("ADAPTERS"))
    return args


def response_row(index: int, data: Dict[str, Any], resp: Any, labels: Optional[str], elapsed: float) -> Tuple[Dict[str, Any], Dict[str, int]]:
    response = resp.choices[0].message.content
    row_data = copy.deepcopy(data)
    row_data.setdefault("messages", []).append({"role": "assistant", "content": response})
    usage = safe_usage(resp)
    row = {
        "__index__": index,
        "__ok__": True,
        "__elapsed_s__": elapsed,
        "response": response,
        "labels": labels,
        "logprobs": resp.choices[0].logprobs,
        **row_data,
    }
    return row, usage


def error_row(index: int, data: Dict[str, Any], error: BaseException) -> Dict[str, Any]:
    return {
        "__index__": index,
        "__ok__": False,
        "response": "",
        "labels": None,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": "".join(traceback.format_exception_only(type(error), error)).strip(),
        **copy.deepcopy(data),
    }


# Fields accepted by swift InferRequest. Extra dataset columns (e.g. "model",
# "id", "label") must not leak into the infer input dict: with the transformers
# backend they collide with template.generate(model=...) raising a TypeError.
INFER_REQUEST_FIELDS = ("messages", "images", "audios", "videos", "tools", "objects")


def to_infer_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keys swift InferRequest accepts; drop extra dataset columns."""
    return {key: data[key] for key in INFER_REQUEST_FIELDS if key in data}


def prepare_item(index: int, raw: Dict[str, Any], task_type: str) -> Tuple[int, Dict[str, Any], Optional[str]]:
    data = copy.deepcopy(raw)
    if task_type == "causal_lm":
        labels = InferRequest.remove_response(data["messages"])
    else:
        labels = data.pop("label", None)
    return index, data, labels


def main() -> int:
    result_path = Path(os.environ["RESULT_PATH"])
    failed_path = Path(os.environ.get("FAILED_PATH", str(result_path) + ".failed.jsonl"))
    reset_output = env_bool("RESET_OUTPUT", False)
    resume = env_bool("RESUME", True)
    initial_backend = normalize_backend(os.environ.get("INFER_BACKEND", "vllm"))
    if initial_backend == "transformers":
        batch_size = env_int("SWIFT_TRANSFORMERS_MAX_BATCH_SIZE", env_int("TRANSFORMERS_BATCH_SIZE", 1))
    else:
        batch_size = env_int("BATCH_SIZE", 8)
    max_retries = env_int("MAX_RETRIES", 2)
    retry_sleep = float(os.environ.get("RETRY_SLEEP", "5"))

    if reset_output:
        result_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)

    max_length, truncation_strategy = resolve_length_policy()
    print(json.dumps({
        "event": "environment",
        "package_versions": package_versions(),
        "infer_backend": normalize_backend(os.environ.get("INFER_BACKEND", "vllm")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_attention_backend": os.environ.get("SWIFT_VLLM_ATTENTION_BACKEND"),
        "vllm_engine_kwargs": os.environ.get("SWIFT_VLLM_ENGINE_KWARGS"),
        "model_path": os.environ.get("MODEL_PATH"),
        "adapters": os.environ.get("ADAPTERS"),
        "val_dataset": os.environ.get("VAL_DATASET"),
        "result_path": str(result_path),
        "failed_path": str(failed_path),
    }, ensure_ascii=False), flush=True)
    print(json.dumps({
        "event": "length_policy",
        "vllm_max_model_len": env_int("SWIFT_VLLM_MAX_MODEL_LEN", 16384),
        "max_new_tokens": env_int("MAX_NEW_TOKENS", 1024),
        "max_input_margin": env_int("MAX_INPUT_MARGIN", 32),
        "max_length": int(max_length),
        "truncation_strategy": truncation_strategy,
        "overlong_policy": os.environ.get("OVERLONG_POLICY", "skip"),
    }, ensure_ascii=False), flush=True)

    try:
        runner = SwiftInfer(build_args())
    except BaseException as exc:
        infer_backend = normalize_backend(os.environ.get("INFER_BACKEND", "vllm"))
        if (
            infer_backend == "vllm"
            and env_bool("AUTO_FALLBACK_TO_TRANSFORMERS", True)
            and should_fallback_to_transformers(exc)
        ):
            print(json.dumps({
                "event": "backend_fallback",
                "from": "vllm",
                "to": "transformers",
                "error_type": type(exc).__name__,
                "error_message": str(exc).splitlines()[0] if str(exc) else "",
            }, ensure_ascii=False), flush=True)
            os.environ["INFER_BACKEND"] = "transformers"
            os.environ["BATCH_SIZE"] = os.environ.get("TRANSFORMERS_BATCH_SIZE", "1")
            batch_size = env_int("SWIFT_TRANSFORMERS_MAX_BATCH_SIZE", env_int("TRANSFORMERS_BATCH_SIZE", 1))
            try:
                runner = SwiftInfer(build_args())
            except BaseException as fallback_exc:
                print(json.dumps({
                    "event": "init_failed",
                    "infer_backend": "transformers",
                    "error_type": type(fallback_exc).__name__,
                    "error_message": str(fallback_exc).splitlines()[0] if str(fallback_exc) else "",
                }, ensure_ascii=False), flush=True)
                mark_init_failure(result_path, failed_path, fallback_exc)
                return 2
        else:
            print(json.dumps({
                "event": "init_failed",
                "infer_backend": infer_backend,
                "error_type": type(exc).__name__,
                "error_message": str(exc).splitlines()[0] if str(exc) else "",
            }, ensure_ascii=False), flush=True)
            mark_init_failure(result_path, failed_path, exc)
            return 2
    if hasattr(runner.infer_engine, "strict"):
        runner.infer_engine.strict = False
    request_config = runner.args.get_request_config()
    dataset = list(runner._prepare_val_dataset())

    completed = count_jsonl(result_path) if resume else 0
    if completed > len(dataset):
        raise RuntimeError(f"result has {completed} rows, but dataset has only {len(dataset)} rows: {result_path}")

    pending = [(i, dataset[i]) for i in range(completed, len(dataset))]
    print(json.dumps({
        "event": "start",
        "dataset_rows": len(dataset),
        "completed_rows": completed,
        "pending_rows": len(pending),
        "batch_size": batch_size,
        "split_on_error": env_bool("SPLIT_ON_ERROR", True),
        "max_retries": max_retries,
        "allow_failed": env_bool("ALLOW_FAILED", True),
    }, ensure_ascii=False), flush=True)

    global_pbar = None
    if env_bool("GLOBAL_TQDM", True) and tqdm is not None:
        global_pbar = tqdm(total=len(dataset), initial=completed, dynamic_ncols=True, desc="samples", unit="sample")

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_ok = 0
    total_failed = 0
    infer_time = 0.0
    wall_start = time.perf_counter()

    def run_batch(items: List[Tuple[int, Dict[str, Any]]]) -> None:
        nonlocal total_prompt_tokens, total_completion_tokens, total_ok, total_failed, infer_time
        prepared = [prepare_item(i, raw, runner.args.task_type) for i, raw in items]
        infer_inputs = [to_infer_input(data) for _, data, _ in prepared]
        labels_list = [labels for _, _, labels in prepared]

        last_error: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                start = time.perf_counter()
                metrics = [InferStats()]
                responses = runner.infer(infer_inputs, request_config, use_tqdm=True, metrics=metrics, **runner.infer_kwargs)
                elapsed = time.perf_counter() - start
                rows: List[Dict[str, Any]] = []
                failed_rows: List[Dict[str, Any]] = []
                prompt_tokens = 0
                completion_tokens = 0
                ok_count = 0
                failed_count = 0
                for (index, data, labels), resp in zip(prepared, responses):
                    if isinstance(resp, BaseException):
                        row = error_row(index, data, resp)
                        rows.append(row)
                        failed_rows.append(row)
                        failed_count += 1
                        continue
                    row, usage = response_row(index, data, resp, labels, elapsed)
                    rows.append(row)
                    prompt_tokens += usage["prompt_tokens"]
                    completion_tokens += usage["completion_tokens"]
                    ok_count += 1
                append_jsonl(result_path, rows)
                if failed_rows:
                    append_jsonl(failed_path, failed_rows)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_ok += ok_count
                total_failed += failed_count
                infer_time += elapsed
                if global_pbar is not None:
                    global_pbar.update(len(items))
                    global_pbar.set_postfix(ok=total_ok, failed=total_failed, refresh=False)
                print(json.dumps({
                    "event": "batch_done",
                    "start_index": items[0][0],
                    "end_index": items[-1][0],
                    "batch_rows": len(items),
                    "ok": ok_count,
                    "failed": failed_count,
                    "elapsed_s": elapsed,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_s": (prompt_tokens + completion_tokens) / elapsed if elapsed > 0 else None,
                    "samples_per_s": ok_count / elapsed if elapsed > 0 else None,
                }, ensure_ascii=False), flush=True)
                return
            except BaseException as exc:  # Keep KeyboardInterrupt/SystemExit visible after retries.
                last_error = exc
                print(json.dumps({
                    "event": "batch_error",
                    "start_index": items[0][0],
                    "end_index": items[-1][0],
                    "attempt": attempt + 1,
                    "max_attempts": max_retries + 1,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }, ensure_ascii=False), flush=True)
                if attempt < max_retries:
                    time.sleep(retry_sleep)

        if len(items) > 1 and env_bool("SPLIT_ON_ERROR", True):
            mid = len(items) // 2
            run_batch(items[:mid])
            run_batch(items[mid:])
            return

        assert last_error is not None
        error_rows = [error_row(i, raw, last_error) for i, raw in items]
        append_jsonl(result_path, error_rows)
        append_jsonl(failed_path, error_rows)
        total_failed += len(error_rows)
        if global_pbar is not None:
            global_pbar.update(len(error_rows))
            global_pbar.set_postfix(ok=total_ok, failed=total_failed, refresh=False)

    try:
        for start in range(0, len(pending), batch_size):
            run_batch(pending[start:start + batch_size])
    finally:
        if global_pbar is not None:
            global_pbar.close()

    wall_elapsed = time.perf_counter() - wall_start
    summary = {
        "event": "done",
        "ok": total_ok,
        "failed": total_failed,
        "new_rows": total_ok + total_failed,
        "result_rows": count_jsonl(result_path),
        "infer_time_s": infer_time,
        "wall_time_s": wall_elapsed,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "tokens_per_s_infer_only": (total_prompt_tokens + total_completion_tokens) / infer_time if infer_time > 0 else None,
        "samples_per_s_infer_only": total_ok / infer_time if infer_time > 0 else None,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if total_failed == 0 or env_bool("ALLOW_FAILED", True) else 2


if __name__ == "__main__":
    sys.exit(main())
