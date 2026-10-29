"""Versioned full-scope native evaluation with continuous batching and live health.

Original runs/checkpoints are immutable. This run evaluates every selected row
under one new protocol; it does not mix old-engine outputs into the new scores.
An exact sustained token cycle can be aborted, but the example remains in the
denominator with failure metrics. The declared DoF model decoder uses generated values and prompt context only; no reference-fed retry.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import threading
import time

DIAGNOSTIC = Path(__file__).resolve().parent
sys.path.insert(0,str(DIAGNOSTIC.parent/'eval_code'))
from dof_codebook_interface import model_prompt, decode_response, VERSION as CODEC_VERSION
from copy import deepcopy
from eval_core import BUDGETS, canonical, check_saved, digest, repeat_period, select_files, validate_row
from probe_vllm import render

CODE = DIAGNOSTIC.parent / 'eval_code'
ROOT = Path("/Knowin/sim/weilinruan/Anvil/recovery_20261020")
EXPORT = ROOT.parent / "arxiv/native_evaluation_export_20261012_v4"
SCORE_CODE = CODE / "score_code"
METRICS = None
REGIONS = None


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def init_scorer(manifest):
    global METRICS, REGIONS
    sys.path.insert(0, str(SCORE_CODE))
    from eval import native_metrics
    METRICS = native_metrics
    REGIONS = None  # Initialize source masks lazily only when needed.


def score(job, manifest):
    global REGIONS
    row, item = job
    region = None
    if row["native_form"] == "point" and row["evaluation_meta"]["source"] == "coco_train2017":
        if REGIONS is None:
            REGIONS = METRICS.CocoPointReference(manifest)
        region = REGIONS(row)
    if row['native_form']=='dof':
        item['latent_response']=item['response']
        item['latent_response_sha256']=item['response_sha256']
        item['model_decoder']=CODEC_VERSION
        try:
            item['response']=decode_response(item['latent_response'],row['messages'][0]['content'])
            item['decoder_error']=None
        except (ValueError,KeyError,TypeError,OverflowError) as exc:
            item['response']=''
            item['decoder_error']=type(exc).__name__+': '+str(exc)
        item['response_sha256']=hashlib.sha256(item['response'].encode()).hexdigest()
    item["metrics"] = METRICS.score_row(row, item["response"],
                                      finish_reason=item["finish_reason"], point_region=region)
    return item


def inference_row(row):
    result=deepcopy(row)
    # Remove gold answer even from the wrapper; render only uses user content.
    result['messages']=[deepcopy(row['messages'][0])]
    if row['native_form']=='dof':
        result['messages'][0]['content']=model_prompt(row['messages'][0]['content'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["G", "S", "A", "joint"], required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--memory", type=float, default=.85)
    parser.add_argument("--repetition-penalty", type=float, default=1.)
    parser.add_argument("--disable-cycle-guard", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=0)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--forms", nargs="+", choices=sorted(BUDGETS))
    parser.add_argument("--training-preflight", action="store_true")
    args = parser.parse_args()
    assert os.environ["ANVIL_EXPECTED_GPU_UUID"] == args.gpu_uuid
    assert args.gpu_uuid.startswith("GPU-") and 1 <= args.concurrency <= 64 and .4 <= args.memory <= .9
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lease = open(ROOT.parent / ".gpu_locks" / (args.gpu_uuid + ".lock"), "a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = open(args.output / "run.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stat = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    save(args.output / "process.json", {"pid": os.getpid(), "start_ticks": stat[19],
                                       "cmdline_sha256": sha("/proc/self/cmdline")})
    state = {"stage": "preflight", "pid": os.getpid(), "gpu_uuid": args.gpu_uuid,
             "model": args.name, "started_at": time.time(), "completed": 0,
             "active": 0, "last_completed_at": None, "last_engine_step_at": None}
    state_lock = threading.Lock()
    stop_requested = threading.Event()
    heartbeat_done = threading.Event()

    def heartbeat():
        while not heartbeat_done.is_set():
            with state_lock:
                snap = {**state, "heartbeat_at": time.time()}
            save(args.output / "status.json", snap)
            heartbeat_done.wait(10)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
    signal.signal(signal.SIGINT, lambda *_: stop_requested.set())
    handles, pool, engine = {}, None, None
    try:
        manifest = json.loads((EXPORT / "manifest.json").read_text())
        files = select_files(manifest, args.scope, args.shard, args.shards)
        if args.forms:
            files = [(name, info) for name, info in files if info["form"] in args.forms]
            assert {info["form"] for _, info in files} == set(args.forms)
        total = sum(i["shard_records"] for _, i in files)
        for name, info in files:
            assert sha(EXPORT / name) == info["sha256"], f"Changed evaluation input: {name}"
        merge = args.model / "merge_receipt.json"
        if merge.exists():
            binding = json.loads(merge.read_text())
            assert binding["state"] == "verified_derived_inference_copy"
            for name, item in binding["shards"].items():
                assert sha(args.model / name) == item["derived_sha256"], f"Changed model shard: {name}"
            source = json.loads((ROOT.parent / "arxiv/native_a_specialist_20261012_v4/protocol.json").read_text())
            metadata = {name: expected for name, expected in source["base_files_sha256"].items()
                        if ".safetensors" not in name or name.endswith(".index.json")}
            for name, expected in metadata.items():
                assert sha(args.model / name) == expected, f"Changed model metadata: {name}"
            binding["metadata_sha256"] = metadata
        else:
            source = json.loads((ROOT.parent / "arxiv/native_a_specialist_20261012_v4/protocol.json").read_text())
            assert str(args.model) == source["model"]
            binding = {"base_files_sha256": source["base_files_sha256"]}
            for name, expected in source["base_files_sha256"].items():
                assert sha(args.model / name) == expected, f"Changed base file: {name}"
        code_files = [CODE / n for n in ("run_vllm_eval.py", "eval_core.py", "probe_vllm.py", "training_preflight.py", "run_isolated.sh")]
        code_files += list(SCORE_CODE.rglob("*.py"))
        code_files += [DIAGNOSTIC/n for n in ['dof_codebook_eval.py','dof_codebook_interface.py','dof_codec_prototype.py','dof_rotation_codebook.json','prepare_dof_rotation_codebook.py']]
        protocol = {
            "schema": "anvil-dof-codebook-model-evaluation-v1", "model_decoder": CODEC_VERSION, "name": args.name,
            "model": str(args.model), "model_binding": binding,
            "scope": args.scope, "selected_forms": args.forms, "shard": args.shard, "shards": args.shards, "expected_records": total,
            "export_manifest_sha256": sha(EXPORT / "manifest.json"), "selection_sha256": manifest["selection_sha256"],
            "files": dict(files), "context_length": 32768, "image_max_tokens": 1536,
            "dtype": "bfloat16", "seed": 17, "temperature": 0, "thinking": False,
            "token_budgets": BUDGETS, "concurrency": args.concurrency,
            "gpu_memory_utilization": args.memory, "enforce_eager": args.eager,
            "prefix_caching": True, "max_num_batched_tokens": 8192,
            "multimodal_max_pixels": 1536 * 32 * 32,
            "repetition_penalty": args.repetition_penalty,
            "cycle_guard": None if args.disable_cycle_guard else
                {"minimum_output_tokens": 512, "minimum_identical_span": 256,
                 "minimum_repeats": 16, "max_period": 32, "check_interval_tokens": 64,
                 "outcome": "abort and retain as failed example; never synthesize an answer"},
            "response_policy": "DoF model predicts 12-number latent; declared deterministic decoder produces native geometry. Store both raw latent and model output; invalid/aborted stay in denominator. Other forms unchanged.",
            "environment": {k: importlib.metadata.version(k) for k in
                            ("vllm", "torch", "transformers", "safetensors", "pycocotools")},
            "code_sha256": {str(p): sha(p) for p in code_files},
            "smoke_rows": args.smoke_rows, "old_results_imported": False, "reference_targets_changed": False,
            "malformed_and_aborted_examples_in_denominator": True,
            "final_mopd": False, "published_paper_results_reproduced": False}
        path = args.output / "protocol.json"
        if args.resume:
            assert json.loads(path.read_text()) == protocol, "Resume protocol changed"
        else:
            save(path, protocol)
        run_sha = sha(path)
        prior = {}
        counts, metric_groups = Counter(), {}

        def accumulate(item):
            counts["completed"] += 1
            counts[item["native_form"]] += 1
            counts["completion_tokens"] += item["usage"]["completion_tokens"]
            counts["truncated"] += item["finish_reason"] == "length"
            counts["cycle_aborted"] += item["finish_reason"] == "abort"
            counts["contract_valid"] += bool(item["metrics"]["contract_valid"])
            key = (item["native_form"], item["source"])
            group = metric_groups.setdefault(key, {"n": 0, "sums": Counter()})
            group["n"] += 1
            group["sums"].update(item["metrics"])

        for _, info in files:
            form = info["form"]
            path = args.output / f"predictions_{form}.jsonl"
            saved = {}
            if path.exists():
                assert args.resume
                with path.open("rb") as f:
                    for line in f:
                        assert line.endswith(b"\n"), "Partial output tail retained; explicit recovery required"
                        item = json.loads(line)
                        assert item["ordinal"] not in saved and item["protocol_sha256"] == run_sha
                        assert item["native_form"] == form
                        assert hashlib.sha256(item["response"].encode()).hexdigest() == item["response_sha256"]
                        assert digest({k: v for k, v in item.items() if k != "record_sha256"}) == item["record_sha256"]
                        saved[item["ordinal"]] = (item["id"], item["evaluation_content_sha256"])
                        accumulate(item)
            prior[form] = saved
            handles[form] = path.open("a", encoding="utf-8")
        started_generation = time.monotonic()
        initial_completed = counts["completed"]
        image_cache = {}

        def rows():
            yielded = 0
            for name, info in files:
                form = info["form"]
                found = set()
                with (EXPORT / name).open("rb") as f:
                    for ordinal, line in enumerate(f):
                        if ordinal % args.shards != args.shard:
                            continue
                        row = json.loads(line)
                        validate_row(row, form, manifest["selection_sha256"])
                        if ordinal in prior[form]:
                            assert prior[form][ordinal] == (row["id"], row["evaluation_meta"]["evaluation_content_sha256"])
                            found.add(ordinal)
                            continue
                        if args.smoke_rows and yielded >= args.smoke_rows:
                            return
                        yield ordinal, row
                        yielded += 1
                assert found == set(prior[form]), "Saved ordinals outside scope"

        with state_lock:
            state.update(stage="loading_model", expected_records=total, completed=counts["completed"])
        from PIL import Image
        import torch
        assert str(torch.cuda.get_device_properties(0).uuid) == args.gpu_uuid.removeprefix("GPU-"), "CUDA GPU identity mismatch"
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind
        torch.set_num_threads(4)
        processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
        llm = LLM(model=str(args.model), dtype="bfloat16", max_model_len=32768,
                  gpu_memory_utilization=args.memory, max_num_seqs=args.concurrency,
                  max_num_batched_tokens=8192, enable_prefix_caching=True,
                  limit_mm_per_prompt={"image": 1, "video": 0},
                  mm_processor_kwargs={"max_pixels": 1536 * 32 * 32},
                  enforce_eager=args.eager, seed=17, trust_remote_code=False)
        engine = llm.llm_engine
        preflight_path = args.output / "training_preflight.json"
        preflight_complete = preflight_path.exists() and json.loads(preflight_path.read_text()).get("state") == "complete"
        if args.training_preflight and not preflight_complete:
            with state_lock:
                state.update(stage="training_only_preflight")
            sys.path.insert(0, str(SCORE_CODE))
            from training_preflight import run as run_training_preflight
            run_training_preflight(llm, processor, preflight_path)
        pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=init_scorer, initargs=(manifest,))
        iterator = iter(rows())
        active, scoring = {}, {}
        exhausted = False
        seq = 0
        started_generation = time.monotonic()
        initial_tokens = counts["completion_tokens"]
        with state_lock:
            state.update(stage="inference", generation_started_at=time.time())

        def persist_ready(block=False):
            pending = list(scoring)
            if block and pending and not any(f.done() for f in pending):
                pending[0].result()
            written = 0
            for future in pending:
                if not future.done():
                    continue
                item = future.result()  # A scoring/reference failure stops the run.
                item["record_sha256"] = digest(item)
                form = item["native_form"]
                handles[form].write(canonical(item) + "\n")
                del scoring[future]
                accumulate(item)
                written += 1
            if written:
                for h in handles.values():
                    h.flush()
                    os.fsync(h.fileno())
                seconds = max(.001, time.monotonic() - started_generation)
                with state_lock:
                    state.update(completed=counts["completed"], per_form={f: counts[f] for f in BUDGETS},
                                 last_completed_at=time.time(), cycle_aborted=counts["cycle_aborted"],
                                 truncated=counts["truncated"], contract_valid=counts["contract_valid"],
                                 records_per_second=(counts["completed"] - initial_completed) / seconds,
                                 generated_tokens_per_second=(counts["completion_tokens"] - initial_tokens) / seconds)

        while active or not exhausted or scoring:
            if stop_requested.is_set():
                if active:
                    engine.abort_request(list(active))
                active.clear()
                exhausted = True
            while not exhausted and len(active) < args.concurrency and len(scoring) < args.concurrency * 4:
                try:
                    ordinal, row = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                image_path = row["images"][0]
                if image_path not in image_cache:
                    image_cache[image_path] = sha(image_path)
                assert image_cache[image_path] == row["evaluation_meta"]["image_sha256"]
                with Image.open(image_path) as im:
                    request = {"prompt": render(processor, inference_row(row)), "multi_modal_data": {"image": im.convert("RGB")}}
                rid = f"{args.name}-{seq}"
                seq += 1
                params = SamplingParams(temperature=0, max_tokens=BUDGETS[row["native_form"]], seed=17,
                                        repetition_penalty=args.repetition_penalty,
                                        output_kind=RequestOutputKind.CUMULATIVE)
                engine.add_request(rid, request, params)
                active[rid] = {"row": row, "ordinal": ordinal, "last_check": 0, "started": time.time()}
            if active:
                outputs = engine.step()
                with state_lock:
                    state.update(active=len(active), scoring_pending=len(scoring), last_engine_step_at=time.time())
                for output in outputs:
                    rid = output.request_id
                    if rid not in active:
                        continue
                    task = active[rid]
                    result = output.outputs[0]
                    cycle = None
                    if not args.disable_cycle_guard and len(result.token_ids) - task["last_check"] >= 64:
                        task["last_check"] = len(result.token_ids)
                        cycle = repeat_period(result.token_ids)
                    if not output.finished and not cycle:
                        continue
                    if cycle and not output.finished:
                        engine.abort_request([rid])
                    row = task["row"]
                    finish = "abort" if cycle and not output.finished else result.finish_reason
                    item = {"id": row["id"], "ordinal": task["ordinal"], "native_form": row["native_form"],
                            "source": row["evaluation_meta"]["source"],
                            "evaluation_content_sha256": row["evaluation_meta"]["evaluation_content_sha256"],
                            "protocol_sha256": run_sha, "response": result.text,
                            "response_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
                            "finish_reason": finish, "cycle_detection": cycle,
                            "usage": {"completion_tokens": len(result.token_ids),
                                      "prompt_tokens": len(output.prompt_token_ids or [])},
                            "seconds_in_engine": time.time() - task["started"]}
                    future = pool.submit(score, (row, item), manifest)
                    scoring[future] = rid
                    del active[rid]
            persist_ready(block=not active and bool(scoring))
        for h in handles.values():
            h.flush()
            os.fsync(h.fileno())
        for name, expected in protocol["code_sha256"].items():
            assert sha(CODE / name) == expected, "Evaluator code changed during the run"
        if stop_requested.is_set():
            with state_lock:
                state.update(stage="paused_recoverable", active=0)
        else:
            assert counts["completed"] == (initial_completed + args.smoke_rows if args.smoke_rows else total)
            result = {
                "state": "smoke_complete" if args.smoke_rows else "complete_scope_shard",
                "n": counts["completed"], "expected_records": total, "protocol_sha256": run_sha,
                "groups": [{"form": k[0], "source": k[1], "n": v["n"],
                            "mean": {m: value / v["n"] for m, value in sorted(v["sums"].items())}}
                           for k, v in sorted(metric_groups.items())],
                "prediction_sha256": {Path(h.name).name: sha(h.name) for h in handles.values()},
                "global_composite_score": None, "final_mopd": False,
                "published_paper_results_reproduced": False}
            save(args.output / "metrics.json", result)
            with state_lock:
                state.update(stage=result["state"], active=0, completed_at=time.time())
    except BaseException as error:
        with state_lock:
            state.update(stage="failed", error_type=type(error).__name__, error=str(error)[:1000])
        save(args.output / "failure.json", {"type": type(error).__name__, "reason": str(error),
                                           "completed_predictions_preserved": True, "time": time.time()})
        raise
    finally:
        for h in handles.values():
            h.close()
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        heartbeat_done.set()
        thread.join(timeout=12)
        with state_lock:
            save(args.output / "status.json", {**state, "heartbeat_at": time.time()})


if __name__ == "__main__":
    main()
