"""Training-only runtime/quality probe; never select settings using held-out answers."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
PANEL = Path("/Knowin/sim/weilinruan/Anvil/arxiv/native_teacher_panel_20261012")
BUDGETS = {"bbox": 2048, "detection": 8192, "mask": 16384, "point": 16384,
           "direction": 512, "metric_comparison": 512, "relation_graph": 8192,
           "depth_order": 2048, "affordance": 512, "dof": 2048, "action_cot": 2048}


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def render(processor, row):
    prompt = row["messages"][0]["content"]
    assert prompt.startswith("<image>") and prompt.count("<image>") == 1
    messages = [{"role": "user", "content": [{"type": "image"},
                {"type": "text", "text": prompt.removeprefix("<image>")}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--forms", default="depth_order,direction,bbox,point,mask,detection,relation_graph")
    p.add_argument("--per-group", type=int, default=2)
    p.add_argument("--cap", type=int, default=256)
    p.add_argument("--batches", default="1,8,16")
    p.add_argument("--penalties", default="1")
    p.add_argument("--memory", type=float, default=.85)
    p.add_argument("--eager", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((PANEL / "manifest.json").read_text())
    assert manifest["split"] == "train"
    assert hashlib.sha256((PANEL / "panel.jsonl").read_bytes()).hexdigest() == manifest["panel_sha256"]
    rows = []
    counts = defaultdict(int)
    for line in (PANEL / "panel.jsonl").open():
        row = json.loads(line)
        key = (row["gsa_native_form"], row["gsa_meta"]["binding"]["source"])
        if key[0] in args.forms.split(",") and counts[key] < args.per_group:
            assert row["gsa_meta"]["split"] == "train"
            rows.append(row)
            counts[key] += 1
    import torch
    assert str(torch.cuda.get_device_properties(0).uuid) == os.environ["ANVIL_EXPECTED_GPU_UUID"].removeprefix("GPU-"), "CUDA GPU identity mismatch"
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    report = {"split": "train", "model": str(args.model), "panel_sha256": manifest["panel_sha256"],
              "rows": len(rows), "cap": args.cap, "cases": [], "timings": [],
              "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_at": time.time(),
              "benchmark_score_claimed": False}
    save(args.output / "report.json", report)
    llm = LLM(model=str(args.model), dtype="bfloat16", max_model_len=32768,
              gpu_memory_utilization=args.memory, max_num_seqs=32,
              max_num_batched_tokens=8192, enable_prefix_caching=True,
              limit_mm_per_prompt={"image": 1, "video": 0},
              mm_processor_kwargs={"max_pixels": 1536 * 32 * 32},
              enforce_eager=args.eager, seed=17, trust_remote_code=False)
    requests = []
    for row in rows:
        with Image.open(row["images"][0]) as im:
            requests.append({"prompt": render(processor, row),
                             "multi_modal_data": {"image": im.convert("RGB")}})
    # Keep warmup distinct from measured runs.
    llm.generate(requests[:1], SamplingParams(temperature=0, max_tokens=16, seed=17), use_tqdm=False)
    for penalty in map(float, args.penalties.split(",")):
        for batch in map(int, args.batches.split(",")):
            started = time.perf_counter()
            tokens = 0
            cases = []
            for offset in range(0, len(rows), batch):
                part = rows[offset:offset + batch]
                params = [SamplingParams(temperature=0, max_tokens=min(args.cap, BUDGETS[r["gsa_native_form"]]),
                                         repetition_penalty=penalty, seed=17) for r in part]
                responses = llm.generate(requests[offset:offset + batch], params, use_tqdm=False)
                for row, response in zip(part, responses):
                    out = response.outputs[0]
                    tokens += len(out.token_ids)
                    item = {"id": row["id"], "form": row["gsa_native_form"], "batch": batch,
                            "repetition_penalty": penalty, "finish_reason": out.finish_reason,
                            "completion_tokens": len(out.token_ids), "response": out.text,
                            "prompt_tokens": len(response.prompt_token_ids or [])}
                    cases.append(item)
            elapsed = time.perf_counter() - started
            timing = {"batch": batch, "repetition_penalty": penalty, "seconds": elapsed,
                      "completion_tokens": tokens, "tokens_per_second": tokens / elapsed,
                      "records_per_second": len(rows) / elapsed}
            report["cases"].extend(cases)
            report["timings"].append(timing)
            save(args.output / "report.json", report)
            print(json.dumps(timing), flush=True)
    report.update(state="complete", completed_at=time.time())
    save(args.output / "report.json", report)


if __name__ == "__main__":
    main()
