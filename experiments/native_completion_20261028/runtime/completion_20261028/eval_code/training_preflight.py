"""Training-only throughput and full-budget output checks; no held-out tuning."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from eval_core import BUDGETS
from probe_vllm import render


def run(llm, processor, destination):
    from PIL import Image
    from vllm import SamplingParams
    from trains.flywheel.native_training import strict_json, validate_target

    panel = Path("/Knowin/sim/weilinruan/Anvil/arxiv/native_teacher_panel_20261012")
    manifest = json.loads((panel / "manifest.json").read_text())
    assert manifest["split"] == "train"
    assert hashlib.sha256((panel / "panel.jsonl").read_bytes()).hexdigest() == manifest["panel_sha256"]
    rows, counts = [], defaultdict(int)
    for line in (panel / "panel.jsonl").open():
        row = json.loads(line)
        key = (row["gsa_native_form"], row["gsa_meta"]["binding"]["source"])
        if key[0] in {"bbox", "mask", "point", "depth_order", "direction"} and counts[key] < 8:
            assert row["gsa_meta"]["split"] == "train"
            rows.append(row)
            counts[key] += 1
    requests = []
    for row in rows:
        with Image.open(row["images"][0]) as im:
            requests.append({"prompt": render(processor, row),
                             "multi_modal_data": {"image": im.convert("RGB")}})
    report = {"split": "train", "panel_sha256": manifest["panel_sha256"],
              "benchmark_score_claimed": False, "records": len(rows), "timings": [], "full_budget_cases": []}

    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    llm.generate(requests[:1], SamplingParams(temperature=0, max_tokens=16, seed=17), use_tqdm=False)
    for batch in (8, 16, 32):
        start = time.perf_counter()
        tokens = 0
        for offset in range(0, len(rows), batch):
            out = llm.generate(requests[offset:offset + batch],
                               SamplingParams(temperature=0, max_tokens=128, seed=17), use_tqdm=False)
            tokens += sum(len(x.outputs[0].token_ids) for x in out)
        seconds = time.perf_counter() - start
        report["timings"].append({"batch": batch, "max_tokens": 128, "seconds": seconds,
                                  "completion_tokens": tokens, "tokens_per_second": tokens / seconds,
                                  "records_per_second": len(rows) / seconds})
        save()
    chosen, seen = [], set()
    for i, row in enumerate(rows):
        key = (row["gsa_native_form"], row["gsa_meta"]["binding"]["source"])
        if key not in seen:
            chosen.append(i)
            seen.add(key)
    start = time.perf_counter()
    out = llm.generate([requests[i] for i in chosen],
                       [SamplingParams(temperature=0, max_tokens=BUDGETS[rows[i]["gsa_native_form"]], seed=17)
                        for i in chosen], use_tqdm=False)
    for i, output in zip(chosen, out):
        row, answer = rows[i], output.outputs[0]
        valid = False
        try:
            validate_target(row["gsa_native_form"], row["gsa_native_variant"],
                            strict_json(answer.text), row["messages"][0]["content"])
            valid = True
        except (ValueError, TypeError, KeyError, IndexError, OverflowError):
            pass
        report["full_budget_cases"].append({"id": row["id"], "form": row["gsa_native_form"],
                                            "response": answer.text, "contract_valid": valid,
                                            "finish_reason": answer.finish_reason,
                                            "tokens": len(answer.token_ids)})
    report["full_budget_seconds"] = time.perf_counter() - start
    report["state"] = "complete"
    save()
    return report
