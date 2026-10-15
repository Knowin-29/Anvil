#!/usr/bin/env python3
"""Unified benchmark entry point.

    python eval/run_benchmark.py --model <path> --benchmark cv_bench --out r.json [--limit N]

One interface for every benchmark, replacing nine external runners with mutually
incompatible argparse surfaces. Loads the dataset, runs inference through swift's
TransformersEngine and scores in-process. A benchmark either emits a trusted number or
records why it could not; nothing is estimated.

Output JSON always carries: benchmark, score, n_evaluated, parse_failures, parsed,
notes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.gsa_eval.registry import BENCHMARKS, PRIORITY  # noqa: E402
from eval.gsa_eval.runner import (evaluate, generate_shard_multi,  # noqa: E402
                                  score_shards)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="HF checkpoint directory (merged weights)")
    ap.add_argument("--benchmark", help=f"one of: {', '.join(sorted(BENCHMARKS))}")
    ap.add_argument("--out", help="output JSON path")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate a seeded stratified subset; 0 = full split")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 uses the preregistered task-semantic budget; a positive "
                         "value is an explicit diagnostic override")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", help="override dataset path (gsa_bench_test only)")
    ap.add_argument("--scratch", default=None,
                    help="scratch dir for extracted images (default .cache/eval_assets)")
    ap.add_argument("--dump-predictions", help="also write raw responses as jsonl")
    ap.add_argument("--list", action="store_true", help="list benchmarks and exit")
    # Data parallelism: N processes, each pinned to one GPU via CUDA_VISIBLE_DEVICES,
    # each generating its stride slice. Scoring happens once on the merged shards.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-out", help="worker mode: write this shard's raw responses "
                                       "here and skip scoring")
    ap.add_argument("--score-shards", nargs="+",
                    help="parent mode: merge these shard files and score, no inference")
    # Multi-benchmark worker mode. Loading the 9B checkpoint costs ~105 s while a shard's
    # generation for the cheap benchmarks costs 63-230 s, so one load per (benchmark,
    # model) cell burned ~40% of the wall clock with the GPU idle. With this flag one
    # process loads the weights once and walks its stride slice of every listed benchmark.
    ap.add_argument("--benchmarks", nargs="+",
                    help="worker mode: generate this shard's slice of each of these "
                         "benchmarks from a single loaded model")
    ap.add_argument("--shard-dir",
                    help="with --benchmarks: write <shard-dir>/<benchmark>/shard_N.jsonl")
    ap.add_argument("--device-map", default="cuda:0",
                    help="'cuda:0' pins the whole model to one visible GPU (default). "
                         "'auto' restores swift's pipeline-parallel split, which idled "
                         "7 of 8 GPUs; only useful for the equivalence check.")
    # The qwen3_5 template force-prefixes '<think>\n' on every assistant turn, so the
    # model reasons even though the SFT answers are bare JSON. --no-thinking clears that
    # prefix, which is the only lever that reaches it (a '/no_think' prompt suffix does
    # not: measured 11/12 responses still opened a think span).
    ap.add_argument("--thinking", dest="thinking", action="store_true", default=None,
                    help="force the template's thinking prefix on")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false",
                    help="clear the thinking prefix; matches the SFT training format")
    ap.add_argument("--elapsed-s", type=float, default=None,
                    help="wall time of the sharded generation, recorded in the result")
    a = ap.parse_args()
    # Display-name migration only: preserve the frozen v1 data and scoring protocol.
    if a.benchmark == "anvil_eval":
        a.benchmark = "gsa_bench_test"
    if a.benchmarks:
        a.benchmarks = ["gsa_bench_test" if b == "anvil_eval" else b for b in a.benchmarks]

    if a.list:
        print(f"{'benchmark':20} {'avail':6} {'n':>6}  metric / skip reason")
        for name, s in sorted(BENCHMARKS.items()):
            star = "*" if name in PRIORITY else " "
            info = s.metric if s.available else f"SKIP: {s.skip_reason}"
            print(f"{star}{name:19} {str(s.available):6} {s.n_expected:>6}  {info}")
        print("\n* = self-hosted and verified end-to-end")
        return 0

    root = Path(__file__).resolve().parents[1]
    scratch = Path(a.scratch) if a.scratch else root / ".cache" / "eval_assets"
    scratch.mkdir(parents=True, exist_ok=True)

    if a.benchmarks:
        for f in ("model", "out", "shard_dir"):
            if not getattr(a, f):
                ap.error(f"--benchmarks requires --{f.replace('_', '-')}")
        unknown = [b for b in a.benchmarks if b not in BENCHMARKS]
        if unknown:
            ap.error(f"unknown benchmark(s): {unknown}; known: {sorted(BENCHMARKS)}")
        res = generate_shard_multi(
            a.benchmarks, a.model, shard=a.shard, num_shards=a.num_shards,
            shard_dir=Path(a.shard_dir), limit=a.limit, max_tokens=a.max_tokens,
            batch_size=a.batch_size, scratch=scratch, data=a.data, seed=a.seed,
            device_map=(None if a.device_map == "none" else a.device_map),
            thinking=a.thinking)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"wrote {out}")
        # A worker that lost even one benchmark must fail: the parent would otherwise
        # score a set with responses missing and count them as wrong answers.
        return 1 if res.get("n_failed") else 0

    missing = [f for f in ("model", "benchmark", "out") if not getattr(a, f)]
    if missing:
        ap.error("missing required arguments: " + ", ".join("--" + m for m in missing))

    out = Path(a.out)
    dump = Path(a.dump_predictions) if a.dump_predictions else None
    try:
        if a.score_shards:
            res = score_shards(a.benchmark, a.model,
                               [Path(p) for p in a.score_shards], limit=a.limit,
                               max_tokens=a.max_tokens, scratch=scratch, data=a.data,
                               seed=a.seed, dump_predictions=dump,
                               elapsed_s=a.elapsed_s, thinking=a.thinking,
                               generation_batch_size=a.batch_size)
        else:
            res = evaluate(a.benchmark, a.model, limit=a.limit,
                           max_tokens=a.max_tokens, batch_size=a.batch_size,
                           scratch=scratch, data=a.data, seed=a.seed,
                           dump_predictions=dump, shard=a.shard,
                           num_shards=a.num_shards,
                           device_map=(None if a.device_map == "none"
                                       else a.device_map),
                           shard_out=(Path(a.shard_out) if a.shard_out else None),
                           thinking=a.thinking)
    except Exception as e:
        import traceback
        traceback.print_exc()
        res = {"benchmark": a.benchmark, "model": a.model, "score": None,
               "n_evaluated": 0, "parse_failures": 0, "parsed": False,
               "error": f"{type(e).__name__}: {e}",
               "notes": f"failed: {type(e).__name__}: {e}"}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "detail"}, indent=2,
                     ensure_ascii=False))
    print(f"wrote {out}")
    if a.shard_out:
        # A worker only generates; "parsed" is decided later by the parent, so success
        # here means the shard file exists and every sample produced a response.
        return 0 if res.get("n_evaluated") and not res.get("error") else 1
    return 0 if res.get("parsed") or res.get("skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
