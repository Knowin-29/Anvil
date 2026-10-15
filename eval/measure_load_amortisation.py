#!/usr/bin/env python3
"""Measure the throughput won by loading the model once per worker.

A/B on the same GPUs, same benchmarks, same subset, same batch size. The ONLY difference
is how many times each worker loads the checkpoint:

  A (old, per-cell)  : one `run_eval_dp.sh` invocation per benchmark
                       -> n_benchmarks x n_gpus model loads
  B (new, batched)   : one invocation with the whole benchmark list
                       -> n_gpus model loads

Both paths write the same per-benchmark shard files and are scored by the same parent, so
the comparison is wall clock only; scoring equivalence is proved separately and exactly by
eval/verify_shard_scoring.py.

    python eval/measure_load_amortisation.py --limit 20 \
        --benchmarks where2place refspatial_bench embspatial
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOAD_RE = re.compile(r"\[engine\] loaded .* in ([0-9.]+)s")
REUSE_RE = re.compile(r"\[engine\] reusing loaded")


def run_dp(tag: str, model: str, benches: list[str], subset: int, batch: int,
           log: Path) -> tuple[int, float]:
    env = dict(os.environ)
    env["THINKING"] = env.get("THINKING", "off")
    cmd = ["bash", str(ROOT / "scripts" / "run_eval_dp.sh"), tag, model,
           " ".join(benches), str(subset), str(batch)]
    t = time.time()
    with log.open("w") as f:
        rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return rc, round(time.time() - t, 1)


def scan_loads(stamp_glob: str) -> dict:
    """Count model loads and sum their cost from the worker logs of one run."""
    loads, reuses = [], 0
    for p in sorted((ROOT / "logs").glob(stamp_glob)):
        txt = p.read_text(errors="ignore")
        loads += [float(m.group(1)) for m in LOAD_RE.finditer(txt)]
        reuses += len(REUSE_RE.findall(txt))
    return {"n_loads": len(loads), "load_seconds_total": round(sum(loads), 1),
            "load_seconds_median": (round(sorted(loads)[len(loads) // 2], 1)
                                    if loads else None),
            "n_engine_reuses": reuses}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "output" / "merged_e1"))
    ap.add_argument("--benchmarks", nargs="+",
                    default=["where2place", "refspatial_bench", "embspatial"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "eval_results" /
                                        "load_amortisation.json"))
    a = ap.parse_args()

    if not Path(a.model, "config.json").is_file():
        print(f"FATAL: no model at {a.model}", file=sys.stderr)
        return 2

    import torch
    ng = torch.cuda.device_count()
    print(f"[measure] gpus={ng} benches={a.benchmarks} limit={a.limit}", flush=True)

    report = {"model": a.model, "benchmarks": a.benchmarks, "limit": a.limit,
              "batch_size": a.batch_size, "n_gpus": ng, "arms": {}}

    # --- A: old scheme, one invocation per benchmark -------------------------
    per_bench = {}
    a_total = 0.0
    a_loads = {"n_loads": 0, "load_seconds_total": 0.0, "n_engine_reuses": 0}
    for b in a.benchmarks:
        before = set((ROOT / "logs").glob("dp_measA_multi_shard*.log"))
        log = ROOT / "logs" / f"measure_A_{b}.log"
        rc, wall = run_dp("measA", a.model, [b], a.limit, a.batch_size, log)
        new = sorted(set((ROOT / "logs").glob("dp_measA_multi_shard*.log")) - before)
        loads = {"n_loads": 0, "load_seconds_total": 0.0, "n_engine_reuses": 0}
        ls = []
        for p in new:
            txt = p.read_text(errors="ignore")
            ls += [float(m.group(1)) for m in LOAD_RE.finditer(txt)]
            loads["n_engine_reuses"] += len(REUSE_RE.findall(txt))
        loads["n_loads"] = len(ls)
        loads["load_seconds_total"] = round(sum(ls), 1)
        per_bench[b] = {"wall_s": wall, "rc": rc, **loads}
        a_total += wall
        a_loads["n_loads"] += loads["n_loads"]
        a_loads["load_seconds_total"] = round(
            a_loads["load_seconds_total"] + loads["load_seconds_total"], 1)
        a_loads["n_engine_reuses"] += loads["n_engine_reuses"]
        print(f"[measure] A {b}: wall={wall}s loads={loads['n_loads']} "
              f"load_s={loads['load_seconds_total']} rc={rc}", flush=True)
    report["arms"]["A_per_cell"] = {"total_wall_s": round(a_total, 1),
                                    "per_benchmark": per_bench, **a_loads}

    # --- B: new scheme, one invocation for the whole list --------------------
    before = set((ROOT / "logs").glob("dp_measB_multi_shard*.log"))
    log = ROOT / "logs" / "measure_B_batched.log"
    rc, wall = run_dp("measB", a.model, a.benchmarks, a.limit, a.batch_size, log)
    new = sorted(set((ROOT / "logs").glob("dp_measB_multi_shard*.log")) - before)
    ls, reuses = [], 0
    for p in new:
        txt = p.read_text(errors="ignore")
        ls += [float(m.group(1)) for m in LOAD_RE.finditer(txt)]
        reuses += len(REUSE_RE.findall(txt))
    report["arms"]["B_batched"] = {
        "total_wall_s": wall, "rc": rc, "n_loads": len(ls),
        "load_seconds_total": round(sum(ls), 1),
        "load_seconds_median": (round(sorted(ls)[len(ls) // 2], 1) if ls else None),
        "n_engine_reuses": reuses}
    print(f"[measure] B batched: wall={wall}s loads={len(ls)} "
          f"load_s={round(sum(ls), 1)} reuses={reuses} rc={rc}", flush=True)

    A, B = report["arms"]["A_per_cell"], report["arms"]["B_batched"]
    report["result"] = {
        "wall_A_s": A["total_wall_s"], "wall_B_s": B["total_wall_s"],
        "wall_saved_s": round(A["total_wall_s"] - B["total_wall_s"], 1),
        "speedup": (round(A["total_wall_s"] / B["total_wall_s"], 2)
                    if B["total_wall_s"] else None),
        "loads_A": A["n_loads"], "loads_B": B["n_loads"],
        "load_seconds_A": A["load_seconds_total"],
        "load_seconds_B": B["load_seconds_total"],
        "load_seconds_saved": round(A["load_seconds_total"] - B["load_seconds_total"], 1),
        "engine_reuses_B": B["n_engine_reuses"],
    }
    # Scores must agree between the arms: same model, same subset, same seed. This is a
    # sanity check on the batched worker, not the exactness proof.
    scores = {}
    for b in a.benchmarks:
        row = {}
        for tag in ("measA", "measB"):
            p = ROOT / "eval_results" / tag / f"{b}.json"
            if p.is_file():
                d = json.loads(p.read_text())
                row[tag] = {"score": d.get("score"),
                            "score_if_untrusted": d.get("score_if_untrusted"),
                            "n_evaluated": d.get("n_evaluated")}
        scores[b] = row
    report["scores_per_arm"] = scores

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("\n" + json.dumps(report["result"], indent=2))
    print(json.dumps(scores, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
