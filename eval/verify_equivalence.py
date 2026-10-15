#!/usr/bin/env python3
"""Prove the fast path scores identically to the old one.

Runs the same seeded subset twice on the same checkpoint:

  A. baseline  -- device_map='auto', one process. This is the old behaviour: swift
                  splits one model across every visible GPU as a pipeline, so one card
                  computes while the rest idle (measured 12-20% utilisation).
  B. fast      -- one process per GPU, device_map='cuda:0', stride-sharded samples,
                  merged and scored once.

Greedy decoding (temperature=0) makes the two runs comparable, but they are not
bit-identical by construction: batch composition changes, and bf16 matmul reductions
are not associative, so a token near a decision boundary can flip. The check therefore
reports both the score delta and per-sample response agreement, and asserts the score
delta is within tolerance rather than demanding identical text.

    python eval/verify_equivalence.py --model <dir> --benchmark where2place --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _env() -> dict:
    e = dict(os.environ)
    e.update({"HF_HOME": str(ROOT / ".cache/hf"),
              "XDG_CACHE_HOME": str(ROOT / ".cache"),
              "TMPDIR": str(ROOT / ".cache/tmp"),
              "MPLCONFIGDIR": str(ROOT / ".cache/mpl"),
              "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    return e


def _run(cmd: list[str], env: dict, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        return subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)


def baseline(model: str, bench: str, limit: int, batch: int, work: Path) -> dict:
    """Old path: single process, pipeline-parallel across all visible GPUs."""
    out = work / "baseline.json"
    dump = work / "baseline_preds.jsonl"
    cmd = [sys.executable, str(ROOT / "eval/run_benchmark.py"),
           "--model", model, "--benchmark", bench, "--out", str(out),
           "--limit", str(limit), "--batch-size", str(batch),
           "--device-map", "auto", "--dump-predictions", str(dump)]
    t = time.time()
    rc = _run(cmd, _env(), work / "baseline.log")
    el = time.time() - t
    res = json.loads(out.read_text()) if out.is_file() else {"error": "no output"}
    res["_wall_s"] = round(el, 1)
    res["_rc"] = rc
    res["_dump"] = str(dump)
    return res


def fast(model: str, bench: str, limit: int, batch: int, work: Path,
         gpus: list[int]) -> dict:
    """New path: one process per GPU, each pinned to a single card."""
    shard_dir = work / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    t = time.time()
    for i, gpu in enumerate(gpus):
        env = _env()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [sys.executable, str(ROOT / "eval/run_benchmark.py"),
               "--model", model, "--benchmark", bench,
               "--out", str(shard_dir / f"result_{i}.json"),
               "--shard", str(i), "--num-shards", str(len(gpus)),
               "--shard-out", str(shard_dir / f"shard_{i}.jsonl"),
               "--limit", str(limit), "--batch-size", str(batch),
               "--device-map", "cuda:0"]
        log = (work / f"fast_shard{i}.log").open("w")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT),
                      log))
    rc = 0
    for p, log in procs:
        if p.wait() != 0:
            rc = 1
        log.close()
    el = time.time() - t
    out = work / "fast.json"
    dump = work / "fast_preds.jsonl"
    if rc == 0:
        cmd = [sys.executable, str(ROOT / "eval/run_benchmark.py"),
               "--model", model, "--benchmark", bench, "--out", str(out),
               "--limit", str(limit), "--elapsed-s", str(round(el, 1)),
               "--score-shards"] + [str(p) for p in sorted(shard_dir.glob("shard_*.jsonl"))] \
              + ["--dump-predictions", str(dump)]
        rc = _run(cmd, _env(), work / "fast_score.log")
    res = json.loads(out.read_text()) if out.is_file() else {"error": "no output"}
    res["_wall_s"] = round(el, 1)
    res["_rc"] = rc
    res["_dump"] = str(dump)
    return res


def compare_dumps(a: Path, b: Path) -> dict:
    """Per-uid response agreement between the two runs."""
    def load(p):
        return {str(json.loads(l)["uid"]): json.loads(l)["response"]
                for l in p.read_text().splitlines() if l.strip()}
    if not (a.is_file() and b.is_file()):
        return {"error": "dump missing"}
    da, db = load(a), load(b)
    shared = sorted(set(da) & set(db))
    identical = sum(1 for u in shared if da[u] == db[u])
    diffs = [u for u in shared if da[u] != db[u]][:3]
    return {"n_compared": len(shared),
            "identical_text": identical,
            "identical_pct": round(100.0 * identical / len(shared), 2) if shared else 0.0,
            "only_in_baseline": len(set(da) - set(db)),
            "only_in_fast": len(set(db) - set(da)),
            "example_differing_uids": diffs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", default="where2place")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpus", default=None, help="comma list, default all visible")
    # A fixed tolerance is not meaningful across sample sizes: at n=24 a single sample is
    # worth 4.17 points, so any one flipped answer must breach 0.5 no matter how correct
    # the merge path is. Measured on cv_bench n=24: sharding shifted the score by 4.16,
    # and so did merely changing batch size from 8 to 12 with NO sharding at all, while
    # the same config re-run twice was 24/24 identical. The perturbation is batch
    # composition under non-associative bf16 reductions, not the shard split. Default is
    # therefore "one sample's worth", and exact scoring equivalence is proved separately
    # and without tolerance by eval/verify_shard_scoring.py.
    ap.add_argument("--tolerance", type=float, default=None,
                    help="max allowed absolute score difference in points; default is "
                         "one sample's worth (100/n), which is the smallest difference "
                         "a single flipped answer can produce")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.gpus:
        gpus = [int(x) for x in a.gpus.split(",")]
    else:
        import torch
        gpus = list(range(torch.cuda.device_count()))
    if len(gpus) < 2:
        print("WARNING: fewer than 2 GPUs visible; the speedup measurement is "
              "meaningless here, only the score equivalence is.", flush=True)

    work = ROOT / ".cache" / "equiv" / f"{a.benchmark}_limit{a.limit}"
    work.mkdir(parents=True, exist_ok=True)
    print(f"[equiv] work dir {work}\n[equiv] gpus {gpus}", flush=True)

    print("[equiv] A: baseline (device_map=auto, pipeline-parallel, 1 process)",
          flush=True)
    base = baseline(a.model, a.benchmark, a.limit, a.batch_size, work)
    print(f"        score={base.get('score')} untrusted={base.get('score_if_untrusted')} "
          f"wall={base.get('_wall_s')}s rc={base.get('_rc')}", flush=True)

    print(f"[equiv] B: fast ({len(gpus)} processes, 1 GPU each)", flush=True)
    fst = fast(a.model, a.benchmark, a.limit, a.batch_size, work, gpus)
    print(f"        score={fst.get('score')} untrusted={fst.get('score_if_untrusted')} "
          f"wall={fst.get('_wall_s')}s rc={fst.get('_rc')}", flush=True)

    agree = compare_dumps(Path(base.get("_dump", "")), Path(fst.get("_dump", "")))

    def eff(d):
        s = d.get("score")
        return s if s is not None else d.get("score_if_untrusted")

    sb, sf = eff(base), eff(fst)
    delta = None if (sb is None or sf is None) else round(abs(sb - sf), 4)
    speedup = (round(base["_wall_s"] / fst["_wall_s"], 2)
               if fst.get("_wall_s") else None)

    n_eval = base.get("n_evaluated") or fst.get("n_evaluated") or a.limit or 0
    tolerance = a.tolerance
    if tolerance is None:
        tolerance = round(100.0 / n_eval, 4) if n_eval else 0.5

    verdict = "FAIL"
    if base.get("_rc") != 0 or fst.get("_rc") != 0:
        verdict = "INCONCLUSIVE (a run failed; see logs)"
    elif delta is not None and delta <= tolerance:
        verdict = "PASS"

    report = {
        "benchmark": a.benchmark, "model": a.model, "limit": a.limit,
        "gpus": gpus,
        "baseline": {k: base.get(k) for k in
                     ("score", "score_if_untrusted", "n_evaluated", "parse_failures",
                      "parse_failure_rate", "truncation_rate", "max_tokens", "_wall_s",
                      "_rc")},
        "fast": {k: fst.get(k) for k in
                 ("score", "score_if_untrusted", "n_evaluated", "parse_failures",
                  "parse_failure_rate", "truncation_rate", "max_tokens", "_wall_s",
                  "_rc")},
        "score_delta": delta, "tolerance": tolerance,
        "tolerance_basis": ("one sample = 100/n points" if a.tolerance is None
                            else "explicit --tolerance"),
        "speedup_wall": speedup,
        "response_agreement": agree,
        "verdict": verdict,
        "note": ("This check varies TWO things at once: device placement (auto pipeline "
                 "vs one GPU per process) AND the sample split, so a text difference is "
                 "not attributable on its own. Measured context: re-running one identical "
                 "config twice gave 24/24 identical text, while changing only the batch "
                 "size (no sharding) shifted cv_bench n=24 by the same 4.16 points that "
                 "sharding did -- the perturbation is batch composition under "
                 "non-associative bf16 reductions. Exact scoring equivalence is proved "
                 "without tolerance by eval/verify_shard_scoring.py (15/15 field-"
                 "identical); eval/verify_shard_generation.sh isolates the generation "
                 "side with device placement held constant."),
    }
    print("\n" + json.dumps(report, indent=2, ensure_ascii=False))
    out = Path(a.out) if a.out else (ROOT / "eval_results" /
                                     f"equivalence_{a.benchmark}_limit{a.limit}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
