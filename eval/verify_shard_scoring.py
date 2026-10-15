#!/usr/bin/env python3
"""Prove the shard merge path scores identically to the single-process path.

This is the half of the data-parallel change that can be verified *exactly*, with no
GPU and no sampling noise, so it is checked here rather than inferred from a small
end-to-end run.

The risk being tested: `evaluate()` slices samples with a stride (`samples[shard::n]`),
so each worker sees a different subset in a different order, and the parent then rebuilds
the sample list and re-attaches responses. If anything in that path aligned responses to
ground truth *by position* instead of by uid, every score would be silently wrong while
still looking plausible. Real recorded model responses are replayed through both paths
and the full result records are compared field by field.

    python eval/verify_shard_scoring.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gsa_eval.datasets import stratified_subset  # noqa: E402
from eval.gsa_eval.registry import get_spec  # noqa: E402
from eval.gsa_eval.runner import _score_and_package, score_shards  # noqa: E402

# (benchmark, dump file) pairs of real recorded responses.
CASES = [
    ("where2place", "runs/eval_v2_20260817/replay_predictions/preds_base_where2place.jsonl"),
    ("where2place", "runs/eval_v2_20260817/replay_predictions/preds_sft_9b_lora_where2place.jsonl"),
    ("refspatial_bench", "runs/eval_v2_20260817/replay_predictions/preds_base_refspatial_bench.jsonl"),
    ("refspatial_bench", "runs/eval_v2_20260817/replay_predictions/preds_sft_9b_lora_refspatial_bench.jsonl"),
    ("cv_bench", "runs/eval_v2_20260817/replay_predictions/preds_base_cv_bench.jsonl"),
]

# Fields that legitimately differ between the two paths (wall clock) or restate it.
IGNORE = {"elapsed_s", "notes", "model"}


def single_process(spec, samples, by_uid, cap_by_uid, n_total, limit, max_tokens):
    """The unsharded path: score the ordered sample list in one go."""
    base = {"benchmark": spec.name, "model": "replay", "metric": spec.metric,
            "score": None, "n_evaluated": 0, "parse_failures": 0, "parsed": False}
    texts = [by_uid[str(s.uid)] for s in samples]
    capped = [cap_by_uid.get(str(s.uid), False) for s in samples]
    return _score_and_package(base, spec, samples, texts, n_total=n_total, limit=limit,
                              max_tokens=max_tokens, started=0.0,
                              dump_predictions=None, capped=capped)


def run_case(bench: str, dump: Path, num_shards: int, limit: int) -> dict:
    spec = get_spec(bench)
    scratch = ROOT / ".cache" / "eval_assets"
    all_samples = spec.loader(scratch)
    n_total = len(all_samples)
    samples = stratified_subset(all_samples, limit, seed=0)

    rows = [json.loads(l) for l in dump.read_text().splitlines() if l.strip()]
    by_uid = {str(r["uid"]): (r.get("response") or "") for r in rows}
    cap_by_uid = {str(r["uid"]): bool(r.get("hit_token_cap")) for r in rows}
    # Only keep samples the dump actually covers, so a partial dump does not look like
    # a shard failure.
    samples = [s for s in samples if str(s.uid) in by_uid]
    if not samples:
        return {"benchmark": bench, "dump": dump.name, "skipped": "no overlap"}
    # A dump may cover only part of a split (cv_bench was run with --limit). score_shards
    # rebuilds the full sample list and treats absent uids as a dead shard, which is the
    # right production behaviour, so the comparison is restricted to the covered subset
    # by passing that subset in as the shard content and as the reference.
    if len(samples) < len(stratified_subset(all_samples, limit, seed=0)):
        by_uid = {str(s.uid): by_uid[str(s.uid)] for s in samples}
        cap_by_uid = {str(s.uid): cap_by_uid.get(str(s.uid), False) for s in samples}
        limit = len(samples)
        covered = {str(s.uid) for s in samples}
        if {str(s.uid) for s in stratified_subset(all_samples, limit, seed=0)} != covered:
            return {"benchmark": bench, "dump": dump.name,
                    "skipped": f"partial dump ({len(samples)}/{n_total}) is not a "
                               f"reproducible stratified subset; needs a full-split dump"}

    max_tokens = spec.max_tokens
    ref = single_process(spec, samples, by_uid, cap_by_uid, n_total, limit, max_tokens)

    # Write stride shards exactly as the workers would, then score through the parent.
    with tempfile.TemporaryDirectory() as td:
        files = []
        for i in range(num_shards):
            p = Path(td) / f"shard_{i}.jsonl"
            with p.open("w") as f:
                for s in samples[i::num_shards]:
                    f.write(json.dumps({"uid": s.uid,
                                        "response": by_uid[str(s.uid)],
                                        "hit_token_cap": cap_by_uid.get(str(s.uid),
                                                                        False)},
                                       ensure_ascii=False) + "\n")
            files.append(p)
        # Shards are deliberately handed over in reverse order: a real run finishes them
        # out of order, and the parent must not depend on that order.
        got = score_shards(bench, "replay", list(reversed(files)), limit=limit,
                           max_tokens=max_tokens, scratch=scratch)

    diffs = {}
    for k in sorted(set(ref) | set(got)):
        if k in IGNORE:
            continue
        if ref.get(k) != got.get(k):
            diffs[k] = {"single_process": ref.get(k), "sharded": got.get(k)}
    return {"benchmark": bench, "dump": dump.name, "num_shards": num_shards,
            "n_scored": len(samples), "score": ref.get("score"),
            "score_if_untrusted": ref.get("score_if_untrusted"),
            "parse_failures": ref.get("parse_failures"),
            "identical": not diffs, "diffs": diffs}


def main() -> int:
    reports = []
    for bench, rel in CASES:
        dump = ROOT / rel
        if not dump.is_file():
            reports.append({"benchmark": bench, "dump": rel, "skipped": "missing dump"})
            continue
        for num_shards in (2, 3, 8):
            r = run_case(bench, dump, num_shards, limit=0)
            reports.append(r)
            tick = "OK  " if r.get("identical") else "FAIL"
            if r.get("skipped"):
                tick = "SKIP"
            print(f"[{tick}] {bench:18} {dump.name:42} shards={num_shards} "
                  f"n={r.get('n_scored')} score={r.get('score')}"
                  f"{'' if r.get('identical') else ' DIFFS: ' + json.dumps(r.get('diffs'))}",
                  flush=True)

    checked = [r for r in reports if "identical" in r]
    ok = all(r["identical"] for r in checked)
    out = {"verdict": "PASS" if ok and checked else "FAIL",
           "n_comparisons": len(checked),
           "note": ("real recorded responses replayed through the single-process scorer "
                    "and through the stride-shard merge; shard files are handed to the "
                    "parent in reverse order to prove uid matching, not position, is "
                    "what aligns a response to its ground truth"),
           "cases": reports}
    dest = ROOT / "eval_results" / "equivalence_shard_scoring.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\n{out['verdict']}: {len(checked)} comparisons\nwrote {dest}")
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
