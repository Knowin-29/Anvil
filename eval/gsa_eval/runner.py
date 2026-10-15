"""Inference + scoring driver.

Inference goes through swift's TransformersEngine: vLLM is not installed in this
environment, and TransformersEngine is the path already verified to load the merged
Qwen3.5 checkpoint correctly.

Parallelism: one process per GPU, each holding a full copy of the model and taking a
stride-`n` slice of the samples. The previous code passed no `device_map`, so swift's
`get_default_device_map()` saw >=2 visible GPUs, returned `'auto'` and split the single
model across every card as a pipeline -- with one stage active at a time, measured GPU
utilisation sat at 12-20%, matching 1/n_gpu. The 9B bf16 weights are 19 GiB, so a
single card holds the whole model and pipelining buys nothing.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .registry import get_spec
from .scorers import score_gsa_geometry as sc_geom
from .scorers import score_multiple_choice as sc_mc

# Above these rates a score reflects the generation budget rather than the model, so it is
# withheld instead of being reported. Measured basis: at 2048 tokens where2place truncated
# 44% (base) / 47% (SFT) of samples and produced 35.35 vs 20.52, which reads as a large
# regression but is an artefact of cutting reasoning off mid-sentence.
TRUNCATION_LIMIT = 15.0      # percent of samples with an unterminated <think> span
PARSE_FAILURE_LIMIT = 25.0   # percent of samples whose answer could not be parsed

# Stamped into every scored result so a resumable driver can tell "already done" from
# "produced by an older harness". Measured need: base/cv_bench and sft/cv_bench survived
# a whole re-run because the skip check only looked at `score`/`parsed`, so a stale
# single-process result (6416 s vs the 63-230 s the same node needs under 8-way data
# parallel) was treated as finished and cv_bench never went through the fast path.
# Bump this whenever a change alters the scores a benchmark produces.
#   3: prompt-driven scale for point-in-mask + referent_type in the registry
#   4: gsa_bench_test prompts now declare the output form (the bbox rows' prompt was
#      `Locate "X" in the image.` and named none, so every point answer was booked a
#      parse failure -- 105/300 base, 232/300 SFT); parse_points/parse_boxes also
#      recognise the corpus's own key aliases explicitly instead of via a regex fallback
#   5: result records include the evaluation-code fingerprint and complete generation
#      protocol; trust-gate remediation distinguishes truncation from format failure
#   6: non-thinking, task-semantic output budgets (128 MC; 256 structured) replace
#      2048/4096-token free-form budgets and are checked by the result registry
#   7: CV-Bench receives the same direct-letter response instruction as EmbSpatial.
#      The source prompt omitted an output constraint, causing otherwise answerable
#      multiple-choice generations to fill the concise budget with prose.
RESULT_SCHEMA = 7
EVALUATION_PROTOCOL = "gsa-eval-v7"


def _resolve_max_tokens(spec, requested: int) -> int:
    """Use the preregistered task budget unless the caller explicitly overrides it."""
    if requested <= 0:
        return spec.max_tokens
    return requested


def evaluation_code_sha256() -> str:
    """Fingerprint every local module that can change sample selection or scores."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("datasets.py", "parsing.py", "registry.py", "runner.py", "scorers.py"):
        path = root / name
        digest.update(name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _engine(model: str, max_batch_size: int, device_map: str | None = "cuda:0",
            thinking: bool | None = None):
    """Build the inference engine, pinned to one device by default.

    `device_map` is forwarded to swift's loader. Leaving it None restores the old
    `'auto'` pipeline-parallel behaviour and is kept only so the equivalence check can
    run both paths back to back.

    `thinking=False` clears the template's response prefix. Measured need: the qwen3_5
    chat template appends `'<think>\\n'` to every assistant turn unconditionally
    (chat_template.jinja:152, and swift logs `response_prefix: '<think>\\n'` at load),
    so 100/100 recorded responses opened a reasoning span even though the SFT training
    answers are bare JSON arrays with no `<think>` at all. The reasoning span is what
    overruns the token budget. Appending `/no_think` to the prompt does not help
    (measured 11/12 still emitted `<think>`) because the prefix is added after the
    prompt, on the assistant side; this switch is the only thing that reaches it.
    """
    from swift import RequestConfig  # noqa: F401  (imported for callers' convenience)
    from swift.infer_engine import TransformersEngine
    kwargs = {} if device_map is None else {"device_map": device_map}
    engine = TransformersEngine(model, max_batch_size=max_batch_size, **kwargs)
    if thinking is not None:
        tmpl = getattr(engine, "template", None)
        if tmpl is None:
            raise RuntimeError("engine exposes no template; cannot set thinking mode")
        meta = getattr(tmpl, "template_meta", None)
        prefix = ""
        if thinking:
            prefix = getattr(meta, "thinking_prefix", "") or "<think>\n"
        else:
            prefix = getattr(meta, "non_thinking_prefix", "") or ""
        tmpl.response_prefix = prefix
        tmpl.enable_thinking = thinking
        print(f"[engine] thinking={thinking} response_prefix={prefix!r}", flush=True)
    return engine


def _assert_weights_loaded(engine) -> None:
    """Refuse to score a model whose weights did not actually load.

    Guards the failure mode from findings section 4.3: an InternVL-style AutoModel load
    of a Qwen3.5 checkpoint returned exit code 0 while every language_model tensor was
    randomly initialised, producing a plausible-looking low score. A random-init model
    is worse than no number at all, because the number looks usable.
    """
    import torch
    model = getattr(engine, "model", None)
    if model is None:
        return
    bad = []
    for name, p in model.named_parameters():
        if p.is_meta:
            bad.append(name)
        elif p.numel() and not torch.isfinite(p.reshape(-1)[:8].float()).all():
            bad.append(name)
        if len(bad) >= 5:
            break
    if bad:
        raise RuntimeError(f"model weights not properly loaded, e.g. {bad[:5]}")


def _infer(engine, samples, *, max_tokens: int) -> tuple[list[str], list[bool]]:
    """Greedy-decode one response per sample on an already-built engine.

    Split out from `_generate` so one process can serve several benchmarks from a single
    set of loaded weights. Measured motivation: loading the 9B checkpoint takes ~105 s
    while a shard's generation for the cheap benchmarks takes 63-230 s, so paying the
    load once per (benchmark, model) cell spent ~40% of the wall clock with the GPU idle
    -- 80 loads across 10 cells x 8 workers.
    """
    from swift import InferRequest, RequestConfig
    cfg = RequestConfig(max_tokens=max_tokens, temperature=0.0)
    requests = [InferRequest(messages=[{"role": "user", "content": s.prompt}],
                             images=list(s.image_paths)) for s in samples]
    responses = engine.infer(requests, cfg)
    texts: list[str] = []
    capped: list[bool] = []
    for r in responses:
        try:
            texts.append(r.choices[0].message.content or "")
            capped.append(r.choices[0].finish_reason == "length")
        except Exception:
            texts.append("")
            capped.append(False)
    return texts, capped


# One engine per (model, batch_size, device_map, thinking) within a process. Keyed on
# every argument that changes the weights or the template, so a cache hit cannot
# silently serve a differently-configured engine.
_ENGINE_CACHE: dict[tuple, object] = {}


def get_engine(model: str, batch_size: int, device_map: str | None,
               thinking: bool | None):
    """Build the engine once per process and reuse it across benchmarks."""
    key = (model, batch_size, device_map, thinking)
    eng = _ENGINE_CACHE.get(key)
    if eng is None:
        t0 = time.time()
        eng = _engine(model, batch_size, device_map=device_map, thinking=thinking)
        _assert_weights_loaded(eng)
        print(f"[engine] loaded {model} in {time.time() - t0:.1f}s", flush=True)
        _ENGINE_CACHE[key] = eng
    else:
        print(f"[engine] reusing loaded {model} (no reload)", flush=True)
    return eng


def _generate(model: str, samples, *, max_tokens: int, batch_size: int,
              device_map: str | None,
              thinking: bool | None = None) -> tuple[list[str], list[bool]]:
    """Decode one response per sample, reusing a cached engine when possible."""
    engine = get_engine(model, batch_size, device_map, thinking)
    return _infer(engine, samples, max_tokens=max_tokens)


def evaluate(benchmark: str, model: str, *, limit: int = 0, max_tokens: int = 0,
             batch_size: int = 8, scratch: Path, data: str | None = None,
             seed: int = 0, dump_predictions: Path | None = None,
             shard: int = 0, num_shards: int = 1, device_map: str | None = "cuda:0",
             shard_out: Path | None = None, thinking: bool | None = None) -> dict:
    """Run one benchmark end-to-end and return the result record.

    With `num_shards > 1` this process generates only its own stride slice and, if
    `shard_out` is given, writes the raw responses there instead of scoring; the parent
    merges the shards and scores once, so the score is identical to a single-process run
    (verified by eval/verify_equivalence.py).
    """
    spec = get_spec(benchmark)
    started = time.time()
    base = {"benchmark": benchmark, "model": model, "metric": spec.metric,
            "score": None, "n_evaluated": 0, "parse_failures": 0, "parsed": False}

    if not spec.available:
        base["notes"] = f"skipped: {spec.skip_reason}"
        base["skipped"] = spec.skip_reason
        return base

    kwargs = {"data": data} if (data and benchmark == "gsa_bench_test") else {}
    samples = spec.loader(scratch, **kwargs)
    n_total = len(samples)
    if not samples:
        base["notes"] = "skipped: dataset loaded zero samples"
        base["skipped"] = "empty dataset"
        return base

    from .datasets import stratified_subset
    samples = stratified_subset(samples, limit, seed=seed)
    max_tokens = _resolve_max_tokens(spec, max_tokens)
    print(f"[{benchmark}] {len(samples)}/{n_total} samples, max_tokens={max_tokens}",
          flush=True)

    # Stride slicing, not contiguous blocks: every shard then sees a comparable mix of
    # groups, so a shard that dies leaves a subset that is still stratified.
    all_samples = samples
    if num_shards > 1:
        samples = all_samples[shard::num_shards]
        print(f"[{benchmark}] shard {shard}/{num_shards}: {len(samples)} samples",
              flush=True)
        if not samples:
            base["notes"] = "empty shard"
            return base

    texts, capped = _generate(model, samples, max_tokens=max_tokens,
                              batch_size=batch_size, device_map=device_map,
                              thinking=thinking)

    # Shard workers stop here: scoring happens once in the parent on the merged set, so
    # per-group rates and the trust gate are computed over the whole benchmark rather
    # than per shard.
    if shard_out is not None:
        shard_out.parent.mkdir(parents=True, exist_ok=True)
        with shard_out.open("w") as f:
            for s, t, c in zip(samples, texts, capped):
                f.write(json.dumps({"uid": s.uid, "response": t, "hit_token_cap": c},
                                   ensure_ascii=False) + "\n")
        base.update({"shard": shard, "num_shards": num_shards,
                     "n_evaluated": len(samples),
                     "elapsed_s": round(time.time() - started, 1),
                     "notes": f"shard {shard}/{num_shards} written to {shard_out}"})
        return base

    return _score_and_package(base, spec, samples, texts, n_total=n_total, limit=limit,
                              max_tokens=max_tokens, started=started,
                              dump_predictions=dump_predictions, capped=capped,
                              thinking=thinking, seed=seed,
                              generation_batch_size=batch_size)


def generate_shard_multi(benchmarks: list[str], model: str, *, shard: int,
                         num_shards: int, shard_dir: Path, limit: int = 0,
                         max_tokens: int = 0, batch_size: int = 8, scratch: Path,
                         data: str | None = None, seed: int = 0,
                         device_map: str | None = "cuda:0",
                         thinking: bool | None = None) -> dict:
    """Generate this shard's slice of SEVERAL benchmarks from one loaded model.

    The per-benchmark shard files this writes are byte-for-byte the same shape as the
    single-benchmark worker's, so the parent's uid-keyed `score_shards` merge is
    unchanged and stays covered by eval/verify_shard_scoring.py.

    Existing behaviour is preserved exactly per benchmark: the same loader, the same
    seeded `stratified_subset`, the same `spec.max_tokens` promotion and the same
    `samples[shard::num_shards]` stride. Only the engine is shared.

    A benchmark that fails is recorded and the loop continues, because losing one
    benchmark must not discard the shards already generated for the others -- that would
    reintroduce the reload cost it exists to avoid.
    """
    from .datasets import stratified_subset
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    per: dict[str, dict] = {}
    engine = None
    for bench in benchmarks:
        t0 = time.time()
        try:
            spec = get_spec(bench)
            if not spec.available:
                per[bench] = {"status": "skipped", "reason": spec.skip_reason}
                continue
            kwargs = {"data": data} if (data and bench == "gsa_bench_test") else {}
            samples = spec.loader(scratch, **kwargs)
            n_total = len(samples)
            samples = stratified_subset(samples, limit, seed=seed)
            mt = _resolve_max_tokens(spec, max_tokens)
            mine = samples[shard::num_shards] if num_shards > 1 else samples
            if not mine:
                per[bench] = {"status": "empty_shard", "n": 0}
                continue
            print(f"[{bench}] shard {shard}/{num_shards}: {len(mine)}/{n_total} "
                  f"samples, max_tokens={mt}", flush=True)
            # Built lazily so a dataset that fails to load costs nothing, and cached so
            # every later benchmark in this list reuses the same weights.
            if engine is None:
                engine = get_engine(model, batch_size, device_map, thinking)
            texts, capped = _infer(engine, mine, max_tokens=mt)
            out = shard_dir / bench / f"shard_{shard}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as f:
                for s, t, c in zip(mine, texts, capped):
                    f.write(json.dumps({"uid": s.uid, "response": t,
                                        "hit_token_cap": c},
                                       ensure_ascii=False) + "\n")
            per[bench] = {"status": "ok", "n": len(mine), "max_tokens": mt,
                          "elapsed_s": round(time.time() - t0, 1),
                          "shard_file": str(out)}
            print(f"[{bench}] shard {shard} done in "
                  f"{per[bench]['elapsed_s']}s -> {out}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            per[bench] = {"status": "error", "error": f"{type(e).__name__}: {e}",
                          "elapsed_s": round(time.time() - t0, 1)}
    n_ok = sum(1 for v in per.values() if v.get("status") == "ok")
    return {"model": model, "shard": shard, "num_shards": num_shards,
            "benchmarks": per, "n_ok": n_ok,
            "n_failed": sum(1 for v in per.values() if v.get("status") == "error"),
            "engine_loads": 0 if engine is None else 1,
            "thinking": thinking}


def _parse_success(spec, samples, texts: list[str]) -> list[bool]:
    """Per-sample: did an answer come out at all, independently of correctness.

    Recomputed with the same parsers the scorer uses, so "parsed" here means exactly what
    it means inside the score. Needed for the paired-subset comparison (audit defect F3):
    the two models truncate at different rates, so the full-set gap partly measures the
    token budget, and the honest comparison also has to be reported on the samples where
    both models produced an answer.
    """
    from .parsing import parse_boxes, parse_choice, parse_points
    out: list[bool] = []
    for s, t in zip(samples, texts):
        try:
            if spec.scorer is sc_mc:
                n = len(s.meta.get("choices") or []) or 4
                out.append(parse_choice(t, n) is not None)
            elif spec.scorer is sc_geom:
                out.append(bool(parse_boxes(t) if s.meta.get("form") == "bbox"
                                else parse_points(t)))
            else:
                out.append(bool(parse_points(t)))
        except Exception:
            out.append(False)
    return out


def _score_and_package(base: dict, spec, samples, texts: list[str], *, n_total: int,
                       limit: int, max_tokens: int, started: float,
                       dump_predictions: Path | None,
                       capped: list[bool] | None = None,
                       thinking: bool | None = None, seed: int = 0,
                       generation_batch_size: int | None = None) -> dict:
    """Score a full set of responses and apply the trust gate."""
    # Point scorers need the spec to know what the prompt declares (coordinate scale,
    # referent type). Passed by keyword only to scorers that accept it, so the
    # multiple-choice scorers keep their two-argument signature.
    import inspect
    if "spec" in inspect.signature(spec.scorer).parameters:
        score, detail, fails = spec.scorer(samples, texts, spec=spec)
    else:
        score, detail, fails = spec.scorer(samples, texts)
    elapsed = round(time.time() - started, 1)
    # Two independent truncation signals, unioned. The unclosed-<think> heuristic alone
    # under-counts: a response can close its reasoning and still be cut off mid-JSON,
    # which scores 0 while looking like a well-formed wrong answer. finish_reason
    # 'length' catches those; the tag check still covers dumps that carry no flag.
    unclosed = [("<think>" in t.lower() and "</think>" not in t.lower())
                for t in texts]
    if capped is None:
        capped = [False] * len(texts)
    truncated = sum(1 for u, c in zip(unclosed, capped) if u or c)
    n_unclosed = sum(unclosed)
    n_capped = sum(capped)
    parsed_ok = _parse_success(spec, samples, texts)

    if dump_predictions is not None:
        dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        with dump_predictions.open("w") as f:
            for s, t, c in zip(samples, texts, capped):
                f.write(json.dumps({"uid": s.uid, "group": s.group,
                                    "gt": s.gt if isinstance(s.gt, (str, int, list)) else str(s.gt),
                                    "hit_token_cap": c,
                                    "response": t}, ensure_ascii=False) + "\n")

    notes = [f"n_total={n_total}", f"subset={'stratified' if limit else 'full'}",
             f"elapsed_s={elapsed}"]
    trunc_rate = 100.0 * truncated / len(samples)
    fail_rate = 100.0 * fails / len(samples)
    if truncated:
        notes.append(f"truncated={truncated} ({trunc_rate:.0f}%; unclosed_think="
                     f"{n_unclosed}, hit_token_cap={n_capped})")

    # A score computed mostly from truncated generations measures the token budget, not the
    # model. This gate exists because a 2048-token run produced where2place 35.35 (base)
    # vs 20.52 (SFT) with 44% and 47% of samples cut off mid-reasoning -- a difference that
    # looked like a regression but was an artefact. Such a run must not silently feed the
    # paper's tables, so the score is withheld and the reason recorded.
    trustworthy = trunc_rate <= TRUNCATION_LIMIT and fail_rate <= PARSE_FAILURE_LIMIT
    if not trustworthy:
        remedies = []
        if trunc_rate > TRUNCATION_LIMIT:
            remedies.append(
                "audit concise task compliance; do not expand the preregistered "
                "task-semantic budget post hoc")
        if fail_rate > PARSE_FAILURE_LIMIT:
            remedies.append("audit task/format compliance; do not relax the parser post hoc")
        notes.append(
            f"UNTRUSTWORTHY: truncation {trunc_rate:.0f}% (limit {TRUNCATION_LIMIT}%), "
            f"parse failures {fail_rate:.0f}% (limit {PARSE_FAILURE_LIMIT}%); "
            + "; ".join(remedies))

    base.update({
        "schema_version": RESULT_SCHEMA,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "evaluation_code_sha256": evaluation_code_sha256(),
        "score": score if trustworthy else None,
        "score_if_untrusted": None if trustworthy else score,
        "n_evaluated": len(samples), "n_total": n_total,
        "parse_failures": fails,
        "parse_failure_rate": round(fail_rate, 2),
        "parsed": trustworthy, "detail": detail, "max_tokens": max_tokens,
        "truncated_reasoning": truncated,
        "truncated_unclosed_think": n_unclosed,
        "truncated_hit_token_cap": n_capped,
        "truncation_rate": round(trunc_rate, 2),
        # F3: unparsed responses score 0 (deliberately -- a model that cannot produce the
        # requested format is not doing the task). But the rate differs between models,
        # so the full-set score partly measures the generation budget. The per-uid
        # parse-success list lets eval/compare_models.py recompute both models on the
        # subset where BOTH parsed, which is the truncation-free comparison. Recorded here
        # rather than derived later so it cannot drift from the scored run.
        #
        # Sorted because these are SETS: the sharded path rebuilds the sample list from
        # the loader while the single-process path keeps its own order, so an unsorted
        # list would report a spurious difference in the equivalence check even though
        # every score matches. (It did: 3/15 comparisons flagged only these two fields,
        # with identical membership.)
        "parsed_uids": sorted(str(s.uid) for s, ok in zip(samples, parsed_ok) if ok),
        "unparsed_uids": sorted(str(s.uid) for s, ok in zip(samples, parsed_ok)
                                if not ok),
        "thinking": thinking,
        "subset_seed": seed,
        "subset_limit": limit,
        "generation_batch_size": generation_batch_size,
        "elapsed_s": elapsed,
        "notes": "; ".join(notes)})
    return base


def score_shards(benchmark: str, model: str, shard_files: list[Path], *, limit: int = 0,
                 max_tokens: int = 0, scratch: Path, data: str | None = None,
                 seed: int = 0, dump_predictions: Path | None = None,
                 elapsed_s: float | None = None, thinking: bool | None = None,
                 generation_batch_size: int | None = None) -> dict:
    """Merge per-GPU shard dumps and score them as one run.

    The sample list is rebuilt with the same loader, limit and seed as the workers used,
    then responses are matched by uid rather than by position, so a shard finishing out
    of order cannot silently misalign a response with another sample's ground truth.
    """
    spec = get_spec(benchmark)
    started = time.time() - (elapsed_s or 0.0)
    base = {"benchmark": benchmark, "model": model, "metric": spec.metric,
            "score": None, "n_evaluated": 0, "parse_failures": 0, "parsed": False}
    if not spec.available:
        base["notes"] = f"skipped: {spec.skip_reason}"
        base["skipped"] = spec.skip_reason
        return base

    kwargs = {"data": data} if (data and benchmark == "gsa_bench_test") else {}
    samples = spec.loader(scratch, **kwargs)
    n_total = len(samples)
    from .datasets import stratified_subset
    samples = stratified_subset(samples, limit, seed=seed)
    max_tokens = _resolve_max_tokens(spec, max_tokens)

    by_uid: dict[str, str] = {}
    cap_by_uid: dict[str, bool] = {}
    for p in shard_files:
        if not Path(p).is_file():
            continue
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            by_uid[str(row["uid"])] = row.get("response") or ""
            cap_by_uid[str(row["uid"])] = bool(row.get("hit_token_cap"))

    missing = [s.uid for s in samples if str(s.uid) not in by_uid]
    if missing:
        # Scoring a partial set as if it were complete would count absent responses as
        # wrong answers and understate the model, so this is a hard failure.
        raise RuntimeError(
            f"{len(missing)}/{len(samples)} responses missing from shards "
            f"(e.g. {missing[:5]}); a shard failed -- re-run it before scoring")

    texts = [by_uid[str(s.uid)] for s in samples]
    capped = [cap_by_uid.get(str(s.uid), False) for s in samples]
    return _score_and_package(base, spec, samples, texts, n_total=n_total, limit=limit,
                              max_tokens=max_tokens, started=started,
                              dump_predictions=dump_predictions, capped=capped,
                              thinking=thinking, seed=seed,
                              generation_batch_size=generation_batch_size)
