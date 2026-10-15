#!/usr/bin/env bash
# run_eval_v2_dlc.sh — evaluate base and E1 on the self-hosted harness, data-parallel.
#
# Platform entrypoint (one line):
#   bash /path/to/Anvil/scripts/run_eval_v2_dlc.sh
#
# THROUGHPUT FIX (this is the reason for v2 of this script)
# The harness built TransformersEngine without device_map, so swift's
# get_default_device_map() saw >=2 visible GPUs, returned 'auto' and split ONE model
# across every card as a PIPELINE. Only one stage computes at a time, so measured GPU
# utilisation sat at 12-20% -- i.e. 1/n_gpu, exactly as pipelining predicts. The 9B bf16
# weights are 19 GiB and fit on a single card, so the split bought nothing.
# Now: one process per GPU (CUDA_VISIBLE_DEVICES=i), each with a full model copy and a
# stride slice of the samples; the parent merges the shards and scores once.
#
# Verified, not assumed. eval/verify_shard_scoring.py replays real recorded responses
# through the single-process scorer and through the stride-shard merge and compares every
# field: 15/15 identical over where2place / refspatial_bench / cv_bench at 2, 3 and 8
# shards, with shard files handed to the parent in REVERSE order to prove responses are
# aligned by uid and not by position. Result: eval_results/equivalence_shard_scoring.json.
# Generation-side control: eval/verify_shard_generation.sh (same GPU, same device_map,
# only the sharding differs).
#
# TIME ESTIMATES BELOW ARE NOT MEASURED AT 8 GPUs. The local box has 2 shared A800s that
# were externally occupied throughout (30-57 GiB resident, a 2-GPU attempt died with
# CUDA OOM caused by other tenants), so no 8-GPU speedup was measured. The 8-GPU figures
# are EXTRAPOLATED from the measured single-process cost and are labelled as such.
#
# TOKEN BUDGET
# The first DLC run used 2048 tokens and truncated 44-63% of samples on the point
# benchmarks, so where2place 35.35 (base) vs 20.52 (SFT) measured truncation, not
# ability. registry.py now gives those benchmarks 4096 and runner.py withholds any score
# whose truncation exceeds 15% or parse failures exceed 25%. Truncation is detected from
# the engine's finish_reason as well as an unclosed <think>, because a response can close
# its reasoning and still be cut off mid-JSON.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || { echo "FATAL: $W missing (check the vePFS mount)"; exit 1; }
# Must precede any python call: torch lives in this venv, not in the container python.
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

# /path/to/workspace/foundation is read-only, so every cache has to live under the project.
export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs" "$W/eval_results" "$W/save/logs"

SUBSET="${SUBSET:-600}"      # per-benchmark cap; 0 = full split
HOURS="${HOURS:-6}"
BATCH="${BATCH:-16}"
BASE=/path/to/environment/hf_models/Qwen/Qwen3.5-9B
STAMP=$(date -u +%Y%m%d_%H%M%S)
MASTER="$W/logs/eval_v2_${STAMP}.log"
# Per-cell wall clock ceiling. A hung cell must not eat the whole deadline.
export CELL_TIMEOUT="${CELL_TIMEOUT:-3h}"

say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$MASTER"; }

# Read from the harness rather than hardcoded, so bumping it in one place is enough.
RESULT_SCHEMA=$(python -c \
  'import sys;sys.path.insert(0,"'"$W"'");from eval.gsa_eval.runner import RESULT_SCHEMA;print(RESULT_SCHEMA)' \
  2>/dev/null || echo 0)
[[ "$RESULT_SCHEMA" != "0" ]] || { echo "FATAL: cannot import the harness"; exit 1; }

# torch, not `nvidia-smi -L | wc -l`: the latter ignores CUDA_VISIBLE_DEVICES and
# over-reports on a sliced node, which previously made an 8-GPU job run on 2.
NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
say "=================================================="
say " GSA-Forge evaluation v2 (data-parallel, 1 model per GPU)"
say " GPUs=$NG  subset=$SUBSET  batch=$BATCH  cap=${HOURS}h  cell_timeout=$CELL_TIMEOUT"
python -c "import torch;print(' GPU0:',torch.cuda.get_device_name(0))" 2>/dev/null | tee -a "$MASTER"
df -h "$W" 2>/dev/null | awk 'NR==2{print " disk free: "$4}' | tee -a "$MASTER"
say "=================================================="
[[ "$NG" -ge 1 ]] || { say "FATAL: torch sees 0 GPUs"; nvidia-smi -L 2>&1 | head -3; exit 1; }
[[ "$NG" -ge 8 ]] || say " NOTE: only $NG GPUs visible; the time estimates below assume 8."

# ---- resolve the E1 checkpoint and merge its LoRA ---------------------------
ADAPTER=$(ls -d "$W"/output/sft_9b_lora_l20/v*/checkpoint-* "$W"/output/sft_9b_lora_l20/checkpoint-* \
                "$W"/output/sft_9b_lora/v*/checkpoint-*    "$W"/output/sft_9b_lora/checkpoint-* \
          2>/dev/null | sort -t- -k2 -n | tail -1)
# E1 was already merged by an earlier run; reuse whichever copy exists before paying
# for another 19 GiB export.
MERGED=""
for cand in "$W/output/merged_e1" "$W/output/merged_sft_9b_lora"; do
  [[ -f "$cand/config.json" ]] && { MERGED="$cand"; break; }
done
if [[ -z "$MERGED" ]]; then
  MERGED="$W/output/merged_e1"
  if [[ -n "$ADAPTER" ]]; then
    say " merging LoRA $ADAPTER -> $MERGED"
    swift export --adapters "$ADAPTER" --merge_lora true --output_dir "$MERGED" \
      >> "$MASTER" 2>&1 || say " WARN: merge failed; E1 rows will be skipped"
  else
    say " WARN: no E1 adapter found; only the base rows will be produced"
  fi
else
  say " E1 merged model: $MERGED"
fi

# ---- run one (model, benchmark) cell, skipping finished work ---------------
# A result counts as done only if it was produced by the CURRENT harness. Measured bug
# this fixes: base/cv_bench and sft/cv_bench were written by the pre-data-parallel run of
# 2026-08-17 08:25 (single process, 6416 s and 4364 s -- 26-102x the 63-230 s the other
# benchmarks take under 8-way DP). Because they carried `score` and `parsed`, the old
# check skipped them, so cv_bench silently kept a stale single-process number: it never
# had a shard directory and its `truncation_rate` field was absent, unlike every
# benchmark that actually ran. `schema_version` makes staleness explicit instead of
# leaving it to be inferred from a wall-clock outlier.
scored () {  # 0 if the json carries a trusted score FROM THIS HARNESS VERSION
  python - "$1" "$RESULT_SCHEMA" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
if d.get("score") is None or not d.get("parsed"):
    sys.exit(1)
if str(d.get("schema_version")) != sys.argv[2]:
    print(f"    stale result (schema {d.get('schema_version')} != {sys.argv[2]}); re-running")
    sys.exit(1)
sys.exit(0)
PY
}

report_cell () {
  local tag="$1" bench="$2"
  local out="$W/eval_results/$tag/${bench}.json"
  local s
  s=$(python - "$out" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("score"), "| untrusted:", d.get("score_if_untrusted"),
          "| trunc%:", d.get("truncation_rate"), "| parsefail%:", d.get("parse_failure_rate"))
except Exception:
    print("none")
PY
)
  say "  ->    $tag/$bench score=$s"
}

# One pass per MODEL, not per (model, benchmark) cell: each worker loads the 9B weights
# once (measured ~105 s) and walks its stride slice of every pending benchmark. The old
# per-cell loop paid that load n_benchmarks x n_gpus times -- 80 loads for 5 benchmarks
# x 2 models x 8 GPUs, all of it with the GPU idle.
run_model () {
  local tag="$1" model="$2"
  mkdir -p "$W/eval_results/$tag"
  [[ -f "$model/config.json" ]] || { say "  SKIP  $tag (model missing)"; return 0; }

  local pending=() b out
  for b in $BENCHES; do
    out="$W/eval_results/$tag/${b}.json"
    if [[ -s "$out" ]] && scored "$out"; then
      say "  SKIP  $tag/$b (already scored, current schema)"
    else
      pending+=("$b")
    fi
  done
  if [[ "${#pending[@]}" -eq 0 ]]; then
    say "  SKIP  $tag (all benchmarks already scored)"; return 0
  fi

  say "  RUN   $tag: ${pending[*]}  (${NG}-way data parallel, 1 model load per worker)"
  bash "$W/scripts/run_eval_dp.sh" "$tag" "$model" "${pending[*]}" "$SUBSET" "$BATCH" \
      >> "$MASTER" 2>&1
  local rc=$?
  say "  ->    $tag batch rc=$rc"
  for b in "${pending[@]}"; do report_cell "$tag" "$b"; done
}

# Cheapest first: a failure surfaces in minutes rather than hours.
BENCHES="${BENCHES:-where2place refspatial_bench cv_bench embspatial gsa_bench_test}"
run_model base "$BASE"
[[ -n "$MERGED" && -f "$MERGED/config.json" ]] && run_model sft_9b_lora "$MERGED"

# ---- summaries + LaTeX backfill -------------------------------------------
for tag in base sft_9b_lora; do
  d="$W/eval_results/$tag"
  [[ -d "$d" ]] || continue
  python - "$d" <<'PY' | tee -a "$MASTER"
import json, glob, os, sys
d = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(d, '*.json'))):
    if f.endswith('summary.json'):
        continue
    try:
        j = json.load(open(f))
    except Exception:
        continue
    rows[j.get('benchmark', os.path.basename(f)[:-5])] = {
        'score': j.get('score'),
        'score_if_untrusted': j.get('score_if_untrusted'),
        'truncation_rate': j.get('truncation_rate'),
        'parse_failure_rate': j.get('parse_failure_rate'),
        'n_evaluated': j.get('n_evaluated')}
json.dump(rows, open(os.path.join(d, 'summary.json'), 'w'), indent=2)
got = {k: v for k, v in rows.items() if v['score'] is not None}
print(f"=== {os.path.basename(d)}: {len(got)}/{len(rows)} trusted ===")
for k, v in rows.items():
    if v['score'] is not None:
        print(f"  {k:<20}{v['score']:.2f}")
    else:
        u = v['score_if_untrusted']
        why = f"WITHHELD (raw {u}, trunc {v['truncation_rate']}%, parsefail {v['parse_failure_rate']}%)" \
              if u is not None else "NOT SCORED"
        print(f"  {k:<20}{why}")
PY
done

say " backfilling LaTeX tables"
python "$W/eval/backfill_tables.py" --results "$W/eval_results" >> "$MASTER" 2>&1 \
  && say " tables written" || say " WARN: backfill failed"

say "=================================================="
say " done. log: $MASTER"
