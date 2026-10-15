#!/usr/bin/env bash
# run_eval_all_dlc.sh — produce every number the paper's tables need.
#
#   CKPT=output/sft_9b_lora bash scripts/run_eval_all_dlc.sh
#   CKPT=output/grpo_9b_lora TAG=grpo bash scripts/run_eval_all_dlc.sh
#   CKPT=base TAG=base bash scripts/run_eval_all_dlc.sh          # zero-shot row
#
# For a LoRA checkpoint the adapter is merged into a standalone model first, because
# the public benchmark runners take a plain --model path and know nothing about PEFT.
#
# Writes one JSON per benchmark to eval_results/<TAG>/ plus a combined summary, so
# eval/backfill_tables.py can turn them into LaTeX.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || exit 1
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs" "$W/eval_results"

BASE=/path/to/environment/hf_models/Qwen/Qwen3.5-9B
CKPT="${CKPT:-output/sft_9b_lora}"
TAG="${TAG:-$(basename "$CKPT")}"
HOURS="${HOURS:-10}"
OUTD="$W/eval_results/$TAG"
mkdir -p "$OUTD"
LOG="$W/logs/eval_${TAG}_$(date -u +%Y%m%d_%H%M%S).log"

echo "==================================================" | tee -a "$LOG"
echo " eval run : $TAG" | tee -a "$LOG"

# ---- resolve the model path (merging LoRA if needed) -----------------------
if [[ "$CKPT" == "base" ]]; then
  MODEL="$BASE"
  echo " model    : BASE (zero-shot)" | tee -a "$LOG"
else
  ADAPTER=$(ls -d "$W/$CKPT"/checkpoint-* "$W/$CKPT"/v*/checkpoint-* 2>/dev/null \
            | sort -t- -k2 -n | tail -1)
  [[ -n "$ADAPTER" ]] || { echo "FATAL: no checkpoint under $CKPT" | tee -a "$LOG"; exit 1; }
  if [[ -f "$ADAPTER/adapter_model.safetensors" ]]; then
    MERGED="$W/output/merged_$TAG"
    if [[ ! -f "$MERGED/config.json" ]]; then
      echo " merging LoRA -> $MERGED" | tee -a "$LOG"
      swift export --adapters "$ADAPTER" --merge_lora true \
        --output_dir "$MERGED" 2>&1 | tail -5 | tee -a "$LOG"
    fi
    MODEL="$MERGED"
  else
    MODEL="$ADAPTER"          # full-parameter checkpoint: usable directly
  fi
  echo " model    : $MODEL" | tee -a "$LOG"
fi
[[ -f "$MODEL/config.json" ]] || { echo "FATAL: bad model dir $MODEL" | tee -a "$LOG"; exit 1; }

NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 1)
echo " GPUs     : $NG" | tee -a "$LOG"
echo " out      : $OUTD" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

RR=/path/to/datasets/datasets/benchmark_eval_all/src
DR=/path/to/datasets/datasets/transferred_datasets/EmbodiedAI_benchmark

run_one () {   # run_one <name> <runner-relative> <extra args...>
  local name="$1"; shift
  local runner="$RR/$1"; shift
  local res="$OUTD/${name}.json"
  if [[ -s "$res" ]]; then echo "  [skip] $name (already done)" | tee -a "$LOG"; return 0; fi
  [[ -f "$runner" ]] || { echo "  [miss] $name: runner not found" | tee -a "$LOG"; return 0; }
  echo "  [run ] $name" | tee -a "$LOG"
  # The model flag is NOT uniform across runners: some declare --model, others declare
  # --model-path plus --model-name (where a bare --model is an ambiguous prefix and
  # argparse hard-errors). The caller passes the exact flag, so nothing is guessed here.
  ( cd "$(dirname "$runner")" && timeout 2h python "$(basename "$runner")" "$@" ) \
      > "$OUTD/${name}.raw.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  [fail] $name rc=$rc (see ${name}.raw.log)" | tee -a "$LOG"
  fi
  # Harvest whatever score the runner printed. A regex scrape is only trustworthy when the
  # runner exited 0, so rc is recorded and `trusted` gates the number for the paper tables.
  python - "$OUTD/${name}.raw.log" "$res" "$name" "$rc" <<'PY'
import json, os, re, sys
raw, out, name, rc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
txt = open(raw, errors='ignore').read() if os.path.exists(raw) else ''
pats = [r'(?:overall|final|total|average|acc(?:uracy)?)\D{0,20}?([0-9]+\.[0-9]+)',
        r'([0-9]+\.[0-9]+)\s*%']
score = None
for p in pats:
    m = re.findall(p, txt, re.I)
    if m:
        score = float(m[-1]); break
if score is not None and score <= 1.0:
    score *= 100
json.dump({'benchmark': name, 'score': score, 'parsed': score is not None,
           'runner_rc': rc, 'trusted': bool(rc == 0 and score is not None)},
          open(out, 'w'), indent=2)
print(f"    -> score={score} rc={rc} trusted={rc == 0 and score is not None}")
PY
}

timeout "${HOURS}h" bash -c 'true'   # keep the overall cap explicit in the log

EB="$DR"

run_one vsibench         vsibench_qwen.py \
        --model "$MODEL" --res_dir "$OUTD/vsibench_out" \
        --parquet_file "$EB/VSI-Bench/test-00000-of-00001.parquet" \
        --video_path "$EB/VSI-Bench"
run_one cv_bench         cv_bench_qwen_local.py \
        --model-path "$MODEL" --out-dir "$OUTD/cv_bench_out"
run_one where2place      where2place_qwen_local.py \
        --model-path "$MODEL" --output-dir "$OUTD/where2place_out"
run_one refspatial_bench refsptialeval_qwen_local.py \
        --model-path "$MODEL" --output-dir "$OUTD/refspatial_out" \
        --datapath "$EB/RefSpatial-Bench"
run_one embspatial       embspatial/inference.py \
        --backend local --model-path "$MODEL" \
        --data-path "$EB/EmbSpatial-Bench/embspatial_bench.json" \
        --output-path "$OUTD/embspatial_out.json"

# erqa_api.py / BLINK test_benchmark.py are API-client scripts (they need a served
# endpoint, and erqa_api.py additionally imports tensorflow, absent here). robovqa and
# openeqa have no local data tree under $DR. Left unrun rather than faked.
echo "  [skip] erqa      (erqa_api.py needs tensorflow + a served endpoint)"    | tee -a "$LOG"
echo "  [skip] blink     (test_benchmark.py is an API client: needs --base-url)" | tee -a "$LOG"
echo "  [skip] robovqa   (no local RoboVQA data under \$DR)"                    | tee -a "$LOG"
echo "  [skip] openeqa   (no local OpenEQA data under \$DR)"                     | tee -a "$LOG"

# ---- GSA-Bench held-out test (human GT labels) -----------------------------
if [[ -f "$W/eval/eval_gsa_test.py" ]]; then
  echo "  [run ] gsa_bench_test" | tee -a "$LOG"
  # eval_gsa_test.py is now a thin wrapper over gsa_eval/runner.py; --dump-predictions
  # exists so the scored coordinates can be traced back, which the old script could not.
  python "$W/eval/eval_gsa_test.py" --model "$MODEL" \
    --data "$W/trains/gsa_bench/test.jsonl" \
    --dump-predictions "$W/save/logs/preds_$(basename "$OUTD")_gsa_bench_test.jsonl" \
    --out "$OUTD/gsa_bench_test.json" 2>&1 | tail -12 | tee -a "$LOG"
fi

# ---- summary ---------------------------------------------------------------
python - "$OUTD" <<'PY'
import json, glob, os, sys
d = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(d, '*.json'))):
    if f.endswith('summary.json'): continue
    try: j = json.load(open(f))
    except Exception: continue
    name = j.get('benchmark', os.path.basename(f)[:-5])
    # gsa_bench_test is computed by our own scorer, so it has no runner_rc field
    trusted = j.get('trusted', j.get('score') is not None)
    rows[name] = {'score': j.get('score'), 'trusted': bool(trusted),
                  'runner_rc': j.get('runner_rc')}
json.dump(rows, open(os.path.join(d, 'summary.json'), 'w'), indent=2)
ok = {k: v['score'] for k, v in rows.items() if v['trusted'] and v['score'] is not None}
print("\n=== summary ===")
for k, v in rows.items():
    s = '%.2f' % v['score'] if v['score'] is not None else 'NO SCORE'
    print(f"  {k:<20}{s:<10}{'' if v['trusted'] else '<-- UNTRUSTED (runner failed)'}")
print(f"  trusted: {len(ok)}/{len(rows)}")
if ok:
    print(f"  {'MEAN(trusted)':<20}{sum(ok.values())/len(ok):.2f}")
PY
echo "==================================================" | tee -a "$LOG"
echo " results: $OUTD/summary.json" | tee -a "$LOG"
