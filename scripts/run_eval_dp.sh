#!/usr/bin/env bash
# run_eval_dp.sh — data-parallel evaluation of one model over one or more benchmarks.
#
#   bash scripts/run_eval_dp.sh <tag> <model_dir> "<bench> [bench ...]" [subset] [batch]
#
# One process per GPU, each pinned with CUDA_VISIBLE_DEVICES to a single card and
# holding a full copy of the model; each takes a stride slice of the samples. The
# parent then merges the shards and scores once, so the score is identical to a
# single-process run (see eval/verify_shard_scoring.py).
#
# Why not swift's default: TransformersEngine was constructed without device_map, so
# swift's get_default_device_map() saw >=2 visible GPUs and returned 'auto', splitting
# ONE model across every card as a pipeline. Only one stage computes at a time, so
# measured GPU utilisation was 12-20% -- i.e. 1/n_gpu. The 9B bf16 weights are 19 GiB
# and fit on a single card, so pipelining bought nothing and cost n_gpu-fold throughput.
#
# WHY A BENCHMARK LIST (the second throughput fix)
# Loading the 9B checkpoint is measured at ~105 s, while a shard's generation for the
# cheap benchmarks is 63-230 s. With one process per (benchmark, model) cell that load
# was ~40% of the wall clock with the GPU idle, and 5 benchmarks x 2 models x 8 workers
# meant 80 loads. Passing several benchmarks makes each worker load the weights ONCE and
# walk its own stride slice of every benchmark in turn.
#
# Equivalence is untouched by this: the worker writes the same per-benchmark
# shard_<i>.jsonl files, and the parent still merges them by uid.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || { echo "FATAL: $W missing"; exit 1; }
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DeepSpeed is imported indirectly by Transformers/Accelerate while loading and
# saving checkpoints.  This node has working PyTorch CUDA wheels but no local
# CUDA toolkit, so prevent the optional op builder from probing for nvcc.
export DS_IGNORE_CUDA_DETECTION=1 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=PIX
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs" "$W/save/logs"

TAG="${1:?usage: run_eval_dp.sh <tag> <model_dir> \"<bench> [bench ...]\" [subset] [batch]}"
MODEL="${2:?model dir}"
BENCH_LIST="${3:?benchmark(s), space separated}"
SUBSET="${4:-0}"
BATCH="${5:-16}"
CELL_TIMEOUT="${CELL_TIMEOUT:-3h}"

# shellcheck disable=SC2206
BENCHES=(${BENCH_LIST})
[[ "${#BENCHES[@]}" -ge 1 ]] || { echo "FATAL: empty benchmark list"; exit 2; }

# Thinking mode. The qwen3_5 chat template unconditionally appends '<think>\n' to the
# assistant turn, so every response opens a reasoning span even though the SFT training
# answers are bare JSON arrays. That span is what overruns the token budget.
#
# Measured on 10 where2place samples with the merged E1 model:
#   thinking on : 6/10 truncated -> those 6 have no parseable answer at all
#   thinking off: 0/10 truncated, 7/10 parseable
# (A naive count of "any JSON array in the raw text" flatters thinking mode, but those
# arrays sit inside the unterminated reasoning and the parser rightly discards them:
# strip_reasoning() returns empty for 44/44 recorded truncated responses.)
#
# So thinking off is both faster and yields more scoreable answers, and it matches the
# format the model was trained to emit. Override with THINKING=on to compare.
case "${THINKING:-off}" in
  off|0|false) THINKING_FLAG="--no-thinking" ;;
  on|1|true)   THINKING_FLAG="--thinking" ;;
  auto)        THINKING_FLAG="" ;;
  *) echo "[dp] FATAL: THINKING must be off|on|auto, got '${THINKING}'" >&2; exit 2 ;;
esac

# torch, never `nvidia-smi -L | wc -l`: the latter ignores CUDA_VISIBLE_DEVICES and
# over-reports on a sliced node, which previously made an 8-GPU job run on 2 cards.
VISIBLE_NG="$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
[[ "$VISIBLE_NG" -ge 1 ]] || { echo "FATAL: torch sees 0 GPUs"; exit 1; }
NG="${NUM_GPUS:-$VISIBLE_NG}"
[[ "$NG" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: NUM_GPUS must be a positive integer, got '$NG'"; exit 2; }
[[ "$NG" -le "$VISIBLE_NG" ]] || {
  echo "FATAL: requested NUM_GPUS=$NG, but torch sees only $VISIBLE_NG visible GPU(s)" >&2
  exit 2
}

# Preserve an upstream GPU slice. Overwriting CUDA_VISIBLE_DEVICES with 0..N-1
# previously made two independently sliced launchers collide on physical GPU 0.
GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  for ((i = 0; i < VISIBLE_NG; i++)); do GPU_IDS+=("$i"); done
fi
[[ "${#GPU_IDS[@]}" -ge "$NG" ]] || {
  echo "FATAL: only ${#GPU_IDS[@]} GPU id(s) in CUDA_VISIBLE_DEVICES for NG=$NG" >&2
  exit 2
}

STAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$W/.cache/shards"
SHARD_ROOT=$(mktemp -d "$W/.cache/shards/eval_${STAMP}_XXXXXX")

echo "[dp] $TAG  benches='${BENCHES[*]}'  gpus=$NG ids='${GPU_IDS[*]:0:$NG}' subset=$SUBSET batch=$BATCH"
START=$(date +%s)

pids=()
cleanup_workers() {
  status=$?
  trap - EXIT INT TERM
  for p in "${pids[@]}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
  for p in "${pids[@]}"; do
    wait "$p" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup_workers EXIT INT TERM

for ((i = 0; i < NG; i++)); do
  log="$W/logs/dp_${TAG}_multi_shard${i}_${STAMP}.log"
  # Each worker sees exactly one physical GPU, so device_map cuda:0 is that card and
  # swift's is_mp() sees a single device and cannot fall back to pipeline parallel.
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" timeout "$CELL_TIMEOUT" \
    python "$W/eval/run_benchmark.py" \
      --model "$MODEL" --benchmarks "${BENCHES[@]}" \
      --shard-dir "$SHARD_ROOT" \
      --out "$SHARD_ROOT/worker_$i.json" \
      --shard "$i" --num-shards "$NG" \
      --limit "$SUBSET" --batch-size "$BATCH" --device-map cuda:0 \
      ${THINKING_FLAG} \
      > "$log" 2>&1 &
  pids+=("$!")
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
GEN_ELAPSED=$(( $(date +%s) - START ))
echo "[dp] generation done in ${GEN_ELAPSED}s for ${#BENCHES[@]} benchmark(s) (rc=$rc)"

# run_benchmark records per-benchmark errors in worker JSON so that one bad dataset
# does not hide later diagnostics. Formal DP evaluation is stricter: every worker and
# every expected shard must exist before any aggregate score is produced.
if ! python - "$SHARD_ROOT" "$NG" "${BENCHES[@]}" <<'PY'
import json
import os
import sys

root, ng, *benches = sys.argv[1:]
ng = int(ng)
errors = []
for i in range(ng):
    path = os.path.join(root, f"worker_{i}.json")
    if not os.path.isfile(path):
        errors.append(f"missing worker result: {path}")
        continue
    try:
        worker = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        errors.append(f"unreadable worker result {path}: {exc}")
        continue
    results = worker.get("benchmarks") or {}
    for bench in benches:
        status = (results.get(bench) or {}).get("status")
        if status != "ok":
            errors.append(f"worker {i} benchmark {bench}: status={status!r}")
        shard = os.path.join(root, bench, f"shard_{i}.jsonl")
        if not os.path.isfile(shard):
            errors.append(f"missing shard: {shard}")
if errors:
    print("[dp] completeness gate failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
print(f"[dp] completeness gate passed: {ng} workers x {len(benches)} benchmarks")
PY
then
  rc=1
fi

if [[ "$rc" -ne 0 ]]; then
  echo "[dp] FATAL: a shard failed; not scoring a partial set. Logs:"
  ls -1 "$W"/logs/dp_"${TAG}"_multi_shard*_"${STAMP}".log
  tail -n 20 -- "$W"/logs/dp_"${TAG}"_multi_shard*_"${STAMP}".log
  exit 1
fi

# Report the measured load cost that this mode amortises, straight from the workers.
python - "$SHARD_ROOT" "$GEN_ELAPSED" "${#BENCHES[@]}" "$NG" <<'PY'
import glob, json, os, sys
root, gen, nb, ng = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
loads = 0
per = {}
for f in sorted(glob.glob(os.path.join(root, "worker_*.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    loads += int(d.get("engine_loads") or 0)
    for b, v in (d.get("benchmarks") or {}).items():
        if v.get("status") == "ok":
            per.setdefault(b, []).append(v.get("elapsed_s") or 0.0)
print(f"[dp] engine loads this run: {loads} "
      f"(one per worker; the per-cell scheme would have used {nb * ng})")
for b, xs in sorted(per.items()):
    print(f"[dp]   {b:18} shard gen max={max(xs):7.1f}s  sum={sum(xs):8.1f}s  n_shards={len(xs)}")
print(f"[dp] total generation wall: {gen:.0f}s")
PY

# Score once per benchmark, on the merged set: the trust gate and per-group rates must be
# computed over the whole benchmark, not per shard.
src=0
for b in "${BENCHES[@]}"; do
  OUT="$W/eval_results/$TAG/${b}.json"
  mkdir -p "$(dirname "$OUT")"
  if ! compgen -G "$SHARD_ROOT/$b/shard_*.jsonl" > /dev/null; then
    echo "[dp] $b: no shard files (benchmark skipped or empty); not scoring"
    continue
  fi
  python "$W/eval/run_benchmark.py" \
      --model "$MODEL" --benchmark "$b" --out "$OUT" \
      --limit "$SUBSET" --elapsed-s "$GEN_ELAPSED" --batch-size "$BATCH" \
      --score-shards "$SHARD_ROOT/$b"/shard_*.jsonl \
      --dump-predictions "$W/save/logs/preds_${TAG}_${b}.jsonl" \
      ${THINKING_FLAG}
  brc=$?
  echo "[dp] scored $OUT (rc=$brc)"
  # A withheld score (trust gate) is a legitimate outcome, not a script failure, so it
  # must not abort the remaining benchmarks.
  [[ "$brc" -ne 0 ]] && src=1
done
exit "$src"
