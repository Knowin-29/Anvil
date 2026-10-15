#!/usr/bin/env bash
# launch_cvbench_pair.sh — score base and the E1 SFT model on CV-Bench concurrently.
#
# Two things the upstream runner cannot do, and why they are needed:
#   1. It writes scratch frames to a hardcoded /tmp/cv_bench_<idx>.jpg, so two concurrent
#      runs delete each other's frames. eval/vendor/cv_bench_local_patched.py uses a
#      per-process scratch dir instead.
#   2. Base emits long <think> chains (~10 s/sample vs ~1.2 s post-SFT), so the full
#      2,638-item split does not fit the eval budget. --per-source keeps the SAME seeded
#      subset for both models, which is what makes the delta comparable.
#
#   bash scripts/launch_cvbench_pair.sh            # 200/source (600 items)
#   PER_SOURCE=0 bash scripts/launch_cvbench_pair.sh   # full split
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || exit 1
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub" HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export TRITON_CACHE_DIR="$W/.cache/triton" PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$W"/.cache/{hf,triton,tmp,mpl}

PER_SOURCE="${PER_SOURCE:-200}"
BASE=/path/to/environment/hf_models/Qwen/Qwen3.5-9B
SFT="$W/output/merged_sft_9b_lora"
RUNNER="$W/eval/vendor/cv_bench_local_patched.py"

for m in "$BASE" "$SFT"; do
  [[ -f "$m/config.json" ]] || { echo "FATAL: missing model $m"; exit 1; }
done

cd "$(dirname "$RUNNER")" || exit 1
CUDA_VISIBLE_DEVICES=0 nohup timeout 3h python "$(basename "$RUNNER")" \
  --model-path "$BASE" --model-name base --per-source "$PER_SOURCE" \
  --out-dir "$W/eval_results/base/cv_bench_out" \
  > "$W/eval_results/base/cv_bench.raw.log" 2>&1 &
echo "base pid=$!"

CUDA_VISIBLE_DEVICES=1 nohup timeout 3h python "$(basename "$RUNNER")" \
  --model-path "$SFT" --model-name sft_9b_lora --per-source "$PER_SOURCE" \
  --out-dir "$W/eval_results/sft_9b_lora/cv_bench_out" \
  > "$W/eval_results/sft_9b_lora/cv_bench.raw.log" 2>&1 &
echo "sft  pid=$!"
