#!/usr/bin/env bash
# run_sft_9b_full_dlc.sh
#   9B  = Qwen3.5-9B
#   full = full-parameter (vision tower TRAINED -> this is the anti-forgetting CONTRAST;
#          LoRA cannot demonstrate that claim because it structurally cannot forget)
#   dlc  = submitted as a Volcengine custom task
#
# Platform entrypoint (one line):
#   bash /path/to/Anvil/scripts/run_sft_9b_full_dlc.sh
#
# GPU count, grad accumulation and step count are derived at runtime; the
# effective batch is pinned to 32 so results stay comparable across allocations.
set -uo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W" || { echo "FATAL: $W missing (check the vePFS mount)"; exit 1; }
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

# caches must not touch read-only /path/to/workspace/foundation
export HF_HOME="$W/.cache/hf" HF_HUB_CACHE="$W/.cache/hf/hub"
export TORCH_HOME="$W/.cache/torch" TRITON_CACHE_DIR="$W/.cache/triton"
export XDG_CACHE_HOME="$W/.cache" TMPDIR="$W/.cache/tmp" MPLCONFIGDIR="$W/.cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="$W/logs" WANDB_PROJECT=GSA_Forge
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs" "$W/output"

HOURS="${HOURS:-46}"
MODEL=/path/to/environment/hf_models/Qwen/Qwen3.5-9B
OUT="$W/output/sft_9b_full"

# GPU count from torch, NOT nvidia-smi: nvidia-smi -L ignores CUDA_VISIBLE_DEVICES
# and over-reports on a sliced node, which makes torchrun spawn too many ranks.
NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
[[ "$NG" -ge 1 ]] || { echo "FATAL: torch sees 0 GPUs"; nvidia-smi -L | head -3; exit 1; }
# Memory budget, from E2's own logging.jsonl ("9409.8M Params, 8953.8M Trainable"):
#   bf16 weights 17.53 + bf16 grads 16.68 + Adam m,v fp32 66.71 + fp32 master 33.36
#   = 134.27 GiB of persistent state. zero3 over 8 ranks = 16.78 GiB/GPU, leaving
#   ~27.7 of the 44.53 GiB the driver actually exposes. Step 1 already reported
#   29.69 GiB, so the headroom was ~15 GiB and the first backward pass (which must
#   all-gather a full transformer block on top of the activations) asked for 3.14 more
#   than remained. BS=2 was never viable here; BS=1 alone only halves activations
#   while the 16.78 GiB floor stays, so the default is zero3_offload: it moves the
#   optimizer moments and the fp32 master copy to host RAM, dropping the GPU floor to
#   ~4.3 GiB/GPU at the cost of ~100 GiB pinned host RAM (DLC nodes have 918-1967 GiB).
BS="${BS:-1}"
GA=$(( 32 / (NG * BS) )); [[ "$GA" -lt 1 ]] && GA=1   # effective batch stays 32
STEPS=$(( NG * 500 ))
DS="${DS:-zero3_offload}"
MAXLEN="${MAXLEN:-3072}"

# The per-form files are concatenated in order, so the unshuffled corpus is locally
# clustered (each tenth holds only ~4 of 8 forms). Prefer the shuffled copy.
TRAIN="$W/trains/gsa_bench/train_shuf.jsonl"
[[ -s "$TRAIN" ]] || TRAIN="$W/trains/gsa_bench/train.jsonl"
VAL="$W/trains/gsa_bench/val_small.jsonl"
[[ -s "$VAL" ]] || VAL="$W/trains/gsa_bench/val.jsonl"

RESUME=()
# Rank by numeric step: `sort -t- -k2 -n` splits v0-20260816-044214/checkpoint-N on the
# first '-' and compares the date field, so it resumed from the OLDEST checkpoint.
LAST=$(ls -d "$OUT"/checkpoint-* "$OUT"/v*/checkpoint-* 2>/dev/null \
       | sed 's|.*/checkpoint-\([0-9]\{1,\}\)$|\1 &|' | sort -n -k1 \
       | awk 'NF==2' | tail -1 | cut -d' ' -f2)
[[ -n "$LAST" ]] && RESUME=(--resume_from_checkpoint "$LAST")

LOG="$W/logs/sft_9b_full_dlc_$(date -u +%Y%m%d_%H%M%S).log"
echo "=================================================="
echo " run   : sft_9b_full_dlc (anti-forgetting contrast)"
echo " GPUs  : $NG   per_device=$BS grad_accum=$GA  eff_batch=$(( NG * BS * GA ))"
echo " ds    : $DS   max_length=$MAXLEN"
echo " steps : $STEPS   cap=${HOURS}h"
echo " train : $(basename "$TRAIN")  ($(wc -l < "$TRAIN") rows)"
echo " val   : $(basename "$VAL")  ($(wc -l < "$VAL") rows)"
echo " resume: ${LAST:-none}"
free -g 2>/dev/null | awk 'NR==2{print " host RAM: "$2"G"}'
df -h "$W" 2>/dev/null | awk 'NR==2{print " disk free: "$4}'
echo " log   : $LOG"
echo "=================================================="

NPROC_PER_NODE="$NG" timeout "${HOURS}h" swift sft \
  --model "$MODEL" \
  --dataset "$TRAIN" \
  --val_dataset "$VAL" \
  --tuner_type full \
  --freeze_vit false --freeze_aligner false \
  --torch_dtype bfloat16 --deepspeed "$DS" \
  --max_steps "$STEPS" \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
  --learning_rate 1e-5 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_length "$MAXLEN" --gradient_checkpointing true \
  --logging_steps 20 --eval_steps 500 --save_steps 500 --save_total_limit 3 \
  --load_best_model_at_end true --metric_for_best_model loss \
  --greater_is_better false \
  --output_dir "$OUT" --dataloader_num_workers 16 --dataset_num_proc 32 \
  --report_to wandb "${RESUME[@]}" \
  2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
echo "=================================================="
case "$RC" in
  0)   echo " done. checkpoints in $OUT" ;;
  124) echo " hit ${HOURS}h cap — re-run the SAME command to resume" ;;
  *)   echo " FAILED rc=$RC"; grep -iE 'error|assert|out of memory|Signal 9' "$LOG" | tail -5 ;;
esac
exit "$RC"
