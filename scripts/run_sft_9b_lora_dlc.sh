#!/usr/bin/env bash
# run_sft_9b_lora_dlc.sh
#   9B  = Qwen3.5-9B
#   lora = LoRA r64 (vision tower frozen)
#   dlc  = submitted as a Volcengine custom task
#
# Platform entrypoint (one line):
#   bash /path/to/Anvil/scripts/run_sft_9b_lora_dlc.sh
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
OUT="$W/output/sft_9b_lora"

# GPU count from torch, NOT nvidia-smi: nvidia-smi -L ignores CUDA_VISIBLE_DEVICES
# and over-reports on a sliced node, which makes torchrun spawn too many ranks.
NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
[[ "$NG" -ge 1 ]] || { echo "FATAL: torch sees 0 GPUs"; nvidia-smi -L | head -3; exit 1; }
# Per-device batch is the main throughput lever: the previous run used batch 1 and
# left GPUs under 50% utilised with only 30/80 GiB of memory in use. If a node OOMs,
# submit with BS=2 (or 1) and the effective batch stays 32.
BS="${BS:-4}"
GA=$(( 32 / (NG * BS) )); [[ "$GA" -lt 1 ]] && GA=1   # effective batch stays 32 at per-device 4
# Calibrated against the 8xL20 measurement (10.89 s/step at effective batch 32).
# An A800 is roughly 2.5x an L20 per card, so expect ~5-6 s/step here.
# 8 GPUs -> 16000 steps, i.e. 512,000 samples (1.5 epoch) in ~24h.
STEPS=$(( NG * 2000 ))

# The per-form files are concatenated in order, so the unshuffled corpus is locally
# clustered (each tenth holds only ~4 of 8 forms). Prefer the shuffled copy.
TRAIN="$W/trains/gsa_bench/train_shuf.jsonl"
[[ -s "$TRAIN" ]] || TRAIN="$W/trains/gsa_bench/train.jsonl"
VAL="$W/trains/gsa_bench/val_small.jsonl"
[[ -s "$VAL" ]] || VAL="$W/trains/gsa_bench/val.jsonl"

RESUME=()
LAST=$(ls -d "$OUT"/checkpoint-* "$OUT"/v*/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[[ -n "$LAST" ]] && RESUME=(--resume_from_checkpoint "$LAST")

LOG="$W/logs/sft_9b_lora_dlc_$(date -u +%Y%m%d_%H%M%S).log"
echo "=================================================="
echo " run   : sft_9b_lora_dlc"
echo " GPUs  : $NG   grad_accum=$GA   eff_batch=$(( NG * GA ))"
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
  --tuner_type lora --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 \
  --torch_dtype bfloat16 --deepspeed zero2 \
  --max_steps "$STEPS" \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
  --learning_rate 1e-4 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_length 3072 --gradient_checkpointing true \
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
