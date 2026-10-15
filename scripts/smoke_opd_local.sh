#!/usr/bin/env bash
# smoke_opd_local.sh — prove the OPD/GKD path actually ENTERS the training loop
# on the local 2xA800 box, where both cards are shared with external processes.
#
# Deliberately exercises the SEPARATE-teacher branch of gkd_trainer.compute_loss
# (the `else` at "assert self.teacher_model is not None"), because that is the
# branch the DLC runs will take. Self-distillation via disable_adapter() is a
# different branch and would not prove anything about a real teacher.
#
# Teacher: output/merged_e1 (E1 LoRA already merged into a standalone 9B).
#   merged rather than --teacher_adapters on purpose: `teacher_adapters` is
#   declared in RLHFArguments but read by NO code path (verified), so passing it
#   silently trains against the BASE model. See docs/DLC_NEXT.md.
# Student: base Qwen3.5-9B + LoRA.
#
#   bash scripts/smoke_opd_local.sh
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
mkdir -p "$W"/.cache/{hf,torch,triton,tmp,mpl} "$W/logs"

# One card only: the box has 2 shared A800s and a 2-rank attempt has already been
# killed by external processes (OOM). MASTER_PORT is moved off the default 29500.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29733}"

STUDENT=/path/to/environment/hf_models/Qwen/Qwen3.5-9B
TEACHER="${TEACHER:-$W/output/merged_e1}"
[[ -f "$TEACHER/config.json" ]] || { echo "FATAL: no teacher at $TEACHER"; exit 1; }

STEPS="${STEPS:-2}"
OUT="$W/output/_smoke_opd_local"
LOG="$W/logs/smoke_opd_local_$(date -u +%Y%m%d_%H%M%S).log"

NG=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
[[ "$NG" -ge 1 ]] || { echo "FATAL: torch sees 0 GPUs"; exit 1; }

echo "=================================================="
echo " smoke : opd/gkd separate-teacher path"
echo " GPUs  : $NG (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)  port=$MASTER_PORT"
echo " teach : $TEACHER"
echo " steps : $STEPS"
echo " log   : $LOG"
echo "=================================================="

# lmbda 1.0 forces student self-sampling on EVERY step, so the on-policy
# generate path is exercised rather than skipped by the RNG draw.
NPROC_PER_NODE="$NG" timeout 28m swift rlhf --rlhf_type gkd \
  --model "$STUDENT" \
  --teacher_model "$TEACHER" \
  --lmbda 1.0 --beta 0.5 \
  --dataset "$W/trains/gsa_bench/val_small.jsonl" \
  --split_dataset_ratio 0 \
  --tuner_type lora --lora_rank 8 --lora_alpha 16 \
  --torch_dtype bfloat16 \
  --max_steps "$STEPS" \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --max_length 1024 --max_completion_length 32 \
  --gradient_checkpointing true \
  --logging_steps 1 --save_steps 100000 --save_only_model true \
  --output_dir "$OUT" --dataloader_num_workers 1 --dataset_num_proc 1 \
  --report_to none \
  2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
echo "=================================================="
echo " rc=$RC"
# The point of the smoke is "did a real optimizer step happen", so assert on the
# loss line rather than on rc alone: a 2-second clean exit also returns 0.
if grep -qE "'loss':" "$LOG"; then
  echo " PASS: reached the training loop"
  grep -oE "\{'loss':[^}]*\}" "$LOG" | tail -3
else
  echo " FAIL: never produced a loss line"
  grep -iE 'error|traceback|out of memory|assert' "$LOG" | tail -8
fi
exit "$RC"
