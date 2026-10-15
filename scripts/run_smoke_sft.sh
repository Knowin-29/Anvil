#!/bin/bash
# SMOKE: minimal 9B SFT to verify the training pipeline end-to-end (few steps).
# Uses the fully-downloaded Qwen3.5-9B + local flywheel-derived jsonl.
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env
mkdir -p logs output

SMOKE_MODEL="${SMOKE_MODEL:-${MODEL_9B}}"
SMOKE_STEPS="${SMOKE_STEPS:-10}"
OUTPUT_DIR="${OUTPUT_BASE}/smoke_sft_9b"

echo "=== [$(date)] SMOKE SFT ==="
echo "Model:  ${SMOKE_MODEL}"
echo "Data:   ${GSA_FORGE_ROOT}/trains/gsa_sft_train.jsonl"
echo "Steps:  ${SMOKE_STEPS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
# Smoke uses LoRA (low memory; validates the pipeline on a shared box). Set
# SMOKE_TUNER=full to exercise full-parameter SFT when GPUs are free.
swift sft \
    --model "${SMOKE_MODEL}" \
    --dataset "${GSA_FORGE_ROOT}/trains/gsa_sft_train.jsonl" \
    --tuner_type "${SMOKE_TUNER:-lora}" \
    --torch_dtype bfloat16 \
    --max_steps "${SMOKE_STEPS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_length 2048 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 100000 \
    --save_total_limit 1 \
    --output_dir "${OUTPUT_DIR}" \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2 \
    --save_only_model true \
    --report_to none

echo "=== [$(date)] SMOKE SFT complete (if no error above, pipeline OK) ==="
