#!/bin/bash
# Real 9B SFT on clean GSA data (LoRA by default; SFT_TUNER=full for full-param).
# Produces a post-trained checkpoint to compare against the base-9B baseline on the
# local grounding eval (the tab:main "before -> after" delta).
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env

TUNER="${SFT_TUNER:-lora}"
EPOCHS="${SFT_EPOCHS:-1}"
OUTPUT_DIR="${OUTPUT_BASE}/sft_9b_gsa_${TUNER}"
mkdir -p logs output

echo "=== [$(date)] 9B GSA SFT (${TUNER}) ==="
echo "Model:  ${MODEL_9B}"
echo "Data:   ${GSA_FORGE_ROOT}/trains/gsa_sft_train.jsonl"
echo "Output: ${OUTPUT_DIR}"

EXTRA=()
if [ "${TUNER}" = "lora" ]; then EXTRA=(--lora_rank 16 --lora_alpha 32 --learning_rate 1e-4); \
  else EXTRA=(--deepspeed zero3 --learning_rate 2e-5); fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-3}" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift sft \
    --model "${MODEL_9B}" \
    --dataset "${GSA_FORGE_ROOT}/trains/gsa_sft_train.jsonl" \
    --tuner_type "${TUNER}" \
    --torch_dtype bfloat16 \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --max_length 2048 --gradient_checkpointing true \
    --logging_steps 10 --save_steps 500 --save_total_limit 2 \
    --output_dir "${OUTPUT_DIR}" \
    --dataloader_num_workers 4 --dataset_num_proc 4 \
    --save_only_model true --report_to wandb \
    "${EXTRA[@]}" \
    2>&1

echo "=== [$(date)] 9B GSA SFT complete -> ${OUTPUT_DIR} ==="
