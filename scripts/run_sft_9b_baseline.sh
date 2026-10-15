#!/bin/bash
# Baseline: Qwen3.5-9B Direct SFT (for comparison with distilled model)
# Local: 4x A800-80GB | Framework: ms-swift | DeepSpeed: ZeRO-3
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/sft_9b_baseline"
LOG_FILE="./logs/run_sft_9b_baseline_${TIMESTAMP}.log"

echo "=== [$(date)] Baseline: 9B SFT ===" | tee -a ${LOG_FILE}
echo "Model:   ${MODEL_9B}" | tee -a ${LOG_FILE}
echo "Output:  ${OUTPUT_DIR}" | tee -a ${LOG_FILE}

cd ${MS_SWIFT_ROOT}

NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift sft \
    --model ${MODEL_9B} \
    --dataset ${GSA_FORGE_ROOT}/trains/gsa_sft_train.jsonl \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_length 4096 \
    --gradient_checkpointing true \
    --logging_steps 5 \
    --save_steps 200 \
    --save_total_limit 3 \
    --output_dir ${OUTPUT_DIR} \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --save_only_model true \
    --report_to wandb \
    2>&1 | tee -a ${LOG_FILE}

echo "=== [$(date)] Baseline 9B SFT Complete ===" | tee -a ${LOG_FILE}
