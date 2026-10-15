#!/bin/bash
# Stage 1: Qwen3.6-27B Teacher SFT Training
# Local: 4x A800-80GB | Framework: ms-swift 4.2 | DeepSpeed: ZeRO-3
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/sft_27b_teacher"
LOG_FILE="./logs/run_sft_27b_${TIMESTAMP}.log"
mkdir -p logs output

echo "=== [$(date)] Stage 1: 27B Teacher SFT ===" | tee -a ${LOG_FILE}
echo "Model:   ${MODEL_27B}" | tee -a ${LOG_FILE}
echo "Data:    ${GSA_FORGE_ROOT}/trains/gsa_sft_train.jsonl" | tee -a ${LOG_FILE}
echo "Output:  ${OUTPUT_DIR}" | tee -a ${LOG_FILE}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7} \
NPROC_PER_NODE=${NPROC_PER_NODE:-6} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift sft \
    --model ${MODEL_27B} \
    --dataset ${GSA_FORGE_ROOT}/trains/gsa_sft_train.jsonl \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --deepspeed zero3 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_length 2048 \
    --gradient_checkpointing true \
    --logging_steps 5 \
    --save_steps 200 \
    --save_total_limit 3 \
    --output_dir ${OUTPUT_DIR} \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --save_only_model true \
    2>&1 | tee -a ${LOG_FILE}

echo "=== [$(date)] Stage 1 Complete ===" | tee -a ${LOG_FILE}
