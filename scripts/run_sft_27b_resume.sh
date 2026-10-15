#!/bin/bash
# Resume from a prior SFT checkpoint as base model (no optimizer resume, fresh optimizer)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Resume from the latest local SFT checkpoint (v3 completed run) unless overridden
RESUME_CKPT="${RESUME_CKPT:-${TEACHER_SFT_CKPT:-${OUTPUT_BASE}/sft_27b_teacher/v3-20260607-135945/checkpoint-1740}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" swift sft \
    --model "${RESUME_CKPT}" \
    --dataset "${GSA_FORGE_ROOT}/trains/gsa_sft_train.jsonl" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --deepspeed zero3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --max_length 2048 \
    --gradient_checkpointing true \
    --logging_steps 5 \
    --save_steps 500 \
    --save_total_limit 5 \
    --output_dir "${OUTPUT_BASE}/sft_27b_teacher_resume" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --save_only_model true
