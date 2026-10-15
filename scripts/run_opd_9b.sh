#!/bin/bash
# Stage 3: On-Policy Distillation (GKD) - Distill 27B Teacher to 9B Student
# Local: 4x A800-80GB | Framework: ms-swift GKD | teacher generates in-process (no vLLM on driver 535)
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env

# --- Distillation backend policy (local 4x A800, driver 535) ---
# vLLM teacher rollout is unavailable on this driver (needs CUDA 13 / driver 575+ for qwen3_5).
# We run GKD with the teacher generating via the in-process transformers backend (--use_vllm false).
# This is slower per step but keeps the full teacher->student distillation on 4x A800.
OPD_USE_VLLM="${OPD_USE_VLLM:-false}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEACHER_CKPT="${OUTPUT_BASE}/grpo_27b_teacher"
OUTPUT_DIR="${OUTPUT_BASE}/opd_9b_student"
LOG_FILE="./logs/run_opd_9b_${TIMESTAMP}.log"

# Determine teacher model path
if [ -f "${TEACHER_CKPT}/config.json" ]; then
    TEACHER_MODEL="${TEACHER_CKPT}"
    echo "Using GRPO teacher: ${TEACHER_CKPT}" | tee -a ${LOG_FILE}
elif [ -f "${OUTPUT_BASE}/sft_27b_teacher/config.json" ]; then
    TEACHER_MODEL="${OUTPUT_BASE}/sft_27b_teacher"
    echo "Fallback to SFT teacher: ${TEACHER_MODEL}" | tee -a ${LOG_FILE}
else
    TEACHER_MODEL="${MODEL_27B}"
    echo "WARNING: No trained teacher, using base 27B" | tee -a ${LOG_FILE}
fi

echo "=== [$(date)] Stage 3: OPD/GKD Distillation ===" | tee -a ${LOG_FILE}
echo "Teacher: ${TEACHER_MODEL}" | tee -a ${LOG_FILE}
echo "Student: ${MODEL_9B}" | tee -a ${LOG_FILE}
echo "Output:  ${OUTPUT_DIR}" | tee -a ${LOG_FILE}

cd ${MS_SWIFT_ROOT}

# GKD distillation: teacher + student both in-process across all 4 GPUs (no vLLM server).
# seq_kd=true => teacher generates sequences (transformers backend) used as distillation targets.
VLLM_ARGS=(--use_vllm false)
if [ "${OPD_USE_VLLM}" = "true" ] && python -c "import vllm" 2>/dev/null; then
    VLLM_ARGS=(--use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port 8000)
fi

echo "[$(date)] Starting GKD distillation (use_vllm=${OPD_USE_VLLM}) ..." | tee -a ${LOG_FILE}
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model ${MODEL_9B} \
    --teacher_model ${TEACHER_MODEL} \
    --dataset ${GSA_FORGE_ROOT}/trains/gsa_sft_train.jsonl \
    --tuner_type full \
    --seq_kd true \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --deepspeed zero3 \
    --teacher_deepspeed zero3 \
    --attn_impl ${ATTN_IMPL:-flash_attn} \
    "${VLLM_ARGS[@]}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --max_length 4096 \
    --max_completion_length 2048 \
    --gradient_checkpointing true \
    --logging_steps 5 \
    --save_steps 200 \
    --save_total_limit 3 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --output_dir ${OUTPUT_DIR} \
    --save_only_model true \
    --report_to wandb \
    2>&1 | tee -a ${LOG_FILE}

echo "=== [$(date)] Stage 3 Complete ===" | tee -a ${LOG_FILE}
echo "Student model: ${OUTPUT_DIR}" | tee -a ${LOG_FILE}
