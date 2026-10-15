#!/bin/bash
# SMOKE: minimal GKD distillation (9B teacher -> 9B student, in-process, no vLLM)
# to verify the OPD/GKD path. Uses 9B as both to keep memory low for the smoke.
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env
mkdir -p logs output

SMOKE_TEACHER="${SMOKE_TEACHER:-${MODEL_9B}}"
SMOKE_STUDENT="${SMOKE_STUDENT:-${MODEL_9B}}"
SMOKE_STEPS="${SMOKE_STEPS:-6}"
OUTPUT_DIR="${OUTPUT_BASE}/smoke_opd_9b"

echo "=== [$(date)] SMOKE OPD/GKD (in-process teacher) ==="
echo "Teacher: ${SMOKE_TEACHER}"
echo "Student: ${SMOKE_STUDENT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model "${SMOKE_STUDENT}" \
    --teacher_model "${SMOKE_TEACHER}" \
    --dataset "${GSA_FORGE_ROOT}/trains/gsa_sft_train.jsonl" \
    --tuner_type "${SMOKE_TUNER:-lora}" \
    --seq_kd "${SMOKE_SEQ_KD:-false}" \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --deepspeed zero3 \
    --teacher_deepspeed zero3 \
    --use_vllm false \
    --max_steps "${SMOKE_STEPS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length 2048 \
    --max_completion_length 256 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 100000 \
    --output_dir "${OUTPUT_DIR}" \
    --save_only_model true \
    --report_to none

echo "=== [$(date)] SMOKE OPD complete ==="
