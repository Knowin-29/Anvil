#!/bin/bash
# SMOKE: minimal 9B GRPO (PT backend, no vLLM) to verify reward plugin + RLHF path.
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env
mkdir -p logs output

SMOKE_MODEL="${SMOKE_MODEL:-${MODEL_9B}}"
SMOKE_STEPS="${SMOKE_STEPS:-6}"
OUTPUT_DIR="${OUTPUT_BASE}/smoke_grpo_9b"

echo "=== [$(date)] SMOKE GRPO (PT backend) ==="
echo "Model: ${SMOKE_MODEL}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type grpo \
    --model "${SMOKE_MODEL}" \
    --dataset "${GSA_FORGE_ROOT}/trains/gsa_grpo_train.jsonl" \
    --external_plugins "${GSA_FORGE_ROOT}/scripts/gsa_reward_plugin.py" \
    --reward_funcs gsa_reward format \
    --use_vllm false \
    --tuner_type "${SMOKE_TUNER:-lora}" \
    --torch_dtype bfloat16 \
    --max_steps "${SMOKE_STEPS}" \
    --max_completion_length 256 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-7 \
    --logging_steps 1 \
    --save_steps 100000 \
    --num_generations 2 \
    --temperature 0.7 \
    --beta 0.01 \
    --log_completions true \
    --output_dir "${OUTPUT_DIR}" \
    --report_to none

echo "=== [$(date)] SMOKE GRPO complete ==="
