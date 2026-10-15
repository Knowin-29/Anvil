#!/bin/bash
# Baseline: Qwen3.5-9B GRPO (after SFT baseline)
# Architecture: 1 GPU vLLM rollout + remaining GPUs training (local 4x A800)
# Framework: ms-swift
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env

# --- vLLM policy (local 4x A800, driver 535): PT backend, no vLLM ---
BASE_USE_VLLM="${BASE_USE_VLLM:-false}"
if [ "${BASE_USE_VLLM}" = "true" ] && ! python -c "import vllm" 2>/dev/null; then
    echo "BASE_USE_VLLM=true but vllm not importable; falling back to PT backend."
    BASE_USE_VLLM="false"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SFT_CKPT="${OUTPUT_BASE}/sft_9b_baseline"
OUTPUT_DIR="${OUTPUT_BASE}/grpo_9b_baseline"
LOG_FILE="./logs/run_grpo_9b_baseline_${TIMESTAMP}.log"

if [ -f "${SFT_CKPT}/config.json" ]; then
    TRAIN_MODEL="${SFT_CKPT}"
else
    TRAIN_MODEL="${MODEL_9B}"
    echo "WARNING: 9B SFT checkpoint not found, using base model" | tee -a ${LOG_FILE}
fi

echo "=== [$(date)] Baseline: 9B GRPO ===" | tee -a ${LOG_FILE}
echo "Model:   ${TRAIN_MODEL}" | tee -a ${LOG_FILE}
echo "Output:  ${OUTPUT_DIR}" | tee -a ${LOG_FILE}

cd ${MS_SWIFT_ROOT}

# PT-backend GRPO across all 4 GPUs (no vLLM rollout server on driver 535).
VLLM_ARGS=(--use_vllm false)
if [ "${BASE_USE_VLLM}" = "true" ]; then
    ROLLOUT_GPU="${BASE_ROLLOUT_GPU:-3}"
    CUDA_VISIBLE_DEVICES=${ROLLOUT_GPU} swift rollout --model ${TRAIN_MODEL} \
        --vllm_max_model_len 4096 --vllm_gpu_memory_utilization 0.9 &
    ROLLOUT_PID=$!
    sleep 20
    VLLM_ARGS=(--use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port 8000)
fi

# Training
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type grpo \
    --model ${TRAIN_MODEL} \
    --dataset ${GSA_FORGE_ROOT}/trains/gsa_grpo_train.jsonl \
    --external_plugins ${GSA_FORGE_ROOT}/scripts/gsa_reward_plugin.py \
    --reward_funcs gsa_reward format \
    "${VLLM_ARGS[@]}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --deepspeed zero3 \
    --attn_impl ${ATTN_IMPL:-flash_attn} \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 4 \
    --temperature 0.7 \
    --num_iterations 1 \
    --beta 0.01 \
    --log_completions true \
    --output_dir ${OUTPUT_DIR} \
    --report_to wandb \
    --save_only_model true \
    2>&1 | tee -a ${LOG_FILE}

[ -n "${ROLLOUT_PID:-}" ] && kill ${ROLLOUT_PID} 2>/dev/null || true
echo "=== [$(date)] Baseline 9B GRPO Complete ===" | tee -a ${LOG_FILE}
