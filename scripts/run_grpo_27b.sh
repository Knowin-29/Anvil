#!/bin/bash
# Stage 2: Qwen3.6-27B Teacher GRPO Training
# Local: 4x A800-80GB, colocate vLLM rollout (train + infer share GPUs)
# Framework: ms-swift + vLLM (>=0.17.2 for qwen3_5). Override via env:
#   GRPO_USE_VLLM / GRPO_VLLM_MODE / GRPO_VLLM_GPU_UTIL / CUDA_VISIBLE_DEVICES / NPROC_PER_NODE
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env
mkdir -p logs

# --- vLLM policy (local 4x A800, driver 535) ---
# This box's driver (535, CUDA 12.9) cannot run a qwen3_5-capable vLLM (needs CUDA 13 / driver 575+).
# ms-swift 4.2 (gsa_train) lazily imports vllm, so GRPO runs WITHOUT vllm using the PT/transformers
# generation backend (--use_vllm false). Slower rollout but fully local. Set GRPO_USE_VLLM=true only
# on a CUDA13/driver575+ box.
GRPO_USE_VLLM="${GRPO_USE_VLLM:-false}"
if [ "${GRPO_USE_VLLM}" = "true" ] && ! python -c "import vllm" 2>/dev/null; then
    echo "GRPO_USE_VLLM=true but vllm not importable; falling back to PT backend." 
    GRPO_USE_VLLM="false"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SFT_CKPT="${OUTPUT_BASE}/sft_27b_teacher"
OUTPUT_DIR="${OUTPUT_BASE}/grpo_27b_teacher"
LOG_FILE="./logs/run_grpo_27b_${TIMESTAMP}.log"

# Use SFT checkpoint if available, fallback to base model
SFT_CKPT_FULL=$(find ${OUTPUT_BASE}/sft_27b_teacher -name "config.json" -path "*/checkpoint-*" | sort | tail -1 | xargs dirname 2>/dev/null)
if [ -n "${SFT_CKPT_FULL}" ] && [ -f "${SFT_CKPT_FULL}/config.json" ]; then
    TRAIN_MODEL="${SFT_CKPT_FULL}"
    echo "Using SFT checkpoint: ${TRAIN_MODEL}" 
else
    TRAIN_MODEL="${MODEL_27B}"
    echo "WARNING: SFT checkpoint not found, using base model" 
fi

echo "=== [$(date)] Stage 2: 27B Teacher GRPO ===" 
echo "Model:   ${TRAIN_MODEL}" 
echo "Data:    ${GSA_FORGE_ROOT}/trains/gsa_grpo_train.jsonl" 
echo "Output:  ${OUTPUT_DIR}" 

cd ${MS_SWIFT_ROOT}

# Build backend-specific args: vLLM (colocate) only if explicitly enabled + available.
VLLM_ARGS=(--use_vllm false)
if [ "${GRPO_USE_VLLM}" = "true" ]; then
    VLLM_ARGS=(--use_vllm true --vllm_mode "${GRPO_VLLM_MODE:-colocate}" \
               --vllm_gpu_memory_utilization "${GRPO_VLLM_GPU_UTIL:-0.4}")
fi

echo "[$(date)] Starting GRPO training (use_vllm=${GRPO_USE_VLLM})..." 

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
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 2 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.7 \
    --top_p 0.9 \
    --num_iterations 1 \
    --beta 0.01 \
    --log_completions true \
    --output_dir ${OUTPUT_DIR} \
    --report_to wandb \
    --save_only_model true \
    2>&1 

echo "=== [$(date)] Stage 2 Complete ===" 
