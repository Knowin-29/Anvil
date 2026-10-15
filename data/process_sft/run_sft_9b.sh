#!/bin/bash
# EmbodiedForge SFT Training (Standalone, no Rex-Omni engine dependency)
# Model: Qwen3.5-9B | Env: qwen36 | 8x A100-80G
set -e

source /path/to/datasets/dataflywheel/.env
export PATH="/path/to/environment/env/qwen36/bin:$PATH"
export WANDB_PROJECT="GSA_Forge_Train_volengine"

MODEL_PATH="/path/to/environment/hf_models/Qwen/Qwen3.5-9B"
TSV_BASE="/path/to/datasets/dataflywheel/output/rex_omni_tsv"
OUTPUT_DIR="/path/to/datasets/dataflywheel/output/checkpoints/sft_9b"
LOG_DIR="/path/to/datasets/dataflywheel/output/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

echo "[$(date)] Starting SFT training..."
echo "Model: ${MODEL_PATH}"
echo "Data: ${TSV_BASE}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_DIR}/sft_${TIMESTAMP}.log"

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=29501 \
    /path/to/datasets/dataflywheel/process_sft/train_standalone.py \
    --model_path ${MODEL_PATH} \
    --tsv_base ${TSV_BASE} \
    --output_dir ${OUTPUT_DIR} \
    --epochs 1 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 1e-5 \
    2>&1 | tee ${LOG_DIR}/sft_${TIMESTAMP}.log
