#!/bin/bash
# Usage: bash run_vlabel_gpu.sh <ckpt> <tag> <gpus>
CKPT=$1; TAG=$2; GPUS=$3
cd ${GSA_FORGE_ROOT}
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
swift infer \
    --model "$CKPT" \
    --val_dataset ${GSA_FORGE_ROOT}/eval/vlabel_input.jsonl \
    --infer_backend pt \
    --enable_thinking false \
    --max_new_tokens 256 \
    --max_batch_size 8 \
    --result_path ${GSA_FORGE_ROOT}/eval/vlabel_out_${TAG}.jsonl
