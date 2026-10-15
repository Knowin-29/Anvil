#!/bin/bash
# Replay correct-balanced test set on a checkpoint
# Usage: bash run_vlabel_replay.sh <ckpt_path> <tag>
CKPT=$1
TAG=$2
cd ${GSA_FORGE_ROOT}
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
swift infer \
    --model "$CKPT" \
    --val_dataset ${GSA_FORGE_ROOT}/eval/vlabel_input.jsonl \
    --infer_backend pt \
    --enable_thinking false \
    --max_new_tokens 256 \
    --max_batch_size 8 \
    --result_path ${GSA_FORGE_ROOT}/eval/vlabel_out_${TAG}.jsonl
