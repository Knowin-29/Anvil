#!/bin/bash
cd /path/to/storage/anvil/GSA_Forge
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
swift infer \
    --model /path/to/storage/anvil/GSA_Forge/eval/ckpt_eval_1000 \
    --val_dataset /path/to/storage/anvil/GSA_Forge/eval/replay_100.jsonl \
    --infer_backend pt \
    --enable_thinking false \
    --max_new_tokens 96 \
    --max_batch_size 16 \
    --result_path /path/to/storage/anvil/GSA_Forge/eval/replay_100_out.jsonl
