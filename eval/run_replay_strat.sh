#!/bin/bash
# Stratified replay eval (bbox IoU). Runnable locally; EVAL_MODEL overridable.
# vLLM can't load qwen3_5 on driver 535 -> use PT infer backend.
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source .env 2>/dev/null || true

EVAL_MODEL="${EVAL_MODEL:-${MODEL_9B}}"
EVAL_GPUS="${EVAL_GPUS:-0}"
VAL="${EVAL_VAL:-eval/replay_strat.jsonl}"
OUT="${EVAL_OUT:-eval/replay_strat_out.jsonl}"

echo "=== [$(date)] Replay eval ==="
echo "Model: ${EVAL_MODEL}"
echo "Val:   ${VAL}"

CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift infer \
    --model "${EVAL_MODEL}" \
    --val_dataset "${VAL}" \
    --infer_backend pt \
    --max_new_tokens 96 \
    --max_batch_size 8 \
    --result_path "${OUT}"

echo "=== [$(date)] Eval done -> ${OUT} ==="
echo "Run: python eval/eval_iou_report.py --pred ${OUT} --gt eval/replay_strat_gt.jsonl"
