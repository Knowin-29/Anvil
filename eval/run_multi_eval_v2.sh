#!/bin/bash
# ??????? v2: ?? v0616 ??? + v4 scorer
# ??: bash eval/run_multi_eval_v2.sh [N_SAMPLES]
set -e
cd ${GSA_FORGE_ROOT}

N=${1:-1000}
EVAL_INPUT="${GSA_EVAL_ROOT}/v0616/vlabel_input_v0616.jsonl"
EVAL_GT="${GSA_EVAL_ROOT}/v0616/vlabel_gt_v0616.jsonl"
OUT_DIR="${GSA_EVAL_ROOT}/v0616/outputs"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Subset
if [ "$N" -gt 0 ] && [ "$N" -lt 100000 ]; then
    INPUT="$OUT_DIR/input_n${N}.jsonl"
    GT="$OUT_DIR/gt_n${N}.jsonl"
    head -n $N "$EVAL_INPUT" > "$INPUT"
    head -n $N "$EVAL_GT" > "$GT"
else
    INPUT="$EVAL_INPUT"
    GT="$EVAL_GT"
    N=100000
fi

echo "============================================"
echo " GSA_Forge Multi-Model Eval (N=$N)"
echo " GPU: 4,5,6,7 | Scorer: v4"
echo "============================================"

declare -A MODELS
MODELS[base]="${MODEL_ROOT}/Qwen/Qwen3.6-27B"
MODELS[v7_3000]="${GSA_FORGE_ROOT}/output/sft_27b_v7_fresh/v0-20260612-145559/checkpoint-3000"
MODELS[v7_5000]="${GSA_FORGE_ROOT}/output/ckpt_27b_v7_5000"
MODELS[v5_7000]="${GSA_FORGE_ROOT}/output/sft_27b_teacher_v5_resume/v1-20260611-001757/checkpoint-7000"

for TAG in base v7_3000 v7_5000 v5_7000; do
    MODEL="${MODELS[$TAG]}"
    OUT="$OUT_DIR/out_${TAG}_n${N}.jsonl"

    if [ -f "$OUT" ] && [ $(wc -l < "$OUT") -ge $N ]; then
        echo "[$TAG] Already done ($(wc -l < "$OUT") lines). Skipping."
        continue
    fi

    echo ""
    echo "[$TAG] === Starting inference ==="
    echo "  Model: $MODEL"
    echo "  Output: $OUT"
    echo "  Time: $(date '+%H:%M:%S')"

    swift infer \
        --model "$MODEL" \
        --val_dataset "$INPUT" \
        --max_batch_size 1 \
        --max_length 4096 \
        --truncation_strategy left \
        --result_path "$OUT" \
        --max_new_tokens 1024 \
        2>&1 | grep -E "100%|Error|result|Train|infer" | tail -3

    LINES=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    echo "  [$TAG] Complete: $LINES lines at $(date '+%H:%M:%S')"
done

echo ""
echo "============================================"
echo " Scoring all models..."
echo "============================================"

for TAG in base v7_3000 v7_5000 v5_7000; do
    OUT="$OUT_DIR/out_${TAG}_n${N}.jsonl"
    if [ -f "$OUT" ]; then
        echo ""
        python3 eval/score_vlabel_v4.py "$OUT" "$GT" "${TAG}_n${N}"
    fi
done

echo ""
echo "============================================"
echo " All done! $(date)"
echo "============================================"
