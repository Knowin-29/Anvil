#!/bin/bash
# ?????????: ? GPU 4-7 ?? swift infer ???
# ??: bash eval/run_multi_eval.sh [subset_size]
# subset_size=0 ????, ??100(smoke test)

set -e
cd ${GSA_FORGE_ROOT}

SUBSET=${1:-100}
EVAL_INPUT="${GSA_EVAL_ROOT}/v0616/vlabel_input_v0616.jsonl"
EVAL_GT="${GSA_EVAL_ROOT}/v0616/vlabel_gt_v0616.jsonl"
OUT_DIR="${GSA_EVAL_ROOT}/v0616/outputs"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export IMAGE_MAX_TOKEN_NUM=1024

# ?? subset > 0, ????
if [ "$SUBSET" -gt 0 ]; then
    INPUT_FILE="$OUT_DIR/input_subset_${SUBSET}.jsonl"
    GT_FILE="$OUT_DIR/gt_subset_${SUBSET}.jsonl"
    head -n $SUBSET "$EVAL_INPUT" > "$INPUT_FILE"
    head -n $SUBSET "$EVAL_GT" > "$GT_FILE"
    echo "[INFO] Using subset of $SUBSET samples"
else
    INPUT_FILE="$EVAL_INPUT"
    GT_FILE="$EVAL_GT"
    echo "[INFO] Using FULL dataset (100k samples)"
fi

# ????
declare -A MODELS
MODELS[base]="${MODEL_ROOT}/Qwen/Qwen3.6-27B"
MODELS[v7_3000]="${GSA_FORGE_ROOT}/output/sft_27b_v7_fresh/v0-20260612-145559/checkpoint-3000"
MODELS[v7_5000]="${GSA_FORGE_ROOT}/output/ckpt_27b_v7_5000"
MODELS[v5_7000]="${GSA_FORGE_ROOT}/output/sft_27b_teacher_v5_resume/v1-20260611-001757/checkpoint-7000"

echo "========================================="
echo " Multi-Model Evaluation (subset=$SUBSET)"
echo "========================================="
echo ""

for TAG in base v7_3000 v7_5000 v5_7000; do
    MODEL_PATH="${MODELS[$TAG]}"
    OUT_FILE="$OUT_DIR/out_${TAG}_n${SUBSET}.jsonl"

    if [ -f "$OUT_FILE" ] && [ $(wc -l < "$OUT_FILE") -ge $SUBSET ]; then
        echo "[$TAG] Output exists ($OUT_FILE), skipping inference."
        continue
    fi

    echo "[$TAG] Running inference: $MODEL_PATH"
    echo "  -> $OUT_FILE"

    swift infer \
        --model "$MODEL_PATH" \
        --val_dataset "$INPUT_FILE" \
        --max_batch_size 1 \
        --max_length 4096 \
        --truncation_strategy left \
        --result_path "$OUT_FILE" \
        --max_new_tokens 1024 \
        2>&1 | tail -5

    echo "[$TAG] Done. Lines: $(wc -l < "$OUT_FILE")"
    echo ""
done

echo "========================================="
echo " All inferences complete. Running scoring..."
echo "========================================="

# ??
python3 eval/score_vlabel_v3.py "$OUT_DIR/out_base_n${SUBSET}.jsonl" "$GT_FILE" "base_n${SUBSET}"
python3 eval/score_vlabel_v3.py "$OUT_DIR/out_v7_3000_n${SUBSET}.jsonl" "$GT_FILE" "v7_3000_n${SUBSET}"
python3 eval/score_vlabel_v3.py "$OUT_DIR/out_v7_5000_n${SUBSET}.jsonl" "$GT_FILE" "v7_5000_n${SUBSET}"
python3 eval/score_vlabel_v3.py "$OUT_DIR/out_v5_7000_n${SUBSET}.jsonl" "$GT_FILE" "v5_7000_n${SUBSET}"

echo ""
echo "Done! Check eval/vlabel_v3_*.json for results."
