#!/bin/bash
# Rex-Embodied v3.0 Verifier 训练脚本 (ms-swift)
# 三段式: SFT → DPO → GRPO
#
# 用法: bash apps/data_flywheel/verifier/train_v0.sh [stage]
#   stage: sft / dpo / grpo / all (默认 all)
#
# 前置条件:
#   - conda activate rex
#   - pip install ms-swift vllm
#   - export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80

set -e

# ---- 配置 ----
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATA_DIR="verifier_data"
CKPT_DIR="checkpoints"
STAGE="${1:-all}"

# 从 .env 读取 WANDB_API_KEY
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep WANDB_API_KEY | xargs)
fi

echo "=== Rex-Embodied Verifier Training ==="
echo "Model: $MODEL"
echo "Data: $DATA_DIR"
echo "Stage: $STAGE"
echo ""

# ---- Stage 1: SFT ----
run_sft() {
    echo ">>> Stage 1: SFT"
    swift sft \
        --model "$MODEL" \
        --train_type lora \
        --lora_rank 64 \
        --lora_alpha 128 \
        --dataset "${DATA_DIR}/v0_sft.jsonl" \
        --num_train_epochs 2 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-4 \
        --warmup_ratio 0.05 \
        --max_length 2048 \
        --output_dir "${CKPT_DIR}/verifier_sft_v0" \
        --logging_steps 10 \
        --save_strategy epoch \
        --report_to wandb \
        --run_name "verifier_sft_v0"
    echo ">>> SFT complete: ${CKPT_DIR}/verifier_sft_v0"
}

# ---- Stage 2: DPO ----
run_dpo() {
    echo ">>> Stage 2: DPO"
    swift rlhf \
        --rlhf_type dpo \
        --model "${CKPT_DIR}/verifier_sft_v0" \
        --dataset "${DATA_DIR}/v0_dpo.jsonl" \
        --beta 0.1 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --max_length 2048 \
        --output_dir "${CKPT_DIR}/verifier_dpo_v0" \
        --logging_steps 10 \
        --save_strategy epoch \
        --report_to wandb \
        --run_name "verifier_dpo_v0"
    echo ">>> DPO complete: ${CKPT_DIR}/verifier_dpo_v0"
}

# ---- Stage 3: GRPO ----
run_grpo() {
    echo ">>> Stage 3: GRPO"
    swift rlhf \
        --rlhf_type grpo \
        --model "${CKPT_DIR}/verifier_dpo_v0" \
        --dataset "${DATA_DIR}/v0_sft.jsonl" \
        --reward_funcs format consistency downstream \
        --reward_funcs_path "apps/data_flywheel/verifier/rewards.py" \
        --num_generations 4 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-5 \
        --max_length 2048 \
        --output_dir "${CKPT_DIR}/verifier_grpo_v0" \
        --logging_steps 5 \
        --save_strategy epoch \
        --report_to wandb \
        --run_name "verifier_grpo_v0"
    echo ">>> GRPO complete: ${CKPT_DIR}/verifier_grpo_v0"
}

# ---- 执行 ----
case "$STAGE" in
    sft)  run_sft ;;
    dpo)  run_dpo ;;
    grpo) run_grpo ;;
    all)
        run_sft
        run_dpo
        run_grpo
        echo "=== All stages complete ==="
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 [sft|dpo|grpo|all]"
        exit 1
        ;;
esac
