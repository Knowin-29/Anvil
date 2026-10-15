#!/bin/bash
# Train all 6 per-dimension teachers sequentially (Qwen3.5-9B)
# Each teacher specializes in one GSA dimension
# Environment: qwen36
#
# After SFT teachers are trained, run GRPO on each to further improve.

export PATH="/path/to/environment/env/qwen36/bin:$PATH"

NNODES=1
GPUS_PER_NODE=8
export MASTER_ADDR=localhost

BASE_MODEL="/path/to/environment/hf_models/Qwen/Qwen3.5-9B"
TSV_BASE="/path/to/datasets/dataflywheel/output/rex_omni_tsv"
CKPT_BASE="/path/to/datasets/dataflywheel/output/checkpoints/teachers"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd /path/to/environment/Rex-Omni/finetuning

DIMENSIONS=("G_grounding" "S_spatial" "A_identity_attr" "A_obj_state" "A_action_precondition" "A_action_interface")

for DIM in "${DIMENSIONS[@]}"; do
    echo "============================================="
    echo "Training teacher: ${DIM} (9B)"
    echo "============================================="

    OUTPUT_DIR="${CKPT_BASE}/teacher_${DIM}_9b"
    mkdir -p ${OUTPUT_DIR}

    export MASTER_PORT=$((29500 + RANDOM % 100))

    torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        ${SCRIPT_DIR}/train_adaptive.py \
        --config ${SCRIPT_DIR}/configs/sft_9b.py \
        --model_name_or_path ${BASE_MODEL} \
        --deepspeed scripts/zero2.json \
        --data_flatten False \
        --tune_mm_vision True \
        --tune_mm_mlp True \
        --tune_mm_llm True \
        --bf16 \
        --output_dir ${OUTPUT_DIR} \
        --num_train_epochs 2 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --mm_projector_lr 2e-5 \
        --vision_tower_lr 2e-6 \
        --optim adamw_torch \
        --warmup_ratio 0.05 \
        --weight_decay 0.01 \
        --max_grad_norm 1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 5 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --run_name "teacher_${DIM}_9b" \
        --report_to none

    echo "Teacher ${DIM} done. Checkpoint: ${OUTPUT_DIR}"
    echo ""
done

echo "All teachers trained!"
echo "Next: Run GRPO on each teacher, then OPD distillation."
