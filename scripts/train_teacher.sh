#!/bin/bash
# GSA Teacher Training — 8×A100 Distributed SFT
# Usage: bash scripts/train_teacher.sh

set -e
cd /path/to/environment/Rex-Embodied

echo "=== GSA Teacher Training ==="
echo "Model: Qwen/Qwen2.5-VL-7B-Instruct"
echo "Data: data/output/gsa_v0/latest/training/gsa_sft_data.jsonl"
echo "Output: data/output/teacher/gsa_teacher_v0"
echo ""

# Prepare training data from latest GSA run
echo "Step 1: Preparing training data..."
PYTHONPATH=. python -c "
from apps.teacher_training.data_prep import prepare_training_data
from pathlib import Path
import glob

# Find latest GSA run
runs = sorted(glob.glob('data/output/gsa_v0/gsa_v0_*'))
if not runs:
    print('ERROR: No GSA runs found. Run scripts/run_gsa_production.py first.')
    exit(1)
latest = runs[-1]
print(f'Using latest run: {latest}')

n = prepare_training_data(
    labels_dir=f'{latest}/labels',
    output_path='data/output/gsa_v0/latest/training/gsa_sft_data.jsonl',
)
print(f'Prepared {n} training samples')
"

if [ ! -f "data/output/gsa_v0/latest/training/gsa_sft_data.jsonl" ]; then
    echo "ERROR: Training data not found at data/output/gsa_v0/latest/training/gsa_sft_data.jsonl"
    exit 1
fi

TRAIN_SIZE=$(wc -l < "data/output/gsa_v0/latest/training/gsa_sft_data.jsonl")
echo "Training samples: $TRAIN_SIZE"
echo ""

# Generate DeepSpeed config
echo "Step 2: Generating DeepSpeed config..."
PYTHONPATH=. python -c "
from apps.teacher_training.train import generate_deepspeed_config
generate_deepspeed_config('scripts/configs/ds_zero2.json')
print('DeepSpeed config saved.')
"

# Launch distributed training
echo "Step 3: Launching 8-GPU training..."
echo ""

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

torchrun --nproc_per_node=8 --master_port=29500 \
    -m apps.teacher_training.sft_trainer \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --data_path data/output/gsa_v0/latest/training/gsa_sft_data.jsonl \
    --output_dir data/output/teacher/gsa_teacher_v0 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-05 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_length 2048 \
    --bf16 true \
    --deepspeed scripts/configs/ds_zero2.json \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 3 \
    --gradient_checkpointing true \
    --report_to none

echo ""
echo "=== Training Complete ==="
echo "Model saved to: data/output/teacher/gsa_teacher_v0"
