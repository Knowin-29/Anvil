"""Teacher Training — 8-GPU distributed SFT with DeepSpeed.

Trains a VLM teacher model on GSA data using:
- DeepSpeed ZeRO-2 for memory efficiency
- 8×A100-80GB parallel training
- Conversation-format GSA data
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("teacher.train")


@dataclass
class TrainConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    data_path: str = "data/output/gsa_v0/latest/training/gsa_sft_data.jsonl"
    output_dir: str = "data/output/teacher/gsa_teacher_v0"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_length: int = 2048
    bf16: bool = True
    deepspeed: str = "scripts/configs/ds_zero2.json"
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    gradient_checkpointing: bool = True
    report_to: str = "none"


def generate_deepspeed_config(output_path: str = "scripts/configs/ds_zero2.json"):
    """Generate DeepSpeed ZeRO-2 config for 8×A100."""
    config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": 4,
        "gradient_clipping": 1.0,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return output_path


def generate_train_script(config: TrainConfig, output_path: str = "scripts/train_teacher.sh"):
    """Generate the training launch script."""
    script = f"""#!/bin/bash
# GSA Teacher Training — 8×A100 Distributed SFT
# Usage: bash scripts/train_teacher.sh

set -e
cd /path/to/environment/Rex-Embodied

echo "=== GSA Teacher Training ==="
echo "Model: {config.model_name_or_path}"
echo "Data: {config.data_path}"
echo "Output: {config.output_dir}"
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
print(f'Using latest run: {{latest}}')

n = prepare_training_data(
    labels_dir=f'{{latest}}/labels',
    output_path='{config.data_path}',
)
print(f'Prepared {{n}} training samples')
"

if [ ! -f "{config.data_path}" ]; then
    echo "ERROR: Training data not found at {config.data_path}"
    exit 1
fi

TRAIN_SIZE=$(wc -l < "{config.data_path}")
echo "Training samples: $TRAIN_SIZE"
echo ""

# Generate DeepSpeed config
echo "Step 2: Generating DeepSpeed config..."
PYTHONPATH=. python -c "
from apps.teacher_training.train import generate_deepspeed_config
generate_deepspeed_config('{config.deepspeed}')
print('DeepSpeed config saved.')
"

# Launch distributed training
echo "Step 3: Launching 8-GPU training..."
echo ""

export PYTHONPATH="$(pwd):$PYTHONPATH"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

torchrun --nproc_per_node=8 --master_port=29500 \\
    -m apps.teacher_training.sft_trainer \\
    --model_name_or_path {config.model_name_or_path} \\
    --data_path {config.data_path} \\
    --output_dir {config.output_dir} \\
    --num_train_epochs {config.num_train_epochs} \\
    --per_device_train_batch_size {config.per_device_train_batch_size} \\
    --gradient_accumulation_steps {config.gradient_accumulation_steps} \\
    --learning_rate {config.learning_rate} \\
    --warmup_ratio {config.warmup_ratio} \\
    --lr_scheduler_type {config.lr_scheduler_type} \\
    --max_length {config.max_length} \\
    --bf16 {str(config.bf16).lower()} \\
    --deepspeed {config.deepspeed} \\
    --logging_steps {config.logging_steps} \\
    --save_steps {config.save_steps} \\
    --save_total_limit {config.save_total_limit} \\
    --gradient_checkpointing {str(config.gradient_checkpointing).lower()} \\
    --report_to {config.report_to}

echo ""
echo "=== Training Complete ==="
echo "Model saved to: {config.output_dir}"
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(script)
    os.chmod(output_path, 0o755)
    return output_path
