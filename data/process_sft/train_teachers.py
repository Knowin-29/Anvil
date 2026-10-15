"""
Per-Dimension Teacher Training Pipeline

Strategy:
  - Generator: Qwen3.5-397B-A17B API (already running, generates high-quality data)
  - Teacher Training: SFT on dimension-specific verified data, then GRPO with verifier reward
  - Result: 6 specialized teachers (one per GSA dimension)

Teacher sizes:
  - Large teacher: Qwen3.6-27B (primary)
  - Small teacher: Qwen3.5-9B (for faster iteration / ablation)

Each teacher learns to:
  1. Understand the specific dimension's annotation format
  2. Judge quality of annotations (verifier role)
  3. Generate high-quality annotations (generator role)

Usage:
    # Train all 6 teachers (27B)
    python train_teachers.py --base_model 27b --dimensions all

    # Train single dimension teacher
    python train_teachers.py --base_model 9b --dimensions G_grounding
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DIMENSIONS = [
    "G_grounding",
    "S_spatial",
    "A_identity_attr",
    "A_obj_state",
    "A_action_precondition",
    "A_action_interface",
]

BASE_MODELS = {
    "9b": "/path/to/environment/hf_models/Qwen/Qwen3.5-9B",
    "27b": "/path/to/environment/hf_models/Qwen/Qwen3.6-27B",
}

VERIFIED_DIR = "/path/to/Anvil/data/output/gsa_flywheel_exp/verified"
OUTPUT_BASE = "/path/to/datasets/dataflywheel/output/checkpoints/teachers"
DATA_BASE = "/path/to/datasets/dataflywheel/output/teacher_data"
ENV_PYTHON = "/path/to/environment/env/qwen36/bin/python"


def prepare_dimension_data(dimension: str, output_dir: str):
    """Extract dimension-specific data and format for SFT training."""
    dim_dir = os.path.join(VERIFIED_DIR, dimension)
    if not os.path.isdir(dim_dir):
        print(f"  [SKIP] No data for {dimension}")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dimension}_sft.jsonl")

    count = 0
    with open(output_file, "w") as out_f:
        for jf in sorted(Path(dim_dir).glob("*.jsonl")):
            with open(jf) as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        out_f.write(line)
                        count += 1
                    except:
                        continue

    print(f"  [{dimension}] {count} samples → {output_file}")
    return count


def build_teacher_sft_config(dimension: str, base_model: str, data_dir: str, output_dir: str):
    """Generate a training config for one dimension's teacher."""
    config_content = f'''# Auto-generated teacher SFT config for {dimension}
# Base: {base_model}

from dataset import ConcatDataset, DataCollatorForSupervisedDataset, GroundingTSVDataset
from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import GROUNDING_SINGLE_REGION_STAGE_XYXY

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

model_name_or_path = "{base_model}"

TSV_BASE = "/path/to/datasets/dataflywheel/output/rex_omni_tsv"

# Use the full grounding data for G_grounding teacher
embodiedforge_grounding = dict(
    type=GroundingTSVDataset,
    img_tsv_file=f"{{TSV_BASE}}/embodiedforge_grounding.images.tsv",
    ann_tsv_file=f"{{TSV_BASE}}/embodiedforge_grounding.annotations.tsv",
    ann_lineidx_file=f"{{TSV_BASE}}/embodiedforge_grounding.annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="teacher_{dimension}",
)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[embodiedforge_grounding],
)

data_collator = dict(type=DataCollatorForSupervisedDataset)
'''
    config_path = os.path.join(output_dir, f"teacher_{dimension}.py")
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


def train_teacher(dimension: str, base_model_key: str, gpu_ids: str = "0,1,2,3,4,5,6,7"):
    """Launch SFT training for one dimension's teacher."""
    base_model = BASE_MODELS[base_model_key]
    teacher_output = os.path.join(OUTPUT_BASE, f"teacher_{dimension}_{base_model_key}")
    data_dir = os.path.join(DATA_BASE, dimension)

    os.makedirs(teacher_output, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Prepare data
    print(f"\n{'='*60}")
    print(f"Training Teacher: {dimension} ({base_model_key})")
    print(f"{'='*60}")

    count = prepare_dimension_data(dimension, data_dir)
    if count == 0:
        print(f"  [ABORT] No data for {dimension}")
        return

    print(f"  Base model: {base_model}")
    print(f"  Output: {teacher_output}")
    print(f"  Training samples: {count}")

    # Step 2: Training would be launched here
    print(f"\n  To train, run:")
    print(f"    CUDA_VISIBLE_DEVICES={gpu_ids} bash process_sft/run_sft_9b.sh")
    print(f"    (with MODEL_PATH={base_model} OUTPUT_DIR={teacher_output})")


def main():
    parser = argparse.ArgumentParser(description="Train per-dimension teachers")
    parser.add_argument("--base_model", choices=["9b", "27b"], default="9b")
    parser.add_argument("--dimensions", nargs="+", default=["all"])
    parser.add_argument("--prepare_only", action="store_true", help="Only prepare data, don't train")
    args = parser.parse_args()

    dims = DIMENSIONS if "all" in args.dimensions else args.dimensions

    print("=" * 60)
    print("EmbodiedForge Per-Dimension Teacher Training")
    print(f"Base model: {args.base_model} ({BASE_MODELS[args.base_model]})")
    print(f"Dimensions: {dims}")
    print("=" * 60)

    for dim in dims:
        if dim not in DIMENSIONS:
            print(f"[WARN] Unknown dimension: {dim}, skipping")
            continue

        data_dir = os.path.join(DATA_BASE, dim)
        prepare_dimension_data(dim, data_dir)

        if not args.prepare_only:
            train_teacher(dim, args.base_model)

    print("\n" + "=" * 60)
    print("Data preparation complete.")
    print(f"Teacher data at: {DATA_BASE}")
    print(f"Checkpoints will be at: {OUTPUT_BASE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
