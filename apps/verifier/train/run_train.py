"""Verifier 训练脚本 — 基于 ms-swift 的三阶段训练 (SFT → DPO → GRPO)。

用法:
    # Stage 1: SFT
    python -m apps.verifier.train.run_train --stage sft --data data/verifier/train_data/sft_train.jsonl

    # Stage 2: DPO
    python -m apps.verifier.train.run_train --stage dpo --data data/verifier/train_data/dpo_train.jsonl

    # Stage 3: GRPO
    python -m apps.verifier.train.run_train --stage grpo --data data/verifier/train_data/grpo_train.jsonl

环境: conda activate roboevolve_vlm_train (或 roboevolve_wan_dpo/grpo)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [verifier.train] %(message)s")
logger = logging.getLogger("verifier.train")

# ─── 配置 ────────────────────────────────────────────────────────────────────
BASE_MODEL = "/path/to/environment/hf_models/Qwen/Qwen3-32B"
OUTPUT_ROOT = "data/verifier/checkpoints"

TRAIN_CONFIGS = {
    "sft": {
        "env": "roboevolve_vlm_train",
        "script": "swift sft",
        "args": {
            "--model": BASE_MODEL,
            "--train_type": "lora",
            "--lora_rank": "16",
            "--lora_alpha": "32",
            "--dataset": "",  # filled at runtime
            "--output_dir": "",  # filled at runtime
            "--num_train_epochs": "3",
            "--per_device_train_batch_size": "2",
            "--gradient_accumulation_steps": "8",
            "--learning_rate": "1e-4",
            "--max_length": "4096",
            "--logging_steps": "10",
            "--save_steps": "100",
            "--save_total_limit": "3",
            "--bf16": "true",
            "--gradient_checkpointing": "true",
        },
    },
    "dpo": {
        "env": "roboevolve_wan_dpo",
        "script": "swift rlhf",
        "args": {
            "--model": BASE_MODEL,
            "--rlhf_type": "dpo",
            "--train_type": "lora",
            "--lora_rank": "16",
            "--lora_alpha": "32",
            "--dataset": "",
            "--output_dir": "",
            "--num_train_epochs": "2",
            "--per_device_train_batch_size": "1",
            "--gradient_accumulation_steps": "16",
            "--learning_rate": "5e-5",
            "--max_length": "4096",
            "--logging_steps": "10",
            "--save_steps": "50",
            "--save_total_limit": "2",
            "--bf16": "true",
            "--gradient_checkpointing": "true",
            "--beta": "0.1",
        },
    },
    "grpo": {
        "env": "roboevolve_wan_grpo",
        "script": "swift rlhf",
        "args": {
            "--model": BASE_MODEL,
            "--rlhf_type": "grpo",
            "--train_type": "lora",
            "--lora_rank": "16",
            "--lora_alpha": "32",
            "--dataset": "",
            "--output_dir": "",
            "--num_train_epochs": "1",
            "--per_device_train_batch_size": "1",
            "--gradient_accumulation_steps": "16",
            "--learning_rate": "1e-5",
            "--max_length": "4096",
            "--logging_steps": "5",
            "--save_steps": "50",
            "--save_total_limit": "2",
            "--bf16": "true",
            "--gradient_checkpointing": "true",
            "--num_generations": "4",
        },
    },
}


def generate_train_command(stage: str, data_path: str, output_dir: str) -> List[str]:
    """生成训练命令。"""
    config = TRAIN_CONFIGS[stage]
    args = dict(config["args"])
    args["--dataset"] = data_path
    args["--output_dir"] = output_dir

    # 如果有 SFT checkpoint，DPO/GRPO 从 SFT 继续
    if stage in ("dpo", "grpo"):
        sft_ckpt = os.path.join(OUTPUT_ROOT, "sft", "best")
        if os.path.exists(sft_ckpt):
            args["--model"] = sft_ckpt
            logger.info(f"Using SFT checkpoint: {sft_ckpt}")

    cmd_parts = ["conda", "run", "--no-capture-output", "-n", config["env"]]
    cmd_parts += config["script"].split()
    for k, v in args.items():
        if v:
            cmd_parts.extend([k, v])

    return cmd_parts


def run_training(stage: str, data_path: str):
    """执行训练。"""
    output_dir = os.path.join(OUTPUT_ROOT, stage)
    os.makedirs(output_dir, exist_ok=True)

    cmd = generate_train_command(stage, data_path, output_dir)
    logger.info(f"Stage: {stage}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Command: {' '.join(cmd)}")

    # 保存命令到文件
    cmd_file = os.path.join(output_dir, "train_cmd.sh")
    with open(cmd_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd /path/to/environment/Rex-Embodied\n")
        f.write(f"export PYTHONPATH=.\n")
        f.write(" ".join(cmd) + "\n")
    os.chmod(cmd_file, 0o755)
    logger.info(f"Command saved to: {cmd_file}")

    # 执行
    proc = subprocess.run(cmd, cwd="/path/to/environment/Rex-Embodied")
    if proc.returncode != 0:
        logger.error(f"Training failed with code {proc.returncode}")
        sys.exit(1)

    logger.info(f"Training complete: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Verifier Training")
    parser.add_argument("--stage", choices=["sft", "dpo", "grpo", "all"], required=True)
    parser.add_argument("--data", required=True, help="Path to training data jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Only print command, don't execute")
    args = parser.parse_args()

    if args.stage == "all":
        stages = ["sft", "dpo", "grpo"]
    else:
        stages = [args.stage]

    for stage in stages:
        data_path = args.data
        # For 'all' mode, auto-detect data files
        if args.stage == "all":
            data_dir = os.path.dirname(args.data)
            data_path = os.path.join(data_dir, f"{stage}_train.jsonl")
            if not os.path.exists(data_path):
                logger.warning(f"Skipping {stage}: {data_path} not found")
                continue

        if args.dry_run:
            output_dir = os.path.join(OUTPUT_ROOT, stage)
            cmd = generate_train_command(stage, data_path, output_dir)
            print(f"\n[{stage}] Would run:")
            print(f"  {' '.join(cmd)}")
        else:
            run_training(stage, data_path)


if __name__ == "__main__":
    main()
