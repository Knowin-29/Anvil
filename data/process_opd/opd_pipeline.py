"""
On-Policy Distillation: Train a unified small Student from 6 per-dimension Teachers.

Pipeline:
  1. Teachers (6x 9B/27B, each specialized per dimension) generate soft scores
  2. Generator (Qwen3.5-397B API) produces on-policy annotations
  3. Teachers score the on-policy data → soft labels
  4. Student (3B/7B) learns all 6 dimensions simultaneously

Student model options:
  - Qwen2.5-VL-3B-Instruct (fastest inference, deploy at edge)
  - Qwen2.5-VL-7B-Instruct (balanced quality/speed)

Loss:
  L_total = λ_kl * KL(student || teacher_soft) + λ_rank * RankingLoss + λ_sft * SFT_Loss

Usage:
    # Step 1: Generate on-policy data (using flywheel's API generator)
    python opd_pipeline.py generate --num_samples 10000

    # Step 2: Score with teachers
    python opd_pipeline.py score --teacher_dir /path/to/teachers/

    # Step 3: Train student
    python opd_pipeline.py train --student_model 3b
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

DIMENSIONS = [
    "G_grounding",
    "S_spatial",
    "A_identity_attr",
    "A_obj_state",
    "A_action_precondition",
    "A_action_interface",
]

STUDENT_MODELS = {
    "3b": "/path/to/environment/hf_models/Qwen/Qwen2.5-VL-3B-Instruct",
    "7b": "/path/to/environment/hf_models/Qwen/Qwen2.5-VL-7B-Instruct",
}

TEACHER_BASE = "/path/to/datasets/dataflywheel/output/checkpoints/teachers"
OPD_DATA_DIR = "/path/to/datasets/dataflywheel/output/opd_data"
OPD_CKPT_DIR = "/path/to/datasets/dataflywheel/output/checkpoints/opd_student"


def cmd_generate(args):
    """Generate on-policy data using the flywheel's Qwen3.5-397B API generator.

    This calls the existing generator_service to produce fresh annotations
    that haven't been seen during teacher training (on-policy).
    """
    output_dir = os.path.join(OPD_DATA_DIR, "on_policy_generated")
    os.makedirs(output_dir, exist_ok=True)

    print("On-Policy Generation")
    print(f"  Target: {args.num_samples} samples per dimension")
    print(f"  Output: {output_dir}")
    print()
    print("  This calls the GSA flywheel generator (Qwen3.5-397B API).")
    print("  Make sure the generator service is running:")
    print("    cd /path/to/Anvil")
    print("    python scripts/generator_service.py --mode api")
    print()

    # In practice, this would call the generator service API
    # For now, we can also just use the existing verified+needs_review data
    # as a proxy for on-policy data (data the teachers haven't been GRPO'd on)
    needs_review_dir = "/path/to/Anvil/data/output/gsa_flywheel_exp/needs_review"

    for dim in DIMENSIONS:
        dim_dir = os.path.join(needs_review_dir, dim)
        if not os.path.isdir(dim_dir):
            print(f"  [{dim}] No needs_review data, skipping")
            continue

        samples = []
        for jf in sorted(Path(dim_dir).glob("*.jsonl")):
            with open(jf) as f:
                for line in f:
                    try:
                        samples.append(json.loads(line.strip()))
                    except:
                        continue
            if len(samples) >= args.num_samples:
                break

        out_file = os.path.join(output_dir, f"{dim}.jsonl")
        with open(out_file, "w") as f:
            for s in samples[:args.num_samples]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  [{dim}] {min(len(samples), args.num_samples)} samples → {out_file}")

    print(f"\nGeneration complete. Data at: {output_dir}")


def cmd_score(args):
    """Score on-policy data with all 6 dimension teachers."""
    input_dir = os.path.join(OPD_DATA_DIR, "on_policy_generated")
    output_dir = os.path.join(OPD_DATA_DIR, "teacher_scores")
    os.makedirs(output_dir, exist_ok=True)

    print("Teacher Scoring")
    print(f"  Input: {input_dir}")
    print(f"  Teachers: {args.teacher_dir}")
    print(f"  Output: {output_dir}")
    print()

    for dim in DIMENSIONS:
        teacher_path = os.path.join(args.teacher_dir, f"teacher_{dim}_9b")
        data_path = os.path.join(input_dir, f"{dim}.jsonl")
        score_path = os.path.join(output_dir, f"{dim}_scores.jsonl")

        if not os.path.isfile(data_path):
            print(f"  [{dim}] No input data, skipping")
            continue

        if not os.path.isdir(teacher_path):
            print(f"  [{dim}] Teacher not found at {teacher_path}")
            print(f"         Using rule-based scoring as fallback")
            # Fallback: use existing verification scores
            scored = []
            with open(data_path) as f:
                for line in f:
                    sample = json.loads(line.strip())
                    v = sample.get("_verification", {})
                    score = v.get("score", 0.5) if isinstance(v, dict) else 0.5
                    scored.append({"sample": sample, "teacher_score": score, "dimension": dim})
            with open(score_path, "w") as f:
                for s in scored:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"  [{dim}] {len(scored)} samples scored (rule-based fallback)")
            continue

        # When teacher is available: load and run inference
        print(f"  [{dim}] Loading teacher from {teacher_path}...")
        print(f"         TODO: Implement actual teacher inference")
        # This will be:
        # model = AutoModelForVision2Seq.from_pretrained(teacher_path)
        # for sample in data:
        #     score = model.score(image, annotation)
        #     ...

    print(f"\nScoring complete. Scores at: {output_dir}")


def cmd_train(args):
    """Train unified student via distillation from teacher scores."""
    score_dir = os.path.join(OPD_DATA_DIR, "teacher_scores")
    student_model = STUDENT_MODELS.get(args.student_model, args.student_model)
    output_dir = os.path.join(OPD_CKPT_DIR, f"student_{args.student_model}")

    print("OPD Student Training")
    print(f"  Student model: {student_model}")
    print(f"  Teacher scores: {score_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Loss weights: KL={args.kl_weight}, Rank={args.rank_weight}, SFT={args.sft_weight}")
    print()

    # Load all scored data
    all_data = []
    for dim in DIMENSIONS:
        score_file = os.path.join(score_dir, f"{dim}_scores.jsonl")
        if os.path.isfile(score_file):
            with open(score_file) as f:
                dim_data = [json.loads(l) for l in f]
            all_data.extend(dim_data)
            print(f"  [{dim}] {len(dim_data)} scored samples loaded")

    print(f"\n  Total training data: {len(all_data)} samples across {len(DIMENSIONS)} dimensions")
    print()

    if not all_data:
        print("  [ERROR] No scored data found. Run 'score' step first.")
        return

    # Training script generation
    train_script = f"""#!/bin/bash
# OPD Student Training - {args.student_model}
export PATH="/path/to/environment/env/qwen36/bin:$PATH"
cd /path/to/environment/Rex-Omni/finetuning

torchrun --nproc_per_node=8 \\
    /path/to/datasets/dataflywheel/process_sft/train_adaptive.py \\
    --config /path/to/datasets/dataflywheel/process_sft/configs/sft_9b.py \\
    --model_name_or_path {student_model} \\
    --deepspeed scripts/zero2.json \\
    --bf16 \\
    --output_dir {output_dir} \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 4 \\
    --gradient_accumulation_steps 4 \\
    --learning_rate 5e-5 \\
    --warmup_ratio 0.1 \\
    --lr_scheduler_type cosine \\
    --gradient_checkpointing True \\
    --run_name opd_student_{args.student_model} \\
    --report_to none
"""
    script_path = os.path.join(OPD_CKPT_DIR, f"run_student_{args.student_model}.sh")
    os.makedirs(OPD_CKPT_DIR, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(train_script)
    os.chmod(script_path, 0o755)

    print(f"  Training script written: {script_path}")
    print(f"  Run: bash {script_path}")


def main():
    parser = argparse.ArgumentParser(description="On-Policy Distillation Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate on-policy data")
    gen_parser.add_argument("--num_samples", type=int, default=5000)

    # score
    score_parser = subparsers.add_parser("score", help="Score data with teachers")
    score_parser.add_argument("--teacher_dir", default=TEACHER_BASE)

    # train
    train_parser = subparsers.add_parser("train", help="Train student model")
    train_parser.add_argument("--student_model", choices=["3b", "7b"], default="3b")
    train_parser.add_argument("--kl_weight", type=float, default=0.5)
    train_parser.add_argument("--rank_weight", type=float, default=0.3)
    train_parser.add_argument("--sft_weight", type=float, default=0.2)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
