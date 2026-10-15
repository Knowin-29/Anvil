"""GRPO Trainer for Verifier — Group Relative Policy Optimization。

GRPO 通过对同一 prompt 生成多个 response，用 rule-based reward 排序后做 policy gradient。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger("verifier.grpo")


def load_grpo_dataset(path: str) -> Dataset:
    """加载 GRPO 数据 (只需要 prompt)。"""
    data = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data.append({"prompt": obj.get("instruction", obj.get("prompt", ""))})
    return Dataset.from_list(data)


def rule_based_reward(response: str, **kwargs) -> float:
    """Rule-based reward function for verifier outputs.
    
    Rewards:
    - Valid JSON output: +0.3
    - Score in [0, 1]: +0.2
    - Has reason field: +0.2
    - Reason length > 10 chars: +0.1
    - Consistent (score matches reason sentiment): +0.2
    """
    reward = 0.0
    
    try:
        obj = json.loads(response)
        reward += 0.3  # Valid JSON
        
        score = obj.get("score", -1)
        if 0 <= score <= 1:
            reward += 0.2  # Valid score range
        
        reason = obj.get("reason", "")
        if reason:
            reward += 0.2  # Has reason
            if len(reason) > 10:
                reward += 0.1  # Meaningful reason
        
        # Consistency check
        if reason:
            positive_words = ["correct", "good", "accurate", "specific", "grounded", "valid"]
            negative_words = ["incorrect", "wrong", "generic", "invalid", "nonsensical", "low"]
            has_positive = any(w in reason.lower() for w in positive_words)
            has_negative = any(w in reason.lower() for w in negative_words)
            
            if (score >= 0.6 and has_positive) or (score < 0.4 and has_negative):
                reward += 0.2  # Consistent
    except (json.JSONDecodeError, TypeError):
        reward = 0.0  # Invalid JSON gets 0
    
    return reward


class RuleBasedRewardModel:
    """Wrapper for rule-based reward to work with GRPOTrainer."""
    
    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        return [rule_based_reward(c) for c in completions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--bf16", type=str, default="True")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--reward_model", default="rule_based")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    dataset = load_grpo_dataset(args.data_path)
    logger.info(f"GRPO dataset: {len(dataset)} prompts")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16.lower() == "true",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_completion_length=512,
        num_generations=args.num_generations,
        gradient_checkpointing=True,
        report_to="wandb",
    )

    reward_fn = RuleBasedRewardModel()

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        reward_funcs=reward_fn,
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
