#!/usr/bin/env python3
"""
Standalone SFT training script for EmbodiedForge data.
Does NOT depend on Rex-Omni engine imports.
Uses transformers Trainer directly with Qwen3.5 VL model.

Usage:
    torchrun --nproc_per_node=8 train_standalone.py
"""
import argparse
import json
import os
import sys
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class EmbodiedForgeGroundingDataset(Dataset):
    """Loads grounding TSV data for SFT training."""

    def __init__(self, img_tsv, ann_tsv, ann_idx, processor, max_length=2048):
        self.img_tsv = img_tsv
        self.ann_tsv = ann_tsv
        self.processor = processor
        self.max_length = max_length

        with open(ann_idx) as f:
            self.offsets = [int(line.strip()) for line in f]

        self.img_handle = None
        self.ann_handle = None

    def __len__(self):
        return len(self.offsets)

    def _load(self, idx):
        if self.ann_handle is None:
            self.ann_handle = open(self.ann_tsv)
        self.ann_handle.seek(self.offsets[idx])
        line = self.ann_handle.readline().strip()
        img_offset_str, ann_json = line.split("\t", 1)
        img_offset = int(img_offset_str)

        if self.img_handle is None:
            self.img_handle = open(self.img_tsv)
        self.img_handle.seek(img_offset)
        img_b64 = self.img_handle.readline().strip()
        image = Image.open(BytesIO(b64decode(img_b64))).convert("RGB")
        annotation = json.loads(ann_json)
        return image, annotation

    def __getitem__(self, idx):
        try:
            image, ann = self._load(idx)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        if "boxes" in ann:
            boxes = ann["boxes"]
            phrases = [b["phrase"] for b in boxes if b.get("phrase")]
            bboxes = [b["bbox"] for b in boxes if b.get("bbox")]
            if not phrases:
                return self.__getitem__((idx + 1) % len(self))
            text = f"Detect: {', '.join(phrases)}"
            bbox_str = "; ".join([f"[{b[0]},{b[1]},{b[2]},{b[3]}]" for b in bboxes])
            response = f"Found objects: {bbox_str}"
        elif "conversations" in ann:
            convs = ann["conversations"]
            text = convs[0]["value"].replace("<image>\n", "")
            response = convs[1]["value"] if len(convs) > 1 else ""
        else:
            return self.__getitem__((idx + 1) % len(self))

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ]},
            {"role": "assistant", "content": response},
        ]

        try:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, return_tensors="pt",
                max_length=self.max_length, truncation=True,
                return_dict=True, add_generation_prompt=False,
            )
            # Flatten batch dim
            return {k: v.squeeze(0) if hasattr(v, 'squeeze') else v for k, v in inputs.items()}
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/path/to/environment/hf_models/Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--tsv_base", default="/path/to/datasets/dataflywheel/output/rex_omni_tsv")
    parser.add_argument("--output_dir", default="/path/to/datasets/dataflywheel/output/checkpoints/sft_9b")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--local_rank", type=int, default=-1)
    args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[SFT] Loading model: {args.model_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"[SFT] Loading dataset from: {args.tsv_base}")
    dataset = EmbodiedForgeGroundingDataset(
        img_tsv=f"{args.tsv_base}/embodiedforge_grounding.images.tsv",
        ann_tsv=f"{args.tsv_base}/embodiedforge_grounding.annotations.tsv",
        ann_idx=f"{args.tsv_base}/embodiedforge_grounding.annotations.tsv.lineidx",
        processor=processor,
    )
    print(f"[SFT] Dataset size: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="wandb",
        run_name="embodiedforge_sft_9b",
        deepspeed="zero2.json" if os.path.exists("zero2.json") else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[SFT] Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
