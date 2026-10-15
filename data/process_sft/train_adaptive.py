"""
Adaptive training script that wraps Rex-Omni's train.py to support
both Qwen2.5-VL and Qwen3.5 model architectures.

Usage:
    torchrun ... train_adaptive.py --config ... --model_name_or_path ...
"""
import sys
import os

# Add Rex-Omni finetuning to path
REX_OMNI_FT = "/path/to/environment/Rex-Omni/finetuning"
sys.path.insert(0, REX_OMNI_FT)
os.chdir(REX_OMNI_FT)

# Patch flash_attn import before anything else loads engine/
import types
import importlib
import importlib.util

# Create a proper mock that passes all transformers checks
flash_spec = importlib.machinery.ModuleSpec("flash_attn", None)
flash_mock = types.ModuleType("flash_attn")
flash_mock.__version__ = "2.7.0"
flash_mock.__spec__ = flash_spec
flash_mock.__file__ = "/fake/flash_attn/__init__.py"
flash_mock.__path__ = ["/fake/flash_attn"]

flash_iface = types.ModuleType("flash_attn.flash_attn_interface")
flash_iface.__spec__ = importlib.machinery.ModuleSpec("flash_attn.flash_attn_interface", None)
flash_iface.flash_attn_varlen_func = lambda *a, **kw: None

flash_mock.flash_attn_interface = flash_iface
sys.modules["flash_attn"] = flash_mock
sys.modules["flash_attn.flash_attn_interface"] = flash_iface

import json
import torch
import transformers
from pathlib import Path
from mmengine.config import Config

from engine.argument import DataArguments, ModelArguments, TrainingArguments
from engine.registry import BUILDER
from engine.utils import is_main_process


def detect_model_type(model_path: str) -> str:
    """Detect model architecture from config.json."""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")
        archs = cfg.get("architectures", [])
        if "qwen3_5" in model_type or "Qwen3_5" in str(archs):
            return "qwen3.5"
        elif "qwen2_5_vl" in model_type or "Qwen2_5_VL" in str(archs):
            return "qwen2.5vl"
    return "qwen2.5vl"


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    cfg = Config.fromfile(training_args.config)
    model_type = detect_model_type(model_args.model_name_or_path)

    if is_main_process():
        print(f"[train_adaptive] Detected model type: {model_type}")
        print(f"[train_adaptive] Model path: {model_args.model_name_or_path}")

    if model_type == "qwen3.5":
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
        try:
            from liger_kernel.transformers import apply_liger_kernel
            apply_liger_kernel()
        except Exception:
            pass

        model = AutoModelForVision2Seq.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="sdpa",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        data_args.image_processor = processor.image_processor
        data_args.model_type = "qwen3.5"

    else:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
        from engine.trainer import replace_qwen2_vl_attention_class

        apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="sdpa",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path
        ).image_processor
        data_args.model_type = "qwen2.5vl"

        if data_args.data_flatten:
            replace_qwen2_vl_attention_class()

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Set trainable params
    from train import set_model
    set_model(model_args, model)

    if is_main_process():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[train_adaptive] Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    # Build dataset
    train_dataset = BUILDER.build(
        cfg.train_dataset,
        default_args=dict(tokenizer=tokenizer, data_args=data_args),
    )
    data_collator = BUILDER.build(
        cfg.data_collator,
        default_args=dict(tokenizer=tokenizer),
    )

    if is_main_process():
        print(f"[train_adaptive] Dataset size: {len(train_dataset)}")

    # Copy chat_template if exists
    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    if os.path.exists(source_path) and training_args.output_dir:
        os.makedirs(training_args.output_dir, exist_ok=True)
        import shutil
        shutil.copy(source_path, os.path.join(training_args.output_dir, "chat_template.json"))

    model.config.use_cache = True

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
