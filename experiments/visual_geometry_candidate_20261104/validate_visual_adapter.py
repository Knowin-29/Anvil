"""CPU-only adapter compatibility and zero-delta/gradient checks.

Construct the real 9B module graph on meta tensors, then exercise an isolated
visual LoRA projection on CPU. This is not a model-quality or GPU experiment.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from peft.tuners.lora.layer import Linear
from safetensors import safe_open
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

from expand_visual_adapter import sha


def validate(initialization):
    torch.set_num_threads(4)
    receipt = json.loads((initialization / 'initialization.json').read_text())
    assert sha(initialization / 'adapter_model.safetensors') == receipt['adapter_sha256']
    assert sha(initialization / 'adapter_config.json') == receipt['config_sha256']
    config = AutoConfig.from_pretrained(receipt['base'], local_files_only=True)
    adapter_config = LoraConfig.from_pretrained(initialization)
    # Match PeftModel.from_pretrained(..., is_trainable=True), which overrides
    # the inference_mode flag written by save_pretrained.
    adapter_config.inference_mode = False
    with init_empty_weights():
        model = Qwen3_5ForConditionalGeneration(config)
        model = get_peft_model(model, adapter_config)
    expected = get_peft_model_state_dict(model)
    with safe_open(initialization / 'adapter_model.safetensors', framework='pt', device='cpu') as stored:
        assert set(expected) == set(stored.keys()), 'Saved/expected adapter key mismatch'
        for name, tensor in expected.items():
            assert list(tensor.shape) == stored.get_slice(name).get_shape(), name
    trainable = [(name, tensor) for name, tensor in model.named_parameters() if tensor.requires_grad]
    assert sum(t.numel() for _, t in trainable) == receipt['total_adapter_parameters']
    vision = [(name, tensor) for name, tensor in trainable if '.visual.' in name]
    assert len(vision) == 68 and sum(t.numel() for _, t in vision) == receipt['visual_parameters']
    assert all('.lora_' in name for name, _ in trainable), 'Unexpected full-weight training'
    for name, module in model.named_modules():
        if '.visual.' in name and hasattr(module, 'lora_A'):
            assert module.r['default'] == 32 and module.lora_alpha['default'] == 64
    del expected, model, trainable, vision
    # Use a real configured visual-projection shape, but independent CPU-only
    # tensors; no candidate parameter is modified by this gradient test.
    module_name = 'model.visual.blocks.19.attn.qkv'
    out_features, in_features = receipt['visual_modules'][module_name]
    torch.manual_seed(17)
    base = torch.nn.Linear(in_features, out_features, bias=False)
    base.requires_grad_(False)
    projection_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                                   target_modules=['qkv'])
    layer = Linear(base, 'default', config=projection_config, r=32, lora_alpha=64)
    x = torch.randn(2, in_features)
    baseline = base(x).detach()
    initial = layer(x)
    assert torch.equal(initial, baseline), 'Zero-delta initialization changes output'
    initial.square().mean().backward()
    gradient = layer.lora_B['default'].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
    gradient_norm = float(gradient.norm())
    with torch.no_grad():
        layer.lora_B['default'].weight.add_(gradient, alpha=-0.01)
    assert not torch.equal(layer(x).detach(), baseline), 'Visual delta cannot learn'
    return dict(state='cpu_compatibility_zero_delta_and_gradient_passed', checked=time.time(),
        initialization=str(initialization), adapter_sha256=receipt['adapter_sha256'],
        model_graph='Qwen3_5ForConditionalGeneration', meta_key_count=564,
        language_parameters=receipt['language_parameters'], visual_parameters=receipt['visual_parameters'],
        total_adapter_parameters=receipt['total_adapter_parameters'],
        isolated_visual_gradient_norm=gradient_norm, gpu_used=False,
        quality_validation_performed=False, training_started=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--initialization', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = validate(args.initialization)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
