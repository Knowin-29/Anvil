"""Extend an existing language adapter with zero-delta visual LoRA modules.

This prepares a new initialization, not a trained checkpoint. Existing adapter
tensors remain identical; fresh visual B matrices are zero. No source files,
optimizer states, evaluation inputs or evaluation scorers are changed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def visual_shapes(base):
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    selected = {}
    pattern = re.compile(r'model\.visual\.(?:blocks\.(?:19|2[0-6])\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])|merger\.linear_fc[12])\.weight')
    for name, shard in index['weight_map'].items():
        if pattern.fullmatch(name):
            with safe_open(base / shard, framework='pt', device='cpu') as source:
                shape = source.get_slice(name).get_shape()
            assert len(shape) == 2
            selected[name.removesuffix('.weight')] = shape
    assert len(selected) == 34, f'Expected 32 final-block and 2 merger projections, got {len(selected)}'
    return dict(sorted(selected.items()))


def expand(source, base, output, expected_source_sha, rank=32, seed=20261104):
    assert not output.exists(), 'Initialization output already exists'
    source_file = source / 'adapter_model.safetensors'
    assert sha(source_file) == expected_source_sha, 'Changed initialization adapter'
    seal = json.loads((source / 'checkpoint_complete.json').read_text())
    assert seal['files']['adapter_model.safetensors']['sha256'] == expected_source_sha
    assert sha(source / 'adapter_config.json') == seal['files']['adapter_config.json']['sha256']
    config = json.loads((source / 'adapter_config.json').read_text())
    assert isinstance(config['target_modules'], str)
    assert config['r'] == 64 and config['lora_alpha'] == 128 and not config['use_rslora']
    original = load_file(source_file, device='cpu')
    assert len(original) == 496 and not any('.visual.' in name for name in original)
    assert sum(value.numel() for value in original.values()) == 173113344
    shapes = visual_shapes(base)
    state = dict(original)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    for module, (out_features, in_features) in shapes.items():
        prefix = 'base_model.model.' + module
        # PEFT's ordinary Kaiming-uniform A / zero B initialization.
        a = torch.empty((rank, in_features), dtype=torch.float32)
        a.uniform_(-1 / math.sqrt(in_features), 1 / math.sqrt(in_features), generator=generator)
        state[prefix + '.lora_A.weight'] = a
        state[prefix + '.lora_B.weight'] = torch.zeros((out_features, rank), dtype=torch.float32)
    config['target_modules'] = '(?:' + config['target_modules'] + ')|(?:^(' + '|'.join(re.escape(name) for name in shapes) + ')$)'
    config['rank_pattern'] = {**config.get('rank_pattern', {}), **{name: rank for name in shapes}}
    config['alpha_pattern'] = {**config.get('alpha_pattern', {}), **{name: 2 * rank for name in shapes}}
    config['base_model_name_or_path'] = str(base)
    staging = output.with_name(output.name + '.building-' + str(os.getpid()))
    staging.mkdir(parents=True, exist_ok=False)
    (staging / 'adapter_config.json').write_text(json.dumps(config, indent=2) + '\n')
    save_file(state, staging / 'adapter_model.safetensors', metadata={'format': 'pt'})
    del state
    restored = load_file(staging / 'adapter_model.safetensors', device='cpu')
    assert all(torch.equal(restored[key], value) for key, value in original.items())
    additions = set(restored) - set(original)
    assert len(additions) == 68
    assert all(torch.isfinite(restored[key]).all() for key in additions)
    assert all(torch.count_nonzero(restored[key]) == 0 for key in additions if '.lora_B.' in key)
    receipt = dict(schema='anvil-zero-delta-visual-adapter-initialization-v1',
        state='prepared_not_trained', created=time.time(), source=str(source),
        source_adapter_sha256=expected_source_sha, base=str(base), output=str(output),
        preserved_language_tensors=len(original), added_visual_tensors=len(additions),
        language_parameters=sum(t.numel() for t in original.values()),
        visual_parameters=sum(restored[key].numel() for key in additions),
        total_adapter_parameters=sum(t.numel() for t in restored.values()),
        visual_rank=rank, visual_alpha=2 * rank, visual_modules=shapes,
        initial_visual_delta_exactly_zero=True, language_tensors_exactly_unchanged=True,
        seed=seed, source_modified=False, optimizer_resumed=False,
        training_performed=False, quality_accepted=False,
        adapter_sha256=sha(staging / 'adapter_model.safetensors'),
        config_sha256=sha(staging / 'adapter_config.json'))
    (staging / 'initialization.json').write_text(json.dumps(receipt, indent=2) + '\n')
    staging.rename(output)
    return receipt


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--source-sha256', required=True)
    args = ap.parse_args()
    print(json.dumps(expand(args.source, args.base, args.output, args.source_sha256)))
