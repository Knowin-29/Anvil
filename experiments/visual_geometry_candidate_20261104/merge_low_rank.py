"""Map mixed-rank language/visual LoRA tensors into the unchanged base graph."""
import math

import torch


def collect_pairs(weights, config, base_weight_map):
    if config.get('use_dora') or config.get('use_rslora') or config.get('modules_to_save'):
        raise ValueError('Unsupported adapter merge semantics')
    if config.get('bias', 'none') != 'none' or config.get('lora_bias', False):
        raise ValueError('Bias adapters are not supported')
    ranks, alphas = config.get('rank_pattern', {}), config.get('alpha_pattern', {})
    pairs, used, modules = {}, set(), set()
    for key, a in weights.items():
        if not key.endswith('.lora_A.weight'):
            continue
        if not key.startswith('base_model.model.'):
            raise ValueError('Unexpected adapter prefix')
        module = key.removeprefix('base_model.model.').removesuffix('.lora_A.weight')
        target = module + '.weight'
        bkey = key.replace('.lora_A.weight', '.lora_B.weight')
        if target not in base_weight_map or bkey not in weights:
            raise ValueError('Unmapped or incomplete adapter pair: ' + module)
        b = weights[bkey]
        rank, alpha = ranks.get(module, config['r']), alphas.get(module, config['lora_alpha'])
        if not isinstance(rank, int) or rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('Invalid per-module rank or scale')
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError('Adapter rank does not match tensor shapes: ' + module)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Nonfinite adapter pair')
        pairs[target] = (a, b, alpha / rank)
        modules.add(module)
        used.update((key, bkey))
    if used != set(weights):
        raise ValueError('Unmapped adapter tensors')
    # The expanded adapter stores exact module names. Fail rather than silently
    # treating an unrecognized regex pattern as a default-rank module.
    if not set(ranks).issubset(modules) or not set(alphas).issubset(modules):
        raise ValueError('Unresolved per-module adapter options')
    return pairs


def merge_tensor(base, pair):
    a, b, scale = pair
    if base.ndim != 2 or (b.shape[0], a.shape[1]) != tuple(base.shape):
        raise ValueError('Adapter/base projection shape mismatch')
    result = (base.float() + (b.float() @ a.float()) * scale).to(base.dtype)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite merged tensor')
    return result.contiguous()
