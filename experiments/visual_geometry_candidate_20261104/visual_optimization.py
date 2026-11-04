"""Separate visual/language optimization and verify real visual learning."""
import hashlib
import math

import torch


def parameter_groups(model, expected):
    named = {'language': [], 'visual': []}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if '.lora_' not in name:
            raise ValueError('Unexpected trainable base parameter: ' + name)
        family = 'visual' if '.visual.' in name else 'language' if '.language_model.' in name else None
        if family is None:
            raise ValueError('Unknown trainable adapter parameter: ' + name)
        named[family].append((name, value))
    for family in named:
        count = sum(value.numel() for _, value in named[family])
        if count != expected[family + '_parameters'] or not count:
            raise ValueError(f'{family} trainable count changed: {count}')
    groups = [dict(params=[value for _, value in named[family]], family=family)
              for family in ('language', 'visual')]
    return groups, named


def set_learning_rates(optimizer, config, ratio, targeted):
    if not 0 <= ratio <= 1:
        raise ValueError('Invalid learning-rate schedule ratio')
    if [g.get('family') for g in optimizer.param_groups] != ['language', 'visual']:
        raise ValueError('Optimizer groups changed, including during resume')
    language = config.get('targeted_geometry_lr', config['lr']) if targeted else config['lr']
    visual = config['visual_lr']
    if not all(math.isfinite(x) and x > 0 for x in [language, visual]):
        raise ValueError('Each adapter group needs a positive finite peak learning rate')
    for group, peak in zip(optimizer.param_groups, [language, visual]):
        group['lr'] = peak * ratio


def gradient_norms(named):
    norms = {}
    for family, items in named.items():
        gradients = []
        for name, value in items:
            if value.grad is None:
                raise ValueError('Missing adapter gradient: ' + name)
            gradients.append(torch.linalg.vector_norm(value.grad.detach(), dtype=torch.float32))
        total = torch.linalg.vector_norm(torch.stack(gradients))
        if not torch.isfinite(total):
            raise ValueError('Nonfinite adapter gradient: ' + family)
        norms[family] = float(total)
    return norms


def visual_state(named):
    result = {}
    for name, value in named['visual']:
        tensor = value.detach().float().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite visual adapter tensor: ' + name)
        result[name] = dict(sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                            nonzero=int(torch.count_nonzero(tensor)), parameters=tensor.numel())
    return result


def visual_update_receipt(named, baseline, baseline_step, step):
    current = visual_state(named)
    if set(current) != set(baseline) or step <= baseline_step:
        raise ValueError('Visual update lineage changed')
    changed = [name for name in current if current[name]['sha256'] != baseline[name]['sha256']]
    changed_b = [name for name in changed if '.lora_B.' in name]
    if not changed_b:
        raise ValueError('No visual B matrix changed during real training')
    return dict(schema='anvil-visual-training-update-v1', baseline_step=baseline_step, step=step,
                tensors=len(current), changed_tensors=len(changed), changed_b_tensors=len(changed_b),
                baseline=baseline, current=current, real_training_update_verified=True)
