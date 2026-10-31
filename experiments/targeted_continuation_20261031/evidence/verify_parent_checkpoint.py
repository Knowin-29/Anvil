"""CPU-only audit of the sealed recovery MOPD endpoint; never edits live inputs."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from safetensors.torch import load_file

C = Path('/Knowin/sim/weilinruan/Anvil/recovery_20261020/completion_20261028')
O = C / 'direct_recovery_20261030'
RUN = O / 'runs/mopd'
STEP = 1536
CKPT = RUN / f'checkpoint-{STEP}'
OUT = O / 'mopd_checkpoint1536_verification.json'
SPEC_SHA = '3be2abe0bd7d4753c0bc97411bada5ba04280fa8a52e24ddcea7aa56841e9cde'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def require(value, message):
    if not value:
        raise RuntimeError(message)

def finite(value):
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(x) for x in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True

def main():
    torch.set_num_threads(2)
    require(not OUT.exists(), 'Existing verification evidence must not be overwritten')
    require(sha(O / 'spec.json') == SPEC_SHA, 'Training specification changed')
    spec = read(O / 'spec.json')
    protocol = read(RUN / 'protocol.json')
    state = read(CKPT / 'trainer_state.json')
    seal = read(CKPT / 'checkpoint_complete.json')
    complete = read(RUN / 'complete.json')
    required = {'README.md', 'adapter_config.json', 'adapter_model.safetensors',
                'optimizer.pt', 'scheduler.pt', 'trainer_state.json'}
    required |= {f'rng_state_{i}.pth' for i in range(32)}
    require(set(seal['files']) == required, 'Checkpoint payload membership differs')
    require(seal['global_step'] == STEP and seal['world_size'] == 32, 'Wrong sealed endpoint')
    for name, expected in seal['files'].items():
        p = CKPT / name
        require(p.is_file() and not p.is_symlink(), f'Missing or indirect payload: {name}')
        require(p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'],
                f'Payload integrity failure: {name}')
    print('All 38 checkpoint payload hashes verified', flush=True)
    require(state['global_step'] == STEP and state['max_steps'] == STEP and
            state['stage'] == 'mopd' and state['examples_seen'] == STEP * 64 and
            state['final_mopd'] and not state['smoke'], 'Wrong trainer state')
    require(complete['steps'] == STEP and Path(complete['checkpoint']).resolve() == CKPT.resolve()
            and complete['stage'] == 'mopd_training_complete' and not complete['smoke'],
            'Wrong completion receipt')
    require(complete['adapter_sha256'] == sha(CKPT / 'adapter_model.safetensors') and
            complete['spec_sha256'] == SPEC_SHA, 'Completion binding differs')
    require(protocol['spec_sha256'] == SPEC_SHA and protocol['world_size'] == 32 and
            protocol['global_batch'] == 64 and protocol['updates'] == STEP and
            protocol['code_sha256'] == sha(O / 'train.py'), 'Protocol binding differs')
    parent = read(O / 'runs/repair/complete.json')
    initial = Path(protocol['initial'])
    require(initial.resolve() == Path(parent['checkpoint']).resolve() and parent['steps'] == 2048,
            'Wrong repair ancestry')
    require(sha(initial / 'adapter_model.safetensors') == parent['adapter_sha256'] ==
            protocol['initial_adapter_sha256'], 'Repair adapter changed')
    adapter = load_file(str(CKPT / 'adapter_model.safetensors'), device='cpu')
    before = load_file(str(initial / 'adapter_model.safetensors'), device='cpu')
    require(set(adapter) == set(before) and len(adapter) == 496, 'Adapter tensor schema differs')
    changed = 0
    for name, value in adapter.items():
        require(value.shape == before[name].shape and bool(torch.isfinite(value).all()),
                'Invalid adapter tensor: ' + name)
        changed += int(not torch.equal(value, before[name]))
    require(changed > 0, 'No adapter update occurred')
    adapter_parameters = sum(t.numel() for t in adapter.values())
    require(adapter_parameters == 173113344, 'Adapter parameter count differs')
    del before, adapter
    optimizer = torch.load(CKPT / 'optimizer.pt', map_location='cpu', weights_only=True)
    param_ids = [p for group in optimizer['param_groups'] for p in group['params']]
    require(len(param_ids) == len(set(param_ids)) == 496 and set(param_ids) == set(optimizer['state']),
            'Optimizer parameter membership differs')
    count = 0
    for value in optimizer['state'].values():
        require(int(value['step'].item()) == STEP, 'Optimizer update step differs')
        m, v = value['exp_avg'], value['exp_avg_sq']
        require(m.shape == v.shape and bool(torch.isfinite(m).all()) and
                bool(torch.isfinite(v).all()) and bool((v >= 0).all()), 'Invalid optimizer moments')
        count += m.numel()
    require(count == adapter_parameters, 'Optimizer parameter count differs')
    cfg = spec['stages']['mopd']
    warmup = math.ceil(STEP * .03)
    expected_lr = cfg['lr'] * .5 * (1 + math.cos(math.pi * (STEP - 1 - warmup) / (STEP - warmup)))
    require(all(math.isclose(g['lr'], expected_lr, rel_tol=1e-10, abs_tol=1e-20)
                for g in optimizer['param_groups']), 'Optimizer learning rate differs')
    del optimizer
    scheduler = torch.load(CKPT / 'scheduler.pt', map_location='cpu', weights_only=True)
    require(scheduler == {'step': STEP, 'steps': STEP, 'schedule': 'warmup3%-cosine'},
            'Scheduler differs')
    for rank in range(32):
        rng = torch.load(CKPT / f'rng_state_{rank}.pth', map_location='cpu', weights_only=False)
        require(set(rng) == {'cpu', 'cuda', 'python', 'numpy'}, 'RNG components missing')
        torch.Generator(device='cpu').set_state(rng['cpu'])
        require(rng['cuda'].dtype == torch.uint8 and rng['cuda'].ndim == 1 and
                rng['cuda'].numel() > 0, 'Invalid CUDA RNG blob')
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
    logs = [json.loads(line) for line in (RUN / 'logging.jsonl').open()]
    require(logs == state['log_history'] and len(logs) == STEP and
            [r['step'] for r in logs] == list(range(1, STEP + 1)) and finite(logs),
            'Training history missing, inconsistent or nonfinite')
    positive_mtd = sum(row['mtd'] > 0 for row in logs)
    require(positive_mtd > 0, 'No positive MTD updates')
    print('Adapter, optimizer, scheduler, RNG and training history verified on CPU', flush=True)
    sources, families = Counter(), Counter()
    ids, images = set(), set()
    rows = native = teacher_rows = teacher_positions = native_tokens = 0
    exposure_digest = hashlib.sha256()
    for rank in range(32):
        seen = set()
        for line in (RUN / f'exposure_rank{rank}.jsonl').open('rb'):
            row = json.loads(line)
            key = (row['step'], row['micro'])
            require(row['rank'] == rank and not row['smoke'] and row['stage'] == 'mopd' and
                    1 <= row['step'] <= STEP and row['micro'] in (0, 1) and key not in seen,
                    f'Invalid or repeated exposure at rank {rank}')
            seen.add(key)
            exposure_digest.update(line)
            rows += 1
            sources[row['source_category']] += 1
            ids.add(row['canonical_uid'])
            rgb = row['source_binding']['decoded_rgb_sha256']
            require(bool(rgb), 'Missing image identity')
            images.add(rgb)
            if row['source_category'] == 'native_distillation':
                native += 1
                families[row['family']] += 1
                teacher_rows += int(row['teacher_scored_positions'] > 0)
                teacher_positions += row['teacher_scored_positions']
                native_tokens += row['rollout_tokens']
                require(row['rollout_role'] == 'on_policy_mtd', 'Wrong native rollout role')
        require(seen == {(s, m) for s in range(1, STEP + 1) for m in (0, 1)},
                f'Incomplete rank {rank} exposure sequence')
    require(rows == STEP * 64 and native == teacher_rows == 49152, 'Wrong exposure or teacher coverage')
    require(dict(sources) == state['source_exposures'] == complete['source_exposures'] ==
            spec['planned_source_exposures']['mopd'], 'Source exposure totals differ')
    result = dict(state='key_mopd_checkpoint1536_integrity_verified',
        checked_at=datetime.now(timezone.utc).isoformat(), checkpoint=str(CKPT), files=38,
        total_bytes=sum(v['bytes'] for v in seal['files'].values()), all_file_hashes_verified=True,
        adapter_tensors_finite=496, changed_adapter_tensor_count=changed,
        adapter_changed_since_repair2048=True, checkpoint_payloads_cpu_loadable=True,
        optimizer_parameters_verified=count, optimizer_step=STEP, scheduler_verified=scheduler,
        all32_rng_files_loadable=True, cuda_rng_blobs_loaded_on_cpu_only=True,
        gpu_resume_experiment=False, initial_repair_checkpoint_and_spec_binding_verified=True,
        finite_training_history=True, positive_mtd_updates=positive_mtd,
        committed_exposure_prefix=dict(rows=rows, all32_rank_prefixes_complete=True,
            unique_records=len(ids), unique_image_bindings=len(images), source_exposures=dict(sources),
            rank_order_prefix_sha256=exposure_digest.hexdigest(), native_rows=native,
            teacher_scored_rows=teacher_rows, native_families=dict(families),
            teacher_scored_positions=teacher_positions, native_rollout_tokens=native_tokens,
            counts_are_not_dataset_inventory=True),
        verification_script_sha256=sha(Path(__file__)), heldout_quality_verified=False, weights_uploaded=False)
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
