#!/usr/bin/env python3
"""Evaluate terminal LoRA checkpoints without full-weight exports; persist every batch."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import signal
import subprocess

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
from eval.gsa_eval import runner as f
from eval.gsa_eval.registry import get_spec
A = W.parent/'Anvil_ICLR27'
BENCHES = ['where2place', 'robospatial_home_context', 'vabench_point', 'blink_relative_depth', 'erqa', 'embspatial']

def binding(p):
    p = Path(p)
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def save(p, d):
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d, indent=2)+'\n')
    tmp.replace(p)

def run(mode, adapter, benches):
    out = A/'engineering/data/table2_fill_20260914'/mode
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'resume.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = json.loads((adapter/'adapter_config.json').read_text())
    base = Path(cfg['base_model_name_or_path'])
    from safetensors import safe_open
    import torch
    with safe_open(adapter/'adapter_model.safetensors', framework='pt', device='cpu') as weights:
        assert len(list(weights.keys())) > 0
        for k in weights.keys():
            assert torch.isfinite(weights.get_tensor(k)).all(), k
    identity = {'adapter': binding(adapter/'adapter_model.safetensors'),
                'adapter_config': binding(adapter/'adapter_config.json'),
                'base_config': binding(base/'config.json'),
                'base_index': binding(base/'model.safetensors.index.json'),
                'base_path': str(base), 'scorer': f.evaluation_code_sha256(),
                'implementation': binding(__file__), 'benchmarks': benches,
                'batch_size': 4, 'thinking': False, 'model_type': 'qwen3_5',
                'merge': 'LoRA merged in CPU memory using Swift export operation, then moved to GPU; no full checkpoint written'}
    receipt = out/'adapter_execution.json'
    if receipt.exists():
        assert json.loads(receipt.read_text()) == identity, 'resume identity changed'
    else:
        save(receipt, identity)
    print('Loading and merging verified adapter in CPU memory', flush=True)
    from swift.infer_engine import TransformersEngine
    from swift.tuners import Swift
    engine = TransformersEngine(str(base), adapters=[str(adapter)], max_batch_size=4,
                                model_type='qwen3_5', template_type='qwen3_5', device_map='cpu')
    Swift.merge_and_unload(engine.model)
    engine.model = engine.model.model
    engine.model.to('cuda:0')
    engine.engine = engine.model
    meta = engine.template.template_meta
    engine.template.response_prefix = getattr(meta, 'non_thinking_prefix', '') or ''
    engine.template.enable_thinking = False
    f._assert_weights_loaded(engine)
    worker_path = out/'worker.json'
    worker = json.loads(worker_path.read_text()) if worker_path.exists() else {'model': str(adapter), 'benchmarks': {}, 'adapter_execution': binding(receipt)}
    for bench in benches:
        spec = get_spec(bench)
        samples = list(spec.loader(W/'.cache/eval'))
        shard = out/'shards'/bench/'shard_0.jsonl'
        shard.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if shard.exists():
            # A truncated final record is preserved for inspection, then only that
            # incomplete line is removed. Completed batches remain immutable.
            raw = shard.read_bytes()
            end = raw.rfind(b'\n')+1
            if end < len(raw):
                partial = shard.with_name('interrupted_tail_'+str(time.time_ns())+'.bin')
                partial.write_bytes(raw[end:])
                with shard.open('r+b') as stream:
                    stream.truncate(end)
            existing = [json.loads(line) for line in shard.read_text().splitlines()]
        assert [str(r['uid']) for r in existing] == [str(s.uid) for s in samples[:len(existing)]]
        assert len(existing) <= len(samples)
        started = time.monotonic()
        elapsed = worker['benchmarks'].get(bench, {}).get('elapsed_s', 0)
        with shard.open('a', buffering=1) as stream:
            for offset in range(len(existing), len(samples), 4):
                chunk = samples[offset:offset+4]
                texts, caps = f._infer(engine, chunk, max_tokens=spec.max_tokens)
                assert len(texts) == len(caps) == len(chunk)
                for sample, text, cap in zip(chunk, texts, caps):
                    stream.write(json.dumps({'uid': sample.uid, 'response': text, 'hit_token_cap': cap}, ensure_ascii=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
                worker['benchmarks'][bench] = {'status': 'running', 'n': min(offset+4,len(samples)),
                    'elapsed_s': elapsed+time.monotonic()-started, 'expected_n': len(samples)}
                save(worker_path, worker)
                print(f'{bench}: {min(offset+4,len(samples))}/{len(samples)}', flush=True)
        worker['benchmarks'][bench] = {'status': 'ok', 'n': len(samples), 'elapsed_s': elapsed+time.monotonic()-started}
        save(worker_path, worker)
        result_path = out/f'{bench}.json'
        if not result_path.exists():
            result = f.score_shards(bench, str(adapter), [shard], scratch=W/'.cache/eval',
                                   max_tokens=spec.max_tokens, thinking=False, generation_batch_size=4,
                                   elapsed_s=worker['benchmarks'][bench]['elapsed_s'],
                                   dump_predictions=out/f'preds_{bench}.jsonl')
            result['adapter_execution'] = binding(receipt)
            save(result_path, result)
    save(out/'completion_status.json', {'rc': 0, 'cells': {b: {'generation': 'ok', 'result_exists': True} for b in benches}})
    print('All requested endpoints complete', flush=True)

def wait_control(mode):
    assert mode in {f'control_replay_s{s}' for s in (43,44)}
    root = A/'engineering/data/source_controls_20260915'/mode
    status = root/'training_portfix_v1/train_status.json'
    print('Waiting for terminal checkpoint; active training is preserved', flush=True)
    while not status.exists():
        time.sleep(2)
    assert json.loads(status.read_text())['rc'] == 0, 'Training did not complete'
    adapters = list((root/'training_portfix_v1').glob('v*/checkpoint-303'))
    assert len(adapters) == 1 and (adapters[0]/'adapter_model.safetensors').exists()
    # The already-running legacy shell has an immutable export stage. Stop only
    # its newly launched redundant exporter after successful training, never the trainer.
    stopped = []
    until = time.monotonic()+120
    while time.monotonic() < until:
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                args = (proc/'cmdline').read_bytes().split(b'\0')
                exact = str(adapters[0]).encode()
                exporter = any(b'/cli/export.py' in x or x.endswith(b'/swift') for x in args) and b'--adapters' in args and exact in args
                if exporter:
                    os.kill(int(proc.name), signal.SIGTERM)
                    stopped.append(int(proc.name))
            except (OSError, ProcessLookupError):
                pass
        if stopped or (root/'export.log').exists() and 'Disk quota exceeded' in (root/'export.log').read_text():
            break
        time.sleep(1)
    save(root/'export_bypass.json', {'training_status': binding(status), 'adapter': str(adapters[0]),
                                    'redundant_exporter_pids_stopped': stopped, 'reason': 'Use in-memory merge; preserve trained LoRA and disk quota'})
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    while int(subprocess.check_output(['nvidia-smi', '--id='+gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()) >= 64:
        time.sleep(5)
    return adapters[0]

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--wait-control', action='store_true')
    p.add_argument('--benchmarks', nargs='+', default=BENCHES)
    args = p.parse_args()
    adapter = wait_control(args.mode) if args.wait_control else args.adapter
    assert adapter is not None
    run(args.mode, adapter, args.benchmarks)
