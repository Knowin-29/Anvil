"""Evaluate an immutable local candidate using the unchanged public protocol.

No training, publication, automatic result promotion, or wall-clock kill limit.
Each worker loads once for all benchmarks. Completed cells survive a later failure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CURRENT_OUTPUT = None
BENCHMARKS = ('cv_bench', 'embspatial', 'refspatial_bench', 'where2place',
              'blink_relative_depth', 'erqa', 'vabench_point', 'gsa_bench_test')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def acquire(args, run):
    args.lock_root.mkdir(parents=True, exist_ok=True)
    while True:
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                        '--format=csv,noheader'], text=True).splitlines()
        for selector in args.gpus.split(','):
            raw = subprocess.check_output(['nvidia-smi', '-i', selector,
                '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
            device, memory = next(csv.reader([raw.strip()]))
            device = device.strip()
            if device in busy or int(memory.strip()) > 1024:
                continue
            handle = (args.lock_root / (device + '.lock')).open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            # Recheck after acquiring our cooperative lock; unrelated jobs do not
            # share it, so this is a best-effort idle check, not a global reservation.
            busy_now = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                               '--format=csv,noheader'], text=True).splitlines()
            used_now = subprocess.check_output(['nvidia-smi', '-i', device,
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()
            if device in busy_now or int(used_now) > 1024:
                handle.close()
                continue
            save(run / 'gpu.json', {'uuid': device, 'at': stamp()})
            return device, handle
        save(run / 'status.json', {'state': 'waiting_for_idle_gpu', 'pid': os.getpid(), 'at': stamp()})
        time.sleep(15)


def command(run, name, argv, env):
    save(run / (name + '.command.json'), argv)
    started = time.monotonic()
    with (run / (name + '.log')).open('x') as stream:
        child = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT)
        save(run / (name + '.status.json'), {'state': 'running', 'pid': child.pid,
                                           'supervisor_pid': os.getpid(), 'at': stamp()})
        code = child.wait()
    result = {'exit_code': code, 'wall_seconds': time.monotonic() - started,
              'state': 'complete' if code == 0 else 'failed', 'at': stamp()}
    save(run / (name + '.status.json'), result)
    return code


def validate_shard_uids(shards):
    seen = set()
    for path in shards:
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line)['uid']
                if not isinstance(value, (str, int)) or isinstance(value, bool):
                    raise ValueError('invalid sample identity in shard')
                uid = str(value)
                if not uid or uid in seen:
                    raise ValueError('duplicate or empty sample identity across shards')
                seen.add(uid)
    # Full-set membership/completeness is checked by the unchanged CPU scorer.
    return len(seen)


def result_is_complete(results, worker_codes, score_codes, unchanged):
    return bool(unchanged and worker_codes and not any(worker_codes) and not any(score_codes)
                and len(score_codes) == len(BENCHMARKS)
                and len(results) == len(BENCHMARKS)
                and {row['benchmark'] for row in results} == set(BENCHMARKS)
                and all(row.get('state') == 'scored'
                        and row.get('metrics', {}).get('n_evaluated', 0) > 0
                        and not row.get('metrics', {}).get('skipped') for row in results))


def main():
    global CURRENT_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--benchg-data', type=Path, required=True)
    parser.add_argument('--lock-root', type=Path, required=True)
    parser.add_argument('--runtime-parent', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--candidate-label', required=True,
                        help='Descriptive checkpoint identity; not a qualification or release claim')
    parser.add_argument('--selection-description', required=True,
                        help='How this checkpoint was selected before seeing these test results')
    args = parser.parse_args()
    if args.workers < 1 or args.workers > len(set(args.gpus.split(','))):
        parser.error('workers must fit the explicitly selected GPU set')
    args.output.mkdir(parents=True, exist_ok=False)
    CURRENT_OUTPUT = args.output
    args.runtime_parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    save(args.output / 'status.json', {'state': 'hashing_and_snapshotting',
                                      'pid': os.getpid(), 'at': stamp()})
    model = args.model.resolve(strict=True)
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    names = sorted(set(index['weight_map'].values()))
    if any(Path(name).name != name or not name.endswith('.safetensors') for name in names):
        raise ValueError('unsafe shard index')
    expected = {}
    for line in (model / 'SHA256SUMS').read_text().splitlines():
        checksum, name = line.split()
        expected[name.lstrip('*')] = checksum
    weights = {name: sha(model / name) for name in names}
    if any(expected.get(name) != checksum for name, checksum in weights.items()):
        raise ValueError('weight bytes differ from source checksum receipt')
    metadata = {name: sha(model / name) for name in (
        'config.json', 'model.safetensors.index.json', 'tokenizer_config.json',
        'tokenizer.json', 'processor_config.json', 'preprocessor_config.json',
        'chat_template.jinja', 'args.json', 'trainer_state.json')}
    code = args.output / 'code'
    shutil.copytree(ROOT / 'eval', code / 'eval', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, code / 'evaluate_release_candidate.py')
    code_hashes = {str(path.relative_to(code)): sha(path) for path in sorted(code.rglob('*.py'))}
    save(args.output / 'binding.json', {'model': str(model), 'weights_sha256': weights,
        'metadata_sha256': metadata, 'code_sha256': code_hashes, 'driver_sha256': sha(__file__),
        'benchg_sha256': sha(args.benchg_data), 'benchmarks': BENCHMARKS,
        'candidate_label': args.candidate_label,
        'candidate_role': 'unqualified_evaluation_candidate',
        'training_eval_separation': 'not_yet_established_for_this_candidate',
        'publication_eligible': False, 'selection': args.selection_description,
        'protocol': {'id': 'gsa-eval-v7', 'batch_size': 4, 'thinking': False,
                     'max_tokens_override': 0, 'limit': 0, 'seed': 0}, 'at': stamp()})
    env = dict(os.environ, ANVIL_BENCHMARK_ROOT=str(args.benchmark_root),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        WANDB_MODE='disabled', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'NPROC_PER_NODE'):
        env.pop(key, None)
    runner = str(code / 'eval/run_benchmark.py')
    common = [sys.executable, '-B', runner, '--model', str(model), '--batch-size', '4',
              '--device-map', 'cuda:0', '--no-thinking', '--max-tokens', '0', '--seed', '0']

    def worker(number, smoke=False):
        run = args.output / ('smoke' if smoke else f'worker_{number}')
        run.mkdir()
        device, lock = acquire(args, run)
        try:
            save(run / 'status.json', {'state': 'gpu_acquired', 'pid': os.getpid(), 'at': stamp()})
            for directory in ('hf', 'triton', 'torch/kernels'):
                (run / 'cache' / directory).mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='anvil-eval-', dir=args.runtime_parent) as runtime:
                local = dict(env, CUDA_VISIBLE_DEVICES=device, TMPDIR=runtime,
                             XDG_CACHE_HOME=str(run / 'cache'), HF_HOME=str(run / 'cache/hf'),
                             TRITON_CACHE_DIR=str(run / 'cache/triton'))
                argv = common + ['--scratch', str(run / 'assets'), '--out', str(run / 'result.json')]
                if smoke:
                    argv += ['--benchmark', 'cv_bench', '--limit', '4',
                             '--shard-out', str(run / 'predictions.jsonl')]
                else:
                    argv += ['--benchmarks', *BENCHMARKS, '--limit', '0',
                             '--shard', str(number), '--num-shards', str(args.workers),
                             '--shard-dir', str(args.output / 'shards'), '--data', str(args.benchg_data)]
                result = command(run, 'inference', argv, local)
                save(run / 'status.json', {'state': 'complete' if result == 0 else 'failed',
                                           'exit_code': result, 'at': stamp()})
                return result
        finally:
            lock.close()

    if worker(0, smoke=True):
        save(args.output / 'status.json', {'state': 'smoke_failed_no_full_queue_started', 'at': stamp()})
        return 1
    save(args.output / 'status.json', {'state': 'full_evaluation_running', 'pid': os.getpid(), 'at': stamp()})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(worker, range(args.workers)))
    save(args.output / 'status.json', {'state': 'cpu_scoring', 'pid': os.getpid(), 'at': stamp()})
    results = []
    score_codes = []
    for bench in BENCHMARKS:
        run = args.output / bench
        run.mkdir()
        shards = [args.output / 'shards' / bench / f'shard_{i}.jsonl' for i in range(args.workers)]
        if not all(path.is_file() for path in shards):
            results.append({'benchmark': bench, 'state': 'missing_shards', 'score': None})
            continue
        validate_shard_uids(shards)
        argv = common + ['--benchmark', bench, '--limit', '0', '--out', str(run / 'metrics.json'),
            '--scratch', str(run / 'assets'), '--score-shards', *map(str, shards),
            '--dump-predictions', str(run / 'predictions.jsonl'), '--data', str(args.benchg_data)]
        score_code = command(run, 'score', argv, dict(env, CUDA_VISIBLE_DEVICES=''))
        score_codes.append(score_code)
        if score_code or not (run / 'metrics.json').is_file():
            results.append({'benchmark': bench, 'state': 'score_failed', 'score': None})
            continue
        results.append({'benchmark': bench, 'state': 'scored',
                        'metrics': json.loads((run / 'metrics.json').read_text())})
    save(args.output / 'status.json', {'state': 'rechecking_model_bytes', 'pid': os.getpid(), 'at': stamp()})
    unchanged = all(sha(model / name) == checksum for name, checksum in {**metadata, **weights}.items())
    complete = result_is_complete(results, codes, score_codes, unchanged)
    save(args.output / 'summary.json', {'results': results, 'worker_exit_codes': codes,
        'score_exit_codes': score_codes, 'all_benchmarks_scored': complete,
        'model_bytes_unchanged': unchanged, 'wall_seconds': time.monotonic() - started,
        'publication_eligible': False, 'replace_manuscript_results': False,
        'binding_sha256': sha(args.output / 'binding.json'), 'at': stamp()})
    save(args.output / 'status.json', {'state': 'finished_requires_acceptance' if complete
                                      else 'finished_with_failures', 'at': stamp()})
    return int(not complete)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        if CURRENT_OUTPUT is not None:
            save(CURRENT_OUTPUT / 'status.json', {'state': 'failed', 'at': stamp(),
                'error_type': type(error).__name__, 'reason': str(error)})
        raise
