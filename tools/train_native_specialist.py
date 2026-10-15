#!/usr/bin/env python3
"""Step-bounded native SFT from a complete training export, not final MOPD.

Loads every admitted file in the requested scope; never creates a substitute
small dataset. A fixed update budget is NOT a claim of completing an epoch.
The joint control includes QA at its corpus proportion; specialists exclude it.
No held-out tuning, implicit resume, silent row skipping, GPU eviction, or hard
wall-clock limit. Terminal weights require separate matched evaluation before
teacher qualification or publication.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import fcntl
import hashlib
import heapq
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trains.flywheel.native_training import FAMILIES, strict_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def selected_streams(manifest, scope, sampling):
    require(manifest.get('state') == 'all_selected_training_rows_exported'
            and manifest.get('dropped_records') == 0, 'Complete unchanged training export required')
    streams = []
    for name, info in sorted(manifest['files'].items()):
        form = info['form']; family = FAMILIES.get(form, 'QA' if form == 'QA' else None)
        require(family and info['family'] == family and name == f'train_{form}.jsonl'
                and type(info['records']) is int and info['records'] > 0, 'Invalid training export stream')
        if scope == 'joint' or family == scope: streams.append((name, info))
    expected = {f for f, family in FAMILIES.items() if scope == 'joint' or family == scope}
    if scope == 'joint': expected.add('QA')
    require({info['form'] for _, info in streams} == expected, 'Requested scope lacks a native form')
    total = sum(info['records'] for _, info in streams)
    if sampling == 'record':
        probabilities = [info['records'] / total for _, info in streams]
    else:
        families = Counter(info['family'] for _, info in streams if info['family'] != 'QA')
        qa = sum(info['records'] for _, info in streams if info['family'] == 'QA') / total
        probabilities = [qa if info['family'] == 'QA' else (1 - qa) / len(families) / families[info['family']]
                         for _, info in streams]
    require(abs(sum(probabilities) - 1) < 1e-10, 'Invalid stream probabilities')
    return streams, probabilities


def training_command(args, streams, probabilities, *, smoke=False):
    command = [sys.executable, '-B', '-m', 'swift.cli.sft', '--model', str(args.model),
        '--model_type', 'qwen3_5', '--template', 'qwen3_5', '--dataset',
        *[str(args.export / name) for name, _ in streams], '--interleave_prob', *map(str, probabilities),
        '--streaming', 'false', '--lazy_tokenize', 'true', '--stopping_strategy', 'all_exhausted',
        '--dataset_shuffle', 'true', '--split_dataset_ratio', '0', '--remove_unused_columns', 'true',
        '--strict', 'true', '--truncation_strategy', 'delete',
        '--tuner_type', 'lora', '--torch_dtype', 'bfloat16', '--enable_thinking', 'false',
        '--attn_impl', 'sdpa', '--freeze_vit', 'true', '--freeze_aligner', 'true',
        '--lora_rank', '64', '--lora_alpha', '128', '--lora_dropout', '.05', '--target_modules', 'all-linear',
        '--max_steps', '2' if smoke else str(args.steps), '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', str(args.gradient_accumulation), '--seed', str(args.seed),
        '--data_seed', str(args.seed), '--learning_rate', str(args.learning_rate), '--warmup_ratio', '.03',
        '--lr_scheduler_type', 'cosine', '--max_length', str(args.max_length),
        '--gradient_checkpointing', 'true', '--logging_steps', '1' if smoke else '8',
        '--save_steps', '2' if smoke else '128', '--save_total_limit', '1' if smoke else '4',
        '--output_dir', str(args.output / ('smoke' if smoke else 'training')), '--add_version', 'false',
        '--eval_strategy', 'no', '--dataloader_num_workers', '2', '--dataset_num_proc', '4', '--report_to', 'none']
    # Swift maps delete -> template raise; strict=True propagates that error,
    # rather than silently filtering or truncating oversized examples.
    return command


def prepare(args, streams, probabilities):
    require(not args.output.exists(), 'Use a new run directory; resume requires an explicit reviewed checkpoint')
    args.output.mkdir(parents=True)
    args.runtime_parent.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix='anvil-native-', dir=args.runtime_parent))
    env = dict(os.environ)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'NPROC_PER_NODE', 'PYTHONPATH', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN'):
        env.pop(key, None)
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', USE_HF='1', WANDB_MODE='disabled', DS_IGNORE_CUDA_DETECTION='1', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', IMAGE_MAX_TOKEN_NUM=str(args.image_tokens),
        CUDA_VISIBLE_DEVICES='', TMPDIR=str(runtime), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key, folder in [('HF_HOME', 'hf'), ('HF_DATASETS_CACHE', 'datasets'), ('MODELSCOPE_CACHE', 'modelscope'), ('XDG_CACHE_HOME', 'xdg'),
                        ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor')]:
        path = args.output / 'cache' / folder; path.mkdir(parents=True, exist_ok=True); env[key] = str(path)
    model_files = sorted({*args.model.glob('*.json'), *args.model.glob('*.safetensors'),
                          *args.model.glob('*.jinja'), *args.model.glob('*.txt')})
    require(any(path.suffix == '.safetensors' for path in model_files), 'Local safetensors base weights required')
    save(args.output / 'protocol.json', {'scope': args.scope, 'sampling': args.sampling,
        'streams': [{**info, 'file': name, 'probability': prob} for (name, info), prob in zip(streams, probabilities)],
        'steps': args.steps, 'seed': args.seed, 'gradient_accumulation': args.gradient_accumulation,
        'max_length': args.max_length, 'image_tokens': args.image_tokens, 'learning_rate': args.learning_rate,
        'export': str(args.export), 'export_manifest_sha256': sha(args.export / 'manifest.json'),
        'runner_sha256': sha(Path(__file__)), 'model': str(args.model), 'runtime': str(runtime),
        'base_files_sha256': {path.name: sha(path) for path in model_files},
        'environment': {k: importlib.metadata.version(k) for k in
                        ['torch', 'transformers', 'ms-swift', 'peft', 'datasets', 'accelerate']},
        'method': 'source-native SFT preparation; not MOPD or qualified-teacher evidence',
        'complete_epoch_claimed': False, 'heldout_used_for_training_or_selection': False,
        'row_order': 'full map-style dataset shuffle, not a small prefix of a sequential stream',
        'smoke': 'two separate updates, main restarts from base; unchanged full input streams',
        'code_policy': 'runner and admission modules are copied before GPU work'})
    code = args.output / 'code'; code.mkdir()
    import shutil
    shutil.copy2(__file__, code / Path(__file__).name)
    shutil.copy2(Path(__file__).parents[1] / 'trains/flywheel/native_training.py', code / 'native_training.py')
    return env


def preflight(args, streams):
    """Hash every selected stream and encode its first/longest representatives.

    This is a compatibility check, not a false claim of encoding all 5.22M rows.
    All actual training rows subsequently pass Swift's strict encoder.
    """
    import torch
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(str(args.model), load_model=False, model_type='qwen3_5',
                                        torch_dtype=torch.bfloat16, use_hf=True)
    template = get_template(processor, template_type='qwen3_5', max_length=args.max_length,
                            truncation_strategy='raise', enable_thinking=False)
    template.set_mode('train'); report = {}
    for name, info in streams:
        digest = hashlib.sha256(); count = 0; longest = []; first = []
        with (args.export / name).open('rb') as stream:
            for count, raw in enumerate(stream, 1):
                digest.update(raw)
                # Length comparison is a byte-based compatibility sample only.
                item = (len(raw), count, raw)
                if len(longest) < 16: heapq.heappush(longest, item)
                elif item[:2] > longest[0][:2]: heapq.heapreplace(longest, item)
                if count <= 16: first.append(item)
        require(count == info['records'] and digest.hexdigest() == info['sha256'], 'Training file changed: ' + name)
        selected = {number: raw for _, number, raw in first + longest}; lengths = []
        for raw in selected.values():
            row = strict_json(raw)
            encoded = template.encode({'messages': copy.deepcopy(row['messages']), 'images': row['images']})
            require(any(x != -100 for x in encoded['labels']), 'No supervised target tokens')
            lengths.append(len(encoded['input_ids']))
        report[info['form']] = {'rows_hash_checked': count, 'rows_encoded': len(lengths),
                              'sample_max_sequence_length': max(lengths), 'sha256': digest.hexdigest()}
        print(json.dumps({'stage': 'encoding_preflight', 'form': info['form'], **report[info['form']]}), flush=True)
    save(args.output / 'preflight.json', {'streams': report, 'complete_encoding_claimed': False,
        'sampling': 'first 16 and 16 longest serialized rows per form; duplicates removed',
        'model_weights_loaded': False, 'runtime_encoding': 'strict; errors stop rather than drop or replace records'})


def gpu_snapshot(uuid):
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    occupied = {row[0].strip() for row in csv.reader(raw.splitlines()) if row}
    values = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=uuid,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True).strip().split(',')
    require(values[0].strip() == uuid, 'GPU identity changed')
    return {'uuid': uuid, 'compute_pid_present': uuid in occupied,
            'memory_mib': int(values[1]), 'utilization_percent': int(values[2])}


def idle(snapshot):
    return not snapshot['compute_pid_present'] and snapshot['memory_mib'] <= 1024 and snapshot['utilization_percent'] == 0


def acquire_gpu(args):
    uuid = subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    require(uuid.startswith('GPU-') and '\n' not in uuid, 'One physical GPU is required')
    args.lock_root.mkdir(parents=True, exist_ok=True)
    lock = (args.lock_root / (uuid + '.lock')).open('a')
    while True:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            save(args.output / 'status.json', {'stage': 'waiting_for_gpu_lock', 'pid': os.getpid(), 'uuid': uuid})
            time.sleep(20); continue
        snapshot = gpu_snapshot(uuid)
        if idle(snapshot):
            time.sleep(2); second = gpu_snapshot(uuid)
            if idle(second):
                save(args.output / 'gpu_allocation.json', {'checks': [snapshot, second]})
                return lock, uuid
        fcntl.flock(lock, fcntl.LOCK_UN)
        save(args.output / 'status.json', {'stage': 'waiting_for_gpu', 'pid': os.getpid(), **snapshot})
        time.sleep(20)


def run_command(args, name, command, env):
    start = time.monotonic(); save(args.output / (name + '.command.json'), command)
    with (args.output / (name + '.log')).open('x') as log:
        child = subprocess.Popen(command, env=env, cwd=args.output, stdout=log, stderr=subprocess.STDOUT)
        save(args.output / 'status.json', {'stage': name, 'pid': os.getpid(), 'child_pid': child.pid})
        rc = child.wait()
    save(args.output / (name + '.result.json'), {'exit_code': rc, 'seconds': time.monotonic() - start})
    require(rc == 0, f'{name} exited {rc}; see {name}.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export', 'model', 'output', 'lock-root', 'runtime-parent'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scope', choices=['G', 'S', 'A', 'joint'], required=True)
    parser.add_argument('--sampling', choices=['record', 'family-form'], default='record')
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--image-tokens', type=int, default=1536)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    require(args.steps > 0 and args.gradient_accumulation > 0 and args.max_length > args.image_tokens > 0,
            'Invalid step or sequence budgets')
    require(0 < args.learning_rate < 1, 'Invalid learning rate')
    for name in ('export', 'model', 'output', 'lock_root', 'runtime_parent'):
        setattr(args, name, getattr(args, name).resolve())
    manifest = strict_json((args.export / 'manifest.json').read_bytes())
    streams, probabilities = selected_streams(manifest, args.scope, args.sampling)
    env = prepare(args, streams, probabilities)
    # Encode on CPU before acquiring a GPU; this process never loads model weights.
    os.environ.update(env)
    try:
        preflight(args, streams)
        if args.preflight_only:
            save(args.output / 'status.json', {'stage': 'preflight_complete_no_training', 'pid': os.getpid()}); return
        lock, uuid = acquire_gpu(args)
        with lock:
            env['CUDA_VISIBLE_DEVICES'] = uuid
            run_command(args, 'smoke', training_command(args, streams, probabilities, smoke=True), env)
            run_command(args, 'training', training_command(args, streams, probabilities), env)
        final = args.output / 'training' / f'checkpoint-{args.steps}' / 'adapter_model.safetensors'
        require(final.is_file(), 'Terminal checkpoint absent')
        state = strict_json((final.parent / 'trainer_state.json').read_bytes())
        require(state.get('global_step') == args.steps, 'Training did not reach its declared update budget')
        save(args.output / 'status.json', {'stage': 'sft_complete_evaluation_pending', 'pid': os.getpid(),
            'terminal_adapter': str(final), 'adapter_sha256': sha(final), 'teacher_qualified': False,
            'mopd_complete': False, 'public_release_eligible': False})
    except BaseException as error:
        save(args.output / 'failure.json', {'stage': 'failed', 'type': type(error).__name__, 'reason': str(error)})
        raise


if __name__ == '__main__':
    main()
