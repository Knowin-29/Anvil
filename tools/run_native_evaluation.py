#!/usr/bin/env python3
"""Full source-native inference, with terminal-checkpoint queues and explicit resume.

Every selected row in the scope/shard is evaluated. Shards partition each form
by its frozen file ordinal; they are not smaller replacement benchmarks. Runtime
errors stop with saved predictions, while malformed model answers remain in the
metric denominator. No checkpoint selection, hard timeout, or implicit restart.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import native_metrics
from tools.response_text import POLICY as RESPONSE_POLICY, RAW_POLICY, SWIFT_NON_THINKING_PREFIX, generated_text
from tools.train_native_specialist import acquire_gpu as acquire_one_gpu, gpu_snapshot, idle, save, sha
from trains.flywheel.native_training import FAMILIES, canonical, digest, require, strict_json, validate_target

VERSION = 'anvil-native-evaluation-run-v4'
TOKEN_BUDGETS = {'bbox': 2048, 'detection': 8192, 'mask': 16384, 'point': 16384,
    'direction': 512, 'metric_comparison': 512, 'relation_graph': 8192, 'depth_order': 2048,
    'affordance': 512, 'dof': 2048, 'action_cot': 2048}
CODE = ('tools/run_native_evaluation.py', 'tools/train_native_specialist.py',
        'tools/map_prism_native.py', 'tools/arxiv_source_scale.py',
        'trains/flywheel/native_training.py', 'eval/native_metrics.py', 'tools/response_text.py')


def runtime_environment():
    """Optional kernels change numerical execution and must be bound on resume."""
    result = {name: importlib.metadata.version(name) for name in
              ('torch', 'transformers', 'ms-swift', 'peft', 'numpy', 'scipy')}
    for name in ('triton', 'flash-linear-attention', 'fla-core', 'causal-conv1d', 'pycocotools'):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = None
    return result


def verify_metric_dependencies(files):
    """Fail before GPU allocation, not hours later on the first valid mask."""
    if not {info['form'] for _, info in files}.intersection({'mask', 'point'}): return
    try:
        mask_api = importlib.import_module('pycocotools.mask')
    except (ImportError, OSError) as error:
        raise ValueError('Mask/point evaluation requires importable pycocotools before inference') from error
    require(all(callable(getattr(mask_api, name, None)) for name in ('frPyObjects', 'merge', 'decode')),
            'Incomplete pycocotools mask API')


def verify_fast_kernels(module):
    """Do not silently fall back when the caller explicitly requests acceleration."""
    names = ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
             'fused_recurrent_gated_delta_rule', 'FusedRMSNormGated')
    require(getattr(module, 'is_fast_path_available', False)
            and all(callable(getattr(module, name, None)) for name in names),
            'Requested Qwen3.5 fast kernels are unavailable; no fallback or inference was started')


def acquire_gpu(args):
    """A baseline may use any explicitly authorized card as it becomes idle."""
    if ',' not in args.gpu: return acquire_one_gpu(args)
    values = subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-gpu=uuid',
                                     '--format=csv,noheader'], text=True).splitlines()
    uuids = [value.strip() for value in values]
    require(len(uuids) == len(set(uuids)) == len(args.gpu.split(','))
            and all(value.startswith('GPU-') for value in uuids), 'Invalid GPU pool')
    args.lock_root.mkdir(parents=True, exist_ok=True)
    while True:
        observations = []
        for uuid in uuids:
            lock = (args.lock_root / (uuid + '.lock')).open('a')
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close(); observations.append({'uuid': uuid, 'locked': True}); continue
            try:
                first = gpu_snapshot(uuid); observations.append(first)
                if idle(first):
                    time.sleep(2); second = gpu_snapshot(uuid)
                    if idle(second):
                        save(args.output / 'gpu_allocation.json', {'pool': uuids, 'checks': [first, second]})
                        return lock, uuid
            except BaseException:
                lock.close(); raise
            lock.close()
        save(args.output / 'status.json', {'stage': 'waiting_for_gpu_pool', 'pid': os.getpid(), 'checks': observations})
        time.sleep(20)


def selected_files(manifest, scope, shard, shards):
    require(type(shards) is int and type(shard) is int and 0 <= shard < shards, 'Invalid shard partition')
    require(manifest.get('state') == 'all_selected_native_evaluation_rows_exported'
            and manifest.get('dropped_records') == 0 and manifest.get('targets_repaired') == 0,
            'Complete unchanged evaluation export required')
    require(set(manifest['by_form']) == set(FAMILIES), 'Evaluation export lacks native forms')
    files = []
    for name, info in manifest['files'].items():
        form = info['form']
        require(form in FAMILIES and name == f'eval_{form}.jsonl' and info['family'] == FAMILIES[form]
                and type(info['records']) is int and info['records'] > 0
                and info['records'] == manifest['by_form'][form], 'Invalid evaluation file')
        if scope == 'joint' or info['family'] == scope:
            files.append((name, {**info, 'shard_records': max(0, (info['records'] - shard + shards - 1) // shards)}))
    expected = {f for f in FAMILIES if scope == 'joint' or FAMILIES[f] == scope}
    require({i['form'] for _, i in files} == expected, 'Requested scope missing a form')
    require(sum(i['records'] for i in manifest['files'].values()) == manifest['records'], 'Export total mismatch')
    return sorted(files, key=lambda item: (item[1]['records'], item[0]))


def validate_row(row, form, selection_sha):
    meta = row['evaluation_meta']
    require(row.get('schema') == 'anvil-source-native-evaluation-v1' and row.get('native_form') == form
            and row.get('family') == FAMILIES[form] and row.get('training_approved') is False
            and meta.get('split') == 'benchmark' and meta.get('selection_sha256') == selection_sha,
            'Evaluation row split/form binding mismatch')
    require(isinstance(row.get('id'), str) and bool(row['id']), 'Missing evaluation identity')
    require(meta['evaluation_content_sha256'] == digest({k: v for k, v in row.items() if k != 'evaluation_meta'}),
            'Evaluation content changed')
    require(len(row['messages']) == 2 and [m['role'] for m in row['messages']] == ['user', 'assistant']
            and len(row['images']) == 1, 'Expected one native image prompt/reference pair')
    validate_target(form, row['native_variant'], strict_json(row['messages'][1]['content']), row['messages'][0]['content'])


def shard_rows(path, form, selection_sha, shard, shards):
    with path.open('rb') as stream:
        for ordinal, raw in enumerate(stream):
            if ordinal % shards != shard: continue
            row = strict_json(raw); validate_row(row, form, selection_sha)
            yield ordinal, row


def process_identity(pid):
    """PID alone is not a live job identity (the namespace can wrap)."""
    try:
        root = Path('/proc') / str(pid)
        stat = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
        if stat[0] == 'Z': return None
        return {'pid': int(pid), 'start_ticks': stat[19], 'cmdline_sha256': sha(root / 'cmdline')}
    except (OSError, ValueError): return None


def terminal_adapter(root, protocol):
    status = strict_json((root / 'status.json').read_bytes())
    if status.get('stage') != 'sft_complete_evaluation_pending': return None
    require(strict_json((root / 'training.result.json').read_bytes())['exit_code'] == 0, 'Training result failed')
    checkpoint = root / 'training' / f"checkpoint-{protocol['steps']}"
    adapter = checkpoint / 'adapter_model.safetensors'
    require(Path(status['terminal_adapter']).resolve() == adapter.resolve()
            and sha(adapter) == status['adapter_sha256'], 'Terminal adapter binding mismatch')
    require(strict_json((checkpoint / 'trainer_state.json').read_bytes())['global_step'] == protocol['steps'],
            'Not the declared terminal checkpoint')
    config = strict_json((checkpoint / 'adapter_config.json').read_bytes())
    require(config.get('peft_type') == 'LORA'
            and config.get('base_model_name_or_path') == protocol['model'], 'Expected bound base/LoRA adapter')
    require((checkpoint / 'args.json').is_file(), 'Adapter inference configuration absent')
    return {'path': str(checkpoint), 'files_sha256': {p.name: sha(p) for p in
        sorted({adapter, *checkpoint.glob('*.json')})}}


def wait_for_training(args, protocol, expected_identity):
    while True:
        require(sha(args.training_run / 'protocol.json') == args.training_protocol_sha, 'Training protocol changed')
        require(not (args.training_run / 'failure.json').exists(), 'Training failed; inspect its saved artifacts')
        adapter = terminal_adapter(args.training_run, protocol)
        if adapter: return adapter
        require(expected_identity is not None and process_identity(expected_identity['pid']) == expected_identity,
                'Training supervisor no longer live and no terminal checkpoint receipt exists')
        save(args.output / 'status.json', {'stage': 'waiting_for_terminal_training', 'pid': os.getpid(),
             'training_process': expected_identity, 'training_run': str(args.training_run)})
        time.sleep(20)


def checked_image(row, cache):
    path = row['images'][0]; expected = row['evaluation_meta']['image_sha256']
    if path not in cache: cache[path] = sha(path)
    require(cache[path] == expected, 'Evaluation image bytes changed')


def prediction_item(row, ordinal, response, finish_reason, usage, run_sha, region=None, *, sdk_response_prefix=None):
    require(isinstance(response, str) and finish_reason in {'stop', 'length'}, 'Unexpected engine response')
    body = generated_text(response, sdk_response_prefix) if sdk_response_prefix is not None else response
    item = {'id': row['id'], 'ordinal': ordinal, 'native_form': row['native_form'],
        'source': row['evaluation_meta']['source'], 'evaluation_content_sha256': row['evaluation_meta']['evaluation_content_sha256'],
        'protocol_sha256': run_sha, 'response': response, 'finish_reason': finish_reason, 'usage': usage,
        'metrics': native_metrics.score_row(row, body, finish_reason=finish_reason, point_region=region)}
    if sdk_response_prefix is not None:
        item.update(sdk_response_prefix=sdk_response_prefix, model_response_sha256=hashlib.sha256(body.encode()).hexdigest())
    return item


def validate_saved(item, row, ordinal, run_sha, region=None):
    require(item.get('id') == row['id'] and item.get('ordinal') == ordinal
            and item.get('protocol_sha256') == run_sha
            and item.get('evaluation_content_sha256') == row['evaluation_meta']['evaluation_content_sha256'],
            'Saved prediction is not the exact bound prefix of this shard')
    expected = prediction_item(row, ordinal, item['response'], item['finish_reason'], item['usage'], run_sha, region,
                               sdk_response_prefix=item.get('sdk_response_prefix'))
    require(item == expected, 'Saved prediction fields or metrics changed')


def run_form(args, name, info, manifest, run_sha, engine, region_loader, image_cache):
    from swift.infer_engine import InferRequest, RequestConfig
    form = info['form']; path = args.output / ('predictions_' + form + '.jsonl')
    rows = iter(shard_rows(args.export / name, form, manifest['selection_sha256'], args.shard, args.shards))
    def region(row):
        return region_loader(row) if form == 'point' and row['evaluation_meta']['source'] == 'coco_train2017' else None
    completed = 0
    if path.exists():
        require(args.resume, 'Existing predictions require explicit --resume')
        with path.open('rb') as stream:
            for raw in stream:
                require(raw.endswith(b'\n'), 'Incomplete prediction tail; retain it for reviewed recovery')
                pair = next(rows, None); require(pair is not None, 'Too many saved predictions')
                ordinal, row = pair; checked_image(row, image_cache)
                validate_saved(strict_json(raw), row, ordinal, run_sha, region(row)); completed += 1
    batch_size = 1 if TOKEN_BUDGETS[form] > 4096 else args.batch_size
    config = RequestConfig(max_tokens=TOKEN_BUDGETS[form], temperature=0., seed=17)
    with path.open('a' if args.resume else 'x', encoding='utf-8') as out:
        while batch := list(itertools.islice(rows, batch_size)):
            for _, row in batch: checked_image(row, image_cache)
            requests = [InferRequest(messages=copy.deepcopy(row['messages'][:1]), images=row['images']) for _, row in batch]
            responses = engine.infer(requests, config, use_tqdm=False)
            require(len(responses) == len(batch), 'Engine omitted a prediction')
            for (ordinal, row), response in zip(batch, responses):
                if isinstance(response, Exception): raise response
                require(len(response.choices) == 1, 'Expected one greedy response')
                choice = response.choices[0]
                item = prediction_item(row, ordinal, choice.message.content or '', choice.finish_reason,
                    asdict(response.usage) if response.usage else None, run_sha, region(row),
                    sdk_response_prefix=engine.template.response_prefix)
                out.write(canonical(item) + '\n'); completed += 1
            out.flush(); os.fsync(out.fileno())
            save(args.output / 'progress.json', {'stage': 'inference', 'form': form, 'completed': completed,
                'form_shard_total': info['shard_records'], 'protocol_sha256': run_sha, 'pid': os.getpid()})
            if completed % 100 < batch_size: print(json.dumps({'form': form, 'completed': completed, 'total': info['shard_records']}), flush=True)
    require(completed == info['shard_records'], 'Not all selected rows were evaluated')
    save(args.output / ('complete_' + form + '.json'), {'records': completed, 'prediction_sha256': sha(path),
        'dataset_sha256': info['sha256'], 'protocol_sha256': run_sha})


def validate_pretrained_binding(model, binding):
    """A public checkpoint has no Anvil training run; keep its identity separate."""
    require(binding.get('schema') == 'anvil-pretrained-baseline-binding-v1'
            and binding.get('state') == 'local_files_match_pinned_hub_revision'
            and binding.get('model') == str(model), 'Verified pretrained model binding required')
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', binding.get('repository', ''))
            and re.fullmatch(r'[a-f0-9]{40}', binding.get('revision', '')), 'Immutable public model identity required')
    require(binding.get('model_type') == 'qwen3_vl', 'Unsupported pretrained architecture')
    names = {p.name for pattern in ('*.json', '*.safetensors', '*.jinja', '*.txt')
             for p in model.glob(pattern) if not p.name.startswith('.')}
    hashes = binding['base_files_sha256']
    require(set(hashes) == names and all(re.fullmatch(r'[a-f0-9]{64}', v) for v in hashes.values()),
            'Pretrained binding must cover every local inference file')
    require(all(Path(n).name == n for n in names), 'Unsafe pretrained file identity')
    config = strict_json((model/'config.json').read_bytes())
    require(config.get('model_type') == 'qwen3_vl'
            and config.get('architectures') == ['Qwen3VLForConditionalGeneration']
            and not config.get('auto_map'), 'Unexpected or custom pretrained architecture')
    index = strict_json((model/'model.safetensors.index.json').read_bytes())
    require(set(index['weight_map'].values()) == {n for n in names if n.endswith('.safetensors')}
            and bool(index['weight_map']), 'Incomplete pretrained weight index')
    return binding


def pretrained_binding(model, binding_path):
    return validate_pretrained_binding(model, strict_json(binding_path.read_bytes()))


def bind_public_model(model, repository, revision, output):
    """Read public Hub metadata only; do not download or rewrite model files."""
    require(not output.exists(), 'Use a new pretrained binding path')
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository)
            and re.fullmatch(r'[a-f0-9]{40}', revision), 'Immutable public model identity required')
    from huggingface_hub import HfApi
    info = HfApi(token=False).model_info(repository, revision=revision, files_metadata=True)
    require(info.sha == revision, 'Hub revision differs from the requested commit')
    remote = {entry.rfilename: entry for entry in info.siblings}
    paths = sorted(p for pattern in ('*.json', '*.safetensors', '*.jinja', '*.txt')
                   for p in model.glob(pattern) if not p.name.startswith('.'))
    hashes = {}; upstream = {}
    for path in paths:
        require(path.name in remote, 'Local inference file absent from pinned Hub revision: ' + path.name)
        entry = remote[path.name]
        require(path.stat().st_size == entry.size, 'Pretrained file size differs: ' + path.name)
        checksum = sha(path)
        if entry.lfs:
            require(checksum == entry.lfs.sha256, 'Pretrained LFS bytes differ: ' + path.name)
        else:
            raw = path.read_bytes()
            require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == entry.blob_id,
                    'Pretrained Git bytes differ: ' + path.name)
        hashes[path.name] = checksum
        upstream[path.name] = {'size': entry.size, 'git_blob': entry.blob_id,
                               'lfs_sha256': entry.lfs.sha256 if entry.lfs else None}
    binding = {'schema': 'anvil-pretrained-baseline-binding-v1',
        'state': 'local_files_match_pinned_hub_revision', 'model': str(model),
        'repository': repository, 'revision': revision, 'model_type': 'qwen3_vl',
        'base_files_sha256': hashes, 'upstream_files': upstream, 'local_weights_changed': False,
        'full_evaluation_complete': False, 'model_calls': 0, 'training_claimed': False}
    validate_pretrained_binding(model, binding)
    output.parent.mkdir(parents=True, exist_ok=True); save(output, binding)
    return binding


def evaluate(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # No model allocation before the cooperative GPU lock.
    manifest_path = args.export / 'manifest.json'; manifest = strict_json(manifest_path.read_bytes())
    files = selected_files(manifest, args.scope, args.shard, args.shards)
    verify_metric_dependencies(files)
    baseline_path = getattr(args, 'pretrained_binding', None)
    model_type = 'qwen3_vl' if baseline_path else 'qwen3_5'
    response_policy = RAW_POLICY if baseline_path else RESPONSE_POLICY
    if baseline_path:
        require(not args.training_run and not args.base_protocol, 'Pretrained baseline cannot claim Anvil training')
        require(not getattr(args, 'require_fast_kernels', False), 'Qwen3.5 kernels do not apply to this baseline')
        protocol = pretrained_binding(args.model, baseline_path); args.training_protocol_sha = None
    else:
        training_path = args.training_run / 'protocol.json' if args.training_run else args.base_protocol
        protocol = strict_json(training_path.read_bytes()); args.training_protocol_sha = sha(training_path)
        require(str(args.model) == protocol['model'], 'Base model path differs from training binding')
        training_manifest = Path(protocol['export']) / 'manifest.json'
        require(sha(training_manifest) == protocol['export_manifest_sha256'], 'Training export binding changed')
        training_data = strict_json(training_manifest.read_bytes())
        require(training_data['selection_sha256'] == manifest['selection_sha256']
                and training_data['state'] == 'all_selected_training_rows_exported', 'Training/evaluation use different frozen selections')
        if args.training_run:
            require(protocol['scope'] in {args.scope, 'joint'}, 'Specialist evaluated outside its declared scope')
    request = {'schema': VERSION, 'export_manifest_sha256': sha(manifest_path), 'scope': args.scope,
        'files': dict(files), 'shard': args.shard, 'shards': args.shards, 'model': str(args.model),
        'training_protocol_sha256': args.training_protocol_sha, 'training_run': str(args.training_run) if args.training_run else None,
        'pretrained_binding_sha256': sha(baseline_path) if baseline_path else None,
        'pretrained_repository': protocol['repository'] if baseline_path else None,
        'pretrained_revision': protocol['revision'] if baseline_path else None,
        'model_type': model_type, 'template_type': model_type,
        'base_files_sha256': protocol['base_files_sha256'], 'token_budgets': TOKEN_BUDGETS,
        'batch_size': args.batch_size, 'long_form_batch_size': 1, 'image_tokens': 1536, 'context_length': 32768,
        'seed': 17, 'decoding': 'greedy', 'thinking': False, 'dtype': 'bfloat16', 'attention': 'sdpa',
        'require_fast_kernels': bool(getattr(args, 'require_fast_kernels', False)),
        'response_text_policy': response_policy,
        'metrics_version': native_metrics.VERSION, 'environment': runtime_environment(),
        'code_sha256': {name: sha(Path(__file__).parents[1] / name) for name in CODE},
        'row_subset': 'all rows in declared scope; ordinal modulo shard count partitions without sampling',
        'public_release_eligible': False, 'official_leaderboard_claimed': False}
    if args.resume:
        require(strict_json((args.output / 'request.json').read_bytes()) == request, 'Resume request differs from frozen run')
    else:
        require(not (args.output / 'request.json').exists(), 'Existing request requires explicit --resume')
        args.output.mkdir(parents=True, exist_ok=True); save(args.output / 'request.json', request)
    with (args.output / 'run.lock').open('a') as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(args.output / 'process.json', process_identity(os.getpid()))
        try:
            expected = None
            if args.training_run:
                state = strict_json((args.training_run / 'status.json').read_bytes())
                expected = process_identity(state['pid'])
                if expected is not None:
                    command = (Path('/proc') / str(state['pid']) / 'cmdline').read_bytes().split(b'\0')
                    require(str(args.training_run).encode() in command
                            and any(p.endswith(b'/train_native_specialist.py') for p in command),
                            'Training PID belongs to a different process')
            adapter = wait_for_training(args, protocol, expected) if args.training_run else None
            for name, expected_sha in protocol['base_files_sha256'].items():
                require(sha(args.model / name) == expected_sha, 'Base file changed: ' + name)
            for name, info in files:
                require(sha(args.export / name) == info['sha256'], 'Evaluation file changed: ' + name)
                with (args.export / name).open('rb') as stream: count = sum(1 for _ in stream)
                require(count == info['records'], 'Evaluation count mismatch')
            frozen = {**request, 'adapter': adapter}
            if args.resume and (args.output / 'protocol.json').exists():
                require(strict_json((args.output / 'protocol.json').read_bytes()) == frozen, 'Resume checkpoint changed')
            else:
                require(not list(args.output.glob('predictions_*.jsonl')), 'Predictions exist without a bound protocol')
                save(args.output / 'protocol.json', frozen)
            run_sha = sha(args.output / 'protocol.json')
            region_loader = (native_metrics.CocoPointReference(manifest) if any(i['form'] == 'point' for _, i in files)
                             and any(i['source'] == 'coco_train2017' for i in manifest['source_inputs']) else None)
            lock, uuid = acquire_gpu(args)
            with lock:
                save(args.output / 'status.json', {'stage': 'loading_model', 'pid': os.getpid(), 'uuid': uuid})
                os.environ['CUDA_VISIBLE_DEVICES'] = uuid
                import torch
                if request['require_fast_kernels']:
                    from transformers.models.qwen3_5 import modeling_qwen3_5
                    verify_fast_kernels(modeling_qwen3_5)
                from swift.infer_engine import TransformersEngine
                # Construct the same explicit inference template for base and all adapters;
                # Swift otherwise restores differing training lengths from adapter args.json.
                from swift.model import get_model_processor
                from swift.template import get_template
                _, processor = get_model_processor(str(args.model), load_model=False, model_type=model_type, use_hf=True)
                template = get_template(processor, template_type=model_type, max_length=32768,
                    truncation_strategy='raise', enable_thinking=False)
                require(template.response_prefix == response_policy['sdk_response_prefix'], 'SDK template scaffold differs from protocol')
                engine = TransformersEngine(str(args.model), adapters=[adapter['path']] if adapter else [],
                    model_type=model_type, template_type=model_type, torch_dtype=torch.bfloat16,
                    device_map='cuda:0', attn_impl='sdpa', max_batch_size=args.batch_size, use_hf=True)
                engine.template = template; engine.strict = True; engine.model.eval()
                save(args.output / 'status.json', {'stage': 'inference', 'pid': os.getpid(), 'uuid': uuid,
                    'scope': args.scope, 'shard': args.shard, 'shards': args.shards})
                image_cache = {}
                for name, info in files: run_form(args, name, info, manifest, run_sha, engine, region_loader, image_cache)
                if region_loader: region_loader.close()
                for path, expected_sha in image_cache.items(): require(sha(path) == expected_sha, 'Image changed during inference')
            require(sha(manifest_path) == request['export_manifest_sha256'], 'Export manifest changed during inference')
            if baseline_path: require(sha(baseline_path) == request['pretrained_binding_sha256'], 'Pretrained binding changed during inference')
            for name, info in files: require(sha(args.export / name) == info['sha256'], 'Evaluation file changed during inference')
            for name, expected_sha in request['base_files_sha256'].items(): require(sha(args.model / name) == expected_sha, 'Base changed during inference')
            for name, expected_sha in request['code_sha256'].items(): require(sha(Path(__file__).parents[1] / name) == expected_sha, 'Frozen code changed during inference')
            if adapter:
                for name, expected_sha in adapter['files_sha256'].items(): require(sha(Path(adapter['path']) / name) == expected_sha, 'Adapter changed during inference')
            def scored():
                for _, info in files:
                    with (args.output / ('predictions_' + info['form'] + '.jsonl')).open(encoding='utf-8') as stream:
                        yield from (strict_json(raw) for raw in stream)
            metrics = native_metrics.summarize(scored())
            require(metrics['n'] == sum(info['shard_records'] for _, info in files), 'Metric denominator mismatch')
            save(args.output / 'metrics.json', {**metrics, 'state': 'complete_scope_shard', 'run_protocol_sha256': run_sha,
                'scope': args.scope, 'shard': args.shard, 'shards': args.shards, 'adapter': adapter,
                'export_manifest_sha256': request['export_manifest_sha256'],
                'prediction_files': {info['form']: sha(args.output / ('predictions_' + info['form'] + '.jsonl')) for _, info in files},
                'teacher_qualified': False, 'mopd_complete': False, 'public_release_eligible': False})
            save(args.output / 'status.json', {'stage': 'complete_scope_shard', 'pid': os.getpid(), 'records': metrics['n']})
        except BaseException as error:
            save(args.output / 'failure.json', {'type': type(error).__name__, 'reason': str(error),
                'partial_predictions_preserved': True, 'restart_policy': 'inspect live process, then explicit --resume only'})
            raise


def launch(args):
    """Freeze code before waiting for training; later source edits cannot change this queue."""
    if not args.worker:
        if not args.resume: require(not args.output.exists(), 'Existing run requires explicit --resume')
        snapshot = args.output / 'code'
        if not args.resume:
            snapshot.mkdir(parents=True)
            for name in CODE:
                target = snapshot / name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(__file__).parents[1] / name, target)
            # Regular packages prevent an unrelated checkout on sys.path from
            # overriding these frozen namespace directories.
            for directory in ('tools', 'eval', 'trains', 'trains/flywheel'):
                (snapshot / directory / '__init__.py').touch()
        command = [sys.executable, '-B', str(snapshot / 'tools/run_native_evaluation.py'), *sys.argv[1:], '--worker']
        # The parent creates only the immutable snapshot; worker owns request/protocol files.
        env = dict(os.environ)
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                    'NPROC_PER_NODE', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN'):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
            PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', USE_HF='1',
            IMAGE_MAX_TOKEN_NUM='1536', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
            DS_IGNORE_CUDA_DETECTION='1', WANDB_MODE='disabled', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        for key, folder in [('TMPDIR','runtime'), ('HF_HOME','cache/hf'), ('HF_DATASETS_CACHE','cache/datasets'),
            ('MODELSCOPE_CACHE','cache/modelscope'), ('XDG_CACHE_HOME','cache/xdg'),
            ('TRITON_CACHE_DIR','cache/triton'), ('TORCHINDUCTOR_CACHE_DIR','cache/torchinductor')]:
            path = args.output / folder; path.mkdir(parents=True, exist_ok=True); env[key] = str(path)
        raise SystemExit(subprocess.call(command, env=env))
    evaluate(args)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'bind-pretrained':
        parser = argparse.ArgumentParser(description='Bind existing Qwen3-VL baseline files to a public Hub commit; no inference')
        for name in ('model', 'output'): parser.add_argument('--'+name, type=Path, required=True)
        for name in ('repository', 'revision'): parser.add_argument('--'+name, required=True)
        args = parser.parse_args(sys.argv[2:])
        result = bind_public_model(args.model.resolve(), args.repository, args.revision, args.output.resolve())
        print(json.dumps({'state': result['state'], 'files': len(result['base_files_sha256']), 'model_calls': 0}))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export', 'model', 'output', 'lock-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--training-run', type=Path)
    group.add_argument('--base-protocol', type=Path)
    group.add_argument('--pretrained-binding', type=Path, help='Verified immutable public checkpoint; no Anvil training claim')
    parser.add_argument('--scope', choices=['G', 'S', 'A', 'joint'], required=True)
    parser.add_argument('--gpu', required=True, help='One GPU or comma-separated authorized idle-GPU pool')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--require-fast-kernels', action='store_true',
                        help='Require installed Qwen3.5 fused operators; fail instead of silently falling back')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.batch_size > 0 and 0 <= args.shard < args.shards, 'Invalid batch or shard settings')
    for name in ('export', 'model', 'output', 'lock_root', 'training_run', 'base_protocol', 'pretrained_binding'):
        value = getattr(args, name)
        if value is not None: setattr(args, name, value.resolve())
    launch(args)


if __name__ == '__main__': main()
