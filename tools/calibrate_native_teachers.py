#!/usr/bin/env python3
"""Training-side reference NLL, not held-out accuracy or teacher qualification.

Select a deterministic source/form-stratified panel from the COMPLETE training
export, then compare terminal specialists with joint SFT on exactly those rows.
No test input, autoregressive generation, silent truncation or automatic resume.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_native_evaluation import CODE as EVAL_CODE, acquire_gpu, process_identity, terminal_adapter
from tools.train_native_specialist import save, selected_streams, sha
from trains.flywheel.native_training import FAMILIES, canonical, digest, require, strict_json, validate_training_record

VERSION = 'anvil-training-reference-calibration-v1'
CODE = (*EVAL_CODE, 'tools/calibrate_native_teachers.py')


def group(row):
    return (row['gsa_family'], row['gsa_form'], row['gsa_meta']['binding']['source'])


def rank(row, seed):
    return int(digest([VERSION, seed, *group(row), row['canonical_uid']]), 16)


def retain(heap, row, raw, ordinal, size, seed):
    item = (-rank(row, seed), -ordinal, raw)
    if len(heap) < size: heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]: heapq.heapreplace(heap, item)


def select(args):
    require(args.per_group > 0, 'Positive panel size required')
    require(not args.output.exists(), 'Selection requires a new output directory')
    args.output.mkdir(parents=True)
    code_hashes = {name: sha(Path(__file__).parents[1] / name) for name in CODE}
    for name in CODE:
        target = args.output / 'code' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parents[1] / name, target)
    source = args.export / 'manifest.json'
    manifest_sha = sha(source); manifest = strict_json(source.read_bytes())
    streams, _ = selected_streams(manifest, 'joint', 'record')
    require(sum(info['records'] for _, info in streams) == manifest['records'], 'Export total mismatch')
    counts = Counter(); heaps = defaultdict(list)
    for name, info in streams:
        checksum = hashlib.sha256(); number = 0
        with (args.export / name).open('rb') as stream:
            for number, raw in enumerate(stream, 1):
                checksum.update(raw)
                require(raw.endswith(b'\n'), 'Incomplete training row')
                if info['family'] == 'QA': continue  # Replay only, never a spatial teacher.
                row = strict_json(raw)
                require(isinstance(row, dict) and isinstance(row.get('gsa_meta'), dict)
                        and isinstance(row['gsa_meta'].get('binding'), dict), 'Malformed training row')
                binding = row['gsa_meta']['binding']
                require(row.get('gsa_form') == info['form'] and row.get('gsa_family') == info['family']
                        and binding.get('selection_sha256') == manifest['selection_sha256']
                        and binding.get('split') == 'train', 'Nontraining or mismatched input')
                key = group(row); counts[key] += 1
                retain(heaps[key], row, raw, number, args.per_group, args.seed)
        require(number == info['records'] and checksum.hexdigest() == info['sha256'], 'Changed training file: ' + name)
        save(args.output / 'progress.json', {'stage': 'selecting', 'completed_file': name})
    require(sha(source) == manifest_sha, 'Training manifest changed')
    ids = set(); image_hashes = {}; selected = Counter()
    with (args.output / 'panel.jsonl').open('wb') as stream:
        for key in sorted(heaps):
            for _, _, raw in sorted(heaps[key], reverse=True):
                row = strict_json(raw); validate_training_record(row)
                identity = (row['gsa_form'], row['canonical_uid'])
                require(identity not in ids, 'Repeated panel identity'); ids.add(identity)
                image = row['images'][0]; expected = row['gsa_meta']['binding']['image_sha256']
                if image not in image_hashes: image_hashes[image] = sha(image)
                require(image_hashes[image] == expected, 'Panel image changed')
                selected[key] += 1; stream.write(raw)
        stream.flush(); os.fsync(stream.fileno())
    require(all(sha(args.output / 'code' / name) == checksum
                and sha(Path(__file__).parents[1] / name) == checksum for name, checksum in code_hashes.items()),
            'Selection code changed during the scan')
    save(args.output / 'manifest.json', {'schema': VERSION, 'split': 'train', 'seed': args.seed,
        'per_source_form': args.per_group, 'records': len(ids), 'panel_sha256': sha(args.output / 'panel.jsonl'),
        'training_export': str(args.export), 'training_export_manifest_sha256': manifest_sha,
        'training_files': manifest['files'], 'complete_export_records': manifest['records'],
        'qa_records_excluded': sum(i['records'] for _, i in streams if i['family'] == 'QA'),
        'selection': 'lowest SHA256(seed, family, form, source, canonical_uid) within each source/form',
        'groups': [{'family': k[0], 'form': k[1], 'source': k[2], 'population': counts[k], 'selected': selected[k]}
                   for k in sorted(counts)],
        'code_sha256': code_hashes,
        'heldout': False, 'training_rows_may_have_been_seen_by_models': True, 'teacher_qualified': False})


def load_panel(path):
    manifest = strict_json((path / 'manifest.json').read_bytes())
    require(manifest['schema'] == VERSION and manifest['split'] == 'train' and manifest['heldout'] is False,
            'Only training-side calibration is supported')
    require(sha(path / 'panel.jsonl') == manifest['panel_sha256'], 'Panel bytes changed')
    rows = [strict_json(raw) for raw in (path / 'panel.jsonl').read_bytes().splitlines()]
    require(len(rows) == manifest['records'] and len({(r['gsa_form'], r['canonical_uid']) for r in rows}) == len(rows),
            'Panel size/identity mismatch')
    for row in rows: validate_training_record(row)
    expected = {(g['family'], g['form'], g['source']): g['selected'] for g in manifest['groups']}
    require(Counter(map(group, rows)) == expected and {r['gsa_form'] for r in rows} == set(FAMILIES),
            'Panel group/form coverage changed')
    return manifest, rows


def response_positions(labels, attention):
    import torch
    require(labels.ndim == 2 and labels.shape[0] == 1 and labels.shape == attention.shape,
            'Calibration requires an unpadded single-row label layout')
    positions = torch.where((labels[0, 1:] != -100) & attention[0, 1:].bool())[0]
    require(positions.numel() > 0, 'No supervised response tokens')
    return positions, labels[0, 1:][positions]


def token_scores(logits, target):
    import torch
    require(logits.ndim == 3 and logits.shape[:2] == (1, target.numel()), 'Next-token logits are misaligned')
    total = 0.; correct = 0
    # Do not upcast a potentially 16K x vocabulary tensor all at once.
    for start in range(0, target.numel(), 128):
        block = logits[0, start:start + 128].float(); labels = target[start:start + 128]
        total += float(torch.nn.functional.cross_entropy(block, labels, reduction='sum'))
        correct += int((block.argmax(-1) == labels).sum())
    require(math.isfinite(total) and total >= 0, 'Nonfinite NLL')
    return {'nll_sum': total, 'tokens': int(target.numel()), 'correct_tokens': correct}


def check_score(item, row, ordinal, protocol_sha):
    require(item['ordinal'] == ordinal and item['row_sha256'] == digest(row)
            and item['protocol_sha256'] == protocol_sha and item['group'] == list(group(row)), 'Saved score binding mismatch')
    for name in ('expert', 'joint'):
        value = item[name]
        require(type(value['tokens']) is int and value['tokens'] > 0
                and type(value['correct_tokens']) is int and 0 <= value['correct_tokens'] <= value['tokens']
                and isinstance(value['nll_sum'], (int, float)) and math.isfinite(value['nll_sum'])
                and value['nll_sum'] >= 0, 'Invalid saved token score')
    require(item['expert']['tokens'] == item['joint']['tokens'], 'Paired target lengths differ')


def summarize(items):
    groups = defaultdict(list)
    for item in items: groups[tuple(item['group'])].append(item)
    output = []
    for key, values in sorted(groups.items()):
        entry = dict(zip(('family', 'form', 'source'), key)); entry['records'] = len(values)
        for arm in ('expert', 'joint'):
            entry[arm + '_record_mean_nll'] = sum(v[arm]['nll_sum'] / v[arm]['tokens'] for v in values) / len(values)
            entry[arm + '_token_nll'] = sum(v[arm]['nll_sum'] for v in values) / sum(v[arm]['tokens'] for v in values)
        entry['paired_mean_nll_difference_expert_minus_joint'] = entry['expert_record_mean_nll'] - entry['joint_record_mean_nll']
        output.append(entry)
    families = {}
    for family in ('G', 'S', 'A'):
        values = [v for v in output if v['family'] == family]
        if not values: continue
        delta = sum(v['paired_mean_nll_difference_expert_minus_joint'] for v in values) / len(values)
        families[family] = {'equal_source_form_mean_nll_difference': delta,
                            'training_reference_nll_improved': delta < 0, 'groups': len(values)}
    return {'schema': VERSION, 'records': len(items), 'groups': output, 'families': families,
        'verdict': 'CALIBRATION_COMPLETE', 'teacher_qualified': False, 'heldout_accuracy': False,
        'interpretation': 'Training-reference fit only. Not semantic accuracy, unseen generalization, or a teacher eligibility report.'}


def score(args):
    manifest, rows = load_panel(args.panel)
    runs = {}; adapters = {}; protocols = {}
    for value in args.training_run:
        family, path = value.split('=', 1)
        require(family in ('G', 'S', 'A', 'joint') and family not in runs, 'Specify each training scope once')
        runs[family] = Path(path).resolve()
    require(set(runs) == {'G', 'S', 'A', 'joint'}, 'Four completed native SFT runs required')
    for family, root in runs.items():
        protocol = strict_json((root / 'protocol.json').read_bytes())
        require(protocol['scope'] == family and Path(protocol['model']).resolve() == args.model
                and protocol['export_manifest_sha256'] == manifest['training_export_manifest_sha256'],
                'Training/model/export identity mismatch')
        adapter = terminal_adapter(root, protocol)
        require(adapter is not None, 'Calibration requires terminal successful training, not partial checkpoints')
        adapters[family] = adapter; protocols[family] = protocol
    base_files = protocols['joint']['base_files_sha256']
    require(all(p['base_files_sha256'] == base_files for p in protocols.values()), 'Unmatched base models')
    for name, checksum in base_files.items(): require(sha(args.model / name) == checksum, 'Base bytes changed')
    protocol = {'schema': VERSION, 'panel_manifest_sha256': sha(args.panel / 'manifest.json'),
        'training_export_manifest_sha256': manifest['training_export_manifest_sha256'], 'base_files_sha256': base_files,
        'training_protocol_sha256': {k: sha(v / 'protocol.json') for k, v in runs.items()}, 'adapters': adapters,
        'model': str(args.model), 'max_length': 16384, 'image_tokens': 1536, 'batch_size': 1,
        'thinking': False, 'dtype': 'bfloat16', 'attention': 'sdpa', 'seed': 17,
        'metric': 'next-token NLL on all template-supervised assistant tokens; no truncation',
        'comparison': 'paired record-mean NLL, equal weighting across source/form strata per family',
        'teacher_qualified': False, 'heldout_used': False,
        'environment': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'ms-swift', 'peft')},
        'code_sha256': {name: sha(Path(__file__).parents[1] / name) for name in CODE}}
    target = args.output / 'protocol.json'
    if args.resume: require(strict_json(target.read_bytes()) == protocol, 'Resume protocol differs')
    else:
        require(not target.exists(), 'Existing score run requires explicit resume'); save(target, protocol)
    protocol_sha = sha(target); completed = []; predictions = args.output / 'scores.jsonl'
    if predictions.exists():
        require(args.resume, 'Existing scores require explicit resume')
        with predictions.open('rb') as stream:
            for ordinal, raw in enumerate(stream):
                require(raw.endswith(b'\n') and ordinal < len(rows), 'Incomplete/excess score tail')
                item = strict_json(raw); check_score(item, rows[ordinal], ordinal, protocol_sha); completed.append(item)
    images = {}
    for row in rows:
        path = row['images'][0]
        if path not in images: images[path] = sha(path)
        require(images[path] == row['gsa_meta']['binding']['image_sha256'], 'Calibration image changed')
    if len(completed) < len(rows):
        lock, uuid = acquire_gpu(args)
        with lock:
            save(args.output / 'status.json', {'stage': 'loading_model', 'pid': os.getpid(), 'uuid': uuid})
            os.environ['CUDA_VISIBLE_DEVICES'] = uuid
            import torch
            from peft import PeftModel
            from swift.model import get_model_processor
            from swift.template import get_template
            from swift.utils import to_device
            torch.manual_seed(17)
            model, processor = get_model_processor(str(args.model), model_type='qwen3_5', use_hf=True,
                torch_dtype=torch.bfloat16, device_map='cuda:0', attn_impl='sdpa')
            model = PeftModel.from_pretrained(model, adapters['joint']['path'], adapter_name='joint', is_trainable=False)
            for family in ('G', 'S', 'A'): model.load_adapter(adapters[family]['path'], adapter_name=family, is_trainable=False)
            model.eval(); model.requires_grad_(False)
            template = get_template(processor, template_type='qwen3_5', max_length=16384,
                                    truncation_strategy='raise', enable_thinking=False)
            template.set_mode('train')
            with predictions.open('a', encoding='utf-8') as stream, torch.inference_mode():
                for ordinal in range(len(completed), len(rows)):
                    row = rows[ordinal]
                    batch = to_device(template.data_collator([template.encode({k: row[k] for k in ('messages', 'images')})]), 'cuda:0')
                    positions, labels = response_positions(batch['labels'], batch['attention_mask'])
                    inputs = {k: v for k, v in batch.items() if k not in ('labels', 'loss_scale') and not k.startswith('_')}
                    inputs.update(logits_to_keep=positions, use_cache=False)
                    item = {'ordinal': ordinal, 'row_sha256': digest(row), 'group': list(group(row)), 'protocol_sha256': protocol_sha}
                    for name, family in (('expert', row['gsa_family']), ('joint', 'joint')):
                        model.set_adapter(family); model.requires_grad_(False); model.eval()
                        with template.forward_context(model, batch): result = model(**inputs)
                        item[name] = token_scores(result.logits, labels); del result
                    check_score(item, row, ordinal, protocol_sha)
                    stream.write(canonical(item) + '\n'); stream.flush(); os.fsync(stream.fileno()); completed.append(item)
                    save(args.output / 'status.json', {'stage': 'scoring', 'pid': os.getpid(), 'uuid': uuid,
                        'records_completed': len(completed), 'records_total': len(rows)})
    for path, checksum in images.items(): require(sha(path) == checksum, 'Image changed during calibration')
    for adapter in adapters.values():
        for name, checksum in adapter['files_sha256'].items(): require(sha(Path(adapter['path']) / name) == checksum, 'Adapter changed')
    report = summarize(completed); report.update(protocol_sha256=protocol_sha, scores_sha256=sha(predictions))
    save(args.output / 'metrics.json', report)
    save(args.output / 'status.json', {'stage': 'calibration_complete', 'records': len(completed), 'teacher_qualified': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('select', 'score'))
    for name in ('export', 'panel', 'model', 'lock-root'): parser.add_argument('--' + name, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-group', type=int, default=64); parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--training-run', action='append', default=[])
    parser.add_argument('--gpu'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ('export', 'panel', 'model', 'output', 'lock_root'):
        if getattr(args, key): setattr(args, key, getattr(args, key).resolve())
    if args.mode == 'select':
        require(args.export and not args.resume, 'Selection needs a complete export and new output'); select(args); return
    require(args.panel and args.model and args.lock_root and args.gpu, 'Scoring needs panel, model and GPU lock/pool')
    if not args.worker:
        require(args.resume or not args.output.exists(), 'Existing run requires explicit resume')
        snapshot = args.output / 'code'
        if not args.resume:
            for name in CODE:
                target = snapshot / name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(__file__).parents[1] / name, target)
            for name in ('tools', 'eval', 'trains', 'trains/flywheel'): (snapshot / name / '__init__.py').touch()
        env = dict(os.environ)
        for key in ('PYTHONPATH', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN',
                    'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'NPROC_PER_NODE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
        env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                   CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                   USE_HF='1', IMAGE_MAX_TOKEN_NUM='1536', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   TOKENIZERS_PARALLELISM='false', DS_IGNORE_CUDA_DETECTION='1', WANDB_MODE='disabled')
        for key, name in (('TMPDIR', 'runtime'), ('HF_HOME', 'cache/hf'), ('MODELSCOPE_CACHE', 'cache/modelscope')):
            directory = args.output / name; directory.mkdir(parents=True, exist_ok=True); env[key] = str(directory)
        raise SystemExit(subprocess.call([sys.executable, '-B', str(snapshot / 'tools/calibrate_native_teachers.py'),
                                         *sys.argv[1:], '--worker'], env=env))
    with (args.output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(args.output / 'process.json', process_identity(os.getpid()))
        try: score(args)
        except BaseException as exc:
            save(args.output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'automatic_restart': False})
            raise


if __name__ == '__main__': main()
