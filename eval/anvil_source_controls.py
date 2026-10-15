#!/usr/bin/env python3
"""Frozen source controls and single-GPU retention for the September 15 closeout."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
A = W.parent / 'Anvil_ICLR27'
ROOT = A / 'engineering/data/source_controls_20260915'
REPAIR = A / 'engineering/data/vacancy_repair_20260914'
CONTROL_MODES = tuple(f'control_{kind}_s{seed}' for kind in ('replay', 'vacancy') for seed in (43, 44))


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def bind(p):
    p = Path(p)
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def prepare():
    source = REPAIR / 'split/train.jsonl'
    dev = REPAIR / 'split/dev.jsonl'
    manifest = read(REPAIR / 'split/manifest.json')
    assert bind(source)['sha256'] == manifest['train']['sha256']
    assert bind(dev)['sha256'] == manifest['dev']['sha256']
    train, valid = rows(source), rows(dev)
    assert len(train) == 7264 and len(valid) == 360
    dev_images = {im for r in valid for im in r['images']}
    plan = {'schema': 'anvil-source-controls-v1', 'source_train': bind(source),
            'shared_dev': bind(dev), 'script': bind(__file__), 'seeds': [43, 44],
            'steps': 303, 'effective_batch': 24, 'initialization': str(A / 'engineering/data/p5_retention_clean_run_seed17_v2/runtime/merged'),
            'scope': 'Matched optimizer updates and row exposures; not matched token count or unique examples. Source substitution, not full MTD or flywheel causality.',
            'selection': 'Two fixed seeds; all six endpoints reported; terminal checkpoint303 only.', 'arms': {}}
    for kind, expected in (('vacancy', 3608), ('replay', 3656)):
        selected = [r for r in train if (r['gsa_meta']['source'] == 'refspatial_3d_vacant') == (kind == 'vacancy')]
        assert len(selected) == expected
        assert not dev_images.intersection(im for r in selected for im in r['images'])
        rng = random.Random(20260915)
        expanded = []
        while len(expanded) < len(train):
            cycle = list(selected)
            rng.shuffle(cycle)
            expanded.extend(cycle[:len(train)-len(expanded)])
        rng.shuffle(expanded)
        target = ROOT / kind / 'train.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as f:
            for row in expanded:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        plan['arms'][kind] = {'unique_source_rows': expected, 'training_rows': len(expanded),
                             'train': bind(target), 'sources': dict(Counter(r['gsa_meta']['source'] for r in selected))}
    write(ROOT / 'plan.json', plan)
    print(json.dumps(plan, indent=2))


def retention(out):
    # Reuse the immutable official suite and strict scorers, with explicitly local
    # execution metadata. Do not inherit the old four-GPU admission declaration.
    from eval import general_retention_formal_v3 as f
    old = A / 'engineering/data/p5_retention_clean_run_seed17_v2/retention'
    receipt_path = old / 'inputs/input_receipt.json'
    receipt = read(receipt_path)
    runtime = f._runtime_rows(receipt)
    assert len(runtime) == 10105
    f._verify_runtime_images(runtime)
    model = REPAIR / 'merged'
    old_results = {role: f._verify_result(old / f'results/{role}.json') for role in ('base', 'e1')}
    assert all(r['input_receipt'] == f.file_binding(receipt_path) for r in old_results.values())
    out.mkdir(parents=True, exist_ok=False)
    launch = {'schema': 'anvil-local-retention-v1', 'model': f.model_inventory(model),
              'suite_receipt': bind(receipt_path), 'runtime_index': bind(receipt['runtime_index']['path']),
              'script': bind(__file__), 'frozen_runner': bind(f.__file__),
              'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'), 'expected_n': 10105,
              'execution': 'one GPU; same prompts, strict scorers, greedy non-thinking, 4096 visual tokens, MC batch16, TextVQA batch1; all cases retained',
              'scope': 'Local retention observation; not the old four-GPU admission or full MTD non-inferiority.'}
    write(out / 'launch.json', launch)
    config = f.load_yaml(Path(receipt['config']['path']))
    digit_article = f._reference_digit_article(Path(config['source_evidence']['vqa_reference_path']))
    predictions, credits = [], []
    start = time.monotonic()
    with (out / 'predictions.jsonl').open('x', buffering=1) as stream:
        for dataset in f.DATASET_ORDER:
            selected = [r for r in runtime if r['dataset'] == dataset]
            batch = f.DATASET_BATCH_SIZES[dataset]
            for offset in range(0, len(selected), batch):
                part = selected[offset:offset+batch]
                public = [{'messages': [{'role': 'user', 'content': r['prompt']}], 'images': [r['image_path']]} for r in part]
                assert len({r['max_tokens'] for r in part}) == 1
                responses = f._real_infer(public, part[0]['max_tokens'], model=str(model), engine_batch_size=16,
                                          execution_batch_size=batch, request_ids=[r['global_uid'] for r in part])
                for sample, (response, capped) in zip(part, responses):
                    pred = {'global_uid': sample['global_uid'], 'dataset': dataset, 'response': response, 'hit_token_cap': capped}
                    predictions.append(pred)
                    stream.write(json.dumps(pred, ensure_ascii=False)+'\n')
                    if sample['metric'] == 'multiple_choice_accuracy':
                        parsed, unclosed = f.parse_multiple_choice(response, sample['choices'])
                        credit = float(parsed == sample['answer'])
                    else:
                        assert sample['metric'] == 'official_vqa_soft_accuracy'
                        parsed, unclosed = f._answer_span(response)
                        credit = f._vqa_soft_credit(parsed, sample['answer'], digit_article) if parsed is not None else 0.0
                    credits.append({'global_uid': sample['global_uid'], 'dataset': dataset, 'credit': credit,
                                    'malformed': parsed is None, 'truncated': bool(capped or unclosed)})
                if offset % (batch*20) == 0 or offset+batch >= len(selected):
                    print(f'{dataset} {min(offset+batch,len(selected))}/{len(selected)} elapsed={time.monotonic()-start:.0f}s', flush=True)
    assert len(predictions) == len({r['global_uid'] for r in predictions}) == 10105
    assert {r['global_uid'] for r in predictions} == {r['global_uid'] for r in runtime}
    benchmarks = {}
    for dataset in f.DATASET_ORDER:
        group = [r for r in credits if r['dataset'] == dataset]
        score = 100*sum(r['credit'] for r in group)/len(group)
        benchmarks[dataset] = {'n': len(group), 'score': score,
            'malformed': sum(r['malformed'] for r in group), 'truncated': sum(r['truncated'] for r in group),
            'base': old_results['base']['benchmarks'][dataset]['raw_score_percent'],
            'P5': old_results['e1']['benchmarks'][dataset]['raw_score_percent']}
    report = {'schema': 'anvil-local-retention-v1', 'benchmarks': benchmarks, 'per_sample': credits,
              'macro_score': sum(r['score'] for r in benchmarks.values())/3,
              'elapsed_s': time.monotonic()-start, 'launch': bind(out/'launch.json'),
              'predictions': bind(out/'predictions.jsonl'),
              'comparators': {role: bind(old/f'results/{role}.json') for role in old_results}}
    write(out/'result.json', report)
    write(out/'completion_status.json', {'rc': 0, 'n': 10105})
    print(json.dumps({'benchmarks': benchmarks, 'macro_score': report['macro_score']}, indent=2))


def summarize(out, wait_hours):
    from eval import anvil_closeout_acceptance as accept
    modes = (*accept.MODES, *CONTROL_MODES)
    deadline = time.monotonic()+3600*wait_hours
    while not all((accept.ROOT/m/'completion_status.json').exists() for m in (*CONTROL_MODES, 'vacancy_seed44', 'vacancy_remaining')):
        if time.monotonic() >= deadline:
            break
        time.sleep(30)
    accept.MODES = modes
    report = accept.accept()
    report['source_control_plan'] = bind(ROOT/'plan.json')
    report['summarizer'] = bind(__file__)
    table = {(r['model'], r['benchmark']): r['score'] for r in report['cells']}
    report['source_controls'] = []
    for seed, mixed in ((43, 'vacancy_repair'), (44, 'vacancy_seed44')):
        for bench in ('where2place', 'robospatial_home_context', 'vabench_point', 'blink_relative_depth', 'erqa', 'embspatial'):
            names = {'mixed': mixed, 'replay': f'control_replay_s{seed}', 'vacancy': f'control_vacancy_s{seed}'}
            if all((m, bench) in table for m in names.values()):
                values = {k: table[m, bench] for k, m in names.items()}
                report['source_controls'].append({'seed': seed, 'benchmark': bench, **values,
                                                 'mixed_minus_replay': values['mixed']-values['replay']})
    write(out, report)
    print(json.dumps({'accepted_cells': len(report['cells']), 'pending': report['pending'], 'source_controls': report['source_controls']}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['prepare', 'retention', 'summarize'])
    ap.add_argument('--out', type=Path)
    ap.add_argument('--wait-hours', type=float, default=10)
    args = ap.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'retention':
        retention(args.out or ROOT/'retention')
    else:
        summarize(args.out or ROOT/'acceptance.json', args.wait_hours)
