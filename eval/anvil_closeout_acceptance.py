#!/usr/bin/env python3
"""Accept complete saved predictions and summarize Anvil closeout without model calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
from eval.gsa_eval.registry import get_spec
from eval.gsa_eval.parsing import parse_points

A = W.parent / 'Anvil_ICLR27'
ROOT = A / 'engineering/data/table2_fill_20260914'
MODES = ('p4_seed23', 'mimo_g_completion', 'cosmos', 'vacancy_repair',
         'vacancy_remaining', 'vacancy_seed44')


def read(p):
    return json.loads(Path(p).read_text())


def binding(p):
    p = Path(p)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def predictions(p):
    rows = [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    ids = [str(row['uid']) for row in rows]
    assert len(ids) == len(set(ids)), f'duplicate IDs: {p}'
    return dict(zip(ids, rows))


def score_saved(spec, samples, responses):
    kwargs = {'spec': spec} if 'spec' in inspect.signature(spec.scorer).parameters else {}
    return spec.scorer(samples, responses, **kwargs)


def accept():
    out = {'schema': 'anvil-closeout-acceptance-v1', 'cells': [], 'pending': [],
           'script': binding(__file__), 'scope': 'Observed local SFT/checkpoint results; not full MTD.'}
    for mode in MODES:
        p = ROOT / mode
        if not (p / 'worker.json').exists():
            out['pending'].append(mode)
            continue
        worker = read(p / 'worker.json')
        for bench, item in worker['benchmarks'].items():
            result = p / f'{bench}.json'
            if item['status'] != 'ok' or not result.exists():
                out['pending'].append(f'{mode}/{bench}')
                continue
            d = read(result)
            spec = get_spec(bench)
            samples = list(spec.loader(W / '.cache/eval'))
            shard = p / 'shards' / bench / 'shard_0.jsonl'
            rows = predictions(shard)
            assert set(rows) == {str(s.uid) for s in samples}, (mode, bench, 'coverage')
            assert d['n_evaluated'] == item['n'] == len(samples), (mode, bench, 'count')
            responses = [rows[str(s.uid)]['response'] for s in samples]
            score, _, fails = score_saved(spec, samples, responses)
            observed = d['score'] if d['score'] is not None else d['score_if_untrusted']
            assert abs(score - observed) < .015 and fails == d['parse_failures'], (mode, bench, 'rescore')
            out['cells'].append({'model': mode, 'benchmark': bench, 'score': observed,
                'formal_score_accepted': d['score'] is not None, 'n': len(samples),
                'parse_failures': fails, 'capped': d['truncated_hit_token_cap'],
                'result': binding(result), 'shard': binding(shard)})

    # Paired image bootstrap: inference uncertainty at fixed checkpoints, not seed uncertainty.
    summary = read(A / 'engineering/data/public_eval_expansion_20260914/summary.json')
    lookup = {(x['model_id'], x['benchmark']): x['prediction_path'] for x in summary['cells']}
    lookup['p5', 'where2place'] = str(W / 'save/logs/preds_anvil_p5_retention_clean_seed17_final_v7_where2place.jsonl')
    out['paired_vacancy'] = []
    for bench in ('where2place', 'robospatial_home_context'):
        newpath = ROOT / 'vacancy_repair' / f'preds_{bench}.jsonl'
        oldpath = Path(lookup['p5', bench])
        if not oldpath.exists():
            out['pending'].append(f'paired/{oldpath}')
            continue
        old, new = predictions(oldpath), predictions(newpath)
        spec = get_spec(bench)
        samples = list(spec.loader(W / '.cache/eval'))
        groups = defaultdict(list)
        for s in samples:
            key = str(s.uid)
            delta = score_saved(spec, [s], [new[key]['response']])[0] - score_saved(spec, [s], [old[key]['response']])[0]
            groups[tuple(s.image_paths)].append(delta)
        sums = np.array([sum(v) for v in groups.values()])
        sizes = np.array([len(v) for v in groups.values()])
        rng = np.random.default_rng(20260915)
        draws = rng.integers(0, len(sums), size=(5000, len(sums)))
        boot = sums[draws].sum(1) / sizes[draws].sum(1)
        out['paired_vacancy'].append({'benchmark': bench, 'n': len(samples),
            'image_clusters': len(groups), 'delta_pp': float(sums.sum()/sizes.sum()),
            'image_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(),
            'repetitions': 5000, 'bootstrap_seed': 20260915,
            'scope': 'Fixed-checkpoint paired image uncertainty; not across-training-seed CI.',
            'old': binding(oldpath), 'new': binding(newpath)})

    common = read(A / 'engineering/data/public_eval_expansion_20260914/common_subset_cpu.json')
    out['common_subset'] = []
    for bench in ('erqa', 'vabench_point'):
        spec = get_spec(bench)
        samples = [s for s in spec.loader(W / '.cache/eval') if str(s.uid) not in common['excluded_uids'][bench]]
        rows = predictions(ROOT / 'vacancy_repair' / f'preds_{bench}.jsonl')
        responses = [rows[str(s.uid)]['response'] for s in samples]
        score = score_saved(spec, samples, responses)[0]
        row = {'benchmark': bench, 'n': len(samples), 'score': score,
               'scope': 'Same historical P4/P5 image-exclusion set; no new scene-disjoint assertion.'}
        if bench == 'vabench_point':
            first = [json.dumps(parse_points(t)[:1]) for t in responses]
            row['first_point'] = score_saved(spec, samples, first)[0]
        out['common_subset'].append(row)

    corpus = W / 'trains/gsa_bench/train.jsonl'
    forms, images = Counter(), set()
    for line in corpus.open():
        r = json.loads(line)
        forms[r['gsa_meta']['form']] += 1
        images.update(r['images'])
    out['historical_corpus'] = {'rows': sum(forms.values()), 'images': len(images),
        'forms': dict(forms), 'source': binding(corpus),
        'G': sum(forms[k] for k in ('bbox', 'detection', 'mask', 'point')),
        'S': forms['depth'], 'A': sum(forms[k] for k in ('action_cot', 'affordance', 'dof'))}
    dof = W / 'data/flywheel_v2/full500/raw/dof.jsonl'
    rows = [json.loads(line) for line in dof.read_text().splitlines()]
    out['historical_dof'] = {'n': len(rows), 'dominant_axis_records': sum(
        any(item.get('jaw_axis_2d') == [.71, .71] for item in r['payload']) for r in rows),
        'source': binding(dof), 'scope': 'Frequency only; not proof of semantic collapse or later replacement.'}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--wait-mode', action='append', default=[])
    ap.add_argument('--wait-hours', type=float, default=8)
    a = ap.parse_args()
    deadline = time.monotonic() + a.wait_hours*3600
    while a.wait_mode and not all((ROOT/m/'completion_status.json').exists() for m in a.wait_mode):
        if time.monotonic() >= deadline:
            break
        time.sleep(30)
    assert not a.out.exists(), f'preserve existing acceptance: {a.out}'
    out = accept()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({'accepted': len(out['cells']), 'pending': out['pending'],
                      'paired_vacancy': out['paired_vacancy'], 'common_subset': out['common_subset'],
                      'historical_corpus': out['historical_corpus'], 'historical_dof': out['historical_dof']}, indent=2))


if __name__ == '__main__':
    main()
