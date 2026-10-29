"""Paired G/S diagnostics for a nonfinal objective control, never final validation.

Reuse the exact 35,989 baseline rows and the unchanged native evaluator. This
independently declared diagnostic does not alter the running training plan.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
import yaml

C = Path(__file__).resolve().parent
D = C / 'diagnosis_32gpu'
R = C.parent
sys.path[:0] = [str(C / 'objective_controls_20261029'), str(D), str(C / 'eval_code')]
from objective_control_common import bound, load_spec as load_midrun_spec, read, require, save, sha
from student_matched_native import plan as baseline_selection
from retention_node import environment, run, EVAL
from eval_core import digest

MATCHED_PROTOCOL_FIELDS = [
    'export_manifest_sha256', 'selection_sha256', 'context_length', 'image_max_tokens',
    'dtype', 'seed', 'temperature', 'thinking', 'token_budgets', 'concurrency',
    'max_num_batched_tokens', 'multimodal_max_pixels', 'repetition_penalty',
    'cycle_guard', 'response_policy', 'malformed_and_aborted_examples_in_denominator',
    'code_sha256', 'gpu_memory_utilization', 'enforce_eager', 'prefix_caching',
    'scope', 'selected_forms', 'shard', 'shards', 'expected_records', 'files']


def load_spec(path):
    spec = read(path)
    require(spec['schema'] == 'anvil-objective-control-GS-diagnostic-v1', 'Wrong GS diagnostic spec')
    require(spec['step'] == 128 and spec['full_validation'] is False
            and spec['final_quality_accepted'] is False, 'Unexpected diagnostic scope')
    for filename, expected in spec['frozen_files'].items():
        require(sha(filename) == expected, 'Diagnostic dependency changed: ' + filename)
    binding = spec['public_spec']
    require(sha(binding['path']) == binding['sha256'], 'Public checkpoint binding changed')
    public = load_midrun_spec(binding['path'])
    require(public['step'] == spec['step'] and public['checkpoint'] == spec['checkpoint']
            and public['model'] == spec['model'], 'Diagnostic model mismatch')
    require(spec['selection'] == baseline_selection(), 'Fixed baseline selection changed')
    require('dof' not in spec['selection']['forms'], 'G/S diagnostic must not use latent DoF outputs')
    return spec


def prepare(public_path, out):
    public = load_midrun_spec(public_path)
    require(public['step'] == 128, 'Only an audited128-update control is supported')
    selection = read(D / 'student_matched_native_selection.json')
    require(selection == baseline_selection(), 'Baseline sample changed')
    baseline = read(D / 'student_matched_native_comparison.json')
    require(baseline['n'] == selection['expected_records'] == 35989, 'Incomplete baseline')
    require(baseline['spec_sha256'] == sha(D / 'student_matched_native_spec.json'), 'Baseline spec changed')
    frozen = [Path(__file__), D / 'student_matched_native.py', D / 'retention_node.py',
              D / 'student_matched_native_selection.json', D / 'student_matched_native_spec.json',
              D / 'student_matched_native_comparison.json', R / 'cloud_a800/checkpoints.py',
              C / 'objective_controls_20261029/objective_control_common.py'] + list((C / 'eval_code').rglob('*.py'))
    spec = dict(schema='anvil-objective-control-GS-diagnostic-v1', step=128, arm=public['arm'], objective_control=True, final_mopd=False,
                checkpoint=public['checkpoint'], model=public['model'],
                public_spec=bound(public_path), selection=selection,
                baseline_root=str(D / 'student_matched_native'),
                baseline_adapter=read(D / 'student_matched_native_spec.json')['checkpoint'],
                frozen_files={str(p): sha(p) for p in sorted(set(frozen))},
                full_validation=False, final_quality_accepted=False, weights_uploaded=False,
                note='Exact same fixed G/S sample as the initial student. Diagnostic only; final580K remains required.')
    path = out / 'spec.json'
    if path.exists():
        require(read(path) == spec, 'Existing immutable GS diagnostic spec differs')
    else:
        save(path, spec)
    config = yaml.safe_load((D / 'g_priority_native_node0_cli.yaml').read_text())
    config['TaskName'] = 'task_anvil_objective_' + public['arm'] + '_GS_A800x8'
    config['Description'] = 'Paired fixed35989 G/S diagnostic of exact nonfinal objective-control128; unchanged scorer'
    config['ActiveDeadlineSeconds'] = 14400
    argv = [str(R / '.venv-train/bin/python'), '-u', str(Path(__file__).resolve()),
            '--run', '--spec', str(path), '--spec-sha256', sha(path), '--output', str(out)]
    config['Entrypoint'] = 'set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n' + shlex.join(argv)
    content = yaml.safe_dump(config, sort_keys=False)
    cli = out / 'cli.yaml'
    if cli.exists():
        require(cli.read_text() == content, 'Existing GS diagnostic config differs')
    else:
        cli.write_text(content)
    return path


def checked_rows(folder, expected_n, evidence):
    receipt = read(folder / 'complete.json')
    output = folder / 'evaluation'
    metrics = read(output / 'metrics.json')
    require(sha(output / 'metrics.json') == receipt['metrics_sha256'], 'Changed saved metrics')
    protocol = read(output / 'protocol.json')
    protocol_sha = sha(output / 'protocol.json')
    require(metrics['state'] == 'complete_scope_shard' and metrics['n'] == expected_n,
            'Incomplete native diagnostic shard')
    require(metrics['protocol_sha256'] == protocol_sha and protocol['smoke_rows'] == 0,
            'Protocol mismatch or smoke output')
    evidence[str(output / 'metrics.json')] = sha(output / 'metrics.json')
    evidence[str(output / 'protocol.json')] = protocol_sha
    n = 0
    for filename, expected in metrics['prediction_sha256'].items():
        path = output / filename
        require(sha(path) == expected, 'Changed saved predictions')
        evidence[str(path)] = expected
        for line in path.open():
            row = json.loads(line)
            require(row['protocol_sha256'] == protocol_sha, 'Prediction protocol mismatch')
            require(hashlib.sha256(row['response'].encode()).hexdigest() == row['response_sha256'],
                    'Prediction text changed')
            require(digest({k: v for k, v in row.items() if k != 'record_sha256'}) == row['record_sha256'],
                    'Prediction record changed')
            require(all(math.isfinite(float(v)) for v in row['metrics'].values()), 'Nonfinite metric')
            n += 1
            yield row, protocol
    require(n == expected_n, 'Incorrect saved prediction denominator')


def paired_group_rows(baseline, current):
    require(baseline.keys() == current.keys(), 'Paired example IDs differ')
    groups = defaultdict(lambda: dict(n=0, baseline=defaultdict(float), current=defaultdict(float)))
    for identity, (row, protocol) in current.items():
        reference, base_protocol = baseline[identity]
        require(row['evaluation_content_sha256'] == reference['evaluation_content_sha256']
                and row['source'] == reference['source'], 'Paired reference differs')
        for field in MATCHED_PROTOCOL_FIELDS:
            require(protocol[field] == base_protocol[field], 'Unmatched inference protocol: ' + field)
        require(row['metrics'].keys() == reference['metrics'].keys(), 'Metric definitions differ')
        group = groups[(row['native_form'], row['source'])]
        group['n'] += 1
        for key, value in row['metrics'].items():
            group['current'][key] += float(value)
            group['baseline'][key] += float(reference['metrics'][key])
    return [dict(form=f, source=s, n=g['n'], metrics={key: dict(
                initial_student=g['baseline'][key] / g['n'], objective_control_checkpoint=value / g['n'],
                checkpoint_minus_initial=(value-g['baseline'][key]) / g['n'])
                for key, value in g['current'].items()}) for (f, s), g in sorted(groups.items())]


def aggregate(spec_path, output):
    spec = load_spec(spec_path)
    sets = []
    evidence = {str(spec_path): sha(spec_path)}
    for root in [Path(spec['baseline_root']), output / 'shards']:
        rows = {}
        for task in spec['selection']['tasks']:
            folder = root / str(task['shard'])
            if root == output / 'shards':
                require(read(folder / 'complete.json')['spec_sha256'] == sha(spec_path), 'Different GS spec')
            for row, protocol in checked_rows(folder, task['records'], evidence):
                key = (row['native_form'], row['id'])
                require(key not in rows, 'Duplicate sample identity')
                rows[key] = (row, protocol)
        require(len(rows) == spec['selection']['expected_records'], 'Incomplete sample')
        sets.append(rows)
    groups = paired_group_rows(*sets)
    report = dict(state='complete_paired_GS_diagnostic_requires_review', step=128, objective_control=True, final_mopd=False,
                  checkpoint=spec['checkpoint'], initial_student=spec['baseline_adapter'], n=35989,
                  full_validation=False, final_quality_accepted=False, weights_uploaded=False,
                  spec_sha256=sha(spec_path), groups=groups, evidence=evidence,
                  uncertainty='Fixed8of128ordinalstrata reused from baseline; descriptive matched estimates, not an independent final test.')
    save(output / 'report.json', report)
    return report


def node(spec_path, spec_sha, output):
    require(sha(spec_path) == spec_sha, 'Submitted spec content changed')
    spec = load_spec(spec_path)
    devices = []
    for line in subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                                        '--format=csv,noheader,nounits'], text=True).splitlines():
        index, uid, name, memory = map(str.strip, line.split(','))
        require('A800' in name and int(memory) >= 79000, 'Unexpected GPU')
        devices.append((int(index), uid))
    devices.sort()
    require(len(devices) == 8, 'Expected eight GPUs')
    save(output / 'status.json', dict(stage='running', started=time.time(), devices=devices))

    def evaluate(task):
        gpu, uid = devices[task['local_gpu']]
        out = output / 'shards' / str(task['shard'])
        out.mkdir(parents=True, exist_ok=False)
        env = environment([gpu], 'objective128_matched_' + str(task['shard']))
        env['ANVIL_EXPECTED_GPU_UUID'] = uid
        (Path(env['XDG_CACHE_HOME']) / 'torch/kernels').mkdir(parents=True, exist_ok=True)
        argv = [EVAL, str(C / 'eval_code/run_vllm_eval.py'), '--model', spec['model'],
                '--name', 'objective_control128_GS_sample', '--scope', 'joint', '--gpu-uuid', uid,
                '--shard', str(task['shard']), '--shards', '128', '--concurrency', '64',
                '--memory', '.88', '--output', str(out / 'evaluation'), '--forms', *spec['selection']['forms']]
        save(out / 'launch.json', dict(argv=argv, task=task, gpu=gpu, uuid=uid, spec_sha256=spec_sha))
        run(argv, env, out / 'console.log')
        metrics = read(out / 'evaluation/metrics.json')
        require(metrics['state'] == 'complete_scope_shard' and metrics['n'] == task['records'],
                'Incomplete GS worker output')
        save(out / 'complete.json', dict(task=task, metrics_sha256=sha(out / 'evaluation/metrics.json'),
                                        spec_sha256=spec_sha))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(evaluate, spec['selection']['tasks']))
    aggregate(spec_path, output)
    save(output / 'status.json', dict(stage='complete_paired_GS_diagnostic_requires_review',
                                     completed=time.time(), report_sha256=sha(output / 'report.json')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--public-spec', type=Path)
    ap.add_argument('--spec', type=Path)
    ap.add_argument('--spec-sha256')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.run:
            require(args.spec is not None and args.spec_sha256, 'Run requires a bound spec')
            node(args.spec, args.spec_sha256, args.output)
        else:
            require(args.public_spec is not None, 'Preparation requires a sealed public checkpoint')
            print(prepare(args.public_spec, args.output))
    except BaseException as exc:
        save(args.output / 'failure.json', dict(error=repr(exc), time=time.time()))
        raise
