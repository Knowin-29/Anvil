"""Bind exact real first128 source rows and unchanged scoring before GPU submission."""
import ast
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import yaml
from objective_control_common import HERE, C, D, R, TRAIN, bound, read, require, save, sha
sys.path[:0] = [str(C), str(C / 'formal_mopd')]
from formal_common import load_plan, sealed_checkpoint


def ast_body(path, name):
    tree = ast.parse(path.read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name), include_attributes=False)


def main():
    require(not (HERE / 'experiment.json').exists(), 'Experiment already sealed')
    formal = C / 'formal_mopd/runs/candidate1_20261029'
    plan = load_plan(formal / 'plan.json')
    v6, v7 = C / 'native_mopd_v6.py', C / 'native_mopd_v7.py'
    for name in ['FullPool', 'ReplayPool', 'install_hidden_forward', 'move', 'forward_inputs']:
        require(ast_body(v6, name) == ast_body(v7, name), 'Changed helper: ' + name)
    require(ast_body(HERE / 'score.py', 'public_score') == ast_body(C / 'formal_mopd/score_checkpoint.py', 'public_score'),
            'Public scoring changed')
    require(ast_body(HERE / 'public_worker.py', 'main') == ast_body(C / 'formal_mopd/checkpoint_public_worker.py', 'main'),
            'Public inference changed')
    for p in [v7, *HERE.glob('*.py')]:
        compile(p.read_text(), str(p), 'exec')
    for p in [v7, HERE / 'node.py', HERE / 'merge.py', HERE / 'public_worker.py', HERE / 'score.py']:
        subprocess.run([TRAIN, str(p), '--help'], stdout=subprocess.DEVNULL, check=True, timeout=60)
    import importlib.util
    submission_spec = importlib.util.spec_from_file_location('objective_submit_validation', HERE / 'submit.py')
    submission_module = importlib.util.module_from_spec(submission_spec)
    submission_spec.loader.exec_module(submission_module)
    # Real-source audit, not a synthetic duplicate of the sampler formula alone.
    from native_mopd_v7 import ReplayPool
    pool = ReplayPool(True)
    baseline = defaultdict(list); seen = set()
    for rank in range(16):
        for line in (formal / 'training' / ('exposure_rank%d.jsonl' % rank)).open():
            row = json.loads(line)
            if row['step'] > 128:
                continue
            key = (row['step'], row['micro'], row['rank'])
            require(key not in seen and row['rank'] == rank, 'Duplicate original exposure')
            seen.add(key)
            baseline[str(row['step'])].append([row['canonical_uid'], row['source_category'],
                                               row['family'], row['form'], row['source_binding']])
    require(len(seen) == 4096 and len(baseline) == 128, 'Incomplete original first128 source rows')
    canonical = lambda items: sorted(json.dumps(x, sort_keys=True) for x in items)
    actual = Counter()
    for step in range(128):
        sampled = []
        for micro in range(4):
            categories = set()
            for rank in range(8):
                global_slot = micro * 8 + rank
                row = pool.sample(step * 2 + global_slot // 16, global_slot % 16, 16)
                category = 'general_qa' if row['gsa_family'] == 'QA' else 'spatial_reference' if row.get('anvil_replay_source') else 'native_distillation'
                sampled.append([row['canonical_uid'], category, row['gsa_family'], row['gsa_form'], row['gsa_meta']['binding']])
                actual[category] += 1; categories.add(category)
            require(len(categories) == 1, 'Ranks would enter mismatched collectives')
        require(canonical(sampled) == canonical(baseline[str(step+1)]), 'Actual source sampling differs')
    require(actual == dict(native_distillation=2048, general_qa=1024, spatial_reference=1024), 'Changed source mixture')
    with tempfile.TemporaryDirectory() as temp:
        adapter = Path(temp) / 'checkpoint-128'; adapter.mkdir()
        save(adapter.parent / 'protocol.json', dict(smoke=False, quality_qualified=False, full_budget_probe=False, objective_control=True))
        try:
            sealed_checkpoint(adapter)
        except ValueError:
            pass
        else:
            raise AssertionError('A control was admitted to formal delivery')
    # Zero-distillation without an explicit diagnostic flag must fail before CUDA.
    reject = subprocess.run([TRAIN, str(v7), '--output', '/tmp/anvil-never-created-control',
                             '--steps', '128', '--mtd-weight', '0'], capture_output=True, text=True, timeout=60)
    require(reject.returncode != 0 and 'Zero-distillation controls cannot be formal MOPD' in reject.stderr,
            'Zero-MTD production guard did not reject')
    save(HERE / 'baseline_source_rows.json', dict(schema='anvil-formal-first128-source-rows-v1',
        records=4096, rows=dict(baseline), formal_checkpoint=bound(formal / 'training/checkpoint-128/checkpoint_complete.json'),
        data_is_training_only=True))
    derivation = dict(original=bound(v6), derived=bound(v7), changes=[
        'Explicit native-reference SFT and MTD coefficients; supervised replay unchanged.',
        'Zero-MTD control skips teacher and sampled rollout; never a final MOPD.',
        'Short controls preserve1664-update optimizer schedule.',
        'Phase2 compares native SFT0/MTD1 and native SFT.1/MTD0 with the same original student and teachers.',
        'Eight-GPU virtual16 sampler matches all4096 real first128 source rows exactly.',
        'Runtime proof binds trainer/student/teachers/topology/weights/schedule; metadata excludes all controls from final delivery.'],
        limitations='Eight versus16 physical GPUs changes rank generation RNG and floating-point reduction order; all-eligible per-example weights are equal. These are diagnostic comparisons, not exact stochastic replications.')
    save(HERE / 'derivation.json', derivation)
    arms = dict(no_native_sft=dict(native_sft_weight=0., mtd_weight=1.), low_native_sft_no_mtd=dict(native_sft_weight=.1, mtd_weight=0.))
    configs = []
    for arm in arms:
        config = yaml.safe_load((D / 'mopd_v6_candidate1/cli.yaml').read_text())
        config['TaskName'] = 'task_anvil_objective_' + arm + '_128_A800x8'
        config['Description'] = 'Nonfinal matched128-update loss diagnostic; real GPU1to3resume; full7public plus10105general. Source rows match formal16GPU.'
        config['ActiveDeadlineSeconds'] = (8 if arm == 'no_native_sft' else 2)*3600
        config['Entrypoint'] = 'set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n' + TRAIN + ' -u ' + str(HERE / 'node.py') + ' --arm ' + arm
        folder = HERE / 'submissions' / arm; folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'cli.yaml'; path.write_text(yaml.safe_dump(config, sort_keys=False)); configs.append(path)
    files = set(map(Path, plan['frozen_files'])) | set(HERE.glob('*.py')) | set(configs)
    files |= {v7, HERE / 'baseline_source_rows.json', HERE / 'derivation.json', formal / 'eligibility.json',
              D / 'inputs/manifest.json', D / 'retention_node.py', D / 'student_adaptation_public_comparison.json'}
    manifest = read(D / 'inputs/manifest.json')
    files |= {D / 'code' / name for name in manifest['code']}
    files.add(Path(manifest['retention']['vqa_reference']))
    frozen = {str(p):sha(p) for p in sorted(files)}
    validation = dict(state='passed', timestamp=time.time(), frozen_files=frozen,
        real_source_rows_matched=4096, updates=128, source_categories=dict(actual),
        all_ranks_same_collective_branch=True, original_v6_unchanged=True,
        public_score_AST_unchanged=True, public_inference_AST_unchanged=True,
        helper_AST_unchanged=True, control_final_admission_rejected=True, zero_mtd_production_rejected=True,
        syntax_and_cli_passed=True, gpu_runtime_tested=False, weights_uploaded=False)
    save(HERE / 'cpu_validation.json', validation)
    experiment = dict(schema='anvil-native-objective-controls-v1', arms=arms, steps=128,
        world=8, sampler_world=16, global_batch=32, schedule_steps=1664, final_mopd=False,
        runtime_bindings=str(D / 'mopd_v6_candidate1/bindings.json'),
        initialization_lineage=str(D / 'mopd_v6_candidate1/initialization_lineage.json'),
        original_eligibility=str(formal / 'eligibility.json'), baseline_source_rows=str(HERE / 'baseline_source_rows.json'),
        frozen_files=frozen, cpu_validation=bound(HERE / 'cpu_validation.json'),
        heldout_used_adaptively=True, formal_training_unmodified=True,
        comparison_limitations=derivation['limitations'], created=time.time())
    save(HERE / 'experiment.json', experiment)
    print(json.dumps({k:v for k,v in validation.items() if k != 'frozen_files'}, indent=2))


if __name__ == '__main__':
    main()
