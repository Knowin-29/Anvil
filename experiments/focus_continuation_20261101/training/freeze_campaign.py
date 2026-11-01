"""Seal one substantive continuation after complete exact-parent evaluation."""
import ast
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
import yaml

O = Path(__file__).resolve().parent
C = O.parent
R = C.parent
OLD = C / 'targeted_repair_20261031'
H = C / 'final_delivery_focus_v1'
PARENT_FD = C / 'delivery_runs/targeted_repair_20261031'
FD = C / 'delivery_runs/focus_repair_20261101'
TRAIN = R / '.venv-train/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bound(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def counts(schedule, steps, forms):
    sources, native, roles = Counter(), Counter(), Counter()
    for step in range(steps):
        cycle, slot = divmod(step, len(schedule))
        kind = schedule[slot]
        occurrence = cycle * schedule.count(kind) + schedule[:slot].count(kind)
        if kind == 'native':
            source = 'native'
            native[forms[occurrence % len(forms)]] += 64
        elif kind == 'qa':
            source = ('ai2d', 'scienceqa', 'textvqa')[occurrence % 3]
        else:
            source = {'depth': 'depth_replay', 'point': 'point_replay', 'mask': 'mask_replay'}.get(kind, kind)
        sources[source] += 64
        roles[kind] += 64
    assert sum(sources.values()) == steps * 64
    return sources, native, roles


def advance(offsets, sources, native):
    result = dict(offsets)
    for source, amount in sources.items():
        if source in {'native', 'native_distillation', 'original_native'}:
            continue
        key = {'depth_replay': 'depth', 'point_replay': 'point', 'mask_replay': 'mask'}.get(source, source)
        result[key] = result.get(key, 0) + amount
    for form, amount in native.items():
        key = 'native:' + form
        result[key] = result.get(key, 0) + amount
    return result


def main():
    assert not (O / 'spec.json').exists(), 'Never overwrite a frozen specification'
    old = read(OLD / 'spec.json')
    for path, digest in old['frozen_files'].items():
        assert sha(path) == digest, path
    complete = read(OLD / 'runs/mopd/complete.json')
    parent = Path(complete['checkpoint'])
    assert complete['steps'] == 768 and complete['stage'] == 'mopd_training_complete'
    assert sha(parent / 'adapter_model.safetensors') == complete['adapter_sha256']
    integrity = read(OLD / 'mopd_checkpoint768_verification.json')
    assert integrity['all_file_hashes_verified'] and integrity['optimizer_step'] == 768
    report_path = PARENT_FD / 'evaluation/final_report.json'
    report = read(report_path)
    assert sha(report_path) == '6fb7c10b0a83bc2c3cd9fb79c3c3669ce601bac1d0f3165f29d68f0caf450462'
    assert report['native']['n'] == 580000 and sum(x['n'] for x in report['general'].values()) == 10105
    assert report['state'] == 'complete_requires_quality_review'
    assert report['checkpoint'] == str(parent)
    lineage = read(PARENT_FD / 'training_lineage.json')
    assert lineage['state'] == 'complete' and lineage['checkpoint'] == str(parent)
    supplement_path = C / 'refspatial_grounding_20261101/manifest.json'
    supplement = read(supplement_path)
    assert supplement['state'] == 'verified_train_only' and supplement['all_current_image_hashes_verified']
    assert supplement['new_images_vs_admitted_pool'] == 0 and not supplement['evaluation_answers_consumed']
    encoding = read(O / 'supplement_encoding_validation.json')
    assert encoding['state'] == 'passed_cpu_only' and encoding['manifest_sha256'] == sha(supplement_path)
    assert encoding['validator_sha256'] == sha(O / 'validate_supplement_encoding.py')
    module_spec = importlib.util.spec_from_file_location('focus_pool', O / 'pool.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert read(O / 'schedules.json') == module.SCHEDULES
    steps = {'repair': 8192, 'mopd': 512}
    initial_offsets = advance(old['source_offsets']['mopd'], old['planned_source_exposures']['mopd'],
                              old['planned_native_form_exposures']['mopd'])
    initial_offsets.update(grounding=0, spatialqa=0)
    offsets, planned_sources, planned_forms, planned_roles = {}, {}, {}, {}
    offset = initial_offsets
    for stage in ('repair', 'mopd'):
        offsets[stage] = offset
        source, native, roles = counts(module.SCHEDULES[stage], steps[stage], module.FORMS)
        offset = advance(offset, source, native)
        source['original_native' if stage == 'repair' else 'native_distillation'] = source.pop('native')
        planned_sources[stage], planned_forms[stage], planned_roles[stage] = dict(source), dict(native), dict(roles)
    spec = copy.deepcopy(old)
    spec.update(initial=dict(adapter=str(parent), adapter_sha256=complete['adapter_sha256']),
                initialization_lineage=bound(PARENT_FD / 'training_lineage.json'),
                source_offsets=offsets, sampling_seed=old['sampling_seed'],
                planned_source_exposures=planned_sources, planned_native_form_exposures=planned_forms,
                planned_role_exposures=planned_roles, final_reserve_hours=8)
    spec['stages'] = dict(repair=dict(steps=8192, lr=2e-5, save_every=64, ul_weight=.1, ul_rollout_cap=768, ul_every_mask_steps=8),
                          mopd=dict(steps=512, lr=5e-7, save_every=32, native_sft_weight=1.,
                                    mtd_weight=.25, ul_weight=.1, ul_rollout_cap=768, ul_every_mask_steps=8))
    for name, item in supplement['files'].items():
        assert sha(item['path']) == item['sha256']
        spec['datasets'][name] = {key: item[key] for key in ('path', 'sha256', 'records')}
    spec['supplemental_source_admission'] = bound(supplement_path)
    spec['continuation_parent'] = dict(checkpoint=str(parent), adapter_sha256=complete['adapter_sha256'],
        lineage=bound(PARENT_FD / 'training_lineage.json'), training_complete=bound(OLD / 'runs/mopd/complete.json'),
        public_report=bound(report_path))
    trigger = dict(schema='anvil-complete-evaluation-continuation-trigger-v1',
        public_general_report=str(report_path), public_general_report_sha256=sha(report_path),
        scores=report['paper_aligned_public'], final_quality_accepted=False,
        rationale='Complete exact-model macro70.1171 vs paper74.27. Where2Place60.40 vs68, BLINK85.48 vs91.13, VA74.33 vs80.67 and RefSpatial Location54.57 vs59 remain material gaps. Full native pointing/mask geometry remains weak. Continue substantive source-grounded training; no invented tolerance or scorer change.')
    save_new(O / 'public_trigger.json', trigger)
    spec['recovery'] = dict(reason='Substantive grounding/placement continuation after complete parent evaluation.',
        parent_pipeline=bound(OLD / 'spec.json'), parent_training_complete=bound(OLD / 'runs/mopd/complete.json'),
        parent_integrity=bound(OLD / 'mopd_checkpoint768_verification.json'),
        parent_lineage=bound(PARENT_FD / 'training_lineage.json'), trigger=bound(O / 'public_trigger.json'),
        original_sft_retrained=False, parent_model_weights_inherited=True, new_stage_optimizers=True,
        full_generation_budgets_preserved=True)
    spec['rationale'] = ('Continue exact MOPD768 parent. Expand original RefSpatial natural-language point/depth pairs only on previously admitted images; '
        'finish coverage of the182878 vacancy records through continued permutation offsets. Retain all11 native forms, general QA, mask repair, '
        'and genuine full-budget on-policy MOPD512. No new GPU diagnostic arms, heldout training rows, metric changes or automatic acceptance.')
    spec['key_checkpoints'] = old['key_checkpoints'] + ['focus_repair/checkpoint-4096', 'focus_repair/checkpoint-8192',
        'focus_mopd/checkpoint-128', 'focus_mopd/checkpoint-256', 'focus_mopd/checkpoint-512']
    spec['key_checkpoint_sources'] = {name: str((OLD / 'runs' / name).resolve()) for name in old['key_checkpoints']}
    spec['key_checkpoint_sources'].update({name: str(O / 'runs' / name.removeprefix('focus_')) for name in spec['key_checkpoints'][9:]})
    # Budget uses measured completed updates by source role, with explicit proxies for the two short new sources.
    estimates = {}
    efficiency_evidence = {}
    old_schedules = read(OLD / 'sampling_config.json')['schedules']
    for stage in ('repair', 'mopd'):
        logs = [json.loads(line) for line in (OLD / f'runs/{stage}/logging.jsonl').open()]
        timings = defaultdict(list)
        for entry in logs:
            timings[old_schedules[stage][(entry['step'] - 1) % len(old_schedules[stage])]].append(entry['seconds'])
        means = {key: statistics.mean(values) for key, values in timings.items()}
        parent_mask = [row for row in logs if old_schedules[stage][(row['step'] - 1) % len(old_schedules[stage])] == 'mask']
        assert all(row['ul'] == 0 and row['cycle_fraction'] == 0 for row in parent_mask)
        efficiency_evidence[stage] = dict(log=bound(OLD / f'runs/{stage}/logging.jsonl'),
            mask_updates=len(parent_mask), generated_sequences=len(parent_mask) * 32,
            elapsed_mask_update_seconds=sum(row['seconds'] for row in parent_mask),
            nonzero_unlikelihood_updates=0, cycle_fraction=0,
            new_mask_supervision_updates=planned_roles[stage]['mask'] // 64,
            new_auxiliary_generation_updates=planned_roles[stage]['mask'] // (64 * 8),
            generation_token_cap_unchanged=768, native_mopd_generation_unchanged=True)
        proxies = {'grounding': 'vacancy', 'spatialqa': 'qa'}
        original_mask_seconds = means['mask']
        # Pure CE mask steps use the measured original point CE timing as a declared proxy.
        mask_ce_proxy = means['point']
        means['mask'] = mask_ce_proxy + (original_mask_seconds - mask_ce_proxy) / 8
        mean = statistics.mean(means[proxies.get(kind, kind)] for kind in module.SCHEDULES[stage])
        overhead = max(0, logs[-1]['elapsed'] - sum(row['seconds'] for row in logs)) / len(logs)
        estimates[stage] = dict(projected_role_seconds=means, parent_mask_seconds=original_mask_seconds,
            mask_ce_proxy_seconds=mask_ce_proxy, auxiliary_mask_generation_every=8, new_source_proxy=proxies,
            estimated_seconds_per_update=mean + overhead, planned_updates=steps[stage],
            estimated_hours=steps[stage] * (mean + overhead) / 3600)
    estimated_hours = sum(item['estimated_hours'] for item in estimates.values())
    remaining = (datetime.fromisoformat(spec['deadline_utc']).timestamp() - time.time()) / 3600
    assert remaining > estimated_hours + 8, 'Insufficient nominal budget; inspect before submission'
    spec['budget_estimate'] = dict(stages=estimates, projected_training_hours=estimated_hours,
        remaining_hours_at_preparation=remaining, validation_upload_reserve_hours=8,
        queue_and_long_generation_time_not_guaranteed=True, parent_mopd_updates_preserved=2304,
        new_mopd_updates=512)
    spec['auxiliary_generation_efficiency'] = efficiency_evidence
    old_gate = "is_mtd=stage=='mopd' and kind=='native';is_ul=kind=='mask' and micro==0"
    new_gate = "is_mtd=stage=='mopd' and kind=='native';is_ul=(kind=='mask' and micro==0 and (step//len(SCHEDULES[stage]))%cfg['ul_every_mask_steps']==0)"
    assert (O / 'train.py').read_text().replace(new_gate, old_gate) == (OLD / 'train.py').read_text(), 'Unexpected training implementation change'
    subprocess.run([str(TRAIN), str(O / 'test_resume_state.py')], check=True)
    for path in [*O.glob('*.py'), *H.glob('*.py'), C / 'focus_continuation_controller.py', C / 'focus_final_handoff.py']:
        ast.parse(path.read_text(), filename=str(path))
    # Existing sampler is exercised on every source at stage boundaries; no GPU is initialized.
    pool = module.Pool(spec)
    checked = 0
    for stage, count in steps.items():
        for step in list(range(16)) + list(range(count - 16, count)):
            for micro, rank in [(0, 0), (1, 31)]:
                row, kind, source = pool.sample(stage, step, micro, rank, 32, 64)
                assert row['gsa_meta']['split'] == 'train' and row['images'] and len(row['messages']) == 2
                assert row['messages'][0]['content'].startswith('<image>') and row['messages'][1]['content']
                assert source in planned_sources[stage]
                checked += 1
    import numpy as np
    vacancy_count = spec['datasets']['vacancy']['records']
    covered_through = offsets['mopd']['vacancy']
    assert initial_offsets['vacancy'] == 55296 and covered_through >= vacancy_count
    indices = [module.permute(i, vacancy_count, spec['sampling_seed']) for i in range(covered_through)]
    assert len(set(indices)) == vacancy_count
    validation = dict(state='passed_cpu_only', sampler_sha256=sha(O / 'pool.py'),
        boundary_samples_checked=checked, original_native_inventory=sum(map(len, pool.offsets.values())),
        native_forms_reachable=list(module.FORMS), prior_and_planned_vacancy_unique_records=vacancy_count,
        prior_and_planned_vacancy_exposures=covered_through, these_are_planned_not_observed_counts=True,
        source_admission=bound(supplement_path), resume_regression_checks_passed=True,
        training_losses_unchanged_except_auxiliary_generation_frequency=True, auxiliary_mask_rollout_every=8, gpu_diagnostic_jobs=0)
    assert validation['original_native_inventory'] == 4756000
    save_new(O / 'sampler_validation.json', validation)
    interface = read(H / 'interface_validation/rotation128_interface_validation.json')
    assert interface['state'] == 'passed' and interface['package_builder_sha256'] == sha(H / 'package.py')
    save_new(O / 'sampling_config.json', dict(schema='anvil-focus-sampling-v1', sampling_seed=spec['sampling_seed'],
        source_offsets=offsets, schedules=module.SCHEDULES, planned_steps=steps, native_forms=list(module.FORMS)))
    spec['frozen_files'].update({str(path.resolve()): sha(path) for path in [*O.glob('*.py'), O / 'schedules.json',
        O / 'public_trigger.json', O / 'sampling_config.json', O / 'sampler_validation.json',
        O / 'supplement_encoding_validation.json', supplement_path,
        C / 'prepare_refspatial_grounding_20261101.py', OLD / 'runs/mopd/complete.json',
        OLD / 'runs/mopd/protocol.json', PARENT_FD / 'training_lineage.json', report_path]})
    save_new(O / 'spec.json', spec)
    config = yaml.safe_load((OLD / 'cli.yaml').read_text())
    config['TaskName'] = 'task_train_anvil_grounding_continuation_A800x32'
    config['Description'] = 'Continue MOPD768: grounding/placement repair8192 then genuine MOPD512; preserve 9 parent +5 new key checkpoints'
    config['ActiveDeadlineSeconds'] = int((datetime.fromisoformat(spec['deadline_utc']).timestamp() - time.time()))
    config['Entrypoint'] = 'set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n' + shlex.join([
        str(TRAIN), '-u', str(O / 'node.py'), '--spec', str(O / 'spec.json'), '--spec-sha256', sha(O / 'spec.json')])
    with (O / 'cli.yaml').open('x') as stream:
        stream.write(yaml.safe_dump(config, sort_keys=False))
    FD.mkdir(parents=True, exist_ok=True)
    policy = dict(schema='anvil-direct-final-handoff-v1', pipeline_spec=str(O / 'spec.json'),
        pipeline_spec_sha256=sha(O / 'spec.json'), nodes=4, gpus=32,
        files={str(path): sha(path) for path in [C / 'focus_final_handoff.py', *H.glob('*.py'),
                                               H / 'interface_validation/cpu_validation.json']},
        deadline=datetime.fromisoformat(spec['deadline_utc']).timestamp(),
        final_quality_autoaccepted=False, key_checkpoints=spec['key_checkpoints'])
    save_new(FD / 'policy.json', policy)
    submit = dict(schema='anvil-targeted-continuation-submission-v1', gpus=32,
        spec=str(O / 'spec.json'), config=str(O / 'cli.yaml'), trigger=str(O / 'public_trigger.json'),
        parent_report=str(report_path), parent_evaluation_spec_sha256=report['final_spec_sha256'],
        parent_jobs=['t-20261101054208-ggqmg', 't-20261101054227-k6xrx', 't-20261101054246-xbkzr', 't-20261101054306-mf4tc'],
        handoff_source=str(C / 'focus_final_handoff.py'), handoff_policy=str(FD / 'policy.json'))
    submit['files'] = {str(path): sha(path) for path in [C / 'focus_continuation_controller.py',
        O / 'spec.json', O / 'cli.yaml', O / 'public_trigger.json', FD / 'policy.json', C / 'focus_final_handoff.py']}
    save_new(O / 'submit_policy.json', submit)
    print(json.dumps(dict(spec_sha256=sha(O / 'spec.json'), stages=spec['stages'], planned_sources=planned_sources,
                         budget=spec['budget_estimate'], key_checkpoints=len(spec['key_checkpoints']), submitted=False), indent=2))


if __name__ == '__main__':
    main()
