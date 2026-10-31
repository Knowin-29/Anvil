"""Freeze a substantive continuation and its complete final-delivery configuration."""
from collections import Counter
from datetime import datetime
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import yaml

O = Path(__file__).resolve().parent
C = O.parent
R = C.parent
OLD = C/'direct_recovery_20261030'
H = C/'final_delivery_targeted_v1'
FD = C/'delivery_runs/targeted_repair_20261031'
PARENT_FD = C/'delivery_runs/direct_recovery_20261030'


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


def main():
    assert not (O/'spec.json').exists(), 'Do not change a frozen training specification'
    old = read(OLD/'spec.json')
    for path, digest in old['frozen_files'].items():
        assert sha(path) == digest, path
    complete = read(OLD/'runs/mopd/complete.json')
    parent = Path(complete['checkpoint'])
    assert complete['steps'] == 1536 and complete['stage'] == 'mopd_training_complete'
    assert sha(parent/'adapter_model.safetensors') == complete['adapter_sha256']
    integrity = read(OLD/'mopd_checkpoint1536_verification.json')
    assert integrity['all_file_hashes_verified'] and integrity['optimizer_step'] == 1536
    lineage = read(PARENT_FD/'training_lineage.json')
    assert lineage['state'] == 'complete' and lineage['checkpoint'] == str(parent)
    trigger = read(O/'public_trigger.json')
    assert sha(trigger['public_general_report']) == trigger['public_general_report_sha256']
    assert trigger['final_quality_accepted'] is False
    sampling = read(O/'sampling_config.json')
    validation = read(O/'sampler_validation.json')
    assert validation['sampler_sha256'] == sha(O/'pool.py')
    assert validation['actual_unique_depth_records_in_planned_schedule'] == 14849
    # Parent throughput leaves approximately one day for final evaluation/upload.
    # The sampler's repair coverage validation depends on4096; its MOPD prefix is unchanged.
    sampling['planned_steps']['mopd'] = 768
    (O/'sampling_config.json').write_text(json.dumps(sampling, indent=2)+'\n')
    module_spec = importlib.util.spec_from_file_location('candidate_pool', O/'pool.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert sampling['schedules'] == module.SCHEDULES
    subprocess.run([sys.executable, str(O/'test_resume_state.py')], check=True)
    for p in [*O.glob('*.py'), *H.glob('*.py'), C/'targeted_final_handoff.py']:
        ast.parse(p.read_text(), filename=str(p))
    inherited = (OLD/'train.py').read_text()
    prepared = (O/'train.py').read_text()
    start = '            for micro in range(accum):'
    end = "            norm=torch.nn.utils.clip_grad_norm_(trainable,1.);assert torch.isfinite(norm)"
    assert inherited.split(start)[1].split(end)[0] == prepared.split(start)[1].split(end)[0], 'Training loss implementation changed unexpectedly'
    interface = read(H/'interface_validation/cpu_validation.json')
    assert interface['state'] == 'passed_cpu_only'
    assert interface['interface_parity']['rotation128']['package_builder_sha256'] == sha(H/'package.py')
    planned_sources, planned_native_forms, role_counts = {}, {}, {}
    for stage, steps in sampling['planned_steps'].items():
        counts, forms, roles = Counter(), Counter(), Counter()
        schedule = module.SCHEDULES[stage]
        for step in range(steps):
            cycle, slot = divmod(step, len(schedule))
            kind = schedule[slot]
            occurrence = cycle*schedule.count(kind)+schedule[:slot].count(kind)
            if kind == 'native':
                source = 'native_distillation' if stage == 'mopd' else 'original_native'
                forms[module.FORMS[occurrence % len(module.FORMS)]] += 64
            elif kind == 'qa':
                source = ('ai2d','scienceqa','textvqa')[occurrence % 3]
            else:
                source = {'depth':'depth_replay','point':'point_replay','mask':'mask_replay'}.get(kind, kind)
            counts[source] += 64
            roles[kind] += 64
        assert sum(counts.values()) == steps*64
        planned_sources[stage] = dict(counts)
        planned_native_forms[stage] = dict(forms)
        role_counts[stage] = dict(roles)
    spec = copy.deepcopy(old)
    spec['initial'] = dict(adapter=str(parent), adapter_sha256=complete['adapter_sha256'])
    spec['initialization_lineage'] = bound(PARENT_FD/'training_lineage.json')
    spec['continuation_parent'] = dict(checkpoint=str(parent), adapter_sha256=complete['adapter_sha256'],
        lineage=bound(PARENT_FD/'training_lineage.json'), training_complete=bound(OLD/'runs/mopd/complete.json'),
        public_report=bound(PARENT_FD/'evaluation/public_general_report.json'))
    spec['recovery'] = dict(reason='Substantive corrective continuation after the completed parent public evaluation fell short.',
        parent_pipeline=bound(OLD/'spec.json'), parent_training_complete=bound(OLD/'runs/mopd/complete.json'),
        parent_integrity=bound(OLD/'mopd_checkpoint1536_verification.json'), parent_lineage=spec['initialization_lineage'],
        trigger=bound(O/'public_trigger.json'), original_sft_retrained=False, parent_model_weights_inherited=True,
        new_stage_optimizers=True, full_generation_budgets_preserved=True)
    spec['sampling_seed'] = sampling['sampling_seed']
    spec['source_offsets'] = sampling['source_offsets']
    spec['balanced_native_forms'] = list(module.FORMS)
    spec['stages'] = dict(
        repair=dict(steps=4096, lr=1e-5, save_every=64, ul_weight=.1, ul_rollout_cap=768),
        mopd=dict(steps=768, lr=1e-6, save_every=32, native_sft_weight=1., mtd_weight=.25,
                  ul_weight=.1, ul_rollout_cap=768))
    spec['planned_source_exposures'] = planned_sources
    spec['planned_native_form_exposures'] = planned_native_forms
    spec['planned_role_exposures'] = role_counts
    spec['key_checkpoints'] = ['ancestor_repair/checkpoint-2048',
        'ancestor_mopd/checkpoint-512', 'ancestor_mopd/checkpoint-1024', 'ancestor_mopd/checkpoint-1536',
        'repair/checkpoint-2048', 'repair/checkpoint-4096',
        'mopd/checkpoint-256', 'mopd/checkpoint-512', 'mopd/checkpoint-768']
    spec['teacher_assessment']['decision'] = ('Reuse the bound G/S/A teachers as imperfect distillation references. '
        'Increase native ground-truth supervision and reduce the distillation coefficient; no universal teacher superiority claim.')
    spec['rationale'] = ('Parent macro69.367vs74.27; depth received only101/49152 native exposures. '
        'Cover all14849 existing train-only depth records in corrective replay, balance all11 native forms, '
        'continue supervised placement/point/QA/mask/PRISM replay, and retain genuine full-budget on-policy MOPD. '
        'Lower continuation learning rates and stronger reference supervision reduce regression risk. '
        'No new GPU diagnostic arms, new benchmark training records, scorer changes or quality autoacceptance.')
    spec['budget_estimate'] = dict(parent_repair_seconds_per_update=9.65325,
        parent_mopd_seconds_per_update=59.4, projected_training_hours=(4096*9.65325+768*59.4)/3600,
        generation_variability_and_queue_time_not_guaranteed=True, validation_upload_reserve_hours=24,
        new_mopd_updates=768, parent_mopd_updates_preserved=1536)
    save_new(O/'cpu_training_validation.json', dict(state='passed_cpu_only',
        inherited_loss_implementation_unchanged=True, resume_prefix_regression_checks_passed=True,
        sampler_validation=bound(O/'sampler_validation.json'), full_generation_budgets_preserved=True,
        python_syntax_valid=True, gpu_diagnostic_tasks=0,
        real_training_checkpoint_and_final_quality_not_yet_verified=True, created=time.time()))
    spec['frozen_files'].update({str(p.resolve()):sha(p) for p in [*O.glob('*.py'),
        O/'sampling_config.json', O/'sampler_validation.json', O/'cpu_training_validation.json', O/'public_trigger.json',
        OLD/'runs/mopd/complete.json', OLD/'runs/mopd/protocol.json',
        PARENT_FD/'training_lineage.json', PARENT_FD/'evaluation/public_general_report.json']})
    (O/'runs').mkdir(exist_ok=True)
    for name, source in [('ancestor_repair',OLD/'runs/repair'), ('ancestor_mopd',OLD/'runs/mopd')]:
        link = O/'runs'/name
        if link.is_symlink():
            assert link.resolve() == source.resolve()
        else:
            link.symlink_to(source.resolve(), target_is_directory=True)
    save_new(O/'spec.json', spec)
    config = yaml.safe_load((OLD/'cli.yaml').read_text())
    config['TaskName'] = 'task_train_anvil_targeted_continuation_A800x32'
    config['Description'] = 'Continue parentMOPD1536: balanced repair4096 then genuineMOPD768; full depth replay; complete final evaluation and key checkpoints'
    config['ActiveDeadlineSeconds'] = 129600
    config['Entrypoint'] = 'set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n' + shlex.join([
        str(R/'.venv-train/bin/python'), '-u', str(O/'node.py'), '--spec', str(O/'spec.json'),
        '--spec-sha256', sha(O/'spec.json')])
    with (O/'cli.yaml').open('x') as stream:
        stream.write(yaml.safe_dump(config, sort_keys=False))
    FD.mkdir(parents=True, exist_ok=True)
    policy = dict(schema='anvil-direct-final-handoff-v1', pipeline_spec=str(O/'spec.json'),
        pipeline_spec_sha256=sha(O/'spec.json'), nodes=4, gpus=32,
        files={str(p):sha(p) for p in [C/'targeted_final_handoff.py', *H.glob('*.py'),
            H/'interface_validation/cpu_validation.json', O/'cpu_training_validation.json']},
        deadline=datetime.fromisoformat(spec['deadline_utc']).timestamp(),
        final_quality_autoaccepted=False, key_checkpoints=spec['key_checkpoints'])
    save_new(FD/'policy.json', policy)
    print(json.dumps(dict(spec=str(O/'spec.json'), spec_sha256=sha(O/'spec.json'),
        stages=spec['stages'], planned_source_exposures=planned_sources,
        planned_native_form_exposures=planned_native_forms, budget=spec['budget_estimate'],
        dlc_config=str(O/'cli.yaml'), handoff_policy=str(FD/'policy.json'), submitted=False), indent=2))


if __name__ == '__main__':
    main()
