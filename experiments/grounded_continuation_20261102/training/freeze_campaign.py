"""Continue the evaluated balanced parent on complete admitted point pools.

The original requested deadline has elapsed. The new execution expiry is an
operational bound, not a claim that the original deadline was met.
"""
import ast
from collections import defaultdict
import copy
from datetime import datetime, timezone, timedelta
import importlib.util
import json
from pathlib import Path
import runpy
import shlex
import statistics
import subprocess
import time
import yaml

O = Path(__file__).resolve().parent
C = O.parent
R = C.parent
OLD = C / 'balanced_repair_20261102'
PFD = C / 'delivery_runs/balanced_repair_20261102'
H = C / 'final_delivery_grounded_v1'
FD = C / 'delivery_runs/grounded_repair_20261102'
TRAIN = R / '.venv-train/bin/python'
helpers = runpy.run_path(str(C / 'focus_repair_20261101/freeze_campaign.py'))
read, sha, bound, save_new, counts, advance = [helpers[k] for k in ('read', 'sha', 'bound', 'save_new', 'counts', 'advance')]


def main():
    assert not (O / 'spec.json').exists(), 'Never overwrite a frozen pipeline'
    old = read(OLD / 'spec.json')
    for path, digest in old['frozen_files'].items():
        assert sha(path) == digest, path
    complete = read(OLD / 'runs/mopd/complete.json')
    parent = Path(complete['checkpoint'])
    integrity_path = OLD / 'mopd_checkpoint512_verification.json'
    integrity = read(integrity_path)
    assert complete['steps'] == 512 and complete['stage'] == 'mopd_training_complete'
    assert sha(parent / 'adapter_model.safetensors') == complete['adapter_sha256']
    assert integrity['all_file_hashes_verified'] and integrity['optimizer_step'] == 512
    report_path = PFD / 'evaluation/final_report.json'
    report = read(report_path)
    assert report['state'] == 'complete_requires_quality_review'
    assert report['checkpoint'] == str(parent) and report['native']['n'] == 580000
    assert sum(x['n'] for x in report['public'].values()) == 7366
    assert sum(x['n'] for x in report['general'].values()) == 10105
    review = PFD / 'quality_review.json'
    assert read(review)['accepted'] is False and read(review)['report_sha256'] == sha(report_path)
    ancestor_backup = C / 'balanced_checkpoint_backup_20261102/mopd_512/verification_receipt.json'
    assert len(read(ancestor_backup)['key_checkpoints']) == 19
    assert read(ancestor_backup)['revision'] == '8eb2eead706097d838ac61e349c4121e25e5436b'

    modspec = importlib.util.spec_from_file_location('grounded_pool', O / 'pool.py')
    module = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(module)
    assert read(O / 'schedules.json') == module.SCHEDULES
    steps = dict(repair=8192, mopd=256)
    offset = advance(old['source_offsets']['mopd'], old['planned_source_exposures']['mopd'], old['planned_native_form_exposures']['mopd'])
    offsets, planned_sources, planned_forms, planned_roles = {}, {}, {}, {}
    for stage in ('repair', 'mopd'):
        offsets[stage] = offset
        sources, forms, roles = counts(module.SCHEDULES[stage], steps[stage], module.FORMS)
        offset = advance(offset, sources, forms)
        sources['original_native' if stage == 'repair' else 'native_distillation'] = sources.pop('native')
        planned_sources[stage], planned_forms[stage], planned_roles[stage] = map(dict, (sources, forms, roles))
    spec = copy.deepcopy(old)
    spec.update(initial=dict(adapter=str(parent), adapter_sha256=complete['adapter_sha256']),
                initialization_lineage=bound(PFD / 'training_lineage.json'),
                source_offsets=offsets, planned_source_exposures=planned_sources,
                planned_native_form_exposures=planned_forms, planned_role_exposures=planned_roles)
    spec['stages'] = dict(
        repair=dict(steps=8192, lr=2e-6, save_every=64, ul_weight=.1, ul_rollout_cap=768, ul_every_mask_steps=8),
        mopd=dict(steps=256, lr=2.5e-7, save_every=32, native_sft_weight=1., mtd_weight=.25,
                  ul_weight=.1, ul_rollout_cap=768, ul_every_mask_steps=8))
    spec['original_user_deadline_utc'] = old.get('original_user_deadline_utc', old['deadline_utc'])
    spec['original_deadline_missed'] = True
    spec['deadline_utc'] = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    spec['execution_expiry_note'] = '24h operational expiry for the already-authorized quality correction; original five-day deadline was missed, not extended retroactively.'
    spec['continuation_parent'] = dict(checkpoint=str(parent), adapter_sha256=complete['adapter_sha256'],
        lineage=bound(PFD/'training_lineage.json'), training_complete=bound(OLD/'runs/mopd/complete.json'), public_report=bound(report_path))
    rationale = ('Current full evaluation macro70.442857 with retained general ability, but material RefSpatial53.08/59, '
                 'Where2Place58.18/68, VABench72.67/80.67 localization/placement gaps. Cover all existing admitted '
                 'grounding97674, vacancy182878 and FSD29448 records with lower repair LR2e-6. Preserve native, '
                 'SAT and general replay. Then genuine MOPD256 at2.5e-7, all11 forms and original generation budgets. '
                 'Existing training annotations unchanged; no benchmark answers or evaluator changes. No guaranteed improvement.')
    trigger = dict(schema='anvil-complete-evaluation-continuation-trigger-v1', public_general_report=str(report_path),
        public_general_report_sha256=sha(report_path), scores=report['paper_aligned_public'], final_quality_accepted=False, rationale=rationale)
    save_new(O/'public_trigger.json', trigger)
    spec['recovery'] = dict(reason=rationale, parent_pipeline=bound(OLD/'spec.json'),
        parent_training_complete=bound(OLD/'runs/mopd/complete.json'), parent_integrity=bound(integrity_path),
        parent_lineage=bound(PFD/'training_lineage.json'), trigger=bound(O/'public_trigger.json'),
        original_sft_retrained=False, parent_model_weights_inherited=True, new_stage_optimizers=True,
        full_generation_budgets_preserved=True)
    spec['rationale'] = rationale + ' Original5.22M SFT and source composition disclosures remain unchanged. Inventory and exposures remain separate.'
    newkeys = ['grounded_repair/checkpoint-4096', 'grounded_repair/checkpoint-8192',
               'grounded_mopd/checkpoint-128', 'grounded_mopd/checkpoint-256']
    spec['key_checkpoints'] = old['key_checkpoints'] + newkeys
    spec['key_checkpoint_sources'].update({name:str(O/'runs'/name.removeprefix('grounded_')) for name in newkeys})
    assert len(spec['key_checkpoints']) == 23
    spec['retained_nonancestor_checkpoints'] = old['retained_nonancestor_checkpoints']
    spec['parent_quality_review'] = bound(review)
    # The unchanged permutation covers every pool member when exposures >= N.
    for name in ('grounding', 'vacancy', 'fsd', 'sat'):
        assert planned_sources['repair'][name] >= spec['datasets'][name]['records']

    estimates = {}
    for stage in steps:
        logs = [json.loads(line) for line in (OLD/f'runs/{stage}/logging.jsonl').open()]
        old_schedule = read(OLD/'schedules.json')[stage]
        durations = defaultdict(list)
        for row in logs:
            durations[old_schedule[(row['step']-1) % len(old_schedule)]].append(row['seconds'])
        means = {k:statistics.mean(v) for k,v in durations.items()}
        overhead = max(0, logs[-1]['elapsed']-sum(x['seconds'] for x in logs))/len(logs)
        per_update = statistics.mean(means[k] for k in module.SCHEDULES[stage])+overhead
        estimates[stage] = dict(source_log=bound(OLD/f'runs/{stage}/logging.jsonl'), measured_role_seconds=means,
            estimated_seconds_per_update=per_update, planned_updates=steps[stage], estimated_hours=steps[stage]*per_update/3600)
    spec['budget_estimate'] = dict(stages=estimates, projected_training_hours=sum(x['estimated_hours'] for x in estimates.values()),
        validation_upload_reserve_hours=2, queue_and_long_generation_time_not_guaranteed=True,
        original_user_deadline_missed=True, additional_gpu_ceiling=32)
    assert sha(O/'train.py') == sha(OLD/'train.py'), 'Training loss implementation must remain unchanged'
    for path in [*O.glob('*.py'), *H.glob('*.py'), C/'grounded_continuation_controller.py', C/'grounded_final_handoff.py']:
        ast.parse(path.read_text(), filename=str(path))
    pool = module.Pool(spec)
    checked = 0
    for stage, limit in steps.items():
        for step in [*range(16), *range(limit-16,limit)]:
            for micro,rank in [(0,0),(1,31)]:
                row,kind,source = pool.sample(stage, step, micro, rank, 32, 64)
                assert row['gsa_meta']['split']=='train' and len(row['images'])==1 and len(row['messages'])==2
                assert source in planned_sources[stage] and row['messages'][0]['content'].startswith('<image>')
                checked += 1
    coverage = {}
    for name in ('grounding','vacancy','fsd','sat'):
        n=spec['datasets'][name]['records']; exposures=planned_sources['repair'][name]
        unique=len({module.permute(i+offsets['repair'][name],n,spec['sampling_seed']) for i in range(exposures)})
        assert unique==n
        coverage[name]=dict(inventory=n,planned_exposures=exposures,planned_unique_records=unique)
    save_new(O/'sampler_validation.json',dict(state='passed_cpu_only',boundary_samples_checked=checked,
        original_native_inventory=sum(map(len,pool.offsets.values())),native_forms_reachable=list(module.FORMS),
        supplementary_pool_coverage=coverage,these_are_planned_not_observed_counts=True,
        training_loss_implementation_unchanged=True,gpu_diagnostic_jobs=0))
    interface = read(H/'interface_validation/rotation128_interface_validation.json')
    assert interface['state']=='passed' and interface['package_builder_sha256']==sha(H/'package.py')
    save_new(O/'sampling_config.json',dict(schema='anvil-grounded-sampling-v1',sampling_seed=spec['sampling_seed'],
        source_offsets=offsets,schedules=module.SCHEDULES,planned_steps=steps,native_forms=list(module.FORMS)))
    frozen = [*O.glob('*.py'),O/'schedules.json',O/'public_trigger.json',O/'sampling_config.json',O/'sampler_validation.json',
              OLD/'spec.json',OLD/'runs/mopd/complete.json',OLD/'runs/mopd/protocol.json',integrity_path,
              PFD/'training_lineage.json',report_path,review,ancestor_backup]
    spec['frozen_files'].update({str(path.resolve()):sha(path) for path in frozen})
    save_new(O/'spec.json',spec)
    config=yaml.safe_load((OLD/'cli.yaml').read_text())
    config['TaskName']='task_train_anvil_grounded_continuation_A800x32'
    config['Description']='Balanced MOPD512 parent; full admitted grounding/vacancy/FSD coverage at lower LR, repair8192 then genuine MOPD256;23 key backups'
    config['ActiveDeadlineSeconds']=86400
    config['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+shlex.join([str(TRAIN),'-u',str(O/'node.py'),'--spec',str(O/'spec.json'),'--spec-sha256',sha(O/'spec.json')])
    with (O/'cli.yaml').open('x') as stream:
        stream.write(yaml.safe_dump(config,sort_keys=False))
    handoff=dict(schema='anvil-direct-final-handoff-v1',pipeline_spec=str(O/'spec.json'),pipeline_spec_sha256=sha(O/'spec.json'),
        nodes=4,gpus=32,files={str(path):sha(path) for path in [C/'grounded_final_handoff.py',*H.glob('*.py'),H/'interface_validation/cpu_validation.json']},
        deadline=datetime.fromisoformat(spec['deadline_utc']).timestamp(),final_quality_autoaccepted=False,key_checkpoints=spec['key_checkpoints'])
    save_new(FD/'policy.json',handoff)
    jobs=[read(PFD/'submissions'/str(i)/'submission.json')['response']['Id'] for i in range(4)]
    policy=dict(schema='anvil-targeted-continuation-submission-v1',gpus=32,spec=str(O/'spec.json'),config=str(O/'cli.yaml'),
        trigger=str(O/'public_trigger.json'),parent_report=str(report_path),parent_evaluation_spec_sha256=report['final_spec_sha256'],
        parent_jobs=jobs,handoff_source=str(C/'grounded_final_handoff.py'),handoff_policy=str(FD/'policy.json'))
    policy['files']={str(path):sha(path) for path in [C/'grounded_continuation_controller.py',O/'spec.json',O/'cli.yaml',
        O/'public_trigger.json',FD/'policy.json',C/'grounded_final_handoff.py']}
    save_new(O/'submit_policy.json',policy)
    print(json.dumps(dict(spec_sha256=sha(O/'spec.json'),coverage=coverage,budget=spec['budget_estimate'],submitted=False)))


if __name__=='__main__':
    main()
