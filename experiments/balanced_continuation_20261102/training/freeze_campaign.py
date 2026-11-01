"""Freeze a lower-LR SAT continuation from the stronger, fully evaluated parent."""
import ast
from collections import defaultdict
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import shlex
import statistics
import subprocess
import time
import yaml
O=Path(__file__).resolve().parent
C=O.parent
R=C.parent
OLD=C/'targeted_repair_20261031'
FOCUS=C/'focus_repair_20261101'
H=C/'final_delivery_balanced_v1'
PFD=C/'delivery_runs/targeted_repair_20261031'
FFD=C/'delivery_runs/focus_repair_20261101'
FD=C/'delivery_runs/balanced_repair_20261102'
TRAIN=R/'.venv-train/bin/python'
helpers=runpy.run_path(str(FOCUS/'freeze_campaign.py'))
read,sha,bound,save_new,counts,advance=[helpers[n] for n in ('read','sha','bound','save_new','counts','advance')]

def main():
    assert not (O/'spec.json').exists(), 'Never overwrite frozen sources'
    old=read(OLD/'spec.json'); focus=read(FOCUS/'spec.json')
    for path,digest in old['frozen_files'].items():
        assert sha(path)==digest,path
    complete=read(OLD/'runs/mopd/complete.json');parent=Path(complete['checkpoint'])
    assert complete['steps']==768 and complete['stage']=='mopd_training_complete'
    assert sha(parent/'adapter_model.safetensors')==complete['adapter_sha256']=='c7813b969b2aec281c962e79188e5dcaf335f4bb34be9c432a312fa221478a29'
    integrity=read(OLD/'mopd_checkpoint768_verification.json')
    assert integrity['all_file_hashes_verified'] and integrity['optimizer_step']==768
    parent_report=PFD/'evaluation/final_report.json'
    assert sha(parent_report)=='6fb7c10b0a83bc2c3cd9fb79c3c3669ce601bac1d0f3165f29d68f0caf450462'
    report_path=FFD/'evaluation/final_report.json';report=read(report_path)
    assert sha(report_path)=='9a3243b5b8f18d9b268bb6ff404c5cd6a841285e532503f2c54cd7539ed1c6b3'
    assert report['state']=='complete_requires_quality_review' and report['native']['n']==580000
    assert sum(x['n'] for x in report['general'].values())==10105
    rejection=FFD/'quality_review.json';assert read(rejection)['accepted'] is False
    sat_path=C/'sat_replay_20261102/manifest.json';sat=read(sat_path)
    ground_path=C/'refspatial_grounding_20261101/manifest.json';ground=read(ground_path)
    for m in (sat,ground):
        assert m['state']=='verified_train_only' and not m['evaluation_answers_consumed']
    encoding=read(O/'supplement_encoding_validation.json')
    assert encoding['state']=='passed_cpu_only' and encoding['manifest_sha256']==sha(sat_path)
    assert encoding['validator_sha256']==sha(O/'validate_supplement_encoding.py')
    modspec=importlib.util.spec_from_file_location('balanced_pool',O/'pool.py')
    module=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(module)
    assert read(O/'schedules.json')==module.SCHEDULES
    steps={'repair':2560,'mopd':512}
    initial_offsets=advance(old['source_offsets']['mopd'],old['planned_source_exposures']['mopd'],old['planned_native_form_exposures']['mopd'])
    initial_offsets.update(grounding=0,sat=0)
    offsets={};planned_sources={};planned_forms={};planned_roles={};offset=initial_offsets
    for stage in ('repair','mopd'):
        offsets[stage]=offset
        source,native,roles=counts(module.SCHEDULES[stage],steps[stage],module.FORMS)
        offset=advance(offset,source,native)
        source['original_native' if stage=='repair' else 'native_distillation']=source.pop('native')
        planned_sources[stage]=dict(source);planned_forms[stage]=dict(native);planned_roles[stage]=dict(roles)
    spec=copy.deepcopy(old)
    spec.update(initial=dict(adapter=str(parent),adapter_sha256=complete['adapter_sha256']),
        initialization_lineage=bound(PFD/'training_lineage.json'),source_offsets=offsets,
        planned_source_exposures=planned_sources,planned_native_form_exposures=planned_forms,
        planned_role_exposures=planned_roles,final_reserve_hours=2)
    spec['stages']=dict(repair=dict(steps=2560,lr=5e-6,save_every=64,ul_weight=.1,ul_rollout_cap=768,ul_every_mask_steps=8),
        mopd=dict(steps=512,lr=5e-7,save_every=32,native_sft_weight=1.,mtd_weight=.25,ul_weight=.1,ul_rollout_cap=768,ul_every_mask_steps=8))
    for name,item in {'grounding':ground['files']['grounding'],'sat':sat['files']['sat']}.items():
        assert sha(item['path'])==item['sha256']
        spec['datasets'][name]={key:item[key] for key in ('path','sha256','records')}
    assert 'spatialqa' not in spec['datasets']
    spec['supplemental_source_admission']=bound(sat_path)
    spec['additional_grounding_source_admission']=bound(ground_path)
    spec['continuation_parent']=dict(checkpoint=str(parent),adapter_sha256=complete['adapter_sha256'],
        lineage=bound(PFD/'training_lineage.json'),training_complete=bound(OLD/'runs/mopd/complete.json'),public_report=bound(parent_report))
    trigger=dict(schema='anvil-complete-evaluation-continuation-trigger-v1',public_general_report=str(report_path),
        public_general_report_sha256=sha(report_path),scores=report['paper_aligned_public'],final_quality_accepted=False,
        rationale='Focus branch completed without runtime crash but macro regressed70.1171->67.6957, especially EmbSpatial/BLINK. Branch from stronger MOPD768 parent using lower LR, source-original SAT multi-choice/spatial reasoning, native/FSD/general replay. No new GPU diagnostic arms or scorer changes. Attribution remains uncertain; no metric success guarantee.')
    save_new(O/'public_trigger.json',trigger)
    spec['recovery']=dict(reason='Lower-LR SAT replay from stronger evaluated MOPD768; rejected focus branch is not an ancestor.',
        parent_pipeline=bound(OLD/'spec.json'),parent_training_complete=bound(OLD/'runs/mopd/complete.json'),
        parent_integrity=bound(OLD/'mopd_checkpoint768_verification.json'),parent_lineage=bound(PFD/'training_lineage.json'),
        trigger=bound(O/'public_trigger.json'),original_sft_retrained=False,parent_model_weights_inherited=True,
        new_stage_optimizers=True,full_generation_budgets_preserved=True)
    spec['rationale']='Continue stronger parent with repair2560 at5e-6 then genuine MOPD512. SAT51322 admitted train records, protected image exclusion; source annotations/splits not independently certified. Original5.22M SFT retained. Original source composition differs from report; do not claim exact reproduction. No focus spatialqa replay; no rejected-focus exposures in final ancestry. Final acceptance requires full validation and honest quality review.'
    newkeys=['balanced_repair/checkpoint-1280','balanced_repair/checkpoint-2560','balanced_mopd/checkpoint-128','balanced_mopd/checkpoint-256','balanced_mopd/checkpoint-512']
    spec['key_checkpoints']=focus['key_checkpoints']+newkeys
    spec['key_checkpoint_sources']=dict(focus['key_checkpoint_sources'])
    spec['key_checkpoint_sources'].update({name:str(O/'runs'/name.removeprefix('balanced_')) for name in newkeys})
    spec['retained_nonancestor_checkpoints']=[name for name in focus['key_checkpoints'] if name.startswith('focus_')]
    spec['rejected_experiment_review']=bound(rejection)
    assert len(spec['key_checkpoints'])==19 and len(spec['retained_nonancestor_checkpoints'])==5
    estimates={}
    previous_schedules=read(FOCUS/'schedules.json')
    for stage in steps:
        logs=[json.loads(line) for line in (FOCUS/f'runs/{stage}/logging.jsonl').open()]
        timings=defaultdict(list)
        for entry in logs:timings[previous_schedules[stage][(entry['step']-1)%16]].append(entry['seconds'])
        means={key:statistics.mean(values) for key,values in timings.items()}
        proxies={'sat':'qa'}
        overhead=max(0,logs[-1]['elapsed']-sum(row['seconds'] for row in logs))/len(logs)
        per=statistics.mean(means[proxies.get(kind,kind)] for kind in module.SCHEDULES[stage])+overhead
        estimates[stage]=dict(source_log=bound(FOCUS/f'runs/{stage}/logging.jsonl'),measured_role_seconds=means,
            new_source_proxy=proxies,estimated_seconds_per_update=per,planned_updates=steps[stage],estimated_hours=steps[stage]*per/3600)
    hours=sum(v['estimated_hours'] for v in estimates.values())
    remaining=(datetime.fromisoformat(spec['deadline_utc']).timestamp()-time.time())/3600
    assert remaining>hours+2, ('Insufficient nominal budget',remaining,hours)
    spec['budget_estimate']=dict(stages=estimates,projected_training_hours=hours,remaining_hours_at_preparation=remaining,
        validation_upload_reserve_hours=2,queue_and_long_generation_time_not_guaranteed=True,parent_mopd_updates_preserved=2304,new_mopd_updates=512)
    spec['auxiliary_generation_efficiency']=focus['auxiliary_generation_efficiency']
    assert sha(O/'train.py')==sha(FOCUS/'train.py')
    subprocess.run([str(TRAIN),str(O/'test_resume_state.py')],check=True)
    for path in [*O.glob('*.py'),*H.glob('*.py'),C/'balanced_continuation_controller.py',C/'balanced_final_handoff.py']:ast.parse(path.read_text(),filename=str(path))
    pool=module.Pool(spec);checked=0
    for stage,count in steps.items():
        for step in list(range(16))+list(range(count-16,count)):
            for micro,rank in [(0,0),(1,31)]:
                row,kind,source=pool.sample(stage,step,micro,rank,32,64)
                assert row['gsa_meta']['split']=='train' and len(row['images'])==1 and len(row['messages'])==2
                assert row['messages'][0]['content'].startswith('<image>') and row['messages'][1]['content'] and source in planned_sources[stage]
                checked+=1
    sat_count=spec['datasets']['sat']['records'];sat_exposures=planned_sources['repair']['sat']
    assert sat_exposures>=sat_count
    assert len({module.permute(i,sat_count,spec['sampling_seed']) for i in range(sat_exposures)})==sat_count
    validation=dict(state='passed_cpu_only',sampler_sha256=sha(O/'pool.py'),boundary_samples_checked=checked,
        original_native_inventory=sum(map(len,pool.offsets.values())),native_forms_reachable=list(module.FORMS),
        planned_sat_unique_records=sat_count,planned_sat_repair_exposures=sat_exposures,these_are_planned_not_observed_counts=True,
        source_admission=bound(sat_path),resume_regression_checks_passed=True,training_loss_implementation_unchanged=True,gpu_diagnostic_jobs=0)
    assert validation['original_native_inventory']==4756000
    save_new(O/'sampler_validation.json',validation)
    interface=read(H/'interface_validation/rotation128_interface_validation.json')
    assert interface['state']=='passed' and interface['package_builder_sha256']==sha(H/'package.py')
    save_new(O/'sampling_config.json',dict(schema='anvil-balanced-sampling-v1',sampling_seed=spec['sampling_seed'],source_offsets=offsets,
        schedules=module.SCHEDULES,planned_steps=steps,native_forms=list(module.FORMS)))
    spec['frozen_files'].update({str(path.resolve()):sha(path) for path in [*O.glob('*.py'),O/'schedules.json',O/'public_trigger.json',
        O/'sampling_config.json',O/'sampler_validation.json',O/'supplement_encoding_validation.json',sat_path,ground_path,
        C/'prepare_sat_replay_20261102.py',OLD/'runs/mopd/complete.json',OLD/'runs/mopd/protocol.json',
        PFD/'training_lineage.json',parent_report,report_path,rejection]})
    save_new(O/'spec.json',spec)
    config=yaml.safe_load((FOCUS/'cli.yaml').read_text())
    config['TaskName']='task_train_anvil_balanced_continuation_A800x32'
    config['Description']='MOPD768 parent -> low-LR SAT repair2560 -> genuine MOPD512; retain19 key checkpoints, 5 rejected-focus nonancestors'
    config['ActiveDeadlineSeconds']=int(datetime.fromisoformat(spec['deadline_utc']).timestamp()-time.time())
    config['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+shlex.join([str(TRAIN),'-u',str(O/'node.py'),'--spec',str(O/'spec.json'),'--spec-sha256',sha(O/'spec.json')])
    with (O/'cli.yaml').open('x') as stream:stream.write(yaml.safe_dump(config,sort_keys=False))
    policy=dict(schema='anvil-direct-final-handoff-v1',pipeline_spec=str(O/'spec.json'),pipeline_spec_sha256=sha(O/'spec.json'),nodes=4,gpus=32,
        files={str(path):sha(path) for path in [C/'balanced_final_handoff.py',*H.glob('*.py'),H/'interface_validation/cpu_validation.json']},
        deadline=datetime.fromisoformat(spec['deadline_utc']).timestamp(),final_quality_autoaccepted=False,key_checkpoints=spec['key_checkpoints'])
    save_new(FD/'policy.json',policy)
    submit=dict(schema='anvil-targeted-continuation-submission-v1',gpus=32,spec=str(O/'spec.json'),config=str(O/'cli.yaml'),trigger=str(O/'public_trigger.json'),
        parent_report=str(report_path),parent_evaluation_spec_sha256=report['final_spec_sha256'],
        parent_jobs=['t-20261101231815-j4jgt','t-20261101231845-9d5z9','t-20261101231914-j5xpk','t-20261101231941-z2bkj'],
        handoff_source=str(C/'balanced_final_handoff.py'),handoff_policy=str(FD/'policy.json'))
    submit['files']={str(path):sha(path) for path in [C/'balanced_continuation_controller.py',O/'spec.json',O/'cli.yaml',O/'public_trigger.json',FD/'policy.json',C/'balanced_final_handoff.py']}
    save_new(O/'submit_policy.json',submit)
    print(json.dumps(dict(spec_sha256=sha(O/'spec.json'),stages=spec['stages'],planned_sources=planned_sources,budget=spec['budget_estimate'],key_checkpoints=19,submitted=False),indent=2))

if __name__=='__main__':main()
