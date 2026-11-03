"""Freeze a new coordinate-objective campaign; never modify prior campaigns."""
import ast
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import sys
import time
import yaml

O=Path(__file__).resolve().parent;C=O.parent;R=C.parent
OLD=C/'geometry_repair_candidate_20261103';PARENT=C/'format_repair_20261103'
H=C/'final_delivery_geometry_objective_v1';FD=C/'delivery_runs/geometry_objective_20261104'
PD=C/'delivery_runs/geometry_repair_candidate_20261103'
sys.path.insert(0,str(O))
from native_io_pool import SCHEDULES,FORMS

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def counts(stage,steps):
    sources,forms,roles=Counter(),Counter(),Counter();schedule=SCHEDULES[stage]
    for step in range(steps):
        cycle,slot=divmod(step,len(schedule));kind=schedule[slot]
        occurrence=cycle*schedule.count(kind)+schedule[:slot].count(kind)
        if kind=='native':
            source='original_native' if stage=='repair' else 'native_distillation'
            forms[FORMS[occurrence%len(FORMS)]]+=64
        elif kind=='qa':source=('ai2d','scienceqa','textvqa')[occurrence%3]
        else:source={'depth':'depth_replay','point':'point_replay','mask':'mask_replay','affordance':'affordance_replay'}.get(kind,kind)
        sources[source]+=64;roles[kind]+=64
    return dict(sources),dict(forms),dict(roles)

def main():
    assert not (O/'spec.json').exists()
    spec=read(O/'candidate_settings.json')
    for p,d in spec['frozen_files'].items():assert sha(p)==d,p
    validation=read(O/'source_validation.json')
    assert validation['state']=='cpu_geometry_objective_source_checks_passed'
    for p,d in validation['files'].items():assert sha(p)==d,p
    assert read(O/'unit_test_result.json')['passed']==8
    report_path=PD/'evaluation/final_report.json';report=read(report_path)
    review_path=C/'geometry_delivery_control_20261103/quality_review.json';review=read(review_path)
    assert review['accepted'] is False
    assert report['native']['n']==580000 and sum(v['n'] for v in report['public'].values())==7366
    assert sum(v['n'] for v in report['general'].values())==10105
    assert sha(report_path)=='4f8547402087413a526ccd6bff2c788a779c340d85dbf838b8cd34a0902fd7af'
    parent=read(PARENT/'runs/mopd/complete.json')
    assert parent['checkpoint']==spec['initial']['adapter']
    assert parent['adapter_sha256']==spec['initial']['adapter_sha256']==sha(Path(parent['checkpoint'])/'adapter_model.safetensors')
    backup=C/'geometry_checkpoint_backup_20261103/mopd_256/verification_receipt.json'
    assert read(backup)['revision']=='6bca12d523756a32ac20e9eaf4943adb57a5975a'
    assert len(read(backup)['key_checkpoints'])==31
    spec['stages']['repair'].update(steps=8192,lr=5e-6,targeted_geometry_lr=2e-5,geometry_aux_weight=.5)
    spec['stages']['mopd'].update(steps=256,lr=2.5e-7,geometry_aux_weight=.25)
    for field in ['planned_source_exposures','planned_native_form_exposures','planned_role_exposures']:spec[field]={}
    for stage,cfg in spec['stages'].items():
        a,b,c=counts(stage,cfg['steps'])
        spec['planned_source_exposures'][stage]=a;spec['planned_native_form_exposures'][stage]=b;spec['planned_role_exposures'][stage]=c
    rationale=('The full geometry-replay experiment failed native quality: mask IoU regressed, '
      'affordance/rotation gains were negligible. Branch from the better fully evaluated format MOPD256 parent. '
      'Retain full-target CE, angular DoF supervision, all original data and all11 forms; add scalar-normalized '
      'coordinate CE emphasizing integer image coordinates and leading camera-position decimals. '
      'Only dedicated mask/affordance/PRISM steps use2e-5 peak LR; other repair steps retain5e-6. '
      'Repair8192 then genuine on-policy MOPD256 with unchanged teachers, rollout budgets and scorer. '
      'No GPU diagnostic arm. Effectiveness is an unproven optimization hypothesis requiring full evaluation.')
    spec['rationale']=rationale
    spec['geometry_objective']=dict(code=bound(O/'geometry_loss.py'),source_checks=bound(O/'source_validation.json'),
        tests=bound(O/'unit_test_result.json'),targets_modified=False,geometry_scoring_modified=False,
        full_answer_supervision_retained=True,effectiveness_proven=False)
    spec['retained_nonancestor_checkpoints'] += spec['key_checkpoints'][27:31]
    spec['rejected_experiment_reviews']['geometry_replay']=bound(review_path)
    spec['parent_quality_assessment']=bound(C/'format_eval_quota_recovery_20261103/quality_assessment_pending_scope.json')
    spec['recovery'].update(reason=rationale,parent_pipeline=bound(PARENT/'spec.json'),
        original_sft_retrained=False,parent_model_weights_inherited=True,new_stage_optimizers=True,
        rejected_replay_exposures_inherited=False)
    spec['point_contract_policy']['code']=bound(O/'point_contracts.py')
    spec['native_io_policy']['implementation']=bound(O/'native_io_pool.py')
    newkeys=['coordinate_repair/checkpoint-4096','coordinate_repair/checkpoint-8192',
             'coordinate_mopd/checkpoint-128','coordinate_mopd/checkpoint-256']
    spec['key_checkpoints']+=newkeys
    spec['key_checkpoint_sources'].update({key:str(O/'runs'/key.removeprefix('coordinate_')) for key in newkeys})
    estimates={}
    for stage,cfg in spec['stages'].items():
        log=OLD/f'runs/{stage}/logging.jsonl';rows=[json.loads(line) for line in log.open()]
        seconds=rows[-1]['elapsed']/len(rows)
        # The auxiliary projects only coordinate-token states. Allow 25%
        # overhead until the substantive job provides measured throughput.
        estimates[stage]=dict(source_log=bound(log),measured_seconds_per_update=seconds,
            overhead_allowance=1.25,updates=cfg['steps'],estimated_hours=seconds*1.25*cfg['steps']/3600)
    hours=sum(v['estimated_hours'] for v in estimates.values())
    spec['budget_estimate']=dict(stages=estimates,projected_training_hours=hours,
        validation_upload_reserve_hours=3,queue_and_long_generation_time_not_guaranteed=True,
        original_user_deadline_missed=True,additional_gpu_ceiling=32)
    assert hours+3<24
    deadline=datetime.now(timezone.utc)+timedelta(hours=24)
    spec['deadline_utc']=deadline.isoformat();spec['final_reserve_hours']=3
    spec['execution_expiry_note']='New24h operational window; original five-day user deadline remains missed. Not a promise of quality or completion.'
    trigger=dict(schema='anvil-complete-evaluation-continuation-trigger-v1',public_general_report=str(report_path),
        public_general_report_sha256=sha(report_path),scores=report['paper_aligned_public'],
        final_quality_accepted=False,rationale=rationale,rejected_review=bound(review_path))
    save(O/'public_trigger.json',trigger)
    spec['recovery']['trigger']=bound(O/'public_trigger.json')
    save(O/'schedules.json',SCHEDULES)
    interface=H/'interface_validation/cpu_validation.json'
    assert read(interface)['state']=='passed_cpu_only'
    frozen=[*O.glob('*.py'),*H.glob('*.py'),O/'candidate_settings.json',O/'source_validation.json',
        O/'unit_test_result.json',O/'schedules.json',O/'public_trigger.json',report_path,review_path,backup,
        interface,H/'interface_validation/rotation128_interface_validation.json']
    for p in [*O.glob('*.py'),*H.glob('*.py')]:ast.parse(p.read_text())
    spec['frozen_files'].update({str(p.resolve()):sha(p) for p in frozen})
    save(O/'spec.json',spec)
    config=yaml.safe_load((OLD/'cli.yaml').read_text())
    config['TaskName']='task_train_anvil_coordinate_objective_A800x32'
    config['Description']='Format parent; coordinate CE auxiliary and targeted geometry LR; repair8192 plus real MOPD256; full native/public/general evaluation'
    config['ActiveDeadlineSeconds']=86400
    config['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+shlex.join([str(R/'.venv-train/bin/python'),'-u',str(O/'node.py'),'--spec',str(O/'spec.json'),'--spec-sha256',sha(O/'spec.json')])
    with (O/'cli.yaml').open('x') as f:yaml.safe_dump(config,f,sort_keys=False)
    handoff=dict(schema='anvil-direct-final-handoff-v1',pipeline_spec=str(O/'spec.json'),pipeline_spec_sha256=sha(O/'spec.json'),
        nodes=4,gpus=32,files={str(p):sha(p) for p in [C/'coordinate_final_handoff.py',*H.glob('*.py'),interface]},
        deadline=deadline.timestamp(),final_quality_autoaccepted=False,key_checkpoints=spec['key_checkpoints'])
    save(FD/'policy.json',handoff)
    jobs=[read(PD/'submissions'/str(i)/'submission.json')['response']['Id'] for i in range(4)]
    policy=dict(schema='anvil-targeted-continuation-submission-v1',gpus=32,spec=str(O/'spec.json'),config=str(O/'cli.yaml'),
        trigger=str(O/'public_trigger.json'),parent_report=str(report_path),parent_evaluation_spec_sha256=report['final_spec_sha256'],
        parent_jobs=jobs,handoff_source=str(C/'coordinate_final_handoff.py'),handoff_policy=str(FD/'policy.json'))
    policy['files']={str(p):sha(p) for p in [C/'coordinate_continuation_controller.py',C/'coordinate_final_handoff.py',O/'spec.json',O/'cli.yaml',O/'public_trigger.json',FD/'policy.json']}
    save(O/'submit_policy.json',policy)
    print(json.dumps(dict(spec_sha256=sha(O/'spec.json'),training_estimate_hours=hours,deadline_utc=spec['deadline_utc'],
                          keys=35,submitted=False)))

if __name__=='__main__':main()
