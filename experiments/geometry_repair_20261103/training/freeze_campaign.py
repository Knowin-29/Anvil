"""Seal a geometry-focused continuation from the fully evaluated format MOPD model."""
import ast,copy,hashlib,json,shlex,statistics,sys,time
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import yaml
O=Path(__file__).resolve().parent;C=O.parent;R=C.parent;OLD=C/'format_repair_20261103'
Q=C/'format_eval_quota_recovery_20261103';PFD=C/'delivery_runs/format_repair_20261103'
H=C/'final_delivery_geometry_v1';FD=C/'delivery_runs/geometry_repair_candidate_20261103'
TRAIN=R/'.venv-train/bin/python'
sys.path.insert(0,str(O))
from native_io_pool import SCHEDULES,FORMS

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def counts(stage,steps):
 sources,forms,roles=Counter(),Counter(),Counter()
 schedule=SCHEDULES[stage]
 for step in range(steps):
  cycle,slot=divmod(step,len(schedule));kind=schedule[slot]
  occurrence=cycle*schedule.count(kind)+schedule[:slot].count(kind)
  if kind=='native':
   source='original_native' if stage=='repair' else 'native_distillation';forms[FORMS[occurrence%len(FORMS)]]+=64
  elif kind=='qa':source=('ai2d','scienceqa','textvqa')[occurrence%3]
  else:source={'depth':'depth_replay','mask':'mask_replay','point':'point_replay','affordance':'affordance_replay'}.get(kind,kind)
  sources[source]+=64;roles[kind]+=64
 return dict(sources),dict(forms),dict(roles)

def main():
 assert not (O/'spec.json').exists(),'Never overwrite a frozen campaign'
 old=read(OLD/'spec.json');cfg=read(O/'sampling_config.json')
 for path,digest in old['frozen_files'].items():assert sha(path)==digest,path
 validation=read(O/'sampler_validation.json');assert validation['records_checked']==262144 and validation['all16544_admitted_prism_records_reachable'] and validation['all5907_admitted_affordance_records_reachable']
 for path,digest in validation['files'].items():assert sha(path)==digest,path
 io=read(O/'native_io_validation.json');assert io['state']=='exact_indexed_read_equivalence_passed'
 for path,digest in io['files'].items():assert sha(path)==digest,path
 complete=read(OLD/'runs/mopd/complete.json');parent=Path(complete['checkpoint'])
 assert complete['steps']==256 and complete['stage']=='mopd_training_complete' and sha(parent/'adapter_model.safetensors')==complete['adapter_sha256']
 integrity=OLD/'mopd_checkpoint256_verification.json';assert read(integrity)['all_file_hashes_verified'] and read(integrity)['optimizer_step']==256
 report_path=Q/'evaluation/final_report.json';report=read(report_path)
 assert report['checkpoint']==str(parent) and report['state']=='complete_requires_quality_review'
 assert report['native']['n']==580000 and sum(x['n'] for x in report['public'].values())==7366 and sum(x['n'] for x in report['general'].values())==10105
 assert read(Q/'status.json')['stage']=='full_final_report_ready_requires_quality_review'
 backup=C/'format_checkpoint_backup_20261103/mopd_256/verification_receipt.json';receipt=read(backup)
 assert len(receipt['key_checkpoints'])==27 and receipt['revision']=='e5b8da0e1c9c5346d28359405b98c21713b81511'
 assert [r['path'] for r in receipt['key_checkpoints']]==old['key_checkpoints']
 stages=dict(repair=4096,mopd=256);spec=copy.deepcopy(old)
 spec['initial']=dict(adapter=str(parent),adapter_sha256=complete['adapter_sha256'])
 spec['initialization_lineage']=bound(PFD/'training_lineage.json')
 spec['source_offsets']=cfg['source_offsets'];spec['sampling_seed']=cfg['sampling_seed']
 spec['planned_source_exposures']={};spec['planned_native_form_exposures']={};spec['planned_role_exposures']={}
 for stage,steps in stages.items():
  a,b,c=counts(stage,steps);spec['planned_source_exposures'][stage]=a;spec['planned_native_form_exposures'][stage]=b;spec['planned_role_exposures'][stage]=c
 spec['stages']['repair'].update(steps=4096,lr=5e-6)
 spec['stages']['mopd'].update(steps=256,lr=2.5e-7)
 spec['continuation_parent']=dict(checkpoint=str(parent),adapter_sha256=complete['adapter_sha256'],lineage=bound(PFD/'training_lineage.json'),training_complete=bound(OLD/'runs/mopd/complete.json'),public_report=bound(report_path))
 rationale=('Continue the exact format MOPD256 parent after full580000 native,7366public and10105general evaluation. '
 'Public macro73.481429 vs paper74.267143 is close, but native maskIoU.158590, affordanceIoU.126352 and DoFrotation111.553995deg remain weak. '
 'Previous format repair had5952 exposures each of mask/affordance/DoF and no dedicated expanded-PRISM sampling. '
 'Repair4096 at5e-6 allocates65536 direct mask,32768 PRISM covering all16544 admitted rows,16384 affordance covering all5907 rows, '
 'plus balanced all11 native forms, unchanged point-format grounding/vacancy/FSD, SAT and general QA replay. '
 'Then genuine MOPD256 with all11 forms, existing objective/teachers and generation budgets. '
 'Exact-length indexed reads avoid shared-filesystem read amplification without changing selected records or annotations. '
 'Coverage limitation is not established as the sole cause; improvement is not guaranteed. No small diagnostic GPU jobs or heldout-label training.')
 trigger=dict(schema='anvil-complete-evaluation-continuation-trigger-v1',public_general_report=str(report_path),public_general_report_sha256=sha(report_path),scores=report['paper_aligned_public'],final_quality_accepted=False,rationale=rationale)
 save(O/'public_trigger.json',trigger)
 spec['rationale']=rationale+' Original5.22M SFT,5.8M corpus and source-mixture disclosures remain unchanged.'
 spec['recovery']=dict(reason=rationale,parent_pipeline=bound(OLD/'spec.json'),parent_training_complete=bound(OLD/'runs/mopd/complete.json'),parent_integrity=bound(integrity),parent_lineage=bound(PFD/'training_lineage.json'),trigger=bound(O/'public_trigger.json'),original_sft_retrained=False,parent_model_weights_inherited=True,new_stage_optimizers=True,full_generation_budgets_preserved=True)
 spec.pop('parent_quality_review',None);spec['parent_quality_assessment']=bound(Q/'quality_assessment_pending_scope.json')
 spec['point_contract_policy']['code']=bound(O/'point_contracts.py')
 spec['execution_expiry_note']='Retain the preceding operational expiry; the original five-day deadline remains missed.'
 newkeys=['geometry_repair/checkpoint-2048','geometry_repair/checkpoint-4096','geometry_mopd/checkpoint-128','geometry_mopd/checkpoint-256']
 spec['key_checkpoints']=old['key_checkpoints']+newkeys
 spec['key_checkpoint_sources']=dict(old['key_checkpoint_sources'])
 spec['key_checkpoint_sources'].update({k:str(O/'runs'/k.removeprefix('geometry_')) for k in newkeys})
 assert len(spec['key_checkpoints'])==31
 spec['native_io_policy']=dict(implementation=bound(O/'native_io_pool.py'),equivalence=bound(O/'native_io_validation.json'),gpu_throughput_improvement_not_yet_measured=True)
 # Conservative estimates from actual completed role timings; no promised speedup from I/O changes.
 estimates={}
 old_schedules=read(OLD/'schedules.json')
 for stage,steps in stages.items():
  rows=[json.loads(x) for x in (OLD/f'runs/{stage}/logging.jsonl').open()];times=defaultdict(list)
  schedule=old_schedules[stage]
  for row in rows:
   step=row['step']-1;cycle,slot=divmod(step,len(schedule));kind=schedule[slot];times[kind].append(row['seconds'])
   if kind=='native':
    occurrence=cycle*schedule.count(kind)+schedule[:slot].count(kind);times['native:'+FORMS[occurrence%len(FORMS)]].append(row['seconds'])
  means={k:statistics.mean(v) for k,v in times.items()}
  if stage=='repair':
   means['prism']=means['native:dof'];means['affordance']=means['native:affordance']
   means['mask']=max(means['native:mask'],old['budget_estimate']['stages']['repair']['measured_role_seconds']['mask'])
  overhead=max(0,rows[-1]['elapsed']-sum(r['seconds'] for r in rows))/len(rows)
  seconds=statistics.mean(means[k] for k in SCHEDULES[stage])+overhead
  estimates[stage]=dict(source_log=bound(OLD/f'runs/{stage}/logging.jsonl'),measured_role_seconds=means,estimated_seconds_per_update=seconds,planned_updates=steps,estimated_hours=steps*seconds/3600)
 hours=sum(x['estimated_hours'] for x in estimates.values())
 spec['budget_estimate']=dict(stages=estimates,projected_training_hours=hours,validation_upload_reserve_hours=2,queue_and_long_generation_time_not_guaranteed=True,original_user_deadline_missed=True,additional_gpu_ceiling=32)
 deadline=datetime.fromisoformat(spec['deadline_utc']).timestamp()
 assert time.time()+(hours+2)*3600<deadline,'Insufficient operational budget; review before sealing'
 assert (O/'train.py').read_text().replace('from native_io_pool import Pool,SCHEDULES,FORMS','from pool import Pool,SCHEDULES,FORMS')==(OLD/'train.py').read_text()
 interface=read(H/'interface_validation/rotation128_interface_validation.json');assert interface['state']=='passed' and interface['package_builder_sha256']==sha(H/'package.py')
 for p in [*O.glob('*.py'),*H.glob('*.py'),C/'geometry_continuation_controller.py',C/'geometry_final_handoff.py']:ast.parse(p.read_text())
 frozen=[*O.glob('*.py'),*H.glob('*.py'),O/'schedules.json',O/'sampling_config.json',O/'sampler_validation.json',O/'native_io_validation.json',O/'training_io_source_validation.json',O/'public_trigger.json',OLD/'spec.json',OLD/'runs/mopd/complete.json',OLD/'runs/mopd/protocol.json',integrity,report_path,Q/'quality_assessment_pending_scope.json',PFD/'training_lineage.json',backup,H/'interface_validation/rotation128_interface_validation.json']
 spec['frozen_files'].update({str(p.resolve()):sha(p) for p in frozen});save(O/'spec.json',spec)
 config=yaml.safe_load((OLD/'cli.yaml').read_text());config['TaskName']='task_train_anvil_geometry_continuation_A800x32'
 config['Description']='Format MOPD256 parent; full admitted PRISM and affordance coverage, stronger mask replay; repair4096 then genuine MOPD256; full evaluation and31 key backups'
 config['ActiveDeadlineSeconds']=max(1,int(deadline-time.time()))
 config['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+shlex.join([str(TRAIN),'-u',str(O/'node.py'),'--spec',str(O/'spec.json'),'--spec-sha256',sha(O/'spec.json')])
 with (O/'cli.yaml').open('x') as f:f.write(yaml.safe_dump(config,sort_keys=False))
 handoff=dict(schema='anvil-direct-final-handoff-v1',pipeline_spec=str(O/'spec.json'),pipeline_spec_sha256=sha(O/'spec.json'),nodes=4,gpus=32,files={str(p):sha(p) for p in [C/'geometry_final_handoff.py',*H.glob('*.py'),H/'interface_validation/cpu_validation.json']},deadline=deadline,final_quality_autoaccepted=False,key_checkpoints=spec['key_checkpoints'])
 save(FD/'policy.json',handoff)
 jobs=[read(Q/'submissions'/str(i)/'submission.json')['response']['Id'] for i in range(4)]
 policy=dict(schema='anvil-targeted-continuation-submission-v1',gpus=32,spec=str(O/'spec.json'),config=str(O/'cli.yaml'),trigger=str(O/'public_trigger.json'),parent_report=str(report_path),parent_evaluation_spec_sha256=report['final_spec_sha256'],parent_jobs=jobs,handoff_source=str(C/'geometry_final_handoff.py'),handoff_policy=str(FD/'policy.json'))
 policy['files']={str(p):sha(p) for p in [C/'geometry_continuation_controller.py',O/'spec.json',O/'cli.yaml',O/'public_trigger.json',FD/'policy.json',C/'geometry_final_handoff.py']};save(O/'submit_policy.json',policy)
 print(json.dumps(dict(spec_sha256=sha(O/'spec.json'),projected_training_hours=hours,original_deadline_missed=True,operational_expiry_utc=spec['deadline_utc'],key_checkpoints=31,submitted=False)))
if __name__=='__main__':main()
