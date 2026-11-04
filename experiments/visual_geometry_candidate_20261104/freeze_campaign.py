"""Freeze visual/language continuation and full evaluation without altering ancestors."""
import ast
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib,json,shlex,sys,time
from pathlib import Path
import yaml
V=Path(__file__).resolve().parent;C=V.parent;R=C.parent
O=C/'geometry_objective_v2_20261104';D=C/'delivery_runs/geometry_objective_v2_20261104'
H=C/'final_delivery_visual_geometry_v1';FD=C/'delivery_runs/visual_geometry_20261104'
sys.path.insert(0,str(V))
from native_io_pool import SCHEDULES,FORMS

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def counts(stage,steps):
 sources,forms,roles,offsets=Counter(),Counter(),Counter(),Counter();schedule=SCHEDULES[stage]
 for step in range(steps):
  cycle,slot=divmod(step,len(schedule));kind=schedule[slot];occurrence=cycle*schedule.count(kind)+schedule[:slot].count(kind)
  if kind=='native':
   source='original_native' if stage=='repair' else 'native_distillation';form=FORMS[occurrence%len(FORMS)];forms[form]+=64;key='native:'+form
  elif kind=='qa':source=('ai2d','scienceqa','textvqa')[occurrence%3];key=source
  else:source={'depth':'depth_replay','point':'point_replay','mask':'mask_replay','affordance':'affordance_replay'}.get(kind,kind);key=kind
  sources[source]+=64;roles[kind]+=64;offsets[key]+=64
 return dict(sources),dict(forms),dict(roles),dict(offsets)

def main():
 assert not (V/'spec.json').exists()
 parent=read(O/'spec.json');spec=json.loads(json.dumps(parent))
 for p,h in parent['frozen_files'].items():assert sha(p)==h,p
 report_path=D/'evaluation/final_report.json';report=read(report_path)
 review_path=C/'coordinate_delivery_control_v2_20261104/quality_review.json';review=read(review_path)
 assert review['accepted'] is False and review['report_sha256']==sha(report_path)
 assert report['native']['n']==580000 and sum(x['n'] for x in report['public'].values())==7366 and sum(x['n'] for x in report['general'].values())==10105
 init=read(V/'initialization/initialization.json');complete=read(O/'runs/mopd/complete.json')
 assert init['source']==complete['checkpoint'] and init['source_adapter_sha256']==complete['adapter_sha256']
 spec['initial']=dict(adapter=init['output'],adapter_sha256=init['adapter_sha256'])
 spec['visual_adaptation']=dict(initialization=bound(V/'initialization/initialization.json'),interface=bound(V/'protocol.json'),language_parameters=173113344,visual_parameters=5136384,visual_blocks=list(range(19,27)),merger=True,rank=32,alpha=64,initial_delta_zero=True)
 spec['initialization_lineage']=bound(D/'training_lineage.json')
 spec['continuation_parent']=dict(checkpoint=complete['checkpoint'],adapter_sha256=complete['adapter_sha256'],lineage=bound(D/'training_lineage.json'),training_complete=bound(O/'runs/mopd/complete.json'),public_report=bound(report_path))
 spec['stages']['repair'].update(steps=8192,lr=1e-6,targeted_geometry_lr=2e-6,visual_lr=2e-5,first_checkpoint_step=64,save_every=256)
 spec['stages']['mopd'].update(steps=256,lr=2.5e-7,visual_lr=2.5e-7,first_checkpoint_step=32,save_every=32)
 rationale='Continue the completed coordinate MOPD256 weights. Full public benchmarks pass the per-benchmark 10% threshold; native geometry fails. Preserve all language weights and introduce zero-delta rank32 adapters in final eight visual blocks and merger. Joint visual/language repair8192 then genuine MOPD256. Full original pools, all11 forms, general/public replay, scorers and generation budgets retained. This intervention tests whether trainable visual features improve grounding; improvement requires full heldout evaluation.'
 spec['rationale']=rationale
 spec['recovery']=dict(reason=rationale,parent_pipeline=bound(O/'spec.json'),parent_training_complete=bound(O/'runs/mopd/complete.json'),parent_integrity=bound(O/'mopd_checkpoint256_verification.json'),parent_lineage=bound(D/'training_lineage.json'),original_sft_retrained=False,parent_model_weights_inherited=True,new_stage_optimizers=True,full_generation_budgets_preserved=True,parent_exposures_inherited=True,initialization=bound(V/'initialization/initialization.json'))
 spec['parent_quality_assessment']=bound(review_path);spec['rejected_experiment_reviews']['coordinate_v2']=bound(review_path)
 spec['point_contract_policy']['code']=bound(V/'point_contracts.py');spec['native_io_policy']['implementation']=bound(V/'native_io_pool.py');spec['geometry_objective']['code']=bound(V/'geometry_loss.py')
 for field in ['planned_source_exposures','planned_native_form_exposures','planned_role_exposures']:spec[field]={}
 endpoints={}
 for stage,cfg in parent['stages'].items():
  used=counts(stage,cfg['steps'])[3]
  for source,start in parent['source_offsets'][stage].items():endpoints[source]=max(endpoints.get(source,0),start+used.get(source,0))
 spec['source_offsets']={}
 for stage,cfg in spec['stages'].items():
  a,b,c,used=counts(stage,cfg['steps']);spec['planned_source_exposures'][stage]=a;spec['planned_native_form_exposures'][stage]=b;spec['planned_role_exposures'][stage]=c
  spec['source_offsets'][stage]=dict(endpoints)
  for source,n in used.items():endpoints[source]=endpoints.get(source,0)+n
  assert sum(a.values())==cfg['steps']*64 and len(b)==11
 keys=['visual_repair/checkpoint-4096','visual_repair/checkpoint-8192','visual_mopd/checkpoint-128','visual_mopd/checkpoint-256']
 spec['key_checkpoints']+=keys;spec['key_checkpoint_sources'].update({k:str(V/'runs'/k.removeprefix('visual_')) for k in keys});assert len(spec['key_checkpoints'])==39
 backup=C/'coordinate_checkpoint_backup_v2_20261104/mopd_256/verification_receipt.json'
 assert len(read(backup)['key_checkpoints'])==35 and read(backup)['revision']=='574164bea67033e6117348584076c9627f4d97e8'
 estimates={}
 for stage,cfg in spec['stages'].items():
  log=O/f'runs/{stage}/logging.jsonl';rows=[json.loads(l) for l in log.open()];seconds=sum(r['seconds'] for r in rows)/len(rows)
  estimates[stage]=dict(source_log=bound(log),measured_seconds_per_update=seconds,overhead_allowance=1.5,updates=cfg['steps'],estimated_hours=seconds*1.5*cfg['steps']/3600)
 hours=sum(v['estimated_hours'] for v in estimates.values())
 spec['budget_estimate']=dict(stages=estimates,projected_training_hours=hours,validation_upload_reserve_hours=4,queue_and_visual_overhead_unmeasured=True,original_user_deadline_missed=True,additional_gpu_ceiling=32)
 deadline=datetime.now(timezone.utc)+timedelta(hours=48)
 spec['deadline_utc']=deadline.isoformat();spec['final_reserve_hours']=4;spec['execution_expiry_note']='48h operational expiry; original five-day deadline remains missed. Estimate is not a quality guarantee.'
 trigger=dict(schema='anvil-complete-evaluation-continuation-trigger-v1',public_general_report=str(report_path),public_general_report_sha256=sha(report_path),scores=report['paper_aligned_public'],final_quality_accepted=False,rationale=rationale,rejected_review=bound(review_path))
 save(V/'public_trigger.json',trigger);spec['recovery']['trigger']=bound(V/'public_trigger.json');save(V/'schedules.json',SCHEDULES)
 # Keep immutable ancestor bindings and add actual new implementation dependencies.
 frozen=[*V.glob('*.py'),*H.glob('*.py'),*H.joinpath('interface_validation').glob('*.json'),V/'protocol.json',V/'initialization/initialization.json',V/'initialization/adapter_config.json',V/'public_trigger.json',V/'schedules.json',V/'cpu_validation.json',V/'optimizer_cpu_tests.json',V/'merge_cpu_validation.json',V/'trainer_interface_validation.json',report_path,review_path,backup,C/'visual_continuation_controller.py',C/'visual_final_handoff.py',C/'backup_visual_checkpoints.py']
 for p in frozen:
  if p.suffix=='.py':ast.parse(p.read_text())
 spec['frozen_files'].update({str(p.resolve()):sha(p) for p in frozen})
 sys.path.insert(0,str(H));from common import initialization_transform
 initialization_transform(spec,read(D/'training_lineage.json'))
 save(V/'candidate_settings.json',spec);spec['frozen_files'][str(V/'candidate_settings.json')]=sha(V/'candidate_settings.json');save(V/'spec.json',spec)
 config=yaml.safe_load((O/'cli.yaml').read_text());config['TaskName']='task_train_anvil_visual_geometry_A800x32';config['Description']='Visual-language geometry repair8192 and genuine MOPD256; completed coordinate parent; all11 native forms; full evaluation';config['ActiveDeadlineSeconds']=172800
 config['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+shlex.join([str(R/'.venv-train/bin/python'),'-u',str(V/'node.py'),'--spec',str(V/'spec.json'),'--spec-sha256',sha(V/'spec.json')])
 with (V/'cli.yaml').open('x') as f:yaml.safe_dump(config,f,sort_keys=False)
 handoff=dict(schema='anvil-direct-final-handoff-v1',pipeline_spec=str(V/'spec.json'),pipeline_spec_sha256=sha(V/'spec.json'),nodes=4,gpus=32,files={str(p):sha(p) for p in [C/'visual_final_handoff.py',*H.glob('*.py'),H/'interface_validation/cpu_validation.json']},deadline=deadline.timestamp(),final_quality_autoaccepted=False,key_checkpoints=spec['key_checkpoints'])
 save(FD/'policy.json',handoff)
 jobs=['t-20261104230112-7zx4q','t-20261104230159-5nzrb','t-20261104230247-665gd','t-20261104230330-mlz8c']
 policy=dict(schema='anvil-targeted-continuation-submission-v1',gpus=32,spec=str(V/'spec.json'),config=str(V/'cli.yaml'),trigger=str(V/'public_trigger.json'),parent_report=str(report_path),parent_evaluation_spec_sha256=report['final_spec_sha256'],parent_jobs=jobs,handoff_source=str(C/'visual_final_handoff.py'),handoff_policy=str(FD/'policy.json'))
 policy['files']={str(p):sha(p) for p in [C/'visual_continuation_controller.py',C/'visual_final_handoff.py',V/'spec.json',V/'cli.yaml',V/'public_trigger.json',FD/'policy.json']};save(V/'submit_policy.json',policy)
 save(V/'backup_policy.json',dict(spec_sha256=sha(V/'spec.json'),files={str(p):sha(p) for p in [C/'backup_visual_checkpoints.py',V/'verify_checkpoint.py']},parent_hf_revision=read(backup)['revision'],previous_keys=35,planned_keys=39,credentials_modified=False))
 print(json.dumps(dict(spec_sha256=sha(V/'spec.json'),projected_training_hours=hours,deadline_utc=spec['deadline_utc'],keys=39,submitted=False)))
if __name__=='__main__':main()
