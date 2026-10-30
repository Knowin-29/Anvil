from pathlib import Path
from datetime import datetime, timezone
import ast, copy, hashlib, json, os, re, shutil, shlex, sys, time

C=Path('/Knowin/sim/weilinruan/Anvil/recovery_20261020/completion_20261028')
R=C.parent; OLD=C/'direct_completion_20261029'; O=C/'direct_recovery_20261030'
H0=C/'final_delivery_direct_v1'; H=C/'final_delivery_recovery_v1'
FD=C/'delivery_runs/direct_recovery_20261030'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.pending');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
assert not O.exists() and not H.exists() and not FD.exists(), 'Preserve prior attempts'
old=read(OLD/'spec.json')
for p,d in old['frozen_files'].items():assert sha(p)==d,p
repair=read(OLD/'runs/repair/complete.json')
assert repair['steps']==2048 and repair['stage']=='repair_training_complete'
assert sha(Path(repair['checkpoint'])/'adapter_model.safetensors')==repair['adapter_sha256']
assert not list((OLD/'runs/mopd').glob('checkpoint-*/checkpoint_complete.json'))
logs=[json.loads(x) for x in (OLD/'runs/mopd/logging.jsonl').read_text().splitlines()]
assert logs[-1]['step']==126
exposures=0;uncommitted_rows=0
for rank in range(32):
    rows=[json.loads(x) for x in (OLD/f'runs/mopd/exposure_rank{rank}.jsonl').read_text().splitlines()]
    committed=[x for x in rows if x['step']<=126]
    suffix=[x for x in rows if x['step']>126]
    assert [(x['step'],x['micro']) for x in committed]==[(s,m) for s in range(1,127) for m in range(2)]
    assert all(x['step']==127 and x['micro']==0 for x in suffix)
    exposures+=len(committed);uncommitted_rows+=len(suffix)
assert exposures==8064
O.mkdir();(O/'runs').mkdir();(O/'runs/repair').symlink_to(OLD/'runs/repair',target_is_directory=True)
FD.mkdir(parents=True)
failure=dict(schema='anvil-terminal-runtime-failure-v1',task='t-20261029174537-bvfnb',cloud_status='Failed',
    last_committed_step=126,failed_step=127,uncheckpointed_exposures=8064,uncommitted_microbatch_rows=uncommitted_rows,
    first_collective_timeout_ms=600000,collective='ALLREDUCE',elements=1,sequence=5385,
    established='Other ranks reached the scalar reduction while rank1 had not enqueued it; no earlier OOM/numerical exception found.',
    likely_cause='Long on-policy generation exceeded the default collective wait; phase telemetry will distinguish generation from other precollective work.',
    original_files_preserved=True,final_optimizer_ancestry_includes_failed_attempt=False,
    log_bindings={str(p):sha(p) for p in sorted((OLD/'runs/mopd').glob('console_node*.log'))},
    checked_at=datetime.now(timezone.utc).isoformat())
save(O/'failure_analysis.json',failure)
original=(OLD/'train.py').read_text();training=original
changes=[
 ('import argparse\n','import argparse\nfrom datetime import timedelta\n'),
 ("dist.init_process_group('nccl');torch.set_num_threads(4)","assert 600 < spec['distributed_timeout_seconds'] <= 7200\n    dist.init_process_group('nccl',timeout=timedelta(seconds=spec['distributed_timeout_seconds']));torch.set_num_threads(4)"),
 ("teacher_assessment=spec['teacher_assessment'],code_sha256=sha(__file__))", "teacher_assessment=spec['teacher_assessment'],code_sha256=sha(__file__),\n        distributed_timeout_seconds=spec['distributed_timeout_seconds'],recovery=spec['recovery'])"),
 ("begin=time.monotonic();qualified=dict.fromkeys('GSA',True)","begin=time.monotonic();qualified=dict.fromkeys('GSA',True)\n    def phase(name,**fields):\n        save(out/f'phase_rank{rank}.json',dict(rank=rank,stage=stage,phase=name,updated=time.time(),**fields))"),
 ("                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):\n                        ids=student.generate", "                    rollout_started=time.monotonic()\n                    phase('rollout',step=step+1,micro=micro,form=row['gsa_form'],max_new_tokens=BUDGETS[row['gsa_form']] if is_mtd else cfg['ul_rollout_cap'])\n                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):\n                        ids=student.generate"),
 ("                    cycle=guard.cycle;generated=ids.shape[1]-prompt_length", "                    cycle=guard.cycle;generated=ids.shape[1]-prompt_length\n                    phase('rollout_complete',step=step+1,micro=micro,form=row['gsa_form'],tokens=generated,seconds=time.monotonic()-rollout_started)"),
 ("                        eligible=(plan.selected_stages>=0).sum().float();total=eligible.clone();dist.all_reduce(total)", "                        phase('mtd_normalization',step=step+1,micro=micro,form=row['gsa_form'],tokens=generated)\n                        eligible=(plan.selected_stages>=0).sum().float();total=eligible.clone();dist.all_reduce(total)"),
 ("            optimizer.step();dist.all_reduce(stats);stats/=world", "            optimizer.step();dist.all_reduce(stats);stats/=world\n            phase('update_complete',step=step+1)")]
for before,after in changes:
    assert training.count(before)==1,before
    training=training.replace(before,after)
restored=training
for before,after in reversed(changes):restored=restored.replace(after,before)
assert restored==original,'Unexpected training-math change'
(O/'train.py').write_text(training);shutil.copyfile(OLD/'pool.py',O/'pool.py')
node=(OLD/'node.py').read_text().replace('Four-worker repair→MOPD pipeline, with production checkpoints and no ablations.','Four-worker MOPD recovery from the completed corrective checkpoint.')
assert "for stage in ['repair','mopd']:" in node
node=node.replace("for stage in ['repair','mopd']:","for stage in ['mopd']:")
(O/'node.py').write_text(node)
shutil.copyfile(__file__,O/'prepare_recovery.py')
spec=copy.deepcopy(old)
spec['distributed_timeout_seconds']=3600
spec['stages']['mopd']['save_every']=32
spec['recovery']=dict(reason='Explicit recovery after confirmed terminal NCCL timeout; restart MOPD from verified completed repair.',
    previous_pipeline={'path':str(OLD/'spec.json'),'sha256':sha(OLD/'spec.json')},
    failure_analysis={'path':str(O/'failure_analysis.json'),'sha256':sha(O/'failure_analysis.json')},
    repair_complete={'path':str(OLD/'runs/repair/complete.json'),'sha256':sha(OLD/'runs/repair/complete.json')},
    discarded_attempt_updates=126,discarded_attempt_exposures=8064,discarded_attempt_in_final_optimizer_ancestry=False,
    repair_retrained=False,mopd_restart_step=0,mopd_planned_steps=1536,planned_mopd_exposures=98304,
    timeout_seconds=3600,checkpoint_interval=32,full_generation_budgets_preserved=True,
    data_loss_teacher_and_optimizer_math_unchanged=True)
assert {k:v for k,v in spec['stages']['mopd'].items() if k!='save_every'}=={k:v for k,v in old['stages']['mopd'].items() if k!='save_every'}
for key in ('initial','teachers','datasets','planned_source_exposures','planned_role_exposures','key_checkpoints'):
    assert spec[key]==old[key],key
for p in O.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
save(O/'cpu_recovery_validation.json',dict(state='passed',training_change_reversal_exact=True,
    data_pool_bytes_unchanged=sha(O/'pool.py')==sha(OLD/'pool.py'),training_budget_and_mixture_unchanged=True,
    inherited_repair_checkpoint_binding_verified=True,failed_attempt_prefix_exposures_verified=8064,
    old_frozen_sources_unchanged=True,python_syntax_valid=True,extra_gpu_test_tasks=0,
    runtime_recovery_still_requires_actual_execution=True,created=time.time()))
spec['frozen_files'].update({str(p):sha(p) for p in list(O.glob('*.py'))+[O/'failure_analysis.json',O/'cpu_recovery_validation.json',OLD/'runs/repair/complete.json',OLD/'runs/repair/protocol.json']})
save(O/'spec.json',spec)
import yaml
conf=yaml.safe_load((OLD/'cli.yaml').read_text())
conf['TaskName']='task_train_anvil_MOPD_recovery_A800x32'
conf['Description']='MOPD1536 from completed repair2048; NCCL3600s, phase telemetry, checkpoint32; same full pool, losses, teachers, generation budgets'
conf['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+shlex.join([str(R/'.venv-train/bin/python'),'-u',str(O/'node.py'),'--spec',str(O/'spec.json'),'--spec-sha256',sha(O/'spec.json')])
(O/'cli.yaml').write_text(yaml.safe_dump(conf,sort_keys=False))
shutil.copytree(H0,H,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
p=H/'common.py';s=p.read_text();assert s.count("pipeline=C/'direct_completion_20261029'")==1
s=s.replace("pipeline=C/'direct_completion_20261029'","pipeline=C/'direct_recovery_20261030'")
s=s.replace("spec=read(pipeline/'spec.json')", "spec=read(pipeline/'spec.json')\n    for item in spec['recovery'].values():\n        if isinstance(item,dict) and 'path' in item and 'sha256' in item:check_binding(item)")
p.write_text(s)
p=H/'lineage.py';s=p.read_text();s=s.replace("warm_start=initial['warm_start'], initial_sft=initial['initial_sft'],","warm_start=initial['warm_start'], initial_sft=initial['initial_sft'], recovery=pipeline_spec.get('recovery'),");p.write_text(s)
p=H/'package.py';s=p.read_text();s=s.replace("Their identities are in `evidence/key_checkpoints.json`.","Their identities are in `evidence/key_checkpoints.json`. An earlier MOPD attempt failed before its first checkpoint; its 126 uncheckpointed updates are excluded from this model's optimizer ancestry. The completed corrective stage was reused without retraining.");p.write_text(s)
handoff=C/'direct_recovery_handoff.py'
handoff.write_text((C/'direct_final_handoff.py').read_text().replace("H=C/'final_delivery_direct_v1'","H=C/'final_delivery_recovery_v1'"))
for p in [*H.glob('*.py'),handoff]:ast.parse(p.read_text(),filename=str(p))
old_policy=read(C/'delivery_runs/direct_completion_20261029/policy.json');policy=copy.deepcopy(old_policy)
policy['pipeline_spec']=str(O/'spec.json');policy['pipeline_spec_sha256']=sha(O/'spec.json')
files={}
for name in old_policy['files']:
    path=Path(name)
    if path==C/'direct_final_handoff.py':path=handoff
    elif path.is_relative_to(H0):path=H/path.relative_to(H0)
    files[str(path)]=sha(path)
policy['files']=files;save(FD/'policy.json',policy)
monitor=C/'campaign_monitor_recovery.py'
monitor.write_text((C/'campaign_monitor.py').read_text().replace('direct_completion_20261029','direct_recovery_20261030'))
ast.parse(monitor.read_text())
print(json.dumps({'pipeline':str(O),'spec_sha256':sha(O/'spec.json'),'reused_repair_checkpoint':repair['checkpoint'],'mopd_steps':1536,'global_batch':64,'world':32,'timeout_seconds':3600,'save_every':32,'handoff':str(handoff),'policy':str(FD/'policy.json')},indent=2))
