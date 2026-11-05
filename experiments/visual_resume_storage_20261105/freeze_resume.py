from pathlib import Path
import ast,hashlib,json,shlex,time,yaml
W=Path(__file__).resolve().parent;C=W.parent;V=C/'visual_geometry_candidate_20261104';J=C/'visual_delivery_control_20261104';D=C/'delivery_runs/visual_geometry_20261104';H=C/'final_delivery_visual_geometry_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
for p in W.glob('*.py'):ast.parse(p.read_text())
verification=read(W/'resume768_verification.json');assert verification['state']=='visual_checkpoint_integrity_verified' and verification['step']==768
assert read(W/'optimizer_cleanup_complete.json')['files_removed']==411
files=[W/'node.py',W/'storage_budget.py',W/'test_resume.py',W/'resume768_verification.json',W/'optimizer_cleanup_complete.json',W/'optimizer_cleanup_preview.json',W/'failure_archive.json',V/'spec.json']
policy=dict(schema='anvil-exact-visual-resume-v1',pipeline_spec_sha256=sha(V/'spec.json'),resume_checkpoint=str(V/'runs/repair/checkpoint-768'),resume_step=768,target_repair_step=8192,target_mopd_step=256,resume_seal_sha256=sha(V/'runs/repair/checkpoint-768/checkpoint_complete.json'),files={str(p):sha(p) for p in files},optimizer_scheduler_rng_resumed=True,original_trainer_unchanged=True,gpus=32,required_reserve_gib=256,created=time.time())
save(W/'resume_policy.json',policy)
config=yaml.safe_load((V/'cli.yaml').read_text());config['TaskName']='task_train_anvil_visual_resume768_A800x32';config['Description']='Exact visual repair checkpoint768 resume with optimizer and32 RNGs; finish8192 then MOPD256; reserved storage; full evaluation';config['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+shlex.join([str(C.parent/'.venv-train/bin/python'),'-u',str(W/'node.py')])
with (W/'cli.yaml').open('x') as f:yaml.safe_dump(config,f,sort_keys=False)
handoff=read(D/'policy.json');handoff.update(files={str(p):sha(p) for p in [W/'handoff.py',W/'storage_budget.py',W/'resume_policy.json',*H.glob('*.py'),H/'interface_validation/cpu_validation.json']},resume_submission=str(W/'submission.json'),resume_policy=str(W/'resume_policy.json'))
save(W/'handoff_policy.json',handoff)
release=read(J/'release_policy.json');release['files'].update({str(p):sha(p) for p in [W/'finish_delivery.py',W/'handoff.py',W/'storage_budget.py',W/'resume_policy.json',W/'handoff_policy.json',W/'cleanup_verified_weights.py']});release.update(recovery_policy=str(W/'resume_policy.json'),original_attempt_failed=True);save(W/'release_policy.json',release)
cleanup=read(J/'cleanup_policy.json');cleanup['files'].update({str(p):sha(p) for p in [W/'cleanup_verified_weights.py',W/'release_policy.json']})
for name in ['preview_argv','apply_argv']:cleanup[name]=[str(W/'cleanup_verified_weights.py') if x==str(J/'cleanup_verified_weights.py') else x for x in cleanup[name]]
cleanup.update(recovery_submission=str(W/'submission.json'),additional_quota_ledger=str(W/'optimizer_cleanup_complete.json'));save(W/'cleanup_policy.json',cleanup)
print(json.dumps(dict(state='frozen_exact_resume',step=768,target_repair=8192,target_mopd=256,pipeline_spec_sha256=policy['pipeline_spec_sha256'])))
