"""Verify the first substantive checkpoint; no diagnostic training or GPU work."""
from pathlib import Path
import fcntl,hashlib,json,subprocess,time,os
J=Path(__file__).resolve().parent;C=J.parent;V=C/'visual_geometry_candidate_20261104';TRAIN=C.parent/'.venv-train/bin/python'
def save(p,v):
 t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
 lock=(J/'first_checkpoint.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 policy=json.loads((V/'backup_policy.json').read_text());expected=policy['spec_sha256'];output=V/'repair_checkpoint64_verification.json'
 if output.exists():return
 spec=json.loads((V/'spec.json').read_text());from datetime import datetime
 deadline=datetime.fromisoformat(spec['deadline_utc']).timestamp()
 while time.time()<deadline:
  if list(V.glob('node*_failure.json')):raise RuntimeError('Production node failed')
  if (V/'runs/repair/checkpoint-64/checkpoint_complete.json').exists():break
  save(J/'first_checkpoint_status.json',dict(state='waiting_production_checkpoint64',pid=os.getpid(),updated=time.time()));time.sleep(30)
 else:raise TimeoutError('No first production checkpoint')
 assert hashlib.sha256((V/'spec.json').read_bytes()).hexdigest()==expected
 assert hashlib.sha256((V/'verify_checkpoint.py').read_bytes()).hexdigest()==policy['files'][str(V/'verify_checkpoint.py')]
 with (J/'first_checkpoint_verification.log').open('a') as log:subprocess.run([str(TRAIN),str(V/'verify_checkpoint.py'),'--spec-sha256',expected,'--stage','repair','--step','64','--output',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
 result=json.loads(output.read_text());assert result['state']=='visual_checkpoint_integrity_verified'
 save(J/'first_checkpoint_status.json',dict(state=result['state'],step=64,visual_parameters=result['visual_parameters_verified'],changed_visual_b_tensors=result['changed_visual_b_tensors'],receipt=str(output),updated=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as e:save(J/'first_checkpoint_failure.json',dict(error_type=type(e).__name__,time=time.time()));raise
