"""Submit one exact-resume32-GPU job under the campaign capacity lock."""
from pathlib import Path
import fcntl,json,subprocess,sys,time,hashlib
W=Path(__file__).resolve().parent;C=W.parent
sys.path.insert(0,str(C));from final_delivery_handoff_v1 import cloud_snapshot,submission_response
sys.path.insert(0,str(C/'formal_mopd'));from controller import register

def read(p):return json.loads(p.read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 t=p.with_suffix('.pending');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def main():
 own=(W/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 policy=read(W/'resume_policy.json')
 for p,h in policy['files'].items():assert sha(p)==h,p
 reserve=read(W/'storage_reserve.json');s=(W/'storage_reserve.bin').stat();assert reserve['initial_bytes']==256*1024**3 and s.st_size==reserve['remaining_bytes'] and s.st_ino==reserve['inode']
 spec=read(C/'visual_geometry_candidate_20261104/spec.json')
 for p,h in spec['frozen_files'].items():assert sha(p)==h,p
 assert not (W/'submission.json').exists() and not (W/'submission_intent.json').exists()
 with open('/tmp/anvil-formal-mopd-submit.lock','a') as shared:
  fcntl.flock(shared,fcntl.LOCK_EX)
  cloud,active=cloud_snapshot();assert active==0
  assert next(x for x in cloud if x['id']=='t-20261105001409-lb4r4')['status']=='Failed'
  save(W/'capacity.json',dict(cloud=cloud,active=active,requested=32,cap=32,checked=time.time()))
  binding=dict(config_sha256=sha(W/'cli.yaml'),resume_policy_sha256=sha(W/'resume_policy.json'),pipeline_spec_sha256=policy['pipeline_spec_sha256'],gpus=32,time=time.time())
  save(W/'submission_intent.json',binding)
  proc=subprocess.run(['/opt/volc/bin/volc','--log-level','error','ml_task','submit','--conf',str(W/'cli.yaml'),'--output','json'],capture_output=True,text=True,timeout=90)
  (W/'submission.log').write_text(proc.stdout+proc.stderr);receipt=dict(binding,response=submission_response(proc.stdout,proc.returncode));save(W/'submission.json',receipt)
  register(receipt,32,'visual_exact_resume768_repair8192_mopd256',W/'cli.yaml')
 print(json.dumps(dict(task=receipt['response']['Id'],resume_step=768)),flush=True)
 save(W/'controller_status.json',dict(stage='submitted_handoff_running',task_id=receipt['response']['Id'],updated=time.time()))
 with (W/'handoff.log').open('a') as log:subprocess.run([str(C.parent/'.venv-train/bin/python'),'-u',str(W/'handoff.py'),'--policy',str(W/'handoff_policy.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
 save(W/'controller_status.json',dict(stage='full_final_report_ready_requires_quality_review',updated=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as e:save(W/'controller_failure.json',dict(error=repr(e),time=time.time()));raise
