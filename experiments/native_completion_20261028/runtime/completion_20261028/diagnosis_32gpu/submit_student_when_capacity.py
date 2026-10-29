"""Submit the already-sealed student task once campaign GPU capacity is free."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl,hashlib,json,subprocess,time,os
ROOT=Path(__file__).resolve().parent;C=ROOT.parent
TERMINAL={'Success','Succeeded','Failed','Killed','Stopped'}

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def check(job):
 p=subprocess.run(['/root/.volc/bin/volc','--log-level','error','ml_task','get','--id',job['id'],'--output','json','--format','Id,Name,Status'],capture_output=True,text=True,timeout=30)
 values=[json.loads(line) for line in p.stdout.splitlines() if line.startswith('[{"Id"')]
 if p.returncode or len(values)!=1:raise RuntimeError('Cannot resolve status for '+job['id'])
 return dict(job,status=values[0][0]['Status'])

def main():
 with open('/tmp/anvil-student-adaptation-submit.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  receipt=ROOT/'student_adaptation_submission.json'
  if receipt.exists():return
  config=ROOT/'student_adaptation_cli.yaml';spec_path=ROOT/'student_adaptation_spec.json'
  expected_config,expected_spec=sha(config),sha(spec_path);deadline=time.time()+3600
  while time.time()<deadline:
   spec=json.loads(spec_path.read_text());assert sha(config)==expected_config and sha(spec_path)==expected_spec
   for name,digest in spec['frozen_files'].items():assert sha(name)==digest,name
   assert sha(ROOT/'student_adaptation_cpu_validation.json')==spec['cpu_validation_sha256']
   jobs=json.loads((ROOT/'jobs.json').read_text())+json.loads((C/'jobs_v2.json').read_text())
   assert not any(j['scope']=='student_rotation128_adaptation_full_public_general_A' for j in jobs)
   with ThreadPoolExecutor(max_workers=4) as pool:cloud=list(pool.map(check,jobs))
   assert len({j['id'] for j in cloud})==len(cloud)
   active=sum(j['gpus'] for j in cloud if j['status'] not in TERMINAL)
   assert active<=32
   save(ROOT/'student_adaptation_submission_status.json',dict(state='waiting_for_capacity',gpus=active,additional=8,updated=time.time(),pid=os.getpid(),spec_sha256=expected_spec))
   if active+8<=32:
    save(ROOT/'student_adaptation_submission_capacity.json',dict(cloud=cloud,allocated_or_pending_gpus=active,requested=8,ceiling=32,checked=time.time()))
    p=subprocess.run(['/root/.volc/bin/volc','--log-level','error','ml_task','submit','--conf',str(config),'--output','json'],capture_output=True,text=True,timeout=120)
    (ROOT/'student_adaptation_submission.log').write_text(p.stdout+p.stderr)
    # Never repeat an ambiguous mutation automatically.
    values=[]
    for line in p.stdout.splitlines():
     try:value=json.loads(line)
     except json.JSONDecodeError:continue
     if isinstance(value,dict) and isinstance(value.get('Id'),str):values.append(value)
    assert p.returncode==0 and len(values)==1,'Submission outcome ambiguous; inspect once, no automatic retry'
    response=values[0];save(receipt,dict(response=response,submitted=time.time(),spec_sha256=expected_spec,config_sha256=expected_config))
    registry=json.loads((ROOT/'jobs.json').read_text());assert response['Id'] not in {j['id'] for j in registry}
    registry.append(dict(id=response['Id'],gpus=8,scope='student_rotation128_adaptation_full_public_general_A',config=str(config)))
    save(ROOT/'jobs.json',registry)
    save(ROOT/'student_adaptation_submission_status.json',dict(state='submitted',id=response['Id'],updated=time.time(),gpus_after_submission=active+8))
    print(response['Id'],flush=True);return
   time.sleep(30)
  raise TimeoutError('Capacity did not become free within one hour; no submission')

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  save(ROOT/'student_adaptation_submission_failure.json',dict(error=repr(exc),time=time.time()));raise
