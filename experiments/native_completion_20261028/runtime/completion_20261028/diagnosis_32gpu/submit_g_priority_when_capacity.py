"""Submit two prepared independent validation nodes while respecting the32GPU cap."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import fcntl,json,subprocess,time,os
from submit_student_when_capacity import check,save,sha,TERMINAL
ROOT=Path(__file__).resolve().parent;C=ROOT.parent

def main():
 with open('/tmp/anvil-G-priority-submit.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  spec_path=ROOT/'g_priority_native_spec.json';spec_sha=sha(spec_path);spec=json.loads(spec_path.read_text())
  configs=[ROOT/('g_priority_native_node'+str(n)+'_cli.yaml') for n in range(2)];config_sha=[sha(p) for p in configs]
  end=time.time()+7200
  for node in range(2):
   receipt=ROOT/('g_priority_native_node'+str(node)+'_submission.json')
   if receipt.exists():continue
   while time.time()<end:
    assert sha(spec_path)==spec_sha and sha(configs[node])==config_sha[node]
    for path,digest in spec['frozen_files'].items():assert sha(path)==digest,path
    jobs=json.loads((ROOT/'jobs.json').read_text())+json.loads((C/'jobs_v2.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:cloud=list(pool.map(check,jobs))
    active=sum(j['gpus'] for j in cloud if j['status'] not in TERMINAL);assert active<=32
    save(ROOT/'g_priority_submission_status.json',dict(state='waiting_for_capacity',node=node,gpus=active,additional=8,updated=time.time(),pid=os.getpid()))
    if active+8>32:time.sleep(30);continue
    save(ROOT/('g_priority_native_node'+str(node)+'_capacity.json'),dict(cloud=cloud,gpus_before=active,requested=8,ceiling=32,checked=time.time()))
    p=subprocess.run(['/root/.volc/bin/volc','--log-level','error','ml_task','submit','--conf',str(configs[node]),'--output','json'],capture_output=True,text=True,timeout=120)
    (ROOT/('g_priority_native_node'+str(node)+'_submission.log')).write_text(p.stdout+p.stderr)
    values=[]
    for line in p.stdout.splitlines():
     try:value=json.loads(line)
     except json.JSONDecodeError:continue
     if isinstance(value,dict) and isinstance(value.get('Id'),str):values.append(value)
    assert p.returncode==0 and len(values)==1,'Ambiguous submission; no automatic retry'
    response=values[0];save(receipt,dict(response=response,submitted=time.time(),spec_sha256=spec_sha,config_sha256=config_sha[node]))
    registry=json.loads((ROOT/'jobs.json').read_text());assert response['Id'] not in {j['id'] for j in registry}
    registry.append(dict(id=response['Id'],gpus=8,scope='priority_full_G_spatial_native_node'+str(node),config=str(configs[node])));save(ROOT/'jobs.json',registry)
    save(ROOT/'g_priority_submission_status.json',dict(state='node_submitted',node=node,id=response['Id'],gpus_after=active+8,updated=time.time()))
    print(node,response['Id'],flush=True);break
   else:raise TimeoutError('Capacity did not free within2hours; remainingnode not submitted')
  save(ROOT/'g_priority_submission_status.json',dict(state='both_nodes_submitted',updated=time.time()))

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  save(ROOT/'g_priority_submission_failure.json',dict(error=repr(exc),time=time.time()));raise
