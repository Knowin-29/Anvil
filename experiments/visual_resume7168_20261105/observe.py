"""Concise live observation of the resumed production job and its delivery workers."""
from pathlib import Path
import argparse,collections,json,subprocess,time
W=Path(__file__).resolve().parent;C=W.parent;V=C/'visual_geometry_candidate_20261104'
ap=argparse.ArgumentParser();ap.add_argument('--cloud',action='store_true');args=ap.parse_args()
submission=json.loads((W/'submission.json').read_text());identity=submission['response']['Id']
obs=dict(checked=time.time(),task_id=identity,stages={},failures=[str(p) for p in [*W.glob('node*_failure.json'),W/'controller_failure.json',C/'delivery_runs/visual_geometry_20261104/failure.json'] if p.exists()])
if args.cloud:
 proc=subprocess.run(['/opt/volc/bin/volc','--log-level','error','ml_task','get','--id',identity,'--output','json'],capture_output=True,text=True,timeout=60)
 values=[]
 for line in proc.stdout.splitlines():
  try:item=json.loads(line)
  except ValueError:continue
  if isinstance(item,list):values.extend(x for x in item if isinstance(x,dict) and x.get('JobId')==identity)
 assert proc.returncode==0 and len(values)==1
 obs['cloud']={k:values[0][k] for k in ['JobId','Status','Start','End','Elapsed']}
for stage in ['repair','mopd']:
 run=V/'runs'/stage;path=run/'logging.jsonl';rows=[]
 if path.exists():
  for line in path.open():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
 if rows:
  r=rows[-1];obs['stages'][stage]={k:r.get(k) for k in ['step','max_steps','elapsed','sft','mtd','group_grad_norms','peak_gpu_gib']}
  obs['stages'][stage]['log_updated_after_submission']=path.stat().st_mtime>submission['time']
  latest=run/'latest.json'
  if latest.exists():obs['stages'][stage]['latest_checkpoint']=json.loads(latest.read_text())['step']
  phases=[]
  for p in run.glob('phase_rank*.json'):
   x=json.loads(p.read_text())
   if x['updated']>submission['time']:phases.append(x)
  obs['stages'][stage]['fresh_rank_phases']=dict(collections.Counter(x['phase'] for x in phases))
  if phases:obs['stages'][stage]['latest_phase_age_seconds']=time.time()-max(x['updated'] for x in phases)
state=json.loads((C/'visual_resume_storage_20261105/storage_reserve.json').read_text());st=(C/'visual_resume_storage_20261105/storage_reserve.bin').stat();assert st.st_size==state['remaining_bytes'];obs['reserved_gib_remaining']=st.st_size/1024**3;obs['runtime_resume_verified']=(W/'runtime_resume_verified.json').exists()
obs['emergency_reserved_gib_remaining']=(W/'emergency_reserve.bin').stat().st_size/1024**3
p=W/'runtime_observation.json';t=p.with_suffix('.pending');t.write_text(json.dumps(obs,indent=2)+'\n');t.replace(p);print(json.dumps(obs,indent=2))
