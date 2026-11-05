"""Prepare and submit one8-GPU full evaluation of the selected7168 checkpoint."""
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,sys,time
Z=Path(__file__).resolve().parent;C=Z.parent;H=C/'final_delivery_checkpoint7168_v1';TRAIN=str(C.parent/'.venv-train/bin/python')
sys.path[:0]=[str(C),str(C/'formal_mopd')]
from final_delivery_handoff_v1 import cloud_snapshot,submission_response
from controller import register,TERMINAL

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pending');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def status(stage,**fields):save(Z/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**fields))
def tool(script,*argv):
 with (Z/(Path(script).stem+'.controller.log')).open('a') as f:subprocess.run([TRAIN,'-u',str(script),*map(str,argv)],stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 own=(Z/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 model=Z/'merged_model';receipt=model/'merge_receipt.json';spec=Z/'evaluation_spec.json';out=Z/'evaluation'
 launch=read(Z/'merge_launch.json')
 while not receipt.exists() or read(receipt).get('state')!='verified_derived_inference_copy':
  process=Path(f"/proc/{launch['pid']}/cmdline")
  assert process.exists() and str(H/'merge.py') in process.read_text(),'Merge process ended without complete receipt'
  status('waiting_selected_model_merge');time.sleep(15)
 if not spec.exists():tool(H/'prepare.py','--training-run',C/'visual_geometry_candidate_20261104/runs/repair','--model',model,'--output',spec,'--native-shards',32,'--public-shards',8)
 tool(H/'configs.py','--spec',spec,'--output',out,'--nodes',1)
 sys.path.insert(0,str(C/'visual_resume_storage_20261105'));from storage_budget import release
 release('checkpoint7168_full_evaluation_headroom',32)
 submission=Z/'evaluation_submission.json';config=out/'node_0_cli.yaml'
 shared=open('/tmp/anvil-formal-mopd-submit.lock','a');fcntl.flock(shared,fcntl.LOCK_EX)
 if submission.exists():submitted=read(submission)
 else:
  assert not (Z/'evaluation_submission_intent.json').exists(),'Inspect ambiguous prior submission'
  cloud,active=cloud_snapshot();assert active==0,'Wait for cancelled32-GPU task to become terminal'
  assert next(j for j in cloud if j['id']=='t-20261105190848-9cphl')['status'] in TERMINAL
  save(Z/'capacity_before_eval.json',dict(active=active,requested=8,cap=8,cloud=cloud,time=time.time()))
  binding=dict(final_spec_sha256=sha(spec),config_sha256=sha(config),gpus=8,time=time.time())
  save(Z/'evaluation_submission_intent.json',binding)
  proc=subprocess.run(['/opt/volc/bin/volc','--log-level','error','ml_task','submit','--conf',str(config),'--output','json'],capture_output=True,text=True,timeout=90)
  (Z/'evaluation_submission.log').write_text(proc.stdout+proc.stderr);submitted=dict(binding,response=submission_response(proc.stdout,proc.returncode));save(submission,submitted)
 register(submitted,8,'selected_visual_checkpoint7168_complete_evaluation',config)
 fcntl.flock(shared,fcntl.LOCK_UN);identity=submitted['response']['Id'];print(json.dumps(dict(task=identity,gpus=8)),flush=True)
 while True:
  cloud,active=cloud_snapshot();job=next(j for j in cloud if j['id']==identity)
  assert job['status'] not in TERMINAL or job['status']=='Success','Evaluation terminal without success: '+job['status']
  status('full_evaluation_running' if job['status']=='Running' else 'full_evaluation_waiting',task_id=identity,cloud_status=job['status'],active_campaign_gpus=active)
  if job['status']=='Success':break
  time.sleep(45)
 node=read(out/'node_0.json');assert node['stage']=='node_complete' and not node['failures']
 if not (out/'final_report.json').exists():tool(H/'score.py','--spec',spec,'--output',out)
 scorer=C/'anvil_eval_score_v1_1_20261104';seal=read(scorer/'protocol_seal.json')
 for name,h in seal['files'].items():assert sha(scorer/name)==h
 if not (Z/'native_score.json').exists():tool(scorer/'score_records.py','--report',out/'final_report.json','--spec',spec,'--output',Z/'native_score.json')
 report=read(out/'final_report.json');native=read(Z/'native_score.json')
 assert report['native']['n']==580000 and sum(x['n'] for x in report['public'].values())==7366 and sum(x['n'] for x in report['general'].values())==10105
 public={k:dict(measured=v,reference=report['paper_target'][k],minimum=.9*report['paper_target'][k],passed=v>=.9*report['paper_target'][k]) for k,v in report['paper_aligned_public'].items()}
 tolerance=dict(schema='anvil-selected-checkpoint-tolerance-v1',report_sha256=sha(out/'final_report.json'),evaluation_spec_sha256=sha(spec),checkpoint=report['checkpoint'],native=dict(measured=native['score'],reference=65.35,minimum=58.815,passed=native['score']>=58.815),public=public,all_numeric_tolerances_pass=all(x['passed'] for x in public.values()) and native['score']>=58.815,full_counts_verified=True,general=report['general'],native_groups=report['native']['groups'],native_score_sha256=sha(Z/'native_score.json'),requires_native_and_general_review=True)
 save(Z/'tolerance_assessment.json',tolerance);status('full_report_and_tolerance_ready',all_numeric_tolerances_pass=tolerance['all_numeric_tolerances_pass'])
if __name__=='__main__':
 try:main()
 except BaseException as e:save(Z/'controller_failure.json',dict(error=repr(e),time=time.time()));raise
