"""Single 8-GPU adaptation; real resume test and disjoint full evaluations."""
from concurrent.futures import ThreadPoolExecutor
import argparse,json,hashlib,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent
sys.path.insert(0,str(R/'cloud_a800'))
from checkpoints import save,verify
from dof_pose_node import environment,run
from student_adaptation_pool import load_spec
TRAIN=str(R/'.venv-train/bin/python');EVAL=str(R/'.venv-vllm/bin/python')

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True);args=ap.parse_args();spec=load_spec(args.spec)
 devices=[]
 for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines():
  index,uid,name,memory=map(str.strip,row.split(','));assert 'A800' in name and int(memory)>=79000
  devices.append((int(index),uid))
 devices.sort();assert len(devices)==8
 out=ROOT/'student_adaptation_run/main';out.mkdir(parents=True,exist_ok=True)
 status=ROOT/'student_adaptation_node_status.json'
 env=environment([d[0] for d in devices],'student_adaptation')
 prefix=[TRAIN,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=8','--master_port','29830',str(ROOT/'student_adaptation_train.py'),'--spec',str(args.spec)]
 try:
  save(status,dict(stage='gpu_resume_preflight',devices=devices,started=time.time(),spec_sha256=sha(args.spec)))
  smoke=ROOT/'student_adaptation_smoke'
  run(prefix+['--output',str(smoke),'--smoke','--stop-after','1'],env,out/'smoke.log')
  run(prefix+['--output',str(smoke),'--smoke','--stop-after','2','--resume',str(smoke/'checkpoint-1')],env,out/'smoke_resume.log')
  one,two=verify(smoke/'checkpoint-1',8),verify(smoke/'checkpoint-2',8)
  assert one['global_step']==1 and two['global_step']==2
  assert one['files']['adapter_model.safetensors']['sha256']!=two['files']['adapter_model.safetensors']['sha256']
  run([TRAIN,'-c',"import sys,torch; s=torch.load(sys.argv[1],map_location='cpu',weights_only=True)['state']; assert s and {int(v['step']) for v in s.values()}=={2}; print('optimizer resumed to 2')",str(smoke/'checkpoint-2/optimizer.pt')],env,out/'resume_validation.log')
  save(out/'runtime_verified.json',dict(state='passed',checkpoint1=one['files']['adapter_model.safetensors']['sha256'],checkpoint2=two['files']['adapter_model.safetensors']['sha256'],optimizer_steps=2,final_mopd=False))
  save(status,dict(stage='training',updated=time.time()))
  run(prefix+['--output',str(out)],env,out/'training.log')
  receipt=json.loads((out/'complete.json').read_text());assert receipt['step']==spec['steps'] and not receipt['smoke']
  run([TRAIN,str(C/'final_delivery/exposure_audit.py'),'--run',str(out),'--output',str(out/'exposure_audit.json')],env,out/'exposure_audit.log')
  merged=ROOT/'merged_student_adaptation'
  run([TRAIN,str(ROOT/'merge_student_adaptation.py'),'--adapter',receipt['checkpoint'],'--output',str(merged)],env,out/'merge.log')
  save(status,dict(stage='full_public_general_and_A_evaluation',updated=time.time()))
  def evaluate(index):
   gpu,uid=devices[index];kind='public' if index<4 else 'A';shard=index%4
   destination=ROOT/('student_adaptation_benchmarks' if kind=='public' else 'student_adaptation_evaluations')/'student_adaptation'/str(shard)
   destination.mkdir(parents=True,exist_ok=True);local_env=environment([gpu],'student_adaptation_eval_'+str(gpu));local_env['ANVIL_EXPECTED_GPU_UUID']=uid
   if kind=='public':
    argv=[TRAIN,str(ROOT/'student_adaptation_benchmark_worker.py'),'--model',str(merged),'--role','student_adaptation','--shard',str(shard),'--shards','4']
   else:
    argv=[EVAL,str(ROOT/'dof_codebook_eval.py'),'--model',str(merged),'--name','student_adaptation','--scope','A','--gpu-uuid',uid,'--shard',str(shard),'--shards','4','--concurrency','64','--memory','.88','--output',str(destination/'evaluation')]
   save(destination/'launch.json',dict(argv=argv,gpu=gpu,uuid=uid,started=time.time()))
   try:
    run(argv,local_env,destination/'console.log')
    if kind=='public':assert json.loads((destination/'complete.json').read_text())['state']=='complete'
    else:
     metrics=json.loads((destination/'evaluation/metrics.json').read_text());assert metrics['state']=='complete_scope_shard'
     save(destination/'complete.json',dict(n=metrics['n'],metrics_sha256=sha(destination/'evaluation/metrics.json')))
   except BaseException as exc:
    save(destination/'failure.json',dict(error=str(exc),time=time.time()));raise
  with ThreadPoolExecutor(max_workers=8) as workers:list(workers.map(evaluate,range(8)))
  run([TRAIN,str(ROOT/'score_student_adaptation_public.py')],env,out/'public_scoring.log')
  run([TRAIN,str(ROOT/'score_student_adaptation_A.py')],env,out/'A_scoring.log')
  save(status,dict(stage='complete_requires_quality_review',completed=time.time(),final_mopd=False))
 except BaseException as exc:
  save(status,dict(stage='failed',error=str(exc),time=time.time()));raise

if __name__=='__main__':main()
