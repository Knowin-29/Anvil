"""Two statically disjoint 8-GPU nodes evaluate the useful G candidate completely."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent
sys.path.insert(0,str(R/'cloud_a800'))
from checkpoints import save
from retention_node import environment,run,EVAL

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--node',type=int,choices=[0,1],required=True);args=ap.parse_args()
 spec=json.loads((ROOT/'g_priority_native_spec.json').read_text())
 for path,digest in spec['frozen_files'].items():assert sha(path)==digest,path
 for name,digest in spec['model_files'].items():assert sha(Path(spec['model'])/name)==digest,name
 devices=[]
 for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines():
  index,uid,name,memory=map(str.strip,row.split(','));assert 'A800' in name and int(memory)>=79000;devices.append((int(index),uid))
 devices.sort();assert len(devices)==8
 tasks=[t for t in spec['tasks'] if t['node']==args.node];assert len(tasks)==8
 status=ROOT/('g_priority_native_node'+str(args.node)+'_status.json');save(status,dict(stage='running',started=time.time(),devices=devices,tasks=tasks))
 def evaluate(task):
  gpu,uid=devices[task['local_gpu']];out=ROOT/'G_spatial_priority_native'/str(task['shard']);out.mkdir(parents=True,exist_ok=True)
  env=environment([gpu],'G_priority_'+str(task['shard']));env['ANVIL_EXPECTED_GPU_UUID']=uid
  (Path(env['XDG_CACHE_HOME'])/'torch/kernels').mkdir(parents=True,exist_ok=True)
  argv=[EVAL,str(C/'eval_code/run_vllm_eval.py'),'--model',spec['model'],'--name','G_spatial_priority','--scope','G','--gpu-uuid',uid,'--shard',str(task['shard']),'--shards','16','--concurrency','64','--memory','.88','--output',str(out/'evaluation')]
  save(out/'launch.json',dict(argv=argv,task=task,gpu=gpu,uuid=uid,started=time.time(),spec_sha256=sha(ROOT/'g_priority_native_spec.json')))
  try:
   run(argv,env,out/'console.log');metrics=json.loads((out/'evaluation/metrics.json').read_text())
   assert metrics['state']=='complete_scope_shard' and metrics['n']==task['records']
   save(out/'complete.json',dict(task=task,metrics_sha256=sha(out/'evaluation/metrics.json'),spec_sha256=sha(ROOT/'g_priority_native_spec.json')))
  except BaseException as exc:
   save(out/'failure.json',dict(error=repr(exc),time=time.time()));raise
 try:
  with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(evaluate,tasks))
  save(status,dict(stage='node_complete_requires_aggregate',completed=time.time(),shards=[t['shard'] for t in tasks]))
 except BaseException as exc:
  save(status,dict(stage='failed',error=repr(exc),time=time.time()));raise

if __name__=='__main__':main()
