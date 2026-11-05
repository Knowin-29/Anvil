"""Resume exact visual checkpoint768, then finish the unchanged repair and MOPD."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent;C=W.parent;V=C/'visual_geometry_candidate_20261104'
sys.path[:0]=[str(W),str(C/'diagnosis_32gpu'),str(C.parent/'cloud_a800')]
from retention_node import environment,TRAIN
from checkpoints import save,sha,verify
from storage_budget import release,before_checkpoint

def main():
 node=int(os.environ['MLP_ROLE_INDEX']);assert int(os.environ['MLP_WORKER_NUM'])==4 and 0<=node<4
 policy=json.loads((W/'resume_policy.json').read_text());spec=json.loads((V/'spec.json').read_text())
 for p,h in policy['files'].items():assert sha(p)==h,p
 assert sha(V/'spec.json')==policy['pipeline_spec_sha256']
 for p,h in spec['frozen_files'].items():assert sha(p)==h,p
 resume=Path(policy['resume_checkpoint']);assert resume==V/'runs/repair/checkpoint-768' and sha(resume/'checkpoint_complete.json')==policy['resume_seal_sha256']
 assert not (V/'runs/repair/complete.json').exists() and not (V/'runs/mopd/complete.json').exists()
 if node==0:assert verify(resume,32)['global_step']==768;release('training_log_headroom',8)
 devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines();assert len(devices)==8
 for line in devices:
  _,_,model,memory=map(str.strip,line.split(','));assert 'A800' in model and int(memory)>=79000
 env=environment(range(8),'visual_resume_'+str(node));env['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1'
 for key,leaf in [('TRITON_CACHE_DIR','triton'),('XDG_CACHE_HOME','xdg'),('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:
  cache=Path('/tmp/anvil-visual-resume')/str(node)/leaf;cache.mkdir(parents=True,exist_ok=True);env[key]=str(cache)
 (Path(env['XDG_CACHE_HOME'])/'torch/kernels').mkdir(parents=True,exist_ok=True)
 probe=V/'runs/repair'/('.resume_storage_probe_'+str(node));blocks=3072 if node==0 else 64
 try:
  with probe.open('xb') as f:
   block=os.urandom(1024**2)
   for _ in range(blocks):f.write(block)
   f.flush();os.fsync(f.fileno())
  assert probe.stat().st_size==blocks*1024**2
 finally:probe.unlink(missing_ok=True)
 save(W/f'node{node}_preflight.json',dict(state='passed',write_bytes=blocks*1024**2,time=time.time()))
 address=os.environ['MLP_WORKER_0_HOST'];assert address not in ('localhost','127.0.0.1')
 for stage in ['repair','mopd']:
  if node==0:before_checkpoint(stage,768 if stage=='repair' else 0,spec)
  argv=[TRAIN,'-m','torch.distributed.run','--nnodes=4','--nproc_per_node=8','--node_rank='+str(node),'--master_addr='+address,'--master_port='+os.environ.get('MLP_WORKER_0_PORT','29890'),'--max_restarts=0','--rdzv_conf=timeout=600',str(V/'train.py'),'--spec',str(V/'spec.json'),'--stage',stage]
  if stage=='repair':argv+=['--resume',str(resume)]
  save(W/f'node{node}_{stage}_launch.json',dict(node=node,argv=argv,devices=devices,started=time.time(),original_trainer_unchanged=True))
  log=W/f'{stage}_console_node{node}.log'
  with log.open('x') as stream:
   proc=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   while proc.poll() is None:
    time.sleep(2)
    if node==0:
     p=V/'runs'/stage/'status.json'
     if p.exists():
      try:status=json.loads(p.read_text())
      except json.JSONDecodeError:status={}
      before_checkpoint(stage,status.get('step',768 if stage=='repair' else 0),spec)
    with log.open('rb') as f:f.seek(max(0,log.stat().st_size-32768));tail=f.read()
    local_fatal=any(x in tail for x in [b'Disk quota exceeded',b'No space left on device',b'OutOfMemoryError:',b'SafetensorError:',b']: Traceback'])
    if local_fatal:save(W/f'fatal_node{node}.json',dict(node=node,stage=stage,time=time.time()))
    if local_fatal or list(W.glob('fatal_node*.json')):
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
     raise RuntimeError('Fatal training exception; torchrun group stopped promptly')
   assert proc.returncode==0,f'{stage} torchrun exited {proc.returncode}'
  complete=json.loads((V/'runs'/stage/'complete.json').read_text());assert complete['steps']==spec['stages'][stage]['steps']
  if node==0:verify(complete['checkpoint'],32)
  save(W/f'node{node}_{stage}_complete.json',dict(state=stage+'_complete',checkpoint=complete['checkpoint'],time=time.time()))
  deadline=time.time()+600
  while not all((W/f'node{i}_{stage}_complete.json').exists() for i in range(4)):
   assert time.time()<deadline;time.sleep(2)
 save(W/f'node{node}_complete.json',dict(state='repair8192_mopd256_complete',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as e:save(W/('node'+os.environ.get('MLP_ROLE_INDEX','unknown')+'_failure.json'),dict(error=repr(e),time=time.time()));raise
