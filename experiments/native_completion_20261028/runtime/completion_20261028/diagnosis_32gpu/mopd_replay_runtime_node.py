"""Execution-only8GPU MOPD+QA/resume check; no teacher quality bypass for production."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(R/'cloud_a800'))
from repair_node import environment,run,TRAIN
from checkpoints import save,verify
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines();assert len(rows)==8
for line in rows:
 i,u,n,m=map(str.strip,line.split(','));assert 'A800' in n and int(m)>=79000
out=ROOT/'mopd_replay_runtime';out.mkdir(exist_ok=True)
env=environment(list(range(8)),'mopd_replay_runtime')
prefix=[TRAIN,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=8','--master_port','29750',str(C/'native_mopd_v3.py'),'--output',str(out),'--global-batch','32','--smoke','--smoke-forms','QA,bbox,direction,affordance','--smoke-token-cap','64','--save-every','2']
run(prefix+['--steps','2'],env,out/'first.log')
run(prefix+['--steps','3','--resume',str(out/'checkpoint-2')],env,out/'resume.log')
a=verify(out/'checkpoint-2',8);b=verify(out/'checkpoint-3',8)
assert a['global_step']==2 and b['global_step']==3 and a['files']['adapter_model.safetensors']['sha256']!=b['files']['adapter_model.safetensors']['sha256']
run([TRAIN,'-c',"import torch,sys; s=torch.load(sys.argv[1],map_location='cpu',weights_only=True)['state']; assert s and {int(v['step']) for v in s.values()}=={3}",str(out/'checkpoint-3/optimizer.pt')],env,out/'optimizer_verify.log')
from collections import Counter
counts=Counter();spatial_mtd=False
for rank in range(8):
 for row in map(json.loads,(out/f'exposure_rank{rank}.jsonl').open()):
  counts[row['family']]+=1
  if row['family']=='QA':assert row['rollout_tokens']==row['teacher_scored_positions']==0
  else:spatial_mtd|=row['teacher_scored_positions']>0
assert counts==dict(QA=24,G=24,S=24,A=24) and spatial_mtd
save(out/'runtime_verified.json',dict(state='passed_execution_only',world=8,optimizer_step=3,changed_adapter=True,exposures=dict(counts),qa_reference_only=True,spatial_mtd_positions_nonzero=spatial_mtd,production_throughput_measured=False,teachers_qualified=False,formal_mopd_started=False))
