"""Two four-GPU codec candidates, followed by all 4100 native A examples each.

GPU ownership is node-local memory, never cross-node filesystem locking.
A released training pair immediately joins evaluation of any ready candidate.
A failed arm does not silently erase controls or stop other independent arms.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent
sys.path.insert(0,str(R/'cloud_a800'))
from checkpoints import save,verify
TRAIN=str(R/'.venv-train/bin/python');EVAL=str(R/'.venv-vllm/bin/python')
ARMS=['A_prism_union','A_prism_balanced']
condition=threading.Condition();available=[];pending=[];finished=0;failures=[]

def environment(indices,tag):
    env=os.environ.copy();cache=ROOT/'repair_cache'/('prism_expanded_'+tag)
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(str(x) for x in indices),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',USE_HF='1',USE_HUB_KERNELS='NO',
        IMAGE_MAX_TOKEN_NUM='1536',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USE_FLASHINFER_SAMPLER='0')
    for key in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','GH_TOKEN','GITHUB_TOKEN']:env.pop(key,None)
    for key,leaf in [('TRITON_CACHE_DIR','triton'),('XDG_CACHE_HOME','xdg'),('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:
        path=cache/leaf;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    return env

def run(cmd,env,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def train_arm(arm,pair,port):
    global finished
    out=ROOT/'prism_expanded_repairs'/arm;out.mkdir(parents=True,exist_ok=True)
    env=environment([x[0] for x in pair],arm)
    prefix=[TRAIN,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=4','--master_port',str(port),str(ROOT/'prism_expanded_train.py'),'--arm',arm]
    try:
        smoke=ROOT/'prism_expanded_smokes'/arm
        run(prefix+['--output',str(smoke),'--smoke','--stop-after','1'],env,out/'smoke.log')
        run(prefix+['--output',str(smoke),'--smoke','--stop-after','2','--resume',str(smoke/'checkpoint-1')],env,out/'smoke_resume.log')
        one=verify(smoke/'checkpoint-1',4);two=verify(smoke/'checkpoint-2',4)
        assert one['global_step']==1 and two['global_step']==2
        assert one['files']['adapter_model.safetensors']['sha256']!=two['files']['adapter_model.safetensors']['sha256']
        # Check real optimizer resume, not just a changing filename/global_step.
        check=[TRAIN,'-c',"import sys,torch; s=torch.load(sys.argv[1],map_location='cpu',weights_only=True)['state']; assert s and {int(v['step']) for v in s.values()}=={2}; print('optimizer resumed to2')",str(smoke/'checkpoint-2/optimizer.pt')]
        run(check,env,out/'resume_validation.log')
        save(out/'runtime_verified.json',dict(state='passed',checkpoint1=one['files']['adapter_model.safetensors']['sha256'],checkpoint2=two['files']['adapter_model.safetensors']['sha256'],optimizer_steps=2,final_mopd=False))
        run(prefix+['--output',str(out)],env,out/'training.log')
        receipt=json.loads((out/'complete.json').read_text());assert receipt['step']==512 and not receipt['smoke']
        merged=ROOT/('merged_prism_expanded_'+arm)
        run([TRAIN,str(ROOT/'merge_prism_expanded.py'),'--adapter',receipt['checkpoint'],'--output',str(merged)],env,out/'merge.log')
        shards=4
        with condition:
            pending.extend(dict(arm=arm,model=str(merged),shard=i,shards=shards) for i in range(shards))
            pending.sort(key=lambda x:(not x['arm'].startswith('A'),x['shard'],x['arm']))
            condition.notify_all()
    except BaseException as exc:
        with condition:failures.append(dict(stage='training',arm=arm,error=str(exc)))
        save(out/'failure.json',dict(stage='training',error=str(exc),time=time.time()))
    finally:
        with condition:
            available.extend(pair);finished+=1;condition.notify_all()

def eval_worker():
    while True:
        with condition:
            while not (available and pending):
                if finished==len(ARMS) and not pending:return
                condition.wait(20)
            gpu,uuid=available.pop(0);task=pending.pop(0)
        arm=task['arm'];out=ROOT/'prism_expanded_evaluations'/arm/str(task['shard']);out.mkdir(parents=True,exist_ok=True)
        env=environment([gpu],'eval_'+str(gpu));env['ANVIL_EXPECTED_GPU_UUID']=uuid
        try:
            argv=[EVAL,str(ROOT/'dof_codebook_eval.py'),'--model',task['model'],'--name',arm,'--scope','A',
                '--gpu-uuid',uuid,'--shard',str(task['shard']),'--shards',str(task['shards']),
                '--concurrency','64','--memory','.88','--output',str(out/'evaluation')]
            save(out/'launch.json',dict(argv=argv,gpu=gpu,uuid=uuid,started=time.time()))
            run(argv,env,out/'console.log')
            metrics=json.loads((out/'evaluation/metrics.json').read_text());assert metrics['state']=='complete_scope_shard'
            save(out/'complete.json',dict(task=task,n=metrics['n'],metrics_sha256=hashlib.sha256((out/'evaluation/metrics.json').read_bytes()).hexdigest()))
        except BaseException as exc:
            with condition:failures.append(dict(stage='evaluation',arm=arm,shard=task['shard'],error=str(exc)))
            save(out/'failure.json',dict(error=str(exc),time=time.time()))
        finally:
            with condition:available.append((gpu,uuid));condition.notify_all()

def main():
    frozen=json.loads((ROOT/'prism_expanded_experiment.json').read_text())
    for path,expected in frozen['frozen_files'].items():
        with Path(path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,path
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines();assert len(rows)==8
    devices=[]
    for row in rows:
        i,u,n,m=map(str.strip,row.split(','));assert 'A800' in n and int(m)>=79000;devices.append((int(i),u))
    devices.sort();save(ROOT/'prism_expanded_node_status.json',dict(stage='running',devices=devices,started=time.time(),gpus=8))
    # Separate executors avoid evaluation waits consuming training thread slots.
    with ThreadPoolExecutor(max_workers=8) as evaluators,ThreadPoolExecutor(max_workers=2) as trainers:
        ef=[evaluators.submit(eval_worker) for _ in range(8)]
        tf=[trainers.submit(train_arm,arm,devices[4*i:4*i+4],29810+i) for i,arm in enumerate(ARMS)]
        for f in as_completed(tf):f.result()
        for f in as_completed(ef):f.result()
    save(ROOT/'prism_expanded_node_status.json',dict(stage='failed' if failures else 'complete_requires_quality_review',failures=failures,completed=time.time()))
    if failures:raise RuntimeError(str(failures))
    subprocess.run([TRAIN,str(ROOT/'score_prism_expanded.py')],check=True)

if __name__=='__main__':main()
