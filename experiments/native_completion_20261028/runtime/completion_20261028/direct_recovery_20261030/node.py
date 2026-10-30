"""Four-worker MOPD recovery from the completed corrective checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
O=Path(__file__).resolve().parent;C=O.parent;D=C/'diagnosis_32gpu';R=C.parent
sys.path[:0]=[str(D),str(R/'cloud_a800')]
from retention_node import environment,run,TRAIN
from checkpoints import save,sha,verify


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--spec-sha256',required=True)
    args=ap.parse_args();assert sha(args.spec)==args.spec_sha256
    spec=json.loads(args.spec.read_text())
    for filename,digest in spec['frozen_files'].items():assert sha(filename)==digest,filename
    nodes=spec['world']//8;node=int(os.environ.get('MLP_ROLE_INDEX','0'))
    assert int(os.environ.get('MLP_WORKER_NUM',str(nodes)))==nodes==4 and 0<=node<nodes
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(lines)==8
    for line in lines:
        i,u,n,m=map(str.strip,line.split(','));assert 'A800' in n and int(m)>=79000
    env=environment(range(8),'direct_completion_'+str(node));env['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1'
    (Path(env['XDG_CACHE_HOME'])/'torch/kernels').mkdir(parents=True,exist_ok=True)
    address=os.environ['MLP_WORKER_0_HOST'];port=os.environ.get('MLP_WORKER_0_PORT','29890')
    assert address not in ('localhost','127.0.0.1')
    save(O/f'node{node}_launch.json',dict(node=node,devices=lines,host=__import__('socket').gethostname(),
        spec_sha256=args.spec_sha256,started=time.time()))
    for stage in ['mopd']:
        out=O/'runs'/stage;out.mkdir(parents=True,exist_ok=True)
        assert not (out/'protocol.json').exists(),'Existing run needs explicit recovery; never overwrite'
        argv=[TRAIN,'-m','torch.distributed.run','--nnodes=4','--nproc_per_node=8',
            '--node_rank='+str(node),'--master_addr='+address,'--master_port='+port,
            '--max_restarts=0','--rdzv_conf=timeout=600',str(O/'train.py'),'--spec',str(args.spec),'--stage',stage]
        save(O/f'node{node}_status.json',dict(stage=stage,started=time.time(),argv=argv))
        run(argv,env,out/f'console_node{node}.log')
        receipt=json.loads((out/'complete.json').read_text());assert receipt['steps']==spec['stages'][stage]['steps']
        if node==0:verify(receipt['checkpoint'],spec['world'])
        save(O/f'node{node}_{stage}_complete.json',dict(completed=time.time(),checkpoint=receipt['checkpoint']))
        # All workers finished the prior torchrun before reusing its port.
        deadline=time.time()+600
        while not all((O/f'node{i}_{stage}_complete.json').exists() for i in range(nodes)):
            assert time.time()<deadline,'Stage handoff timed out'
            time.sleep(2)
    save(O/f'node{node}_status.json',dict(stage='training_complete_requires_full_validation',completed=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(O/('node'+os.environ.get('MLP_ROLE_INDEX','0')+'_failure.json'),dict(error=repr(exc),time=time.time()))
        raise
