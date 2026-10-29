"""Two-node execution proof and equal-global-batch throughput comparison.

This remains an execution diagnostic with unqualified candidate teachers.
No file locking is used to allocate or arbitrate GPUs across nodes.
"""
import argparse,hashlib,json,math,os,subprocess,sys,time
from collections import Counter
from pathlib import Path
C=Path(__file__).resolve().parent;D=C/'diagnosis_32gpu';R=C.parent
sys.path[:0]=[str(D),str(R/'cloud_a800')]
from repair_node import environment,run,TRAIN
from checkpoints import save,verify


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def topology(env):
    nodes=int(env.get('MLP_WORKER_NUM',env.get('NNODES','0')))
    rank=int(env.get('MLP_ROLE_INDEX',env.get('NODE_RANK','-1')))
    gpus=int(env.get('MLP_WORKER_GPU',env.get('NPROC_PER_NODE','0')))
    address=env.get('MLP_WORKER_0_HOST') or env.get('MASTER_ADDR')
    port=env.get('MLP_WORKER_0_PORT') or env.get('MASTER_PORT','29500')
    assert nodes==2 and gpus==8 and rank in (0,1),'Requires two distinct eight-GPU workers'
    assert address and address not in ('127.0.0.1','localhost'),'Cross-node master address missing'
    assert str(port).isdigit() and 1<=int(port)<=65535,'Invalid rendezvous port'
    return rank,address,int(port)


def inspect_exposures(out,steps,world,expected):
    seen=set();counts=Counter();families=Counter();forms=Counter();tokens=[]
    for rank in range(world):
        for row in map(json.loads,(out/f'exposure_rank{rank}.jsonl').open()):
            key=(row['step'],row['micro'],row['rank']);assert key not in seen;seen.add(key)
            assert 1<=row['step']<=steps and row['rank']==rank and row['smoke']
            counts[row['source_category']]+=1
            if row['source_category']=='native_distillation':
                assert row['teacher_scored_positions']>0
                families[row['family']]+=1;forms[row['form']]+=1;tokens.append(row['rollout_tokens'])
            else:assert row['teacher_scored_positions']==row['rollout_tokens']==0
    assert dict(counts)==expected
    return dict(source_exposures=dict(counts),native_families=dict(families),native_forms=dict(forms),rollout_tokens=tokens)


def finished_phase(root,phase,node,signature):
    # Synchronize completed child processes before reusing the cloud rendezvous
    # port. These immutable receipts coordinate stages, not GPU allocation.
    folder=root/'phases';folder.mkdir(exist_ok=True)
    path=folder/f'{phase}_node{node}.json';assert not path.exists()
    save(path,dict(phase=phase,node=node,experiment_sha256=signature,finished=time.time()))
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        assert not list(root.glob('node*_failure.json')),'Peer node failed'
        paths=[folder/f'{phase}_node{i}.json' for i in range(2)]
        if all(p.is_file() for p in paths):
            for i,p in enumerate(paths):
                value=json.loads(p.read_text());assert value['node']==i and value['experiment_sha256']==signature and value['phase']==phase
            return
        time.sleep(2)
    raise TimeoutError('Peer did not finish phase '+phase)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',type=Path,required=True);args=ap.parse_args()
    spec=json.loads(args.experiment.read_text());node,address,port=topology(os.environ)
    assert spec['world']==16 and spec['probe_steps']==24 and spec['global_batch']==32
    assert spec['formal_mopd'] is False and spec['teacher_qualified'] is False
    for path,digest in spec['frozen_files'].items():assert sha(path)==digest,path
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(devices)==8
    for row in devices:
        index,uid,name,memory=map(str.strip,row.split(','));assert 'A800' in name and int(memory)>=79000
    root=Path(spec['output']);root.mkdir(parents=True,exist_ok=True)
    env=environment(list(range(8)),'mopd_v6_multinode_'+str(node))
    env.update(NNODES='2',NODE_RANK=str(node),NPROC_PER_NODE='8',MASTER_ADDR=address,MASTER_PORT=str(port),
               TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    (Path(env['XDG_CACHE_HOME'])/'torch/kernels').mkdir(parents=True,exist_ok=True)
    launcher=[TRAIN,'-m','torch.distributed.run','--nnodes=2','--nproc_per_node=8',f'--node_rank={node}',
        f'--master_addr={address}',f'--master_port={port}','--max_restarts=0','--rdzv_conf=timeout=300']
    status=root/f'node{node}_status.json';assert not status.exists(),'Do not reuse an execution attempt'
    signature=sha(args.experiment)
    save(status,dict(stage='collective_preflight',node=node,world=16,started=time.time()))
    run(launcher+[str(C/'mopd_distributed_preflight.py'),'--output',str(root/'collectives')],env,root/f'collectives_node{node}.log')
    finished_phase(root,'collectives',node,signature)
    out=root/'resume';out.mkdir(exist_ok=True)
    prefix=launcher+[str(C/'native_mopd_v6.py'),'--output',str(out),'--global-batch','80','--smoke',
        '--smoke-forms','QA,bbox,direction,dof,point','--smoke-dataset',spec['fixtures'],'--smoke-token-cap','64',
        '--save-every','2','--dof-representation','rotation128','--spatial-replay',
        '--runtime-bindings',spec['runtime_bindings'],'--initialization-lineage',spec['initialization_lineage']]
    save(status,dict(stage='real_optimizer_resume',node=node,updated=time.time()))
    run(prefix+['--steps','2'],env,out/f'first_node{node}.log')
    finished_phase(root,'first_two_updates',node,signature)
    run(prefix+['--steps','3','--resume',str(out/'checkpoint-2')],env,out/f'resume_node{node}.log')
    if node==0:
        one,two=verify(out/'checkpoint-2',16),verify(out/'checkpoint-3',16)
        assert one['global_step']==2 and two['global_step']==3
        assert one['files']['adapter_model.safetensors']['sha256']!=two['files']['adapter_model.safetensors']['sha256']
        run([TRAIN,'-c',"import sys,torch; s=torch.load(sys.argv[1],map_location='cpu',weights_only=True)['state']; assert s and {int(v['step']) for v in s.values()}=={3}",str(out/'checkpoint-3/optimizer.pt')],env,out/'optimizer_verify.log')
        counts=inspect_exposures(out,3,16,dict(native_distillation=144,general_qa=48,spatial_reference=48))
        assert counts['native_families']==dict(G=48,S=48,A=48)
        state=json.loads((out/'checkpoint-3/trainer_state.json').read_text())
        assert all(math.isfinite(x[k]) for x in state['log_history'] for k in ['sft','mtd','grad_norm'])
        assert all(x['mtd']>0 and x['grad_norm']>0 for x in state['log_history'])
        protocol=json.loads((out/'protocol.json').read_text())
        save(out/'runtime_verified.json',dict(state='passed_execution_only',world=16,nodes=2,optimizer_step=3,
            changed_adapter=True,representation_sha256=protocol['model_representation_sha256'],
            trainer_code_sha256=sha(C/'native_mopd_v6.py'),student_adapter_sha256=sha(Path(protocol['initial'])/'adapter_model.safetensors'),
            teacher_adapter_sha256={f:sha(Path(p)/'adapter_model.safetensors') for f,p in protocol['teachers'].items()},
            **counts,teacher_qualified=False,formal_mopd_started=False,protocol_sha256=sha(out/'protocol.json'),completed=time.time()))
    finished_phase(root,'resume_verified',node,signature)
    probe=root/'probe';probe.mkdir(exist_ok=True)
    save(status,dict(stage='equal_batch_full_budget_probe',node=node,updated=time.time()))
    command=launcher+[str(C/'native_mopd_v6.py'),'--output',str(probe),'--global-batch','32','--smoke','--full-budget-probe',
        '--steps','24','--save-every','8','--dof-representation','rotation128','--spatial-replay',
        '--runtime-bindings',spec['runtime_bindings'],'--initialization-lineage',spec['initialization_lineage']]
    run(command,env,probe/f'console_node{node}.log')
    if node==0:
        final=verify(probe/'checkpoint-24',16);assert final['global_step']==24
        state=json.loads((probe/'checkpoint-24/trainer_state.json').read_text())
        counts=inspect_exposures(probe,24,16,dict(native_distillation=384,general_qa=192,spatial_reference=192))
        assert counts['source_exposures']==state['source_exposures'] and len(counts['native_forms'])==11
        assert all(math.isfinite(x[k]) for x in state['log_history'] for k in ['sft','mtd','grad_norm','seconds'])
        save(probe/'probe_verified.json',dict(state='passed_full_budget_execution_probe',world=16,nodes=2,steps=24,
            global_batch=32,**counts,seconds_per_update=[r['seconds'] for r in state['log_history']],
            peak_gpu_gib=max(r['peak_gpu_gib'] for r in state['log_history']),
            trainer_code_sha256=sha(C/'native_mopd_v6.py'),runtime_bindings_sha256=sha(spec['runtime_bindings']),
            qualified_final_candidate=False,production_budget_frozen=False,formal_mopd=False,completed=time.time()))
    save(status,dict(stage='complete_execution_only',node=node,completed=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if '--experiment' in sys.argv:
            spec=json.loads(Path(sys.argv[sys.argv.index('--experiment')+1]).read_text());root=Path(spec['output']);root.mkdir(parents=True,exist_ok=True)
            node=os.environ.get('MLP_ROLE_INDEX',os.environ.get('NODE_RANK','unknown'))
            save(root/f'node{node}_failure.json',dict(error=repr(exc),time=time.time()))
        raise
