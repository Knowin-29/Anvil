"""Eight-GPU interface/resume proof, followed by a full-budget execution probe.

Candidate teacher weights are explicitly unqualified. Nothing here starts a
formal MOPD run, chooses the production update budget, or uploads a model.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent
sys.path[:0]=[str(ROOT),str(C),str(R/'cloud_a800')]
from repair_node import environment,run,TRAIN
from checkpoints import save,verify


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    manifest=json.loads((ROOT/'mopd_v5_runtime_experiment.json').read_text())
    for path,digest in manifest['frozen_files'].items():assert sha(path)==digest,path
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(devices)==8
    for line in devices:
        i,u,n,m=map(str.strip,line.split(','));assert 'A800' in n and int(m)>=79000
    env=environment(list(range(8)),'mopd_v5_runtime')
    (ROOT/'repair_cache/mopd_v5_runtime/xdg/torch/kernels').mkdir(parents=True,exist_ok=True)
    out=ROOT/'mopd_v5_runtime';out.mkdir(exist_ok=True)
    save(out/'node_status.json',dict(stage='runtime_preflight',started=time.time(),gpus=8))
    prefix=[TRAIN,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=8','--master_port','29785',
        str(C/'native_mopd_v5.py'),'--output',str(out),'--global-batch','40','--smoke',
        '--smoke-forms','QA,bbox,direction,dof,point','--smoke-dataset',str(ROOT/'mopd_v5_runtime_fixtures.jsonl'),
        '--smoke-token-cap','64','--save-every','2','--dof-representation','rotation128','--spatial-replay',
        '--runtime-bindings',str(ROOT/'mopd_v5_runtime_bindings.json')]
    run(prefix+['--steps','2'],env,out/'first.log')
    run(prefix+['--steps','3','--resume',str(out/'checkpoint-2')],env,out/'resume.log')
    a,b=verify(out/'checkpoint-2',8),verify(out/'checkpoint-3',8)
    assert a['global_step']==2 and b['global_step']==3
    assert a['files']['adapter_model.safetensors']['sha256']!=b['files']['adapter_model.safetensors']['sha256']
    run([TRAIN,'-c',"import torch,sys; state=torch.load(sys.argv[1],map_location='cpu',weights_only=True)['state']; assert state and {int(v['step']) for v in state.values()}=={3}",str(out/'checkpoint-3/optimizer.pt')],env,out/'optimizer_verify.log')
    counts=Counter();native=Counter();seen=set()
    for rank in range(8):
        for row in map(json.loads,(out/f'exposure_rank{rank}.jsonl').open()):
            key=(row['step'],row['micro'],row['rank']);assert key not in seen;seen.add(key)
            counts[row['source_category']]+=1
            if row['source_category']!='native_distillation':
                assert row['rollout_tokens']==row['teacher_scored_positions']==0
            else:
                assert row['teacher_scored_positions']>0
                native[row['family']]+=1
    assert counts==dict(general_qa=24,native_distillation=72,spatial_reference=24)
    assert native==dict(G=24,S=24,A=24)
    state=json.loads((out/'checkpoint-3/trainer_state.json').read_text())
    assert state['source_exposures']==counts
    assert all(__import__('math').isfinite(item[key]) for item in state['log_history'] for key in ['sft','mtd','grad_norm'])
    assert all(item['mtd']>0 and item['grad_norm']>0 for item in state['log_history'])
    protocol=json.loads((out/'protocol.json').read_text())
    receipt=dict(state='passed_execution_only',world=8,optimizer_step=3,changed_adapter=True,
        representation_sha256=protocol['model_representation_sha256'],
        source_exposures=dict(counts),native_families=dict(native),replay_reference_only=True,
        native_mtd_positions_nonzero=True,teacher_qualified=False,formal_mopd_started=False,
        production_throughput_measured=False,protocol_sha256=sha(out/'protocol.json'),
        checkpoint3_manifest_sha256=sha(out/'checkpoint-3/checkpoint_complete.json'),completed=time.time())
    save(out/'runtime_verified.json',receipt)
    # Full budgets and actual-pool sampling, but still the explicitly unqualified
    # diagnostic checkpoints. Its throughput cannot set the final model budget.
    probe=ROOT/'mopd_v5_full_budget_probe';probe.mkdir(exist_ok=True)
    save(out/'node_status.json',dict(stage='full_budget_execution_probe',runtime_verified=True,started=time.time()))
    full=[TRAIN,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=8','--master_port','29786',
        str(C/'native_mopd_v5.py'),'--output',str(probe),'--global-batch','32','--smoke','--full-budget-probe',
        '--steps','4','--save-every','2','--dof-representation','rotation128','--spatial-replay',
        '--runtime-bindings',str(ROOT/'mopd_v5_runtime_bindings.json')]
    try:
        run(full,env,probe/'console.log')
        final=verify(probe/'checkpoint-4',8)
        assert final['global_step']==4
        s=json.loads((probe/'checkpoint-4/trainer_state.json').read_text())
        assert s['source_exposures']==dict(native_distillation=64,general_qa=32,spatial_reference=32)
        counts=Counter();forms=Counter();tokens=[]
        for rank in range(8):
            for row in map(json.loads,(probe/f'exposure_rank{rank}.jsonl').open()):
                counts[row['source_category']]+=1
                if row['source_category']=='native_distillation':
                    forms[row['form']]+=1;tokens.append(row['rollout_tokens'])
        assert dict(counts)==s['source_exposures']
        save(probe/'probe_verified.json',dict(state='passed_full_budget_execution_probe',steps=4,
            source_exposures=dict(counts),native_forms=dict(forms),rollout_tokens=tokens,
            seconds_per_update=[r['seconds'] for r in s['log_history']],
            peak_gpu_gib=max(r['peak_gpu_gib'] for r in s['log_history']),
            checkpoints_and_decoder_bound=True,qualified_final_candidate=False,
            production_budget_frozen=False,formal_mopd=False,completed=time.time()))
        save(out/'node_status.json',dict(stage='complete_execution_only',completed=time.time()))
    except BaseException as exc:
        save(probe/'failure.json',dict(stage='full_budget_execution_probe',error=str(exc),time=time.time()))
        raise


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        out=ROOT/'mopd_v5_runtime';out.mkdir(exist_ok=True)
        save(out/'failure.json',dict(error=str(exc),time=time.time()))
        raise
