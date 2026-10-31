"""CPU verification of an immutable checkpoint and its committed log prefixes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from safetensors.torch import load_file

C = Path('/Knowin/sim/weilinruan/Anvil/recovery_20261020/completion_20261028')
PIPELINE = C/'targeted_repair_20261031'
SPEC_SHA = 'ad69923106cb9dd8d995e8363071867f63fb1c2d887e5a49f1d9d925ae8b1511'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list,tuple)):
        return all(finite(item) for item in value)
    return math.isfinite(value) if isinstance(value,float) else True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['repair','mopd'],required=True)
    ap.add_argument('--step',type=int,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();step=args.step;stage=args.stage;run=PIPELINE/'runs'/stage;ckpt=run/f'checkpoint-{step}'
    require(not args.output.exists(),'Do not overwrite verification evidence')
    torch.set_num_threads(2)
    require(sha(PIPELINE/'spec.json')==SPEC_SHA,'Training specification changed')
    spec=read(PIPELINE/'spec.json');cfg=spec['stages'][stage]
    require(0<step<=cfg['steps'],'Step outside planned stage')
    state=read(ckpt/'trainer_state.json');seal=read(ckpt/'checkpoint_complete.json');protocol=read(run/'protocol.json')
    required={'README.md','adapter_config.json','adapter_model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json'}
    required|={f'rng_state_{rank}.pth' for rank in range(32)}
    require(set(seal['files'])==required and seal['global_step']==step and seal['world_size']==32,'Wrong checkpoint membership or step')
    for name,meta in seal['files'].items():
        file=ckpt/name
        require(file.is_file() and not file.is_symlink() and file.stat().st_size==meta['bytes'] and sha(file)==meta['sha256'],'Changed checkpoint payload: '+name)
    print('All 38 payload hashes verified',flush=True)
    require(state['global_step']==step and state['max_steps']==cfg['steps'] and state['stage']==stage and
            state['examples_seen']==step*64 and state['final_mopd']==(stage=='mopd') and state['smoke'] is False,'Wrong trainer state')
    require(protocol['spec_sha256']==SPEC_SHA and protocol['world_size']==32 and protocol['global_batch']==64 and
            protocol['updates']==cfg['steps'] and protocol['stage']==stage and protocol['code_sha256']==sha(PIPELINE/'train.py'),'Wrong protocol binding')
    initial=Path(protocol['initial'])
    if stage=='repair':
        require(initial==Path(spec['initial']['adapter']) and protocol['initial_adapter_sha256']==spec['initial']['adapter_sha256'],'Wrong completed-parent initialization')
    else:
        parent=read(PIPELINE/'runs/repair/complete.json')
        require(initial==Path(parent['checkpoint']) and parent['steps']==spec['stages']['repair']['steps'] and
                protocol['initial_adapter_sha256']==parent['adapter_sha256'],'Wrong repair to MOPD initialization')
    require(sha(initial/'adapter_model.safetensors')==protocol['initial_adapter_sha256'],'Initial adapter changed')
    adapter=load_file(str(ckpt/'adapter_model.safetensors'),device='cpu');before=load_file(str(initial/'adapter_model.safetensors'),device='cpu')
    require(set(adapter)==set(before) and len(adapter)==496,'Changed adapter tensor schema')
    changed=0
    for name,value in adapter.items():
        require(value.shape==before[name].shape and bool(torch.isfinite(value).all()),'Invalid adapter tensor: '+name)
        changed+=int(not torch.equal(value,before[name]))
    parameters=sum(t.numel() for t in adapter.values())
    require(changed>0 and parameters==173113344,'No parameter update or changed parameter count')
    del adapter,before
    optimizer=torch.load(ckpt/'optimizer.pt',map_location='cpu',weights_only=True)
    ids=[p for group in optimizer['param_groups'] for p in group['params']]
    require(len(ids)==len(set(ids))==496 and set(ids)==set(optimizer['state']),'Wrong optimizer membership')
    count=0
    for value in optimizer['state'].values():
        require(int(value['step'].item())==step,'Wrong optimizer step')
        m,v=value['exp_avg'],value['exp_avg_sq']
        require(m.shape==v.shape and bool(torch.isfinite(m).all()) and bool(torch.isfinite(v).all()) and bool((v>=0).all()),'Invalid optimizer moments')
        count+=m.numel()
    require(count==parameters,'Wrong optimizer parameter count')
    warmup=math.ceil(cfg['steps']*.03)
    rate=cfg['lr']*((step/warmup) if step-1<warmup else .5*(1+math.cos(math.pi*(step-1-warmup)/(cfg['steps']-warmup))))
    require(all(math.isclose(g['lr'],rate,rel_tol=1e-10,abs_tol=1e-20) for g in optimizer['param_groups']),'Wrong optimizer learning rate')
    del optimizer
    scheduler=torch.load(ckpt/'scheduler.pt',map_location='cpu',weights_only=True)
    require(scheduler==dict(step=step,steps=cfg['steps'],schedule='warmup3%-cosine'),'Wrong scheduler')
    for rank in range(32):
        rng=torch.load(ckpt/f'rng_state_{rank}.pth',map_location='cpu',weights_only=False)
        require(set(rng)=={'cpu','cuda','python','numpy'},'Missing RNG component')
        torch.Generator(device='cpu').set_state(rng['cpu'])
        require(rng['cuda'].dtype==torch.uint8 and rng['cuda'].ndim==1 and rng['cuda'].numel()>0,'Invalid CUDA RNG blob')
        random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
    with (run/'logging.jsonl').open() as stream:
        logs=[json.loads(line) for line in itertools.islice(stream,step)]
    require(logs==state['log_history'] and len(logs)==step and [r['step'] for r in logs]==list(range(1,step+1)) and finite(logs),'Wrong or nonfinite committed training history')
    positive=sum(row['mtd']>0 for row in logs)
    require(positive>0 if stage=='mopd' else positive==0,'Unexpected distillation updates for stage')
    print('Finite adapter, optimizer, scheduler, 32 RNG states and committed logs verified',flush=True)
    sources=Counter();forms=Counter();families=Counter();uids=set();images=set();source_ids=defaultdict(set);source_images=defaultdict(set)
    digest=hashlib.sha256();rows=native=teacher_rows=teacher_positions=native_tokens=0
    for rank in range(32):
        seen=set();expected=iter((s,m) for s in range(1,step+1) for m in (0,1))
        with (run/f'exposure_rank{rank}.jsonl').open('rb') as stream:
            for line in stream:
                row=json.loads(line)
                if row['step']>step:
                    break
                key=(row['step'],row['micro'])
                require(row['rank']==rank and row['smoke'] is False and row['stage']==stage and key==next(expected,None) and key not in seen,'Invalid or reordered exposure prefix')
                seen.add(key);digest.update(line);rows+=1
                category=row['source_category'];uid=row['canonical_uid'];rgb=row['source_binding']['decoded_rgb_sha256']
                require(bool(rgb),'Missing image binding')
                sources[category]+=1;uids.add(uid);images.add(rgb);source_ids[category].add(uid);source_images[category].add(rgb)
                if category=='native_distillation':
                    native+=1;forms[row['form']]+=1;families[row['family']]+=1
                    teacher_rows+=int(row['teacher_scored_positions']>0);teacher_positions+=row['teacher_scored_positions'];native_tokens+=row['rollout_tokens']
                    require(row['rollout_role']=='on_policy_mtd','Wrong native rollout role')
        require(len(seen)==step*2 and next(expected,None) is None,'Missing rank exposure prefix')
    require(rows==step*64 and dict(sources)==state['source_exposures'],'Exposure totals differ')
    require(native==teacher_rows,'Unscored native-distillation exposure')
    if step==cfg['steps']:
        complete=read(run/'complete.json')
        require(complete['steps']==step and Path(complete['checkpoint']).resolve()==ckpt.resolve() and complete['stage']==stage+'_training_complete' and not complete['smoke'],'Wrong stage completion receipt')
        require(complete['adapter_sha256']==sha(ckpt/'adapter_model.safetensors') and complete['spec_sha256']==SPEC_SHA,'Wrong final checkpoint binding')
        require(dict(sources)==complete['source_exposures']==spec['planned_source_exposures'][stage],'Wrong completed source mixture')
    result=dict(state='targeted_checkpoint_integrity_verified',checked_at=datetime.now(timezone.utc).isoformat(),stage=stage,step=step,
        checkpoint=str(ckpt),spec_sha256=SPEC_SHA,files=38,total_bytes=sum(v['bytes'] for v in seal['files'].values()),all_file_hashes_verified=True,
        adapter_tensors_finite=496,changed_adapter_tensor_count=changed,optimizer_parameters_verified=count,optimizer_step=step,scheduler_verified=scheduler,
        initial_adapter=str(initial),initial_adapter_sha256=protocol['initial_adapter_sha256'],all32_rng_files_loadable=True,
        cuda_rng_blobs_loaded_on_cpu_only=True,gpu_resume_experiment=False,finite_committed_training_history=True,positive_mtd_updates=positive,
        committed_exposure_prefix=dict(rows=rows,all32_rank_prefixes_complete=True,unique_records=len(uids),unique_image_bindings=len(images),
            source_exposures=dict(sources),source_unique_records={k:len(v) for k,v in source_ids.items()},source_unique_image_bindings={k:len(v) for k,v in source_images.items()},
            rank_order_prefix_sha256=digest.hexdigest(),native_rows=native,native_forms=dict(forms),native_families=dict(families),teacher_scored_rows=teacher_rows,
            teacher_scored_positions=teacher_positions,native_rollout_tokens=native_tokens,counts_are_not_dataset_inventory=True),
        verification_script_sha256=sha(Path(__file__)),heldout_quality_verified=False)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
