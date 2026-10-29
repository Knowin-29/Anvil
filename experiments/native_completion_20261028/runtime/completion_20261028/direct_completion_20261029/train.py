"""Two substantive stages, real on-policy MOPD and continued proven mask repair.

No diagnostic arms, test-row training, scorer edits, or automatic quality claims.
The first production checkpoint is also the real execution/resume evidence.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

O=Path(__file__).resolve().parent;C=O.parent;D=C/'diagnosis_32gpu';R=C.parent
sys.path[:0]=[str(O),str(C/'Anvil'),str(C),str(D),str(R),str(R/'cloud_a800')]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import fla.ops.gated_delta_rule
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration,StoppingCriteriaList
from training_data import BASE,EMPTY_THINK,encode,processor
from checkpoints import sha,save,seal,verify
from memory_loss import chunked_ce
from native_mopd_v6 import install_hidden_forward,move,forward_inputs
from mopd_representation import Representation,RepresentationRouter,ExactCycleStop,install_linear_bytelevel_spans
from trains.mopd.chunked_heads import stage_sft_from_hidden,selected_mtd_from_hidden
from repair_train import token_unlikelihood,cycle_endpoint
from dof_pose_loss import PoseLoss
from eval_core import BUDGETS
from pool import Pool,SCHEDULES


def load_spec(path):
    spec=json.loads(Path(path).read_text())
    assert spec['schema']=='anvil-direct-corrective-MOPD-v1'
    for filename,digest in spec['frozen_files'].items():assert sha(filename)==digest,filename
    return spec


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True)
    ap.add_argument('--stage',choices=['repair','mopd'],required=True)
    ap.add_argument('--resume',type=Path);args=ap.parse_args()
    spec=load_spec(args.spec);stage=args.stage;cfg=spec['stages'][stage]
    rank,world,local=[int(os.environ.get(k,v)) for k,v in [('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')]]
    assert world==spec['world']==32 and spec['global_batch']==64
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl');torch.set_num_threads(4)
    torch.manual_seed(17+rank);random.seed(17+rank);np.random.seed(17+rank)
    out=O/'runs'/stage;out.mkdir(parents=True,exist_ok=True)
    initial=spec['initial']['adapter']
    if stage=='mopd':
        parent=json.loads((O/'runs/repair/complete.json').read_text())
        assert parent['stage']=='repair_training_complete' and parent['steps']==spec['stages']['repair']['steps']
        initial=parent['checkpoint']
        assert sha(Path(initial)/'adapter_model.safetensors')==parent['adapter_sha256']
    else:assert sha(Path(initial)/'adapter_model.safetensors')==spec['initial']['adapter_sha256']
    adapters={f:v['adapter'] for f,v in spec['teachers'].items()}
    for family,path in adapters.items():assert sha(Path(path)/'adapter_model.safetensors')==spec['teachers'][family]['adapter_sha256']
    representation=Representation('rotation128');install_linear_bytelevel_spans()
    interfaces=representation.validate_adapters(adapters,initial)
    assert not interfaces['student']['student_requires_representation_adaptation']
    pool=Pool(spec);accum=spec['global_batch']//world
    os.environ.update(ANVIL_DATA_CONTRACT='anvil-source-native-train-v1',TRAIN=str(pool.export))
    protocol=dict(schema='anvil-direct-corrective-MOPD-v1',stage=stage,spec_sha256=sha(args.spec),
        initial=initial,initial_adapter_sha256=sha(Path(initial)/'adapter_model.safetensors'),
        teachers=adapters,world_size=world,global_batch=spec['global_batch'],updates=cfg['steps'],
        schedule=SCHEDULES[stage],loss=cfg,base=BASE,model_representation=representation.descriptor['version'],
        model_representation_sha256=representation.digest,representation=representation.descriptor,
        decode_required_for_native_inference=True,scope='all_11_native_forms',
        original_full_corpus_records=5800000,sft_pool_already_covered=5220000,spatial_pool=4756000,
        datasets=spec['datasets'],inventory_is_actual_training_coverage=False,
        original_initialization_lineage=spec['initialization_lineage'],smoke=False,
        final_mopd=stage=='mopd',quality_qualified=False,final_quality_accepted=False,
        teacher_assessment=spec['teacher_assessment'],code_sha256=sha(__file__))
    if rank==0:
        path=out/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==protocol
        else:save(path,protocol)
        save(out/'status.json',dict(stage='loading',training_stage=stage,updated=time.time(),smoke=False))
    dist.barrier();install_hidden_forward()
    student=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,
        device_map={'':device},attn_implementation='sdpa',local_files_only=True)
    student=PeftModel.from_pretrained(student,str(args.resume or initial),is_trainable=True)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    student.enable_input_require_grads()
    trainable=[p for p in student.parameters() if p.requires_grad];assert sum(p.numel() for p in trainable)==173113344
    teacher=None
    if stage=='mopd':
        teacher=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,
            device_map={'':device},attn_implementation='sdpa',local_files_only=True)
        teacher=PeftModel.from_pretrained(teacher,adapters['G'],adapter_name='G',is_trainable=False)
        for family in 'SA':teacher.load_adapter(adapters[family],adapter_name=family,is_trainable=False)
        teacher.requires_grad_(False);teacher.eval()
    p=processor();router=RepresentationRouter(representation);pose=PoseLoss(p.tokenizer,'angular')
    head=student.get_output_embeddings();t_head=teacher.get_output_embeddings() if teacher else None
    optimizer=torch.optim.AdamW(trainable,lr=cfg['lr'],weight_decay=.01,fused=True)
    start=0;logs=[];exposures=Counter()
    if args.resume:
        assert args.resume.resolve().parent==out.resolve()
        verify(args.resume,world)
        state=json.loads((args.resume/'trainer_state.json').read_text())
        assert state['max_steps']==cfg['steps'] and state['stage']==stage
        optimizer.load_state_dict(torch.load(args.resume/'optimizer.pt',map_location=device,weights_only=True))
        start=state['global_step'];logs=state['log_history'];exposures.update(state['source_exposures'])
        assert {int(v['step']) for v in optimizer.state.values()}=={start}
        rng=torch.load(args.resume/f'rng_state_{rank}.pth',map_location='cpu',weights_only=False)
        torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],device)
        random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    model=DDP(student,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    begin=time.monotonic();qualified=dict.fromkeys('GSA',True)

    def checkpoint(step):
        dest=out/f'checkpoint-{step}';dest.mkdir(exist_ok=True)
        if rank==0:
            student.save_pretrained(dest,safe_serialization=True)
            torch.save(optimizer.state_dict(),dest/'optimizer.pt')
            torch.save(dict(step=step,steps=cfg['steps'],schedule='warmup3%-cosine'),dest/'scheduler.pt')
            save(dest/'trainer_state.json',dict(global_step=step,max_steps=cfg['steps'],stage=stage,
                log_history=logs,examples_seen=step*spec['global_batch'],source_exposures=dict(exposures),
                smoke=False,final_mopd=stage=='mopd',final_quality_accepted=False))
        torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),
            python=random.getstate(),numpy=np.random.get_state()),dest/f'rng_state_{rank}.pth')
        dist.barrier()
        if rank==0:
            seal(dest,world);save(out/'latest.json',dict(path=str(dest),step=step,stage=stage,smoke=False))
        dist.barrier();return dest

    exposure_path=out/f'exposure_rank{rank}.jsonl'
    assert args.resume or not exposure_path.exists(),'Use explicit resume; do not duplicate exposures'
    if args.resume:
        # A declared resume preserves the uncommitted suffix before trimming it.
        raw=exposure_path.read_bytes();lines=raw.splitlines(keepends=True);keep=[];suffix=[]
        for line in lines:
            try: row=json.loads(line)
            except ValueError:suffix.append(line);continue
            (keep if row['step']<=start else suffix).append(line)
        assert len(keep)==start*accum
        if suffix:
            backup=out/f'exposure_rank{rank}.uncommitted_before_resume_{time.time_ns()}.jsonl'
            backup.write_bytes(b''.join(suffix));exposure_path.write_bytes(b''.join(keep))
    with exposure_path.open('a') as stream:
        for step in range(start,cfg['steps']):
            optimizer.zero_grad(set_to_none=True);began=time.monotonic()
            stats=torch.zeros(6,device=device,dtype=torch.float64);local_counts=Counter()
            warmup=math.ceil(cfg['steps']*.03)
            ratio=min(1.,(step+1)/warmup) if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/(cfg['steps']-warmup)))
            for group in optimizer.param_groups:group['lr']=cfg['lr']*ratio
            for micro in range(accum):
                source,kind,category=pool.sample(stage,step,micro,rank,world,spec['global_batch'])
                row=representation.row(source);data=move(encode(p,row),device)
                is_mtd=stage=='mopd' and kind=='native';is_ul=kind=='mask' and micro==0
                sampled=None;cycle=None;positions=[];generated=0;teacher_positions=0
                if is_mtd or is_ul:
                    target_start=int((data['labels'][0]!=-100).nonzero()[0])
                    prompt_length=target_start+len(p.tokenizer.encode(EMPTY_THINK,add_special_tokens=False))
                    prompt=forward_inputs(data)
                    for key in ['input_ids','attention_mask','mm_token_type_ids']:
                        if key in prompt:prompt[key]=prompt[key][:,:prompt_length]
                    student.eval();guard=ExactCycleStop(prompt_length)
                    generation=dict(do_sample=True,temperature=.9,top_p=1.,top_k=0) if is_mtd else dict(do_sample=False)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        ids=student.generate(**prompt,max_new_tokens=BUDGETS[row['gsa_form']] if is_mtd else cfg['ul_rollout_cap'],
                            **generation,use_cache=True,logits_to_keep=1,eos_token_id=p.tokenizer.eos_token_id,
                            pad_token_id=p.tokenizer.pad_token_id,stopping_criteria=StoppingCriteriaList([guard]))
                    cycle=guard.cycle;generated=ids.shape[1]-prompt_length
                    if is_ul and cycle:positions=list(range(ids.shape[1]-min(64,cycle['span_tokens']),ids.shape[1]))
                    sampled=dict(prompt,input_ids=ids,attention_mask=torch.ones_like(ids))
                    if 'mm_token_type_ids' in prompt:
                        sampled['mm_token_type_ids']=torch.cat((prompt['mm_token_type_ids'],torch.zeros_like(ids[:,prompt_length:])),dim=1)
                    if is_mtd:
                        sampled['labels']=ids.clone();sampled['labels'][:,:prompt_length]=-100
                student.train();sync=model.no_sync() if micro<accum-1 else nullcontext()
                with sync,torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=model(**forward_inputs(data),anvil_hidden_only=True,use_cache=False)
                    if is_mtd:
                        reference=router.plan([row],data,p.tokenizer,reference=True,qualified=qualified,reference_prefix=EMPTY_THINK)
                        loss=stage_sft_from_hidden(hidden[:,:-1],head,reference.token_ids,reference.stage_ids,chunk_size=128)
                        weight=cfg['native_sft_weight']
                    elif row['gsa_form']=='dof':
                        loss=pose(hidden,head,data['labels'],row,source['messages'][1]['content'])[0];weight=1.
                    else:loss=chunked_ce(hidden,head,data['labels']);weight=1.
                    assert torch.isfinite(loss);(weight*loss/accum).backward();stats[0]+=loss.detach()/accum
                    del hidden,loss
                    if is_mtd:
                        plan=router.plan([row],sampled,p.tokenizer,reference=False,qualified=qualified)
                        teacher.set_adapter(row['gsa_family']);teacher.requires_grad_(False);teacher.eval()
                        student_hidden=model(**forward_inputs(sampled),anvil_hidden_only=True,use_cache=False)
                        with torch.no_grad():teacher_hidden=teacher(**forward_inputs(sampled),anvil_hidden_only=True,use_cache=False)
                        mtd=selected_mtd_from_hidden(student_hidden[:,:-1],teacher_hidden[:,:-1],head,t_head,
                            plan.token_ids,plan.stage_ids,plan.selected_stages,plan.structural_mask,
                            topk=128,beta=.5,structural_alpha=2.,temperature=.9,chunk_size=16)
                        eligible=(plan.selected_stages>=0).sum().float();total=eligible.clone();dist.all_reduce(total)
                        weighted=mtd*(eligible*world/total.clamp_min(1))
                        assert torch.isfinite(mtd);(cfg['mtd_weight']*weighted/accum).backward()
                        stats[1]+=mtd.detach()/accum;stats[4]+=1;teacher_positions=int((plan.stage_ids>=0).sum())
                        del student_hidden,teacher_hidden,mtd,weighted,plan,reference
                    if is_ul:
                        hidden=model(**forward_inputs(sampled),anvil_hidden_only=True,use_cache=False)
                        ul=token_unlikelihood(hidden,head,ids,positions);assert torch.isfinite(ul)
                        (cfg['ul_weight']*ul).backward();stats[2]+=ul.detach();stats[3]+=bool(cycle);stats[5]+=1
                        del hidden,ul
                local_counts[category]+=1
                stream.write(json.dumps(dict(step=step+1,micro=micro,rank=rank,stage=stage,
                    canonical_uid=source['canonical_uid'],form=source['gsa_form'],family=source['gsa_family'],
                    source_category=category,source_binding=source['gsa_meta']['binding'],
                    original_target_sha256=hashlib.sha256(source['messages'][1]['content'].encode()).hexdigest(),
                    derived_target_sha256=hashlib.sha256(row['messages'][1]['content'].encode()).hexdigest(),
                    reference_tokens=int((data['labels']!=-100).sum()),rollout_tokens=generated,
                    rollout_role='on_policy_mtd' if is_mtd else 'greedy_cycle_unlikelihood' if is_ul else None,
                    teacher_scored_positions=teacher_positions,cycle=cycle,smoke=False))+'\n');stream.flush()
                if sampled is not None:del sampled,ids,prompt
                del data,row,source
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.);assert torch.isfinite(norm)
            optimizer.step();dist.all_reduce(stats);stats/=world
            # Stage schedule guarantees the same source category on every rank.
            for key,n in local_counts.items():exposures[key]+=n*world
            entry=dict(step=step+1,max_steps=cfg['steps'],training_stage=stage,sft=float(stats[0]),
                mtd=float(stats[1]),ul=float(stats[2]),cycle_fraction=float(stats[3]),
                mtd_rows_per_rank=float(stats[4]),ul_rows_per_rank=float(stats[5]),
                grad_norm=float(norm),lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-began,
                elapsed=time.monotonic()-begin,examples_seen=(step+1)*spec['global_batch'],
                source_exposures=dict(exposures),peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
            logs.append(entry)
            if rank==0:
                with (out/'logging.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
                save(out/'status.json',dict(stage='training',updated=time.time(),smoke=False,**entry));print(json.dumps(entry),flush=True)
            if (step+1)%cfg['save_every']==0 or step+1==cfg['steps']:checkpoint(step+1)
    if rank==0:
        dest=out/f"checkpoint-{cfg['steps']}"
        save(out/'complete.json',dict(stage=stage+'_training_complete',steps=cfg['steps'],checkpoint=str(dest),
            adapter_sha256=sha(dest/'adapter_model.safetensors'),spec_sha256=sha(args.spec),
            examples_seen=cfg['steps']*spec['global_batch'],source_exposures=dict(exposures),
            final_mopd=stage=='mopd',smoke=False,final_quality_accepted=False,completed=time.time()))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
