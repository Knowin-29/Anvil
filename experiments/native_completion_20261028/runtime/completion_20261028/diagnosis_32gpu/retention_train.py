"""Matched narrow/broad QA replay from current joint; original spatial pool retained.
No heldout training;512 fixed updates; separate source provenance and exposure logs.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parent
C=ROOT.parent;R=C.parent
sys.path[:0]=[str(C),str(R),str(R/'cloud_a800')]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as recompute
import fla.ops.gated_delta_rule
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration
from training_data import BASE,EMPTY_THINK,encode,processor
from memory_loss import chunked_ce
from native_mopd_v2 import install_hidden_forward,move,forward_inputs,sha
from checkpoints import save,seal,verify
from eval_core import repeat_period
from audit_targets import compact

ARMS={'joint_narrow':('joint',False,False),'joint_broad':('joint',False,False)}

class Pool:
    def __init__(self,arm):
        from native_mopd_v2 import FullPool
        self.spatial=FullPool();self.index=json.loads((R/'training_repair/full_training_index/manifest.json').read_text())
        self.broad=arm=='joint_broad';self.schedule=['QA','G','QA','S','QA','A']
        self.replay={}
        replay=json.loads((ROOT/'retention_replay/manifest.json').read_text());assert replay['state']=='verified_train_only'
        self.replay_manifest_sha256=sha(ROOT/'retention_replay/manifest.json')
        if self.broad:
            for dataset in ['ai2d','scienceqa','textvqa']:
                p=ROOT/'retention_replay'/(dataset+'.jsonl');assert sha(p)==replay['files_sha256'][dataset]
                self.replay[dataset]=[json.loads(l) for l in p.open()]
                assert len(self.replay[dataset])==replay['additional_training_records'][dataset]
        else:
            info=self.index['files']['train_QA.jsonl'];self.offsets=np.load(info['index_file'],mmap_mode='r')
            assert sha(info['index_file'])==info['index_sha256']
            file=Path(self.index['export'])/'train_QA.jsonl';assert sha(file)==info['source_sha256'];self.handle=file.open('rb')
    def sample(self,micro,rank,world):
        if micro%2:return self.spatial.sample(micro//2,rank,world)
        ordinal=micro//2
        if self.broad:
            dataset=['ai2d','scienceqa','textvqa'][ordinal%3];pool=self.replay[dataset]
            count=len(pool);ordinal=(ordinal//3)*world+rank
        else:count=len(self.offsets);ordinal=ordinal*world+rank
        multiplier=1000003
        while math.gcd(multiplier,count)!=1:multiplier+=2
        selected=(ordinal*multiplier+17)%count
        if self.broad:row=pool[selected]
        else:
            offset,size=self.offsets[selected];self.handle.seek(int(offset));row=json.loads(self.handle.read(int(size)))
        assert row['gsa_meta']['split']=='train' and row['gsa_form']=='QA'
        return row

def cycle_endpoint(tokens):
    # Same conservative cycle definition as unchanged evaluator; never reference-fed.
    for end in range(512,len(tokens)+1,32):
        cycle=repeat_period(tokens[:end])
        if cycle:return end,cycle
    return None,None

def token_unlikelihood(hidden,head,ids,positions):
    if not positions:return hidden.sum()*0
    states=hidden[0,torch.tensor(positions,device=hidden.device)-1]
    targets=ids[0,positions]
    def score(x,y):
        logits=head(x).float();chosen=logits.gather(1,y[:,None]).squeeze(1)
        # log(1 + exp(chosen - logsumexp(other logits))) = -log(1-p(chosen)).
        others=logits.scatter(1,y[:,None],float('-inf')).logsumexp(-1)
        return torch.nn.functional.softplus(chosen-others).sum()
    total=states.sum()*0
    for start in range(0,len(positions),32):
        total=total+recompute(score,states[start:start+32],targets[start:start+32],use_reentrant=False)
    return total/len(positions)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=512)
    ap.add_argument('--stop-after',type=int);ap.add_argument('--resume',type=Path);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();family,rounded,use_ul=ARMS[args.arm]
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    assert world==4 and args.steps==512
    if args.smoke:assert args.stop_after in (1,2)
    else:assert args.stop_after is None
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl')
    torch.set_num_threads(4);torch.manual_seed(17+rank);random.seed(17+rank);np.random.seed(17+rank)
    args.output.mkdir(parents=True,exist_ok=True);pool=Pool(args.arm)
    initial=json.loads((R.parent/'a800_repair_20261020/pipeline_status.json').read_text())['checkpoints'][family]
    adapter=Path(initial['checkpoint']);assert sha(adapter/'adapter_model.safetensors')==initial['adapter_sha256']
    protocol=dict(schema='anvil-retention-repair-controlled-v1',arm=args.arm,initial=initial,
        source_export_manifest_sha256=pool.index['export_manifest_sha256'],source_index_sha256=sha(R/'training_repair/full_training_index/manifest.json'),
        source_pool=5220000,full_original_sft_completed=True,world=world,global_batch=32,steps=512,seed=17,
        form_schedule=pool.schedule,qa_replay_fraction=.5,lr=2e-5,warmup_fraction=.03,
        schedule='cosine',trainable_parameters=173113344,context=16384,image_tokens=1536,
        intervention='official broad train-only QA replay' if pool.broad else 'original CLEVR QA replay control',
        additional_replay_manifest_sha256=pool.replay_manifest_sha256,
        ul=dict(weight=.1,every_updates=4,rollout_cap=768,greedy=True,minimum_exact_span=256) if use_ul else None,
        source_data_modified=False,heldout_training=False,final_mopd=False,smoke=args.smoke,
        code_sha256={n:sha(ROOT/n) for n in ['retention_train.py']})
    if rank==0:
        pp=args.output/'protocol.json'
        if pp.exists():assert json.loads(pp.read_text())==protocol
        else:save(pp,protocol)
        save(args.output/'status.json',dict(stage='loading',arm=args.arm,smoke=args.smoke,started=time.time()))
    dist.barrier();install_hidden_forward()
    model=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,
            device_map={'':device},attn_implementation='sdpa',local_files_only=True)
    model=PeftModel.from_pretrained(model,str(args.resume or adapter),is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    parameters=[p for p in model.parameters() if p.requires_grad];assert sum(p.numel() for p in parameters)==173113344
    optimizer=torch.optim.AdamW(parameters,lr=2e-5,weight_decay=.01,fused=True)
    start=0;logs=[]
    if args.resume:
        verify(args.resume,world)
        optimizer.load_state_dict(torch.load(args.resume/'optimizer.pt',map_location=device,weights_only=True))
        state=json.loads((args.resume/'trainer_state.json').read_text());start=state['global_step'];logs=state['log_history']
        rng=torch.load(args.resume/f'rng_state_{rank}.pth',map_location='cpu',weights_only=False)
        torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],device);random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    p=processor();head=model.get_output_embeddings();ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    accum=32//world;began=time.monotonic()
    def save_checkpoint(step):
        dest=args.output/f'checkpoint-{step}';dest.mkdir(exist_ok=True)
        if rank==0:
            model.save_pretrained(dest,safe_serialization=True);torch.save(optimizer.state_dict(),dest/'optimizer.pt')
            torch.save(dict(step=step,schedule='cosine',steps=512),dest/'scheduler.pt')
            save(dest/'trainer_state.json',dict(global_step=step,max_steps=512,examples_seen=step*32,log_history=logs,smoke=args.smoke,final_mopd=False))
        torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),python=random.getstate(),numpy=np.random.get_state()),dest/f'rng_state_{rank}.pth')
        dist.barrier()
        if rank==0:seal(dest,world);save(args.output/'latest.json',dict(path=str(dest),step=step))
        dist.barrier()
    end=args.stop_after or args.steps
    with (args.output/f'exposure_rank{rank}.jsonl').open('a') as exposure:
        for step in range(start,end):
            t0=time.monotonic();optimizer.zero_grad(set_to_none=True);stats=torch.zeros(4,device=device)
            warmup=math.ceil(args.steps*.03)
            ratio=min(1.,(step+1)/warmup) if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/(args.steps-warmup)))
            for group in optimizer.param_groups:group['lr']=2e-5*ratio
            for micro in range(accum):
                ordinal=step*accum+micro;row=deepcopy(pool.sample(ordinal,rank,world));original=row['messages'][1]['content']
                if rounded and row['gsa_form']=='dof':
                    row['messages'][1]['content']=json.dumps(compact(json.loads(original)),sort_keys=True,separators=(',',':'),ensure_ascii=False)
                data=move(encode(p,row),device);model.train();sampled=None;positions=[];cycle=None
                # Same branch and forward/backward count across ranks, including no-cycle zero loss.
                do_ul=use_ul and step%4==0 and micro==0 and row['gsa_form']=='mask'
                if do_ul:
                    target_start=int((data['labels'][0]!=-100).nonzero()[0]);prompt_length=target_start+len(p.tokenizer.encode(EMPTY_THINK,add_special_tokens=False))
                    prompt=forward_inputs(data)
                    for key in ['input_ids','attention_mask','mm_token_type_ids']:
                        if key in prompt:prompt[key]=prompt[key][:,:prompt_length]
                    model.eval()
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        ids=model.generate(**prompt,max_new_tokens=768,do_sample=False,use_cache=True,logits_to_keep=1,
                            eos_token_id=p.tokenizer.eos_token_id,pad_token_id=p.tokenizer.pad_token_id)
                    endpoint,cycle=cycle_endpoint(ids[0,prompt_length:].tolist())
                    if cycle:
                        ids=ids[:,:prompt_length+endpoint]
                        positions=list(range(ids.shape[1]-min(64,cycle['span_tokens']),ids.shape[1]))
                    sampled=dict(prompt,input_ids=ids,attention_mask=torch.ones_like(ids))
                    if 'mm_token_type_ids' in prompt:sampled['mm_token_type_ids']=torch.cat((prompt['mm_token_type_ids'],torch.zeros_like(ids[:,prompt_length:])),dim=1)
                    model.train()
                sync=ddp.no_sync() if micro<accum-1 else nullcontext()
                with sync,torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=ddp(**forward_inputs(data),anvil_hidden_only=True,use_cache=False)
                    ce=chunked_ce(hidden,head,data['labels']);assert torch.isfinite(ce);(ce/accum).backward();del hidden
                    stats[0]+=ce.detach()/accum
                    if do_ul:
                        hidden=ddp(**sampled,anvil_hidden_only=True,use_cache=False)
                        ul=token_unlikelihood(hidden,head,sampled['input_ids'],positions);assert torch.isfinite(ul)
                        # One rollout penalty per eligible update, separate from the16-example mean CE.
                        (.1*ul).backward();stats[1]+=ul.detach();stats[2]+=bool(cycle);stats[3]+=1;del hidden,ul
                exposure.write(json.dumps(dict(step=step+1,micro=micro,rank=rank,canonical_uid=row['canonical_uid'],form=row['gsa_form'],
                    original_target_sha256=__import__('hashlib').sha256(original.encode()).hexdigest(),
                    derived_target_sha256=__import__('hashlib').sha256(row['messages'][1]['content'].encode()).hexdigest(),
                    source_binding=row['gsa_meta']['binding'],tokens=data['input_ids'].shape[1],cycle=cycle,smoke=args.smoke))+'\n');exposure.flush()
                del ce,data,sampled
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm),'Non-finite gradient'
            optimizer.step();dist.all_reduce(stats);stats/=world
            entry=dict(step=step+1,max_steps=512,ce=float(stats[0]),ul=float(stats[1]),cycle_fraction=float(stats[2]),rollout_fraction=float(stats[3]),
                grad_norm=float(norm),lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-t0,elapsed=time.monotonic()-began,
                examples_seen=(step+1)*32,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
            logs.append(entry)
            if rank==0:
                with (args.output/'logging.jsonl').open('a') as log:log.write(json.dumps(entry)+'\n')
                save(args.output/'status.json',dict(stage='training',arm=args.arm,smoke=args.smoke,updated=time.time(),**entry));print(json.dumps(entry),flush=True)
            if (step+1)%128==0 or step+1==end:save_checkpoint(step+1)
    if rank==0:
        cp=args.output/f'checkpoint-{end}'
        save(args.output/'complete.json',dict(state='smoke_complete' if args.smoke else 'repair_complete_requires_evaluation',arm=args.arm,
            checkpoint=str(cp),adapter_sha256=sha(cp/'adapter_model.safetensors'),step=end,smoke=args.smoke,final_mopd=False,teacher_qualified=False,completed=time.time()))
    dist.destroy_process_group()

if __name__=='__main__':main()
