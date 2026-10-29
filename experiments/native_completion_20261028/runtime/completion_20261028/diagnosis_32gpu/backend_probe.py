"""Inference-only matched training/heldout diagnostic, merged vs adapter HF.
Gold answers are used only by the unchanged scorer after generation.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent
sys.path[:0]=[str(C),str(R),str(R/'cloud_a800'),str(C/'eval_code/score_code')]
import torch
import fla.ops.gated_delta_rule
from transformers import Qwen3_5ForConditionalGeneration,StoppingCriteria,StoppingCriteriaList
from peft import PeftModel
from training_data import processor,encode,EMPTY_THINK,BASE
from native_mopd_v2 import move,forward_inputs,sha
from checkpoints import save
from eval_core import repeat_period,BUDGETS
from eval.native_metrics import score_row

class Guard(StoppingCriteria):
    def __init__(self,start):self.start=start;self.cycle=None
    def __call__(self,input_ids,scores,**kwargs):
        n=input_ids.shape[1]-self.start
        if n>=512 and n%64==0:self.cycle=repeat_period(input_ids[0,self.start:].tolist())
        return torch.full((input_ids.shape[0],),self.cycle is not None,device=input_ids.device,dtype=torch.bool)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);args=ap.parse_args()
    jobs=json.loads((ROOT/'backend_jobs.json').read_text());job=jobs[args.arm]
    out=ROOT/'backend_probes'/args.arm;out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in (ROOT/('backend_inputs_'+job['family']+'.jsonl')).open()]
    torch.set_num_threads(4);torch.manual_seed(17)
    assert str(torch.cuda.get_device_properties(0).uuid)==os.environ['ANVIL_EXPECTED_GPU_UUID'].removeprefix('GPU-')
    path=Path(job['model']);binding={}
    if job['adapter']:
        assert sha(path/'adapter_model.safetensors')==job['sha256'];binding['adapter']=job['sha256']
        model=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True)
        model=PeftModel.from_pretrained(model,str(path),is_trainable=False)
    else:
        receipt=json.loads((path/'merge_receipt.json').read_text());assert receipt['state']=='verified_derived_inference_copy'
        for name,info in receipt['shards'].items():assert sha(path/name)==info['derived_sha256']
        binding=receipt
        model=Qwen3_5ForConditionalGeneration.from_pretrained(str(path),torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True)
    model.requires_grad_(False);model.eval();p=processor();start=time.monotonic();groups={}
    save(out/'protocol.json',dict(job=job,model_binding=binding,inputs_sha256=sha(ROOT/('backend_inputs_'+job['family']+'.jsonl')),code_sha256=sha(__file__),gold_in_model_input=False,training_performed=False,seed=17,temperature=0,image_tokens=1536,budgets=BUDGETS))
    with (out/'predictions.jsonl').open('w') as f:
        for i,item in enumerate(rows):
            row=item['row'];assert sha(row['images'][0])==item['image_sha256']
            # Synthetic encoder wrapper contains no gold answer and never enters any training pool.
            shell=dict(id=row['id'],messages=[row['messages'][0],dict(role='assistant',content='')],images=row['images'],gsa_meta=dict(split='train'))
            data=move(encode(p,shell),'cuda:0');length=int((data['labels'][0]!=-100).nonzero()[0])+len(p.tokenizer.encode(EMPTY_THINK,add_special_tokens=False))
            prompt=forward_inputs(data)
            for k in ['input_ids','attention_mask','mm_token_type_ids']:
                if k in prompt:prompt[k]=prompt[k][:,:length]
            guard=Guard(length);budget=BUDGETS[row['native_form']]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                ids=model.generate(**prompt,max_new_tokens=budget,do_sample=False,use_cache=True,logits_to_keep=1,eos_token_id=p.tokenizer.eos_token_id,pad_token_id=p.tokenizer.pad_token_id,stopping_criteria=StoppingCriteriaList([guard]))
            new=ids[0,length:].tolist();response=p.tokenizer.decode(new,skip_special_tokens=True)
            reason='abort' if guard.cycle else ('stop' if new[-1]==p.tokenizer.eos_token_id else 'length')
            metrics=score_row(row,response,finish_reason=reason)
            record=dict(id=row['id'],split=item['split'],response=response,finish_reason=reason,cycle=guard.cycle,metrics=metrics,tokens=len(new),prompt_token_sha256=hashlib.sha256(prompt['input_ids'].cpu().numpy().tobytes()).hexdigest())
            f.write(json.dumps(record)+'\n');f.flush();g=groups.setdefault(item['split'],dict(n=0,sums={}));g['n']+=1
            for k,v in metrics.items():g['sums'][k]=g['sums'].get(k,0)+v
            save(out/'status.json',dict(stage='inference',completed=i+1,total=len(rows),elapsed=time.monotonic()-start,updated=time.time()))
    save(out/'complete.json',dict(state='diagnostic_complete',groups={k:dict(n=v['n'],mean={a:b/v['n'] for a,b in v['sums'].items()}) for k,v in groups.items()},prediction_sha256=sha(out/'predictions.jsonl'),not_teacher_qualification=True))

if __name__=='__main__':main()
