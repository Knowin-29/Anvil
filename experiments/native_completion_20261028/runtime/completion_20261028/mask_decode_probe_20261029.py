"""Read-only paired HF decoding probe; never training or final evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

C = Path(__file__).resolve().parent
D = C / 'diagnosis_32gpu'
R = C.parent
O = C / 'mask_decode_probe_20261029'
sys.path[:0] = [str(C/'Anvil'), str(C), str(D), str(R), str(R/'cloud_a800'), str(C/'eval_code/score_code')]
from checkpoints import save, sha


class CycleStop:
    """Same512-token minimum and32-token cadence; original cycle detector."""
    def __init__(self, length):
        self.length=length
        self.cycle=None

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        from eval_core import repeat_period
        assert len(input_ids)==1
        generated=input_ids.shape[1]-self.length
        if generated>=512 and generated%32==0:
            self.cycle=repeat_period(input_ids[0,self.length:].tolist())
        return torch.tensor([self.cycle is not None],device=input_ids.device,dtype=torch.bool)


def load_spec(path, expected):
    assert sha(path) == expected
    spec = json.loads(Path(path).read_text())
    assert spec['schema'] == 'anvil-paired-mask-decoding-probe-v1'
    for filename, digest in spec['files'].items():
        assert sha(filename) == digest, filename
    return spec


def worker(path, expected, arm):
    import torch
    import fla.ops.gated_delta_rule
    from peft import PeftModel
    from transformers import Qwen3_5ForConditionalGeneration, StoppingCriteriaList
    from training_data import BASE, EMPTY_THINK, processor, encode
    from eval_core import BUDGETS
    from eval.native_metrics import score_row

    spec = load_spec(path, expected)
    task = spec['tasks'][arm]
    model_spec = spec['models'][task['model']]
    adapter = Path(model_spec['adapter'])
    assert sha(adapter/'adapter_model.safetensors') == model_spec['adapter_sha256']
    torch.set_num_threads(4)
    assert str(torch.cuda.get_device_properties(0).uuid) == os.environ['ANVIL_EXPECTED_GPU_UUID'].removeprefix('GPU-')
    out = O / 'workers' / arm
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'predictions.jsonl').exists(), 'Preserve partial diagnostic output; no silent overwrite'
    save(out/'status.json', dict(stage='loading', updated=time.time()))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16,
        device_map={'':'cuda:0'}, attn_implementation='sdpa', local_files_only=True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    p = processor()
    items = [json.loads(line) for line in Path(spec['inputs']).open()]
    items = [item for i,item in enumerate(items) if i % 2 == task['shard']]
    save(out/'protocol.json', dict(spec_sha256=expected, task=task, model=model_spec,
        image_tokens=1536, do_sample=task['mode']=='sample', temperature=.9 if task['mode']=='sample' else 0,
        top_p=1., top_k=0, budgets=BUDGETS, repetition_penalty=1.,
        seed_rule='17+sha256(example_id)[:8] modulo2**31; reseeded per example',
        gold_in_model_input=False, training_performed=False, final_validation=False))
    began = time.monotonic()
    with (out/'predictions.jsonl').open('x') as stream:
        for i,item in enumerate(items):
            row = item['row']
            assert row['native_form'] == 'mask' and sha(row['images'][0]) == item['image_sha256']
            # Gold remains only in row for the original scorer after generation.
            shell = dict(id=row['id'], messages=[row['messages'][0],dict(role='assistant',content='')],
                images=row['images'], gsa_meta=dict(split='train'))
            data = encode(p,shell)
            length = int((data['labels'][0] != -100).nonzero()[0]) + len(p.tokenizer.encode(EMPTY_THINK,add_special_tokens=False))
            prompt = {k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in data.items()
                      if k not in {'labels','token_type_ids'}}
            for key in ['input_ids','attention_mask','mm_token_type_ids']:
                if key in prompt: prompt[key] = prompt[key][:,:length]
            seed = (17 + int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)) % 2**31
            torch.manual_seed(seed)
            guard = CycleStop(length)
            sampling = dict(do_sample=True,temperature=.9,top_p=1.,top_k=0) if task['mode']=='sample' else dict(do_sample=False)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                ids = model.generate(**prompt, max_new_tokens=BUDGETS['mask'], **sampling,
                    use_cache=True, logits_to_keep=1, eos_token_id=p.tokenizer.eos_token_id,
                    pad_token_id=p.tokenizer.pad_token_id, stopping_criteria=StoppingCriteriaList([guard]))
            new = ids[0,length:].tolist()
            response = p.tokenizer.decode(new,skip_special_tokens=True)
            reason = 'abort' if guard.cycle else 'stop' if new[-1]==p.tokenizer.eos_token_id else 'length'
            record = dict(id=row['id'],split=item['split'],model=task['model'],mode=task['mode'],
                response=response,finish_reason=reason,cycle=guard.cycle,tokens=len(new),seed=seed,
                metrics=score_row(row,response,finish_reason=reason),
                prompt_sha256=hashlib.sha256(prompt['input_ids'].cpu().numpy().tobytes()).hexdigest())
            stream.write(json.dumps(record)+'\n');stream.flush()
            save(out/'status.json',dict(stage='inference',completed=i+1,total=len(items),updated=time.time(),elapsed=time.monotonic()-began))
    save(out/'complete.json',dict(state='complete_readonly_diagnostic',n=len(items),
        spec_sha256=expected,predictions_sha256=sha(out/'predictions.jsonl'),protocol_sha256=sha(out/'protocol.json')))


def aggregate(spec, expected):
    sets={};evidence={};summary={}
    for arm,task in spec['tasks'].items():
        out=O/'workers'/arm;receipt=json.loads((out/'complete.json').read_text())
        assert receipt['spec_sha256']==expected
        for name,field in [('predictions.jsonl','predictions_sha256'),('protocol.json','protocol_sha256')]:
            assert sha(out/name)==receipt[field]
            evidence[str(out/name)]=receipt[field]
        key=task['model']+'/'+task['mode'];rows=sets.setdefault(key,{})
        for line in (out/'predictions.jsonl').open():
            row=json.loads(line);assert row['id'] not in rows;rows[row['id']]=row
    first=next(iter(sets.values()))
    assert len(first)==32
    for key,rows in sets.items():
        assert rows.keys()==first.keys()
        for identity,row in rows.items():assert row['prompt_sha256']==first[identity]['prompt_sha256']
        summary[key]={}
        for split in ['train','heldout']:
            group=[r for r in rows.values() if r['split']==split];assert len(group)==16
            summary[key][split]=dict(n=16,cycle_stops=sum(bool(r['cycle']) for r in group),
                mean_tokens=sum(r['tokens'] for r in group)/16,
                metrics={k:sum(r['metrics'][k] for r in group)/16 for k in group[0]['metrics']})
    save(O/'report.json',dict(state='complete_requires_review',spec_sha256=expected,summary=summary,
        evidence=evidence,training_performed=False,final_quality_accepted=False,
        limitations=spec['limitations'],completed=time.time()))


def node(path, expected):
    spec=load_spec(path,expected)
    from retention_node import environment,run,TRAIN
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(lines)==len(spec['tasks'])==8
    devices=[]
    for line in lines:
        i,u,n,m=map(str.strip,line.split(','));assert 'A800' in n and int(m)>=79000
        devices.append((int(i),u))
    assert len({u for i,u in devices})==8
    save(O/'status.json',dict(stage='running',devices=devices,started=time.time()))
    def execute(pair):
        (gpu,uid),arm=pair;env=environment([gpu],'mask_decode_'+str(gpu));env['ANVIL_EXPECTED_GPU_UUID']=uid
        out=O/'workers'/arm;out.mkdir(parents=True,exist_ok=True)
        run([TRAIN,'-u',str(Path(__file__)), '--spec',str(path),'--spec-sha256',expected,'--worker',arm],env,out/'console.log')
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(execute,zip(sorted(devices),spec['tasks'])))
    aggregate(spec,expected)
    save(O/'status.json',dict(stage='complete_requires_review',completed=time.time(),report_sha256=sha(O/'report.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--spec-sha256',required=True)
    ap.add_argument('--worker');ap.add_argument('--node',action='store_true');args=ap.parse_args()
    assert bool(args.worker)!=args.node
    try:
        worker(args.spec,args.spec_sha256,args.worker) if args.worker else node(args.spec,args.spec_sha256)
    except BaseException as exc:
        out=O/'workers'/args.worker if args.worker else O;out.mkdir(parents=True,exist_ok=True)
        save(out/'failure.json',dict(error=repr(exc),time=time.time()))
        raise
