"""Matched full public benchmark and general-VQA comparison, exact frozen inputs."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT.parent.parent/'cloud_a800'))
from checkpoints import save
from gsa_eval.datasets import Sample

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--role',required=True)
    ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,default=4)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    run=ROOT/('student_adaptation_benchmark_smoke' if args.smoke else 'student_adaptation_benchmarks')/args.role/str(args.shard)
    run.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    for name,digest in manifest['code'].items():assert sha(ROOT/'code'/name)==digest
    import torch
    import fla.ops.gated_delta_rule
    from gsa_eval.runner import get_engine,_infer
    torch.set_num_threads(4)
    expected=os.environ.get('ANVIL_EXPECTED_GPU_UUID')
    assert expected and str(torch.cuda.get_device_properties(0).uuid)==expected.removeprefix('GPU-')
    save(run/'status.json',dict(stage='loading',started=time.time()))
    engine=get_engine(args.model,8,'cuda:0',False)
    index=json.loads((Path(args.model)/'model.safetensors.index.json').read_text())
    model_binding={n:sha(Path(args.model)/n) for n in sorted(set(index['weight_map'].values()))}
    protocol=dict(model=args.model,weights=model_binding,inputs_sha256=sha(ROOT/'inputs/manifest.json'),
        code=manifest['code'],driver_sha256=sha(__file__),role=args.role,shard=args.shard,shards=args.shards,
        image_max_tokens=1536,thinking=False,temperature=0,batch=8,smoke=args.smoke)
    pp=run/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:save(pp,protocol)
    done=0;began=time.monotonic()
    for name in [*manifest['public'],'retention']:
        path=ROOT/'inputs'/(name+'.jsonl')
        info=manifest['retention'] if name=='retention' else manifest['public'][name]
        assert sha(path)==info['sha256']
        rows=[json.loads(line) for line in path.open()][args.shard::args.shards]
        if args.smoke:rows=rows[:2]
        output=run/(name+'.jsonl');saved={}
        if output.exists():
            for line in output.open():
                item=json.loads(line);assert item['uid'] not in saved;saved[item['uid']]=item
        pending=[r for r in rows if str(r.get('global_uid',r.get('uid'))) not in saved]
        # VQA budgets differ by dataset. Preserve stable order within each budget.
        groups=defaultdict(list)
        for row in pending:groups[row['max_tokens']].append(row)
        with output.open('a') as f:
            for budget,group in groups.items():
                for start in range(0,len(group),8):
                    part=group[start:start+8];samples=[]
                    for r in part:
                        paths=[r['image_path']] if name=='retention' else r['image_paths']
                        for path in paths:assert sha(path)==manifest['images'][path]
                        samples.append(Sample(uid=str(r.get('global_uid',r.get('uid'))),prompt=r['prompt'],image_paths=paths,gt=None))
                    texts,caps=_infer(engine,samples,max_tokens=budget)
                    assert len(texts)==len(part)==len(caps)
                    for sample,text,cap in zip(samples,texts,caps):
                        item=dict(uid=sample.uid,response=text,hit_token_cap=bool(cap),response_sha256=hashlib.sha256(text.encode()).hexdigest())
                        f.write(json.dumps(item,ensure_ascii=False)+'\n');saved[item['uid']]=item;done+=1
                    f.flush()
                    save(run/'status.json',dict(stage='inference',benchmark=name,completed_this_session=done,seconds=time.monotonic()-began))
        assert set(saved)=={str(r.get('global_uid',r.get('uid'))) for r in rows}
        print(name,len(saved),'complete',flush=True)
    save(run/'complete.json',dict(state='smoke_complete' if args.smoke else 'complete',role=args.role,shard=args.shard,protocol_sha256=sha(pp),completed=time.time()))

if __name__=='__main__':main()
