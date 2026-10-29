"""Training-only fit diagnosis of completed codebook candidates, not new training.

Generate all 3008 source training examples to distinguish failure to learn the
pose code from poor heldout-scene generalization. Full heldout scores stay fixed.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent
sys.path[:0]=[str(ROOT),str(C/'eval_code'),str(R/'cloud_a800')]
from checkpoints import save
from dof_codebook_interface import model_prompt,decode_response,encode_target
from probe_vllm import render


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def worker(args,manifest):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    torch.set_num_threads(4)
    assert str(torch.cuda.get_device_properties(0).uuid)==os.environ['ANVIL_EXPECTED_GPU_UUID'].removeprefix('GPU-')
    arm=manifest['models'][args.arm];model=Path(arm['model'])
    for file,digest in arm['files'].items():assert sha(model/file)==digest
    out=ROOT/'codebook_trainfit'/args.arm/str(args.shard);out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(l) for l in Path(manifest['source']).open()][args.shard::4]
    assert all(r['gsa_meta']['split']=='train' and r['gsa_form']=='dof' for r in rows)
    save(out/'status.json',dict(stage='loading',expected=len(rows),started=time.time()))
    p=AutoProcessor.from_pretrained(model,local_files_only=True)
    llm=LLM(model=str(model),dtype='bfloat16',max_model_len=32768,gpu_memory_utilization=.88,max_num_seqs=64,
        max_num_batched_tokens=8192,enable_prefix_caching=True,limit_mm_per_prompt={'image':1,'video':0},
        mm_processor_kwargs={'max_pixels':1536*32*32},seed=17,trust_remote_code=False)
    params=SamplingParams(temperature=0,max_tokens=2048,repetition_penalty=1.,seed=17)
    records=[];began=time.time()
    with (out/'predictions.jsonl').open('x') as f:
        for start in range(0,len(rows),64):
            part=rows[start:start+64];requests=[]
            for row in part:
                assert sha(row['images'][0])==row['gsa_meta']['binding']['image_sha256']
                prompt=model_prompt(row['messages'][0]['content'])
                with Image.open(row['images'][0]) as im:
                    requests.append(dict(prompt=render(p,dict(messages=[dict(content=prompt)])),multi_modal_data={'image':im.convert('RGB')}))
            outputs=llm.generate(requests,params,use_tqdm=False)
            assert len(outputs)==len(part)
            for row,output in zip(part,outputs):
                text=output.outputs[0].text;gold=json.loads(row['messages'][1]['content']);prompt=row['messages'][0]['content']
                target_code=json.loads(encode_target(gold,prompt))['rotation_bin']
                valid=False;rotation=180.;contact=2**.5;pred_code=None;error=None
                try:
                    value=json.loads(decode_response(text,prompt));pred_code=json.loads(text)['rotation_bin']
                    a=np.asarray(value['camera_from_gripper'])[:3,:3];b=np.asarray(gold['camera_from_gripper'])[:3,:3]
                    # Same numerical proper-rotation projection and pi symmetry.
                    u,_,vt=np.linalg.svd(a);a=u@vt;u,_,vt=np.linalg.svd(b);b=u@vt
                    angle=lambda m:float(np.degrees(np.arccos(np.clip((np.trace(a.T@m)-1)/2,-1,1))))
                    rotation=min(angle(b),angle(b@np.diag([-1.,-1.,1.])))
                    wh=json.loads(prompt.removeprefix('<image>'))['input']['image_size_wh']
                    contact=min(2**.5,float(np.linalg.norm((np.asarray(value['contact_point_px'])-gold['contact_point_px'])/wh)))
                    valid=output.outputs[0].finish_reason=='stop'
                    if not valid:rotation=180.;contact=2**.5
                except (ValueError,TypeError,KeyError,OverflowError) as exc:error=str(exc)
                item=dict(id=row['canonical_uid'],split='train',response=text,response_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    source_content_sha256=row['gsa_meta']['training_content_sha256'],finish_reason=output.outputs[0].finish_reason,
                    valid=valid,target_bin=target_code,predicted_bin=pred_code,rotation_bin_exact=valid and pred_code==target_code,
                    pi_symmetric_rotation_error_degrees=rotation,normalized_projected_contact_l2=contact,decoder_error=error,
                    completion_tokens=len(output.outputs[0].token_ids))
                records.append(item);f.write(json.dumps(item)+'\n')
            f.flush();save(out/'status.json',dict(stage='inference',completed=len(records),expected=len(rows),elapsed=time.time()-began))
    save(out/'complete.json',dict(state='training_only_diagnostic_complete',n=len(records),arm=args.arm,shard=args.shard,
        manifest_sha256=sha(ROOT/'dof_codebook_trainfit_manifest.json'),predictions_sha256=sha(out/'predictions.jsonl'),
        heldout_benchmark=False,teacher_qualified=False,completed=time.time()))


def node(manifest):
    from dof_codebook_node import environment,run
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines();assert len(rows)==8
    tasks=[(arm,shard) for arm in manifest['models'] for shard in range(4)]
    def execute(pair):
        device,(arm,shard)=pair;i,u,n,m=map(str.strip,device.split(','));assert 'A800' in n and int(m)>=79000
        env=environment([int(i)],'trainfit_'+i);env['ANVIL_EXPECTED_GPU_UUID']=u
        run([str(R/'.venv-vllm/bin/python'),__file__,'--arm',arm,'--shard',str(shard)],env,ROOT/'codebook_trainfit_logs'/(arm+'_'+str(shard)+'.log'))
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(execute,zip(rows,tasks)))
    result={}
    for arm in manifest['models']:
        records=[];evidence={}
        for shard in range(4):
            out=ROOT/'codebook_trainfit'/arm/str(shard);receipt=json.loads((out/'complete.json').read_text());file=out/'predictions.jsonl'
            assert receipt['manifest_sha256']==sha(ROOT/'dof_codebook_trainfit_manifest.json') and sha(file)==receipt['predictions_sha256']
            evidence[str(file)]=sha(file);records.extend(json.loads(l) for l in file.open())
        assert len(records)==len({r['id'] for r in records})==3008
        result[arm]=dict(n=3008,mean={k:sum(r[k] for r in records)/3008 for k in ['valid','rotation_bin_exact','pi_symmetric_rotation_error_degrees','normalized_projected_contact_l2']},
            predicted_bins=dict(Counter(r['predicted_bin'] for r in records)),target_bins=dict(Counter(r['target_bin'] for r in records)),evidence=evidence)
    save(ROOT/'dof_codebook_trainfit_comparison.json',dict(state='complete_training_fit_diagnostic',results=result,teacher_qualified=False,heldout_scores_unchanged=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--shard',type=int);args=ap.parse_args()
    manifest=json.loads((ROOT/'dof_codebook_trainfit_manifest.json').read_text())
    for path,digest in manifest['frozen_files'].items():assert sha(path)==digest,path
    assert sha(manifest['source'])==manifest['source_sha256']
    if args.arm:
        assert args.arm in manifest['models'] and 0<=args.shard<4;worker(args,manifest)
    else:node(manifest)


if __name__=='__main__':main()
