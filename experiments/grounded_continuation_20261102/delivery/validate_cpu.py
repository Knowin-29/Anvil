"""CPU checks of standalone inference parity and fail-closed delivery bindings."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from common import C,D,HERE,TRAIN,formal_training,require,save,sha
from node import task_list
from package import copy_inference_interface
from upload import verify_local,safe_relative


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for nodes in range(1,5):
        tasks=task_list(dict(public_shards=8,native_shards=32))
        partitions=[tasks[i::nodes] for i in range(nodes)]
        flattened=[(x['kind'],x['shard']) for group in partitions for x in group]
        require(len(flattened)==40 and len(set(flattened))==40,'Missing/duplicated static shards')
    try:
        formal_training(D/'mopd_v5_runtime',full_checkpoint=False)
    except ValueError as exc:
        require('formal MOPD' in str(exc),'Unexpected smoke rejection')
    else:raise AssertionError('Smoke accepted as final')
    for name in ['../secret','/absolute','x/../../escape']:
        try:safe_relative(name)
        except ValueError:pass
        else:raise AssertionError('Unsafe upload path accepted')
    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        p=Path(temporary);save(p/'release_manifest.json',dict(schema='anvil-sealed-final-release-v1',quality_review_accepted=False))
        try:verify_local(p)
        except ValueError:pass
        else:raise AssertionError('Unreviewed package accepted')
    code='''
import importlib,json,sys
from pathlib import Path
pack,diag,mode,source=map(str,sys.argv[1:])
sys.path.insert(0,pack)
from anvil_interface import prepare_native_prompt,decode_native_response
rows=[json.loads(x) for x in Path(source).open()]
expected=[]
for row in rows:
    prompt=row['messages'][0]['content']
    assert prepare_native_prompt('bbox',prompt)==prompt
    assert decode_native_response('bbox','unchanged',prompt)=='unchanged'
if mode=='native':
    for row in rows:
        prompt=row['messages'][0]['content'];answer=row['messages'][1]['content']
        assert prepare_native_prompt('dof',prompt)==prompt
        assert decode_native_response('dof',answer,prompt)==answer
else:
    # Cache packaged functions before importing the ORIGINAL modules separately.
    for name in ['dof_codebook_interface','dof_codec_interface','dof_codec_prototype']:
        sys.modules.pop(name,None)
    sys.path.insert(0,diag)
    original=importlib.import_module('dof_codebook_interface' if mode=='rotation128' else 'dof_codec_interface')
    for row in rows:
        prompt=row['messages'][0]['content']
        latent=original.encode_target(json.loads(row['messages'][1]['content']),prompt)
        assert prepare_native_prompt('dof',prompt)==original.model_prompt(prompt)
        assert decode_native_response('dof',latent,prompt)==original.decode_response(latent,prompt)
    bad=['{}','null','{"rotation_bin":128,"position":[0,0,0,0,0,0]}','{"latent":[0,0,0,0,0,0,0,0,0,0,0,0]}']
    for value in bad:
        states=[]
        for fun in [lambda:decode_native_response('dof',value,prompt),lambda:original.decode_response(value,prompt)]:
            try:states.append(('accepted',fun()))
            except Exception as exc:states.append(('rejected',type(exc).__name__))
        assert states[0]==states[1]
assert len(rows)==3008
print(json.dumps(dict(rows=len(rows),mode=mode,valid_parity=True,malformed_parity=True)))
'''
    results={}
    source=C.parent.parent/'arxiv/native_training_export_20261012_v4_r2/train_dof.jsonl'
    for mode in ['native','rotation6d','rotation128']:
        out=args.output/mode;copy_inference_interface(out,mode)
        completed=subprocess.run([TRAIN,'-c',code,str(out),str(D),mode,str(source)],text=True,capture_output=True,check=True)
        result=json.loads(completed.stdout.strip().splitlines()[-1])
        result.update(state='passed',package_builder_sha256=sha(HERE/'package.py'),
            standalone_files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
            source_sha256=sha(source),gpu_inference_validated=False)
        save(args.output/(mode+'_interface_validation.json'),result);results[mode]=result
    save(args.output/'cpu_validation.json',dict(state='passed_cpu_only',static_partitions_1_to_4_nodes=True,
        smoke_rejected_as_final=True,unreviewed_package_rejected=True,path_escape_rejected=True,
        interface_parity=results,final_gpu_evaluation_not_yet_run=True,hf_upload_not_attempted=True))
    print(args.output/'cpu_validation.json')


if __name__=='__main__':main()
