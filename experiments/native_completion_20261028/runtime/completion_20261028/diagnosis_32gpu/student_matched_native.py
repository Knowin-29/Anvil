"""Predeclared G/S validation sample of the exact adapted student.

Eight randomly selected ordinal strata out of128, using the existing unmodified
native evaluator. These sampled scores are never described as full validation.
The student's complete4100A and public evaluations run separately.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import yaml

D=Path(__file__).resolve().parent;C=D.parent;R=C.parent
sys.path[:0]=[str(C/'eval_code'),str(R/'cloud_a800')]
from checkpoints import save
from eval_core import select_files
from retention_node import environment,run,EVAL


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def plan():
    export=R.parent/'arxiv/native_evaluation_export_20261012_v4'
    manifest=json.loads((export/'manifest.json').read_text())
    selected=sorted(random.Random(170629).sample(range(128),8))
    tasks=[];forms=['bbox','detection','mask','point','direction','metric_comparison','relation_graph','depth_order']
    for gpu,shard in enumerate(selected):
        files=[(name,info) for name,info in select_files(manifest,'joint',shard,128) if info['form'] in forms]
        tasks.append(dict(local_gpu=gpu,shard=shard,shards=128,
                          records=sum(info['shard_records'] for _,info in files),files=dict(files)))
    return dict(schema='anvil-predeclared-student-GS-sample-v1',seed=170629,tasks=tasks,forms=forms,
        selection='Eight ordinal strata sampled without replacement from128; fixed before adapted-student results',
        expected_records=sum(t['records'] for t in tasks),full_validation=False,
        export_manifest=str(export/'manifest.json'),export_manifest_sha256=sha(export/'manifest.json'),
        scorer_changed=False,reference_targets_changed=False)


def prepare():
    selection=json.loads((D/'student_matched_native_selection.json').read_text())
    assert selection==plan()
    complete=json.loads((D/'student_adaptation_run/main/complete.json').read_text())
    assert complete['step']==1024 and not complete['smoke']
    model=D/'merged_student_adaptation';merge=json.loads((model/'merge_receipt.json').read_text())
    assert merge['state']=='verified_derived_inference_copy' and merge['original_adapter']==complete['checkpoint']
    assert merge['adapter_sha256']==complete['adapter_sha256']==sha(Path(complete['checkpoint'])/'adapter_model.safetensors')
    for name,info in merge['shards'].items():assert sha(model/name)==info['derived_sha256']
    frozen=[Path(__file__),D/'student_matched_native_selection.json',D/'retention_node.py',
            D/'student_adaptation_run/main/complete.json',D/'student_adaptation_run/main/protocol.json',
            R/'cloud_a800/checkpoints.py']+list((C/'eval_code').rglob('*.py'))
    spec=dict(schema='anvil-exact-student-GS-sample-v1',model=str(model),checkpoint=complete['checkpoint'],
        adapter_sha256=complete['adapter_sha256'],selection=selection,full_validation=False,
        teacher_qualified=False,formal_mopd=False,
        model_files={p.name:sha(p) for p in model.iterdir() if p.is_file()},
        frozen_files={str(p):sha(p) for p in sorted(set(frozen))})
    destination=D/'student_matched_native_spec.json';assert not destination.exists();save(destination,spec)
    config=yaml.safe_load((D/'g_priority_native_node0_cli.yaml').read_text())
    config['TaskName']='task_anvil_adapted_student_matched_GS_A800x8'
    config['Description']='Predeclared8of128nativeG/Sstrata;exactadaptedstudent;unchangedscorer;notfullvalidationorqualification'
    config['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+str(R/'.venv-train/bin/python')+' -u '+str(Path(__file__))+' --run'
    (D/'student_matched_native_cli.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    print(destination)


def aggregate(spec):
    groups=defaultdict(lambda:dict(n=0,sums=defaultdict(float)));seen=set();evidence={}
    for task in spec['selection']['tasks']:
        out=D/'student_matched_native'/str(task['shard']);receipt=json.loads((out/'complete.json').read_text())
        metrics_path=out/'evaluation/metrics.json';assert sha(metrics_path)==receipt['metrics_sha256']
        metrics=json.loads(metrics_path.read_text());assert metrics['n']==task['records']
        for name,digest in metrics['prediction_sha256'].items():
            path=out/'evaluation'/name;assert sha(path)==digest;evidence[str(path)]=digest
            for row in map(json.loads,path.open()):
                key=(row['native_form'],row['id']);assert key not in seen;seen.add(key)
        for group in metrics['groups']:
            value=groups[(group['form'],group['source'])];value['n']+=group['n']
            for key,score in group['mean'].items():value['sums'][key]+=score*group['n']
    assert len(seen)==spec['selection']['expected_records']
    save(D/'student_matched_native_comparison.json',dict(state='complete_sample_requires_matched_teacher_review',
        n=len(seen),full_validation=False,teacher_qualified=False,formal_mopd=False,
        spec_sha256=sha(D/'student_matched_native_spec.json'),
        groups=[dict(form=f,source=s,n=g['n'],mean={k:v/g['n'] for k,v in g['sums'].items()}) for (f,s),g in groups.items()],
        evidence=evidence))


def node():
    path=D/'student_matched_native_spec.json';spec=json.loads(path.read_text())
    for file,digest in spec['frozen_files'].items():assert sha(file)==digest,file
    for name,digest in spec['model_files'].items():assert sha(Path(spec['model'])/name)==digest
    devices=[]
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines():
        index,uid,name,memory=map(str.strip,line.split(','));assert 'A800' in name and int(memory)>=79000;devices.append((int(index),uid))
    devices.sort();assert len(devices)==8
    status=D/'student_matched_native_node_status.json';save(status,dict(stage='running',started=time.time(),devices=devices))
    def evaluate(task):
        gpu,uid=devices[task['local_gpu']];out=D/'student_matched_native'/str(task['shard']);out.mkdir(parents=True,exist_ok=False)
        env=environment([gpu],'student_matched_'+str(task['shard']));env['ANVIL_EXPECTED_GPU_UUID']=uid
        (Path(env['XDG_CACHE_HOME'])/'torch/kernels').mkdir(parents=True,exist_ok=True)
        argv=[EVAL,str(C/'eval_code/run_vllm_eval.py'),'--model',spec['model'],'--name','student_adaptation_GS_sample',
            '--scope','joint','--gpu-uuid',uid,'--shard',str(task['shard']),'--shards','128',
            '--concurrency','64','--memory','.88','--output',str(out/'evaluation'),'--forms',*spec['selection']['forms']]
        save(out/'launch.json',dict(argv=argv,task=task,gpu=gpu,uuid=uid,spec_sha256=sha(path)))
        try:
            run(argv,env,out/'console.log');metrics=json.loads((out/'evaluation/metrics.json').read_text())
            assert metrics['state']=='complete_scope_shard' and metrics['n']==task['records']
            save(out/'complete.json',dict(task=task,metrics_sha256=sha(out/'evaluation/metrics.json'),spec_sha256=sha(path)))
        except BaseException as exc:
            save(out/'failure.json',dict(error=repr(exc),time=time.time()));raise
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(evaluate,spec['selection']['tasks']))
        aggregate(spec);save(status,dict(stage='sample_complete_requires_matched_teacher_review',completed=time.time()))
    except BaseException as exc:
        save(status,dict(stage='failed',error=repr(exc),time=time.time()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    assert args.prepare != args.run
    prepare() if args.prepare else node()
