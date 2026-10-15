"""Train native G/S prerequisites on disjoint source scenes, then collect results.

Four fixed-update SFT arms are a prerequisite to routed learning, not an automatic
teacher-qualification certificate or a token-matched MOPD comparison.
"""
from __future__ import annotations

import argparse
import collections
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

from native_completion import A, W, D, MODEL, sha, save, rows, read_binding, commit_bytes
from native_completion import prediction_spool, spool_batch_ready, commit_prediction

ROOT = D/'linked_sensor_training_v1'
SOURCE = D/'linked_sensor_gs_v1'
ARMS = ['G', 'S', 'joint', 'linked']
SETTINGS = {'steps':512, 'seed':17, 'lr':1e-4, 'batch':1, 'accumulation':4,
            'rank':64, 'alpha':128, 'dropout':0.05, 'image_tokens':1536,
            'max_length':3072, 'max_new_tokens':512}


def write_rows(path, values):
    commit_bytes(path, ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values).encode())


def prepare():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'prepare.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source=read_binding(SOURCE/'manifest.json')
        for name,b in source['files'].items():
            assert sha(SOURCE/name)==b['sha256']
        binding={'source_manifest_sha256':sha(SOURCE/'manifest.json'),'settings':SETTINGS,
            'split_rule':'SHA256(linked-qualification-v1|physical_scene_group), first 64 bits modulo 10 == 0 reserves qualification from previously unused train scenes; original dev is unchanged.',
            'model':str(MODEL),'model_config_sha256':sha(MODEL/'config.json'),
            'training':'Fresh base per arm; G/S use the corresponding atomic view; joint mixes both atomic views; linked uses both stage targets in one response. Equal optimizer updates and row exposures, measured tokens are not assumed equal.',
            'selection':'Terminal checkpoint-512; no qualification score selects a checkpoint.',
            'teacher_eligible':False,'full_MTD_complete':False}
        if (ROOT/'manifest.json').exists():
            old=read_binding(ROOT/'manifest.json');assert old['binding']==binding
            for name,b in old['files'].items():assert sha(ROOT/name)==b['sha256']
            return old
        import hashlib
        train=rows(SOURCE/'train.jsonl');dev=rows(SOURCE/'dev.jsonl')
        def qualify(r):
            text='linked-qualification-v1|'+r['gsa_meta']['scene_group']
            return int(hashlib.sha256(text.encode()).hexdigest()[:16],16)%10==0
        splits={'train':[r for r in train if not qualify(r)],
                'qualification':[r for r in train if qualify(r)],'dev':dev}
        groups={k:{r['gsa_meta']['scene_group'] for r in v} for k,v in splits.items()}
        for a in groups:
            assert groups[a]
            for b in groups:
                if a!=b:assert not groups[a]&groups[b]
        ids={k:{r['sample_id'] for r in v} for k,v in splits.items()}
        atomic={stage:{r['sample_id'][:-2]:r for split in ['train','dev']
                      for r in rows(SOURCE/f'{split}_{stage}.jsonl')} for stage in ['G','S']}
        files={}
        for split,linked in splits.items():
            values={s:[atomic[s][r['sample_id']] for r in linked] for s in ['G','S']}
            values['joint']=values['G']+values['S'];values['linked']=linked
            for arm,records in values.items():
                native=ROOT/f'{split}_{arm}.jsonl'
                records=deepcopy(records)
                for r in records:r['gsa_meta']['split']=split
                write_rows(native,records)
                files[native.name]={'sha256':sha(native),'records':len(records)}
                if split in ['train','dev']:
                    # Only prompt/image/answer reach Swift; references stay outside
                    # the model request and are retained in the native files above.
                    messages=[{k:r[k] for k in ['messages','images']} for r in records]
                    if split=='train':random.Random(SETTINGS['seed']).shuffle(messages)
                    path=ROOT/f'{split}_{arm}_messages.jsonl';write_rows(path,messages)
                    files[path.name]={'sha256':sha(path),'records':len(messages)}
        all_hashes={k:{r['gsa_meta']['decoded_rgb_sha256'] for r in v} for k,v in splits.items()}
        for a in all_hashes:
            for b in all_hashes:
                if a!=b:assert not all_hashes[a]&all_hashes[b]
        result={'binding':binding,'files':files,
            'split_counts':{k:{'images':len(v),'physical_scene_groups':len(groups[k])} for k,v in splits.items()},
            'qualification_ids':sorted(ids['qualification']),
            'scope':'Native two-object localization and relative depth; no Action supervision or full teacher eligibility.'}
        save(ROOT/'manifest.json',result);return result


def train(arm, gpu):
    manifest=prepare();folder=ROOT/arm;folder.mkdir(exist_ok=True)
    if (folder/'training_done.json').exists():
        done=read_binding(folder/'training_done.json')
        assert sha(Path(done['adapter'])/'adapter_model.safetensors')==done['adapter_sha256']
        return
    launch=W/'scripts/run_sft_9b_lora_l20_dlc.sh'
    env={**os.environ,'GSA_ROOT':str(W),'MODEL':str(MODEL),'CUDA_VISIBLE_DEVICES':str(gpu),
        'MASTER_PORT':str(29850+gpu),'TRAIN':str(ROOT/f'train_{arm}_messages.jsonl'),
        'VAL':str(ROOT/f'dev_{arm}_messages.jsonl'),'OUT':str(folder/'training'),
        'BS':'1','GA':'4','RANK':'64','ALPHA':'128','DROPOUT':'0.05','SEED':'17',
        'LR':'0.0001','STEPS':'512','EVAL_STEPS':'128','SAVE_STEPS':'128','HOURS':'0',
        'REPORT_TO':'none','COUNT_INPUT_TOKENS':'1','DATALOADER_NUM_WORKERS':'2',
        'DATASET_NUM_PROC':'2','IMAGE_MAX_TOKEN_NUM':'1536','PYTHONUNBUFFERED':'1'}
    env.pop('INITIAL_ADAPTER',None)
    binding={'manifest_sha256':sha(ROOT/'manifest.json'),'arm':arm,
        'settings':SETTINGS,'train_sha256':sha(Path(env['TRAIN'])),'dev_sha256':sha(Path(env['VAL'])),
        'launcher_sha256':sha(launch),'gpu':gpu,'no_wall_clock_kill':True}
    if (folder/'binding.json').exists():assert read_binding(folder/'binding.json')==binding
    else:save(folder/'binding.json',binding)
    started=time.monotonic()
    with (folder/'training.log').open('a') as log:
        subprocess.run(['bash',str(launch)],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    checkpoints=list((folder/'training').glob('v*/checkpoint-512'))
    assert len(checkpoints)==1,'Expected one completed fixed-budget terminal checkpoint'
    selected=checkpoints[0];state=read_binding(selected/'trainer_state.json')
    actual=read_binding(selected.parent/'args.json')
    expected={'model':str(MODEL),'dataset':[env['TRAIN']],'val_dataset':[env['VAL']],
              'max_steps':512,'seed':17,'lora_rank':64,'lora_alpha':128,'learning_rate':1e-4,
              'per_device_train_batch_size':1,'gradient_accumulation_steps':4}
    assert all(actual[k]==v for k,v in expected.items()),'Resolved training settings drift'
    assert state['global_step']==512 and state['max_steps']==512
    save(folder/'training_done.json',{'adapter':str(selected),'adapter_sha256':sha(selected/'adapter_model.safetensors'),
        'trainer_state_sha256':sha(selected/'trainer_state.json'),'args_sha256':sha(selected.parent/'args.json'),
        'global_step':512,'num_input_tokens_seen':state.get('num_input_tokens_seen'),
        'current_invocation_seconds':time.monotonic()-started,'teacher_eligible':False})


def evaluate(arm):
    from anvil import _native_engine, _native_infer_batches
    from eval.g_family_qualification import native_diagnostic
    folder=ROOT/arm;done=read_binding(folder/'training_done.json');adapter=Path(done['adapter'])
    assert sha(adapter/'adapter_model.safetensors')==done['adapter_sha256']
    if (folder/'done.json').exists():return
    attempt=Path(tempfile.mkdtemp(prefix='infer-',dir=folder))
    engine=_native_engine(MODEL,adapter,attempt,max_new_tokens=512)
    for view in ['joint','linked']:
        source=ROOT/f'qualification_{view}.jsonl';values=rows(source)
        dest=folder/view;dest.mkdir(exist_ok=True);chunks=dest/'chunks';chunks.mkdir(exist_ok=True)
        binding={'source_sha256':sha(source),'adapter_sha256':done['adapter_sha256'],'max_new_tokens':512}
        spool=prediction_spool(dest,binding)
        for i in range(0,len(values),4):
            batch=values[i:i+4];path=chunks/f'{i:06d}.jsonl';partial=spool/f'{i:06d}.jsonl'
            if path.exists():assert [r['sample_id'] for r in rows(path)]==[r['sample_id'] for r in batch]
            else:
                if not spool_batch_ready(partial,batch):_native_infer_batches(engine,batch,partial)
                commit_prediction(partial,path,batch)
            save(dest/'progress.json',{'completed':i+len(batch),'total':len(values),'time':time.time()})
        prediction=dest/'predictions.jsonl';commit_bytes(prediction,b''.join(p.read_bytes() for p in sorted(chunks.glob('*.jsonl'))))
        if view=='joint':save(dest/'result.json',native_diagnostic(source,[(arm,prediction)]))
        else:save(dest/'result.json',score_linked(values,rows(prediction)))
        save(dest/'done.json',{'source_sha256':sha(source),'prediction_sha256':sha(prediction),'result_sha256':sha(dest/'result.json')})
    save(folder/'done.json',{'training_done_sha256':sha(folder/'training_done.json'),'time':time.time(),'teacher_eligible':False})


def score_linked(source, predictions):
    from trains.mopd.anvil_runtime import reference_geometry_evaluator
    from trains.flywheel.validators import validate
    assert [r['sample_id'] for r in source]==[r['sample_id'] for r in predictions]
    values=[]
    for row,pred in zip(source,predictions):
        text=re.sub(r'^\s*<think>\s*</think>\s*','',pred['response']).strip()
        match=re.fullmatch(r'<GROUND>(.*?)</GROUND>\s*<RELATE>(.*?)</RELATE>',text,re.S)
        scores=[0.,0.];valid=[False,False]
        if match and pred['finish_reason']=='stop':
            try:
                payloads=[json.loads(match.group(i)) for i in [1,2]]
                for i,(stage,form) in enumerate(zip(['G','S'],['detection','depth'])):
                    valid[i]=bool(validate(form,payloads[i],row['gsa_stage_contexts'][i])[0])
                    if valid[i]:
                        score=reference_geometry_evaluator(row=row,stage=stage,form=form,payload=payloads[i],
                            upstream={} if i==0 else {'G':payloads[0]},reference=row['gsa_references'][i])
                        scores[i]=0. if score is None else float(score)
            except (ValueError,TypeError,KeyError,IndexError):pass
        values.append({'sample_id':row['sample_id'],'scene':row['gsa_meta']['scene_group'],
            'G':scores[0],'S':scores[1],'strict_chain':bool(valid[0] and valid[1] and scores[0]>=.75 and scores[1]>.5),
            'finish_reason':pred['finish_reason']})
    return {'records':len(values),'mean':{k:sum(r[k] for r in values)/len(values) for k in ['G','S','strict_chain']},
            'items':values,'scope':'Two-stage held-out source task; no Action score or full benchmark aggregate.'}


def close_stage_tags(text):
    """Diagnostic only: insert missing closing tags; never change JSON content."""
    text=re.sub(r'^\s*<think>\s*</think>\s*','',text).strip()
    match=re.fullmatch(r'<GROUND>(.*?)\s*(?:</GROUND>)?\s*<RELATE>(.*?)\s*(?:</RELATE>)?',text,re.S)
    if not match:return None
    try:
        for i in (1,2):json.loads(match.group(i))
    except (ValueError,TypeError):return None
    return '<GROUND>'+match.group(1)+'</GROUND><RELATE>'+match.group(2)+'</RELATE>'


def collect():
    manifest=read_binding(ROOT/'manifest.json');status={};observed={}
    for arm in ARMS:
        folder=ROOT/arm
        if (folder/'done.json').exists():
            training=read_binding(folder/'training_done.json');adapter=Path(training['adapter'])
            assert sha(adapter/'adapter_model.safetensors')==training['adapter_sha256']
            assert sha(adapter/'trainer_state.json')==training['trainer_state_sha256']
            assert sha(adapter.parent/'args.json')==training['args_sha256']
            assert sha(folder/'training_done.json')==read_binding(folder/'done.json')['training_done_sha256']
            for view in ['joint','linked']:
                bound=read_binding(folder/view/'done.json')
                assert sha(ROOT/f'qualification_{view}.jsonl')==bound['source_sha256']
                assert sha(folder/view/'predictions.jsonl')==bound['prediction_sha256']
                assert sha(folder/view/'result.json')==bound['result_sha256']
            status[arm]='complete'
            raw=rows(folder/'linked/predictions.jsonl');normalized=deepcopy(raw)
            changed=0
            for pred in normalized:
                closed=close_stage_tags(pred['response'])
                original=re.sub(r'^\s*<think>\s*</think>\s*','',pred['response']).strip()
                if closed is not None:
                    changed+=closed!=original;pred['response']=closed
            tag_only=score_linked(rows(ROOT/'qualification_linked.jsonl'),normalized)
            observed[arm]={'atomic':read_binding(folder/'joint/result.json')['models'][arm]['per_form'],
                           'linked':read_binding(folder/'linked/result.json')['mean'],
                           'training':training,
                           'closing_tag_analysis':{'mean':tag_only['mean'],'normalized_rows':changed,
                             'rule':'Post hoc closing-tag insertion only; JSON, labels, geometry, order and finish reason unchanged. Primary strict scores are unchanged.'},
                           'capped_outputs':sum(p['finish_reason']!='stop' for p in raw)}
        elif (folder/'failure.json').exists():status[arm]='failed'
        elif (folder/'training_done.json').exists():status[arm]='evaluating'
        else:status[arm]='training_or_queued'
    result={'status':status,'observed':observed,'manifest_sha256':sha(ROOT/'manifest.json'),
            'split_counts':manifest['split_counts'],'teacher_eligible':False,'full_MTD_complete':False}
    save(ROOT/'acceptance.json',result)
    if len(observed)==4:
        tex=[r'\begin{table}[!htbp]',r'\centering\small',
            r'\caption{\textbf{Learning from source-native linked supervision.} Four fresh-base SFT arms use 512 updates and 2,048 row exposures each. Evaluation covers 282 images in 50 held-out physical scenes. Columns report atomic detection F1, atomic depth accuracy, and G$\to$S success (\%). The final column is a post hoc sensitivity analysis inserting missing closing stage tags without changing either JSON payload. Training tokens are recorded rather than matched. This two-stage source task does not measure full G/S/A or distillation.}',
            r'\label{tab:linked-source-training}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',
            r'Training view & Detection & Depth & Strict G$\to$S & Tags closed \\',r'\midrule']
        for arm in ARMS:
            r=observed[arm];g=r['atomic']['detection']['reference_diagnostics']['primary'];s=r['atomic']['depth']['reference_diagnostics']['mean_per_item_pair_accuracy']
            tex.append(arm+f" & {100*g:.2f} & {100*s:.2f} & {100*r['linked']['strict_chain']:.2f} & {100*r['closing_tag_analysis']['mean']['strict_chain']:.2f}"+r' \\')
        tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
        commit_bytes(A/'overleaf/source/tables/linked_source_training.tex',('\n'.join(tex)+'\n').encode())
    print(json.dumps({'status':status,'split_counts':manifest['split_counts']}),flush=True)
    return result


def worker(arm, gpu):
    folder=ROOT/arm;folder.mkdir(exist_ok=True)
    with (folder/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'done.json').exists():return
        while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))>512:time.sleep(30)
        if (folder/'failure.json').exists():(folder/'failure.json').replace(folder/f'failure_history_{time.time_ns()}.json')
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
        try:
            train(arm,gpu);evaluate(arm)
        except Exception as error:
            save(folder/'failure.json',{'error':repr(error),'time':time.time()});raise


def prepare_contexts():
    """Freeze localization interventions, keeping image-based depth targets fixed."""
    root=ROOT/'contexts_v1';root.mkdir(exist_ok=True)
    original=ROOT/'qualification_S.jsonl'
    student=ROOT/'joint/joint/predictions.jsonl'
    source=rows(original)
    assert sha(student)==read_binding(ROOT/'joint/joint/done.json')['prediction_sha256']
    binding={'source_sha256':sha(original),'student_prediction_sha256':sha(student),
        'student_adapter_sha256':read_binding(ROOT/'joint/training_done.json')['adapter_sha256'],
        'conditions':['reference','student','shifted'],
        'intervention':'Student uses saved joint atomic G output; malformed G gives empty supplied groundings, never oracle substitution. Shifted moves each reference box by a seeded signed 150/1000 displacement per axis with boundary clipping, retaining names and box sizes.',
        'target':'Original scene-relative depth order, including cases with imperfect supplied localization. This is robustness to upstream localization, not action qualification.',
        'generation':{'batch':4,'max_new_tokens':512,'image_tokens':1536,'temperature':0},
        'selection':'All 282 previously frozen qualification scenes/images; descriptive follow-up after the initial source evaluation, not an untouched confirmatory test.'}
    if (root/'manifest.json').exists():
        old=read_binding(root/'manifest.json');assert old['binding']==binding
        for name,b in old['files'].items():assert sha(root/name)==b['sha256']
        return old
    pred={r['sample_id']:r for r in rows(student)};files={};invalid=0
    transformed={'student':[],'shifted':[]}
    import hashlib
    for row in source:
        p=pred[row['sample_id'][:-1]+'G']
        try:
            payload=json.loads(re.sub(r'^\s*<think>\s*</think>\s*','',p['response']).strip())
            assert p['finish_reason']=='stop' and isinstance(payload,list)
            assert all(isinstance(q,dict) and isinstance(q.get('label'),str)
                and len(q.get('bbox_2d',[]))==4 and all(isinstance(v,(int,float)) for v in q['bbox_2d']) for q in payload)
            generated=[{'object_class':q['label'],'bbox':q['bbox_2d']} for q in payload]
        except (ValueError,TypeError,AssertionError):
            generated=[];invalid+=1
        shifted=deepcopy(row['gsa_context']['groundings'])
        seed=int(hashlib.sha256(('context-shift-v1|'+row['sample_id']).encode()).hexdigest()[:16],16)
        rng=random.Random(seed)
        for q in shifted:
            box=q['bbox']
            for axis in (0,1):
                size=box[axis+2]-box[axis]
                box[axis]=max(0,min(1000-size,box[axis]+rng.choice([-150,150])))
                box[axis+2]=box[axis]+size
        for condition,groundings in [('student',generated),('shifted',shifted)]:
            new=deepcopy(row)
            new['messages'][0]['content']=('<image>Order these grounded entities from nearest to farthest in camera depth: '+json.dumps(groundings)+
                '. Output labels, ranks and ordinal relative_depth values. Return only a JSON array; image coordinates use the 0–1000 scale.')
            new['gsa_meta']['supplied_groundings']=groundings
            new['gsa_meta']['context_intervention']=condition
            # gsa_context and gsa_payload remain the scene truth for scoring;
            # only supplied prompt localization changes across conditions.
            transformed[condition].append(new)
    for name,values in transformed.items():
        path=root/(name+'.jsonl');write_rows(path,values)
        files[path.name]={'sha256':sha(path),'records':len(values)}
    result={'binding':binding,'files':files,'invalid_student_groundings':invalid,'rows':len(source),
        'scenes':len({r['gsa_meta']['scene_group'] for r in source}),'teacher_eligible':False}
    save(root/'manifest.json',result);return result


def context_worker(arm,gpu):
    from anvil import _native_engine, _native_infer_batches
    from eval.g_family_qualification import native_diagnostic
    root=ROOT/'contexts_v1';folder=root/arm;folder.mkdir(exist_ok=True)
    with (folder/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'done.json').exists():return
        while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))>512:time.sleep(30)
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
        done=read_binding(ROOT/arm/'training_done.json');adapter=Path(done['adapter'])
        assert sha(adapter/'adapter_model.safetensors')==done['adapter_sha256']
        engine=_native_engine(MODEL,adapter,Path(tempfile.mkdtemp(prefix='infer-',dir=folder)),max_new_tokens=512)
        for condition in ['student','shifted']:
            source=root/(condition+'.jsonl');values=rows(source);dest=folder/condition;dest.mkdir(exist_ok=True)
            chunks=dest/'chunks';chunks.mkdir(exist_ok=True)
            binding={'source_sha256':sha(source),'adapter_sha256':done['adapter_sha256'],
                'generation':read_binding(root/'manifest.json')['binding']['generation']}
            if (dest/'binding.json').exists():assert read_binding(dest/'binding.json')==binding
            else:save(dest/'binding.json',binding)
            spool=prediction_spool(dest,binding)
            for i in range(0,len(values),4):
                batch=values[i:i+4];path=chunks/f'{i:06d}.jsonl';partial=spool/f'{i:06d}.jsonl'
                if path.exists():assert [r['sample_id'] for r in rows(path)]==[r['sample_id'] for r in batch]
                else:
                    if not spool_batch_ready(partial,batch):_native_infer_batches(engine,batch,partial)
                    commit_prediction(partial,path,batch)
                save(dest/'progress.json',{'completed':i+len(batch),'total':len(values),'time':time.time()})
            prediction=dest/'predictions.jsonl';commit_bytes(prediction,b''.join(p.read_bytes() for p in sorted(chunks.glob('*.jsonl'))))
            save(dest/'result.json',native_diagnostic(source,[(arm,prediction)]))
            save(dest/'done.json',{'source_sha256':sha(source),'prediction_sha256':sha(prediction),'result_sha256':sha(dest/'result.json')})
        save(folder/'done.json',{'time':time.time(),'adapter_sha256':done['adapter_sha256']})


def context_status():
    root=ROOT/'contexts_v1';result={}
    for arm in ARMS:
        result[arm]={}
        for condition in ['student','shifted']:
            folder=root/arm/condition
            if (folder/'done.json').exists():
                bound=read_binding(folder/'done.json')
                assert sha(root/(condition+'.jsonl'))==bound['source_sha256']
                assert sha(folder/'predictions.jsonl')==bound['prediction_sha256']
                assert sha(folder/'result.json')==bound['result_sha256']
                score=read_binding(folder/'result.json')['models'][arm]['per_form']['depth']['reference_diagnostics']['mean_per_item_pair_accuracy']
                result[arm][condition]={'state':'complete','accuracy':score}
            else:result[arm][condition]={'state':'running_or_queued',**(read_binding(folder/'progress.json') if (folder/'progress.json').exists() else {})}
    save(root/'status.json',result);print(json.dumps(result),flush=True);return result


def context_supervise():
    prepare_contexts();root=ROOT/'contexts_v1'
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpus=[int(v) for v in os.environ.get('ANVIL_CONTEXT_GPUS','4,5,6,7').split(',')]
        assert len(gpus)==4 and len(set(gpus))==4
        handles=[]
        for gpu,arm in zip(gpus,ARMS):
            log=(root/f'{arm}.log').open('a')
            handles.append((subprocess.Popen([sys.executable,__file__,'context-worker','--arm',arm,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT),log))
        codes=[p.wait() for p,_ in handles]
        for _,log in handles:log.close()
        context_status();save(root/'queue_finished.json',{'exit_codes':codes,'time':time.time()})
        if any(codes):raise RuntimeError('Context comparison incomplete; saved batches retained')


def supervise():
    prepare()
    with (ROOT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handles=[]
        for gpu,arm in enumerate(ARMS):
            log=(ROOT/f'{arm}.log').open('a')
            handles.append((subprocess.Popen([sys.executable,__file__,'worker','--arm',arm,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT),log))
        codes=[p.wait() for p,_ in handles]
        for _,log in handles:log.close()
        result=collect();save(ROOT/'queue_finished.json',{'exit_codes':codes,'time':time.time()})
        if any(codes):raise RuntimeError('Linked-source queue incomplete; rerun same command to resume')
        subprocess.run([sys.executable,str(A/'engineering/experiments/submission_closeout.py'),'publish'],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','supervise','status','worker','prepare-contexts','context-supervise','context-status','context-worker'])
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--gpu',type=int,choices=range(8));args=parser.parse_args()
    {'prepare':prepare,'supervise':supervise,'status':collect,'worker':lambda:worker(args.arm,args.gpu),
     'prepare-contexts':prepare_contexts,'context-supervise':context_supervise,'context-status':context_status,
     'context-worker':lambda:context_worker(args.arm,args.gpu)}[args.command]()
