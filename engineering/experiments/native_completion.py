#!/usr/bin/env python3
"""Finite native-transfer queue; saved batches survive restarts, no wall-clock kill.

COCO polygon transfer extends the very small mask qualification set. This is not
an action-plan benchmark or an automatic certificate of teacher superiority.
"""
from __future__ import annotations
import argparse
import collections
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

A = Path(os.environ.get('ANVIL_ROOT', Path(__file__).resolve().parents[2]))
W = Path(os.environ.get('ANVIL_WORKTREE', A))
D = Path(os.environ.get('ANVIL_DATA', A / 'engineering/data'))
OUT = D / 'native_completion_20260915'
if os.environ.get('ANVIL_NATIVE_COHORT')=='sensor-mask': OUT=OUT/'sensor_mask'
MODEL = Path(os.environ.get('MODEL_9B', '/path/to/Qwen3.5-9B'))
ROLES = ['G', 'S', 'A', 'joint']
sys.path[:0] = [str(A/'engineering'), str(W)]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def save(p, d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    commit_bytes(p,(json.dumps(d,indent=2,ensure_ascii=False)+'\n').encode())

def commit_bytes(p,payload):
    """Retain the complete payload while quota is unavailable; never infer twice."""
    p=Path(p)
    while True:
        pending=None
        try:
            with tempfile.NamedTemporaryFile(mode='wb',prefix=p.name+'.',suffix='.pending',dir=p.parent,delete=False) as f:
                pending=Path(f.name);f.write(payload);f.flush();os.fsync(f.fileno())
            pending.replace(p)
            return
        except OSError as exc:
            if pending is not None:pending.unlink(missing_ok=True)
            if exc.errno not in [errno.EDQUOT,errno.ENOSPC]:raise
            try:print(f'Storage unavailable; retaining payload and retrying in 30 s: {p}',flush=True)
            except OSError:pass
            time.sleep(30)

def prediction_spool(folder,binding):
    identity=json.dumps({'folder':str(folder),'binding':binding},sort_keys=True).encode()
    spool=Path('/tmp/anvil-native-prediction-spool')/hashlib.sha256(identity).hexdigest()
    spool.mkdir(parents=True,exist_ok=True)
    return spool

def commit_prediction(partial,target,batch):
    assert [r['sample_id'] for r in rows(partial)]==[r['sample_id'] for r in batch]
    with partial.open('rb') as saved:os.fsync(saved.fileno())
    commit_bytes(target,partial.read_bytes())
    partial.unlink()

def spool_batch_ready(partial,batch):
    if not partial.exists():return False
    try:
        assert [r['sample_id'] for r in rows(partial)]==[r['sample_id'] for r in batch]
        return True
    except (ValueError,KeyError,AssertionError):
        partial.replace(partial.with_suffix(f'.incomplete-{time.time_ns()}'))
        return False

def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

def jobs():
    result=[]
    for gpu,role in enumerate(ROLES):
        for seed,arm in [(0,'initial'),(17,'control'),(17,'feedback'),(23,'control'),(23,'feedback')]:
            if arm=='initial': root=D/'native_teacher_pilot_v2'/role
            else:
                root=D/('native_teacher_refresh_recovery_v1' if seed==17 else 'native_teacher_refresh_seed23_v1')/(role+('_control_r1' if arm=='control' else '_feedback'))
                if not (root/'selected_adapter').exists():
                    root=D/('native_teacher_refresh_v1' if seed==17 else 'native_teacher_refresh_seed23_v1')/(role+('_control_r1' if arm=='control' else '_feedback'))
                if not (root/'selected_adapter').exists() and role=='joint' and seed==23 and arm=='feedback':
                    root=D/'native_teacher_refresh_seed23_v1/joint_feedback_retry1'
            result.append(dict(id=f'{role}_{arm}_{seed}',gpu=gpu,role=role,seed=seed,arm=arm,adapter=str(root/'selected_adapter')))
    return result

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if os.environ.get('ANVIL_NATIVE_COHORT')=='sensor-mask':
        p=D/'scannet_native_sensor_mask_v2/dev.jsonl'
        kept=[r for r in rows(p) if r['gsa_form']=='mask']
        source=OUT/'evaluation.jsonl'
        payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept)
        if source.exists():assert source.read_text()==payload
        else:source.write_text(payload)
        manifest={'schema':'anvil-native-sensor-mask-transfer-v1','source':str(source),'sha256':sha(source),'rows':len(kept),
            'generation':{'batch':4,'max_new_tokens':512,'max_length':3072,'temperature':0,'image_max_token_num':1536},
            'sources':{str(p):{'sha256':sha(p)}},'jobs':jobs(),'full_teacher_qualification':False}
        save(OUT/'manifest.json',manifest)
        return manifest
    target=OUT/'coco_expansion'
    if not (target/'manifest.json').is_file():
        # The existing converter uses official polygons, content hashes and a
        # deterministic image partition. Download caching makes retries cheap.
        if target.exists():
            raise RuntimeError('Incomplete source preparation retained; inspect it before retrying.')
        command=[sys.executable,str(A/'engineering/anvil.py'),'prepare-coco-masks',
            '--worktree',str(W),'--annotations-zip',str(D/'coco_official_v1/annotations_trainval2017.zip'),
            '--expected-annotations-sha256','113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268',
            '--image-cache',str(D/'coco_official_v1/train2017'),'--output',str(target),
            '--limit','2048','--workers','8','--seed','anvil-coco-mask-transfer-20260915',
            '--dev-permille','500','--mask-min-iou','.985','--polygon-max-vertices','96',
            '--min-box-side-px','50','--direct-download',
            '--categories','bottle','bowl','cup','fork','knife','scissors','spoon','teddy bear','vase','wine glass']
        for name in ['coco_native_mask_canary_v1','coco_native_mask_refined_v2']:
            command += ['--exclude-corpus',str(D/name/'audit_corpus.jsonl')]
        env={**os.environ,'PYTHONPATH':str(D/'coco_official_v1/python_deps')+os.pathsep+str(W)}
        with (OUT/'prepare.log').open('a') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    sources=[target/'dev.jsonl',D/'scannet_native_sensor_mask_v2/dev.jsonl',D/'reviewed_native_teachers_v2/qualification.jsonl']
    reserved=set()
    for name in ['reviewed_native_teachers_v2','native_feedback_v35_clean_v1']:
        for r in rows(D/name/'audit_corpus.jsonl'):
            if r.get('gsa_meta',{}).get('image_sha256'):reserved.add(r['gsa_meta']['image_sha256'])
    merged=[]
    for p in sources:
        for row in rows(p):
            if p.parent==target and row['gsa_meta']['image_sha256'] in reserved:continue
            row['gsa_eval_source']=p.parent.name
            merged.append(row)
    ids=[r['sample_id'] for r in merged]
    assert len(ids)==len(set(ids))
    source=OUT/'evaluation.jsonl'
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged)
    if source.exists(): assert source.read_text()==payload,'Frozen evaluation drift'
    else: source.write_text(payload)
    manifest={'schema':'anvil-native-transfer-completion-v1','source':str(source),'sha256':sha(source),
        'rows':len(merged),'forms':dict(collections.Counter(r['gsa_form'] for r in merged)),
        'sources':{str(p):{'sha256':sha(p),'rows':len(rows(p))} for p in sources},'jobs':jobs(),
        'generation':{'batch':4,'max_new_tokens':512,'max_length':3072,'temperature':0,'image_max_token_num':1536},
        'purpose':'Expanded source-native mask transfer and fixed historical/sensor references across initial and refreshed teachers.',
        'independent_action_plan_truth':False,'full_teacher_qualification':False,
        'core_dependencies':['independent action-plan alternatives and checklists','qualified G/S/A superiority under three contexts','implemented five-arm matched-budget routing comparison','matched-call filtering precision experiment']}
    for job in jobs():
        assert Path(job['adapter'],'adapter_model.safetensors').is_file(),job
    save(OUT/'manifest.json',manifest)
    return manifest

def worker(jobid):
    from anvil import _native_engine, _native_infer_batches
    from eval.g_family_qualification import native_diagnostic
    job=next(j for j in jobs() if j['id']==jobid)
    folder=OUT/jobid;folder.mkdir(exist_ok=True)
    with (folder/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((OUT/'manifest.json').read_text())
        source=Path(manifest['source']);assert sha(source)==manifest['sha256']
        adapter=Path(job['adapter'])
        binding={'source_sha256':sha(source),'adapter':str(adapter),'weights_sha256':sha(adapter/'adapter_model.safetensors'),
                 'adapter_config_sha256':sha(adapter/'adapter_config.json'),'generation':manifest['generation']}
        receipt=folder/'binding.json'
        if receipt.exists(): assert json.loads(receipt.read_text())==binding,'Run binding changed'
        else: save(receipt,binding)
        spool=prediction_spool(folder,binding)
        allrows=rows(source)
        chunks=folder/'chunks';chunks.mkdir(exist_ok=True)
        missing=[]
        for i in range(0,len(allrows),4):
            batch=allrows[i:i+4];p=chunks/f'{i:06d}.jsonl'
            if p.exists():
                assert [r['sample_id'] for r in rows(p)]==[r['sample_id'] for r in batch]
            elif spool_batch_ready(spool/f'{i:06d}.jsonl',batch):
                commit_prediction(spool/f'{i:06d}.jsonl',p,batch)
            else: missing.append((i,batch,p))
        if missing:
            attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=folder))
            engine=_native_engine(MODEL,adapter,attempt)
            for i,batch,p in missing:
                partial=spool/f'{i:06d}.jsonl'
                _native_infer_batches(engine,batch,partial)
                assert len(rows(partial))==len(batch)
                commit_prediction(partial,p,batch)
                save(folder/'progress.json',{'completed':i+len(batch),'total':len(allrows),'updated':time.time()})
                print(jobid,i+len(batch),'/',len(allrows),flush=True)
        prediction=folder/'predictions.jsonl'
        commit_bytes(prediction,b''.join(p.read_bytes() for p in sorted(chunks.glob('*.jsonl'))))
        report=native_diagnostic(source,[(jobid,prediction)])
        save(folder/'result.json',report)
        save(folder/'done.json',{'job':job,'rows':len(allrows),'prediction_sha256':sha(prediction),'result_sha256':sha(folder/'result.json'),'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})

def inventory():
    found=[]
    for p in sorted((A/'overleaf/source').rglob('*.tex')):
        for n,line in enumerate(p.read_text().splitlines(),1):
            line=re.sub(r'(?<!\\)%.*','',line)
            if '\\needcheck{' in line: found.append({'file':str(p.relative_to(A)),'line':n,'count':line.count('\\needcheck{'),'text':line.strip()})
    return found

def repair_caps(jobid):
    """Only retry length-truncated responses; keep the fixed-512 result intact."""
    from anvil import _native_engine, _native_infer_batches
    from eval.g_family_qualification import native_diagnostic
    job=next(j for j in jobs() if j['id']==jobid)
    source=Path(json.loads((OUT/'manifest.json').read_text())['source'])
    original=OUT/jobid/'predictions.jsonl'
    folder=OUT/'cap_repair'/jobid;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'done.json').exists():return
        predicted=rows(original);byid={r['sample_id']:r for r in predicted}
        retry=[r for r in rows(source) if byid[r['sample_id']]['finish_reason']=='length']
        binding={'primary_sha256':sha(original),'source_sha256':sha(source),'adapter_weights_sha256':sha(Path(job['adapter'])/'adapter_model.safetensors'),
            'retry_rule':'finish_reason == length; no reference or score enters retry selection','max_new_tokens':1024,'batch':4}
        if (folder/'binding.json').exists():assert read_binding(folder/'binding.json')==binding
        else:save(folder/'binding.json',binding)
        spool=prediction_spool(folder,binding)
        chunks=folder/'chunks';chunks.mkdir(exist_ok=True)
        missing=[]
        for i in range(0,len(retry),4):
            batch=retry[i:i+4];target=chunks/f'{i:06d}.jsonl';partial=spool/f'{i:06d}.jsonl'
            if target.exists():continue
            if spool_batch_ready(partial,batch):commit_prediction(partial,target,batch)
            else:missing.append((i,batch,target))
        if missing:
            attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=folder))
            engine=_native_engine(MODEL,Path(job['adapter']),attempt,max_new_tokens=1024)
            for i,batch,p in missing:
                partial=spool/f'{i:06d}.jsonl';_native_infer_batches(engine,batch,partial)
                assert len(rows(partial))==len(batch)
                commit_prediction(partial,p,batch)
                save(folder/'progress.json',{'completed':i+len(batch),'total':len(retry),'updated':time.time()})
        repaired=[r for p in sorted(chunks.glob('*.jsonl')) for r in rows(p)]
        assert {r['sample_id'] for r in repaired}=={r['sample_id'] for r in retry}
        byid.update({r['sample_id']:r for r in repaired})
        combined=folder/'predictions.jsonl'
        commit_bytes(combined,(''.join(json.dumps(byid[r['sample_id']],ensure_ascii=False)+'\n' for r in predicted)).encode())
        save(folder/'result.json',native_diagnostic(source,[(jobid,combined)]))
        save(folder/'done.json',{'retry_count':len(retry),'total':len(predicted),'result_sha256':sha(folder/'result.json'),
            'remaining_caps':sum(r['finish_reason']=='length' for r in byid.values()),'prediction_sha256':sha(combined)})

def read_binding(p):return json.loads(Path(p).read_text())

def collect_mask_contrasts(inputs, adaptive=False):
    """Pair saved IoUs by identity; resample source images or physical scenes.

    Intervals condition on the stated seed/checkpoints and do not estimate
    training-seed uncertainty. Every mask record remains in its denominator.
    """
    import numpy as np
    root=OUT/'cap_repair' if adaptive else OUT
    required=['G_initial_0','joint_initial_0']+[
        f'{role}_{arm}_{seed}' for role in ['G','joint']
        for seed in [17,23] for arm in ['control','feedback']]
    if not all((root/j/'done.json').exists() for j in required):return
    inputs=[r for r in inputs if r['gsa_form']=='mask']
    ids=[r['sample_id'] for r in inputs];assert len(ids)==len(set(ids))
    values={};bindings={}
    for name in required:
        folder=root/name;done=read_binding(folder/'done.json')
        assert done['result_sha256']==sha(folder/'result.json')
        assert done['prediction_sha256']==sha(folder/'predictions.jsonl')
        doc=read_binding(folder/'result.json');model=doc['models'][name]
        assert doc['source']['sha256']==sha(Path(doc['source']['path']))
        observations={r['sample_id']:r for r in model['items'] if r['form']=='mask'}
        assert set(observations)==set(ids)
        vector=np.asarray([observations[i]['g_reference_scores']['mask_iou'] for i in ids],dtype=float)
        assert np.isfinite(vector).all() and ((vector>=0)&(vector<=1)).all()
        assert abs(float(vector.mean())-model['per_form']['mask']['reference_diagnostics']['mask_iou'])<1e-12
        values[name]=vector
        bindings[name]={'result_sha256':done['result_sha256'],'prediction_sha256':done['prediction_sha256']}
    sources={}
    for source,unit in [('coco_expansion','image'),('scannet_native_sensor_mask_v2','physical_scene')]:
        indices=[i for i,r in enumerate(inputs) if r.get('gsa_eval_source','scannet_native_sensor_mask_v2')==source]
        if not indices:continue
        groups=[]
        for i in indices:
            meta=inputs[i]['gsa_meta']
            groups.append(str(meta['source_image_id']) if unit=='image' else meta['scene_group'])
        labels=sorted(set(groups));assert len(labels)>1
        positions={g:k for k,g in enumerate(labels)}
        group_ids=np.asarray([positions[g] for g in groups]);counts=np.bincount(group_ids)
        contrasts=[('Initial G minus joint','G_initial_0','joint_initial_0')]+[
            (f'{role} feedback minus replay, s{seed}',f'{role}_feedback_{seed}',f'{role}_control_{seed}')
            for role in ['G','joint'] for seed in [17,23]]
        results=[]
        for label,left,right in contrasts:
            differences=values[left][indices]-values[right][indices]
            sums=np.bincount(group_ids,weights=differences,minlength=len(labels))
            rng=np.random.default_rng(20260915);estimates=[]
            for start in range(0,5000,100):
                draw=rng.integers(0,len(labels),size=(min(100,5000-start),len(labels)))
                estimates.extend((sums[draw].sum(axis=1)/counts[draw].sum(axis=1)).tolist())
            lo,hi=np.quantile(estimates,[.025,.975])
            results.append({'contrast':label,'left':left,'right':right,
                'left_mean':float(values[left][indices].mean()),'right_mean':float(values[right][indices].mean()),
                'difference':float(differences.mean()),'cluster_ci95':[float(lo),float(hi)]})
        sources[source]={'records':len(indices),'clusters':len(labels),'resampling_unit':unit,'contrasts':results}
    protocol='adaptive' if adaptive else 'fixed512'
    report={'schema':'anvil-native-mask-paired-contrasts-v1','protocol':protocol,
        'source_sha256':sha(OUT/'evaluation.jsonl'),'bindings':bindings,'sources':sources,
        'bootstrap':{'replicates':5000,'seed':20260915,'estimator':'record-weighted mean paired difference',
            'interval':'percentile, conditional on fixed checkpoints; not adjusted for multiple comparisons'},
        'scope':'Descriptive transfer contrasts; initial teachers have different original training budgets. No three-context qualification.'}
    save(OUT/f'mask_contrasts_{protocol}.json',report)
    # The expanded cohort contains exactly the same RGB-D source rows as its
    # separately reported sensor cohort; avoid a duplicated statistical table.
    if os.environ.get('ANVIL_NATIVE_COHORT')=='sensor-mask':return
    labels={'coco_expansion':'COCO','scannet_native_sensor_mask_v2':'RGB-D'}
    tex=[r'\begin{table}[!htbp]',r'\centering\small',
        r'\caption{\textbf{Paired mask-transfer differences ('+('selective continuation' if adaptive else '512-token limit')+r').} Polygon-IoU percentage points and descriptive 95\% percentile intervals from 5,000 paired cluster resamples. COCO clusters are images; RGB-D clusters are physical scenes. Each seed is reported separately; these intervals condition on the saved checkpoints and do not estimate seed uncertainty.}',
        r'\label{tab:native-mask-contrasts-'+protocol+r'}',r'\begin{tabular}{@{}lrr@{}}',r'\toprule',
        r'Contrast & COCO & RGB-D \\',r'\midrule']
    for index in range(5):
        cells=[]
        for source in labels:
            item=sources[source]['contrasts'][index];lo,hi=item['cluster_ci95']
            cells.append(f"{100*item['difference']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
        tex.append(sources['coco_expansion']['contrasts'][index]['contrast']+' & '+' & '.join(cells)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip\footnotesize '+
        '; '.join(f"{labels[s]}: {r['records']} masks, {r['clusters']} clusters" for s,r in sources.items())+'.',r'\end{table}']
    commit_bytes(A/'overleaf/source/tables'/f'native_mask_contrasts_{protocol}.tex',('\n'.join(tex)+'\n').encode())

def collect():
    result={}
    for job in jobs():
        p=OUT/job['id']/'done.json'
        if p.exists():
            done=json.loads(p.read_text());assert done['result_sha256']==sha(p.with_name('result.json'))
            d=json.loads(p.with_name('result.json').read_text())['models'][job['id']]
            result[job['id']]={'state':'complete','per_form':d['per_form'],'output_health':d['output_health']}
        elif (p.parent/'failure.json').exists():result[job['id']]={'state':'failed','details':json.loads((p.parent/'failure.json').read_text())}
        elif (p.parent/'progress.json').exists():result[job['id']]={'state':'running_or_interrupted',**json.loads((p.parent/'progress.json').read_text())}
        else:result[job['id']]={'state':'queued'}
    marks=inventory()
    d={'jobs':result,'needcheck_calls':sum(x['count'] for x in marks),'needcheck_locations':marks,
       'full_MTD_complete':False,'automatic_numeric_substitution':False,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    plans=D/'alfred_native_plan_sources_v1/plan_judgment_v1'
    if (plans/'binding.json').exists():
        d['action_plan_verification']={'source_pairs':len(list((plans/'records').glob('*.json'))),
            'prepared':(plans/'manifest.json').exists(),'published':(plans/'published.json').exists(),
            'jobs':{j['id']:'complete' if (plans/j['id']/'done.json').exists() else
                'failed' if (plans/j['id']/'failure.json').exists() else
                read_binding(plans/j['id']/'progress.json') if (plans/j['id']/'progress.json').exists() else 'queued'
                for j in alfred_plan_jobs()}}
    save(OUT/'acceptance.json',d)
    if all(j['state']=='complete' for j in result.values()):
        cohort='sensor_mask' if os.environ.get('ANVIL_NATIVE_COHORT')=='sensor-mask' else 'coco_transfer'
        manifest=json.loads((OUT/'manifest.json').read_text());inputs=rows(Path(manifest['source']))
        collect_mask_contrasts(inputs)
        collect_mask_contrasts(inputs,adaptive=True)
        source_counts=collections.Counter(r.get('gsa_eval_source','scannet_native_sensor_mask_v2') for r in inputs if r['gsa_form']=='mask')
        repaired={}
        for job in jobs():
            folder=OUT/'cap_repair'/job['id']
            if (folder/'done.json').exists():
                done=read_binding(folder/'done.json');assert done['result_sha256']==sha(folder/'result.json')
                repaired[job['id']]={'result':read_binding(folder/'result.json')['models'][job['id']],**done}
        if repaired:save(OUT/'cap_repair/acceptance.json',repaired)
        # This table reports source-specific polygon overlap, not a complete
        # native benchmark or human semantic precision. Keep every target case.
        tex=['% Generated from complete, bound prediction journals.',r'\begin{table}[!htbp]',r'\centering\small',
            r'\caption{\textbf{Native mask transfer ('+('RGB-D' if cohort=='sensor_mask' else 'expanded source pool')+r').} Mean polygon IoU (\%) at a 512-token output budget; the five rows use the same evaluation records. S/A columns test cross-family transfer. Replay and feedback continuations use 512 updates; the initial checkpoints use their original training budgets. This does not measure action-plan semantics or a complete G/S/A aggregate.}',
            r'\label{tab:native-'+cohort.replace('_','-')+r'}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Checkpoint & G & S & A & Joint \\',r'\midrule']
        for seed,arm in [(0,'initial'),(17,'control'),(17,'feedback'),(23,'control'),(23,'feedback')]:
            label={'initial':'Initial','control':'Replay','feedback':'Feedback'}[arm]+('' if seed==0 else f' (seed {seed})')
            vals=[result[f'{role}_{arm}_{seed}']['per_form']['mask']['reference_diagnostics']['mask_iou']*100 for role in ROLES]
            tex.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
        source_names={'coco_expansion':'COCO train2017','scannet_native_sensor_mask_v2':'RGB-D instances','reviewed_native_teachers_v2':'historical references'}
        tex += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip\footnotesize Sources: '+', '.join(source_names[name]+f' ({n})' for name,n in source_counts.items())+'.',r'\end{table}']
        table=A/'overleaf/source/tables'/f'native_{cohort}_observation.tex'
        content='\n'.join(tex)+'\n'
        if not table.exists() or table.read_text()!=content:table.write_text(content)
        if len(repaired)==len(jobs()):
            tex=[r'\begin{table}[!htbp]',r'\centering\small',
                 r'\caption{\textbf{Native mask transfer with selective continuation.} The same source pool and checkpoints as Table~\ref{tab:native-'+cohort.replace('_','-')+r'}. Only outputs that reached 512 tokens are retried with a 1,024-token limit; stopped responses are reused. Entries report polygon IoU (\%). Retry decisions use termination status, not reference answers.}',
                 r'\label{tab:native-adaptive-'+cohort.replace('_','-')+r'}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Checkpoint & G & S & A & Joint \\',r'\midrule']
            for seed,arm in [(0,'initial'),(17,'control'),(17,'feedback'),(23,'control'),(23,'feedback')]:
                label={'initial':'Initial','control':'Replay','feedback':'Feedback'}[arm]+('' if seed==0 else f' (seed {seed})')
                vals=[repaired[f'{role}_{arm}_{seed}']['result']['per_form']['mask']['reference_diagnostics']['mask_iou']*100 for role in ROLES]
                tex.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
            tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
            table=A/'overleaf/source/tables'/f'native_{cohort}_adaptive.tex';content='\n'.join(tex)+'\n'
            if not table.exists() or table.read_text()!=content:table.write_text(content)
    print(json.dumps({'jobs':{k:v['state'] for k,v in result.items()},'needcheck':d['needcheck_calls'],
        **({'action_plan_verification':d['action_plan_verification']} if 'action_plan_verification' in d else {})}),flush=True)

def lane(gpu,repair=False):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)}
    # Prefer the assigned family, then take work from other lanes. A fast A
    # lane can help with long G polygon outputs instead of leaving its GPU idle.
    ordered=sorted(jobs(),key=lambda j:j['gpu']!=gpu)
    for job in ordered:
        folder=(OUT/'cap_repair'/job['id']) if repair else OUT/job['id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'done.json').exists():continue
        while True:
            used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used<=512:break
            if all(((OUT/'cap_repair'/j['id']) if repair else (OUT/j['id'])).joinpath('done.json').exists() for j in jobs()):return
            print(f'Lane {gpu} waiting: {used} MiB occupied; other lanes continue',flush=True)
            time.sleep(30)
        with (folder/'dispatch.lock').open('w') as claim:
            try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (folder/'done.json').exists():continue
            if (folder/'failure.json').exists():
                (folder/'failure.json').replace(folder/f'failure_history_{time.time_ns()}.json')
            save(folder/'dispatch.json',{'actual_gpu':gpu,'pid':os.getpid(),'time':time.time()})
            with (folder/'run.log').open('a') as log:
                code=subprocess.run([sys.executable,__file__,'repair' if repair else 'worker','--job',job['id']],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:save(folder/'failure.json',{'exit_code':code,'retry':'rerun same queue; only missing four-item chunks are regenerated','time':time.time()})
            collect()

def collect_all():
    """One CPU-only entry for saved cohorts and the completed closeout suite."""
    env={k:v for k,v in os.environ.items() if k!='ANVIL_NATIVE_COHORT'}
    subprocess.run([sys.executable,__file__,'status'],env=env,check=True)
    if (D/'native_completion_20260915/sensor_mask/manifest.json').is_file():
        subprocess.run([sys.executable,__file__,'status'],
                       env={**env,'ANVIL_NATIVE_COHORT':'sensor-mask'},check=True)
    subprocess.run([sys.executable,str(A/'engineering/experiments/submission_closeout.py'),'collect'],
                   env=env,check=True)

def inspect_linked_sources():
    """Inventory recoverable links; do not invent targets, edges or calibration."""
    from trains.flywheel.validators import validate_structural
    root=D/'reviewed_native_teachers_v2';report={'reviewed_splits':{}}
    for split in ['train','dev','qualification']:
        p=root/(split+'.jsonl');groups=collections.defaultdict(dict)
        if not p.exists():continue
        for r in rows(p):
            image=r['gsa_meta']['image_sha256'];form=r['gsa_form']
            assert form not in groups[image],'Duplicate reviewed image/form'
            groups[image][form]=r
        candidates=[]
        for h,g in groups.items():
            if not {'detection','depth'}<=g.keys():continue
            ground,depth=g['detection'],g['depth'];deps=depth['gsa_meta'].get('upstream_reviewed_sample_ids',[])
            candidates.append({'image_sha256':h,'ground_id':ground['sample_id'],'depth_id':depth['sample_id'],
                'dependency_confirmed':ground['sample_id'] in deps,
                'entity_set_equal':{v['label'] for v in ground['gsa_payload']}=={v['label'] for v in depth['gsa_payload']},
                'available_action_forms':sorted(k for k in g if k in ['affordance','dof','action_cot'])})
        report['reviewed_splits'][split]={'source_sha256':sha(p),'images':len(groups),'pairs':len(candidates),
            'confirmed_pairs':sum(c['dependency_confirmed'] and c['entity_set_equal'] for c in candidates),'candidates':candidates}
    p=W/'trains/anvil_p4_image_disjoint_v1/train.jsonl'
    groups=collections.defaultdict(set);targets=collections.defaultdict(lambda:collections.defaultdict(list));counts=collections.Counter()
    with p.open() as handle:
        for line in handle:
            r=json.loads(line);form=r.get('gsa_meta',{}).get('form');counts[form]+=1;images=tuple(r.get('images',[]))
            if len(images)!=1:continue
            groups[images].add(form)
            if form in ['detection','depth']:
                targets[images][form].append(r['messages'][-1]['content'])
    pairs=[key for key,value in groups.items() if {'detection','depth'}<=value]
    outcomes=collections.Counter();compatible=[];validations=collections.Counter()
    for key in pairs:
        pair=targets[key]
        if any(len(pair[f])!=1 for f in ['detection','depth']):outcomes['ambiguous_duplicate_form']+=1;continue
        try:g,s=[json.loads(pair[f][0]) for f in ['detection','depth']]
        except (ValueError,TypeError):outcomes['non_json_target']+=1;continue
        g_ok,g_reason=validate_structural('detection',g);s_ok,s_reason=validate_structural('depth',s)
        if not g_ok:outcomes['invalid_detection']+=1;validations['detection:'+str(g_reason)]+=1;continue
        if not s_ok:outcomes['invalid_depth']+=1;validations['depth:'+str(s_reason)]+=1;continue
        if {v['label'] for v in g}!={v['label'] for v in s}:outcomes['different_entity_sets']+=1;continue
        if len(s)<2:outcomes['fewer_than_two_depth_entities']+=1;continue
        outcomes['compatible_targets']+=1
        compatible.append({'image':key[0],'detection_target_sha256':hashlib.sha256(pair['detection'][0].encode()).hexdigest(),
            'depth_target_sha256':hashlib.sha256(pair['depth'][0].encode()).hexdigest(),'entities':len(s)})
    report['p4']={'source':str(p),'source_sha256':sha(p),'rows':sum(counts.values()),'forms':dict(counts),
        'images_by_path':len(groups),'same_path_detection_depth':len(pairs),
        'with_action_form':sum(bool({'affordance','dof','action_cot'}&groups[key]) for key in pairs),
        'compatibility_outcomes':dict(outcomes),'validation_reasons':dict(validations),'compatible_candidates':compatible,
        'validator_sha256':sha(W/'trains/flywheel/validators.py'),
        'scope':'Current native structure and entity-set compatibility only; original consumed edges, semantic correctness, pixel identity and calibration are not certified.'}
    assert sum(outcomes.values())==len(pairs)
    save(D/'native_teacher_pilot_v2/linked_source_inventory.json',report)
    print(json.dumps({'reviewed_splits':{k:{x:v[x] for x in ['images','pairs','confirmed_pairs']} for k,v in report['reviewed_splits'].items()},
        'p4_rows':sum(counts.values()),'cooccurring_pairs':len(pairs),'compatibility_outcomes':dict(outcomes)}),flush=True)

def bind_alfred_planner_sources(root):
    """Keep the upstream definitions as data; never execute downloaded scripts."""
    manifest=read_binding(root/'manifest.json')
    if manifest.get('planner_sources'):
        for item in manifest['planner_sources']['files']:
            assert sha(item['path'])==item['sha256'],'ALFRED upstream reference changed'
        return
    import urllib.request
    revision='f91f4c0c96c7a29f33d0557f86b0a21035379b3b'
    names=['gen/planner/domains/PutTaskExtended_domain.pddl','gen/planner/ff_planner_handler.py',
           'gen/scripts/augment_trajectories.py','LICENSE']
    # The workspace proxy returns 503 for GitHub. Direct public HTTPS is scoped
    # to this pinned reference download and does not alter any training process.
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    files=[];folder=root/'planner_reference';folder.mkdir(exist_ok=True)
    for name in names:
        url=f'https://raw.githubusercontent.com/askforalfred/alfred/{revision}/{name}'
        with opener.open(url,timeout=30) as response:payload=response.read()
        assert payload and len(payload)<1_000_000
        dest=folder/Path(name).name;pending=dest.with_suffix(dest.suffix+'.pending')
        pending.write_bytes(payload);pending.replace(dest)
        files.append({'path':str(dest),'url':url,'sha256':sha(dest)})
    manifest['planner_sources']={'repository':'https://github.com/askforalfred/alfred',
        'revision':revision,'files':files,'execution_checked':False}
    save(root/'manifest.json',manifest)

def audit_alfred_metadata(root):
    """Check source bindings and initial-pose metadata, without simulator claims."""
    manifest=read_binding(root/'manifest.json');packets=root/'expert_sources.jsonl'
    assert sha(packets)==manifest['expert_sources_sha256']
    records=rows(packets);scenes=collections.defaultdict(set);actions=collections.Counter()
    failures=[];pose_matches=0;first_indices=0;source_bytes=0
    for row in records:
        for item in [row['trajectory'],row['frame'],*row['pddl_states']]:
            assert sha(item['path'])==item['sha256'],'ALFRED source bytes changed'
            source_bytes+=Path(item['path']).stat().st_size
        doc=read_binding(row['trajectory']['path']);scenes[row['split']].add(row['scene'])
        actions.update(a['action'] for a in row['expert_high_actions'])
        init=doc['scene']['init_action']
        expected=[round(init['x']/.25),round(init['z']/.25),round(init['rotation']/90)%4,round(init['horizon'])]
        pddl=Path(row['pddl_states'][0]['path']).read_text().split('(:goal',1)[0]
        locations=re.findall(r'\(atLocation\s+agent1\s+(\S+)\)',pddl,re.IGNORECASE)
        decoded=[]
        for location in locations:
            text=location.replace('_bar_','|').replace('_minus_','-').replace('_dot_','.')
            decoded.append([float(v) for v in text.split('|')[1:]])
        pose_ok=len(decoded)==1 and decoded[0]==expected
        index_ok=(row['frame']['metadata_entry']['low_idx'],row['frame']['metadata_entry']['high_idx'])==(0,0)
        pose_matches+=pose_ok;first_indices+=index_ok
        if not pose_ok or not index_ok:
            failures.append({'task_id':row['task_id'],'expected_pose':expected,'pddl_pose':decoded,'first_frame_index_ok':index_ok})
    report={'schema':'anvil-alfred-source-metadata-audit-v1','expert_sources_sha256':sha(packets),
        'records':len(records),'source_bytes':source_bytes,'unique_scenes':len(set().union(*scenes.values())),
        'scenes_per_split':{k:len(v) for k,v in scenes.items()},
        'seen_unseen_scene_overlap':sorted(scenes['valid_seen']&scenes['valid_unseen']),
        'initial_pose_matches':pose_matches,'first_frame_zero_indices':first_indices,
        'high_level_entries':sum(actions.values()),'high_level_actions':dict(actions),'failures':failures,
        'scope':'PDDL initial agent pose matches trajectory initialization; first frame is indexed before low/high action zero. No simulator or pixel-state replay.',
        'simulator_replay_checked':False,'expert_transition_replay_checked':False}
    save(root/'metadata_audit.json',report)
    manifest['metadata_audit']={'path':str(root/'metadata_audit.json'),'sha256':sha(root/'metadata_audit.json'),
        'initial_pose_matches':pose_matches,'records':len(records),'failures':len(failures)}
    save(root/'manifest.json',manifest)
    return report

def alfred_boolean_problem(packet, domain_path):
    """Preserve source Boolean transitions; normalize FF-specific PDDL syntax.

    Costs never appear in a source precondition or goal. Their projection is
    suitable for applicability/goal checks, not shortest-path or cost scoring.
    """
    from unified_planning.environment import get_environment
    from unified_planning.io import PDDLReader
    def parse(text):
        tokens=re.findall(r'\(|\)|[^\s()]+',re.sub(r';[^\n]*','',text).lower())
        stack=[];result=None
        for token in tokens:
            if token=='(':
                value=[]
                if stack:stack[-1].append(value)
                stack.append(value)
            elif token==')':
                result=stack.pop()
            else:stack[-1].append(token)
        assert not stack and result[0]=='define'
        return result
    def serialize(tree):
        return '('+' '.join(serialize(v) if isinstance(v,list) else v for v in tree)+')'
    domain=parse(Path(domain_path).read_text());problem=parse(Path(packet['pddl_states'][0]['path']).read_text())
    # The released FF files place these symbols in problem objects even though
    # domain actions refer to them as constants. Move their declarations only.
    constants={'sinkbasintype':'rtype','microwavetype':'rtype','fridgetype':'rtype',
               'knifetype':'otype','butterknifetype':'otype'}
    objects=next(x for x in problem if isinstance(x,list) and x[0]==':objects')
    assert all(objects[i+1]=='-' for i in range(1,len(objects),3))
    kept=[':objects'];moved=[];declared={};duplicates=[]
    for i in range(1,len(objects),3):
        name,_,kind=objects[i:i+3]
        if name in declared:
            if declared[name]!=kind:
                raise ValueError(f'Conflicting source object types for {name}: {declared[name]} / {kind}')
            duplicates.append({'name':name,'type':kind})
            continue
        declared[name]=kind
        if name in constants:
            assert constants[name]==kind;moved.extend([name,'-',kind])
        else:kept.extend([name,'-',kind])
    assert len(moved)==3*len(constants)
    objects[:]=kept
    predicate_index=next(i for i,x in enumerate(domain) if isinstance(x,list) and x[0]==':predicates')
    predicates=domain[predicate_index]
    assert not any(p[0]=='cancontain' for p in predicates[1:])
    # This static relation occurs in all 506 problems but is undeclared and unused
    # by upstream actions. Declaring it retains the facts without adding a rule.
    predicates.append(['cancontain','?r','-','rtype','?o','-','otype'])
    domain.insert(predicate_index,[':constants',*moved])
    def project(tree):
        if not isinstance(tree,list):return tree
        if tree and tree[0] in [':functions',':metric']:return None
        if tree and tree[0]=='increase':
            assert tree[1]==['totalcost']
            return None
        if tree and tree[0]=='=' and isinstance(tree[1],list) and tree[1][0] in ['distance','totalcost']:
            return None
        return [y for x in tree if (y:=project(x)) is not None]
    # Reject any future source whose costs affect applicability or the goal.
    for action in [x for x in domain if isinstance(x,list) and x[0]==':action']:
        condition=serialize(action[action.index(':precondition')+1])
        assert not re.search(r'\b(totalcost|distance)\b',condition)
    goal=next(x for x in problem if isinstance(x,list) and x[0]==':goal')
    assert not re.search(r'\b(totalcost|distance)\b',serialize(goal))
    normalized_domain=serialize(project(domain));normalized_problem=serialize(project(problem))
    assert not re.search(r'\b(totalcost|distance)\b',normalized_domain+normalized_problem)
    # UP 1.3's quantified-effect reader creates variables in its default
    # environment; use that same environment and recycle CPU worker processes.
    env=get_environment()
    parsed=PDDLReader(environment=env).parse_problem_string(normalized_domain,normalized_problem)
    return parsed,{'domain_sha256':sha(domain_path),'problem_sha256':packet['pddl_states'][0]['sha256'],
        'normalized_domain_sha256':hashlib.sha256(normalized_domain.encode()).hexdigest(),
        'normalized_problem_sha256':hashlib.sha256(normalized_problem.encode()).hexdigest(),
        'normalization':['Move five existing problem objects to domain constants',
                         'Declare upstream unused static canContain relation',
                         'Deduplicate identical typed object declarations; reject conflicting types',
                         'Project cost-only functions, assignments, effects and metric; preserve all Boolean conditions/effects/goals'],
        'duplicate_object_declarations':duplicates,
        'cost_optimality_scored':False}

def alfred_goal_satisfied(problem,state):
    """Short-circuit logic; delegate leaf evaluation and typing to UP.

    UP's DAG walk visits nested existentials before testing sibling type guards.
    Lazy conjunctions avoid enumerating irrelevant object pairs in ALFRED goals.
    """
    from itertools import product
    from unified_planning.model.walkers.state_evaluator import StateEvaluator
    evaluator=StateEvaluator(problem)
    def visit(node,bindings):
        if node.is_and():return all(visit(x,bindings) for x in node.args)
        if node.is_or():return any(visit(x,bindings) for x in node.args)
        if node.is_not():return not visit(node.arg(0),bindings)
        if node.is_exists() or node.is_forall():
            variables=node.variables()
            values=(visit(node.arg(0),{**bindings,**dict(zip(variables,objects))})
                    for objects in product(*(problem.objects(v.type) for v in variables)))
            return any(values) if node.is_exists() else all(values)
        return evaluator.evaluate(node,state,bindings).is_true()
    return all(visit(goal,{}) for goal in problem.goals)

def replay_alfred_expert(packet, domain_path, high_plan=None, binding_choices=None):
    """Resolve source coordinates, then let UP check every grounded transition."""
    import math
    problem,binding=alfred_boolean_problem(packet,domain_path)
    doc=read_binding(packet['trajectory']['path']);agent=problem.object('agent1')
    typed=collections.defaultdict(list)
    for obj in problem.all_objects:typed[obj.type.name].append(obj)
    def decode(text):
        for old,new in [('_minus_','-'),('_bar_','|'),('_plus_','+'),('_dot_','.'),('_comma_',',')]:
            text=text.replace(old,new)
        return text
    def coordinate(value,kind):
        found=[]
        for obj in typed[kind]:
            name,separator,tail=decode(obj.name).partition('_')
            if not separator or name!=value[0].lower():continue
            numbers=tail.split(',')
            if len(numbers)==len(value[1]) and all(math.isclose(float(a),b,abs_tol=1e-7,rel_tol=0) for a,b in zip(numbers,value[1])):
                found.append(obj)
        if len(found)!=1:raise ValueError(f'Coordinate binding {kind}/{value[0]} has {len(found)} matches')
        return found[0]
    steps=[];binding_choices={} if binding_choices is None else binding_choices
    with problem.environment.factory.SequentialSimulator(problem=problem) as sim:
        state=sim.get_initial_state()
        def truth(name,*args):return state.get_value(problem.fluent(name)(*args)).is_true()
        for high_index,entry in enumerate(doc['plan']['high_pddl'] if high_plan is None else high_plan):
            action=entry['planner_action'];name=action['action']
            if name=='End':
                goal=alfred_goal_satisfied(problem,state)
                steps.append({'high_index':high_index,'source_action':name,'goal':goal})
                if not goal:return {'complete':False,'reason':'expert_end_before_goal','steps':steps,'binding':binding}
                continue
            locations=[l for l in typed['location'] if truth('atlocation',agent,l)]
            if len(locations)!=1:raise ValueError('Agent does not have one current location')
            loc=locations[0];candidates=[]
            def candidate(label,*args):candidates.append(problem.action(label)(*args))
            if name=='GotoLocation':
                dest=next(o for o in typed['location'] if decode(o.name)==action['location'].lower())
                candidate('gotolocation',agent,loc,dest)
            elif name=='PickupObject':
                obj=coordinate(action['coordinateObjectId'],'object')
                if 'coordinateReceptacleObjectId' in action:
                    rec=coordinate(action['coordinateReceptacleObjectId'],'receptacle')
                    candidate('pickupobjectinreceptacle1',agent,loc,obj,rec)
                else:candidate('pickupobjectnoreceptacle',agent,loc,obj)
            elif name=='PutObject':
                obj=coordinate(action['coordinateObjectId'],'object')
                types=[t for t in typed['otype'] if truth('objecttype',obj,t)]
                assert len(types)==1
                target=action['coordinateReceptacleObjectId']
                try:rec=coordinate(target,'receptacle')
                except ValueError:rec=None
                if rec is not None:
                    candidate('putobjectinreceptacle1',agent,loc,types[0],obj,rec)
                    candidate('putreceptacleobjectinreceptacle1',agent,loc,types[0],obj,rec)
                else:
                    outer=coordinate(target,'object')
                    for rec in typed['receptacle']:
                        if truth('inreceptacle',outer,rec):
                            candidate('putobjectinreceptacleobject1',agent,loc,types[0],obj,outer,rec)
            elif name in ['CleanObject','HeatObject','CoolObject']:
                rec=coordinate(action['coordinateReceptacleObjectId'],'receptacle')
                held=[o for o in typed['object'] if truth('holds',agent,o)]
                if name=='CleanObject':held=[coordinate(action['coordinateObjectId'],'object')]
                for obj in held:candidate(name.lower(),agent,loc,rec,obj)
            elif name=='ToggleObject':
                candidate('toggleobject',agent,loc,coordinate(action['coordinateObjectId'],'object'))
            elif name=='SliceObject':
                obj=coordinate(action['coordinateObjectId'],'object')
                for knife in typed['object']:
                    if truth('holds',agent,knife):candidate('sliceobject',agent,loc,obj,knife)
            else:raise ValueError(f'Unsupported expert action {name}')
            valid=[a for a in candidates if sim.is_applicable(state,a)]
            step={'high_index':high_index,'source_action':name,'applicable_candidates':[str(a) for a in valid]}
            steps.append(step)
            if high_index in binding_choices:
                selected=binding_choices[high_index]
                chosen=[a for a in valid if str(a)==selected]
                if len(chosen)!=1:raise ValueError(f'Inapplicable supplied source binding at step {high_index}')
                step['selected_binding']=selected
                valid=chosen
            if len(valid)!=1:
                step['considered_candidates']=[str(a) for a in candidates]
                return {'complete':False,'reason':'nonunique_or_inapplicable_expert_transition','steps':steps,'binding':binding}
            state=sim.apply(state,valid[0]);assert state is not None
        goal=alfred_goal_satisfied(problem,state)
        return {'complete':goal,'reason':'goal_reached' if goal else 'final_goal_unsatisfied',
                'steps':steps,'binding':binding,'scope':'Symbolic source-domain replay; no AI2-THOR physics replay'}

def replay_alfred_alternatives(packet,domain_path,high_plan=None):
    """Enumerate every applicable grounding; retain successful and failed branches.

    Multiple containers can contain the same source object. This represents an
    ambiguity in grounding a fixed macro plan, not an alternative action order.
    No branch is selected from model performance, and no physics is inferred.
    """
    pending=[{}];branches=[];forks=[]
    while pending:
        choices=pending.pop()
        result=replay_alfred_expert(packet,domain_path,high_plan,binding_choices=choices)
        if result.get('reason')=='nonunique_or_inapplicable_expert_transition':
            step=result['steps'][-1];candidates=step['applicable_candidates']
            if len(candidates)>1:
                index=step['high_index']
                assert index not in choices
                forks.append({'prefix_choices':choices,'high_index':index,'candidates':candidates})
                pending.extend({**choices,index:action} for action in reversed(candidates))
                continue
        branches.append({'binding_choices':choices,**result})
    successes=sum(b['complete'] for b in branches)
    return {'complete':successes>0,'all_groundings_reach_goal':successes==len(branches),
        'reason':'goal_reached_all_groundings' if successes==len(branches) else
                 'goal_reached_some_groundings' if successes else 'no_grounding_reaches_goal',
        'grounding_branches':len(branches),'successful_groundings':successes,
        'forks':forks,'branches':branches,
        'scope':'All source-coordinate groundings of the recorded macro plan; no physical execution or alternative-order certification'}

def _alfred_replay_task(packet):
    root=D/'alfred_native_plan_sources_v1'
    sys.path.insert(0,str(root/'python_deps'))
    try:
        result=replay_alfred_alternatives(packet,root/'planner_reference/PutTaskExtended_domain.pddl')
    except Exception as exc:
        result={'complete':False,'reason':'source_or_replay_error','error':f'{type(exc).__name__}: {exc}'}
    return {'task_id':packet['task_id'],'task_type':packet['task_type'],'split':packet['split'],
            'trajectory_sha256':packet['trajectory']['sha256'],**result}

def alfred_grounded_replay(problem, actions):
    """Score a supplied grounded plan, with no expert-order comparison."""
    with problem.environment.factory.SequentialSimulator(problem=problem) as sim:
        state=sim.get_initial_state()
        for index,text in enumerate(actions):
            match=re.fullmatch(r'(\w+)\(([^()]*)\)',text)
            if match is None:raise ValueError('Malformed grounded source action')
            action=problem.action(match[1])(*(problem.object(x.strip()) for x in match[2].split(',')))
            if not sim.is_applicable(state,action):
                return {'valid':False,'reason':'inapplicable_transition','first_failure':index,'steps':len(actions)}
            state=sim.apply(state,action)
            assert state is not None
        goal=alfred_goal_satisfied(problem,state)
        return {'valid':goal,'reason':'goal_reached' if goal else 'final_goal_unsatisfied',
                'first_failure':None,'steps':len(actions)}

def alfred_plan_prompt(problem,packet,actions):
    """Lossless Boolean state/rule view with compact, type-preserving identifiers.

    Only canContain is omitted: it is absent from every operator and goal. The
    complete typed universe is retained because quantified rules can inspect it.
    The source instruction and first frame are shared by both labels in a pair.
    """
    aliases={};counters=collections.Counter();typed=collections.defaultdict(list)
    for obj in sorted(problem.all_objects,key=lambda o:o.name):
        kind=obj.type.name
        if kind in ['object','receptacle','location']:
            label='loc' if kind=='location' else re.split('_',obj.name,1)[0]
            prefix=('o' if kind=='object' else 'r' if kind=='receptacle' else 'l')+'_'+label
            counters[prefix]+=1;alias=f'{prefix}_{counters[prefix]}'
        else:alias=obj.name
        assert alias not in aliases.values()
        aliases[obj.name]=alias;typed[kind].append(alias)
    def rename(text):
        return re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',lambda m:aliases.get(m[0],m[0]),text)
    facts=sorted(rename(str(f)) for f,v in problem.explicit_initial_values.items()
                 if v.is_true() and f.fluent().name!='cancontain')
    names=sorted({s.partition('(')[0] for s in actions})
    rules='\n'.join(rename(str(problem.action(name))) for name in names)
    goal=rename(' and '.join(str(g) for g in problem.goals))
    universe='\n'.join(kind+': '+', '.join(values) for kind,values in sorted(typed.items()))
    plan='\n'.join(f'{i+1}. {rename(a)}' for i,a in enumerate(actions))
    prompt=('<image>\nCheck a household robot plan under the supplied symbolic rules. '
        'A plan is VALID exactly when every listed action is applicable in order and the final state satisfies the goal. '
        'Use the supplied complete symbolic state even for objects outside the initial camera view. '
        'All unlisted Boolean facts are false. Effects update only their named facts; all other facts persist. '
        'Quantifiers range over the typed universe. Agent, location, and receptacle are subtypes of object. '
        'The rules define high-level actions, not a physical robot rollout. '
        'Return exactly VALID or INVALID, with no explanation.\n'
        'Instruction: '+packet['goals'][0]+'\nTyped universe:\n'+universe+
        '\nInitially true facts:\n'+'\n'.join(facts)+'\nGoal:\n'+goal+
        '\nAction rules:\n'+rules+'\nCandidate plan:\n'+plan)
    return prompt,{'object_aliases':aliases,'initial_true_facts':len(facts),'rules':names}

def _alfred_plan_pair_task(packet):
    root=D/'alfred_native_plan_sources_v1';sys.path.insert(0,str(root/'python_deps'))
    for item in [packet['trajectory'],packet['pddl_states'][0],packet['frame']]:
        assert sha(item['path'])==item['sha256'],'ALFRED source bytes changed'
    problem,binding=alfred_boolean_problem(packet,root/'planner_reference/PutTaskExtended_domain.pddl')
    source=read_binding(root/'symbolic_replay_v2/records'/f"{packet['task_id']}.json")
    assert source['complete'] and source['all_groundings_reach_goal']
    assert source['trajectory_sha256']==packet['trajectory']['sha256']
    branch=source['branches'][0]
    positive=[step.get('selected_binding',step.get('applicable_candidates',[None])[0])
              for step in branch['steps'] if step['source_action']!='End']
    assert all(positive)
    pos_score=alfred_grounded_replay(problem,positive);assert pos_score['valid']
    swaps=[(i,j) for i in range(len(positive)) for j in range(i+1,len(positive)) if positive[i]!=positive[j]]
    swaps.sort(key=lambda ij:hashlib.sha256(f"anvil-plan-swap-v1|{packet['task_id']}|{ij}".encode()).digest())
    attempts=[];negative=None
    for i,j in swaps:
        candidate=positive.copy();candidate[i],candidate[j]=candidate[j],candidate[i]
        score=alfred_grounded_replay(problem,candidate)
        attempts.append({'swap':[i,j],**score})
        if not score['valid']:negative=candidate;break
    if negative is None:raise ValueError(f"No invalid same-multiset permutation for {packet['task_id']}")
    assert collections.Counter(positive)==collections.Counter(negative)
    examples=[];views=[]
    for label,plan in [('VALID',positive),('INVALID',negative)]:
        prompt,view=alfred_plan_prompt(problem,packet,plan);views.append(view)
        # IDs and dataset order conceal label and source split from the request.
        sample_id=hashlib.sha256((packet['task_id']+'|'+label+'|plan-judgment-v1').encode()).hexdigest()
        examples.append({'sample_id':sample_id,'task_id':packet['task_id'],'scene':packet['scene'],
            'task_type':packet['task_type'],'source_split':packet['split'],
            'images':[packet['frame']['path']],
            'messages':[{'role':'user','content':prompt},{'role':'assistant','content':label}]})
    assert views[0]==views[1]
    assert examples[0]['messages'][0]['content'].partition('\nCandidate plan:\n')[0]==examples[1]['messages'][0]['content'].partition('\nCandidate plan:\n')[0]
    return {'task_id':packet['task_id'],'scene':packet['scene'],'task_type':packet['task_type'],
        'source_split':packet['split'],'binding':binding,'source_replay_sha256':sha(root/'symbolic_replay_v2/records'/f"{packet['task_id']}.json"),
        'frame_sha256':packet['frame']['sha256'],'positive':positive,'negative':negative,
        'positive_score':pos_score,'negative_attempts':attempts,'prompt_view':views[0],'examples':examples}

def prepare_alfred_plans():
    """Source-certified matched judgments; no generated annotation or model calls."""
    import inspect
    import multiprocessing as mp
    root=D/'alfred_native_plan_sources_v1';target=root/'plan_judgment_v1';target.mkdir(exist_ok=True)
    packets=rows(root/'expert_sources.jsonl')
    code='\n'.join(inspect.getsource(f) for f in [alfred_boolean_problem,alfred_goal_satisfied,
        alfred_grounded_replay,alfred_plan_prompt,_alfred_plan_pair_task])
    binding={'schema':'anvil-alfred-state-conditioned-plan-judgment-v1','expert_sources_sha256':sha(root/'expert_sources.jsonl'),
        'reference_replay_binding_sha256':sha(root/'symbolic_replay_v2/binding.json'),
        'source_domain_sha256':sha(root/'planner_reference/PutTaskExtended_domain.pddl'),
        'preparation_code_sha256':hashlib.sha256(code.encode()).hexdigest(),
        'negative_rule':'First invalid two-position swap in source-ID seeded hash order. Same length and action multiset; no model output is consulted.',
        'positive_rule':'First saved source grounding; all source grounding branches already validated.',
        'scope':'Action-order/goal verification given full Boolean state and action rules; not standard ALFRED robot success, visual-only planning, or eight-form native Action score.',
        'generation':{'batch':4,'max_new_tokens':16,'max_length':12288,'temperature':0,'seed':17,'image_max_token_num':1536},
        'model_calls_during_construction':0,'annotation_calls':0}
    if (target/'binding.json').exists():assert read_binding(target/'binding.json')==binding
    else:save(target/'binding.json',binding)
    records=target/'records';records.mkdir(exist_ok=True)
    with (target/'prepare.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        missing=[p for p in packets if not (records/f"{p['task_id']}.json").exists()]
        with mp.get_context('spawn').Pool(4,maxtasksperchild=16) as pool:
            for result in pool.imap_unordered(_alfred_plan_pair_task,missing,chunksize=1):
                save(records/f"{result['task_id']}.json",result)
                print('ALFRED_PLAN_PAIR',result['task_id'],flush=True)
        results=[read_binding(records/f"{p['task_id']}.json") for p in packets]
        allrows=sorted([e for result in results for e in result['examples']],key=lambda r:r['sample_id'])
        assert len({r['sample_id'] for r in allrows})==2*len(packets)
        source=target/'evaluation.jsonl'
        commit_bytes(source,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows).encode())
        save(target/'manifest.json',{'schema':binding['schema'],'binding_sha256':sha(target/'binding.json'),
            'source':str(source),'source_sha256':sha(source),'rows':len(allrows),'trajectories':len(packets),
            'scenes':len({str(r['scene']) for r in allrows}),
            'task_types':dict(collections.Counter(r['task_type'] for r in allrows)),
            'labels':dict(collections.Counter(r['messages'][-1]['content'] for r in allrows)),
            'negative_reasons':dict(collections.Counter(r['negative_attempts'][-1]['reason'] for r in results)),
            'positive_steps':sum(len(r['positive']) for r in results),'generation':binding['generation'],'scope':binding['scope']})

def alfred_plan_jobs():
    return [{'id':'base','model':str(MODEL),'adapter':None},
        {'id':'p5','model':str(MODEL),'adapter':str(D/'p5_retention_clean_run_seed17_v2/selected_adapter')},
        {'id':'A_initial','model':str(MODEL),'adapter':str(D/'native_teacher_pilot_v2/A/selected_adapter')},
        {'id':'joint_initial','model':str(MODEL),'adapter':str(D/'native_teacher_pilot_v2/joint/selected_adapter')}]

def parse_alfred_verdict(response, finish_reason):
    """Accept one exact label after the template's optional empty think prefix.

    Swift exposes the non-thinking chat-template prefix in response text. Strip
    that empty wrapper only; reasoning, extra answers and capped output fail.
    """
    answer=re.sub(r'^<think>\s*</think>\s*','',response.strip(),count=1)
    return answer if finish_reason=='stop' and answer in ['VALID','INVALID'] else None

def alfred_plan_worker(jobid):
    from anvil import _native_infer_batches, _INFERENCE_DISTRIBUTED_ENV
    from swift.pipelines.infer.infer import SwiftInfer
    target=D/'alfred_native_plan_sources_v1/plan_judgment_v1';manifest=read_binding(target/'manifest.json')
    source=Path(manifest['source']);assert sha(source)==manifest['source_sha256']
    job=next(j for j in alfred_plan_jobs() if j['id']==jobid)
    folder=target/jobid;folder.mkdir(exist_ok=True)
    with (folder/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=Path(job['model']);adapter=Path(job['adapter']) if job['adapter'] else None
        if jobid=='p5':
            lineage=read_binding(D/'p5_retention_clean_run_seed17_v2/completed_lineage.json')
            for key,name in [('selected_adapter_config','adapter_config.json'),('selected_adapter_weights','adapter_model.safetensors')]:
                assert sha(adapter/name)==lineage['files'][key]['sha256'],'P5 adapter differs from its selected checkpoint'
        model_files=sorted(p for p in model.iterdir() if p.is_file() and p.suffix in ['.safetensors','.json','.jinja'])
        assert any(p.suffix=='.safetensors' for p in model_files)
        adapter_files=[] if adapter is None else [adapter/'adapter_model.safetensors',adapter/'adapter_config.json']
        binding={'source_sha256':sha(source),'model':str(model),'adapter':str(adapter) if adapter else None,
            'model_files':{p.name:sha(p) for p in model_files},'adapter_files':{p.name:sha(p) for p in adapter_files},
            'generation':manifest['generation']}
        if (folder/'binding.json').exists():assert read_binding(folder/'binding.json')==binding
        else:save(folder/'binding.json',binding)
        allrows=rows(source);chunks=folder/'chunks';chunks.mkdir(exist_ok=True);spool=prediction_spool(folder,binding)
        missing=[]
        for i in range(0,len(allrows),4):
            batch=allrows[i:i+4];p=chunks/f'{i:06d}.jsonl';partial=spool/f'{i:06d}.jsonl'
            if p.exists():assert [r['sample_id'] for r in rows(p)]==[r['sample_id'] for r in batch]
            elif spool_batch_ready(partial,batch):commit_prediction(partial,p,batch)
            else:missing.append((i,batch,p))
        if missing:
            for key in _INFERENCE_DISTRIBUTED_ENV:os.environ.pop(key,None)
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
            lengths=[len(tokenizer.encode(r['messages'][0]['content'],add_special_tokens=False)) for r in allrows]
            # Conservative allowance for the initial image, chat wrappers and output.
            assert max(lengths)+1536+128+16<=12288,'Source-state prompts exceed the frozen context budget'
            save(folder/'input_budget.json',{'text_tokens_max':max(lengths),'image_allowance':1536,
                'wrapper_allowance':128,'max_new_tokens':16,'max_length':12288,'all_rows_checked':len(allrows)})
            command=['--model',str(model),'--infer_backend','transformers','--max_batch_size','4',
                '--max_new_tokens','16','--max_length','12288','--temperature','0','--seed','17',
                '--enable_thinking','false','--stream','false','--load_args','false','--torch_dtype','bfloat16',
                '--truncation_strategy','delete']
            if adapter:command+=['--adapters',str(adapter)]
            engine=SwiftInfer(command)
            actual={k:getattr(engine.args,k) for k in ['model','adapters','max_new_tokens','max_length','enable_thinking','truncation_strategy']}
            assert actual['max_new_tokens']==16 and actual['max_length']==12288
            assert engine.template.truncation_strategy=='raise','Never silently truncate a source-state prompt'
            assert actual['model']==str(model) and actual['adapters']==([] if adapter is None else [str(adapter)])
            save(folder/'inference_args.json',{'command':command,'actual':actual})
            for i,batch,p in missing:
                partial=spool/f'{i:06d}.jsonl';_native_infer_batches(engine,batch,partial)
                commit_prediction(partial,p,batch)
                save(folder/'progress.json',{'completed':i+len(batch),'total':len(allrows),'updated':time.time()})
        prediction=folder/'predictions.jsonl'
        commit_bytes(prediction,b''.join(p.read_bytes() for p in sorted(chunks.glob('*.jsonl'))))
        save(folder/'done.json',{'prediction_sha256':sha(prediction),'rows':len(allrows),'time':time.time()})
    collect_alfred_plans()

def collect_alfred_plans():
    import numpy as np
    target=D/'alfred_native_plan_sources_v1/plan_judgment_v1'
    if not (target/'manifest.json').exists():return
    manifest=read_binding(target/'manifest.json');source=Path(manifest['source'])
    assert sha(source)==manifest['source_sha256'];allrows=rows(source);result={}
    scenes=sorted({str(r['scene']) for r in allrows});names={'base':'Qwen3.5-9B','p5':'Anvil P5','A_initial':'Action specialist','joint_initial':'Joint SFT'}
    for job in alfred_plan_jobs():
        folder=target/job['id']
        if not (folder/'done.json').exists():result[job['id']]={'state':'pending'};continue
        p=folder/'predictions.jsonl';assert sha(p)==read_binding(folder/'done.json')['prediction_sha256']
        pred=rows(p);assert [r['sample_id'] for r in pred]==[r['sample_id'] for r in allrows]
        decisions=[]
        for r,v in zip(allrows,pred):
            label=r['messages'][-1]['content']
            parsed=parse_alfred_verdict(v['response'],v['finish_reason'])
            decisions.append({'sample_id':r['sample_id'],'scene':str(r['scene']),'label':label,
                'prediction':parsed,'correct':parsed==label,'finish_reason':v['finish_reason']})
        label_scores={label:sum(x['correct'] for x in decisions if x['label']==label)/sum(x['label']==label for x in decisions)
                      for label in ['VALID','INVALID']}
        successes=np.array([sum(x['correct'] for x in decisions if x['scene']==s) for s in scenes])
        sizes=np.array([sum(x['scene']==s for x in decisions) for s in scenes]);rng=np.random.default_rng(20260915)
        indices=rng.integers(0,len(scenes),size=(5000,len(scenes)))
        samples=successes[indices].sum(axis=1)/sizes[indices].sum(axis=1)
        score={'state':'complete','accuracy':sum(x['correct'] for x in decisions)/len(decisions),
            'label_accuracy':label_scores,'scene_cluster_ci95':np.quantile(samples,[.025,.975]).tolist(),
            'unparsed_or_truncated':sum(x['prediction'] is None for x in decisions),'rows':len(decisions),
            'prompt_tokens_max':max(v['prompt_tokens'] for v in pred),'prediction_sha256':sha(p)}
        save(folder/'decisions.json',decisions);save(folder/'result.json',score);result[job['id']]=score
    save(target/'acceptance.json',{'manifest_sha256':sha(target/'manifest.json'),'jobs':result,'scope':manifest['scope'],
        'scoring':'Exact VALID/INVALID after at most one leading empty think wrapper; finish_reason must be stop',
        'scorer_revision':'empty-template-prefix-v2'})
    if not all(r['state']=='complete' for r in result.values()):return
    tex=[r'\begin{table}[!htbp]',r'\centering\small',
        r'\caption{\textbf{State-conditioned action-plan verification on ALFRED sources.} '+str(manifest['rows'])+
        r' balanced judgments over '+str(manifest['trajectories'])+r' trajectories and '+str(manifest['scenes'])+
        r' scenes. Each source plan is paired with an invalid permutation of exactly the same actions. Models receive the initial frame, complete Boolean state, goal, and source action rules; accuracy is reported in \%. Intervals resample scenes. This is a constructed verification task, not the standard ALFRED execution benchmark.}',
        r'\label{tab:alfred-plan-verification}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',
        r'Model & Valid & Invalid & Overall & 95\% CI \\',r'\midrule']
    for job in alfred_plan_jobs():
        r=result[job['id']];lo,hi=r['scene_cluster_ci95']
        tex.append(names[job['id']]+' & '+f"{r['label_accuracy']['VALID']*100:.2f} & {r['label_accuracy']['INVALID']*100:.2f} & {r['accuracy']*100:.2f} & [{lo*100:.2f}, {hi*100:.2f}]"+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    commit_bytes(A/'overleaf/source/tables/alfred_plan_verification.tex',('\n'.join(tex)+'\n').encode())

def alfred_plan_lane(gpu):
    target=D/'alfred_native_plan_sources_v1/plan_judgment_v1'
    for job in alfred_plan_jobs()[gpu:gpu+1]:
        folder=target/job['id'];folder.mkdir(exist_ok=True)
        if (folder/'done.json').exists():continue
        while True:
            used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used<=512:break
            time.sleep(30)
        with (folder/'dispatch.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (folder/'done.json').exists():continue
            if (folder/'failure.json').exists():
                (folder/'failure.json').replace(folder/f'failure_history_{time.time_ns()}.json')
            with (folder/'run.log').open('a') as log:
                code=subprocess.run([sys.executable,__file__,'alfred-plan-worker','--job',job['id']],
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT).returncode
            if code:
                save(folder/'failure.json',{'exit_code':code,'time':time.time()});raise RuntimeError('Plan verification cell failed; completed batches retained')

def run_alfred_plans():
    """Finish source preparation on CPU, then follow the current GPU queue."""
    target=D/'alfred_native_plan_sources_v1/plan_judgment_v1';target.mkdir(exist_ok=True)
    with (target/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare_alfred_plans()
        primary=D/'native_completion_20260915'
        # Preserve the existing primary/cap-repair priority and avoid GPU races.
        while not (primary/'published.json').exists():
            save(target/'queue_state.json',{'state':'waiting_for_native_publication','time':time.time()})
            time.sleep(30)
        processes=[subprocess.Popen([sys.executable,__file__,'alfred-plan-lane','--gpu',str(gpu)]) for gpu in range(4)]
        codes=[p.wait() for p in processes]
        collect_alfred_plans()
        if any(codes):raise RuntimeError('ALFRED plan verification incomplete; resume same command')
        subprocess.run([sys.executable,str(A/'engineering/experiments/submission_closeout.py'),'publish'],check=True)
        save(target/'published.json',{'pdf_sha256':sha(A/'paper/Anvil_Submission_Draft.pdf'),
            'zip_sha256':sha(A/'overleaf/Anvil_Submission_Draft.zip'),'time':time.time()})

def verify_alfred_sources():
    """A resumable CPU queue of expert-reference checks, with no model calls."""
    import inspect
    import multiprocessing
    prepare_alfred_sources()
    root=D/'alfred_native_plan_sources_v1';sys.path.insert(0,str(root/'python_deps'))
    import unified_planning
    assert unified_planning.__version__=='1.3.0'
    folder=root/'symbolic_replay_v2';folder.mkdir(exist_ok=True)
    code='\n'.join(inspect.getsource(f) for f in [alfred_boolean_problem,alfred_goal_satisfied,replay_alfred_expert,replay_alfred_alternatives])
    binding={'sources_sha256':sha(root/'expert_sources.jsonl'),
        'domain_sha256':sha(root/'planner_reference/PutTaskExtended_domain.pddl'),
        'replay_code_sha256':hashlib.sha256(code.encode()).hexdigest(),
        'unified_planning':unified_planning.__version__,'coordinate_absolute_tolerance':1e-7,
        'grounding_policy':'Enumerate all applicable source-coordinate bindings; retain failed branches',
        'scope':'Boolean source-domain expert replay; no physics, plan optimality, alternative-order coverage or model scores'}
    with (folder/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'binding.json').exists():assert read_binding(folder/'binding.json')==binding,'Expert replay binding changed'
        else:save(folder/'binding.json',binding)
        packets=rows(root/'expert_sources.jsonl');completed=folder/'records';completed.mkdir(exist_ok=True)
        missing=[r for r in packets if not (completed/(r['task_id']+'.json')).exists()]
        done=len(packets)-len(missing)
        if missing:
            with multiprocessing.get_context('spawn').Pool(4,maxtasksperchild=16) as pool:
                for result in pool.imap_unordered(_alfred_replay_task,missing,chunksize=1):
                    save(completed/(result['task_id']+'.json'),result);done+=1
                    save(folder/'progress.json',{'completed':done,'total':len(packets),'updated':time.time()})
                    if done%25==0:print('ALFRED symbolic replay',done,'/',len(packets),flush=True)
        results=[read_binding(completed/(r['task_id']+'.json')) for r in packets]
        for packet,result in zip(packets,results):assert packet['trajectory']['sha256']==result['trajectory_sha256']
        unchanged=0
        for result in results:
            previous=root/'symbolic_replay_v1/records'/(result['task_id']+'.json')
            if previous.exists():
                old=read_binding(previous)
                if old['complete']:
                    assert result['complete'] and result['grounding_branches']==1
                    assert old['steps']==result['branches'][0]['steps']
                    unchanged+=1
        counts=collections.defaultdict(lambda:collections.Counter())
        for result in results:
            counts[result['task_type']]['records']+=1
            counts[result['task_type']]['goal_reached']+=bool(result['complete'])
        report={'binding':binding,'records':len(results),'goal_reached':sum(r['complete'] for r in results),
            'per_task_type':dict(counts),'reasons':dict(collections.Counter(r['reason'] for r in results)),
            'all_groundings_reach_goal':sum(r.get('all_groundings_reach_goal',False) for r in results),
            'grounding_branches':sum(r.get('grounding_branches',0) for r in results),
            'source_ambiguities':sum(bool(r.get('forks')) for r in results),
            'unchanged_successful_v1_records':unchanged,
            'records_with_duplicate_declarations':sum(any(b.get('binding',{}).get('duplicate_object_declarations') for b in r.get('branches',[])) for r in results),
            'model_calls':0,'new_annotation_calls':0,'benchmark_ready':False,
            'remaining':['Independent alternatives and forbidden-transition scoring',
                         'Freeze model-visible observation and native Action answer protocol',
                         'Run model evaluation; source expert success is not a model score']}
        save(folder/'acceptance.json',report)
        manifest=read_binding(root/'manifest.json')
        manifest['expert_replay']={'path':str(folder/'acceptance.json'),'sha256':sha(folder/'acceptance.json'),
            'records':len(results),'goal_reached':report['goal_reached'],'scope':binding['scope']}
        save(root/'manifest.json',manifest)
        print(json.dumps(report),flush=True)

def prepare_alfred_sources():
    """Extract existing expert evidence without inventing plan checklists."""
    root=D/'alfred_native_plan_sources_v1';root.mkdir(parents=True,exist_ok=True)
    archive=Path('/path/to/external_data/datasets/full_2.1.0.7z')
    tar='/path/to/workspace/miniconda3/bin/bsdtar'
    pattern=re.compile(r'^full_2[.]1[.]0/(valid_seen|valid_unseen)/[^/]+/[^/]+/(traj_data[.]json|problem_[^/]+[.]pddl)$')
    with (root/'prepare.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'manifest.json').exists():
            bind_alfred_planner_sources(root)
            audit_alfred_metadata(root)
            print('ALFRED source extraction already complete',flush=True);return
        stat=archive.stat();identity={'path':str(archive),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        catalog=root/'archive_members.json'
        if catalog.exists():
            saved=read_binding(catalog);assert saved['archive']==identity
            members=saved['members']
        else:
            members=[]
            proc=subprocess.Popen([tar,'-tf',str(archive)],stdout=subprocess.PIPE,text=True)
            for line in proc.stdout:
                name=line.rstrip('\n')
                if pattern.fullmatch(name):members.append(name)
            assert proc.wait()==0,'ALFRED archive listing failed'
            members=sorted(set(members))
            assert members and any(n.endswith('traj_data.json') for n in members)
            save(catalog,{'archive':identity,'members':members})
        assert all(pattern.fullmatch(n) for n in members)
        raw=root/'source';raw.mkdir(exist_ok=True)

        def extract(names,label):
            missing=[n for n in names if not (raw/n).is_file()]
            if not missing:return
            targets=root/(label+'_members.txt');targets.write_text('\n'.join(missing)+'\n')
            attempt=Path(tempfile.mkdtemp(prefix=label+'-',dir=root))
            save(root/'progress.json',{'phase':label,'requested_files':len(missing),'started':time.time()})
            print('ALFRED',label,'extracting',len(missing),'files',flush=True)
            # -q stops after every exact requested member has been found. Without
            # it, bsdtar keeps scanning the remaining 109 GB archive after a hit.
            subprocess.run([tar,'-x','-q','-m','-f',str(archive),'-C',str(attempt),'-T',str(targets)],check=True)
            for n in missing:
                src=attempt/n;assert src.is_file() and not src.is_symlink(),n
                dest=raw/n;dest.parent.mkdir(parents=True,exist_ok=True);src.replace(dest)
            for directory in sorted(attempt.rglob('*'),key=lambda p:len(p.parts),reverse=True):
                if directory.is_dir():directory.rmdir()
            attempt.rmdir()

        extract(members,'metadata')
        records=[];frames=[]
        for name in members:
            if not name.endswith('traj_data.json'):continue
            src=raw/name;doc=read_binding(src);parts=Path(name).parts
            assert doc['task_id']==parts[-2]
            assert doc['images'] and doc['plan']['high_pddl']
            frame=Path(name).parent/'raw_images'/Path(doc['images'][0]['image_name']).with_suffix('.jpg').name
            frames.append(str(frame))
            states=[n for n in members if Path(n).parent==Path(name).parent and n.endswith('.pddl')]
            for state in states:
                value=(raw/state).read_text();assert '(:init' in value and '(:goal' in value,state
            records.append({'task_id':doc['task_id'],'task_type':doc['task_type'],'split':parts[1],
                'scene':doc['scene']['floor_plan'],
                'goals':[ann['task_desc'] for ann in doc['turk_annotations']['anns']],
                'expert_high_actions':[p['discrete_action'] for p in doc['plan']['high_pddl']],
                'low_action_count':len(doc['plan']['low_actions']),
                'trajectory':{'path':str(src),'sha256':sha(src)},
                'pddl_states':[{'path':str(raw/n),'sha256':sha(raw/n)} for n in states],
                'frame':{'path':str(raw/frame),'metadata_entry':doc['images'][0]},
                'independent_action_checklist_complete':False})
        assert len({r['task_id'] for r in records})==len(records),'Repeated canonical task identity'
        assert all(re.fullmatch(r'full_2[.]1[.]0/(valid_seen|valid_unseen)/[^/]+/[^/]+/raw_images/[0-9]+[.]jpg',n) for n in frames)
        extract(sorted(set(frames)),'frames')
        from PIL import Image
        for row in records:
            frame=Path(row['frame']['path']);row['frame']['sha256']=sha(frame)
            with Image.open(frame) as im:im.verify()
        packets=root/'expert_sources.jsonl'
        packets.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        save(root/'manifest.json',{'schema':'anvil-alfred-expert-source-v1','archive':identity,
            'official_documentation':'https://github.com/askforalfred/alfred/blob/master/data/README.md',
            'records':len(records),'splits':dict(collections.Counter(r['split'] for r in records)),
            'task_types':dict(collections.Counter(r['task_type'] for r in records)),
            'expert_sources_sha256':sha(packets),'model_calls':0,'new_annotation_calls':0,
            'benchmark_ready':False,'source_scene_and_task_ids_available':True,
            'remaining':['Verify temporal alignment of source observations and PDDL initial states',
                         'Bind the authoritative planning domain and check expert transitions',
                         'Derive and verify valid alternatives, forbidden transitions and terminal goals',
                         'Define native Action scoring before model evaluation']})
        bind_alfred_planner_sources(root)
        audit_alfred_metadata(root)
        print('ALFRED source extraction complete:',len(records),'expert trajectories',flush=True)

def supervise():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
        processes=[]
        for gpu in range(4):
            processes.append(subprocess.Popen([sys.executable,__file__,'lane','--gpu',str(gpu)]))
        codes=[p.wait() for p in processes]
        collect();save(OUT/'queue_finished.json',{'lane_exit_codes':codes,'finished':time.time()})
        if not any(codes) and os.environ.get('ANVIL_NATIVE_COHORT')!='sensor-mask':finalize()

def finalize():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'publish.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (OUT/'published.json').exists():return
        # Own the final stage before waiting, so a supervisor started with an
        # older script cannot publish while continuation jobs are still pending.
        while not all((OUT/j['id']/'done.json').exists() for j in jobs()):
            if (OUT/'queue_finished.json').exists():
                raise RuntimeError('The queue ended with incomplete cells; saved results remain intact.')
            time.sleep(30)
        processes=[subprocess.Popen([sys.executable,__file__,'repair-lane','--gpu',str(gpu)]) for gpu in range(4)]
        codes=[p.wait() for p in processes]
        if any(codes) or not all((OUT/'cap_repair'/j['id']/'done.json').exists() for j in jobs()):
            raise RuntimeError('Selective cap repair is incomplete; primary predictions remain intact.')
        # Reload the collector in a fresh process after a potentially long run.
        subprocess.run([sys.executable,__file__,'status'],check=True)
        sensor=OUT/'sensor_mask'
        if sensor.exists():
            subprocess.run([sys.executable,__file__,'status'],env={**os.environ,'ANVIL_NATIVE_COHORT':'sensor-mask'},check=True)
        with (OUT/'publish.log').open('a') as log:
            subprocess.run([sys.executable,str(A/'engineering/experiments/submission_closeout.py'),'publish'],stdout=log,stderr=subprocess.STDOUT,check=True)
        save(OUT/'published.json',{'pdf_sha256':sha(A/'paper/Anvil_Submission_Draft.pdf'),
            'zip_sha256':sha(A/'overleaf/Anvil_Submission_Draft.zip'),'time':time.time(),'full_MTD_complete':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['supervise','prepare','prepare-alfred','verify-alfred','prepare-linked','inspect-linked','alfred-plans','alfred-plan-lane','alfred-plan-worker','lane','worker','status','collect','finalize','repair','repair-lane']);p.add_argument('--gpu',type=int,choices=range(8));p.add_argument('--job');a=p.parse_args()
    {'supervise':supervise,'prepare':prepare,'lane':lambda:lane(a.gpu),'worker':lambda:worker(a.job),'status':collect,'finalize':finalize,
     'collect':collect_all,'prepare-alfred':prepare_alfred_sources,'verify-alfred':verify_alfred_sources,
     'inspect-linked':inspect_linked_sources,
     'prepare-linked':lambda:__import__('linked_sensor').prepare(),
     'alfred-plans':run_alfred_plans,'alfred-plan-lane':lambda:alfred_plan_lane(a.gpu),'alfred-plan-worker':lambda:alfred_plan_worker(a.job),
     'repair':lambda:repair_caps(a.job),'repair-lane':lambda:lane(a.gpu,repair=True)}[a.command]()
