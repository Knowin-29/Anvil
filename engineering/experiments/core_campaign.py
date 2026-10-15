"""Finite G-to-S mechanism campaign; no full-G/S/A qualification is inferred."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from native_completion import A, W, D, MODEL, sha, save, rows, read_binding, commit_bytes
from native_completion import prediction_spool, spool_batch_ready, commit_prediction
from linked_training import ROOT as SOURCE, score_linked, context_status

ROOT=D/'core_gs_20260916'
QUALITY=D/'quality_gs_20260916'
RETENTION=ROOT/'retention'
PY=Path(os.environ.get('ANVIL_PYTHON',sys.executable))
ARMS={'semantic_weighted':('semantic',2),'flat':('flat',0),
      'declared':('declared',2),'structural':('structural',2),'semantic_uniform':('semantic',0)}
JOBS=[{'id':f'{arm}_{seed}','arm':arm,'seed':seed,'steps':256} for seed in (17,23) for arm in ARMS]


def scene_interval(items, field, other=None):
    """Paired physical-scene bootstrap; does not invent training replicates."""
    import numpy as np
    if other is not None:
        assert [(r['sample_id'],r['scene']) for r in items]==[(r['sample_id'],r['scene']) for r in other]
    scenes=sorted({r['scene'] for r in items});index={s:i for i,s in enumerate(scenes)}
    groups=np.array([index[r['scene']] for r in items])
    values=np.array([float(r[field]) for r in items])
    if other is not None:values-=np.array([float(r[field]) for r in other])
    totals=np.bincount(groups,weights=values,minlength=len(scenes))
    counts=np.bincount(groups,minlength=len(scenes))
    draws=np.random.default_rng(20260916).integers(len(scenes),size=(5000,len(scenes)))
    samples=totals[draws].sum(1)/counts[draws].sum(1)
    return {'mean':float(values.mean()),'low':float(np.quantile(samples,.025)),
        'high':float(np.quantile(samples,.975)),'scenes':len(scenes),'records':len(items)}


def collect_context():
    context_status();root=SOURCE/'contexts_v1';source=rows(SOURCE/'qualification_S.jsonl')
    values={};bindings={}
    for arm in ('S','joint'):
        values[arm]={}
        for condition in ('reference','student','shifted'):
            folder=SOURCE/arm/'joint' if condition=='reference' else root/arm/condition
            binding=read_binding(folder/'done.json')
            for name,key in [('predictions.jsonl','prediction_sha256'),('result.json','result_sha256')]:
                assert sha(folder/name)==binding[key];bindings[str(folder/name)]=binding[key]
            result=read_binding(folder/'result.json')['models'][arm]
            byid={r['sample_id']:r for r in result['items'] if r['form']=='depth'}
            assert set(byid)=={r['sample_id'] for r in source}
            measured=[]
            for r in source:
                item=byid[r['sample_id']];pairs=item.get('depth_pairs') or {}
                score=pairs.get('correct',0)/pairs['total'] if pairs.get('total') else 0.
                measured.append({'sample_id':r['sample_id'],'scene':r['gsa_meta']['scene_group'],'accuracy':score})
            expected=result['per_form']['depth']['reference_diagnostics']['mean_per_item_pair_accuracy']
            assert abs(sum(r['accuracy'] for r in measured)/len(measured)-expected)<1e-10
            values[arm][condition]=measured
    contrasts={c:scene_interval(values['S'][c],'accuracy',values['joint'][c]) for c in values['S']}
    result={'bindings':bindings,'contrast':contrasts,'items':values,'resamples':5000,'seed':20260916,
        'scope':'Source-specific S versus joint comparison; previously observed 282-image/50-scene split. G has no predecessor. No full-GSA eligibility claim.'}
    save(root/'acceptance.json',result)
    tex=[r'\begin{table}[!htbp]',r'\centering\small',
        r'\caption{\textbf{Depth reasoning under changed upstream localization.} The same 282 images and 50 physical scenes are evaluated with reference boxes, joint-model boxes, and deterministically shifted boxes. Depth targets remain fixed. Entries are accuracy (\%); paired differences include 95\% scene-bootstrap intervals (5,000 draws). Both teachers use the same 512-update source-SFT budget. These observations do not establish a specialist advantage.}',
        r'\label{tab:source-context}',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
        r'Upstream boxes & S specialist & Joint & S $-$ joint \\',r'\midrule']
    for condition,title in [('reference','Reference'),('student','Student-produced'),('shifted','Shifted')]:
        means=[100*sum(r['accuracy'] for r in values[a][condition])/len(source) for a in ('S','joint')]
        v=contrasts[condition]
        tex.append(f"{title} & {means[0]:.2f} & {means[1]:.2f} & {100*v['mean']:+.2f} [{100*v['low']:+.2f}, {100*v['high']:+.2f}]"+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    commit_bytes(A/'overleaf/source/tables/source_context.tex',('\n'.join(tex)+'\n').encode())
    print(json.dumps(contrasts),flush=True);return result


def verified_result(folder):
    done=read_binding(folder/'done.json');training=read_binding(folder/'training_done.json')
    adapter=Path(training['adapter'])
    for file,key in [(folder/'training_done.json','training_done_sha256'),
                     (folder/'evaluation/predictions.jsonl','prediction_sha256'),
                     (folder/'evaluation/result.json','result_sha256')]:
        assert sha(file)==done[key],str(file)
    assert sha(SOURCE/'qualification_linked.jsonl')==done['source_sha256']
    assert sha(adapter/'adapter_model.safetensors')==training['adapter_sha256']
    state_path=Path(training.get('trainer_state_path',str(adapter/'trainer_state.json')))
    assert sha(state_path)==training['trainer_state_sha256']
    state=read_binding(state_path);assert state['global_step']==256
    measured=score_linked(rows(SOURCE/'qualification_linked.jsonl'),rows(folder/'evaluation/predictions.jsonl'))
    assert measured==read_binding(folder/'evaluation/result.json')
    return measured,training


def checked_retention(name):
    folder=RETENTION/name;done=read_binding(folder/'done.json')
    assert sha(folder/'predictions.json')==done['prediction_sha256']
    assert sha(folder/'result.json')==done['result_sha256']
    value=read_binding(folder/'result.json')
    assert len(value['items'])==1011 and len({r['uid'] for r in value['items']})==1011
    for dataset,expected in value['benchmarks'].items():
        selected=[r['credit'] for r in value['items'] if r['dataset']==dataset]
        assert abs(sum(selected)/len(selected)-expected)<1e-10
    return value


def collect_core():
    from copy import deepcopy
    baseline=checked_retention('initial');observed={};combined={}
    for job in JOBS:
        folder=ROOT/job['id'];measured,training=verified_result(folder)
        retention=checked_retention(job['id'])
        binding=read_binding(RETENTION/job['id']/'binding.json')
        assert binding['adapter_sha256']==training['adapter_sha256']
        adapter=Path(training['adapter']);logs=list(adapter.parent.glob('anvil_routes_rank*.jsonl'))
        assert len(logs)==2
        routes=[r for p in logs for r in rows(p)]
        assert sum(r['budget']['examples'] for r in routes)==1024
        decisions=[v for r in routes for v in r['routes']]
        eligible=sum(r['eligible_local'] for r in routes)
        assert len(decisions)==1024
        budgets={k:sum(r['budget'][k] for r in routes) for k in routes[0]['budget']
                 if isinstance(routes[0]['budget'][k],int)}
        observed[job['id']]={'native':measured,'retention':retention['benchmarks'],
            'retention_ratio':retention['macro']/baseline['macro'],
            'eligible':eligible/1024,'G_route_among_eligible':sum(r['stage']=='G' for r in decisions)/eligible if eligible else 0.,
            'budgets':budgets,'training':training,'route_log_sha256':{str(p):sha(p) for p in logs}}
    for arm in ARMS:
        a=observed[arm+'_17']['native']['items'];b=observed[arm+'_23']['native']['items']
        assert [(r['sample_id'],r['scene']) for r in a]==[(r['sample_id'],r['scene']) for r in b]
        combined[arm]=deepcopy(a)
        for target,x,y in zip(combined[arm],a,b):
            for field in ('G','S','strict_chain'):target[field]=(x[field]+y[field])/2
    report={'jobs':observed,'contrast':{arm:{other:scene_interval(combined[arm],'strict_chain',combined[other])
            for other in ('flat','declared')} for arm in ARMS},
        'inference':'Paired scene bootstrap of the two-seed mean. This interval conditions on the two observed seeds; no population training-seed interval is asserted.'}
    save(ROOT/'acceptance.json',report)
    tex=[r'\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xccccc@{}}',r'\toprule',
         r'Method & G$\to$S & $\Delta$ flat [95\% CI] & $\Delta$ declared [95\% CI] & Eligible / G route & VQA retain. \\',r'\midrule']
    titles={'flat':'Flat declared-teacher','structural':'Structural route','declared':'Declared route',
        'semantic_uniform':'Semantic (uniform)','semantic_weighted':r'\model: semantic + weights'}
    for arm in ('flat','structural','declared','semantic_uniform','semantic_weighted'):
        score=100*sum(r['strict_chain'] for r in combined[arm])/282
        interval=lambda other:('reference' if arm==other else '{mean:+.2f} [{low:+.2f}, {high:+.2f}]'.format(
            **{k:100*report['contrast'][arm][other][k] for k in ('mean','low','high')}))
        mean=lambda key:sum(observed[f'{arm}_{seed}'][key] for seed in (17,23))/2
        tex.append(f"{titles[arm]} & {score:.2f} & {interval('flat')} & {interval('declared')} & "
                   f"{100*mean('eligible'):.1f} / {100*mean('G_route_among_eligible'):.1f} & {100*mean('retention_ratio'):.1f}\\%"+r' \\')
    tex += [r'\bottomrule',r'\end{tabularx}']
    commit_bytes(A/'overleaf/source/tables/tab_mtd_ablation.tex',('\n'.join(tex)+'\n').encode())
    lines=[r'\begin{table}[!htbp]',r'\centering\small',
        r'\caption{\textbf{Per-seed routing observations.} Strict G$\to$S success on 282 images / 50 scenes and sampled VQA retention, reported as seed 17 / seed 23. Teacher positions count scored logits including distributed padding; realized budgets are measured, not matched.}',
        r'\label{tab:core-seeds}',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
        r'Method & G$\to$S (\%) & VQA ratio (\%) & Teacher positions \\',r'\midrule']
    for arm in titles:
        pair=[observed[f'{arm}_{s}'] for s in (17,23)]
        lines.append(titles[arm]+' & '+' / '.join(f"{100*v['native']['mean']['strict_chain']:.2f}" for v in pair)
            +' & '+' / '.join(f"{100*v['retention_ratio']:.2f}" for v in pair)
            +' & '+' / '.join(str(v['budgets']['teacher_scored_positions']) for v in pair)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    commit_bytes(A/'overleaf/source/tables/core_seeds.tex',('\n'.join(lines)+'\n').encode())
    return report


def collect_component_contrasts():
    """Compare measured routing components on the same two-seed predictions."""
    core=read_binding(ROOT/'acceptance.json')
    combined={}
    for arm in ARMS:
        pair=[core['jobs'][f'{arm}_{seed}']['native']['items'] for seed in (17,23)]
        assert [(r['sample_id'],r['scene']) for r in pair[0]]==[(r['sample_id'],r['scene']) for r in pair[1]]
        combined[arm]=[{'sample_id':a['sample_id'],'scene':a['scene'],
            'strict_chain':(a['strict_chain']+b['strict_chain'])/2} for a,b in zip(*pair)]
    contrasts={arm:scene_interval(combined['semantic_weighted'],'strict_chain',combined[arm])
               for arm in ('flat','declared','structural','semantic_uniform')}
    for arm in ('flat','declared'):
        assert contrasts[arm]==core['contrast']['semantic_weighted'][arm]
    report={'core_acceptance_sha256':sha(ROOT/'acceptance.json'),'contrast':contrasts,
        'scope':'Paired scene intervals conditional on the two observed training seeds; structural routing changes both selected examples and spans, whereas the two semantic arms differ in token weighting.'}
    save(ROOT/'component_contrasts.json',report)
    lines=[r'\begin{table}[!htbp]',r'\centering\small',
        r'\caption{\textbf{Direct component contrasts.} Semantic routing with weighted spans minus each control, in strict G$\to$S percentage points. The paired scene bootstrap averages predictions across seeds 17/23 before resampling the same 50 scenes (5,000 draws); it does not estimate population training-seed variation.}',
        r'\label{tab:core-components}',r'\begin{tabular}{@{}lrr@{}}',r'\toprule',
        r'Control & Difference & 95\% scene interval \\',r'\midrule']
    for arm,title in [('flat','Flat declared-teacher'),('declared','Declared route'),
                      ('structural','Structural route'),('semantic_uniform','Semantic, uniform')]:
        v=contrasts[arm]
        lines.append(f"{title} & {100*v['mean']:+.2f} & [{100*v['low']:+.2f}, {100*v['high']:+.2f}]"+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    commit_bytes(A/'overleaf/source/tables/core_components.tex',('\n'.join(lines)+'\n').encode())
    print(json.dumps(report),flush=True);return report


def collect_quality(allow_partial=False):
    manifest=read_binding(QUALITY/'training_manifest.json');observed={}
    for name in QUALITY_ARMS:
        if allow_partial and not (QUALITY/'training'/name/'done.json').exists():continue
        measured,training=verified_result(QUALITY/'training'/name)
        source=manifest['training_files'][name]
        assert sha(source['path'])==source['sha256']==training['source_sha256']
        observed[name]={'native':measured,'training':training,'pool':manifest['full_pool'][name]}
    assert 'raw' in observed,'Raw baseline is required for every downstream contrast'
    complete=set(observed)==set(QUALITY_ARMS)
    assert complete or allow_partial
    report={'arms':observed,'manifest_sha256':sha(QUALITY/'training_manifest.json'),
        'contrast':{name:scene_interval(observed[name]['native']['items'],'strict_chain',observed['raw']['native']['items'])
                    for name in observed}}
    save(QUALITY/('acceptance.json' if complete else 'partial_acceptance.json'),report)
    names=['Raw generation','+ syntax parser','+ typed contracts','+ round 2, no feedback','+ rejection feedback']
    tex=['% anvil-source-quality-v1',r'\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xcccc@{}}',r'\toprule',
        r'Data construction & Retained & Source correctness & G$\to$S gain & Named feedback \\',r'\midrule']
    for name,title in zip(QUALITY_ARMS,names):
        value=manifest['full_pool'][name];n=value['retained'];assert n>0
        suffix=r'$^\dagger$' if name in ('round2','feedback') else ''
        gain=('reference' if name=='raw' else f"{100*report['contrast'][name]['mean']:+.2f}"
              if name in observed else r'\needcheck{pending}')
        tex.append(f"{title} & {100*n/value['candidates']:.1f}\\% & {100*value['source_correct']/n:.1f}\\%{suffix} & "
            + gain
            +(' & '+(r'\ding{51}' if name=='feedback' else r'\ding{55}'))+r' \\')
    tex += [r'\bottomrule',r'\end{tabularx}']
    commit_bytes(A/'overleaf/source/tables/tab_pipeline_ablation.tex',('\n'.join(tex)+'\n').encode())
    return report


def quality_results():
    """Publish verified downstream cells without waiting for the routing campaign."""
    with (QUALITY/'downstream_publication.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for required in (QUALITY_ARMS[:4],QUALITY_ARMS):
            while not all((QUALITY/'training'/name/'done.json').exists() for name in required):
                if (QUALITY/'queue_finished.json').exists() and any(read_binding(QUALITY/'queue_finished.json')['exit_codes']):
                    raise RuntimeError('Quality queue failed; downstream cells remain pending')
                time.sleep(30)
            if (ROOT/'published.json').exists():return
            receipt=QUALITY/'downstream_summary_published.json'
            if receipt.exists() and set(required)<=set(read_binding(receipt)['completed_arms']):continue
            report=collect_quality(allow_partial=True)
            if len(report['arms'])==len(QUALITY_ARMS):
                source=A/'overleaf/source/sections/experiments.tex'
                commit_bytes(source,backfill_quality_text(source.read_text(),report).encode())
            from submission_closeout import publish
            publish()
            accepted=QUALITY/('acceptance.json' if len(report['arms'])==len(QUALITY_ARMS) else 'partial_acceptance.json')
            save(receipt,{'time':time.time(),'completed_arms':list(report['arms']),
                'acceptance':str(accepted),'acceptance_sha256':sha(accepted),
                'scope':'Measured table cells only; routing and remaining prose retain their independent dependencies.'})
        receipt=QUALITY/'downstream_summary_published.json'
        if receipt.exists() and read_binding(receipt).get('scaling_appendix_complete'):return
        while not all((QUALITY/'training'/name/'done.json').exists() for name in SCALE_ARMS):
            if (QUALITY/'queue_finished.json').exists() and any(read_binding(QUALITY/'queue_finished.json')['exit_codes']):
                raise RuntimeError('Nested coverage queue failed; keep its evidence pending')
            time.sleep(30)
        collect_scaling()
        source=A/'overleaf/source/sections/appendix_results.tex';content=source.read_text()
        old=r'''\needcheck{Matched-size filtering and matched-call feedback; fixed-token nested-data
scales; semantic precision with the audited denominator and end-to-end generation time.}'''
        new=r'''Matched-size filtering and equal-call repair (Table~\ref{tab:pipeline-ablation}); nested 25/50/100\% source coverage at fixed updates (Table~\ref{tab:measured-data-scaling}). Source correctness and retained-candidate denominators are separate from historical human judgments.'''
        if old in content:content=content.replace(old,new)
        else:assert new in content
        # Future core tables are included only once their real files exist.
        # The existing evidence finalizer detects these same input directives.
        for name in ('core_seeds','core_aggregates','measured_data_scaling'):
            if r'\input{tables/'+name+'}' not in content:
                content+='\n'+r'\IfFileExists{tables/'+name+r'.tex}{\input{tables/'+name+r'}}{}'+'\n'
        commit_bytes(source,content.encode())
        from submission_closeout import publish
        publish()
        value=read_binding(receipt)
        value.update(time=time.time(),scaling_appendix_complete=True,
            scaling_acceptance_sha256=sha(QUALITY/'scaling_acceptance.json'))
        save(receipt,value)


def collect_quality_initial():
    """Backfill observed first-pass cells while repair/training run independently."""
    import re
    source=rows(SOURCE/'train_linked.jsonl');byid={r['sample_id']:r for r in source}
    predictions=[];bindings={};worker_seconds=0.
    for shard in (0,1):
        folder=QUALITY/f'shard_{shard}'/'initial';done=read_binding(folder/'done.json')
        path=folder/'predictions.jsonl';assert sha(path)==done['prediction_sha256']
        values=rows(path);assert len(values)==done['rows']
        assert [p['sample_id'] for p in values]==[r['sample_id'] for r in source[shard::2]]
        worker_seconds+=sum(p['batch_seconds']/p['batch_size'] for p in values)
        bindings[str(path)]=sha(path);predictions+=values
    assert len(predictions)==len({r['sample_id'] for r in predictions})==2011
    checks=[quality_check(byid[p['sample_id']],p) for p in predictions]
    pools={name:[v for v in checks if name=='raw' or v[name]] for name in ('raw','syntax','typed')}
    report={'source_sha256':sha(SOURCE/'train_linked.jsonl'),'prediction_sha256':bindings,
        'candidates':2011,'pool':{name:{'retained':len(values),'source_correct':sum(v['source_correct'] for v in values)}
             for name,values in pools.items()},
        'model':'Qwen3.6-27B','initial_generation_calls':2011,'new_human_annotation':0,
        'inference_worker_seconds':worker_seconds,'allocated_pair_gpu_hours':2*worker_seconds/3600,
        'timing_scope':'Sum of measured batch inference times on two workers, each assigned two GPUs; excludes model initialization and later repairs.'}
    save(QUALITY/'initial_acceptance.json',report)
    path=A/'overleaf/source/tables/tab_pipeline_ablation.tex';text=path.read_text()
    if not text.startswith('% anvil-source-quality-v1\n'):text='% anvil-source-quality-v1\n'+text
    text=text.replace('Structural retention & Human precision','Retained & Source correctness')
    for name,label in [('raw','Raw generation'),('syntax',r'\quad + syntax parser'),('typed',r'\quad + typed contracts')]:
        value=report['pool'][name]
        pattern=r'(?m)^('+re.escape(label)+r'\n\s*& )[^\n]*?( & )[^\n]*?( & )'
        def replacement(m):
            return m[1]+f"{100*value['retained']/2011:.1f}\\%"+m[2]+f"{100*value['source_correct']/value['retained']:.1f}\\%"+m[3]
        text,n=re.subn(pattern,replacement,text);assert n==1,(label,n)
    commit_bytes(path,text.encode())
    path=A/'overleaf/source/sections/experiments.tex';text=path.read_text()
    caption=r'''\caption{\textbf{Progressive data-pipeline ablation.} The first three rows reuse
the same 2,011 initial G$\to$S candidates from Qwen3.6-27B. Retention uses all
candidates; correctness is measured among retained candidates against official
instance and depth targets. Repair and downstream comparisons use this same source
pool and are filled as their matched-call and matched-size runs complete.
The historical 2,618-response replay is reported separately in Table~\ref{tab:filter-replay}.}'''
    text,n=re.subn(r'\\caption\{\\textbf\{Progressive data-pipeline ablation\.\}.*?(?=\n\\label\{tab:pipeline-ablation\})',lambda _:caption,text,flags=re.S)
    assert n==1;commit_bytes(path,text.encode())
    print(json.dumps({k:v for k,v in report.items() if k!='prediction_sha256'}),flush=True)
    return report


def collect_quality_repair():
    """Accept a complete repair arm without waiting for the other arm or SFT."""
    import re
    source=rows(SOURCE/'train_linked.jsonl');byid={r['sample_id']:r for r in source}
    initial=[];bindings={}
    for shard in (0,1):
        folder=QUALITY/f'shard_{shard}/initial';path=folder/'predictions.jsonl'
        assert sha(path)==read_binding(folder/'done.json')['prediction_sha256']
        initial+=rows(path);bindings[str(path)]=sha(path)
    assert len(initial)==len({p['sample_id'] for p in initial})==len(source)==2011
    failed={p['sample_id'] for p in initial if not quality_check(byid[p['sample_id']],p)['source_correct']}
    assert len(failed)==885
    result={'source_sha256':sha(SOURCE/'train_linked.jsonl'),'initial_correct':len(source)-len(failed),
        'condition':{},'predictions':bindings,'scope':'Correctness is the source-based admission rule; no independent verifier precision claim.'}
    table=A/'overleaf/source/tables/tab_pipeline_ablation.tex';content=table.read_text()
    for condition,title in (('no_feedback',r'\quad + round 2, no feedback'),('feedback',r'\quad + rejection feedback')):
        if not all((QUALITY/f'shard_{s}'/condition/'done.json').exists() for s in (0,1)):continue
        predictions=[]
        for shard in (0,1):
            folder=QUALITY/f'shard_{shard}'/condition;path=folder/'predictions.jsonl'
            assert sha(path)==read_binding(folder/'done.json')['prediction_sha256']
            predictions+=rows(path);bindings[str(path)]=sha(path)
        assert len(predictions)==len(failed) and {p['sample_id'] for p in predictions}==failed
        recovered=sum(quality_check(byid[p['sample_id']],p)['source_correct'] for p in predictions)
        retained=result['initial_correct']+recovered
        result['condition'][condition]={'calls':len(predictions),'newly_correct':recovered,
            'retained':retained,'candidates':len(source),'conditional_source_correctness':1.0}
        pattern=r'(?m)^('+re.escape(title)+r'\n[ \t]*&[ \t]*)[^&\n]*( & )[^&\n]*( &[ \t]*(?:\n[ \t]*)?)'
        content,count=re.subn(pattern,lambda m:m[1]+f'{100*retained/len(source):.1f}'+r'\%'+m[2]+r'100.0\%$^\dagger$'+m[3],content)
        assert count==1,title
    assert result['condition'],'No repair arm is complete'
    save(QUALITY/'repair_acceptance.json',result);commit_bytes(table,content.encode())
    path=A/'overleaf/source/sections/experiments.tex';content=path.read_text()
    sentence=r'$^\dagger$Source correctness is the admission rule, not an independent estimate of verifier accuracy.'
    if sentence not in content:
        pattern=r'(\\caption\{\\textbf\{Progressive data-pipeline ablation\.\}.*?)(\}\n\\label\{tab:pipeline-ablation\})'
        content,count=re.subn(pattern,lambda m:m[1]+'\n'+sentence+m[2],content,flags=re.S)
        assert count==1;commit_bytes(path,content.encode())
    print(json.dumps(result['condition']),flush=True)
    return result


def quality_generation_summary():
    """Publish generation evidence independently of the slower downstream training."""
    import re
    with (QUALITY/'generation_summary.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (QUALITY/'generation_summary_published.json').exists():return
        while not (QUALITY/'generation_finished.json').exists():time.sleep(30)
        assert not any(read_binding(QUALITY/'generation_finished.json')['exit_codes'])
        started=time.perf_counter();manifest=quality_materialize()
        source=rows(SOURCE/'train_linked.jsonl');byid={r['sample_id']:r for r in source}
        phases={name:[] for name in ('initial','no_feedback','feedback')}
        bindings={}
        for shard in (0,1):
            for condition in phases:
                folder=QUALITY/f'shard_{shard}'/condition
                path=folder/'predictions.jsonl';done=read_binding(folder/'done.json')
                assert sha(path)==done['prediction_sha256']
                bindings[str(path)]=sha(path);phases[condition]+=rows(path)
        initial={p['sample_id']:p for p in phases['initial']}
        assert len(initial)==len(phases['initial'])==2011
        checks={uid:quality_check(byid[uid],pred) for uid,pred in initial.items()}
        failed={uid for uid,check in checks.items() if not check['source_correct']}
        assert len(failed)==885
        measured={}
        for condition,predictions in phases.items():
            assert len({p['sample_id'] for p in predictions})==len(predictions)
            assert {p['sample_id'] for p in predictions}==(set(initial) if condition=='initial' else failed)
            correct=sum(quality_check(byid[p['sample_id']],p)['source_correct'] for p in predictions)
            seconds=sum(p['batch_seconds']/p['batch_size'] for p in predictions)
            assert seconds>0
            measured[condition]={'calls':len(predictions),'source_correct':correct,
                'inference_worker_seconds':seconds,'allocated_gpu_hours':seconds*2/3600,
                'source_correct_per_allocated_gpu_hour':correct/(seconds*2/3600),
                'prompt_tokens':sum(p['prompt_tokens'] for p in predictions),
                'completion_tokens':sum(p['completion_tokens'] for p in predictions)}
        for arm,condition in (('round2','no_feedback'),('feedback','feedback')):
            assert manifest['full_pool'][arm]['retained']==measured['initial']['source_correct']+measured[condition]['source_correct']
        report={'source_sha256':sha(SOURCE/'train_linked.jsonl'),'prediction_sha256':bindings,
            'phases':measured,'new_human_annotation':0,'paid_calls':0,
            'scope':'Each phase uses two-GPU workers; measured inference time excludes model loading and idle time. Repair phases each address exactly the same 885 initial failures. Source correctness determines admission and is not an independent verifier precision estimate.',
            'cpu_collection_wall_seconds':time.perf_counter()-started}
        save(QUALITY/'generation_acceptance.json',report)
        tex=[r'\begin{table}[!htbp]',r'\centering\small',
            r'\caption{\textbf{Measured generation cost and repair yield.} Qwen3.6-27B generates a common 2,011-item pool. Each repair arm receives one extra call on the same 885 initial failures; successes in a repair row count newly admitted items. GPU-hours sum measured batch inference time multiplied by the two GPUs assigned to each worker, excluding loading and idle time. Source-derived checks require no new human annotation.}',
            r'\label{tab:generation-cost}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',
            r'Phase & Calls & Source-correct & GPU-hours & Correct / GPU-hour \\',r'\midrule']
        for condition,title in (('initial','Initial generation'),('no_feedback','Repair, no feedback'),('feedback','Repair, named feedback')):
            v=measured[condition]
            tex.append(f"{title} & {v['calls']:,} & {v['source_correct']:,} & {v['allocated_gpu_hours']:.2f} & {v['source_correct_per_allocated_gpu_hour']:.1f}"+r' \\')
        tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        commit_bytes(A/'overleaf/source/tables/generation_cost.tex',('\n'.join(tex)+'\n').encode())
        path=A/'overleaf/source/sections/appendix_results.tex';content=path.read_text()
        if r'\input{tables/generation_cost}' not in content:
            commit_bytes(path,(content+'\n'+r'\input{tables/generation_cost}'+'\n').encode())
        path=A/'overleaf/source/tables/tab_pipeline_ablation.tex';content=path.read_text()
        for name,title in (('round2',r'\quad + round 2, no feedback'),('feedback',r'\quad + rejection feedback')):
            pool=manifest['full_pool'][name]
            pattern=r'(?m)^('+re.escape(title)+r'\n[ \t]*&[ \t]*)[^&\n]*( & )[^&\n]*( &[ \t]*(?:\n[ \t]*)?)'
            def replace(m):
                return (m[1]+f"{100*pool['retained']/pool['candidates']:.1f}\\%"+m[2]
                    +f"{100*pool['source_correct']/pool['retained']:.1f}\\%"+r'$^\dagger$'+m[3])
            content,count=re.subn(pattern,replace,content)
            assert count==1 or (ROOT/'published.json').exists(),title
        commit_bytes(path,content.encode())
        path=A/'overleaf/source/sections/experiments.tex';content=path.read_text()
        if not (ROOT/'published.json').exists():
            caption=r'''\caption{\textbf{Progressive data-pipeline ablation.} All rows start from
2,011 common G$\to$S candidates. The repair arms each use one extra call on the
same 885 initial failures. Retention uses all candidates; correctness uses retained
candidates against source instance/depth targets. $^\dagger$Source correctness is
the admission rule, not an independent estimate of verifier accuracy. Generation
costs appear in Table~\ref{tab:generation-cost}; downstream gains await the
matched-size training runs. Historical human judgments are reported separately.}'''
            content,count=re.subn(r'\\caption\{\\textbf\{Progressive data-pipeline ablation\.\}.*?(?=\n\\label\{tab:pipeline-ablation\})',lambda _:caption,content,flags=re.S)
            assert count==1;commit_bytes(path,content.encode())
            from submission_closeout import publish
            publish()
        save(QUALITY/'generation_summary_published.json',{'time':time.time(),'acceptance_sha256':sha(QUALITY/'generation_acceptance.json')})


def collect_scaling():
    manifest=read_binding(QUALITY/'scaling_manifest.json');observed={}
    full=read_binding(QUALITY/'training_manifest.json')['training_files']['feedback']
    assert sha(full['path'])==full['sha256']==manifest['full_source_sha256']
    for name in (*SCALE_ARMS,'feedback'):
        result,training=verified_result(QUALITY/'training'/name)
        source=full if name=='feedback' else manifest['training_files'][name]
        assert sha(source['path'])==source['sha256']==training['source_sha256']
        observed[name]={'native':result,'training':training,'records':source['rows'],
            'score_interval':scene_interval(result['items'],'strict_chain')}
    assert set(manifest['training_files'][SCALE_ARMS[0]]['sample_ids']) < set(manifest['training_files'][SCALE_ARMS[1]]['sample_ids']) < set(full['sample_ids'])
    report={'manifest_sha256':sha(QUALITY/'scaling_manifest.json'),'arms':observed,
        'scope':'Nested source coverage; same updates and row exposures, one seed. Actual input tokens are not forced equal.'}
    save(QUALITY/'scaling_acceptance.json',report)
    tex=[r'\begin{table}[!htbp]',r'\centering\small',
        r'\caption{\textbf{Nested source coverage at fixed updates.} The feedback corpus is frozen before downstream evaluation. Nested scene-stratified subsets use the same fresh-base initialization, 256 updates, 1,024 row exposures, and seed 17. Scores are strict G$\to$S success on 282 images / 50 scenes with 95\% scene-bootstrap intervals. Input tokens are measured; the experiment does not fit an equal-token scaling law.}',
        r'\label{tab:measured-data-scaling}',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
        r'Pool fraction & Unique records & Input tokens & Success (\%) \\',r'\midrule']
    for name,label in zip((*SCALE_ARMS,'feedback'),('25\%','50\%','100\%')):
        value=observed[name];score=value['score_interval'];tokens=value['training']['num_input_tokens_seen']
        assert isinstance(tokens,int) and tokens>0
        tex.append(f"{label} & {value['records']} & {tokens:,} & {100*score['mean']:.2f} [{100*score['low']:.2f}, {100*score['high']:.2f}]"+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    commit_bytes(A/'overleaf/source/tables/measured_data_scaling.tex',('\n'.join(tex)+'\n').encode())
    return report


def finish_evidence():
    """Follow the existing publisher without touching any GPU process/session."""
    with (ROOT/'finish_evidence.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (ROOT/'published.json').exists():
            if list(ROOT.glob('*/failure.json')):raise RuntimeError('A core job failed before evidence publication')
            time.sleep(30)
        while not all((QUALITY/'training'/name/'done.json').exists() for name in SCALE_ARMS):
            if (QUALITY/'queue_finished.json').exists() and any(read_binding(QUALITY/'queue_finished.json')['exit_codes']):
                raise RuntimeError('Nested source coverage did not finish')
            time.sleep(30)
        with (ROOT/'finalize.lock').open('w') as publication_lock:
            fcntl.flock(publication_lock,fcntl.LOCK_EX)
            core=collect_core();quality=collect_quality();scaling=collect_scaling()
            backfill_prose(core,quality)
            source=A/'overleaf/source/sections/appendix_results.tex';text=source.read_text()
            replacements={
                r'''\needcheck{Flat, declared-family, structural-first, and semantic-first routing;
selector-only and route--weight/reference-access controls; route coverage and F1.}''':
                r'''Five routing arms, two seeds, strict source-linked success, routed-example fractions and realized teacher budgets (Tables~\ref{tab:mtd-ablation} and~\ref{tab:core-seeds}). Additional reference-access and selector-only designs are not measured.''',
                r'''\needcheck{Matched-size filtering and matched-call feedback; fixed-token nested-data
scales; semantic precision with the audited denominator and end-to-end generation time.}''':
                r'''Matched-size filtering and equal-call repair (Table~\ref{tab:pipeline-ablation}); nested 25/50/100\% source coverage at fixed updates (Table~\ref{tab:measured-data-scaling}). Source correctness and retained-candidate denominators are separate from historical human judgments.''',
                r'''\needcheck{Paired seed--scene intervals for completed core runs; alternative aggregates
computed from the same predictions, without inventing missing seeds.}''':
                r'''Per-seed scores and paired scene intervals for the observed seeds 17/23; stage-mean and scene-equal summaries use the same saved predictions (Tables~\ref{tab:core-seeds} and~\ref{tab:core-aggregates}). No population training-seed interval is inferred from two runs.'''}
            for old,new in replacements.items():
                if old in text:text=text.replace(old,new)
            includes='\n'.join(r'\input{tables/'+name+'}' for name in ('core_seeds','core_aggregates','measured_data_scaling'))
            if r'\input{tables/core_seeds}' not in text:text+='\n'+includes+'\n'
            commit_bytes(source,text.encode())
            tex=[r'\begin{table}[!htbp]',r'\centering\small',
                r'\caption{\textbf{Aggregation sensitivity on the same linked predictions.} Entries average seeds 17/23. Stage mean averages grounding F1 and depth accuracy; strict success requires both stage thresholds. Scene-equal success first averages within each of the 50 physical scenes. These are G$\to$S summaries, not a full G/S/A macro.}',
                r'\label{tab:core-aggregates}',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
                r'Route & Stage mean & Strict success & Scene-equal success \\',r'\midrule']
            for arm in ARMS:
                per_seed=[]
                for seed in (17,23):
                    native=core['jobs'][f'{arm}_{seed}']['native'];items=native['items']
                    scenes=sorted({r['scene'] for r in items})
                    equal=sum(sum(r['strict_chain'] for r in items if r['scene']==s)/sum(r['scene']==s for r in items) for s in scenes)/len(scenes)
                    per_seed.append(((native['mean']['G']+native['mean']['S'])/2,native['mean']['strict_chain'],equal))
                metrics=[sum(v[i] for v in per_seed)*50 for i in range(3)]
                tex.append(arm.replace('_',' ')+''.join(f' & {v:.2f}' for v in metrics)+r' \\')
            tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
            commit_bytes(A/'overleaf/source/tables/core_aggregates.tex',('\n'.join(tex)+'\n').encode())
            from submission_closeout import publish
            publish()
            save(ROOT/'evidence_published.json',{'time':time.time(),'core_sha256':sha(ROOT/'acceptance.json'),
                'quality_sha256':sha(QUALITY/'acceptance.json'),'scaling_sha256':sha(QUALITY/'scaling_acceptance.json'),
                'remaining_scope':'Final abstract/conclusion, native benchmark scope, model/artifact registry and submission cleanup remain separate acceptance tasks.'})


def finish_submission():
    """Package completed evidence, then reconcile six repeated/scoped placeholders."""
    import hashlib, re, tarfile, zipfile
    with (ROOT/'finish_submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'artifacts_published.json').exists():return
        while not (ROOT/'evidence_published.json').exists():
            if list(ROOT.glob('*/failure.json')):
                raise RuntimeError('A core experiment failed; submission backfill is withheld')
            for path in (ROOT/'queue_finished.json',QUALITY/'queue_finished.json',QUALITY/'generation_finished.json'):
                if path.exists() and any(read_binding(path)['exit_codes']):
                    if path.parent==ROOT:
                        # A new queue can legitimately supersede an old failed run.
                        # The live supervisor owns this lock throughout its retry.
                        with (ROOT/'queue.lock').open('a') as queue_lock:
                            try:fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                            except BlockingIOError:continue
                    raise RuntimeError('An experiment queue failed: '+str(path))
            save(ROOT/'submission_progress.json',{'state':'waiting_for_verified_experiments','time':time.time()})
            time.sleep(30)
        with (ROOT/'finalize.lock').open('w') as publication_lock:
            fcntl.flock(publication_lock,fcntl.LOCK_EX)
            core=read_binding(ROOT/'acceptance.json')
            package=A/'release/Anvil_Submission_Supplement_20260916'
            archive=package.with_suffix('.tar.gz')
            if not archive.exists():
                # An interrupted, partial export is retained for diagnosis; never overwritten.
                assert not package.exists(),'Partial supplement exists; inspect before choosing a fresh export path'
                save(ROOT/'submission_progress.json',{'state':'exporting_anonymous_supplement','time':time.time()})
                subprocess.run([str(PY),str(W/'tools/export_anvil_release.py'),'--worktree',str(W),
                    '--package',str(A),'--output',str(package),'--skip-manuscript','--include-campaign',
                    '--path-prefix','/path/to/source-archives=/path/to/source-archives',
                    '--path-prefix',os.environ.get('ANVIL_CHECKPOINT_PATH_MAPPING','/path/to/local-checkpoints=/path/to/checkpoint-archive')],check=True)
            manifest=read_binding(package/'release_manifest.json')
            assert manifest['campaign_included'] and not manifest['manuscript_included']
            replay=manifest['checks']['candidate_execution']['exported_campaign_scoring_replay']
            assert replay=={'replayed_cells':17,'records_per_cell':282,'predictions_recomputed':4794}
            expected={v['path']:v for v in manifest['files']}
            with tarfile.open(archive,'r:gz') as bundle:
                members=bundle.getmembers()
                assert len(members)==len(expected)+1
                assert all(m.isfile() and m.name.startswith('Anvil/') for m in members)
                assert {m.name[6:] for m in members}==set(expected)|{'release_manifest.json'}
                for member in members:
                    name=member.name[6:]
                    target=package/name
                    with bundle.extractfile(member) as handle:
                        checksum=hashlib.file_digest(handle,'sha256').hexdigest()
                    assert checksum==sha(target),name
                    if name in expected:assert checksum==expected[name]['sha256'],name
            registry=read_binding(A/'engineering/model_registry.json')
            public_registry=read_binding(package/'registry/model_registry.json')
            registry['final_routed_student']=public_registry
            registry['source_linked_components']=read_binding(ROOT/'study_binding.json')
            registry['anonymous_local_bundle']={'path':str(archive.relative_to(A)),'sha256':sha(archive)}
            save(A/'engineering/model_registry.json',registry)

            # These statements use the completed study, without asserting favorable results.
            contrast=core['contrast']['semantic_weighted']['flat']
            result_sentence=(r'Two-seed G$\to$S routing changes strict success by '
                f"{100*contrast['mean']:+.2f} points relative to flat distillation "
                f"(95\\% scene interval [{100*contrast['low']:+.2f}, {100*contrast['high']:+.2f}]); "
                'native-mask refresh does not improve over replay in either seed.')
            source=A/'overleaf/source/sections'
            replacements={
                'abstract.tex':{r'\needcheck{Complete matched-budget distillation and teacher-refresh comparisons.}':result_sentence},
                'conclusion.tex':{r'\needcheck{matched-budget routing and teacher-refresh benefits remain to be measured.}':
                    r'the routed G$\to$S study isolates teacher selection at fixed updates. '+result_sentence},
                'experiments.tex':{r'''\needcheck{The complete \bench evaluation will report G/S/A, strict-chain success, VQA
retention, and predicted-versus-oracle gaps.} The appendix specifies per-form scores and
aggregation sensitivity.''':r'''Native evaluation separates the available targets: 282 source-linked G$\to$S
images, a 1,024-item mask panel, and 1,012 state-conditioned Action judgments.
Tables~\ref{tab:mtd-ablation}, \ref{tab:native-coco-transfer}, and
\ref{tab:alfred-plan-verification} report the corresponding results.
Context interventions and sampled VQA retention accompany the linked study;
these separate panels do not define a complete eight-form G/S/A aggregate.'''},
                'appendix_results.tex':{r'''\needcheck{Independent per-form targets and actual test counts; per-form scores,
review reliability, ceilings, and oracle/dependency interventions.}''':r'''Source-specific counts and scores: 282 linked images / 50 scenes, 1,024 mask
items, and 1,012 Action judgments / 92 scenes. Historical reliability uses 320
paired human judgments (Table~\ref{tab:historical-human-review}); changed-context
results appear in Table~\ref{tab:source-context}. A full eight-form human ceiling
and complete G/S/A-chain evaluation are outside these measured studies.'''}}
            for name,mapping in replacements.items():
                path=source/name;content=path.read_text()
                for old,new in mapping.items():
                    if old in content:content=content.replace(old,new)
                    else:assert new in content,'Changed manuscript placeholder: '+name
                if name=='appendix_results.tex':
                    content=content.replace(r'''\caption{\textbf{Remaining evidence packages, not estimated results.}
The former numerical forecast tables are consolidated here. Each row names the
experiment still needed; removing a forecast does not complete that experiment.
Main matched-budget tables retain their existing layout.}''',r'''\caption{\textbf{Measured evidence and scope.} The completed source-specific
studies are listed with their relevant tables. Broader unperformed comparisons
are distinguished from the reported observations.}''')
                commit_bytes(path,content.encode())
            binding=read_binding(ROOT/'study_binding.json')
            identities=[(r'\basemodel','base revision','c20223623576'),
                (r'$q_G$','source G adapter',binding['teachers']['G']['sha256'][:12]),
                (r'$q_S$','source S adapter',binding['teachers']['S']['sha256'][:12]),
                ('Linked SFT','routing initialization',binding['student']['sha256'][:12]),
                (r'\model','routed G'+r'$\to$'+'S student',public_registry['weights_sha256'][:12])]
            handoff=r'''\section{Anonymous Reproduction Supplement}
\label{app:artifacts}

The companion supplementary archive contains the source-linked study's frozen
requests and targets, saved predictions, scoring code, training settings, and
selected student adapter. The manuscript source and figure assets are delivered
separately as an Overleaf project. Source images and base weights are obtained
from their respective providers. Historical human spot checks remain separate
from automated data production.

\begin{table}[!htbp]
\centering\small
\caption{Contents of the anonymous supplementary archive; paths are relative to its root.}
\label{tab:artifact-map}
\begin{tabularx}{\textwidth}{@{}l>{\raggedright\arraybackslash}X@{}}
\toprule
Artifact & Path \\
\midrule
Commands and settings & \path{engineering/worktree/README.md} \\
Data and evaluation code & \path{engineering/experiments/} \\
Frozen linked targets & \path{results/linked_sensor_training_v1/} \\
Routing results and budgets & \path{results/core_gs_20260916/} \\
Generation and quality results & \path{results/quality_gs_20260916/} \\
Final student adapter & \path{registry/anvil-9b/} \\
Model identity and selection & \path{registry/model_registry.json} \\
File inventory and checksums & \path{release_manifest.json} \\
\bottomrule
\end{tabularx}
\end{table}

The archive's scoring code reproduces all 17 linked evaluation cells from 4,794
saved responses on the common 282-image panel. This checks scoring reproducibility;
GPU retraining from the exported copy is not claimed. Server-specific path prefixes
are replaced, while responses, targets, and measured values are preserved. The file
inventory records the original and distributed checksums for transformed files.

\subsection{Model registry}
The routed student is the semantic-weighted arm, seed 17, at its prespecified
terminal update 256. Selection does not depend on evaluation performance. It
continues the common source-linked SFT initialization. Separate source-specific
specialists and checkpoints used in earlier transfer experiments retain their own
identities; their scores are not attributed to this routed adapter.

\begin{table}[!htbp]
\centering\small
\caption{Source-linked training components. The base uses a repository revision;
adapter entries show SHA-256 prefixes. Full identities accompany the results.}
\label{tab:model-registry}
\begin{tabular}{@{}lll@{}}
\toprule
Component & Role & Revision / hash \\
\midrule
'''
            handoff+='\n'.join(f'{name} & {role} & '+r'\texttt{'+digest+r'} \\' for name,role,digest in identities)
            handoff+='\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{table}'+'\n'
            commit_bytes(source/'appendix_handoff.tex',handoff.encode())
            benchmark=source/'benchmark.tex';content=benchmark.read_text()
            content=content.replace(
                'The remaining native-form test releases and linked-task manifest are not yet complete.',
                'The linked G'+r'$\to$'+'S panel adds 282 images from 50 scenes; a separate mask panel has '
                '1,024 items. Action verification uses 1,012 judgments from 506 trajectories and 92 scenes. '
                'These panels measure different tasks and are reported separately.')
            content=content.replace(
                r'''release counts, not design quotas; ``--'' denotes an unreleased test set. Depth's
$40^{\dagger}$ cases are calibration-only. External benchmarks are reported separately.''',
                r'''source-specific counts, not equal design quotas; ``--'' denotes an unavailable
generation test set. Depth's $40^{\dagger}$ historical cases are calibration-only.
The 1,012-item Action verification study is separate from action-plan generation.
External benchmarks are reported separately.''')
            content=content.replace(r' & detection & -- & image + category; emits target ID & COCO mAP',
                r' & detection & 282 & two source objects; linked G stage & detection set-F1')
            content=content.replace(r' & mask & -- & image + referent; emits geometry & mask IoU',
                r' & mask & 1,024 & source instances; emits geometry & polygon IoU')
            content=content.replace(r'\taskS{S: Relation} & relative depth & $40^{\dagger}$',
                r'\taskS{S: Relation} & relative depth & $282+40^{\dagger}$')
            content=content.replace(
                r'$S_{\mathrm{GSA}}$ is a secondary summary accompanied by alternative-weight sensitivity.',
                r'$S_{\mathrm{GSA}}$ defines the complete-interface summary. We do not report this '
                'aggregate for the present source-specific panels. The linked study instead reports '
                'strict success, stage means, and scene-equal summaries.')
            start=content.find('For independently validated linked traces, strict-chain success')
            if start>=0:
                end=content.index('\n% 中文：原子与关联面板',start)
                content=content[:start]+r'''For the measured G$\to$S task, strict success requires a stopped response with
valid stage tags and typed payloads, detection set-F1 of at least 0.75, and depth
accuracy above 0.5 under the fixed source-derived evaluator. These operational
thresholds are held fixed across all arms; they are not estimated from the
historical human pilot. Table~\ref{tab:source-context} compares reference,
student-produced, and shifted boxes while keeping depth targets fixed. The
independent Action study measures plan validity under source transition rules;
it does not extend these observations into a measured G$\to$S$\to$A chain.
''' + content[end:]
            else:
                assert r'For the measured G$\to$S task' in content
            commit_bytes(benchmark,content.encode())
            assert not any(re.search(r'\\needcheck\{',re.sub(r'(?<!\\)%[^\n]*','',p.read_text()))
                           for p in (A/'overleaf/source').rglob('*.tex')),'Unresolved placeholders remain'
            from submission_closeout import publish
            publish()
            with zipfile.ZipFile(A/'overleaf/Anvil_Submission_Draft.zip') as bundle:
                for name in bundle.namelist():
                    assert bundle.read(name)==(A/'overleaf/source'/name).read_bytes(),name
            log=(A/'overleaf/source/main.log').read_text()
            assert not re.search(r'Overfull \\[hv]box|There were undefined|multiply defined|Fatal error',log)
            receipt={'time':time.time(),'experiment_needcheck_remaining':0,
                'supplement':str(archive),'supplement_sha256':sha(archive),
                'pdf_sha256':sha(A/'paper/Anvil_Submission_Draft.pdf'),
                'zip_sha256':sha(A/'overleaf/Anvil_Submission_Draft.zip'),
                'scoring_replay':replay,'publicly_published':False,
                'remaining_acceptance':'Final claim/layout review and recoverable cleanup; zero placeholders alone is not submission acceptance.'}
            save(ROOT/'artifacts_published.json',receipt)
            save(ROOT/'submission_progress.json',{'state':'metrics_and_local_artifacts_complete_pending_final_review','time':time.time()})


def finalize_campaign():
    with (ROOT/'finalize.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            core=all((ROOT/j['id']/'done.json').exists() for j in JOBS)
            quality=all((QUALITY/'training'/n/'done.json').exists() for n in QUALITY_ARMS)
            if core and quality and (RETENTION/'initial/done.json').exists():break
            failures=list(ROOT.glob('*/failure.json'))
            if failures:raise RuntimeError('Campaign has failed jobs: '+','.join(str(p) for p in failures))
            if (QUALITY/'queue_finished.json').exists() and any(read_binding(QUALITY/'queue_finished.json')['exit_codes']):
                raise RuntimeError('Quality training failed; do not publish incomplete results')
            time.sleep(30)
        core=collect_core();quality=collect_quality();collect_context()
        backfill_prose(core,quality)
        from submission_closeout import publish
        publish()
        save(ROOT/'published.json',{'time':time.time(),'acceptance_sha256':sha(ROOT/'acceptance.json'),
            'quality_acceptance_sha256':sha(QUALITY/'acceptance.json'),
            'scope':'Measured table backfill; remaining manuscript placeholders must still be assessed.'})


def backfill_prose(core,quality):
    source=A/'overleaf/source/sections/experiments.tex';text=source.read_text()
    old=r'''\caption{\textbf{Core matched-budget routing test.} Rows match student/teacher tokens,
queries, parameters, and updates; eligibility and route changes are reported. VQA is the
AI2D/ScienceQA/TextVQA post/pre macro; per-seed results appear in the appendix.}'''
    new=r'''\caption{\textbf{Routing at a fixed update budget.} Five continuations share the
initial student, source split, 256 updates, effective batch four, and maximum rollout
budget. Entries average seeds 17/23; intervals resample 50 physical scenes while
conditioning on these two seeds. Eligible is the fraction receiving teacher loss;
G route is its grounding share. VQA is the post/pre three-task macro on fixed
AI2D/ScienceQA/TextVQA panels (309/202/500). Realized token/call budgets and per-seed
scores are reported in Table~\ref{tab:core-seeds}.}'''
    if old in text:text=text.replace(old,new)
    old=r'''\needcheck{Measure matched-budget gains over flat, declared-family, and structural-first
routing, with eligibility and teacher-change rates (Table~\ref{tab:mtd-ablation}).}'''
    c=core['contrast']['semantic_weighted']['flat']
    new=('At the same update budget, semantic routing with weighted spans changes strict '
        r'G$\to$S success relative to flat declared-teacher distillation by '+f"{100*c['mean']:+.2f} points "
        +f"(95\\% scene interval [{100*c['low']:+.2f}, {100*c['high']:+.2f}]). "
        +r'Table~\ref{tab:mtd-ablation} reports all five arms, including non-improving contrasts.')
    if old in text:text=text.replace(old,new)
    commit_bytes(source,backfill_quality_text(text,quality).encode())


def backfill_quality_text(text,quality):
    new=r'''\caption{\textbf{Progressive data-pipeline ablation.} All arms start from 2,011
source-grounded G$\to$S candidates. Syntax and typed filtering reuse first outputs;
the two repair arms use exactly one additional call on each initially failing item.
Retention uses all candidates; correctness uses retained candidates.
$^\dagger$Admission itself uses source correctness, so this conditional value is
not an independent verifier-accuracy estimate. Downstream SFT matches record counts
within physical scenes and 256 updates (seed 17); individual images and realized
tokens can differ. Gains measure strict success on the same 282 held-out images.}'''
    import re
    text,n=re.subn(r'\\caption\{\\textbf\{Progressive data-pipeline ablation\.\}.*?(?=\n\\label\{tab:pipeline-ablation\})',lambda _:new,text,flags=re.S)
    assert n==1
    old=r'''\needcheck{Measure semantic precision, matched-size downstream gains, and fixed-call
feedback yield (Table~\ref{tab:pipeline-ablation}).} Nested-data scaling separates additional
coverage from fixed-size quality effects (Table~\ref{tab:data-scaling}).'''
    manifest=read_binding(QUALITY/'training_manifest.json');n=manifest['matched_records_per_arm']
    c=quality['contrast']['feedback']
    new=(f'Each downstream arm uses {n:,} retained examples, matched by scene count. '
         'Named-feedback data change strict success relative to raw generated data by '
         f"{100*c['mean']:+.2f} points (95\\% scene interval [{100*c['low']:+.2f}, {100*c['high']:+.2f}]). "
         'This comparison measures data selection and repair at matched size; it does not isolate a data-scaling law.')
    if old in text:text=text.replace(old,new)
    return text


def retention_prepare():
    import hashlib, math
    from linked_training import write_rows
    root=W/'artifacts/dlc_preflight/general_retention_v3_current_20260827/inputs'
    receipt=read_binding(root/'input_receipt.json');index=root/'runtime_index.jsonl'
    assert sha(index)==receipt['runtime_index']['sha256']
    values=rows(index);chosen=[];counts={}
    for dataset in ('ai2d_test','scienceqa_test','textvqa_val'):
        group=[r for r in values if r['dataset']==dataset]
        group.sort(key=lambda r:hashlib.sha256(('anvil-core-retention-20260916|'+r['global_uid']).encode()).hexdigest())
        selected=group[:math.ceil(len(group)/10)];chosen.extend(selected)
        counts[dataset]={'population':len(group),'evaluated':len(selected)}
    RETENTION.mkdir(exist_ok=True)
    manifest={'source_index_sha256':sha(index),'source_receipt_sha256':sha(root/'input_receipt.json'),
        'counts':counts,'selection':'Fixed SHA256 UID ordering, ceiling of 10% per benchmark, frozen before core-checkpoint VQA inference.',
        'image_tokens':1536,'thinking':False,'batch':4,'temperature':0,'max_output':'32 for multiple choice; 64 for TextVQA',
        'scope':'Matched initial/post retention panel; these are sampled retention scores, not replacement full-benchmark leaderboard scores.'}
    if (RETENTION/'manifest.json').exists():assert read_binding(RETENTION/'manifest.json')==manifest
    else:
        for r in chosen:assert sha(r['image_path'])==r['image_sha256']
        write_rows(RETENTION/'evaluation.jsonl',chosen);save(RETENTION/'manifest.json',manifest)
    return chosen


def retention_infer(engine,name,adapter_sha):
    from swift import InferRequest, RequestConfig
    from eval.general_retention_formal_v3 import parse_multiple_choice, _answer_span, _vqa_soft_credit, _reference_digit_article
    values=retention_prepare();folder=RETENTION/name;folder.mkdir(exist_ok=True)
    binding={'manifest_sha256':sha(RETENTION/'manifest.json'),'adapter_sha256':adapter_sha,
        'evaluation_sha256':sha(RETENTION/'evaluation.jsonl')}
    if (folder/'binding.json').exists():assert read_binding(folder/'binding.json')==binding
    else:save(folder/'binding.json',binding)
    if (folder/'done.json').exists():return
    predictions=[]
    for dataset in ('ai2d_test','scienceqa_test','textvqa_val'):
        selected=[r for r in values if r['dataset']==dataset];dest=folder/dataset;dest.mkdir(exist_ok=True)
        for i in range(0,len(selected),4):
            batch=selected[i:i+4];path=dest/f'{i:06d}.json'
            if path.exists():
                out=read_binding(path);assert [r['uid'] for r in out]==[r['global_uid'] for r in batch]
            else:
                requests=[InferRequest(messages=[{'role':'user','content':r['prompt']}],images=[r['image_path']]) for r in batch]
                config=RequestConfig(max_tokens=64 if dataset=='textvqa_val' else 32,temperature=0)
                response=engine.infer(requests,config,use_tqdm=False);assert len(response)==len(batch)
                out=[{'uid':r['global_uid'],'response':v.choices[0].message.content or '',
                    'finish_reason':v.choices[0].finish_reason,'completion_tokens':v.usage.completion_tokens} for r,v in zip(batch,response)]
                save(path,out)
            predictions.extend(out)
            save(folder/'progress.json',{'completed':len(predictions),'total':len(values),'time':time.time()})
    assert [r['uid'] for r in predictions]==[r['global_uid'] for r in values]
    reference=W/'data/general_retention_v1/sources/vlmevalkit/vqa_eval.py'
    assert sha(reference)=='c04e0a1cad29cd8caa786ea1a96d2d945630304d756d1bba234b36e66c83f06d'
    normalize=_reference_digit_article(reference);items=[]
    for row,pred in zip(values,predictions):
        if row['metric']=='multiple_choice_accuracy':
            parsed,unclosed=parse_multiple_choice(pred['response'],row['choices']);credit=float(parsed==row['answer'])
        else:
            parsed,unclosed=_answer_span(pred['response']);credit=_vqa_soft_credit(parsed,row['answer'],normalize) if parsed is not None else 0.
        if pred['finish_reason']!='stop' or unclosed:credit=0.
        items.append({'uid':row['global_uid'],'dataset':row['dataset'],'credit':credit,
            'image_sha256':row['image_sha256'],'capped':pred['finish_reason']!='stop','parsed':parsed is not None})
    by={name:sum(r['credit'] for r in items if r['dataset']==name)/sum(r['dataset']==name for r in items)
        for name in ('ai2d_test','scienceqa_test','textvqa_val')}
    save(folder/'predictions.json',predictions);save(folder/'result.json',{'benchmarks':by,'macro':sum(by.values())/3,'items':items})
    save(folder/'done.json',{'prediction_sha256':sha(folder/'predictions.json'),'result_sha256':sha(folder/'result.json')})


def retention_initial():
    RETENTION.mkdir(parents=True,exist_ok=True)
    with (RETENTION/'initial.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RETENTION/'initial/done.json').exists():return
        # Start in the first lane's evaluation interval, after its two-GPU
        # training has released GPU 6. The next training checks both GPUs.
        while not (ROOT/'semantic_weighted_17/training_done.json').exists():
            if (ROOT/'semantic_weighted_17/failure.json').exists():
                raise RuntimeError('Core training failed before the baseline inference slot')
            time.sleep(20)
        while int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))>512:time.sleep(20)
        os.environ['CUDA_VISIBLE_DEVICES']='6'
        from anvil import _native_engine
        initial=read_binding(SOURCE/'linked/training_done.json')
        engine=_native_engine(MODEL,Path(initial['adapter']),Path(tempfile.mkdtemp(prefix='initial-infer-',dir=RETENTION)),max_new_tokens=64)
        retention_infer(engine,'initial',initial['adapter_sha256'])


def quality_check(row,pred):
    import re
    from trains.flywheel.validators import validate
    text=re.sub(r'^\s*<think>\s*</think>\s*','',pred['response']).strip()
    match=re.fullmatch(r'<GROUND>(.*?)</GROUND>\s*<RELATE>(.*?)</RELATE>',text,re.S)
    syntax=False;typed=False
    if match and pred['finish_reason']=='stop':
        try:
            payloads=[json.loads(match.group(i)) for i in (1,2)];syntax=True
            typed=all(validate(form,payload,context)[0] for form,payload,context in
                      zip(('detection','depth'),payloads,row['gsa_stage_contexts']))
        except (ValueError,TypeError,KeyError,IndexError):pass
    score=score_linked([row],[pred])['items'][0]
    reason=('invalid stage serialization or JSON' if not syntax else
            'output violates typed field or entity requirements' if not typed else
            'G: localization does not match the requested entities' if score['G']<.75 else
            'S: depth order is inconsistent with the scene' if score['S']<=.5 else 'accepted')
    return {'syntax':syntax,'typed':bool(typed),'source_correct':bool(score['strict_chain']),
            'G':score['G'],'S':score['S'],'feedback':reason}


def quality_prepare():
    QUALITY.mkdir(exist_ok=True)
    model=Path('/path/to/external_environment/hf_models/Qwen/Qwen3.6-27B')
    manifest={'source':str(SOURCE/'train_linked.jsonl'),'source_sha256':sha(SOURCE/'train_linked.jsonl'),
        'records':2011,'model':str(model),'model_config_sha256':sha(model/'config.json'),
        'generation':{'batch':4,'max_new_tokens':512,'temperature':0,'seed':17,'image_tokens':1536,'max_length':4096},
        'split':'Only previously frozen training scenes; dev and evaluation never generate training candidates.',
        'rounds':'One common initial output; exactly one additional call per failed source check in each of no-feedback and named-feedback arms. Already correct first outputs reused.',
        'metric':'Correctness against independent official instance/depth targets, not human precision. Final source-verified retention and conditional correctness have the same acceptance criterion and are not an independent verifier-accuracy estimate.',
        'human_review':'Historical 320 judgments reported separately; no new review or paid calls.'}
    if (QUALITY/'manifest.json').exists():assert read_binding(QUALITY/'manifest.json')==manifest
    else:save(QUALITY/'manifest.json',manifest)
    return manifest


def quality_generate(shard):
    from copy import deepcopy
    import torch
    manifest=quality_prepare();root=QUALITY/f'shard_{shard}';root.mkdir(exist_ok=True)
    with (root/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'done.json').exists():return
        gpus=[0,1] if shard==0 else [2,3]
        while True:
            out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(a):int(b) for a,b in (line.split(',') for line in out.splitlines())}
            if all(memory[g]<512 for g in gpus):break
            time.sleep(20)
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
        from anvil import _native_infer_batches
        from swift.pipelines.infer.infer import SwiftInfer
        command=['--model',manifest['model'],'--infer_backend','transformers','--device_map','auto',
            '--max_batch_size','4','--max_new_tokens','512','--max_length','4096','--temperature','0',
            '--seed','17','--enable_thinking','false','--stream','false','--load_args','false','--torch_dtype','bfloat16']
        engine=SwiftInfer(command)
        source=rows(Path(manifest['source']))[shard::2]
        predictions={};checks={}
        for condition in ('initial','no_feedback','feedback'):
            selected=source if condition=='initial' else [r for r in source if not checks[r['sample_id']]['source_correct']]
            dest=root/condition;dest.mkdir(exist_ok=True);chunks=dest/'chunks';chunks.mkdir(exist_ok=True)
            requests=[]
            for row in selected:
                row=deepcopy(row)
                if condition!='initial':
                    old=predictions[row['sample_id']]['response']
                    note=('Please review the previous answer and return a corrected complete response.' if condition=='no_feedback'
                          else 'The verifier reports: '+checks[row['sample_id']]['feedback']+'. Correct this failure and return a complete response.')
                    row['messages'][0]['content']+='\nPrevious answer:\n'+old+'\n'+note
                requests.append(row)
            binding={'manifest_sha256':sha(QUALITY/'manifest.json'),'shard':shard,'condition':condition,
                'request_ids':[r['sample_id'] for r in requests]}
            save(dest/'binding.json',binding);spool=prediction_spool(dest,binding)
            for i in range(0,len(requests),4):
                batch=requests[i:i+4];path=chunks/f'{i:06d}.jsonl';partial=spool/f'{i:06d}.jsonl'
                if not path.exists():
                    if not spool_batch_ready(partial,batch):_native_infer_batches(engine,batch,partial)
                    commit_prediction(partial,path,batch)
                assert [r['sample_id'] for r in rows(path)]==[r['sample_id'] for r in batch]
                save(dest/'progress.json',{'completed':i+len(batch),'total':len(requests),'time':time.time()})
            output=dest/'predictions.jsonl';commit_bytes(output,b''.join(p.read_bytes() for p in sorted(chunks.glob('*.jsonl'))))
            result={r['sample_id']:r for r in rows(output)}
            if condition=='initial':
                predictions=result;checks={r['sample_id']:quality_check(r,result[r['sample_id']]) for r in source}
                save(root/'initial_checks.json',checks)
            save(dest/'done.json',{'rows':len(result),'prediction_sha256':sha(output)})
        save(root/'done.json',{'source_rows':len(source),'time':time.time()})


def quality_supervise():
    quality_prepare()
    with (QUALITY/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handles=[]
        for i in (0,1):
            log=(QUALITY/f'shard_{i}.log').open('a')
            handles.append((subprocess.Popen([sys.executable,__file__,'quality-generate','--lane',str(i)],stdout=log,stderr=subprocess.STDOUT),log))
        codes=[p.wait() for p,_ in handles]
        for _,log in handles:log.close()
        save(QUALITY/'generation_finished.json',{'exit_codes':codes,'time':time.time()})
        if any(codes):raise RuntimeError('Quality generation failed; all completed chunks retained')
        quality_finish()


def quality_finish():
    QUALITY.mkdir(exist_ok=True)
    with (QUALITY/'training_queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (QUALITY/'queue_finished.json').exists() and not any(read_binding(QUALITY/'queue_finished.json')['exit_codes']):return
        while not (QUALITY/'generation_finished.json').exists():time.sleep(20)
        assert not any(read_binding(QUALITY/'generation_finished.json')['exit_codes']),'Generation must complete before downstream training'
        quality_materialize()
        handles=[]
        for i in range(4):
            log=(QUALITY/f'training_lane_{i}.log').open('a')
            handles.append((subprocess.Popen([sys.executable,__file__,'quality-lane','--lane',str(i)],stdout=log,stderr=subprocess.STDOUT),log))
        codes=[p.wait() for p,_ in handles]
        for _,log in handles:log.close()
        save(QUALITY/'queue_finished.json',{'exit_codes':codes,'time':time.time()})
        if any(codes):raise RuntimeError('Quality training incomplete; saved checkpoints retained')


def recover_quality_feedback():
    """Reuse a complete terminal adapter after optimizer-only save failure."""
    import torch
    from safetensors.torch import load_file
    name='feedback';folder=QUALITY/'training'/name
    with (folder/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (folder/'training_done.json').exists():
            checkpoints=list((folder/'training').glob('v*/checkpoint-256'))
            assert len(checkpoints)==1
            adapter=checkpoints[0]
            assert not (adapter/'trainer_state.json').exists()
            model_state=adapter/'global_step256/mp_rank_00_model_states.pt'
            stored=torch.load(model_state,map_location='cpu',weights_only=False)
            assert stored['global_steps']==256 and stored['global_samples']==1024
            assert stored['skipped_steps']==0
            weights=load_file(adapter/'adapter_model.safetensors')
            expected={k.replace('.default.','.'):v for k,v in stored['module'].items()}
            assert set(weights)==set(expected)
            assert all(torch.equal(v,expected[k]) and torch.isfinite(v).all().item()
                       for k,v in weights.items())
            log=adapter.parent/'logging.jsonl'
            terminal=[r for r in rows(log) if r.get('global_step/max_steps')=='256/256']
            assert len(terminal)==1 and 'eval_loss' in terminal[0]
            tokens=terminal[0]['num_input_tokens_seen'];assert isinstance(tokens,int) and tokens>0
            source=read_binding(QUALITY/'training_manifest.json')['training_files'][name]
            assert sha(source['path'])==source['sha256']
            evidence={'global_step':256,'global_samples':1024,'num_input_tokens_seen':tokens,
                'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
                'model_state_sha256':sha(model_state),'terminal_log_sha256':sha(log),
                'matching_adapter_tensors':len(weights),'optimizer_checkpoint_complete':False,
                'provenance':'Observed DeepSpeed model state and terminal evaluation log; not a reconstructed Hugging Face trainer state. Disk quota interrupted subsequent optimizer serialization. Every saved adapter tensor matches the step-256 model state exactly.'}
            state_path=folder/'terminal_state_recovery.json';save(state_path,evidence)
            save(folder/'training_done.json',{'adapter':str(adapter),
                'adapter_sha256':evidence['adapter_sha256'],'trainer_state_path':str(state_path),
                'trainer_state_sha256':sha(state_path),'num_input_tokens_seen':tokens,
                'global_step':256,'source_sha256':source['sha256'],
                'optimizer_checkpoint_complete':False,'terminal_adapter_recovered':True})
            del weights,expected,stored
        if not (folder/'done.json').exists():
            while int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True))<32000:time.sleep(20)
            subprocess.run([sys.executable,__file__,'quality-evaluate','--job',name,'--gpu','0'],check=True)
    # The original parent finishes its other live evaluations before this lock is available.
    with (QUALITY/'training_queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert all((QUALITY/'training'/arm/'done.json').exists() for arm in (*QUALITY_ARMS,*SCALE_ARMS))
        for arm in (*QUALITY_ARMS,*SCALE_ARMS):verified_result(QUALITY/'training'/arm)
        old=QUALITY/'queue_finished.json'
        if old.exists():old.replace(QUALITY/f'queue_before_terminal_recovery_{time.time_ns()}.json')
        save(old,{'exit_codes':[0,0,0,0],'time':time.time(),
            'recovery':'Original failed queue retained; feedback terminal adapter verified and evaluated without retraining; all seven cells independently accepted.'})


def storage_recovery():
    """Restart only failed/idle lanes; preserve the already running declared arm."""
    with (ROOT/'storage_recovery.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('structural_17','semantic_uniform_17'):
            folder=ROOT/name;failure=folder/'failure.json'
            if failure.exists():
                assert 'Disk quota exceeded' in (folder/'train.log').read_text()
                failure.replace(folder/f'failure_storage_history_{time.time_ns()}.json')
        handles=[]
        def start_lane(index):
            with (ROOT/f'lane_{index}.lock').open('a') as lane_lock:
                try:fcntl.flock(lane_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:return
            log=(ROOT/f'storage_retry_lane_{index}.log').open('a')
            handles.append((subprocess.Popen([sys.executable,__file__,'lane','--lane',str(index)],
                                            stdout=log,stderr=subprocess.STDOUT),log))
        start_lane(1)
        recover_quality_feedback()
        start_lane(2);start_lane(3)
        quality_results()
        while not all((ROOT/j['id']/'done.json').exists() for j in JOBS):
            if list(ROOT.glob('*/failure.json')):
                raise RuntimeError('A retried core cell failed; completed results remain available')
            if any(p.poll() not in (None,0) for p,_ in handles):
                raise RuntimeError('A recovery lane exited unsuccessfully')
            time.sleep(30)
        for process,log in handles:
            assert process.wait()==0;log.close()
        # Preserve the old supervisor's terminal codes before writing the recovered queue.
        with (ROOT/'queue.lock').open('a') as queue_lock:
            fcntl.flock(queue_lock,fcntl.LOCK_EX)
            for job in JOBS:
                verified_result(ROOT/job['id']);checked_retention(job['id'])
            old=ROOT/'queue_finished.json'
            if old.exists():old.replace(ROOT/f'queue_before_storage_recovery_{time.time_ns()}.json')
            save(old,{'exit_codes':[0,0,0,0],'time':time.time(),
                'recovery':'Failed storage attempts retained; all ten completed cells verified after retries. Healthy declared training was not interrupted.'})
        finalize_campaign();finish_evidence();finish_submission()


QUALITY_ARMS=('raw','syntax','typed','round2','feedback')
SCALE_ARMS=('feedback_25','feedback_50')


def scaling_prepare():
    """Nested source coverage at fixed updates, frozen before downstream results."""
    import collections, hashlib
    from linked_training import write_rows
    with (QUALITY/'scaling_prepare.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (QUALITY/'scaling_manifest.json').exists():return read_binding(QUALITY/'scaling_manifest.json')
        manifest=quality_materialize();full=manifest['training_files']['feedback']
        assert sha(full['path'])==full['sha256']
        ordered=sorted(full['sample_ids'],key=lambda uid:hashlib.sha256(('quality-order|'+uid).encode()).hexdigest())
        values=rows(full['path']);assert len(ordered)==len(values)
        byid=dict(zip(ordered,values));source={r['sample_id']:r for r in rows(SOURCE/'train_linked.jsonl')}
        heldout={r['sample_id'] for r in rows(SOURCE/'qualification_linked.jsonl')}
        assert set(ordered).isdisjoint(heldout)
        for uid,value in byid.items():
            assert value['images']==source[uid]['images']
            assert value['messages'][0]==source[uid]['messages'][0]
        groups=collections.defaultdict(list)
        for uid in ordered:groups[source[uid]['gsa_meta']['scene_group']].append(uid)
        scenes=sorted(groups,key=lambda s:hashlib.sha256(('scale-17|'+s).encode()).hexdigest())
        nested=[]
        for i in range(max(map(len,groups.values()))):
            nested.extend(groups[s][i] for s in scenes if i<len(groups[s]))
        assert set(nested)==set(ordered)
        files={}
        for name,fraction in zip(SCALE_ARMS,(.25,.5)):
            ids=nested[:max(1,int(len(nested)*fraction))]
            path=QUALITY/(name+'_train.jsonl');write_rows(path,[byid[uid] for uid in ids])
            files[name]={'path':str(path),'sha256':sha(path),'rows':len(ids),'sample_ids':ids,
                'physical_scenes':len({source[uid]['gsa_meta']['scene_group'] for uid in ids})}
        result={'full_source_sha256':full['sha256'],'full_records':len(values),'training_files':files,
            'seed':17,'steps':256,'effective_batch':4,'learning_rate':1e-4,
            'selection':'Deterministic scene round-robin; nested 25% and 50% of the same accepted feedback pool, before downstream evaluation.',
            'budget':'Equal updates and row exposures; actual input tokens measured. This is a source-coverage experiment, not an equal-token scaling law.'}
        save(QUALITY/'scaling_manifest.json',result);return result


def quality_materialize():
    import collections
    import hashlib
    from linked_training import write_rows
    if (QUALITY/'training_manifest.json').exists():return read_binding(QUALITY/'training_manifest.json')
    source=rows(SOURCE/'train_linked.jsonl');byid={r['sample_id']:r for r in source}
    pools={name:{} for name in QUALITY_ARMS};bound={};stats={name:[] for name in QUALITY_ARMS}
    for shard in (0,1):
        root=QUALITY/f'shard_{shard}';predictions={}
        for condition in ('initial','no_feedback','feedback'):
            path=root/condition/'predictions.jsonl';done=read_binding(root/condition/'done.json')
            assert sha(path)==done['prediction_sha256'];bound[str(path)]=sha(path)
            predictions[condition]={r['sample_id']:r for r in rows(path)}
        for uid,first in predictions['initial'].items():
            row=byid[uid];check=quality_check(row,first)
            for name,keep in [('raw',True),('syntax',check['syntax']),('typed',check['typed'])]:
                if keep:pools[name][uid]=first;stats[name].append(check)
            for name,condition in [('round2','no_feedback'),('feedback','feedback')]:
                value=first if check['source_correct'] else predictions[condition][uid]
                after=quality_check(row,value)
                if after['source_correct']:pools[name][uid]=value;stats[name].append(after)
    assert len(pools['raw'])==len(source)==2011
    buckets={name:collections.defaultdict(list) for name in QUALITY_ARMS}
    for name,pool in pools.items():
        for uid in pool:buckets[name][byid[uid]['gsa_meta']['scene_group']].append(uid)
    scenes=sorted(set.intersection(*[set(bucket) for bucket in buckets.values()]))
    counts={scene:min(len(buckets[name][scene]) for name in QUALITY_ARMS) for scene in scenes}
    assert sum(counts.values())>=128,'Too few matched-scene samples for the declared downstream study'
    files={}
    for name in QUALITY_ARMS:
        chosen=[]
        for scene,count in counts.items():
            # Same seeded UID order in each arm; retained candidates differ.
            ordered=sorted(buckets[name][scene],key=lambda uid:hashlib.sha256(('quality-match-v1|17|'+uid).encode()).hexdigest())
            chosen+=ordered[:count]
        train=[]
        for uid in sorted(chosen,key=lambda uid:hashlib.sha256(('quality-order|'+uid).encode()).hexdigest()):
            row=byid[uid]
            train.append({'images':row['images'],'messages':[row['messages'][0],{'role':'assistant','content':pools[name][uid]['response']}]})
        path=QUALITY/(name+'_train.jsonl');write_rows(path,train)
        files[name]={'path':str(path),'sha256':sha(path),'rows':len(train),'sample_ids':chosen}
    report={'source_sha256':sha(SOURCE/'train_linked.jsonl'),'predictions':bound,'training_files':files,
        'matched_physical_scene_counts':counts,'matched_records_per_arm':sum(counts.values()),
        'matching':'Equal retained records per physical scene, seeded UID selection; individual images need not match. Same updates; actual tokens recorded.',
        'precision_scope':'Conditional correctness against the source. Round2/feedback use the source criterion for admission, so their accepted correctness is not independent verifier accuracy.',
        'full_pool':{name:{'candidates':len(source),'retained':len(stats[name]),'source_correct':sum(v['source_correct'] for v in stats[name])}
                     for name in QUALITY_ARMS},'no_new_human_review':True}
    save(QUALITY/'training_manifest.json',report);return report


def quality_lane(index):
    manifest=quality_materialize();scaling=scaling_prepare()
    toolkit=Path('/path/to/external_environment/env/anvil_cuda121')
    assigned=list(QUALITY_ARMS[index::4])
    if index in (1,2):assigned.append(SCALE_ARMS[index-1])
    for name in assigned:
        folder=QUALITY/'training'/name;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (folder/'done.json').exists():continue
            while int(subprocess.check_output(['nvidia-smi',f'--id={index}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))>512:time.sleep(20)
            source=(scaling if name in SCALE_ARMS else manifest)['training_files'][name]
            assert sha(source['path'])==source['sha256']
            if not (folder/'training_done.json').exists():
                env={**os.environ,'PATH':str(PY.parent)+os.pathsep+str(toolkit/'bin')+os.pathsep+os.environ['PATH'],
                    'CUDA_HOME':str(toolkit),'LD_LIBRARY_PATH':str(toolkit/'lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''),
                    'MAX_JOBS':'4','GSA_ROOT':str(W),'MODEL':str(MODEL),'CUDA_VISIBLE_DEVICES':str(index),
                    'MASTER_PORT':str(29930+index),'TRAIN':source['path'],'VAL':str(SOURCE/'dev_linked_messages.jsonl'),
                    'OUT':str(folder/'training'),'BS':'1','GA':'4','RANK':'64','ALPHA':'128','DROPOUT':'0.05',
                    'SEED':'17','LR':'0.0001','STEPS':'256','EVAL_STEPS':'128','SAVE_STEPS':'128','HOURS':'0',
                    'REPORT_TO':'none','COUNT_INPUT_TOKENS':'1','DATALOADER_NUM_WORKERS':'2','DATASET_NUM_PROC':'2',
                    'IMAGE_MAX_TOKEN_NUM':'1536','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
                env.pop('INITIAL_ADAPTER',None)
                save(folder/'launch.json',{'training_manifest_sha256':sha(QUALITY/('scaling_manifest.json' if name in SCALE_ARMS else 'training_manifest.json')),
                    'arm':name,'gpu':index,'steps':256,'seed':17,'matched_records':source['rows'],'no_wallclock_kill':True})
                with (folder/'train.log').open('a') as log:
                    subprocess.run(['bash',str(W/'scripts/run_sft_9b_lora_l20_dlc.sh')],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                checkpoints=list((folder/'training').glob('v*/checkpoint-256'));assert len(checkpoints)==1
                adapter=checkpoints[0];state=read_binding(adapter/'trainer_state.json');assert state['global_step']==256
                save(folder/'training_done.json',{'adapter':str(adapter),'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
                    'trainer_state_sha256':sha(adapter/'trainer_state.json'),'num_input_tokens_seen':state.get('num_input_tokens_seen'),
                    'global_step':256,'source_sha256':source['sha256']})
            subprocess.run([sys.executable,__file__,'quality-evaluate','--job',name,'--gpu',str(index)],check=True)


def prepare():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'prepare.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (ROOT/'manifest.json').exists():return read_binding(ROOT/'manifest.json')
        contexts=context_status()
        if not all(contexts[f][c]['state']=='complete' for f in ('S','joint') for c in ('student','shifted')):
            raise RuntimeError('WAIT_CONTEXTS')
        evidence={f:contexts[f] for f in ('S','joint')}
        evidence['root_G']={'source':str(SOURCE/'G/joint/result.json'),'sha256':sha(SOURCE/'G/joint/result.json'),
            'note':'G has no upstream stage; its atomic grounding comparison is reused.'}
        save(ROOT/'context_evidence.json',evidence)
        teachers={f:{'adapter':read_binding(SOURCE/f/'training_done.json')['adapter'],
                     'sha256':read_binding(SOURCE/f/'training_done.json')['adapter_sha256']} for f in ('G','S')}
        initial=read_binding(SOURCE/'linked/training_done.json')
        binding={'schema':'anvil-gs-mechanism-binding-v1','full_GSA_qualified':False,
            'scope':'Two-stage source-grounded mechanism comparison, independent Action results remain separate.',
            'teacher_policy':'Run measured G/S specialists even when a superiority contrast is inconclusive; report quality rather than require favorable evidence.',
            'teachers':teachers,'student':{'adapter':initial['adapter'],'sha256':initial['adapter_sha256']},
            'training':{'path':str(SOURCE/'train_linked.jsonl'),'sha256':sha(SOURCE/'train_linked.jsonl')},
            'context_evidence':{'path':str(ROOT/'context_evidence.json'),'sha256':sha(ROOT/'context_evidence.json')}}
        save(ROOT/'study_binding.json',binding)
        manifest={'study_binding_sha256':sha(ROOT/'study_binding.json'),'jobs':JOBS,
            'evaluation':{'path':str(SOURCE/'qualification_linked.jsonl'),'sha256':sha(SOURCE/'qualification_linked.jsonl'),
                'records':282,'physical_scenes':50,'previously_observed_source_split':True},
            'common':{'updates':256,'effective_batch':4,'lr':5e-6,'max_length':3072,'max_completion_length':256,
                'image_tokens':1536,'topk':128,'jsd_beta':0.5,'SFT_weight':1,'MTD_weight':1,
                'selection':'terminal checkpoint; no evaluation-dependent arm/checkpoint selection',
                'budget':'Equal updates, common maximum rollout/context budget; actual tokens/calls logged, not claimed identical.'},
            'full_MTD_GSA_completed':False,'no_new_human_annotation':True}
        save(ROOT/'manifest.json',manifest);return manifest


def train(job,gpus,teacher_deepspeed='zero3'):
    manifest=prepare();folder=ROOT/job['id'];folder.mkdir(exist_ok=True)
    if (folder/'training_done.json').exists():return
    binding=read_binding(ROOT/'study_binding.json');selector,alpha=ARMS[job['arm']]
    assert teacher_deepspeed in ('zero3','zero3_offload')
    toolkit=Path('/path/to/external_environment/env/anvil_cuda121')
    env={**os.environ,'PATH':str(PY.parent)+os.pathsep+str(toolkit/'bin')+os.pathsep+os.environ['PATH'],'GSA_ROOT':str(W),'ANVIL_ROOT':str(A),
        'CUDA_HOME':str(toolkit),'LD_LIBRARY_PATH':str(toolkit/'lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''),
        'MAX_JOBS':'4','TEACHER_DEEPSPEED':teacher_deepspeed,
        'MODEL':str(MODEL),'TRAIN':binding['training']['path'],'OUT':str(folder/'training'),
        'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)),'MASTER_PORT':str(29900+JOBS.index(job)),
        'ANVIL_EXPERIMENT_SCOPE':'GS_MECHANISM','ANVIL_MTD':'1','ANVIL_ROUTING_MODE':selector,
        'ANVIL_STRUCTURAL_ALPHA':str(alpha),'ANVIL_STAGE_EVALUATOR':'trains.mopd.anvil_runtime:reference_geometry_evaluator',
        'TEACHER_ELIGIBILITY_JSON':str(ROOT/'study_binding.json'),'STUDENT_ADAPTER':binding['student']['adapter'],
        'GSA_TEACHER_G_ADAPTER':binding['teachers']['G']['adapter'],'GSA_TEACHER_S_ADAPTER':binding['teachers']['S']['adapter'],
        'BS':'1','GA':str(4//len(gpus)),'STEPS':'256','SEED':str(job['seed']),'DATA_SEED':str(job['seed']),
        'LR':'0.000005','HOURS':'0','REPORT_TO':'none','SAVE_STEPS':'128','SAVE_TOTAL_LIMIT':'2',
        'MAX_LENGTH':'3072','MAX_COMPLETION_LENGTH':'256','IMAGE_MAX_TOKEN_NUM':'1536',
        'WORKERS':'2','DATA_PROC':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
    command=W/'scripts/run_mopd_9b_lora_l20.sh'
    receipt={'job':job,'manifest_sha256':sha(ROOT/'manifest.json'),'launcher_sha256':sha(command),
        'runtime_sha256':sha(W/'trains/mopd/anvil_runtime.py'),'plugin_sha256':sha(W/'scripts/gsa_mopd_plugin.py'),
        'gpus':gpus,'settings':manifest['common'],'cuda_compiler_sha256':sha(toolkit/'bin/nvcc')}
    if teacher_deepspeed!='zero3':
        config=Path('/path/to/external_environment/ms-swift/swift/config')/(teacher_deepspeed+'.json')
        receipt['execution']={'teacher_deepspeed':teacher_deepspeed,'config_sha256':sha(config),
            'reason':'Frozen teacher parameter CPU offload to share GPUs; model, objective, data and update budget unchanged.'}
    if (folder/'launch.json').exists() and read_binding(folder/'launch.json')!=receipt:
        assert not list((folder/'training').glob('v*/checkpoint-*/trainer_state.json')),'Refuse changed launch over a saved optimizer state'
        (folder/'launch.json').replace(folder/f'launch_before_environment_repair_{time.time_ns()}.json')
    if not (folder/'launch.json').exists():save(folder/'launch.json',receipt)
    # A scheduler can restart while its already-running trainer completes.
    # Reuse a verified terminal checkpoint instead of launching training again.
    checkpoints=list((folder/'training').glob('v*/checkpoint-256'))
    if checkpoints:
        assert len(checkpoints)==1
        adapter=checkpoints[0];state=read_binding(adapter/'trainer_state.json')
        assert state['global_step']==256
        assert (adapter/'optimizer.pt').exists() or any(adapter.glob('global_step*'))
        save(folder/'training_done.json',{'adapter':str(adapter),'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
            'trainer_state_sha256':sha(adapter/'trainer_state.json'),'seconds':None,
            'global_step':256,'job':job,'full_GSA_qualified':False,
            'recovered_terminal_checkpoint':True,'duration_source':'original trainer log; scheduler wall time unavailable'})
        return
    started=time.monotonic()
    with (folder/'train.log').open('a') as log:
        subprocess.run(['bash',str(command)],env=env,cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
    checkpoints=list((folder/'training').glob('v*/checkpoint-256'))
    assert len(checkpoints)==1,'Missing unique terminal checkpoint'
    adapter=checkpoints[0];state=read_binding(adapter/'trainer_state.json')
    assert state['global_step']==256
    save(folder/'training_done.json',{'adapter':str(adapter),'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
        'trainer_state_sha256':sha(adapter/'trainer_state.json'),'seconds':time.monotonic()-started,
        'global_step':state['global_step'],'job':job,'full_GSA_qualified':False})


def shared_core_job(job,gpus):
    """Run one existing, unlaunched cell with frozen-teacher CPU offload."""
    folder=ROOT/job['id'];folder.mkdir(parents=True,exist_ok=True)
    with (folder/'job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'done.json').exists():return
        if (folder/'launch.json').exists():
            launch=read_binding(folder/'launch.json')
            assert launch['gpus']==gpus and launch.get('execution',{}).get('teacher_deepspeed')=='zero3_offload'
        while True:
            out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(a):int(b) for a,b in (line.split(',') for line in out.splitlines())}
            if all(free[g]>=40000 for g in gpus):break
            time.sleep(20)
        save(folder/'shared_start.json',{'time':time.time(),'gpus':gpus,
            'free_memory_mib':{str(g):free[g] for g in gpus},'minimum_free_mib':40000,
            'foreign_workloads_stopped':False})
        try:
            train(job,gpus,teacher_deepspeed='zero3_offload')
            subprocess.run([sys.executable,__file__,'evaluate','--job',job['id'],'--gpu',str(gpus[0])],check=True)
        except Exception as exc:
            save(folder/'failure.json',{'error':repr(exc),'time':time.time()});raise


def retention_model(name,gpu):
    """Use the released partner GPU while native evaluation runs on the first."""
    folder=RETENTION/name;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'done.json').exists():return
        done=read_binding(ROOT/name/'training_done.json')
        assert sha(Path(done['adapter'])/'adapter_model.safetensors')==done['adapter_sha256']
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
        from anvil import _native_engine
        engine=_native_engine(MODEL,Path(done['adapter']),Path(tempfile.mkdtemp(prefix='infer-',dir=folder)),max_new_tokens=64)
        retention_infer(engine,name,done['adapter_sha256'])


def evaluate(job,gpu,folder_override=None):
    from contextlib import ExitStack
    from anvil import _native_engine, _native_infer_batches
    folder=folder_override or ROOT/job['id'];done=read_binding(folder/'training_done.json')
    if (folder/'done.json').exists():return
    source=SOURCE/'qualification_linked.jsonl';values=rows(source)
    assert sha(source)==read_binding(ROOT/'manifest.json')['evaluation']['sha256']
    adapter=Path(done['adapter']);assert sha(adapter/'adapter_model.safetensors')==done['adapter_sha256']
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    with ExitStack() as stack:
        retention_worker=None
        # GPU 6 is reserved for initial-model retention in the first interval.
        # Subsequent jobs own their recorded pair until both evaluations finish.
        if folder_override is None and job['id']!='semantic_weighted_17' and not (RETENTION/job['id']/'done.json').exists():
            pair=read_binding(folder/'launch.json')['gpus'];assert gpu in pair and len(pair)==2
            partner=next(g for g in pair if g!=gpu)
            free=int(subprocess.check_output(['nvidia-smi',f'--id={partner}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            # This lane still owns the pair. A small unrelated resident process
            # need not serialize independent panels when inference has ample room.
            # Use the same conservative free-memory floor as shared core training.
            if free>=40000:
                log=stack.enter_context((folder/'retention_parallel.log').open('a'))
                retention_worker=stack.enter_context(subprocess.Popen(
                    [sys.executable,__file__,'retention-model','--job',job['id'],'--gpu',str(partner)],
                    stdout=log,stderr=subprocess.STDOUT))
                save(folder/'evaluation_schedule.json',{'native_gpu':gpu,'retention_gpu':partner,
                    'retention_initial_free_mib':free,'retention_min_free_mib':40000,
                    'protocol':'Same fixed manifests, batch sizes and decoding; concurrent independent panels.'})
        engine=_native_engine(MODEL,adapter,Path(tempfile.mkdtemp(prefix='infer-',dir=folder)),max_new_tokens=512)
        dest=folder/'evaluation';dest.mkdir(exist_ok=True);chunks=dest/'chunks';chunks.mkdir(exist_ok=True)
        binding={'source_sha256':sha(source),'adapter_sha256':done['adapter_sha256'],'output_tokens':512}
        spool=prediction_spool(dest,binding)
        for i in range(0,len(values),4):
            batch=values[i:i+4];path=chunks/f'{i:06d}.jsonl';partial=spool/f'{i:06d}.jsonl'
            if path.exists():assert [r['sample_id'] for r in rows(path)]==[r['sample_id'] for r in batch]
            else:
                if not spool_batch_ready(partial,batch):_native_infer_batches(engine,batch,partial)
                commit_prediction(partial,path,batch)
            save(dest/'progress.json',{'completed':i+len(batch),'total':len(values),'time':time.time()})
        pred=dest/'predictions.jsonl';commit_bytes(pred,b''.join(p.read_bytes() for p in sorted(chunks.glob('*.jsonl'))))
        save(dest/'result.json',score_linked(values,rows(pred)))
        if folder_override is None:
            if retention_worker is not None:
                assert retention_worker.wait()==0,'Parallel VQA evaluation failed; saved outputs retained'
                checked_retention(job['id'])
            else:retention_infer(engine,job['id'],done['adapter_sha256'])
        save(folder/'done.json',{'source_sha256':sha(source),'prediction_sha256':sha(pred),
            'result_sha256':sha(dest/'result.json'),'training_done_sha256':sha(folder/'training_done.json'),
            'native_evaluation_complete':True,'VQA_retention_complete':folder_override is None})


def status():
    result={}
    for job in JOBS:
        folder=ROOT/job['id'];state='queued'
        if (folder/'done.json').exists():state='complete' if read_binding(folder/'done.json').get('VQA_retention_complete') else 'native_complete_VQA_pending'
        elif (folder/'failure.json').exists():state='failed'
        elif (folder/'training_done.json').exists():state='evaluating'
        elif (folder/'launch.json').exists():
            with (folder/'job.lock').open('a') as job_lock:
                try:
                    fcntl.flock(job_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    state='queued_retry'
                except BlockingIOError:state='training'
        result[job['id']]={'state':state}
        if (folder/'training_done.json').exists():
            result[job['id']]['saved_terminal_update']=read_binding(folder/'training_done.json')['global_step']
        for panel,path in (('native',folder/'evaluation/progress.json'),
                           ('VQA',RETENTION/job['id']/'progress.json')):
            if path.exists():
                progress=read_binding(path)
                result[job['id']][panel+'_progress']={k:progress[k] for k in ('completed','total')}
        histories=list((folder/'training').glob('v*/logging.jsonl'))
        if histories:
            latest=max(histories,key=lambda p:p.stat().st_mtime)
            history=rows(latest)
            updates=[r for r in history if 'global_step/max_steps' in r and 'loss' in r]
            if updates:
                last=updates[-1]
                result[job['id']]['last_logged_update']={k:last[k] for k in
                    ('global_step/max_steps','loss','grad_norm','elapsed_time','remaining_time','train_speed(s/it)') if k in last}
                result[job['id']]['last_log_utc_seconds']=latest.stat().st_mtime
        failed_at=max((read_binding(p).get('time',0)
                       for p in folder.glob('failure*history*.json')),default=0)
        stale_retry_log=(state=='training' and failed_at>
                         result[job['id']].get('last_log_utc_seconds',0))
        if stale_retry_log:result[job['id']]['state']='retry_initializing'
        if state=='queued_retry' or stale_retry_log:
            if 'last_logged_update' in result[job['id']]:
                result[job['id']]['previous_attempt_last_update']=result[job['id']].pop('last_logged_update')
            saved=list((folder/'training').glob('v*/checkpoint-*/trainer_state.json'))
            result[job['id']]['retry_starts_from_update']=max(
                (read_binding(path)['global_step'] for path in saved),default=0)
        if (folder/'evaluation/result.json').exists():result[job['id']]['score']=read_binding(folder/'evaluation/result.json')['mean']
    ROOT.mkdir(exist_ok=True);save(ROOT/'status.json',result);print(json.dumps(result),flush=True);return result


def lane(index):
    pairs=[[5,6],[4,7],[0,1],[2,3]];gpus=pairs[index]
    teacher_deepspeed=os.environ.get('ANVIL_CORE_TEACHER_DEEPSPEED','zero3_offload')
    assert teacher_deepspeed in ('zero3','zero3_offload')
    ROOT.mkdir(exist_ok=True)
    with (ROOT/f'lane_{index}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if index>=2:
            while not (QUALITY/'queue_finished.json').exists():time.sleep(20)
            if any(read_binding(QUALITY/'queue_finished.json')['exit_codes']):
                raise RuntimeError('Quality queue failed; its GPU ownership requires inspection')
        order=(JOBS[index::2]+JOBS[1-index::2]) if index<2 else list(reversed(JOBS))
        while not all((ROOT/j['id']/'done.json').exists() for j in JOBS):
            progressed=False
            for job in order:
                folder=ROOT/job['id'];folder.mkdir(exist_ok=True)
                if (folder/'done.json').exists():continue
                with (folder/'job.lock').open('w') as job_lock:
                    try:fcntl.flock(job_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if (folder/'done.json').exists():continue
                    # A saved launch owns its original pair until completion.
                    # Helpers take only unlaunched jobs, avoiding placement drift.
                    if (folder/'launch.json').exists() and read_binding(folder/'launch.json')['gpus']!=gpus:continue
                    while True:
                        used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
                        memory={int(a):{'used':int(b),'free':int(c)} for a,b,c in (line.split(',') for line in used.splitlines())}
                        if all(memory[g]['free']>=40000 if teacher_deepspeed=='zero3_offload' else memory[g]['used']<512 for g in gpus):break
                        time.sleep(20)
                    if (folder/'failure.json').exists():(folder/'failure.json').replace(folder/f'failure_history_{time.time_ns()}.json')
                    try:
                        train(job,gpus,teacher_deepspeed=teacher_deepspeed)
                        subprocess.run([sys.executable,__file__,'evaluate','--job',job['id'],'--gpu',str(gpus[0])],check=True)
                    except Exception as exc:
                        save(folder/'failure.json',{'error':repr(exc),'time':time.time()});raise
                    progressed=True;status()
            if not progressed:
                failures=[j['id'] for j in JOBS if (ROOT/j['id']/'failure.json').exists()]
                if failures:raise RuntimeError('Core jobs failed: '+','.join(failures))
                time.sleep(20)


def supervise():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:prepare();break
            except RuntimeError as e:
                if str(e)!='WAIT_CONTEXTS':raise
                time.sleep(20)
        handles=[]
        for i in range(4):
            log=(ROOT/f'lane_{i}.log').open('a')
            handles.append((subprocess.Popen([sys.executable,__file__,'lane','--lane',str(i)],stdout=log,stderr=subprocess.STDOUT),log))
        codes=[p.wait() for p,_ in handles]
        for _,log in handles:log.close()
        status();save(ROOT/'queue_finished.json',{'exit_codes':codes,'time':time.time()})
        if any(codes):raise RuntimeError('Core campaign needs a diagnosed fix; checkpoints retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','supervise','lane','status','evaluate','shared-core-job','quality-prepare','quality-supervise','quality-generate','quality-materialize','quality-lane','quality-evaluate','quality-finish','quality-recover-feedback','storage-recovery','quality-results','retention-prepare','retention-initial','retention-model','collect-context','collect-core','collect-quality','collect-quality-initial','collect-quality-repair','quality-generation-summary','collect-scaling','finish-evidence','finish-submission','finalize'])
    parser.add_argument('--lane',type=int,choices=range(4));parser.add_argument('--gpu',type=int)
    parser.add_argument('--job',choices=[j['id'] for j in JOBS]+list(QUALITY_ARMS)+list(SCALE_ARMS));args=parser.parse_args()
    {'prepare':prepare,'supervise':supervise,'lane':lambda:lane(args.lane),'status':status,'collect-context':collect_context,
     'collect-core':collect_core,'collect-quality':collect_quality,'collect-quality-initial':collect_quality_initial,'finalize':finalize_campaign,
     'collect-scaling':collect_scaling,'finish-evidence':finish_evidence,'finish-submission':finish_submission,
     'quality-generation-summary':quality_generation_summary,'quality-results':quality_results,
     'collect-quality-repair':collect_quality_repair,
     'quality-prepare':quality_prepare,'quality-supervise':quality_supervise,'quality-generate':lambda:quality_generate(args.lane),
     'quality-materialize':quality_materialize,'quality-lane':lambda:quality_lane(args.lane),
     'quality-finish':quality_finish,'quality-recover-feedback':recover_quality_feedback,
     'storage-recovery':storage_recovery,
     'retention-prepare':retention_prepare,'retention-initial':retention_initial,
     'retention-model':lambda:retention_model(args.job,args.gpu),
     'shared-core-job':lambda:shared_core_job(next(j for j in JOBS if j['id']==args.job),[[5,6],[4,7]][args.lane]),
     'quality-evaluate':lambda:evaluate({'id':args.job},args.gpu,QUALITY/'training'/args.job),
     'evaluate':lambda:evaluate(next(j for j in JOBS if j['id']==args.job),args.gpu)}[args.command]()
