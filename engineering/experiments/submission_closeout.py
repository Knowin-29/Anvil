#!/usr/bin/env python3
"""Finite, resumable local experiments and evidence-only LaTeX backfill.

Visual controls keep every question and target, replacing only its image. They
are controls, never replacement scores for the primary benchmark table.
"""
from __future__ import annotations
import argparse
import errno
from dataclasses import replace
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
import traceback

A = Path(__file__).resolve().parents[2]
W = Path(os.environ.get('ANVIL_WORKTREE', A))
OUT = A / 'engineering/data/submission_closeout_20260915'
sys.path.insert(0, str(W))
BASE = os.environ.get('MODEL_9B', '/path/to/Qwen3.5-9B')
P5 = os.environ.get('ANVIL_P5_MODEL', '/path/to/Anvil-P5-merged')
BENCHES = ['gsa_bench_test', 'blink_relative_depth', 'robospatial_home_context']
JOBS = [dict(id=f'{m}_{b}', kind='visual', model=m, benchmark=b)
        for m in ['base', 'p5'] for b in BENCHES] + [dict(id='compression', kind='compression')]

def save_bytes(p, value):
    """Preserve the previous artifact until a complete replacement is durable."""
    p.parent.mkdir(parents=True, exist_ok=True)
    while True:
        pending = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=p.parent,
                    prefix=p.name+'.', suffix='.pending', delete=False) as stream:
                pending = Path(stream.name)
                stream.write(value); stream.flush(); os.fsync(stream.fileno())
            pending.replace(p)
            return
        except OSError as exc:
            if pending is not None: pending.unlink(missing_ok=True)
            if exc.errno not in (errno.EDQUOT, errno.ENOSPC): raise
            time.sleep(30)

def save_text(p, value):
    save_bytes(p,value.encode('utf-8'))

def save(p, value):
    save_text(p, json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def identity(p, value):
    if p.exists():
        assert json.loads(p.read_text()) == value, f'Resume identity changed: {p}'
    else:
        save(p, value)

def rows(p, repair=False):
    if not p.exists(): return []
    raw = p.read_bytes()
    end = raw.rfind(b'\n') + 1
    if end < len(raw) and repair:
        p.with_name(p.name + f'.interrupted.{time.time_ns()}').write_bytes(raw[end:])
        with p.open('r+b') as f: f.truncate(end)
        raw=raw[:end]
    return [json.loads(s) for s in raw.splitlines() if s.strip()]

def append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())

def frozen_model(path):
    p = Path(path)
    return {'path': str(p.resolve()), 'config': sha(p/'config.json'),
            'index': sha(p/'model.safetensors.index.json')}

def visual(job, folder):
    import torch
    from PIL import Image
    from eval.gsa_eval import runner as f
    from eval.gsa_eval.registry import get_spec
    bench = job['benchmark']; model = BASE if job['model'] == 'base' else P5
    spec = get_spec(bench); samples = list(spec.loader(W/'.cache/eval'))
    # A cycle over unique image hashes prevents a repeated-scene identity mapping.
    images = {sha(p): str(p) for s in samples for p in s.image_paths}
    ordered = sorted(images, key=lambda h: hashlib.sha256(('20260915/'+h).encode()).hexdigest())
    assert len(ordered) > 1
    shuffled = {h: images[ordered[(i+1) % len(ordered)]] for i,h in enumerate(ordered)}
    binds = [{'uid': str(s.uid), 'prompt': s.prompt, 'gt': str(s.gt),
              'images': [sha(p) for p in s.image_paths]} for s in samples]
    identity(folder/'identity.json', {'model': frozen_model(model), 'benchmark': bench,
        'samples': binds, 'shuffle': {h: sha(p) for h,p in shuffled.items()},
        'max_tokens': spec.max_tokens, 'scorer': f.evaluation_code_sha256(),
        'implementation': sha(__file__), 'seed': 20260915, 'thinking': False,
        'controls': ['blank', 'shuffled'], 'decode': 'greedy', 'initial_batch': 4,
        'blank': 'RGB 127; original image dimensions', 'shuffled_fit': 'resize to original dimensions'})
    engine = f._engine(model, max_batch_size=4, thinking=False)
    f._assert_weights_loaded(engine)
    elapsed_start = time.monotonic()
    for condition in ['blank', 'shuffled']:
        resultfile = folder/f'{condition}.json'
        if resultfile.exists(): continue
        transformed = []
        for s in samples:
            paths = []
            for p in s.image_paths:
                h = sha(p)
                target = folder/'images'/condition/(h+'.png')
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(p) as im: size = im.size
                    if condition == 'blank': image = Image.new('RGB', size, (127,127,127))
                    else:
                        with Image.open(shuffled[h]) as im: image = im.convert('RGB').resize(size)
                    image.save(target)
                paths.append(str(target))
            transformed.append(replace(s, image_paths=paths))
        shard = folder/f'{condition}.jsonl'; existing = rows(shard, repair=True)
        assert [str(r['uid']) for r in existing] == [str(s.uid) for s in samples[:len(existing)]]
        offset = len(existing); batch = 4
        with shard.open('a', buffering=1) as stream:
            while offset < len(samples):
                chunk = transformed[offset:offset+batch]
                try:
                    responses, caps = f._infer(engine, chunk, max_tokens=spec.max_tokens)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch == 1: raise
                    batch //= 2
                    print(f'OOM: resume at {offset}, batch={batch}', flush=True)
                    continue
                assert len(responses) == len(caps) == len(chunk)
                for s, response, cap in zip(chunk, responses, caps):
                    append(stream, {'uid': str(s.uid), 'response': response, 'hit_token_cap': cap})
                offset += len(chunk)
                save(folder/'progress.json', {'condition': condition, 'completed': offset,
                    'total': len(samples), 'batch': batch, 'elapsed_s': time.monotonic()-elapsed_start})
                print(f'{condition} {offset}/{len(samples)}', flush=True)
        result = f.score_shards(bench, model, [shard], scratch=W/'.cache/eval',
            max_tokens=spec.max_tokens, thinking=False, generation_batch_size=batch,
            dump_predictions=folder/f'preds_{condition}.jsonl')
        result.update(condition=condition, original_question_and_target=True,
                      output_sha256=sha(shard), identity_sha256=sha(folder/'identity.json'))
        save(resultfile, result)

def compression(folder):
    import torch
    from swift.infer_engine import TransformersEngine
    from swift.utils import to_device
    data = A/'engineering/data/reviewed_native_teachers_v2'
    for family in ['G', 'S', 'A']:
        adapter = A/f'engineering/data/native_teacher_pilot_v2/{family}/selected_adapter'
        source = data/f'qualification_{family}.jsonl'
        examples = rows(source)
        identity(folder/f'{family}_identity.json', {'adapter': sha(adapter/'adapter_model.safetensors'),
            'config': sha(adapter/'adapter_config.json'), 'source': sha(source),
            'implementation': sha(__file__), 'K': [64,128], 'positions': 'all supervised response positions',
            'scope': 'teacher-forced historical pilot; compression fidelity, not downstream MTD'})
        shard = folder/f'{family}.jsonl'; done = rows(shard, repair=True)
        assert [r['uid'] for r in done] == [r['sample_id'] for r in examples[:len(done)]]
        if len(done) == len(examples): continue
        engine = TransformersEngine(BASE, adapters=[str(adapter)], max_batch_size=1,
            model_type='qwen3_5', template_type='qwen3_5', device_map='cuda:0')
        engine.template.set_mode('train')
        engine.template.enable_thinking = False
        engine.template.response_prefix = ''
        engine.model.eval()
        with shard.open('a', buffering=1) as stream:
            for r in examples[len(done):]:
                inputs = engine.template.encode({'messages': r['messages'], 'images': r['images']})
                batch = to_device(engine.template.data_collator([inputs]), 'cuda:0')
                labels = batch.pop('labels')
                positions = torch.where(labels[0] != -100)[0]
                positions = positions[positions > 0]
                with torch.inference_mode():
                    outputs = engine.model(**batch, use_cache=False)
                    logits = outputs.logits[0, positions-1].float()
                    probs = logits.softmax(-1)
                    targets = labels[0,positions]
                    values, indices = probs.topk(128, dim=-1)
                    rec = {'uid': r['sample_id'], 'family': family, 'n_tokens': len(positions), 'K': {}}
                    for k in [64,128]:
                        mass = values[:,:k].sum(-1)
                        hits = (indices[:,:k] == targets[:,None]).any(-1)
                        rec['K'][str(k)] = {'mass_sum': mass.sum().item(),
                            'mass_min': mass.min().item(), 'target_hits': hits.sum().item(),
                            'tail_max': (1-mass).max().item()}
                append(stream, rec)
                save(folder/'progress.json', {'family': family, 'uid': r['sample_id']})
                print(f'compression {family} {r["sample_id"]}', flush=True)
                del outputs, logits, probs, batch, labels
        del engine
        import gc
        gc.collect(); torch.cuda.empty_cache()

def worker(jobid):
    job = next(j for j in JOBS if j['id'] == jobid)
    folder = OUT/jobid; folder.mkdir(parents=True, exist_ok=True)
    with (folder/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder/'complete.json').exists(): return
        try:
            if job['kind'] == 'visual': visual(job, folder)
            else: compression(folder)
            save(folder/'complete.json', {'rc': 0, 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        except Exception:
            save(folder/'failure.json', {'traceback': traceback.format_exc()})
            raise

def collect():
    result = {'jobs': {}, 'blocked_evidence': {
        'full_semantic_MTD': 'Qualified G/S/A specialists and independent Action targets are absent.',
        'semantic_precision_95': 'Requires independent correctness labels; CPU structural yield is a different metric.',
        'teacher_coevolution': 'Existing refresh pilots do not establish qualified-teacher superiority.'}}
    table = [r'\begin{table}[t]', r'\centering\small',
        r'\caption{Visual dependence with frozen questions and targets. Blank images use constant gray; shuffled images come from another scene. Full-split scores (\%). All output failures remain \mbox{in the denominator}.}',
        r'\label{tab:visual-controls}', r'\begin{tabular}{llrrr}', r'\toprule',
        r'Model & Benchmark & Original & Blank & Shuffled \\', r'\midrule']
    expansion=json.loads((A/'engineering/data/public_eval_expansion_20260914/summary.json').read_text())
    originals={(r['model_id'],r['benchmark']):Path(r['result_path']) for r in expansion['cells']}
    index=json.loads((A/'engineering/data/p5_retention_clean_run_seed17_v2/spatial_registry_20260914/result_index.json').read_text())
    for m,rid in [('base','qwen3_5_9b_base'),('p5','anvil_p5_retention_clean_seed17')]:
        run=next(r for r in index['runs'] if r['id']==rid)
        originals[m,'gsa_bench_test']=Path(run['results']['gsa_bench_test']['path'])
    for job in JOBS:
        folder = OUT/job['id']
        state = 'complete' if (folder/'complete.json').exists() else 'failed' if (folder/'failure.json').exists() else 'pending'
        result['jobs'][job['id']] = {'state': state}
        if job['kind'] != 'visual': continue
        scores = []
        for condition in ['blank', 'shuffled']:
            p = folder/f'{condition}.json'
            if p.exists():
                r = json.loads(p.read_text())
                assert sha(folder/f'{condition}.jsonl') == r['output_sha256']
                verified=folder/f'{condition}_cpu_verified.json'
                if not verified.exists() or json.loads(verified.read_text())['result_sha256']!=sha(p):
                    from eval.gsa_eval import runner as scorer
                    from eval.gsa_eval.registry import get_spec
                    spec=get_spec(job['benchmark']); samples=list(spec.loader(W/'.cache/eval'))
                    predictions=rows(folder/f'{condition}.jsonl')
                    assert len(predictions)==len(samples)==r['n_evaluated']
                    assert [str(v['uid']) for v in predictions]==[str(s.uid) for s in samples]
                    check=scorer.score_shards(job['benchmark'],r['model'],[folder/f'{condition}.jsonl'],
                        scratch=W/'.cache/eval',max_tokens=spec.max_tokens,thinking=False,generation_batch_size=4)
                    for key in ['score','score_if_untrusted','n_evaluated','parse_failures']:
                        assert check.get(key)==r.get(key),(job['id'],condition,key,check.get(key),r.get(key))
                    save(verified,{'result_sha256':sha(p),'predictions_sha256':r['output_sha256'],
                        'n':len(samples),'all_uids_exact':True,'independent_cpu_rescore':True})
                score = r['score'] if r['score'] is not None else r['score_if_untrusted']
                scores.append(f'{score:.2f}')
                result['jobs'][job['id']][condition] = {'score': score, 'n': r['n_evaluated'], 'sha256': sha(p)}
            else: scores.append('--')
        names = {'gsa_bench_test': 'G (1,463)', 'blink_relative_depth': 'BLINK (124)', 'robospatial_home_context': 'RoboSpatial (122)'}
        original=originals[job['model'],job['benchmark']]
        original_data=json.loads(original.read_text())
        original_score=original_data['score'] if original_data['score'] is not None else original_data['score_if_untrusted']
        result['jobs'][job['id']]['original']={'path':str(original),'sha256':sha(original),'score':original_score}
        table.append(('Base' if job['model']=='base' else 'Anvil P5')+' & '+names[job['benchmark']]+' & '+f'{original_score:.2f}'+' & '+' & '.join(scores)+r' \\')
    table.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    target = A/'overleaf/source/tables/tab_visual_controls.tex'
    # Show completed cells immediately; never substitute scores for pending cells.
    if any('blank' in v or 'shuffled' in v for v in result['jobs'].values()):
        if any(result['jobs'][j['id']]['state']!='complete' for j in JOBS if j['kind']=='visual'):
            table[2]=table[2][:-1]+r" ``--'' denotes a run that has not yet completed.}"
        target.write_text('\n'.join(table)+'\n')
    if all(result['jobs'][j['id']]['state']=='complete' for j in JOBS if j['kind']=='visual'):
        g=result['jobs']['p5_gsa_bench_test'];depth=result['jobs']['p5_blink_relative_depth']
        text=(f"\\paragraph{{Visual dependence.}} On all 1,463 G-precursor examples, P5 scores "
            f"{g['original']['score']:.2f} with the original image, {g['blank']['score']:.2f} with a blank image, "
            f"and {g['shuffled']['score']:.2f} with an image from another scene. "
            f"On BLINK depth (124), the corresponding scores are {depth['original']['score']:.2f}, "
            f"{depth['blank']['score']:.2f}, and {depth['shuffled']['score']:.2f}. "
            "These full-split controls preserve every question, target, and decoding limit. "
            "They support visual dependence on the evaluated tasks; they do not establish "
            "the full eight-form dependency claim (Table~\\ref{tab:visual-controls}).\n")
        (A/'overleaf/source/tables/visual_dependence_observation.tex').write_text(text)
    calibration = [r for fam in ['G','S','A'] for r in rows(OUT/'compression'/f'{fam}.jsonl')]
    if (OUT/'compression/complete.json').exists():
        n = sum(r['n_tokens'] for r in calibration)
        result['compression'] = {'records': len(calibration), 'tokens': n, 'K': {}}
        for k in ['64','128']:
            result['compression']['K'][k] = {'mass_pct': 100*sum(r['K'][k]['mass_sum'] for r in calibration)/n,
                'reference_token_coverage_pct': 100*sum(r['K'][k]['target_hits'] for r in calibration)/n}
        c=result['compression']; k64=c['K']['64']; k128=c['K']['128']
        text = (f"On {c['records']} historical pilot records ({n:,} teacher-forced response tokens), "
            f"raw top-$64$ and top-$128$ supports retain {k64['mass_pct']:.3f}\\% and {k128['mass_pct']:.3f}\\% "
            f"of teacher probability mass, respectively. Their reference-token coverage is "
            f"{k64['reference_token_coverage_pct']:.2f}\\% and {k128['reference_token_coverage_pct']:.2f}\\%. "
            "This measures distribution compression on pilot specialists; it is not a downstream distillation comparison.\n")
        (A/'overleaf/source/tables/compression_observation.tex').write_text(text)
    result['historical_human_review'] = historical_human_review()
    result['historical_filter_replay'] = historical_filter_replay()
    from native_completion import collect_alfred_plans
    collect_alfred_plans()
    plans=A/'engineering/data/alfred_native_plan_sources_v1/plan_judgment_v1/acceptance.json'
    if plans.exists():result['action_plan_verification']=json.loads(plans.read_text())
    save(OUT/'acceptance.json', result)
    return result

def historical_human_review():
    """Reuse the original judgments; no reannotation or benchmark certification."""
    import numpy as np
    from eval.compile_semantic_audit import compile_report, content_hash, load_latest
    pack=W/'data/semantic_audit_v3'
    manifest=json.loads((pack/'manifest.json').read_text())
    report,_=compile_report(argparse.Namespace(pack=pack, adjudication=pack/'adjudication.jsonl',
        min_accuracy=.8, min_agreement=.8))
    assert not report['errors'], report['errors']
    previous=json.loads((W/'artifacts/audits/pilot_semantic_audit_v3_adjudicated.json').read_text())
    assert report['by_form']==previous['by_form']
    a,errors_a=load_latest(pack/'annotator_a/annotations/annotations.jsonl')
    b,errors_b=load_latest(pack/'annotator_b/annotations/annotations.jsonl')
    assert not errors_a and not errors_b and a.keys()==b.keys()
    adjudications=rows(pack/'adjudication.jsonl')
    adjudicated={r['sample_id']:r for r in adjudications}
    assert len(adjudicated)==len(adjudications)
    for sid,r in adjudicated.items():
        assert r['source_audit_id']==manifest['audit_id']
        assert r['label_sha256']==content_hash(a[sid]['label_snapshot'])
        assert a[sid]['correct']!=b[sid]['correct']
        assert r['reviewer_key'] not in {a[sid]['annotator_key'],b[sid]['annotator_key']}
    forms=['bbox','detection','mask','point','depth','affordance','dof','action_cot']
    records=[]
    for item in manifest['selected']:
        for form in forms:
            sid=item['sample_ids'][form];va=a[sid]['correct'];vb=b[sid]['correct']
            resolved=va if va==vb else adjudicated.get(sid,{}).get('correct')
            records.append({'sample_id':sid,'image_sha256':item['image_sha256'],
                'form':form,'a':va,'b':vb,'resolved':resolved})
    assert len(records)==320 and len(manifest['selected'])==40
    def summarize(items):
        n=len(items);agreement=sum(r['a']==r['b'] for r in items)
        pa=sum(r['a'] for r in items)/n;pb=sum(r['b'] for r in items)/n
        chance=pa*pb+(1-pa)*(1-pb)
        return {'n':n,'agreed':agreement,'raw_agreement':agreement/n,
            'kappa':(agreement/n-chance)/(1-chance) if chance<1 else None,
            'supported':sum(r['resolved'] is True for r in items),
            'rejected':sum(r['resolved'] is False for r in items),
            'unresolved':sum(r['resolved'] is None for r in items)}
    per_form={f:summarize([r for r in records if r['form']==f]) for f in forms}
    total=summarize(records)
    agreements=np.array([r['a']==r['b'] for r in records]).reshape(40,8)
    rng=np.random.default_rng(20260915)
    intervals=np.quantile(agreements[rng.integers(0,40,size=(5000,40))].mean(axis=(1,2)),[.025,.975])
    paths=[pack/'manifest.json',pack/'annotator_a/annotations/annotations.jsonl',
        pack/'annotator_b/annotations/annotations.jsonl',pack/'adjudication.jsonl',
        W/'artifacts/audits/pilot_semantic_audit_v3_adjudicated.json']
    result={'scope':'Historical pilot review; not final benchmark reliability or matched-pipeline precision',
        'bindings':{str(p):sha(p) for p in paths},'images':40,'per_form':per_form,'total':total,
        'agreement_ci95':intervals.tolist(),'bootstrap':{'unit':'image','replicates':5000,'seed':20260915},
        'adjudications':len(adjudicated),'edited_target_correction_rate':None,
        'not_judgeable_rate':None,'final_benchmark_reliability':False,'new_annotation_calls':0,
        'source_audit_verdict':report['verdict']}
    save(OUT/'historical_human_review.json',result)
    labels={'bbox':'Box','detection':'Detection','mask':'Mask','point':'Point','depth':'Depth',
        'affordance':'Affordance','dof':'DoF','action_cot':'Action plan'}
    table=[r'\begin{table}[!htbp]',r'\centering\small',
        r"\caption{\textbf{Historical human spot check, reported separately from automated production.} Two reviewers assessed 40 images in each of eight forms (320 paired decisions). Agreement and Cohen's $\kappa$ use the original votes; supported/rejected counts additionally use 18 third-reviewer adjudications. These judgments concern only the historical candidates.}",
        r'\label{tab:historical-human-review}',r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',
        r'Form & $n$ & Agree (\%) & $\kappa$ & Supported & Rejected & Unresolved \\',r'\midrule']
    for form in forms+['all']:
        if form=='all':table.append(r'\midrule')
        v=total if form=='all' else per_form[form]
        k='--' if v['kappa'] is None else f"{v['kappa']:.2f}"
        table.append((labels.get(form,'All'))+f" & {v['n']} & {100*v['raw_agreement']:.1f} & {k} & {v['supported']} & {v['rejected']} & {v['unresolved']}"+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}',
        f"Pooled raw agreement is {total['agreed']}/{total['n']} "
        f"({100*total['raw_agreement']:.1f}\\%; 95\\% image-bootstrap interval "
        f"{100*intervals[0]:.1f}--{100*intervals[1]:.1f}\\%, 5,000 resamples). "
        "All 14 unresolved decisions concern depth. Reviewers made binary judgments; "
        "target-edit and not-judgeable rates were not collected.\n"]
    save_text(A/'overleaf/source/tables/historical_human_review.tex','\n'.join(table)+'\n')
    return result

def historical_filter_replay():
    """Compare filters on every first-attempt generation response in the pilot.

    Verification queries and adaptive retries are separate populations. The two
    derived forms (box and point) have no generation query in this historical run.
    Recorded upstream contexts remain fixed; this is not end-to-end regeneration.
    """
    import copy
    import gzip
    import inspect
    from collections import Counter
    from trains.flywheel import validators, contracts
    from trains.flywheel.vlm_client import extract_json
    folder=OUT/'filter_replay';folder.mkdir(exist_ok=True)
    source=W/'logs/vlm_requests_v2_full500.jsonl.gz'
    raw_root=W/'data/flywheel_v2/full500/raw'
    binding={'schema':'anvil-first-attempt-filter-replay-v1','requests_sha256':sha(source),
        'context_sources':{f:sha(raw_root/(f+'.jsonl')) for f in ['detection','depth']},
        'validators_sha256':sha(Path(validators.__file__)),
        'contracts_sha256':sha(Path(contracts.__file__)),
        'parser_sha256':hashlib.sha256(inspect.getsource(extract_json).encode()).hexdigest(),
        'selection':'Every detection query at temperature 0.1 and mask/depth/affordance/dof/action query at 0.2; all must have their first-attempt cache. Verification and retries are excluded by query type/temperature, never by response.',
        'cache_key':'SHA256(model + | + temperature + | + prompt + | + image_sha256)',
        'upstream_policy':'Original recorded inputs, checked against the actual prompt; no upstream refiltering or regeneration',
        'new_model_calls':0,'human_precision_established':False}
    identity(folder/'binding.json',binding)
    snapshot=folder/'first_candidates.jsonl'
    if not snapshot.exists() or not (folder/'population.json').exists():
        with gzip.open(source,'rt') as stream:requests=[json.loads(line) for line in stream if line.strip()]
        context_rows={f:{r['image_sha256']:r for r in rows(raw_root/(f+'.jsonl'))}
                      for f in ['detection','depth']}
        candidates=[];seen=set();images={};other=Counter()
        prefixes=[('Detect ','detection'),('Verify ','verification'),('Segment ','mask'),
            ('Order ','depth'),('The ','affordance'),('For ','dof'),('A robot ','action_cot')]
        for index,request in enumerate(requests):
            prompt=request['prompt']
            form=next((f for prefix,f in prefixes if prompt.startswith(prefix)),None)
            assert form is not None,'Unknown historical query type'
            expected_temperature=.1 if form=='detection' else .2
            if form=='verification' or request['temperature']!=expected_temperature:
                other[form]+=1;continue
            assert request['attempt']==1,'Initial query contains transport retries; resolve chronology first'
            image_sha=request['image_sha256'];identity_key=(image_sha,form)
            assert identity_key not in seen,'Repeated first-attempt candidate'
            seen.add(identity_key)
            if image_sha not in images:
                assert sha(Path(request['image_path_local_only']))==image_sha
                images[image_sha]=True
            key=hashlib.sha256(f"{request['model']}|{request['temperature']}|{prompt}|{image_sha}".encode()).hexdigest()
            cache=W/'logs/vlm_cache'/(key+'.json')
            assert cache.is_file(),f'Missing initial response: {index}/{form}'
            answer=json.loads(cache.read_text())
            assert extract_json(answer['content'])==answer['parsed'],'Historical parser/cache mismatch'
            context={}
            if form!='detection':
                context=copy.deepcopy(context_rows['detection'][image_sha]['gsa_context'])
                primary=context['primary'];label=primary['object_class'];box=primary['bbox']
                if form=='depth':
                    expected='Order these objects from nearest to farthest from the camera: '+', '.join(
                        f"{g['object_class']}@{g['bbox']}" for g in context['groundings'])+'.'
                else:
                    expected={'mask':f'Segment the {label} located at bounding box {box}',
                        'affordance':f'The {label} occupies {box}.',
                        'dof':f'For the {label} at {box}, specify',
                        'action_cot':f'A robot arm must pick up the {label} at {box}.'}[form]
                assert prompt.startswith(expected),'Recorded grounding context differs from prompt'
                if form in ['affordance','dof','action_cot']:
                    spatial=context_rows['depth'][image_sha]['gsa_context']['spatial']
                    assert f'Verified spatial context: {spatial}.' in prompt
                    context['spatial']=copy.deepcopy(spatial)
            candidates.append({'request_index':index,'form':form,'image_sha256':image_sha,
                'request':request,'cache_sha256':sha(cache),'content':answer['content'],
                'parsed':answer['parsed'],'context':context})
        assert len(candidates)==2618 and len(images)==500
        snapshot_text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in candidates)
        if snapshot.exists():assert snapshot.read_text()==snapshot_text,'Interrupted snapshot differs from source cache'
        else:save_text(snapshot,snapshot_text)
        save(folder/'population.json',{'snapshot_sha256':sha(snapshot),'requests':len(requests),
            'candidates':len(candidates),'images':len(images),'other_queries':dict(other)})
    population=json.loads((folder/'population.json').read_text())
    assert sha(snapshot)==population['snapshot_sha256']
    candidates=rows(snapshot);counts={};decisions=[]
    for candidate in candidates:
        form=candidate['form'];parsed=extract_json(candidate['content'])
        assert parsed==candidate['parsed']
        items=parsed if isinstance(parsed,list) else [parsed] if isinstance(parsed,dict) else []
        syntactic=parsed is not None
        shape,reason=validators.validate_structural(form,copy.deepcopy(items),candidate['context'])
        typed=False
        if syntactic and shape:
            typed,reason=validators.validate(form,copy.deepcopy(items),candidate['context'])
        v=counts.setdefault(form,Counter());v['raw']+=1;v['syntax']+=syntactic;v['typed']+=bool(typed)
        decisions.append({'request_index':candidate['request_index'],'form':form,
            'syntax':syntactic,'typed':bool(typed),'reason':reason if not typed else ''})
    totals={k:sum(v[k] for v in counts.values()) for k in ['raw','syntax','typed']}
    assert totals['raw']==2618 and totals['typed']<=totals['syntax']<=totals['raw']
    result={'binding':binding,'population':population,'per_form':counts,'totals':totals,
        'scope':'Conditional candidate-level filtering on six generated forms; box/point were derived, not independently queried. Fixed recorded upstream contexts; no end-to-end cascade or semantic precision claim.',
        'rejection_reasons':dict(Counter(d['form']+':'+d['reason'] for d in decisions if not d['typed']))}
    save(folder/'decisions.json',decisions);save(folder/'acceptance.json',result)
    table=[r'\begin{table}[!htbp]',r'\centering\small',
        r'\caption{\textbf{Filtering the same first-attempt candidates.} Every initial generation response is retained before filtering. JSON parsing and typed contracts use the same 2,618 candidates and their recorded upstream inputs. Box and point were derived in this historical run; verifier calls and adaptive retries are separate. Entries are candidate counts, not semantic correctness.}',
        r'\label{tab:filter-replay}',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
        r'Form & Raw & JSON parsed & Typed contracts \\',r'\midrule']
    for form in ['detection','mask','depth','affordance','dof','action_cot']:
        v=counts[form];label={'dof':'DoF','action_cot':'Action plan'}.get(form,form.capitalize())
        table.append(f"{label} & {v['raw']} & {v['syntax']} & {v['typed']}"+r' \\')
    table += [r'\midrule',f"All & {totals['raw']} & {totals['syntax']} & {totals['typed']}"+r' \\',
        r'\bottomrule',r'\end{tabular}',r'\end{table}']
    save_text(A/'overleaf/source/tables/filter_replay_observation.tex','\n'.join(table)+'\n')
    target=A/'overleaf/source/tables/tab_pipeline_ablation.tex';text=target.read_text()
    # The later matched-source experiment has a different candidate population.
    # Preserve its measurements; historical replay remains in its own table.
    if text.startswith('% anvil-source-quality-v1\n'):
        return result
    for label,metric in [('Raw generation','raw'),(r'\quad + syntax parser','syntax'),
                         (r'\quad + typed contracts','typed')]:
        pattern=r'(?m)^('+re.escape(label)+r'\n\s*& )(?:\\needcheck\{[\d.]+\\%\}|[\d.]+\\%)( &)'
        value=f"{100*totals[metric]/totals['raw']:.1f}"+r'\%'
        text,n=re.subn(pattern,lambda m:m[1]+value+m[2],text)
        assert n==1,f'Cannot locate measured structural cell: {label}'
    save_text(target,text)
    return result

def historical_audit():
    """Recount historical source overlaps from image contents and retained records."""
    from PIL import Image
    from collections import Counter
    from eval.gsa_eval.registry import get_spec
    def rgb(p):
        with Image.open(p) as im:
            im=im.convert('RGB')
            return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
    manifest=W/'artifacts/upload_review_500_v2/manifest.jsonl'
    candidates=rows(manifest)
    retained=Counter(r['image_sha256'] for p in (W/'data/flywheel_v2/full500/raw').glob('*.jsonl') for r in rows(p))
    result={'manifest_sha256': sha(manifest), 'benchmarks': {}}
    for bench in ['where2place','refspatial_bench']:
        samples=list(get_spec(bench).loader(W/'.cache/eval'))
        exact={(Path(p).name,sha(p)) for s in samples for p in s.image_paths}
        decoded={rgb(p) for s in samples for p in s.image_paths}
        hits=[]
        for r in candidates:
            source=Path(r['source_path'])
            assert source.exists(), str(source)
            if bench=='where2place': match=(source.name,sha(source)) in exact
            else: match=rgb(source) in decoded
            if match: hits.append({'id':r['id'],'source':str(source),'source_sha256':sha(source),
                'upload_sha256':r['upload_sha256'],'records':retained[r['upload_sha256']]})
        result['benchmarks'][bench]={'n':len(samples),'colliding_sources':len(hits),
            'retained_images':sum(h['records']>0 for h in hits),'retained_records':sum(h['records'] for h in hits),'matches':hits}
    save(OUT/'historical_audit.json',result)
    log=A.parent/'GSA_Forge/output/sft_9b_lora_l20/v0-20260815-004807/logging.jsonl'
    history=rows(log);terminal=next(r for r in history if 'train_runtime' in r)
    ev=[r for r in history if 'eval_loss' in r][-1]
    result['legacy_training']={'path':str(log),'sha256':sha(log),
        'steps':terminal['global_step/max_steps'],'hours':terminal['train_runtime']/3600,
        'epoch':terminal['epoch'],'validation_loss':ev['eval_loss']}
    indexpath=A/'engineering/data/p5_retention_clean_run_seed17_v2/spatial_registry_20260914/result_index.json'
    index=json.loads(indexpath.read_text());cells=[]
    for run in index['runs']:
        if run['id'].endswith('_legacy'):continue
        for bench,n in [('cv_bench',2638),('embspatial',3640),('refspatial_bench',277),('where2place',100),('gsa_bench_test',1463)]:
            item=run['results'][bench];p=Path(item['path']);obs=json.loads(p.read_text())
            assert item['n_evaluated']==obs['n_evaluated']==n
            cells.append({'id':run['id'],'benchmark':bench,'n':n,'path':str(p),'sha256':sha(p)})
    assert len(cells)==35
    result['baseline_inventory']={'registry_sha256':sha(indexpath),'complete_cells':cells}
    save(OUT/'historical_audit.json',result)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='matches'} for k,v in result['benchmarks'].items()},indent=2))

def publish():
    # Independent evidence followers may become ready at the same time.
    # Serialize compilation and replacement of the shared PDF/ZIP pair.
    with (OUT/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _publish()


def _publish():
    result = collect()
    if not (OUT/'paper_ready').exists() and not (OUT/'review_ready').exists():
        print('Results collected; manuscript editing latch still active.', flush=True); return
    wrapper = os.environ.get('ANVIL_LATEX_COMPILER', '/path/to/latex_compile.py')
    uv = '/path/to/workspace/env/uv/uv'
    source = A/'overleaf/source'
    with (OUT/'compile.log').open('a') as log:
        subprocess.run([uv,'run','--python',sys.executable,'python','-B',wrapper,'main.tex'],
                       cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
    import zipfile
    save_bytes(A/'paper/Anvil_Submission_Draft.pdf',(source/'main.pdf').read_bytes())
    includes = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',
        '\n'.join(p.read_text() for p in source.rglob('*.tex'))))
    target = A/'overleaf/Anvil_Submission_Draft.zip'
    with zipfile.ZipFile(target.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source.rglob('*')):
            if not p.is_file(): continue
            rel=str(p.relative_to(source))
            if p.suffix in {'.tex','.bib','.sty','.cls','.bst'} or rel in includes or rel in {'README_OVERLEAF.md','latexmkrc'}:
                z.write(p, rel)
    target.with_suffix('.tmp').replace(target)
    marks=sum(len(re.findall(r'\\needcheck\{',re.sub(r'(?<!\\)%[^\n]*','',p.read_text()))) for p in source.rglob('*.tex'))
    published={'pdf': str(A/'paper/Anvil_Submission_Draft.pdf'),
        'pdf_sha256': sha(A/'paper/Anvil_Submission_Draft.pdf'), 'zip': str(target),
        'zip_sha256': sha(target), 'needcheck': marks, 'jobs': result['jobs']}
    pages=int(re.search(r'^Pages:\s+(\d+)',subprocess.check_output(['pdfinfo',str(source/'main.pdf')],text=True),re.M)[1])
    body=subprocess.check_output(['pdftotext','-layout',str(source/'main.pdf'),'-'],text=True)
    conclusion=next((i for i,t in enumerate(body.split('\f'),1) if re.search(r'^\s*(?:\d+\s+)?\d+\s+C\s*ONCLUSION',t,re.M|re.I)),None)
    assert conclusion is not None, 'Cannot locate conclusion page in compiled PDF'
    published.update(pages=pages,conclusion_page=conclusion,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(OUT/'published.json',published)
    manifest_path=A/'CURRENT_RELEASE_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text())
    manifest.update(updated_utc=published['updated_utc'],submission_ready=False,
        status='figure_variants_native_transfer_core_evidence_pending')
    with zipfile.ZipFile(target) as z: file_count=len(z.namelist())
    manifest['active_artifacts']={'pdf':{'path':'paper/Anvil_Submission_Draft.pdf','sha256':published['pdf_sha256']},
        'overleaf_zip':{'path':'overleaf/Anvil_Submission_Draft.zip','sha256':published['zip_sha256'],'file_count':file_count,'figure_count':len(includes)},
        'remaining_needcheck':marks}
    figure_manifest=A/'engineering/data/figure_variants_20260915/manifest.json'
    if figure_manifest.is_file():
        figure_release=json.loads(figure_manifest.read_text())
        deck=A/'figures/Anvil_V6_RoboRefer_ALL_FIGURES_EDITABLE.pptx'
        assert sha(deck)==figure_release['pptx']['sha256']
        manifest['active_artifacts']['editable_figures']={
            'path':str(deck.relative_to(A)),'sha256':sha(deck),
            'original_pages':6,'appended_comparison_pages':figure_release['pptx']['appended_comparison_pages'],
            'candidates_per_figure':{name:sum(v['figure']==name for v in figure_release['figures'])
                for name in {v['figure'] for v in figure_release['figures']}},
            'selected_variant':figure_release['selected_variant'],
            'selected_per_figure':figure_release.get('selected_per_figure',{})}
    manifest['overleaf'].update(main_pages=conclusion,paper_pages=pages)
    save(manifest_path,manifest)
    complete=sum(j['state']=='complete' for j in result['jobs'].values())
    native_root=A/'engineering/data/native_completion_20260915'
    native_counts={}
    for cohort,root in [('expanded',native_root),('sensor_mask',native_root/'sensor_mask')]:
        native_counts[cohort]=len(list(root.glob('*/done.json')))
    status=(f"Updated {published['updated_utc']}. Current PDF: **{pages} pages**, conclusion on page **{conclusion}**; "
        f"**{marks} non-comment needcheck calls** remain. This is a revised draft, not a completed full-MTD submission.\n\n"
        f"Closeout jobs complete: **{complete}/{len(JOBS)}**. Results and exact paths: "
        "`engineering/data/submission_closeout_20260915/acceptance.json`; live queue: `bash engineering/run_remaining.sh status`. "
        "Current remaining core experiments and their declared source-specific scope are recorded in `todo_exp.md`; historical human review is reported separately.\n\n"
        f"Active native-transfer queue: **{native_counts['sensor_mask']}/20 RGB-D cells** and **{native_counts['expanded']}/20 expanded-source cells** complete at packaging time. "
        "Run `bash scripts/run_native_completion.sh status`; the experiment board is `todo_exp.md`. "
        f"Selective cap repair: **{len(list((native_root/'cap_repair').glob('*/done.json')))}/20 cells**. "
        f"ALFRED plan verification: **{len(list((A/'engineering/data/alfred_native_plan_sources_v1/plan_judgment_v1').glob('*/done.json')))}/4 models**; "
        f"linked G-to-S training/evaluation: **{len(list((A/'engineering/data/linked_sensor_training_v1').glob('*/done.json')))}/4 arms**. "
        "Completed results are included in appendix tables; all launchers reuse completed work.\n\n"
        f"PDF SHA256: `{published['pdf_sha256']}`. ZIP SHA256: `{published['zip_sha256']}`.\n")
    readme=A/'README.md'; text=readme.read_text()
    block='<!-- ANVIL_CLOSEOUT_STATUS_BEGIN -->\n'+status+'<!-- ANVIL_CLOSEOUT_STATUS_END -->'
    text=re.sub(r'<!-- ANVIL_CLOSEOUT_STATUS_BEGIN -->.*?<!-- ANVIL_CLOSEOUT_STATUS_END -->',lambda _:block,text,flags=re.S)
    save_text(readme,text)
    registry=A/'engineering/experiments/result_registry.yaml'; text=registry.read_text()
    entry={'evidence':'engineering/data/submission_closeout_20260915/acceptance.json',
        'completed_jobs':complete,'total_jobs':len(JOBS),'needcheck':marks,'paper_pages':pages,
        'main_conclusion_page':conclusion,'paper_sha256':published['pdf_sha256'],'full_MTD_completed':False}
    block='# BEGIN CLOSEOUT_STATUS\nsubmission_closeout_20260915: '+json.dumps(entry)+'\n# END CLOSEOUT_STATUS'
    if '# BEGIN CLOSEOUT_STATUS' in text:
        text=re.sub(r'# BEGIN CLOSEOUT_STATUS.*?# END CLOSEOUT_STATUS',lambda _:block,text,flags=re.S)
    else:text=block+'\n'+text
    save_text(registry,text)
    print(json.dumps({'published_pdf':published['pdf'],'published_zip':published['zip'],
        'completed_jobs':complete,'needcheck':marks,'pages':pages,'conclusion_page':conclusion}),flush=True)

def finalize():
    # Also supports an already-running older supervisor without touching its workers.
    while True:
        state=OUT/'supervisor.json'
        pid=json.loads(state.read_text())['pid'] if state.exists() else 0
        alive=pid and (Path('/proc')/str(pid)).exists()
        terminal=all((OUT/j['id']/'complete.json').exists() or (OUT/j['id']/'failure.json').exists() for j in JOBS)
        if terminal and not alive and (OUT/'review_ready').exists(): break
        time.sleep(15)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'publish'],check=True)

def supervise():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(OUT/'plan.json', {'jobs': JOBS, 'hard_runtime_timeout': False,
            'resume': 'batch journal + immutable model/protocol/input identity',
            'gpu_claim': 'memory < 64 MiB and no compute process; UUID device binding',
            'paid_calls': False})
        # Immutable executable snapshot: editing the live driver never corrupts workers.
        snapshot = OUT/'worker_snapshot.py'
        if not snapshot.exists(): snapshot.write_bytes(Path(__file__).read_bytes())
        # Its project root is injected explicitly, independent of the snapshot location.
        pending=[j for j in JOBS if not (OUT/j['id']/'complete.json').exists()]
        running={}; attempted=set()
        while pending or running:
            for jid,(proc,gpu,stream) in list(running.items()):
                if proc.poll() is not None:
                    stream.close(); del running[jid]
                    print(f'{jid} exited {proc.returncode}', flush=True)
                    collect()
            query=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
            compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
            used={v[1] for v in running.values()}
            free=[line.split(',')[0].strip() for line in query.splitlines()
                  if int(line.split(',')[1].strip())<64 and line.split(',')[0].strip() not in compute
                  and line.split(',')[0].strip() not in used]
            while pending and free:
                job=pending.pop(0); gpu=free.pop(0); folder=OUT/job['id']; folder.mkdir(exist_ok=True)
                stream=(folder/'worker.log').open('a')
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, ANVIL_CLOSEOUT_ROOT=str(A),
                         OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
                proc=subprocess.Popen([sys.executable,'-u',str(snapshot),'worker','--job',job['id']],
                    env=env, stdout=stream, stderr=subprocess.STDOUT)
                running[job['id']]=(proc,gpu,stream); attempted.add(job['id'])
                print(f'Started {job["id"]} PID={proc.pid} GPU={gpu}', flush=True)
            save(OUT/'supervisor.json', {'pid': os.getpid(), 'running': {j:{'pid':p.pid,'gpu':g} for j,(p,g,s) in running.items()},
                'pending': [j['id'] for j in pending], 'attempted': sorted(attempted)})
            if pending or running: time.sleep(15)
        publish()

if __name__ == '__main__':
    if os.environ.get('ANVIL_CLOSEOUT_ROOT'):
        A=Path(os.environ['ANVIL_CLOSEOUT_ROOT']); W=A.parent/'GSA_collect'
        OUT=A/'engineering/data/submission_closeout_20260915'; sys.path.insert(0,str(W))
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['supervise','worker','collect','publish','status','audit','finalize']); p.add_argument('--job')
    args=p.parse_args()
    if args.command=='worker': worker(args.job)
    elif args.command=='supervise': supervise()
    elif args.command=='publish': publish()
    elif args.command=='collect': print(json.dumps(collect(),indent=2))
    elif args.command=='audit': historical_audit()
    elif args.command=='finalize': finalize()
    else:
        for name in ['supervisor.json','acceptance.json','published.json']:
            path=OUT/name
            if path.exists(): print(name+'\n'+path.read_text())
