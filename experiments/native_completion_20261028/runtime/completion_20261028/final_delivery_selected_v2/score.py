"""Summarize one exact model using unchanged native/public/general scorers."""
import argparse
from collections import Counter,defaultdict
import hashlib
import inspect
import math
from pathlib import Path
import sys
from common import C,D,EXPORT,check_protocol,load_spec,read,require,save,sha
sys.path[:0]=[str(D/'code'),str(C/'eval_code')]
from eval_core import digest
from gsa_eval.registry import BENCHMARKS
from gsa_eval.datasets import Sample
from general_retention_formal_v3 import parse_multiple_choice,_answer_span,_reference_digit_article,_vqa_soft_credit

PAPER=dict(cv_bench=88.21,embspatial=82.34,refspatial_bench=59.,where2place=68.,
           blink_relative_depth=91.13,erqa=50.52,vabench_point=80.67)


def native_score(root,spec_path,spec,artifacts):
    manifest=read(EXPORT/'manifest.json')
    expected={}
    for name,info in manifest['files'].items():
        require(sha(EXPORT/name)==info['sha256'],'Changed native input')
        for ordinal,line in enumerate((EXPORT/name).open()):
            row=__import__('json').loads(line)
            key=(row['native_form'],ordinal)
            require(key not in expected,'Duplicate native source ordinal')
            expected[key]=(row['id'],row['evaluation_meta']['evaluation_content_sha256'])
    require(len(expected)==580000,'Native denominator changed')
    seen=set();finishes=Counter();groups=defaultdict(lambda:dict(n=0,sums=Counter()))
    for shard in range(spec['native_shards']):
        out=root/'native'/str(shard);file=out/'metrics.json'
        metrics=read(file);protocol=read(out/'protocol.json');check_protocol(protocol,spec_path,spec)
        require(metrics['state']=='complete_scope_shard' and metrics['protocol_sha256']==sha(out/'protocol.json'),'Incomplete native shard')
        require(protocol['scope']=='joint' and protocol['shard']==shard and protocol['shards']==spec['native_shards'],'Native partition changed')
        artifacts[str(file)]=sha(file);local=defaultdict(lambda:dict(n=0,sums=Counter()))
        for name,expected_sha in metrics['prediction_sha256'].items():
            require(Path(name).name==name,'Unsafe prediction path')
            p=out/name;require(sha(p)==expected_sha,'Native prediction changed');artifacts[str(p)]=expected_sha
            for line in p.open():
                row=__import__('json').loads(line);key=(row['native_form'],row['ordinal'])
                require(key in expected and key not in seen and row['ordinal']%spec['native_shards']==shard,'Duplicate/out-of-scope native output')
                require(expected[key]==(row['id'],row['evaluation_content_sha256']),'Native source identity changed')
                require(row['protocol_sha256']==metrics['protocol_sha256'],'Mixed native protocol')
                require(digest({k:v for k,v in row.items() if k!='record_sha256'})==row['record_sha256'],'Native record hash changed')
                require(hashlib.sha256(row['response'].encode()).hexdigest()==row['response_sha256'],'Native response hash changed')
                require(all(math.isfinite(v) for v in row['metrics'].values()),'Nonfinite native metric')
                seen.add(key);finishes[row['finish_reason']]+=1
                group_key=(row['native_form'],row['source'])
                for destination in [local,groups]:
                    destination[group_key]['n']+=1;destination[group_key]['sums'].update(row['metrics'])
        require(sum(v['n'] for v in local.values())==metrics['n']==metrics['expected_records'],'Native count mismatch')
        require(set(local)=={(g['form'],g['source']) for g in metrics['groups']},'Missing native metric group')
        for group in metrics['groups']:
            actual=local[(group['form'],group['source'])]
            require(actual['n']==group['n'],'Group count mismatch')
            require(set(actual['sums'])==set(group['mean']),'Metric keys changed')
            for key,value in group['mean'].items():
                require(math.isclose(actual['sums'][key]/actual['n'],value,rel_tol=1e-10,abs_tol=1e-10),'Native aggregation mismatch')
    require(seen==set(expected),'Incomplete native evaluation')
    return dict(n=len(seen),finish_reasons=dict(finishes),groups=[dict(form=f,source=s,n=g['n'],mean={k:v/g['n'] for k,v in g['sums'].items()}) for (f,s),g in sorted(groups.items())])


def public_score(root,spec_path,spec,artifacts):
    manifest=read(D/'inputs/manifest.json');result={'public':{},'general':{}}
    for name in [*manifest['public'],'retention']:
        by_id={}
        for shard in range(spec['public_shards']):
            out=root/'public'/str(shard);complete=read(out/'complete.json');protocol=read(out/'protocol.json')
            check_protocol(protocol,spec_path,spec)
            require(complete['state']=='complete' and complete['protocol_sha256']==sha(out/'protocol.json'),'Incomplete public shard')
            require(protocol['shard']==shard and protocol['shards']==spec['public_shards'],'Public partition changed')
            file=out/(name+'.jsonl');require(sha(file)==complete['prediction_sha256'][name],'Public predictions changed')
            artifacts[str(file)]=sha(file)
            for line in file.open():
                row=__import__('json').loads(line);uid=row['uid']
                require(uid not in by_id,'Duplicate public prediction')
                require(hashlib.sha256(row['response'].encode()).hexdigest()==row['response_sha256'],'Public response changed')
                by_id[uid]=row
        info=manifest['retention'] if name=='retention' else manifest['public'][name]
        file=D/'inputs'/(name+'.jsonl');require(sha(file)==info['sha256'],'Public inputs changed')
        source=[__import__('json').loads(line) for line in file.open()]
        require(set(by_id)=={str(r.get('global_uid',r.get('uid'))) for r in source},'Incomplete public evaluation')
        if name!='retention':
            samples=[Sample(**{k:v for k,v in row.items() if k!='max_tokens'}) for row in source]
            scorer=BENCHMARKS[name]
            kwargs={'spec':scorer} if 'spec' in inspect.signature(scorer.scorer).parameters else {}
            score,detail,fails=scorer.scorer(samples,[by_id[str(r.uid)]['response'] for r in samples],**kwargs)
            require(len(samples)==spec['public_records'][name],'Public denominator changed')
            result['public'][name]=dict(n=len(samples),score=score,detail=detail,parse_failures=fails,
                capped=sum(by_id[str(r.uid)]['hit_token_cap'] for r in samples))
        else:
            require(len(source)==10105,'General denominator changed')
            require(sha(info['vqa_reference'])==info['vqa_reference_sha256'],'VQA reference changed')
            helper=_reference_digit_article(Path(info['vqa_reference']));values=defaultdict(list);invalid=Counter()
            for row in source:
                text=by_id[row['global_uid']]['response']
                if row['metric']=='multiple_choice_accuracy':
                    parsed,_=parse_multiple_choice(text,row['choices']);credit=float(parsed==row['answer'])
                else:
                    parsed,_=_answer_span(text);credit=_vqa_soft_credit(parsed,row['answer'],helper) if parsed else 0.
                values[row['dataset']].append(credit);invalid[row['dataset']]+=int(parsed is None)
            result['general']={k:dict(n=len(v),score=100*sum(v)/len(v),parse_failures=invalid[k]) for k,v in values.items()}
    aligned={k:(v['detail']['per_group']['location']['mean_in_mask'] if k=='refspatial_bench' else v['score']) for k,v in result['public'].items()}
    result.update(paper_aligned_public=aligned,paper_aligned_macro=sum(aligned.values())/7,
                  paper_target=PAPER,final_minus_paper_percentage_points={k:v-PAPER[k] for k,v in aligned.items()},
                  paper_protocol_note='RefSpatial Location100; full277 also reported. Historical runtime equivalence is not established.')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();spec=load_spec(args.spec);artifacts={}
    native=native_score(args.output,args.spec,spec,artifacts)
    public=public_score(args.output,args.spec,spec,artifacts)
    result=dict(schema='anvil-final-evaluation-report-v1',state='complete_requires_quality_review',
                final_spec_sha256=sha(args.spec),model=spec['model'],checkpoint=spec['checkpoint'],
                native=native,**public,artifacts_sha256=artifacts,
                final_quality_accepted=False,published_results_reproduced=False,weights_uploaded=False)
    save(args.output/'final_report.json',result)
    print(args.output/'final_report.json')


if __name__=='__main__':main()
