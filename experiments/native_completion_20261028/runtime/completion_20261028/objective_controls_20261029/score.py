"""Full seven-public/general diagnostics for a sealed formal intermediate checkpoint."""
from collections import Counter,defaultdict
import argparse,hashlib,inspect,json,sys
from pathlib import Path
from objective_control_common import D,check_protocol,load_spec,read,require,save,sha
sys.path.insert(0,str(D/'code'))
from gsa_eval.registry import BENCHMARKS
from gsa_eval.datasets import Sample
from general_retention_formal_v3 import parse_multiple_choice,_answer_span,_reference_digit_article,_vqa_soft_credit
PAPER=dict(cv_bench=88.21,embspatial=82.34,refspatial_bench=59.,where2place=68.,blink_relative_depth=91.13,erqa=50.52,vabench_point=80.67)

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
    result=public_score(args.output,args.spec,spec,artifacts)
    baseline=read(D/'student_adaptation_public_comparison.json')['results']['student_adaptation']['paper_aligned_public']
    report=dict(schema='anvil-objective-control-public-report-v1',state='complete_objective_control_requires_review',objective_control=True,final_mopd=False,
      checkpoint_spec_sha256=sha(args.spec),checkpoint=spec['checkpoint'],step=spec['step'],model=spec['model'],
      **result,student_initial_public=baseline,checkpoint_minus_initial_percentage_points={k:v-baseline[k] for k,v in result['paper_aligned_public'].items()},
      artifacts_sha256=artifacts,full_native_validation=False,final_quality_accepted=False,weights_uploaded=False)
    save(args.output/'report.json',report);print(args.output/'report.json')

if __name__=='__main__':main()
