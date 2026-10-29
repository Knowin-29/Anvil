"""Score full matched public/VQA runs with the original task scorers."""
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from gsa_eval.registry import BENCHMARKS
from gsa_eval.datasets import Sample
from general_retention_formal_v3 import parse_multiple_choice,_answer_span,_reference_digit_article,_vqa_soft_credit

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text());result={};artifacts={}
    for role in ['base','joint']:
        result[role]={'public':{},'retention':{}}
        for name in [*manifest['public'],'retention']:
            by_id={}
            for shard in range(4):
                run=ROOT/'benchmarks'/role/str(shard);assert json.loads((run/'complete.json').read_text())['state']=='complete'
                file=run/(name+'.jsonl');artifacts[str(file)]=sha(file)
                for line in file.open():
                    row=json.loads(line);uid=row['uid'];assert uid not in by_id
                    assert hashlib.sha256(row['response'].encode()).hexdigest()==row['response_sha256']
                    by_id[uid]=row
            source=[json.loads(l) for l in (ROOT/'inputs'/(name+'.jsonl')).open()]
            assert set(by_id)=={str(r.get('global_uid',r.get('uid'))) for r in source}
            if name!='retention':
                samples=[Sample(**{k:v for k,v in r.items() if k!='max_tokens'}) for r in source]
                texts=[by_id[str(r.uid)]['response'] for r in samples];spec=BENCHMARKS[name]
                kwargs={'spec':spec} if 'spec' in inspect.signature(spec.scorer).parameters else {}
                score,detail,fails=spec.scorer(samples,texts,**kwargs)
                result[role]['public'][name]=dict(n=len(samples),score=score,detail=detail,parse_failures=fails,
                    capped=sum(by_id[str(r.uid)]['hit_token_cap'] for r in samples))
            else:
                helper=_reference_digit_article(Path(manifest['retention']['vqa_reference']));values=defaultdict(list);invalid=defaultdict(int)
                for row in source:
                    prediction=by_id[row['global_uid']];text=prediction['response']
                    if row['metric']=='multiple_choice_accuracy':
                        parsed,unclosed=parse_multiple_choice(text,row['choices']);credit=float(parsed==row['answer'])
                    else:
                        parsed,unclosed=_answer_span(text);credit=_vqa_soft_credit(parsed,row['answer'],helper) if parsed else 0.
                    values[row['dataset']].append(credit);invalid[row['dataset']]+=int(parsed is None)
                for name,scores in values.items():result[role]['retention'][name]=dict(n=len(scores),score=100*sum(scores)/len(scores),parse_failures=invalid[name])
        result[role]['public_macro']=sum(x['score'] for x in result[role]['public'].values())/7
        result[role]['retention_macro']=sum(x['score'] for x in result[role]['retention'].values())/3
    deltas={group:{name:result['joint'][group][name]['score']-item['score'] for name,item in result['base'][group].items()} for group in ['public','retention']}
    report=dict(state='complete',results=result,joint_minus_base_percentage_points=deltas,prediction_sha256=artifacts,
        protocol='Matched full seven public benchmarks and10105 official VQA examples; current runtime1536 image tokens; not the older sampled1111-panel',
        paper_main_table_target=dict(cv_bench=88.21,embspatial=82.34,refspatial_bench=59.,where2place=68.,blink_relative_depth=91.13,erqa=50.52,vabench_point=80.67,macro=74.27),
        current_model_role='Full-corpus SFT joint; not final MOPD',publication_results_reproduced=False)
    (ROOT/'benchmark_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'public_macro':{k:v['public_macro'] for k,v in result.items()},'deltas':deltas},indent=2))

if __name__=='__main__':main()
