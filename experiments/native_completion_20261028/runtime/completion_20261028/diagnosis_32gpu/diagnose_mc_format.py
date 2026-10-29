"""Supplemental format diagnosis; does not replace or change frozen benchmark scores."""
from collections import defaultdict
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'code'))
from general_retention_formal_v3 import parse_multiple_choice,_answer_span
inputs={r['global_uid']:r for r in map(json.loads,(ROOT/'inputs/retention.jsonl').open())}
result={}
for role in ['base','joint']:
    stats=defaultdict(lambda:dict(n=0,strict_correct=0,strict_invalid=0,unambiguous_letter_plus_exact_option=0,additional_correct=0))
    for path in (ROOT/'benchmarks'/role).glob('*/retention.jsonl'):
        for pred in map(json.loads,path.open()):
            row=inputs[pred['uid']]
            if row['metric']!='multiple_choice_accuracy':continue
            s=stats[row['dataset']];s['n']+=1;parsed,_=parse_multiple_choice(pred['response'],row['choices'])
            s['strict_correct']+=parsed==row['answer'];s['strict_invalid']+=parsed is None
            if parsed is None:
                span,_=_answer_span(pred['response']);m=re.fullmatch(r'([A-Z])[.)]\s+(.+)',span or '',re.DOTALL)
                if m and ord(m[1])-65<len(row['choices']):
                    normalize=lambda text:' '.join(text.casefold().split()).rstrip('.')
                    if normalize(m[2])==normalize(row['choices'][ord(m[1])-65]):
                        s['unambiguous_letter_plus_exact_option']+=1;s['additional_correct']+=m[1]==row['answer']
    result[role]=dict(stats)
(ROOT/'mc_format_diagnosis.json').write_text(json.dumps(dict(results=result,official_scores_changed=False,rule='Supplementary: exact option letter plus matching full option text only; no ground-truth-dependent parsing'),indent=2)+'\n')
print(json.dumps(result,indent=2))
