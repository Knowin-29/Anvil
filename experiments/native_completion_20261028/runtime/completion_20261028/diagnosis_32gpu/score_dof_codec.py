"""Summarize all controls and interventions; never auto-qualify a teacher."""
from collections import defaultdict,Counter
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
results={}
for arm in ['A_codec_specialist','A_codec_joint']:
    groups=defaultdict(lambda:dict(n=0,sums=defaultdict(float)));seen=set();counts=Counter();evidence={}
    for shard in range(4):
        out=ROOT/'codec_evaluations'/arm/str(shard);file=out/'evaluation/metrics.json';receipt=json.loads((out/'complete.json').read_text())
        digest=hashlib.sha256(file.read_bytes()).hexdigest();assert digest==receipt['metrics_sha256'];evidence[str(file)]=digest
        metrics=json.loads(file.read_text());assert metrics['state']=='complete_scope_shard'
        for name,expected in metrics['prediction_sha256'].items():
            pred=out/'evaluation'/name
            with pred.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
            for line in pred.open():
                row=json.loads(line);key=(row['native_form'],row['id']);assert key not in seen;seen.add(key)
                counts[row['finish_reason']]+=1
        for row in metrics['groups']:
            g=groups[(row['form'],row['source'])];g['n']+=row['n']
            for k,v in row['mean'].items():g['sums'][k]+=v*row['n']
    assert len(seen)==(4100 if arm[0]=='A' else 390528)
    results[arm]=dict(n=len(seen),finish_reasons=dict(counts),groups=[dict(form=f,source=s,n=g['n'],mean={k:v/g['n'] for k,v in g['sums'].items()}) for (f,s),g in groups.items()],evidence=evidence)
(ROOT/'dof_codec_comparison.json').write_text(json.dumps(dict(state='complete_requires_quality_review',results=results,teacher_qualified=False,final_mopd=False),indent=2)+'\n')
print('Both geometry-decoder candidates fully evaluated; quality approval still requires review.')
