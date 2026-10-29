"""Score only fully generated public sets while general/native evaluations continue."""
from collections import Counter
import hashlib,inspect,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from gsa_eval.registry import BENCHMARKS
from gsa_eval.datasets import Sample

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 manifest=json.loads((ROOT/'inputs/manifest.json').read_text());results={}
 for role in ['joint_control','joint_spatial','G_control','G_spatial']:
  result={}
  for name,info in manifest['public'].items():
   source_path=ROOT/'inputs'/(name+'.jsonl');assert sha(source_path)==info['sha256']
   source=[json.loads(line) for line in source_path.open()];expected={str(r['uid']) for r in source};saved={};files={};pending=False
   for shard in range(4):
    path=ROOT/'spatial_benchmarks'/role/str(shard)/(name+'.jsonl')
    if not path.exists():pending=True;break
    # A producer may be writing a final line. Never score an incomplete set.
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):pending=True;break
    try:rows=[json.loads(line) for line in raw.splitlines()]
    except json.JSONDecodeError:pending=True;break
    expected_shard={str(r['uid']) for r in source[shard::4]}
    if {row['uid'] for row in rows}!=expected_shard:pending=True;break
    assert len(rows)==len(expected_shard)
    for row in rows:
     assert row['uid'] not in saved and hashlib.sha256(row['response'].encode()).hexdigest()==row['response_sha256']
     saved[row['uid']]=row
    files[str(path)]=hashlib.sha256(raw).hexdigest()
   if pending:continue
   assert set(saved)==expected
   samples=[Sample(**{k:v for k,v in row.items() if k!='max_tokens'}) for row in source]
   scorer=BENCHMARKS[name];kwargs={'spec':scorer} if 'spec' in inspect.signature(scorer.scorer).parameters else {}
   score,detail,failed=scorer.scorer(samples,[saved[str(s.uid)]['response'] for s in samples],**kwargs)
   value=detail['per_group']['location']['mean_in_mask'] if name=='refspatial_bench' else score
   result[name]=dict(n=len(source),full_score=score,paper_aligned_score=value,parse_failures=failed,capped=sum(x['hit_token_cap'] for x in saved.values()),prediction_sha256=files)
  results[role]=dict(completed_public_sets=result,full_seven_public_complete=len(result)==7,
   paper_aligned_macro=sum(x['paper_aligned_score'] for x in result.values())/7 if len(result)==7 else None,
   complete_general_or_native_evaluation_claimed=False,teacher_qualified=False)
 report=dict(state='interim_only_complete_public_sets',results=results,updated=time.time(),scorers_changed=False,final_mopd=False)
 path=ROOT/'spatial_completed_public_interim.json';path.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({role:{'macro':r['paper_aligned_macro'],'scores':{n:i['paper_aligned_score'] for n,i in r['completed_public_sets'].items()}} for role,r in results.items()},indent=2))

if __name__=='__main__':main()
