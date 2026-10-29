"""Real-source schedule and encoding preflight, without claiming GPU/model quality."""
from collections import Counter,defaultdict
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent
sys.path[:0]=[str(ROOT),str(C),str(R),str(R/'cloud_a800')]
import torch
from student_adaptation_pool import Pool
from dof_codebook_interface import training_row
from dof_pose_loss import PoseLoss
from training_data import processor,encode
from checkpoints import save

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 torch.set_num_threads(8);pool=Pool();p=processor();counts=Counter();unique=defaultdict(set);forms=Counter();examples={};snapshots={}
 for step in range(1024):
  for micro in range(4):
   for rank in range(8):
    row,source=pool.sample(step*4+micro,rank,8)
    counts[source]+=1;unique[source].add(row['canonical_uid']);forms[row['gsa_form']]+=1
    examples.setdefault((source,row['gsa_form']),row)
  if step+1 in (512,1024):snapshots[str(step+1)]=dict(planned_exposures=dict(counts),planned_unique_records={k:len(v) for k,v in unique.items()},by_form=dict(forms))
 assert sum(counts.values())==32768 and len(forms)==12 and len(unique['supplemental_prism'])>12000
 assert counts['original_native']==8192 and sum(counts[k] for k in ['ai2d','scienceqa','textvqa'])==4096
 assert counts['fsd_goal']==3072 and counts['refspatial_vacancy']==1024
 assert counts['original_dof']+counts['supplemental_prism']==16384
 assert len({row['canonical_uid'] for row in pool.union})==16544
 encodings=[]
 for (source,form),row in examples.items():
  original=row['messages'][1]['content'];derived=training_row(row) if form=='dof' else row
  data=encode(p,derived);length=data['input_ids'].shape[1];assert length<=16384
  if form=='dof':
   for mode in ['hard','angular']:
    meta=PoseLoss(p.tokenizer,mode).target(derived,original)
    assert data['labels'][0,-len(meta['ids']):].tolist()==meta['ids']
  encodings.append(dict(source=source,form=form,sequence_tokens=length))
 result=dict(state='passed_cpu',world=8,global_batch=32,schedules=snapshots,actual_image_encodings=encodings,source_manifests=pool.source_manifests,
  originals_modified=False,training_started=False,gpu_runtime_validation=False,final_mopd=False,
  files_sha256={str(path):sha(path) for path in [Path(__file__),ROOT/'student_adaptation_train.py',ROOT/'student_adaptation_pool.py',ROOT/'dof_pose_loss.py']})
 save(ROOT/'student_adaptation_cpu_validation.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
