"""Fixed-codebook error and expanded-train priors; never modify model scores."""
from collections import defaultdict,Counter
import json,hashlib,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent
sys.path[:0]=[str(C/'eval_code/score_code'),str(C/'Anvil/tools'),str(R/'cloud_a800')]
from eval.native_metrics import score_row
from checkpoints import save
from dof_codebook_interface import ROTATIONS
from dof_codec_interface import context
from dof_codec_prototype import encode,decode
from prepare_dof_rotation_codebook import distances

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def medoid(values):
 rotations=np.asarray([v[1] for v in values]);sums=[]
 for start in range(0,len(values),128):sums.extend(distances(rotations[start:start+128],rotations).sum(1))
 return rotations[int(np.argmin(sums))]

def main():
 start=time.time();paths={'original':R.parent/'arxiv/native_training_export_20261012_v4_r2/train_dof.jsonl','supplement':ROOT/'prism_expansion/train_bound.jsonl','heldout':R.parent/'arxiv/native_evaluation_export_20261012_v4/eval_dof.jsonl'}
 datasets={name:[json.loads(l) for l in p.open()] for name,p in paths.items()};assert [len(datasets[k]) for k in paths]==[3008,13536,580]
 quantization={};groups=defaultdict(list);labels={}
 for name,rows in datasets.items():
  rotations=np.asarray([json.loads(row['messages'][1]['content'])['camera_from_gripper'] for row in rows])[:,:3,:3]
  error=np.degrees(distances(rotations,ROTATIONS).min(1))
  quantization[name]=dict(n=len(rows),mean_degrees=float(error.mean()),median_degrees=float(np.median(error)),p95_degrees=float(np.percentile(error,95)),max_degrees=float(error.max()))
  labels[name]=dict(Counter(context(row['messages'][0]['content'])['entity']['label'] for row in rows))
  if name=='heldout':continue
  for row in rows:
   c=context(row['messages'][0]['content']);y=json.loads(row['messages'][1]['content']);entry=(encode(y,c),np.asarray(y['camera_from_gripper'])[:3,:3])
   groups[c['entity']['label']].append(entry);groups['ALL'].append(entry)
 priors={}
 for label,values in groups.items():
  latent=np.median([v[0] for v in values],axis=0);rotation=medoid(values);latent[:3]=rotation[:,0];latent[3:6]=rotation[:,1];priors[label]=latent
 sums=Counter();output=ROOT/'prism_expanded_train_prior_predictions.jsonl'
 with output.open('x') as f:
  for row in datasets['heldout']:
   c=context(row['messages'][0]['content']);latent=priors.get(c['entity']['label'],priors['ALL']).copy();b=c['entity']['bbox_2d']
   uv=np.clip((np.asarray(b[:2])+b[2:])/2000,1e-8,1-1e-8);latent[9:11]=np.log(uv/(1-uv))
   response=json.dumps(decode(latent,c));metrics=score_row(row,response);sums.update(metrics)
   f.write(json.dumps(dict(id=row['id'],response=response,metrics=metrics))+'\n')
 report=dict(state='complete_diagnostic_only',fixed_original_codebook_quantization=quantization,source_label_counts=labels,
  expanded_training_only_category_prior={k:v/580 for k,v in sums.items()},n_heldout=580,n_prior_training=16544,
  no_pixels_used_by_prior=True,codebook_refitted=False,heldout_used_to_fit=False,teacher_qualified=False,
  model_predictions_modified=False,elapsed=time.time()-start,bindings={str(p):sha(p) for p in [*paths.values(),ROOT/'dof_rotation_codebook.json',Path(__file__),output]})
 save(ROOT/'prism_pose_coverage_audit.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
