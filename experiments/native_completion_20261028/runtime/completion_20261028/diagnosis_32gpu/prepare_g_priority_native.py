"""Prepare full G candidate validation; preserve completed matched public evidence."""
from pathlib import Path
import hashlib,json,sys
import yaml
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;R=C.parent
sys.path.insert(0,str(C/'eval_code'))
from eval_core import select_files

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 public=ROOT/'spatial_comparison_G.json';assert json.loads(public.read_text())['state']=='complete'
 model=ROOT/'merged_spatial_G_spatial';merge=json.loads((model/'merge_receipt.json').read_text());assert merge['state']=='verified_derived_inference_copy'
 completion=ROOT/'spatial_repairs/G_spatial/complete.json';receipt=json.loads(completion.read_text());assert receipt['step']==1024 and not receipt['smoke']
 assert merge['adapter_sha256']==receipt['adapter_sha256'] and merge['original_adapter']==receipt['checkpoint']
 export=R.parent/'arxiv/native_evaluation_export_20261012_v4';manifest=json.loads((export/'manifest.json').read_text())
 tasks=[dict(shard=i,shards=16,node=i//8,local_gpu=i%8,records=sum(info['shard_records'] for _,info in select_files(manifest,'G',i,16))) for i in range(16)]
 assert sum(t['records'] for t in tasks)==390528 and len({(t['node'],t['local_gpu']) for t in tasks})==16
 files=[ROOT/'g_priority_native_node.py',ROOT/'score_g_priority_native.py',ROOT/'retention_node.py',completion,public,export/'manifest.json',model/'merge_receipt.json']+list((C/'eval_code').rglob('*.py'))
 spec=dict(schema='anvil-priority-full-G-validation-v1',model=str(model),native_export=str(export),tasks=tasks,records=390528,
  model_files={name:info['derived_sha256'] for name,info in merge['shards'].items()},frozen_files={str(p):sha(p) for p in files},
  replaced_task='t-20261029053626-h2glq',reason='Both matched public/general reports complete. Concentrate full native validation on useful G candidate; cancelled old native partial outputs preserved and never pooled into this run.',
  scorer_changed=False,heldout_answers_used_in_generation=False,teacher_qualified=False)
 target=ROOT/'g_priority_native_spec.json';assert not target.exists();target.write_text(json.dumps(spec,indent=2)+'\n')
 for node in range(2):
  config=yaml.safe_load((ROOT/'prism_expanded_cli.yaml').read_text());config['TaskName']='task_anvil_G_priority_native_A800x8_node'+str(node)
  config['Description']='Full390528 nativeG validation on selected G_spatial model; static16shards across2nodes; original public/control evidence preserved'
  config['Entrypoint']='set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n'+str(R/'.venv-train/bin/python')+' -u '+str(ROOT/'g_priority_native_node.py')+' --node '+str(node)
  (ROOT/('g_priority_native_node'+str(node)+'_cli.yaml')).write_text(yaml.safe_dump(config,sort_keys=False))
 print('16 disjoint shards cover exactly390528 original G examples; two8GPU configs prepared')

if __name__=='__main__':main()
