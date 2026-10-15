#!/usr/bin/env python3
"""Recompute the accepted 17 native result cells from saved text, without GPUs."""
from pathlib import Path
import json,sys,os
root=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
os.environ['ANVIL_CONTRACT_MODULE']=str(root/'engineering/experiments/contract_v2.py')
sys.path[:0]=[str(root/'engineering/experiments'),str(root),str(root/'engineering')]
from linked_training import score_linked
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
source=rows(root/'results/linked_sensor_training_v1/qualification_linked.jsonl')
count=0
for group in ['core_gs_20260916','quality_gs_20260916']:
 for p in sorted((root/'results'/group).rglob('predictions.jsonl')):
  expected=p.parent/'result.json'
  if not expected.exists():continue
  e=json.loads(expected.read_text())
  if not ('records' in e and 'mean' in e and 'strict_chain' in e['mean']):continue
  actual=score_linked(source,rows(p))
  assert actual==e,str(p)
  count+=1
assert count==17,count
print(json.dumps({'verified_cells':count,'records_per_cell':len(source),'predictions_recomputed':count*len(source)}))
