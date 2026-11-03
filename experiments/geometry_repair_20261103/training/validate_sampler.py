"""CPU-only coverage check for an unsubmitted sampling candidate."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,time
from pool import Pool,SCHEDULES,FORMS
O=Path(__file__).resolve().parent
spec=json.loads((O/'sampling_config.json').read_text())
assert spec['submitted'] is False and spec['training_source_ready'] is False
assert spec['schedules']==SCHEDULES
start=time.time();pool=Pool(spec);counts=Counter();forms=Counter();uids=defaultdict(set);native_forms=set()
for step in range(spec['repair_steps']):
 for micro in range(spec['global_batch']//spec['world']):
  for rank in range(spec['world']):
   row,kind,source=pool.sample('repair',step,micro,rank,spec['world'],spec['global_batch'])
   assert row['gsa_meta']['split']=='train' and row['canonical_uid']
   counts[kind]+=1;forms[row['gsa_form']]+=1;uids[kind].add(row['canonical_uid'])
   if kind=='native':native_forms.add(row['gsa_form'])
assert sum(counts.values())==262144
assert counts['mask']==65536 and len(uids['mask'])==65536
assert counts['prism']==32768 and uids['prism']=={x['canonical_uid'] for x in pool.rows['prism']}
assert len(uids['prism'])==16544
assert len(uids['affordance'])==len(pool.offsets['affordance'])==5907
assert native_forms==set(FORMS)
# On-policy MOPD schedule stays exactly the frozen schedule; every native form is reachable.
original=json.loads((O.parent/'format_repair_20261103/schedules.json').read_text())
# The archived schedule file can include additional metadata; compare frozen Python AST instead.
import ast
p=O.parent/'format_repair_20261103/pool.py';tree=ast.parse(p.read_text())
old=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCHEDULES' for t in n.targets))
assert SCHEDULES['mopd']==old['mopd']
result=dict(state='candidate_full_repair_sampler_cpu_validated',candidate_submitted=False,gpu_jobs=0,records_checked=sum(counts.values()),role_exposures=dict(counts),form_exposures=dict(forms),unique_records_per_role={k:len(v) for k,v in uids.items()},all11_native_forms_retained=True,all16544_admitted_prism_records_reachable=True,all5907_admitted_affordance_records_reachable=True,mopd_schedule_unchanged=True,elapsed_seconds=time.time()-start,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [O/'pool.py',O/'point_contracts.py',O/'sampling_config.json',Path(__file__)]},limitations=['Coverage does not establish quality improvement.','No GPU execution or extra diagnostic task.','Training, submission, checkpoint-backup and final-delivery bindings still need preparation before launch.'])
(O/'sampler_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}))
