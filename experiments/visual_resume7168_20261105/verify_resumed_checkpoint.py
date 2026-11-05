"""Check the first regular production save after exact7168 recovery."""
from pathlib import Path
import json,subprocess,time,hashlib
W=Path(__file__).resolve().parent;C=W.parent;V=C/'visual_geometry_candidate_20261104'
policy=json.loads((W/'resume_policy.json').read_text());checkpoint=V/'runs/repair/checkpoint-7424';out=W/'resume7424_verification.json'
assert not out.exists()
while not (checkpoint/'checkpoint_complete.json').exists():
 if list(W.glob('node*_failure.json')) or (W/'controller_failure.json').exists():raise RuntimeError('Recovery failed')
 time.sleep(30)
visual=json.loads((checkpoint/'visual_update.json').read_text());assert visual['baseline_step']==7168
with (W/'resume7424_verification.log').open('a') as log:
 subprocess.run([str(C.parent/'.venv-train/bin/python'),str(V/'verify_checkpoint.py'),'--spec-sha256',policy['pipeline_spec_sha256'],'--stage','repair','--step','7424','--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
receipt=json.loads(out.read_text());assert receipt['state']=='visual_checkpoint_integrity_verified'
(W/'runtime_resume_verified.json').write_text(json.dumps(dict(state='exact7168_resume_verified_at7424',checkpoint=str(checkpoint),baseline=7168,optimizer_step=receipt['optimizer_step'],all32_rng_files_loadable=receipt['all32_rng_files_loadable'],visual_b_tensors_changed=receipt['changed_visual_b_tensors'],receipt_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),time=time.time()),indent=2)+'\n')
