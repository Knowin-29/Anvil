"""Observe the existing task. Never submit, stop, retry or mutate GPU work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
C=Path(__file__).resolve().parent.parent
P=C/'grounded_repair_20261102'
TASK='t-20261102141249-rmwhl'
parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=int,default=10800)
parser.add_argument('--milestone',choices=['repair_4096','repair_8192','mopd_128','mopd_256'],default='repair_4096')
args=parser.parse_args()
start=time.monotonic()
print(json.dumps(dict(observer_pid=os.getpid(),task=TASK,window_seconds=args.seconds,gpu_mutations=False)),flush=True)
while time.monotonic()-start<args.seconds:
    try:
        proc=subprocess.run(['/opt/volc/bin/volc','--log-level','error','ml_task','get','--id',TASK,
            '--output','json','--format','Id,Name,Status'],capture_output=True,text=True,timeout=30)
        rows=[json.loads(line.strip())[0] for line in proc.stdout.splitlines() if line.strip().startswith('[{"Id"')]
        cloud=rows[0]['Status'] if len(rows)==1 and proc.returncode==0 else 'UNKNOWN'
    except (subprocess.TimeoutExpired,json.JSONDecodeError):
        cloud='UNKNOWN'
    stage='mopd' if (P/'runs/mopd/status.json').exists() else 'repair'
    run=P/'runs'/stage
    state=json.loads((run/'status.json').read_text())
    saved=json.loads((run/'latest.json').read_text()) if (run/'latest.json').exists() else {}
    backup=json.loads((C/'grounded_checkpoint_backup_20261102/status.json').read_text())
    phases=[json.loads(f.read_text()) for f in run.glob('phase_rank*.json')]
    result=dict(observed=time.time(),task=TASK,cloud=cloud,stage=stage,step=state.get('step'),saved=saved.get('step'),
        log_age_seconds=round(time.time()-state['updated']),rank_phase_max_age_seconds=round(max((time.time()-d['updated'] for d in phases),default=0)),
        verified_hf_checkpoints=backup.get('verified_checkpoints'),backup_state=backup['stage'])
    result['long_step_observation_warning']=result['log_age_seconds']>900
    print(json.dumps(result),flush=True)
    failure=list(P.glob('node*_failure.json'))+[f for f in [C/'grounded_checkpoint_backup_20261102/failure.json',
        C/'delivery_runs/grounded_repair_20261102/failure.json',C/'verification_20261101/grounded_merged_blob_preupload/failure.json'] if f.exists()]
    if failure or cloud in ['Success','Failed','Stopped','Canceled']:
        print('Observation requires review; no GPU action taken',flush=True);break
    if (C/'grounded_checkpoint_backup_20261102'/args.milestone/'verification_receipt.json').exists():
        print('Milestone: new key checkpoint remotely verified',flush=True);break
    time.sleep(50)
else:
    print('Observation window expired; GPU task and background workers continue independently',flush=True)
