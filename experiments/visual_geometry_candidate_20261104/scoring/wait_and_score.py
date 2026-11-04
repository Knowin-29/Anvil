"""Run one CPU aggregation after the exact current full evaluation completes."""
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import time
from score_records import aggregate,sha,require

Q=Path(__file__).resolve().parent;C=Q.parent
D=C/'delivery_runs/visual_geometry_20261104';J=C/'visual_delivery_control_20261104'
def read(p):return json.loads(p.read_text())
def save(p,v):
    t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)

def main():
    handle=(Q/'worker.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original=C/'anvil_eval_score_v1_1_20261104'
    seal=read(original/'protocol_seal.json')
    for name,digest in seal['files'].items():require(sha(original/name)==digest,'Changed sealed scorer')
    for name in ['score_records.py','PROTOCOL.md','test_score_records.py']:
        require(sha(Q/name)==seal['files'][name],'Changed candidate score definition')
    expiry=datetime.fromisoformat(read(C/'visual_geometry_candidate_20261104/spec.json')['deadline_utc']).timestamp()+3600
    report=D/'evaluation/final_report.json';out=Q/'final_native_score.json'
    if out.exists():return
    while time.time()<expiry:
        if report.exists() and read(D/'status.json')['stage']=='full_final_report_ready_requires_quality_review':break
        save(Q/'status.json',{'stage':'waiting_complete_current_evaluation','pid':os.getpid(),'updated':time.time()})
        time.sleep(45)
    else:raise TimeoutError('Full evaluation not ready')
    result=aggregate(report,D/'evaluation_spec.json',out)
    save(Q/'complete.json',{'state':'full_native_score_computed','score':result['score'],'n':result['n'],
        'artifact':str(out),'artifact_sha256':sha(out),'report_sha256':sha(report),'completed':time.time()})
    save(Q/'status.json',{'stage':'full_native_score_computed','pid':os.getpid(),'updated':time.time()})

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(Q/'failure.json',{'error_type':type(exc).__name__,'failed':time.time()});raise
