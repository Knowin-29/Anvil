"""Submit the fixed student G/S diagnostic after current evaluations and capacity."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import fcntl,json,os,subprocess,time
from submit_student_when_capacity import check,save,sha,TERMINAL
D=Path(__file__).resolve().parent;C=D.parent;R=C.parent


def main():
    with open('/tmp/anvil-student-matched-submit.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=D/'student_matched_native_submission.json'
        if receipt.exists():return
        frozen={str(p):sha(p) for p in [Path(__file__),D/'student_matched_native.py',D/'student_matched_native_selection.json']}
        end=time.time()+7200
        while time.time()<end:
            for path,digest in frozen.items():assert sha(path)==digest,path
            status=json.loads((D/'student_adaptation_node_status.json').read_text())
            assert status['stage']!='failed','Student task failed; no dependent submission'
            if status['stage']!='complete_requires_quality_review':
                save(D/'student_matched_native_submission_status.json',dict(state='waiting_for_student_evaluation',student_stage=status['stage'],pid=os.getpid(),updated=time.time()))
                time.sleep(30);continue
            spec_path=D/'student_matched_native_spec.json'
            if not spec_path.exists():
                with (D/'student_matched_native_preparation.log').open('a') as log:
                    subprocess.run([str(R/'.venv-train/bin/python'),str(D/'student_matched_native.py'),'--prepare'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            spec=json.loads(spec_path.read_text());config=D/'student_matched_native_cli.yaml'
            for path,digest in spec['frozen_files'].items():assert sha(path)==digest,path
            jobs=json.loads((D/'jobs.json').read_text())+json.loads((C/'jobs_v2.json').read_text())
            assert not any(j['scope']=='adapted_student_predeclared_GS_35989_sample' for j in jobs)
            with ThreadPoolExecutor(max_workers=8) as pool:cloud=list(pool.map(check,jobs))
            active=sum(j['gpus'] for j in cloud if j['status'] not in TERMINAL);assert active<=32
            save(D/'student_matched_native_submission_status.json',dict(state='waiting_for_capacity',active=active,requested=8,pid=os.getpid(),updated=time.time()))
            if active+8>32:time.sleep(30);continue
            save(D/'student_matched_native_capacity.json',dict(cloud=cloud,gpus_before=active,requested=8,ceiling=32,checked=time.time()))
            proc=subprocess.run(['/root/.volc/bin/volc','--log-level','error','ml_task','submit','--conf',str(config),'--output','json'],capture_output=True,text=True,timeout=60)
            (D/'student_matched_native_submission.log').write_text(proc.stdout+proc.stderr)
            responses=[]
            for line in proc.stdout.splitlines():
                try:value=json.loads(line)
                except json.JSONDecodeError:continue
                if isinstance(value,dict) and isinstance(value.get('Id'),str):responses.append(value)
            assert proc.returncode==0 and len(responses)==1,'Ambiguous submission: inspect, never resubmit automatically'
            response=responses[0];save(receipt,dict(response=response,spec_sha256=sha(spec_path),config_sha256=sha(config),submitted=time.time()))
            registry=json.loads((D/'jobs.json').read_text());assert response['Id'] not in {j['id'] for j in registry}
            registry.append(dict(id=response['Id'],gpus=8,scope='adapted_student_predeclared_GS_35989_sample',config=str(config)));save(D/'jobs.json',registry)
            save(D/'student_matched_native_submission_status.json',dict(state='submitted',id=response['Id'],gpus_after=active+8,updated=time.time()))
            print(response['Id'],flush=True);return
        raise TimeoutError('Student/capacity did not become ready within2hours; no task submitted')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(D/'student_matched_native_submission_failure.json',dict(error=repr(exc),time=time.time()));raise
