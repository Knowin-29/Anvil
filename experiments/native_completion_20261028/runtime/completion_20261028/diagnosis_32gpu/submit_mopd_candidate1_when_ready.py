"""Queue execution-only MOPD after the student diagnostic has reserved its GPUs.

The dependency serializes cloud submissions with the existing student controller.
This never qualifies teachers, starts formal MOPD, or publishes weights.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import fcntl,json,os,subprocess,time
from submit_student_when_capacity import check,save,sha,TERMINAL
D=Path(__file__).resolve().parent;C=D.parent;O=D/'mopd_v6_candidate1'


def main():
    with open('/tmp/anvil-mopd-v6-candidate1-submit.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=O/'submission.json'
        if receipt.exists():return
        spec_path=O/'experiment.json';config=O/'cli.yaml'
        spec_sha,config_sha=sha(spec_path),sha(config)
        spec=json.loads(spec_path.read_text())
        assert spec['formal_mopd'] is False and spec['teacher_qualified'] is False and spec['probe_steps']==32
        for path,digest in spec['frozen_files'].items():assert sha(path)==digest,path
        end=time.time()+10800
        while time.time()<end:
            assert sha(spec_path)==spec_sha and sha(config)==config_sha
            assert not (D/'student_matched_native_submission_failure.json').exists(),'Student diagnostic submission failed'
            dependency=D/'student_matched_native_submission.json'
            registry=json.loads((D/'jobs.json').read_text())
            if not dependency.exists() or json.loads(dependency.read_text())['response']['Id'] not in {j['id'] for j in registry}:
                save(O/'submission_status.json',dict(state='waiting_for_student_diagnostic_submission',updated=time.time(),pid=os.getpid()))
                time.sleep(30);continue
            jobs=registry+json.loads((C/'jobs_v2.json').read_text())
            assert not any(j['scope']=='execution_only_exact_MOPDv6_candidate1_resume_32probe' for j in jobs)
            with ThreadPoolExecutor(max_workers=8) as pool:cloud=list(pool.map(check,jobs))
            active=sum(j['gpus'] for j in cloud if j['status'] not in TERMINAL);assert active<=32
            save(O/'submission_status.json',dict(state='waiting_for_capacity',active=active,requested=8,updated=time.time(),pid=os.getpid()))
            if active+8>32:time.sleep(30);continue
            for path,digest in spec['frozen_files'].items():assert sha(path)==digest,path
            save(O/'capacity.json',dict(cloud=cloud,gpus_before=active,requested=8,ceiling=32,checked=time.time()))
            proc=subprocess.run(['/root/.volc/bin/volc','--log-level','error','ml_task','submit','--conf',str(config),'--output','json'],capture_output=True,text=True,timeout=60)
            (O/'submission.log').write_text(proc.stdout+proc.stderr)
            responses=[]
            for line in proc.stdout.splitlines():
                try:value=json.loads(line)
                except json.JSONDecodeError:continue
                if isinstance(value,dict) and isinstance(value.get('Id'),str):responses.append(value)
            assert proc.returncode==0 and len(responses)==1,'Ambiguous submission; inspect, never automatically retry'
            response=responses[0];save(receipt,dict(response=response,submitted=time.time(),spec_sha256=spec_sha,config_sha256=config_sha))
            registry=json.loads((D/'jobs.json').read_text());assert response['Id'] not in {j['id'] for j in registry}
            registry.append(dict(id=response['Id'],gpus=8,scope='execution_only_exact_MOPDv6_candidate1_resume_32probe',config=str(config)));save(D/'jobs.json',registry)
            save(O/'submission_status.json',dict(state='submitted',id=response['Id'],gpus_after=active+8,updated=time.time()))
            print(response['Id'],flush=True);return
        raise TimeoutError('Dependency/capacity unavailable within3hours; no submission')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(O/'submission_failure.json',dict(error=repr(exc),time=time.time()));raise
