"""Submit one authorized continuation after the parent's full evaluation finishes.

Reuse the existing campaign registry and capacity lock. Unknown cloud states
remain charged; an uncertain submission is never automatically repeated.
"""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

C = Path(__file__).resolve().parent
H = C/'final_delivery_targeted_v1'
sys.path.insert(0, str(H))
from common import TRAIN, read, require, save, sha
sys.path.insert(0, str(C))
from final_delivery_handoff_v1 import cloud_snapshot, submission_response
sys.path.insert(0, str(C/'formal_mopd'))
from controller import register


def validate_policy(policy):
    require(policy['schema'] == 'anvil-targeted-continuation-submission-v1', 'Unknown submission policy')
    require(policy['gpus'] == 32, 'Wrong GPU allocation')
    for path, digest in policy['files'].items():
        require(sha(path) == digest, 'Prepared source or policy changed: ' + path)
    spec = read(policy['spec'])
    require(spec['world'] == 32 and spec['global_batch'] == 64, 'Wrong distributed training budget')
    for path, digest in spec['frozen_files'].items():
        require(sha(path) == digest, 'Training dependency changed: ' + path)
    require(sha(Path(spec['initial']['adapter'])/'adapter_model.safetensors') ==
            spec['initial']['adapter_sha256'], 'Parent checkpoint changed')
    return spec


def parent_report_ready(policy):
    path = Path(policy['parent_report'])
    if not path.exists():
        return False
    report = read(path)
    require(report['state'] == 'complete_requires_quality_review' and
            report['final_spec_sha256'] == policy['parent_evaluation_spec_sha256'], 'Different parent report')
    require(report['native']['n'] == 580000 and sum(v['n'] for v in report['general'].values()) == 10105,
            'Incomplete parent evaluation')
    trigger = read(policy['trigger'])
    require(report['paper_aligned_public'] == trigger['scores'] and
            trigger['final_quality_accepted'] is False, 'Changed corrective-training trigger')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--inspect', action='store_true')
    args = ap.parse_args()
    policy = read(args.policy)
    spec = validate_policy(policy)
    root = Path(policy['spec']).parent
    deadline = datetime.fromisoformat(spec['deadline_utc']).timestamp()
    if args.inspect:
        print('Submission policy and frozen sources verified; no cloud mutation')
        return
    own = (root/'controller.lock').open('a')
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **values):
        save(root/'controller_status.json', dict(stage=stage, updated=time.time(), pid=os.getpid(),
                                                weights_uploaded=False, **values))
    receipt_path = root/'submission.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['spec_sha256'] == sha(policy['spec']) and
                receipt['config_sha256'] == sha(policy['config']), 'Existing submission differs')
    else:
        require(not (root/'submission_intent.json').exists(), 'Uncertain prior submission; inspect cloud before retry')
        while time.time() < deadline:
            if not parent_report_ready(policy):
                status('waiting_parent_full_evaluation')
                time.sleep(45)
                continue
            lock = open('/tmp/anvil-formal-mopd-submit.lock', 'a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                status('waiting_campaign_submission_lock')
                time.sleep(30)
                continue
            try:
                cloud, active = cloud_snapshot()
                parent_jobs = {j['id']:j for j in cloud if j['id'] in policy['parent_jobs']}
                require(len(parent_jobs) == 4, 'Parent evaluation jobs missing from registry')
                if not all(j['status'] == 'Success' for j in parent_jobs.values()) or active + 32 > 32:
                    status('waiting_campaign_capacity', active_campaign_gpus=active,
                           parent_jobs={k:v['status'] for k,v in parent_jobs.items()})
                else:
                    validate_policy(policy)
                    save(root/'capacity.json', dict(cloud=cloud, active=active, requested=32, cap=32, checked=time.time()))
                    intent = dict(config=str(policy['config']), config_sha256=sha(policy['config']),
                                  spec_sha256=sha(policy['spec']), gpus=32, time=time.time())
                    save(root/'submission_intent.json', intent)
                    proc = subprocess.run(['/root/.volc/bin/volc','--log-level','error','ml_task','submit',
                                           '--conf',policy['config'],'--output','json'],
                                          capture_output=True,text=True,timeout=90)
                    (root/'submission.log').write_text(proc.stdout+proc.stderr)
                    receipt = dict(intent,response=submission_response(proc.stdout,proc.returncode))
                    save(receipt_path,receipt)
                    register(receipt,32,'targeted_repair4096_MOPD768',policy['config'])
                    break
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN)
                lock.close()
            time.sleep(45)
        else:
            raise TimeoutError('No authorized continuation submission before campaign deadline')
    register(receipt,32,'targeted_repair4096_MOPD768',policy['config'])
    status('submitted_handoff_running',task_id=receipt['response']['Id'])
    handoff_log = Path(policy['handoff_policy']).parent/'supervisor.log'
    with handoff_log.open('a') as stream:
        subprocess.run([TRAIN,'-u',policy['handoff_source'],'--policy',policy['handoff_policy']],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    status('full_final_report_ready_requires_quality_review', task_id=receipt['response']['Id'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if '--policy' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--policy')+1]).parent
            save(root/'controller_failure.json',dict(error=repr(exc),time=time.time()))
        raise
