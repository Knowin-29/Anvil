"""Conditionally deepen a promising objective control without exceeding32 GPUs."""
import fcntl
import os
from pathlib import Path
import sys
import time
C = Path(__file__).resolve().parent
sys.path[:0] = [str(C), str(C / 'formal_mopd')]
import controller
from formal_common import read, require, save, sha
from final_delivery_handoff_v1 import observe
from submit_midrun_gs_diagnostic import available_with_public_reservation
import objective_native_diagnostic as diagnostic

ROOT = C / 'objective_native_diagnostics/low_native_sft'
O = C / 'objective_controls_20261029'
SCOPE = 'objective_low_native_sft_GS35989_diagnostic'


def register(receipt, gpus, scope, config):
    require(gpus == 8 and scope == SCOPE, 'Unexpected diagnostic registration')
    path = C / 'jobs_v2.json'; jobs = read(path); identity = receipt['response']['Id']
    if identity not in {j['id'] for j in jobs}:
        jobs.append(dict(id=identity, gpus=8, scope=scope, config=str(config)))
        save(path, jobs)


def main():
    policy = read(ROOT / 'policy.json')
    for path, expected in policy['frozen_files'].items():
        require(sha(path) == expected, 'Diagnostic policy dependency changed')
    require(sha(policy['cpu_validation']['path']) == policy['cpu_validation']['sha256']
            and read(policy['cpu_validation']['path'])['state'] == 'passed', 'CPU validation missing')
    lock = (ROOT / 'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    controller.check = observe; controller.register = register
    public_path = O / 'evaluation/low_native_sft/spec.json'
    report_path = public_path.parent / 'report.json'
    while time.time() < policy['launch_deadline']:
        if (ROOT / 'submission.json').exists():
            receipt=read(ROOT/'submission.json');register(receipt,8,SCOPE,ROOT/'cli.yaml')
            save(ROOT/'controller_status.json',dict(stage='submitted',id=receipt['response']['Id'],updated=time.time(),pid=os.getpid()))
            return
        if not report_path.exists():
            save(ROOT/'controller_status.json',dict(stage='waiting_complete_public_control',updated=time.time(),pid=os.getpid()))
            time.sleep(30);continue
        report = read(report_path)
        require(report['state']=='complete_objective_control_requires_review' and report['final_mopd'] is False,
                'Invalid control report')
        require(report['checkpoint_spec_sha256']==sha(public_path),'Mixed public evidence')
        if report['paper_aligned_macro'] < policy['minimum_macro_for_diagnostic']:
            save(ROOT/'controller_status.json',dict(stage='not_submitted_public_diagnostic_condition_unmet',
                macro=report['paper_aligned_macro'],minimum=policy['minimum_macro_for_diagnostic'],
                quality_rejection_inferred=False,note='Budget rule for an additional native diagnostic, not final model admission.',
                updated=time.time(),pid=os.getpid()))
            return
        cloud,active=controller.cloud_snapshot();by_id={j['id']:j for j in cloud}
        capacity,allocation=available_with_public_reservation(cloud)
        done=by_id[policy['control_task']]['status'] in {'Success','Succeeded'}
        if not capacity or not done:
            save(ROOT/'controller_status.json',dict(stage='waiting_capacity_and_control_success',allocation=allocation,
                control_status=by_id[policy['control_task']]['status'],updated=time.time(),pid=os.getpid()))
            time.sleep(30);continue
        for path,expected in policy['frozen_files'].items():
            require(sha(path)==expected,'Diagnostic dependency changed before submission')
        spec=diagnostic.prepare(public_path,ROOT);diagnostic.load_spec(spec)
        save(ROOT/'reserved_capacity.json',dict(cloud=cloud,**allocation,checked=time.time()))
        identity=controller.submit(ROOT,ROOT/'cli.yaml',8,SCOPE,
            C/'formal_mopd/runs/candidate1_20261029/plan.json')
        if identity:
            save(ROOT/'controller_status.json',dict(stage='submitted',id=identity,updated=time.time(),pid=os.getpid()))
            return
        time.sleep(30)
    save(ROOT/'controller_status.json',dict(stage='not_submitted_launch_window_expired',updated=time.time(),pid=os.getpid()))


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        save(ROOT/'controller_failure.json',dict(error=repr(exc),time=time.time()))
        raise
