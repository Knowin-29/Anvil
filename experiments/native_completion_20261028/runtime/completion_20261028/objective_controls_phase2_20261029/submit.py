"""Schedule phase2 diagnostics around the existing formal512 public evaluation."""
from datetime import datetime
import fcntl
import os
import sys
import time
from objective_control_common import HERE,C,load_experiment,read,require,save,sha
sys.path[:0]=[str(C),str(C/'formal_mopd')]
import controller
from final_delivery_handoff_v1 import observe
from submit_midrun_gs_diagnostic import available_with_public_reservation

FORMAL=C/'formal_mopd/runs/candidate1_20261029'
OLD_NATIVE=C/'objective_native_diagnostics/low_native_sft'


def register(receipt,gpus,scope,config):
    require(gpus==8 and scope.startswith('objective_phase2_'),'Wrong phase2 registration')
    path=C/'jobs_v2.json';jobs=read(path);identity=receipt['response']['Id']
    if identity not in {j['id'] for j in jobs}:
        jobs.append(dict(id=identity,gpus=8,scope=scope,config=str(config)));save(path,jobs)


def ready(arm,cloud):
    by_id={job['id']:job for job in cloud}
    if arm=='no_native_sft':
        return available_with_public_reservation(cloud)[0], 'reserve8_for_formal_public'
    receipt_path=FORMAL/'evaluation/512/submission.json'
    if not receipt_path.exists() or not (receipt_path.parent/'report.json').exists():
        return False,'wait_complete_formal512_public_report'
    identity=read(receipt_path)['response']['Id']
    if by_id.get(identity,{}).get('status') not in {'Success','Succeeded'}:
        return False,'wait_formal512_public_cloud_success'
    step=read(FORMAL/'training/status.json')['step']
    if step>=768 or (FORMAL/'training/checkpoint-1024').exists():
        return available_with_public_reservation(cloud)[0],'reserve8_near_formal1024'
    active=sum(job['gpus'] for job in cloud if job['status'] not in controller.TERMINAL)
    return active+8<=32,'short_control_after512_before1024'


def main():
    experiment=load_experiment();lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    controller.check=observe;controller.register=register
    deadline=datetime.fromisoformat('2026-10-29T20:00:00+08:00').timestamp()
    while time.time()<deadline:
        # Serialize secondary-registry writers: the earlier conditional native
        # controller must finish its no-submission decision before this one acts.
        previous=read(OLD_NATIVE/'controller_status.json')
        if previous['stage']!='not_submitted_public_diagnostic_condition_unmet':
            save(HERE/'controller_status.json',dict(stage='waiting_prior_native_controller_decision',
                previous_stage=previous['stage'],updated=time.time(),pid=os.getpid()))
            time.sleep(30);continue
        all_submitted=True
        for arm in experiment['arms']:
            folder=HERE/'submissions'/arm;scope='objective_phase2_'+arm
            if (folder/'submission.json').exists():
                register(read(folder/'submission.json'),8,scope,folder/'cli.yaml');continue
            all_submitted=False
            cloud,active=controller.cloud_snapshot();admit,reason=ready(arm,cloud)
            save(HERE/'controller_status.json',dict(stage='checking_phase2_capacity',arm=arm,reason=reason,
                active_gpus=active,updated=time.time(),pid=os.getpid()))
            if not admit:break
            load_experiment()
            identity=controller.submit(folder,folder/'cli.yaml',8,scope,FORMAL/'plan.json')
            if identity:
                save(folder/'phase2_submission.json',dict(id=identity,reason=reason,
                    experiment_sha256=sha(HERE/'experiment.json'),submitted=time.time()))
                print(arm,identity,flush=True)
            break
        if all_submitted:
            save(HERE/'controller_status.json',dict(stage='both_controls_submitted',updated=time.time(),pid=os.getpid()));return
        time.sleep(30)
    save(HERE/'controller_status.json',dict(stage='launch_window_expired_inspect_submitted_tasks',updated=time.time(),pid=os.getpid()))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(HERE/'controller_failure.json',dict(error=repr(exc),time=time.time()));raise
