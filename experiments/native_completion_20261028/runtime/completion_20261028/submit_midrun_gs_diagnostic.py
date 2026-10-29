"""Submit one fixed native diagnostic after A controls release their eight GPUs.

The formal controller can own at most one eight-GPU public evaluation. Reserve
that capacity even before its task appears, so this independent diagnostic does
not race the formal controller for the last eight cards. No training is changed.
"""
import fcntl
import os
from pathlib import Path
import sys
import time

C = Path(__file__).resolve().parent
sys.path.insert(0, str(C / 'formal_mopd'))
import controller
from formal_common import read, require, save, sha
from final_delivery_handoff_v1 import observe
import midrun_gs_diagnostic as diagnostic

ROOT = C / 'native_diagnostics/candidate1_step128'
FORMAL = C / 'formal_mopd/runs/candidate1_20261029'
TRAIN_ID = 't-20261029095820-px5rk'
A_ID = 't-20261029081704-zgf22'
SCOPE = 'formal_step128_fixed_GS35989_diagnostic'


def register_diagnostic(receipt, gpus, scope, config):
    # The running formal controller owns diagnosis_32gpu/jobs.json. This single
    # supplementary controller uses the other existing registry, avoiding a
    # read/modify/write collision with checkpoint-public submissions.
    require(scope == SCOPE and gpus == 8, 'Unexpected supplementary task')
    identity = receipt['response']['Id']
    path = C / 'jobs_v2.json'
    jobs = read(path)
    if identity not in {job['id'] for job in jobs}:
        require(identity not in {job['id'] for job in read(C / 'diagnosis_32gpu/jobs.json')},
                'Diagnostic is already registered in the primary registry')
        jobs.append(dict(id=identity, gpus=8, scope=scope, config=str(config)))
        save(path, jobs)


def available_with_public_reservation(cloud):
    active = sum(j['gpus'] for j in cloud if j['status'] not in controller.TERMINAL)
    public = sum(j['gpus'] for j in cloud if j['status'] not in controller.TERMINAL
                 and j.get('scope', '').startswith('formal_MOPD_public_step'))
    require(public <= 8 and active <= 32, 'Unexpected existing controller allocation')
    reserved = max(0, 8-public)
    return active + 8 + reserved <= 32, dict(active=active, requested=8,
        public_active=public, public_reserved=reserved, projected_with_reservation=active+8+reserved, cap=32)


def main():
    policy = read(ROOT / 'policy.json')
    for path, expected in policy['frozen_files'].items():
        require(sha(path) == expected, 'Native diagnostic policy dependency changed')
    checked = read(policy['cpu_validation']['path'])
    require(sha(policy['cpu_validation']['path']) == policy['cpu_validation']['sha256']
            and checked['state'] == 'passed', 'Native diagnostic validation missing')
    for path, expected in checked['code_sha256'].items():
        require(sha(path) == expected, 'CPU validation belongs to another diagnostic version')
    controller.check = observe
    controller.register = register_diagnostic
    lease = (ROOT / 'controller.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    public_spec = FORMAL / 'evaluation/128/spec.json'
    while time.time() < policy['launch_deadline']:
        if (ROOT / 'submission.json').exists():
            receipt = read(ROOT / 'submission.json')
            controller.register(receipt, 8, SCOPE, ROOT / 'cli.yaml')
            save(ROOT / 'controller_status.json', dict(stage='submitted',
                id=receipt['response']['Id'], updated=time.time(), pid=os.getpid()))
            return
        if not public_spec.exists():
            save(ROOT / 'controller_status.json', dict(stage='waiting_sealed_step128_public_model',
                updated=time.time(), pid=os.getpid(), public_slots_reserved=8))
            time.sleep(30)
            continue
        cloud, active = controller.cloud_snapshot()
        by_id = {j['id']: j for j in cloud}
        capacity, allocation = available_with_public_reservation(cloud)
        require(by_id[TRAIN_ID]['status'] not in controller.TERMINAL,
                'Formal run already ended; inspect before adding a midrun diagnostic')
        a_done = by_id[A_ID]['status'] in {'Success', 'Succeeded'}
        if not a_done or not capacity:
            save(ROOT / 'controller_status.json', dict(stage='waiting_A_completion_and_capacity',
                A_status=by_id[A_ID]['status'], allocation=allocation,
                updated=time.time(), pid=os.getpid()))
            time.sleep(30)
            continue
        state = read(FORMAL / 'training/status.json')
        require(state['step'] < 256, 'Step128 diagnostic launch window expired')
        for path, expected in policy['frozen_files'].items():
            require(sha(path) == expected, 'Native diagnostic policy dependency changed')
        spec = diagnostic.prepare(public_spec, ROOT)
        diagnostic.load_spec(spec)
        save(ROOT / 'reserved_capacity.json', dict(cloud=cloud, **allocation,
            exact_baseline_records=35989, checked=time.time()))
        identity = controller.submit(ROOT, ROOT / 'cli.yaml', 8, SCOPE, FORMAL / 'plan.json')
        if identity:
            save(ROOT / 'controller_status.json', dict(stage='submitted', id=identity,
                diagnostic_spec_sha256=sha(spec), updated=time.time(), pid=os.getpid(),
                original_formal_plan_unchanged=True, final_validation=False))
            return
        time.sleep(30)
    raise TimeoutError('Fixed step128 diagnostic launch window expired; no blind resubmission')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(ROOT / 'controller_failure.json', dict(error=repr(exc), time=time.time()))
        raise
