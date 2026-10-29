"""Submit two bounded diagnostics only in the idle gap before formal step512."""
import fcntl
from pathlib import Path
import sys
import time
from objective_control_common import HERE, C, load_experiment, read, require, save, sha
sys.path[:0] = [str(C), str(C / 'formal_mopd')]
import controller
from final_delivery_handoff_v1 import observe

FORMAL = C / 'formal_mopd/runs/candidate1_20261029'
TRAIN_ID = 't-20261029095820-px5rk'


def register(receipt, gpus, scope, config):
    require(gpus == 8 and scope.startswith('objective_control_'), 'Wrong control registration')
    path = C / 'jobs_v2.json'; jobs = read(path)
    identity = receipt['response']['Id']
    if identity not in {j['id'] for j in jobs}:
        jobs.append(dict(id=identity, gpus=gpus, scope=scope, config=str(config)))
        save(path, jobs)


def main():
    experiment = load_experiment()
    lock = (HERE / 'submit.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller.check = observe; controller.register = register
    for arm in experiment['arms']:
        folder = HERE / 'submissions' / arm
        if (folder / 'submission.json').exists():
            register(read(folder / 'submission.json'), 8, 'objective_control_' + arm, folder / 'cli.yaml')
            continue
        require(not (folder / 'submission_intent.json').exists(), 'Ambiguous previous submission; do not retry')
        cloud, active = controller.cloud_snapshot(); by_id = {j['id']:j for j in cloud}
        require(by_id[TRAIN_ID]['status'] == 'Running', 'Formal state uncertain or ended')
        require(all((FORMAL / 'evaluation/128/report.json').exists() for _ in [0]), 'First evaluation incomplete')
        # The formal controller holds its own lock continuously. It may submit
        # only at a sealed scheduled checkpoint. This fresh guard excludes that
        # window; final handoff is impossible while this early run is Running.
        status = read(FORMAL / 'training/status.json')
        require(status['step'] < 256 and not (FORMAL / 'training/checkpoint-512').exists(),
                'Early launch window closed; reserve public-evaluation capacity instead')
        require(all(j['status'] in controller.TERMINAL for j in cloud
                    if j.get('scope', '').startswith('formal_MOPD_public_step')), 'Public evaluation still active')
        require(active + 8 <= 32, 'No capacity for control')
        load_experiment()
        identity = controller.submit(folder, folder / 'cli.yaml', 8, 'objective_control_' + arm, FORMAL / 'plan.json')
        require(identity is not None, 'Capacity changed; inspect before continuing')
        save(folder / 'control_submission.json', dict(id=identity, experiment_sha256=sha(HERE / 'experiment.json'),
            formal_step_observed=status['step'], active_before=active, cap=32, time=time.time()))
        print(arm, identity, flush=True)


if __name__ == '__main__':
    main()
