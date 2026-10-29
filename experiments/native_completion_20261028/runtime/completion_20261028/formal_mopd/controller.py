"""Capacity-bounded formal training and predeclared checkpoint evaluations.

One local controller owns this run's cloud submissions. Cloud state is queried
before every submission; ambiguous submissions are never automatically retried.
This controller cannot accept final quality or upload weights.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from formal_common import C, D, load_plan, read, require, save, sha
sys.path.insert(0, str(D))
from submit_student_when_capacity import check, TERMINAL


def registry():
    jobs = read(D / 'jobs.json') + read(C / 'jobs_v2.json')
    require(len({j['id'] for j in jobs}) == len(jobs), 'Duplicate registry IDs')
    return jobs


def cloud_snapshot():
    with ThreadPoolExecutor(max_workers=8) as pool:
        cloud = list(pool.map(check, registry()))
    active = sum(j['gpus'] for j in cloud if j['status'] not in TERMINAL)
    require(active <= 32, 'Campaign GPU cap already exceeded')
    return cloud, active


def register(receipt, gpus, scope, config):
    jobs = read(D / 'jobs.json'); identity = receipt['response']['Id']
    if identity not in {j['id'] for j in jobs}:
        jobs.append(dict(id=identity, gpus=gpus, scope=scope, config=str(config)))
        save(D / 'jobs.json', jobs)


def submit(folder, config, gpus, scope, plan_path):
    receipt_path = folder / 'submission.json'
    if receipt_path.exists():
        receipt = read(receipt_path); register(receipt, gpus, scope, config)
        return receipt['response']['Id']
    require(not (folder / 'submission_intent.json').exists(), 'Ambiguous earlier attempt; inspect cloud before retry')
    load_plan(plan_path)
    cloud, active = cloud_snapshot()
    save(folder / 'capacity.json', dict(cloud=cloud, active=active, requested=gpus, cap=32, time=time.time()))
    if active + gpus > 32:
        return None
    save(folder / 'submission_intent.json', dict(config=str(config), config_sha256=sha(config),
                                               plan_sha256=sha(plan_path), time=time.time()))
    proc = subprocess.run(['/root/.volc/bin/volc', '--log-level', 'error', 'ml_task', 'submit',
                           '--conf', str(config), '--output', 'json'], capture_output=True, text=True, timeout=60)
    (folder / 'submission.log').write_text(proc.stdout + proc.stderr)
    values = []
    for line in proc.stdout.splitlines():
        try:
            item = __import__('json').loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and isinstance(item.get('Id'), str):
            values.append(item)
    require(proc.returncode == 0 and len(values) == 1, 'Ambiguous submission; do not retry automatically')
    receipt = dict(response=values[0], plan_sha256=sha(plan_path), config_sha256=sha(config),
                   gpus=gpus, submitted=time.time())
    save(receipt_path, receipt); register(receipt, gpus, scope, config)
    return values[0]['Id']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', type=Path, required=True)
    args = ap.parse_args(); plan = load_plan(args.plan); root = args.plan.resolve().parent
    deadline = datetime.fromisoformat(plan['budget']['deadline_utc']).timestamp()
    with open('/tmp/anvil-formal-mopd-submit.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train_id = None
        while time.time() < deadline:
            train_id = submit(root, root / 'training_cli.yaml', plan['world'],
                              'formal_MOPDv6_' + root.name, args.plan)
            if train_id:
                break
            save(root / 'controller_status.json', dict(stage='waiting_training_capacity', updated=time.time(), pid=os.getpid()))
            time.sleep(60)
        require(train_id, 'No formal submission before deadline')
        while time.time() < deadline:
            cloud, active = cloud_snapshot(); by_id = {j['id']:j for j in cloud}
            training_status = by_id[train_id]['status']
            require(training_status not in TERMINAL or training_status == 'Success',
                    'Formal job terminated abnormally: ' + training_status)
            all_done = True; next_step = None; evaluation_active = False
            for step in plan['evaluation_steps']:
                folder = root / 'evaluation' / str(step)
                require(not (folder / 'failure.json').exists(), 'Checkpoint evaluation failed at step%d' % step)
                if (folder / 'submission.json').exists():
                    receipt = read(folder / 'submission.json')
                    register(receipt, 8, 'formal_MOPD_public_step%d_' % step + root.name, folder / 'cli.yaml')
                    task = by_id.get(receipt['response']['Id'])
                    if task is None or task['status'] not in TERMINAL:
                        evaluation_active = True; all_done = False
                    else:
                        require(task['status'] == 'Success' and (folder / 'report.json').exists(),
                                'Evaluation terminated without report at step%d' % step)
                    continue
                all_done = False
                if next_step is None and (Path(plan['training_run']) / ('checkpoint-%d' % step) / 'checkpoint_complete.json').exists():
                    next_step = step
            if all_done and training_status == 'Success':
                save(root / 'controller_status.json', dict(stage='formal_and_public_diagnostics_complete_requires_full_final_review',
                    updated=time.time(), pid=os.getpid(), weights_uploaded=False))
                return
            if next_step is not None and not evaluation_active:
                folder = root / 'evaluation' / str(next_step)
                submit(folder, folder / 'cli.yaml', 8, 'formal_MOPD_public_step%d_' % next_step + root.name, args.plan)
            save(root / 'controller_status.json', dict(stage='monitoring_formal_and_checkpoints', train_id=train_id,
                training_status=training_status, active_campaign_gpus=active, next_ready_step=next_step,
                evaluation_in_progress=evaluation_active, updated=time.time(), pid=os.getpid()))
            time.sleep(60)
        raise TimeoutError('Campaign deadline reached; inspect running jobs and preserve outputs')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if '--plan' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--plan') + 1]).resolve().parent
            save(root / 'controller_failure.json', dict(error=repr(exc), time=time.time()))
        raise
