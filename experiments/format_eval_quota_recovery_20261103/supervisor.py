"""Resume full evaluation after storage-quota failure, preserving the failed run.

The trained model, evaluation spec, workers, scoring and generation budgets are
unchanged. A new output directory and node-local compiler caches isolate retry.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

Q = Path(__file__).resolve().parent
C = Q.parent
H = C / 'final_delivery_format_v1'
D = C / 'delivery_runs/format_repair_20261103'
sys.path.insert(0, str(H))
from common import TRAIN, read, require, save, sha
sys.path.insert(0, str(C))
from final_delivery_handoff_v1 import cloud_snapshot, submission_response
sys.path.insert(0, str(C / 'formal_mopd'))
from controller import register, TERMINAL


def validate():
    policy = read(Q / 'policy.json')
    require(policy['schema'] == 'anvil-evaluation-quota-recovery-v1', 'Wrong recovery policy')
    for name, digest in policy['files'].items():
        require(sha(name) == digest, 'Recovery dependency changed: ' + name)
    require(sha(D / 'evaluation_spec.json') == policy['evaluation_spec_sha256'], 'Different evaluation spec')
    require(read(C / 'format_repair_20261103/runs/mopd/complete.json')['steps'] == 256, 'Training incomplete')
    cleanup = read(policy['cleanup_receipt'])
    require(cleanup['state'] == 'completed' and cleanup['all27_keys_retained'], 'Storage cleanup incomplete')
    require(read(Q / 'storage_write_probe.json')['state'] == 'passed', 'Shared-storage write probe missing')
    return policy


def status(stage, **values):
    save(Q / 'status.json', dict(stage=stage, pid=os.getpid(), updated=time.time(),
                               final_quality_accepted=False, weights_uploaded=False, **values))


def main():
    policy = validate()
    own = (Q / 'supervisor.lock').open('a')
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = policy['deadline']
    identities = {}
    with open('/tmp/anvil-formal-mopd-submit.lock', 'a') as shared:
        fcntl.flock(shared, fcntl.LOCK_EX)
        for node in range(4):
            folder = Q / 'submissions' / str(node)
            folder.mkdir(parents=True, exist_ok=True)
            config = Q / 'configs' / f'node_{node}.yaml'
            if (folder / 'submission.json').exists():
                submitted = read(folder / 'submission.json')
                require(submitted['final_spec_sha256'] == policy['evaluation_spec_sha256'], 'Different submission spec')
                require(submitted['config_sha256'] == sha(config), 'Different submitted config')
            else:
                require(not (folder / 'submission_intent.json').exists(), 'Ambiguous prior submission; reconcile cloud')
                while time.time() < deadline:
                    cloud, active = cloud_snapshot()
                    by_id = {r['id']: r for r in cloud}
                    require(all(by_id[j]['status'] in TERMINAL for j in policy['failed_evaluation_jobs']),
                            'Original evaluation still active or UNKNOWN')
                    if active + 8 <= 32:
                        break
                    status('waiting_final_capacity', active_campaign_gpus=active)
                    time.sleep(45)
                else:
                    raise TimeoutError('No evaluation capacity before expiry')
                validate()
                save(folder / 'capacity.json', dict(active=active, requested=8, cloud=cloud, checked=time.time()))
                binding = dict(final_spec_sha256=policy['evaluation_spec_sha256'], config_sha256=sha(config),
                               gpus=8, time=time.time(), recovery_policy_sha256=sha(Q / 'policy.json'))
                save(folder / 'submission_intent.json', binding)
                proc = subprocess.run(['/root/.volc/bin/volc', '--log-level', 'error', 'ml_task', 'submit',
                                       '--conf', str(config), '--output', 'json'], capture_output=True, text=True, timeout=90)
                (folder / 'submission.log').write_text(proc.stdout + proc.stderr)
                submitted = dict(binding, response=submission_response(proc.stdout, proc.returncode))
                save(folder / 'submission.json', submitted)
            register(submitted, 8, 'format_quota_recovery_full_evaluation_node' + str(node), config)
            identities[node] = submitted['response']['Id']
    while time.time() < deadline:
        cloud, active = cloud_snapshot()
        by_id = {r['id']: r for r in cloud}
        finished = []
        for node, identity in identities.items():
            observed = by_id[identity]['status']
            require(observed not in TERMINAL or observed == 'Success', 'Recovery evaluation failed: ' + identity)
            if observed == 'Success':
                state = read(Q / 'evaluation' / f'node_{node}.json')
                require(state['stage'] == 'node_complete' and not state['failures'], 'Incomplete evaluation node')
                finished.append(node)
        status('full_final_evaluation_running', jobs=identities, finished_nodes=finished, active_campaign_gpus=active)
        if len(finished) == 4:
            break
        time.sleep(60)
    else:
        raise TimeoutError('Recovery evaluation expiry')
    validate()
    report = Q / 'evaluation/final_report.json'
    if not report.exists():
        with (Q / 'score.log').open('a') as stream:
            subprocess.run([TRAIN, '-u', str(H / 'score.py'), '--spec', str(D / 'evaluation_spec.json'),
                            '--output', str(Q / 'evaluation')], stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = read(report)
    require(result['state'] == 'complete_requires_quality_review' and
            result['final_spec_sha256'] == policy['evaluation_spec_sha256'], 'Wrong or incomplete final report')
    status('full_final_report_ready_requires_quality_review', report=str(report), report_sha256=sha(report),
           lineage=str(D / 'training_lineage.json'))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(Q / 'failure.json', dict(error=repr(exc), time=time.time(), pid=os.getpid()))
        raise
