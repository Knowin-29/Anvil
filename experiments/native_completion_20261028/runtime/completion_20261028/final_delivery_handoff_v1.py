"""Continue a completed formal run into full evaluation, stopping for quality review.

This supervisor never trains a replacement model, changes the selected checkpoint,
accepts model quality, packages a release or uploads credentials/weights. It shares
the formal controller's submission lock and charges unknown cloud states to the
32-GPU limit. A submission with an uncertain outcome requires explicit recovery.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

C = Path(__file__).resolve().parent
sys.path.insert(0, str(C / 'final_delivery'))
from common import TRAIN, formal_training, load_spec, read, require, save, sha
sys.path.insert(0, str(C / 'formal_mopd'))
import controller as formal_controller
from formal_common import load_plan

READY = 'formal_and_public_diagnostics_complete_requires_full_final_review'
SUCCESS = {'Success', 'Succeeded'}


def ready_for_final(plan, controller_status, complete):
    """Both the trained budget and the completed diagnostic handoff are required."""
    return (controller_status.get('stage') == READY
            and complete.get('stage') == 'mopd_training_complete'
            and complete.get('smoke') is False
            and complete.get('steps') == plan['steps'])


def charged_gpus(cloud):
    require(len({item['id'] for item in cloud}) == len(cloud), 'Duplicate cloud job IDs')
    return sum(item['gpus'] for item in cloud
               if item['status'] not in formal_controller.TERMINAL)


def observe(job, check=formal_controller.check, sleep=time.sleep):
    errors = []
    for attempt in range(3):
        try:
            return check(job)
        except Exception as exc:
            errors.append(repr(exc))
            if attempt < 2:
                sleep(2 ** (attempt + 1))
    return dict(job, status='UNKNOWN', observation_errors=errors)


def cloud_snapshot():
    with ThreadPoolExecutor(max_workers=8) as pool:
        cloud = list(pool.map(observe, formal_controller.registry()))
    active = charged_gpus(cloud)
    require(active <= 32, 'Campaign GPU cap exceeded')
    return cloud, active


def submission_response(stdout, returncode):
    values = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and isinstance(item.get('Id'), str):
            values.append(item)
    require(returncode == 0 and len(values) == 1,
            'Ambiguous submission: inspect cloud before any retry')
    return values[0]


def assert_policy(policy):
    require(policy['schema'] == 'anvil-final-handoff-v1', 'Unknown handoff policy')
    require(sha(policy['formal_plan']) == policy['formal_plan_sha256'], 'Formal plan changed')
    require(sha(__file__) == policy['supervisor_sha256'], 'Handoff supervisor changed')
    require(policy['nodes'] == 4 and policy['gpu_cap'] == 32, 'Unexpected final topology')
    validation = policy['cpu_validation']
    require(sha(validation['path']) == validation['sha256'], 'Handoff validation changed')
    checked = read(validation['path'])
    require(checked['state'] == 'passed' and checked['supervisor_sha256'] == sha(__file__),
            'Handoff fault-injection validation does not match this supervisor')
    for filename, digest in policy['delivery_sources_sha256'].items():
        require(sha(filename) == digest, 'Final delivery source changed: ' + filename)
    return load_plan(policy['formal_plan'])


def run_tool(root, script, *args):
    log = root / 'logs' / (Path(script).stem + '.log')
    log.parent.mkdir(exist_ok=True)
    argv = [TRAIN, '-u', str(C / 'final_delivery' / script), *map(str, args)]
    with log.open('a') as stream:
        proc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
    require(proc.returncode == 0, 'Final preparation failed; inspect ' + str(log))


def submit_node(root, policy, spec_path, node):
    folder = root / 'submissions' / str(node)
    folder.mkdir(parents=True, exist_ok=True)
    config = root / 'evaluation' / ('node_%d_cli.yaml' % node)
    receipt_path = folder / 'submission.json'
    scope = 'full_final_evaluation_' + root.name + '_node%d' % node
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['final_spec_sha256'] == sha(spec_path), 'Submitted spec changed')
        require(receipt['config_sha256'] == sha(config), 'Submitted config changed')
        formal_controller.register(receipt, 8, scope, config)
        return receipt['response']['Id']
    require(not (folder / 'submission_intent.json').exists(),
            'Ambiguous earlier final submission; recover explicitly')
    assert_policy(policy)
    load_spec(spec_path, verify_model=False)
    cloud, active = cloud_snapshot()
    save(folder / 'capacity.json', dict(cloud=cloud, active=active, requested=8,
                                       cap=32, checked=time.time()))
    if active + 8 > 32:
        return None
    binding = dict(final_spec_sha256=sha(spec_path), config_sha256=sha(config),
                   config=str(config), gpus=8, time=time.time())
    save(folder / 'submission_intent.json', binding)
    proc = subprocess.run(['/root/.volc/bin/volc', '--log-level', 'error', 'ml_task',
                           'submit', '--conf', str(config), '--output', 'json'],
                          capture_output=True, text=True, timeout=90)
    (folder / 'submission.log').write_text(proc.stdout + proc.stderr)
    receipt = dict(binding, response=submission_response(proc.stdout, proc.returncode))
    save(receipt_path, receipt)
    formal_controller.register(receipt, 8, scope, config)
    return receipt['response']['Id']


def status(root, stage, **fields):
    save(root / 'status.json', dict(stage=stage, updated=time.time(), pid=os.getpid(),
                                   final_quality_accepted=False, weights_uploaded=False,
                                   **fields))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path, required=True)
    args = ap.parse_args()
    policy = read(args.policy)
    root = args.policy.resolve().parent
    plan = assert_policy(policy)
    formal_root = Path(policy['formal_plan']).parent
    deadline = datetime.fromisoformat(plan['budget']['deadline_utc']).timestamp()
    own_lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared_lock = open('/tmp/anvil-formal-mopd-submit.lock', 'a')
    while time.time() < deadline:
        state_path = formal_root / 'controller_status.json'
        complete_path = Path(plan['training_run']) / 'complete.json'
        state = read(state_path) if state_path.exists() else {}
        complete = read(complete_path) if complete_path.exists() else {}
        ready = ready_for_final(plan, state, complete)
        if ready:
            try:
                fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
        status(root, 'waiting_formal_and_public_diagnostics',
               formal_controller_stage=state.get('stage'), formal_ready=ready)
        time.sleep(30)
    else:
        raise TimeoutError('Deadline before formal final-validation handoff')

    # No original training process is restarted and no intermediate checkpoint
    # is substituted. Existing formal integrity checks bind the completed tip.
    assert_policy(policy)
    require(not (formal_root / 'controller_failure.json').exists(),
            'Unresolved formal controller failure')
    complete, protocol, checkpoint, receipt = formal_training(plan['training_run'])
    require(complete['steps'] == plan['steps'], 'Final step differs from frozen plan')
    require(shutil.disk_usage(root).free >= 200 * 1024 ** 3,
            'Less than 200 GiB shared storage free for final evaluation/package')
    merged = root / 'merged_model'
    spec = root / 'evaluation_spec.json'
    evaluation = root / 'evaluation'
    status(root, 'preparing_completed_formal_checkpoint', checkpoint=str(checkpoint))
    if not merged.exists():
        run_tool(root, 'merge.py', '--adapter', checkpoint, '--output', merged)
    else:
        merged_receipt = read(merged / 'merge_receipt.json')
        require(merged_receipt.get('state') == 'verified_derived_inference_copy',
                'Partial merge preserved; explicit recovery required')
        require(Path(merged_receipt['original_adapter']).resolve() == checkpoint,
                'Prior merge belongs to another checkpoint')
    if not spec.exists():
        run_tool(root, 'prepare.py', '--training-run', plan['training_run'],
                 '--model', merged, '--output', spec, '--native-shards', 32, '--public-shards', 8)
    load_spec(spec)
    run_tool(root, 'configs.py', '--spec', spec, '--output', evaluation, '--nodes', 4)
    lineage = root / 'training_lineage.json'
    if not lineage.exists():
        run_tool(root, 'lineage.py', '--spec', spec, '--eligibility', formal_root / 'eligibility.json',
                 '--output', lineage)
    require(read(lineage)['final_spec_sha256'] == sha(spec), 'Different final lineage')

    identities = {}
    while time.time() < deadline:
        for node in range(4):
            if node not in identities:
                identity = submit_node(root, policy, spec, node)
                if identity is not None:
                    identities[node] = identity
        cloud, active = cloud_snapshot()
        by_id = {item['id']: item for item in cloud}
        finished = []
        for node, identity in identities.items():
            observed = by_id[identity]['status']
            require(observed not in formal_controller.TERMINAL or observed in SUCCESS,
                    'Final evaluation terminated abnormally: ' + identity + ' ' + observed)
            if observed in SUCCESS:
                state = read(evaluation / ('node_%d.json' % node))
                require(state['stage'] == 'node_complete' and not state['failures'],
                        'Cloud success without complete final node evidence')
                finished.append(node)
        status(root, 'full_final_evaluation_running', jobs=identities,
               finished_nodes=finished, active_campaign_gpus=active,
               final_spec_sha256=sha(spec))
        if len(finished) == 4:
            break
        time.sleep(60)
    else:
        raise TimeoutError('Deadline before complete final evaluation')
    report = evaluation / 'final_report.json'
    if not report.exists():
        status(root, 'scoring_complete_final_outputs', final_spec_sha256=sha(spec))
        run_tool(root, 'score.py', '--spec', spec, '--output', evaluation)
    result = read(report)
    require(result['state'] == 'complete_requires_quality_review'
            and result['final_spec_sha256'] == sha(spec), 'Final report identity mismatch')
    status(root, 'full_final_report_ready_requires_quality_review', report=str(report),
           report_sha256=sha(report), final_spec_sha256=sha(spec), lineage=str(lineage))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if '--policy' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--policy') + 1]).resolve().parent
            save(root / 'failure.json', dict(error=repr(exc), time=time.time(),
                                           weights_uploaded=False, final_quality_accepted=False))
        raise
