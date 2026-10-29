"""Apply the declared first-formal-run budget after exact8/16GPU probes finish.

Teacher usefulness was reviewed separately. This launcher does not manufacture
quality results, change active experiments, or accept/upload a final model.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
C = Path(__file__).resolve().parent
F = C / 'formal_mopd'; D = C / 'diagnosis_32gpu'; R = C.parent
sys.path.insert(0, str(F))
from formal_common import read, require, save, sha
TRAIN = str(R / '.venv-train/bin/python')


def main():
    policy_path = F / 'launch_policy.json'; policy = read(policy_path); policy_sha256 = sha(policy_path)
    require(policy['schema'] == 'anvil-first-formal-budget-policy-v1', 'Unknown budget policy')
    for path, digest in policy['frozen_files'].items():
        require(sha(path) == digest, 'Changed predeclared launcher dependency')
    root = Path(policy['run_directory'])
    with open('/tmp/anvil-first-formal-readiness.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (F / 'formal_controller_process.json').exists():
            prior = read(F / 'formal_controller_process.json')
            require((Path('/proc') / str(prior['pid'])).exists(), 'Prior controller exited; inspect before restart')
            return
        deadline = datetime.fromisoformat(policy['deadline_utc']).timestamp()
        timeout = time.time() + policy['readiness_timeout_hours'] * 3600
        while time.time() < timeout:
            require(sha(policy_path) == policy_sha256, 'Declared budget policy changed during wait')
            for path, digest in policy['frozen_files'].items():
                require(sha(path) == digest, 'Readiness dependency changed')
            failures = [D / name for name in ['mopd_v6_candidate1/resume/failure.json',
                'mopd_v6_candidate1/probe/failure.json', 'mopd_v6_candidate1_multinode/actual/node0_failure.json',
                'mopd_v6_candidate1_multinode/actual/node1_failure.json','mopd_scaling_comparison_failure.json']]
            require(not any(p.exists() for p in failures), 'Execution/scaling failure requires diagnosis')
            report_path = D / 'mopd_scaling_comparison.json'
            if not report_path.exists():
                save(F / 'readiness_status.json', dict(stage='waiting_exact_runtime_and_scaling', pid=os.getpid(), updated=time.time()))
                time.sleep(30)
                continue
            report = read(report_path)
            require(report['state'] == 'complete_equal_source_throughput_comparison_requires_budget_review', 'Incomplete scaling')
            nodes = 2 if report['wall_time_speedup_16_over8'] >= policy['minimum_16GPU_speedup'] else 1
            timing = report['world%d' % (nodes * 8)]
            measured = timing['elapsed_including_intermediate_saves_seconds'] / timing['measured_updates']
            guarded = max(measured * 1.5, timing['p90_update_seconds'])
            seconds = min(policy['maximum_guarded_training_hours'] * 3600,
                          deadline - time.time() - policy['final_reserve_hours'] * 3600) - 3600
            updates = min(policy['maximum_updates'], math.floor(seconds / guarded / 128) * 128)
            require(updates >= 128, 'No defensible first-formal budget before deadline reserve')
            decision = dict(state='selected_from_completed_equal_source_throughput', policy_sha256=sha(policy_path),
                scaling_sha256=sha(report_path), nodes=nodes, updates=updates,
                guarded_seconds_per_update=guarded, measured_seconds_per_update=measured,
                estimated_training_hours=updates * measured / 3600,
                guarded_training_hours=updates * guarded / 3600 + 1,
                final_reserve_hours=policy['final_reserve_hours'], selected=time.time(),
                rationale='Bounded first formal run, up to2048updates and48guardedhours, preserving time for quality-driven correction and final delivery.')
            save(F / 'budget_decision.json', decision)
            require(not root.exists(), 'A formal plan already exists; inspect it rather than overwrite')
            cmd = [TRAIN, str(F / 'prepare_plan.py'), '--output', str(root), '--nodes', str(nodes),
                   '--steps', str(updates), '--reserve-hours', str(policy['final_reserve_hours'])]
            with (F / 'formal_prepare.log').open('a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (root / 'controller.log').open('a') as log:
                process = subprocess.Popen([TRAIN, '-u', str(F / 'controller.py'), '--plan', str(root / 'plan.json')],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(F / 'formal_controller_process.json', dict(pid=process.pid, plan=str(root / 'plan.json'), started=time.time()))
            save(F / 'readiness_status.json', dict(stage='formal_controller_started', pid=process.pid,
                                                 plan=str(root / 'plan.json'), updated=time.time()))
            return
        raise TimeoutError('Exact runtime/scaling not complete within readiness window; no formal submission')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(F / 'readiness_failure.json', dict(error=repr(exc), time=time.time()))
        raise
