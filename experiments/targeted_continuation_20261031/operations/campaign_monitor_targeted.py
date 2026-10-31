"""Read-only observer for the completed parent and targeted continuation.

Writes a separate status file; never submits jobs, edits frozen sources,
accepts quality, uploads weights, or deletes artifacts.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import time

C = Path(__file__).resolve().parent
NEW = C / 'targeted_repair_20261031'
PARENT = C / 'direct_recovery_20261030'
PFD = C / 'delivery_runs/direct_recovery_20261030'
NFD = C / 'delivery_runs/targeted_repair_20261031'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, value):
    temp = path.with_name(path.name + '.monitor-pending')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def stage(folder, planned):
    result = {'planned_updates': planned}
    for name in ['status', 'latest', 'complete']:
        result[name] = read(folder / (name + '.json'))
    log = folder / 'logging.jsonl'
    if log.exists():
        with log.open() as stream:
            tail = list(deque(stream, maxlen=65))
        rows = []
        for i, line in enumerate(tail):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(tail) - 1:
                    raise
        if rows:
            recent = rows[-64:]
            result['last_update'] = rows[-1]
            result['recent_numeric_values_finite'] = all(
                math.isfinite(value) for row in recent for value in row.values()
                if isinstance(value, (int, float)))
            seconds = sum(row['seconds'] for row in recent) / len(recent)
            result['seconds_per_update_recent'] = seconds
            result['remaining_stage_hours_recent_throughput'] = max(0, planned - rows[-1]['step']) * seconds / 3600
            result['eta_excludes_future_stages_queue_evaluation_upload'] = True
            result['logging_age_seconds'] = time.time() - log.stat().st_mtime
    return result


def evaluation(folder):
    shards = []
    for path in (folder / 'evaluation/native').glob('*/status.json'):
        value = read(path)
        shards.append({'shard': path.parent.name, **value})
    return dict(handoff=read(folder / 'status.json'), failure=read(folder / 'failure.json'),
                native_completed=sum(x.get('completed', 0) for x in shards),
                native_expected=580000,
                native_complete_shards=sum(x.get('stage') == 'complete_scope_shard' for x in shards),
                native_observed_shards=len(shards),
                native_engine_heartbeat_max_age_seconds=max(
                    [time.time()-x['heartbeat_at'] for x in shards
                     if 'heartbeat_at' in x and x.get('stage') != 'complete_scope_shard'], default=0),
                public_completed_shards=len(list((folder/'evaluation/public').glob('*/complete.json'))),
                final_report_present=(folder/'evaluation/final_report.json').exists(),
                worker_failures=[str(p) for p in (folder/'evaluation').glob('*/*/node_failure.json')])


def snapshot():
    spec = read(NEW / 'spec.json')
    result = dict(schema='anvil-targeted-campaign-status-v1', updated_utc=datetime.now(timezone.utc).isoformat(),
                  pid=os.getpid(), gpu_ceiling=32,
                  deadline_utc=spec['deadline_utc'], final_quality_accepted=False, weights_uploaded=False,
                  original_inventory=dict(records=5800000, images=160000, sft_records=5220000, spatial_pool=4756000),
                  parent_training_complete=read(PARENT/'runs/mopd/complete.json'),
                  parent_evaluation=evaluation(PFD),
                  continuation_controller=read(NEW/'controller_status.json'),
                  continuation_failure=read(NEW/'controller_failure.json'),
                  submission=read(NEW/'submission.json'),
                  continuation_stages={name:stage(NEW/'runs'/name, cfg['steps']) for name,cfg in spec['stages'].items()},
                  continuation_evaluation=evaluation(NFD),
                  node_failures=[dict(path=str(p), detail=read(p)) for p in NEW.glob('node*_failure.json')],
                  hf_last_observation=read(C/'verification_20261031/hf_inventory_latest.json'),
                  hf_observation_is_not_a_release_verification=True)
    result['currently_observed_stage'] = (result['continuation_evaluation']['handoff'] or
                                         result['continuation_controller'] or {}).get('stage', 'unknown')
    for name in ['repair','mopd']:
        item = result['continuation_stages'][name]
        if item.get('last_update') and not item.get('complete'):
            result['currently_observed_stage'] = name + '_training'
    # The controller/handoff owns fresh cloud-capacity checks. Preserve its age
    # instead of converting an old count into an assertion of current capacity.
    status = (result['continuation_evaluation']['handoff'] or result['parent_evaluation']['handoff'] or {})
    result['last_handoff_capacity_observation'] = dict(
        active_campaign_gpus=status.get('active_campaign_gpus'), updated=status.get('updated'),
        age_seconds=time.time()-status['updated'] if status.get('updated') else None)
    for name,path in [('system','/'),('shared','/Knowin/sim')]:
        stat=os.statvfs(path)
        result[name+'_available_bytes']=stat.f_bavail*stat.f_frsize
    save(C/'campaign_targeted_status.json', result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    with (C/'monitor_targeted.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                snapshot()
                save(C/'monitor_targeted_health.json',dict(state='healthy',pid=os.getpid(),updated=time.time()))
            except Exception as exc:
                save(C/'monitor_targeted_health.json',dict(state='error_retrying',error=repr(exc),pid=os.getpid(),updated=time.time()))
                if args.once:
                    raise
            if args.once:
                return
            time.sleep(45)


if __name__ == '__main__':
    main()
