"""Explicit selection of a sealed formal checkpoint after its run is terminal.

No training-completion record is fabricated. Prefix updates, the original
budget, and its actual disposition remain separate in every delivery artifact.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from common import C, bound, check_binding, read, require, save, sha
sys.path.insert(0, str(C))
from checkpoint_prefix_audit import validate as validate_prefix

TERMINAL = {'Success', 'Stopped', 'Killed'}


def observe(task_id):
    proc = subprocess.run(['/root/.volc/bin/volc', '--log-level', 'error', 'ml_task', 'get',
                           '--id', task_id, '--output', 'json', '--format', 'Id,Name,Status'],
                          capture_output=True, text=True, timeout=45)
    parsed = []
    for line in proc.stdout.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, list) and len(value) == 1 and value[0].get('Id') == task_id:
            parsed.append(value[0])
    require(proc.returncode == 0 and len(parsed) == 1, 'Cloud state unknown; do not select or restart')
    return dict(task=parsed[0], observed=time.time())


def prepare(plan_path, prefix_path, disposition, rationale, cloud):
    plan_path, prefix_path = Path(plan_path).resolve(), Path(prefix_path).resolve()
    plan, prefix = read(plan_path), validate_prefix(prefix_path)
    require(prefix['evidence']['plan'] == bound(plan_path), 'Prefix belongs to another plan')
    run = Path(plan['training_run']).resolve()
    submission = read(plan_path.parent/'submission.json')
    require(submission['plan_sha256']==sha(plan_path),'Cloud submission uses another plan')
    require(cloud['task']['Id'] == submission['response']['Id'], 'Wrong cloud training task')
    require(cloud['task']['Status'] in TERMINAL, 'Selection requires a terminal cloud run')
    require(0 <= time.time()-cloud['observed'] <= 120, 'Stale cloud observation')
    require(bool(rationale.strip()), 'Checkpoint selection needs an explicit rationale')
    require(disposition in {'planned_run_completed_select_recorded_checkpoint',
                            'planned_run_stopped_select_committed_prefix'}, 'Unknown budget disposition')
    evidence = dict(plan=bound(plan_path), prefix=bound(prefix_path), submission=bound(plan_path.parent/'submission.json'))
    completed = cloud['task']['Status'] == 'Success'
    if completed:
        complete = read(run/'complete.json')
        require(disposition == 'planned_run_completed_select_recorded_checkpoint', 'Success needs completed-run disposition')
        require(complete['stage'] == 'mopd_training_complete' and not complete['smoke']
                and complete['steps'] == plan['steps'], 'Success without actual planned training completion')
        evidence['original_run_completion'] = bound(run/'complete.json')
    else:
        require(disposition == 'planned_run_stopped_select_committed_prefix', 'Stopped run must disclose incomplete budget')
        require(not (run/'complete.json').exists(), 'Completed run must be reconciled explicitly')
    return dict(schema='anvil-explicit-formal-checkpoint-selection-v2',
        state='selected_for_full_evaluation_not_quality_accepted', training_run=str(run),
        checkpoint=prefix['checkpoint'], adapter_sha256=prefix['adapter_sha256'],
        selected_step=prefix['selected_step'], planned_steps=plan['steps'],
        original_planned_run_completed=completed, selected_checkpoint_is_planned_tip=prefix['selected_step']==plan['steps'],
        budget_disposition=disposition, rationale=rationale, cloud=cloud,
        final_quality_accepted=False, weights_uploaded=False, evidence=evidence, created=time.time())


def validate(path):
    selection=read(path)
    require(selection['schema']=='anvil-explicit-formal-checkpoint-selection-v2'
            and selection['state']=='selected_for_full_evaluation_not_quality_accepted'
            and selection['final_quality_accepted'] is False and selection['weights_uploaded'] is False,
            'Selection is provenance, not final quality acceptance')
    for item in selection['evidence'].values():check_binding(item)
    plan=read(selection['evidence']['plan']['path'])
    prefix=validate_prefix(selection['evidence']['prefix']['path'])
    require(prefix['evidence']['plan']==selection['evidence']['plan'], 'Different prefix plan')
    require(selection['training_run']==str(Path(plan['training_run']).resolve()), 'Wrong training run')
    require(selection['checkpoint']==prefix['checkpoint'] and selection['adapter_sha256']==prefix['adapter_sha256'], 'Wrong selected adapter')
    require(selection['selected_step']==prefix['selected_step'] and selection['planned_steps']==prefix['planned_steps']==plan['steps'], 'Wrong update budget')
    require(selection['selected_checkpoint_is_planned_tip']==(selection['selected_step']==plan['steps']), 'False planned-tip claim')
    task=selection['cloud']['task'];submission=read(selection['evidence']['submission']['path'])
    require(submission['plan_sha256']==selection['evidence']['plan']['sha256'],'Submission plan mismatch')
    require(task['Id']==submission['response']['Id'] and task['Status'] in TERMINAL, 'Nonterminal/wrong cloud run')
    completed=task['Status']=='Success'
    require(selection['original_planned_run_completed']==completed, 'False original completion claim')
    if completed:
        require(selection['budget_disposition']=='planned_run_completed_select_recorded_checkpoint', 'Wrong completed budget disposition')
        complete=read(selection['evidence']['original_run_completion']['path'])
        require(complete['steps']==plan['steps'] and complete['stage']=='mopd_training_complete'
                and complete['smoke'] is False, 'Invalid original run completion')
    else:
        require(selection['budget_disposition']=='planned_run_stopped_select_committed_prefix'
                and 'original_run_completion' not in selection['evidence'], 'Stopped run cannot claim completion')
    require(selection['rationale'].strip(), 'Missing selection rationale')
    return selection,prefix


def main():
    ap=argparse.ArgumentParser()
    for name in ['plan','prefix-audit','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--disposition',required=True);ap.add_argument('--rationale',required=True)
    args=ap.parse_args();require(not args.output.exists(),'Use a new immutable selection')
    task_id=read(args.plan.parent/'submission.json')['response']['Id']
    receipt=prepare(args.plan,args.prefix_audit,args.disposition,args.rationale,observe(task_id))
    save(args.output,receipt);validate(args.output);print(args.output)


if __name__=='__main__':main()
