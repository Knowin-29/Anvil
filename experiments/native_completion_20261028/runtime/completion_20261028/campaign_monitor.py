"""Persist real campaign progress and aggregate complete evidence without GPU use.

This monitor never manufactures teacher approval or submits training after a
quality failure. A completed DLC runtime is distinct from a qualified model.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
TASKS=json.loads((ROOT/'evaluation_tasks.json').read_text())
RUN=ROOT/'qualification_v2'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,value):
    tmp=path.with_suffix('.pending')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def snapshot():
    now=time.time();shards=[];failures=[]
    for task in TASKS:
        out=RUN/task['id'];state={}
        if (out/'evaluation/status.json').exists():
            try:state=json.loads((out/'evaluation/status.json').read_text())
            except json.JSONDecodeError:pass
        if (out/'worker_failure.json').exists():failures.append(task['id'])
        done=(out/'complete.json').exists()
        completed=task['records'] if done else state.get('completed',0)
        speed=state.get('records_per_second',0)
        heartbeat=state.get('heartbeat_at',now)
        shards.append(dict(id=task['id'],model=task['model'],expected=task['records'],
            completed=completed,complete=done,stage='complete' if done else state.get('stage','queued'),
            last_heartbeat_age_seconds=max(0,now-heartbeat),records_per_second=speed,
            current_shard_remaining_hours=(task['records']-completed)/speed/3600 if speed>0 and not done else None))
    complete=all(x['complete'] for x in shards)
    result=dict(updated_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),
        campaign_gpu_ceiling=json.loads((ROOT/'resource_authorization.json').read_text())['gpu_ceiling'],submitted_jobs=json.loads((ROOT/'jobs_v2.json').read_text()) + (json.loads((ROOT/'diagnosis_32gpu/jobs.json').read_text()) if (ROOT/'diagnosis_32gpu/jobs.json').exists() else []),
        evaluation=dict(expected=1160000,completed=sum(x['completed'] for x in shards),
            complete_shards=sum(x['complete'] for x in shards),total_shards=29,all_complete=complete,
            failures=failures,shards=shards),
        final_training=dict(state='blocked_on_teacher_quality',updates=None,
            sft_retention_is_admission_blocker=False,
            current_runtime='native_mopd_v6.py',runtime_preparation_state='CPU_verified_exact_candidate_GPU_pending',
            runtime_validation_state='passed_execution_only_candidate1' if (ROOT/'diagnosis_32gpu/mopd_v6_candidate1/resume/runtime_verified.json').exists() else 'pending_exact_candidate_v6_GPU',
            previous_v5_runtime_validation_state='passed_execution_only' if (ROOT/'diagnosis_32gpu/mopd_v5_runtime/runtime_verified.json').exists() else 'pending',
            budget_rule='Sample complete4756000 spatial training pool; select updates from measured full-budget throughput, reserving final evaluation/upload time',
            actual_mopd_training_started=False),
        final_validation=dict(state='pending_final_model',priority='final_MOPD_paper_benchmark_performance',retention_role='report_and_diagnose'),hf_upload=dict(state='pending_final_model',weights_uploaded=False),
        deadline_utc='2026-11-02T05:48:02.205167+00:00',five_day_completion_guaranteed=False,
        known_quality_failures=['Original A DoF0/580valid; newcodebook580/580valid but heldoutrotation81.80/89.15deg versus train-onlycategoryprior73.91deg. Trainfitlow82deg/high17.86deg versusheldout81.80/89.15; angular80.65deg;expandedunion/balanced81.36/81.57degdidnotimprove;expandedlearningstrengthcontrolplanned;noqualifiedAteacher yet',
            'G repaired full81250 masks98.304%valid, IoU0.13761; actual geometry quality and final MOPD still pending'],
        sft_retention_diagnostic='Broad QA candidate AI2D87.63, ScienceQA97.42, TextVQA85.88. Intermediate scores are diagnostic, not an admission blocker.',
        full_data_scale=dict(total_records=5800000,sft_records=5220000,spatial_pool=4756000,heldout_records=580000,images=160000),
        hf_server_credentials_modified=False)
    diagnostic=ROOT/'diagnosis_32gpu'
    # Read an explicit, evidence-backed decision. Runtime PASS files alone never
    # create teacher approval or final model acceptance.
    review_path=ROOT/'formal_mopd/candidate1_quality_review.json'
    if review_path.exists():
        review=json.loads(review_path.read_text())
        assert review['decision']=='admit_exact_candidates_for_formal_MOPD_with_intermediate_evaluation'
        assert all(review['family_assessments'][f]['eligible'] for f in 'GSA')
        for path,digest in review['evidence'].items():assert sha(Path(path))==digest
        result['final_training'].update(state='teacher_usefulness_reviewed_waiting_throughput_budget',
            teacher_review=dict(path=str(review_path),sha256=sha(review_path)),
            runtime_preparation_state='exact_candidate_tests_running',
            universal_teacher_superiority_claimed=False)
        result['known_quality_failures']=[
            'A pose remains weak:80.65deg, barely better than adaptedstudent80.90deg and worse than categoryprior; affordance teacher also weaker. Admission is supported by action+5.59pp, not solvedgeometry.',
            'G mask matched validity98.31%vs50.17%,IoU.13336vs.08636; teacher bbox/detection/point scores slightly lower than student. Negative transfer must be checked.',
            'Student public macro69.982857 vs paper74.27. Formal final benchmark performance remains unverified.']
        result['sft_retention_diagnostic']='Adaptedstudent AI2D88.0181,ScienceQA98.0664,TextVQA86.7540. Diagnostic only.'
    result['formal_runs']={}
    for plan_path in (ROOT/'formal_mopd/runs').glob('*/plan.json'):
        plan=json.loads(plan_path.read_text());folder=plan_path.parent;run=Path(plan['training_run'])
        item=dict(plan=str(plan_path),plan_sha256=sha(plan_path),world=plan['world'],updates=plan['steps'],
                  budget=plan['budget'],state='prepared',public_reports=[],failures=[])
        if (folder/'submission.json').exists():
            item.update(state='submitted',submission=json.loads((folder/'submission.json').read_text()))
        if (run/'status.json').exists():
            state=json.loads((run/'status.json').read_text());assert state['smoke'] is False
            item.update(state='training',training=state)
            logs=[json.loads(line) for line in (run/'logging.jsonl').open()][-128:]
            if logs:
                seconds=sum(row['seconds'] for row in logs)/len(logs)
                item['remaining_training_hours_recent_throughput']=(plan['steps']-state['step'])*seconds/3600
        if (run/'complete.json').exists():
            item.update(state='training_complete_requires_final_validation',complete=json.loads((run/'complete.json').read_text()))
        for report_path in folder.glob('evaluation/*/report.json'):
            report=json.loads(report_path.read_text())
            item['public_reports'].append(dict(path=str(report_path),sha256=sha(report_path),step=report['step'],
                macro=report['paper_aligned_macro'],checkpoint_minus_initial=report['checkpoint_minus_initial_percentage_points'],
                full_native_validation=False,final_quality_accepted=False))
        for pattern in ['*failure.json','training/node*_failure.json','evaluation/*/failure.json']:
            for failure in folder.glob(pattern):item['failures'].append(dict(path=str(failure),detail=json.loads(failure.read_text())))
        if (folder/'controller_status.json').exists():item['controller']=json.loads((folder/'controller_status.json').read_text())
        result['formal_runs'][folder.name]=item
        result['final_training'].update(state=item['state'],updates=plan['steps'],world=plan['world'],
            actual_mopd_training_started=(run/'status.json').exists(),training_run=str(run))
    policy=ROOT/'completion_policy.json'
    if policy.exists():result['completion_policy']=json.loads(policy.read_text())
    result['diagnosis']={'benchmarks':{},'repairs':{}}
    for path in diagnostic.glob('benchmarks/*/*/status.json'):
        try:result['diagnosis']['benchmarks'][str(path.parent.relative_to(diagnostic))]=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):pass
    for path in diagnostic.glob('repairs/*/status.json'):
        try:
            value=json.loads(path.read_text())
            if (path.parent/'complete.json').exists():value.update(stage='complete',completion=json.loads((path.parent/'complete.json').read_text()))
            result['diagnosis']['repairs'][path.parent.name]=value
        except (OSError,json.JSONDecodeError):pass
    for label,pattern in [('retention_repairs','retention_repairs/*/status.json'),('retention_benchmarks','retention_benchmarks/*/*/status.json'),('backend_probes','backend_probes_v2/*/status.json'),('teacher_candidate_evaluation','repair_evaluations/*/*/evaluation/status.json'),('codec_repairs','codec_repairs/*/status.json'),('codec_evaluation','codec_evaluations/*/*/evaluation/status.json'),('spatial_repairs','spatial_repairs/*/status.json'),('spatial_benchmarks','spatial_benchmarks/*/*/status.json'),('spatial_native','spatial_native/*/*/evaluation/status.json'),('codebook_repairs','codebook_repairs/*/status.json'),('codebook_evaluation','codebook_evaluations/*/*/evaluation/status.json'),('mopd_v5_runtime','mopd_v5_runtime/status.json'),('mopd_v5_probe','mopd_v5_full_budget_probe/status.json'),('codebook_trainfit','codebook_trainfit/*/*/status.json'),('pose_repairs','pose_repairs/*/status.json'),('pose_evaluation','pose_evaluations/*/*/evaluation/status.json'),('prism_expanded_repairs','prism_expanded_repairs/*/status.json'),('prism_expanded_evaluation','prism_expanded_evaluations/*/*/evaluation/status.json'),('student_adaptation','student_adaptation_run/*/status.json'),('student_adaptation_smoke','student_adaptation_smoke/status.json'),('student_adaptation_benchmarks','student_adaptation_benchmarks/*/*/status.json'),('student_adaptation_native_A','student_adaptation_evaluations/*/*/evaluation/status.json'),('student_adaptation_node','student_adaptation_node_status.json'),('student_adaptation_submission','student_adaptation_submission_status.json'),('G_priority_native','G_spatial_priority_native/*/evaluation/status.json'),('G_priority_nodes','g_priority_native_node*_status.json'),('G_priority_submission','g_priority_submission_status.json'),('prism_learning_repairs','prism_learning_repairs/*/status.json'),('prism_learning_evaluation','prism_learning_evaluations/*/*/evaluation/status.json'),('prism_learning_node','prism_learning_node_status.json'),('student_matched_native','student_matched_native/*/evaluation/status.json'),('student_matched_submission','student_matched_native_submission_status.json'),('student_matched_node','student_matched_native_node_status.json'),('mopd_v6_candidate_runtime','mopd_v6_candidate*/resume/status.json'),('mopd_v6_candidate_probe','mopd_v6_candidate*/probe/status.json'),('mopd_v6_submission','mopd_v6_candidate*/submission_status.json'),('mopd_v6_multinode_nodes','mopd_v6_candidate1_multinode/actual/node*_status.json'),('mopd_v6_multinode_training','mopd_v6_candidate1_multinode/actual/*/status.json')]:
        result['diagnosis'][label]={}
        for path in diagnostic.glob(pattern):
            try:
                value=json.loads(path.read_text())
                if (path.parent/'complete.json').exists():value.update(stage='complete',completion=json.loads((path.parent/'complete.json').read_text()))
                elif (path.parent/'metrics.json').exists():value.update(stage='complete',metrics=str(path.parent/'metrics.json'))
                result['diagnosis'][label][str(path.parent.relative_to(diagnostic))]=value
            except (OSError,json.JSONDecodeError):pass
    result['diagnosis']['failures']=[]
    failure_patterns=['repairs/*/failure.json','retention_repairs/*/failure.json','repair_evaluations/*/*/failure.json','retention_benchmarks/*/*/failure.json','backend_probes_v2/*/failure.json','codec_repairs/*/failure.json','codec_evaluations/*/*/failure.json','spatial_repairs/*/failure.json','spatial_benchmarks/*/*/failure.json','spatial_native/*/*/failure.json','codebook_repairs/*/failure.json','codebook_evaluations/*/*/failure.json','mopd_v5_runtime/failure.json','mopd_v5_full_budget_probe/failure.json','pose_repairs/*/failure.json','pose_evaluations/*/*/failure.json','prism_expanded_repairs/*/failure.json','prism_expanded_evaluations/*/*/failure.json','prism_expansion/preparation_failure.json','student_adaptation_submission_failure.json','student_adaptation_benchmarks/*/*/failure.json','student_adaptation_evaluations/*/*/failure.json','G_spatial_priority_native/*/failure.json','g_priority_submission_failure.json','g_priority_aggregation_failure.json','prism_learning_repairs/*/failure.json','prism_learning_evaluations/*/*/failure.json','student_matched_native/*/failure.json','student_matched_native_submission_failure.json','student_matched_teacher_scoring_failure.json','mopd_v6_candidate*/resume/failure.json','mopd_v6_candidate*/probe/failure.json','mopd_v6_candidate*/submission_failure.json','mopd_v6_candidate1_multinode/actual/node*_failure.json','mopd_scaling_comparison_failure.json']
    for path in [p for pattern in failure_patterns for p in diagnostic.glob(pattern)]:
        try:result['diagnosis']['failures'].append(dict(path=str(path),detail=json.loads(path.read_text())))
        except (OSError,json.JSONDecodeError):pass
    result['diagnosis']['reports']=[str(diagnostic/name) for name in ['benchmark_comparison.json','mc_format_diagnosis.json','repair_comparison.json','retention_recovery_comparison.json','backend_complete_v2.json','matched_mask_interim.json','mopd_replay_runtime/runtime_verified.json','data_scale_reconciliation.json','paper_protocol_reconciliation.json','dof_codec_comparison.json','dof_generalization_audit.json','spatial_replay/manifest.json','spatial_comparison_joint.json','spatial_comparison_G.json','spatial_native_comparison.json','dof_codebook_comparison.json','mopd_v5_runtime/runtime_verified.json','mopd_v5_full_budget_probe/probe_verified.json','augmented_data_scale_reconciliation.json','dof_codebook_trainfit_comparison.json','dof_codebook_distribution_audit.json','dof_pose_comparison.json','dof_pose_cpu_validation.json','prism_expansion/manifest.json','prism_expanded_cpu_validation.json','prism_expanded_comparison.json','augmented_data_scale_reconciliation_v2.json','spatial_completed_public_interim.json','student_adaptation_cpu_validation.json','student_adaptation_public_comparison.json','student_adaptation_native_A_comparison.json','G_spatial_priority_native_comparison.json','prism_pose_coverage_audit.json','joint_spatial_initialization_lineage.json','prism_learning_cpu_validation.json','prism_learning_comparison.json','prospective_teacher_lineage_validation.json','student_matched_native_comparison.json','student_matched_teachers_comparison.json','A_adapted_student_matched_review.json','mopd_v6_candidate1/resume/runtime_verified.json','mopd_v6_candidate1/probe/probe_verified.json','mopd_v6_candidate1_multinode/actual/resume/runtime_verified.json','mopd_v6_candidate1_multinode/actual/probe/probe_verified.json','mopd_scaling_comparison.json'] if (diagnostic/name).exists()]
    if (diagnostic/'prism_expansion/preparation_status.json').exists():result['diagnosis']['prism_preparation']=json.loads((diagnostic/'prism_expansion/preparation_status.json').read_text())
    runtime=diagnostic/'mopd_replay_runtime/status.json'
    if runtime.exists():result['diagnosis']['mopd_replay_execution_only']=json.loads(runtime.read_text())
    cache=ROOT/'cloud_job_status.json' 
    cached_ids={j['id'] for j in json.loads(cache.read_text())['jobs']} if cache.exists() else set()
    if not cache.exists() or now-cache.stat().st_mtime>300 or cached_ids!={j['id'] for j in result['submitted_jobs']}:
        cloud=[]
        for job in result['submitted_jobs']:
            try:
                proc=subprocess.run(['/root/.volc/bin/volc','--log-level','error','ml_task','get','--id',job['id'],'--output','json','--format','Id,Name,Status'],capture_output=True,text=True,timeout=20)
                rows=[json.loads(line) for line in proc.stdout.splitlines() if line.startswith('[{"Id"')]
                assert proc.returncode==0 and len(rows)==1
                cloud.append(dict(**job,status=rows[0][0]['Status']))
            except Exception as exc:cloud.append(dict(**job,status='UNKNOWN',error=type(exc).__name__))
        save(cache,dict(checked_utc=datetime.now(timezone.utc).isoformat(),jobs=cloud))
    result['cloud']=json.loads(cache.read_text())
    result['allocated_or_pending_gpus']=sum(j['gpus'] for j in result['cloud']['jobs'] if j['status'] not in ['Success','Succeeded','Failed','Killed','Stopped'])
    result['resource_ceiling_respected']=result['allocated_or_pending_gpus']<=result['campaign_gpu_ceiling']
    if (diagnostic/'benchmark_comparison.json').exists():
        result['diagnosis']['benchmark_comparison']=str(diagnostic/'benchmark_comparison.json')
    save(ROOT/'campaign_status.json',result)
    return result

def aggregate():
    groups=defaultdict(lambda:dict(n=0,sums=defaultdict(float)))
    seen=defaultdict(set);evidence=[]
    for task in TASKS:
        out=RUN/task['id'];mfile=out/'evaluation/metrics.json'
        receipt=json.loads((out/'complete.json').read_text())
        assert receipt['task']==task and receipt['metrics_sha256']==sha(mfile)
        metrics=json.loads(mfile.read_text())
        assert metrics['state']=='complete_scope_shard'
        assert metrics['n']==metrics['expected_records']==task['records']
        for name,digest in metrics['prediction_sha256'].items():
            file=out/'evaluation'/name;assert sha(file)==digest
            for line in file.open():
                row=json.loads(line);identity=(row['native_form'],row['id'])
                assert identity not in seen[task['model']], 'Duplicate within model evaluation'
                seen[task['model']].add(identity)
        for g in metrics['groups']:
            group=groups[(task['model'],g['form'],g['source'])];group['n']+=g['n']
            for key,value in g['mean'].items():group['sums'][key]+=g['n']*value
        evidence.append(dict(shard=task['id'],metrics_sha256=sha(mfile)))
    assert len(seen['joint'])==580000
    assert seen['G'].isdisjoint(seen['S']) and seen['G'].isdisjoint(seen['A']) and seen['S'].isdisjoint(seen['A'])
    assert seen['joint']==seen['G']|seen['S']|seen['A']
    rows=[dict(model=m,form=f,source=s,n=g['n'],mean={k:v/g['n'] for k,v in g['sums'].items()})
          for (m,f,s),g in sorted(groups.items())]
    impossible=[dict(model=g['model'],form=g['form'],source=g['source'],n=g['n']) for g in rows
                if g['model']!='joint' and g['mean'].get('contract_valid',0)==0]
    save(ROOT/'qualification_summary.json',dict(state='complete_evaluation',records=1160000,
        groups=rows,evidence=evidence,global_composite_score=None,
        teacher_verdict='FAIL' if impossible else 'REQUIRES_QUALITY_REVIEW',
        necessary_quality_condition_failed=impossible,
        quality_rule='A required output form with zero valid expert outputs cannot qualify; nonzero validity alone never establishes eligibility',
        full_GSA_qualified=False,final_mopd=False,weights_uploaded=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');args=ap.parse_args()
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                result=snapshot()
                if result['evaluation']['all_complete'] and not (ROOT/'qualification_summary.json').exists():
                    aggregate()
                priority=ROOT/'diagnosis_32gpu'
                if (all((priority/'G_spatial_priority_native'/str(i)/'complete.json').is_file() for i in range(16))
                        and not (priority/'G_spatial_priority_native_comparison.json').exists()
                        and not (priority/'g_priority_aggregation_failure.json').exists()):
                    proc=subprocess.run(['python3',str(priority/'score_g_priority_native.py')],capture_output=True,text=True,timeout=300)
                    if proc.returncode:save(priority/'g_priority_aggregation_failure.json',dict(error=proc.stderr[-4000:],time=time.time()))
                if ((priority/'student_matched_native_comparison.json').is_file()
                        and (priority/'G_spatial_priority_native_comparison.json').is_file()
                        and not (priority/'student_matched_teachers_comparison.json').exists()
                        and not (priority/'student_matched_teacher_scoring_failure.json').exists()):
                    proc=subprocess.run(['python3',str(priority/'score_student_matched_teachers.py')],capture_output=True,text=True,timeout=300)
                    if proc.returncode:save(priority/'student_matched_teacher_scoring_failure.json',dict(error=proc.stderr[-4000:],time=time.time()))
                if ((priority/'mopd_v6_candidate1/probe/probe_verified.json').is_file()
                        and (priority/'mopd_v6_candidate1_multinode/actual/probe/probe_verified.json').is_file()
                        and not (priority/'mopd_scaling_comparison.json').exists()
                        and not (priority/'mopd_scaling_comparison_failure.json').exists()):
                    proc=subprocess.run(['python3',str(ROOT/'compare_mopd_scaling.py'),'--output',str(priority/'mopd_scaling_comparison.json')],capture_output=True,text=True,timeout=300)
                    if proc.returncode:save(priority/'mopd_scaling_comparison_failure.json',dict(error=proc.stderr[-4000:],time=time.time()))
                save(ROOT/'monitor_health.json',dict(state='healthy',updated_utc=datetime.now(timezone.utc).isoformat()))
            except Exception as exc:
                save(ROOT/'monitor_health.json',dict(state='error_retrying',error=repr(exc),updated_utc=datetime.now(timezone.utc).isoformat()))
                import traceback
                traceback.print_exc()
                if args.once:raise
            if args.once:break
            time.sleep(60)

if __name__=='__main__':main()
