"""Real-prefix CPU tests; terminal states are simulated only in temporary files.

The live run is not stopped or selected. No final admission, GPU validation,
quality review, merge, package or upload is performed by this test.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import subprocess
from common import C,HERE,TRAIN,bound,formal_training,load_spec,read,require,save,sha
from selection import prepare,validate,observe
from exposure_audit import audit
from lineage import logged_totals


def main():
    plan=C/'formal_mopd/runs/candidate1_20261029/plan.json'
    prefix=C/'checkpoint_prefixes/candidate1_step512/report.json'
    run=Path(read(plan)['training_run'])
    task_id=read(plan.parent/'submission.json')['response']['Id']
    live=observe(task_id)
    require(live['task']['Status']=='Running','This preflight is designed to prove refusal of the current live run')
    checks=[]
    def rejected(name,fn):
        try:fn()
        except (ValueError,KeyError,FileNotFoundError):checks.append(name)
        else:raise AssertionError('Unsafe case accepted: '+name)
    args=[plan,prefix,'planned_run_stopped_select_committed_prefix','CPU test only; never a delivery decision']
    rejected('actual_running_cloud_rejected',lambda:prepare(*args,live))
    stopped=dict(task=dict(Id=task_id,Status='Stopped'),observed=time.time())
    for status in ['UNKNOWN','Pending','Failed','Running']:
        rejected('cloud_'+status+'_rejected',lambda status=status:prepare(*args,dict(task=dict(Id=task_id,Status=status),observed=time.time())))
    rejected('stale_cloud_rejected',lambda:prepare(*args,dict(task=dict(Id=task_id,Status='Stopped'),observed=time.time()-121)))
    rejected('wrong_task_rejected',lambda:prepare(*args,dict(task=dict(Id='t-wrong',Status='Stopped'),observed=time.time())))
    rejected('success_without_completed_run_rejected',lambda:prepare(plan,prefix,'planned_run_completed_select_recorded_checkpoint',args[-1],dict(task=dict(Id=task_id,Status='Success'),observed=time.time())))
    receipt=prepare(*args,stopped)
    with tempfile.TemporaryDirectory(prefix='selection_cpu_only_',dir=C) as temporary:
        path=Path(temporary)/'SIMULATED_NOT_A_REAL_SELECTION.json'
        save(path,receipt)
        selection,verified_prefix=validate(path)
        require(selection['selected_step']==512 and selection['planned_steps']==1664
                and selection['original_planned_run_completed'] is False,'Prefix inflated to original budget')
        endpoint,protocol,checkpoint,_=formal_training(run,full_checkpoint=False,selection_path=path)
        require(endpoint['stage']=='selected_formal_checkpoint' and endpoint['examples_seen']==16384,'False completion or exposure')
        stage=audit(run,selection_path=path)
        initial=read(protocol['student_initialization_lineage']['path'])
        original=read(C.parent.parent/'arxiv/native_training_export_20261012_v4_r2/manifest.json')
        totals=logged_totals(initial['post_sft_stages']+[stage],original['selection_sha256'])
        require(totals['exposures']==initial['post_sft_logged_exposures']+16384,'Cross-stage exposure inflation')
        checks+=['real16384_prefix_validated','actual_cross_stage_counts_verified','future_live_rows_excluded']
        # Exercise the real merged checkpoint binding with a temporary simulated
        # terminal-state fixture, never an actual delivery selection.
        spec_path=Path(temporary)/'SIMULATED_SPEC_DO_NOT_SUBMIT.json'
        model=plan.parent/'evaluation/512/merged'
        subprocess.run([TRAIN,str(HERE/'prepare.py'),'--training-run',str(run),
                        '--selection',str(path),'--model',str(model),'--output',str(spec_path)],
                       capture_output=True,text=True,check=True)
        spec=load_spec(spec_path,verify_model=False)
        require(spec['checkpoint']==str(checkpoint) and spec['selected_step']==512
                and spec['native_records']==580000 and spec['general_records']==10105,
                'Selected final specification altered evaluation scope')
        checks.append('real_merged512_full_final_spec_preparation_passed_in_temporary_fixture')
        badspec=deepcopy(spec);badspec['planned_steps']=512;save(spec_path,badspec)
        rejected('spec_budget_mutation_rejected',lambda:load_spec(spec_path,verify_model=False))
        save(spec_path,spec)
        for key,value in [('selected_step',1664),('planned_steps',512),('original_planned_run_completed',True),
                          ('selected_checkpoint_is_planned_tip',True),('final_quality_accepted',True),
                          ('checkpoint',str(run/'checkpoint-128'))]:
            changed=deepcopy(receipt);changed[key]=value;save(path,changed)
            rejected('mutated_'+key+'_rejected',lambda:validate(path))
        save(path,receipt)
        expanded=deepcopy(stage);expanded['exposure_files']=[str(run/('exposure_rank%d.jsonl'%rank)) for rank in range(protocol['world_size'])]
        rejected('future_suffix_in_lineage_rejected',lambda:logged_totals([expanded],original['selection_sha256']))
    # Original helpers are protected by a live handoff: this alternative must
    # not change even one byte of them.
    derivation=read(HERE/'derivation.json')
    for file,digest in derivation['source_files_sha256'].items():require(sha(file)==digest,'Live delivery dependency changed')
    checks.append('all_original_delivery_files_unchanged')
    result=dict(state='passed_cpu_selection_preflight_only',checks=checks,
        actual_cloud_observation=live,real_prefix_step=512,real_prefix_exposures=16384,
        combined_post_sft_exposures=totals['exposures'],combined_unique_supplements=totals['unique_supplemental_records'],
        terminal_state_simulated_only_in_deleted_temporary_fixture=True,
        real_merged_checkpoint_binding_checked=True,
        checkpoint_selected_for_delivery=False,live_run_stopped=False,final_quality_accepted=False,
        actual_selected_final_gpu_evaluation=False,package_created=False,weights_uploaded=False,
        evidence={str(p):sha(p) for p in [plan,prefix,HERE/'derivation.json',*sorted(HERE.glob('*.py'))]},completed=time.time())
    save(HERE/'selection_cpu_validation.json',result)
    print(__import__('json').dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))


if __name__=='__main__':main()
