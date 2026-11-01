"""Refresh user-facing pointers without modifying immutable runtime evidence."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
V=Path(__file__).resolve().parent;C=V.parent;P=C/'balanced_repair_20261102';FD=C/'delivery_runs/balanced_repair_20261102'
D=Path('/Knowin/demo/weilinruan/Anvil/03_Delivery');p=D/'CURRENT_CAMPAIGN.json'
def read(x):return json.loads(x.read_text())
def bound(x):return dict(path=str(x),sha256=hashlib.sha256(x.read_bytes()).hexdigest())
def write(x,v):
 t=x.with_suffix(x.suffix+'.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(x)
previous=V/'CURRENT_CAMPAIGN_before_balanced.json'
if not previous.exists():previous.write_bytes(p.read_bytes())
current=read(p);spec=read(P/'spec.json');sub=read(P/'submission.json');focus=C/'delivery_runs/focus_repair_20261101'
report=read(focus/'evaluation/final_report.json');backup=C/'focus_checkpoint_backup_20261101/mopd_512/verification_receipt.json'
b=read(backup);deleted=read(C/'storage_cleanup_rejected_focus_20261102/deletion_receipt.json')
current.update(checked_at=datetime.now(timezone.utc).isoformat(),previous_snapshot_retained=str(previous),
 live_monitor=str(C/'campaign_balanced_status.json'),live_monitor_health=str(C/'monitor_balanced_health.json'))
current['rejected_completed_experiment']=dict(task_id='t-20261101095411-z67h9',training_status='Success',
 final_report=bound(focus/'evaluation/final_report.json'),quality_review=bound(focus/'quality_review.json'),
 paper_aligned_public=report['paper_aligned_public'],paper_aligned_macro=sum(report['paper_aligned_public'].values())/7,
 accepted_as_final=False,in_new_model_ancestry=False,retained_nonancestor_checkpoints=spec['retained_nonancestor_checkpoints'],
 derived_inference_cleanup=bound(C/'storage_cleanup_rejected_focus_20261102/deletion_receipt.json'))
current['continuation']=dict(task_id=sub['response']['Id'],gpus=32,spec=str(P/'spec.json'),spec_sha256=bound(P/'spec.json')['sha256'],
 submission=str(P/'submission.json'),controller=str(P/'controller_status.json'),cloud_state_at_snapshot=read(FD/'status.json').get('cloud_status'),
 inherits_completed_parent_weights=True,original_sft_rerun=False,fresh_optimizer_per_new_stage=True,
 parent_adapter=spec['initial'],planned_stages=spec['stages'],planned_exposures_not_actual=spec['planned_source_exposures'],budget_estimate=spec['budget_estimate'],
 first_production_checkpoint_integrity=bound(P/'repair_checkpoint64_verification.json'))
current['final_delivery']=dict(root=str(FD),policy=str(FD/'policy.json'),status=str(FD/'status.json'),quality_accepted=False,weights_uploaded=False,
 full_validation=dict(native=580000,public=7366,general=10105),review_required_after_report=True)
current['checkpoint_backups']=dict(verified_count=14,declared_count=19,existing_revision=b['revision'],
 live_status=str(C/'balanced_checkpoint_backup_20261102/status.json'),failure=str(C/'balanced_checkpoint_backup_20261102/failure.json'),
 completion_receipt=str(C/'balanced_checkpoint_backup_20261102/complete.json'),final_upload_requires_worker_terminal=True)
current['hf_inventory']=dict(repository=b['repository'],revision=b['revision'],verified_files=len(b['files']),intermediate_checkpoints_verified=14,
 weights_present=True,final_model_uploaded=False,verification_receipt=str(backup),credentials_modified=False)
current['entire_final_hub_verifier']=dict(source=str(V/'verify_entire_balanced_hub.py'),requires_declared_checkpoints=19,
 execution_required_after_final_upload=True,actual_release_verified=False)
current['merged_weights_uncommitted_preupload']=dict(process=str(V/'balanced_merged_blob_preupload/process.json'),
 status=str(V/'balanced_merged_blob_preupload/status.json'),failure=str(V/'balanced_merged_blob_preupload/failure.json'),
 completion=str(V/'balanced_merged_blob_preupload/complete.json'),required_backups=19,final_upload_requires_worker_terminal=True,published=False)
current['post_release_weight_cleanup']=dict(source=str(V/'cleanup_verified_balanced_weights.py'),sha256=bound(V/'cleanup_verified_balanced_weights.py')['sha256'],
 default='dry_run',execute_after='accepted final model + independent all19 whole-Hub verification + idle cloud/local campaign writers',executed=False,cpu_temporary_file_checks_passed=10)
current['storage_last_review'].update(checked_at=datetime.now(timezone.utc).isoformat(),deleted_bytes_this_review=deleted['bytes_removed'],
 cumulative_deleted_bytes=424738842000+deleted['bytes_removed'],deletion_receipt=str(C/'storage_cleanup_rejected_focus_20261102/deletion_receipt.json'))
for name,root in [('system','/'),('shared','/Knowin/sim')]:
 s=os.statvfs(root);current['storage_last_review'][name+'_free_bytes']=s.f_bavail*s.f_frsize
current['latest_failure_resume_storage_review']=str(V/'failure_resume_storage_review_balanced.md')
current['latest_monitoring_note']='Balanced32-A800 continuation is running from stronger MOPD768 parent. Focus completed but regressed and was rejected. First repair64 state integrity passed.19-checkpoint backup and full evaluation handoff active. No final quality acceptance or publication.'
current.pop('live_foreground_observer',None)
write(p,current)
print(json.dumps(dict(pointer=str(p),task_id=sub['response']['Id'],verified_checkpoint_backups=14,declared_checkpoint_backups=19)))
