"""Publish only the explicitly accepted exact candidate, then verify the whole Hub.

Geometry continuation uses its own exact-model evaluation,31-key backups and review.
No GPU submission, quality decision, credential-store mutation or local deletion.
An interrupted/ambiguous upload requires reconciliation instead of blind retry.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
Q=Path(__file__).resolve().parent
C=Q.parent;V=C/'verification_20261101'
P=C/'geometry_repair_candidate_20261103';D=C/'delivery_runs/geometry_repair_candidate_20261103';H=C/'final_delivery_geometry_v1'
B=C/'geometry_checkpoint_backup_20261103';PRE=V/'geometry_merged_blob_preupload'
OUT=Q/'final_release';TRAIN=C.parent/'.venv-train/bin/python'
POLICY=Q/'release_policy_v2.json'

def read(path):return json.loads(Path(path).read_text())
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def status(stage,**values):save(OUT/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**values))

def quality_decision(review,report,report_sha,spec_sha):
 require(review['schema']=='anvil-final-quality-review-v1','Unknown review schema')
 require(review['report_sha256']==report_sha and review['final_spec_sha256']==spec_sha,'Review binds a different model/report')
 require(report['state']=='complete_requires_quality_review' and report['final_spec_sha256']==spec_sha,'Incomplete or different report')
 require(report['native']['n']==580000 and sum(x['n'] for x in report['general'].values())==10105,'Incomplete evaluation scope')
 require(isinstance(review['accepted'],bool) and bool(review.get('rationale')),'Missing quality decision/rationale')
 require(set(review['benchmark_assessments'])==set(report['paper_aligned_public']) and len(report['paper_aligned_public'])==7,'Every benchmark must be reviewed')
 for name,score in report['paper_aligned_public'].items():
  item=review['benchmark_assessments'][name]
  require(item['score']==score and item['paper']==report['paper_target'][name],'Review changes measured benchmark scores')
 return review['accepted']

def validate_policy():
 policy=read(POLICY)
 require(policy['schema']=='anvil-reviewed-final-delivery-v1' and policy['key_checkpoints']==31,'Wrong release policy')
 for path,digest in policy['files'].items():require(sha(path)==digest,'Changed release dependency: '+path)
 require(sha(P/'spec.json')==policy['pipeline_spec_sha256'],'Different pipeline')
 return policy

def lease(path):
 handle=Path(path).open('a')
 try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BaseException:
  handle.close();raise
 return handle

def run_tool(script,*args):
 log=OUT/'logs'/(Path(script).stem+'.log');log.parent.mkdir(exist_ok=True)
 with log.open('a') as stream:
  subprocess.run([str(TRAIN),'-u',str(script),*map(str,args)],stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--inspect',action='store_true');args=parser.parse_args()
 policy=validate_policy()
 if args.inspect:
  print(json.dumps(dict(state='prepared',quality_decision_automated=False,requires_checkpoints=31,hub_calls=False,gpu_calls=False,cleanup=False)));return
 OUT.mkdir(exist_ok=True);own=lease(OUT/'worker.lock')
 require(not (OUT/'complete.json').exists(),'Release already complete; inspect existing receipt')
 deadline=datetime.fromisoformat(read(P/'spec.json')['deadline_utc']).timestamp()
 review_path=Q/'quality_review.json';report_path=D/'evaluation/final_report.json';spec_path=D/'evaluation_spec.json'
 while time.time()<deadline:
  failures=[p for p in [D/'failure.json',P/'controller_failure.json',B/'failure.json',PRE/'failure.json'] if p.exists()]+list(P.glob('node*_failure.json'))
  require(not failures,'Training/evaluation/backup dependency failed; inspect before proceeding')
  if review_path.exists():
   accepted=quality_decision(read(review_path),read(report_path),sha(report_path),sha(spec_path))
   if not accepted:
    status('quality_rejected_no_publication',review_sha256=sha(review_path),final_model_published=False)
    return
   ready=(B/'complete.json').exists() and (PRE/'complete.json').exists()
   handoff=read(D/'status.json')
   ready=ready and handoff['stage']=='full_final_report_ready_requires_quality_review'
   if ready:break
  status('waiting_full_evaluation_and_explicit_quality_review',report_ready=report_path.exists(),review_ready=review_path.exists(),
   checkpoint_backups_ready=(B/'complete.json').exists(),uncommitted_preupload_ready=(PRE/'complete.json').exists(),final_model_published=False)
  time.sleep(45)
 else:raise TimeoutError('Final delivery dependencies unavailable before deadline')
 # Acquire all relevant writer leases after workers reach their completion receipts.
 # Completion may be recorded just before process exit; wait without duplicate publication.
 held=[]
 while time.time()<deadline:
  try:
   for path in [C/'hub_writer.lock',B/'worker.lock',PRE/'worker.lock',D/'supervisor.lock',P/'controller.lock']:held.append(lease(path))
   break
  except BlockingIOError:
   for h in held:h.close()
   held=[];status('waiting_completed_writer_exit',final_model_published=False);time.sleep(10)
 else:raise TimeoutError('Completed campaign writers did not release their leases')
 validate_policy()
 sys.path.insert(0,str(C));from final_delivery_handoff_v1 import cloud_snapshot
 cloud,active=cloud_snapshot();require(active==0,'Campaign jobs are active or UNKNOWN')
 ids=[read(P/'submission.json')['response']['Id']]+[read(D/'submissions'/str(i)/'submission.json')['response']['Id'] for i in range(4)]
 by_id={row['id']:row for row in cloud};require(all(by_id[i]['status']=='Success' for i in ids),'Training/evaluation not successful')
 require(quality_decision(read(review_path),read(report_path),sha(report_path),sha(spec_path)),'Quality was rejected')
 backup_path=B/'mopd_256/verification_receipt.json';backup=read(backup_path);pipeline=read(P/'spec.json')
 require(backup['state']=='complete_remote_verified_intermediate_checkpoints' and len(backup['key_checkpoints'])==31,'Incomplete checkpoint backup')
 require({k['path'] for k in backup['key_checkpoints']}==set(pipeline['key_checkpoints']),'Different checkpoint inventory')
 require(backup['manifest_sha256']==sha(backup_path.parent/'manifest.json'),'Changed backup manifest')
 pre=read(PRE/'complete.json')
 require(pre.get('required_backups_verified')==31 and pre.get('source_sha256')==sha(V/'preupload_geometry_merged_blobs_v2.py'),'Different corrected preupload worker')
 require(pre['stage']=='uncommitted_merged_blobs_preuploaded' and pre['final_spec_sha256']==sha(spec_path)
  and pre['backup_receipt_sha256']==sha(backup_path) and pre['revision_after']==backup['revision'],'Different merged preupload')
 binding=dict(review_sha256=sha(review_path),report_sha256=sha(report_path),evaluation_spec_sha256=sha(spec_path),
  backup_receipt_sha256=sha(backup_path),policy_sha256=sha(POLICY),active_campaign_gpus=active)
 save(OUT/'accepted_release_binding.json',binding)
 package=D/'final_package';manifest=package/'release_manifest.json'
 if not package.exists():
  status('building_accepted_package',**binding)
  run_tool(H/'package.py','--spec',spec_path,'--report',report_path,'--review',review_path,'--lineage',D/'training_lineage.json',
   '--interface-validation',H/'interface_validation/rotation128_interface_validation.json','--output',package)
 require(manifest.exists(),'Partial package exists; reconcile instead of overwriting')
 sys.path.insert(0,str(H));from upload import verify_local
 sealed=verify_local(package)
 require(sealed['review_sha256']==sha(review_path) and sealed['final_spec_sha256']==sha(spec_path),'Package differs from accepted model')
 token=os.environ.get('ANVIL_HF_TOKEN');require(bool(token),'Missing transient authorized token')
 from huggingface_hub import HfApi
 api=HfApi(token=token);require(api.whoami()['name']=='Knowin-29','Wrong account')
 final=OUT/'upload_verification.json';uploaded=final.with_suffix('.uploaded.json');intent=OUT/'upload_intent.json'
 if not final.exists():
  if uploaded.exists():
   receipt=read(uploaded);require(receipt['manifest_sha256']==sha(manifest),'Ambiguous different upload')
   status('verifying_existing_upload',revision=receipt['revision'])
   run_tool(H/'upload.py','--package',package,'--receipt',final,'--verify-revision',receipt['revision'])
  else:
   require(not intent.exists(),'Uncertain prior upload; reconcile the Hub before retrying')
   info=api.model_info('Knowin-29/Anvil-MOPD-9B')
   require(info.sha==backup['revision'] and {x.rfilename for x in info.siblings}==set(backup['files']),'Remote changed after sequential backups')
   save(intent,dict(parent_revision=info.sha,manifest_sha256=sha(manifest),created=time.time()))
   status('publishing_accepted_model',parent_revision=info.sha,**binding)
   run_tool(H/'upload.py','--package',package,'--receipt',final)
 receipt=read(final)
 require(receipt['state']=='complete_remote_verified' and receipt['manifest_sha256']==sha(manifest),'Final upload not verified')
 whole=OUT/'whole_hub_verification.json'
 if not whole.exists():
  status('verifying_entire_pinned_repository',revision=receipt['revision'])
  run_tool(V/'verify_entire_geometry_hub.py','--manifest',manifest,'--final-receipt',final,'--backup-receipt',backup_path,
   '--pipeline-spec',P/'spec.json','--evaluation-spec',spec_path,'--output',whole)
 audit=read(whole)
 require(audit['state']=='complete_entire_remote_release_verified' and audit['revision']==receipt['revision'],'Whole repository not verified')
 require(len(audit['key_checkpoints'])==31 and set(audit['key_checkpoints'])==set(pipeline['key_checkpoints']),'Missing final checkpoints')
 require(api.model_info('Knowin-29/Anvil-MOPD-9B').sha==audit['revision'],'Head moved after verification')
 result=dict(state='accepted_final_model_and31_checkpoints_remotely_verified',revision=audit['revision'],
  repository='Knowin-29/Anvil-MOPD-9B',url=receipt['url'],package_manifest=str(manifest),
  upload_receipt=str(final),whole_hub_receipt=str(whole),files=audit['remote_file_count'],completed=time.time(),
  quality_review=read(review_path),quality_review_sha256=sha(review_path),cleanup_performed=False,credentials_modified=False)
 save(OUT/'complete.json',result);status(result['state'],revision=result['revision'],final_model_published=True)
 print(json.dumps(dict(state=result['state'],url=result['url'],files=result['files'])),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if '--inspect' not in sys.argv:save(OUT/'failure.json',dict(error_type=type(exc).__name__,failed=time.time(),pid=os.getpid(),inspect_logs_and_remote_before_retry=True))
  print('Final delivery stopped: '+type(exc).__name__,flush=True)
  raise SystemExit(1)
