"""Free only non-key optimizer payloads in completed, HF-backed historical stages."""
from pathlib import Path
import fcntl,hashlib,json,os,runpy,stat,sys,time
W=Path(__file__).resolve().parent;C=W.parent;V=C/'visual_geometry_candidate_20261104';B=C/'coordinate_checkpoint_backup_v2_20261104/mopd_256'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 assert not (W/'optimizer_cleanup_complete.json').exists()
 lock=open('/tmp/anvil-formal-mopd-submit.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
 sys.path.insert(0,str(C));from final_delivery_handoff_v1 import cloud_snapshot
 cloud,active=cloud_snapshot();assert active==0
 assert next(x for x in cloud if x['id']=='t-20261105001409-lb4r4')['status']=='Failed'
 save(W/'cloud_before_cleanup.json',dict(cloud=cloud,active=active,checked=time.time()))
 pipeline=read(V/'spec.json');receipt=read(B/'verification_receipt.json');manifest=read(B/'manifest.json');assert receipt['manifest_sha256']==sha(B/'manifest.json')
 assert len(receipt['key_checkpoints'])==35
 from huggingface_hub import HfApi
 token=runpy.run_path(str(C/'verification_20261101/preupload_merged_blobs.py'))['authorized_token']();api=HfApi(token=token)
 info=api.model_info('Knowin-29/Anvil-MOPD-9B',revision=receipt['revision'],files_metadata=True);remote={x.rfilename:x for x in info.siblings}
 assert info.sha==receipt['revision'] and set(remote)==set(receipt['files'])
 keys={Path(v):k for k,v in pipeline['key_checkpoint_sources'].items()};candidates=[];completed=[]
 for run in sorted({p.parent for p in keys if not p.is_relative_to(V)}):
  complete=read(run/'complete.json');last=Path(complete['checkpoint']);assert last in keys and last.parent==run and complete['smoke'] is False
  remote_name='checkpoints/'+keys[last]+'/adapter_model.safetensors';meta=remote[remote_name]
  assert meta.lfs.sha256==complete['adapter_sha256']==receipt['files'][remote_name]['sha256']
  completed.append(dict(run=str(run),final_checkpoint=str(last),remote_path=remote_name,adapter_sha256=complete['adapter_sha256']))
  for p in sorted(run.glob('checkpoint-*/optimizer.pt')):
   if p.parent in keys:continue
   assert p.resolve()==p and p.is_file() and p.parent.name.removeprefix('checkpoint-').isdigit()
   seal=read(p.parent/'checkpoint_complete.json');assert seal['global_step']<complete['steps'] and seal['world_size']==32
   expected=seal['files']['optimizer.pt'];st=p.stat();assert stat.S_ISREG(st.st_mode) and st.st_size==expected['bytes'] and st.st_nlink==1
   candidates.append(dict(path=str(p),bytes=st.st_size,sealed_sha256=expected['sha256'],inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns))
 assert candidates and sum(x['bytes'] for x in candidates)>400*1024**3
 names={x['path'] for x in candidates}
 for proc in Path('/proc').glob('[0-9]*'):
  try:
   for fd in (proc/'fd').iterdir():
    try:assert os.readlink(fd) not in names,'Optimizer still open'
    except FileNotFoundError:pass
  except (FileNotFoundError,PermissionError):pass
 preview=dict(state='eligible_superseded_nonkey_optimizers',created=time.time(),candidates=candidates,total_bytes=sum(x['bytes'] for x in candidates),completed_stages=completed,pinned_hf_revision=receipt['revision'],preserved='All adapters, all key checkpoints, all visual-candidate states, datasets, logs, metadata, teachers, credentials',eligibility='Completed stage final adapter verified in pinned HF revision; non-key optimizer never used by current resume or final lineage')
 save(W/'optimizer_cleanup_preview.json',preview)
 with (W/'optimizer_cleanup_deleted.jsonl').open('x') as log:
  for x in candidates:
   p=Path(x['path']);st=p.lstat();assert (st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==(x['inode'],x['bytes'],x['mtime_ns'],x['ctime_ns'])
   p.unlink();log.write(json.dumps(dict(x,deleted=time.time()))+'\n');log.flush();os.fsync(log.fileno())
 save(W/'optimizer_cleanup_complete.json',dict(state='complete',files_removed=len(candidates),bytes_freed=preview['total_bytes'],preview_sha256=sha(W/'optimizer_cleanup_preview.json'),completed=time.time(),all_adapter_weights_retained=True,all_key_checkpoints_retained=True,visual_candidate_untouched=True))
 print(json.dumps(dict(files=len(candidates),freed_gib=preview['total_bytes']/1024**3,revision=receipt['revision'])),flush=True)
if __name__=='__main__':main()
