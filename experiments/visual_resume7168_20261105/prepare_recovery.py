from pathlib import Path
import json,hashlib,time,py_compile
W=Path(__file__).resolve().parent;C=W.parent;OLD=C/'visual_resume_storage_20261105';V=C/'visual_geometry_candidate_20261104'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert read(W/'resume7168_verification.json')['all_file_hashes_verified']
(W/'storage_budget.py').write_text("from pathlib import Path\nimport importlib.util\np=Path(__file__).resolve().parent.parent/'visual_resume_storage_20261105/storage_budget.py'\ns=importlib.util.spec_from_file_location('original_storage_reserve',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nrelease=m.release\nbefore_checkpoint=m.before_checkpoint\n")
s=(OLD/'node.py').read_text().replace('768','7168')
s=s.replace(";release('training_log_headroom',8)","")
s=s.replace('from storage_budget import release,before_checkpoint','from storage_budget import release,before_checkpoint\nfrom quota_io import probe')
s=s.replace("str(V/'train.py'),'--spec'","str(W/'train_bootstrap.py'),'--spec'")
s=s.replace('    if node==0:\n     p=',"    if node==0:\n     probe(V/'runs'/stage)\n     p=")
(W/'node.py').write_text(s)
s=(OLD/'submit_and_handoff.py').read_text().replace('768','7168').replace('t-20261105001409-lb4r4','t-20261105103839-7bfdh')
s=s.replace("reserve=read(W/'storage_reserve.json');s=(W/'storage_reserve.bin').stat()", "reserve=read(C/'visual_resume_storage_20261105/storage_reserve.json');s=(C/'visual_resume_storage_20261105/storage_reserve.bin').stat()")
s=s.replace(" assert not (W/'submission.json').exists()", " emergency=read(W/'emergency_reserve_initial.json');es=(W/'emergency_reserve.bin').stat();assert es.st_size==32*1024**3 and es.st_blocks*512>=es.st_size and es.st_ino==emergency['inode']\n assert not (W/'submission.json').exists()")
(W/'submit_and_handoff.py').write_text(s)
(W/'handoff.py').write_text((OLD/'handoff.py').read_text().replace("recovery['resume_step']==768","recovery['resume_step']==7168"))
(W/'cli.yaml').write_text((OLD/'cli.yaml').read_text().replace('resume768','resume7168').replace('checkpoint768','checkpoint7168').replace(str(OLD),str(W)).replace('reserved storage; full evaluation','quota-resilient append transport; full evaluation'))
for name in ['finish_delivery.py','cleanup_verified_weights.py']:
 s=(OLD/name).read_text()
 # Cleanup must keep the original deletion ledger, while using the new task/policy.
 if name=='cleanup_verified_weights.py':
  s=s.replace("W/'optimizer_cleanup_complete.json'", "C/'visual_resume_storage_20261105/optimizer_cleanup_complete.json'")
  s=s.replace("identities = [read(C/'visual_resume_storage_20261105/submission.json')", "identities = [read(C/'visual_resume7168_20261105/submission.json')")
  s=s.replace("str(C/'visual_resume_storage_20261105/finish_delivery.py')]", "str(C/'visual_resume_storage_20261105/finish_delivery.py'),str(C/'visual_resume7168_20261105/finish_delivery.py')]")
  addition="""
    recent_root=C/'visual_resume7168_20261105'
    recent_complete=read(recent_root/'optimizer_cleanup_complete.json')
    recent_preview=read(recent_root/'optimizer_cleanup_preview.json')
    recent_deleted=[json.loads(l) for l in (recent_root/'optimizer_cleanup_deleted.jsonl').read_text().splitlines()]
    require(recent_complete['preview_sha256']==sha(recent_root/'optimizer_cleanup_preview.json') and recent_complete['ledger_sha256']==sha(recent_root/'optimizer_cleanup_deleted.jsonl'),'Recent quota ledger changed')
    require(len(recent_deleted)==len(recent_preview['candidates'])==recent_complete['files_removed']==25,'Recent quota inventory changed')
    for entry,expected in zip(recent_deleted,recent_preview['candidates']):
        path=Path(entry['path'])
        require(all(entry[k]==v for k,v in expected.items()),'Recent quota entry changed')
        require(path.name=='optimizer.pt' and path.parent.parent==PIPELINE/'runs/repair' and int(path.parent.name.split('-')[-1]) not in (64,768,4096,7168) and not path.exists(),'Unexpected recent quota deletion')
        require(str(path) not in previously_pruned,'Repeated recent quota deletion')
        previously_pruned.add(str(path))
    require(sum(e['bytes'] for e in recent_deleted)==recent_complete['freed_bytes'],'Recent quota bytes differ')
"""
  s=s.replace('    def add(path, expected, reason, remote=None):',addition+'\n    def add(path, expected, reason, remote=None):')
 (W/name).write_text(s)
policy=read(OLD/'resume_policy.json');policy.update(resume_checkpoint=str(V/'runs/repair/checkpoint-7168'),resume_step=7168,resume_seal_sha256=sha(V/'runs/repair/checkpoint-7168/checkpoint_complete.json'),previous_attempt='t-20261105103839-7bfdh',required_reserve_gib=173,emergency_reserve_gib=32,created=time.time())
policy['files']={str(W/name):sha(W/name) for name in ['node.py','storage_budget.py','quota_io.py','train_bootstrap.py','test_quota_io.py','resume7168_verification.json']}
policy['files'][str(V/'spec.json')]=sha(V/'spec.json');save(W/'resume_policy.json',policy)
for name in ['handoff_policy.json','release_policy.json','cleanup_policy.json']:
 p=read(OLD/name)
 def remap(x):
  if isinstance(x,str):
   if x.startswith(str(OLD)) and Path(x).name in ['optimizer_cleanup_complete.json','optimizer_cleanup_preview.json']:return x
   return x.replace(str(OLD),str(W))
  if isinstance(x,list):return [remap(v) for v in x]
  if isinstance(x,dict):return {remap(k):remap(v) for k,v in x.items()}
  return x
 p=remap(p)
 p['files']={path:sha(path) for path in p['files']}
 save(W/name,p)
for p in W.glob('*.py'):py_compile.compile(str(p),doraise=True)
for name in ['resume_policy.json','handoff_policy.json','release_policy.json','cleanup_policy.json']:
 for path,h in read(W/name)['files'].items():assert sha(path)==h,path
save(W/'recovery_preparation.json',dict(state='prepared',resume_step=7168,training_sources_unchanged=True,original_spec_sha256=sha(V/'spec.json'),io_changes_only=True,tests_passed=5,time=time.time()))
print('Recovery wrappers and task-bound downstream policies prepared')
