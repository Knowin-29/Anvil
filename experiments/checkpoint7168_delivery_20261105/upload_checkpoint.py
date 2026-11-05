"""Publish the user-selected full checkpoint, preserving the verified36-key parent."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import fcntl,hashlib,json,runpy,time,os
Z=Path(__file__).resolve().parent;C=Z.parent;P=C/'visual_geometry_candidate_20261104';OUT=Z/'checkpoint_upload';REPO='Knowin-29/Anvil-MOPD-9B'
def read(p):return json.loads(p.read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix('.pending');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def main():
 from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_url
 from huggingface_hub.utils import get_session
 OUT.mkdir(exist_ok=True);lock=(C/'hub_writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'upload_intent.json').exists(),'Reconcile an existing intent before retry'
 token=runpy.run_path(str(C/'verification_20261101/preupload_merged_blobs.py'))['authorized_token']();api=HfApi(token=token);assert api.whoami()['name']=='Knowin-29'
 prior=C/'visual_checkpoint_backup_20261104/repair_4096';previous=read(prior/'verification_receipt.json');manifest=read(prior/'manifest.json')
 assert previous['revision']=='403febab48a34f2c1ac5e22fb1b0fc39897d8989' and sha(prior/'manifest.json')==previous['manifest_sha256'] and len(previous['key_checkpoints'])==36
 assert api.model_info(REPO).sha==previous['revision']
 selected=read(Z/'selection.json');folder=Path(selected['checkpoint']);seal=read(folder/'checkpoint_complete.json');assert sha(folder/'checkpoint_complete.json')==selected['seal_sha256']
 assert seal['global_step']==7168 and len(seal['files'])==39
 added={};relative='visual_repair/checkpoint-7168';prefix='checkpoints/'+relative+'/'
 for name,item in seal['files'].items():
  source=folder/name;assert source.is_file() and not source.is_symlink() and source.stat().st_size==item['bytes'] and sha(source)==item['sha256']
  added[prefix+name]=dict(source=str(source),**item)
 for name,source in [('checkpoint_complete.json',folder/'checkpoint_complete.json'),('training_protocol.json',folder.parent/'protocol.json')]:
  added[prefix+name]=dict(source=str(source),bytes=source.stat().st_size,sha256=sha(source))
 keys=previous['key_checkpoints']+[dict(path=relative,step=7168,adapter_sha256=seal['files']['adapter_model.safetensors']['sha256'],selected_release_checkpoint=True,final_evaluated_checkpoint=False,in_final_optimizer_ancestry=True)]
 card=OUT/'README.md'
 card.write_text('''---
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
- anvil
- mopd
- spatial-reasoning
- embodied-ai
datasets:
- Knowin-29/Anvil-Ben
---

# Anvil-MOPD-9B — Spatial Intelligence

**11 native spatial forms · 5.22M-record SFT ancestry · multi-teacher on-policy distillation**

Anvil brings visual perception, spatial reasoning and action-oriented prediction into one Qwen3.5-9B model. Its training combines specialist knowledge transfer with joint visual-language geometry adaptation.

[Paper](https://arxiv.org/abs/2610.21882) · [Code](https://github.com/Knowin-29/Anvil) · [Dataset](https://huggingface.co/datasets/Knowin-29/Anvil-Ben)

## Selected checkpoint

[**Visual geometry · step 7168**](checkpoints/visual_repair/checkpoint-7168) continues the completed coordinate MOPD model with 7,168 visual-language repair updates. The checkpoint includes the LoRA adapter, optimizer, scheduler and all 32 rank RNG states.

- **4.756M-record spatial training pool**, covering all 11 native task forms.
- **178.25M trainable adapter parameters**, including 5.14M visual adapter parameters.
- **458,752 sample exposures** in the selected visual repair stage.

## Key stages

| Stage | Checkpoint |
|---|---|
| Initial adaptation | [step 2048](checkpoints/ancestor_repair/checkpoint-2048) |
| Balanced MOPD | [step 512](checkpoints/balanced_mopd/checkpoint-512) |
| Format-enhanced MOPD | [step 256](checkpoints/format_mopd/checkpoint-256) |
| Coordinate MOPD | [step 256](checkpoints/coordinate_mopd/checkpoint-256) |
| Visual geometry | [step 7168](checkpoints/visual_repair/checkpoint-7168) |

Load the selected adapter with PEFT on `Qwen/Qwen3.5-9B`. Native DoF inference uses the project's rotation128 prompt and decoding interface. Browse the [training stages](checkpoints/) and [source code](https://github.com/Knowin-29/Anvil) for integration details.
''')
 index=OUT/'checkpoint_index.json';save(index,dict(checkpoints=keys,selected_checkpoint=relative,training_stage='repair',step=7168))
 for name,p in [('README.md',card),('docs/provenance/selected_checkpoint.json',Z/'selection.json'),('docs/provenance/checkpoint_index.json',index)]:added[name]=dict(source=str(p),bytes=p.stat().st_size,sha256=sha(p))
 entries=dict(manifest['files']);entries.update(added)
 combined=dict(schema='anvil-selected-checkpoint-backup-v1',repository=REPO,files=entries,key_checkpoints=keys,parent_revision=previous['revision'],selected_checkpoint=relative)
 save(OUT/'manifest.json',combined);save(OUT/'upload_intent.json',dict(parent_revision=previous['revision'],manifest_sha256=sha(OUT/'manifest.json'),time=time.time()))
 save(OUT/'status.json',dict(stage='uploading',files=len(added),bytes=sum(x['bytes'] for x in added.values()),time=time.time()))
 commit=api.create_commit(repo_id=REPO,repo_type='model',parent_commit=previous['revision'],operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=x['source']) for n,x in added.items()],num_threads=8,commit_message='Publish selected visual geometry checkpoint7168')
 revision=commit.oid;save(OUT/'uploaded.json',dict(revision=revision,time=time.time()))
 info=api.model_info(REPO,revision=revision,files_metadata=True);remote={x.rfilename:x for x in info.siblings};assert set(remote)==set(entries)|{'.gitattributes'}
 verified=dict(previous['files'])
 def verify(name):
  x=remote[name];expected=added[name];assert x.size==expected['bytes'];lfs=x.lfs;digest=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
  if not digest:
   h=hashlib.sha256()
   with get_session().stream('GET',hf_hub_url(REPO,name,revision=revision),headers={'Authorization':'Bearer '+token},follow_redirects=True) as r:
    r.raise_for_status()
    for chunk in r.iter_bytes():h.update(chunk)
   digest=h.hexdigest()
  assert digest==expected['sha256'],name
  return name,dict(bytes=x.size,sha256=digest)
 with ThreadPoolExecutor(max_workers=8) as pool:verified.update(dict(pool.map(verify,added)))
 for name in set(remote)-set(added):
  assert remote[name].size==previous['files'][name]['bytes']
  lfs=remote[name].lfs
  if lfs:assert (lfs.get('sha256') if isinstance(lfs,dict) else lfs.sha256)==previous['files'][name]['sha256']
 assert api.model_info(REPO).sha==revision
 save(OUT/'verification_receipt.json',dict(state='complete_remote_verified_selected_checkpoint',repository=REPO,revision=revision,parent_revision=previous['revision'],files=verified,key_checkpoints=keys,manifest_sha256=sha(OUT/'manifest.json'),selected_checkpoint=relative,completed=time.time(),credentials_modified=False))
 save(OUT/'status.json',dict(stage='remote_verified',revision=revision,checkpoints=37,time=time.time()));print(json.dumps(dict(revision=revision,checkpoints=37)),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:save(OUT/'failure.json',dict(error_type=type(e).__name__,time=time.time()));raise
