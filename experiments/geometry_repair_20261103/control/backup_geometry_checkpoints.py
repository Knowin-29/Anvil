"""Verify and back up the four declared new checkpoints as training progresses.

Never submits GPU work or accepts/publishes a final merged model. An ambiguous
Hub mutation stops for explicit reconciliation instead of retrying. Final
release upload must wait until this worker is terminal.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

C=Path(__file__).resolve().parent
P=C/'geometry_repair_candidate_20261103'
OUT=C/'geometry_checkpoint_backup_20261103'
ANCESTORS=C/'format_checkpoint_backup_20261103/mopd_256'
TRAIN=str(C.parent/'.venv-train/bin/python')
VERIFIER=C/'verification_20261101/verify_geometry_checkpoint.py'
SPEC_SHA='eabf085162304648de505cd364bf351e61b16bb6a998fe6b3e19c18b4607dacc'
REPO='Knowin-29/Anvil-MOPD-9B'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def status(stage,**values):
    save(OUT/'status.json',dict(stage=stage,updated=time.time(),pid=os.getpid(),
         final_model_uploaded=False,final_quality_accepted=False,**values))


def main():
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_url
    from huggingface_hub.utils import get_session
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    hub_lock=(C/'hub_writer.lock').open('a');fcntl.flock(hub_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    own_sha=sha(Path(__file__));verifier_sha=sha(VERIFIER)
    assert sha(P/'spec.json')==SPEC_SHA
    spec=read(P/'spec.json');deadline=datetime.fromisoformat(spec['deadline_utc']).timestamp()
    assert spec['key_checkpoints'][27:]==['geometry_repair/checkpoint-2048','geometry_repair/checkpoint-4096',
        'geometry_mopd/checkpoint-128','geometry_mopd/checkpoint-256']
    assert set(spec['key_checkpoint_sources'])==set(spec['key_checkpoints'])
    token=os.environ['ANVIL_HF_TOKEN'];api=HfApi(token=token)
    assert api.whoami(token=token)['name']=='Knowin-29'
    previous=read(ANCESTORS/'verification_receipt.json');manifest=read(ANCESTORS/'manifest.json')
    assert previous['state']=='complete_remote_verified_intermediate_checkpoints'
    assert previous['manifest_sha256']==sha(ANCESTORS/'manifest.json')
    entries=dict(manifest['files']);key_records=list(previous['key_checkpoints'])
    assert [item['path'] for item in key_records]==spec['key_checkpoints'][:27]
    assert manifest['key_checkpoints']==key_records
    assert previous['revision']=='e5b8da0e1c9c5346d28359405b98c21713b81511'
    for relative in spec['key_checkpoints'][27:]:
        remote_stage,name=relative.split('/');stage=remote_stage.removeprefix('geometry_')
        step=int(name.split('-')[-1]);folder=Path(spec['key_checkpoint_sources'][relative])
        assert stage in ('repair','mopd') and folder==P/'runs'/stage/name
        batch=OUT/(stage+'_'+str(step));batch.mkdir(exist_ok=True)
        if (batch/'verification_receipt.json').exists():
            completed=read(batch/'verification_receipt.json');local=read(batch/'manifest.json')
            assert completed['manifest_sha256']==sha(batch/'manifest.json')
            assert completed['parent_revision']==previous['revision']
            previous=completed;entries=dict(local['files']);key_records=list(local['key_checkpoints'])
            continue
        assert not (batch/'upload_intent.json').exists(),'Uncertain prior upload; inspect remote before recovery'
        while time.time()<deadline:
            if (folder/'checkpoint_complete.json').exists() and (
                step<spec['stages'][stage]['steps'] or (folder.parent/'complete.json').exists()):
                break
            status('waiting_for_sealed_key_checkpoint',checkpoint=relative,
                   verified_checkpoints=len(key_records),last_verified_revision=previous['revision'])
            time.sleep(45)
        else:
            raise TimeoutError('Key checkpoint was not available before the campaign deadline')
        assert sha(Path(__file__))==own_sha and sha(VERIFIER)==verifier_sha and sha(P/'spec.json')==SPEC_SHA
        integrity=P/(stage+'_checkpoint'+str(step)+'_verification.json')
        if not integrity.exists():
            status('verifying_checkpoint_state',checkpoint=relative,verified_checkpoints=len(key_records))
            with (batch/'integrity.log').open('a') as stream:
                subprocess.run([TRAIN,str(VERIFIER),'--stage',stage,'--step',str(step),'--output',str(integrity)],
                               stdout=stream,stderr=subprocess.STDOUT,check=True)
        audit=read(integrity)
        assert audit['state']=='geometry_checkpoint_integrity_verified' and audit['stage']==stage and audit['step']==step
        assert Path(audit['checkpoint']).resolve()==folder.resolve() and audit['spec_sha256']==SPEC_SHA
        assert audit['verification_script_sha256']==verifier_sha and audit['all_file_hashes_verified'] and audit['all32_rng_files_loadable']
        seal=read(folder/'checkpoint_complete.json')
        assert seal['world_size']==32 and seal['global_step']==step and len(seal['files'])==38
        added={}
        for filename,item in seal['files'].items():
            source=folder/filename
            assert source.is_file() and not source.is_symlink() and source.stat().st_size==item['bytes'] and sha(source)==item['sha256']
            added['checkpoints/'+relative+'/'+filename]=dict(source=str(source),**item)
        for filename,source in [('checkpoint_complete.json',folder/'checkpoint_complete.json'),
                                ('training_protocol.json',folder.parent/'protocol.json')]:
            added['checkpoints/'+relative+'/'+filename]=dict(source=str(source),bytes=source.stat().st_size,sha256=sha(source))
        key_records.append(dict(path=relative,step=step,adapter_sha256=seal['files']['adapter_model.safetensors']['sha256'],final_evaluated_checkpoint=False))
        for record in key_records:
            record['in_final_optimizer_ancestry']=record['path'] not in spec['retained_nonancestor_checkpoints']
        evidence=batch/'key_checkpoints.json'
        save(evidence,dict(checkpoints=key_records,note='Intermediate training backups only. Full adapter, optimizer, scheduler and all rank RNG states. Final model quality and merged release have not been accepted.'))
        card=batch/'README.md'
        listing='\n'.join('- `'+record['path']+'`' for record in key_records)
        card.write_text('''---
base_model: Qwen/Qwen3.5-9B
tags:
- anvil
- mopd
- intermediate-checkpoints
---

# Anvil MOPD 9B — training and validation in progress

This repository contains **complete intermediate training checkpoints**, not an accepted final model or a standalone merged inference release. The checkpoints include their LoRA adapters, optimizer, scheduler, all 32 rank RNG files, sealed checksum inventories and training protocols.

Currently backed-up checkpoints:

'''+listing+'''

Identities are listed in `evidence/key_checkpoints.json`. The earlier MOPD1536 parent scored 69.3671% and the subsequent MOPD768 continuation scored 70.1171% on the seven-benchmark macro versus the paper target of 74.27%. Neither was accepted as the final release. The rejected focus branch scored67.6957% and is retained solely as a nonancestor experiment backup. The completed balanced MOPD512 parent scored70.442857% with retained general ability but material localization and placement gaps; it was not accepted as the final release. The format continuation branches from balanced MOPD512 with fresh stage optimizers. The rejected grounded branch scored69.797143 and its four checkpoints are retained as nonancestor backups alongside the five rejected-focus keys. Point output-contract variants use only previously admitted training annotations, with exact coordinate conversions and preserved source UIDs. Rejected branches' exposures are excluded from the new model ancestry. New checkpoint availability establishes saved training progress, not improved benchmark performance.

The completed format MOPD256 parent scored73.481429 on the public macro, but native maskIoU.158590, affordanceIoU.126352 and DoFrotation111.553995deg remain weak. The geometry continuation inherits this parent and increases previously admitted geometry training coverage while retaining public/general replay. These intermediate checkpoints do not establish improvement.

The final merged model will be published only after complete final evaluation and evidence-based quality review. This intermediate repository state does not claim reproduction of the paper results.
''')
        for filename,source in [('README.md',card),('evidence/key_checkpoints.json',evidence)]:
            added[filename]=dict(source=str(source),bytes=source.stat().st_size,sha256=sha(source))
        entries.update(added)
        combined=dict(schema='anvil-intermediate-geometry-checkpoint-backup-v1',repository=REPO,
            files=entries,key_checkpoints=key_records,parent_revision=previous['revision'],
            integrity_receipt=dict(path=str(integrity),sha256=sha(integrity)),spec_sha256=SPEC_SHA,
            final_model_uploaded=False,final_quality_accepted=False)
        save(batch/'manifest.json',combined)
        current=api.model_info(REPO,token=token)
        assert current.sha==previous['revision'],'Remote changed outside this sequential backup chain'
        save(batch/'upload_intent.json',dict(parent_commit=previous['revision'],manifest_sha256=sha(batch/'manifest.json'),time=time.time()))
        status('uploading_key_checkpoint',checkpoint=relative,verified_checkpoints=len(key_records)-1,
               previous_revision=previous['revision'],new_payload_bytes=sum(v['bytes'] for v in added.values()))
        operations=[CommitOperationAdd(path_in_repo=filename,path_or_fileobj=entry['source']) for filename,entry in added.items()]
        commit=api.create_commit(repo_id=REPO,repo_type='model',token=token,parent_commit=previous['revision'],
            operations=operations,num_threads=8,commit_message='Back up '+relative+'; final model quality review pending')
        revision=commit.oid
        save(batch/'uploaded.json',dict(revision=revision,parent_revision=previous['revision'],time=time.time()))
        status('verifying_pinned_remote_files',checkpoint=relative,revision=revision,verified_checkpoints=len(key_records)-1)
        info=api.model_info(REPO,revision=revision,files_metadata=True,token=token)
        assert info.sha==revision
        remote={item.rfilename:item for item in info.siblings}
        assert set(remote)==set(entries)|{'.gitattributes'}
        expected={**{name:dict(bytes=item['bytes'],sha256=item['sha256']) for name,item in entries.items()},
                  '.gitattributes':previous['files']['.gitattributes']}

        def verify(filename):
            meta=remote[filename];item=expected[filename]
            assert meta.size==item['bytes'],'Remote size mismatch: '+filename
            lfs=meta.lfs
            digest=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
            method='hub_lfs_sha256'
            if not digest:
                h=hashlib.sha256();url=hf_hub_url(REPO,filename,revision=revision,repo_type='model')
                with get_session().stream('GET',url,headers={'Authorization':'Bearer '+token},follow_redirects=True) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():h.update(chunk)
                digest=h.hexdigest();method='downloaded_sha256'
            assert digest==item['sha256'],'Remote checksum mismatch: '+filename
            return filename,dict(bytes=meta.size,sha256=digest,method=method)

        with ThreadPoolExecutor(max_workers=8) as pool:verified=dict(pool.map(verify,sorted(remote)))
        result=dict(state='complete_remote_verified_intermediate_checkpoints',repository=REPO,revision=revision,
                    parent_revision=previous['revision'],files=verified,key_checkpoints=key_records,
                    manifest_sha256=sha(batch/'manifest.json'),completed=time.time(),
                    final_model_uploaded=False,final_quality_accepted=False,credential_store_modified=False)
        save(batch/'verification_receipt.json',result);previous=result
        save(C/'verification_20261101/hf_inventory_latest.json',dict(repository=REPO,revision=revision,files=sorted(remote),
             weights_present=True,final_model_uploaded=False,intermediate_checkpoints_verified=len(key_records),
             checked_at=time.time(),credentials_modified=False,verification_receipt=str(batch/'verification_receipt.json')))
        status('key_checkpoint_remotely_verified',checkpoint=relative,verified_checkpoints=len(key_records),revision=revision)
        print(json.dumps(dict(checkpoint=relative,revision=revision,verified_checkpoints=len(key_records),verified_files=len(verified))),flush=True)
    assert len(key_records)==len(spec['key_checkpoints'])==31
    status('all_declared_checkpoints_remotely_verified',verified_checkpoints=31,revision=previous['revision'],verified_files=len(previous['files']))
    save(OUT/'complete.json',dict(state='all_declared_checkpoints_remotely_verified',revision=previous['revision'],
         key_checkpoints=key_records,completed=time.time(),final_model_uploaded=False,final_quality_accepted=False))


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        save(OUT/'failure.json',dict(error_type=type(exc).__name__,time=time.time(),pid=os.getpid()))
        print('Balanced checkpoint backup stopped: '+type(exc).__name__,flush=True)
        raise SystemExit(1)
