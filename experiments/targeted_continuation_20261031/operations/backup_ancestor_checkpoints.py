"""Back up declared full ancestor checkpoints, without claiming a final release.

Only explicit transient ANVIL_HF_TOKEN is used. Server credentials are never
read or changed. Uploaded paths match the already frozen final package layout.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

C = Path(__file__).resolve().parent
OUT = C / 'ancestor_checkpoint_backup_20261031'
REPO = 'Knowin-29/Anvil-MOPD-9B'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, value):
    target = OUT / name
    temp = target.with_suffix(target.suffix + '.pending')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(target)


def status(stage, **fields):
    save('status.json', dict(stage=stage, updated=time.time(), pid=os.getpid(),
                            final_model_uploaded=False, final_quality_accepted=False, **fields))


def main():
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_url
    from huggingface_hub.utils import get_session
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'upload_intent.json').exists(), 'Existing intent needs explicit remote reconciliation'
    token = os.environ['ANVIL_HF_TOKEN']
    api = HfApi(token=token)
    assert api.whoami(token=token)['name'] == 'Knowin-29'
    previous = api.model_info(REPO, token=token)
    assert {f.rfilename for f in previous.siblings} == {'.gitattributes', 'README.md'}, 'Remote already changed'
    spec = read(C/'targeted_repair_20261031/spec.json')
    selected = spec['key_checkpoints'][:4]
    assert selected == ['ancestor_repair/checkpoint-2048', 'ancestor_mopd/checkpoint-512',
                        'ancestor_mopd/checkpoint-1024', 'ancestor_mopd/checkpoint-1536']
    files = {}
    key_records = []
    status('verifying_local_ancestor_payloads')
    for relative in selected:
        folder = C/'targeted_repair_20261031/runs'/relative
        seal = read(folder/'checkpoint_complete.json')
        assert seal['world_size'] == 32
        assert len(seal['files']) == 38
        assert seal['global_step'] == int(folder.name.split('-')[-1])
        assert {p.name for p in folder.iterdir()} == set(seal['files']) | {'checkpoint_complete.json'}
        for name, item in seal['files'].items():
            source = folder/name
            assert source.is_file() and not source.is_symlink()
            assert source.stat().st_size == item['bytes'] and sha(source) == item['sha256'], 'Changed checkpoint payload'
            files['checkpoints/'+relative+'/'+name] = dict(source=str(source), **item)
        for remote_name, source in [('checkpoint_complete.json', folder/'checkpoint_complete.json'),
                                    ('training_protocol.json', folder.parent/'protocol.json')]:
            files['checkpoints/'+relative+'/'+remote_name] = dict(source=str(source), bytes=source.stat().st_size, sha256=sha(source))
        key_records.append(dict(path=relative, step=seal['global_step'],
                                adapter_sha256=seal['files']['adapter_model.safetensors']['sha256'],
                                final_evaluated_checkpoint=False))
    evidence = OUT/'key_checkpoints.json'
    evidence.write_text(json.dumps(dict(checkpoints=key_records,
        note='Intermediate ancestor backups only. Complete adapters, optimizer, scheduler and all rank RNG files. Final corrective MOPD training, validation, acceptance and merged-model release remain incomplete.'), indent=2)+'\n')
    card = OUT/'README.md'
    card.write_text('''---
base_model: Qwen/Qwen3.5-9B
tags:
- anvil
- mopd
- intermediate-checkpoints
---

# Anvil MOPD 9B — training in progress

This repository currently contains **full intermediate ancestor checkpoints only**, not an accepted final model or a standalone merged inference release.

The retained corrective checkpoint 2048 and parent MOPD checkpoints 512/1024/1536 include their LoRA adapters, optimizer, scheduler, all 32 rank RNG files, sealed checksum inventories, and training protocols. Their identities are listed in `evidence/key_checkpoints.json`. The parent completed 1536 MOPD updates; its seven-benchmark macro was 69.3671%, compared with the paper target of 74.27%. These backups do not establish that the paper results have been reproduced.

Further corrective training inherits the parent checkpoint. The accepted final merged model and additional key checkpoints will be uploaded only after complete final validation and quality review. Do not treat this intermediate repository state as the finished release.
''')
    for name, source in [('README.md', card), ('evidence/key_checkpoints.json', evidence)]:
        files[name] = dict(source=str(source), bytes=source.stat().st_size, sha256=sha(source))
    manifest = dict(schema='anvil-intermediate-ancestor-backup-v1', repository=REPO, files=files,
                    key_checkpoints=key_records, previous_revision=previous.sha,
                    targeted_spec_sha256=sha(C/'targeted_repair_20261031/spec.json'),
                    final_model_uploaded=False, final_quality_accepted=False, prepared=time.time())
    save('manifest.json', manifest)
    save('upload_intent.json', dict(parent_commit=previous.sha, manifest_sha256=sha(OUT/'manifest.json'), time=time.time()))
    status('uploading_ancestor_checkpoints', files=len(files), bytes=sum(f['bytes'] for f in files.values()))
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=item['source']) for name,item in files.items()]
    commit = api.create_commit(repo_id=REPO, repo_type='model', token=token, parent_commit=previous.sha,
        operations=operations, num_threads=8, commit_message='Back up full ancestor checkpoints; final model validation still pending')
    revision = commit.oid
    save('uploaded.json', dict(revision=revision, repository=REPO, time=time.time(), final_model_uploaded=False))
    status('verifying_pinned_remote_checkpoint_files', revision=revision)
    info = api.model_info(REPO, revision=revision, files_metadata=True, token=token)
    assert info.sha == revision
    remote = {f.rfilename:f for f in info.siblings}
    assert set(remote) == set(files) | {'.gitattributes'}, 'Unexpected remote contents'

    def verify(name):
        meta = remote[name]
        if name in files:
            assert meta.size == files[name]['bytes'], 'Remote size differs: '+name
        lfs = meta.lfs
        digest = (lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
        method = 'hub_lfs_sha256'
        if not digest:
            url = hf_hub_url(REPO, name, revision=revision, repo_type='model')
            h = hashlib.sha256()
            with get_session().stream('GET',url,headers={'Authorization':'Bearer '+token},follow_redirects=True) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    h.update(chunk)
            digest=h.hexdigest();method='downloaded_sha256'
        if name in files:
            assert digest == files[name]['sha256'], 'Remote digest differs: '+name
        return name,dict(sha256=digest,bytes=meta.size,method=method)

    with ThreadPoolExecutor(max_workers=8) as pool:
        verified = dict(pool.map(verify, sorted(remote)))
    receipt = dict(state='complete_remote_verified_ancestor_checkpoints_only', repository=REPO,
                   revision=revision,url='https://huggingface.co/'+REPO+'/tree/'+revision,
                   files=verified,manifest_sha256=sha(OUT/'manifest.json'),key_checkpoints=key_records,
                   credential_store_modified=False,final_model_uploaded=False,final_quality_accepted=False,completed=time.time())
    save('verification_receipt.json',receipt)
    status('complete_remote_verified_ancestor_checkpoints_only',revision=revision,checkpoint_count=4,verified_files=len(verified))
    inventory=dict(repository=REPO,revision=revision,files=sorted(remote),weights_present=True,
                   final_model_uploaded=False,ancestor_checkpoints_verified=4,checked_at=time.time(),credentials_modified=False,
                   verification_receipt=str(OUT/'verification_receipt.json'))
    dest=C/'verification_20261031/hf_inventory_latest.json'
    temp=dest.with_suffix('.backup-pending');temp.write_text(json.dumps(inventory,indent=2)+'\n');temp.replace(dest)
    print(json.dumps(dict(state=receipt['state'],revision=revision,verified_files=len(verified),key_checkpoints=4)),flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        OUT.mkdir(exist_ok=True)
        # Keep failure diagnostics credential-free. Remote reconciliation is
        # required before retrying any ambiguous create_commit result.
        save('failure.json',dict(error_type=type(exc).__name__,time=time.time(),pid=os.getpid()))
        print('Ancestor checkpoint backup failed: '+type(exc).__name__,flush=True)
        raise SystemExit(1)
