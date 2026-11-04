"""Upload an explicitly sealed model package and verify every remote file.

Token is read only from ANVIL_HF_TOKEN, passed explicitly to Hub calls, and never
saved. No login, logout, credential-store changes, git credential helpers, or
implicit server HF tokens are used. Re-running verification pins a commit SHA.
"""
import argparse
import hashlib
import os
from pathlib import Path,PurePosixPath
import time
from common import read,require,save,sha


def safe_relative(name):
    path=PurePosixPath(name)
    require(bool(name) and not path.is_absolute() and all(part not in {'..','.'} for part in path.parts),
            'Unsafe package path')
    return str(path)


def verify_local(root):
    root=Path(root).resolve();manifest=read(root/'release_manifest.json')
    require(manifest['schema']=='anvil-sealed-final-release-v1' and manifest['quality_review_accepted'] is True,
            'Final release must bind an accepted, completed final-model review')
    actual=set()
    for p in root.rglob('*'):
        require(not p.is_symlink(),'Package must not contain symlinks')
        if p.is_file():actual.add(p.relative_to(root).as_posix())
    require(actual==set(manifest['files'])|{'release_manifest.json'},'Unsealed or missing package files')
    for name,item in manifest['files'].items():
        name=safe_relative(name);p=root/name
        require(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Package file changed: '+name)
    require(manifest['repository']=='Knowin-29/Anvil-MOPD-9B','Unexpected model destination')
    index=read(root/'model.safetensors.index.json')
    require(set(index['weight_map'].values()) <= set(manifest['files']),'Missing merged weight shard')
    require('adapter/adapter_model.safetensors' in manifest['files'],'Missing exact MOPD adapter')
    return manifest


def field(item,key,default=None):
    return item.get(key,default) if isinstance(item,dict) else getattr(item,key,default)


def remote_digest(api,repo,revision,name,metadata,token):
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import get_session
    lfs=field(metadata,'lfs')
    expected=field(lfs,'sha256') if lfs else None
    if expected:
        return expected,'hub_lfs_sha256'
    # Git blob IDs are not SHA256(content). Download small non-LFS files at the
    # exact commit and hash their bytes; never compare SHA256 to a Git SHA1.
    url=hf_hub_url(repo_id=repo,filename=name,revision=revision,repo_type='model')
    with get_session().stream('GET',url,headers={'Authorization':'Bearer '+token},follow_redirects=True) as response:
        response.raise_for_status();h=hashlib.sha256()
        for chunk in response.iter_bytes():h.update(chunk)
    return h.hexdigest(),'downloaded_sha256'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True)
    ap.add_argument('--receipt',type=Path,required=True)
    ap.add_argument('--verify-revision',help='Only verify this existing exact Hub commit')
    args=ap.parse_args();manifest=verify_local(args.package)
    token=os.environ.get('ANVIL_HF_TOKEN')
    require(bool(token),'Set transient ANVIL_HF_TOKEN; saved server credentials are never used')
    from huggingface_hub import HfApi
    api=HfApi(token=token);who=api.whoami(token=token)
    require(who['name']=='Knowin-29','Token does not identify the authorized Knowin-29 account')
    repo=manifest['repository']
    if args.verify_revision:
        revision=args.verify_revision
        require(len(revision)==40 and all(c in '0123456789abcdef' for c in revision),'Pin a full commit SHA')
    else:
        previous=api.model_info(repo,token=token).sha
        commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=str(args.package),
            token=token,parent_commit=previous,
            allow_patterns=[*manifest['files'],'release_manifest.json'],
            commit_message='Upload validated final MOPD model, adapter, decoder and evidence')
        revision=commit.oid
        save(args.receipt.with_suffix('.uploaded.json'),dict(state='uploaded_pending_remote_verification',
            repository=repo,revision=revision,manifest_sha256=sha(args.package/'release_manifest.json'),
            credential_store_modified=False,time=time.time()))
    info=api.model_info(repo,revision=revision,files_metadata=True,token=token)
    require(info.sha==revision,'Remote revision changed')
    siblings={s.rfilename:s for s in info.siblings};files=dict(manifest['files'])
    file=args.package/'release_manifest.json';files[file.name]=dict(bytes=file.stat().st_size,sha256=sha(file))
    verified={}
    for name,item in sorted(files.items()):
        require(name in siblings,'Missing remote file: '+name)
        meta=siblings[name];require(meta.size==item['bytes'],'Remote file size mismatch: '+name)
        digest,method=remote_digest(api,repo,revision,name,meta,token)
        require(digest==item['sha256'],'Remote SHA256 mismatch: '+name)
        verified[name]=dict(sha256=digest,bytes=item['bytes'],method=method)
        print('verified',name,flush=True)
    unexpected_weights=[name for name in siblings if name.endswith(('.safetensors','.bin','.pt','.pth','.index.json')) and name not in files]
    require(not unexpected_weights,'Unrelated remote weight files require explicit review: '+repr(unexpected_weights))
    save(args.receipt,dict(state='complete_remote_verified',repository=repo,revision=revision,
        url=f'https://huggingface.co/{repo}/tree/{revision}',files=verified,
        manifest_sha256=sha(file),credential_store_modified=False,completed=time.time()))
    print(f'https://huggingface.co/{repo}/tree/{revision}')


if __name__=='__main__':main()
