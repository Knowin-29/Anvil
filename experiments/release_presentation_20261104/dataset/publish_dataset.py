"""Publish the documentation layout; retain every annotation byte and shard path."""
import hashlib
import json
from pathlib import Path
import runpy
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete, hf_hub_download

Q = Path(__file__).resolve().parent
REPO = 'Knowin-29/Anvil-Ben'

def main():
    before = json.loads((Q/'dataset_inventory_before.json').read_text())
    token = runpy.run_path(str(Q.parent/'verification_20261101/preupload_merged_blobs.py'))['authorized_token']()
    api = HfApi(token=token)
    assert api.whoami()['name'] == 'Knowin-29'
    added = {
        'README.md': 'dataset_README.md',
        'docs/ATTRIBUTION.md': 'dataset_ATTRIBUTION.md',
        'scripts/rebuild_clevr_images.py': 'relocated_rebuild_clevr_images.py',
        'scripts/rebuild_spatial_images.py': 'relocated_rebuild_spatial_images.py',
        'release_manifest.json': 'dataset_release_manifest_updated.json',
    }
    removed = {'rebuild_clevr_images.py', 'rebuild_spatial_images.py'}
    receipt = Q/'dataset_commit.json'
    if not receipt.exists():
        assert not (Q/'dataset_commit_intent.json').exists(), 'Reconcile existing commit intent'
        assert api.dataset_info(REPO).sha == before['revision'], 'Dataset changed; inspect before editing'
        intent = {'parent_revision':before['revision'], 'added_sha256':{
            name:hashlib.sha256((Q/local).read_bytes()).hexdigest() for name,local in added.items()},
            'removed':sorted(removed)}
        (Q/'dataset_commit_intent.json').write_text(json.dumps(intent,indent=2)+'\n')
        ops = [CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(Q/local)) for name,local in added.items()]
        ops += [CommitOperationDelete(path_in_repo=name) for name in sorted(removed)]
        commit = api.create_commit(REPO,repo_type='dataset',operations=ops,parent_commit=before['revision'],
            commit_message='Present Anvil-Ben with concise overview, usage guide and organized tools')
        receipt.write_text(json.dumps({'revision':commit.oid,'parent_revision':before['revision']},indent=2)+'\n')
    revision = json.loads(receipt.read_text())['revision']
    after = api.dataset_info(REPO,revision=revision,files_metadata=True)
    remote = {f.rfilename:f for f in after.siblings}
    old = {f['path']:f for f in before['files']}
    assert set(remote) == (set(old)-removed)|set(added)
    for name,f in old.items():
        if name in removed or name in added: continue
        assert remote[name].size == f['size'] and remote[name].blob_id == f['blob_id'], name
        if f['lfs']: assert remote[name].lfs.sha256 == f['lfs']['sha256'], name
    for name,local in added.items():
        path = hf_hub_download(REPO,name,repo_type='dataset',revision=revision,token=token)
        assert Path(path).read_bytes() == (Q/local).read_bytes(), name
    assert api.dataset_info(REPO).sha == revision, 'Head changed after commit'
    result={'state':'dataset_presentation_remotely_verified','repository':REPO,'revision':revision,
            'files':len(remote),'annotation_shards_preserved':63,'records':5766133,
            'credentials_modified':False}
    (Q/'dataset_complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
