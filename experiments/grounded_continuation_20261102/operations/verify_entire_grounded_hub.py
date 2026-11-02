"""Read-only, pinned-commit reconciliation after the frozen final uploader.

The uploader verifies all sealed release files. This additional check requires
the remote inventory to be exactly the release plus the previously verified
.gitattributes, so unlisted non-weight files cannot escape the final audit.
No login, upload, credential-store access, model loading or GPU work occurs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time

REPOSITORY = 'Knowin-29/Anvil-MOPD-9B'
PIPELINE_SPEC_SHA256 = '47d0f55e3948fc0c77a418373177d18d00bb18b443e986746d6a9d8dc6fe2062'
KEYS = {
    'ancestor_repair/checkpoint-2048',
    'ancestor_mopd/checkpoint-512',
    'ancestor_mopd/checkpoint-1024',
    'ancestor_mopd/checkpoint-1536',
    'repair/checkpoint-2048', 'repair/checkpoint-4096',
    'mopd/checkpoint-256', 'mopd/checkpoint-512', 'mopd/checkpoint-768',
    'focus_repair/checkpoint-4096', 'focus_repair/checkpoint-8192',
    'focus_mopd/checkpoint-128', 'focus_mopd/checkpoint-256', 'focus_mopd/checkpoint-512',
    'balanced_repair/checkpoint-1280', 'balanced_repair/checkpoint-2560',
    'balanced_mopd/checkpoint-128', 'balanced_mopd/checkpoint-256', 'balanced_mopd/checkpoint-512',
    'grounded_repair/checkpoint-4096', 'grounded_repair/checkpoint-8192',
    'grounded_mopd/checkpoint-128', 'grounded_mopd/checkpoint-256',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(manifest, manifest_identity, final, backup):
    require(manifest['schema'] == 'anvil-sealed-final-release-v1'
            and manifest['quality_review_accepted'] is True, 'Unaccepted release')
    require(final['state'] == 'complete_remote_verified', 'Final upload not verified')
    require(backup['state'] == 'complete_remote_verified_intermediate_checkpoints',
            'Key-checkpoint backup not verified')
    require(all(x['repository'] == REPOSITORY for x in (manifest, final, backup)),
            'Different repository')
    require(final['manifest_sha256'] == manifest_identity['sha256'], 'Different final manifest')
    for receipt in (final, backup):
        require(re.fullmatch(r'[0-9a-f]{40}', receipt['revision']) is not None,
                'Unpinned receipt')
    expected = dict(manifest['files'])
    require('release_manifest.json' not in expected, 'Self-referential manifest')
    expected['release_manifest.json'] = manifest_identity
    require(set(final['files']) == set(expected), 'Incomplete final upload receipt')
    for name, item in expected.items():
        path = PurePosixPath(name)
        require(name == str(path) and not path.is_absolute()
                and '..' not in path.parts and name not in {'', '.'}, 'Unsafe file path')
        require(item['bytes'] >= 0 and re.fullmatch(r'[0-9a-f]{64}', item['sha256']),
                'Invalid expected identity')
        require(all(final['files'][name][k] == item[k] for k in ('bytes', 'sha256')),
                'Final receipt differs from manifest: ' + name)
    keys = backup['key_checkpoints']
    require(len(keys) == len(KEYS) and {k['path'] for k in keys} == KEYS,
            'All twenty-three declared checkpoints are required')
    for key in keys:
        prefix = 'checkpoints/' + key['path'] + '/'
        names = {name for name in backup['files'] if name.startswith(prefix)}
        require({prefix + name for name in ('adapter_model.safetensors', 'optimizer.pt',
                'scheduler.pt', 'trainer_state.json', 'checkpoint_complete.json',
                'training_protocol.json')} <= names, 'Incomplete checkpoint payload')
        require({prefix + f'rng_state_{rank}.pth' for rank in range(32)} <= names,
                'Missing checkpoint RNG state')
        for name in names:
            require(name in expected and all(expected[name][k] == backup['files'][name][k]
                    for k in ('bytes', 'sha256')), 'Checkpoint changed after backup: ' + name)
        require(expected[prefix + 'adapter_model.safetensors']['sha256'] == key['adapter_sha256'],
                'Checkpoint adapter identity mismatch')
    extras = set(backup['files']) - set(expected)
    require(extras <= {'.gitattributes'}, 'Unreviewed retained files: ' + repr(sorted(extras)))
    for name in extras:
        expected[name] = backup['files'][name]
    return expected


def bind_final_model(manifest, pipeline, pipeline_sha, evaluation, evaluation_sha):
    require(pipeline_sha == PIPELINE_SPEC_SHA256, 'Different frozen training campaign')
    require(set(pipeline['key_checkpoints']) == KEYS and len(pipeline['key_checkpoints']) == 23,
            'Different declared checkpoint inventory')
    require(set(pipeline['key_checkpoint_sources']) == KEYS, 'Missing checkpoint source mapping')
    require(manifest['final_spec_sha256'] == evaluation_sha, 'Different final evaluation')
    require(manifest['files']['evidence/corrective_pipeline_spec.json']['sha256'] == pipeline_sha
            and manifest['files']['evidence/evaluation_spec.json']['sha256'] == evaluation_sha,
            'Packaged specifications differ')
    expected = Path(pipeline['key_checkpoint_sources']['grounded_mopd/checkpoint-256'])
    require(Path(evaluation['checkpoint']) == expected
            and Path(evaluation['training_run']) == expected.parent,
            'Evaluated checkpoint is not the completed grounded MOPD256')
    adapter = manifest['files']['checkpoints/grounded_mopd/checkpoint-256/adapter_model.safetensors']
    require(manifest['files']['adapter/adapter_model.safetensors']['sha256'] == adapter['sha256'],
            'Final standalone adapter differs from the declared last checkpoint')
    for name, digest in evaluation['model_files'].items():
        if name != 'README.md':
            require(manifest['files'][name]['sha256'] == digest, 'Merged model differs: ' + name)


def field(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'final-receipt', 'backup-receipt', 'pipeline-spec', 'evaluation-spec', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new output receipt path')
    manifest, final, backup = [json.loads(p.read_text()) for p in
                              (args.manifest, args.final_receipt, args.backup_receipt)]
    expected = inventory(manifest, dict(bytes=args.manifest.stat().st_size,
                         sha256=sha(args.manifest)), final, backup)
    pipeline = json.loads(args.pipeline_spec.read_text())
    evaluation = json.loads(args.evaluation_spec.read_text())
    bind_final_model(manifest, pipeline, sha(args.pipeline_spec), evaluation, sha(args.evaluation_spec))
    token = os.environ.get('ANVIL_HF_TOKEN')
    require(bool(token), 'Provide transient ANVIL_HF_TOKEN')
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import get_session
    api = HfApi(token=token)
    require(api.whoami(token=token)['name'] == 'Knowin-29', 'Wrong authorized account')
    revision = final['revision']
    info = api.model_info(REPOSITORY, revision=revision, files_metadata=True, token=token)
    require(info.sha == revision, 'Different remote commit')
    remote = {entry.rfilename: entry for entry in info.siblings}
    require(set(remote) == set(expected), 'Remote inventory differs from the complete release')
    verified = {}
    for name, item in sorted(expected.items()):
        meta = remote[name]
        require(meta.size == item['bytes'], 'Remote size mismatch: ' + name)
        digest = field(field(meta, 'lfs'), 'sha256')
        method = 'hub_lfs_sha256'
        if not digest:
            method = 'downloaded_sha256'
            url = hf_hub_url(REPOSITORY, name, revision=revision, repo_type='model')
            with get_session().stream('GET', url, headers={'Authorization': 'Bearer ' + token},
                                      follow_redirects=True) as response:
                response.raise_for_status()
                hasher = hashlib.sha256()
                for chunk in response.iter_bytes():
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        require(digest == item['sha256'], 'Remote content mismatch: ' + name)
        verified[name] = dict(bytes=item['bytes'], sha256=digest, method=method)
    require(api.model_info(REPOSITORY, token=token).sha == revision,
            'Repository head changed during final verification')
    receipt = dict(state='complete_entire_remote_release_verified', repository=REPOSITORY,
                   revision=revision, remote_file_count=len(verified), files=verified,
                   key_checkpoints=sorted(KEYS), completed=time.time(),
                   credential_store_modified=False, remote_writes_performed=False,
                   evidence_sha256={str(p): sha(p) for p in
                       (args.manifest, args.final_receipt, args.backup_receipt, args.pipeline_spec, args.evaluation_spec, Path(__file__))})
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
