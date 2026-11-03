"""Stage exact final-model LFS blobs without creating a Hub commit or release."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
import sys
import threading
import time

os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import logging

logging.set_verbosity_error()
ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
PIPELINE = CAMPAIGN / 'geometry_repair_candidate_20261103'
PIPELINE_SHA256 = 'eabf085162304648de505cd364bf351e61b16bb6a998fe6b3e19c18b4607dacc'
SPEC = CAMPAIGN / 'delivery_runs/geometry_repair_candidate_20261103/evaluation_spec.json'
BACKUP_ROOT = CAMPAIGN / 'geometry_checkpoint_backup_20261103'
BACKUP_RECEIPT = BACKUP_ROOT / 'mopd_256/verification_receipt.json'
REPOSITORY = 'Knowin-29/Anvil-MOPD-9B'
OUTPUT = ROOT / 'geometry_merged_blob_preupload'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_identity(spec, pipeline, backup):
    expected = PIPELINE/'runs/mopd/checkpoint-256'
    require(spec['schema'] == 'anvil-final-evaluation-spec-v1'
            and Path(spec['training_run']) == expected.parent
            and Path(spec['checkpoint']) == expected
            and Path(spec['model']) == SPEC.parent/'merged_model', 'Different final candidate')
    require(spec['native_records'] == 580000 and spec['general_records'] == 10105,
            'Incomplete final evaluation scope')
    require(backup['state'] == 'complete_remote_verified_intermediate_checkpoints'
            and backup['repository'] == REPOSITORY, 'Unverified final checkpoint backups')
    require(len(backup['key_checkpoints']) == len(pipeline['key_checkpoints']) == 31
            and {x['path'] for x in backup['key_checkpoints']} == set(pipeline['key_checkpoints']),
            'All thirty-one checkpoints must be backed up before merged preupload')
    names = [name for name in spec['model_files'] if name.endswith('.safetensors')]
    require(len(names) == 4 and all(Path(name).name == name for name in names),
            'Expected exactly four safe merged-shard names')
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    require(sha(PIPELINE/'spec.json') == PIPELINE_SHA256, 'Different frozen campaign')
    pipeline = json.loads((PIPELINE/'spec.json').read_text())
    if args.inspect:
        print(json.dumps(dict(state='prepared', pipeline_sha256=PIPELINE_SHA256,
              expected_checkpoint=str(PIPELINE/'runs/mopd/checkpoint-256'),
              required_checkpoint_backups=len(pipeline['key_checkpoints']),
              no_gpu_work=True, no_hub_calls=True, no_commit_operations=True)))
        return
    OUTPUT.mkdir(exist_ok=True)
    lease = (OUTPUT/'worker.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (OUTPUT/'complete.json').exists(), 'Preupload already completed')
    own_sha = sha(__file__)
    deadline = datetime.fromisoformat(pipeline['deadline_utc']).timestamp()
    hub_lease = (CAMPAIGN/'hub_writer.lock').open('a')
    while time.time() < deadline:
        require(sha(__file__) == own_sha and sha(PIPELINE/'spec.json') == PIPELINE_SHA256,
                'Preupload source or frozen campaign changed')
        failures = list(PIPELINE.glob('node*_failure.json'))
        failures += [p for p in [BACKUP_ROOT/'failure.json', SPEC.parent/'failure.json'] if p.exists()]
        require(not failures, 'Training, backup or delivery dependency requires review')
        ready = SPEC.exists() and BACKUP_RECEIPT.exists() and (BACKUP_ROOT/'complete.json').exists()
        if ready:
            try:
                fcntl.flock(hub_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
        save(OUTPUT/'status.json', dict(stage='waiting_exact_model_and_all31_checkpoint_backups',
             pid=os.getpid(), heartbeat_at=time.time(), spec_ready=SPEC.exists(),
             all31_backup_complete=(BACKUP_ROOT/'complete.json').exists(),
             final_model_published=False, server_credentials_modified=False))
        time.sleep(45)
    else:
        raise TimeoutError('Candidate was unavailable before the campaign deadline')
    sys.path.insert(0, str(CAMPAIGN/'final_delivery_geometry_v1'))
    from common import load_spec
    spec_digest = sha(SPEC)
    spec = load_spec(SPEC, verify_model=False)
    backup = json.loads(BACKUP_RECEIPT.read_text())
    names = validate_identity(spec, pipeline, backup)
    manifest_path = BACKUP_RECEIPT.parent/'manifest.json'
    require(backup['manifest_sha256'] == sha(manifest_path), 'Changed checkpoint backup manifest')
    complete = json.loads((PIPELINE/'runs/mopd/complete.json').read_text())
    require(backup['files']['checkpoints/format_mopd/checkpoint-256/adapter_model.safetensors']['sha256']
            == complete['adapter_sha256'], 'Different final checkpoint in Hub backup')
    token = os.environ.get('ANVIL_HF_TOKEN')
    require(bool(token), 'Provide the authorized transient credential')
    api = HfApi(token=token)
    require(api.whoami()['name'] == 'Knowin-29', 'Wrong authorized account')
    before = api.model_info(REPOSITORY)
    inventory = sorted(x.rfilename for x in before.siblings)
    require(before.sha == backup['revision'] and set(inventory) == set(backup['files']),
            'Remote differs from the thirty-one-checkpoint verified backup')
    state = dict(schema='anvil-uncommitted-format-merged-blob-preupload-v1', repository=REPOSITORY,
                 final_spec_sha256=spec_digest, source_model=spec['model'],
                 pid=os.getpid(), started=time.time(), stage='preuploading',
                 pipeline_spec_sha256=PIPELINE_SHA256, checkpoint=spec['checkpoint'],
                 required_backups_verified=27, backup_receipt_sha256=sha(BACKUP_RECEIPT),
                 revision_before=before.sha, files={}, create_commit_called=False,
                 final_model_published=False, final_quality_accepted=False,
                 server_credentials_modified=False)
    lock = threading.Lock()
    done = threading.Event()

    def heartbeat():
        while not done.is_set():
            with lock:
                save(OUTPUT / 'status.json', {**state, 'heartbeat_at': time.time()})
            done.wait(15)

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()

    def upload(name):
        path = Path(spec['model']) / name
        original = path.stat()
        operation = CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
        digest = operation.upload_info.sha256.hex()
        require(digest == spec['model_files'][name], 'Merged model changed: ' + name)
        require(operation.upload_info.size == original.st_size, 'Shard size changed')
        api.preupload_lfs_files(REPOSITORY, additions=[operation], repo_type='model',
                               revision='main', create_pr=False, num_threads=1)
        require(operation._is_uploaded and operation._upload_mode == 'lfs'
                and not operation._should_ignore, 'Shard was not preuploaded')
        after = path.stat()
        require((after.st_size, after.st_mtime_ns, after.st_ino) ==
                (original.st_size, original.st_mtime_ns, original.st_ino), 'Local shard changed during upload')
        item = dict(bytes=original.st_size, sha256=digest, completed=time.time(),
                    sdk_preupload_completed=True, committed_to_repository=False)
        with lock:
            state['files'][name] = item
            save(OUTPUT / 'status.json', {**state, 'heartbeat_at': time.time()})
        print(json.dumps({'preuploaded': name, 'bytes': original.st_size}), flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for future in concurrent.futures.as_completed([pool.submit(upload, name) for name in names]):
                future.result()
        require(sha(SPEC) == spec_digest, 'Final evaluation specification changed')
        after = api.model_info(REPOSITORY)
        require(after.sha == before.sha and sorted(x.rfilename for x in after.siblings) == inventory,
                'Repository changed during preupload; no release is asserted')
        with lock:
            state.update(stage='uncommitted_merged_blobs_preuploaded', completed=time.time(),
                         revision_after=after.sha, remote_inventory_unchanged=True,
                         total_bytes=sum(x['bytes'] for x in state['files'].values()),
                         note='Unreferenced upload staging only. Final package, quality review, commit and pinned full-repository verification remain required.')
            save(OUTPUT / 'complete.json', state)
        print(json.dumps({'state': state['stage'], 'total_bytes': state['total_bytes'],
                          'repository_revision_unchanged': after.sha}), flush=True)
    except BaseException as exc:
        # Avoid persisting a signed URL, authorization header or credential in an exception.
        with lock:
            state.update(stage='preupload_failed', error_type=type(exc).__name__, failed=time.time())
            save(OUTPUT / 'failure.json', state)
        print(json.dumps({'state': 'preupload_failed', 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1)
    finally:
        done.set()
        monitor.join()
        with lock:
            save(OUTPUT / 'status.json', {**state, 'heartbeat_at': time.time()})


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        if '--inspect' not in sys.argv:
            OUTPUT.mkdir(exist_ok=True)
            save(OUTPUT/'failure.json', dict(stage='preupload_stopped', error_type=type(exc).__name__,
                 failed=time.time(), final_model_published=False, server_credentials_modified=False))
        print(json.dumps(dict(state='preupload_stopped', error_type=type(exc).__name__)), flush=True)
        raise SystemExit(1)
