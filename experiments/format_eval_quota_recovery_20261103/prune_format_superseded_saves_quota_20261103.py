"""Prune only non-key payloads from the completed format repair stage.

Keep every key checkpoint, every MOPD save, all metadata and all datasets.
These non-key saves were superseded, not individually uploaded. Their complete
stage endpoint and midpoint have verified remote backups. Default: inventory.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

C = Path('/data/anvil/recovery_20261020/completion_20261028')
V = C / 'verification_20261101'
P = C / 'format_repair_20261103'
D = C / 'delivery_runs/format_repair_20261103'
B = C / 'format_checkpoint_backup_20261103'
OUT = Path('/opt/anvil-private/anvil-quota-recovery-20261103/format_superseded_cleanup')
PAYLOAD = re.compile(r'(adapter_model\.safetensors|optimizer\.pt|scheduler\.pt|rng_state_\d+\.pth)')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = read(P/'spec.json')
    backup = read(B/'mopd_256/verification_receipt.json')
    remote = read(V/'format_first27_remote_recheck.json')
    assert backup['state'] == 'complete_remote_verified_intermediate_checkpoints'
    assert len(backup['key_checkpoints']) == remote['keys'] == 27
    assert backup['revision'] == remote['revision'] == remote['head_revision']
    assert remote['remote_files'] == len(backup['files']) == 1083
    assert read(P/'runs/mopd/complete.json')['steps'] == 256
    assert read(B/'complete.json')['state'] == 'all_declared_checkpoints_remotely_verified'
    complete = read(P/'runs/repair/complete.json')
    assert complete['steps'] == 8192 and complete['stage'] == 'repair_training_complete'
    keys = {Path(p) for p in spec['key_checkpoint_sources'].values()}
    endpoint = Path(complete['checkpoint'])
    assert endpoint in keys
    for step in (4096, 8192):
        folder = P/f'runs/repair/checkpoint-{step}'
        audit = read(P/f'repair_checkpoint{step}_verification.json')
        assert audit['all_file_hashes_verified'] and audit['optimizer_step'] == step
        prefix = f'checkpoints/format_repair/checkpoint-{step}/'
        seal = read(folder/'checkpoint_complete.json')
        assert sha(folder/'checkpoint_complete.json') == backup['files'][prefix+'checkpoint_complete.json']['sha256']
        for name, item in seal['files'].items():
            assert item['sha256'] == backup['files'][prefix+name]['sha256']
            assert (folder/name).stat().st_size == item['bytes']
        assert sha(folder/'adapter_model.safetensors') == backup['files'][prefix+'adapter_model.safetensors']['sha256']
    items = []
    required = {'adapter_model.safetensors','optimizer.pt','scheduler.pt'} | {f'rng_state_{i}.pth' for i in range(32)}
    for folder in sorted((P/'runs/repair').glob('checkpoint-*')):
        if folder in keys:
            continue
        assert folder == folder.resolve() and folder.is_dir()
        step = int(folder.name.removeprefix('checkpoint-'))
        assert 0 < step < 8192
        seal = read(folder/'checkpoint_complete.json')
        assert seal['global_step'] == step and seal['world_size'] == 32
        names = {name for name in seal['files'] if PAYLOAD.fullmatch(name)}
        assert names == required
        for name in sorted(names):
            path = folder/name
            st = path.lstat()
            assert stat.S_ISREG(st.st_mode) and st.st_nlink == 1
            assert st.st_size == seal['files'][name]['bytes']
            items.append(dict(path=str(path), bytes=st.st_size, allocated_bytes=st.st_blocks*512,
                              inode=st.st_ino, device=st.st_dev, mtime_ns=st.st_mtime_ns,
                              ctime_ns=st.st_ctime_ns, sealed_sha256=seal['files'][name]['sha256']))
    assert len(items) == 126*35
    inventory = dict(reason='superseded_nonkey_saves_of_completed_stage', individually_uploaded=False,
                     retained_remote_revision=backup['revision'], retained_all27_keys=True,
                     retained_all_mopd_saves=True, retained_metadata_logs_and_data=True,
                     bytes=sum(i['bytes'] for i in items), files=items, created=time.time())
    if not args.apply:
        print(json.dumps(dict(files=len(items), bytes=inventory['bytes'], apply=False)))
        return
    assert not (OUT/'intent.json').exists(), 'Reconcile any partial cleanup explicitly'
    sys.path.insert(0, str(C))
    from final_delivery_handoff_v1 import cloud_snapshot
    with ExitStack() as stack:
        for lock in [Path('/tmp/anvil-formal-mopd-submit.lock'), C/'hub_writer.lock',
                     P/'controller.lock', D/'supervisor.lock', B/'worker.lock',
                     V/'format_final_release/worker.lock']:
            f = stack.enter_context(lock.open('a'))
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cloud, active = cloud_snapshot()
        assert active == 0, 'Campaign still active or UNKNOWN'
        by_id = {r['id']: r for r in cloud}
        assert by_id[read(P/'submission.json')['response']['Id']]['status'] == 'Success'
        # Independently reconcile the current remote inventory before local pruning.
        import runpy
        from huggingface_hub import HfApi
        token = runpy.run_path(str(V/'preupload_merged_blobs.py'))['authorized_token']()
        info = HfApi(token=token).model_info(backup['repository'], files_metadata=True)
        assert info.sha == backup['revision'], 'HF head changed'
        actual = {f.rfilename: f for f in info.siblings}
        assert set(actual) == set(backup['files'])
        for name, expected in backup['files'].items():
            meta = actual[name]
            assert meta.size == expected['bytes']
            if meta.lfs:
                digest = meta.lfs.get('sha256') if isinstance(meta.lfs, dict) else meta.lfs.sha256
                assert digest == expected['sha256']
        del token
        names = {i['path'] for i in items}
        for process in Path('/proc').glob('[0-9]*'):
            if int(process.name) == os.getpid():
                continue
            try:
                argv = (process/'cmdline').read_bytes().split(b'\0')
                assert os.fsencode(P/'train.py') not in argv
                for fd in (process/'fd').iterdir():
                    try:
                        assert os.readlink(fd) not in names, 'Payload still open'
                    except FileNotFoundError:
                        pass
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
        inventory['cloud_snapshot'] = cloud
        save(OUT/'intent.json', inventory)
        removed = []
        with (OUT/'deleted.jsonl').open('x') as journal:
            for item in items:
                path = Path(item['path'])
                st = path.lstat()
                assert (st.st_ino, st.st_dev, st.st_size, st.st_mtime_ns, st.st_ctime_ns) == (
                    item['inode'],item['device'],item['bytes'],item['mtime_ns'],item['ctime_ns'])
                assert stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and path.parent not in keys
                path.unlink()
                removed.append(item)
                journal.write(json.dumps(dict(path=str(path),bytes=item['bytes'],removed_at=time.time()))+'\n')
                journal.flush()
        for key in keys:
            assert (key/'adapter_model.safetensors').is_file()
            assert (key/'optimizer.pt').is_file()
        result = dict(state='completed', files=len(removed), bytes=sum(i['bytes'] for i in removed),
                      allocated_bytes=sum(i['allocated_bytes'] for i in removed), completed=time.time(),
                      intent_sha256=sha(OUT/'intent.json'), all27_keys_retained=True,
                      all_mopd_saves_retained=True, server_credentials_modified=False,
                      system_disk_bytes_removed=0, storage='shared /Knowin/sim',
                      open_file_inspection_scope='Visible local processes; inaccessible unrelated processes excluded; cloud campaign independently terminal')
        save(OUT/'complete.json', result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
