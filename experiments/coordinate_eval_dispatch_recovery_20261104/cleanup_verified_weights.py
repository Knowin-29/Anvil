"""Remove campaign weight payloads only after the accepted release is verified.

Default is a dry run. No training files, metadata, logs, datasets, base models,
teachers or credentials are removed. --apply additionally requires all campaign
jobs and writers to have finished. Superseded non-key saves are eligible only
when their completed stage's last checkpoint is present in the verified release.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import stat
import sys
import time

C = Path('/data/anvil/recovery_20261020/completion_20261028')
V = C / 'verification_20261101'
PIPELINE = C / 'geometry_objective_v2_20261104'
DELIVERY = C / 'delivery_runs/geometry_objective_v2_20261104'
RECOVERY = DELIVERY
VERIFIER = V / 'verify_entire_coordinate_hub_v2.py'
REPOSITORY = 'Knowin-29/Anvil-MOPD-9B'
DISPATCH = Path(__file__).resolve().parent
RESUME = C / 'coordinate_resume_storage_20261104'
CONTROL = C / 'coordinate_delivery_control_v2_20261104'
sys.path.insert(0,str(CONTROL))
from quality_gate import quality_decision
PAYLOAD = re.compile(r'(?:adapter_model\.safetensors|optimizer\.pt|scheduler\.pt|rng_state_\d+\.pth)')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def within(path, root):
    path, root = Path(path), Path(root)
    require(path.is_absolute() and path == path.resolve(), 'Symlink or noncanonical path: ' + str(path))
    require(path.is_relative_to(root.resolve()) and path != root.resolve(), 'Path outside allowed output: ' + str(path))
    return path


def identity(path, expected, root):
    path = within(path, root)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'Not a regular payload: ' + str(path))
    require(before.st_size == expected['bytes'] and sha(path) == expected['sha256'],
            'Local payload differs from its verified identity: ' + str(path))
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, k) == getattr(after, k) for k in fields), 'Payload changed while reading')
    return dict(path=str(path), bytes=after.st_size, sha256=expected['sha256'],
                device=after.st_dev, inode=after.st_ino, mtime_ns=after.st_mtime_ns,
                ctime_ns=after.st_ctime_ns, links=after.st_nlink)


def require_release_complete(record, revision, review_sha):
    require(record['state']=='accepted_final_model_and35_checkpoints_remotely_verified'
            and record['revision']==revision and record['quality_review_sha256']==review_sha,
            'Final accepted publication is not complete or differs')


def load_release(args):
    published = read(CONTROL/'final_release/complete.json')
    review_path=CONTROL/'quality_review.json'
    report_path=DELIVERY/'evaluation/final_report.json'
    review=read(review_path);report=read(report_path)
    require(quality_decision(review,report,sha(report_path),sha(DELIVERY/'evaluation_spec.json')),
            'Native/public/general quality was not accepted')
    require(args.final_receipt.resolve()==CONTROL/'final_release/upload_verification.json'
            and args.verification_receipt.resolve()==CONTROL/'final_release/whole_hub_verification.json',
            'Use this exact campaign publication receipts')
    check = runpy.run_path(str(VERIFIER))
    paths = [args.manifest, args.final_receipt, args.backup_receipt,
             PIPELINE / 'spec.json', DELIVERY / 'evaluation_spec.json']
    manifest, final, backup, pipeline, evaluation = map(read, paths)
    require(within(args.manifest, DELIVERY).name == 'release_manifest.json', 'Unexpected package manifest')
    expected = check['inventory'](manifest, dict(bytes=args.manifest.stat().st_size,
                                              sha256=sha(args.manifest)), final, backup)
    check['bind_final_model'](manifest, pipeline, sha(paths[3]), evaluation, sha(paths[4]))
    whole = read(args.verification_receipt)
    require_release_complete(published,whole['revision'],sha(review_path))
    require(manifest['review_sha256']==sha(review_path),'Packaged quality review differs')
    require(whole['state'] == 'complete_entire_remote_release_verified'
            and whole['repository'] == REPOSITORY and whole['revision'] == final['revision'],
            'Missing complete final release verification')
    require(set(whole['key_checkpoints']) == check['KEYS'] and len(whole['key_checkpoints']) == 35,
            'All thirty-five remote checkpoints are required')
    require(set(whole['files']) == set(expected), 'Incomplete whole-Hub receipt')
    for name, item in expected.items():
        require(all(whole['files'][name][k] == item[k] for k in ('bytes', 'sha256')),
                'Whole-Hub receipt differs: ' + name)
    for path in paths + [VERIFIER]:
        require(whole['evidence_sha256'].get(str(path)) == sha(path), 'Changed verified evidence: ' + str(path))
    require(Path(evaluation['model']) == DELIVERY / 'merged_model', 'Unexpected final merged model')
    return manifest, pipeline, evaluation, whole


def candidate_sources(manifest, pipeline, evaluation, package):
    """Produce an explicit list; never recursively delete a directory."""
    candidates = {}
    previously_pruned = set()
    for ledger_name in ('grounded_superseded_cleanup_20261103', 'format_superseded_cleanup_quota_20261103'):
        prior_root = V / ledger_name
        prior_complete = read(prior_root / 'complete.json')
        require(prior_complete['state'] == 'completed' and
                prior_complete['intent_sha256'] == sha(prior_root / 'intent.json'),
                'Changed prior cleanup ledger: ' + ledger_name)
        prior_intent = read(prior_root / 'intent.json')
        ledger_paths = {item['path'] for item in prior_intent['files']}
        require(len(ledger_paths) == prior_complete['files'] == 4410,
                'Incomplete prior cleanup inventory: ' + ledger_name)
        require(not (ledger_paths & previously_pruned), 'Overlapping cleanup ledgers')
        previously_pruned.update(ledger_paths)

    # Bound early quota cleanup: only the126 superseded repair optimizers.
    quota_root = RESUME
    quota_complete = read(quota_root / 'optimizer_cleanup_complete.json')
    require(quota_complete['state']=='complete' and quota_complete['preview_sha256']==sha(quota_root/'optimizer_cleanup_preview.json'), 'Changed quota cleanup preview')
    quota_preview = read(quota_root/'optimizer_cleanup_preview.json')
    quota_deleted = [json.loads(line) for line in (quota_root/'optimizer_cleanup_deleted.jsonl').read_text().splitlines()]
    require(len(quota_deleted)==len(quota_preview['candidates'])==quota_complete['files_removed']==126, 'Incomplete quota cleanup ledger')
    expected_quota = {e['path']:e for e in quota_preview['candidates']}
    require(len(expected_quota)==126 and len({e['path'] for e in quota_deleted})==126, 'Duplicate quota cleanup entry')
    for entry in quota_deleted:
        expected = expected_quota[entry['path']]
        require(all(entry[k]==v for k,v in expected.items()), 'Quota deletion differs from preview')
        path = Path(entry['path'])
        require(path.name=='optimizer.pt' and path.parent.parent==PIPELINE/'runs/repair', 'Unexpected quota cleanup scope')
        require(str(path.parent) not in quota_preview['protected'] and not path.exists(), 'Protected or recreated quota payload')
        require(str(path) not in previously_pruned, 'Overlapping quota cleanup')
        previously_pruned.add(str(path))
    require(sum(e['bytes'] for e in quota_deleted)==quota_complete['bytes_freed'], 'Quota cleanup byte count differs')

    def add(path, expected, reason, remote=None):
        path = within(path, C)
        require(path.exists(), 'Missing payload before cleanup: ' + str(path))
        require(str(path) not in candidates, 'Duplicate cleanup path')
        candidates[str(path)] = dict(expected=expected, reason=reason, remote_path=remote)

    sources = {Path(path): key for key, path in pipeline['key_checkpoint_sources'].items()}
    for run in sorted({path.parent for path in sources}):
        within(run, C)
        complete = read(run / 'complete.json')
        last = Path(complete['checkpoint'])
        require(complete['stage'] == run.name + '_training_complete' and complete['smoke'] is False,
                'Training stage is not complete: ' + str(run))
        require(last.parent == run and last in sources
                and last.name == 'checkpoint-' + str(complete['steps']), 'Last stage checkpoint is not backed up')
        final_remote = 'checkpoints/' + sources[last] + '/adapter_model.safetensors'
        require(manifest['files'][final_remote]['sha256'] == complete['adapter_sha256'],
                'Completed stage differs from its remote final checkpoint')
        for folder in sorted(run.glob('checkpoint-*')):
            require(re.fullmatch(r'checkpoint-[0-9]+', folder.name), 'Unknown checkpoint directory')
            within(folder, run)
            seal = read(folder / 'checkpoint_complete.json')
            step = int(folder.name.split('-')[1])
            require(seal['schema'] == 'anvil-complete-checkpoint-v1' and seal['world_size'] == 32
                    and seal['global_step'] == step and 0 < step <= complete['steps'], 'Unsealed or future checkpoint')
            key = sources.get(folder)
            if key:
                remote_seal = manifest['files']['checkpoints/' + key + '/checkpoint_complete.json']
                require(sha(folder / 'checkpoint_complete.json') == remote_seal['sha256'], 'Local key seal differs')
            expected_names = {name for name in seal['files'] if PAYLOAD.fullmatch(name)}
            require({'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt'}
                    | {f'rng_state_{rank}.pth' for rank in range(32)} == expected_names,
                    'Incomplete checkpoint payload inventory')
            for name in sorted(expected_names):
                expected = seal['files'][name]
                remote = 'checkpoints/' + key + '/' + name if key else None
                if remote:
                    require(expected == manifest['files'][remote], 'Local key seal differs from release')
                if not (folder / name).exists() and not key and str(folder / name) in previously_pruned:
                    continue
                add(folder / name, expected, 'uploaded_key_copy' if key else 'superseded_nonkey_save', remote)

    for name, item in manifest['files'].items():
        path = Path(name)
        if path.suffix in {'.safetensors', '.pt', '.pth', '.bin'}:
            add(package / path, item, 'uploaded_package_copy', name)
    for name, digest in evaluation['model_files'].items():
        if Path(name).suffix == '.safetensors':
            require(manifest['files'][name]['sha256'] == digest, 'Merged shard differs from release')
            add(Path(evaluation['model']) / name, manifest['files'][name], 'uploaded_merged_copy', name)
    return candidates


def idle_campaign():
    sys.path.insert(0, str(C))
    from final_delivery_handoff_v1 import cloud_snapshot
    cloud, active = cloud_snapshot()
    require(cloud and active == 0, 'Campaign jobs are still active or UNKNOWN')
    by_id = {row['id']: row for row in cloud}
    require(by_id[read(PIPELINE / 'submission.json')['response']['Id']]['status'] == 'Killed', 'Original failed task not stopped')
    identities = [read(RESUME / 'submission.json')['response']['Id']]
    identities += [read(DISPATCH / 'submissions' / str(n) / 'submission.json')['response']['Id'] for n in range(4)]
    require(all(by_id[task]['status'] == 'Success' for task in identities), 'Final training/evaluation did not succeed')
    require(read(RECOVERY / 'status.json')['stage'] == 'full_final_report_ready_requires_quality_review',
            'Final handoff has not completed')
    require(read(C / 'coordinate_checkpoint_backup_v2_20261104/complete.json')['state']
            == 'all_declared_checkpoints_remotely_verified', 'Checkpoint backup has not completed')
    require(read(V / 'coordinate_v2_merged_blob_preupload/complete.json')['stage']
            == 'uncommitted_merged_blobs_preuploaded', 'Merged preupload has not completed')
    return [dict(id=row['id'], status=row['status'], gpus=row['gpus']) for row in cloud]


def no_local_readers(items):
    names = {item['path'] for item in items}
    other_users = []
    protected_unrelated = []
    writers = [str(C / 'final_delivery_geometry_objective_v2' / name) for name in ('merge.py', 'package.py', 'upload.py')]
    writers += [str(C / 'backup_coordinate_checkpoints_v2.py'), str(V / 'preupload_coordinate_merged_blobs_v2.py')]
    writers += [str(CONTROL/'finish_delivery.py'), str(RESUME/'finish_delivery.py'), str(DISPATCH/'finish_delivery.py')]
    for process in Path('/proc').glob('[0-9]*'):
        if int(process.name) == os.getpid():
            continue
        argv = []
        try:
            argv = (process / 'cmdline').read_bytes().split(b'\0')
            require(not any(os.fsencode(name) in argv for name in writers), 'Local campaign writer still running: PID ' + process.name)
            # This campaign runs under the current Unix user. Other users' fd
            # links are kernel-protected even for this container's root. Do not
            # mistake an uninspectable unrelated process for a campaign reader;
            # retain the inspection scope in the receipt. Cloud jobs are checked
            # independently above, including UNKNOWN jobs.
            if process.stat().st_uid != os.geteuid():
                require(not any(os.fsencode(C) in arg for arg in argv),
                        'Another Unix user has a campaign process: PID ' + process.name)
                other_users.append(int(process.name))
                continue
            for fd in (process / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                require(target not in names, 'Local process has a cleanup payload open: PID ' + process.name)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            require(argv and not any(os.fsencode(C) in arg for arg in argv),
                    'Cannot inspect a possible local campaign process: PID ' + process.name)
            protected_unrelated.append(int(process.name))
    return dict(inspected_unix_uid=os.geteuid(), unrelated_other_user_pids=other_users,
                protected_processes_without_campaign_arguments=protected_unrelated,
                scope='Campaign writers plus readable owner-user file descriptors; cloud jobs checked separately')


def check_remote(whole):
    token = os.environ.get('ANVIL_HF_TOKEN')
    require(bool(token), 'Provide transient ANVIL_HF_TOKEN; no server credential is read')
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    require(api.whoami(token=token)['name'] == 'Knowin-29', 'Wrong authorized account')
    info = api.model_info(REPOSITORY, files_metadata=True, token=token)
    require(info.sha == whole['revision'], 'HF head changed after verification')
    remote = {f.rfilename: f for f in info.siblings}
    require(set(remote) == set(whole['files']), 'HF inventory changed')
    for name, item in whole['files'].items():
        meta = remote[name]
        require(meta.size == item['bytes'], 'Remote size changed')
        if meta.lfs:
            digest = meta.lfs.get('sha256') if isinstance(meta.lfs, dict) else meta.lfs.sha256
            require(digest == item['sha256'], 'Remote LFS identity changed')


def unchanged(item):
    current = Path(item['path']).lstat()
    require(stat.S_ISREG(current.st_mode) and all(getattr(current, a) == item[b] for a, b in
            [('st_dev', 'device'), ('st_ino', 'inode'), ('st_size', 'bytes'), ('st_mtime_ns', 'mtime_ns'),
             ('st_ctime_ns', 'ctime_ns')]), 'Payload changed since preflight: ' + item['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'final-receipt', 'backup-receipt', 'verification-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new cleanup receipt directory')
    manifest, pipeline, evaluation, whole = load_release(args)
    with ExitStack() as stack:
        for path in [CONTROL/'final_release/worker.lock', Path('/tmp/anvil-formal-mopd-submit.lock'), C / 'hub_writer.lock',
                     PIPELINE / 'controller.lock', RECOVERY / 'supervisor.lock',
                     C / 'coordinate_checkpoint_backup_v2_20261104/worker.lock',
                     V / 'coordinate_v2_merged_blob_preupload/worker.lock']:
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cloud = idle_campaign()
        check_remote(whole)
        sources = candidate_sources(manifest, pipeline, evaluation, args.manifest.parent)
        items = [dict(identity(Path(path), value['expected'], C), reason=value['reason'],
                      remote_path=value['remote_path']) for path, value in sorted(sources.items())]
        local_inspection = no_local_readers(items)
        # Recheck after potentially lengthy local hashing, before any unlink.
        idle_campaign()
        check_remote(whole)
        for item in items:
            unchanged(item)
        require(len({(x['device'], x['inode']) for x in items}) == len(items),
                'Cleanup candidates share a hard link; review storage accounting before deletion')
        args.output.mkdir(parents=True, exist_ok=False)
        plan = dict(schema='anvil-verified-release-weight-cleanup-v1', revision=whole['revision'],
                    created=time.time(), dry_run=not args.apply, cloud=cloud, candidates=items,
                    total_logical_bytes=sum(x['bytes'] for x in items),
                    verification_receipt=str(args.verification_receipt),
                    verification_receipt_sha256=sha(args.verification_receipt),
                    script_sha256=sha(__file__),
                    local_process_inspection=local_inspection,
                    retained='All metadata, logs, data, bases, teachers, source and Codex records')
        (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        if args.apply:
            # Journal each unlink intent/result, so an interrupted cleanup is auditable.
            with (args.output / 'deletions.jsonl').open('x') as stream:
                for item in items:
                    unchanged(item)
                    stream.write(json.dumps(dict(event='unlink_intent', path=item['path'], time=time.time())) + '\n')
                    stream.flush(); os.fsync(stream.fileno())
                    Path(item['path']).unlink()
                    stream.write(json.dumps(dict(event='unlinked', path=item['path'], bytes=item['bytes'], time=time.time())) + '\n')
                    stream.flush(); os.fsync(stream.fileno())
            (args.output / 'complete.json').write_text(json.dumps(dict(state='eligible_local_weight_payloads_removed',
                revision=whole['revision'], removed_files=len(items), logical_bytes=sum(x['bytes'] for x in items),
                plan_sha256=sha(args.output / 'plan.json'), completed=time.time()), indent=2) + '\n')
        print(json.dumps(dict(dry_run=not args.apply, files=len(items), logical_bytes=sum(x['bytes'] for x in items))))


if __name__ == '__main__':
    main()
