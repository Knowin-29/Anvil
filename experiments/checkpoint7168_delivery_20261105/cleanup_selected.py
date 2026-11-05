"""Remove verified local weight copies after accepted 7168 publication.

Default is a dry run. This version handles the actual 37-checkpoint release;
it never assumes the abandoned visual repair8192/MOPD256 stages completed.
Logs, datasets, source, metadata, base models and Codex history are preserved.
"""
import argparse,fcntl,importlib.util,json,os,re,sys,time
from contextlib import ExitStack
from pathlib import Path
from release_io import Z,C,REPO,read,save,sha,auth

PAYLOAD=re.compile(r'(?:adapter_model\.safetensors|optimizer\.pt|scheduler\.pt|rng_state_\d+\.pth)')
HELPER=C/'visual_resume7168_20261105/cleanup_verified_weights.py'
HELPER_SHA='d7d45a41f9c668a165003ceffed86bf710f185bfb0d1d1b6220698bb61326afa'

def helpers():
    assert sha(HELPER)==HELPER_SHA,'Cleanup helper changed'
    spec=importlib.util.spec_from_file_location('prior_cleanup_helpers',HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def require_idle():
    sys.path.insert(0,str(C))
    from final_delivery_handoff_v1 import cloud_snapshot
    cloud,active=cloud_snapshot()
    assert cloud and active==0,'Cloud campaign still active or unknown'
    identity=read(Z/'evaluation_submission.json')['response']['Id']
    assert next(x for x in cloud if x['id']==identity)['status']=='Success'
    assert read(Z/'evaluation/node_0.json')['stage']=='node_complete'
    assert read(Z/'status.json')['stage']=='full_report_and_tolerance_ready'
    return cloud

def verified_release():
    p=Z/'final_release/verification_receipt.json';whole=read(p)
    assert whole['state']=='accepted_selected_model_and37_checkpoints_remotely_verified'
    assert whole['repository']==REPO and whole['checkpoints']==37 and whole['selected_step']==7168
    assert whole['final_spec_sha256']==sha(Z/'evaluation_spec.json')
    assert whole['quality_review_sha256']==sha(Z/'quality_review.json')
    assert read(Z/'quality_review.json')['accepted'] is True
    assert read(Z/'tolerance_assessment.json')['all_numeric_tolerances_pass'] is True
    return whole

def remote_check(api,whole):
    info=api.model_info(REPO,files_metadata=True)
    assert info.sha==whole['revision'],'Hub changed after verification'
    files={x.rfilename:x for x in info.siblings}
    assert set(files)==set(whole['files'])
    for name,item in whole['files'].items():
        meta=files[name];assert meta.size==item['bytes'],name
        if meta.lfs:assert meta.lfs.sha256==item['sha256'],name

def candidates(whole):
    mapping=read(Z/'final_release/layout_map.json')
    backup=read(Z/'checkpoint_upload/manifest.json')
    prior=read(Z/'checkpoint_upload/verification_receipt.json')
    assert prior['manifest_sha256']==sha(Z/'checkpoint_upload/manifest.json')
    assert prior['revision']==whole['parent_revision'] and len(prior['key_checkpoints'])==37
    payload=read(Z/'release_payload/payload_manifest.json')
    assert payload['final_spec_sha256']==whole['final_spec_sha256']
    output={}
    for source_manifest,is_checkpoint in [(backup['files'],True),(payload['files'],False)]:
        for name,item in source_manifest.items():
            if is_checkpoint:
                if not name.startswith('checkpoints/') or not PAYLOAD.fullmatch(Path(name).name):continue
            elif Path(name).suffix!='.safetensors':continue
            remote=mapping[name]
            expected={k:item[k] for k in ['bytes','sha256']}
            assert whole['files'][remote]==expected,'Payload not byte-identical to accepted remote copy'
            p=Path(item['source'])
            assert p.is_absolute() and p==p.resolve() and p.is_relative_to(C) and p.is_file()
            if str(p) in output:
                assert output[str(p)]['expected']==expected
                output[str(p)]['remote_paths'].append(remote)
            else:output[str(p)]=dict(expected=expected,remote_paths=[remote])
    assert sum(Path(n).name=='optimizer.pt' for n in output)==37
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    assert out.parent==Z and not out.exists(),'Use a new direct child of the delivery directory'
    whole=verified_release();helper=helpers()
    with ExitStack() as stack:
        for p in [Z/'controller.lock',Z/'preupload.lock',C/'hub_writer.lock',
                  C/'visual_geometry_candidate_20261104/controller.lock',
                  C/'visual_checkpoint_backup_20261104/worker.lock',
                  Path('/tmp/anvil-formal-mopd-submit.lock')]:
            lock=stack.enter_context(p.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cloud=require_idle();api,token=auth();remote_check(api,whole)
        sources=candidates(whole)
        items=[dict(helper.identity(Path(name),item['expected'],C),remote_paths=item['remote_paths'])
            for name,item in sorted(sources.items())]
        inspection=helper.no_local_readers(items)
        assert len({(x['device'],x['inode']) for x in items})==len(items),'Hard-linked candidates need review'
        assert all(x['links']==1 for x in items),'Shared hard links need separate storage accounting'
        require_idle();remote_check(api,whole)
        for item in items:helper.unchanged(item)
        out.mkdir()
        save(out/'plan.json',dict(schema='anvil-selected7168-local-copy-cleanup-v1',revision=whole['revision'],
            dry_run=not args.apply,candidates=items,logical_bytes=sum(x['bytes'] for x in items),
            final_receipt_sha256=sha(Z/'final_release/verification_receipt.json'),
            script_sha256=sha(__file__),helper_sha256=HELPER_SHA,cloud=cloud,local_inspection=inspection,
            retained='Non-key checkpoints, all logs, source, metadata, datasets, bases, teachers and Codex records',
            created=time.time()))
        if args.apply:
            with (out/'deletions.jsonl').open('x') as log:
                for item in items:
                    helper.unchanged(item)
                    log.write(json.dumps(dict(event='unlink_intent',path=item['path'],time=time.time()))+'\n')
                    log.flush();os.fsync(log.fileno())
                    Path(item['path']).unlink()
                    log.write(json.dumps(dict(event='unlinked',path=item['path'],bytes=item['bytes'],time=time.time()))+'\n')
                    log.flush();os.fsync(log.fileno())
            save(out/'complete.json',dict(state='verified_local_checkpoint_and_inference_copies_removed',
                revision=whole['revision'],files=len(items),logical_bytes=sum(x['bytes'] for x in items),
                plan_sha256=sha(out/'plan.json'),completed=time.time()))
        print(json.dumps(dict(dry_run=not args.apply,files=len(items),logical_bytes=sum(x['bytes'] for x in items))))

if __name__=='__main__':main()
