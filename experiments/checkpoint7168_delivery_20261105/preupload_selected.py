"""Upload merged inference blobs while evaluation runs, without changing the HF head."""
import fcntl,os,time
from huggingface_hub import CommitOperationAdd
from release_io import Z,C,REPO,auth,prepare_payload,read,save,sha

def main():
    lock=(Z/'preupload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (Z/'preupload_complete.json').exists():return
    save(Z/'preupload_status.json',dict(stage='validating_selected_inference_payload',pid=os.getpid(),time=time.time()))
    payload=prepare_payload()
    receipt=read(Z/'checkpoint_upload/verification_receipt.json')
    assert receipt['state']=='complete_remote_verified_selected_checkpoint' and len(receipt['key_checkpoints'])==37
    api,token=auth()
    with (C/'hub_writer.lock').open('a') as writer:
        fcntl.flock(writer,fcntl.LOCK_EX)
        assert api.model_info(REPO).sha==receipt['revision']
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=item['source']) for name,item in payload.items()]
        save(Z/'preupload_status.json',dict(stage='uploading_uncommitted_inference_blobs',pid=os.getpid(),bytes=sum(x['bytes'] for x in payload.values()),time=time.time()))
        api.preupload_lfs_files(REPO,additions=operations,num_threads=4)
        assert api.model_info(REPO).sha==receipt['revision']
        save(Z/'preupload_complete.json',dict(state='selected_inference_blobs_preuploaded',
            revision_unchanged=receipt['revision'],payload_manifest_sha256=sha(Z/'release_payload/payload_manifest.json'),
            final_spec_sha256=sha(Z/'evaluation_spec.json'),completed=time.time(),final_model_published=False))
        save(Z/'preupload_status.json',dict(stage='uncommitted_upload_complete',time=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(Z/'preupload_failure.json',dict(error_type=type(e).__name__,time=time.time()))
        raise
