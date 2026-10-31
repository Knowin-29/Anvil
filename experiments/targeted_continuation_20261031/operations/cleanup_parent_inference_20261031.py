"""Remove four reconstructible, completed-parent inference shards only."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

C=Path(__file__).resolve().parent
FOLDER=C/'delivery_runs/direct_recovery_20261030/merged_model'
OUT=C/'storage_cleanup_20261031'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def cloud(task):
    proc=subprocess.run(['/root/.volc/bin/volc','--log-level','error','ml_task','get','--id',task,
        '--output','json','--format','Id,Name,Status'],capture_output=True,text=True,timeout=25)
    rows=[json.loads(line) for line in proc.stdout.splitlines() if line.startswith('[{"Id"')]
    assert proc.returncode==0 and len(rows)==1
    return rows[0][0]


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'deletion_receipt.json').exists()
    assert not FOLDER.is_symlink()
    parent=C/'delivery_runs/direct_recovery_20261030'
    handoff=read(parent/'status.json');assert handoff['stage']=='full_final_report_ready_requires_quality_review'
    assert not (Path('/proc')/str(handoff['pid'])).exists()
    report=read(parent/'evaluation/final_report.json')
    assert sha(parent/'evaluation/final_report.json')==handoff['report_sha256'] and report['native']['n']==580000
    tasks=['t-20261031115815-69g2z','t-20261031115835-mdw9z','t-20261031115855-mwgn9','t-20261031115919-5d4f8']
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs=list(pool.map(cloud,tasks+['t-20261031125657-rhdzz']))
    assert all(job['Status']=='Success' for job in jobs[:4]) and jobs[-1]['Status']=='Running'
    spec=read(C/'targeted_repair_20261031/spec.json');policy=read(C/'delivery_runs/targeted_repair_20261031/policy.json')
    lineage=read(parent/'training_lineage.json')
    bindings={**spec['frozen_files'],**policy['files'],**{x['path']:x['sha256'] for x in lineage['evidence']}}
    assert not any(str(FOLDER) in path for path in bindings)
    merged=read(FOLDER/'merge_receipt.json');assert merged['state']=='verified_derived_inference_copy'
    adapter=Path(merged['original_adapter']);base=Path(merged['original_base'])
    assert sha(adapter/'adapter_model.safetensors')==merged['adapter_sha256']==spec['initial']['adapter_sha256']
    backup=read(C/'targeted_checkpoint_backup_20261031/repair_2048/verification_receipt.json')
    assert backup['state']=='complete_remote_verified_intermediate_checkpoints'
    assert backup['files']['checkpoints/ancestor_mopd/checkpoint-1536/adapter_model.safetensors']['sha256']==merged['adapter_sha256']
    assert {p.name for p in FOLDER.glob('*.safetensors')}==set(merged['shards'])
    candidates=[]
    for name,expected in merged['shards'].items():
        file=FOLDER/name;st=file.lstat();assert file.is_file() and not file.is_symlink() and st.st_nlink==1
        assert st.st_size==expected['bytes'] and sha(file)==expected['derived_sha256']
        assert sha(base/name)==expected['original_sha256']
        candidates.append(dict(path=str(file),bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns,
                               derived_sha256=expected['derived_sha256'],base_sha256=expected['original_sha256']))
        print('Verified derived shard and retained base:',name,flush=True)
    paths={x['path'] for x in candidates}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            command=(proc/'cmdline').read_bytes().decode(errors='replace')
            assert str(FOLDER) not in command,'Local process references the derived model'
            for fd in (proc/'fd').iterdir():
                try:assert os.readlink(fd) not in paths,'Open payload descriptor'
                except OSError:pass
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    audit=dict(created=time.time(),cloud=jobs,candidates=candidates,retained_base=str(base),retained_adapter=str(adapter),
        adapter_sha256=merged['adapter_sha256'],backup_revision=backup['revision'],merge_receipt_sha256=sha(FOLDER/'merge_receipt.json'),
        full_parent_report_sha256=handoff['report_sha256'],frozen_dependency_paths=list(bindings),
        merged_weights_themselves_uploaded=False,all_training_checkpoints_logs_data_preserved=True)
    save(OUT/'preflight.json',audit)
    result=dict(started=time.time(),preflight_sha256=sha(OUT/'preflight.json'),removed=[],bytes_removed=0,
        scope='Only the four completed-parent derived inference shards; no checkpoint, source dataset, log or server credential removed')
    save(OUT/'deletion_receipt.json',result)
    for item in candidates:
        file=Path(item['path']);st=file.lstat()
        assert (st.st_ino,st.st_size,st.st_mtime_ns,st.st_nlink)==(item['inode'],item['bytes'],item['mtime_ns'],1)
        file.unlink();result['removed'].append(item);result['bytes_removed']+=item['bytes'];save(OUT/'deletion_receipt.json',result)
    assert not any(Path(path).exists() for path in paths)
    for path,digest in bindings.items():assert sha(Path(path))==digest,'Frozen dependency changed: '+path
    assert sha(adapter/'adapter_model.safetensors')==merged['adapter_sha256']
    save(FOLDER/'INFERENCE_SHARDS_PRUNED.json',dict(pruned_at=time.time(),deletion_receipt=str(OUT/'deletion_receipt.json'),
        reason='Completed parent evaluation; inference copy is reproducible from retained base and remotely backed-up adapter',
        retained_base=str(base),retained_adapter=str(adapter),restore_source=str(C/'final_delivery_recovery_v1/merge.py'),
        restore_instruction='Rebuild in a new directory using the retained adapter and original merger; compare shard hashes with merge_receipt.json before using the parent again.',
        loadable_model_directory=False))
    result['completed']=time.time();result['frozen_dependencies_preserved']=len(bindings);save(OUT/'deletion_receipt.json',result)
    print(json.dumps(dict(bytes_removed=result['bytes_removed'],GiB_removed=result['bytes_removed']/2**30,
                          frozen_dependencies_preserved=len(bindings))),flush=True)


if __name__=='__main__':main()
