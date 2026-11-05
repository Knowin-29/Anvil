"""Check the final remote publication, cleanup ledger and source delivery."""
import json
import subprocess
import time
from pathlib import Path
from release_io import Z, C, REPO, auth, read, save, sha


def audit():
    release=read(Z/'final_release/verification_receipt.json')
    review=read(Z/'public_quality_review.json')
    cleanup=read(Z/'verified_local_cleanup/complete.json')
    plan=read(Z/'verified_local_cleanup/plan.json')
    source=read(Z/'source_git_sync.json');bundle=read(Z/'source_bundle.json')
    assert release['state']=='accepted_selected_model_and37_checkpoints_remotely_verified'
    assert release['quality_review_sha256']==sha(Z/'public_quality_review.json')
    assert review['accepted'] and review['public_mean']>68
    assert len(review['benchmark_assessments'])==7
    assert all(x['accepted'] and x['relative_difference']<=0.10 for x in review['benchmark_assessments'].values())
    assert review['report_sha256']==sha(Z/'evaluation/final_report.json')
    assert review['native_score_sha256']==sha(Z/'native_score.json')
    assert cleanup['revision']==release['revision']
    assert cleanup['plan_sha256']==sha(Z/'verified_local_cleanup/plan.json')
    assert plan['final_receipt_sha256']==sha(Z/'final_release/verification_receipt.json')
    events=[json.loads(line) for line in (Z/'verified_local_cleanup/deletions.jsonl').read_text().splitlines()]
    deleted=[x for x in events if x['event']=='unlinked']
    assert len(deleted)==cleanup['files']==len(plan['candidates'])==1299
    assert len(events)==2*len(deleted)
    assert {x['path'] for x in deleted}=={x['path'] for x in plan['candidates']}
    assert sum(x['bytes'] for x in deleted)==cleanup['logical_bytes']
    assert all(not Path(x['path']).exists() for x in deleted)
    assert sha(bundle['path'])==bundle['sha256']
    assert Path(bundle['link']).resolve()==Path(bundle['path'])
    assert bundle['commit']==source['commit'] and source['branch']=='main'
    repo=Path('/Knowin/demo/weilinruan/Anvil/01_Code')
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==source['commit']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    registry=read(repo/'registry/models.json')['final_release']
    assert registry['revision']==release['revision'] and registry['uploaded']
    api,token=auth()
    model=api.model_info(REPO,files_metadata=True)
    assert model.sha==release['revision'] and not model.private
    remote={x.rfilename:x for x in model.siblings}
    assert set(remote)==set(release['files'])
    for name,item in release['files'].items():
        meta=remote[name];assert meta.size==item['bytes'], name
        if meta.lfs: assert meta.lfs.sha256==item['sha256'], name
    data=read(C/'hub_presentation_20261104/dataset_complete.json')
    dataset=api.dataset_info(data['repository'],files_metadata=True)
    assert dataset.sha==data['revision'] and not dataset.private
    assert len(dataset.siblings)==data['files']==71
    assert len([x for x in dataset.siblings if x.rfilename.endswith('.jsonl.gz')])==63
    result=dict(state='complete_under_latest_public_benchmark_acceptance',
        model_repository=REPO,model_revision=model.sha,model_files=len(remote),
        selected_step=7168,checkpoints=37,featured_stages=5,
        public_mean=review['public_mean'],public_benchmarks_within_tolerance=7,
        native_is_final_acceptance_gate=False,native_score=read(Z/'native_score.json')['score'],
        dataset_repository=data['repository'],dataset_revision=dataset.sha,
        dataset_files=len(dataset.siblings),dataset_records=data['records'],
        source_branch='main',source_commit=source['commit'],source_bundle=bundle,
        local_weight_files_removed=cleanup['files'],local_weight_logical_bytes_removed=cleanup['logical_bytes'],
        local_weight_allocated_bytes_unlinked=cleanup['allocated_bytes_unlinked'],
        unused_reserve_bytes_released=read(Z/'unused_reserve_release.json')['allocated_bytes_released'],
        new_dlc_submissions=0,credentials_modified=False,completed=time.time())
    save(Z/'completion_receipt.json',result)
    pointer=repo.parent/'03_Delivery/CURRENT_CAMPAIGN.json'
    current=read(pointer)
    current.update(state=result['state'],updated=time.time(),source_git=source,source_bundle=bundle,
        source_sync_complete=True,final_model_uploaded=True,final_quality_accepted=True,
        final_revision=model.sha,completion_receipt=str(Z/'completion_receipt.json'),
        cleanup_receipt=str(Z/'verified_local_cleanup/complete.json'),
        local_weight_logical_bytes_removed=cleanup['logical_bytes'],
        publication_prerequisites=[],next_compute_resources='Complete; no further compute required')
    current['storage'].update(current_reserved_bytes=0,current_emergency_reserved_bytes=0)
    save(pointer,current)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_bundle','unused_reserve_bytes_released']},indent=2))


if __name__=='__main__':audit()
