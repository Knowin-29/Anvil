"""Publish an accepted 7168 release and five-stage layout in one atomic HF commit.

--prepare writes a reviewable local plan. --apply publishes that same plan after
an evidence-bound review. Neither mode deletes local checkpoints or submits GPUs.
"""
import argparse,fcntl,json,math,time
from pathlib import Path
from huggingface_hub import CommitOperationAdd,CommitOperationCopy,CommitOperationDelete,hf_hub_download
from release_io import Z,C,REPO,auth,read,save,sha,validated_payload
from model_presentation import build_plan,digest

def accepted_report():
    spec_path=Z/'evaluation_spec.json';report_path=Z/'evaluation/final_report.json'
    review=read(Z/'quality_review.json');report=read(report_path);native=read(Z/'native_score.json')
    policy_path=C/'hub_presentation_visual_20261104/user_acceptance_policy.json'
    policy=read(policy_path)
    assert review['accepted'] is True and review['report_sha256']==sha(report_path)
    assert review['final_spec_sha256']==report['final_spec_sha256']==sha(spec_path)
    assert review['user_acceptance_policy_sha256']==sha(policy_path)
    assert review['native_score_sha256']==sha(Z/'native_score.json')
    assert report['state']=='complete_requires_quality_review'
    assert report['checkpoint']==read(Z/'selection.json')['checkpoint']
    assert report['native']['n']==580000
    assert sum(x['n'] for x in report['public'].values())==7366
    assert sum(x['n'] for x in report['general'].values())==10105
    assert set(review['benchmark_assessments'])==set(report['paper_aligned_public'])
    for name,score in report['paper_aligned_public'].items():
        assert math.isfinite(score) and score>=.9*report['paper_target'][name],name
        assert review['benchmark_assessments'][name]['accepted'] is True
    assert set(review['native_group_assessments'])=={x['form']+'::'+x['source'] for x in report['native']['groups']}
    assert all(x['accepted'] is True and x['rationale'] for x in review['native_group_assessments'].values())
    assert set(review['general_assessments'])==set(report['general'])
    assert all(x['accepted'] is True and x['rationale'] for x in review['general_assessments'].values())
    assert native['schema']==policy['native_aggregate_protocol'] and native['state']=='complete' and native['n']==580000
    assert native['report_sha256']==sha(report_path) and native['final_spec_sha256']==sha(spec_path)
    assert math.isfinite(native['score']) and native['score']>=policy['native_minimum']
    scorer=C/'anvil_eval_score_v1_1_20261104'
    assert sha(scorer/'protocol_seal.json')==policy['native_protocol_seal_sha256']
    assert native['scorer_sha256']==sha(scorer/'score_records.py')
    assert native['protocol_sha256']==sha(scorer/'PROTOCOL.md')
    for name,h in read(scorer/'protocol_seal.json')['files'].items():assert sha(scorer/name)==h
    for name,h in report['artifacts_sha256'].items():assert sha(name)==h,'Evaluation output changed'
    return dict(report,anvil_eval_native_score=native['score'])

def plan(api,token):
    report=accepted_report();payload=validated_payload()
    prior=read(Z/'checkpoint_upload/verification_receipt.json')
    assert prior['state']=='complete_remote_verified_selected_checkpoint' and len(prior['key_checkpoints'])==37
    revision=prior['revision'];files={name:{k:v[k] for k in ['bytes','sha256']} for name,v in prior['files'].items()}
    original_manifest=read(Z/'checkpoint_upload/manifest.json')
    seals={};originals={}
    for name in files:
        if name.startswith('checkpoints/') and name.endswith('/checkpoint_complete.json'):
            source=Path(original_manifest['files'][name]['source'])
            if not source.is_file() or sha(source)!=files[name]['sha256']:
                source=Path(hf_hub_download(REPO,name,revision=revision,token=token))
            data=source.read_bytes();assert digest(data)==files[name]
            originals[name]=data
            seals[name.removeprefix('checkpoints/').removesuffix('/checkpoint_complete.json')]=json.loads(data)
    files.update({name:{k:item[k] for k in ['bytes','sha256']} for name,item in payload.items()})
    scorer=C/'anvil_eval_score_v1_1_20261104'
    documents={
        'docs/evaluation/evaluation_report.json':Z/'evaluation/final_report.json',
        'docs/evaluation/evaluation_spec.json':Z/'evaluation_spec.json',
        'docs/evaluation/quality_review.json':Z/'quality_review.json',
        'docs/evaluation/anvil_eval_score.json':Z/'native_score.json',
        'docs/evaluation/ANVIL_EVAL_PROTOCOL.md':scorer/'PROTOCOL.md',
        'docs/evaluation/score_records.py':scorer/'score_records.py',
        'docs/evaluation/scoring_protocol_seal.json':scorer/'protocol_seal.json',
    }
    pipeline=read(C/'visual_geometry_candidate_20261104/spec.json')
    mapping,added,expected=build_plan(files,report,read(Z/'evaluation_spec.json')['representation'],
        revision,seals,originals,set(pipeline['retained_nonancestor_checkpoints']),
        score_artifacts={name:p.read_bytes() for name,p in documents.items()})
    assert not any(name.startswith(('backup/','evidence/')) for name in expected)
    assert sum(name.startswith('checkpoints/') and name.endswith('/adapter_model.safetensors') for name in expected)==37
    preview=Z/'final_release/preview';preview.mkdir(parents=True,exist_ok=True)
    for name,data in added.items():
        path=preview/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    save(Z/'final_release/expected_inventory.json',expected)
    save(Z/'final_release/layout_map.json',mapping)
    save(Z/'final_release/plan.json',dict(parent_revision=revision,files=len(expected),checkpoints=37,
        featured_stages=5,quality_review_sha256=sha(Z/'quality_review.json'),
        expected_inventory_sha256=sha(Z/'final_release/expected_inventory.json')))
    return prior,payload,mapping,added,expected

def apply():
    out=Z/'final_release';out.mkdir(exist_ok=True)
    with (C/'hub_writer.lock').open('a') as writer:
        fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        api,token=auth();prior,payload,mapping,added,expected=plan(api,token)
        parent=prior['revision'];committed=out/'commit.json'
        before={x.rfilename:x for x in api.model_info(REPO,revision=parent,files_metadata=True).siblings}
        assert set(before)==set(prior['files'])
        if not committed.exists():
            assert not (out/'commit_intent.json').exists(),'Reconcile ambiguous publication before retry'
            assert api.model_info(REPO).sha==parent,'HF head changed'
            operations=[CommitOperationCopy(src_path_in_repo=old,path_in_repo=new,src_revision=parent)
                for old,new in mapping.items() if old in prior['files'] and old!=new and new not in added and new not in payload]
            operations.extend(CommitOperationAdd(path_in_repo=name,path_or_fileobj=item['source'])
                for name,item in payload.items() if name not in added)
            operations.extend(CommitOperationAdd(path_in_repo=name,path_or_fileobj=data) for name,data in added.items())
            operations.extend(CommitOperationDelete(path_in_repo=old) for old,new in mapping.items()
                if old in prior['files'] and old!=new and old not in expected)
            save(out/'commit_intent.json',dict(parent_revision=parent,operations=len(operations),
                plan_sha256=sha(out/'plan.json'),time=time.time()))
            result=api.create_commit(REPO,operations=operations,parent_commit=parent,num_threads=4,
                commit_message='Release selected Anvil visual geometry model and five featured training stages')
            save(committed,dict(revision=result.oid,parent_revision=parent,time=time.time()))
        revision=read(committed)['revision']
        remote={x.rfilename:x for x in api.model_info(REPO,revision=revision,files_metadata=True).siblings}
        assert set(remote)==set(expected),'Remote inventory mismatch'
        reverse={new:old for old,new in mapping.items()}
        for name,item in expected.items():
            meta=remote[name];assert meta.size==item['bytes'],name
            if meta.lfs:assert meta.lfs.sha256==item['sha256'],name
            elif name in added or name in payload:
                assert sha(hf_hub_download(REPO,name,revision=revision,token=token))==item['sha256'],name
            else:
                old=reverse[name]
                assert old in before and meta.blob_id==before[old].blob_id,name
        assert api.model_info(REPO).sha==revision
        save(out/'verification_receipt.json',dict(state='accepted_selected_model_and37_checkpoints_remotely_verified',
            repository=REPO,revision=revision,parent_revision=parent,files=expected,checkpoints=37,
            featured_stages=5,selected_step=7168,final_spec_sha256=sha(Z/'evaluation_spec.json'),
            quality_review_sha256=sha(Z/'quality_review.json'),credentials_modified=False,completed=time.time()))
        pointer=Path('/Knowin/demo/weilinruan/Anvil/03_Delivery/CURRENT_CAMPAIGN.json')
        current=read(pointer);assert current['delivery']==str(Z)
        current.update(updated=time.time(),state='accepted_selected7168_remotely_verified',
            final_quality_accepted=True,final_model_uploaded=True,final_revision=revision)
        save(pointer,current)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if args.apply:apply()
    else:
        api,token=auth();plan(api,token)
