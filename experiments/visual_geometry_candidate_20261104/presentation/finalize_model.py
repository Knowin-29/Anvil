"""Apply the final presentation after acceptance, verified upload and local cleanup.

Default invocation inspects readiness. --wait queues a single postrelease action;
--apply performs it once. Frozen training/release sources remain untouched.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, hf_hub_download
from model_presentation import REPO, build_plan, digest, encode

Q=Path(__file__).resolve().parent
C=Q.parent
J=C/'visual_delivery_control_20261104'
P=C/'visual_geometry_candidate_20261104'
D=C/'delivery_runs/visual_geometry_20261104'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v: raise ValueError(m)
def save(p,v):
    temp=Path(str(p)+'.pending');temp.write_bytes(encode(v));temp.replace(p)

def ready():
    return all(p.exists() for p in [J/'final_release/complete.json',J/'post_release_cleanup/complete.json'])

def acceptance():
    review=read(J/'quality_review.json');report=read(D/'evaluation/final_report.json')
    sys.path.insert(0,str(J));from quality_gate import quality_decision
    require(quality_decision(review,report,sha(D/'evaluation/final_report.json'),sha(D/'evaluation_spec.json')),
            'Candidate not accepted')
    require(review.get('user_acceptance_policy_sha256')==sha(Q/'user_acceptance_policy.json'),
            'Review must apply the user 10 percent acceptance policy')
    for key,value in report['paper_aligned_public'].items():
        require(value>=.90*report['paper_target'][key], 'Public benchmark exceeds tolerance: '+key)
    native=review.get('paper_native_tolerance_assessment',{})
    require(native.get('paper_score')==65.39 and native.get('scoring_protocol_verified') is True
            and native.get('accepted') is True and native.get('measured_score',-1)>=58.815,
            'Native versioned score does not meet the user-designated threshold')
    score_path=C/'visual_native_score_20261104/final_native_score.json'
    score=read(score_path);policy=read(Q/'user_acceptance_policy.json')
    require(score['schema']==policy['native_aggregate_protocol'] and score['state']=='complete' and score['n']==580000,
            'Incomplete versioned native score')
    require(score['report_sha256']==sha(D/'evaluation/final_report.json') and score['final_spec_sha256']==sha(D/'evaluation_spec.json'),
            'Native score binds different model or report')
    require(score['score']==native['measured_score'] and native.get('score_artifact')==str(score_path)
            and native.get('score_artifact_sha256')==sha(score_path), 'Review changes computed native score')
    require(native.get('scoring_reference')=='anvil-eval-record-score-v1.1', 'Different native scoring version')
    source=C/'anvil_eval_score_v1_1_20261104'
    require(sha(source/'protocol_seal.json')==policy['native_protocol_seal_sha256'], 'Protocol seal changed')
    seal=read(source/'protocol_seal.json')
    require(all(sha(source/n)==h for n,h in seal['files'].items()), 'Sealed scorer changed')
    require(score['scorer_sha256']==sha(source/'score_records.py') and score['protocol_sha256']==sha(source/'PROTOCOL.md'),
            'Score used different definitions')
    report={**report,'anvil_eval_native_score':score['score']}
    return report

def verify_remote(api,token,revision,expected):
    info=api.model_info(REPO,revision=revision,files_metadata=True)
    remote={f.rfilename:f for f in info.siblings}
    require(set(remote)==set(expected),'Remote inventory differs')
    for name,item in expected.items():
        meta=remote[name];require(meta.size==item['bytes'],'Remote size differs: '+name)
        if meta.lfs: actual=meta.lfs.sha256
        else:
            file=hf_hub_download(REPO,name,revision=revision,token=token)
            actual=sha(file)
        require(actual==item['sha256'],'Remote hash differs: '+name)
    return len(remote)

def apply():
    require(ready(),'Wait for accepted release and verified local cleanup')
    report=acceptance()
    release=read(J/'final_release/complete.json')
    cleanup=read(J/'post_release_cleanup/complete.json')
    require(release['state']=='accepted_final_model_and39_checkpoints_remotely_verified','Wrong release state')
    require(cleanup['state']=='eligible_local_weight_payloads_removed' and cleanup['revision']==release['revision'],
            'Cleanup is not complete for this release')
    whole=read(J/'final_release/whole_hub_verification.json')
    require(whole['revision']==release['revision'] and len(whole['key_checkpoints'])==39,'Wrong verified inventory')
    with ExitStack() as stack:
        for p in [C/'hub_writer.lock',J/'final_release/worker.lock',P/'controller.lock',D/'supervisor.lock',
                  C/'visual_checkpoint_backup_20261104/worker.lock',
                  C/'verification_20261101/visual_merged_blob_preupload/worker.lock']:
            lock=stack.enter_context(p.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        token=runpy.run_path(str(C/'verification_20261101/preupload_merged_blobs.py'))['authorized_token']()
        api=HfApi(token=token);require(api.whoami()['name']=='Knowin-29','Wrong account')
        revision=release['revision']
        files={name:{k:item[k] for k in ['bytes','sha256']} for name,item in whole['files'].items()}
        committed=Q/'model_commit.json'
        if not committed.exists():
            require(not (Q/'model_commit_intent.json').exists(),'Reconcile uncertain prior commit')
            require(api.model_info(REPO).sha==revision,'Hub changed after accepted publication')
            verify_remote(api,token,revision,files)
            seals={};originals={}
            for name in files:
                if name.startswith('checkpoints/') and name.endswith('/checkpoint_complete.json'):
                    path=hf_hub_download(REPO,name,revision=revision,token=token)
                    originals[name]=Path(path).read_bytes()
                    seals[name.removeprefix('checkpoints/').removesuffix('/checkpoint_complete.json')]=json.loads(originals[name])
            spec=read(D/'evaluation_spec.json');pipeline=read(P/'spec.json')
            mapping,added,expected=build_plan(files,report,spec['representation'],revision,seals,originals,
                                            set(pipeline['retained_nonancestor_checkpoints']),
                                            score_artifacts={
                                                'docs/evaluation/anvil_eval_score.json':(C/'visual_native_score_20261104/final_native_score.json').read_bytes(),
                                                'docs/evaluation/ANVIL_EVAL_PROTOCOL.md':(C/'anvil_eval_score_v1_1_20261104/PROTOCOL.md').read_bytes(),
                                                'docs/evaluation/score_records.py':(C/'anvil_eval_score_v1_1_20261104/score_records.py').read_bytes(),
                                                'docs/evaluation/scoring_protocol_seal.json':(C/'anvil_eval_score_v1_1_20261104/protocol_seal.json').read_bytes()})
            preview=Q/'model_preview';preview.mkdir(exist_ok=True)
            for name,data in added.items():
                path=preview/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            save(Q/'model_expected_inventory.json',expected)
            save(Q/'model_layout_map.json',mapping)
            operations=[]
            for old,new in mapping.items():
                if old!=new and new not in added:
                    operations.append(CommitOperationCopy(src_path_in_repo=old,path_in_repo=new,src_revision=revision))
            operations.extend(CommitOperationAdd(path_in_repo=name,path_or_fileobj=data) for name,data in added.items())
            # Remove a source path only when its destination is present in this atomic commit.
            operations.extend(CommitOperationDelete(path_in_repo=old) for old,new in mapping.items()
                              if old!=new and old not in expected)
            require(not any(n.startswith(('backup/','evidence/')) for n in expected),'Unexpected root metadata')
            require(sum(n.endswith('/adapter_model.safetensors') and n.startswith('checkpoints/') for n in expected)==39,
                    'Checkpoint inventory loss')
            save(Q/'model_commit_intent.json',{'parent_revision':revision,'operations':len(operations),
                'expected_inventory_sha256':sha(Q/'model_expected_inventory.json'),
                'quality_review_sha256':sha(J/'quality_review.json'),'policy_sha256':sha(Q/'user_acceptance_policy.json')})
            commit=api.create_commit(REPO,operations=operations,parent_commit=revision,
                commit_message='Organize Anvil release around final inference and five featured training stages',num_threads=8)
            save(committed,{'revision':commit.oid,'parent_revision':revision})
        new_revision=read(committed)['revision'];expected=read(Q/'model_expected_inventory.json')
        count=verify_remote(api,token,new_revision,expected)
        require(api.model_info(REPO).sha==new_revision,'Head moved after presentation verification')
        save(Q/'model_complete.json',{'state':'accepted_model_presentation_remotely_verified',
            'repository':REPO,'revision':new_revision,'parent_revision':revision,'files':count,
            'checkpoints':39,'featured_stages':5,'weights_and_training_state_unchanged':True,
            'credentials_modified':False,'completed':time.time()})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--wait',action='store_true');mode.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if not args.wait and not args.apply:
        print(json.dumps({'ready':ready(),'state':'prepared_postrelease_presentation','model_repo_modified':False}));return
    lock=(Q/'model_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (Q/'model_complete.json').exists(): return
    if args.wait:
        # This bounds the presentation worker only; it never changes a GPU task deadline.
        expiry=datetime.fromisoformat(read(P/'spec.json')['deadline_utc']).timestamp()+86400
        while not ready() and time.time()<expiry:
            if (J/'quality_review.json').exists() and read(J/'quality_review.json').get('accepted') is False:
                save(Q/'model_status.json',{'stage':'waiting_for_next_accepted_candidate','updated':time.time()});return
            save(Q/'model_status.json',{'stage':'waiting_accepted_release_and_cleanup','updated':time.time(),
                'final_model_published_by_this_worker':False})
            time.sleep(45)
        require(ready(),'Postrelease prerequisites unavailable before presentation worker expiry')
    apply()
    save(Q/'model_status.json',{'stage':'presentation_complete','updated':time.time()})

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        save(Q/'model_failure.json',{'error_type':type(exc).__name__,'failed':time.time(),
            'action':'Inspect prerequisites and remote state before retry'})
        print('Model presentation stopped: '+type(exc).__name__,flush=True)
        raise SystemExit(1)
