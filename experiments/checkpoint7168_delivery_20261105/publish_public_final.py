"""Publish the evaluated 7168 under the user's revised seven-benchmark gate.

Preserves the historical native-gate rejection and the already organized 37
checkpoints. No training submissions, scoring changes, or credential changes.
"""
import argparse
import fcntl
import math
import time
from pathlib import Path
from huggingface_hub import CommitOperationAdd, hf_hub_download
from release_io import Z, C, REPO, read, save, sha, auth, validated_payload
from model_presentation import render, encode, digest

OUT = Z/'final_release'
POLICY = Z/'public_acceptance_policy.json'
REVIEW = Z/'public_quality_review.json'
BENCHMARKS = {'cv_bench', 'embspatial', 'refspatial_bench', 'where2place',
              'blink_relative_depth', 'erqa', 'vabench_point'}


def decision():
    report = read(Z/'evaluation/final_report.json')
    spec_sha = sha(Z/'evaluation_spec.json')
    assert report['final_spec_sha256'] == spec_sha
    assert report['checkpoint'] == read(Z/'selection.json')['checkpoint']
    assert report['state'] == 'complete_requires_quality_review'
    assert report['native']['n'] == 580000
    assert sum(v['n'] for v in report['public'].values()) == 7366
    assert sum(v['n'] for v in report['general'].values()) == 10105
    assert set(report['paper_aligned_public']) == BENCHMARKS
    for path, expected in report['artifacts_sha256'].items():
        assert sha(path) == expected, 'Changed evaluated artifact: '+path
    mean = sum(report['paper_aligned_public'].values()) / 7
    assert math.isclose(mean, report['paper_aligned_macro'], abs_tol=1e-10)
    policy = dict(schema='anvil-public-final-acceptance-v1', selected_step=7168,
        user_instruction='最终checkpoint在CV-Bench、EmbSpatial、RefSpatial、Where2Place、BLINK depth、ERQA、VABench-Point几个benchmark都得到相对接近的成绩即可（mean在68以上）。最后一天，最后5次提交dlc，训练步数不需要太多。',
        benchmarks=sorted(BENCHMARKS), public_mean_strictly_greater_than=68.0,
        maximum_per_benchmark_relative_difference=0.10,
        interpretation='Retain the earlier 10% per-benchmark tolerance as the concrete meaning of relatively close; use the latest public-only acceptance scope.',
        superseded_policy_review_sha256=sha(Z/'quality_review.json'),
        native_is_final_acceptance_gate=False,
        native_results='Preserved unchanged in the measured report and historical review.',
        new_dlc_submissions=0, final_spec_sha256=spec_sha)
    save(POLICY, policy)
    checks = {}
    for name, value in report['paper_aligned_public'].items():
        target = report['paper_target'][name]
        difference = abs(value-target)/target
        checks[name] = dict(score=value, paper_reference=target,
            relative_difference=difference,
            accepted=math.isfinite(value) and difference <= 0.10)
    review = dict(schema='anvil-public-final-quality-review-v1',
        accepted=mean>68 and all(x['accepted'] for x in checks.values()),
        acceptance_scope='Seven named public benchmarks under the latest user instruction',
        rationale='Full existing evaluation exceeds mean 68 and all seven scores are within 10% of their paper references. Release the evaluated checkpoint without new training or DLC submissions.',
        selected_step=7168, public_mean=mean, benchmark_assessments=checks,
        report_sha256=sha(Z/'evaluation/final_report.json'), final_spec_sha256=spec_sha,
        user_acceptance_policy_sha256=sha(POLICY),
        historical_native_quality_review_sha256=sha(Z/'quality_review.json'),
        native_score_sha256=sha(Z/'native_score.json'),
        native_score=read(Z/'native_score.json')['score'], native_gate_passed=False,
        native_is_final_acceptance_gate=False, new_dlc_submissions=0)
    assert review['accepted']
    save(REVIEW, review)
    return report


def plan():
    report = decision()
    payload = validated_payload()
    prior = read(Z/'checkpoint_collection/complete.json')
    assert prior['state'] == 'selected_checkpoint_collection_remotely_verified'
    assert prior['checkpoint_count'] == 37
    parent = prior['revision']
    added = render(report, read(Z/'evaluation_spec.json')['representation'], parent, [])
    added.update({
        'docs/evaluation/public_acceptance_policy.json': POLICY.read_bytes(),
        'docs/evaluation/quality_review.json': REVIEW.read_bytes(),
        'docs/evaluation/historical_native_quality_review.json': (Z/'quality_review.json').read_bytes(),
        'docs/evaluation/release_status.json': encode(dict(
            state='accepted_under_revised_public_benchmark_policy', selected_step=7168,
            evaluation_report='evaluation_report.json', report_sha256=sha(Z/'evaluation/final_report.json'),
            quality_review='quality_review.json', native_gate_passed=False,
            note='The immutable inference-time report retains its original pre-publication state flags. The separate release review records the subsequent acceptance decision.')),
    })
    native_page = added['docs/evaluation/README.md'].decode()
    native_page += '\n## Release acceptance\n\nThe final release uses the seven-benchmark public acceptance policy: mean **72.2729%**, with all seven scores within 10% of the manuscript reference. The native aggregate is **50.623671** under Anvil-Eval v1.1. Native metrics are not part of the revised final gate.\n\n[Current release review](quality_review.json) · [Acceptance policy](public_acceptance_policy.json) · [Historical native review](historical_native_quality_review.json) · [Native scoring protocol](ANVIL_EVAL_PROTOCOL.md)\n'
    added['docs/evaluation/README.md'] = native_page.encode()
    expected = {name:dict(item) for name,item in prior['files'].items() if name!='release_manifest.json'}
    expected.update({name:{k:item[k] for k in ['bytes','sha256']} for name,item in payload.items()})
    expected.update({name:digest(data) for name,data in added.items()})
    manifest = dict(schema='anvil-final-public-release-v1', repository=REPO,
        source_revision=parent, selected_step=7168, checkpoint_count=37,
        featured_stages=5, acceptance_policy_sha256=sha(POLICY),
        quality_review_sha256=sha(REVIEW), final_spec_sha256=sha(Z/'evaluation_spec.json'),
        files=dict(expected))
    added['release_manifest.json'] = encode(manifest)
    expected['release_manifest.json'] = digest(added['release_manifest.json'])
    assert sum(n.startswith('checkpoints/') and n.endswith('/adapter_model.safetensors') for n in expected)==37
    assert not any(n.startswith(('backup/','evidence/')) for n in expected)
    for name,data in added.items():
        p=OUT/'preview'/name; p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    save(OUT/'expected_inventory.json', expected)
    save(OUT/'plan.json', dict(parent_revision=parent, files=len(expected), checkpoints=37,
        featured_stages=5, quality_review_sha256=sha(REVIEW),
        expected_inventory_sha256=sha(OUT/'expected_inventory.json')))
    return prior,payload,added,expected


def apply():
    with (C/'hub_writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior,payload,added,expected=plan()
        api,token=auth(); parent=prior['revision']
        before={x.rfilename:x for x in api.model_info(REPO,revision=parent,files_metadata=True).siblings}
        assert set(before)==set(prior['files'])
        committed=OUT/'commit.json'
        if not committed.exists():
            assert not (OUT/'commit_intent.json').exists(), 'Reconcile ambiguous publication before retry'
            assert api.model_info(REPO).sha==parent, 'HF head changed'
            operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=v['source'])
                for n,v in payload.items() if n not in added]
            operations += [CommitOperationAdd(path_in_repo=n,path_or_fileobj=data) for n,data in added.items()]
            save(OUT/'commit_intent.json', dict(parent_revision=parent,operations=len(operations),
                plan_sha256=sha(OUT/'plan.json'),time=time.time()))
            result=api.create_commit(REPO,operations=operations,parent_commit=parent,num_threads=4,
                commit_message='Release evaluated Anvil checkpoint7168 merged model with 72.27 public benchmark mean')
            save(committed,dict(revision=result.oid,parent_revision=parent,time=time.time()))
        revision=read(committed)['revision']
        remote={x.rfilename:x for x in api.model_info(REPO,revision=revision,files_metadata=True).siblings}
        assert set(remote)==set(expected), 'Remote inventory mismatch'
        for name,item in expected.items():
            meta=remote[name]; assert meta.size==item['bytes'], name
            if meta.lfs:
                assert meta.lfs.sha256==item['sha256'], name
            elif name in added or name in payload:
                assert sha(hf_hub_download(REPO,name,revision=revision,token=token))==item['sha256'], name
            else:
                assert meta.blob_id==before[name].blob_id, name
        assert api.model_info(REPO).sha==revision
        save(OUT/'verification_receipt.json', dict(
            state='accepted_selected_model_and37_checkpoints_remotely_verified',
            repository=REPO,revision=revision,parent_revision=parent,files=expected,
            checkpoints=37,featured_stages=5,selected_step=7168,
            acceptance_scope='public_benchmarks',final_spec_sha256=sha(Z/'evaluation_spec.json'),
            quality_review_sha256=sha(REVIEW),user_acceptance_policy_sha256=sha(POLICY),
            credentials_modified=False,completed=time.time()))
        pointer=Path('/workspace/anvil-project/03_Delivery/CURRENT_CAMPAIGN.json')
        current=read(pointer);assert current['delivery']==str(Z)
        current.update(updated=time.time(),state='accepted_selected7168_remotely_verified',
            final_quality_accepted=True, final_model_uploaded=True,final_revision=revision,
            acceptance_scope='public_benchmarks',public_mean=read(REVIEW)['public_mean'],
            quality_review=str(REVIEW),publication_script=str(Path(__file__).resolve()),
            publication_prerequisites=['full_evaluation','public_mean_above_68','each_public_score_within_10_percent'],
            next_compute_resources='No new compute required; evaluated7168 meets latest user gate',
            native_is_final_acceptance_gate=False)
        save(pointer,current)
        print('Verified final revision:',revision, 'files:',len(expected),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if args.apply: apply()
    else:
        prior,payload,added,expected=plan()
        print('Prepared final publication:',len(expected),'files;',len(payload),'payload entries')
