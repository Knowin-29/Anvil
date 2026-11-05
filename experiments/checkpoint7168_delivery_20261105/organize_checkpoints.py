"""Organize the uploaded checkpoint collection and publish measured evaluation.

This is independent of final-model acceptance: it does not publish root merged
weights, change quality decisions, delete local files or schedule training.
"""
import argparse,fcntl,json,time
from pathlib import Path
from huggingface_hub import CommitOperationAdd,CommitOperationCopy,CommitOperationDelete,hf_hub_download
from release_io import Z,C,REPO,read,save,sha,auth
import model_presentation as presentation

OUT=Z/'checkpoint_collection'

def build():
    prior=read(Z/'checkpoint_upload/verification_receipt.json')
    assert prior['state']=='complete_remote_verified_selected_checkpoint' and len(prior['key_checkpoints'])==37
    manifest=read(Z/'checkpoint_upload/manifest.json')
    assert sha(Z/'checkpoint_upload/manifest.json')==prior['manifest_sha256']
    files={n:{k:x[k] for k in ['bytes','sha256']} for n,x in prior['files'].items()}
    originals={};seals={}
    for name in files:
        if name.startswith('checkpoints/') and name.endswith('/checkpoint_complete.json'):
            data=Path(manifest['files'][name]['source']).read_bytes()
            assert presentation.digest(data)==files[name]
            originals[name]=data
            seals[name.removeprefix('checkpoints/').removesuffix('/checkpoint_complete.json')]=json.loads(data)
    report=read(Z/'evaluation/final_report.json');native=read(Z/'native_score.json')
    assert report['final_spec_sha256']==sha(Z/'evaluation_spec.json')
    assert native['report_sha256']==sha(Z/'evaluation/final_report.json') and native['n']==580000
    assert report['native']['n']==580000 and sum(x['n'] for x in report['public'].values())==7366
    assert sum(x['n'] for x in report['general'].values())==10105
    for p,h in report['artifacts_sha256'].items():assert sha(p)==h
    scorer=C/'anvil_eval_score_v1_1_20261104'
    documents={
        'docs/evaluation/evaluation_report.json':Z/'evaluation/final_report.json',
        'docs/evaluation/evaluation_spec.json':Z/'evaluation_spec.json',
        'docs/evaluation/anvil_eval_score.json':Z/'native_score.json',
        'docs/evaluation/quality_review.json':Z/'quality_review.json',
        'docs/evaluation/ANVIL_EVAL_PROTOCOL.md':scorer/'PROTOCOL.md',
        'docs/evaluation/score_records.py':scorer/'score_records.py',
        'docs/evaluation/scoring_protocol_seal.json':scorer/'protocol_seal.json',
        'docs/provenance/checkpoint_upload_manifest.json':Z/'checkpoint_upload/manifest.json',
        'docs/provenance/training_update_audit.json':Z/'training_update_audit.json',
    }
    payload=read(Z/'release_payload/payload_manifest.json')
    for name,item in payload['files'].items():
        if name.startswith('anvil/') or name in {'docs/provenance/training_lineage.json','docs/provenance/completed_parent_mopd_lineage.json','docs/provenance/interface_validation.json'}:
            assert sha(item['source'])==item['sha256'];documents[name]=Path(item['source'])
    pipeline=read(C/'visual_geometry_candidate_20261104/spec.json')
    mapping,added,_=presentation.build_plan(files,dict(report,anvil_eval_native_score=native['score']),
        'rotation128',prior['revision'],seals,originals,set(pipeline['retained_nonancestor_checkpoints']),
        score_artifacts={n:p.read_bytes() for n,p in documents.items()})
    for key in seals:
        name=mapping['checkpoints/'+key+'/README.md'];card=added[name].decode()
        card=card.replace('Use the [root merged model](https://huggingface.co/'+REPO+') for final-model inference.',
            'Use the [selected visual geometry adapter](https://huggingface.co/'+REPO+'/tree/main/checkpoints/visual-geometry) with the Qwen3.5-9B base model for inference.')
        added[name]=card.encode();seal_name=mapping['checkpoints/'+key+'/checkpoint_complete.json']
        seal=json.loads(added[seal_name]);seal['files']['README.md']=presentation.digest(added[name]);added[seal_name]=presentation.encode(seal)
    added['checkpoints/README.md']=added['checkpoints/README.md'].replace(
        b'Release scores apply to the final merged model and selected visual geometry adapter.',
        b'Release scores were measured on the BF16 inference merge of the selected visual geometry adapter.')
    added['docs/evaluation/README.md']=added['docs/evaluation/README.md'].replace(
        b'# Anvil release evaluation',f'# Anvil checkpoint 7168 evaluation\n\n**Anvil-Eval v1.1: {native["score"]:.4f}** on all 580,000 native records.'.encode())
    rows='\n'.join(f'| {presentation.LABELS[k]} | {value:.2f} |' for k,value in report['paper_aligned_public'].items())
    stages='\n'.join(f'| {title} | [{slug}](checkpoints/{slug}) |' for slug,title in presentation.FEATURED.values())
    groups={g['form']:g['mean'] for g in report['native']['groups']}
    card=f'''---
base_model: Qwen/Qwen3.5-9B
library_name: peft
pipeline_tag: image-text-to-text
tags:
- anvil
- mopd
- spatial-reasoning
- embodied-ai
datasets:
- Knowin-29/Anvil-Ben
---

# Anvil-MOPD-9B

### Structured spatial intelligence in a 9B vision-language model.

**11 native spatial forms · 5.22M-record SFT ancestry · specialist-to-generalist MOPD**

[Paper](https://arxiv.org/abs/2610.21882) · [Code](https://github.com/Knowin-29/Anvil) · [Data](https://huggingface.co/datasets/Knowin-29/Anvil-Ben)

Anvil connects scene-aligned spatial supervision and multi-teacher on-policy distillation. The selected **visual geometry checkpoint 7168** extends the completed coordinate MOPD stage with joint visual-language adaptation for structured spatial outputs.

## Release highlights

- **37 complete checkpoints**, with five featured stages and a compact training archive.
- **178.25M adapter parameters**, including 5.14M visual parameters.
- **4.756M-record spatial training pool**; 458,752 sample exposures in the selected repair stage.
- **Complete evaluation:** 580,000 native spatial examples, 7,366 public benchmark examples and 10,105 general-ability examples.
- **{report['paper_aligned_macro']:.2f}% public benchmark mean** across seven benchmarks.

## Measured results

| Public benchmark | Score (%) |
|---|---:|
{rows}

| Native spatial measure | Value |
|---|---:|
| Anvil-Eval v1.1 | {native['score']:.2f} |
| Mask IoU | {groups['mask']['matched_mask_iou']:.4f} |
| Affordance IoU | {groups['affordance']['region_iou']:.4f} |
| DoF rotation error | {groups['dof']['rotation_error_degrees']:.2f}° |

[Full native and general results](docs/evaluation/README.md) · [Evaluation report](docs/evaluation/evaluation_report.json) · [Native scoring protocol](docs/evaluation/ANVIL_EVAL_PROTOCOL.md)

## Load the selected adapter

```python
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

root = snapshot_download(
    "Knowin-29/Anvil-MOPD-9B",
    allow_patterns=["checkpoints/visual-geometry/adapter*", "anvil/*"],
)
base_id = "Qwen/Qwen3.5-9B"
processor = AutoProcessor.from_pretrained(base_id)
base = Qwen3_5ForConditionalGeneration.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(base, root + "/checkpoints/visual-geometry")
```

Use thinking-disabled generation. The packaged `anvil/anvil_interface.py` provides native DoF integration: call `prepare_native_prompt("dof", prompt)` before generation and `decode_native_response("dof", text, prompt)` afterward. The DoF representation is `rotation128`.

## Featured training stages

| Stage | Directory |
|---|---|
{stages}

Each checkpoint contains adapter weights, optimizer, scheduler and all 32 rank RNG states. See the [checkpoint guide](checkpoints/README.md) for continuation and the [archive](checkpoints/archive) for additional saved stages.

## Repository guide

| Path | Contents |
|---|---|
| `checkpoints/` | Five featured stages and 32 archived checkpoints |
| `anvil/` | Native pose prompt and decoding interface |
| `docs/evaluation/` | Measured results and evaluation settings |
| `docs/provenance/` | Training lineage, original stage identities and release records |
| `release_manifest.json` | File checksums and source revision |
'''
    added['README.md']=card.encode();added.pop('release_manifest.json')
    expected={mapping[n]:item for n,item in files.items()}
    expected.update({n:presentation.digest(data) for n,data in added.items()})
    added['release_manifest.json']=presentation.encode(dict(schema='anvil-selected-checkpoint-collection-v1',
        repository=REPO,source_revision=prior['revision'],files=expected,checkpoint_count=37,
        selected_checkpoint='checkpoints/visual-geometry',featured_checkpoints=list(presentation.FEATURED),
        quality_accepted_final_model=False,final_spec_sha256=sha(Z/'evaluation_spec.json')))
    expected['release_manifest.json']=presentation.digest(added['release_manifest.json'])
    assert not any(n.startswith(('backup/','evidence/')) for n in expected)
    assert sum(n.startswith('checkpoints/') and n.endswith('/adapter_model.safetensors') for n in expected)==37
    for n,item in files.items():
        if Path(n).suffix in {'.safetensors','.pt','.pth'}:assert expected[mapping[n]]==item
    OUT.mkdir(exist_ok=True);preview=OUT/'preview';preview.mkdir(exist_ok=True)
    for name,data in added.items():
        p=preview/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    save(OUT/'expected_inventory.json',expected);save(OUT/'layout_map.json',mapping)
    save(OUT/'plan.json',dict(parent_revision=prior['revision'],files=len(expected),checkpoint_count=37,
        featured_stages=5,expected_inventory_sha256=sha(OUT/'expected_inventory.json'),
        report_sha256=sha(Z/'evaluation/final_report.json'),final_model_acceptance_changed=False))
    return prior,mapping,added,expected

def apply():
    with (C/'hub_writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior,mapping,added,expected=build();api,token=auth();parent=prior['revision']
        before={x.rfilename:x for x in api.model_info(REPO,revision=parent,files_metadata=True).siblings}
        assert set(before)==set(prior['files'])
        for n,x in before.items():
            assert x.size==prior['files'][n]['bytes']
            if x.lfs:assert x.lfs.sha256==prior['files'][n]['sha256']
        if not (OUT/'commit.json').exists():
            assert not (OUT/'commit_intent.json').exists(),'Reconcile ambiguous commit before retry'
            assert api.model_info(REPO).sha==parent
            operations=[CommitOperationCopy(src_path_in_repo=old,path_in_repo=new,src_revision=parent)
                for old,new in mapping.items() if old!=new and new not in added]
            operations.extend(CommitOperationAdd(path_in_repo=n,path_or_fileobj=data) for n,data in added.items())
            operations.extend(CommitOperationDelete(path_in_repo=old) for old,new in mapping.items() if old!=new and old not in expected)
            save(OUT/'commit_intent.json',dict(parent_revision=parent,operations=len(operations),plan_sha256=sha(OUT/'plan.json'),time=time.time()))
            result=api.create_commit(REPO,parent_commit=parent,operations=operations,num_threads=8,
                commit_message='Feature five Anvil training stages and publish full checkpoint7168 evaluation')
            save(OUT/'commit.json',dict(revision=result.oid,parent_revision=parent))
        revision=read(OUT/'commit.json')['revision'];remote={x.rfilename:x for x in api.model_info(REPO,revision=revision,files_metadata=True).siblings}
        assert set(remote)==set(expected);reverse={new:old for old,new in mapping.items()}
        for n,item in expected.items():
            x=remote[n];assert x.size==item['bytes'],n
            if x.lfs:assert x.lfs.sha256==item['sha256'],n
            elif n in added:assert sha(hf_hub_download(REPO,n,revision=revision,token=token))==item['sha256'],n
            else:assert x.blob_id==before[reverse[n]].blob_id,n
        assert api.model_info(REPO).sha==revision
        save(OUT/'complete.json',dict(state='selected_checkpoint_collection_remotely_verified',revision=revision,
            parent_revision=parent,files=expected,checkpoint_count=37,featured_stages=5,
            final_quality_accepted=False,root_merged_model_published=False,credentials_modified=False,completed=time.time()))
        print(json.dumps(dict(revision=revision,files=len(expected),checkpoints=37)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');args=p.parse_args()
    if args.apply:apply()
    else:
        build();print(OUT/'preview/README.md')
