"""Build a standalone final-model package after full evaluation and quality review."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import time
from common import D,HERE,bound,check_binding,load_spec,read,require,save,sha


def copy_inference_interface(output,mode):
    """Keep exact inference functions; omit the codebook fitting dependency."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    files=['dof_codec_interface.py','dof_codec_prototype.py'] if mode!='native' else []
    for name in files:shutil.copyfile(D/name,output/name)
    if mode=='rotation128':
        source=(D/'dof_codebook_interface.py').read_text()
        tree=ast.parse(source);lines=source.splitlines(keepends=True);remove=set()
        for node in tree.body:
            if (isinstance(node,ast.ImportFrom) and node.module=='prepare_dof_rotation_codebook') or (
                isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in {'encode_target','training_row'}):
                remove.update(range(node.lineno-1,node.end_lineno))
        (output/'dof_codebook_interface.py').write_text(''.join(line for i,line in enumerate(lines) if i not in remove))
        shutil.copyfile(D/'dof_rotation_codebook.json',output/'dof_rotation_codebook.json')
    save(output/'anvil_representation.json',dict(mode=mode,decoder_required=mode!='native',
        source_inference_files_sha256={name:sha(D/name) for name in files+(['dof_codebook_interface.py','dof_rotation_codebook.json'] if mode=='rotation128' else [])}))
    (output/'anvil_interface.py').write_text('''"""Native Anvil prompt/output adapter; no scoring code or reference answers."""
import importlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
MODE=json.loads((ROOT/'anvil_representation.json').read_text())['mode']
MODULE=importlib.import_module({'rotation128':'dof_codebook_interface','rotation6d':'dof_codec_interface'}[MODE]) if MODE!='native' else None

def prepare_native_prompt(form,prompt):
    return MODULE.model_prompt(prompt) if form=='dof' and MODULE else prompt

def decode_native_response(form,response,original_prompt):
    # Invalid latent outputs raise an error; no answer is invented.
    return MODULE.decode_response(response,original_prompt) if form=='dof' and MODULE else response
''')


def main():
    ap=argparse.ArgumentParser()
    for name in ['spec','report','review','lineage','interface-validation','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();spec=load_spec(args.spec);report=read(args.report);review=read(args.review)
    require(report['state']=='complete_requires_quality_review' and report['final_spec_sha256']==sha(args.spec),
            'Full final evaluation is missing or belongs to another model')
    require(report['native']['n']==580000 and sum(g['n'] for g in report['general'].values())==10105,
            'Final evaluation is incomplete')
    for path,digest in report['artifacts_sha256'].items():require(sha(path)==digest,'Evaluation output changed')
    require(review['schema']=='anvil-final-quality-review-v1' and review['accepted'] is True
            and review['report_sha256']==sha(args.report) and review['final_spec_sha256']==sha(args.spec),
            'An evidence-backed final-quality review is required')
    require(bool(review.get('rationale')) and set(review['benchmark_assessments'])==set(report['paper_aligned_public']),
            'Review must assess every public benchmark; no automatic invented tolerance')
    lineage=read(args.lineage)
    require(lineage['schema']=='anvil-final-lineage-audit-v1' and lineage['state']=='complete'
            and lineage['final_spec_sha256']==sha(args.spec), 'Final model lineage has not been audited')
    require(lineage['initial_sft']['training_records']==5220000 and lineage['other_experiment_arms_counted_as_final_training'] is False,
            'Incorrect original scale or pooled experimental exposure counts')
    for item in lineage['evidence']:check_binding(item)
    validation=read(args.interface_validation)
    require(validation['state']=='passed' and validation['mode']==spec['representation']
            and validation['package_builder_sha256']==sha(__file__), 'Standalone interface parity validation missing')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name,digest in spec['model_files'].items():
        if name=='README.md':continue
        require(Path(name).name==name,'Unsafe source filename')
        shutil.copyfile(Path(spec['model'])/name,out/name)
        require(sha(out/name)==digest,'Copied model changed')
    adapter=out/'adapter';adapter.mkdir()
    for name in ['adapter_model.safetensors','adapter_config.json']:
        shutil.copyfile(Path(spec['checkpoint'])/name,adapter/name)
    copy_inference_interface(out/'anvil',spec['representation'])
    for name,digest in validation['standalone_files_sha256'].items():
        require(sha(out/'anvil'/name)==digest,'Packaged interface differs from tested interface')
    evidence=out/'evidence';evidence.mkdir()
    pipeline=Path(spec['training_run']).parent.parent
    pipeline_spec=read(pipeline/'spec.json')
    from common import verify
    key_records=[]
    for relative in pipeline_spec['key_checkpoints']:
        original=Path(pipeline_spec['key_checkpoint_sources'][relative])
        require(original.is_dir(),'Required key checkpoint missing: '+relative)
        receipt=verify(original,pipeline_spec['world'])
        destination=out/'checkpoints'/relative
        destination.mkdir(parents=True,exist_ok=False)
        for name,item in receipt['files'].items():
            shutil.copyfile(original/name,destination/name)
            require(sha(destination/name)==item['sha256'],'Copied checkpoint differs')
        shutil.copyfile(original/'checkpoint_complete.json',destination/'checkpoint_complete.json')
        shutil.copyfile(original.parent/'protocol.json',destination/'training_protocol.json')
        key_records.append(dict(path=relative,step=receipt['global_step'],
            in_final_optimizer_ancestry=relative not in pipeline_spec['retained_nonancestor_checkpoints'],
            adapter_sha256=receipt['files']['adapter_model.safetensors']['sha256'],
            final_evaluated_checkpoint=original.resolve()==Path(spec['checkpoint']).resolve()))
    save(evidence/'key_checkpoints.json',dict(checkpoints=key_records,
        note='Complete adapters, optimizer, scheduler and rank RNG state. Only the final checkpoint has the release final evaluation.'))
    shutil.copyfile(pipeline/'spec.json',evidence/'corrective_pipeline_spec.json')
    shutil.copyfile(pipeline/'runs/repair/protocol.json',evidence/'corrective_training_protocol.json')
    for name,path in [('evaluation_spec.json',args.spec),('evaluation_report.json',args.report),
                      ('quality_review.json',args.review),('training_lineage.json',args.lineage),
                      ('interface_validation.json',args.interface_validation),
                      ('training_protocol.json',Path(spec['training_run'])/'protocol.json'),
                      ('training_complete.json',Path(spec['training_run'])/'complete.json')]:
        shutil.copyfile(path,evidence/name)
    scores='\n'.join(f"| {name} | {score:.2f} |" for name,score in report['paper_aligned_public'].items())
    card=f'''---
base_model: Qwen/Qwen3.5-9B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- anvil
- mopd
- spatial-reasoning
---

# Anvil MOPD 9B

**Structured spatial understanding in a 9B vision-language model.** Anvil connects visual perception, spatial relations and action-oriented prediction through 11 native task forms, including boxes, masks, points, affordances and poses.

[Paper](https://arxiv.org/abs/2610.21882) · [Code](https://github.com/Knowin-29/Anvil) · [Dataset](https://huggingface.co/datasets/Knowin-29/Anvil-Ben)

## Highlights

- A 5.22-million-record SFT foundation followed by targeted spatial training and multi-teacher on-policy distillation.
- Joint visual and language adaptation, delivered as ready-to-load BF16 weights and a LoRA adapter.
- Complete evaluation on 580,000 native spatial records, seven public benchmarks and 10,105 general-ability examples.
- Key training stages in [checkpoints/](checkpoints/), with optimizer, scheduler and distributed RNG states for continued training.

## Benchmark results

| Benchmark | Score |
| --- | ---: |
{scores}

RefSpatial reports Location100. [Detailed native, public and general results](evidence/evaluation_report.json) · [Training lineage](evidence/training_lineage.json)

## Quick start

Load the repository with Transformers `Qwen3_5ForConditionalGeneration` and `AutoProcessor`, using BF16 and thinking disabled. Evaluation uses up to 1,536 image tokens.

```python
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

repo = "Knowin-29/Anvil-MOPD-9B"
processor = AutoProcessor.from_pretrained(repo)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    repo, torch_dtype=torch.bfloat16, device_map="auto"
)
```

For native DoF tasks, use the supplied `anvil/anvil_interface.py`: call `prepare_native_prompt("dof", original_prompt)` before generation and `decode_native_response("dof", generated_text, original_prompt)` to obtain the native pose output. The deployed representation is `{spec['representation']}`.

## Repository

- Root model shards and processor assets: full-model inference.
- `adapter/`: final LoRA weights.
- `checkpoints/`: key training stages and recovery state.
- `anvil/`: native pose interface.
- `release_manifest.json`: file inventory and SHA256 checksums.
'''
    (out/'README.md').write_text(card)
    files={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()}
    save(out/'release_manifest.json',dict(schema='anvil-sealed-final-release-v1',repository='Knowin-29/Anvil-MOPD-9B',
        quality_review_accepted=True,final_spec_sha256=sha(args.spec),review_sha256=sha(args.review),
        files=files,created=time.time(),remote_verified=False))
    print(out)


if __name__=='__main__':main()
