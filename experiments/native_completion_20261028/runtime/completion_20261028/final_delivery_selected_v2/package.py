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
    require(lineage['checkpoint']==spec['checkpoint'] and lineage['selected_step']==spec['selected_step']
            and lineage['planned_steps']==spec['planned_steps'] and lineage['later_training_rows_counted'] is False,
            'Lineage does not describe the exact selected prefix')
    for item in lineage['evidence']:check_binding(item)
    from selection import observe,validate
    selected,_=validate(spec['evidence']['checkpoint_selection']['path'])
    current=observe(selected['cloud']['task']['Id'])
    require(current['task']['Status']==selected['cloud']['task']['Status'],
            'Cloud run disposition changed after checkpoint selection')
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
        require(sha(adapter/name)==sha(Path(spec['checkpoint'])/name),'Copied adapter changed')
    copy_inference_interface(out/'anvil',spec['representation'])
    for name,digest in validation['standalone_files_sha256'].items():
        require(sha(out/'anvil'/name)==digest,'Packaged interface differs from tested interface')
    evidence=out/'evidence';evidence.mkdir()
    for name,path in [('evaluation_spec.json',args.spec),('evaluation_report.json',args.report),
                      ('quality_review.json',args.review),('training_lineage.json',args.lineage),
                      ('interface_validation.json',args.interface_validation),
                      ('training_protocol.json',Path(spec['training_run'])/'protocol.json'),
                      ('checkpoint_selection.json',Path(spec['evidence']['checkpoint_selection']['path']))]:
        shutil.copyfile(path,evidence/name)
    save(evidence/'terminal_cloud_recheck.json',current)
    scores='\n'.join(f"| {name} | {score:.4f} | {report['paper_target'][name]:.2f} | {report['final_minus_paper_percentage_points'][name]:+.4f} |" for name,score in report['paper_aligned_public'].items())
    card=f'''---
base_model: Qwen/Qwen3.5-9B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- anvil
- mopd
---

# Anvil MOPD 9B

This release contains the exact selected MOPD checkpoint evaluated in `evidence/evaluation_report.json`, as merged BF16 weights and the original LoRA adapter. All seven public benchmarks, 580,000 native examples and 10,105 general examples use this one model. The historical paper table is a comparison target; it is not substituted for the measured release scores.

The selected checkpoint completed **{spec['selected_step']} updates** from a plan of **{spec['planned_steps']} updates**. The original planned run completed: **{spec['original_planned_run_completed']}**. Its disposition is `{spec['budget_disposition']}`. Only the selected checkpoint's committed exposure prefix enters its model lineage; later updates are excluded. Selection and terminal cloud evidence are included in the package, without fabricating a completed original training budget.

| Benchmark | This checkpoint | Paper target | Difference (pp) |
| --- | ---: | ---: | ---: |
{scores}

RefSpatial paper comparison uses Location100; the complete 277-example score is also in the report. Historical inference settings are not claimed identical. See the quality review for limitations.

The campaign used validation results to select repairs. These reported validation sets are not a new blind test of the selected model.

The original corpus contains 5,800,000 records and 160,000 images, with 5,220,000 training records. Its source mixture differs from the paper description. Supplemental QA/spatial inventories, actual unique exposures, repetitions and model initialization history are disclosed separately in `evidence/training_lineage.json`. Experimental control arms are not added to this model's exposure totals.

## Inference

Load the merged model and processor with Transformers `Qwen3_5ForConditionalGeneration` and `AutoProcessor`, using BF16 and thinking disabled. Evaluation used up to 1,536 image tokens. Install the model-compatible Transformers/Qwen3.5 and FLA versions recorded in the evaluation protocols.

Native DoF representation: `{spec['representation']}`. When the representation is not `native`, raw model output is a latent pose representation. Use `anvil/anvil_interface.py`: call `prepare_native_prompt("dof", original_prompt)` before generation, then `decode_native_response("dof", generated_text, original_prompt)`. The prompt contains the image entity and calibration, not a target answer. Invalid generated values raise an error. Other native forms and public benchmark prompts are unchanged. Loading the model alone does not apply this decoder automatically.

The exact files and SHA256 values are in `release_manifest.json`. Upload verification pins a specific Hub commit; it is performed after this package is created.
'''
    (out/'README.md').write_text(card)
    files={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()}
    save(out/'release_manifest.json',dict(schema='anvil-sealed-final-release-v1',repository='Knowin-29/Anvil-MOPD-9B',
        quality_review_accepted=True,final_spec_sha256=sha(args.spec),review_sha256=sha(args.review),
        selected_step=spec['selected_step'],planned_steps=spec['planned_steps'],
        original_planned_run_completed=spec['original_planned_run_completed'],
        files=files,created=time.time(),remote_verified=False))
    print(out)


if __name__=='__main__':main()
