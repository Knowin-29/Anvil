"""Render the final Hub layout from the accepted, remotely verified release."""
import hashlib
import json

REPO = 'Knowin-29/Anvil-MOPD-9B'
FEATURED = {
    'ancestor_repair/checkpoint-2048': ('initial-adaptation', 'Initial adaptation'),
    'balanced_mopd/checkpoint-512': ('balanced-mopd', 'Balanced MOPD'),
    'format_mopd/checkpoint-256': ('format-mopd', 'Format-enhanced MOPD'),
    'coordinate_repair/checkpoint-8192': ('geometry-repair', 'Geometry-focused training'),
    'coordinate_mopd/checkpoint-256': ('final-mopd', 'Final MOPD'),
}
LABELS = {'cv_bench':'CV-Bench', 'embspatial':'EmbSpatial', 'refspatial_bench':'RefSpatial · Location100',
          'where2place':'Where2Place', 'blink_relative_depth':'BLINK · Relative Depth',
          'erqa':'ERQA', 'vabench_point':'VABench · Point'}

def encode(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()

def digest(data):
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def destination(name):
    if name.startswith('checkpoints/'):
        parts = name.split('/')
        if len(parts)<4: raise ValueError('Unexpected checkpoint path: '+name)
        key = '/'.join(parts[1:3]); tail='/'.join(parts[3:])
        if key in FEATURED: return 'checkpoints/'+FEATURED[key][0]+'/'+tail
        return 'checkpoints/archive/'+key+'/'+tail
    if name.startswith('evidence/'):
        leaf = name.removeprefix('evidence/')
        if leaf in {'evaluation_report.json','quality_review.json','evaluation_spec.json'}:
            return 'docs/evaluation/'+leaf
        return 'docs/provenance/'+leaf
    if name.startswith('backup/'):
        return 'docs/provenance/backup/'+name.removeprefix('backup/')
    if name == 'release_manifest.json': return 'docs/provenance/original_release_manifest.json'
    return name

def stage_card(key, revision):
    title=FEATURED.get(key,('',key.replace('_',' ').replace('/',' · ')))[1]
    step=key.rsplit('-',1)[1]
    return f'''# Anvil — {title}

**Qwen3.5-9B · LoRA adapter · step {step}**

This stage checkpoint includes the adapter and complete training state for continuation: optimizer, scheduler and all 32 rank RNG states.

| File | Purpose |
|---|---|
| `adapter_model.safetensors` + `adapter_config.json` | Load the stage adapter with PEFT |
| `optimizer.pt`, `scheduler.pt`, `rng_state_*.pth` | Restore the training state |
| `trainer_state.json`, `training_protocol.json` | Step history and stage configuration |
| `checkpoint_complete.json` | File checksums |

Use the [root merged model](https://huggingface.co/{REPO}) for final-model inference. See the [checkpoint guide](https://huggingface.co/{REPO}/blob/main/checkpoints/README.md) for the training stages.

[Original checkpoint](https://huggingface.co/{REPO}/tree/{revision}/checkpoints/{key})
'''.encode()

def render(report, representation, revision, keys):
    rows='\n'.join(f'| {LABELS[k]} | {v:.2f} | {report["paper_target"][k]:.2f} |' for k,v in report['paper_aligned_public'].items())
    stages='\n'.join(f'| [{title}](checkpoints/{slug}) | `{key}` |' for key,(slug,title) in FEATURED.items())
    card=f'''---
base_model: Qwen/Qwen3.5-9B
library_name: transformers
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

### Spatial perception, reasoning and action in one 9B model.

**11 native spatial forms · 5.22M-record SFT ancestry · specialist-to-generalist MOPD**

[Paper](https://arxiv.org/abs/2610.21882) · [Project](https://knowin-29.github.io/Anvil.github.io/) · [Code](https://github.com/Knowin-29/Anvil) · [Data](https://huggingface.co/datasets/Knowin-29/Anvil-Ben)

Anvil connects scene-aligned spatial supervision with specialist knowledge transfer. Built on Qwen3.5-9B, it produces structured outputs for visual grounding, spatial relations and action-oriented tasks. MOPD transfers specialist feedback on the student's own rollouts into a unified model.

## Release highlights

- **Ready-to-load BF16 weights** at the repository root, with the final LoRA adapter in `adapter/`.
- **Five featured training stages**, each with adapter weights and complete continuation state.
- **Full release evaluation:** 580,000 native spatial examples, 7,366 public benchmark examples and 10,105 general-ability examples.
- **{report['paper_aligned_macro']:.2f}% public benchmark mean** across the seven benchmark scores below.

## Results

| Benchmark | Released model (%) | Paper reference (%) |
|---|---:|---:|
{rows}

[Native geometry and general-ability results](docs/evaluation/README.md) · [Evaluation report](docs/evaluation/evaluation_report.json) · [Evaluation protocol](docs/evaluation/evaluation_spec.json)

## Get the model

Download the inference package:

```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    "Knowin-29/Anvil-MOPD-9B",
    ignore_patterns=["checkpoints/**", "docs/**"],
)
```

Load with Transformers `Qwen3_5ForConditionalGeneration` and `AutoProcessor`, using BF16 and thinking disabled. Evaluation uses up to 1,536 image tokens. The packaged `anvil/anvil_interface.py` handles native DoF prompts and pose decoding: call `prepare_native_prompt("dof", prompt)` before generation and `decode_native_response("dof", text, prompt)` afterward. The pose representation is `{representation}`; other native forms use their original prompts and responses.

## Featured checkpoints

| Stage | Original stage identifier |
|---|---|
{stages}

Use [the checkpoint guide](checkpoints/README.md) to download an individual stage. Additional saved stages are organized in `checkpoints/archive/`.

## Repository guide

| Path | Contents |
|---|---|
| Root model files | Final merged BF16 model, processor and tokenizer |
| `adapter/` | Final MOPD LoRA adapter |
| `checkpoints/` | Featured stages and training archive |
| `anvil/` | Native output interface |
| `docs/evaluation/` | Measured scores and evaluation settings |
| `docs/provenance/` | Training lineage and release records |
| `release_manifest.json` | File checksums and source revision |

Anvil's full data package contains 5.8M records, including 5.22M training records. Explore the [Anvil-Ben data page](https://huggingface.co/datasets/Knowin-29/Anvil-Ben) for downloadable annotations and image reconstruction tools.
'''
    guide='''# Anvil checkpoints

Select a stage to continue training or study specialist-to-generalist transfer. Each directory contains a LoRA adapter, optimizer, scheduler, rank RNG states and training protocol.

| Stage | Directory |
|---|---|
'''
    guide+='\n'.join(f'| {title} | [{slug}]({slug}) |' for slug,title in FEATURED.values())
    guide+='''

```python
from huggingface_hub import snapshot_download

root = snapshot_download(
    "Knowin-29/Anvil-MOPD-9B",
    allow_patterns=["checkpoints/final-mopd/**"],
)
```

The [archive](archive) groups additional checkpoints by experiment and step. The [stage index](../docs/provenance/checkpoint_index.json) records original identities, ancestry and current paths. Release scores apply to the final merged model and final MOPD adapter.
'''
    native=['# Anvil release evaluation','',f'Public benchmark mean: **{report["paper_aligned_macro"]:.4f}%**. Native coverage: **580,000 records** across all 11 forms. General coverage: **10,105 examples**.','','## Native spatial tasks','','| Form / source | Records | Metric | Value |','|---|---:|---|---:|']
    for g in report['native']['groups']:
        for metric,value in g['mean'].items():
            native.append(f'| {g["form"]} / {g["source"]} | {g["n"]:,} | {metric} | {value:.6f} |')
    native+=['','## General ability','','```json',json.dumps(report['general'],indent=2),'```','',
             '## Evaluation settings','',
             'RefSpatial uses Location100 for the paper comparison; the report also includes the full 277-example run. Native rows retain their original output budgets and task scorers. These release evaluation splits also informed training-stage selection. Paper reference values follow the manuscript table; the exact release decoding settings are recorded in [the protocol](evaluation_spec.json).','',
             '[Measured report](evaluation_report.json) · [Training lineage](../provenance/training_lineage.json)','']
    if 'anvil_eval_native_score' in report:
        highlight=f"**Anvil-Eval v1.1: {report['anvil_eval_native_score']:.2f}** on 580,000 native records. [Scoring protocol](docs/evaluation/ANVIL_EVAL_PROTOCOL.md)\n\n"
        card=card.replace('## Results\n\n','## Results\n\n'+highlight)
    return {'README.md':card.encode(),'checkpoints/README.md':guide.encode(),
            'docs/evaluation/README.md':'\n'.join(native).encode()}

def build_plan(files, report, representation, revision, seals, originals, nonancestors, score_artifacts=None):
    mapping={name:destination(name) for name in files}
    if len(set(mapping.values()))!=len(mapping): raise ValueError('Layout collision')
    keys={name.removeprefix('checkpoints/').rsplit('/',1)[0] for name in files if name.startswith('checkpoints/') and name.endswith('/adapter_model.safetensors')}
    if len(keys)!=35 or not set(FEATURED)<=keys: raise ValueError('Incomplete 35-checkpoint release')
    added=render(report,representation,revision,keys)
    index=[]
    for key in sorted(keys):
        origin='checkpoints/'+key
        dest=destination(origin+'/README.md').rsplit('/',1)[0]
        data=stage_card(key,revision)
        seal=json.loads(json.dumps(seals[key]))
        if digest(originals[origin+'/checkpoint_complete.json'])!=files[origin+'/checkpoint_complete.json']:
            raise ValueError('Changed original checkpoint seal')
        if seal['files']['README.md']!=files[origin+'/README.md']:
            raise ValueError('Original card differs from checkpoint seal')
        seal['files']['README.md']=digest(data)
        seal['presentation_origin']={'revision':revision,'path':origin,
            'original_seal_sha256':files[origin+'/checkpoint_complete.json']['sha256']}
        added[dest+'/README.md']=data
        added[dest+'/checkpoint_complete.json']=encode(seal)
        index.append({'original':origin,'path':dest,'featured':key in FEATURED,
                      'in_final_optimizer_ancestry':key not in nonancestors,
                      'original_revision':revision})
    added['docs/provenance/checkpoint_index.json']=encode({'source_revision':revision,'checkpoints':index})
    added['docs/provenance/layout_map.json']=encode({'source_revision':revision,'paths':mapping,
        'checkpoint_payload_changes':['README.md','checkpoint_complete.json'],
        'weights_and_training_state':'byte-identical to source revision'})
    if score_artifacts: added.update(score_artifacts)
    expected={mapping[name]:dict(item) for name,item in files.items()}
    expected.update({name:digest(data) for name,data in added.items()})
    manifest={'schema':'anvil-hub-presentation-v1','repository':REPO,'source_revision':revision,
              'original_release_manifest':'docs/provenance/original_release_manifest.json',
              'files':expected,'checkpoint_count':35,'featured_checkpoints':list(FEATURED)}
    added['release_manifest.json']=encode(manifest)
    expected={**expected,'release_manifest.json':digest(added['release_manifest.json'])}
    return mapping,added,expected
