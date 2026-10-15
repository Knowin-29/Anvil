"""Build a private, de-identified SFT handoff; never certify a final model.

No remote writes. Original adapters and their runtime configurations are untouched.
Only a separately verified final model may use publish_anvil_model.py.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from tools.publish_anvil_model import REPO, REVISION, digest
from tools.export_anvil_release import SECRET, KEY_UUID, PRIVATE_IDENTITY, has_deployed_endpoint


def portable_config(original, base, revision):
    if not REPO.fullmatch(base) or not REVISION.fullmatch(revision):
        raise ValueError('Public base identity and immutable revision required')
    if original.get('peft_type') != 'LORA' or original.get('task_type') != 'CAUSAL_LM':
        raise ValueError('Expected a causal-LM LoRA adapter')
    result = copy.deepcopy(original)
    result.update(base_model_name_or_path=base, revision=revision)
    scan_metadata(json.dumps(result).encode())
    return result


def scan_metadata(raw):
    if SECRET.search(raw) or KEY_UUID.search(raw) or PRIVATE_IDENTITY.search(raw) or has_deployed_endpoint(raw):
        raise ValueError('Private path, credential or deployed endpoint in staging metadata')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def tensor_compatibility(model_path, config, adapter):
    """CPU-only shape and finite-tensor verification; not generation accuracy."""
    import torch
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText
    with init_empty_weights():
        base = AutoModelForImageTextToText.from_config(
            AutoConfig.from_pretrained(model_path, local_files_only=True), dtype=torch.bfloat16,
            attn_implementation='sdpa')
        model = get_peft_model(base, LoraConfig(**config))
        expected = {name:tuple(t.shape) for name,t in get_peft_model_state_dict(model).items()}
    count = 0; elements = 0
    with safe_open(adapter, framework='pt', device='cpu') as tensors:
        scan_metadata(json.dumps(tensors.metadata() or {}).encode())
        if set(tensors.keys()) != set(expected): raise ValueError('Adapter keys differ from model topology')
        for name, shape in expected.items():
            tensor = tensors.get_tensor(name)
            if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Invalid adapter tensor shape or non-finite values')
            count += 1; elements += tensor.numel()
    return {'tensors':count,'elements':elements,'shape_and_finiteness_verified':True,
            'gpu_calls':0,'new_generation_accuracy_claimed':False}


def stage(args):
    root=args.output.resolve()
    if root.exists(): raise ValueError('Use a new output; do not overwrite a prior handoff')
    registry=json.loads(args.registry.read_text())['supervised_checkpoints']['native_v4_preparation']
    binding=json.loads(args.base_binding.read_text())
    if binding.get('state') != 'all_training_bound_base_files_match_hub_revision':
        raise ValueError('Verified Hub base binding required')
    base,revision=binding['base_model'],binding['base_revision']
    runs=dict(item.split('=',1) for item in args.run)
    if len(args.run)!=4 or set(runs)!={'G','S','A','joint'}: raise ValueError('Exactly four scoped runs required')
    prepared={}; provenance={}
    for family,raw_path in runs.items():
        run=Path(raw_path).resolve(); protocol=json.loads((run/'protocol.json').read_text())
        indexed=registry['adapters'][family]; checkpoint=run/'training'/f"checkpoint-{registry['terminal_update']}"
        if digest(run/'protocol.json')!=indexed['training_protocol_sha256'] or protocol['scope']!=family:
            raise ValueError('Training protocol binding differs from indexed artifact')
        if protocol['export_manifest_sha256']!=registry['training_export_manifest_sha256']:
            raise ValueError('Training export differs from registry')
        if protocol['base_files_sha256']!=binding['files_sha256']:
            raise ValueError('Training base differs from verified Hub revision')
        state=json.loads((checkpoint/'trainer_state.json').read_text())
        if state['global_step']!=registry['terminal_update'] or json.loads((run/'training.result.json').read_text())['exit_code']!=0:
            raise ValueError('Successful terminal training result required')
        weights=checkpoint/'adapter_model.safetensors'
        if digest(weights)!=indexed['weights_sha256']: raise ValueError('Weight identity changed')
        config=portable_config(json.loads((checkpoint/'adapter_config.json').read_text()),base,revision)
        compatibility=tensor_compatibility(protocol['model'],config,str(weights))
        prepared[family]=(weights,config)
        provenance[family]={**indexed,'terminal_update':state['global_step'],
            'training':{k:protocol[k] for k in ('scope','sampling','steps','seed','gradient_accumulation',
                                             'max_length','image_tokens','learning_rate','environment')},
            'portable_config_changes':['base_model_name_or_path','revision'],
            'original_config_sha256':digest(checkpoint/'adapter_config.json'), 'compatibility':compatibility}
        print(json.dumps({'scope':family,**compatibility}),flush=True)
    root.mkdir(parents=True)
    for family,(weights,config) in prepared.items():
        folder=root/family;folder.mkdir();shutil.copyfile(weights,folder/'adapter_model.safetensors')
        save(folder/'adapter_config.json',config)
        if digest(folder/'adapter_model.safetensors')!=registry['adapters'][family]['weights_sha256']:
            raise ValueError('Copied weight checksum mismatch')
    shutil.copyfile(args.base_license,root/'LICENSE.base')
    save(root/'BASE_MODEL.json',binding)
    save(root/'PROVENANCE.json',{'schema':'anvil-private-sft-provenance-v1',
        'training_export_manifest_sha256':registry['training_export_manifest_sha256'], 'adapters':provenance})
    (root/'NOTICE.md').write_text(
        '# Private research staging\n\nThese are source-native SFT preparation adapters, not a final Anvil release.\n'
        'The Qwen base is not bundled; its Apache-2.0 license is preserved in LICENSE.base.\n'
        'No new public license for these project adapter weights is granted by this staging upload.\n'
        'The MIT license of the engineering source does not replace the base-model or data licenses.\n',encoding='utf-8')
    (root/'README.md').write_text(f'''---
library_name: peft
base_model: {base}
base_model_relation: adapter
tags:
- anvil
- lora
- research-checkpoint
---

# Anvil source-native SFT preparation — private staging

This bundle contains four terminal 1,024-update LoRA adapters: `G`, `S`, `A`,
and `joint`. It is not the final MOPD model, a full-epoch result, a qualified
teacher release, or the checkpoint behind the manuscript's main table.
Full matched evaluation is still in progress. These weights are not suitable
for robot execution without additional validation and safety controls.

The reconstruction training export contains 5.22M source-grounded records;
each run made 1,024 updates with micro-batch 1, accumulation 8 and seed 17.
Adapter rank is 64, alpha 128, dropout 0.05 and learning rate 1e-4. Vision
parameters were frozen. G/joint used 16,384 context tokens and S/A used 8,192;
image budget was 1,536. See PROVENANCE.json for exact per-run bindings.
Dataset images, training examples, optimizers and server logs are not included.

## Load one adapter

Use the exact base revision `{revision}` and an immutable staging commit.
The subfolder is required because the bundle is not a merged model.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

base_id = "{base}"
base_revision = "{revision}"
base = AutoModelForImageTextToText.from_pretrained(
    base_id, revision=base_revision, dtype=torch.bfloat16,
    device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained(base_id, revision=base_revision)
model = PeftModel.from_pretrained(
    base, "YOUR_PRIVATE_STAGING_REPOSITORY", subfolder="joint",
    revision="YOUR_IMMUTABLE_STAGING_COMMIT")
model.eval()
```

The packaged config replaces only the private base path and pins its public
revision. All adapter tensor bytes are unchanged. CPU topology, tensor-shape
and finiteness checks are packaging checks, not a new accuracy evaluation.

## Known evaluation limits

The current Action prompt does not spell out every scoring-contract field type
and allowed value; strict-format failures must not be reported as general
reasoning incapacity. In the completed joint DoF panel, predicted transforms
did not meet the rigid-transform contract. These issues remain unresolved;
this package claims neither final benchmark readiness nor improved accuracy.

The repository must remain private until release terms and full validation
are settled. See NOTICE.md and LICENSE.base. Do not infer a public adapter
license from the engineering repository's MIT license.
''',encoding='utf-8')
    files={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}
    manifest={'schema':'anvil-private-sft-staging-v1','artifact_role':'source_native_sft_preparation',
        'base_model':base,'base_revision':revision,'training_export_manifest_sha256':registry['training_export_manifest_sha256'],
        'full_evaluation_complete':False,'final_mopd_complete':False,'public_release_authorized':False,'files':files}
    for name in files:
        if not name.endswith('.safetensors'): scan_metadata((root/name).read_bytes())
    save(root/'STAGING_MANIFEST.json',manifest)
    print(json.dumps({'state':'private_staging_prepared','files':len(files)+1,'weights':4,'remote_writes':0}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('registry','base-binding','base-license','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--run',action='append',required=True,help='Repeat FAMILY=/path/to/terminal/run')
    stage(parser.parse_args())


if __name__=='__main__': main()
