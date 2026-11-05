"""Build and bind the selected checkpoint's inference payload without copying weights."""
from pathlib import Path
import hashlib,json,runpy,sys,time

Z=Path(__file__).resolve().parent
C=Z.parent
REPO='Knowin-29/Anvil-MOPD-9B'

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_suffix(p.suffix+'.pending')
    temporary.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temporary.replace(p)
def auth():
    from huggingface_hub import HfApi
    token=runpy.run_path(str(C/'verification_20261101/preupload_merged_blobs.py'))['authorized_token']()
    api=HfApi(token=token)
    assert api.whoami()['name']=='Knowin-29'
    return api,token

def prepare_payload():
    sys.path.insert(0,str(C/'final_delivery_checkpoint7168_v1'))
    from common import load_spec
    spec=load_spec(Z/'evaluation_spec.json')
    selected=read(Z/'selection.json');checkpoint=Path(selected['checkpoint'])
    assert spec['checkpoint']==str(checkpoint) and spec['selected_step']==7168
    parent=read(C/'visual_geometry_candidate_20261104/spec.json')['continuation_parent']['lineage']
    assert sha(parent['path'])==parent['sha256']
    lineage=read(parent['path']);assert lineage['state']=='complete'
    state=read(checkpoint/'trainer_state.json')
    assert sum(state['source_exposures'].values())==458752
    prepared=Z/'release_payload';prepared.mkdir(exist_ok=True)
    from package_interface import copy_inference_interface
    copy_inference_interface(prepared/'anvil',spec['representation'])
    validation_path=C/'final_delivery_visual_geometry_v1/interface_validation/rotation128_interface_validation.json'
    validation=read(validation_path)
    assert validation['state']=='passed' and validation['mode']==spec['representation']
    for name,h in validation['standalone_files_sha256'].items():
        assert sha(prepared/'anvil'/name)==h,'Inference interface differs: '+name
    save(prepared/'training_lineage.json',dict(schema='anvil-selected-checkpoint-lineage-v1',
        state='verified_selected_checkpoint_ancestry',selected_checkpoint=selected,
        parent_completed_mopd=parent,initial_sft=lineage['initial_sft'],
        selected_stage=dict(stage='repair',steps=7168,planned_steps=8192,planned_stage_completed=False,
            sample_exposures=458752,source_exposures=state['source_exposures']),
        other_experiment_arms_counted_as_final_training=False,final_spec_sha256=sha(Z/'evaluation_spec.json')))
    sources={name:Path(spec['model'])/name for name in spec['model_files'] if name not in {'README.md','merge_receipt.json'}}
    sources.update({'adapter/'+name:checkpoint/name for name in ['adapter_model.safetensors','adapter_config.json']})
    sources.update({'anvil/'+name:prepared/'anvil'/name for name in validation['standalone_files_sha256']})
    sources.update({
        'docs/provenance/merge_receipt.json':Path(spec['model'])/'merge_receipt.json',
        'docs/provenance/training_lineage.json':prepared/'training_lineage.json',
        'docs/provenance/completed_parent_mopd_lineage.json':Path(parent['path']),
        'docs/provenance/selected_checkpoint.json':Z/'selection.json',
        'docs/provenance/interface_validation.json':validation_path,
    })
    payload={name:dict(source=str(p),bytes=p.stat().st_size,sha256=sha(p)) for name,p in sources.items()}
    save(prepared/'payload_manifest.json',dict(schema='anvil-selected-inference-payload-v1',
        final_spec_sha256=sha(Z/'evaluation_spec.json'),files=payload))
    return payload

def validated_payload():
    manifest=read(Z/'release_payload/payload_manifest.json')
    assert manifest['final_spec_sha256']==sha(Z/'evaluation_spec.json')
    for name,item in manifest['files'].items():
        p=Path(item['source'])
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Payload changed: '+name
    return manifest['files']
