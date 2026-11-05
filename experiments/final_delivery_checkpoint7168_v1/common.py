"""Bindings shared by final evaluation and delivery; no credential operations."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
C = HERE.parent
D = C / 'diagnosis_32gpu'
R = C.parent
EXPORT = R.parent / 'arxiv/native_evaluation_export_20261012_v4'
TRAIN = str(R / '.venv-train/bin/python')
EVAL = str(R / '.venv-vllm/bin/python')
sys.path.insert(0, str(R / 'cloud_a800'))
from checkpoints import save, verify


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def check_binding(item):
    require(sha(item['path']) == item['sha256'], 'Changed evidence: ' + item['path'])


def initialization_transform(spec, ancestor):
    policy = spec['visual_adaptation']
    for key in ['initialization', 'interface']:
        check_binding(policy[key])
    receipt = read(policy['initialization']['path'])
    require(receipt['schema']=='anvil-zero-delta-visual-adapter-initialization-v1'
            and receipt['state']=='prepared_not_trained', 'Not a zero-delta initialization')
    require(receipt['source']==ancestor['checkpoint'] and
            receipt['source_adapter_sha256']==ancestor['adapter_sha256'], 'Different trained parent')
    initial = Path(spec['initial']['adapter'])
    require(receipt['output']==str(initial) and receipt['adapter_sha256']==spec['initial']['adapter_sha256'],
            'Different expanded initialization')
    require(sha(initial/'adapter_model.safetensors')==receipt['adapter_sha256'] and
            sha(initial/'adapter_config.json')==receipt['config_sha256'], 'Changed expanded adapter')
    require(receipt['language_tensors_exactly_unchanged'] is True and
            receipt['initial_visual_delta_exactly_zero'] is True and
            receipt['training_performed'] is False, 'Initialization changed parent function or adds training')
    require(receipt['preserved_language_tensors']==496 and receipt['added_visual_tensors']==68,
            'Adapter expansion scope changed')
    require(receipt['language_parameters']==policy['language_parameters']==173113344 and
            receipt['visual_parameters']==policy['visual_parameters']==5136384 and
            receipt['total_adapter_parameters']==178249728, 'Adapter parameter scope changed')
    interface = read(policy['interface']['path'])
    require(interface['decode_required_for_native_inference'] is True and
            interface['model_representation']=='anvil-dof-train-rotation128-v1', 'Changed native interface')
    return dict(kind='zero_delta_visual_lora_extension', source_checkpoint=ancestor['checkpoint'],
                source_adapter_sha256=ancestor['adapter_sha256'], checkpoint=str(initial),
                adapter_sha256=receipt['adapter_sha256'], training_exposures_added=0,
                original_language_tensors_preserved=True, initial_visual_delta_exactly_zero=True,
                receipt=policy['initialization'], interface=policy['interface'])


def formal_training(run, full_checkpoint=True):
    """Validate the user's selected saved state without asserting planned-stage completion."""
    run=Path(run).resolve();pipeline=C/'visual_geometry_candidate_20261104'
    selected=C/'checkpoint7168_delivery_20261105/selection.json'
    choice=read(selected);checkpoint=Path(choice['checkpoint']).resolve()
    require(run==pipeline/'runs/repair' and checkpoint==run/'checkpoint-7168','Different selected checkpoint')
    protocol=read(run/'protocol.json');spec=read(pipeline/'spec.json')
    require(sha(pipeline/'spec.json')==choice['pipeline_spec_sha256']==protocol['spec_sha256'],'Training identity changed')
    require(sha(checkpoint/'checkpoint_complete.json')==choice['seal_sha256'],'Selected state changed')
    for file,digest in spec['frozen_files'].items():require(sha(file)==digest,'Training source changed')
    require(protocol['stage']=='repair' and protocol['scope']=='all_11_native_forms' and not protocol['smoke'],'Unexpected training scope')
    require(protocol['spatial_pool']==4756000 and protocol['sft_pool_already_covered']==5220000,'Changed dataset ancestry')
    require(protocol['initial']==spec['initial']['adapter'] and protocol['initial_adapter_sha256']==spec['initial']['adapter_sha256'],'Broken corrective ancestry')
    require(sha(Path(protocol['initial'])/'adapter_model.safetensors')==protocol['initial_adapter_sha256'],'Changed initial adapter')
    binding=spec['continuation_parent']['lineage'];check_binding(binding);ancestor=read(binding['path'])
    require(ancestor['state']=='complete','Parent MOPD lineage incomplete');initialization_transform(spec,ancestor)
    receipt=verify(checkpoint,32) if full_checkpoint else read(checkpoint/'checkpoint_complete.json')
    state=read(checkpoint/'trainer_state.json');audit=read(choice['integrity']['path']);check_binding(choice['integrity'])
    require(receipt['global_step']==state['global_step']==audit['step']==7168,'Selected step changed')
    require(state['max_steps']==8192 and state['stage']=='repair' and state['final_mopd'] is False,'Incorrect stage identity')
    require(audit['all_file_hashes_verified'] and audit['all32_rng_files_loadable'] and audit['optimizer_step']==7168,'Incomplete integrity audit')
    require(len(state['log_history'])==7168 and state['source_exposures']==audit['committed_exposure_prefix']['source_exposures'],'Exposure prefix differs')
    require(sum(state['source_exposures'].values())==7168*64,'Incorrect actual training coverage')
    complete=dict(stage='user_selected_repair_checkpoint',checkpoint=str(checkpoint),steps=7168,
        planned_stage_steps=8192,stage_plan_completed=False,smoke=False,final_mopd=False,
        source_exposures=state['source_exposures'],adapter_sha256=receipt['files']['adapter_model.safetensors']['sha256'],spec_sha256=choice['pipeline_spec_sha256'])
    return complete,protocol,checkpoint,receipt


def load_spec(path, verify_model=True):
    path = Path(path).resolve()
    spec = read(path)
    require(spec['schema'] == 'anvil-final-evaluation-spec-v1', 'Unknown evaluation spec')
    require(spec['native_records'] == 580000 and spec['general_records'] == 10105,
            'Final evaluation must use complete sets')
    require(spec['native_shards'] >= 1 and spec['public_shards'] >= 1, 'Invalid sharding')
    for item in spec['evidence'].values():
        check_binding(item)
    for file, digest in spec['frozen_files'].items():
        require(sha(file) == digest, 'Evaluation dependency changed: ' + file)
    complete, protocol, checkpoint, _ = formal_training(spec['training_run'], full_checkpoint=False)
    require(str(checkpoint) == spec['checkpoint'], 'Final checkpoint identity changed')
    require(protocol['representation']['mode'] == spec['representation'], 'Model representation mismatch')
    if verify_model:
        for name, digest in spec['model_files'].items():
            require(Path(name).name == name, 'Unsafe model filename')
            require(sha(Path(spec['model'])/name) == digest, 'Final model changed: ' + name)
    return spec


def check_protocol(protocol, spec_path, spec):
    require(protocol['final_spec_sha256'] == sha(spec_path), 'Mixed final evaluation specs')
    require(Path(protocol['model']).resolve() == Path(spec['model']), 'Mixed evaluation models')
    require(protocol.get('smoke', False) is False and protocol.get('smoke_rows', 0) == 0,
            'Smoke results cannot qualify final evaluation')
