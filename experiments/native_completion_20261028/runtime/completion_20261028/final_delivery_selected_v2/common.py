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


def formal_training(run, full_checkpoint=True, selection_path=None):
    run = Path(run).resolve()
    if selection_path is not None:
        from selection import validate
        selection, prefix = validate(selection_path)
        require(selection['training_run']==str(run), 'Selection belongs to another training run')
        protocol=read(run/'protocol.json')
        require(protocol['smoke'] is False and protocol['quality_qualified'] is True
                and protocol.get('full_budget_probe') is False and not protocol.get('objective_control',False),
                'Diagnostic controls cannot become a selected formal model')
        require(protocol['scope']=='all_11_native_forms' and protocol['spatial_pool']==4756000
                and protocol.get('eligibility_sha256') and protocol.get('runtime_bindings_sha256') is None,
                'Missing formal teacher/scope evidence')
        require(protocol.get('mtd_weight',1)>0, 'Zero-MTD model cannot be delivered as MOPD')
        checkpoint=Path(selection['checkpoint']).resolve()
        require(checkpoint.parent==run, 'Selected checkpoint outside training run')
        receipt=verify(checkpoint,protocol['world_size']) if full_checkpoint else read(checkpoint/'checkpoint_complete.json')
        state=read(checkpoint/'trainer_state.json')
        require(receipt['global_step']==state['global_step']==selection['selected_step'], 'Selected checkpoint step mismatch')
        require(state['source_exposures']==prefix['actual']['source_exposures'], 'Selected exposure mismatch')
        endpoint=dict(stage='selected_formal_checkpoint',smoke=False,checkpoint=str(checkpoint),
            steps=selection['selected_step'],planned_steps=selection['planned_steps'],
            source_exposures=prefix['actual']['source_exposures'],
            examples_seen=prefix['actual']['exposures'],
            original_planned_run_completed=selection['original_planned_run_completed'],
            budget_disposition=selection['budget_disposition'],selection=bound(selection_path))
        return endpoint,protocol,checkpoint,receipt
    complete, protocol = read(run/'complete.json'), read(run/'protocol.json')
    require(complete['stage'] == 'mopd_training_complete' and complete['smoke'] is False,
            'Only completed formal MOPD can enter final delivery')
    require(protocol['smoke'] is False and protocol['quality_qualified'] is True
            and protocol.get('full_budget_probe') is False, 'Smoke/probe is not final MOPD')
    require(protocol['scope'] == 'all_11_native_forms' and protocol['spatial_pool'] == 4756000,
            'Unexpected final training scope')
    require(protocol.get('eligibility_sha256') and protocol.get('runtime_bindings_sha256') is None,
            'Formal teacher evidence is required')
    require(complete['steps'] == protocol['updates'], 'Incomplete planned MOPD updates')
    checkpoint = Path(complete['checkpoint']).resolve()
    require(checkpoint.parent == run, 'Final checkpoint outside training run')
    receipt = verify(checkpoint, protocol['world_size']) if full_checkpoint else read(checkpoint/'checkpoint_complete.json')
    state = read(checkpoint/'trainer_state.json')
    require(receipt['global_step'] == state['global_step'] == complete['steps'], 'Step mismatch')
    require(state['source_exposures'] == complete['source_exposures'], 'Exposure mismatch')
    require(sum(complete['source_exposures'].values()) == complete['steps'] * protocol['global_batch'],
            'Exposure count does not cover all updates')
    return complete, protocol, checkpoint, receipt


def load_spec(path, verify_model=True):
    path = Path(path).resolve()
    spec = read(path)
    require(spec['schema'] == 'anvil-selected-final-evaluation-spec-v2', 'Unknown selected evaluation spec')
    require(spec['native_records'] == 580000 and spec['general_records'] == 10105,
            'Final evaluation must use complete sets')
    require(spec['native_shards'] >= 1 and spec['public_shards'] >= 1, 'Invalid sharding')
    for item in spec['evidence'].values():
        check_binding(item)
    for file, digest in spec['frozen_files'].items():
        require(sha(file) == digest, 'Evaluation dependency changed: ' + file)
    selection=spec['evidence']['checkpoint_selection'];check_binding(selection)
    complete, protocol, checkpoint, _ = formal_training(spec['training_run'], full_checkpoint=False,
                                                        selection_path=selection['path'])
    require(str(checkpoint) == spec['checkpoint'], 'Final checkpoint identity changed')
    require(spec['selected_step']==complete['steps'] and spec['planned_steps']==complete['planned_steps'],
            'Selected evaluation update budget changed')
    require(spec['original_planned_run_completed']==complete['original_planned_run_completed']
            and spec['budget_disposition']==complete['budget_disposition'],'Selected budget disposition changed')
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
