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


def formal_training(run, full_checkpoint=True):
    run = Path(run).resolve()
    complete, protocol = read(run/'complete.json'), read(run/'protocol.json')
    require(complete['stage'] == 'mopd_training_complete' and complete['smoke'] is False,
            'Only completed formal MOPD can enter final delivery')
    require(protocol['schema']=='anvil-direct-corrective-MOPD-v1'
            and protocol['smoke'] is False and protocol['stage']=='mopd'
            and protocol['final_mopd'] is True, 'Only the direct production MOPD stage is supported')
    require(protocol['scope'] == 'all_11_native_forms' and protocol['spatial_pool'] == 4756000,
            'Unexpected final training scope')
    pipeline=C/'direct_completion_20261029'
    require(run==pipeline/'runs/mopd','Different direct training run')
    spec=read(pipeline/'spec.json')
    require(sha(pipeline/'spec.json')==protocol['spec_sha256']==complete['spec_sha256'], 'Pipeline identity changed')
    for filename,digest in spec['frozen_files'].items():require(sha(filename)==digest,'Changed training source')
    require(protocol['loss']==spec['stages']['mopd'] and protocol['loss']['mtd_weight']>0, 'Missing genuine distillation')
    require(protocol['updates']==spec['stages']['mopd']['steps'], 'Changed MOPD budget')
    parent=pipeline/'runs/repair';repair=read(parent/'complete.json')
    require(repair['stage']=='repair_training_complete' and repair['steps']==spec['stages']['repair']['steps'], 'Incomplete corrective stage')
    require(protocol['initial']==repair['checkpoint'] and protocol['initial_adapter_sha256']==repair['adapter_sha256'], 'Broken repair→MOPD ancestry')
    require(sha(Path(repair['checkpoint'])/'adapter_model.safetensors')==repair['adapter_sha256'], 'Corrective checkpoint changed')
    review=protocol['teacher_assessment']
    require(sha(review['existing_review'])==review['review_sha256'], 'Teacher evidence changed')
    for family,adapter in protocol['teachers'].items():
        require(adapter==spec['teachers'][family]['adapter'] and sha(Path(adapter)/'adapter_model.safetensors')==spec['teachers'][family]['adapter_sha256'], 'Different teacher')
    require(complete['steps'] == protocol['updates'], 'Incomplete planned MOPD updates')
    checkpoint = Path(complete['checkpoint']).resolve()
    require(checkpoint.parent == run, 'Final checkpoint outside training run')
    receipt = verify(checkpoint, protocol['world_size']) if full_checkpoint else read(checkpoint/'checkpoint_complete.json')
    state = read(checkpoint/'trainer_state.json')
    require(receipt['global_step'] == state['global_step'] == complete['steps'], 'Step mismatch')
    require(state['source_exposures'] == complete['source_exposures'], 'Exposure mismatch')
    require(sum(complete['source_exposures'].values()) == complete['steps'] * protocol['global_batch'],
            'Exposure count does not cover all updates')
    require(complete['source_exposures']==spec['planned_source_exposures']['mopd'], 'Completed source mixture differs')
    require(any(row['mtd']>0 and row['mtd_rows_per_rank']>0 for row in state['log_history']), 'No actual MTD optimization evidence')
    return complete, protocol, checkpoint, receipt


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
