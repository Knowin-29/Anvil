"""Exact formal-run bindings, including sealed intermediate checkpoints."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
C = HERE.parent
D = C / 'diagnosis_32gpu'
R = C.parent
TRAIN = str(R / '.venv-train/bin/python')
sys.path.insert(0, str(R / 'cloud_a800'))
from checkpoints import save, verify


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def bound(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def check_binding(item):
    require(sha(item['path']) == item['sha256'], 'Changed evidence: ' + item['path'])


def load_plan(path):
    plan = read(path)
    require(plan['schema'] == 'anvil-formal-MOPD-plan-v1', 'Wrong training plan')
    require(plan['smoke'] is False and plan['world'] in (8, 16), 'Wrong formal topology')
    require(plan['global_batch'] == 32 and plan['steps'] >= 128, 'Wrong formal budget')
    for file, digest in plan['frozen_files'].items():
        require(sha(file) == digest, 'Frozen dependency changed: ' + file)
    for item in plan['evidence'].values():
        check_binding(item)
    return plan


def sealed_checkpoint(adapter, full=True):
    adapter = Path(adapter).resolve()
    protocol = read(adapter.parent / 'protocol.json')
    require(protocol['smoke'] is False and protocol['quality_qualified'] is True
            and protocol['full_budget_probe'] is False, 'Smoke/probe is not a formal checkpoint')
    require(protocol['scope'] == 'all_11_native_forms' and protocol['spatial_pool'] == 4756000,
            'Training source scope changed')
    require(protocol['eligibility_sha256'] and protocol['runtime_bindings_sha256'] is None,
            'Missing formal teacher decision')
    receipt = verify(adapter, protocol['world_size']) if full else read(adapter / 'checkpoint_complete.json')
    for name in ('trainer_state.json', 'adapter_model.safetensors', 'adapter_config.json'):
        require(sha(adapter / name) == receipt['files'][name]['sha256'], 'Checkpoint bytes changed: ' + name)
    state = read(adapter / 'trainer_state.json')
    require(state['final_mopd'] is True and state['smoke'] is False, 'Nonformal checkpoint')
    require(0 < state['global_step'] <= protocol['updates'] == state['max_steps'], 'Invalid step')
    require(receipt['global_step'] == state['global_step'], 'Checkpoint seal step mismatch')
    require(sum(state['source_exposures'].values()) == state['global_step'] * protocol['global_batch'],
            'Incomplete source exposure accounting')
    return state, protocol, receipt


def load_midrun_spec(path):
    spec = read(path)
    require(spec['schema'] == 'anvil-formal-checkpoint-public-evaluation-v1', 'Wrong midrun spec')
    require(spec['full_native_validation'] is False and spec['final_quality_accepted'] is False,
            'Intermediate diagnostics are not final delivery')
    state, protocol, receipt = sealed_checkpoint(spec['checkpoint'], full=False)
    require(state['global_step'] == spec['step'], 'Changed intermediate step')
    for item in spec['evidence'].values():
        check_binding(item)
    for file, digest in spec['frozen_files'].items():
        require(sha(file) == digest, 'Evaluation code changed: ' + file)
    for name, digest in spec['model_files'].items():
        require(Path(name).name == name and sha(Path(spec['model']) / name) == digest,
                'Merged model changed: ' + name)
    return spec


def check_protocol(protocol, spec_path, spec):
    require(protocol['checkpoint_spec_sha256'] == sha(spec_path), 'Mixed checkpoint spec')
    require(Path(protocol['model']).resolve() == Path(spec['model']), 'Mixed model')
    require(protocol['smoke'] is False, 'Smoke predictions cannot be full public diagnostics')
