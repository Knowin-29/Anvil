"""Strictly nonfinal objective controls and unchanged evaluation bindings."""
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


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bound(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def load_experiment():
    plan = read(HERE / 'experiment.json')
    require(plan['schema'] == 'anvil-native-objective-controls-v1' and plan['final_mopd'] is False,
            'Wrong experiment role')
    for path, expected in plan['frozen_files'].items():
        require(sha(path) == expected, 'Control dependency changed: ' + path)
    require(sha(plan['cpu_validation']['path']) == plan['cpu_validation']['sha256'],
            'Control CPU validation changed')
    require(read(plan['cpu_validation']['path'])['state'] == 'passed', 'Control validation missing')
    return plan


def control_checkpoint(adapter, full=True):
    adapter = Path(adapter).resolve()
    require(adapter.parent.parent == HERE / 'runs', 'Unexpected objective-control path')
    complete, protocol = read(adapter.parent / 'complete.json'), read(adapter.parent / 'protocol.json')
    require(complete['stage'] == 'objective_control_complete' and complete['smoke'] is False
            and complete['objective_control'] is True and complete['final_mopd'] is False,
            'Not a completed nonfinal control')
    require(protocol['objective_control'] is True and protocol['quality_qualified'] is False
            and protocol['schedule_steps'] == 1664 and protocol['updates'] == complete['steps'] == 128,
            'Control schedule or role changed')
    require(Path(complete['checkpoint']).resolve() == adapter, 'Different control tip')
    receipt = verify(adapter, 8) if full else read(adapter / 'checkpoint_complete.json')
    state = read(adapter / 'trainer_state.json')
    require(receipt['global_step'] == state['global_step'] == 128 and state['final_mopd'] is False,
            'Control checkpoint state differs')
    for name in ['adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json']:
        require(sha(adapter / name) == receipt['files'][name]['sha256'], 'Control checkpoint bytes changed')
    require(sum(state['source_exposures'].values()) == 4096, 'Incomplete control exposures')
    return complete, protocol, receipt


def load_spec(path):
    spec = read(path)
    require(spec['schema'] == 'anvil-objective-control-evaluation-v1'
            and spec['final_mopd'] is False and spec['final_quality_accepted'] is False,
            'Wrong evaluation role')
    for filename, expected in spec['frozen_files'].items():
        require(sha(filename) == expected, 'Evaluation dependency changed: ' + filename)
    for item in spec['evidence'].values():
        require(sha(item['path']) == item['sha256'], 'Control evidence changed')
    control_checkpoint(spec['checkpoint'], full=False)
    for name, expected in spec['model_files'].items():
        require(Path(name).name == name and sha(Path(spec['model']) / name) == expected,
                'Control model changed')
    require(spec['step'] == 128 and spec['public_shards'] == 8 and spec['general_records'] == 10105,
            'Incomplete control evaluation scope')
    return spec


def check_protocol(protocol, spec_path, spec):
    require(protocol['checkpoint_spec_sha256'] == sha(spec_path)
            and Path(protocol['model']).resolve() == Path(spec['model'])
            and protocol['smoke'] is False, 'Mixed control evaluation protocol')
