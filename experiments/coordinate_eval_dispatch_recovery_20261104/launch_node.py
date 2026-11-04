"""Bind the frozen evaluation node to its own version of common.py.

The frozen node contains a stale sys.path entry. Load its intended, sealed
dependency before executing it; all inference workers and scorers stay unchanged.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
PACKAGE = CAMPAIGN / 'final_delivery_geometry_objective_v2'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind_common(policy):
    for filename, digest in policy['files'].items():
        if sha(filename) != digest:
            raise ValueError('Changed dispatch dependency: ' + filename)
    if sha(policy['evaluation_spec']) != policy['evaluation_spec_sha256']:
        raise ValueError('Changed exact evaluation specification')
    location = PACKAGE / 'common.py'
    module_spec = importlib.util.spec_from_file_location('common', location)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules['common'] = module
    module_spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != location or module.HERE != PACKAGE:
        raise ValueError('Unexpected evaluation module root')
    return module


def main():
    policy = json.loads((ROOT / 'runtime_policy.json').read_text())
    common = bind_common(policy)
    args = list(sys.argv[1:])
    inspect = '--inspect' in args
    if inspect:
        args.remove('--inspect')
    spec_path = Path(args[args.index('--spec') + 1]).resolve()
    output = Path(args[args.index('--output') + 1]).resolve()
    node = int(args[args.index('--node') + 1])
    nodes = int(args[args.index('--nodes') + 1])
    if spec_path != Path(policy['evaluation_spec']) or output != Path(policy['output']):
        raise ValueError('Different candidate or output directory')
    if nodes != 4 or not 0 <= node < nodes:
        raise ValueError('Expected four eight-GPU full-evaluation tasks')
    binding = dict(common_file=common.__file__, common_sha256=sha(common.__file__),
                   worker_root=str(common.HERE), frozen_node_sha256=sha(PACKAGE / 'node.py'),
                   evaluation_spec_sha256=sha(spec_path), runtime_policy_sha256=sha(ROOT / 'runtime_policy.json'),
                   node=node, nodes=nodes, checked=time.time())
    if inspect:
        namespace = runpy.run_path(str(PACKAGE / 'node.py'), run_name='dispatch_inspection')
        assert namespace['HERE'] == PACKAGE
        assert namespace['load_spec'] is common.load_spec
        spec = common.load_spec(spec_path, verify_model=False)
        tasks = namespace['task_list'](spec)
        partitions = [[task for index, task in enumerate(tasks) if index % nodes == n]
                      for n in range(nodes)]
        assert len(tasks) == 40 and all(len(part) == 10 for part in partitions)
        assert sum(task['kind'] == 'public' for task in tasks) == 8
        assert sum(task['kind'] == 'native' for task in tasks) == 32
        assert spec['native_records'] == 580000 and spec['general_records'] == 10105
        assert sum(spec['public_records'].values()) == 7366
        print(json.dumps(dict(binding, state='dispatch_and_full_scope_verified', gpu_work=False)))
        return
    common.save(ROOT / f'node{node}_import_binding.json', binding)
    sys.argv = [str(PACKAGE / 'node.py'), *args]
    runpy.run_path(str(PACKAGE / 'node.py'), run_name='__main__')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if '--inspect' not in sys.argv:
            node = sys.argv[sys.argv.index('--node') + 1] if '--node' in sys.argv else 'unknown'
            path = ROOT / f'node{node}_dispatch_failure.json'
            path.write_text(json.dumps(dict(error=repr(exc), failed=time.time()), indent=2) + '\n')
        raise
