"""Launch the frozen formal trainer on one or two cloud workers."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from formal_common import C, D, R, TRAIN, load_plan, read, require, save, sha
sys.path.insert(0, str(D))
from repair_node import environment, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--resume', type=Path)
    args = ap.parse_args()
    plan = load_plan(args.plan)
    nodes = plan['world'] // 8
    node = int(os.environ.get('MLP_ROLE_INDEX', '0'))
    require(int(os.environ.get('MLP_WORKER_NUM', str(nodes))) == nodes and 0 <= node < nodes,
            'Cloud topology does not match plan')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                                      '--format=csv,noheader,nounits'], text=True).splitlines()
    require(len(devices) == 8, 'Expected eight GPUs per worker')
    for line in devices:
        index, uid, name, memory = map(str.strip, line.split(','))
        require('A800' in name and int(memory) >= 79000, 'Unexpected GPU')
    out = Path(plan['training_run']); out.mkdir(parents=True, exist_ok=True)
    env = environment(list(range(8)), 'formal_mopd_' + out.name + '_' + str(node))
    (Path(env['XDG_CACHE_HOME']) / 'torch/kernels').mkdir(parents=True, exist_ok=True)
    env['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    address = os.environ.get('MLP_WORKER_0_HOST', '127.0.0.1')
    port = os.environ.get('MLP_WORKER_0_PORT', '29890')
    require(nodes == 1 or address not in ('127.0.0.1', 'localhost'), 'Missing cross-node address')
    cmd = [TRAIN, '-m', 'torch.distributed.run', '--nnodes=' + str(nodes), '--nproc_per_node=8',
           '--node_rank=' + str(node), '--master_addr=' + address, '--master_port=' + port,
           '--max_restarts=0', '--rdzv_conf=timeout=300', str(C / 'native_mopd_v6.py'),
           '--output', str(out), '--steps', str(plan['steps']), '--global-batch', '32',
           '--save-every', '128', '--eligibility', plan['eligibility'],
           '--initialization-lineage', plan['initialization_lineage'],
           '--dof-representation', 'rotation128', '--spatial-replay']
    if args.resume:
        from formal_common import sealed_checkpoint
        _, protocol, _ = sealed_checkpoint(args.resume)
        require(args.resume.resolve().parent == out.resolve(), 'Resume outside exact run')
        require(protocol['world_size'] == plan['world'] and protocol['updates'] == plan['steps'],
                'Cannot change optimizer schedule/topology during resume')
        cmd += ['--resume', str(args.resume)]
    else:
        require(not (out / 'protocol.json').exists(), 'Use explicit verified resume for an existing run')
    save(out / ('launch_node%d.json' % node), dict(argv=cmd, plan_sha256=sha(args.plan),
                                                devices=devices, started=time.time()))
    try:
        run(cmd, env, out / ('console_node%d.log' % node))
        complete = read(out / 'complete.json')
        require(complete['stage'] == 'mopd_training_complete' and complete['steps'] == plan['steps'],
                'Training did not complete the frozen budget')
        save(out / ('node%d_complete.json' % node), dict(completed=time.time(), plan_sha256=sha(args.plan)))
    except BaseException as exc:
        save(out / ('node%d_failure.json' % node), dict(error=repr(exc), time=time.time()))
        raise


if __name__ == '__main__':
    main()
