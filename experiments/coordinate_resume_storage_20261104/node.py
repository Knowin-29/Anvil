"""Resume the unchanged coordinate MOPD from its sealed step160 state."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

W = Path(__file__).resolve().parent
C = W.parent
O = C / 'geometry_objective_v2_20261104'
sys.path[:0] = [str(C / 'diagnosis_32gpu'), str(C.parent / 'cloud_a800')]
from retention_node import environment, TRAIN
from checkpoints import save, sha, verify


def main():
    node = int(os.environ['MLP_ROLE_INDEX'])
    assert int(os.environ['MLP_WORKER_NUM']) == 4 and 0 <= node < 4
    policy = json.loads((W / 'resume_policy.json').read_text())
    for name, digest in policy['files'].items():
        assert sha(name) == digest, name
    spec = json.loads((O / 'spec.json').read_text())
    assert sha(O / 'spec.json') == policy['pipeline_spec_sha256']
    for name, digest in spec['frozen_files'].items():
        assert sha(name) == digest, name
    resume = Path(policy['resume_checkpoint'])
    assert resume == O / 'runs/mopd/checkpoint-160'
    assert sha(resume / 'checkpoint_complete.json') == policy['resume_seal_sha256']
    assert not (O / 'runs/mopd/complete.json').exists()
    assert json.loads((O / 'runs/repair/complete.json').read_text())['steps'] == 8192
    if node == 0:
        assert verify(resume, 32)['global_step'] == 160
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                                      '--format=csv,noheader,nounits'], text=True).splitlines()
    assert len(devices) == 8
    for line in devices:
        _, _, model, memory = map(str.strip, line.split(','))
        assert 'A800' in model and int(memory) >= 79000
    env = environment(range(8), 'coordinate_resume_' + str(node))
    env['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    for key, leaf in [('TRITON_CACHE_DIR', 'triton'), ('XDG_CACHE_HOME', 'xdg'),
                      ('VLLM_CACHE_ROOT', 'vllm'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                      ('CUDA_CACHE_PATH', 'cuda')]:
        cache = Path('/tmp/anvil-coordinate-resume') / str(node) / leaf
        cache.mkdir(parents=True, exist_ok=True)
        env[key] = str(cache)
    (Path(env['XDG_CACHE_HOME']) / 'torch/kernels').mkdir(parents=True, exist_ok=True)
    probe = O / 'runs/mopd' / f'.resume_storage_probe_{node}'
    blocks = 3072 if node == 0 else 64
    try:
        with probe.open('xb') as stream:
            block = os.urandom(1024 ** 2)
            for _ in range(blocks):
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        assert probe.stat().st_size == blocks * 1024 ** 2
    finally:
        probe.unlink(missing_ok=True)
    save(W / f'node{node}_preflight.json', dict(state='passed', node=node,
         write_probe_bytes=blocks * 1024 ** 2, checked=time.time()))
    address = os.environ['MLP_WORKER_0_HOST']
    assert address not in ('localhost', '127.0.0.1')
    argv = [TRAIN, '-m', 'torch.distributed.run', '--nnodes=4', '--nproc_per_node=8',
            '--node_rank=' + str(node), '--master_addr=' + address,
            '--master_port=' + os.environ.get('MLP_WORKER_0_PORT', '29890'),
            '--max_restarts=0', '--rdzv_conf=timeout=600', str(O / 'train.py'),
            '--spec', str(O / 'spec.json'), '--stage', 'mopd', '--resume', str(resume)]
    save(W / f'node{node}_launch.json', dict(node=node, devices=devices, argv=argv,
         started=time.time(), resume_step=160, original_training_source_unchanged=True))
    log = W / f'console_node{node}.log'
    with log.open('x') as stream:
        proc = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        while proc.poll() is None:
            time.sleep(3)
            with log.open('rb') as reader:
                reader.seek(max(0, log.stat().st_size - 16384))
                tail = reader.read()
            if b'SafetensorError:' in tail or b'OutOfMemoryError:' in tail:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise RuntimeError('Fatal training exception; process group stopped. See ' + str(log))
        assert proc.returncode == 0, f'torchrun exited {proc.returncode}'
    complete = json.loads((O / 'runs/mopd/complete.json').read_text())
    assert complete['steps'] == 256
    if node == 0:
        assert verify(complete['checkpoint'], 32)['global_step'] == 256
    save(W / f'node{node}_complete.json', dict(state='mopd256_complete', completed=time.time(),
         checkpoint=complete['checkpoint'], resume_step=160))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(W / ('node' + os.environ.get('MLP_ROLE_INDEX', 'unknown') + '_failure.json'),
             dict(error=repr(exc), time=time.time()))
        raise
