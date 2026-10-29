"""Two 4-GPU training arms per node; then frozen full evaluations.

Node-local GPU ownership; no shared-filesystem allocation lock. Failure of one
arm is preserved and does not erase or cancel the other arm's evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
C, R = ROOT.parent, ROOT.parent.parent
sys.path.insert(0, str(R / 'cloud_a800'))
from checkpoints import save, verify
from retention_node import environment, run
TRAIN, EVAL = str(R / '.venv-train/bin/python'), str(R / '.venv-vllm/bin/python')
condition = threading.Condition()
available, pending, failures = [], [], []
finished = 0


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def train_arm(arm, devices, port):
    global finished
    out = ROOT / 'spatial_repairs' / arm
    out.mkdir(parents=True, exist_ok=True)
    env = environment([d[0] for d in devices], 'spatial_' + arm)
    prefix = [TRAIN, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=4', '--master_port', str(port), str(ROOT / 'spatial_train.py'), '--arm', arm]
    try:
        smoke = ROOT / 'spatial_smokes' / arm
        run(prefix + ['--output', str(smoke), '--smoke', '--stop-after', '1'], env, out / 'smoke.log')
        run(prefix + ['--output', str(smoke), '--smoke', '--stop-after', '2', '--resume', str(smoke / 'checkpoint-1')], env, out / 'smoke_resume.log')
        one, two = verify(smoke / 'checkpoint-1', 4), verify(smoke / 'checkpoint-2', 4)
        assert one['global_step'] == 1 and two['global_step'] == 2
        assert one['files']['adapter_model.safetensors']['sha256'] != two['files']['adapter_model.safetensors']['sha256']
        check = [TRAIN, '-c', "import sys,torch; s=torch.load(sys.argv[1],map_location='cpu',weights_only=True)['state']; assert s and {int(v['step']) for v in s.values()}=={2}; print('optimizer resumed to 2')", str(smoke / 'checkpoint-2/optimizer.pt')]
        run(check, env, out / 'resume_validation.log')
        save(out / 'runtime_verified.json', dict(state='passed', checkpoint1=one['files']['adapter_model.safetensors']['sha256'], checkpoint2=two['files']['adapter_model.safetensors']['sha256'], optimizer_steps=2, final_mopd=False))
        run(prefix + ['--output', str(out)], env, out / 'training.log')
        receipt = json.loads((out / 'complete.json').read_text())
        assert receipt['step'] == 1024 and not receipt['smoke']
        merged = ROOT / ('merged_spatial_' + arm)
        run([TRAIN, str(ROOT / 'merge_spatial.py'), '--adapter', receipt['checkpoint'], '--output', str(merged)], env, out / 'merge.log')
        with condition:
            pending.extend(dict(kind='public', arm=arm, model=str(merged), shard=i, shards=4) for i in range(4))
            if arm.startswith('G_'):
                pending.extend(dict(kind='native', arm=arm, model=str(merged), shard=i, shards=8) for i in range(8))
            pending.sort(key=lambda t:(t['kind'] != 'public', t['shard'], t['arm']))
            condition.notify_all()
    except BaseException as exc:
        with condition:
            failures.append(dict(stage='training', arm=arm, error=str(exc)))
        save(out / 'failure.json', dict(stage='training', error=str(exc), time=time.time()))
    finally:
        with condition:
            available.extend(devices)
            finished += 1
            condition.notify_all()


def evaluation_worker():
    while True:
        with condition:
            while not (available and pending):
                if finished == 2 and not pending:
                    return
                condition.wait(20)
            gpu, uuid = available.pop(0)
            task = pending.pop(0)
        arm = task['arm']
        out = ROOT / ('spatial_benchmarks' if task['kind'] == 'public' else 'spatial_native') / arm / str(task['shard'])
        out.mkdir(parents=True, exist_ok=True)
        env = environment([gpu], 'spatial_eval_' + arm.split('_')[0] + '_' + str(gpu))
        env['ANVIL_EXPECTED_GPU_UUID'] = uuid
        try:
            if task['kind'] == 'public':
                argv = [TRAIN, str(ROOT / 'spatial_benchmark_worker.py'), '--model', task['model'], '--role', arm, '--shard', str(task['shard']), '--shards', '4']
            else:
                argv = [EVAL, str(C / 'eval_code/run_vllm_eval.py'), '--model', task['model'], '--name', arm,
                        '--scope', 'G', '--gpu-uuid', uuid, '--shard', str(task['shard']), '--shards', '8',
                        '--concurrency', '64', '--memory', '.88', '--output', str(out / 'evaluation')]
            save(out / 'launch.json', dict(argv=argv, gpu=gpu, uuid=uuid, started=time.time()))
            run(argv, env, out / 'console.log')
            if task['kind'] == 'public':
                assert json.loads((out / 'complete.json').read_text())['state'] == 'complete'
            else:
                metrics = json.loads((out / 'evaluation/metrics.json').read_text())
                assert metrics['state'] == 'complete_scope_shard'
                save(out / 'complete.json', dict(task=task, n=metrics['n'], metrics_sha256=sha(out / 'evaluation/metrics.json')))
        except BaseException as exc:
            with condition:
                failures.append(dict(stage='evaluation', task=task, error=str(exc)))
            save(out / 'failure.json', dict(error=str(exc), time=time.time()))
        finally:
            with condition:
                available.append((gpu, uuid))
                condition.notify_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=['joint', 'G'], required=True)
    args = ap.parse_args()
    # Fail closed on pre-submission script or source-pool changes.
    manifest = json.loads((ROOT / 'spatial_experiment.json').read_text())
    for path, digest in manifest['frozen_files'].items():
        assert sha(path) == digest, path
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert len(rows) == 8
    devices = []
    for row in rows:
        i, u, n, m = map(str.strip, row.split(','))
        assert 'A800' in n and int(m) >= 79000
        devices.append((int(i), u))
    devices.sort()
    status = ROOT / ('spatial_node_' + args.family + '_status.json')
    save(status, dict(stage='running', devices=devices, started=time.time(), gpus=8, family=args.family))
    arms = [args.family + '_control', args.family + '_spatial']
    with ThreadPoolExecutor(max_workers=8) as evaluators, ThreadPoolExecutor(max_workers=2) as trainers:
        ef = [evaluators.submit(evaluation_worker) for _ in range(8)]
        tf = [trainers.submit(train_arm, arm, devices[4*i:4*i+4], 29750+i) for i, arm in enumerate(arms)]
        for f in as_completed(tf):
            f.result()
        for f in as_completed(ef):
            f.result()
    save(status, dict(stage='failed' if failures else 'evaluations_complete_scoring', failures=failures, completed=time.time()))
    if failures:
        raise RuntimeError(str(failures))
    subprocess.run([TRAIN, str(ROOT / 'score_spatial_benchmarks.py'), '--family', args.family], check=True)
    if args.family == 'G':
        subprocess.run([TRAIN, str(ROOT / 'score_spatial_native.py')], check=True)
    save(status, dict(stage='complete_requires_quality_review', completed=time.time(), family=args.family))


if __name__ == '__main__':
    main()
