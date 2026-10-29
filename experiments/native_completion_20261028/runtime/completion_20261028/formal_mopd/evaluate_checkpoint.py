"""Eight-GPU diagnostics of an immutable formal checkpoint while training continues."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
from formal_common import HERE, C, D, R, TRAIN, bound, load_plan, read, require, save, sha, sealed_checkpoint
sys.path.insert(0, str(D))
from retention_node import environment, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); plan = load_plan(args.plan)
    require(args.step in plan['evaluation_steps'], 'Checkpoint not in the evaluation plan')
    checkpoint = Path(plan['training_run']) / ('checkpoint-%d' % args.step)
    state, protocol, receipt = sealed_checkpoint(checkpoint)
    require(state['global_step'] == args.step and protocol['updates'] == plan['steps'], 'Wrong checkpoint')
    require(protocol['eligibility_sha256'] == sha(plan['eligibility']), 'Wrong eligibility')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    devices = []
    for line in subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                                        '--format=csv,noheader,nounits'], text=True).splitlines():
        index, uid, name, memory = map(str.strip, line.split(','))
        require('A800' in name and int(memory) >= 79000, 'Unexpected GPU')
        devices.append((int(index), uid))
    devices.sort(); require(len(devices) == 8, 'Expected eight local GPUs')
    env = environment([d[0] for d in devices], 'mopd_checkpoint_' + str(args.step))
    model = out / 'merged'
    spec_path = out / 'spec.json'
    try:
        save(out / 'status.json', dict(stage='merge_sealed_checkpoint', step=args.step, started=time.time()))
        if not (model / 'merge_receipt.json').exists():
            run([TRAIN, str(HERE / 'merge_checkpoint.py'), '--adapter', str(checkpoint), '--output', str(model)],
                env, out / 'merge.log')
        merge = read(model / 'merge_receipt.json')
        require(merge['state'] == 'verified_derived_inference_copy', 'Partial merge requires explicit recovery')
        require(merge['adapter_sha256'] == receipt['files']['adapter_model.safetensors']['sha256'], 'Wrong merge')
        for name, info in merge['shards'].items():
            require(sha(model / name) == info['derived_sha256'], 'Changed merge')
        manifest = read(D / 'inputs/manifest.json')
        files = list(HERE.glob('*.py')) + list((D / 'code').rglob('*.py'))
        files += [D / 'retention_node.py', R / 'cloud_a800/checkpoints.py', Path(manifest['retention']['vqa_reference'])]
        spec = dict(schema='anvil-formal-checkpoint-public-evaluation-v1', checkpoint=str(checkpoint.resolve()),
                    step=args.step, model=str(model), public_shards=8,
                    public_records={k:v['n'] for k,v in manifest['public'].items()}, general_records=10105,
                    model_files={p.name:sha(p) for p in model.iterdir() if p.is_file()},
                    evidence={k:bound(p) for k,p in {
                        'checkpoint':checkpoint / 'checkpoint_complete.json',
                        'training_protocol':checkpoint.parent / 'protocol.json',
                        'training_plan':args.plan, 'inputs':D / 'inputs/manifest.json'}.items()},
                    frozen_files={str(p):sha(p) for p in sorted(set(files))},
                    full_native_validation=False, final_quality_accepted=False,
                    note='Intermediate public/general diagnostics. Full580K native final verification remains required.')
        if spec_path.exists():
            require(read(spec_path) == spec, 'Cannot change an evaluation spec')
        else:
            save(spec_path, spec)
        save(out / 'status.json', dict(stage='public_general_evaluation', step=args.step, updated=time.time()))
        def evaluate(shard):
            gpu, uid = devices[shard]; dest = out / 'public' / str(shard)
            local = environment([gpu], 'mopd_checkpoint_%d_eval%d' % (args.step, shard))
            local['ANVIL_EXPECTED_GPU_UUID'] = uid
            run([TRAIN, str(HERE / 'checkpoint_public_worker.py'), '--spec', str(spec_path),
                 '--model', str(model), '--role', 'formal_step%d' % args.step, '--shard', str(shard),
                 '--shards', '8', '--output', str(dest)], local, dest / 'console.log')
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(evaluate, range(8)))
        run([TRAIN, str(HERE / 'score_checkpoint.py'), '--spec', str(spec_path), '--output', str(out)],
            env, out / 'scoring.log')
        save(out / 'status.json', dict(stage='complete_requires_quality_review', step=args.step,
                                      report_sha256=sha(out / 'report.json'), completed=time.time()))
    except BaseException as exc:
        save(out / 'failure.json', dict(error=repr(exc), time=time.time()))
        raise


if __name__ == '__main__':
    main()
