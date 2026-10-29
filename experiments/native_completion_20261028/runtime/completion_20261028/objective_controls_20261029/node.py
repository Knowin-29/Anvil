"""One eight-GPU nonfinal control: real resume, matched128 updates, full public/general."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
import subprocess
import sys
import time
from pathlib import Path
from objective_control_common import HERE, C, D, R, TRAIN, bound, control_checkpoint, load_experiment, read, require, save, sha, verify
sys.path.insert(0, str(D))
from retention_node import environment, run


def audit_exposures(root, steps, mtd, baseline):
    counts = Counter(); forms = Counter(); seen = set(); rows = defaultdict(list)
    for rank in range(8):
        for line in (root / ('exposure_rank%d.jsonl' % rank)).open():
            row = __import__('json').loads(line)
            key = (row['step'], row['micro'], row['rank'])
            require(key not in seen and row['rank'] == rank and 1 <= row['step'] <= steps
                    and 0 <= row['micro'] < 4, 'Duplicate or wrong exposure')
            seen.add(key); counts[row['source_category']] += 1; forms[row['form']] += 1
            rows[row['step']].append([row['canonical_uid'], row['source_category'],
                                      row['family'], row['form'], row['source_binding']])
            if row['source_category'] != 'native_distillation' or mtd == 0:
                require(row['rollout_tokens'] == row['teacher_scored_positions'] == 0,
                        'Unexpected teacher use in reference-only control')
            else:
                require(row['rollout_tokens'] > 0 and row['teacher_scored_positions'] > 0,
                        'Missing real native distillation')
    require(len(seen) == steps * 32, 'Incomplete rank-step coverage')
    canonical = lambda items: sorted(__import__('json').dumps(x, sort_keys=True) for x in items)
    for step, items in rows.items():
        require(canonical(items) == canonical(baseline[str(step)]), 'Source rows differ from formal run')
    state = read(root / ('checkpoint-%d/trainer_state.json' % steps))
    require(state['source_exposures'] == dict(counts), 'Exposure totals do not match checkpoint')
    require(len(state['log_history']) == steps, 'Incomplete update history')
    for entry in state['log_history']:
        require(all(math.isfinite(entry[k]) for k in ['sft', 'mtd', 'grad_norm'])
                and entry['grad_norm'] > 0, 'Nonfinite or zero gradient')
        require(entry['mtd'] > 0 if mtd else entry['mtd'] == 0, 'Unexpected MTD loss')
    return dict(records=len(seen), source_exposures=dict(counts), forms=dict(forms),
                unique_by_category={k:len({r[0] for items in rows.values() for r in items if r[1] == k}) for k in counts},
                matched_source_rows_per_update=True, stochastic_rollouts_identical_claimed=False,
                topology_note='Sources and update schedule match16GPU formal; RNG and native token normalization groups differ on8GPUs.',
                exposure_files={str(root / ('exposure_rank%d.jsonl' % rank)):sha(root / ('exposure_rank%d.jsonl' % rank)) for rank in range(8)})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True)
    args = ap.parse_args(); experiment = load_experiment()
    require(args.arm in experiment['arms'], 'Unknown objective control')
    arm = experiment['arms'][args.arm]
    out = HERE / 'runs' / args.arm; out.mkdir(parents=True, exist_ok=True)
    require(not (out / 'protocol.json').exists(), 'Existing main run needs explicit recovery')
    devices = []
    for line in subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                                        '--format=csv,noheader,nounits'], text=True).splitlines():
        gpu, uid, name, memory = map(str.strip, line.split(','))
        require('A800' in name and int(memory) >= 79000, 'Unexpected GPU')
        devices.append((int(gpu), uid))
    devices.sort(); require(len(devices) == 8, 'Expected eight GPUs')
    env = environment([x[0] for x in devices], 'objective_control_' + args.arm)
    (Path(env['XDG_CACHE_HOME']) / 'torch/kernels').mkdir(parents=True, exist_ok=True)
    baseline = read(experiment['baseline_source_rows'])['rows']
    prefix = [TRAIN, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=8',
              '--master_port', '29870', str(C / 'native_mopd_v7.py'), '--global-batch', '32',
              '--objective-control', '--native-sft-weight', str(arm['native_sft_weight']),
              '--mtd-weight', str(arm['mtd_weight']), '--schedule-steps', '1664',
              '--sampler-world-size', '16', '--spatial-replay', '--dof-representation', 'rotation128',
              '--initialization-lineage', experiment['initialization_lineage']]
    proof = HERE / 'runtime' / args.arm
    save(out / 'node_status.json', dict(stage='real_gpu_resume_proof', devices=devices, started=time.time()))
    smoke = prefix + ['--output', str(proof), '--smoke', '--full-budget-probe', '--save-every', '1',
                      '--runtime-bindings', experiment['runtime_bindings']]
    run(smoke + ['--steps', '1'], env, proof / 'first.log')
    run(smoke + ['--steps', '3', '--resume', str(proof / 'checkpoint-1')], env, proof / 'resume.log')
    first, last = verify(proof / 'checkpoint-1', 8), verify(proof / 'checkpoint-3', 8)
    require(first['global_step'] == 1 and last['global_step'] == 3, 'Resume did not advance')
    require(first['files']['adapter_model.safetensors']['sha256'] != last['files']['adapter_model.safetensors']['sha256'],
            'Adapter did not change')
    run([TRAIN, '-c', "import sys,torch; s=torch.load(sys.argv[1],map_location='cpu',weights_only=True)['state']; assert s and {int(v['step']) for v in s.values()}=={3}; print('actual optimizer resumed from1 to3')",
         str(proof / 'checkpoint-3/optimizer.pt')], env, proof / 'optimizer_verify.log')
    audit = audit_exposures(proof, 3, arm['mtd_weight'], baseline)
    protocol = read(proof / 'protocol.json')
    receipt = dict(state='passed_execution_only', world_size=8, sampler_world_size=16,
        schedule_steps=1664, loss_weights=protocol['loss_weights'], optimizer_step=3, changed_adapter=True,
        representation_sha256=protocol['model_representation_sha256'],
        trainer_code_sha256=sha(C / 'native_mopd_v7.py'),
        student_adapter_sha256=sha(Path(protocol['initial']) / 'adapter_model.safetensors'),
        teacher_adapter_sha256={f:sha(Path(p) / 'adapter_model.safetensors') for f,p in protocol['teachers'].items()},
        source_audit=audit, teacher_loaded=arm['mtd_weight'] > 0, objective_control=True, final_mopd=False,
        protocol_sha256=sha(proof / 'protocol.json'), checkpoint_manifest_sha256=sha(proof / 'checkpoint-3/checkpoint_complete.json'),
        completed=time.time())
    save(proof / 'runtime_verified.json', receipt)
    eligibility = read(experiment['original_eligibility'])
    eligibility['runtime_evidence'] = bound(proof / 'runtime_verified.json')
    eligibility['control_only_note'] = 'Original teacher usefulness assessment preserved; this cloned receipt authorizes only a nonfinal objective control.'
    save(out / 'eligibility.json', eligibility)
    save(out / 'node_status.json', dict(stage='control_training', steps=128, updated=time.time()))
    run(prefix + ['--output', str(out), '--steps', '128', '--save-every', '128',
                  '--eligibility', str(out / 'eligibility.json')], env, out / 'training.log')
    checkpoint = out / 'checkpoint-128'
    complete, protocol, receipt = control_checkpoint(checkpoint)
    audit = audit_exposures(out, 128, arm['mtd_weight'], baseline)
    save(out / 'exposure_audit.json', dict(state='verified_matched4096_exposures', **audit, final_mopd=False))
    evaluation = HERE / 'evaluation' / args.arm; model = evaluation / 'merged'
    save(out / 'node_status.json', dict(stage='merge_and_public_general', updated=time.time()))
    run([TRAIN, str(HERE / 'merge.py'), '--adapter', str(checkpoint), '--output', str(model)], env, evaluation / 'merge.log')
    merge = read(model / 'merge_receipt.json')
    require(merge['state'] == 'verified_derived_inference_copy' and
            merge['adapter_sha256'] == receipt['files']['adapter_model.safetensors']['sha256'], 'Incomplete/wrong merge')
    manifest = read(D / 'inputs/manifest.json')
    spec = dict(schema='anvil-objective-control-evaluation-v1', arm=args.arm, checkpoint=str(checkpoint),
        step=128, model=str(model), public_shards=8, general_records=10105,
        public_records={k:v['n'] for k,v in manifest['public'].items()},
        model_files={p.name:sha(p) for p in model.iterdir() if p.is_file()},
        evidence={k:bound(p) for k,p in dict(experiment=HERE / 'experiment.json',
            checkpoint=checkpoint / 'checkpoint_complete.json', training_protocol=out / 'protocol.json',
            exposure_audit=out / 'exposure_audit.json', inputs=D / 'inputs/manifest.json').items()},
        frozen_files=experiment['frozen_files'], final_mopd=False, final_quality_accepted=False,
        full_native_validation=False, heldout_used_adaptively=True)
    spec_path = evaluation / 'spec.json'; save(spec_path, spec)
    def evaluate(shard):
        gpu, uid = devices[shard]; dest = evaluation / 'public' / str(shard)
        local = environment([gpu], 'objective_' + args.arm + '_eval' + str(shard))
        local['ANVIL_EXPECTED_GPU_UUID'] = uid
        run([TRAIN, str(HERE / 'public_worker.py'), '--spec', str(spec_path), '--model', str(model),
             '--role', 'objective_control_' + args.arm, '--shard', str(shard), '--shards', '8',
             '--output', str(dest)], local, dest / 'console.log')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(evaluate, range(8)))
    run([TRAIN, str(HERE / 'score.py'), '--spec', str(spec_path), '--output', str(evaluation)], env, evaluation / 'score.log')
    save(out / 'node_status.json', dict(stage='control_complete_requires_review', report_sha256=sha(evaluation / 'report.json'), completed=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if '--arm' in sys.argv:
            out = HERE / 'runs' / sys.argv[sys.argv.index('--arm') + 1]
            save(out / 'failure.json', dict(error=repr(exc), time=time.time()))
        raise
