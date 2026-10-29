"""Seal a reviewed formal budget using exact real-GPU execution/scaling evidence."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import shlex
import sys
import time
import yaml
from formal_common import HERE, C, D, R, TRAIN, bound, read, require, save, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--nodes', type=int, choices=[1, 2], required=True)
    ap.add_argument('--steps', type=int, required=True)
    ap.add_argument('--reserve-hours', type=float, default=24)
    args = ap.parse_args()
    require(args.steps >= 128 and args.steps % 128 == 0, 'Budget must use complete128-step checkpoint intervals')
    require(args.reserve_hours >= 24, 'Reserve at least24hours for final review/recovery/upload')
    require(not args.output.exists(), 'Use a new immutable formal run directory')
    review_path = HERE / 'candidate1_quality_review.json'; review = read(review_path)
    require(review['decision'] == 'admit_exact_candidates_for_formal_MOPD_with_intermediate_evaluation',
            'Missing evidence-backed teacher decision')
    for file, digest in review['evidence'].items():
        require(sha(file) == digest, 'Changed quality evidence')
    require(all(review['family_assessments'][f]['eligible'] is True for f in 'GSA'), 'Unreviewed teacher')
    candidate = D / 'mopd_v6_candidate1'
    binding = read(candidate / 'bindings.json')
    require(binding['student'] == review['student'] and binding['teachers'] == review['teachers'],
            'Quality review uses other candidates')
    world = args.nodes * 8
    runtime_root = candidate if world == 8 else D / 'mopd_v6_candidate1_multinode/actual'
    runtime_path = runtime_root / 'resume/runtime_verified.json'; runtime = read(runtime_path)
    require(runtime['state'] == 'passed_execution_only' and runtime['world'] == world,
            'Exact chosen topology has no real optimizer-resume proof')
    sys.path.insert(0, str(C))
    from mopd_lineage import bind_lineage, validate_runtime
    teachers = {f:binding['teachers'][f]['adapter'] for f in 'GSA'}
    initial = binding['student']['adapter']
    validate_runtime(runtime, C / 'native_mopd_v6.py', initial, teachers)
    require(runtime['representation_sha256'] == binding['representation_sha256'], 'Different model interface')
    lineage = candidate / 'initialization_lineage.json'
    bind_lineage(lineage, initial, required=True)
    scaling_path = D / 'mopd_scaling_comparison.json'; scaling = read(scaling_path)
    require(scaling['state'] == 'complete_equal_source_throughput_comparison_requires_budget_review'
            and scaling['same_source_UID_multiset'] and scaling['same_exact_initial_student_and_teachers'],
            'Missing matched throughput comparison')
    timing = scaling['world%d' % world]
    for file, digest in timing['evidence'].items():
        require(sha(file) == digest, 'Changed throughput evidence')
    require(world == 8 or scaling['wall_time_speedup_16_over8'] >= 1.25,
            'Use16GPUs only with at least25percent measured wall-clock gain')
    require(world == 16 or scaling['wall_time_speedup_16_over8'] < 1.25,
            'Prefer measured faster16GPU topology for this deadline')
    measured = timing['elapsed_including_intermediate_saves_seconds'] / timing['measured_updates']
    guard = max(measured * 1.5, timing['p90_update_seconds'])
    deadline = datetime(2026, 11, 2, 5, 48, tzinfo=timezone.utc).timestamp()
    remaining = deadline - time.time()
    projected = args.steps * guard + 3600
    require(projected + args.reserve_hours * 3600 <= remaining,
            'Selected budget consumes final validation/upload reserve')
    root = args.output.resolve(); root.mkdir(parents=True)
    eligibility = dict(schema='anvil-native-teacher-eligibility-v6', verdict='ELIGIBLE',
        data_contract='anvil-source-native-train-v1',
        training_export_manifest_sha256=sha(R.parent / 'arxiv/native_training_export_20261012_v4_r2/manifest.json'),
        teachers={f:dict(**binding['teachers'][f], eligible=True,
                        assessment=review['family_assessments'][f]) for f in 'GSA'},
        joint_adapter=initial, joint_adapter_sha256=binding['student']['adapter_sha256'],
        model_representation_sha256=binding['representation_sha256'],
        runtime_evidence=bound(runtime_path), quality_review=bound(review_path),
        final_model_accepted=False, universal_teacher_superiority_claimed=False,
        note='Admission for actual distillation based on complementary matched knowledge; final outcome requires evaluation.')
    eligibility_path = root / 'eligibility.json'; save(eligibility_path, eligibility)
    original = read(candidate / 'experiment.json')
    frozen = dict(original['frozen_files'])
    for path in [*HERE.glob('*.py'), HERE / 'derivation.json', review_path, eligibility_path,
                 scaling_path, runtime_path]:
        frozen[str(path)] = sha(path)
    budget = dict(measured_seconds_per_update=measured, guarded_seconds_per_update=guard,
                  projected_training_with_startup_hours=projected / 3600,
                  final_reserve_hours=args.reserve_hours, deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
                  source='24 equal-source full-budget real-pool updates, including intermediate saves',
                  unobserved_long_tail_uncertain=True,
                  monitoring_rule='Recompute ETA from formal rolling128updates; preserve24hour final reserve. Any change to frozen update budget requires a separately declared run, not silent schedule mutation.')
    plan = dict(schema='anvil-formal-MOPD-plan-v1', smoke=False, world=world, global_batch=32,
                steps=args.steps, training_run=str(root / 'training'), eligibility=str(eligibility_path),
                initialization_lineage=str(lineage), budget=budget,
                evaluation_steps=sorted({s for s in [128,512,1024,args.steps] if s <= args.steps}),
                planned_exposures=dict(total=args.steps * 32, native_distillation=args.steps * 16,
                                      general_qa=args.steps * 8, spatial_reference=args.steps * 8),
                exposure_counts_are_planned_not_actual=True,
                evidence={k:bound(p) for k,p in {'quality_review':review_path,'runtime':runtime_path,
                         'scaling':scaling_path,'initialization_lineage':lineage}.items()},
                frozen_files=frozen, final_quality_accepted=False, weights_uploaded=False, created=time.time())
    if (HERE / 'launch_policy.json').exists():
        decision = read(HERE / 'budget_decision.json')
        require(decision['nodes'] == args.nodes and decision['updates'] == args.steps,
                'Formal plan differs from the declared measured budget decision')
        require(decision['policy_sha256'] == sha(HERE / 'launch_policy.json')
                and decision['scaling_sha256'] == sha(scaling_path), 'Changed budget evidence')
        for key, file in [('launch_policy', HERE / 'launch_policy.json'),
                          ('budget_decision', HERE / 'budget_decision.json')]:
            plan['evidence'][key] = bound(file)
            plan['frozen_files'][str(file)] = sha(file)
    plan_path = root / 'plan.json'; save(plan_path, plan)
    config = yaml.safe_load((D / 'mopd_v6_candidate1_multinode/cli.yaml').read_text())
    config['TaskName'] = 'task_anvil_formal_MOPD_' + root.name + '_A800x' + str(world)
    config['Description'] = 'Formal full-pool MOPD; reviewed exactteachers; globalbatch32; updates%d; no final quality or upload claim' % args.steps
    config['TaskRoleSpecs'][0]['RoleReplicas'] = args.nodes
    config['ActiveDeadlineSeconds'] = int(remaining - args.reserve_hours * 3600)
    config['Entrypoint'] = 'set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n' + shlex.join(
        [TRAIN, '-u', str(HERE / 'train_node.py'), '--plan', str(plan_path)])
    (root / 'training_cli.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    for step in plan['evaluation_steps']:
        dest = root / 'evaluation' / str(step); dest.mkdir(parents=True)
        config['TaskName'] = 'task_anvil_MOPD_checkpoint%d_public_general_A800x8' % step
        config['Description'] = 'Immutable formal intermediate checkpoint; full seven public and10105general; not final validation'
        config['TaskRoleSpecs'][0]['RoleReplicas'] = 1
        config['ActiveDeadlineSeconds'] = 21600
        config['Entrypoint'] = 'set -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n' + shlex.join(
            [TRAIN, '-u', str(HERE / 'evaluate_checkpoint.py'), '--plan', str(plan_path), '--step', str(step), '--output', str(dest)])
        (dest / 'cli.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(plan_path)
    print(budget)


if __name__ == '__main__':
    main()
