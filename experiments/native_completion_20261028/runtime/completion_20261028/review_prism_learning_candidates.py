"""Audit completed A-learning controls against the current teacher and student."""
import json
from pathlib import Path
import time
import midrun_gs_diagnostic as native

C, D = native.C, native.D


def main():
    roots = dict(current_A=D / 'pose_evaluations/A_pose_angular',
                 initial_student=D / 'student_adaptation_evaluations/student_adaptation',
                 A_prism_lr5e5=D / 'prism_learning_evaluations/A_prism_lr5e5',
                 A_prism_lr1e4=D / 'prism_learning_evaluations/A_prism_lr1e4')
    evidence, models = {}, {}
    for name, root in roots.items():
        rows = {}
        for shard in range(4):
            folder = root / str(shard)
            metrics = native.read(folder / 'evaluation/metrics.json')
            for row, protocol in native.checked_rows(folder, metrics['n'], evidence):
                identity = (row['native_form'], row['id'])
                native.require(identity not in rows, 'Repeated A example')
                native.require(protocol['model_decoder'] == 'anvil-dof-train-rotation128-v1',
                               'Different DoF representation')
                rows[identity] = (row, protocol)
        native.require(len(rows) == 4100, 'Incomplete A validation')
        models[name] = rows
    comparisons = {}
    for candidate in ['A_prism_lr5e5', 'A_prism_lr1e4']:
        comparisons[candidate] = {}
        for baseline in ['current_A', 'initial_student']:
            groups = native.paired_group_rows(models[baseline], models[candidate])
            for group in groups:
                group['metrics'] = {metric: dict(
                    baseline=value['initial_student'], candidate=value['formal_checkpoint'],
                    candidate_minus_baseline=value['checkpoint_minus_initial'])
                    for metric, value in group['metrics'].items()}
            comparisons[candidate][baseline] = groups
    report = dict(state='complete_matched_A_learning_control_review', n_per_model=4100,
        rows_and_references_exactly_matched=True, inference_protocols_matched=True,
        model_roots={k: str(v) for k, v in roots.items()}, comparisons=comparisons,
        candidate_for_future_consideration='A_prism_lr1e4',
        rationale='Among the two controls, LR1e-4 has lower pose/contact error and higher affordance IoU/action accuracy. Compared with current A, geometry/affordance improve while action accuracy falls slightly. It remains weaker than the train-only category pose prior; pose quality is not solved.',
        category_prior_rotation_error_degrees=70.88,
        current_formal_binding_changed=False, subsequent_formal_run_admitted=False,
        final_model_quality_accepted=False, weights_uploaded=False,
        next_decision='Review first formal checkpoint public and paired G/S diagnostics before deciding whether another teacher-bound run is useful. No hot swap in the running candidate.',
        evidence=evidence, code_sha256=native.sha(Path(__file__).resolve()), completed=time.time())
    path = D / 'prism_learning_matched_review.json'
    native.require(not path.exists(), 'Preserve the prior immutable review')
    native.save(path, report)
    print(path)
    for candidate, baselines in comparisons.items():
        print(candidate, json.dumps([{ 'form': g['form'], 'n': g['n'],
            'metrics': {k: v for k, v in g['metrics'].items() if k in [
                'pi_symmetric_rotation_error_degrees', 'normalized_projected_contact_l2',
                'region_iou', 'next_action_exact']}} for g in baselines['current_A']], indent=2))


if __name__ == '__main__':
    main()
