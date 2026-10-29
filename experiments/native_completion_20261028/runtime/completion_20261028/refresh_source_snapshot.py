"""Refresh only the explicit source/evidence archive, never data or live training."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

C = Path(__file__).resolve().parent
R = C.parent
DEST = Path('/Knowin/demo/weilinruan/Anvil/01_Code/experiments/native_completion_20261028')
SECRET = re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})')


def copy(source, target):
    raw = source.read_bytes()
    assert not SECRET.search(raw), 'Credential-like content in ' + str(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    return dict(source=str(source), sha256=digest, bytes=len(raw))


def main():
    manifest = json.loads((DEST / 'snapshot_manifest.json').read_text())
    sources = {Path(item['source']) for item in manifest['files'].values()}
    sources |= set((C / 'objective_controls_20261029').glob('*.py'))
    sources |= set((C / 'objective_controls_phase2_20261029').glob('*.py'))
    sources |= {C / 'native_mopd_v7.py', C / 'review_prism_learning_candidates.py',
                C / 'review_objective_controls.py', C / 'refresh_source_snapshot.py'}
    sources |= {C / name for name in ['checkpoint_prefix_audit.py', 'validate_checkpoint_prefix_audit.py',
                                     'prepare_next_teacher_candidate.py', 'audit_mtd_reduction.py']}
    sources |= {C / name for name in ['objective_native_diagnostic.py', 'submit_objective_native_diagnostic.py',
                                     'validate_objective_native_diagnostic.py']}
    sources |= {C / name for name in ['review_completed_control_public.py', 'review_phase2_objectives.py',
                                     'validate_phase2_scheduling.py']}
    sources |= {C / name for name in ['audit_vacancy_source_lineage.py',
                                     'prepare_vacancy_expansion_candidate.py']}
    selected=C/'final_delivery_selected_v2'
    sources |= set(selected.glob('*.py')) | {selected/'README.md'}
    sources |= {C/'midrun_gs_diagnostic_512.py',C/'submit_midrun_gs512.py'}
    sources |= {C/'mask_decode_probe_20261029.py',C/'prepare_mask_decode_probe_20261029.py'}
    for source in sorted(sources):
        relative = source.relative_to(R)
        manifest['files'][str(relative)] = copy(source, DEST / 'runtime' / relative)
    extra = {
        'formal_checkpoint128_integrity.json': C / 'formal_mopd/runs/candidate1_20261029/checkpoint128_integrity_review.json',
        'formal_checkpoint128_public_report.json': C / 'formal_mopd/runs/candidate1_20261029/evaluation/128/report.json',
        'formal_checkpoint128_native_GS_report.json': C / 'native_diagnostics/candidate1_step128/report.json',
        'prism_learning_matched_review.json': C / 'diagnosis_32gpu/prism_learning_matched_review.json',
        'objective_controls_cpu_validation.json': C / 'objective_controls_20261029/cpu_validation.json',
        'objective_controls_experiment.json': C / 'objective_controls_20261029/experiment.json',
        'objective_controls_derivation.json': C / 'objective_controls_20261029/derivation.json',
        'checkpoint_prefix128_audit.json': C / 'checkpoint_prefixes/candidate1_step128/report.json',
        'checkpoint_prefix_validation.json': C / 'checkpoint_prefix_validation.json',
        'next_teacher_candidate_preparation.json': C / 'next_candidate_prism_20261029/preparation.json',
        'next_A_teacher_lineage.json': C / 'next_candidate_prism_20261029/A_teacher_lineage.json',
        'objective_topology_clarification.json': C / 'objective_controls_20261029/topology_clarification.json',
        'objective_native_cpu_validation.json': C / 'objective_native_diagnostics/cpu_validation.json',
        'objective_native_launch_policy.json': C / 'objective_native_diagnostics/low_native_sft/policy.json',
        'objective_native_schedule_outcome.json': C / 'objective_native_diagnostics/low_native_sft/controller_status.json',
        'objective_phase2_cpu_validation.json': C / 'objective_controls_phase2_20261029/cpu_validation.json',
        'objective_phase2_experiment.json': C / 'objective_controls_phase2_20261029/experiment.json',
        'objective_phase2_scheduling_validation.json': C / 'objective_controls_phase2_20261029/scheduling_validation.json',
        'auxiliary_replay_overlap_427.json': C / 'auxiliary_replay_overlap_20261029.json',
        'formal_checkpoint512_public_report.json': C / 'formal_mopd/runs/candidate1_20261029/evaluation/512/report.json',
        'formal_checkpoint512_review.json': C / 'formal512_review_20261029.json',
        'checkpoint_prefix512_audit.json': C / 'checkpoint_prefixes/candidate1_step512/report.json',
        'vacancy_upstream_fidelity.json': C / 'vacancy_upstream_fidelity_20261029.json',
        'vacancy_expansion_candidate_policy.json': C / 'vacancy_expansion_candidate_20261029/policy.json',
        'vacancy_expansion_visual_review.json': C / 'vacancy_expansion_visual_review_20261029.json',
        'selected_delivery_derivation.json': selected/'derivation.json',
        'selected_delivery_cpu_validation.json': selected/'selection_cpu_validation.json',
        'selected_delivery_interface_cpu_validation.json': selected/'interface_validation/cpu_validation.json',
        'selected_delivery_rotation128_parity.json': selected/'interface_validation/rotation128_interface_validation.json',
        'formal512_native_cpu_validation.json': C/'native_diagnostics/candidate1_step512/cpu_validation.json',
        'formal512_native_policy.json': C/'native_diagnostics/candidate1_step512/policy.json',
        'formal512_native_spec.json': C/'native_diagnostics/candidate1_step512/spec.json',
    }
    for arm in ['low_native_sft', 'no_mtd']:
        for kind, path in {
            'submission':C / 'objective_controls_20261029/submissions' / arm / 'submission.json',
            'runtime':C / 'objective_controls_20261029/runtime' / arm / 'runtime_verified.json'
        }.items():
            if path.exists():
                extra['objective_' + arm + '_' + kind + '.json'] = path
        for kind, path in {
            'public_report':C / 'objective_controls_20261029/evaluation' / arm / 'report.json',
            'exposure_audit':C / 'objective_controls_20261029/runs' / arm / 'exposure_audit.json'
        }.items():
            if path.exists():
                extra['objective_' + arm + '_' + kind + '.json'] = path
    comparison = C / 'objective_controls_20261029/comparison.json'
    if comparison.exists():
        extra['objective_controls_comparison.json'] = comparison
    for arm in ['no_native_sft', 'low_native_sft_no_mtd']:
        for kind,path in {
            'submission':C/'objective_controls_phase2_20261029/submissions'/arm/'submission.json',
            'runtime':C/'objective_controls_phase2_20261029/runtime'/arm/'runtime_verified.json',
            'public_report':C/'objective_controls_phase2_20261029/evaluation'/arm/'report.json'
        }.items():
            if path.exists():extra['phase2_'+arm+'_'+kind+'.json']=path
    expansion = C / 'vacancy_expansion_candidate_20261029/manifest.json'
    if expansion.exists():
        extra['vacancy_expansion_candidate_manifest.json'] = expansion
    for name,path in {
        'phase2_objective_comparison.json':C/'objective_controls_phase2_20261029/comparison.json',
        'formal512_native_submission.json':C/'native_diagnostics/candidate1_step512/submission.json',
        'formal512_native_report.json':C/'native_diagnostics/candidate1_step512/report.json',
        'formal512_native_review.json':C/'native512_review_20261029.json',
        'formal512_rollout_efficiency.json':C/'rollout_efficiency_step512_20261029.json',
        'paired_mask_failure.json':C/'paired_mask_failure_20261029.json',
        'candidate1_quality_stop.json':C/'formal_candidate1_quality_stop_20261029.json',
        'candidate1_controller_retirement.json':C/'candidate1_controller_retirement_20261029.json',
        'mask_decode_probe_spec.json':C/'mask_decode_probe_20261029/spec.json',
        'mask_decode_probe_cpu_validation.json':C/'mask_decode_probe_20261029/cpu_validation.json',
        'mask_decode_probe_submission.json':C/'mask_decode_probe_20261029/submission.json',
    }.items():
        if path.exists():extra[name]=path
    for name, source in extra.items():
        manifest['evidence'][name] = copy(source, DEST / 'evidence' / name)
    for name, item in manifest['evidence'].items():
        assert hashlib.sha256((DEST / 'evidence' / name).read_bytes()).hexdigest() == item['sha256']
    manifest.update(created_utc=datetime.now(timezone.utc).isoformat(),
        first_formal_checkpoint_public_regression=True,
        objective_controls=dict(final_mopd=False, steps=128, source_rows_matched=4096,
            world=8, sampler_world=16, schedule_steps=1664,
            limitation='Physical topology changes generation RNG and floating-point reduction order; all-eligible per-example objective weights are equal.'),
        formal_training_completed=False, final_model_uploaded=False)
    (DEST / 'snapshot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    readme = DEST / 'README.md'
    text = readme.read_text()
    marker = '\n## First formal checkpoint and objective controls\n'
    text = text.split(marker)[0] + marker + '''
The verified formal step128 checkpoint has seven-public macro68.614286 versus
the exact initial student's69.982857 (−1.368571 percentage points). Its general
scores remain essentially stable. On35,989 exact paired native G/S examples,
gains are small and mask validity is essentially unchanged. This is unresolved
public spatial regression, not an accepted final result or proof of collapse.

Two nonfinal128-update controls test native-reference SFT0.1/MTD1 and SFT1/MTD0.
They preserve the1664-update learning-rate horizon and exactly match all4096
training rows in the original first128 updates. Eight physical GPUs with a
virtual16 sampler differ in rank RNG and floating-point reduction order, so
these are diagnostic comparisons rather than exact stochastic replications.
The reduction audit corrects earlier frozen notes: MTD normalizes over eligible
examples; all observed nonempty native rows keep the same per-example weight.
The zero-MTD arm cannot be released as MOPD. Each task requires a genuine GPU
optimizer1→3 resume check before training, then unchanged seven-public and
10,105-example general evaluation. Current task receipts are archived;
source/CPU checks alone do not establish GPU success or quality improvement.

Both phase1 controls have now completed full public/general evaluation. Native
SFT0.1/MTD1 reaches69.278571 public macro; SFT1/MTD0 reaches68.682857. Both remain
below the initial69.982857, while general performance remains stable. The lower
native SFT weight mitigates part of the regression; no sole cause is established.
Further128-update controls test SFT0/MTD1 and SFT0.1/MTD0 with identical initial
student, teachers, data and schedule. The first reserves eight public-evaluation
slots; the second waits for formal512 public success before using spare capacity.
All controls remain excluded from final delivery.
Phase2 nativeSFT0.1/MTD0 now completes at68.098571 macro, versus69.278571
with the same nativeSFT weight and activeMTD. Where2Place is56.53 versus62.52;
general scores remain stable. The zero-native-SFT/activeMTD arm is still running.
These descriptive short-run results do not support simply reducing nativeSFT
without distillation, and do not establish a single cause of the regression.

Formal step512 has now completed full seven-public/general evaluation. Its
public macro69.088571 recovers0.474286pp versus step128 but remains0.894286pp
below initialization. RefSpatial Location100 is51.30 and Where2Place60.70;
ERQA49.83 improves1.05pp over initialization. This is not accepted final quality.
The subsequent matched native512 diagnostic found mask validity falling from
50.17% to43.45% and IoU from0.08636 to0.07216 on5,079 unchanged examples.
The formal run was explicitly stopped for quality review after669 logged
updates; all22 files of the640 recovery checkpoint and its20,480-row committed
prefix were verified. The original1664-step budget was not completed. Its
training controller and final-tip handoff were retired without final submissions.
No checkpoint was selected for release; the final objective remains unfinished.
Paired failures show2,842 exact-cycle stops at512 versus2,497 initially. By
contrast all8,192 committed training rollouts through512 ended with EOS.
A separate fixed32-example HF probe compares greedy versus sampled decoding
at initialization and512. It is inference-only and cannot establish final quality.

A read-only upstream audit matched all3,384 active vacancy records exactly to
source coordinates and image dimensions. The raw source has231,554 records,
including58 out-of-range targets already excluded from the231,496 processed
records. The active conversion is faithful, but semantic correctness is not
independently established. A separate CPU preparation screens the full source
against native/public/general/test/development images and fixed scan reserves.
This candidate inventory is not admitted to active training or counted as
actual coverage; its manifest is archived only when preparation completes.
Preparation retained182,878 records/35,736 images, including180,360 records
absent from the existing3,384-record pool. Stronger scan/development exclusions
remove866 existing rows from this separate candidate. Eight flagged public or
general image pairs affecting28 such rows were visually inspected and found
to show distinct content; conservative exclusions remain in place. These new
reservations do not erase prior student exposure or establish a blind test.

An independent final_delivery_selected_v2 workflow supports explicit selection
of a sealed formal checkpoint after its cloud run is terminal. It freezes only
the committed source-exposure prefix, discloses selected versus planned updates
and the original run's actual disposition, and preserves full580K/seven-public/
10105-general evaluation and explicit quality review. The original live handoff
and all of its delivery files are unchanged. CPU tests use the real512 checkpoint,
its merged model and16384-row prefix; terminal states are simulated only in
deleted temporary fixtures. No checkpoint has been selected for final delivery.

The expanded A learning control improves pose80.65°→76.37°, contact error and
affordance IoU, while action accuracy falls about0.96pp. Pose remains worse
than the train-only category prior. It has not replaced the live formal teacher.
'''
    text = text.replace('The original live handoff\nand all of its delivery files are unchanged.',
        'The original delivery files are unchanged; its former live handoff was\nexplicitly retired after the quality stop recorded above.')
    readme.write_text(text)
    print(json.dumps(dict(source_files=len(manifest['files']), evidence_files=len(manifest['evidence']),
                          secrets_scan_passed=True, archived_bytes_verified=True)))


if __name__ == '__main__':
    main()
