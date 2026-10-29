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
    sources |= {C / 'native_mopd_v7.py', C / 'review_prism_learning_candidates.py',
                C / 'refresh_source_snapshot.py'}
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
    }
    for arm in ['low_native_sft', 'no_mtd']:
        for kind, path in {
            'submission':C / 'objective_controls_20261029/submissions' / arm / 'submission.json',
            'runtime':C / 'objective_controls_20261029/runtime' / arm / 'runtime_verified.json'
        }.items():
            if path.exists():
                extra['objective_' + arm + '_' + kind + '.json'] = path
    for name, source in extra.items():
        manifest['evidence'][name] = copy(source, DEST / 'evidence' / name)
    for name, item in manifest['evidence'].items():
        assert hashlib.sha256((DEST / 'evidence' / name).read_bytes()).hexdigest() == item['sha256']
    manifest.update(created_utc=datetime.now(timezone.utc).isoformat(),
        first_formal_checkpoint_public_regression=True,
        objective_controls=dict(final_mopd=False, steps=128, source_rows_matched=4096,
            world=8, sampler_world=16, schedule_steps=1664,
            limitation='Physical topology changes RNG and token-normalization groups; diagnostic comparisons only.'),
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
virtual16 sampler differ in rank RNG and native token-normalization groups, so
these are diagnostic comparisons rather than exact stochastic replications.
The zero-MTD arm cannot be released as MOPD. Each task requires a genuine GPU
optimizer1→3 resume check before training, then unchanged seven-public and
10,105-example general evaluation. Current task receipts are archived;
source/CPU checks alone do not establish GPU success or quality improvement.

The expanded A learning control improves pose80.65°→76.37°, contact error and
affordance IoU, while action accuracy falls about0.96pp. Pose remains worse
than the train-only category prior. It has not replaced the live formal teacher.
'''
    readme.write_text(text)
    print(json.dumps(dict(source_files=len(manifest['files']), evidence_files=len(manifest['evidence']),
                          secrets_scan_passed=True, archived_bytes_verified=True)))


if __name__ == '__main__':
    main()
