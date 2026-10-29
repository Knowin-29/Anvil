"""Validate real baseline bindings and capacity rules without submitting GPUs."""
import json
from pathlib import Path
import time
from unittest.mock import patch
import midrun_gs_diagnostic as m
import submit_midrun_gs_diagnostic as q


def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Expected mismatch rejection')


def main():
    selection = m.read(m.D / 'student_matched_native_selection.json')
    assert selection == m.baseline_selection()
    evidence, rows = {}, {}
    for task in selection['tasks']:
        for row, protocol in m.checked_rows(m.D / 'student_matched_native' / str(task['shard']),
                                            task['records'], evidence):
            key = (row['native_form'], row['id'])
            assert key not in rows
            rows[key] = (row, protocol)
    assert len(rows) == 35989
    groups = m.paired_group_rows(rows, rows)
    assert all(v['checkpoint_minus_initial'] == 0 for g in groups for v in g['metrics'].values())
    key = next(iter(rows))
    bad = dict(rows)
    bad.pop(key)
    rejects(lambda: m.paired_group_rows(rows, bad))
    row, protocol = rows[key]
    bad = dict(rows)
    bad[key] = ({**row, 'evaluation_content_sha256': 'changed'}, protocol)
    rejects(lambda: m.paired_group_rows(rows, bad))
    bad[key] = (row, {**protocol, 'temperature': 1})
    rejects(lambda: m.paired_group_rows(rows, bad))
    train = dict(id='t', gpus=16, status='Running', scope='formal_MOPDv6_c1')
    public = dict(id='p', gpus=8, status='Running', scope='formal_MOPD_public_step128_c1')
    control = dict(id='a', gpus=8, status='Running', scope='A_control')
    assert q.available_with_public_reservation([train])[0]
    assert q.available_with_public_reservation([train, public])[0]
    assert not q.available_with_public_reservation([train, control])[0]
    assert not q.available_with_public_reservation([train, public, control])[0]
    assert not q.available_with_public_reservation([train, {**control, 'status': 'UNKNOWN'}])[0]
    # No mutations to the formal controller's primary registry, including when
    # a completed submission is registered again during controller recovery.
    saved = []
    existing = []

    def fake_read(path):
        return list(existing) if path == q.C / 'jobs_v2.json' else []

    def fake_save(path, jobs):
        assert path == q.C / 'jobs_v2.json'
        existing[:] = jobs
        saved.append(path)

    with patch.object(q, 'read', side_effect=fake_read), patch.object(q, 'save', side_effect=fake_save):
        receipt = dict(response=dict(Id='fixture_native_diagnostic'))
        q.register_diagnostic(receipt, 8, q.SCOPE, Path('fixture.yaml'))
        q.register_diagnostic(receipt, 8, q.SCOPE, Path('fixture.yaml'))
        assert len(saved) == 1 and len(existing) == 1
    report = dict(state='passed', validated=time.time(), real_baseline_rows=len(rows),
        real_baseline_evidence_sha256=evidence, baseline_self_comparison_all_zero=True,
        checks=['exact fixed baseline sample', 'real prediction and protocol hashes',
                'complete35989 baseline denominator', 'missing paired ID rejected',
                'changed reference hash rejected', 'changed inference protocol rejected',
                'reserve8publicslots even before submission', 'A-active-or-UNKNOWN prevents allocation',
                'formal16+public8+diagnostic8 within32cap',
                'idempotent secondary registry write avoids formal registry collision'],
        gpu_or_cloud_mutations=False,
        code_sha256={str(p): m.sha(p) for p in [m.C / 'midrun_gs_diagnostic.py',
                      m.C / 'submit_midrun_gs_diagnostic.py', Path(__file__).resolve()]})
    m.save(m.C / 'midrun_gs_cpu_validation.json', report)
    print(json.dumps({k: v for k, v in report.items()
                     if k not in ['real_baseline_evidence_sha256', 'code_sha256']}, indent=2))


if __name__ == '__main__':
    main()
