"""CPU fault injection for submission safety; no actual DLC mutations."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('handoff', ROOT / 'final_delivery_handoff_v1.py')
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def expect_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Expected rejection')


def main():
    plan = handoff.read(ROOT / 'formal_mopd/runs/candidate1_20261029/plan.json')
    done = dict(stage='mopd_training_complete', smoke=False, steps=plan['steps'])
    ready = dict(stage=handoff.READY)
    assert handoff.ready_for_final(plan, ready, done)
    assert not handoff.ready_for_final(plan, ready, {**done, 'smoke': True})
    assert not handoff.ready_for_final(plan, ready, {**done, 'steps': 128})
    assert not handoff.ready_for_final(plan, {'stage': 'monitoring_formal_and_checkpoints'}, done)
    live = handoff.read(Path(plan['training_run']) / 'status.json')
    assert not handoff.ready_for_final(plan, ready, live)
    errors = iter([TimeoutError('cloud timeout'), ValueError('not yet visible'), None])
    calls = []

    def transient(job):
        calls.append(job['id'])
        error = next(errors)
        if error:
            raise error
        return dict(job, status='Running')

    job = dict(id='fixture_job', gpus=16)
    assert handoff.observe(job, check=transient, sleep=lambda _: None)['status'] == 'Running'
    assert len(calls) == 3

    def unavailable(_):
        raise TimeoutError('fixture')

    unknown = handoff.observe(job, check=unavailable, sleep=lambda _: None)
    assert unknown['status'] == 'UNKNOWN'
    assert handoff.charged_gpus([unknown, dict(id='done', gpus=8, status='Success')]) == 16
    expect_error(lambda: handoff.charged_gpus([unknown, unknown]))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 'evaluation').mkdir()
        spec_path = root / 'spec.json'
        spec_path.write_text('{}')
        for node in range(3):
            (root / 'evaluation' / ('node_%d_cli.yaml' % node)).write_text('fixture: true\n')
        with patch.object(handoff, 'assert_policy'), patch.object(handoff, 'load_spec'), \
                patch.object(handoff.formal_controller, 'register') as register, \
                patch.object(handoff, 'cloud_snapshot', return_value=([], 32)), \
                patch.object(handoff.subprocess, 'run') as submit:
            assert handoff.submit_node(root, {}, spec_path, 0) is None
            submit.assert_not_called()
            assert not (root / 'submissions/0/submission_intent.json').exists()
            register.assert_not_called()

        def successful_submit(*args, **kwargs):
            assert (root / 'submissions/0/submission_intent.json').exists()
            return SimpleNamespace(stdout='notice\n{"Id":"fixture_final_0"}\n', stderr='', returncode=0)

        with patch.object(handoff, 'assert_policy'), patch.object(handoff, 'load_spec'), \
                patch.object(handoff.formal_controller, 'register') as register, \
                patch.object(handoff, 'cloud_snapshot', return_value=([], 24)), \
                patch.object(handoff.subprocess, 'run', side_effect=successful_submit) as submit:
            assert handoff.submit_node(root, {}, spec_path, 0) == 'fixture_final_0'
            assert handoff.submit_node(root, {}, spec_path, 0) == 'fixture_final_0'
            assert submit.call_count == 1 and register.call_count == 2

        with patch.object(handoff, 'assert_policy'), patch.object(handoff, 'load_spec'), \
                patch.object(handoff.formal_controller, 'register') as register, \
                patch.object(handoff, 'cloud_snapshot', return_value=([], 24)), \
                patch.object(handoff.subprocess, 'run', return_value=SimpleNamespace(
                    stdout='unknown result', stderr='', returncode=0)) as submit:
            expect_error(lambda: handoff.submit_node(root, {}, spec_path, 1))
            expect_error(lambda: handoff.submit_node(root, {}, spec_path, 1))
            assert submit.call_count == 1
            assert (root / 'submissions/1/submission_intent.json').exists()
            assert not (root / 'submissions/1/submission.json').exists()
            register.assert_not_called()
        (root / 'evaluation/node_0_cli.yaml').write_text('changed: true\n')
        expect_error(lambda: handoff.submit_node(root, {}, spec_path, 0))
    result = dict(state='passed', validation='CPU fault injection, no real final model or DLC submission',
        supervisor_sha256=handoff.sha(ROOT / 'final_delivery_handoff_v1.py'),
        checks=['completed non-smoke full-budget and diagnostic handoff required',
                'live incomplete formal run rejected', 'transient cloud observation recovered',
                'unknown cloud state fully charged', 'duplicate cloud ID rejected',
                '32 GPU full capacity prevents submission', '24+8 GPU submission recorded once',
                'existing receipt avoids duplicate submission',
                'ambiguous outcome preserves intent and prevents retry',
                'changed submitted config rejected'],
        formal_plan_sha256=handoff.sha(ROOT / 'formal_mopd/runs/candidate1_20261029/plan.json'))
    handoff.save(ROOT / 'final_handoff_cpu_validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
