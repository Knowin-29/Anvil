"""Validate the new control binding with a real completed control, without GPUs."""
import ast
import datetime
import json
from pathlib import Path
import time
import objective_native_diagnostic as diagnostic
import submit_objective_native_diagnostic as controller
C = Path(__file__).resolve().parent


def main():
    root=C/'objective_native_diagnostics';out=root/'cpu_no_mtd_prepare';out.mkdir(parents=True,exist_ok=True)
    spec=diagnostic.prepare(C/'objective_controls_20261029/evaluation/no_mtd/spec.json',out)
    value=diagnostic.load_spec(spec)
    assert value['final_mopd'] is False and value['selection']['expected_records']==35989
    for fn in ['checked_rows','paired_group_rows']:
        def body(path):
            return ast.dump(next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==fn),include_attributes=False)
        assert body(C/'midrun_gs_diagnostic.py')==body(C/'objective_native_diagnostic.py').replace('objective_control_checkpoint','formal_checkpoint')
    base=[dict(gpus=16,status='Running',scope='formal')]
    for extra,expected in [([],True),([dict(gpus=8,status='Running',scope='control')],False),
                           ([dict(gpus=8,status='UNKNOWN',scope='control')],False),
                           ([dict(gpus=8,status='Running',scope='formal_MOPD_public_step512')],True)]:
        assert controller.available_with_public_reservation(base+extra)[0] is expected
    files=[C/'objective_native_diagnostic.py',C/'submit_objective_native_diagnostic.py',
           C/'submit_midrun_gs_diagnostic.py',C/'final_delivery_handoff_v1.py',C/'formal_mopd/controller.py',
           C/'objective_controls_20261029/objective_control_common.py',Path(__file__)]
    receipt=dict(state='passed',actual_control_spec_load=True,matched_records=35989,
        unchanged_pairing_and_scoring_AST=True,public_capacity_reservation_cases_passed=True,
        unknown_capacity_charged=True,final_mopd=False,code_sha256={str(p):diagnostic.sha(p) for p in files},
        actual_spec=diagnostic.bound(spec),completed=time.time())
    diagnostic.save(root/'cpu_validation.json',receipt)
    folder=root/'low_native_sft';folder.mkdir(exist_ok=True)
    initial=diagnostic.read(C/'diagnosis_32gpu/student_adaptation_public_comparison.json')['results']['student_adaptation']['paper_aligned_public']
    policy=dict(schema='anvil-conditional-objective-native-diagnostic-v1',control_task='t-20261029121655-c9bfj',
        minimum_macro_for_diagnostic=sum(initial.values())/7,final_quality_gate=False,
        reason='Spend8GPUs on paired35989 native diagnostics if the full low-native-SFT public macro reaches its exact initial-student baseline; this threshold only allocates diagnostics.',
        launch_deadline=datetime.datetime.fromisoformat('2026-10-29T18:00:00+08:00').timestamp(),
        frozen_files=receipt['code_sha256'],cpu_validation=diagnostic.bound(root/'cpu_validation.json'),
        baseline_public=diagnostic.bound(C/'diagnosis_32gpu/student_adaptation_public_comparison.json'),gpu_cap=32,public_slots_reserved=8)
    diagnostic.save(folder/'policy.json',policy)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
